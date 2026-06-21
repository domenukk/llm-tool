//! Proc-macro crate for `llm-tool`.
//!
//! Provides the `#[llm_tool]` attribute macro that transforms a plain function
//! into a strongly-typed [`RustTool`](https://docs.rs/llm-tool/latest/llm_tool/trait.RustTool.html)
//! implementation.
//!
//! With the `prompt-templates` feature enabled, tool descriptions can be
//! loaded from `.tmpl.md` template files via `prompt_file = "..."`, and tool
//! responses can be auto-rendered through templates via
//! `response_file = "..."`.

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, GenericArgument, Ident, ItemFn, LitStr, Pat, PatType, PathArguments, Token, Type,
    parse::ParseStream, parse_macro_input,
};

/// Transforms a function into a `RustTool` implementation.
///
/// The macro generates:
/// - A `{FnName}Params` struct deriving `Deserialize` and `JsonSchema`
/// - A `{FnName}` unit struct (`PascalCase`) implementing `RustTool`
///
/// The tool **name** is the function name (`snake_case`).
/// The tool **description** comes from one of the sources below.
/// Parameter names and types come from the function signature.
/// Doc comments on parameters become schema descriptions.
///
/// # Description sources (in priority order)
///
/// | Syntax | Cost | Feature |
/// |--------|------|---------|
/// | `#[llm_tool]` + doc comment | Zero (static `&str`) | — |
/// | `#[llm_tool(prompt = "inline text")]` | Zero (static `&str`) | — |
/// | `#[llm_tool(response_file = "...")]` | Runtime render | `prompt-templates` |
/// | `#[llm_tool(prompt_file = "tools/x.tmpl.md")]` | Zero (compiled) | `prompt-templates` |
/// | `#[llm_tool(prompt_file = "...", params(k = "v"))]` | Zero (compiled) | `prompt-templates` |
/// | `#[llm_tool(prompt_file = "...", context = fn)]` | Runtime `Cow::Owned` | `prompt-templates` |
///
/// ## Inline description
///
/// Override or replace the doc comment with an inline string:
///
/// ```text
/// #[llm_tool(prompt = "Get the current weather for a city.")]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// ## Template descriptions (feature: `prompt-templates`)
///
/// Load the description from a `.tmpl.md` file:
///
/// ```text
/// #[llm_tool(prompt_file = "tools/weather.tmpl.md")]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// For templates with variables, provide **compile-time** key-value pairs:
///
/// ```text
/// #[llm_tool(prompt_file = "tools/weather.tmpl.md", params(api = "v3", env = "prod"))]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// The macro reads the template, validates all declared variables are
/// provided, renders the description, and embeds the result as a static
/// string — **zero runtime cost**.
///
/// For **runtime** context (e.g. values from config), provide a context function:
///
/// ```text
/// #[llm_tool(prompt_file = "tools/weather.tmpl.md", context = build_ctx)]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// The context function signature is `fn(&ToolStruct) -> Context`.
/// Templates are parsed once at startup via `LazyLock`.
///
/// # Typed parameters
///
/// Parameters may use `&str` — the generated params struct stores an owned
/// `String` and the macro auto-borrows it before passing to your function body.
///
/// # Response templates
///
/// When `response_file = "path/to/response.tmpl.md"` is provided, the
/// tool's return value (`T: Serialize`) is used to build a template context
/// via `Context::from_serialize`, rendered through the template, and returned
/// as `ToolOutput`. The struct is also attached as metadata.
///
/// # Return types
///
/// The return type can be `Result<T, E>` or just `T` (infallible):
///
/// - **`T`**: `String` (wrapped as-is), `ToolOutput` (passed through), any
///   `T: Serialize` (auto-serialized to JSON), or any `T: Into<ToolOutput>`
/// - **`E`**: any `E: Into<ToolError>` — built-in for `String`, `ToolError`,
///   `std::io::Error`, `serde_json::Error`
#[proc_macro_attribute]
pub fn llm_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let tool_attr = if attr.is_empty() {
        None
    } else {
        match syn::parse::<ToolAttr>(attr) {
            Ok(parsed) => Some(parsed),
            Err(err) => return err.to_compile_error().into(),
        }
    };
    match tool_impl(&func, tool_attr.as_ref()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// ── Attribute Parsing ───────────────────────────────────────────────────────

/// Parsed `#[llm_tool(...)]` attribute.
///
/// Supports:
/// - `prompt = "inline text"` — static inline description
/// - `prompt_file = "path.tmpl.md"` — template file (requires `prompt-templates`)
/// - `params(key = "value", ...)` — compile-time template variables
/// - `context = path::to::fn` — runtime template context function
/// - `response_file = "path.tmpl.md"` — response rendering template
struct ToolAttr {
    /// Inline description string (mutually exclusive with `prompt_file_path`).
    prompt_inline: Option<LitStr>,
    /// Path to a `.tmpl.md` file (mutually exclusive with `prompt_inline`).
    prompt_file_path: Option<LitStr>,
    /// Path to a response `.tmpl.md` file for auto-rendering tool output.
    response_file_path: Option<LitStr>,
    /// Compile-time key-value pairs for template rendering.
    /// Mutually exclusive with `context_fn`.
    #[cfg(feature = "prompt-templates")]
    inline_params: Vec<(Ident, LitStr)>,
    /// Runtime context function (mutually exclusive with `inline_params`).
    #[cfg(feature = "prompt-templates")]
    context_fn: Option<syn::Path>,
}

impl syn::parse::Parse for ToolAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut prompt_inline = None;
        let mut prompt_file_path = None;
        let mut response_file_path = None;
        #[cfg(feature = "prompt-templates")]
        let mut inline_params = Vec::new();
        #[cfg(feature = "prompt-templates")]
        let mut context_fn = None;
        // Track presence for validation even when not storing values.
        #[cfg(not(feature = "prompt-templates"))]
        let mut has_inline_params = false;
        #[cfg(not(feature = "prompt-templates"))]
        let mut has_context_fn = false;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if ident == "prompt" {
                let _: Token![=] = input.parse()?;
                prompt_inline = Some(input.parse::<LitStr>()?);
            } else if ident == "prompt_file" {
                let _: Token![=] = input.parse()?;
                prompt_file_path = Some(input.parse::<LitStr>()?);
            } else if ident == "response_file" {
                let _: Token![=] = input.parse()?;
                response_file_path = Some(input.parse::<LitStr>()?);
            } else if ident == "params" {
                let content;
                syn::parenthesized!(content in input);
                while !content.is_empty() {
                    let key: Ident = content.parse()?;
                    let _: Token![=] = content.parse()?;
                    let value: LitStr = content.parse()?;
                    #[cfg(feature = "prompt-templates")]
                    inline_params.push((key, value));
                    // Suppress unused-variable warnings in the no-feature path.
                    #[cfg(not(feature = "prompt-templates"))]
                    {
                        drop(key);
                        drop(value);
                    }
                    if !content.is_empty() {
                        let _: Token![,] = content.parse()?;
                    }
                }
                #[cfg(not(feature = "prompt-templates"))]
                {
                    has_inline_params = true;
                }
            } else if ident == "context" {
                let _: Token![=] = input.parse()?;
                #[cfg(feature = "prompt-templates")]
                {
                    context_fn = Some(input.parse::<syn::Path>()?);
                }
                #[cfg(not(feature = "prompt-templates"))]
                {
                    let _path: syn::Path = input.parse()?;
                    has_context_fn = true;
                }
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected `prompt`, `prompt_file`, `response_file`, \
                     `params`, or `context`",
                ));
            }

            if !input.is_empty() {
                let _: Token![,] = input.parse()?;
            }
        }

        #[cfg(feature = "prompt-templates")]
        let (has_inline_params, has_context_fn) = (!inline_params.is_empty(), context_fn.is_some());

        validate_tool_attr(
            prompt_inline.as_ref(),
            prompt_file_path.as_ref(),
            has_inline_params,
            has_context_fn,
        )?;

        // Validate response_file requires prompt-templates feature.
        #[cfg(not(feature = "prompt-templates"))]
        if response_file_path.is_some() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "the `prompt-templates` feature must be enabled to use \
                 `response_file = \"...\"`",
            ));
        }

        Ok(Self {
            prompt_inline,
            prompt_file_path,
            response_file_path,
            #[cfg(feature = "prompt-templates")]
            inline_params,
            #[cfg(feature = "prompt-templates")]
            context_fn,
        })
    }
}

/// Validate mutual-exclusion and presence constraints for parsed `#[llm_tool(...)]`
/// attribute fields.
fn validate_tool_attr(
    prompt_inline: Option<&LitStr>,
    prompt_file_path: Option<&LitStr>,
    has_inline_params: bool,
    has_context_fn: bool,
) -> syn::Result<()> {
    // Mutual exclusion: prompt vs prompt_file.
    if prompt_inline.is_some() && prompt_file_path.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`prompt` and `prompt_file` are mutually exclusive",
        ));
    }

    // params/context only make sense with prompt_file.
    if prompt_file_path.is_none() && has_inline_params {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`params(...)` requires `prompt_file = \"...\"`",
        ));
    }
    if prompt_file_path.is_none() && has_context_fn {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`context = ...` requires `prompt_file = \"...\"`",
        ));
    }

    // params and context are mutually exclusive.
    if has_inline_params && has_context_fn {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`params(...)` and `context = ...` are mutually exclusive; \
             use `params` for compile-time values or `context` for runtime values",
        ));
    }

    // Must have at least prompt or prompt_file (unless only response_file
    // is set, in which case doc comments serve as the description).
    if prompt_inline.is_none()
        && prompt_file_path.is_none()
        && !has_inline_params
        && !has_context_fn
    {
        // This is fine — doc comments will be used as fallback.
    }

    Ok(())
}

// ── Implementation ──────────────────────────────────────────────────────────

/// Parsed information about a single function parameter.
struct ParamInfo {
    name: syn::Ident,
    ty: Box<syn::Type>,
    doc_attrs: Vec<syn::Attribute>,
    is_context: bool,
}

/// Information about the function's return type.
enum ReturnInfo {
    /// `Result<T, E>` — fallible tool.
    ResultType {
        ok_type: Box<syn::Type>,
        err_type: Box<syn::Type>,
    },
    /// Bare `T` — infallible tool.
    BareType,
}

fn tool_impl(func: &ItemFn, attr: Option<&ToolAttr>) -> syn::Result<proc_macro2::TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    let tool_name_str = fn_name.to_string();
    let struct_name = format_ident!("{}", tool_name_str.to_case(Case::Pascal));
    let params_name = format_ident!("{}Params", struct_name);

    // Resolve description: template file OR doc comment.
    let DescriptionInfo {
        static_description,
        helper_tokens,
        description_method,
        dep_tracking,
    } = resolve_description(func, attr)?;

    // Resolve response template (if provided).
    let response_info = resolve_response_template(attr)?;

    // Extract parameters, separating ToolContext from regular params.
    let all_params = extract_params(func)?;
    let ctx_param = all_params.iter().find(|p| p.is_context);
    let params: Vec<&ParamInfo> = all_params.iter().filter(|p| !p.is_context).collect();

    // Enforce doc comments on every non-ToolContext parameter.
    for param in &params {
        if param.doc_attrs.is_empty() {
            return Err(syn::Error::new_spanned(
                &param.name,
                format!(
                    "#[llm_tool] parameter `{}` must have a doc comment \
                      (used as the parameter description in the JSON schema)",
                    param.name
                ),
            ));
        }
    }

    // Parse return type: either Result<T, E> or bare T.
    let return_info = parse_return_type(func)?;

    let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();
    let param_descriptions: Vec<String> = params
        .iter()
        .map(|p| extract_doc_string(&p.doc_attrs))
        .collect();

    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(&params);
    let serde_defaults = build_serde_defaults(&params);
    let body_tokens = build_body_tokens(func, &return_info, &crate_path, &response_info);

    let vis = &func.vis;

    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] tool.");
    let struct_doc = format!(
        "Auto-generated tool struct. See the `#[llm_tool]`-annotated function `{fn_name}` for the implementation."
    );

    // If the user's function takes a ToolContext parameter, bind it from the
    // `_ctx` reference provided by the RustTool::call signature.
    let ctx_binding = if let Some(cp) = ctx_param {
        let ctx_name = &cp.name;
        quote! { let #ctx_name = _ctx; }
    } else {
        quote! {}
    };

    let response_dep_tracking = &response_info.dep_tracking;
    let response_helper_tokens = &response_info.helper_tokens;

    Ok(quote! {
        #dep_tracking
        #response_dep_tracking
        #helper_tokens
        #response_helper_tokens

        #[doc = #params_doc]
        #[derive(::serde::Deserialize, ::schemars::JsonSchema)]
        #vis struct #params_name {
            #(
                #[schemars(description = #param_descriptions)]
                #serde_defaults
                pub #param_names: #param_struct_types,
            )*
        }

        #[doc = #struct_doc]
        #[allow(dead_code)] // Constructed at runtime and used via Box<dyn ErasedTool>
        #vis struct #struct_name;

        #[allow(dead_code)] // Methods called through trait-object dispatch (dyn ErasedTool)
        impl #crate_path::RustTool for #struct_name {
            type Params = #params_name;
            const NAME: &'static str = #tool_name_str;
            const DESCRIPTION: &'static str = #static_description;

            #description_method

            async fn call(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> {
                // Import the fallback trait so `Wrap<T>::__convert()` resolves
                // for `T: Serialize` types that lack an inherent `__convert`.
                use #crate_path::__private::SerializeFallback as _;
                // Destructure params into local bindings matching the original
                // function signature.
                let #params_name { #( #param_names, )* } = params;
                // Auto-borrow &str params from their owned String fields.
                #( #borrow_bindings )*
                #ctx_binding
                #body_tokens
            }
        }
    })
}

// ── Description Resolution ──────────────────────────────────────────────────

/// Structured output from description resolution.
struct DescriptionInfo {
    /// Value for `const DESCRIPTION`. For dynamic descriptions, this contains the raw template body.
    static_description: String,
    /// Helper tokens to emit in the crate scope (e.g. `static TEMPLATE`).
    helper_tokens: proc_macro2::TokenStream,
    /// Implementation of the `description(&self)` method if dynamic.
    description_method: Option<proc_macro2::TokenStream>,
    /// Cargo dependency-tracking tokens.
    dep_tracking: proc_macro2::TokenStream,
}

/// Resolve the tool description from attribute or doc comments.
fn resolve_description(func: &ItemFn, attr: Option<&ToolAttr>) -> syn::Result<DescriptionInfo> {
    match attr {
        // Inline prompt string.
        Some(ToolAttr {
            prompt_inline: Some(desc),
            ..
        }) => Ok(DescriptionInfo {
            static_description: desc.value(),
            helper_tokens: quote! {},
            description_method: None,
            dep_tracking: quote! {},
        }),
        // Template file.
        Some(
            tool_attr @ ToolAttr {
                prompt_file_path: Some(_),
                ..
            },
        ) => resolve_template_description(tool_attr),
        // No attribute, or attribute with only response_file — use doc comment.
        _ => {
            let desc = extract_doc_string(&func.attrs);
            if desc.is_empty() {
                return Err(syn::Error::new_spanned(
                    &func.sig.ident,
                    "#[llm_tool] functions must have a doc comment \
                     (used as the tool description), or use \
                     #[llm_tool(prompt = \"...\")]",
                ));
            }
            Ok(DescriptionInfo {
                static_description: desc,
                helper_tokens: quote! {},
                description_method: None,
                dep_tracking: quote! {},
            })
        }
    }
}

/// Read a `.tmpl.md` template file and extract its body as the tool description.
///
/// Handles three sub-cases:
/// 1. Static template (no declared variables) → `const DESCRIPTION`
/// 2. Template + `params(...)` → compile-time render → `const DESCRIPTION`
/// 3. Template + `context = fn` → runtime render via `description()` method
///
/// Without the `prompt-templates` feature: produces a compile error.
fn resolve_template_description(attr: &ToolAttr) -> syn::Result<DescriptionInfo> {
    #[cfg(not(feature = "prompt-templates"))]
    {
        let span = attr
            .prompt_file_path
            .as_ref()
            .map_or(proc_macro2::Span::call_site(), LitStr::span);
        Err(syn::Error::new(
            span,
            "the `prompt-templates` feature must be enabled to use \
             `#[llm_tool(prompt_file = \"...\")]`. \
             Add `features = [\"prompt-templates\"]` to your llm-tool dependency.",
        ))
    }

    #[cfg(feature = "prompt-templates")]
    resolve_template_description_impl(attr)
}

/// Implementation of template description resolution (feature-gated).
///
/// Handles three sub-cases:
/// 1. Static template (no declared variables) → `const DESCRIPTION`
/// 2. Template + `params(...)` → compile-time render → `const DESCRIPTION`
/// 3. Template + `context = fn` → runtime render via `description()` method
#[cfg(feature = "prompt-templates")]
fn resolve_template_description_impl(attr: &ToolAttr) -> syn::Result<DescriptionInfo> {
    let template_lit = attr
        .prompt_file_path
        .as_ref()
        .expect("prompt_file_path validated");
    let rel_path = template_lit.value();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = std::path::Path::new(&manifest_dir).join(&rel_path);

    let source = std::fs::read_to_string(&full_path).map_err(|e| {
        syn::Error::new(
            template_lit.span(),
            format!("failed to read template '{}': {e}", full_path.display()),
        )
    })?;

    let (fm, body) = prompt_templates::parse_frontmatter(&source).map_err(|e| {
        syn::Error::new(
            template_lit.span(),
            format!("template '{rel_path}' error: {e}"),
        )
    })?;

    let body_str = body.trim().to_string();
    let path_str = full_path.to_string_lossy().to_string();

    // include_str! establishes a file dependency so cargo rebuilds
    // when the template changes.
    let dep_tracking = quote! {
        const _: &str = include_str!(#path_str);
    };

    let has_params = !attr.inline_params.is_empty();
    let has_context = attr.context_fn.is_some();
    let has_declarations = !fm.declarations.is_empty();

    if !has_declarations && !has_params && !has_context {
        // Case 1: Static template — no variables, no params, no context.
        Ok(DescriptionInfo {
            static_description: body_str,
            helper_tokens: quote! {},
            description_method: None,
            dep_tracking,
        })
    } else if has_params {
        // Case 2: Compile-time params — render at build time.
        resolve_template_with_params(
            attr,
            &fm,
            &source,
            &rel_path,
            template_lit.span(),
            dep_tracking,
        )
    } else if has_context {
        // Case 3: Runtime context function.
        resolve_context_description(
            attr,
            &rel_path,
            template_lit,
            &body_str,
            &path_str,
            has_declarations,
            dep_tracking,
        )
    } else {
        // Template declares variables but neither params nor context provided.
        let declared: Vec<&str> = fm.declarations.iter().map(|d| d.name.as_str()).collect();
        Err(syn::Error::new(
            template_lit.span(),
            format!(
                "template '{rel_path}' declares parameters ({}) but neither \
                 `params(...)` nor `context = ...` was provided",
                declared.join(", ")
            ),
        ))
    }
}

/// Resolve a template description with a runtime context function.
///
/// Generates a `description(&self)` method that uses `LazyLock` to parse
/// the template once, then renders it with the user-provided context function
/// on every call.
#[cfg(feature = "prompt-templates")]
fn resolve_context_description(
    attr: &ToolAttr,
    rel_path: &str,
    template_lit: &LitStr,
    body_str: &str,
    path_str: &str,
    has_declarations: bool,
    dep_tracking: proc_macro2::TokenStream,
) -> syn::Result<DescriptionInfo> {
    let context_fn = attr.context_fn.as_ref().ok_or_else(|| {
        syn::Error::new(
            template_lit.span(),
            "internal error: resolve_context_description called without context_fn",
        )
    })?;

    if !has_declarations {
        return Err(syn::Error::new(
            template_lit.span(),
            format!(
                "template '{rel_path}' has no declared parameters, \
                 so `context = ...` is unnecessary. Remove `context` \
                 or add params to the template."
            ),
        ));
    }

    // Generate LazyLock inside description() to avoid name collisions
    // when multiple dynamic-description tools exist in the same module.
    let description_method = quote! {
        fn description(&self) -> ::llm_tool::__private::Cow<'static, str> {
            static TEMPLATE: ::llm_tool::__private::Lazy<::llm_tool::__prompt_templates::Template> =
                ::llm_tool::__private::Lazy::new(|| {
                    ::llm_tool::__prompt_templates::Template::from_source(
                        include_str!(#path_str)
                    ).expect("Valid template (verified at compile time)")
                });
            let ctx = #context_fn(self);
            let rendered = TEMPLATE.render_with(
                &ctx,
                ::llm_tool::__prompt_templates::RenderOptions::default().allow_extra(true),
            )
                .expect("Failed to render tool description template");
            ::llm_tool::__private::Cow::Owned(rendered)
        }
    };

    Ok(DescriptionInfo {
        static_description: body_str.to_owned(),
        helper_tokens: quote! {},
        description_method: Some(description_method),
        dep_tracking,
    })
}

/// Render a template with compile-time `params(...)` values.
///
/// Validates:
/// - Every declared template variable has a matching `params(...)` key
/// - Every `params(...)` key matches a declared template variable
/// - The template renders without errors
#[cfg(feature = "prompt-templates")]
fn resolve_template_with_params(
    attr: &ToolAttr,
    fm: &prompt_templates::Frontmatter,
    source: &str,
    rel_path: &str,
    span: proc_macro2::Span,
    dep_tracking: proc_macro2::TokenStream,
) -> syn::Result<DescriptionInfo> {
    let declared_names: std::collections::HashSet<&str> =
        fm.declarations.iter().map(|d| d.name.as_str()).collect();
    let provided_names: std::collections::HashSet<String> = attr
        .inline_params
        .iter()
        .map(|(k, _)| k.to_string())
        .collect();

    // Check for missing params (declared but not provided).
    let missing: Vec<&str> = declared_names
        .iter()
        .filter(|n| !provided_names.contains(**n))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(syn::Error::new(
            span,
            format!(
                "template '{rel_path}' declares parameters not provided in `params(...)`: {}",
                missing.join(", ")
            ),
        ));
    }

    // Check for extra params (provided but not declared).
    for (key, _) in &attr.inline_params {
        let key_str = key.to_string();
        if !declared_names.contains(key_str.as_str()) {
            return Err(syn::Error::new(
                key.span(),
                format!(
                    "param `{key_str}` is not declared in template '{rel_path}'. \
                     Declared params: {}",
                    declared_names.into_iter().collect::<Vec<_>>().join(", ")
                ),
            ));
        }
    }

    // Build context and render at compile time.
    let template = prompt_templates::Template::from_source(source)
        .map_err(|e| syn::Error::new(span, format!("template '{rel_path}' parse error: {e}")))?;

    let mut ctx = prompt_templates::Context::new();
    for (key, value) in &attr.inline_params {
        ctx.set(key.to_string(), value.value());
    }

    let rendered = template
        .render(&ctx)
        .map_err(|e| syn::Error::new(span, format!("template '{rel_path}' render error: {e}")))?;

    Ok(DescriptionInfo {
        static_description: rendered,
        helper_tokens: quote! {},
        description_method: None,
        dep_tracking,
    })
}

/// Build the struct field types and any auto-borrow bindings for `&str` params.
fn build_param_types_and_borrows(
    params: &[&ParamInfo],
) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    params
        .iter()
        .map(|p| {
            if is_str_ref(&p.ty) {
                // &str → String in struct, auto-borrow in body
                let name = &p.name;
                (quote! { String }, quote! { let #name: &str = &#name; })
            } else {
                let ty = &p.ty;
                (quote! { #ty }, quote! {})
            }
        })
        .unzip()
}

/// Build `#[serde(default)]` annotations for `Option<T>` params.
fn build_serde_defaults(params: &[&ParamInfo]) -> Vec<proc_macro2::TokenStream> {
    params
        .iter()
        .map(|p| {
            if is_option_type(&p.ty) {
                quote! { #[serde(default)] }
            } else {
                quote! {}
            }
        })
        .collect()
}

/// Build the body tokens that wrap the user's function body.
///
/// Uses compile-time dispatch via `__private::Wrap(v).__convert()` —
/// the compiler resolves the correct conversion (inherent method for
/// `String`/`ToolOutput`/`Json<T>`, or `SerializeFallback` trait for
/// `T: Serialize`) without any proc-macro type-name matching.
///
/// When a `response_template` is specified, the return value is instead
/// rendered through the template and returned as `ToolOutput` with the
/// struct attached as metadata.
fn build_body_tokens(
    func: &ItemFn,
    return_info: &ReturnInfo,
    crate_path: &proc_macro2::TokenStream,
    response_info: &ResponseTemplateInfo,
) -> proc_macro2::TokenStream {
    let is_async = func.sig.asyncness.is_some();
    let body_stmts = &func.block.stmts;

    match return_info {
        ReturnInfo::ResultType { ok_type, err_type } => {
            let inner = if is_async {
                quote! {
                    let __r: ::core::result::Result<#ok_type, #err_type> = async move {
                        #( #body_stmts )*
                    }.await;
                }
            } else {
                quote! {
                    let __r: ::core::result::Result<#ok_type, #err_type> = (|| { #( #body_stmts )* })();
                }
            };
            let ok_branch = build_ok_branch(crate_path, response_info);
            quote! {
                #inner
                match __r {
                    ::core::result::Result::Ok(__v) => { #ok_branch },
                    ::core::result::Result::Err(__e) => ::core::result::Result::Err(::core::convert::Into::into(__e)),
                }
            }
        }
        ReturnInfo::BareType => {
            let inner = if is_async {
                quote! {
                    let __v = async move { #( #body_stmts )* }.await;
                }
            } else {
                quote! {
                    let __v = (|| { #( #body_stmts )* })();
                }
            };
            let ok_branch = build_ok_branch(crate_path, response_info);
            quote! {
                #inner
                #ok_branch
            }
        }
    }
}

/// Build the Ok-branch conversion: either the standard `Wrap(v).__convert()`
/// or template-based rendering when `response_template` is set.
fn build_ok_branch(
    crate_path: &proc_macro2::TokenStream,
    response_info: &ResponseTemplateInfo,
) -> proc_macro2::TokenStream {
    if let Some(ref render_tokens) = response_info.render_tokens {
        render_tokens.clone()
    } else {
        quote! { #crate_path::__private::Wrap(__v).__convert() }
    }
}

// ── Response Template Resolution ────────────────────────────────────────────

/// Structured output from response template resolution.
struct ResponseTemplateInfo {
    /// Cargo dependency-tracking tokens.
    dep_tracking: proc_macro2::TokenStream,
    /// Helper tokens (e.g. static `LazyLock` declarations).
    helper_tokens: proc_macro2::TokenStream,
    /// Token stream that converts `__v` into `Result<ToolOutput, ToolError>`
    /// via template rendering. `None` = use default `__convert()` path.
    render_tokens: Option<proc_macro2::TokenStream>,
}

impl Default for ResponseTemplateInfo {
    fn default() -> Self {
        Self {
            dep_tracking: quote! {},
            helper_tokens: quote! {},
            render_tokens: None,
        }
    }
}

/// Resolve response template from the tool attribute.
///
/// When `response_file = "..."` is set (and `prompt-templates` feature is
/// enabled), reads the template file at compile time, validates it, and
/// generates a `Lazy`-based renderer that converts the tool's return
/// value (any `T: Serialize`) into rendered text.
fn resolve_response_template(attr: Option<&ToolAttr>) -> syn::Result<ResponseTemplateInfo> {
    let Some(ToolAttr {
        response_file_path: Some(response_path),
        ..
    }) = attr
    else {
        return Ok(ResponseTemplateInfo::default());
    };

    #[cfg(not(feature = "prompt-templates"))]
    {
        Err(syn::Error::new(
            response_path.span(),
            "the `prompt-templates` feature must be enabled to use \
             `response_file = \"...\"`",
        ))
    }

    #[cfg(feature = "prompt-templates")]
    resolve_response_template_impl(response_path)
}

/// Feature-gated implementation of response template resolution.
#[cfg(feature = "prompt-templates")]
fn resolve_response_template_impl(response_path: &LitStr) -> syn::Result<ResponseTemplateInfo> {
    let rel_path = response_path.value();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = std::path::Path::new(&manifest_dir).join(&rel_path);
    let path_str = full_path.to_string_lossy().to_string();

    // Validate the template file exists and parses at compile time.
    let source = std::fs::read_to_string(&full_path).map_err(|e| {
        syn::Error::new(
            response_path.span(),
            format!(
                "failed to read response template '{}': {e}",
                full_path.display()
            ),
        )
    })?;

    // Parse to validate syntax at compile time.
    prompt_templates::Template::from_source(&source).map_err(|e| {
        syn::Error::new(
            response_path.span(),
            format!("response template '{rel_path}' parse error: {e}"),
        )
    })?;

    let dep_tracking = quote! {
        const _: &str = include_str!(#path_str);
    };

    let render_tokens = quote! {
        {
            static __RESPONSE_TMPL: ::llm_tool::__private::Lazy<::llm_tool::__prompt_templates::Template> =
                ::llm_tool::__private::Lazy::new(|| {
                    ::llm_tool::__prompt_templates::Template::from_source(
                        include_str!(#path_str)
                    ).expect("valid response template (verified at compile time)")
                });
            let __ctx = ::llm_tool::__prompt_templates::Context::from_serialize(&__v)
                .map_err(|e| ::llm_tool::ToolError::new(
                    format!("response template context error: {e}")
                ))?;
            let __rendered = __RESPONSE_TMPL.render(&__ctx)
                .map_err(|e| ::llm_tool::ToolError::new(
                    format!("response template render error: {e}")
                ))?;
            ::llm_tool::ToolOutput::new(__rendered)
                .with_metadata(&__v)
                .map_err(|e| ::llm_tool::ToolError::new(
                    format!("response metadata error: {e}")
                ))
        }
    };

    Ok(ResponseTemplateInfo {
        dep_tracking,
        helper_tokens: quote! {},
        render_tokens: Some(render_tokens),
    })
}

/// Check whether `ty` is `Option<T>` (or `std::option::Option<T>`).
fn is_option_type(ty: &syn::Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(last_seg) = type_path.path.segments.last() else {
        return false;
    };
    if last_seg.ident != "Option" {
        return false;
    }
    matches!(&last_seg.arguments, PathArguments::AngleBracketed(args)
        if args.args.len() == 1
            && matches!(args.args.first(), Some(GenericArgument::Type(_))))
}

/// Check whether `ty` is `ToolContext`, `&ToolContext`, or a qualified path
/// ending in `ToolContext`.
fn is_tool_context_type(ty: &syn::Type) -> bool {
    let inner = match ty {
        Type::Reference(r) => r.elem.as_ref(),
        other => other,
    };
    let Type::Path(type_path) = inner else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "ToolContext")
}

/// Check whether `ty` is `&str`.
fn is_str_ref(ty: &syn::Type) -> bool {
    let Type::Reference(ref_type) = ty else {
        return false;
    };
    if ref_type.mutability.is_some() {
        return false;
    }
    let Type::Path(type_path) = ref_type.elem.as_ref() else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "str" && seg.arguments.is_none())
}

fn is_explicit_context_attr(attr: &syn::Attribute) -> syn::Result<bool> {
    if !attr.path().is_ident("llm_tool") {
        return Ok(false);
    }
    let mut is_context = false;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("context") {
            is_context = true;
            Ok(())
        } else {
            Err(meta.error("unsupported llm_tool attribute"))
        }
    })?;
    Ok(is_context)
}

fn extract_params(func: &ItemFn) -> syn::Result<Vec<ParamInfo>> {
    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "#[llm_tool] functions must be free functions (no `self`)",
                ));
            }
            FnArg::Typed(PatType { pat, ty, attrs, .. }) => {
                let name = match pat.as_ref() {
                    Pat::Ident(ident) => ident.ident.clone(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "#[llm_tool] parameters must be simple identifiers",
                        ));
                    }
                };

                let mut has_context_attr = false;
                for a in attrs {
                    has_context_attr |= is_explicit_context_attr(a)?;
                }
                let is_tool_context = is_tool_context_type(ty);
                let is_context = has_context_attr || is_tool_context;

                if is_tool_context && !matches!(ty.as_ref(), syn::Type::Reference(_)) {
                    return Err(syn::Error::new_spanned(
                        ty,
                        "ToolContext parameter must be a reference type (e.g., `&ToolContext` or `&'a ToolContext`)",
                    ));
                }

                let doc_attrs: Vec<syn::Attribute> = attrs
                    .iter()
                    .filter(|a| a.path().is_ident("doc"))
                    .cloned()
                    .collect();
                params.push(ParamInfo {
                    name,
                    ty: ty.clone(),
                    doc_attrs,
                    is_context,
                });
            }
        }
    }
    Ok(params)
}

fn extract_doc_string(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            if let syn::Meta::NameValue(nv) = &attr.meta
                && let syn::Expr::Lit(lit) = &nv.value
                && let syn::Lit::Str(s) = &lit.lit
            {
                return Some(s.value());
            }
            None
        })
        .collect();
    lines
        .iter()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Parse the return type — either `Result<T, E>` or a bare type `T`.
fn parse_return_type(func: &ItemFn) -> syn::Result<ReturnInfo> {
    let syn::ReturnType::Type(_, ty) = &func.sig.output else {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_tool] functions must have an explicit return type",
        ));
    };

    // Try to parse as Result<T, E>.
    if let Some(result_types) = try_extract_result_types(ty) {
        return Ok(result_types);
    }

    // Not a Result — treat as infallible bare type.
    Ok(ReturnInfo::BareType)
}

/// Try to extract `T` and `E` from a `Result<T, E>` return type.
/// Returns `None` if the type is not a `Result`.
fn try_extract_result_types(ty: &syn::Type) -> Option<ReturnInfo> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    let last_seg = type_path.path.segments.last()?;

    if last_seg.ident != "Result" {
        return None;
    }

    let PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };

    if args.args.len() != 2 {
        return None;
    }

    let GenericArgument::Type(ok_type) = &args.args[0] else {
        return None;
    };

    let GenericArgument::Type(err_type) = &args.args[1] else {
        return None;
    };

    Some(ReturnInfo::ResultType {
        ok_type: Box::new(ok_type.clone()),
        err_type: Box::new(err_type.clone()),
    })
}
