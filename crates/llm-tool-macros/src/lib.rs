//! Proc-macro crate for `llm-tool`.
//!
//! Provides the `#[llm_tool]` attribute macro that transforms a plain function
//! into a strongly-typed [`RustTool`](https://docs.rs/llm-tool/latest/llm_tool/trait.RustTool.html)
//! implementation.
//!
//! With the `md-tmpl` feature enabled, tool descriptions can be
//! loaded from `.tmpl.md` template files via `prompt_file = "..."`, and tool
//! responses can be auto-rendered through templates via
//! `response_file = "..."`.
mod prompt_macro;
mod resource_macro;
#[cfg(feature = "md-tmpl")]
mod response_struct_gen;

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
#[cfg(feature = "md-tmpl")]
use syn::Ident;
use syn::{ItemFn, LitStr, parse_macro_input};

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
/// | `#[llm_tool(response_file = "...")]` | Runtime render | `md-tmpl` |
/// | `#[llm_tool(prompt_file = "tools/x.tmpl.md")]` | Zero (compiled) | `md-tmpl` |
/// | `#[llm_tool(prompt_file = "...", params(k = "v"))]` | Zero (compiled) | `md-tmpl` |
/// | `#[llm_tool(prompt_file = "...", context = fn)]` | Runtime `Cow::Owned` | `md-tmpl` |
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
/// ## Template descriptions (feature: `md-tmpl`)
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

/// Transforms a function into a `RustPrompt` implementation.
#[proc_macro_attribute]
pub fn llm_prompt(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let tool_attr = if attr.is_empty() {
        None
    } else {
        match syn::parse::<ToolAttr>(attr) {
            Ok(parsed) => Some(parsed),
            Err(err) => return err.to_compile_error().into(),
        }
    };
    match prompt_macro::prompt_impl(&func, tool_attr.as_ref()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Transforms a function into a `RustResource` implementation.
#[proc_macro_attribute]
pub fn llm_resource(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let res_attr = match syn::parse::<resource_macro::ResourceAttr>(attr) {
        Ok(parsed) => parsed,
        Err(err) => return err.to_compile_error().into(),
    };
    match resource_macro::resource_impl(&func, &res_attr) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// ── Attribute Parsing ───────────────────────────────────────────────────────

/// Parsed `#[llm_tool(...)]` attribute.
///
/// Supports:
/// - `prompt = "inline text"` — static inline description
/// - `prompt_file = "path.tmpl.md"` — template file (requires `md-tmpl`)
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
    /// Inline response template string (mutually exclusive with `response_file_path`).
    response_inline: Option<LitStr>,
    /// Compile-time key-value pairs for template rendering.
    /// Mutually exclusive with `context_fn`.
    #[cfg(feature = "md-tmpl")]
    inline_params: Vec<(Ident, LitStr)>,
    /// Runtime context function (mutually exclusive with `inline_params`).
    #[cfg(feature = "md-tmpl")]
    context_fn: Option<syn::Path>,
    has_inline_params: bool,
    has_context_fn: bool,
}

const ATTR_PROMPT: &str = "prompt";
const ATTR_PROMPT_FILE: &str = "prompt_file";
const ATTR_RESPONSE_FILE: &str = "response_file";
const ATTR_RESPONSE: &str = "response";
const ATTR_PARAMS: &str = "params";
const ATTR_CONTEXT: &str = "context";
const TYPE_OPTION: &str = "Option";
const TYPE_TOOL_CONTEXT: &str = "ToolContext";
const TYPE_STR: &str = "str";
const ATTR_LLM_TOOL: &str = "llm_tool";

#[derive(Default)]
struct ToolAttrBuilder {
    prompt_inline: Option<syn::LitStr>,
    prompt_file_path: Option<syn::LitStr>,
    response_file_path: Option<syn::LitStr>,
    response_inline: Option<syn::LitStr>,
    #[cfg(feature = "md-tmpl")]
    inline_params: Vec<(syn::Ident, syn::LitStr)>,
    #[cfg(feature = "md-tmpl")]
    context_fn: Option<syn::Path>,
    #[cfg(not(feature = "md-tmpl"))]
    has_inline_params: bool,
    #[cfg(not(feature = "md-tmpl"))]
    has_context_fn: bool,
}

impl ToolAttrBuilder {
    fn parse_single(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
        let ident: syn::Ident = input.parse()?;
        if ident == ATTR_PROMPT {
            let _: syn::Token![=] = input.parse()?;
            self.prompt_inline = Some(input.parse::<syn::LitStr>()?);
        } else if ident == ATTR_PROMPT_FILE {
            let _: syn::Token![=] = input.parse()?;
            self.prompt_file_path = Some(input.parse::<syn::LitStr>()?);
        } else if ident == ATTR_RESPONSE_FILE {
            let _: syn::Token![=] = input.parse()?;
            self.response_file_path = Some(input.parse::<syn::LitStr>()?);
        } else if ident == ATTR_RESPONSE {
            let _: syn::Token![=] = input.parse()?;
            self.response_inline = Some(input.parse::<syn::LitStr>()?);
        } else if ident == ATTR_PARAMS {
            let content;
            syn::parenthesized!(content in input);
            while !content.is_empty() {
                let key: syn::Ident = content.parse()?;
                let _: syn::Token![=] = content.parse()?;
                let value: syn::LitStr = content.parse()?;
                #[cfg(feature = "md-tmpl")]
                self.inline_params.push((key, value));
                #[cfg(not(feature = "md-tmpl"))]
                {
                    drop(key);
                    drop(value);
                }
                if !content.is_empty() {
                    let _: syn::Token![,] = content.parse()?;
                }
            }
            #[cfg(not(feature = "md-tmpl"))]
            {
                self.has_inline_params = true;
            }
        } else if ident == ATTR_CONTEXT {
            let _: syn::Token![=] = input.parse()?;
            #[cfg(feature = "md-tmpl")]
            {
                self.context_fn = Some(input.parse::<syn::Path>()?);
            }
            #[cfg(not(feature = "md-tmpl"))]
            {
                let _path: syn::Path = input.parse()?;
                self.has_context_fn = true;
            }
        } else {
            return Err(syn::Error::new(
                ident.span(),
                "expected `prompt`, `prompt_file`, `response`, `response_file`, `params`, or `context`",
            ));
        }
        Ok(())
    }
}

impl syn::parse::Parse for ToolAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut builder = ToolAttrBuilder::default();

        while !input.is_empty() {
            builder.parse_single(input)?;
            if !input.is_empty() {
                let _: syn::Token![,] = input.parse()?;
            }
        }

        #[cfg(feature = "md-tmpl")]
        let (has_inline_params, has_context_fn) = (
            !builder.inline_params.is_empty(),
            builder.context_fn.is_some(),
        );
        #[cfg(not(feature = "md-tmpl"))]
        let (has_inline_params, has_context_fn) =
            (builder.has_inline_params, builder.has_context_fn);

        validate_tool_attr(
            builder.prompt_inline.as_ref(),
            builder.prompt_file_path.as_ref(),
            has_inline_params,
            has_context_fn,
        )?;

        if builder.response_inline.is_some() && builder.response_file_path.is_some() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "cannot specify both `response` and `response_file`",
            ));
        }

        // Validate response_file requires md-tmpl feature.
        #[cfg(not(feature = "md-tmpl"))]
        if builder.response_file_path.is_some() || builder.response_inline.is_some() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "the `md-tmpl` feature must be enabled to use `response = \"...\"` or `response_file = \"...\"`",
            ));
        }

        Ok(Self {
            prompt_inline: builder.prompt_inline,
            prompt_file_path: builder.prompt_file_path,
            response_file_path: builder.response_file_path,
            response_inline: builder.response_inline,
            #[cfg(feature = "md-tmpl")]
            inline_params: builder.inline_params,
            #[cfg(feature = "md-tmpl")]
            context_fn: builder.context_fn,
            has_inline_params,
            has_context_fn,
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
    let response_info = resolve_response_template(attr, &struct_name, fn_name)?;

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
        #vis struct #struct_name;

        impl #crate_path::RustTool for #struct_name {
            type Params = #params_name;
            const NAME: &'static str = #tool_name_str;
            const DESCRIPTION: &'static str = #static_description;

            #description_method

            
            async fn call(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> {
                ::core::future::ready(()).await;
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

pub(crate) mod desc;
pub(crate) mod helpers;
#[allow(clippy::wildcard_imports)]
pub(crate) use desc::*;
#[allow(clippy::wildcard_imports)]
pub(crate) use helpers::*;
