//! Proc-macro crate for `llm-tool`.
//!
//! Provides the `#[llm_tool]` attribute macro that transforms a plain function
//! into a strongly-typed [`RustTool`](https://docs.rs/llm-tool/latest/llm_tool/trait.RustTool.html)
//! implementation.
//!
//! With the `md-tmpl` feature enabled, tool descriptions can be
//! loaded from `.tmpl.md` template files via `description_file = "..."`, and tool
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
/// | `#[llm_tool(description = "inline text")]` | Zero (static `&str`) | — |
/// | `#[llm_tool(response_file = "...")]` | Runtime render | `md-tmpl` |
/// | `#[llm_tool(description_file = "tools/x.tmpl.md")]` | Zero (compiled) | `md-tmpl` |
/// | `#[llm_tool(description_file = "...", params(k = "v"))]` | Zero (compiled) | `md-tmpl` |
/// | `#[llm_tool(description_file = "...", env(K = "v"))]` | Zero (compiled) | `md-tmpl` |
/// | `#[llm_tool(description_file = "...", context = fn)]` | Runtime `Cow::Owned` | `md-tmpl` |
///
/// ## Inline description
///
/// Override or replace the doc comment with an inline string:
///
/// ```text
/// #[llm_tool(description = "Get the current weather for a city.")]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// ## Template descriptions (feature: `md-tmpl`)
///
/// Load the description from a `.tmpl.md` file:
///
/// ```text
/// #[llm_tool(description_file = "tools/weather.tmpl.md")]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// For templates with variables, provide **compile-time** key-value pairs:
///
/// ```text
/// #[llm_tool(description_file = "tools/weather.tmpl.md", params(api = "v3", env = "prod"))]
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
/// #[llm_tool(description_file = "tools/weather.tmpl.md", context = build_ctx)]
/// fn get_weather(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// The context function signature is `fn(&ToolStruct) -> Context`.
/// Templates are parsed once at startup via `LazyLock`.
///
/// ## Environment variables (feature: `md-tmpl`)
///
/// Templates can declare `env:` variables in their frontmatter. These are
/// separate from `params:` — they represent build-time configuration
/// (deployment environment, API version, etc.) rather than template parameters.
///
/// In the template:
/// ```text
/// ---
/// env:
///   - API_VERSION = str
///   - MAX_RETRIES = int := 3
/// ---
/// Uses API {{ API_VERSION }} with {{ MAX_RETRIES }} retries.
/// ```
///
/// Supply values via the `env(...)` attribute:
/// ```text
/// #[llm_tool(description_file = "tools/api.tmpl.md", env(API_VERSION = "v5"))]
/// fn query_api(/* … */) -> Result<String, ToolError> { /* … */ }
/// ```
///
/// Env values are resolved at compile time, producing a zero-cost static
/// description. They can be combined with `params(...)` or `context = fn`.
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
/// - `description = "inline text"` — static inline description
/// - `description_file = "path.tmpl.md"` — template file (requires `md-tmpl`)
/// - `params(key = "value", ...)` — compile-time template variables
/// - `env(KEY = "value", ...)` — compile-time environment variables for `env:` frontmatter
/// - `context = path::to::fn` — runtime template context function
/// - `response_file = "path.tmpl.md"` — response rendering template
struct ToolAttr {
    /// Inline description string (mutually exclusive with `description_file_path`).
    description_inline: Option<LitStr>,
    /// Path to a `.tmpl.md` file (mutually exclusive with `description_inline`).
    description_file_path: Option<LitStr>,
    /// Path to a response `.tmpl.md` file for auto-rendering tool output.
    response_file_path: Option<LitStr>,
    /// Inline response template string (mutually exclusive with `response_file_path`).
    response_inline: Option<LitStr>,
    /// Compile-time key-value pairs for template rendering.
    /// Mutually exclusive with `context_fn`.
    #[cfg(feature = "md-tmpl")]
    inline_params: Vec<(Ident, LitStr)>,
    /// Compile-time environment variables for `env:` frontmatter declarations.
    #[cfg(feature = "md-tmpl")]
    env_vars: Vec<(Ident, syn::Lit)>,
    /// Runtime context function (mutually exclusive with `inline_params`).
    #[cfg(feature = "md-tmpl")]
    context_fn: Option<syn::Path>,
    has_inline_params: bool,
    has_context_fn: bool,
}

pub(crate) const MACRO_LLM_TOOL: &str = "llm_tool";
pub(crate) const MACRO_LLM_PROMPT: &str = "llm_prompt";
pub(crate) const MACRO_LLM_RESOURCE: &str = "llm_resource";

pub(crate) const ATTR_DESCRIPTION: &str = "description";
pub(crate) const ATTR_DESCRIPTION_FILE: &str = "description_file";
pub(crate) const ATTR_RESPONSE_FILE: &str = "response_file";
pub(crate) const ATTR_RESPONSE: &str = "response";
pub(crate) const ATTR_PARAMS: &str = "params";
pub(crate) const ATTR_CONTEXT: &str = "context";
pub(crate) const ATTR_ENV: &str = "env";
pub(crate) const ATTR_DOC: &str = "doc";

pub(crate) const TYPE_OPTION: &str = "Option";
pub(crate) const TYPE_TOOL_CONTEXT: &str = "ToolContext";
pub(crate) const TYPE_STR: &str = "str";
pub(crate) const TYPE_RESULT: &str = "Result";

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum ToolAttrKey {
    Description,
    DescriptionFile,
    ResponseFile,
    Response,
    Params,
    Env,
    Context,
}

impl ToolAttrKey {
    pub(crate) const ALL: &'static [Self] = &[
        Self::Description,
        Self::DescriptionFile,
        Self::Response,
        Self::ResponseFile,
        Self::Params,
        Self::Env,
        Self::Context,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Description => ATTR_DESCRIPTION,
            Self::DescriptionFile => ATTR_DESCRIPTION_FILE,
            Self::ResponseFile => ATTR_RESPONSE_FILE,
            Self::Response => ATTR_RESPONSE,
            Self::Params => ATTR_PARAMS,
            Self::Env => ATTR_ENV,
            Self::Context => ATTR_CONTEXT,
        }
    }

    pub(crate) fn expected_keys_error(span: proc_macro2::Span) -> syn::Error {
        let mut parts: Vec<String> = Self::ALL
            .iter()
            .map(|k| format!("`{}`", k.as_str()))
            .collect();
        let last = parts.pop().unwrap_or_default();
        let formatted = if parts.is_empty() {
            last
        } else {
            format!("{}, or {last}", parts.join(", "))
        };
        syn::Error::new(span, format!("expected {formatted}"))
    }
}

impl TryFrom<&syn::Ident> for ToolAttrKey {
    type Error = syn::Error;

    fn try_from(ident: &syn::Ident) -> Result<Self, Self::Error> {
        let s = ident.to_string();
        for &variant in Self::ALL {
            if s == variant.as_str() {
                return Ok(variant);
            }
        }
        Err(Self::expected_keys_error(ident.span()))
    }
}

#[derive(Default)]
struct ToolAttrBuilder {
    description_inline: Option<syn::LitStr>,
    description_file_path: Option<syn::LitStr>,
    response_file_path: Option<syn::LitStr>,
    response_inline: Option<syn::LitStr>,
    #[cfg(feature = "md-tmpl")]
    inline_params: Vec<(syn::Ident, syn::LitStr)>,
    #[cfg(feature = "md-tmpl")]
    env_vars: Vec<(syn::Ident, syn::Lit)>,
    #[cfg(feature = "md-tmpl")]
    context_fn: Option<syn::Path>,
    #[cfg(not(feature = "md-tmpl"))]
    has_inline_params: bool,
    #[cfg(not(feature = "md-tmpl"))]
    has_context_fn: bool,
    #[cfg(not(feature = "md-tmpl"))]
    has_env: bool,
}

impl ToolAttrBuilder {
    fn parse_params_attr(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
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
        Ok(())
    }

    fn parse_env_attr(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
        let content;
        syn::parenthesized!(content in input);
        while !content.is_empty() {
            let key: syn::Ident = content.parse()?;
            let _: syn::Token![=] = content.parse()?;
            let value: syn::Lit = content.parse()?;
            match &value {
                syn::Lit::Str(_) | syn::Lit::Int(_) | syn::Lit::Float(_) | syn::Lit::Bool(_) => {}
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "env values must be string, integer, float, or bool literals",
                    ));
                }
            }
            #[cfg(feature = "md-tmpl")]
            self.env_vars.push((key, value));
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
            self.has_env = true;
        }
        Ok(())
    }

    fn parse_single(&mut self, input: syn::parse::ParseStream) -> syn::Result<()> {
        let ident: syn::Ident = input.parse()?;
        let key = ToolAttrKey::try_from(&ident)?;

        match key {
            ToolAttrKey::Description => {
                let _: syn::Token![=] = input.parse()?;
                if self.description_inline.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("duplicate `{}` attribute", key.as_str()),
                    ));
                }
                self.description_inline = Some(input.parse::<syn::LitStr>()?);
            }
            ToolAttrKey::DescriptionFile => {
                let _: syn::Token![=] = input.parse()?;
                if self.description_file_path.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("duplicate `{}` attribute", key.as_str()),
                    ));
                }
                self.description_file_path = Some(input.parse::<syn::LitStr>()?);
            }
            ToolAttrKey::ResponseFile => {
                let _: syn::Token![=] = input.parse()?;
                if self.response_file_path.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("duplicate `{}` attribute", key.as_str()),
                    ));
                }
                self.response_file_path = Some(input.parse::<syn::LitStr>()?);
            }
            ToolAttrKey::Response => {
                let _: syn::Token![=] = input.parse()?;
                if self.response_inline.is_some() {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("duplicate `{}` attribute", key.as_str()),
                    ));
                }
                self.response_inline = Some(input.parse::<syn::LitStr>()?);
            }
            ToolAttrKey::Params => {
                self.parse_params_attr(input)?;
            }
            ToolAttrKey::Env => {
                self.parse_env_attr(input)?;
            }
            ToolAttrKey::Context => {
                let _: syn::Token![=] = input.parse()?;
                #[cfg(feature = "md-tmpl")]
                {
                    if self.context_fn.is_some() {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    self.context_fn = Some(input.parse::<syn::Path>()?);
                }
                #[cfg(not(feature = "md-tmpl"))]
                {
                    let _path: syn::Path = input.parse()?;
                    if self.has_context_fn {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    self.has_context_fn = true;
                }
            }
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
        let has_inline_params = !builder.inline_params.is_empty();
        #[cfg(not(feature = "md-tmpl"))]
        let has_inline_params = builder.has_inline_params;

        #[cfg(feature = "md-tmpl")]
        let has_context_fn = builder.context_fn.is_some();
        #[cfg(not(feature = "md-tmpl"))]
        let has_context_fn = builder.has_context_fn;

        validate_tool_attr(&builder)?;

        Ok(Self {
            description_inline: builder.description_inline,
            description_file_path: builder.description_file_path,
            response_file_path: builder.response_file_path,
            response_inline: builder.response_inline,
            #[cfg(feature = "md-tmpl")]
            inline_params: builder.inline_params,
            #[cfg(feature = "md-tmpl")]
            env_vars: builder.env_vars,
            #[cfg(feature = "md-tmpl")]
            context_fn: builder.context_fn,
            has_inline_params,
            has_context_fn,
        })
    }
}

fn validate_tool_attr(builder: &ToolAttrBuilder) -> syn::Result<()> {
    if builder.description_inline.is_some() && builder.description_file_path.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`description` and `description_file` are mutually exclusive",
        ));
    }

    if builder.response_file_path.is_some() && builder.response_inline.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`response` and `response_file` are mutually exclusive",
        ));
    }

    #[cfg(feature = "md-tmpl")]
    let has_inline_params = !builder.inline_params.is_empty();
    #[cfg(not(feature = "md-tmpl"))]
    let has_inline_params = builder.has_inline_params;

    #[cfg(feature = "md-tmpl")]
    let has_context_fn = builder.context_fn.is_some();
    #[cfg(not(feature = "md-tmpl"))]
    let has_context_fn = builder.has_context_fn;

    #[cfg(feature = "md-tmpl")]
    let has_env = !builder.env_vars.is_empty();
    #[cfg(not(feature = "md-tmpl"))]
    let has_env = builder.has_env;

    if has_inline_params && has_context_fn {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`params(...)` and `context = ...` are mutually exclusive; \
             use `params` for compile-time values or `context` for runtime values",
        ));
    }

    if has_inline_params
        && builder.description_file_path.is_none()
        && builder.description_inline.is_none()
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`params(...)` requires `description_file = \"...\"` or `description = \"...\"`",
        ));
    }

    if has_context_fn
        && builder.description_file_path.is_none()
        && builder.description_inline.is_none()
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`context = ...` requires `description_file = \"...\"` or `description = \"...\"`",
        ));
    }

    if has_env && builder.description_file_path.is_none() && builder.description_inline.is_none() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`env(...)` requires `description_file = \"...\"` or `description = \"...\"`",
        ));
    }

    #[cfg(not(feature = "md-tmpl"))]
    if builder.description_file_path.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`description_file` requires the `md-tmpl` feature of `llm-tool`",
        ));
    }

    #[cfg(not(feature = "md-tmpl"))]
    if builder.response_file_path.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`response_file` requires the `md-tmpl` feature of `llm-tool`",
        ));
    }

    #[cfg(not(feature = "md-tmpl"))]
    if builder.response_inline.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "`response` requires the `md-tmpl` feature of `llm-tool`",
        ));
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
    is_mut: bool,
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
    reject_generic_signature(func, MACRO_LLM_TOOL)?;
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
    let all_params = extract_params(func, MACRO_LLM_TOOL)?;
    let ctx_count = all_params.iter().filter(|p| p.is_context).count();
    if ctx_count > 1 {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_tool] functions can accept at most one ToolContext parameter",
        ));
    }
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
    let return_info = parse_return_type(func, MACRO_LLM_TOOL)?;

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

    let mut_tokens: Vec<proc_macro2::TokenStream> = params
        .iter()
        .map(|p| {
            if p.is_mut {
                quote! { mut }
            } else {
                quote! {}
            }
        })
        .collect();

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


            // NOLINT: macro-generated code — the impl may not be async depending on user's function
            #[allow(unknown_lints, clippy::unused_async_trait_impl)]
            async fn call(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> {

                // Import the fallback trait so `Wrap<T>::__convert()` resolves
                // for `T: Serialize` types that lack an inherent `__convert`.
                use #crate_path::__private::SerializeFallback as _;
                // Destructure params into local bindings matching the original
                // function signature.
                let #params_name { #( #mut_tokens #param_names, )* } = params;
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
pub(crate) use desc::resolve_description;
pub(crate) use helpers::{
    build_body_tokens, build_param_types_and_borrows, build_serde_defaults, extract_doc_string,
    extract_params, parse_return_type, reject_generic_signature, resolve_response_template,
};
