//! Proc-macro crate for `llm-tool`.
//!
//! Provides `#[llm_tool]` (on both free functions and stateful `impl MyType` blocks),
//! `#[llm_prompt]`, and `#[llm_resource]`.

mod attr;
mod builder_derive;
pub(crate) mod desc;
pub(crate) mod helpers;
mod impl_macro;
pub(crate) mod params;
mod prompt_macro;
mod resource_macro;
mod response_struct_gen;

pub(crate) use attr::{
    ATTR_CONTEXT, ATTR_DESCRIPTION, ATTR_DESCRIPTION_FILE, ATTR_EFFECT, ATTR_ENV, ATTR_IDEMPOTENT,
    ATTR_NAME, ATTR_PARAMS, ATTR_PROVENANCE, ATTR_QUARANTINE, ATTR_SANITIZATION, ATTR_SANITIZE,
    MacroOutputSanitization, MacroToolEffect, ToolAttr, ToolAttrBuilder, ToolAttrKey,
};
use convert_case::{Case, Casing};
pub(crate) use desc::{resolve_description, resolve_description_parts};
pub(crate) use helpers::{
    build_body_tokens, build_extension_extractions, build_field_attrs_and_default_helpers,
    build_param_types_and_borrows, extract_doc_string, extract_params, parse_return_type,
    parse_sig_return_type, reject_generic_signature, resolve_response_template,
};
pub(crate) use params::{ParamInfo, destructure_mut_tokens};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, ext::IdentExt as _, parse_macro_input};

pub(crate) const MACRO_LLM_TOOL: &str = "llm_tool";
pub(crate) const MACRO_LLM_PROMPT: &str = "llm_prompt";
pub(crate) const MACRO_LLM_RESOURCE: &str = "llm_resource";
pub(crate) const ATTR_DOC: &str = "doc";

pub(crate) const TYPE_OPTION: &str = "Option";
pub(crate) const TYPE_TOOL_CONTEXT: &str = "ToolContext";
pub(crate) const TYPE_STR: &str = "str";
pub(crate) const TYPE_RESULT: &str = "Result";

/// Transforms a free function or stateful `impl MyType` block into `RustTool` implementation(s).
#[proc_macro_attribute]
pub fn llm_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    match syn::parse::<syn::ItemImpl>(item.clone()) {
        Ok(item_impl) => {
            if !attr.is_empty() {
                return syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[llm_tool] on an `impl` block takes no arguments; place attributes on each `#[llm_tool(...)]` method inside the `impl` block",
                )
                .to_compile_error()
                .into();
            }
            return match impl_macro::tool_impl_block(item_impl) {
                Ok(tokens) => tokens.into(),
                Err(err) => err.to_compile_error().into(),
            };
        }
        Err(not_impl_err) => {
            let has_msg = !not_impl_err.to_string().is_empty();
            debug_assert!(has_msg);
        }
    }

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

/// Internal derive macro used for `md-tmpl` generated parameter and response structs.
#[doc(hidden)]
#[proc_macro_derive(ResponseStructBuilder, attributes(builder))]
pub fn derive_response_struct_builder(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);
    match builder_derive::derive_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Information about the function's return type.
pub(crate) enum ReturnInfo {
    /// `Result<T, E>` — fallible tool/prompt/resource.
    ResultType {
        ok_type: Box<syn::Type>,
        err_type: Box<syn::Type>,
    },
    /// Bare `T` — infallible tool/prompt/resource.
    BareType,
}

/// Structured output from description resolution.
pub(crate) struct DescriptionInfo {
    /// Value for `const DESCRIPTION`. For dynamic descriptions, this contains the raw template body.
    pub static_description: String,
    /// Helper tokens to emit in the crate scope (e.g. `static TEMPLATE`).
    pub helper_tokens: proc_macro2::TokenStream,
    /// Implementation of the `description(&self)` method if dynamic.
    pub description_method: Option<proc_macro2::TokenStream>,
    /// Cargo dependency-tracking tokens.
    pub dep_tracking: proc_macro2::TokenStream,
}

pub(crate) fn validate_params(all_params: &[ParamInfo], sig: &syn::Signature) -> syn::Result<()> {
    let ctx_count = all_params.iter().filter(|p| p.is_context()).count();
    if ctx_count > 1 {
        return Err(syn::Error::new_spanned(
            sig,
            "#[llm_tool] functions can accept at most one ToolContext parameter",
        ));
    }
    for param in all_params.iter().filter(|p| p.is_schema_input()) {
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
    Ok(())
}

pub(crate) fn resolve_sanitization_tokens(
    attr: Option<&ToolAttr>,
    crate_path: &proc_macro2::TokenStream,
) -> (proc_macro2::TokenStream, bool, bool) {
    let san_val = attr.map_or_else(MacroOutputSanitization::default, |a| a.sanitization.clone());
    let san_tokens = match &san_val {
        MacroOutputSanitization::None => quote!(#crate_path::OutputSanitization::None),
        MacroOutputSanitization::SanitizeTokens => {
            quote!(#crate_path::OutputSanitization::SanitizeTokens)
        }
        MacroOutputSanitization::Quarantine => quote!(#crate_path::OutputSanitization::Quarantine),
        MacroOutputSanitization::SanitizeTagged(tag) => {
            quote!(#crate_path::OutputSanitization::SanitizeTagged(#tag))
        }
        MacroOutputSanitization::SanitizeWithNotice { tag, notice } => {
            quote!(#crate_path::OutputSanitization::SanitizeWithNotice { tag: #tag, notice: #notice })
        }
    };
    let sanitize_bool = !matches!(san_val, MacroOutputSanitization::None);
    let quarantine_bool = matches!(
        san_val,
        MacroOutputSanitization::Quarantine
            | MacroOutputSanitization::SanitizeTagged(_)
            | MacroOutputSanitization::SanitizeWithNotice { .. }
    );
    (san_tokens, sanitize_bool, quarantine_bool)
}

pub(crate) fn resolve_effect_and_sanitization_tokens(
    attr: Option<&ToolAttr>,
    crate_path: &proc_macro2::TokenStream,
) -> (
    proc_macro2::TokenStream,
    bool,
    proc_macro2::TokenStream,
    bool,
    bool,
) {
    let effect_val = attr.map_or(MacroToolEffect::Mutating, |a| a.effect);
    let effect_tokens = match effect_val {
        MacroToolEffect::ReadOnly => quote!(#crate_path::ToolEffect::ReadOnly),
        MacroToolEffect::Destructive => quote!(#crate_path::ToolEffect::Destructive),
        MacroToolEffect::Mutating => quote!(#crate_path::ToolEffect::Mutating),
    };
    let idempotent_bool = attr.is_some_and(|a| a.idempotent);
    let (san_tokens, sanitize_bool, quarantine_bool) =
        resolve_sanitization_tokens(attr, crate_path);
    (
        effect_tokens,
        idempotent_bool,
        san_tokens,
        sanitize_bool,
        quarantine_bool,
    )
}

pub(crate) fn build_tool_call_methods(
    is_async: bool,
    crate_path: &proc_macro2::TokenStream,
    inner: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if is_async {
        quote! {
            #[doc(hidden)]
            async fn __call_raw(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> {
                #inner
                __raw_res
            }

            async fn call(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> {
                let __res = self.__call_raw(params, _ctx).await;
                #crate_path::RustTool::output_sanitization(self).apply_result(__res)
            }
        }
    } else {
        quote! {
            #[doc(hidden)]
            fn __call_raw(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::future::Ready<::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError>> {
                ::core::future::ready((|| -> ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> {
                    #inner
                    __raw_res
                })())
            }

            fn call(&self, params: Self::Params, _ctx: &#crate_path::ToolContext) -> ::core::future::Ready<::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError>> {
                let __res = self.__call_raw(params, _ctx).into_inner();
                ::core::future::ready(#crate_path::RustTool::output_sanitization(self).apply_result(__res))
            }
        }
    }
}

pub(crate) fn extract_ok_type<'a>(
    sig: &'a syn::Signature,
    return_info: &'a ReturnInfo,
) -> Option<&'a syn::Type> {
    match return_info {
        ReturnInfo::ResultType { ok_type, .. } => Some(ok_type.as_ref()),
        ReturnInfo::BareType => match &sig.output {
            syn::ReturnType::Type(_, ty) => Some(ty.as_ref()),
            syn::ReturnType::Default => None,
        },
    }
}

fn tool_impl(func: &ItemFn, attr: Option<&ToolAttr>) -> syn::Result<proc_macro2::TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    reject_generic_signature(&func.sig, MACRO_LLM_TOOL)?;
    let fn_unraw = fn_name.unraw().to_string();
    let tool_name_str = attr
        .and_then(|a| a.name_override.as_ref())
        .map_or_else(|| fn_unraw.clone(), syn::LitStr::value);
    let struct_name = format_ident!("{}", fn_unraw.to_case(Case::Pascal));
    let params_name = format_ident!("{}Params", struct_name);

    let DescriptionInfo {
        static_description,
        helper_tokens,
        description_method,
        dep_tracking,
    } = resolve_description(func, attr)?;

    let all_params = extract_params(func, MACRO_LLM_TOOL)?;
    validate_params(&all_params, &func.sig)?;
    let ctx_param = all_params.iter().find(|p| p.is_context());
    let schema_params: Vec<&ParamInfo> =
        all_params.iter().filter(|p| p.is_schema_input()).collect();
    let return_info = parse_return_type(func, MACRO_LLM_TOOL)?;
    let ok_type = extract_ok_type(&func.sig, &return_info);
    let response_info = resolve_response_template(attr, &struct_name, fn_name, ok_type)?;

    let param_names: Vec<_> = schema_params.iter().map(|p| &p.name).collect();
    let param_descriptions: Vec<String> = schema_params
        .iter()
        .map(|p| extract_doc_string(&p.doc_attrs))
        .collect();

    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(&schema_params);
    let (field_attrs, default_helper_fns) =
        build_field_attrs_and_default_helpers(&fn_unraw, &schema_params, &param_struct_types);
    let mut_tokens = destructure_mut_tokens(&schema_params);
    let ext_extractions = build_extension_extractions(&all_params, &crate_path);
    let body_tokens = build_body_tokens(func, &return_info, &crate_path, &response_info);

    let vis = &func.vis;
    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] tool.");
    let struct_doc = format!(
        "Auto-generated tool struct. See the `#[llm_tool]`-annotated function `{fn_unraw}` for the implementation."
    );

    let ctx_binding = ctx_param.map_or_else(
        || quote! {},
        |cp| {
            let ctx_name = &cp.name;
            quote! { let #ctx_name = _ctx; }
        },
    );

    let response_dep_tracking = &response_info.dep_tracking;
    let response_helper_tokens = &response_info.helper_tokens;
    let (effect_tokens, idempotent_bool, san_tokens, sanitize_bool, quarantine_bool) =
        resolve_effect_and_sanitization_tokens(attr, &crate_path);
    let provenance_tokens = attr
        .map_or_else(helpers::MacroToolProvenance::default, |a| a.provenance)
        .to_tokens(&crate_path);

    let call_inner = quote! {
        use #crate_path::__private::SerializeFallback as _;
        let #params_name { #( #mut_tokens #param_names, )* } = params;
        #( #borrow_bindings )*
        #ext_extractions
        #ctx_binding
        let __raw_res: ::core::result::Result<#crate_path::ToolOutput, #crate_path::ToolError> = {
            #body_tokens
        };
    };
    let call_methods =
        build_tool_call_methods(func.sig.asyncness.is_some(), &crate_path, &call_inner);

    Ok(quote! {
        #dep_tracking
        #response_dep_tracking
        #helper_tokens
        #response_helper_tokens
        #default_helper_fns

        #[doc = #params_doc]
        #[derive(::serde::Deserialize, ::schemars::JsonSchema)]
        #vis struct #params_name {
            #(
                #[doc = #param_descriptions]
                #[schemars(description = #param_descriptions)]
                #field_attrs
                pub #param_names: #param_struct_types,
            )*
        }

        #[doc = #struct_doc]
        #vis struct #struct_name;

        impl #crate_path::RustTool for #struct_name {
            type Params = #params_name;
            const NAME: &'static str = #tool_name_str;
            const DESCRIPTION: &'static str = #static_description;
            const IDEMPOTENT: bool = #idempotent_bool;
            const EFFECT: #crate_path::ToolEffect = #effect_tokens;
            const PROVENANCE: #crate_path::ToolProvenance = #provenance_tokens;
            const SANITIZE_OUTPUT: bool = #sanitize_bool;
            const QUARANTINE_OUTPUT: bool = #quarantine_bool;
            const OUTPUT_SANITIZATION: #crate_path::OutputSanitization = #san_tokens;

            #description_method

            #call_methods
        }
    })
}
