use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, ext::IdentExt as _};

use crate::{
    DescriptionInfo, MACRO_LLM_PROMPT, ParamInfo, ReturnInfo, ToolAttr,
    build_extension_extractions, build_field_attrs_and_default_helpers,
    build_param_types_and_borrows, destructure_mut_tokens, extract_doc_string, extract_params,
    helpers, parse_return_type, reject_generic_signature, resolve_description,
    resolve_sanitization_tokens,
};

fn build_prompt_body_tokens(
    func: &ItemFn,
    return_info: &ReturnInfo,
    crate_path: &TokenStream,
) -> TokenStream {
    let ok_expr = quote! { #crate_path::__private::Wrap(__v).__convert_prompt() };
    crate::helpers::build_wrapped_body(func, return_info, &ok_expr)
}

fn build_prompt_render_methods(
    is_async: bool,
    crate_path: &TokenStream,
    inner: &TokenStream,
) -> TokenStream {
    if is_async {
        quote! {
            #[doc(hidden)]
            async fn __render_raw(
                &self,
                params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError> {
                #inner
            }

            async fn render_with_context(
                &self,
                params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError> {
                let __res = self.__render_raw(params, _ctx).await;
                let __pol = #crate_path::RustPrompt::output_sanitization(self);
                match __res {
                    ::core::result::Result::Ok(out) => ::core::result::Result::Ok(__pol.apply_prompt(out)),
                    ::core::result::Result::Err(err) => ::core::result::Result::Err(__pol.apply_error(err)),
                }
            }

            async fn render(
                &self,
                params: Self::Params,
            ) -> ::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError> {
                let __ctx = #crate_path::ToolContext::new();
                self.render_with_context(params, &__ctx).await
            }
        }
    } else {
        quote! {
            #[doc(hidden)]
            fn __render_raw(
                &self,
                params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::future::Ready<::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError>> {
                ::core::future::ready((|| -> ::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError> {
                    #inner
                })())
            }

            fn render_with_context(
                &self,
                params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::future::Ready<::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError>> {
                let __res = self.__render_raw(params, _ctx).into_inner();
                let __pol = #crate_path::RustPrompt::output_sanitization(self);
                let __final = match __res {
                    ::core::result::Result::Ok(out) => ::core::result::Result::Ok(__pol.apply_prompt(out)),
                    ::core::result::Result::Err(err) => ::core::result::Result::Err(__pol.apply_error(err)),
                };
                ::core::future::ready(__final)
            }

            fn render(
                &self,
                params: Self::Params,
            ) -> ::core::future::Ready<::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError>> {
                let __ctx = #crate_path::ToolContext::new();
                self.render_with_context(params, &__ctx)
            }
        }
    }
}

fn validate_prompt_signature_and_attr(
    func: &ItemFn,
    attr: Option<&ToolAttr>,
    all_params: &[ParamInfo],
    schema_params: &[&ParamInfo],
) -> syn::Result<()> {
    if let Some(resp) =
        attr.and_then(|a| a.response_file_path.as_ref().or(a.response_inline.as_ref()))
    {
        return Err(syn::Error::new(
            resp.span(),
            "#[llm_prompt] does not support `response`/`response_file`; \
             response templates apply to #[llm_tool] only",
        ));
    }
    if let Some(eff_span) = attr.and_then(|a| a.explicit_effect_span) {
        return Err(syn::Error::new(
            eff_span,
            "#[llm_prompt] does not support `effect` or `idempotent`; \
             execution effects apply to #[llm_tool] only",
        ));
    }
    if all_params.iter().any(ParamInfo::is_context) {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_prompt] functions do not accept a ToolContext parameter (use `#[llm_tool(ext)]` for typed extension state)",
        ));
    }
    for param in schema_params {
        if param.doc_attrs.is_empty() {
            return Err(syn::Error::new_spanned(
                &param.name,
                format!(
                    "#[llm_prompt] parameter `{}` must have a doc comment \
                      (used as the parameter description in the JSON schema)",
                    param.name
                ),
            ));
        }
    }
    Ok(())
}

pub fn prompt_impl(func: &ItemFn, attr: Option<&ToolAttr>) -> syn::Result<TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    reject_generic_signature(&func.sig, MACRO_LLM_PROMPT)?;

    let fn_unraw = fn_name.unraw().to_string();
    let prompt_name_str = attr
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

    let all_params = extract_params(func, MACRO_LLM_PROMPT)?;
    let schema_params: Vec<&ParamInfo> =
        all_params.iter().filter(|p| p.is_schema_input()).collect();
    validate_prompt_signature_and_attr(func, attr, &all_params, &schema_params)?;

    let return_info = parse_return_type(func, MACRO_LLM_PROMPT)?;
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
    let body_tokens = build_prompt_body_tokens(func, &return_info, &crate_path);

    let vis = &func.vis;
    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] prompt.");
    let struct_doc = format!(
        "Auto-generated prompt struct. See the `#[llm_prompt]`-annotated function `{fn_unraw}` for the implementation."
    );

    let (san_tokens, sanitize_bool, quarantine_bool) =
        resolve_sanitization_tokens(attr, &crate_path);
    let provenance_tokens = attr
        .map_or_else(helpers::MacroToolProvenance::default, |a| a.provenance)
        .to_tokens(&crate_path);

    let render_inner = quote! {
        let #params_name { #( #mut_tokens #param_names, )* } = params;
        #( #borrow_bindings )*
        #ext_extractions
        #body_tokens
    };
    let render_methods =
        build_prompt_render_methods(func.sig.asyncness.is_some(), &crate_path, &render_inner);

    Ok(quote! {
        #dep_tracking
        #helper_tokens
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

        impl #crate_path::RustPrompt for #struct_name {
            type Params = #params_name;
            const NAME: &'static str = #prompt_name_str;
            const DESCRIPTION: &'static str = #static_description;
            const PROVENANCE: #crate_path::ToolProvenance = #provenance_tokens;
            const SANITIZE_OUTPUT: bool = #sanitize_bool;
            const QUARANTINE_OUTPUT: bool = #quarantine_bool;
            const OUTPUT_SANITIZATION: #crate_path::OutputSanitization = #san_tokens;

            #description_method

            #render_methods
        }
    })
}

#[cfg(test)]
mod tests {
    use syn::{parse_quote, parse_str};

    use super::*;

    #[test]
    fn prompt_impl_rejects_effect_and_idempotent() {
        let f: ItemFn = parse_quote! {
            /// A prompt.
            fn p(
                /// Arg.
                x: String,
            ) -> String { x }
        };
        let eff_attr = parse_str::<ToolAttr>(r#"effect = "read_only""#).unwrap();
        let err = prompt_impl(&f, Some(&eff_attr)).unwrap_err();
        assert!(err.to_string().contains("does not support `effect`"));

        let idemp_attr = parse_str::<ToolAttr>("idempotent").unwrap();
        let err2 = prompt_impl(&f, Some(&idemp_attr)).unwrap_err();
        assert!(
            err2.to_string()
                .contains("does not support `effect` or `idempotent`")
        );
    }
}
