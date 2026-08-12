use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

use crate::{
    DescriptionInfo, MACRO_LLM_PROMPT, ParamInfo, ReturnInfo, ToolAttr,
    build_param_types_and_borrows, build_serde_defaults, extract_doc_string, extract_params,
    parse_return_type, reject_generic_signature, resolve_description,
};

/// Build the body tokens for a prompt's `render` method.
///
/// Wraps the user's function body in the appropriate async/sync wrapper and
/// converts the return value via `Wrap(__v).__convert_prompt()`.
fn build_prompt_body_tokens(
    func: &ItemFn,
    return_info: &ReturnInfo,
    crate_path: &TokenStream,
) -> TokenStream {
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
            quote! {
                #inner
                match __r {
                    ::core::result::Result::Ok(__v) => #crate_path::__private::Wrap(__v).__convert_prompt(),
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
            quote! {
                #inner
                #crate_path::__private::Wrap(__v).__convert_prompt()
            }
        }
    }
}

pub fn prompt_impl(func: &ItemFn, attr: Option<&ToolAttr>) -> syn::Result<TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    reject_generic_signature(func, MACRO_LLM_PROMPT)?;
    if let Some(resp) =
        attr.and_then(|a| a.response_file_path.as_ref().or(a.response_inline.as_ref()))
    {
        return Err(syn::Error::new(
            resp.span(),
            "#[llm_prompt] does not support `response`/`response_file`; \
             response templates apply to #[llm_tool] only",
        ));
    }
    let tool_name_str = fn_name.to_string();
    let struct_name = format_ident!("{}", tool_name_str.to_case(Case::Pascal));
    let params_name = format_ident!("{}Params", struct_name);

    let DescriptionInfo {
        static_description,
        helper_tokens,
        description_method,
        dep_tracking,
    } = resolve_description(func, attr)?;

    let all_params = extract_params(func, MACRO_LLM_PROMPT)?;
    if all_params.iter().any(|p| p.is_context) {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_prompt] functions do not accept a ToolContext parameter",
        ));
    }
    let params: Vec<&ParamInfo> = all_params.iter().collect();

    for param in &params {
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

    let return_info = parse_return_type(func, MACRO_LLM_PROMPT)?;

    let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();
    let param_descriptions: Vec<String> = params
        .iter()
        .map(|p| extract_doc_string(&p.doc_attrs))
        .collect();

    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(&params);
    let serde_defaults = build_serde_defaults(&params);
    let body_tokens = build_prompt_body_tokens(func, &return_info, &crate_path);

    let vis = &func.vis;
    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] prompt.");
    let struct_doc = format!(
        "Auto-generated prompt struct. See the `#[llm_prompt]`-annotated function `{fn_name}` for the implementation."
    );

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

    Ok(quote! {
        #dep_tracking
        #helper_tokens

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

        impl #crate_path::RustPrompt for #struct_name {
            type Params = #params_name;
            const NAME: &'static str = #tool_name_str;
            const DESCRIPTION: &'static str = #static_description;

            #description_method

            async fn render(&self, params: Self::Params) -> ::core::result::Result<#crate_path::PromptOutput, #crate_path::ToolError> {
                let #params_name { #( #mut_tokens #param_names, )* } = params;
                #( #borrow_bindings )*
                #body_tokens
            }
        }
    })
}
