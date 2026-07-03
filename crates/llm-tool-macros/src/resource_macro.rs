use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Ident, ItemFn, LitStr, Token,
    parse::{Parse, ParseStream},
};

use crate::{
    ParamInfo, ReturnInfo, build_param_types_and_borrows, build_serde_defaults, extract_doc_string,
    extract_params, parse_return_type,
};

pub struct ResourceAttr {
    pub uri: LitStr,
    pub name: Option<LitStr>,
    pub description: Option<LitStr>,
    pub mime_type: Option<LitStr>,
}

impl Parse for ResourceAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut uri = None;
        let mut name = None;
        let mut description = None;
        let mut mime_type = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let val: LitStr = input.parse()?;

            match ident.to_string().as_str() {
                "uri" | "uri_template" => uri = Some(val),
                "name" => name = Some(val),
                "description" => description = Some(val),
                "mime_type" | "mime" => mime_type = Some(val),
                other => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!("unknown attribute key `{other}`"),
                    ));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let uri = uri.ok_or_else(|| {
            syn::Error::new(
                input.span(),
                "`uri` is required in #[llm_resource(uri = \"...\")]",
            )
        })?;
        Ok(Self {
            uri,
            name,
            description,
            mime_type,
        })
    }
}

pub fn resource_impl(func: &ItemFn, attr: &ResourceAttr) -> syn::Result<TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    let tool_name_str = attr
        .name
        .as_ref()
        .map_or_else(|| fn_name.to_string(), syn::LitStr::value);
    let struct_name = format_ident!("{}", fn_name.to_string().to_case(Case::Pascal));
    let params_name = format_ident!("{}Params", struct_name);

    let description = attr
        .description
        .as_ref()
        .map_or_else(|| extract_doc_string(&func.attrs), syn::LitStr::value);

    let uri_str = attr.uri.value();
    let mime_expr = if let Some(m) = &attr.mime_type {
        let val = m.value();
        quote! { ::core::option::Option::Some(#val) }
    } else {
        quote! { ::core::option::Option::None }
    };

    let all_params = extract_params(func)?;
    let params: Vec<&ParamInfo> = all_params.iter().filter(|p| !p.is_context).collect();
    let return_info = parse_return_type(func)?;

    let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();
    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(&params);
    let serde_defaults = build_serde_defaults(&params);

    let is_async = func.sig.asyncness.is_some();
    let body_stmts = &func.block.stmts;

    let body_tokens = match return_info {
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
                    ::core::result::Result::Ok(__v) => match #crate_path::__private::Wrap(__v).__convert_resource(uri, Self::MIME_TYPE) {
                        ::core::result::Result::Ok(__out) => ::core::result::Result::Ok(__out),
                        ::core::result::Result::Err(__e) => ::core::result::Result::Err(__e),
                    },
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
                match #crate_path::__private::Wrap(__v).__convert_resource(uri, Self::MIME_TYPE) {
                    ::core::result::Result::Ok(__out) => ::core::result::Result::Ok(__out),
                    ::core::result::Result::Err(__e) => ::core::result::Result::Err(__e),
                }
            }
        }
    };

    let vis = &func.vis;
    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] resource.");
    let struct_doc = format!(
        "Auto-generated resource struct. See the `#[llm_resource]`-annotated function `{fn_name}` for the implementation."
    );

    Ok(quote! {
        #[doc = #params_doc]
        #[derive(::serde::Deserialize)]
        #vis struct #params_name {
            #(
                #serde_defaults
                pub #param_names: #param_struct_types,
            )*
        }

        #[doc = #struct_doc]
        #vis struct #struct_name;

        impl #crate_path::RustResource for #struct_name {
            type Params = #params_name;
            const URI_TEMPLATE: &'static str = #uri_str;
            const NAME: &'static str = #tool_name_str;
            const DESCRIPTION: &'static str = #description;
            const MIME_TYPE: ::core::option::Option<&'static str> = #mime_expr;

            async fn read(&self, uri: &str, params: Self::Params) -> ::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError> {
                let #params_name { #( #param_names, )* } = params;
                #( #borrow_bindings )*
                #body_tokens
            }
        }
    })
}
