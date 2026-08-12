use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Ident, ItemFn, LitStr, Token,
    parse::{Parse, ParseStream},
};

use crate::{
    MACRO_LLM_RESOURCE, ParamInfo, ReturnInfo, build_param_types_and_borrows, build_serde_defaults,
    extract_doc_string, extract_params, parse_return_type, reject_generic_signature,
};

pub const ATTR_URI: &str = "uri";
pub const ATTR_URI_TEMPLATE: &str = "uri_template";
pub const ATTR_NAME: &str = "name";
pub const ATTR_MIME_TYPE: &str = "mime_type";
pub const ATTR_MIME: &str = "mime";

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum ResourceAttrKey {
    Uri,
    Name,
    Description,
    MimeType,
}

impl ResourceAttrKey {
    pub(crate) const ALL: &'static [Self] =
        &[Self::Uri, Self::Name, Self::Description, Self::MimeType];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Uri => ATTR_URI,
            Self::Name => ATTR_NAME,
            Self::Description => crate::ATTR_DESCRIPTION,
            Self::MimeType => ATTR_MIME_TYPE,
        }
    }

    pub(crate) const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Uri => &[ATTR_URI, ATTR_URI_TEMPLATE],
            Self::Name => &[ATTR_NAME],
            Self::Description => &[crate::ATTR_DESCRIPTION],
            Self::MimeType => &[ATTR_MIME_TYPE, ATTR_MIME],
        }
    }
}

impl TryFrom<&syn::Ident> for ResourceAttrKey {
    type Error = syn::Error;

    fn try_from(ident: &syn::Ident) -> Result<Self, Self::Error> {
        let s = ident.to_string();
        for &variant in Self::ALL {
            if variant.aliases().contains(&s.as_str()) {
                return Ok(variant);
            }
        }
        let keys = Self::ALL
            .iter()
            .map(|k| format!("`{}`", k.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        Err(syn::Error::new_spanned(
            ident,
            format!("unknown attribute key `{s}`, expected one of: {keys}"),
        ))
    }
}

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
            let key = ResourceAttrKey::try_from(&ident)?;
            input.parse::<Token![=]>()?;
            let val: LitStr = input.parse()?;

            match key {
                ResourceAttrKey::Uri => {
                    if uri.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    uri = Some(val);
                }
                ResourceAttrKey::Name => {
                    if name.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    name = Some(val);
                }
                ResourceAttrKey::Description => {
                    if description.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    description = Some(val);
                }
                ResourceAttrKey::MimeType => {
                    if mime_type.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    mime_type = Some(val);
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

fn build_resource_body_tokens(
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
                    ::core::result::Result::Ok(__v) => #crate_path::__private::Wrap(__v).__convert_resource(__uri, Self::MIME_TYPE),
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
                #crate_path::__private::Wrap(__v).__convert_resource(__uri, Self::MIME_TYPE)
            }
        }
    }
}

pub fn resource_impl(func: &ItemFn, attr: &ResourceAttr) -> syn::Result<TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    reject_generic_signature(func, MACRO_LLM_RESOURCE)?;
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

    let all_params = extract_params(func, MACRO_LLM_RESOURCE)?;
    if all_params.iter().any(|p| p.is_context) {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_resource] functions do not accept a ToolContext parameter",
        ));
    }
    let params: Vec<&ParamInfo> = all_params.iter().collect();
    let return_info = parse_return_type(func, MACRO_LLM_RESOURCE)?;

    let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();
    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(&params);
    let serde_defaults = build_serde_defaults(&params);
    let body_tokens = build_resource_body_tokens(func, &return_info, &crate_path);

    let vis = &func.vis;
    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] resource.");
    let struct_doc = format!(
        "Auto-generated resource struct. See the `#[llm_resource]`-annotated function `{fn_name}` for the implementation."
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

            async fn read(&self, __uri: &str, __params: Self::Params) -> ::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError> {
                let #params_name { #( #mut_tokens #param_names, )* } = __params;
                #( #borrow_bindings )*
                #body_tokens
            }
        }
    })
}
