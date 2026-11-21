use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Ident, ItemFn, LitStr, Token,
    ext::IdentExt as _,
    parse::{Parse, ParseStream},
};

use crate::{
    ATTR_EFFECT, ATTR_IDEMPOTENT, MACRO_LLM_RESOURCE, ParamInfo, ReturnInfo,
    build_extension_extractions, build_field_attrs_and_default_helpers,
    build_param_types_and_borrows, destructure_mut_tokens, extract_params, parse_return_type,
    reject_generic_signature, resolve_sanitization_tokens,
};

pub const ATTR_URI: &str = "uri";
pub const ATTR_URI_TEMPLATE: &str = "uri_template";
pub const ATTR_MIME_TYPE: &str = "mime_type";
pub const ATTR_MIME: &str = "mime";

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ResourceAttrKey {
    Uri,
    Name,
    Description,
    DescriptionFile,
    MimeType,
    Params,
    Env,
    Context,
    Provenance,
    Sanitize,
    Quarantine,
    Sanitization,
}

impl ResourceAttrKey {
    pub const ALL: &'static [Self] = &[
        Self::Uri,
        Self::Name,
        Self::Description,
        Self::DescriptionFile,
        Self::MimeType,
        Self::Params,
        Self::Env,
        Self::Context,
        Self::Provenance,
        Self::Sanitize,
        Self::Quarantine,
        Self::Sanitization,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uri => ATTR_URI,
            Self::Name => crate::ATTR_NAME,
            Self::Description => crate::ATTR_DESCRIPTION,
            Self::DescriptionFile => crate::ATTR_DESCRIPTION_FILE,
            Self::MimeType => ATTR_MIME_TYPE,
            Self::Params => crate::ATTR_PARAMS,
            Self::Env => crate::ATTR_ENV,
            Self::Context => crate::ATTR_CONTEXT,
            Self::Provenance => crate::ATTR_PROVENANCE,
            Self::Sanitize => crate::ATTR_SANITIZE,
            Self::Quarantine => crate::ATTR_QUARANTINE,
            Self::Sanitization => crate::ATTR_SANITIZATION,
        }
    }

    pub const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Uri => &[ATTR_URI, ATTR_URI_TEMPLATE],
            Self::Name => &[crate::ATTR_NAME],
            Self::Description => &[crate::ATTR_DESCRIPTION],
            Self::DescriptionFile => &[crate::ATTR_DESCRIPTION_FILE],
            Self::MimeType => &[ATTR_MIME_TYPE, ATTR_MIME],
            Self::Params => &[crate::ATTR_PARAMS],
            Self::Env => &[crate::ATTR_ENV],
            Self::Context => &[crate::ATTR_CONTEXT],
            Self::Provenance => &[crate::ATTR_PROVENANCE],
            Self::Sanitize => &[crate::ATTR_SANITIZE],
            Self::Quarantine => &[crate::ATTR_QUARANTINE],
            Self::Sanitization => &[crate::ATTR_SANITIZATION],
        }
    }
}

impl TryFrom<&syn::Ident> for ResourceAttrKey {
    type Error = syn::Error;

    fn try_from(ident: &syn::Ident) -> Result<Self, Self::Error> {
        let s = ident.to_string();
        if s == ATTR_EFFECT || s == ATTR_IDEMPOTENT {
            return Err(syn::Error::new_spanned(
                ident,
                "#[llm_resource] does not support `effect` or `idempotent`; \
                 execution effects apply to #[llm_tool] only",
            ));
        }
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
    pub mime_type: Option<LitStr>,
    pub tool_attr: crate::ToolAttr,
}

impl Parse for ResourceAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut uri = None;
        let mut name = None;
        let mut mime_type = None;
        let mut tool_builder = crate::ToolAttrBuilder::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let key = ResourceAttrKey::try_from(&ident)?;

            match key {
                ResourceAttrKey::Uri => {
                    input.parse::<Token![=]>()?;
                    let val: LitStr = input.parse()?;
                    if uri.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    uri = Some(val);
                }
                ResourceAttrKey::Name => {
                    input.parse::<Token![=]>()?;
                    let val: LitStr = input.parse()?;
                    if name.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    name = Some(val);
                }
                ResourceAttrKey::MimeType => {
                    input.parse::<Token![=]>()?;
                    let val: LitStr = input.parse()?;
                    if mime_type.is_some() {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!("duplicate `{}` attribute", key.as_str()),
                        ));
                    }
                    mime_type = Some(val);
                }
                ResourceAttrKey::Description => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Description, input)?;
                }
                ResourceAttrKey::DescriptionFile => {
                    tool_builder.parse_by_key(
                        &ident,
                        crate::ToolAttrKey::DescriptionFile,
                        input,
                    )?;
                }
                ResourceAttrKey::Params => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Params, input)?;
                }
                ResourceAttrKey::Env => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Env, input)?;
                }
                ResourceAttrKey::Context => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Context, input)?;
                }
                ResourceAttrKey::Provenance => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Provenance, input)?;
                }
                ResourceAttrKey::Sanitize => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Sanitize, input)?;
                }
                ResourceAttrKey::Quarantine => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Quarantine, input)?;
                }
                ResourceAttrKey::Sanitization => {
                    tool_builder.parse_by_key(&ident, crate::ToolAttrKey::Sanitization, input)?;
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
        let tool_attr = tool_builder.build()?;
        Ok(Self {
            uri,
            name,
            mime_type,
            tool_attr,
        })
    }
}

fn build_resource_body_tokens(
    func: &ItemFn,
    return_info: &ReturnInfo,
    crate_path: &TokenStream,
) -> TokenStream {
    let ok_expr =
        quote! { #crate_path::__private::Wrap(__v).__convert_resource(__uri, Self::MIME_TYPE) };
    crate::helpers::build_wrapped_body(func, return_info, &ok_expr)
}

fn build_resource_read_methods(
    is_async: bool,
    crate_path: &proc_macro2::TokenStream,
    inner: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if is_async {
        quote! {
            #[doc(hidden)]
            async fn __read_raw(
                &self,
                __uri: &str,
                __params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError> {
                #inner
            }

            async fn read_with_context(
                &self,
                __uri: &str,
                __params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError> {
                let __res = self.__read_raw(__uri, __params, _ctx).await;
                let __pol = #crate_path::RustResource::output_sanitization(self);
                match __res {
                    ::core::result::Result::Ok(out) => ::core::result::Result::Ok(__pol.apply_resource(out)),
                    ::core::result::Result::Err(err) => ::core::result::Result::Err(__pol.apply_error(err)),
                }
            }

            async fn read(
                &self,
                __uri: &str,
                __params: Self::Params,
            ) -> ::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError> {
                let __ctx = #crate_path::ToolContext::new();
                self.read_with_context(__uri, __params, &__ctx).await
            }
        }
    } else {
        quote! {
            #[doc(hidden)]
            fn __read_raw(
                &self,
                __uri: &str,
                __params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::future::Ready<::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError>> {
                ::core::future::ready((|| -> ::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError> {
                    #inner
                })())
            }

            fn read_with_context(
                &self,
                __uri: &str,
                __params: Self::Params,
                _ctx: &#crate_path::ToolContext,
            ) -> ::core::future::Ready<::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError>> {
                let __res = self.__read_raw(__uri, __params, _ctx).into_inner();
                let __pol = #crate_path::RustResource::output_sanitization(self);
                let __final = match __res {
                    ::core::result::Result::Ok(out) => ::core::result::Result::Ok(__pol.apply_resource(out)),
                    ::core::result::Result::Err(err) => ::core::result::Result::Err(__pol.apply_error(err)),
                };
                ::core::future::ready(__final)
            }

            fn read(
                &self,
                __uri: &str,
                __params: Self::Params,
            ) -> ::core::future::Ready<::core::result::Result<#crate_path::ResourceOutput, #crate_path::ToolError>> {
                let __ctx = #crate_path::ToolContext::new();
                self.read_with_context(__uri, __params, &__ctx)
            }
        }
    }
}

pub fn resource_impl(func: &ItemFn, attr: &ResourceAttr) -> syn::Result<TokenStream> {
    let crate_path = quote! { ::llm_tool };
    let fn_name = &func.sig.ident;
    reject_generic_signature(&func.sig, MACRO_LLM_RESOURCE)?;
    let fn_unraw = fn_name.unraw().to_string();
    let tool_name_str = attr
        .name
        .as_ref()
        .map_or_else(|| fn_unraw.clone(), syn::LitStr::value);
    let struct_name = format_ident!("{}", fn_unraw.to_case(Case::Pascal));
    let params_name = format_ident!("{}Params", struct_name);

    let crate::DescriptionInfo {
        static_description,
        helper_tokens,
        description_method,
        dep_tracking,
    } = crate::resolve_description(func, Some(&attr.tool_attr))?;

    let uri_str = attr.uri.value();
    let mime_expr = attr.mime_type.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |m| {
            let val = m.value();
            quote! { ::core::option::Option::Some(#val) }
        },
    );

    let all_params = extract_params(func, MACRO_LLM_RESOURCE)?;
    if all_params.iter().any(ParamInfo::is_context) {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_resource] functions do not accept a ToolContext parameter (use `#[llm_tool(ext)]` for typed extension state)",
        ));
    }
    let schema_params: Vec<&ParamInfo> =
        all_params.iter().filter(|p| p.is_schema_input()).collect();
    let return_info = parse_return_type(func, MACRO_LLM_RESOURCE)?;

    let param_names: Vec<_> = schema_params.iter().map(|p| &p.name).collect();
    let param_descriptions: Vec<String> = schema_params
        .iter()
        .map(|p| {
            let doc = crate::extract_doc_string(&p.doc_attrs);
            if doc.is_empty() {
                format!("Resource parameter `{}`.", p.name.unraw())
            } else {
                doc
            }
        })
        .collect();
    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(&schema_params);
    let (field_attrs, default_helper_fns) =
        build_field_attrs_and_default_helpers(&fn_unraw, &schema_params, &param_struct_types);
    let mut_tokens = destructure_mut_tokens(&schema_params);
    let ext_extractions = build_extension_extractions(&all_params, &crate_path);
    let body_tokens = build_resource_body_tokens(func, &return_info, &crate_path);

    let vis = &func.vis;
    let params_doc = format!("Auto-generated parameters for the [`{struct_name}`] resource.");
    let struct_doc = format!(
        "Auto-generated resource struct. See the `#[llm_resource]`-annotated function `{fn_unraw}` for the implementation."
    );

    let (san_tokens, sanitize_bool, quarantine_bool) =
        resolve_sanitization_tokens(Some(&attr.tool_attr), &crate_path);
    let provenance_tokens = attr.tool_attr.provenance.to_tokens(&crate_path);

    let read_inner = quote! {
        let #params_name { #( #mut_tokens #param_names, )* } = __params;
        #( #borrow_bindings )*
        #ext_extractions
        #body_tokens
    };
    let read_methods =
        build_resource_read_methods(func.sig.asyncness.is_some(), &crate_path, &read_inner);

    Ok(quote! {
        #dep_tracking
        #helper_tokens
        #default_helper_fns

        #[doc = #params_doc]
        #[derive(::serde::Deserialize)]
        #vis struct #params_name {
            #(
                #[doc = #param_descriptions]
                #field_attrs
                pub #param_names: #param_struct_types,
            )*
        }

        #[doc = #struct_doc]
        #vis struct #struct_name;

        impl #crate_path::RustResource for #struct_name {
            type Params = #params_name;
            const URI_TEMPLATE: &'static str = #uri_str;
            const NAME: &'static str = #tool_name_str;
            const DESCRIPTION: &'static str = #static_description;
            const MIME_TYPE: ::core::option::Option<&'static str> = #mime_expr;
            const PROVENANCE: #crate_path::ToolProvenance = #provenance_tokens;
            const SANITIZE_OUTPUT: bool = #sanitize_bool;
            const QUARANTINE_OUTPUT: bool = #quarantine_bool;
            const OUTPUT_SANITIZATION: #crate_path::OutputSanitization = #san_tokens;

            #description_method

            #read_methods
        }
    })
}

#[cfg(test)]
mod tests {
    use syn::parse_str;

    use super::*;

    #[test]
    fn resource_attr_rejects_effect_and_idempotent_with_clear_error() {
        let err = parse_str::<ResourceAttr>(r#"uri = "cfg://a", effect = "read_only""#)
            .err()
            .expect("effect must be rejected on resource");
        assert!(
            err.to_string()
                .contains("does not support `effect` or `idempotent`")
        );

        let err2 = parse_str::<ResourceAttr>(r#"uri = "cfg://a", idempotent = true"#)
            .err()
            .expect("idempotent must be rejected on resource");
        assert!(
            err2.to_string()
                .contains("does not support `effect` or `idempotent`")
        );
    }
}
