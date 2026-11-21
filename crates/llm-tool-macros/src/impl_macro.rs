//! Stateful `#[llm_tool]` expansion on `impl MyType { #[llm_tool(...)] pub async fn foo(&self, ...) }` blocks.

use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, ImplItem, ImplItemFn, ItemImpl, Meta, Type, ext::IdentExt as _};

use crate::{
    DescriptionInfo, MACRO_LLM_TOOL, ParamInfo, ReturnInfo, ToolAttr, build_extension_extractions,
    build_field_attrs_and_default_helpers, build_param_types_and_borrows, build_tool_call_methods,
    destructure_mut_tokens, extract_doc_string, extract_ok_type, helpers, params,
    parse_sig_return_type, reject_generic_signature, resolve_description_parts,
    resolve_effect_and_sanitization_tokens, resolve_response_template, validate_params,
};

fn extract_self_type_pascal_name(self_ty: &Type) -> syn::Result<String> {
    let Type::Path(tp) = self_ty else {
        return Err(syn::Error::new_spanned(
            self_ty,
            "#[llm_tool] `impl` block target must be a concrete named struct/enum type",
        ));
    };
    let Some(last) = tp.path.segments.last() else {
        return Err(syn::Error::new_spanned(
            self_ty,
            "#[llm_tool] `impl` block target path cannot be empty",
        ));
    };
    if !last.arguments.is_none() {
        return Err(syn::Error::new_spanned(
            self_ty,
            "#[llm_tool] `impl` blocks do not support generic type parameters",
        ));
    }
    Ok(last.ident.unraw().to_string().to_case(Case::Pascal))
}

enum MethodToolAttr {
    NotTool,
    Bare,
    Configured(Box<ToolAttr>),
}

impl MethodToolAttr {
    fn as_tool_attr(&self) -> Option<&ToolAttr> {
        match self {
            Self::NotTool | Self::Bare => None,
            Self::Configured(attr) => Some(attr),
        }
    }
}

fn take_method_llm_tool_attr(method: &mut ImplItemFn) -> syn::Result<MethodToolAttr> {
    let mut found = MethodToolAttr::NotTool;
    let mut retained_attrs = Vec::with_capacity(method.attrs.len());

    for attr in method.attrs.drain(..) {
        if params::is_llm_tool_path(attr.path()) {
            if !matches!(found, MethodToolAttr::NotTool) {
                return Err(syn::Error::new_spanned(
                    attr,
                    "duplicate `#[llm_tool]` attribute on method",
                ));
            }
            found = match &attr.meta {
                Meta::Path(_) => MethodToolAttr::Bare,
                Meta::List(list) => MethodToolAttr::Configured(Box::new(syn::parse2::<ToolAttr>(
                    list.tokens.clone(),
                )?)),
                Meta::NameValue(nv) => {
                    return Err(syn::Error::new_spanned(
                        nv,
                        "expected `#[llm_tool]` or `#[llm_tool(...)]`",
                    ));
                }
            };
        } else {
            retained_attrs.push(attr);
        }
    }

    method.attrs = retained_attrs;
    Ok(found)
}

fn strip_method_param_macro_attrs(method: &mut ImplItemFn) {
    for input in &mut method.sig.inputs {
        if let FnArg::Typed(pat_ty) = input {
            pat_ty
                .attrs
                .retain(|a| !params::is_stripped_impl_param_attr(a));
        }
    }
}

fn build_method_call_body(
    method_ident: &syn::Ident,
    is_async: bool,
    call_args: &[TokenStream],
    return_info: &ReturnInfo,
    ok_expr: &TokenStream,
) -> TokenStream {
    let invoke_expr = if is_async {
        quote! { self.0.#method_ident(#( #call_args ),*).await }
    } else {
        quote! { self.0.#method_ident(#( #call_args ),*) }
    };

    match return_info {
        ReturnInfo::ResultType { ok_type, err_type } => quote! {
            use ::llm_tool::__private::DisplayErrorFallback as _;
            let __r: ::core::result::Result<#ok_type, #err_type> = #invoke_expr;
            match __r {
                ::core::result::Result::Ok(__v) => { #ok_expr },
                ::core::result::Result::Err(err) => ::core::result::Result::Err(
                    ::llm_tool::__private::WrapErr(err).__convert_err()
                ),
            }
        },
        ReturnInfo::BareType => quote! {
            let __v = #invoke_expr;
            #ok_expr
        },
    }
}

struct MethodCallBuildArgs<'a> {
    method: &'a ImplItemFn,
    helper_prefix: &'a str,
    params_name: &'a syn::Ident,
    all_params: &'a [ParamInfo],
    schema_params: &'a [&'a ParamInfo],
    return_info: &'a ReturnInfo,
    response_info: &'a helpers::ResponseTemplateInfo,
    crate_path: &'a TokenStream,
}

fn build_method_call_methods_and_helpers(
    args: &MethodCallBuildArgs<'_>,
) -> (Vec<TokenStream>, Vec<TokenStream>, TokenStream, TokenStream) {
    let MethodCallBuildArgs {
        method,
        helper_prefix,
        params_name,
        all_params,
        schema_params,
        return_info,
        response_info,
        crate_path,
    } = *args;
    let param_names: Vec<_> = schema_params.iter().map(|p| &p.name).collect();
    let (param_struct_types, borrow_bindings) = build_param_types_and_borrows(schema_params);
    let (field_attrs, default_helper_fns) =
        build_field_attrs_and_default_helpers(helper_prefix, schema_params, &param_struct_types);
    let mut_tokens = destructure_mut_tokens(schema_params);
    let ext_extractions = build_extension_extractions(all_params, crate_path);
    let ctx_binding = all_params.iter().find(|p| p.is_context()).map_or_else(
        || quote! {},
        |cp| {
            let ctx_name = &cp.name;
            quote! { let #ctx_name = _ctx; }
        },
    );
    let call_args: Vec<TokenStream> = all_params
        .iter()
        .map(|p| {
            let name = &p.name;
            quote! { #name }
        })
        .collect();
    let ok_expr = helpers::build_ok_branch(crate_path, response_info);
    let body_tokens = build_method_call_body(
        &method.sig.ident,
        method.sig.asyncness.is_some(),
        &call_args,
        return_info,
        &ok_expr,
    );
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
        build_tool_call_methods(method.sig.asyncness.is_some(), crate_path, &call_inner);
    (
        param_struct_types,
        field_attrs,
        default_helper_fns,
        call_methods,
    )
}

fn expand_single_method_tool(
    self_ty: &Type,
    type_pascal: &str,
    method: &ImplItemFn,
    attr: Option<&ToolAttr>,
) -> syn::Result<(syn::Ident, TokenStream)> {
    let crate_path = quote! { ::llm_tool };
    reject_generic_signature(&method.sig, MACRO_LLM_TOOL)?;

    let method_unraw = method.sig.ident.unraw().to_string();
    let method_pascal = method_unraw.to_case(Case::Pascal);
    let tool_name_str = attr
        .and_then(|a| a.name_override.as_ref())
        .map_or_else(|| method_unraw.clone(), syn::LitStr::value);

    let struct_name = format_ident!("{}{}Tool", type_pascal, method_pascal);
    let params_name = format_ident!("{}{}Params", type_pascal, method_pascal);
    let helper_prefix = format!("{}_{}", type_pascal.to_case(Case::Snake), method_unraw);
    let helper_fn_ident = format_ident!("{}", helper_prefix);

    let DescriptionInfo {
        static_description,
        helper_tokens,
        description_method,
        dep_tracking,
    } = resolve_description_parts(&method.attrs, &helper_fn_ident, attr)?;

    let all_params = params::extract_fn_inputs(&method.sig.inputs, MACRO_LLM_TOOL, true)?;
    validate_params(&all_params, &method.sig)?;

    let schema_params: Vec<&ParamInfo> =
        all_params.iter().filter(|p| p.is_schema_input()).collect();
    let return_info = parse_sig_return_type(&method.sig, MACRO_LLM_TOOL)?;
    let ok_type = extract_ok_type(&method.sig, &return_info);
    let response_info = resolve_response_template(attr, &struct_name, &helper_fn_ident, ok_type)?;

    let param_names: Vec<_> = schema_params.iter().map(|p| &p.name).collect();
    let param_descriptions: Vec<String> = schema_params
        .iter()
        .map(|p| extract_doc_string(&p.doc_attrs))
        .collect();
    let (param_struct_types, field_attrs, default_helper_fns, call_methods) =
        build_method_call_methods_and_helpers(&MethodCallBuildArgs {
            method,
            helper_prefix: &helper_prefix,
            params_name: &params_name,
            all_params: &all_params,
            schema_params: &schema_params,
            return_info: &return_info,
            response_info: &response_info,
            crate_path: &crate_path,
        });

    let vis = &method.vis;
    let params_doc = format!("Auto-generated parameters for [`{struct_name}`].");
    let struct_doc =
        format!("Auto-generated stateful tool adapter invoking `{type_pascal}::{method_unraw}`.");
    let inner_field_doc = format!("Wrapped `{type_pascal}` service handle.");
    let response_dep_tracking = &response_info.dep_tracking;
    let response_helper_tokens = &response_info.helper_tokens;
    let (effect_tokens, idempotent_bool, san_tokens, sanitize_bool, quarantine_bool) =
        resolve_effect_and_sanitization_tokens(attr, &crate_path);
    let provenance_tokens = attr
        .map_or_else(helpers::MacroToolProvenance::default, |a| a.provenance)
        .to_tokens(&crate_path);

    let generated = quote! {
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
        #[derive(::core::clone::Clone)]
        #vis struct #struct_name(
            #[doc = #inner_field_doc]
            pub #self_ty
        );

        impl #crate_path::RustTool for #struct_name
        where
            #self_ty: ::core::marker::Send + ::core::marker::Sync,
        {
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
    };

    Ok((struct_name, generated))
}

pub fn tool_impl_block(mut item_impl: ItemImpl) -> syn::Result<TokenStream> {
    if let Some((_, ref trait_path, _)) = item_impl.trait_ {
        return Err(syn::Error::new_spanned(
            trait_path,
            "#[llm_tool] can only be applied to inherent `impl MyType { ... }` blocks, not trait impls",
        ));
    }
    if !item_impl.generics.params.is_empty() || item_impl.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &item_impl.generics,
            "#[llm_tool] `impl` blocks do not support generic parameters or `where` clauses; use concrete types",
        ));
    }

    let type_pascal = extract_self_type_pascal_name(&item_impl.self_ty)?;
    let self_ty = (*item_impl.self_ty).clone();

    let mut tool_struct_names = Vec::new();
    let mut generated_items = Vec::new();

    for item in &mut item_impl.items {
        if let ImplItem::Fn(method) = item {
            let parsed_attr = take_method_llm_tool_attr(method)?;
            if !matches!(parsed_attr, MethodToolAttr::NotTool) {
                let (struct_name, tokens) = expand_single_method_tool(
                    &self_ty,
                    &type_pascal,
                    method,
                    parsed_attr.as_tool_attr(),
                )?;
                strip_method_param_macro_attrs(method);
                tool_struct_names.push(struct_name);
                generated_items.push(tokens);
            }
        }
    }

    if tool_struct_names.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_impl.self_ty,
            "#[llm_tool] `impl` block contains no `#[llm_tool]`-annotated methods",
        ));
    }

    let register_helpers: ImplItem = syn::parse_quote! {
        /// Register all `#[llm_tool]` methods of this service into `registry`.
        pub fn register_tools(&self, registry: &mut ::llm_tool::ToolRegistry)
        where
            Self: ::core::clone::Clone + ::core::marker::Send + ::core::marker::Sync + 'static,
        {
            #(
                registry.register(#tool_struct_names(::core::clone::Clone::clone(self)));
            )*
        }
    };
    let registry_builder_helper: ImplItem = syn::parse_quote! {
        /// Build a new [`ToolRegistry`](::llm_tool::ToolRegistry) populated with all `#[llm_tool]` methods of this service.
        #[must_use]
        pub fn tool_registry(&self) -> ::llm_tool::ToolRegistry
        where
            Self: ::core::clone::Clone + ::core::marker::Send + ::core::marker::Sync + 'static,
        {
            let mut registry = ::llm_tool::ToolRegistry::new();
            self.register_tools(&mut registry);
            registry
        }
    };
    item_impl.items.push(register_helpers);
    item_impl.items.push(registry_builder_helper);

    Ok(quote! {
        #item_impl
        #( #generated_items )*
    })
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;

    #[test]
    fn impl_block_rejects_mut_self_and_trait_impls() {
        let mut_self: ItemImpl = parse_quote! {
            impl Svc {
                /// Doc.
                #[llm_tool]
                fn bad(&mut self) -> String { String::new() }
            }
        };
        let err = tool_impl_block(mut_self).unwrap_err();
        assert!(err.to_string().contains("shared `&self` receiver"));

        let trait_impl: ItemImpl = parse_quote! {
            impl MyTrait for Svc {
                /// Doc.
                #[llm_tool]
                fn bad(&self) -> String { String::new() }
            }
        };
        let err2 = tool_impl_block(trait_impl).unwrap_err();
        assert!(err2.to_string().contains("not trait impls"));
    }
}
