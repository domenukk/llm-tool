#[cfg(feature = "md-tmpl")]
use quote::format_ident;
use quote::quote;
#[cfg(feature = "md-tmpl")]
use syn::LitStr;
use syn::{FnArg, GenericArgument, ItemFn, Pat, PatType, PathArguments, Type};

use crate::{
    ATTR_CONTEXT, ATTR_DOC, MACRO_LLM_TOOL, ParamInfo, ReturnInfo, TYPE_OPTION, TYPE_RESULT,
    TYPE_STR, TYPE_TOOL_CONTEXT, ToolAttr,
};
#[cfg(feature = "md-tmpl")]
use crate::{desc, response_struct_gen};

/// Build the struct field types and any auto-borrow bindings for `&str` params.
pub(crate) fn build_param_types_and_borrows(
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
pub(crate) fn build_serde_defaults(params: &[&ParamInfo]) -> Vec<proc_macro2::TokenStream> {
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
pub(crate) fn build_body_tokens(
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
pub(crate) fn build_ok_branch(
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
pub(crate) struct ResponseTemplateInfo {
    /// Cargo dependency-tracking tokens.
    pub(crate) dep_tracking: proc_macro2::TokenStream,
    /// Helper tokens (e.g. static `LazyLock` declarations).
    pub(crate) helper_tokens: proc_macro2::TokenStream,
    /// Token stream that converts `__v` into `Result<ToolOutput, ToolError>`
    /// via template rendering. `None` = use default `__convert()` path.
    pub(crate) render_tokens: Option<proc_macro2::TokenStream>,
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

pub(crate) fn resolve_response_template(
    attr: Option<&ToolAttr>,
    struct_name: &syn::Ident,
    fn_name: &syn::Ident,
) -> syn::Result<ResponseTemplateInfo> {
    // NOLINT: suppress unused-variable warning in non-md-tmpl cfg branch
    let _ = (struct_name, fn_name);
    let Some(attr) = attr else {
        return Ok(ResponseTemplateInfo::default());
    };

    if let Some(response_path) = &attr.response_file_path {
        #[cfg(not(feature = "md-tmpl"))]
        {
            return Err(syn::Error::new(
                response_path.span(),
                "the `md-tmpl` feature must be enabled to use `response_file`",
            ));
        }
        #[cfg(feature = "md-tmpl")]
        {
            return resolve_response_template_file(attr, response_path, struct_name, fn_name);
        }
    }
    if let Some(response_inline) = &attr.response_inline {
        #[cfg(not(feature = "md-tmpl"))]
        {
            return Err(syn::Error::new(
                response_inline.span(),
                "the `md-tmpl` feature must be enabled to use `response`",
            ));
        }
        #[cfg(feature = "md-tmpl")]
        {
            return resolve_response_template_inline(attr, response_inline, struct_name, fn_name);
        }
    }
    Ok(ResponseTemplateInfo::default())
}

/// Feature-gated implementation of response template resolution from file.
#[cfg(feature = "md-tmpl")]
pub(crate) fn resolve_response_template_file(
    attr: &ToolAttr,
    response_path: &LitStr,
    struct_name: &syn::Ident,
    fn_name: &syn::Ident,
) -> syn::Result<ResponseTemplateInfo> {
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

    let dep_tracking = quote! {
        const _: &str = include_str!(#path_str);
    };

    let base_dir = full_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let env_values = desc::env_pairs(attr);
    let env_refs: Vec<(&str, md_tmpl::Value)> = env_values
        .iter()
        .map(|(k, v)| (k.as_str(), md_tmpl::Value::Str(v.clone())))
        .collect();
    let (fm, _) =
        md_tmpl::parse_frontmatter_with_base_dir(&source, base_dir, &env_refs).map_err(|e| {
            syn::Error::new(
                response_path.span(),
                format!("response template '{rel_path}' frontmatter error: {e}"),
            )
        })?;

    let response_struct_name_str = format!("{struct_name}Response");
    let generated_idents = response_struct_gen::collect_generated_type_names(
        &response_struct_name_str,
        &fm.declarations,
    );

    let response_struct_name = format_ident!("{}", response_struct_name_str);
    let response_mod_name = format_ident!("__{}_response_mod", fn_name);

    let env_toks = desc::env_tokens(attr);
    let helper_tokens = quote! {
        ::llm_tool::__md_tmpl_macros::include_template!(
            #response_path as #response_struct_name => #response_mod_name,
            crate = ::llm_tool::__md_tmpl
            #env_toks
        );
        pub use #response_mod_name::{ #( #generated_idents ),* };
    };

    let render_tokens = quote! {
        {
            let __rendered = #response_mod_name::template().render(&__v)
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
        helper_tokens,
        render_tokens: Some(render_tokens),
    })
}

/// Feature-gated implementation of response template resolution from inline string.
#[cfg(feature = "md-tmpl")]
pub(crate) fn resolve_response_template_inline(
    attr: &ToolAttr,
    response_inline: &LitStr,
    struct_name: &syn::Ident,
    fn_name: &syn::Ident,
) -> syn::Result<ResponseTemplateInfo> {
    let source = response_inline.value();

    // Validate the inline template parses at compile time.
    let env_values = desc::env_pairs(attr);
    let env_refs: Vec<(&str, md_tmpl::Value)> = env_values
        .iter()
        .map(|(k, v)| (k.as_str(), md_tmpl::Value::Str(v.clone())))
        .collect();
    let fm = match md_tmpl::parse_frontmatter_with_env(&source, &env_refs) {
        Ok((fm, _)) => fm,
        Err(e) => {
            return Err(syn::Error::new(
                response_inline.span(),
                format!("inline response template error: {e}"),
            ));
        }
    };

    let response_struct_name_str = format!("{struct_name}Response");
    let generated_idents = response_struct_gen::collect_generated_type_names(
        &response_struct_name_str,
        &fm.declarations,
    );

    let response_struct_name = format_ident!("{}", response_struct_name_str);
    let response_mod_name = format_ident!("__{}_response_mod", fn_name);

    let env_toks = desc::env_tokens(attr);
    let helper_tokens = quote! {
        ::llm_tool::__md_tmpl_macros::template!(
            #source as #response_struct_name => #response_mod_name,
            crate = ::llm_tool::__md_tmpl
            #env_toks
        );
        pub use #response_mod_name::{ #( #generated_idents ),* };
    };

    let render_tokens = quote! {
        {
            let __rendered = #response_mod_name::template().render(&__v)
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
        dep_tracking: quote! {},
        helper_tokens,
        render_tokens: Some(render_tokens),
    })
}

/// Check whether `ty` is `Option<T>` (or `std::option::Option<T>`).
pub(crate) fn is_option_type(ty: &syn::Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(last_seg) = type_path.path.segments.last() else {
        return false;
    };
    if last_seg.ident != TYPE_OPTION {
        return false;
    }
    matches!(&last_seg.arguments, PathArguments::AngleBracketed(args)
        if args.args.len() == 1
            && matches!(args.args.first(), Some(GenericArgument::Type(_))))
}

/// Check whether `ty` is `ToolContext`, `&ToolContext`, or a qualified path
/// ending in `ToolContext`.
pub(crate) fn is_tool_context_type(ty: &syn::Type) -> bool {
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
        .is_some_and(|seg| seg.ident == TYPE_TOOL_CONTEXT)
}

/// Check whether `ty` is `&str`.
pub(crate) fn is_str_ref(ty: &syn::Type) -> bool {
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
        .is_some_and(|seg| seg.ident == TYPE_STR && seg.arguments.is_none())
}

pub(crate) fn is_explicit_context_attr(attr: &syn::Attribute) -> syn::Result<bool> {
    if !attr.path().is_ident(MACRO_LLM_TOOL) {
        return Ok(false);
    }
    let mut is_context = false;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident(ATTR_CONTEXT) {
            is_context = true;
            Ok(())
        } else {
            Err(meta.error("unsupported llm_tool attribute"))
        }
    })?;
    Ok(is_context)
}

pub(crate) fn extract_params(func: &ItemFn, macro_name: &str) -> syn::Result<Vec<ParamInfo>> {
    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    format!("#[{macro_name}] functions must be free functions (no `self`)"),
                ));
            }
            FnArg::Typed(PatType { pat, ty, attrs, .. }) => {
                let (name, is_mut) = match pat.as_ref() {
                    Pat::Ident(ident) => (ident.ident.clone(), ident.mutability.is_some()),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            format!("#[{macro_name}] parameters must be simple identifiers"),
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
                    .filter(|a| a.path().is_ident(ATTR_DOC))
                    .cloned()
                    .collect();
                params.push(ParamInfo {
                    name,
                    ty: ty.clone(),
                    doc_attrs,
                    is_context,
                    is_mut,
                });
            }
        }
    }
    Ok(params)
}

/// Reject generic parameters, lifetimes, and `where` clauses on the annotated
/// function.
///
/// The generated `RustTool`/`RustPrompt`/`RustResource` impl is for a concrete,
/// zero-generic unit struct. A generic signature would expand to code that
/// references undeclared type/lifetime parameters, yielding confusing errors
/// pointing at generated code. Fail early with a clear, spanned message.
pub(crate) fn reject_generic_signature(func: &ItemFn, macro_name: &str) -> syn::Result<()> {
    let generics = &func.sig.generics;
    if let Some(where_clause) = &generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            format!("#[{macro_name}] does not support `where` clauses; use concrete types"),
        ));
    }
    if !generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            generics,
            format!(
                "#[{macro_name}] does not support generic parameters or lifetimes; \
                 use concrete types"
            ),
        ));
    }
    Ok(())
}

pub(crate) fn extract_doc_string(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident(ATTR_DOC) {
                return None;
            }
            if let syn::Meta::NameValue(nv) = &attr.meta
                && let syn::Expr::Lit(lit) = &nv.value
                && let syn::Lit::Str(s) = &lit.lit
            {
                return Some(s.value().replace("\r\n", "\n"));
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
pub(crate) fn parse_return_type(func: &ItemFn, macro_name: &str) -> syn::Result<ReturnInfo> {
    let syn::ReturnType::Type(_, ty) = &func.sig.output else {
        return Err(syn::Error::new_spanned(
            &func.sig,
            format!("#[{macro_name}] functions must have an explicit return type"),
        ));
    };

    // Try to parse as Result<T, E>.
    if let Some(result_types) = try_extract_result_types(ty)? {
        return Ok(result_types);
    }

    // Not a Result — treat as infallible bare type.
    Ok(ReturnInfo::BareType)
}

/// Try to extract `T` and `E` from a `Result<T, E>` return type.
/// Returns `None` if the type is not a `Result`.
pub(crate) fn try_extract_result_types(ty: &syn::Type) -> syn::Result<Option<ReturnInfo>> {
    let Type::Path(type_path) = ty else {
        return Ok(None);
    };

    let Some(last_seg) = type_path.path.segments.last() else {
        return Ok(None);
    };

    if last_seg.ident != TYPE_RESULT {
        return Ok(None);
    }

    let PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return Ok(None);
    };

    if args.args.len() == 1 {
        return Err(syn::Error::new_spanned(
            ty,
            "1-argument Result aliases (e.g. `anyhow::Result<T>`, `io::Result<T>`) are not supported directly; \
             please specify both generic arguments: `Result<T, anyhow::Error>` or `Result<T, std::io::Error>`",
        ));
    }

    if args.args.len() != 2 {
        return Ok(None);
    }

    let GenericArgument::Type(ok_type) = &args.args[0] else {
        return Ok(None);
    };

    let GenericArgument::Type(err_type) = &args.args[1] else {
        return Ok(None);
    };

    Ok(Some(ReturnInfo::ResultType {
        ok_type: Box::new(ok_type.clone()),
        err_type: Box::new(err_type.clone()),
    }))
}

#[cfg(test)]
mod tests {
    use syn::{parse_quote, parse_str};

    use super::*;

    fn ty(s: &str) -> Type {
        parse_str::<Type>(s).expect("valid type")
    }

    #[test]
    fn is_option_type_matches_option_variants() {
        assert!(is_option_type(&ty("Option<u32>")));
        assert!(is_option_type(&ty("core::option::Option<String>")));
        assert!(!is_option_type(&ty("u32")));
        assert!(!is_option_type(&ty("Vec<u8>")));
        // Wrong arity is not treated as Option.
        assert!(!is_option_type(&ty("Option<u32, u32>")));
    }

    #[test]
    fn is_str_ref_matches_only_shared_str() {
        assert!(is_str_ref(&ty("&str")));
        assert!(is_str_ref(&ty("&'a str")));
        assert!(!is_str_ref(&ty("&mut str")));
        assert!(!is_str_ref(&ty("String")));
        assert!(!is_str_ref(&ty("&String")));
    }

    #[test]
    fn is_tool_context_type_matches_context_by_name() {
        assert!(is_tool_context_type(&ty("ToolContext")));
        assert!(is_tool_context_type(&ty("&ToolContext")));
        assert!(is_tool_context_type(&ty("&'a ToolContext")));
        assert!(is_tool_context_type(&ty("llm_tool::ToolContext")));
        assert!(!is_tool_context_type(&ty("&MyType")));
    }

    #[test]
    fn try_extract_result_types_requires_two_args() {
        assert!(
            try_extract_result_types(&ty("Result<u32, String>"))
                .unwrap()
                .is_some()
        );
        assert!(
            try_extract_result_types(&ty("std::result::Result<u32, E>"))
                .unwrap()
                .is_some()
        );
        // Single-arg aliases (anyhow::Result, io::Result) produce a compile error.
        assert!(try_extract_result_types(&ty("Result<u32>")).is_err());
        assert!(try_extract_result_types(&ty("String")).unwrap().is_none());
    }

    #[test]
    fn extract_doc_string_joins_and_trims_lines() {
        let f: ItemFn = parse_quote! {
            /// First line.
            /// Second line.
            fn foo() {}
        };
        assert_eq!(extract_doc_string(&f.attrs), "First line.\nSecond line.");

        let none: ItemFn = parse_quote! {
            fn bar() {}
        };
        assert_eq!(extract_doc_string(&none.attrs), "");
    }

    #[test]
    fn reject_generic_signature_flags_generics_lifetimes_and_where() {
        let generic: ItemFn = parse_quote! { fn foo<T>(x: T) -> T { x } };
        assert!(reject_generic_signature(&generic, "llm_tool").is_err());

        let lifetime: ItemFn = parse_quote! { fn foo<'a>(x: &'a str) -> &'a str { x } };
        assert!(reject_generic_signature(&lifetime, "llm_tool").is_err());

        let where_clause: ItemFn = parse_quote! { fn foo(x: u32) -> u32 where u32: Clone { x } };
        assert!(reject_generic_signature(&where_clause, "llm_tool").is_err());

        let concrete: ItemFn = parse_quote! { fn foo(x: u32) -> u32 { x } };
        assert!(reject_generic_signature(&concrete, "llm_tool").is_ok());
    }
}
