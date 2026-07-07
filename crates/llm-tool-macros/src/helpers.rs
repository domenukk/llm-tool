use quote::quote;
use syn::{FnArg, GenericArgument, ItemFn, Pat, PatType, PathArguments, Type};

#[allow(clippy::wildcard_imports)]
use crate::*;

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

    let base_dir = full_path.parent().unwrap_or(std::path::Path::new("."));
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
    if !attr.path().is_ident(ATTR_LLM_TOOL) {
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

pub(crate) fn extract_params(func: &ItemFn) -> syn::Result<Vec<ParamInfo>> {
    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "#[llm_tool] functions must be free functions (no `self`)",
                ));
            }
            FnArg::Typed(PatType { pat, ty, attrs, .. }) => {
                let name = match pat.as_ref() {
                    Pat::Ident(ident) => ident.ident.clone(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "#[llm_tool] parameters must be simple identifiers",
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
                    .filter(|a| a.path().is_ident("doc"))
                    .cloned()
                    .collect();
                params.push(ParamInfo {
                    name,
                    ty: ty.clone(),
                    doc_attrs,
                    is_context,
                });
            }
        }
    }
    Ok(params)
}

pub(crate) fn extract_doc_string(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            if let syn::Meta::NameValue(nv) = &attr.meta
                && let syn::Expr::Lit(lit) = &nv.value
                && let syn::Lit::Str(s) = &lit.lit
            {
                return Some(s.value());
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
pub(crate) fn parse_return_type(func: &ItemFn) -> syn::Result<ReturnInfo> {
    let syn::ReturnType::Type(_, ty) = &func.sig.output else {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[llm_tool] functions must have an explicit return type",
        ));
    };

    // Try to parse as Result<T, E>.
    if let Some(result_types) = try_extract_result_types(ty) {
        return Ok(result_types);
    }

    // Not a Result — treat as infallible bare type.
    Ok(ReturnInfo::BareType)
}

/// Try to extract `T` and `E` from a `Result<T, E>` return type.
/// Returns `None` if the type is not a `Result`.
pub(crate) fn try_extract_result_types(ty: &syn::Type) -> Option<ReturnInfo> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    let last_seg = type_path.path.segments.last()?;

    if last_seg.ident != "Result" {
        return None;
    }

    let PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };

    if args.args.len() != 2 {
        return None;
    }

    let GenericArgument::Type(ok_type) = &args.args[0] else {
        return None;
    };

    let GenericArgument::Type(err_type) = &args.args[1] else {
        return None;
    };

    Some(ReturnInfo::ResultType {
        ok_type: Box::new(ok_type.clone()),
        err_type: Box::new(err_type.clone()),
    })
}
