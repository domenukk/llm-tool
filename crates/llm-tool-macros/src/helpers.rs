use quote::{format_ident, quote};
use syn::{GenericArgument, ItemFn, LitStr, PathArguments, Type};

use crate::{
    ATTR_DOC, ParamInfo, ReturnInfo, TYPE_RESULT, ToolAttr, desc, params, response_struct_gen,
};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum MacroToolProvenance {
    #[default]
    Trusted,
    WorkspaceLocal,
    PrivateData,
    ExternalMcp,
    UntrustedNetwork,
}

impl MacroToolProvenance {
    pub const fn is_untrusted_source(self) -> bool {
        matches!(self, Self::ExternalMcp | Self::UntrustedNetwork)
    }

    pub fn to_tokens(self, crate_path: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        match self {
            Self::Trusted => quote!(#crate_path::ToolProvenance::Trusted),
            Self::WorkspaceLocal => quote!(#crate_path::ToolProvenance::WorkspaceLocal),
            Self::PrivateData => quote!(#crate_path::ToolProvenance::PrivateData),
            Self::ExternalMcp => quote!(#crate_path::ToolProvenance::ExternalMcp),
            Self::UntrustedNetwork => quote!(#crate_path::ToolProvenance::UntrustedNetwork),
        }
    }
}

pub fn parse_provenance_str(s: &str, span: proc_macro2::Span) -> syn::Result<MacroToolProvenance> {
    match s {
        "trusted" | "Trusted" => Ok(MacroToolProvenance::Trusted),
        "workspace_local" | "WorkspaceLocal" => Ok(MacroToolProvenance::WorkspaceLocal),
        "private_data" | "PrivateData" => Ok(MacroToolProvenance::PrivateData),
        "external_mcp" | "ExternalMcp" => Ok(MacroToolProvenance::ExternalMcp),
        "untrusted_network" | "UntrustedNetwork" => Ok(MacroToolProvenance::UntrustedNetwork),
        _ => Err(syn::Error::new(
            span,
            "expected `trusted`, `workspace_local`, `private_data`, `external_mcp`, or `untrusted_network`",
        )),
    }
}

pub use params::{
    build_extension_extractions, build_field_attrs_and_default_helpers,
    build_param_types_and_borrows,
};

pub fn extract_params(func: &ItemFn, macro_name: &str) -> syn::Result<Vec<ParamInfo>> {
    params::extract_fn_inputs(&func.sig.inputs, macro_name, false)
}

/// Wrap a sync or async function body returning either a `Result<T, E>` or a bare
/// value `T`, binding the success value to `__v`, evaluating `ok_expr`, and converting
/// any `E: Display` error via autoref `WrapErr(err).__convert_err()`.
pub fn build_wrapped_body(
    func: &ItemFn,
    return_info: &ReturnInfo,
    ok_expr: &proc_macro2::TokenStream,
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
            quote! {
                use ::llm_tool::__private::DisplayErrorFallback as _;
                #inner
                match __r {
                    ::core::result::Result::Ok(__v) => { #ok_expr },
                    ::core::result::Result::Err(err) => ::core::result::Result::Err(
                        ::llm_tool::__private::WrapErr(err).__convert_err()
                    ),
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
                #ok_expr
            }
        }
    }
}

pub fn build_body_tokens(
    func: &ItemFn,
    return_info: &ReturnInfo,
    crate_path: &proc_macro2::TokenStream,
    response_info: &ResponseTemplateInfo,
) -> proc_macro2::TokenStream {
    let ok_branch = build_ok_branch(crate_path, response_info);
    build_wrapped_body(func, return_info, &ok_branch)
}

/// Build the Ok-branch conversion: either the standard `Wrap(v).__convert()`
/// or template-based rendering when `response_template` is set.
pub fn build_ok_branch(
    crate_path: &proc_macro2::TokenStream,
    response_info: &ResponseTemplateInfo,
) -> proc_macro2::TokenStream {
    response_info.render_tokens.as_ref().map_or_else(
        || quote! { #crate_path::__private::Wrap(__v).__convert() },
        Clone::clone,
    )
}

// ── Response Template Resolution ────────────────────────────────────────────

/// Structured output from response template resolution.
pub struct ResponseTemplateInfo {
    /// Cargo dependency-tracking tokens.
    pub dep_tracking: proc_macro2::TokenStream,
    /// Helper tokens (e.g. static `LazyLock` declarations).
    pub helper_tokens: proc_macro2::TokenStream,
    /// Token stream that converts `__v` into `Result<ToolOutput, ToolError>`
    /// via template rendering. `None` = use default `__convert()` path.
    pub render_tokens: Option<proc_macro2::TokenStream>,
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

pub fn resolve_response_template(
    attr: Option<&ToolAttr>,
    struct_name: &syn::Ident,
    fn_name: &syn::Ident,
    ok_type: Option<&syn::Type>,
) -> syn::Result<ResponseTemplateInfo> {
    let Some(attr) = attr else {
        return Ok(ResponseTemplateInfo::default());
    };

    if let Some(response_path) = &attr.response_file_path {
        return resolve_response_template_file(attr, response_path, struct_name, fn_name, ok_type);
    }
    if let Some(response_inline) = &attr.response_inline {
        return resolve_response_template_inline(
            attr,
            response_inline,
            struct_name,
            fn_name,
            ok_type,
        );
    }
    Ok(ResponseTemplateInfo::default())
}

const SELF_SEGMENT: &str = "self";

/// Returns `true` when the annotated function's return type directly references the
/// compile-time generated `#response_struct_name` (enabling zero-`Context` `__v.render()`).
fn is_generated_response_struct_type(
    ok_type: Option<&syn::Type>,
    response_struct_name: &syn::Ident,
) -> bool {
    let Some(Type::Path(tp)) = ok_type else {
        return false;
    };
    if tp.qself.is_some() {
        return false;
    }
    let segs = &tp.path.segments;
    match segs.len() {
        1 => segs[0].ident == *response_struct_name && segs[0].arguments.is_none(),
        2 => {
            segs[0].ident == SELF_SEGMENT
                && segs[0].arguments.is_none()
                && segs[1].ident == *response_struct_name
                && segs[1].arguments.is_none()
        }
        _ => false,
    }
}

fn build_response_render_tokens(
    response_mod_name: &syn::Ident,
    use_direct_struct_render: bool,
) -> proc_macro2::TokenStream {
    let render_call = if use_direct_struct_render {
        quote! { __v.render() }
    } else {
        quote! { #response_mod_name::template().render(&__v) }
    };
    quote! {
        {
            let __rendered = #render_call
                .map_err(|e| ::llm_tool::ToolError::new(
                    format!("response template render error: {e}")
                ))?;
            ::llm_tool::ToolOutput::new(__rendered)
                .with_metadata(&__v)
                .map_err(|e| ::llm_tool::ToolError::new(
                    format!("response metadata error: {e}")
                ))
        }
    }
}

/// Resolve a template path relative to `CARGO_MANIFEST_DIR` (or absolute path as-is).
pub fn resolve_manifest_path(
    rel_path: &str,
    span: proc_macro2::Span,
) -> syn::Result<std::path::PathBuf> {
    let path = std::path::Path::new(rel_path);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let manifest_dir = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(dir) => dir,
        Err(std::env::VarError::NotPresent) => ".".to_string(),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(syn::Error::new(
                span,
                "CARGO_MANIFEST_DIR contains invalid UTF-8",
            ));
        }
    };
    let candidate = std::path::Path::new(&manifest_dir).join(rel_path);
    if candidate.exists() {
        return Ok(candidate);
    }

    let mut dir = std::path::Path::new(&manifest_dir);
    while let Some(parent) = dir.parent() {
        let direct = parent.join(rel_path);
        if direct.exists() {
            return Ok(direct);
        }
        let crates_dir = parent.join("crates");
        if crates_dir.is_dir() {
            let entries = std::fs::read_dir(&crates_dir).map_err(|err| {
                syn::Error::new(
                    span,
                    format!(
                        "failed to read workspace directory '{}': {err}",
                        crates_dir.display()
                    ),
                )
            })?;
            for entry_res in entries {
                let entry = entry_res.map_err(|err| {
                    syn::Error::new(
                        span,
                        format!(
                            "failed to read directory entry in '{}': {err}",
                            crates_dir.display()
                        ),
                    )
                })?;
                let crate_path = entry.path().join(rel_path);
                if crate_path.exists() {
                    return Ok(crate_path);
                }
            }
        }
        dir = parent;
    }

    Ok(candidate)
}

/// Implementation of response template resolution from file.
pub fn resolve_response_template_file(
    attr: &ToolAttr,
    response_path: &LitStr,
    struct_name: &syn::Ident,
    fn_name: &syn::Ident,
    ok_type: Option<&syn::Type>,
) -> syn::Result<ResponseTemplateInfo> {
    use syn::ext::IdentExt as _;

    let rel_path = response_path.value();
    let full_path = resolve_manifest_path(&rel_path, response_path.span())?;
    let path_str = full_path.to_string_lossy().to_string();

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
    let generated_idents =
        response_struct_gen::collect_generated_type_names(&response_struct_name_str, &fm);

    let response_struct_name = format_ident!("{}", response_struct_name_str);
    let response_mod_name = format_ident!("__{}_response_mod", fn_name.unraw());

    let env_toks = desc::env_tokens(attr);
    let reexport_doc = format!("Auto-generated response types for the [`{struct_name}`] tool.");
    let helper_tokens = quote! {
        ::llm_tool::__md_tmpl_macros::include_template!(
            #response_path as #response_struct_name => #response_mod_name,
            crate = ::llm_tool::__md_tmpl
            #env_toks
        );
        #[doc = #reexport_doc]
        pub use #response_mod_name::{ #( #generated_idents ),* };
    };

    let use_direct = is_generated_response_struct_type(ok_type, &response_struct_name);
    let render_tokens = build_response_render_tokens(&response_mod_name, use_direct);

    Ok(ResponseTemplateInfo {
        dep_tracking,
        helper_tokens,
        render_tokens: Some(render_tokens),
    })
}

/// Implementation of response template resolution from inline string.
pub fn resolve_response_template_inline(
    attr: &ToolAttr,
    response_inline: &LitStr,
    struct_name: &syn::Ident,
    fn_name: &syn::Ident,
    ok_type: Option<&syn::Type>,
) -> syn::Result<ResponseTemplateInfo> {
    use syn::ext::IdentExt as _;

    let source = response_inline.value();

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
    let generated_idents =
        response_struct_gen::collect_generated_type_names(&response_struct_name_str, &fm);

    let response_struct_name = format_ident!("{}", response_struct_name_str);
    let response_mod_name = format_ident!("__{}_response_mod", fn_name.unraw());

    let env_toks = desc::env_tokens(attr);
    let reexport_doc = format!("Auto-generated response types for the [`{struct_name}`] tool.");
    let helper_tokens = quote! {
        ::llm_tool::__md_tmpl_macros::template!(
            #source as #response_struct_name => #response_mod_name,
            crate = ::llm_tool::__md_tmpl
            #env_toks
        );
        #[doc = #reexport_doc]
        pub use #response_mod_name::{ #( #generated_idents ),* };
    };

    let use_direct = is_generated_response_struct_type(ok_type, &response_struct_name);
    let render_tokens = build_response_render_tokens(&response_mod_name, use_direct);

    Ok(ResponseTemplateInfo {
        dep_tracking: quote! {},
        helper_tokens,
        render_tokens: Some(render_tokens),
    })
}

/// Reject generic parameters, lifetimes, and `where` clauses on the annotated signature.
pub fn reject_generic_signature(sig: &syn::Signature, macro_name: &str) -> syn::Result<()> {
    let generics = &sig.generics;
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

pub fn extract_doc_string(attrs: &[syn::Attribute]) -> String {
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

/// Parse the return type from a signature — either `Result<T, E>` or a bare type `T`.
pub fn parse_sig_return_type(sig: &syn::Signature, macro_name: &str) -> syn::Result<ReturnInfo> {
    let syn::ReturnType::Type(_, ty) = &sig.output else {
        return Err(syn::Error::new_spanned(
            sig,
            format!("#[{macro_name}] functions must have an explicit return type"),
        ));
    };

    if let Some(result_types) = try_extract_result_types(ty)? {
        return Ok(result_types);
    }

    Ok(ReturnInfo::BareType)
}

pub fn parse_return_type(func: &ItemFn, macro_name: &str) -> syn::Result<ReturnInfo> {
    parse_sig_return_type(&func.sig, macro_name)
}

/// Try to extract `T` and `E` from a `Result<T, E>` return type.
pub fn try_extract_result_types(ty: &syn::Type) -> syn::Result<Option<ReturnInfo>> {
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
        assert!(reject_generic_signature(&generic.sig, "llm_tool").is_err());

        let lifetime: ItemFn = parse_quote! { fn foo<'a>(x: &'a str) -> &'a str { x } };
        assert!(reject_generic_signature(&lifetime.sig, "llm_tool").is_err());

        let where_clause: ItemFn = parse_quote! { fn foo(x: u32) -> u32 where u32: Clone { x } };
        assert!(reject_generic_signature(&where_clause.sig, "llm_tool").is_err());

        let concrete: ItemFn = parse_quote! { fn foo(x: u32) -> u32 { x } };
        assert!(reject_generic_signature(&concrete.sig, "llm_tool").is_ok());
    }
}
