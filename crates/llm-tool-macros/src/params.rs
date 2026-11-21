//! Function/method parameter extraction, `#[llm_tool(ext | context | default = ...)]`
//! attribute handling, `&str` / `Option<&str>` auto-borrowing, and field attribute forwarding.

use quote::{format_ident, quote};
use syn::{FnArg, GenericArgument, Pat, PatType, PathArguments, Token, Type, ext::IdentExt as _};

use crate::{ATTR_CONTEXT, ATTR_DOC, MACRO_LLM_TOOL, TYPE_OPTION, TYPE_STR, TYPE_TOOL_CONTEXT};

const ATTR_EXT: &str = "ext";
const ATTR_DEFAULT: &str = "default";
const ATTR_SERDE: &str = "serde";
const ATTR_SCHEMARS: &str = "schemars";

/// Classification of a parameter on an `#[llm_tool]`, `#[llm_prompt]`, or `#[llm_resource]` function.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParamKind {
    /// Model-supplied schema parameter (included in the generated `*Params` struct).
    SchemaInput,
    /// Borrowed `&ToolContext` execution context.
    ToolContextRef,
    /// Typed extension extracted via `_ctx.get_ext::<T>()` (`#[llm_tool(ext)]`).
    Extension,
}

/// Parsed information about a single function/method parameter.
pub struct ParamInfo {
    pub name: syn::Ident,
    pub ty: Box<syn::Type>,
    pub doc_attrs: Vec<syn::Attribute>,
    pub forwarded_attrs: Vec<syn::Attribute>,
    pub has_serde_default: bool,
    pub default_expr: Option<syn::Expr>,
    pub kind: ParamKind,
    pub is_mut: bool,
}

impl ParamInfo {
    pub const fn is_schema_input(&self) -> bool {
        matches!(self.kind, ParamKind::SchemaInput)
    }

    pub const fn is_context(&self) -> bool {
        matches!(self.kind, ParamKind::ToolContextRef)
    }

    pub const fn is_extension(&self) -> bool {
        matches!(self.kind, ParamKind::Extension)
    }
}

/// Extract the inner `T` if `ty` is `Option<T>` (`Option<T>`, `core::option::Option<T>`, etc.).
pub fn option_inner_type(ty: &syn::Type) -> Option<&syn::Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segments = &type_path.path.segments;
    let last_seg = segments.last()?;
    if last_seg.ident != TYPE_OPTION {
        return None;
    }
    if segments.len() > 1 {
        let prefixes: Vec<_> = segments
            .iter()
            .take(segments.len() - 1)
            .map(|s| s.ident.to_string())
            .collect();
        let prefix_str = prefixes.join("::");
        if !matches!(
            prefix_str.as_str(),
            "std::option" | "core::option" | "alloc::option"
        ) {
            return None;
        }
    }
    let PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    match args.args.first()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// Check whether `ty` is `Option<T>`.
pub fn is_option_type(ty: &syn::Type) -> bool {
    option_inner_type(ty).is_some()
}

/// Check whether `ty` is `&str` / `&'a str`.
pub fn is_str_ref(ty: &syn::Type) -> bool {
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

/// Check whether `ty` is `Option<&str>` / `Option<&'a str>`.
pub fn is_option_str_ref(ty: &syn::Type) -> bool {
    option_inner_type(ty).is_some_and(is_str_ref)
}

/// Check whether `ty` is `ToolContext`, `&ToolContext`, or a qualified path ending in `ToolContext`.
pub fn is_tool_context_type(ty: &syn::Type) -> bool {
    let mut inner = ty;
    while let Type::Reference(r) = inner {
        inner = r.elem.as_ref();
    }
    let Type::Path(type_path) = inner else {
        return false;
    };
    let segments = &type_path.path.segments;
    let Some(last_seg) = segments.last() else {
        return false;
    };
    if last_seg.ident != TYPE_TOOL_CONTEXT {
        return false;
    }
    if segments.len() > 1 {
        let prefixes: Vec<_> = segments
            .iter()
            .take(segments.len() - 1)
            .map(|s| s.ident.to_string())
            .collect();
        let prefix_str = prefixes.join("::");
        if !matches!(prefix_str.as_str(), "llm_tool" | "crate" | "super" | "self") {
            return false;
        }
    }
    true
}

#[derive(Default)]
struct ParsedParamLlmAttr {
    is_context: bool,
    is_ext: bool,
    default_expr: Option<syn::Expr>,
}

pub fn is_llm_tool_path(path: &syn::Path) -> bool {
    path.segments
        .last()
        .is_some_and(|seg| seg.ident == MACRO_LLM_TOOL)
}

fn parse_param_llm_tool_attr(
    attr: &syn::Attribute,
    out: &mut ParsedParamLlmAttr,
) -> syn::Result<()> {
    if !is_llm_tool_path(attr.path()) {
        return Ok(());
    }
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident(ATTR_CONTEXT) {
            if out.is_ext {
                return Err(meta.error("`context` and `ext` are mutually exclusive"));
            }
            out.is_context = true;
            Ok(())
        } else if meta.path.is_ident(ATTR_EXT) {
            if out.is_context {
                return Err(meta.error("`ext` and `context` are mutually exclusive"));
            }
            out.is_ext = true;
            Ok(())
        } else if meta.path.is_ident(ATTR_DEFAULT) {
            if out.default_expr.is_some() {
                return Err(meta.error("duplicate `default` attribute on parameter"));
            }
            let _: Token![=] = meta.input.parse()?;
            let expr: syn::Expr = meta.input.parse()?;
            out.default_expr = Some(expr);
            Ok(())
        } else {
            Err(meta.error(
                "unsupported parameter attribute; expected `context`, `ext`, or `default = ...`",
            ))
        }
    })
}

fn serde_attr_has_default(attr: &syn::Attribute) -> bool {
    let mut found = false;
    match attr.parse_nested_meta(|meta| {
        if meta.path.is_ident(ATTR_DEFAULT) {
            found = true;
        }
        if meta.input.peek(Token![=]) {
            let _: Token![=] = meta.input.parse()?;
            let _: syn::Expr = meta.input.parse()?;
        } else if meta.input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in meta.input);
            let _: proc_macro2::TokenStream = content.parse()?;
        }
        Ok(())
    }) {
        Ok(()) => found,
        Err(err) => {
            let non_empty = !err.to_string().is_empty();
            debug_assert!(non_empty);
            found
        }
    }
}

/// Returns `true` if `attr` is a parameter-level macro/schema/doc attribute that must be
/// stripped from emitted plain Rust method signatures on `impl` blocks.
pub fn is_stripped_impl_param_attr(attr: &syn::Attribute) -> bool {
    attr.path().is_ident(ATTR_DOC)
        || attr.path().is_ident(ATTR_SERDE)
        || attr.path().is_ident(ATTR_SCHEMARS)
        || is_llm_tool_path(attr.path())
}

/// Extract typed parameter metadata from a punctuated list of `FnArg`s.
///
/// When `allow_self_ref` is `false` (free functions), `self` receivers produce a compile error.
/// When `allow_self_ref` is `true` (`#[llm_tool]` methods inside `#[llm_tool] impl MyType`),
/// a shared `&self` receiver is required and skipped from the returned `Vec<ParamInfo>`.
pub fn extract_fn_inputs<'a, I>(
    inputs: I,
    macro_name: &str,
    allow_self_ref: bool,
) -> syn::Result<Vec<ParamInfo>>
where
    I: IntoIterator<Item = &'a FnArg>,
{
    let mut params = Vec::new();
    let mut saw_self = false;
    for arg in inputs {
        match arg {
            FnArg::Receiver(r) => {
                if !allow_self_ref {
                    return Err(syn::Error::new_spanned(
                        r,
                        format!("#[{macro_name}] functions must be free functions (no `self`)"),
                    ));
                }
                if r.reference.is_none() || r.mutability.is_some() {
                    return Err(syn::Error::new_spanned(
                        r,
                        "#[llm_tool] methods in an `impl` block must take a shared `&self` receiver",
                    ));
                }
                saw_self = true;
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

                let mut llm_attr = ParsedParamLlmAttr::default();
                let mut doc_attrs = Vec::new();
                let mut forwarded_attrs = Vec::new();
                let mut has_serde_default = false;

                for a in attrs {
                    if a.path().is_ident(ATTR_DOC) {
                        doc_attrs.push(a.clone());
                    } else if is_llm_tool_path(a.path()) {
                        parse_param_llm_tool_attr(a, &mut llm_attr)?;
                    } else if a.path().is_ident(ATTR_SERDE) {
                        has_serde_default |= serde_attr_has_default(a);
                        forwarded_attrs.push(a.clone());
                    } else if a.path().is_ident(ATTR_SCHEMARS) {
                        forwarded_attrs.push(a.clone());
                    }
                }

                let is_tool_context = is_tool_context_type(ty);
                if is_tool_context && !matches!(ty.as_ref(), syn::Type::Reference(_)) {
                    return Err(syn::Error::new_spanned(
                        ty,
                        "ToolContext parameter must be a reference type (e.g., `&ToolContext` or `&'a ToolContext`)",
                    ));
                }

                let kind = if llm_attr.is_ext {
                    if matches!(ty.as_ref(), syn::Type::Reference(_)) {
                        return Err(syn::Error::new_spanned(
                            ty,
                            "`#[llm_tool(ext)]` parameters must be owned `Clone + Send + Sync + 'static` types (e.g., `Arc<T>`), not references",
                        ));
                    }
                    ParamKind::Extension
                } else if llm_attr.is_context || is_tool_context {
                    ParamKind::ToolContextRef
                } else {
                    ParamKind::SchemaInput
                };

                if !matches!(kind, ParamKind::SchemaInput) && llm_attr.default_expr.is_some() {
                    return Err(syn::Error::new_spanned(
                        &name,
                        "`#[llm_tool(default = ...)]` cannot be combined with `context` or `ext` parameters",
                    ));
                }

                params.push(ParamInfo {
                    name,
                    ty: ty.clone(),
                    doc_attrs,
                    forwarded_attrs,
                    has_serde_default,
                    default_expr: llm_attr.default_expr,
                    kind,
                    is_mut,
                });
            }
        }
    }
    if allow_self_ref && !saw_self {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[llm_tool] methods in an `impl` block must take `&self`",
        ));
    }
    Ok(params)
}

/// Build `mut` binding tokens for `{FnName}Params` destructuring (omitting `mut` on `&str`
/// and `Option<&str>` fields whose `mut` binding lives on the shadowed borrow statement).
pub fn destructure_mut_tokens(params: &[&ParamInfo]) -> Vec<proc_macro2::TokenStream> {
    params
        .iter()
        .map(|p| {
            if p.is_mut && !is_str_ref(&p.ty) && !is_option_str_ref(&p.ty) {
                quote! { mut }
            } else {
                quote! {}
            }
        })
        .collect()
}

/// Build the struct field types and any auto-borrow bindings for `&str` and `Option<&str>` params.
pub fn build_param_types_and_borrows(
    params: &[&ParamInfo],
) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    params
        .iter()
        .map(|p| {
            let name = &p.name;
            let mut_tok = if p.is_mut {
                quote! { mut }
            } else {
                quote! {}
            };
            if is_str_ref(&p.ty) {
                (
                    quote! { String },
                    quote! { let #mut_tok #name: &str = &#name; },
                )
            } else if is_option_str_ref(&p.ty) {
                (
                    quote! { ::core::option::Option<String> },
                    quote! { let #mut_tok #name: ::core::option::Option<&str> = #name.as_deref(); },
                )
            } else {
                let ty = &p.ty;
                (quote! { #ty }, quote! {})
            }
        })
        .unzip()
}

/// Build per-field `#[serde(default...)]` + forwarded `#[serde]`/`#[schemars]` attributes,
/// plus any helper functions required by `#[llm_tool(default = expr)]`.
pub fn build_field_attrs_and_default_helpers(
    fn_prefix: &str,
    params: &[&ParamInfo],
    param_struct_types: &[proc_macro2::TokenStream],
) -> (Vec<proc_macro2::TokenStream>, proc_macro2::TokenStream) {
    let mut field_attr_tokens = Vec::with_capacity(params.len());
    let mut helper_fns = Vec::new();

    for (p, field_ty) in params.iter().zip(param_struct_types.iter()) {
        let forwarded = &p.forwarded_attrs;
        let default_attr = p.default_expr.as_ref().map_or_else(
            || {
                if !p.has_serde_default && is_option_type(&p.ty) {
                    quote! { #[serde(default)] }
                } else {
                    quote! {}
                }
            },
            |expr| {
                let param_unraw = p.name.unraw().to_string();
                let fn_name_str = format!("__llm_tool_default_{fn_prefix}_{param_unraw}");
                let fn_ident = format_ident!("{}", fn_name_str);
                helper_fns.push(quote! {
                    #[doc(hidden)]
                    fn #fn_ident() -> #field_ty {
                        #expr
                    }
                });
                quote! { #[serde(default = #fn_name_str)] }
            },
        );

        field_attr_tokens.push(quote! {
            #( #forwarded )*
            #default_attr
        });
    }

    let helpers = quote! { #( #helper_fns )* };
    (field_attr_tokens, helpers)
}

/// Generate extraction statements for all `#[llm_tool(ext)]` parameters from `_ctx`.
pub fn build_extension_extractions(
    all_params: &[ParamInfo],
    crate_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let stmts: Vec<proc_macro2::TokenStream> = all_params
        .iter()
        .filter(|p| p.is_extension())
        .map(|p| {
            let name = &p.name;
            let ty = &p.ty;
            let mut_tok = if p.is_mut {
                quote! { mut }
            } else {
                quote! {}
            };
            let param_label = name.unraw().to_string();
            quote! {
                let #mut_tok #name: #ty = match _ctx.get_ext::<#ty>() {
                    ::core::option::Option::Some(__ext_val) => __ext_val,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Err(#crate_path::ToolError::new(
                            concat!(
                                "missing required ToolContext extension `",
                                stringify!(#ty),
                                "` for parameter `",
                                #param_label,
                                "`"
                            )
                        ));
                    }
                };
            }
        })
        .collect();
    quote! { #( #stmts )* }
}

#[cfg(test)]
mod tests {
    use syn::parse_str;

    use super::*;

    fn ty(s: &str) -> Type {
        parse_str::<Type>(s).expect("valid type")
    }

    #[test]
    fn option_and_str_ref_detection() {
        assert!(is_option_type(&ty("Option<u32>")));
        assert!(is_option_type(&ty("core::option::Option<String>")));
        assert!(is_option_str_ref(&ty("Option<&str>")));
        assert!(is_option_str_ref(&ty("core::option::Option<&'a str>")));
        assert!(!is_option_str_ref(&ty("Option<String>")));
        assert!(!is_option_type(&ty("u32")));
        assert!(is_str_ref(&ty("&str")));
        assert!(is_str_ref(&ty("&'a str")));
        assert!(!is_str_ref(&ty("&mut str")));
    }

    #[test]
    fn tool_context_type_detection() {
        assert!(is_tool_context_type(&ty("ToolContext")));
        assert!(is_tool_context_type(&ty("&ToolContext")));
        assert!(is_tool_context_type(&ty("&'a ToolContext")));
        assert!(is_tool_context_type(&ty("llm_tool::ToolContext")));
        assert!(!is_tool_context_type(&ty("&MyType")));
    }
}
