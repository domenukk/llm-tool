use quote::quote;
use syn::{ItemFn, LitStr};

#[allow(clippy::wildcard_imports)]
use crate::*;

pub(crate) fn resolve_description(
    func: &ItemFn,
    attr: Option<&ToolAttr>,
) -> syn::Result<DescriptionInfo> {
    match attr {
        // Inline prompt template or string.
        Some(
            tool_attr @ ToolAttr {
                prompt_inline: Some(_),
                ..
            },
        ) => resolve_inline_description(tool_attr, &func.sig.ident),
        // Template file.
        Some(
            tool_attr @ ToolAttr {
                prompt_file_path: Some(_),
                ..
            },
        ) => resolve_template_description(tool_attr, &func.sig.ident),
        // No attribute, or attribute with only response_file — use doc comment.
        _ => {
            let desc = extract_doc_string(&func.attrs);
            if desc.is_empty() {
                return Err(syn::Error::new_spanned(
                    &func.sig.ident,
                    "#[llm_tool] functions must have a doc comment \
                     (used as the tool description), or use \
                     #[llm_tool(prompt = \"...\")]",
                ));
            }
            Ok(DescriptionInfo {
                static_description: desc,
                helper_tokens: quote! {},
                description_method: None,
                dep_tracking: quote! {},
            })
        }
    }
}

/// Resolve dynamic/static description from inline template string.
pub(crate) fn resolve_inline_description(
    attr: &ToolAttr,
    fn_name: &syn::Ident,
) -> syn::Result<DescriptionInfo> {
    #[cfg(not(feature = "md-tmpl"))]
    {
        let _ = fn_name;
        let span = attr
            .prompt_inline
            .as_ref()
            .map_or(proc_macro2::Span::call_site(), LitStr::span);
        if attr.has_inline_params || attr.has_context_fn {
            return Err(syn::Error::new(
                span,
                "the `md-tmpl` feature must be enabled to use dynamic inline prompts",
            ));
        }
        let desc = attr.prompt_inline.as_ref().unwrap().value();
        Ok(DescriptionInfo {
            static_description: desc,
            helper_tokens: quote! {},
            description_method: None,
            dep_tracking: quote! {},
        })
    }

    #[cfg(feature = "md-tmpl")]
    resolve_inline_description_impl(attr, fn_name)
}

/// Read a `.tmpl.md` template file and extract its body as the tool description.
pub(crate) fn resolve_template_description(
    attr: &ToolAttr,
    fn_name: &syn::Ident,
) -> syn::Result<DescriptionInfo> {
    #[cfg(not(feature = "md-tmpl"))]
    {
        let _ = fn_name;
        let span = attr
            .prompt_file_path
            .as_ref()
            .map_or(proc_macro2::Span::call_site(), LitStr::span);
        Err(syn::Error::new(
            span,
            "the `md-tmpl` feature must be enabled to use \
             `#[llm_tool(prompt_file = \"...\")]`. \
             Add `features = [\"md-tmpl\"]` to your llm-tool dependency.",
        ))
    }

    #[cfg(feature = "md-tmpl")]
    resolve_template_description_impl(attr, fn_name)
}

/// Implementation of template description resolution (feature-gated).
///
/// Handles three sub-cases:
/// 1. Static template (no declared variables) → `const DESCRIPTION`
/// 2. Template + `params(...)` → compile-time render → `const DESCRIPTION`
/// 3. Template + `context = fn` → runtime render via `description()` method
#[cfg(feature = "md-tmpl")]
pub(crate) fn resolve_template_description_impl(
    attr: &ToolAttr,
    fn_name: &syn::Ident,
) -> syn::Result<DescriptionInfo> {
    let template_lit = attr
        .prompt_file_path
        .as_ref()
        .expect("prompt_file_path validated");
    let rel_path = template_lit.value();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = std::path::Path::new(&manifest_dir).join(&rel_path);

    let source = std::fs::read_to_string(&full_path).map_err(|e| {
        syn::Error::new(
            template_lit.span(),
            format!("failed to read template '{}': {e}", full_path.display()),
        )
    })?;

    let base_dir = full_path.parent().unwrap_or(std::path::Path::new("."));
    let (fm, body) = md_tmpl::parse_frontmatter_with_base_dir(&source, base_dir).map_err(|e| {
        syn::Error::new(
            template_lit.span(),
            format!("template '{rel_path}' error: {e}"),
        )
    })?;

    let body_str = body.trim().to_string();
    let path_str = full_path.to_string_lossy().to_string();

    // include_str! establishes a file dependency so cargo rebuilds
    // when the template changes.
    let dep_tracking = quote! {
        const _: &str = include_str!(#path_str);
    };

    let has_params = !attr.inline_params.is_empty();
    let has_context = attr.context_fn.is_some();
    let has_declarations = !fm.declarations.is_empty();

    if !has_declarations && !has_params && !has_context {
        // Case 1: Static template — no variables, no params, no context.
        Ok(DescriptionInfo {
            static_description: body_str,
            helper_tokens: quote! {},
            description_method: None,
            dep_tracking,
        })
    } else if has_params {
        // Case 2: Compile-time params — render at build time.
        resolve_template_with_params(
            attr,
            &fm,
            &source,
            &rel_path,
            template_lit.span(),
            dep_tracking,
        )
    } else if has_context {
        // Case 3: Runtime context function.
        resolve_context_description(ResolveContextArgs {
            attr,
            rel_path: &rel_path,
            template_lit,
            source: &source,
            full_path: &full_path,
            body_str: &body_str,
            has_declarations,
            dep_tracking,
            fn_name,
        })
    } else {
        // Template declares variables but neither params nor context provided.
        let declared: Vec<&str> = fm.declarations.iter().map(|d| d.name.as_str()).collect();
        Err(syn::Error::new(
            template_lit.span(),
            format!(
                "template '{rel_path}' declares parameters ({}) but neither \
                 `params(...)` nor `context = ...` was provided",
                declared.join(", ")
            ),
        ))
    }
}

/// Implementation of inline template description resolution (feature-gated).
#[cfg(feature = "md-tmpl")]
pub(crate) fn resolve_inline_description_impl(
    attr: &ToolAttr,
    fn_name: &syn::Ident,
) -> syn::Result<DescriptionInfo> {
    let template_lit = attr
        .prompt_inline
        .as_ref()
        .expect("prompt_inline validated");
    let source = template_lit.value();
    let trimmed = source.trim_start();
    if !trimmed.starts_with("---") {
        return Ok(DescriptionInfo {
            static_description: source,
            helper_tokens: quote! {},
            description_method: None,
            dep_tracking: quote! {},
        });
    }

    let (fm, body) = md_tmpl::parse_frontmatter(&source)
        .map_err(|e| syn::Error::new(template_lit.span(), format!("inline template error: {e}")))?;

    let body_str = body.trim().to_string();

    let has_params = attr.has_inline_params;
    let has_context = attr.has_context_fn;
    let has_declarations = !fm.declarations.is_empty();

    if !has_declarations && !has_params && !has_context {
        // Case 1: Static template — no variables, no params, no context.
        Ok(DescriptionInfo {
            static_description: body_str,
            helper_tokens: quote! {},
            description_method: None,
            dep_tracking: quote! {},
        })
    } else if has_params {
        // Case 2: Compile-time inline params — render at build time.
        resolve_template_with_params(
            attr,
            &fm,
            &source,
            "<inline>",
            template_lit.span(),
            quote! {},
        )
    } else if has_context {
        // Case 3: Runtime context function.
        let desc_mod_name = format_ident!("__{}_desc_mod", fn_name);
        let helper_tokens = quote! {
            ::llm_tool::__md_tmpl_macros::template!(
                #template_lit => #desc_mod_name,
                crate = ::llm_tool::__md_tmpl
            );
        };
        let context_fn = attr.context_fn.as_ref().unwrap();

        let description_method = quote! {
            fn description(&self) -> ::llm_tool::__private::Cow<'static, str> {
                let ctx = #context_fn(self);
                let rendered = #desc_mod_name::template().render_ctx(&ctx)
                    .expect("Failed to render tool description template");
                ::llm_tool::__private::Cow::Owned(rendered)
            }
        };

        Ok(DescriptionInfo {
            static_description: body_str.clone(),
            helper_tokens,
            description_method: Some(description_method),
            dep_tracking: quote! {},
        })
    } else {
        let declared: Vec<&str> = fm.declarations.iter().map(|d| d.name.as_str()).collect();
        Err(syn::Error::new(
            template_lit.span(),
            format!(
                "inline template declares parameters ({}) but neither \
                 `params(...)` nor `context = ...` was provided",
                declared.join(", ")
            ),
        ))
    }
}

#[cfg(feature = "md-tmpl")]
pub(crate) struct ResolveContextArgs<'a> {
    pub(crate) attr: &'a ToolAttr,
    pub(crate) rel_path: &'a str,
    pub(crate) template_lit: &'a LitStr,
    pub(crate) source: &'a str,
    pub(crate) full_path: &'a std::path::Path,
    pub(crate) body_str: &'a str,
    pub(crate) has_declarations: bool,
    pub(crate) dep_tracking: proc_macro2::TokenStream,
    pub(crate) fn_name: &'a syn::Ident,
}

/// Resolve a template description with a runtime context function.
///
/// Generates a `description(&self)` method that uses `include_template!` to compile
/// the template once, then renders it with the user-provided context function
/// on every call.
#[cfg(feature = "md-tmpl")]
pub(crate) fn resolve_context_description(
    args: ResolveContextArgs<'_>,
) -> syn::Result<DescriptionInfo> {
    let ResolveContextArgs {
        attr,
        rel_path,
        template_lit,
        source: _source,
        full_path: _full_path,
        body_str,
        has_declarations,
        dep_tracking: _dep_tracking,
        fn_name,
    } = args;
    let context_fn = attr.context_fn.as_ref().ok_or_else(|| {
        syn::Error::new(
            template_lit.span(),
            "internal error: resolve_context_description called without context_fn",
        )
    })?;

    if !has_declarations {
        return Err(syn::Error::new(
            template_lit.span(),
            format!(
                "template '{rel_path}' has no declared parameters, \
                 so `context = ...` is unnecessary. Remove `context` \
                 or add params to the template."
            ),
        ));
    }

    let desc_mod_name = format_ident!("__{}_desc_mod", fn_name);
    let rel_path_lit = syn::LitStr::new(rel_path, template_lit.span());
    let helper_tokens = quote! {
        ::llm_tool::__md_tmpl_macros::include_template!(
            #rel_path_lit => #desc_mod_name,
            crate = ::llm_tool::__md_tmpl
        );
    };

    let description_method = quote! {
        fn description(&self) -> ::llm_tool::__private::Cow<'static, str> {
            let ctx = #context_fn(self);
            let rendered = #desc_mod_name::template().render_ctx(&ctx)
                .expect("Failed to render tool description template");
            ::llm_tool::__private::Cow::Owned(rendered)
        }
    };

    Ok(DescriptionInfo {
        static_description: body_str.to_string(),
        helper_tokens,
        description_method: Some(description_method),
        dep_tracking: quote! {},
    })
}

/// Render a template with compile-time `params(...)` values.
///
/// Validates:
/// - Every declared template variable has a matching `params(...)` key
/// - Every `params(...)` key matches a declared template variable
/// - The template renders without errors
#[cfg(feature = "md-tmpl")]
pub(crate) fn resolve_template_with_params(
    attr: &ToolAttr,
    fm: &md_tmpl::Frontmatter,
    source: &str,
    rel_path: &str,
    span: proc_macro2::Span,
    dep_tracking: proc_macro2::TokenStream,
) -> syn::Result<DescriptionInfo> {
    let mut expected_names = std::collections::HashSet::new();
    let mut struct_fields: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for decl in &fm.declarations {
        if let md_tmpl::VarType::Struct(fields) = &decl.var_type {
            for f in fields {
                expected_names.insert(f.name.as_str());
                struct_fields.insert(f.name.clone(), decl.name.clone());
            }
        } else {
            expected_names.insert(decl.name.as_str());
        }
    }

    let provided_names: std::collections::HashSet<String> = attr
        .inline_params
        .iter()
        .map(|(k, _)| k.to_string())
        .collect();

    // Check for missing params (declared but not provided).
    let missing: Vec<&str> = expected_names
        .iter()
        .filter(|n| !provided_names.contains(**n))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(syn::Error::new(
            span,
            format!(
                "template '{rel_path}' declares parameters not provided in `params(...)`: {}",
                missing.join(", ")
            ),
        ));
    }

    // Check for extra params (provided but not declared).
    for (key, _) in &attr.inline_params {
        let key_str = key.to_string();
        if !expected_names.contains(key_str.as_str()) {
            return Err(syn::Error::new(
                key.span(),
                format!(
                    "param `{key_str}` is not declared in template '{rel_path}'. \
                     Declared params: {}",
                    expected_names.into_iter().collect::<Vec<_>>().join(", ")
                ),
            ));
        }
    }

    // Build context and render at compile time.
    let template = md_tmpl::Template::from_source(source)
        .map_err(|e| syn::Error::new(span, format!("template '{rel_path}' parse error: {e}")))?;

    let mut root_values: std::collections::HashMap<String, md_tmpl::Value> =
        std::collections::HashMap::new();
    let mut struct_maps: std::collections::HashMap<
        String,
        std::collections::HashMap<String, md_tmpl::Value>,
    > = std::collections::HashMap::new();

    for (key, value) in &attr.inline_params {
        let key_str = key.to_string();
        if let Some(parent_struct) = struct_fields.get(&key_str) {
            struct_maps
                .entry(parent_struct.clone())
                .or_default()
                .insert(key_str, md_tmpl::Value::Str(value.value()));
        } else {
            root_values.insert(key_str, md_tmpl::Value::Str(value.value()));
        }
    }

    for (struct_name, s_map) in struct_maps {
        root_values.insert(
            struct_name,
            md_tmpl::Value::Struct(std::sync::Arc::new(s_map.into_iter().collect())),
        );
    }

    let mut ctx = md_tmpl::Context::new();
    for (k, v) in root_values {
        ctx.set(k, v);
    }

    let rendered = template
        .render_ctx(&ctx)
        .map_err(|e| syn::Error::new(span, format!("template '{rel_path}' render error: {e}")))?;

    Ok(DescriptionInfo {
        static_description: rendered,
        helper_tokens: quote! {},
        description_method: None,
        dep_tracking,
    })
}
