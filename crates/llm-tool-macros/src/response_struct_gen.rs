//! Helper logic for `response_file` response struct prediction.
//!
//! When `#[llm_tool(response_file = "path.tmpl.md")]` is used, the macro delegates
//! to `include_template!` from `md-tmpl-macros`. This module predicts
//! what type names `include_template!` will generate so `llm-tool-macros` can
//! generate appropriate `pub use` statements to bring them into scope.

use convert_case::{Case, Casing};
use quote::format_ident;

/// Collect all `{struct_name}`-prefixed type names that `include_template!` /
/// `template!` will generate for the given response struct name and frontmatter.
///
/// Returns deduplicated identifiers for the top-level `{struct_name}` and all
/// `{struct_name}`-prefixed nested sub-structs, sub-enums, variant sub-structs,
/// and `pub type` aliases. Unprefixed `types:` aliases remain scoped inside the
/// generated `__{fn_name}_response_mod` module so they never collide with
/// user-defined structs or sibling `#[llm_tool]` definitions in the caller's module.
pub fn collect_generated_type_names(
    struct_name: &str,
    fm: &md_tmpl::Frontmatter,
) -> Vec<syn::Ident> {
    let mut names = vec![format_ident!("{}", struct_name)];
    for decl in &fm.declarations {
        let has_alias = fm
            .imported_type_params
            .get(&decl.name)
            .is_some_and(|(stem, _)| stem.is_empty());
        collect_nested_names(
            struct_name,
            &decl.name,
            &decl.var_type,
            has_alias,
            &mut names,
        );
    }

    let mut seen = std::collections::HashSet::with_capacity(names.len());
    names.retain(|ident| seen.insert(ident.to_string()));
    names
}

/// Recursively collect nested type names from compound `VarType` fields.
fn collect_nested_names(
    parent: &str,
    field: &str,
    ty: &md_tmpl::VarType,
    has_alias_override: bool,
    names: &mut Vec<syn::Ident>,
) {
    use md_tmpl::VarType;
    match ty {
        VarType::List(fields) if fields.len() == 1 && fields[0].name.is_empty() => {
            collect_nested_names(
                parent,
                field,
                &fields[0].var_type,
                has_alias_override,
                names,
            );
        }
        VarType::List(fields) if !fields.is_empty() => {
            let pascal = field.to_case(Case::Pascal);
            let item_name = format!("{parent}{pascal}Item");
            names.push(format_ident!("{}", item_name));
            if !has_alias_override {
                for f in fields {
                    collect_nested_names(&item_name, &f.name, &f.var_type, false, names);
                }
            }
        }
        VarType::Struct(fields) if !fields.is_empty() => {
            let pascal = field.to_case(Case::Pascal);
            let nested_name = format!("{parent}{pascal}");
            names.push(format_ident!("{}", nested_name));
            if !has_alias_override {
                for f in fields {
                    collect_nested_names(&nested_name, &f.name, &f.var_type, false, names);
                }
            }
        }
        VarType::Enum(_) if ty.is_option() => {
            if let Some(inner) = ty.option_inner_type() {
                collect_nested_names(parent, field, inner, has_alias_override, names);
            }
        }
        VarType::Enum(variants) => {
            let pascal = field.to_case(Case::Pascal);
            let enum_name = format!("{parent}{pascal}");
            names.push(format_ident!("{}", enum_name));
            if !has_alias_override {
                for v in variants {
                    let var_prefix = format!("{enum_name}{}", v.name.to_case(Case::Pascal));
                    for f in &v.fields {
                        collect_nested_names(&var_prefix, &f.name, &f.var_type, false, names);
                    }
                }
            }
        }
        VarType::Option(inner) => {
            collect_nested_names(parent, field, inner, has_alias_override, names);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_generated_type_names_includes_optional_and_list_enums_variant_structs_and_aliases() {
        let src = r"---
types:
  - Verdict = enum(Pass, Fail(reason = struct(code = int)))
  - Row = list(col = str, count = int)

params:
  - verdict = Verdict
  - rows = list(Row)
  - status = option(enum(Pending, Done(detail = struct(code = int))))
  - tags = list(enum(Alpha, Beta))
  - items = list(name = str, meta = struct(owner = str))
---
{{ kind(verdict) }}";
        let (fm, _) = md_tmpl::parse_frontmatter(src).expect("valid frontmatter");
        let idents: Vec<String> = collect_generated_type_names("MyToolResponse", &fm)
            .into_iter()
            .map(|i| i.to_string())
            .collect();
        assert_eq!(
            idents,
            vec![
                "MyToolResponse",
                "MyToolResponseVerdict",
                "MyToolResponseVerdictFailReason",
                "MyToolResponseRowsItem",
                "MyToolResponseStatus",
                "MyToolResponseStatusDoneDetail",
                "MyToolResponseTags",
                "MyToolResponseItemsItem",
                "MyToolResponseItemsItemMeta",
            ]
        );
    }
}
