//! Predict auto-generated type names from response template frontmatter.
//!
//! When `response_file = "..."` is used, the macro delegates struct generation
//! to `include_template!` from `prompt-templates-macros`. This module predicts
//! the type names that `include_template!` will generate, so we can emit
//! selective `pub use` re-exports.

use convert_case::{Case, Casing};
use quote::format_ident;

/// Collect all type names that `include_template!` will generate for the given
/// struct name and frontmatter declarations.
///
/// Returns identifiers for the top-level struct and all nested sub-structs
/// (list item structs, struct field structs, etc.).
pub(crate) fn collect_generated_type_names(
    struct_name: &str,
    declarations: &[prompt_templates::VarDecl],
) -> Vec<syn::Ident> {
    let mut names = vec![format_ident!("{}", struct_name)];
    for decl in declarations {
        collect_nested_names(struct_name, &decl.name, &decl.var_type, &mut names);
    }
    names
}

/// Recursively collect nested type names from compound `VarType` fields.
fn collect_nested_names(
    parent: &str,
    field: &str,
    ty: &prompt_templates::VarType,
    names: &mut Vec<syn::Ident>,
) {
    use prompt_templates::VarType;
    match ty {
        VarType::List(fields)
            if !fields.is_empty() && (fields.len() != 1 || !fields[0].name.is_empty()) =>
        {
            let pascal = field.to_case(Case::Pascal);
            let item_name = format!("{parent}{pascal}Item");
            names.push(format_ident!("{}", item_name));
            for f in fields {
                collect_nested_names(&item_name, &f.name, &f.var_type, names);
            }
        }
        VarType::Struct(fields) if !fields.is_empty() => {
            let pascal = field.to_case(Case::Pascal);
            let nested_name = format!("{parent}{pascal}");
            names.push(format_ident!("{}", nested_name));
            for f in fields {
                collect_nested_names(&nested_name, &f.name, &f.var_type, names);
            }
        }
        VarType::Option(inner) => {
            collect_nested_names(parent, field, inner, names);
        }
        _ => {}
    }
}
