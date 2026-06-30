//! Compile-time template parsing and AST extraction.
//!
//! Provides helpers to parse `.tmpl.md` files and compile them into
//! pre-compiled AST nodes at macro expansion time. Used by the
//! `#[llm_tool(response_file = "...")]` and
//! `#[llm_tool(prompt_file = "...", context = fn)]` code paths
//! to emit `Template::from_precompiled(...)` instead of runtime parsing.
//!
//! Adapted from `prompt-templates-macros/src/compile.rs`.

use hashbrown::{HashMap, HashSet};

/// Result of compiling a template at macro expansion time.
pub(crate) struct CompiledTemplateAst {
    pub(crate) frontmatter: prompt_templates::Frontmatter,
    pub(crate) segments: Vec<prompt_templates::compiled::Segment>,
    pub(crate) _inline_templates:
        HashMap<String, prompt_templates::compiled::CompiledInlineTemplate>,
    pub(crate) source_hash: u64,
}

/// FNV-1a hash of the template source, for integrity checking.
pub(crate) fn hash_source(source: &str) -> u64 {
    prompt_templates::__private::fnv1a_hash(source.as_bytes())
}

pub(crate) const KW_LIST: &str = "list";
pub(crate) const KW_STRUCT: &str = "struct";
pub(crate) const KW_OPTION: &str = "option";
pub(crate) const KW_ENUM: &str = "enum";
pub(crate) const KW_TMPL: &str = "tmpl";

pub(crate) const ANGLE_OPEN: char = '<';
pub(crate) const BRACKET_OPEN: char = '[';
#[cfg(feature = "prompt-templates")]
pub(crate) const CHAR_SLASH: char = '/';

pub(crate) const PAREN_SYNTAX: &str = "(...)";
pub(crate) const ANGLE_SYNTAX: &str = "<...>";
pub(crate) const BRACKET_SYNTAX: &str = "[...]";
pub(crate) const FRONTMATTER_DELIM: &str = "---";
pub(crate) const NEWLINE_DELIM_LF: &str = "\n---";
pub(crate) const NEWLINE_DELIM_CRLF: &str = "\r\n---";
pub(crate) const REL_PREFIX_CUR: &str = "./";
#[cfg(feature = "prompt-templates")]
pub(crate) const LABEL_INLINE: &str = "inline";
#[cfg(feature = "prompt-templates")]
pub(crate) const LABEL_INLINE_RESP: &str = "inline response";

pub(crate) fn normalize_and_validate_syntax(
    source: &str,
    rel_path: &str,
) -> Result<String, String> {
    let trimmed_start = source.trim_start();
    if !trimmed_start.starts_with(FRONTMATTER_DELIM) {
        return Ok(source.to_string());
    }

    let after_first_delim = &trimmed_start[FRONTMATTER_DELIM.len()..];
    let end_idx = if let Some(idx) = after_first_delim.find(NEWLINE_DELIM_LF) {
        idx
    } else if let Some(idx) = after_first_delim.find(NEWLINE_DELIM_CRLF) {
        idx
    } else {
        return Ok(source.to_string());
    };

    let prefix_len = source.len() - after_first_delim.len();
    let fm_str = &source[prefix_len..prefix_len + end_idx];

    // Reject legacy angle-bracket and square-bracket syntax for compound types.
    // prompt-templates now requires parentheses exclusively.
    let illegal_keywords = [KW_LIST, KW_STRUCT, KW_OPTION, KW_ENUM, KW_TMPL];
    for kw in &illegal_keywords {
        if fm_str.contains(&format!("{kw}{ANGLE_OPEN}"))
            || fm_str.contains(&format!("{kw} {ANGLE_OPEN}"))
            || fm_str.contains(&format!("{kw}{BRACKET_OPEN}"))
            || fm_str.contains(&format!("{kw} {BRACKET_OPEN}"))
        {
            return Err(format!(
                "template '{rel_path}' uses illegal {ANGLE_SYNTAX} or {BRACKET_SYNTAX} fallback syntax; only full {PAREN_SYNTAX} parentheses are permitted"
            ));
        }
    }

    // Source already uses () syntax which prompt-templates handles natively.
    // No normalization needed.
    Ok(source.to_string())
}

/// Compile a template source string into a pre-compiled AST.
///
/// Performs all the same validation as `prompt-templates-macros`: undeclared
/// variable detection, include resolution, and flow-sensitive type checking.
pub(crate) fn compile_template_to_ast(
    source: &str,
    base_dir: &std::path::Path,
) -> Result<CompiledTemplateAst, String> {
    let source_hash = hash_source(source);
    let (fm, body) = prompt_templates::parse_frontmatter_with_base_dir(source, base_dir)
        .map_err(|e| e.to_string())?;

    let (mut segments, inline_templates) =
        prompt_templates::compiled::compile(body, &fm.type_aliases).map_err(|e| e.to_string())?;

    // Static analysis: enforce that all parameters referenced in the body are declared.
    let referenced = prompt_templates::compiled::collect_referenced_params(&segments);
    let mut declared: HashSet<String> = fm.params.iter().cloned().collect();
    for c in &fm.consts {
        declared.insert(c.name.clone());
    }
    for import in &fm.imports {
        declared.insert(import.stem.clone());
    }
    for inline_name in inline_templates.keys() {
        declared.insert(inline_name.clone());
    }
    let undeclared: Vec<&String> = referenced
        .iter()
        .filter(|v| !declared.contains(v.as_str()))
        .collect();
    if !undeclared.is_empty() {
        let mut names: Vec<&str> = undeclared.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        return Err(format!(
            "undeclared variable(s) referenced in body: {}",
            names.join(", ")
        ));
    }

    // Recursively resolve includes at compile time.
    let tmpl_params: HashSet<String> = fm
        .declarations
        .iter()
        .filter(|d| matches!(d.var_type, prompt_templates::VarType::Tmpl(_)))
        .map(|d| d.name.clone())
        .collect();
    let mut visited_paths = HashSet::new();
    resolve_includes_recursive(
        &mut segments,
        base_dir,
        &mut visited_paths,
        &inline_templates,
        &tmpl_params,
        0,
    )?;

    // Flow-sensitive type check.
    {
        let mut opaque_roots: HashSet<&str> = HashSet::new();
        for import in &fm.imports {
            opaque_roots.insert(&import.stem);
        }
        for c in &fm.consts {
            opaque_roots.insert(&c.name);
        }
        let type_errors = prompt_templates::compiled::validate_field_accesses_with_opaque(
            &segments,
            &fm.declarations,
            &opaque_roots,
        );
        if !type_errors.is_empty() {
            return Err(type_errors.join("\n"));
        }
    }

    Ok(CompiledTemplateAst {
        frontmatter: fm,
        segments,
        _inline_templates: inline_templates,
        source_hash,
    })
}

/// Maximum compile-time include depth.
const DEFAULT_MAX_COMPILE_INCLUDE_DEPTH: usize = 64;

fn max_compile_include_depth() -> usize {
    std::env::var("PROMPT_TEMPLATES_MAX_INCLUDE_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_COMPILE_INCLUDE_DEPTH)
}

fn resolve_includes_recursive(
    segments: &mut [prompt_templates::compiled::Segment],
    base_dir: &std::path::Path,
    visited_paths: &mut HashSet<std::path::PathBuf>,
    inline_templates: &HashMap<String, prompt_templates::compiled::CompiledInlineTemplate>,
    tmpl_params: &HashSet<String>,
    depth: usize,
) -> Result<(), String> {
    let max_depth = max_compile_include_depth();
    if depth > max_depth {
        return Err(format!(
            "compile-time include depth ({depth}) exceeds maximum ({max_depth}). \
             Set PROMPT_TEMPLATES_MAX_INCLUDE_DEPTH to increase the limit"
        ));
    }

    for seg in segments {
        match seg {
            prompt_templates::compiled::Segment::Include(inc) => {
                if tmpl_params.contains(inc.path.as_ref()) {
                    continue;
                }
                if let Some(compiled) = inline_templates.get(inc.path.as_ref()) {
                    inc.inline_compiled = Some(compiled.clone());
                    continue;
                }

                let include_path = base_dir.join(inc.path.as_ref());
                let canonical = include_path
                    .canonicalize()
                    .unwrap_or_else(|_| include_path.clone());

                if !visited_paths.insert(canonical.clone()) {
                    load_include_declarations(inc, &include_path)?;
                    continue;
                }

                resolve_single_include(inc, base_dir, visited_paths, depth + 1)?;
                visited_paths.remove(&canonical);
            }
            prompt_templates::compiled::Segment::ForLoop { body, .. } => {
                resolve_includes_recursive(
                    body,
                    base_dir,
                    visited_paths,
                    inline_templates,
                    tmpl_params,
                    depth,
                )?;
            }
            prompt_templates::compiled::Segment::If {
                branches,
                else_body,
            } => {
                for (_, branch_body) in branches {
                    resolve_includes_recursive(
                        branch_body,
                        base_dir,
                        visited_paths,
                        inline_templates,
                        tmpl_params,
                        depth,
                    )?;
                }
                resolve_includes_recursive(
                    else_body,
                    base_dir,
                    visited_paths,
                    inline_templates,
                    tmpl_params,
                    depth,
                )?;
            }
            prompt_templates::compiled::Segment::Match { arms, .. } => {
                for (_, arm_body) in arms {
                    resolve_includes_recursive(
                        arm_body,
                        base_dir,
                        visited_paths,
                        inline_templates,
                        tmpl_params,
                        depth,
                    )?;
                }
            }
            prompt_templates::compiled::Segment::Static(_)
            | prompt_templates::compiled::Segment::Expr { .. }
            | prompt_templates::compiled::Segment::Raw(_)
            | prompt_templates::compiled::Segment::Comment(_) => {}
        }
    }
    Ok(())
}

fn load_include_declarations(
    inc: &mut prompt_templates::compiled::CompiledInclude,
    include_path: &std::path::Path,
) -> Result<(), String> {
    if inc.inline_compiled.is_some() {
        return Ok(());
    }
    let included_source = std::fs::read_to_string(include_path)
        .map_err(|e| format!("cannot read include {}: {e}", include_path.display()))?;
    let cur_dir = REL_PREFIX_CUR.trim_end_matches(CHAR_SLASH);
    let included_base_dir = include_path
        .parent()
        .unwrap_or(std::path::Path::new(cur_dir));
    let norm_inc_src =
        normalize_and_validate_syntax(&included_source, &include_path.display().to_string())?;
    let (included_fm, included_body) =
        prompt_templates::parse_frontmatter_with_base_dir(&norm_inc_src, included_base_dir)
            .map_err(|e| format!("syntax error in include {}: {e}", include_path.display()))?;
    let (included_segments, _) =
        prompt_templates::compiled::compile(included_body, &included_fm.type_aliases).map_err(
            |e| {
                format!(
                    "compilation error in include {}: {e}",
                    include_path.display()
                )
            },
        )?;
    inc.inline_compiled = Some(prompt_templates::compiled::CompiledInlineTemplate {
        segments: std::sync::Arc::from(included_segments),
        declarations: std::sync::Arc::from(included_fm.declarations),
        consts: std::sync::Arc::new(HashMap::default()),
        imported_consts: std::sync::Arc::new(HashMap::default()),
    });
    Ok(())
}

fn resolve_single_include(
    inc: &mut prompt_templates::compiled::CompiledInclude,
    base_dir: &std::path::Path,
    visited_paths: &mut HashSet<std::path::PathBuf>,
    depth: usize,
) -> Result<(), String> {
    let include_path = base_dir.join(inc.path.as_ref());
    let included_source = std::fs::read_to_string(&include_path)
        .map_err(|e| format!("cannot read include {}: {e}", include_path.display()))?;

    let included_base_dir = include_path.parent().unwrap_or(base_dir);
    let norm_inc_src =
        normalize_and_validate_syntax(&included_source, &include_path.display().to_string())?;
    let (included_fm, included_body) =
        prompt_templates::parse_frontmatter_with_base_dir(&norm_inc_src, included_base_dir)
            .map_err(|e| format!("syntax error in include {}: {e}", include_path.display()))?;

    let (mut included_segments, included_inline_templates) =
        prompt_templates::compiled::compile(included_body, &included_fm.type_aliases).map_err(
            |e| {
                format!(
                    "compilation error in include {}: {e}",
                    include_path.display()
                )
            },
        )?;

    let child_base_dir = include_path.parent().unwrap_or(base_dir);
    {
        let child_tmpl_params: HashSet<String> = included_fm
            .declarations
            .iter()
            .filter(|d| matches!(d.var_type, prompt_templates::VarType::Tmpl(_)))
            .map(|d| d.name.clone())
            .collect();
        resolve_includes_recursive(
            &mut included_segments,
            child_base_dir,
            visited_paths,
            &included_inline_templates,
            &child_tmpl_params,
            depth,
        )?;
    }

    inc.inline_compiled = Some(prompt_templates::compiled::CompiledInlineTemplate {
        segments: std::sync::Arc::from(included_segments),
        declarations: std::sync::Arc::from(included_fm.declarations),
        consts: std::sync::Arc::new(HashMap::default()),
        imported_consts: std::sync::Arc::new(HashMap::default()),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejects_angle_bracket_syntax() {
        let fm = "---\ntypes:\n  - Foo = struct<a = str>\n---\nbody";
        let res = normalize_and_validate_syntax(fm, "test.tmpl.md");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("uses illegal <...> or [...] fallback syntax"),
            "got: {err}"
        );

        let fm_list = "---\nparams:\n  - items = list<str>\n---\nbody";
        let res_list = normalize_and_validate_syntax(fm_list, "test.tmpl.md");
        assert!(res_list.is_err());
    }

    #[test]
    fn test_rejects_bracket_syntax() {
        let fm = "---\ntypes:\n  - Foo = struct[a = str]\n---\nbody";
        let res = normalize_and_validate_syntax(fm, "test.tmpl.md");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(
            err.contains("uses illegal <...> or [...] fallback syntax"),
            "got: {err}"
        );
    }

    #[test]
    fn test_accepts_paren_syntax() {
        let fm =
            "---\ntypes:\n  - Foo = struct(a = str)\n\nparams:\n  - items = list(Foo)\n---\nbody";
        let res = normalize_and_validate_syntax(fm, "test.tmpl.md");
        assert!(res.is_ok(), "failed: {:?}", res.err());
        let norm = res.unwrap();
        assert!(norm.contains("struct(a = str)"), "got: {norm}");
        assert!(norm.contains("list(Foo)"), "got: {norm}");
    }
}
