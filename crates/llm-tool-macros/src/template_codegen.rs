//! Template codegen adapted from `prompt-templates-macros` for `llm-tool-macros`.
//!
//! This module mirrors the codegen functions from `md_tmpl_macros::codegen`,
//! but rewrites all generated runtime paths from `::md_tmpl::` to
//! `::llm_tool::__md_tmpl::` so that downstream crates only need to depend
//! on `llm_tool` (which re-exports `md_tmpl` under the hidden
//! `__md_tmpl` module).
//!
//! Function signatures still reference `md_tmpl::` types directly because
//! `llm-tool-macros` has a direct build-time dependency on `prompt-templates`.

use quote::quote;

pub(crate) fn codegen_segment(seg: &md_tmpl::compiled::Segment) -> proc_macro2::TokenStream {
    use md_tmpl::compiled::Segment;
    match seg {
        Segment::Static(s) => quote! {
            ::llm_tool::__md_tmpl::compiled::Segment::Static(::llm_tool::__md_tmpl::__private::Cow::Borrowed(#s))
        },
        Segment::Expr { expr, filters } => {
            let filters_tokens = filters.iter().map(codegen_parsed_filter);
            let expr_tokens = codegen_compiled_expr(expr);
            quote! {
                ::llm_tool::__md_tmpl::compiled::Segment::Expr {
                    expr: #expr_tokens,
                    filters: ::llm_tool::__md_tmpl::__private::vec![#(#filters_tokens),*],
                }
            }
        }
        Segment::ForLoop {
            binding,
            list_path,
            body,
            else_body,
        } => {
            let body_tokens = body.iter().map(codegen_segment);
            let else_body_tokens = else_body.iter().map(codegen_segment);
            let list_path_tokens = codegen_compiled_path(list_path);
            quote! {
                ::llm_tool::__md_tmpl::compiled::Segment::ForLoop {
                    binding: ::llm_tool::__md_tmpl::__private::Cow::Borrowed(#binding),
                    list_path: #list_path_tokens,
                    body: ::llm_tool::__md_tmpl::__private::vec![#(#body_tokens),*],
                    else_body: ::llm_tool::__md_tmpl::__private::vec![#(#else_body_tokens),*],
                }
            }
        }
        Segment::If {
            branches,
            else_body,
        } => {
            let branch_tokens = branches.iter().map(|(cond, body)| {
                let cond_tokens = codegen_condition(cond);
                let body_tokens = body.iter().map(codegen_segment);
                quote! {
                    (#cond_tokens, ::llm_tool::__md_tmpl::__private::vec![#(#body_tokens),*])
                }
            });
            let else_tokens = else_body.iter().map(codegen_segment);
            quote! {
                ::llm_tool::__md_tmpl::compiled::Segment::If {
                    branches: ::llm_tool::__md_tmpl::__private::vec![#(#branch_tokens),*],
                    else_body: ::llm_tool::__md_tmpl::__private::vec![#(#else_tokens),*],
                }
            }
        }
        Segment::Raw(s) => quote! {
            ::llm_tool::__md_tmpl::compiled::Segment::Raw(::llm_tool::__md_tmpl::__private::Cow::Borrowed(#s))
        },
        Segment::Comment(refs) => {
            quote! {
                ::llm_tool::__md_tmpl::compiled::Segment::Comment(::llm_tool::__md_tmpl::__private::vec![#(::llm_tool::__md_tmpl::__private::Cow::Borrowed(#refs)),*])
            }
        }
        Segment::Include(inc) => codegen_segment_include(inc),
        Segment::Match { expr, arms, .. } => codegen_segment_match(expr, arms),
    }
}

pub(crate) fn codegen_segment_include(
    inc: &md_tmpl::compiled::CompiledInclude,
) -> proc_macro2::TokenStream {
    let path = &inc.path;
    let with_vars = inc.with_vars.iter().map(|(k, v)| {
        quote! { (::llm_tool::__md_tmpl::__private::Cow::Borrowed(#k), ::llm_tool::__md_tmpl::__private::Cow::Borrowed(#v)) }
    });
    let for_each = inc.for_each.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |(b, l)| quote! { ::core::option::Option::Some((::llm_tool::__md_tmpl::__private::Cow::Borrowed(#b), ::llm_tool::__md_tmpl::__private::Cow::Borrowed(#l))) },
    );
    let inline_compiled = inc.inline_compiled.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |ic| {
            let ic_tokens = codegen_compiled_inline_template(ic);
            quote! { ::core::option::Option::Some(#ic_tokens) }
        },
    );
    quote! {
        ::llm_tool::__md_tmpl::compiled::Segment::Include(
            ::llm_tool::__md_tmpl::compiled::CompiledInclude {
                path: ::llm_tool::__md_tmpl::__private::Cow::Borrowed(#path),
                with_vars: ::llm_tool::__md_tmpl::__private::vec![#(#with_vars),*],
                for_each: #for_each,
                inline_compiled: #inline_compiled,
            }
        )
    }
}

pub(crate) fn codegen_segment_match(
    expr: &md_tmpl::compiled::CompiledPath,
    arms: &[(
        Vec<std::borrow::Cow<'static, str>>,
        Vec<md_tmpl::compiled::Segment>,
    )],
) -> proc_macro2::TokenStream {
    let arm_tokens = arms.iter().map(|(variants, body)| {
        let body_tokens = body.iter().map(codegen_segment);
        let variant_tokens = variants.iter().map(|v| {
            quote! { ::llm_tool::__md_tmpl::__private::Cow::Borrowed(#v) }
        });
        quote! {
            (::llm_tool::__md_tmpl::__private::vec![#(#variant_tokens),*], ::llm_tool::__md_tmpl::__private::vec![#(#body_tokens),*])
        }
    });
    let expr_tokens = codegen_compiled_path(expr);
    quote! {
        ::llm_tool::__md_tmpl::compiled::Segment::Match {
            expr: #expr_tokens,
            arms: ::llm_tool::__md_tmpl::__private::vec![#(#arm_tokens),*],
        }
    }
}

pub(crate) fn codegen_parsed_filter(
    f: &md_tmpl::compiled::ParsedFilter,
) -> proc_macro2::TokenStream {
    let kind = codegen_filter_kind(f.kind);
    let args = f.args.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |a| quote! { ::core::option::Option::Some(::llm_tool::__md_tmpl::__private::Cow::Borrowed(#a)) },
    );
    quote! {
        ::llm_tool::__md_tmpl::compiled::ParsedFilter {
            kind: #kind,
            args: #args,
        }
    }
}

pub(crate) fn codegen_filter_kind(k: md_tmpl::compiled::FilterKind) -> proc_macro2::TokenStream {
    use md_tmpl::compiled::FilterKind;
    match k {
        FilterKind::Upper => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Upper },
        FilterKind::Lower => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Lower },
        FilterKind::Trim => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Trim },
        FilterKind::Fixed => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Fixed },
        FilterKind::Join => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Join },
        FilterKind::Limit => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Limit },
        FilterKind::Add => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Add },
        FilterKind::Sub => quote! { ::llm_tool::__md_tmpl::compiled::FilterKind::Sub },
    }
}

pub(crate) fn codegen_condition(c: &md_tmpl::compiled::Condition) -> proc_macro2::TokenStream {
    use md_tmpl::compiled::Condition;
    match c {
        Condition::Truthy(v) => {
            let operand_tokens = codegen_condition_operand(v);
            quote! {
                ::llm_tool::__md_tmpl::compiled::Condition::Truthy(#operand_tokens)
            }
        }
        Condition::Comparison { left, op, right } => {
            let op_tokens = codegen_comparison_op(*op);
            let left_tokens = codegen_condition_operand(left);
            let right_tokens = codegen_condition_operand(right);
            quote! {
                ::llm_tool::__md_tmpl::compiled::Condition::Comparison {
                    left: #left_tokens,
                    op: #op_tokens,
                    right: #right_tokens,
                }
            }
        }
    }
}

pub(crate) fn codegen_compiled_path(
    path: &md_tmpl::compiled::CompiledPath,
) -> proc_macro2::TokenStream {
    let raw = path.as_str();
    quote! { ::llm_tool::__md_tmpl::compiled::CompiledPath::compile(#raw) }
}

pub(crate) fn codegen_compiled_expr(
    expr: &md_tmpl::compiled::CompiledExpr,
) -> proc_macro2::TokenStream {
    use md_tmpl::compiled::CompiledExpr;
    match expr {
        CompiledExpr::Path(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::CompiledExpr::Path(#path_tokens) }
        }
        CompiledExpr::Idx(binding) => {
            quote! { ::llm_tool::__md_tmpl::compiled::CompiledExpr::Idx(::llm_tool::__md_tmpl::__private::String::from(#binding)) }
        }
        CompiledExpr::Len(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::CompiledExpr::Len(#path_tokens) }
        }
        CompiledExpr::Kind(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::CompiledExpr::Kind(#path_tokens) }
        }
        CompiledExpr::Has(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::CompiledExpr::Has(#path_tokens) }
        }
    }
}

pub(crate) fn codegen_condition_operand(
    op: &md_tmpl::compiled::ConditionOperand,
) -> proc_macro2::TokenStream {
    use md_tmpl::compiled::ConditionOperand;
    match op {
        ConditionOperand::Literal(val) => {
            let val_tokens = codegen_value(val);
            quote! { ::llm_tool::__md_tmpl::compiled::ConditionOperand::Literal(#val_tokens) }
        }
        ConditionOperand::Path { path, filters } => {
            let path_tokens = codegen_compiled_path(path);
            let filters_tokens = filters.iter().map(codegen_parsed_filter);
            quote! {
                ::llm_tool::__md_tmpl::compiled::ConditionOperand::Path {
                    path: #path_tokens,
                    filters: ::llm_tool::__md_tmpl::__private::vec![#(#filters_tokens),*],
                }
            }
        }
        ConditionOperand::Idx(binding) => {
            quote! { ::llm_tool::__md_tmpl::compiled::ConditionOperand::Idx(::llm_tool::__md_tmpl::__private::String::from(#binding)) }
        }
        ConditionOperand::Len(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::ConditionOperand::Len(#path_tokens) }
        }
        ConditionOperand::Kind(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::ConditionOperand::Kind(#path_tokens) }
        }
        ConditionOperand::Has(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__md_tmpl::compiled::ConditionOperand::Has(#path_tokens) }
        }
    }
}

pub(crate) fn codegen_comparison_op(
    op: md_tmpl::compiled::ComparisonOp,
) -> proc_macro2::TokenStream {
    use md_tmpl::compiled::ComparisonOp;
    match op {
        ComparisonOp::Eq => quote! { ::llm_tool::__md_tmpl::compiled::ComparisonOp::Eq },
        ComparisonOp::Ne => quote! { ::llm_tool::__md_tmpl::compiled::ComparisonOp::Ne },
        ComparisonOp::Le => quote! { ::llm_tool::__md_tmpl::compiled::ComparisonOp::Le },
        ComparisonOp::Ge => quote! { ::llm_tool::__md_tmpl::compiled::ComparisonOp::Ge },
        ComparisonOp::Lt => quote! { ::llm_tool::__md_tmpl::compiled::ComparisonOp::Lt },
        ComparisonOp::Gt => quote! { ::llm_tool::__md_tmpl::compiled::ComparisonOp::Gt },
    }
}

pub(crate) fn codegen_compiled_inline_template(
    t: &md_tmpl::compiled::CompiledInlineTemplate,
) -> proc_macro2::TokenStream {
    let segments_tokens = t.segments.iter().map(codegen_segment);
    let decls_tokens = t.declarations.iter().map(codegen_var_decl);
    quote! {
        ::llm_tool::__md_tmpl::compiled::CompiledInlineTemplate {
            segments: ::llm_tool::__md_tmpl::__private::Arc::from([#(#segments_tokens),*]),
            declarations: ::llm_tool::__md_tmpl::__private::Arc::from([#(#decls_tokens),*]),
        }
    }
}

pub(crate) fn codegen_var_decl(d: &md_tmpl::VarDecl) -> proc_macro2::TokenStream {
    let name = &d.name;
    let type_tokens = codegen_var_type(&d.var_type);
    let default_tokens = if let Some(v) = &d.default_value {
        let v_tokens = codegen_value(v);
        quote! { ::core::option::Option::Some(#v_tokens) }
    } else {
        quote! { ::core::option::Option::None }
    };
    quote! {
        ::llm_tool::__md_tmpl::VarDecl {
            name: ::llm_tool::__md_tmpl::__private::String::from(#name),
            var_type: #type_tokens,
            default_value: #default_tokens,
        }
    }
}

pub(crate) fn codegen_value(v: &md_tmpl::Value) -> proc_macro2::TokenStream {
    use md_tmpl::Value;
    match v {
        Value::Str(s) => {
            quote! { ::llm_tool::__md_tmpl::Value::Str(::llm_tool::__md_tmpl::__private::String::from(#s)) }
        }
        Value::Int(i) => quote! { ::llm_tool::__md_tmpl::Value::Int(#i) },
        Value::Float(f) => quote! { ::llm_tool::__md_tmpl::Value::Float(#f) },
        Value::Bool(b) => quote! { ::llm_tool::__md_tmpl::Value::Bool(#b) },
        Value::List(l) => {
            let items = l.iter().map(codegen_value);
            quote! { ::llm_tool::__md_tmpl::Value::List(::llm_tool::__md_tmpl::__private::Arc::new(::llm_tool::__md_tmpl::__private::vec![#(#items),*])) }
        }
        Value::Struct(d) => {
            let entries = d.iter().map(|(k, v)| {
                let v_tokens = codegen_value(v);
                quote! { (::llm_tool::__md_tmpl::__private::String::from(#k), #v_tokens) }
            });
            quote! {
                ::llm_tool::__md_tmpl::Value::Struct(
                    ::llm_tool::__md_tmpl::__private::Arc::new([#(#entries),*].into_iter().collect())
                )
            }
        }
        Value::Tmpl(_) => {
            quote! {
                compile_error!("Value::Tmpl cannot be used as a compile-time constant literal")
            }
        }
        Value::None => quote! { ::llm_tool::__md_tmpl::Value::None },
    }
}

pub(crate) fn codegen_var_type(t: &md_tmpl::VarType) -> proc_macro2::TokenStream {
    use md_tmpl::VarType;
    match t {
        VarType::Str => quote! { ::llm_tool::__md_tmpl::VarType::Str },
        VarType::Bool => quote! { ::llm_tool::__md_tmpl::VarType::Bool },
        VarType::Int => quote! { ::llm_tool::__md_tmpl::VarType::Int },
        VarType::Float => quote! { ::llm_tool::__md_tmpl::VarType::Float },
        VarType::List(fields) => {
            let fields_tokens = fields.iter().map(codegen_var_decl);
            quote! { ::llm_tool::__md_tmpl::VarType::List(::llm_tool::__md_tmpl::__private::vec![#(#fields_tokens),*]) }
        }
        VarType::Struct(fields) => {
            let fields_tokens = fields.iter().map(codegen_var_decl);
            quote! { ::llm_tool::__md_tmpl::VarType::Struct(::llm_tool::__md_tmpl::__private::vec![#(#fields_tokens),*]) }
        }
        VarType::Enum(variants) => {
            let variants_tokens = variants.iter().map(codegen_variant_decl);
            quote! { ::llm_tool::__md_tmpl::VarType::Enum(::llm_tool::__md_tmpl::__private::vec![#(#variants_tokens),*]) }
        }
        VarType::Tmpl(fields) => {
            let fields_tokens = fields.iter().map(codegen_var_decl);
            quote! { ::llm_tool::__md_tmpl::VarType::Tmpl(::llm_tool::__md_tmpl::__private::vec![#(#fields_tokens),*]) }
        }
        VarType::Option(inner) => {
            let inner_tokens = codegen_var_type(inner);
            quote! { ::llm_tool::__md_tmpl::VarType::Option(::llm_tool::__md_tmpl::__private::Box::new(#inner_tokens)) }
        }
    }
}

pub(crate) fn codegen_variant_decl(v: &md_tmpl::VariantDecl) -> proc_macro2::TokenStream {
    let name = &v.name;
    let fields_tokens = v.fields.iter().map(codegen_var_decl);
    quote! {
        ::llm_tool::__md_tmpl::VariantDecl {
            name: ::llm_tool::__md_tmpl::__private::String::from(#name),
            fields: ::llm_tool::__md_tmpl::__private::vec![#(#fields_tokens),*],
        }
    }
}

pub(crate) fn codegen_template(
    ast: &crate::template_compile::CompiledTemplateAst,
) -> proc_macro2::TokenStream {
    let segments_tokens = ast.segments.iter().map(codegen_segment);
    let decls_tokens = ast.frontmatter.declarations.iter().map(codegen_var_decl);
    let hash = ast.source_hash;
    let name_tokens = ast.frontmatter.name.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |n| quote! { ::core::option::Option::Some(#n) },
    );
    let desc_tokens = ast.frontmatter.description.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |d| quote! { ::core::option::Option::Some(#d) },
    );
    quote! {
        ::llm_tool::__md_tmpl::Template::from_precompiled(
            &::llm_tool::__md_tmpl::PrecompiledTemplateData {
                segments: &[#(#segments_tokens),*],
                declared_variables: &[#(#decls_tokens),*],
                inline_templates: &[],
                source_hash: #hash,
                consts: &[],
                imported_consts: &[],
                name: #name_tokens,
                description: #desc_tokens,
            }
        )
    }
}
