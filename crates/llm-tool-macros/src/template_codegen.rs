//! Template codegen adapted from `prompt-templates-macros` for `llm-tool-macros`.
//!
//! This module mirrors the codegen functions from `prompt_templates_macros::codegen`,
//! but rewrites all generated runtime paths from `::prompt_templates::` to
//! `::llm_tool::__prompt_templates::` so that downstream crates only need to depend
//! on `llm_tool` (which re-exports `prompt_templates` under the hidden
//! `__prompt_templates` module).
//!
//! Function signatures still reference `prompt_templates::` types directly because
//! `llm-tool-macros` has a direct build-time dependency on `prompt-templates`.

use quote::quote;

pub(crate) fn codegen_segment(
    seg: &prompt_templates::compiled::Segment,
) -> proc_macro2::TokenStream {
    use prompt_templates::compiled::Segment;
    match seg {
        Segment::Static(s) => quote! {
            ::llm_tool::__prompt_templates::compiled::Segment::Static(::llm_tool::__prompt_templates::__private::Cow::Borrowed(#s))
        },
        Segment::Expr { expr, filters } => {
            let filters_tokens = filters.iter().map(codegen_parsed_filter);
            let expr_tokens = codegen_compiled_expr(expr);
            quote! {
                ::llm_tool::__prompt_templates::compiled::Segment::Expr {
                    expr: #expr_tokens,
                    filters: ::llm_tool::__prompt_templates::__private::vec![#(#filters_tokens),*],
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
                ::llm_tool::__prompt_templates::compiled::Segment::ForLoop {
                    binding: ::llm_tool::__prompt_templates::__private::Cow::Borrowed(#binding),
                    list_path: #list_path_tokens,
                    body: ::llm_tool::__prompt_templates::__private::vec![#(#body_tokens),*],
                    else_body: ::llm_tool::__prompt_templates::__private::vec![#(#else_body_tokens),*],
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
                    (#cond_tokens, ::llm_tool::__prompt_templates::__private::vec![#(#body_tokens),*])
                }
            });
            let else_tokens = else_body.iter().map(codegen_segment);
            quote! {
                ::llm_tool::__prompt_templates::compiled::Segment::If {
                    branches: ::llm_tool::__prompt_templates::__private::vec![#(#branch_tokens),*],
                    else_body: ::llm_tool::__prompt_templates::__private::vec![#(#else_tokens),*],
                }
            }
        }
        Segment::Raw(s) => quote! {
            ::llm_tool::__prompt_templates::compiled::Segment::Raw(::llm_tool::__prompt_templates::__private::Cow::Borrowed(#s))
        },
        Segment::Comment(refs) => {
            quote! {
                ::llm_tool::__prompt_templates::compiled::Segment::Comment(::llm_tool::__prompt_templates::__private::vec![#(::llm_tool::__prompt_templates::__private::Cow::Borrowed(#refs)),*])
            }
        }
        Segment::Include(inc) => codegen_segment_include(inc),
        Segment::Match { expr, arms, .. } => codegen_segment_match(expr, arms),
    }
}

pub(crate) fn codegen_segment_include(
    inc: &prompt_templates::compiled::CompiledInclude,
) -> proc_macro2::TokenStream {
    let path = &inc.path;
    let with_vars = inc.with_vars.iter().map(|(k, v)| {
        quote! { (::llm_tool::__prompt_templates::__private::Cow::Borrowed(#k), ::llm_tool::__prompt_templates::__private::Cow::Borrowed(#v)) }
    });
    let for_each = inc.for_each.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |(b, l)| quote! { ::core::option::Option::Some((::llm_tool::__prompt_templates::__private::Cow::Borrowed(#b), ::llm_tool::__prompt_templates::__private::Cow::Borrowed(#l))) },
    );
    let inline_compiled = inc.inline_compiled.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |ic| {
            let ic_tokens = codegen_compiled_inline_template(ic);
            quote! { ::core::option::Option::Some(#ic_tokens) }
        },
    );
    quote! {
        ::llm_tool::__prompt_templates::compiled::Segment::Include(
            ::llm_tool::__prompt_templates::compiled::CompiledInclude {
                path: ::llm_tool::__prompt_templates::__private::Cow::Borrowed(#path),
                with_vars: ::llm_tool::__prompt_templates::__private::vec![#(#with_vars),*],
                for_each: #for_each,
                inline_compiled: #inline_compiled,
            }
        )
    }
}

pub(crate) fn codegen_segment_match(
    expr: &prompt_templates::compiled::CompiledPath,
    arms: &[(
        Vec<std::borrow::Cow<'static, str>>,
        Vec<prompt_templates::compiled::Segment>,
    )],
) -> proc_macro2::TokenStream {
    let arm_tokens = arms.iter().map(|(variants, body)| {
        let body_tokens = body.iter().map(codegen_segment);
        let variant_tokens = variants.iter().map(|v| {
            quote! { ::llm_tool::__prompt_templates::__private::Cow::Borrowed(#v) }
        });
        quote! {
            (::llm_tool::__prompt_templates::__private::vec![#(#variant_tokens),*], ::llm_tool::__prompt_templates::__private::vec![#(#body_tokens),*])
        }
    });
    let expr_tokens = codegen_compiled_path(expr);
    quote! {
        ::llm_tool::__prompt_templates::compiled::Segment::Match {
            expr: #expr_tokens,
            arms: ::llm_tool::__prompt_templates::__private::vec![#(#arm_tokens),*],
        }
    }
}

pub(crate) fn codegen_parsed_filter(
    f: &prompt_templates::compiled::ParsedFilter,
) -> proc_macro2::TokenStream {
    let kind = codegen_filter_kind(f.kind);
    let args = f.args.as_ref().map_or_else(
        || quote! { ::core::option::Option::None },
        |a| quote! { ::core::option::Option::Some(::llm_tool::__prompt_templates::__private::Cow::Borrowed(#a)) },
    );
    quote! {
        ::llm_tool::__prompt_templates::compiled::ParsedFilter {
            kind: #kind,
            args: #args,
        }
    }
}

pub(crate) fn codegen_filter_kind(
    k: prompt_templates::compiled::FilterKind,
) -> proc_macro2::TokenStream {
    use prompt_templates::compiled::FilterKind;
    match k {
        FilterKind::Upper => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Upper },
        FilterKind::Lower => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Lower },
        FilterKind::Trim => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Trim },
        FilterKind::Fixed => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Fixed },
        FilterKind::Join => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Join },
        FilterKind::Limit => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Limit },
        FilterKind::Add => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Add },
        FilterKind::Sub => quote! { ::llm_tool::__prompt_templates::compiled::FilterKind::Sub },
    }
}

pub(crate) fn codegen_condition(
    c: &prompt_templates::compiled::Condition,
) -> proc_macro2::TokenStream {
    use prompt_templates::compiled::Condition;
    match c {
        Condition::Truthy(v) => {
            let operand_tokens = codegen_condition_operand(v);
            quote! {
                ::llm_tool::__prompt_templates::compiled::Condition::Truthy(#operand_tokens)
            }
        }
        Condition::Comparison { left, op, right } => {
            let op_tokens = codegen_comparison_op(*op);
            let left_tokens = codegen_condition_operand(left);
            let right_tokens = codegen_condition_operand(right);
            quote! {
                ::llm_tool::__prompt_templates::compiled::Condition::Comparison {
                    left: #left_tokens,
                    op: #op_tokens,
                    right: #right_tokens,
                }
            }
        }
    }
}

pub(crate) fn codegen_compiled_path(
    path: &prompt_templates::compiled::CompiledPath,
) -> proc_macro2::TokenStream {
    let raw = path.as_str();
    quote! { ::llm_tool::__prompt_templates::compiled::CompiledPath::compile(#raw) }
}

pub(crate) fn codegen_compiled_expr(
    expr: &prompt_templates::compiled::CompiledExpr,
) -> proc_macro2::TokenStream {
    use prompt_templates::compiled::CompiledExpr;
    match expr {
        CompiledExpr::Path(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::CompiledExpr::Path(#path_tokens) }
        }
        CompiledExpr::Idx(binding) => {
            quote! { ::llm_tool::__prompt_templates::compiled::CompiledExpr::Idx(::llm_tool::__prompt_templates::__private::String::from(#binding)) }
        }
        CompiledExpr::Len(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::CompiledExpr::Len(#path_tokens) }
        }
        CompiledExpr::Kind(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::CompiledExpr::Kind(#path_tokens) }
        }
        CompiledExpr::Has(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::CompiledExpr::Has(#path_tokens) }
        }
    }
}

pub(crate) fn codegen_condition_operand(
    op: &prompt_templates::compiled::ConditionOperand,
) -> proc_macro2::TokenStream {
    use prompt_templates::compiled::ConditionOperand;
    match op {
        ConditionOperand::Literal(val) => {
            let val_tokens = codegen_value(val);
            quote! { ::llm_tool::__prompt_templates::compiled::ConditionOperand::Literal(#val_tokens) }
        }
        ConditionOperand::Path { path, filters } => {
            let path_tokens = codegen_compiled_path(path);
            let filters_tokens = filters.iter().map(codegen_parsed_filter);
            quote! {
                ::llm_tool::__prompt_templates::compiled::ConditionOperand::Path {
                    path: #path_tokens,
                    filters: ::llm_tool::__prompt_templates::__private::vec![#(#filters_tokens),*],
                }
            }
        }
        ConditionOperand::Idx(binding) => {
            quote! { ::llm_tool::__prompt_templates::compiled::ConditionOperand::Idx(::llm_tool::__prompt_templates::__private::String::from(#binding)) }
        }
        ConditionOperand::Len(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::ConditionOperand::Len(#path_tokens) }
        }
        ConditionOperand::Kind(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::ConditionOperand::Kind(#path_tokens) }
        }
        ConditionOperand::Has(path) => {
            let path_tokens = codegen_compiled_path(path);
            quote! { ::llm_tool::__prompt_templates::compiled::ConditionOperand::Has(#path_tokens) }
        }
    }
}

pub(crate) fn codegen_comparison_op(
    op: prompt_templates::compiled::ComparisonOp,
) -> proc_macro2::TokenStream {
    use prompt_templates::compiled::ComparisonOp;
    match op {
        ComparisonOp::Eq => quote! { ::llm_tool::__prompt_templates::compiled::ComparisonOp::Eq },
        ComparisonOp::Ne => quote! { ::llm_tool::__prompt_templates::compiled::ComparisonOp::Ne },
        ComparisonOp::Le => quote! { ::llm_tool::__prompt_templates::compiled::ComparisonOp::Le },
        ComparisonOp::Ge => quote! { ::llm_tool::__prompt_templates::compiled::ComparisonOp::Ge },
        ComparisonOp::Lt => quote! { ::llm_tool::__prompt_templates::compiled::ComparisonOp::Lt },
        ComparisonOp::Gt => quote! { ::llm_tool::__prompt_templates::compiled::ComparisonOp::Gt },
    }
}

pub(crate) fn codegen_compiled_inline_template(
    t: &prompt_templates::compiled::CompiledInlineTemplate,
) -> proc_macro2::TokenStream {
    let segments_tokens = t.segments.iter().map(codegen_segment);
    let decls_tokens = t.declarations.iter().map(codegen_var_decl);
    quote! {
        ::llm_tool::__prompt_templates::compiled::CompiledInlineTemplate {
            segments: ::llm_tool::__prompt_templates::__private::Arc::from([#(#segments_tokens),*]),
            declarations: ::llm_tool::__prompt_templates::__private::Arc::from([#(#decls_tokens),*]),
        }
    }
}

pub(crate) fn codegen_var_decl(d: &prompt_templates::VarDecl) -> proc_macro2::TokenStream {
    let name = &d.name;
    let type_tokens = codegen_var_type(&d.var_type);
    let default_tokens = if let Some(v) = &d.default_value {
        let v_tokens = codegen_value(v);
        quote! { ::core::option::Option::Some(#v_tokens) }
    } else {
        quote! { ::core::option::Option::None }
    };
    quote! {
        ::llm_tool::__prompt_templates::VarDecl {
            name: ::llm_tool::__prompt_templates::__private::String::from(#name),
            var_type: #type_tokens,
            default_value: #default_tokens,
        }
    }
}

pub(crate) fn codegen_value(v: &prompt_templates::Value) -> proc_macro2::TokenStream {
    use prompt_templates::Value;
    match v {
        Value::Str(s) => {
            quote! { ::llm_tool::__prompt_templates::Value::Str(::llm_tool::__prompt_templates::__private::String::from(#s)) }
        }
        Value::Int(i) => quote! { ::llm_tool::__prompt_templates::Value::Int(#i) },
        Value::Float(f) => quote! { ::llm_tool::__prompt_templates::Value::Float(#f) },
        Value::Bool(b) => quote! { ::llm_tool::__prompt_templates::Value::Bool(#b) },
        Value::List(l) => {
            let items = l.iter().map(codegen_value);
            quote! { ::llm_tool::__prompt_templates::Value::List(::llm_tool::__prompt_templates::__private::Arc::new(::llm_tool::__prompt_templates::__private::vec![#(#items),*])) }
        }
        Value::Struct(d) => {
            let entries = d.iter().map(|(k, v)| {
                let v_tokens = codegen_value(v);
                quote! { (::llm_tool::__prompt_templates::__private::String::from(#k), #v_tokens) }
            });
            quote! {
                ::llm_tool::__prompt_templates::Value::Struct(
                    ::llm_tool::__prompt_templates::__private::Arc::new([#(#entries),*].into_iter().collect())
                )
            }
        }
        Value::Tmpl(_) => {
            quote! {
                compile_error!("Value::Tmpl cannot be used as a compile-time constant literal")
            }
        }
        Value::None => quote! { ::llm_tool::__prompt_templates::Value::None },
    }
}

pub(crate) fn codegen_var_type(t: &prompt_templates::VarType) -> proc_macro2::TokenStream {
    use prompt_templates::VarType;
    match t {
        VarType::Str => quote! { ::llm_tool::__prompt_templates::VarType::Str },
        VarType::Bool => quote! { ::llm_tool::__prompt_templates::VarType::Bool },
        VarType::Int => quote! { ::llm_tool::__prompt_templates::VarType::Int },
        VarType::Float => quote! { ::llm_tool::__prompt_templates::VarType::Float },
        VarType::List(fields) => {
            let fields_tokens = fields.iter().map(codegen_var_decl);
            quote! { ::llm_tool::__prompt_templates::VarType::List(::llm_tool::__prompt_templates::__private::vec![#(#fields_tokens),*]) }
        }
        VarType::Struct(fields) => {
            let fields_tokens = fields.iter().map(codegen_var_decl);
            quote! { ::llm_tool::__prompt_templates::VarType::Struct(::llm_tool::__prompt_templates::__private::vec![#(#fields_tokens),*]) }
        }
        VarType::Enum(variants) => {
            let variants_tokens = variants.iter().map(codegen_variant_decl);
            quote! { ::llm_tool::__prompt_templates::VarType::Enum(::llm_tool::__prompt_templates::__private::vec![#(#variants_tokens),*]) }
        }
        VarType::Tmpl(fields) => {
            let fields_tokens = fields.iter().map(codegen_var_decl);
            quote! { ::llm_tool::__prompt_templates::VarType::Tmpl(::llm_tool::__prompt_templates::__private::vec![#(#fields_tokens),*]) }
        }
        VarType::Option(inner) => {
            let inner_tokens = codegen_var_type(inner);
            quote! { ::llm_tool::__prompt_templates::VarType::Option(::llm_tool::__prompt_templates::__private::Box::new(#inner_tokens)) }
        }
    }
}

pub(crate) fn codegen_variant_decl(v: &prompt_templates::VariantDecl) -> proc_macro2::TokenStream {
    let name = &v.name;
    let fields_tokens = v.fields.iter().map(codegen_var_decl);
    quote! {
        ::llm_tool::__prompt_templates::VariantDecl {
            name: ::llm_tool::__prompt_templates::__private::String::from(#name),
            fields: ::llm_tool::__prompt_templates::__private::vec![#(#fields_tokens),*],
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
        ::llm_tool::__prompt_templates::Template::from_precompiled(
            &[#(#segments_tokens),*],
            &[#(#decls_tokens),*],
            &[],
            #hash,
            &[],
            &[],
            #name_tokens,
            #desc_tokens,
        )
    }
}
