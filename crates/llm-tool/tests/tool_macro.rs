#[test]
fn tool_macro_compile_tests() {
    let t = trybuild::TestCases::new();
    // Compile-pass cases:
    t.pass("tests/ui/basic_sync.rs");
    t.pass("tests/ui/basic_async.rs");
    t.pass("tests/ui/option_params.rs");
    t.pass("tests/ui/zero_params.rs");
    t.pass("tests/ui/context_passthrough.rs");
    t.pass("tests/ui/multiline_doc.rs");
    t.pass("tests/ui/param_multiline_doc.rs");
    t.pass("tests/ui/str_ref_params.rs");
    t.pass("tests/ui/typed_return.rs");
    t.pass("tests/ui/context_explicit_opt_in.rs");
    t.pass("tests/ui/wrong_return_type.rs");
    t.pass("tests/ui/mut_params.rs");
    // Compile-fail cases:
    t.compile_fail("tests/ui/missing_doc.rs");
    t.compile_fail("tests/ui/missing_param_doc.rs");
    t.compile_fail("tests/ui/self_param.rs");
    t.compile_fail("tests/ui/context_by_value.rs");
    t.compile_fail("tests/ui/conflict_description_and_description_file.rs");
    t.compile_fail("tests/ui/env_without_description.rs");
    t.compile_fail("tests/ui/generic_fn.rs");
    t.compile_fail("tests/ui/unknown_attr_key.rs");
    t.compile_fail("tests/ui/resource_missing_uri.rs");
}

use llm_tool::{RustTool, ToolEffect, llm_tool};

/// Read only tool
#[llm_tool(effect = read_only)]
fn read_only_tool() -> Result<String, String> {
    Ok("read".into())
}

/// Mutating tool
#[llm_tool(effect = mutating)]
fn mutating_tool() -> Result<String, String> {
    Ok("mut".into())
}

/// Destructive tool
#[llm_tool(effect = destructive)]
fn destructive_tool() -> Result<String, String> {
    Ok("dest".into())
}

/// Legacy idempotent true tool
#[llm_tool(idempotent = true)]
fn legacy_idempotent_tool() -> Result<String, String> {
    Ok("legacy".into())
}

/// Legacy idempotent false tool
#[llm_tool(idempotent = false)]
fn legacy_non_idempotent_tool() -> Result<String, String> {
    Ok("legacy_false".into())
}

/// Default tool
#[llm_tool]
fn default_tool() -> Result<String, String> {
    Ok("default".into())
}

#[test]
fn test_tool_effect_macro_derivation() {
    assert_eq!(ReadOnlyTool::EFFECT, ToolEffect::ReadOnly);
    const {
        assert!(ReadOnlyTool::IDEMPOTENT);
    }
    assert!(ReadOnlyTool.is_idempotent());

    assert_eq!(MutatingTool::EFFECT, ToolEffect::Mutating);
    const {
        assert!(!MutatingTool::IDEMPOTENT);
    }
    assert!(!MutatingTool.is_idempotent());

    assert_eq!(DestructiveTool::EFFECT, ToolEffect::Destructive);
    const {
        assert!(!DestructiveTool::IDEMPOTENT);
    }
    assert!(!DestructiveTool.is_idempotent());

    assert_eq!(LegacyIdempotentTool::EFFECT, ToolEffect::ReadOnly);
    const {
        assert!(LegacyIdempotentTool::IDEMPOTENT);
    }
    assert!(LegacyIdempotentTool.is_idempotent());

    assert_eq!(LegacyNonIdempotentTool::EFFECT, ToolEffect::Mutating);
    const {
        assert!(!LegacyNonIdempotentTool::IDEMPOTENT);
    }
    assert!(!LegacyNonIdempotentTool.is_idempotent());

    assert_eq!(DefaultTool::EFFECT, ToolEffect::Mutating);
    const {
        assert!(!DefaultTool::IDEMPOTENT);
    }
    assert!(!DefaultTool.is_idempotent());
}
