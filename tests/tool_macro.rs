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
    // Compile-fail cases:
    t.compile_fail("tests/ui/missing_doc.rs");
    t.compile_fail("tests/ui/missing_param_doc.rs");
    t.compile_fail("tests/ui/self_param.rs");
    t.compile_fail("tests/ui/context_by_value.rs");
}
