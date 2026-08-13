#![cfg(feature = "md-tmpl")]

#[test]
fn template_compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/template_missing_context.rs");
    t.compile_fail("tests/ui/prompt_response_unsupported.rs");
}
