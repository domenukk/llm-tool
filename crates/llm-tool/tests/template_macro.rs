#![cfg(feature = "md-tmpl")]

use std::path::PathBuf;

#[test]
fn template_compile_fail_tests() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_src = manifest_dir.join("tools/dynamic_desc_test.tmpl.md");

    // trybuild generates a test package under target/tests/trybuild/llm-tool/
    // Copy the test fixture so `tools/dynamic_desc_test.tmpl.md` is found in that crate.
    let candidate_dirs = [
        manifest_dir.join("target/tests/trybuild/llm-tool/tools"),
        manifest_dir.join("../../target/tests/trybuild/llm-tool/tools"),
        PathBuf::from("target/tests/trybuild/llm-tool/tools"),
    ];
    for dir in &candidate_dirs {
        if std::fs::create_dir_all(dir).is_ok() {
            let _ = std::fs::copy(&fixture_src, dir.join("dynamic_desc_test.tmpl.md"));
        }
    }

    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/template_missing_context.rs");
    t.compile_fail("tests/ui/prompt_response_unsupported.rs");
}
