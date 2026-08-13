#![cfg(feature = "md-tmpl")]

use std::path::PathBuf;

#[test]
fn template_compile_fail_tests() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_src = manifest_dir.join("tools/dynamic_desc_test.tmpl.md");

    // trybuild generates a test package under <target_dir>/tests/trybuild/llm-tool/
    // Copy the test fixture so `tools/dynamic_desc_test.tmpl.md` is found relative to CARGO_MANIFEST_DIR in that crate.
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map_or_else(|_| manifest_dir.join("target"), PathBuf::from);
    let dest_dir = target_dir.join("tests/trybuild/llm-tool/tools");
    std::fs::create_dir_all(&dest_dir).expect("Failed to create trybuild tools dir");
    std::fs::copy(&fixture_src, dest_dir.join("dynamic_desc_test.tmpl.md"))
        .expect("Failed to copy dynamic_desc_test.tmpl.md fixture to trybuild directory");

    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/template_missing_context.rs");
    t.compile_fail("tests/ui/prompt_response_unsupported.rs");
}
