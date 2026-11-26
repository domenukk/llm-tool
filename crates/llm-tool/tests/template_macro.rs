use std::path::{Path, PathBuf};

/// Cargo's target directory for this build, derived from `CARGO_TARGET_TMPDIR`
/// (`<target-dir>/tmp`), so `--target-dir`, `CARGO_TARGET_DIR`, and
/// `build.target-dir` are all honored.
fn resolve_target_dir() -> PathBuf {
    let tmp_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    tmp_dir
        .parent()
        .map_or_else(|| tmp_dir.to_path_buf(), Path::to_path_buf)
}

#[test]
fn template_compile_fail_tests() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_src = manifest_dir.join("tools/dynamic_desc_test.tmpl.md");

    // trybuild generates a test package under <target_dir>/tests/trybuild/llm-tool/
    // Copy the test fixture so `tools/dynamic_desc_test.tmpl.md` is found relative to CARGO_MANIFEST_DIR in that crate.
    let target_dir = resolve_target_dir();
    let dest_dir = target_dir.join("tests/trybuild/llm-tool/tools");
    std::fs::create_dir_all(&dest_dir).expect("Failed to create trybuild tools dir");
    std::fs::copy(&fixture_src, dest_dir.join("dynamic_desc_test.tmpl.md"))
        .expect("Failed to copy dynamic_desc_test.tmpl.md fixture to trybuild directory");

    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/template_missing_context.rs");
    t.compile_fail("tests/ui/prompt_response_unsupported.rs");
}
