#![cfg(feature = "prompt-templates")]

#[test]
fn template_compile_fail_tests() {
    std::fs::write(
        "/tmp/dynamic_desc_test.tmpl.md",
        "---\nname: dynamic\nparams: [api_version = str]\n---\nRunning on {{ api_version }}\n",
    )
    .unwrap();

    let _guard = OnDrop(Some(|| {
        if let Err(e) = std::fs::remove_file("/tmp/dynamic_desc_test.tmpl.md") {
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("failed to clean up /tmp/dynamic_desc_test.tmpl.md: {e}");
            }
        }
    }));

    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/template_missing_context.rs");
}

struct OnDrop<F: FnOnce()>(Option<F>);
impl<F: FnOnce()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}
