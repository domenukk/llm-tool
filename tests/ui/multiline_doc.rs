// Verifies multi-line doc comments are preserved (not collapsed to one line).
use llm_tool::llm_tool;

/// First line of the description.
///
/// Second paragraph with more details.
#[llm_tool]
fn multiline_tool(
    /// A value.
    x: i64,
) -> Result<String, String> {
    Ok(format!("{x}"))
}

fn main() {
    // Verify the description preserves newlines.
    assert!(
        <MultilineTool as llm_tool::RustTool>::DESCRIPTION.contains('\n'),
        "multi-line doc comments should be joined by newlines, not spaces"
    );
}
