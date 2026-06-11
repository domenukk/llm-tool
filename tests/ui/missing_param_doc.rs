use llm_tool::llm_tool;

/// Has a parameter without a doc comment.
#[llm_tool]
fn undocumented_param(
    /// This one is fine.
    a: i64,
    b: i64,
) -> Result<String, String> {
    Ok(format!("{}", a + b))
}

fn main() {}
