use llm_tool::llm_tool;

#[llm_tool]
fn no_docs(
    /// A value.
    x: i64,
) -> Result<String, String> {
    Ok(format!("{x}"))
}

fn main() {}
