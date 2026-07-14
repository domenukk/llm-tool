use llm_tool::llm_tool;

#[llm_tool(description = "a", description_file = "b")]
fn both_description_and_file(
    /// A param.
    x: String,
) -> Result<String, llm_tool::ToolError> {
    Ok(x)
}

fn main() {}
