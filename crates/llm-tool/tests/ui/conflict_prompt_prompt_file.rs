use llm_tool::llm_tool;

#[llm_tool(prompt = "a", prompt_file = "b")]
fn both_prompt_and_file(
    /// A param.
    x: String,
) -> Result<String, llm_tool::ToolError> {
    Ok(x)
}

fn main() {}
