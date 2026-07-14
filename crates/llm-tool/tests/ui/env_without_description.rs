use llm_tool::llm_tool;

/// This tool has env() but no description_file or description — should fail.
#[llm_tool(env(API_KEY = "secret"))]
fn env_without_description(
    /// A param.
    x: String,
) -> Result<String, llm_tool::ToolError> {
    Ok(x)
}

fn main() {}
