use llm_tool::llm_tool;

/// This tool has env() but no prompt_file or prompt — should fail.
#[llm_tool(env(API_KEY = "secret"))]
fn env_without_prompt(
    /// A param.
    x: String,
) -> Result<String, llm_tool::ToolError> {
    Ok(x)
}

fn main() {}
