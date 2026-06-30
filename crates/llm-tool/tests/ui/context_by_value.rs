use llm_tool::llm_tool;

/// Echoes with context by value.
#[llm_tool]
fn context_echo_by_value(
    /// The greeting.
    greeting: String,
    ctx: llm_tool::ToolContext,
) -> Result<String, String> {
    let _ctx = ctx;
    Ok(greeting)
}

fn main() {}
