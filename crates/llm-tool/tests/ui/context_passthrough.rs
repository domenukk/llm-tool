// Verifies that ToolContext parameters are accepted and passed through.
use llm_tool::llm_tool;

/// Echoes the greeting using the context.
#[llm_tool]
fn context_echo(
    /// The greeting text.
    greeting: String,
    ctx: &llm_tool::ToolContext,
) -> Result<String, String> {
    let _ctx = ctx;
    Ok(greeting)
}

fn main() {
    let _t = ContextEcho;
}
