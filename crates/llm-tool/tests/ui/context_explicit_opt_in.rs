use llm_tool::llm_tool;
use llm_tool::ToolContext;

type CustomContext<'a> = &'a ToolContext;

/// Echoes with explicit type-aliased context.
#[llm_tool]
fn explicit_echo(
    /// The greeting.
    greeting: String,
    #[llm_tool(context)]
    ctx: CustomContext<'_>,
) -> Result<String, String> {
    let _ctx = ctx;
    Ok(greeting)
}

fn main() {
    let _t = ExplicitEcho;
}
