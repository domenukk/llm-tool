use llm_tool::llm_tool;

// The template has variables, but we did NOT provide context = ... or params(...).
#[llm_tool(prompt_file = "/tmp/dynamic_desc_test.tmpl.md")]
fn missing_context(
    x: i64,
) -> Result<String, String> {
    Ok(format!("{x}"))
}

fn main() {}
