use llm_tool::llm_tool;

/// Returns a greeting with no parameters.
#[llm_tool]
fn greet() -> Result<String, String> {
    Ok("Hello!".to_string())
}

fn main() {
    assert!(std::mem::size_of::<Greet>() == 0);
}
