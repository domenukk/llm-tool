use llm_tool::llm_tool;

/// Adds two numbers together.
#[llm_tool]
fn add_numbers(
    /// The first number.
    a: i64,
    /// The second number.
    b: i64,
) -> Result<String, String> {
    Ok(format!("{}", a + b))
}

fn main() {
    assert!(std::mem::size_of::<AddNumbers>() == 0);
}
