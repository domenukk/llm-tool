use llm_tool::llm_tool;

/// Returns the sum of two numbers as a typed i64.
#[llm_tool]
fn typed_add(
    /// The first number.
    a: i64,
    /// The second number.
    b: i64,
) -> Result<i64, String> {
    Ok(a + b)
}

fn main() {
    assert!(std::mem::size_of::<TypedAdd>() == 0);
}
