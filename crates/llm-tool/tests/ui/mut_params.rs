use llm_tool::{llm_prompt, llm_resource, llm_tool};

/// Mutates its inputs.
#[llm_tool]
fn modify_string(
    /// Input string to modify.
    mut input: String,
    /// Number of times to repeat.
    mut count: u32,
) -> Result<String, String> {
    input.push_str("!");
    count += 1;
    Ok(format!("{input} {count}"))
}

/// Prompt with mutable params.
#[llm_prompt]
fn modify_prompt(
    /// Input name.
    mut name: String,
) -> Result<String, String> {
    name.push_str("!");
    Ok(format!("Hello {name}"))
}

/// Resource with mutable params.
#[llm_resource(uri = "test://{val}")]
fn modify_resource(
    mut val: String,
) -> Result<String, String> {
    val.push_str("!");
    Ok(val)
}

fn main() {
    assert_eq!(std::mem::size_of::<ModifyString>(), 0);
    assert_eq!(std::mem::size_of::<ModifyPrompt>(), 0);
    assert_eq!(std::mem::size_of::<ModifyResource>(), 0);
}
