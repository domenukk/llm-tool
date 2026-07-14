use llm_tool::llm_prompt;

/// A prompt cannot use response templates.
#[llm_prompt(response = "unused {{ x }}")]
fn greet(
    /// the name to greet
    name: String,
) -> String {
    format!("Hi {name}")
}

fn main() {}
