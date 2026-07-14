use llm_tool::llm_tool;

/// Uses an unknown attribute key.
#[llm_tool(bogus = "nope")]
fn my_tool() -> String {
    String::new()
}

fn main() {}
