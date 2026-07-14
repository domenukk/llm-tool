use llm_tool::llm_tool;

/// A generic tool is not supported.
#[llm_tool]
fn generic_tool<T: std::fmt::Display>(
    /// the value to render
    value: T,
) -> String {
    value.to_string()
}

fn main() {}
