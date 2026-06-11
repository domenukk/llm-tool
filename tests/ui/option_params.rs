use llm_tool::llm_tool;

/// Searches with optional filters.
#[llm_tool]
fn search(
    /// The search query.
    query: String,
    /// Maximum number of results.
    #[serde(default)]
    max_results: Option<i64>,
    /// Optional tag filter.
    #[serde(default)]
    tag: Option<String>,
) -> Result<String, String> {
    let max = max_results.unwrap_or(10);
    let tag_str = tag.unwrap_or_default();
    Ok(format!("query={query} max={max} tag={tag_str}"))
}

fn main() {
    assert!(std::mem::size_of::<Search>() == 0);
}
