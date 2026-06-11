use llm_tool::llm_tool;

/// Fetches data asynchronously.
#[llm_tool]
async fn fetch_data(
    /// The URL to fetch.
    url: String,
) -> Result<String, String> {
    Ok(format!("fetched: {url}"))
}

fn main() {
    assert!(std::mem::size_of::<FetchData>() == 0);
}
