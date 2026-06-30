use llm_tool::llm_tool;

/// Looks up a user by email.
#[llm_tool]
fn lookup_user(
    /// The email address to look up.
    email: &str,
) -> Result<String, String> {
    Ok(format!("found: {email}"))
}

fn main() {
    assert!(std::mem::size_of::<LookupUser>() == 0);
}
