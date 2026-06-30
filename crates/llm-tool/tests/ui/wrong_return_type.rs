use llm_tool::llm_tool;

/// Does something with a bare return type (no Result wrapper).
#[llm_tool]
fn bare_return(
    /// A value.
    x: i64,
) -> i64 {
    x + 1
}

fn main() {
    assert!(std::mem::size_of::<BareReturn>() == 0);
}
