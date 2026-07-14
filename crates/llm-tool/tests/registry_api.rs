//! Integration tests for the unified registry API introduced in 0.7.0.
//!
//! These lock down the public lookup surface shared by [`ToolRegistry`],
//! [`PromptRegistry`], and [`ResourceRegistry`]: `contains`, `definition`,
//! `iter`/`IntoIterator`, and the `Option<Result<..>>` dispatch/render/read
//! shape where `None` means "not found" and the inner `Result` carries the
//! execution outcome.

use llm_tool::{
    PromptRegistry, ResourceRegistry, ToolContext, ToolError, ToolRegistry, llm_prompt,
    llm_resource, llm_tool,
};

// ── Fixtures ─────────────────────────────────────────────────────────

/// A trivial tool used to exercise the tool registry lookups.
#[llm_tool]
fn echo(
    /// Text to echo back.
    text: String,
) -> Result<String, ToolError> {
    Ok(text)
}

/// A second tool, so iteration and counting cover more than one entry.
#[llm_tool]
fn shout(
    /// Text to shout.
    text: String,
) -> Result<String, ToolError> {
    Ok(text.to_uppercase())
}

/// A prompt used to exercise the prompt registry lookups.
#[llm_prompt]
fn greet(
    /// Who to greet.
    who: String,
) -> Result<String, ToolError> {
    Ok(format!("Hello, {who}!"))
}

/// A resource used to exercise the resource registry lookups.
#[llm_resource(
    uri = "file:///data/{key}.txt",
    name = "blob",
    description = "A keyed blob",
    mime_type = "text/plain"
)]
fn blob(key: String) -> Result<String, ToolError> {
    Ok(format!("value for {key}"))
}

// ── ToolRegistry ─────────────────────────────────────────────────────

#[test]
fn tool_registry_contains_and_definition() {
    let reg = ToolRegistry::new().with_tool(Echo).with_tool(Shout);

    assert!(reg.contains("echo"));
    assert!(reg.contains("shout"));
    assert!(!reg.contains("missing"));

    let def = reg.definition("echo").expect("echo definition present");
    assert_eq!(def.name, "echo");
    assert!(reg.definition("missing").is_none());

    assert_eq!(reg.len(), 2);
    assert!(!reg.is_empty());
}

#[test]
fn tool_registry_iter_matches_definitions() {
    let reg = ToolRegistry::new().with_tool(Echo).with_tool(Shout);

    let mut from_iter: Vec<&str> = reg.iter().map(|(name, _def)| name).collect();
    from_iter.sort_unstable();
    assert_eq!(from_iter, ["echo", "shout"]);

    // `iter()` and `definitions()` must agree on the set of names.
    let mut from_defs: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
    from_defs.sort();
    assert_eq!(from_defs, ["echo", "shout"]);

    // `IntoIterator` for `&ToolRegistry` yields the same pairs.
    let via_into: usize = (&reg).into_iter().count();
    assert_eq!(via_into, 2);
}

#[tokio::test]
async fn tool_registry_dispatch_unknown_is_none() {
    let reg = ToolRegistry::new().with_tool(Echo);
    let ctx = ToolContext::new();

    let out = reg
        .dispatch("echo", serde_json::json!({"text": "hi"}), &ctx)
        .await
        .expect("known tool yields Some")
        .expect("dispatch succeeds");
    assert_eq!(out.content(), "hi");

    // Unknown tool → None, never a spurious Err.
    assert!(
        reg.dispatch("nope", serde_json::json!({}), &ctx)
            .await
            .is_none()
    );
}

#[test]
fn tool_registry_try_register_reports_success() {
    let mut reg = ToolRegistry::new();
    reg.try_register(Echo).expect("echo schema builds");
    assert!(reg.contains("echo"));
}

// ── PromptRegistry ───────────────────────────────────────────────────

#[test]
fn prompt_registry_contains_definition_and_counts() {
    let reg = PromptRegistry::new().with_prompt(Greet);

    assert!(reg.contains("greet"));
    assert!(!reg.contains("missing"));
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    assert!(PromptRegistry::new().is_empty());

    let def = reg.definition("greet").expect("greet definition present");
    assert_eq!(def.name, "greet");
    assert!(reg.definition("missing").is_none());
}

#[test]
fn prompt_registry_iter_matches_definitions() {
    let reg = PromptRegistry::new().with_prompt(Greet);

    let names: Vec<&str> = reg.iter().map(|(name, _def)| name).collect();
    assert_eq!(names, ["greet"]);

    let def_names: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
    assert_eq!(def_names, ["greet"]);

    assert_eq!((&reg).into_iter().count(), 1);
}

#[test]
fn prompt_registry_try_register_reports_success() {
    let mut reg = PromptRegistry::new();
    reg.try_register(Greet).expect("greet schema builds");
    assert!(reg.contains("greet"));
}

#[tokio::test]
async fn prompt_registry_render_known_and_unknown() {
    let reg = PromptRegistry::new().with_prompt(Greet);

    let out = reg
        .render("greet", serde_json::json!({"who": "World"}))
        .await
        .expect("known prompt yields Some")
        .expect("render succeeds");
    assert!(
        out.messages[0].content.contains("Hello, World!"),
        "unexpected rendered content: {:?}",
        out.messages
    );

    // Unknown prompt → None.
    assert!(reg.render("missing", serde_json::json!({})).await.is_none());
}

// ── ResourceRegistry ─────────────────────────────────────────────────

#[test]
fn resource_registry_contains_matches_and_definition() {
    let reg = ResourceRegistry::new().with_resource(Blob);

    // Resources are addressed by name for metadata lookups...
    assert!(reg.contains("blob"));
    assert!(!reg.contains("missing"));

    let def = reg.definition("blob").expect("blob definition present");
    assert_eq!(def.name, "blob");
    assert_eq!(def.uri_template, "file:///data/{key}.txt");
    assert!(reg.definition("missing").is_none());

    // ...but *matched* by URI for reads.
    assert!(reg.matches("file:///data/report.txt"));
    assert!(!reg.matches("file:///other/report.txt"));

    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    assert!(ResourceRegistry::new().is_empty());
}

#[test]
fn resource_registry_iter_matches_definitions() {
    let reg = ResourceRegistry::new().with_resource(Blob);

    let names: Vec<&str> = reg.iter().map(|(name, _def)| name).collect();
    assert_eq!(names, ["blob"]);

    let def_names: Vec<String> = reg.definitions().into_iter().map(|d| d.name).collect();
    assert_eq!(def_names, ["blob"]);

    assert_eq!((&reg).into_iter().count(), 1);
}

#[tokio::test]
async fn resource_registry_read_matching_and_non_matching() {
    let reg = ResourceRegistry::new().with_resource(Blob);

    let out = reg
        .read("file:///data/hello.txt")
        .await
        .expect("matching URI yields Some")
        .expect("read succeeds");
    // The read output should carry the resource contents for the extracted key.
    let json = serde_json::to_value(&out).expect("serialize resource output");
    let text = json["contents"][0]["text"]
        .as_str()
        .expect("resource output should contain a text content block");
    assert!(
        text.contains("value for hello"),
        "unexpected resource content: {json}"
    );

    // A URI that matches no template → None.
    assert!(reg.read("file:///nope/hello.json").await.is_none());
}
