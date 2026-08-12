//! Integration tests for the unified registry API introduced in 0.7.0.
//!
//! These lock down the public lookup surface shared by [`ToolRegistry`],
//! [`PromptRegistry`], and [`ResourceRegistry`]: `contains`, `definition`,
//! `iter`/`IntoIterator`, and the flat `Result<.., ToolError>` dispatch/render/read
//! shape where a missing entry is `Err(ToolError::not_found(..))` (carrying
//! `error_kind == "not_registered"` metadata) and a present entry carries the
//! execution outcome.

use llm_tool::{
    PromptRegistry, RegistryItem, ResourceRegistry, ToolContext, ToolError, ToolRegistry,
    llm_prompt, llm_resource, llm_tool,
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

// ── RegistryItem ─────────────────────────────────────────────────────

#[test]
fn registry_item_enum_public_api() {
    assert_eq!(RegistryItem::Tool.to_string(), "tool");
    assert_eq!(RegistryItem::Prompt.to_string(), "prompt");
    assert_eq!(RegistryItem::Resource.to_string(), "resource");
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
async fn tool_registry_dispatch_unknown_is_not_found() {
    let reg = ToolRegistry::new().with_tool(Echo);
    let ctx = ToolContext::new();

    let out = reg
        .dispatch("echo", serde_json::json!({"text": "hi"}), &ctx)
        .await
        .expect("dispatch succeeds");
    assert_eq!(out.content(), "hi");

    // Unknown tool → not_found Err, never a spurious execution error.
    let err = reg
        .dispatch("nope", serde_json::json!({}), &ctx)
        .await
        .expect_err("unknown tool yields not_found error");
    assert!(err.is_not_found());
    assert_eq!(err.metadata()["error_kind"], "not_registered");
    assert!(err.to_string().contains("nope"), "unexpected error: {err}");
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
        .expect("render succeeds");
    assert!(
        out.messages[0].content.contains("Hello, World!"),
        "unexpected rendered content: {:?}",
        out.messages
    );

    // Unknown prompt → not_found Err.
    let err = reg
        .render("missing", serde_json::json!({}))
        .await
        .expect_err("unknown prompt yields not_found error");
    assert!(err.is_not_found());
    assert_eq!(err.metadata()["error_kind"], "not_registered");
    assert!(
        err.to_string().contains("missing"),
        "unexpected error: {err}"
    );
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

    // A URI that matches no template → not_found Err.
    let uri = "file:///nope/hello.json";
    let err = reg
        .read(uri)
        .await
        .expect_err("non-matching URI yields not_found error");
    assert!(err.is_not_found());
    assert_eq!(err.metadata()["error_kind"], "not_registered");
    assert!(err.to_string().contains(uri), "unexpected error: {err}");
}

// ── Registry Mutation & Shared State ─────────────────────────────────

#[test]
fn registries_remove_clear_and_is_empty() {
    let mut tools = ToolRegistry::new().with_tool(Echo).with_tool(Shout);
    assert_eq!(tools.len(), 2);
    assert!(tools.remove("echo"));
    assert_eq!(tools.len(), 1);
    assert!(!tools.contains("echo"));
    assert!(tools.contains("shout"));
    assert!(!tools.remove("missing"));
    tools.clear();
    assert!(tools.is_empty());

    let mut prompts = PromptRegistry::new().with_prompt(Greet);
    assert_eq!(prompts.len(), 1);
    assert!(prompts.remove("greet"));
    assert!(prompts.is_empty());
    prompts.register(Greet);
    prompts.clear();
    assert!(prompts.is_empty());

    let mut resources = ResourceRegistry::new().with_resource(Blob);
    assert_eq!(resources.len(), 1);
    assert!(resources.remove("blob"));
    assert!(resources.is_empty());
    resources.register(Blob);
    resources.clear();
    assert!(resources.is_empty());
}

#[test]
fn shared_state_direct_methods() {
    let state = llm_tool::SharedState::new();
    assert_eq!(
        state.get_state("key", serde_json::json!("default")),
        serde_json::json!("default")
    );
    state
        .set_state("key", serde_json::json!("value"))
        .expect("set_state succeeds");
    assert_eq!(
        state.get_state("key", serde_json::json!("default")),
        serde_json::json!("value")
    );
    assert!(state.remove_state("key"));
    assert!(!state.remove_state("key"));
    assert_eq!(
        state.get_state("key", serde_json::json!("default")),
        serde_json::json!("default")
    );

    state
        .set_state("k1", serde_json::json!(100))
        .expect("set_state succeeds");
    state
        .set_state("k2", serde_json::json!(200))
        .expect("set_state succeeds");
    state.clear_state();
    assert_eq!(
        state.get_state("k1", serde_json::Value::Null),
        serde_json::Value::Null
    );
    assert_eq!(
        state.get_state("k2", serde_json::Value::Null),
        serde_json::Value::Null
    );
}

#[test]
fn tool_error_metadata_preservation() {
    let err = ToolError::not_found(RegistryItem::Tool, "test")
        .with_meta("initial_key", serde_json::json!("initial_val"))
        .with_metadata(&serde_json::json!({"custom_key": "custom_value"}))
        .expect("with_metadata succeeds");
    assert_eq!(err.metadata()["error_kind"], "not_registered");
    assert_eq!(err.metadata()["initial_key"], "initial_val");
    assert_eq!(err.metadata()["custom_key"], "custom_value");
}

#[test]
fn tool_definition_partial_eq() {
    let reg = ToolRegistry::new().with_tool(Echo);
    let def1 = reg.definition("echo").unwrap();
    let def2 = reg.definition("echo").unwrap();
    assert_eq!(def1, def2);
}

#[test]
fn tool_context_clone_and_debug() {
    let ctx = ToolContext::new().with_conversation_id("conv-123");
    let cloned = ctx.clone();
    assert_eq!(ctx.conversation_id(), cloned.conversation_id());
    let dbg = format!("{ctx:?}");
    assert!(dbg.contains("conv-123"));
}
