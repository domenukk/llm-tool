//! Tests for the renamed `prompt`, `prompt_file`, and `response_file` attributes.
//!
//! Verifies the new canonical attribute names work correctly and produce
//! identical behavior to the pre-rename API.

#![cfg(feature = "prompt-templates")]

use llm_tool::{RustTool, ToolContext, ToolError, ToolRegistry, llm_tool};

// ── prompt = "..." (inline description) ──────────────────────────────

/// This doc comment should be ignored since prompt = "..." takes priority.
#[llm_tool(prompt = "Get the current weather for a given city.")]
fn inline_prompt_tool(
    /// The city name to look up.
    city: String,
) -> Result<String, ToolError> {
    Ok(format!("Weather for {city}: sunny"))
}

#[test]
fn inline_prompt_sets_description() {
    assert_eq!(
        <InlinePromptTool as RustTool>::DESCRIPTION,
        "Get the current weather for a given city."
    );
}

#[test]
fn inline_prompt_tool_has_correct_name() {
    assert_eq!(InlinePromptTool::NAME, "inline_prompt_tool");
}

#[test]
fn inline_prompt_in_registry() {
    let reg = ToolRegistry::new().with_tool(InlinePromptTool);
    let defs = reg.definitions();
    assert_eq!(defs.len(), 1);
    assert_eq!(
        defs[0].description,
        "Get the current weather for a given city."
    );
}

// ── prompt_file = "..." (static template — no variables) ─────────────

#[llm_tool(prompt_file = "tools/static_desc.tmpl.md")]
fn prompt_file_static_tool(
    /// The item identifier.
    item_id: String,
) -> Result<String, ToolError> {
    Ok(format!("Looked up item {item_id}"))
}

#[test]
fn prompt_file_static_sets_description() {
    let desc = <PromptFileStaticTool as RustTool>::DESCRIPTION;
    assert!(
        !desc.is_empty(),
        "prompt_file should embed a non-empty description"
    );
}

// ── prompt_file = "..." + params(...) (compiled template) ────────────

#[llm_tool(
    prompt_file = "tools/parameterized_desc.tmpl.md",
    params(api_version = "v3", env_name = "prod")
)]
fn prompt_file_with_params(
    /// The query string.
    query: String,
) -> Result<String, ToolError> {
    Ok(format!("Searching: {query}"))
}

#[test]
fn prompt_file_with_params_embeds_values() {
    let desc = <PromptFileWithParams as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("v3"),
        "compiled template should contain the param value, got: {desc}",
    );
}

// ── response_file = "..." (response template rendering) ──────────────

#[derive(serde::Serialize)]
struct WeatherResult {
    city: String,
    temp_f: i64,
    condition: String,
    humidity: i64,
}

/// Get weather with response template.
#[llm_tool(response_file = "tools/weather_response.tmpl.md")]
fn response_file_tool(
    /// The city to get weather for.
    city: String,
) -> Result<WeatherResult, ToolError> {
    Ok(WeatherResult {
        city,
        temp_f: 72,
        condition: "Sunny".into(),
        humidity: 45,
    })
}

#[tokio::test]
async fn response_file_renders_struct_fields() {
    let mut reg = ToolRegistry::new();
    reg.register(ResponseFileTool);
    let ctx = ToolContext::new(None);
    let result = reg
        .dispatch(
            "response_file_tool",
            serde_json::json!({"city": "Portland"}),
            &ctx,
        )
        .await
        .expect("dispatch should succeed");

    assert!(
        result.content().contains("Portland"),
        "rendered response should contain the city name, got: {}",
        result.content()
    );
}

#[tokio::test]
async fn response_file_attaches_metadata() {
    let mut reg = ToolRegistry::new();
    reg.register(ResponseFileTool);
    let ctx = ToolContext::new(None);
    let result = reg
        .dispatch(
            "response_file_tool",
            serde_json::json!({"city": "Seattle"}),
            &ctx,
        )
        .await
        .expect("dispatch should succeed");

    assert!(
        result.metadata().contains_key("city"),
        "metadata should contain serialized struct fields"
    );
}

// ── prompt + response_file combined ──────────────────────────────────

/// This doc is overridden by prompt.
#[llm_tool(
    prompt = "Get the forecast.",
    response_file = "tools/weather_response.tmpl.md"
)]
fn combined_prompt_response(
    /// The location.
    city: String,
) -> Result<WeatherResult, ToolError> {
    Ok(WeatherResult {
        city,
        temp_f: 65,
        condition: "Cloudy".into(),
        humidity: 80,
    })
}

#[test]
fn combined_prompt_and_response_file() {
    assert_eq!(
        <CombinedPromptResponse as RustTool>::DESCRIPTION,
        "Get the forecast."
    );
}

// ── doc comment fallback (no prompt/prompt_file) ─────────────────────

/// This description comes from the doc comment.
#[llm_tool]
fn doc_comment_tool(
    /// The input value.
    value: String,
) -> Result<String, ToolError> {
    Ok(value)
}

#[test]
fn doc_comment_fallback_works() {
    assert_eq!(
        <DocCommentTool as RustTool>::DESCRIPTION,
        "This description comes from the doc comment."
    );
}
