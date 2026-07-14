//! Integration tests for response template rendering.
//!
//! Run with: `cargo test --features md-tmpl`

#![cfg(feature = "md-tmpl")]

use llm_tool::{ToolContext, ToolError, ToolRegistry, llm_tool};
use serde::Serialize;

// ── Response template with struct return ──

#[derive(Serialize)]
struct WeatherResponse {
    city: String,
    temp_f: i64,
    condition: String,
    humidity: i64,
}

#[llm_tool(
    description = "Get the weather for a city.",
    response_file = "tools/weather_response.tmpl.md"
)]
fn get_weather_templated(
    /// The city to get weather for.
    city: String,
) -> Result<WeatherResponse, ToolError> {
    Ok(WeatherResponse {
        city,
        temp_f: 72,
        condition: "Sunny".into(),
        humidity: 45,
    })
}

#[tokio::test]
async fn response_template_renders_struct_fields() {
    let registry = ToolRegistry::new().with_tool(GetWeatherTemplated);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "get_weather_templated",
            serde_json::json!({"city": "Seattle"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let content = output.content();
    assert!(
        content.contains("Weather for Seattle"),
        "should render city name: {content}"
    );
    assert!(
        content.contains("72°F"),
        "should render temperature: {content}"
    );
    assert!(
        content.contains("Sunny"),
        "should render condition: {content}"
    );
    assert!(content.contains("45%"), "should render humidity: {content}");
}

#[tokio::test]
async fn response_template_attaches_metadata() {
    let registry = ToolRegistry::new().with_tool(GetWeatherTemplated);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "get_weather_templated",
            serde_json::json!({"city": "Portland"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let meta = output.metadata();
    assert_eq!(
        meta["city"], "Portland",
        "metadata should contain struct fields"
    );
    assert_eq!(meta["temp_f"], 72, "metadata should contain temp_f");
    assert_eq!(
        meta["condition"], "Sunny",
        "metadata should contain condition"
    );
    assert_eq!(meta["humidity"], 45, "metadata should contain humidity");
}

#[test]
fn response_template_tool_has_correct_description() {
    let registry = ToolRegistry::new().with_tool(GetWeatherTemplated);
    let defs = registry.definitions();
    assert_eq!(defs.len(), 1);
    assert_eq!(
        defs[0].description, "Get the weather for a city.",
        "description should come from the inline text, not the response template"
    );
}

// ── Response template with inline string ──

#[llm_tool(
    description = "Get the inline weather.",
    response = r#"
---
params:
  - city = str
  - temp_f = int
  - condition = str
---
Current weather in {{ city }} is {{ condition }} and {{ temp_f }}F.
"#
)]
async fn get_weather_inline(
    /// The city to check weather for.
    city: String,
) -> Result<GetWeatherInlineResponse, llm_tool::ToolError> {
    Ok(GetWeatherInlineResponse {
        city,
        temp_f: 82,
        condition: "Cloudy".to_string(),
    })
}

#[tokio::test]
async fn test_inline_response_template() {
    let registry = ToolRegistry::new().with_tool(GetWeatherInline);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "get_weather_inline",
            serde_json::json!({"city": "Seattle"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    assert_eq!(
        output.content(),
        "Current weather in Seattle is Cloudy and 82F.
"
    );

    let meta = output.metadata();
    assert_eq!(meta["city"], "Seattle");
    assert_eq!(meta["temp_f"], 82);
    assert_eq!(meta["condition"], "Cloudy");
}

#[derive(Serialize)]
struct SearchResultItem {
    title: String,
    score: i64,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    results: Vec<SearchResultItem>,
    total: i64,
}

#[llm_tool(
    description = "Search for things.",
    response_file = "tools/search_response.tmpl.md"
)]
fn search_tool(
    /// The search query.
    query: String,
) -> Result<SearchResponse, ToolError> {
    Ok(SearchResponse {
        query: query.clone(),
        results: vec![SearchResultItem {
            title: "Rust".into(),
            score: 100,
        }],
        total: 1,
    })
}

#[tokio::test]
async fn test_search_response_template() {
    let registry = ToolRegistry::new().with_tool(SearchTool);
    let ctx = ToolContext::new();
    let output = registry
        .dispatch(
            "search_tool",
            serde_json::json!({"query": "language"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();
    assert!(output.content().contains("Search results for \"language\""));
}

// ── Response file + env combination ──

#[llm_tool(
    description = "Search a service.",
    response_file = "tools/env_response.tmpl.md",
    env(SERVICE_NAME = "my-api")
)]
fn env_response_tool(
    /// The search query.
    query: String,
) -> Result<EnvResponseToolResponse, ToolError> {
    Ok(EnvResponseToolResponse {
        result: query,
        count: 5,
    })
}

#[tokio::test]
async fn response_file_with_env_renders_env_var() {
    let registry = ToolRegistry::new().with_tool(EnvResponseTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "env_response_tool",
            serde_json::json!({"query": "test-query"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let content = output.content();
    assert!(
        content.contains("my-api"),
        "should render env SERVICE_NAME in response, got: {content}"
    );
    assert!(
        content.contains('5'),
        "should render count param, got: {content}"
    );
    assert!(
        content.contains("test-query"),
        "should render result param, got: {content}"
    );
}

#[test]
fn env_response_tool_description_comes_from_description() {
    let registry = ToolRegistry::new().with_tool(EnvResponseTool);
    let defs = registry.definitions();
    assert_eq!(defs.len(), 1);
    assert_eq!(
        defs[0].description, "Search a service.",
        "description should come from the description attribute, not the response template"
    );
}

#[tokio::test]
async fn response_file_with_env_attaches_metadata() {
    let registry = ToolRegistry::new().with_tool(EnvResponseTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "env_response_tool",
            serde_json::json!({"query": "metadata-test"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let meta = output.metadata();
    assert_eq!(
        meta["result"], "metadata-test",
        "metadata should contain result field"
    );
    assert_eq!(meta["count"], 5, "metadata should contain count field");
}

// ── Inline response + env combination ──

#[llm_tool(
    description = "Query a service with inline response.",
    response = r#"
---
env:
  - HOST = str

params:
  - status = str
  - code = int
---
[{{ HOST }}] Status: {{ status }} (code={{ code }})
"#,
    env(HOST = "prod.example.com")
)]
fn inline_env_response_tool(
    /// The query.
    query: String,
) -> Result<InlineEnvResponseToolResponse, ToolError> {
    // NOLINT: test tool intentionally ignores query parameter
    let _ = query;
    Ok(InlineEnvResponseToolResponse {
        status: "healthy".to_string(),
        code: 200,
    })
}

#[tokio::test]
async fn inline_response_with_env_renders_env_var() {
    let registry = ToolRegistry::new().with_tool(InlineEnvResponseTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "inline_env_response_tool",
            serde_json::json!({"query": "health"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let content = output.content();
    assert!(
        content.contains("prod.example.com"),
        "should render env HOST in inline response, got: {content}"
    );
    assert!(
        content.contains("healthy"),
        "should render status param, got: {content}"
    );
    assert!(
        content.contains("200"),
        "should render code param, got: {content}"
    );
}

#[test]
fn inline_env_response_description_comes_from_description() {
    assert_eq!(
        <InlineEnvResponseTool as llm_tool::RustTool>::DESCRIPTION,
        "Query a service with inline response."
    );
}

// ── Description + response_file both present ──

#[derive(Serialize)]
struct DescriptionPlusResponseResult {
    city: String,
    temp_f: i64,
    condition: String,
    humidity: i64,
}

#[llm_tool(
    description = "Get weather details for a city.",
    response_file = "tools/weather_response.tmpl.md"
)]
fn description_plus_response_tool(
    /// The city.
    city: String,
) -> Result<DescriptionPlusResponseResult, ToolError> {
    Ok(DescriptionPlusResponseResult {
        city,
        temp_f: 85,
        condition: "Hot".into(),
        humidity: 30,
    })
}

#[test]
fn description_plus_response_description_from_description() {
    let desc = <DescriptionPlusResponseTool as llm_tool::RustTool>::DESCRIPTION;
    assert_eq!(
        desc, "Get weather details for a city.",
        "description should come from the description attribute"
    );
}

#[tokio::test]
async fn description_plus_response_output_from_template() {
    let registry = ToolRegistry::new().with_tool(DescriptionPlusResponseTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "description_plus_response_tool",
            serde_json::json!({"city": "Phoenix"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let content = output.content();
    assert!(
        content.contains("Weather for Phoenix"),
        "response should use response_file template, got: {content}"
    );
    assert!(
        content.contains("85°F"),
        "should render temperature, got: {content}"
    );
    assert!(
        content.contains("Hot"),
        "should render condition, got: {content}"
    );
    assert!(
        content.contains("30%"),
        "should render humidity, got: {content}"
    );
}

// ── Description_file + response_file both present ──

#[derive(Serialize)]
struct DescriptionFileResponseResult {
    city: String,
    temp_f: i64,
    condition: String,
    humidity: i64,
}

#[llm_tool(
    description_file = "tools/description_for_combined.tmpl.md",
    response_file = "tools/weather_response.tmpl.md"
)]
fn description_file_plus_response_tool(
    /// The city name.
    city: String,
) -> Result<DescriptionFileResponseResult, ToolError> {
    Ok(DescriptionFileResponseResult {
        city,
        temp_f: 55,
        condition: "Rainy".into(),
        humidity: 90,
    })
}

#[test]
fn description_file_plus_response_description_from_description_file() {
    let desc = <DescriptionFilePlusResponseTool as llm_tool::RustTool>::DESCRIPTION;
    assert!(
        desc.contains("Look up detailed weather information"),
        "description should come from description_file, got: {desc}"
    );
}

#[tokio::test]
async fn description_file_plus_response_output_from_response_file() {
    let registry = ToolRegistry::new().with_tool(DescriptionFilePlusResponseTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "description_file_plus_response_tool",
            serde_json::json!({"city": "London"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let content = output.content();
    assert!(
        content.contains("Weather for London"),
        "response should use response_file template, got: {content}"
    );
    assert!(
        content.contains("55°F"),
        "should render temperature, got: {content}"
    );
    assert!(
        content.contains("Rainy"),
        "should render condition, got: {content}"
    );
    assert!(
        content.contains("90%"),
        "should render humidity, got: {content}"
    );
}

#[tokio::test]
async fn description_file_plus_response_attaches_metadata() {
    let registry = ToolRegistry::new().with_tool(DescriptionFilePlusResponseTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "description_file_plus_response_tool",
            serde_json::json!({"city": "Berlin"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    let meta = output.metadata();
    assert_eq!(meta["city"], "Berlin");
    assert_eq!(meta["temp_f"], 55);
    assert_eq!(meta["condition"], "Rainy");
    assert_eq!(meta["humidity"], 90);
}

// ── Dispatch and execution tests for env tools ──

#[llm_tool(description_file = "tools/env_desc.tmpl.md", env(API_VERSION = "v7.0"))]
fn dispatch_env_tool(
    /// A query value.
    query: String,
) -> Result<String, ToolError> {
    Ok(format!("executed: {query}"))
}

#[tokio::test]
async fn env_tool_dispatch_returns_correct_output() {
    let registry = ToolRegistry::new().with_tool(DispatchEnvTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "dispatch_env_tool",
            serde_json::json!({"query": "hello"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    assert_eq!(
        output.content(),
        "executed: hello",
        "dispatch should execute the function body"
    );
}

#[test]
fn env_tool_dispatch_description_is_rendered() {
    let registry = ToolRegistry::new().with_tool(DispatchEnvTool);
    let defs = registry.definitions();
    assert_eq!(defs.len(), 1);
    assert!(
        defs[0].description.contains("v7.0"),
        "description should contain rendered env, got: {}",
        defs[0].description
    );
}

#[llm_tool(
    description_file = "tools/multi_env.tmpl.md",
    env(
        SERVICE_NAME = "dispatch-svc",
        REGION = "ap-south-1",
        MAX_CONNECTIONS = 50
    )
)]
fn dispatch_multi_env_tool(
    /// A query.
    query: String,
) -> Result<String, ToolError> {
    Ok(format!("multi-env executed: {query}"))
}

#[tokio::test]
async fn multi_env_tool_dispatch_executes_correctly() {
    let registry = ToolRegistry::new().with_tool(DispatchMultiEnvTool);
    let ctx = ToolContext::new();

    let output = registry
        .dispatch(
            "dispatch_multi_env_tool",
            serde_json::json!({"query": "world"}),
            &ctx,
        )
        .await
        .expect("tool registered")
        .unwrap();

    assert_eq!(output.content(), "multi-env executed: world");
}

#[test]
fn multi_env_tool_dispatch_description_has_all_vars() {
    let registry = ToolRegistry::new().with_tool(DispatchMultiEnvTool);
    let defs = registry.definitions();
    let desc = &defs[0].description;
    assert!(desc.contains("dispatch-svc"), "got: {desc}");
    assert!(desc.contains("ap-south-1"), "got: {desc}");
    assert!(desc.contains("50"), "got: {desc}");
    assert!(
        desc.contains("false"),
        "default DEBUG_MODE should be false, got: {desc}"
    );
}
