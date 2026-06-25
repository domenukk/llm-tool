//! Integration tests for response template rendering.
//!
//! Run with: `cargo test --features prompt-templates`

#![cfg(feature = "prompt-templates")]

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
    prompt = "Get the weather for a city.",
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
    let ctx = ToolContext::new(None);

    let output = registry
        .dispatch(
            "get_weather_templated",
            serde_json::json!({"city": "Seattle"}),
            &ctx,
        )
        .await
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
    let ctx = ToolContext::new(None);

    let output = registry
        .dispatch(
            "get_weather_templated",
            serde_json::json!({"city": "Portland"}),
            &ctx,
        )
        .await
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
    prompt = "Get the inline weather.",
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
    let ctx = ToolContext::new(None);

    let output = registry
        .dispatch(
            "get_weather_inline",
            serde_json::json!({"city": "Seattle"}),
            &ctx,
        )
        .await
        .unwrap();

    assert_eq!(
        output.content(),
        "Current weather in Seattle is Cloudy and 82F.\n"
    );

    let meta = output.metadata();
    assert_eq!(meta["city"], "Seattle");
    assert_eq!(meta["temp_f"], 82);
    assert_eq!(meta["condition"], "Cloudy");
}
