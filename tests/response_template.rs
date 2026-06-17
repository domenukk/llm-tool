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
    description = "Get the weather for a city.",
    response_template = "tools/weather_response.tmpl.md"
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
