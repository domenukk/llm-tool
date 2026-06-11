//! Integration tests for template-based tool descriptions.
//!
//! Run with: `cargo test --features prompt-templates`

#![cfg(feature = "prompt-templates")]

use llm_tool::{RustTool, ToolRegistry, llm_tool};

#[llm_tool(template = "tools/static_desc.tmpl.md")]
fn get_weather(
    /// The city to get weather for.
    city: String,
) -> Result<String, String> {
    Ok(format!("Weather for {city}: sunny"))
}

#[test]
fn template_description_is_embedded() {
    let desc = <GetWeather as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("Fetch the current weather"),
        "description should contain template body, got: {desc}"
    );
    assert!(
        desc.contains("metric and imperial"),
        "description should contain full body, got: {desc}"
    );
}

#[test]
fn template_description_via_description_method() {
    let tool = GetWeather;
    let desc = tool.description();
    assert!(
        desc.contains("Fetch the current weather"),
        "description() should return template body, got: {desc}"
    );
}

#[test]
fn template_description_in_registry() {
    let registry = ToolRegistry::new().with_tool(GetWeather);
    let definitions = registry.definitions();
    assert_eq!(definitions.len(), 1);
    let defn = &definitions[0];
    assert_eq!(defn.name, "get_weather");
    assert!(
        defn.description.contains("Fetch the current weather"),
        "ToolDefinition.description should contain template body, got: {}",
        defn.description
    );
}

/// Doc comments are optional when using template descriptions.
#[llm_tool(template = "tools/static_desc.tmpl.md")]
fn tool_without_docs(
    /// A parameter.
    value: i64,
) -> String {
    format!("{value}")
}

#[test]
fn template_description_no_doc_comment_required() {
    let desc = <ToolWithoutDocs as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("Fetch the current weather"),
        "should work without doc comment, got: {desc}"
    );
}

// ── Dynamic Template Description Tests ──

fn get_weather_context(_tool: &GetWeatherDynamic) -> prompt_templates::Context {
    let mut ctx = prompt_templates::Context::new();
    ctx.set("api_version", "v3.1");
    ctx.set("env_name", "staging");
    ctx
}

#[llm_tool(
    template = "tools/dynamic_desc.tmpl.md",
    context = get_weather_context
)]
fn get_weather_dynamic(
    /// The city to lookup.
    city: String,
) -> Result<String, String> {
    Ok(format!("Weather for {city}: raining"))
}

#[test]
fn dynamic_template_description_renders_at_runtime() {
    let tool = GetWeatherDynamic;
    let desc = tool.description();
    assert!(
        desc.contains("API v3.1"),
        "should render variables, got: {desc}"
    );
    assert!(
        desc.contains("staging environment"),
        "should render variables, got: {desc}"
    );
}

#[test]
fn dynamic_description_propagates_to_registry() {
    let registry = ToolRegistry::new().with_tool(GetWeatherDynamic);
    let definitions = registry.definitions();
    assert_eq!(definitions.len(), 1);
    let defn = &definitions[0];
    assert!(
        defn.description.contains("API v3.1"),
        "ToolDefinition should contain rendered description, got: {}",
        defn.description
    );
    assert!(
        defn.description.contains("staging environment"),
        "ToolDefinition should contain rendered description, got: {}",
        defn.description
    );
}

// ── Inline Description Tests ──

#[llm_tool(description = "Get the current temperature for a location.")]
fn inline_description_tool(
    /// The city name.
    city: String,
) -> String {
    format!("Temp in {city}: 20°C")
}

#[test]
fn inline_description_replaces_doc_comment() {
    let desc = <InlineDescriptionTool as RustTool>::DESCRIPTION;
    assert_eq!(desc, "Get the current temperature for a location.");
}

#[test]
fn inline_description_in_registry() {
    let registry = ToolRegistry::new().with_tool(InlineDescriptionTool);
    let defs = registry.definitions();
    assert_eq!(
        defs[0].description,
        "Get the current temperature for a location."
    );
}

// ── Compile-time Params Tests ──

#[llm_tool(
    template = "tools/parameterized_desc.tmpl.md",
    params(api_version = "v4.2", env_name = "production")
)]
fn parameterized_tool(
    /// A query value.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn compile_time_params_render_into_static_description() {
    let desc = <ParameterizedTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("API v4.2"),
        "should contain rendered api_version, got: {desc}"
    );
    assert!(
        desc.contains("production environment"),
        "should contain rendered env_name, got: {desc}"
    );
}

#[test]
fn compile_time_params_description_is_static() {
    // The description method should return Cow::Borrowed (static str),
    // not Cow::Owned (runtime rendered).
    let tool = ParameterizedTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "compile-time params should produce a static (Borrowed) description"
    );
}

#[test]
fn compile_time_params_in_registry() {
    let registry = ToolRegistry::new().with_tool(ParameterizedTool);
    let defs = registry.definitions();
    assert!(
        defs[0].description.contains("API v4.2"),
        "ToolDefinition should contain rendered params, got: {}",
        defs[0].description
    );
}
