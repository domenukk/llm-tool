//! Integration tests for template-based tool descriptions.
//!
//! Run with: `cargo test --features md-tmpl`

#![cfg(feature = "md-tmpl")]

use llm_tool::{RustTool, ToolRegistry, llm_tool};

#[llm_tool(prompt_file = "tools/static_desc.tmpl.md")]
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
#[llm_tool(prompt_file = "tools/static_desc.tmpl.md")]
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

fn get_weather_context(_tool: &GetWeatherDynamic) -> llm_tool::md_tmpl::Context {
    llm_tool::md_tmpl::Context::from_serialize(&serde_json::json!({
        "context": {
            "api_version": "v3.1",
            "env_name": "staging"
        }
    }))
    .unwrap()
}

#[llm_tool(
    prompt_file = "tools/dynamic_desc.tmpl.md",
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

#[llm_tool(prompt = "Get the current temperature for a location.")]
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
    prompt_file = "tools/parameterized_desc.tmpl.md",
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

// ── Compile-time Env Tests ──

#[llm_tool(prompt_file = "tools/env_desc.tmpl.md", env(API_VERSION = "v5.0"))]
fn env_tool(
    /// A query value.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn env_renders_into_static_description() {
    let desc = <EnvTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("API v5.0"),
        "should contain rendered env API_VERSION, got: {desc}"
    );
    assert!(
        desc.contains("3 retries"),
        "should contain default MAX_RETRIES=3, got: {desc}"
    );
}

#[test]
fn env_description_is_static() {
    let tool = EnvTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "env-based descriptions should be static (Borrowed)"
    );
}

#[llm_tool(
    prompt_file = "tools/env_desc.tmpl.md",
    env(API_VERSION = "v6.0", MAX_RETRIES = "10")
)]
fn env_override_tool(
    /// A query value.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn env_override_default() {
    let desc = <EnvOverrideTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("API v6.0"),
        "should contain provided API_VERSION, got: {desc}"
    );
    assert!(
        desc.contains("10 retries"),
        "should contain overridden MAX_RETRIES=10, got: {desc}"
    );
}

// ── Env + Params Combination Tests ──

#[llm_tool(
    prompt_file = "tools/env_plus_params.tmpl.md",
    params(version = "2.0"),
    env(DEPLOYMENT_ENV = "staging")
)]
fn env_plus_params_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn env_plus_params_renders_both() {
    let desc = <EnvPlusParamsTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("Version 2.0"),
        "should contain rendered param, got: {desc}"
    );
    assert!(
        desc.contains("staging"),
        "should contain rendered env, got: {desc}"
    );
}

#[test]
fn env_plus_params_is_static() {
    let tool = EnvPlusParamsTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "env+params should produce static (Borrowed) description"
    );
}

// ── Env + Context Combination Tests ──

fn env_context_fn(_tool: &EnvPlusContextTool) -> llm_tool::md_tmpl::Context {
    llm_tool::md_tmpl::Context::from_serialize(&serde_json::json!({
        "region": "us-east-1"
    }))
    .unwrap()
}

#[llm_tool(
    prompt_file = "tools/env_plus_context.tmpl.md",
    context = env_context_fn,
    env(CLUSTER = "prod-alpha")
)]
fn env_plus_context_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn env_plus_context_renders_both() {
    let tool = EnvPlusContextTool;
    let desc = tool.description();
    assert!(
        desc.contains("us-east-1"),
        "should contain context param, got: {desc}"
    );
    assert!(
        desc.contains("prod-alpha"),
        "should contain env var, got: {desc}"
    );
}

// ── Inline Env Tests ──

#[llm_tool(
    prompt = r#"
---
env:
  - SERVICE = str
---
Running service {{ SERVICE }}.
"#,
    env(SERVICE = "auth-gateway")
)]
fn inline_env_tool(
    /// A value.
    value: String,
) -> String {
    format!("val: {value}")
}

#[test]
fn inline_env_renders_correctly() {
    let desc = <InlineEnvTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("auth-gateway"),
        "should contain rendered env in inline template, got: {desc}"
    );
}

#[test]
fn inline_env_is_static() {
    let tool = InlineEnvTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "inline env should produce static (Borrowed) description"
    );
}

// ── Typed (Non-String) Env Tests ──

#[llm_tool(
    prompt_file = "tools/env_desc.tmpl.md",
    env(API_VERSION = "v5", MAX_RETRIES = 10)
)]
fn typed_env_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn typed_env_int_renders_correctly() {
    let desc = <TypedEnvTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("10"),
        "should contain int env value, got: {desc}"
    );
    assert!(
        desc.contains("v5"),
        "should contain string env value, got: {desc}"
    );
}

#[llm_tool(
    prompt = r#"
---
env:
  - VERBOSE = bool
---
Verbose: {{ VERBOSE }}.
"#,
    env(VERBOSE = true)
)]
fn bool_env_tool(
    /// A value.
    value: String,
) -> String {
    format!("val: {value}")
}

#[test]
fn bool_env_renders_correctly() {
    let desc = <BoolEnvTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("true"),
        "should contain bool env value, got: {desc}"
    );
}

// ── Float Env Tests ──

#[llm_tool(prompt_file = "tools/float_env.tmpl.md", env(THRESHOLD = 0.95))]
fn float_env_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn float_env_renders_correctly() {
    let desc = <FloatEnvTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("0.95"),
        "should contain float env value, got: {desc}"
    );
    assert!(
        desc.contains("0.5"),
        "should contain default float MIN_SCORE, got: {desc}"
    );
}

#[test]
fn float_env_is_static() {
    let tool = FloatEnvTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "float env should produce static (Borrowed) description"
    );
}

#[llm_tool(
    prompt_file = "tools/float_env.tmpl.md",
    env(THRESHOLD = 0.75, MIN_SCORE = 0.1)
)]
fn float_env_override_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn float_env_override_default() {
    let desc = <FloatEnvOverrideTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("0.75"),
        "should contain overridden THRESHOLD, got: {desc}"
    );
    assert!(
        desc.contains("0.1"),
        "should contain overridden MIN_SCORE, got: {desc}"
    );
}

// ── Multiple Env Vars Tests (3+) ──

#[llm_tool(
    prompt_file = "tools/multi_env.tmpl.md",
    env(
        SERVICE_NAME = "my-api",
        REGION = "us-west-2",
        MAX_CONNECTIONS = 200,
        DEBUG_MODE = true
    )
)]
fn multi_env_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn multi_env_renders_all_vars() {
    let desc = <MultiEnvTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("my-api"),
        "should contain SERVICE_NAME, got: {desc}"
    );
    assert!(
        desc.contains("us-west-2"),
        "should contain REGION, got: {desc}"
    );
    assert!(
        desc.contains("200"),
        "should contain overridden MAX_CONNECTIONS, got: {desc}"
    );
    assert!(
        desc.contains("true"),
        "should contain overridden DEBUG_MODE, got: {desc}"
    );
}

#[test]
fn multi_env_is_static() {
    let tool = MultiEnvTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "multi env should produce static (Borrowed) description"
    );
}

#[llm_tool(
    prompt_file = "tools/multi_env.tmpl.md",
    env(SERVICE_NAME = "cache-svc", REGION = "eu-central-1")
)]
fn multi_env_defaults_tool(
    /// A query.
    query: String,
) -> String {
    format!("query: {query}")
}

#[test]
fn multi_env_uses_defaults_when_not_overridden() {
    let desc = <MultiEnvDefaultsTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("cache-svc"),
        "should contain SERVICE_NAME, got: {desc}"
    );
    assert!(
        desc.contains("eu-central-1"),
        "should contain REGION, got: {desc}"
    );
    assert!(
        desc.contains("100"),
        "should contain default MAX_CONNECTIONS=100, got: {desc}"
    );
    assert!(
        desc.contains("false"),
        "should contain default DEBUG_MODE=false, got: {desc}"
    );
}

// ── Inline Prompt + Context Tests ──

fn inline_context_fn(_tool: &InlineContextTool) -> llm_tool::md_tmpl::Context {
    llm_tool::md_tmpl::Context::from_serialize(&serde_json::json!({
        "model_name": "gemini-pro"
    }))
    .unwrap()
}

#[llm_tool(
    prompt = r#"
---
params:
  - model_name = str
---
Running on model {{ model_name }}.
"#,
    context = inline_context_fn
)]
fn inline_context_tool(
    /// A value.
    value: String,
) -> String {
    format!("val: {value}")
}

#[test]
fn inline_prompt_with_context_renders_at_runtime() {
    let tool = InlineContextTool;
    let desc = tool.description();
    assert!(
        desc.contains("gemini-pro"),
        "should render context variable in inline template, got: {desc}"
    );
}

#[test]
fn inline_prompt_with_context_is_dynamic() {
    let tool = InlineContextTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Owned(_)),
        "inline context descriptions should be dynamic (Owned)"
    );
}

#[test]
fn inline_prompt_with_context_in_registry() {
    let registry = ToolRegistry::new().with_tool(InlineContextTool);
    let defs = registry.definitions();
    assert_eq!(defs.len(), 1);
    assert!(
        defs[0].description.contains("gemini-pro"),
        "registry description should contain runtime value, got: {}",
        defs[0].description
    );
}

// ── Inline Prompt + Params Tests ──

#[llm_tool(
    prompt = r#"
---
params:
  - deployment = str
---
Deployed to {{ deployment }} cluster.
"#,
    params(deployment = "canary")
)]
fn inline_params_tool(
    /// A value.
    value: String,
) -> String {
    format!("val: {value}")
}

#[test]
fn inline_prompt_with_params_renders_at_compile_time() {
    let desc = <InlineParamsTool as RustTool>::DESCRIPTION;
    assert!(
        desc.contains("canary"),
        "should contain rendered param in inline template, got: {desc}"
    );
}

#[test]
fn inline_prompt_with_params_is_static() {
    let tool = InlineParamsTool;
    let desc = tool.description();
    assert!(
        matches!(desc, std::borrow::Cow::Borrowed(_)),
        "inline params should produce static (Borrowed) description"
    );
}

#[test]
fn inline_prompt_with_params_in_registry() {
    let registry = ToolRegistry::new().with_tool(InlineParamsTool);
    let defs = registry.definitions();
    assert_eq!(defs.len(), 1);
    assert!(
        defs[0].description.contains("canary"),
        "registry description should contain rendered param, got: {}",
        defs[0].description
    );
}

// ── Inline Prompt + Env + Context Combination ──

fn inline_env_context_fn(_tool: &InlineEnvContextTool) -> llm_tool::md_tmpl::Context {
    llm_tool::md_tmpl::Context::from_serialize(&serde_json::json!({
        "user_count": 42
    }))
    .unwrap()
}

#[llm_tool(
    prompt = r#"
---
env:
  - INSTANCE = str

params:
  - user_count = int
---
Instance {{ INSTANCE }} serving {{ user_count }} users.
"#,
    context = inline_env_context_fn,
    env(INSTANCE = "prod-3")
)]
fn inline_env_context_tool(
    /// A value.
    value: String,
) -> String {
    format!("val: {value}")
}

#[test]
fn inline_env_plus_context_renders_both() {
    let tool = InlineEnvContextTool;
    let desc = tool.description();
    assert!(
        desc.contains("prod-3"),
        "should contain env var in inline template, got: {desc}"
    );
    assert!(
        desc.contains("42"),
        "should contain context var in inline template, got: {desc}"
    );
}
