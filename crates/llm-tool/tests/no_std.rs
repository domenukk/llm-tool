//! Test suite exercising `llm-tool` core functionality.
//! When run with `--no-default-features`, this tests the `no_std` code paths
//! (`spin::RwLock`, `spin::LazyLock`, `core::any`, `alloc`).

use std::sync::Arc;

use llm_tool::{RustTool, ToolContext, ToolRegistry, llm_tool};

#[derive(Clone, PartialEq, Eq, Debug)]
struct CustomState {
    value: i32,
}

/// Multiplies two numbers.
#[llm_tool]
fn multiply(
    /// First number
    a: i64,
    /// Second number
    b: i64,
) -> Result<String, String> {
    Ok(format!("{}", a * b))
}

/// Gets the custom state value.
#[llm_tool]
fn get_state_val(
    /// Dummy param
    _dummy: bool,
    ctx: &ToolContext,
) -> Result<String, String> {
    let ext = ctx
        .get_ext::<Arc<CustomState>>()
        .expect("CustomState missing");
    Ok(format!("{}", ext.value))
}

#[llm_tool(description_file = "tools/static_desc.tmpl.md")]
fn static_tmpl_tool(
    /// Location
    loc: String,
) -> Result<String, String> {
    Ok(format!("Weather for {loc}"))
}

#[tokio::test]
async fn test_no_std_paths() {
    let mut registry = ToolRegistry::new();
    registry.register(Multiply);
    assert_eq!(Multiply::NAME, "multiply");
    assert_eq!(Multiply::DESCRIPTION, "Multiplies two numbers.");

    registry.register(GetStateVal);

    registry.register(StaticTmplTool);
    assert_eq!(
        StaticTmplTool::DESCRIPTION.replace("\r\n", "\n"),
        "Fetch the current weather for any city worldwide.

Returns structured weather data containing temperature, humidity,
wind speed, and a human-readable conditions summary.
Supports both metric and imperial unit systems."
    );

    assert_eq!(registry.len(), 3);

    let ctx = ToolContext::new().with_conversation_id("test-conv");
    ctx.set_ext(Arc::new(CustomState { value: 42 }))
        .expect("set_ext failed");

    // Test multiply
    let out = registry
        .dispatch("multiply", serde_json::json!({ "a": 6, "b": 7 }), &ctx)
        .await
        .expect("dispatch multiply failed");
    assert_eq!(out.content(), "42");

    // Test get_state_val (tests extension map with core::any / spin lock)
    let out = registry
        .dispatch("get_state_val", serde_json::json!({ "_dummy": true }), &ctx)
        .await
        .expect("dispatch get_state_val failed");
    assert_eq!(out.content(), "42");

    let out = registry
        .dispatch(
            "static_tmpl_tool",
            serde_json::json!({ "loc": "Paris" }),
            &ctx,
        )
        .await
        .expect("dispatch static_tmpl_tool failed");
    assert_eq!(out.content(), "Weather for Paris");
}

#[derive(Clone)]
struct NoStdService {
    prefix: String,
}

#[llm_tool]
impl NoStdService {
    /// Echo sanitized input and extension state.
    #[llm_tool(effect = "read_only")]
    pub fn echo(
        &self,
        /// Sanitized query.
        query: llm_tool::Sanitized<String>,
        /// Typed extension state.
        #[llm_tool(ext)]
        state: Arc<CustomState>,
    ) -> String {
        let clean = query.into_inner();
        let val = Arc::try_unwrap(state).map_or_else(|arc| arc.value, |s| s.value);
        format!("{}:{clean}:{val}", self.prefix)
    }
}

/// Read an item by numeric URI var.
#[llm_tool::llm_resource(uri = "item://{id}")]
fn read_no_std_item(
    /// Numeric ID.
    id: u32,
) -> String {
    format!("item-{id}")
}

#[tokio::test]
async fn test_no_std_0107_ergonomics_and_wire_projections() {
    let svc = NoStdService {
        prefix: "ns".to_string(),
    };
    let reg = svc.tool_registry().read_only();
    assert_eq!(reg.to_openai_tools().len(), 1);
    assert_eq!(reg.to_anthropic_tools().len(), 1);
    assert_eq!(reg.to_gemini_declarations().len(), 1);

    let ctx = ToolContext::new();
    ctx.set_ext(Arc::new(CustomState { value: 7 }))
        .expect("set_ext");
    let out = reg
        .dispatch(
            "echo",
            serde_json::json!({ "query": "hi <|im_start|>" }),
            &ctx,
        )
        .await
        .expect("dispatch echo");
    assert!(!out.content().contains("<|im_start|>"));
    assert!(out.content().starts_with("ns:hi "));

    let res_reg = llm_tool::ResourceRegistry::new().with_resource(ReadNoStdItem);
    let res_out = res_reg
        .read_with_context("item://42", &ctx)
        .await
        .expect("read item://42");
    assert_eq!(res_out.contents.len(), 1);
}
