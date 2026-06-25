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

#[cfg(feature = "prompt-templates")]
#[llm_tool(prompt_file = "tools/static_desc.tmpl.md")]
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

    #[cfg(feature = "prompt-templates")]
    {
        registry.register(StaticTmplTool);
        assert_eq!(
            StaticTmplTool::DESCRIPTION,
            "Fetch the current weather for any city worldwide.\n\nReturns a JSON object containing temperature, humidity,\nwind speed, and a human-readable conditions summary.\nSupports both metric and imperial unit systems."
        );
    }

    let expected_len = if cfg!(feature = "prompt-templates") {
        3
    } else {
        2
    };
    assert_eq!(registry.len(), expected_len);

    let ctx = ToolContext::new(Some("test-conv".into()));
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

    #[cfg(feature = "prompt-templates")]
    {
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
}
