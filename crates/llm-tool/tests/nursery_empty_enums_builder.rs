//! Regression test verifying `#[llm_tool(response = ...)]` generated response structs
//! and `::builder()...build()` compile cleanly under `#![deny(clippy::nursery, clippy::empty_enums)]`
//! with zero uninhabited enums or lint suppressions.

#![deny(clippy::nursery, clippy::empty_enums)]

use llm_tool::{ToolContext, ToolRegistry, llm_tool};

/// Renders a structured build summary using a response template.
#[llm_tool(
    effect = "read_only",
    response = r#"---
params:
  - target = str
  - elapsed_ms = int
  - warnings = list(str)
  - status = option(enum(Clean, Failed))
---
target={{ target }} elapsed={{ elapsed_ms }}ms warnings={{ len(warnings) }} status={% if has(status) %}{{ kind(status) }}{% else %}none{% /if %}"#
)]
fn summarize_build(
    /// Target crate name.
    target: &str,
    /// Elapsed build time in milliseconds.
    elapsed_ms: i64,
) -> SummarizeBuildResponse {
    SummarizeBuildResponse::builder()
        .target(target)
        .elapsed_ms(elapsed_ms)
        .warnings(["none".to_string()])
        .status(Some(SummarizeBuildResponseStatus::Clean))
        .build()
}

/// Renders an empty-param response template via zero-field builder.
#[llm_tool(
    effect = "read_only",
    response = r#"---
params: []
---
pong"#
)]
fn zero_field_response_ping() -> ZeroFieldResponsePingResponse {
    ZeroFieldResponsePingResponse::builder().build()
}

#[tokio::test]
async fn response_struct_builder_compiles_and_dispatches_under_deny_nursery_empty_enums() {
    let reg = ToolRegistry::new()
        .with_tool(SummarizeBuild)
        .with_tool(ZeroFieldResponsePing);
    let ctx = ToolContext::new();

    let out = reg
        .dispatch(
            "summarize_build",
            serde_json::json!({ "target": "llm-tool", "elapsed_ms": 42 }),
            &ctx,
        )
        .await
        .expect("summarize_build dispatch");
    assert_eq!(
        out.content().trim(),
        "target=llm-tool elapsed=42ms warnings=1 status=Clean"
    );
    assert_eq!(out.metadata()["target"], "llm-tool");
    assert_eq!(out.metadata()["elapsed_ms"], 42);

    let defaulted = SummarizeBuildResponse::builder()
        .target("core")
        .elapsed_ms(7)
        .status(None)
        .build();
    assert_eq!(defaulted.warnings.len(), 0);
    assert_eq!(defaulted.status, None);

    let pong = reg
        .dispatch("zero_field_response_ping", serde_json::json!({}), &ctx)
        .await
        .expect("zero_field_response_ping dispatch");
    assert_eq!(pong.content().trim(), "pong");
}
