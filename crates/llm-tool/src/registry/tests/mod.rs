use serde::Deserialize;

use super::{
    super::{EmptyParams, definition_of},
    *,
};
use crate::llm_tool;

/// Create a default `ToolContext` for tests.
fn test_ctx() -> ToolContext {
    ToolContext::new(None)
}

// ── Sample tool structs for tests ────────────────────────────────

#[derive(Deserialize, schemars::JsonSchema)]
struct PathParams {
    /// Filesystem path.
    path: String,
}

struct SampleTool;

impl RustTool for SampleTool {
    type Params = PathParams;
    const NAME: &'static str = "sample";
    const DESCRIPTION: &'static str = "A sample tool";
    async fn call(
        &self,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(params.path.into())
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct RunCommandParams {
    /// Command to run.
    command: String,
    /// Timeout in seconds.
    #[serde(default)]
    timeout: Option<i64>,
    /// Environment variables.
    #[serde(default)]
    env: Option<std::collections::HashMap<String, String>>,
}

struct RunCommandTool;

impl RustTool for RunCommandTool {
    type Params = RunCommandParams;
    const NAME: &'static str = "run_command";
    const DESCRIPTION: &'static str = "Runs a command.";
    async fn call(
        &self,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        assert!(params.timeout.is_none());
        assert!(params.env.is_none());
        Ok(format!("Ran: {}", params.command).into())
    }
}

// ── ToolDefinition tests ─────────────────────────────────────────

mod basic;
mod macros;
