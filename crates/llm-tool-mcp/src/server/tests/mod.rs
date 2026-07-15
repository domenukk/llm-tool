pub use std::io::Cursor;

pub use llm_tool::*;

pub use super::*;

mod basic;
mod blocking_and_dispatch;
mod prompts_resources;
mod rmcp_integration;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct AddParams {
    /// First operand.
    a: i64,
    /// Second operand.
    b: i64,
}

struct AddTool;
impl RustTool for AddTool {
    type Params = AddParams;
    const NAME: &'static str = "add";
    const DESCRIPTION: &'static str = "Adds two numbers.";
    async fn call(
        &self,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::new(format!("{}", params.a + params.b)))
    }
}

struct FailTool;
impl RustTool for FailTool {
    type Params = EmptyParams;
    const NAME: &'static str = "fail";
    const DESCRIPTION: &'static str = "Always fails.";
    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Err(ToolError::new("intentional failure"))
    }
}

struct ContextTool;
impl RustTool for ContextTool {
    type Params = EmptyParams;
    const NAME: &'static str = "whoami";
    const DESCRIPTION: &'static str = "Returns the caller identity from context.";
    async fn call(
        &self,
        _params: Self::Params,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::new(
            ctx.conversation_id().unwrap_or("anonymous").to_owned(),
        ))
    }
}

fn test_server() -> McpServer {
    let registry = ToolRegistry::new()
        .with_tool(AddTool)
        .with_tool(FailTool)
        .with_tool(ContextTool);
    McpServer::new("test-server", "0.0.1", registry)
}

/// A code review prompt.
#[llm_prompt]
fn review_prompt(
    /// Programming language.
    lang: String,
    /// Code to review.
    code: String,
) -> Result<String, ToolError> {
    Ok(format!("Review code for {lang}: {code}"))
}

/// An app config resource.
#[llm_resource(
    uri = "file:///config/{app}.json",
    name = "get_config",
    description = "Get app config",
    mime_type = "application/json"
)]
fn config_resource(app: String) -> Result<String, ToolError> {
    Ok(format!(r#"{{"app":"{app}","enabled":true}}"#))
}

pub fn test_server_with_prompts_and_resources() -> McpServer {
    let registry = ToolRegistry::new()
        .with_tool(AddTool)
        .with_tool(FailTool)
        .with_tool(ContextTool);
    McpServer::builder("test-server", "0.0.1", registry)
        .with_prompt(ReviewPrompt)
        .with_resource(ConfigResource)
        .build()
}
