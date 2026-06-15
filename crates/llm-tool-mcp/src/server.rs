//! MCP stdio server backed by a [`ToolRegistry`].
//!
//! [`McpServer`] wraps a [`ToolRegistry`] and exposes it via the
//! [Model Context Protocol](https://modelcontextprotocol.io/) over any
//! `BufRead`/`Write` pair (typically stdin/stdout).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐   JSON-RPC    ┌───────────┐   dispatch   ┌──────────────┐
//! │  Client  │──────────────▶│ McpServer │─────────────▶│ ToolRegistry │
//! │ (stdin)  │◀──────────────│           │◀─────────────│              │
//! └─────────┘   JSON-RPC    └───────────┘   Result     └──────────────┘
//! ```
//!
//! # Performance
//!
//! - MCP tool schemas are computed **once** at construction and cached.
//! - [`run`](McpServer::run) creates a **single** tokio `current_thread`
//!   runtime, reused for all dispatches.
//! - The `"2.0"` JSON-RPC version is a `&'static str` to avoid allocation.

use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};

use llm_tool::{ToolContext, ToolDefinition, ToolRegistry};
use tracing::{debug, error, info};

use crate::protocol::{
    self, Capabilities, ContentItem, InitializeResult, JsonRpcRequest, JsonRpcResponse,
    McpToolSchema, ServerInfo, ToolCallParams, ToolCallResult, ToolCapabilities, ToolsListResult,
};

/// An MCP server that serves tools from a [`ToolRegistry`] over JSON-RPC.
///
/// # Example
///
/// ```rust
/// use llm_tool::{ToolContext, ToolError, ToolRegistry, llm_tool};
/// use llm_tool_mcp::McpServer;
///
/// /// Adds two numbers.
/// #[llm_tool]
/// fn add(
///     /// First operand.
///     a: i64,
///     /// Second operand.
///     b: i64,
/// ) -> Result<String, ToolError> {
///     Ok(format!("{}", a + b))
/// }
///
/// let registry = ToolRegistry::new().with_tool(Add);
/// let ctx = ToolContext::new(Some("my-agent".into()));
///
/// let server = McpServer::new("my-server", "0.1.0", registry)
///     .with_context(ctx);
///
/// // In production: server.run_stdio().expect("MCP server failed");
/// // Here we prove it works with an in-memory request:
/// let input = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":1,"b":2}}}"#;
/// let reader = std::io::Cursor::new(format!("{input}\n"));
/// let mut output = Vec::new();
/// server.run(reader, &mut output).unwrap();
///
/// let resp: serde_json::Value = serde_json::from_slice(&output).unwrap();
/// assert_eq!(resp["result"]["content"][0]["text"], "3");
/// ```
pub struct McpServer {
    name: String,
    version: String,
    registry: ToolRegistry,
    context: ToolContext,
    /// Pre-computed MCP tool schemas — built once at construction,
    /// wrapped in `Arc` so `tools/list` clones a pointer, not the tree.
    cached_tools_list: Arc<ToolsListResult>,
}

impl McpServer {
    /// Create a new MCP server.
    ///
    /// The `name` and `version` are reported in the MCP `initialize` response.
    /// Tools are served from the given [`ToolRegistry`].
    ///
    /// Tool schemas are computed **once** here and cached for all subsequent
    /// `tools/list` requests.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        let cached_tools_list = Arc::new(build_tools_list_response(&registry));
        Self {
            name: name.into(),
            version: version.into(),
            registry,
            context: ToolContext::new(None),
            cached_tools_list,
        }
    }

    /// Set the [`ToolContext`] used for all tool dispatches.
    ///
    /// The context provides the conversation ID and a shared state store
    /// that persists across tool calls.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = context;
        self
    }

    /// Borrow the underlying [`ToolRegistry`].
    ///
    /// Useful for extracting definitions or dispatching outside MCP.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    // ── Public entry points ─────────────────────────────────────────

    /// Run the server on stdin/stdout.
    ///
    /// Reads JSON-RPC lines from stdin, dispatches them, and writes
    /// responses to stdout.  Blocks until stdin is closed.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    /// Use [`handle_request`](Self::handle_request) instead for async
    /// contexts.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created.
    pub fn run_stdio(&self) -> io::Result<()> {
        self.run(io::stdin().lock(), io::stdout().lock())
    }

    /// Run the server on arbitrary reader/writer streams.
    ///
    /// Creates a single-threaded tokio runtime for async tool dispatch
    /// and reuses it for every request.  The runtime is dropped when the
    /// reader is exhausted.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    /// Use [`handle_request`](Self::handle_request) instead for async
    /// contexts, or use [`run_async`](Self::run_async) with
    /// [`tokio::task::spawn_blocking`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or a fatal
    /// write error occurs.
    pub fn run(&self, reader: impl BufRead, mut writer: impl Write) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        self.run_with_runtime(&rt, reader, &mut writer)
    }

    /// Run the server using an externally-provided tokio runtime.
    ///
    /// Use this when you already have a runtime and want to avoid the
    /// nested-runtime panic.  For the common case (standalone binary),
    /// prefer [`run_stdio`](Self::run_stdio) or [`run`](Self::run).
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal write errors.
    pub fn run_with_runtime(
        &self,
        rt: &tokio::runtime::Runtime,
        reader: impl BufRead,
        writer: &mut impl Write,
    ) -> io::Result<()> {
        for line_result in reader.lines() {
            let line = line_result?;

            if line.trim().is_empty() {
                continue;
            }

            debug!(request = %line, "mcp request");

            let response = rt.block_on(self.handle_request(&line));
            let json = serde_json::to_string(&response).map_err(|e| {
                error!(error = %e, "failed to serialize JSON-RPC response");
                io::Error::other(e)
            })?;

            debug!(response = %json, "mcp response");

            writeln!(writer, "{json}")?;
            writer.flush()?;
        }

        info!("input stream closed — shutting down");
        Ok(())
    }

    /// Handle a single JSON-RPC request string.
    ///
    /// This is the async core used by [`run`](Self::run). Call it directly
    /// when building a custom transport (WebSocket, HTTP, etc.), or for
    /// testing.
    ///
    /// Safe to call from within an existing tokio runtime.
    pub async fn handle_request(&self, line: &str) -> JsonRpcResponse {
        let request: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                return JsonRpcResponse::error(
                    None,
                    protocol::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                );
            }
        };

        let id = request.id.clone();

        match request.method.as_str() {
            "initialize" => self.handle_initialize(id),
            // MCP clients may send `initialized` as a notification — acknowledge it.
            "notifications/initialized" | "initialized" => {
                JsonRpcResponse::success(id, serde_json::Map::new())
            }
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, request.params).await,
            other => JsonRpcResponse::error(
                id,
                protocol::METHOD_NOT_FOUND,
                format!("unknown method: {other}"),
            ),
        }
    }

    // ── Method handlers ─────────────────────────────────────────────

    fn handle_initialize(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        info!(server = %self.name, version = %self.version, "MCP initialize");
        JsonRpcResponse::success(
            id,
            InitializeResult {
                protocol_version: "2024-11-05",
                server_info: ServerInfo {
                    name: self.name.clone(),
                    version: self.version.clone(),
                },
                capabilities: Capabilities {
                    tools: ToolCapabilities {},
                },
            },
        )
    }

    fn handle_tools_list(&self, id: Option<serde_json::Value>) -> JsonRpcResponse {
        info!(count = self.registry.len(), "tools/list");
        // Clone the Arc's inner value; the Serialize impl handles conversion.
        JsonRpcResponse::success(id, (*self.cached_tools_list).clone())
    }

    async fn handle_tools_call(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let Some(raw_params) = params else {
            return JsonRpcResponse::error(
                id,
                protocol::INVALID_PARAMS,
                "tools/call requires params with 'name' and 'arguments'",
            );
        };

        let call_params: ToolCallParams = match serde_json::from_value(raw_params) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    protocol::INVALID_PARAMS,
                    format!("invalid tools/call params: {e}"),
                );
            }
        };

        debug!(tool = %call_params.name, "tools/call");

        match self
            .registry
            .dispatch(&call_params.name, call_params.arguments, &self.context)
            .await
        {
            Ok(output) => JsonRpcResponse::success(
                id,
                ToolCallResult {
                    content: vec![ContentItem {
                        content_type: "text",
                        text: output.content().to_owned(),
                    }],
                    is_error: false,
                },
            ),
            Err(e) => {
                // MCP spec: tool execution errors are returned as success with
                // isError=true, not as JSON-RPC errors.  JSON-RPC errors are
                // reserved for protocol-level failures.
                JsonRpcResponse::success(
                    id,
                    ToolCallResult {
                        content: vec![ContentItem {
                            content_type: "text",
                            text: e.to_string(),
                        }],
                        is_error: true,
                    },
                )
            }
        }
    }
}

// ── Schema helpers ──────────────────────────────────────────────────

/// Build the cached `tools/list` response body.
///
/// Called once at [`McpServer::new`] — the result is wrapped in an
/// `Arc` so `tools/list` clones a pointer, not the full JSON tree.
fn build_tools_list_response(registry: &ToolRegistry) -> ToolsListResult {
    let tools = registry
        .definitions()
        .iter()
        .map(definition_to_mcp_schema)
        .collect();
    ToolsListResult { tools }
}

/// Convert a [`ToolDefinition`] to the MCP `tools/list` schema format.
fn definition_to_mcp_schema(def: &ToolDefinition) -> McpToolSchema {
    McpToolSchema {
        name: def.name.clone(),
        description: def.description.clone(),
        input_schema: def.parameter_schema.clone(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use llm_tool::{EmptyParams, RustTool, ToolError, ToolOutput};

    use super::*;

    // ── Test tools ──────────────────────────────────────────────────

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

    // ── handle_request tests ────────────────────────────────────────

    #[tokio::test]
    async fn initialize_returns_server_info() {
        let server = test_server();
        let resp = server
            .handle_request(r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#)
            .await;

        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "test-server");
        assert_eq!(result["serverInfo"]["version"], "0.0.1");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn tools_list_returns_all_registered_tools() {
        let server = test_server();
        let resp = server
            .handle_request(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
            .await;

        assert!(resp.error.is_none());
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert_eq!(tools.len(), 3);

        let mut names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["add", "fail", "whoami"]);

        // Every tool has the required MCP fields.
        for tool in &tools {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert!(tool["inputSchema"].is_object());
        }
    }

    #[tokio::test]
    async fn tools_list_returns_cached_value() {
        let server = test_server();

        // Two calls should return structurally identical results (from cache).
        let resp1 = server
            .handle_request(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
            .await;
        let resp2 = server
            .handle_request(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#)
            .await;

        assert_eq!(
            resp1.result.unwrap()["tools"],
            resp2.result.unwrap()["tools"]
        );
    }

    #[tokio::test]
    async fn tools_call_success() {
        let server = test_server();
        let resp = server
            .handle_request(
                r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":17,"b":25}}}"#,
            )
            .await;

        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert_eq!(text, "42");
        assert!(result.get("isError").is_none());
    }

    #[tokio::test]
    async fn tools_call_tool_error_returns_is_error() {
        let server = test_server();
        let resp = server
            .handle_request(
                r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fail","arguments":{}}}"#,
            )
            .await;

        // Tool errors are MCP-level, NOT JSON-RPC errors.
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("intentional failure")
        );
    }

    #[tokio::test]
    async fn tools_call_unknown_tool() {
        let server = test_server();
        let resp = server
            .handle_request(
                r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nonexistent","arguments":{}}}"#,
            )
            .await;

        // Unknown tool is also a tool-level error.
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[tokio::test]
    async fn tools_call_missing_name() {
        let server = test_server();
        let resp = server
            .handle_request(
                r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"arguments":{}}}"#,
            )
            .await;

        // Missing "name" is a protocol-level error.
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, protocol::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn tools_call_missing_params() {
        let server = test_server();
        let resp = server
            .handle_request(r#"{"jsonrpc":"2.0","id":7,"method":"tools/call"}"#)
            .await;

        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, protocol::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn tools_call_with_default_arguments() {
        let server = test_server();
        // No "arguments" key — should default to empty object.
        let resp = server
            .handle_request(
                r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"fail"}}"#,
            )
            .await;

        // fail tool takes EmptyParams, so empty args is valid.
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }

    #[tokio::test]
    async fn unknown_method() {
        let server = test_server();
        let resp = server
            .handle_request(r#"{"jsonrpc":"2.0","id":9,"method":"resources/list"}"#)
            .await;

        let err = resp.error.unwrap();
        assert_eq!(err.code, protocol::METHOD_NOT_FOUND);
        assert!(err.message.contains("resources/list"));
    }

    #[tokio::test]
    async fn invalid_json() {
        let server = test_server();
        let resp = server.handle_request("not json at all").await;

        let err = resp.error.unwrap();
        assert_eq!(err.code, protocol::PARSE_ERROR);
    }

    #[tokio::test]
    async fn initialized_notification_is_accepted() {
        let server = test_server();
        // Some MCP clients send this after initialize.
        let resp = server
            .handle_request(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .await;
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn context_is_passed_to_tools() {
        let registry = ToolRegistry::new().with_tool(ContextTool);
        let ctx = ToolContext::new(Some("agent-007".into()));
        let server = McpServer::new("test", "1.0", registry).with_context(ctx);

        let resp = server
            .handle_request(
                r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"whoami","arguments":{}}}"#,
            )
            .await;

        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(text, "agent-007");
    }

    // ── run() integration test ──────────────────────────────────────

    #[test]
    fn run_processes_multiple_requests() {
        let server = test_server();

        let input = [
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":1,"b":2}}}"#,
            "",
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fail","arguments":{}}}"#,
        ];
        let input_str = input.join("\n") + "\n";
        let reader = Cursor::new(input_str.as_bytes());

        let mut output = Vec::new();
        server.run(reader, &mut output).unwrap();

        let responses: Vec<serde_json::Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        // 4 responses (blank line is skipped).
        assert_eq!(responses.len(), 4);

        // 1: initialize
        assert_eq!(responses[0]["result"]["serverInfo"]["name"], "test-server");

        // 2: tools/list — 3 tools
        assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 3);

        // 3: add(1, 2) = "3"
        assert_eq!(responses[2]["result"]["content"][0]["text"], "3");
        assert!(responses[2]["result"].get("isError").is_none());

        // 4: fail — isError
        assert_eq!(responses[3]["result"]["isError"], true);
    }

    // ── run_with_runtime test ───────────────────────────────────────

    #[test]
    fn run_with_runtime_reuses_external_runtime() {
        let server = test_server();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let input = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":10,"b":20}}}"#;
        let reader = Cursor::new(format!("{input}\n"));
        let mut output = Vec::new();

        server.run_with_runtime(&rt, reader, &mut output).unwrap();

        let resp: serde_json::Value =
            serde_json::from_str(String::from_utf8(output).unwrap().trim()).unwrap();
        assert_eq!(resp["result"]["content"][0]["text"], "30");
    }

    // ── Accessor tests ──────────────────────────────────────────────

    #[test]
    fn registry_accessor() {
        let server = test_server();
        assert_eq!(server.registry().len(), 3);
    }

    // ── Schema format test ──────────────────────────────────────────

    #[test]
    fn definition_to_mcp_schema_has_correct_keys() {
        let def = ToolDefinition {
            name: "my_tool".into(),
            description: "Does stuff.".into(),
            parameter_schema: serde_json::json!({"type": "object"}),
        };
        let schema = definition_to_mcp_schema(&def);
        assert_eq!(schema.name, "my_tool");
        assert_eq!(schema.description, "Does stuff.");
        assert_eq!(schema.input_schema["type"], "object");
    }
}
