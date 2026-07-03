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
    collections::HashMap,
    io::{self, BufRead, Write},
    sync::Arc,
};

use llm_tool::{ToolContext, ToolDefinition, ToolRegistry};
use tracing::{debug, error, info};

use crate::protocol::{
    self, Capabilities, ContentItem, InitializeResult, JSONRPC_VERSION, JsonRpcRequest,
    JsonRpcResponse, McpToolSchema, PromptCapabilities, ResourceCapabilities, ServerInfo,
    ToolCallParams, ToolCallResult, ToolCapabilities, ToolsListResult,
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
#[derive(Clone)]
pub struct McpServer {
    name: String,
    version: String,
    registry: Arc<ToolRegistry>,
    context: Arc<ToolContext>,
    /// Pre-computed MCP tool schemas — built once at construction,
    /// wrapped in `Arc` so `tools/list` clones a pointer, not the tree.
    cached_tools_list: Arc<ToolsListResult>,
    prompts: Arc<HashMap<&'static str, Box<dyn llm_tool::ErasedPrompt>>>,
    resources: Arc<Vec<Box<dyn llm_tool::ErasedResource>>>,
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
            registry: Arc::new(registry),
            context: Arc::new(ToolContext::new(None)),
            cached_tools_list,
            prompts: Arc::new(HashMap::new()),
            resources: Arc::new(Vec::new()),
        }
    }

    /// Set the [`ToolContext`] used for all tool dispatches.
    ///
    /// The context provides the conversation ID and a shared state store
    /// that persists across tool calls.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = Arc::new(context);
        self
    }

    /// Register a prompt with the MCP server.
    ///
    /// # Panics
    ///
    /// Panics if called after the server has been cloned.
    #[must_use]
    pub fn with_prompt<P: llm_tool::RustPrompt + 'static>(mut self, prompt: P) -> Self {
        let Ok(mut map) = Arc::try_unwrap(self.prompts) else {
            panic!("cannot add prompt after server has been cloned");
        };
        map.insert(P::NAME, Box::new(prompt));
        self.prompts = Arc::new(map);
        self
    }

    /// Register a resource with the MCP server.
    ///
    /// # Panics
    ///
    /// Panics if called after the server has been cloned.
    #[must_use]
    pub fn with_resource<R: llm_tool::RustResource + 'static>(mut self, resource: R) -> Self {
        let Ok(mut vec) = Arc::try_unwrap(self.resources) else {
            panic!("cannot add resource after server has been cloned");
        };
        vec.push(Box::new(resource));
        self.resources = Arc::new(vec);
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
    /// contexts, or use [`run`](Self::run) inside
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

            let Some(response_val) = rt.block_on(self.handle_message(&line)) else {
                debug!("dropping notification response");
                continue;
            };

            let json = serde_json::to_string(&response_val).map_err(|e| {
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

    /// Run the server asynchronously on Tokio reader/writer streams.
    ///
    /// Reads line-delimited JSON-RPC requests from an async reader, dispatches
    /// them asynchronously without blocking threads or spinning up nested runtimes,
    /// and writes serialized responses back to an async writer.
    ///
    /// Ideal for integrating into existing Tokio applications, network servers,
    /// or when running inside an existing async runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal I/O errors reading requests or writing responses.
    pub async fn run_async(
        &self,
        reader: impl tokio::io::AsyncBufRead + Unpin,
        mut writer: impl tokio::io::AsyncWrite + Unpin,
    ) -> io::Result<()> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let mut lines = reader.lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            debug!(request = %line, "mcp request");

            let Some(response_val) = self.handle_message(&line).await else {
                debug!("dropping notification response");
                continue;
            };

            let json = serde_json::to_string(&response_val).map_err(|e| {
                error!(error = %e, "failed to serialize JSON-RPC response");
                io::Error::other(e)
            })?;

            debug!(response = %json, "mcp response");

            writer.write_all(format!("{json}\n").as_bytes()).await?;
            writer.flush().await?;
        }

        info!("input stream closed — shutting down");
        Ok(())
    }

    /// Listen for TCP connections and serve MCP requests asynchronously on each connection.
    ///
    /// This allows remote clients, IDEs, or multi-client agents to connect over TCP:
    /// - Localhost only: `server.listen_tcp("127.0.0.1:3000").await?`
    /// - External / Docker: `server.listen_tcp("0.0.0.0:8080").await?`
    ///
    /// For each incoming connection, a new Tokio task is spawned running [`run_async`](Self::run_async).
    ///
    /// # Errors
    ///
    /// Returns `Err` if binding to the TCP address fails.
    pub async fn listen_tcp(&self, addr: impl tokio::net::ToSocketAddrs) -> io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!(addr = ?listener.local_addr()?, "listening on TCP for MCP connections");
        self.run_tcp_listener(listener).await
    }

    /// Serve MCP requests asynchronously on an existing [`tokio::net::TcpListener`].
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal accept loop errors.
    pub async fn run_tcp_listener(&self, listener: tokio::net::TcpListener) -> io::Result<()> {
        loop {
            let (mut socket, peer_addr) = listener.accept().await?;
            info!(peer = %peer_addr, "accepted MCP TCP connection");

            let server = self.clone();
            tokio::spawn(async move {
                let (reader, writer) = socket.split();
                let reader = tokio::io::BufReader::new(reader);
                if let Err(e) = server.run_async(reader, writer).await {
                    error!(peer = %peer_addr, error = %e, "MCP TCP connection error");
                }
                info!(peer = %peer_addr, "MCP TCP connection closed");
            });
        }
    }

    #[cfg(unix)]
    /// Listen on a Unix domain socket and serve MCP requests asynchronously on each connection.
    ///
    /// This allows local IPC clients to connect over a filesystem domain socket
    /// (e.g. `/tmp/my-agent.sock`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if binding to the domain socket path fails.
    pub async fn listen_unix(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let path = path.as_ref();
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
        let listener = tokio::net::UnixListener::bind(path)?;
        info!(path = ?path, "listening on Unix domain socket for MCP connections");
        self.run_unix_listener(listener).await
    }

    #[cfg(unix)]
    /// Serve MCP requests asynchronously on an existing [`tokio::net::UnixListener`].
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal accept loop errors.
    pub async fn run_unix_listener(&self, listener: tokio::net::UnixListener) -> io::Result<()> {
        loop {
            let (mut socket, _) = listener.accept().await?;
            info!("accepted MCP Unix domain socket connection");

            let server = self.clone();
            tokio::spawn(async move {
                let (reader, writer) = socket.split();
                let reader = tokio::io::BufReader::new(reader);
                if let Err(e) = server.run_async(reader, writer).await {
                    error!(error = %e, "MCP Unix connection error");
                }
                info!("MCP Unix connection closed");
            });
        }
    }

    /// Handle a single JSON-RPC request string.
    ///
    /// This is the async core used by [`run`](Self::run). Call it directly
    /// when building a custom transport (WebSocket, HTTP, etc.), or for
    /// testing.
    ///
    /// Safe to call from within an existing tokio runtime.
    pub async fn handle_request(&self, line: &str) -> JsonRpcResponse {
        // Detect batch requests (JSON arrays) — redirect to handle_message.
        if let Some(first_non_ws) = line.trim_start().as_bytes().first() {
            if *first_non_ws == b'[' {
                return JsonRpcResponse::error(
                    None,
                    protocol::INVALID_REQUEST,
                    "batch requests must be processed via handle_message or run/run_async",
                );
            }
        }

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

        // JSON-RPC 2.0 §4: the "jsonrpc" field MUST be exactly "2.0".
        if request.version != JSONRPC_VERSION {
            return JsonRpcResponse::error(
                request.id,
                protocol::INVALID_REQUEST,
                format!(
                    "invalid jsonrpc version: expected \"2.0\", got \"{}\"",
                    request.version
                ),
            );
        }

        self.dispatch_method(request).await
    }

    /// Handle a raw JSON-RPC stream line (either a single request or a batch array).
    ///
    /// Returns `Some(Value)` containing the serialized JSON-RPC response(s) to send back,
    /// or `None` if the input was solely a notification (or batch of notifications).
    ///
    /// # Panics
    ///
    /// Panics if the response object cannot be serialized to JSON. This should never
    /// happen for well-formed MCP response types.
    pub async fn handle_message(&self, line: &str) -> Option<serde_json::Value> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(first_non_ws) = trimmed.as_bytes().first() {
            if *first_non_ws == b'[' {
                return self.handle_batch_request(trimmed).await;
            }
        }

        let response = self.handle_request(trimmed).await;
        if response.id.is_none() {
            None
        } else {
            Some(serde_json::to_value(&response).expect("MCP response must be JSON-serializable"))
        }
    }

    /// Handle a JSON-RPC 2.0 batch request array.
    async fn handle_batch_request(&self, line: &str) -> Option<serde_json::Value> {
        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    None,
                    protocol::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                );
                return Some(serde_json::to_value(&resp).expect("serializable"));
            }
        };

        let Some(arr) = val.as_array() else {
            let resp = JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "expected JSON array for batch request",
            );
            return Some(serde_json::to_value(&resp).expect("serializable"));
        };

        if arr.is_empty() {
            let resp = JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "batch request array cannot be empty",
            );
            return Some(serde_json::to_value(&resp).expect("serializable"));
        }

        let mut responses = Vec::with_capacity(arr.len());
        for item in arr {
            let resp_opt = match serde_json::from_value::<JsonRpcRequest>(item.clone()) {
                Ok(request) => {
                    if request.version == JSONRPC_VERSION {
                        let resp = self.dispatch_method(request).await;
                        if resp.id.is_none() { None } else { Some(resp) }
                    } else {
                        Some(JsonRpcResponse::error(
                            request.id,
                            protocol::INVALID_REQUEST,
                            format!(
                                "invalid jsonrpc version: expected \"2.0\", got \"{}\"",
                                request.version
                            ),
                        ))
                    }
                }
                Err(e) => {
                    let id = item.as_object().and_then(|o| o.get("id").cloned());
                    Some(JsonRpcResponse::error(
                        id,
                        protocol::INVALID_REQUEST,
                        format!("invalid request object in batch: {e}"),
                    ))
                }
            };

            if let Some(resp) = resp_opt {
                responses.push(serde_json::to_value(&resp).expect("serializable"));
            }
        }

        if responses.is_empty() {
            None
        } else {
            Some(serde_json::Value::Array(responses))
        }
    }

    /// Dispatch a validated JSON-RPC request to the appropriate method handler.
    async fn dispatch_method(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();

        match request.method.as_str() {
            "initialize" => self.handle_initialize(id, request.params.as_ref()),
            "ping" | "logging/setLevel" => JsonRpcResponse::success(id, protocol::EmptyResult {}),
            // MCP clients may send `initialized` as a notification — acknowledge it.
            "notifications/initialized" | "initialized" => {
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            // Cancellation notifications — acknowledge silently.
            "notifications/cancelled" => {
                debug!("received cancellation notification");
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, request.params).await,
            "resources/list" => {
                let list = protocol::ResourcesListResult {
                    resources: self
                        .resources
                        .iter()
                        .map(|r| {
                            let def = r.definition();
                            protocol::Resource {
                                uri: def.uri_template,
                                name: def.name,
                                description: def.description,
                                mime_type: def.mime_type,
                            }
                        })
                        .collect(),
                };
                JsonRpcResponse::success(id, list)
            }
            "resources/templates/list" => {
                let list = protocol::ResourceTemplatesListResult {
                    resource_templates: self.resources.iter().map(|r| r.definition()).collect(),
                };
                JsonRpcResponse::success(id, list)
            }
            "resources/read" => self.handle_resources_read(id, request.params).await,
            "prompts/list" => {
                let list = protocol::PromptsListResult {
                    prompts: self.prompts.values().map(|p| p.definition()).collect(),
                };
                JsonRpcResponse::success(id, list)
            }
            "prompts/get" => self.handle_prompts_get(id, request.params).await,
            "completion/complete" => {
                JsonRpcResponse::success(id, protocol::CompletionCompleteResult::default())
            }
            "notifications/progress" | "notifications/message" => {
                debug!("received progress/message notification");
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            other => JsonRpcResponse::error(
                id,
                protocol::METHOD_NOT_FOUND,
                format!("unknown method: {other}"),
            ),
        }
    }

    // ── Method handlers ─────────────────────────────────────────────

    /// MCP protocol version supported by this server.
    const PROTOCOL_VERSION: &str = "2024-11-05";

    fn handle_initialize(
        &self,
        id: Option<serde_json::Value>,
        params: Option<&serde_json::Value>,
    ) -> JsonRpcResponse {
        info!(server = %self.name, version = %self.version, "MCP initialize");

        // Protocol version negotiation: if the client sends a
        // `protocolVersion` in params, we report our supported version.
        // The server always responds with the version it actually supports.
        if let Some(p) = params {
            if let Some(client_ver) = p.get("protocolVersion").and_then(|v| v.as_str()) {
                debug!(client_version = %client_ver, server_version = Self::PROTOCOL_VERSION, "protocol version negotiation");
            }
        }

        JsonRpcResponse::success(
            id,
            InitializeResult {
                protocol_version: Self::PROTOCOL_VERSION,
                server_info: ServerInfo {
                    name: self.name.clone(),
                    version: self.version.clone(),
                },
                capabilities: Capabilities {
                    tools: ToolCapabilities {},
                    resources: ResourceCapabilities {},
                    prompts: PromptCapabilities {},
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

    async fn handle_prompts_get(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let Some(p_val) = params else {
            return JsonRpcResponse::error(
                id,
                protocol::INVALID_PARAMS,
                "missing params for prompts/get",
            );
        };
        let get_params: protocol::GetPromptParams = match serde_json::from_value(p_val) {
            Ok(v) => v,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    protocol::INVALID_PARAMS,
                    format!("invalid params: {e}"),
                );
            }
        };
        let Some(prompt) = self.prompts.get(get_params.name.as_str()) else {
            return JsonRpcResponse::error(
                id,
                protocol::INVALID_PARAMS,
                format!("unknown prompt: {}", get_params.name),
            );
        };
        let fut = prompt.render_erased(get_params.arguments);
        match fut.await {
            Ok(output) => {
                let messages = output
                    .messages
                    .into_iter()
                    .map(|m| protocol::PromptMessage {
                        role: m.role.into_owned(),
                        content: protocol::PromptMessageContent::Text { text: m.content },
                    })
                    .collect();
                let res = protocol::GetPromptResult {
                    description: None,
                    messages,
                };
                JsonRpcResponse::success(id, res)
            }
            Err(err) => JsonRpcResponse::error(id, protocol::INVALID_PARAMS, err.message),
        }
    }

    async fn handle_resources_read(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let Some(p_val) = params else {
            return JsonRpcResponse::error(
                id,
                protocol::INVALID_PARAMS,
                "missing params for resources/read",
            );
        };
        let read_params: protocol::ReadResourceParams = match serde_json::from_value(p_val) {
            Ok(v) => v,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    protocol::INVALID_PARAMS,
                    format!("invalid params: {e}"),
                );
            }
        };
        let mut matched_fut = None;
        for res in self.resources.iter() {
            if let Some(fut) = res.read_erased(&read_params.uri) {
                matched_fut = Some(fut);
                break;
            }
        }
        let Some(fut) = matched_fut else {
            return JsonRpcResponse::error(
                id,
                protocol::INVALID_PARAMS,
                format!("resource not found matching URI: {}", read_params.uri),
            );
        };
        match fut.await {
            Ok(output) => {
                let res = protocol::ReadResourceResult {
                    contents: output.contents,
                };
                JsonRpcResponse::success(id, res)
            }
            Err(err) => JsonRpcResponse::error(id, protocol::INVALID_PARAMS, err.message),
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
mod tests;
