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
    fmt,
    io::{self, BufRead, Write},
    sync::Arc,
};

use llm_tool::{PromptRegistry, ResourceRegistry, ToolContext, ToolDefinition, ToolRegistry};
use tracing::{debug, error, info};

use crate::protocol::{
    self, Capabilities, ContentItem, InitializeResult, JSONRPC_VERSION, JsonRpcRequest,
    JsonRpcResponse, McpToolSchema, PromptCapabilities, ResourceCapabilities, ServerInfo,
    ToolCallParams, ToolCallResult, ToolCapabilities, ToolsListResult,
};

/// A transport for the blocking [`McpServer::serve`] entry point.
///
/// Lets a binary select stdio, TCP, or a Unix domain socket at runtime (e.g.
/// from CLI args) and hand it to [`serve`](McpServer::serve) in a single call,
/// instead of matching over transports and calling the individual blocking
/// runners itself.
///
/// This drives the **blocking** convenience runners. Tokio applications should
/// prefer the async [`run_async`](McpServer::run_async),
/// [`listen_tcp`](McpServer::listen_tcp), and
/// [`listen_unix`](McpServer::listen_unix) methods directly.
///
/// # Example
///
/// ```rust
/// use llm_tool_mcp::Transport;
///
/// let tcp = Transport::Tcp("127.0.0.1:3000".parse().unwrap());
/// // A binary might build one of these from `--stdio` / `--tcp <addr>` flags,
/// // then call `server.serve(transport)`.
/// assert!(matches!(tcp, Transport::Tcp(_)));
/// ```
#[derive(Debug, Clone)]
pub enum Transport {
    /// Serve on stdin/stdout — the standard MCP subprocess transport.
    Stdio,
    /// Serve on the given TCP socket address.
    Tcp(std::net::SocketAddr),
    /// Serve on a Unix domain socket at the given filesystem path.
    #[cfg(unix)]
    Unix(std::path::PathBuf),
}

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
/// let ctx = ToolContext::new().with_conversation_id("my-agent");
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
    /// Pre-serialized `tools/list` result body — built once at construction and
    /// wrapped in `Arc` so `tools/list` clones a pointer plus one JSON value,
    /// never re-serializing the schema tree.
    cached_tools_list: Arc<serde_json::Value>,
    prompts: Arc<PromptRegistry>,
    resources: Arc<ResourceRegistry>,
}

/// Builder for [`McpServer`] that registers prompts and resources up front.
///
/// Prefer this over constructing a server and mutating it: the builder owns
/// its [`PromptRegistry`] and [`ResourceRegistry`] outright, so registration
/// is infallible and never has to unwrap a shared `Arc`.
///
/// # Example
///
/// ```rust
/// use llm_tool::{ToolContext, ToolRegistry};
/// use llm_tool_mcp::McpServer;
///
/// let server = McpServer::builder("srv", "0.1.0", ToolRegistry::new())
///     .with_context(ToolContext::new().with_conversation_id("agent"))
///     .build();
/// # let _server = server;
/// ```
pub struct McpServerBuilder {
    name: String,
    version: String,
    registry: ToolRegistry,
    context: ToolContext,
    prompts: PromptRegistry,
    resources: ResourceRegistry,
}

impl McpServerBuilder {
    /// Start building a server that serves tools from `registry`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            registry,
            context: ToolContext::new(),
            prompts: PromptRegistry::new(),
            resources: ResourceRegistry::new(),
        }
    }

    /// Set the [`ToolContext`] used for all tool dispatches.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = context;
        self
    }

    /// Register a prompt template.
    #[must_use]
    pub fn with_prompt<P: llm_tool::RustPrompt + 'static>(mut self, prompt: P) -> Self {
        self.prompts.register(prompt);
        self
    }

    /// Register a resource (or resource template).
    #[must_use]
    pub fn with_resource<R: llm_tool::RustResource + 'static>(mut self, resource: R) -> Self {
        self.resources.register(resource);
        self
    }

    /// Finish building the [`McpServer`].
    ///
    /// Tool schemas are computed and serialized **once** here and cached for
    /// all subsequent `tools/list` requests.
    #[must_use]
    pub fn build(self) -> McpServer {
        let cached_tools_list = Arc::new(build_tools_list_value(&self.registry));
        McpServer {
            name: self.name,
            version: self.version,
            registry: Arc::new(self.registry),
            context: Arc::new(self.context),
            cached_tools_list,
            prompts: Arc::new(self.prompts),
            resources: Arc::new(self.resources),
        }
    }
}

impl McpServer {
    /// Create a new MCP server serving tools from `registry`.
    ///
    /// The `name` and `version` are reported in the MCP `initialize` response.
    /// To also register prompts or resources, use [`builder`](Self::builder).
    ///
    /// Tool schemas are computed **once** here and cached for all subsequent
    /// `tools/list` requests.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        McpServerBuilder::new(name, version, registry).build()
    }

    /// Begin building a server, allowing prompts and resources to be registered
    /// before [`build`](McpServerBuilder::build).
    #[must_use]
    pub fn builder(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> McpServerBuilder {
        McpServerBuilder::new(name, version, registry)
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

    /// Borrow the underlying [`ToolRegistry`].
    ///
    /// Useful for extracting definitions or dispatching outside MCP.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    // ── Public entry points ─────────────────────────────────────────

    /// Run the server on stdin/stdout, blocking until stdin is closed.
    ///
    /// This is a blocking convenience for the common "just serve" binary: it
    /// builds a tokio runtime internally so `main` needs no `async`. If you are
    /// already inside a tokio application, drive the server with the async
    /// [`run_async`](Self::run_async) instead.
    ///
    /// Reads JSON-RPC lines from stdin, dispatches them, and writes
    /// responses to stdout.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    /// Use [`run_async`](Self::run_async) instead for async contexts.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created.
    pub fn run_stdio(&self) -> io::Result<()> {
        self.run(io::stdin().lock(), io::stdout().lock())
    }

    /// Serve MCP over TCP, blocking forever (until a fatal accept error).
    ///
    /// Blocking convenience mirroring [`run_stdio`](Self::run_stdio) for the
    /// TCP transport: it builds a **multi-threaded** tokio runtime internally
    /// (each accepted connection is served on its own task) and blocks on
    /// [`listen_tcp`](Self::listen_tcp). Use it for simple standalone binaries
    /// that only need to serve TCP.
    ///
    /// For tokio applications, prefer the async [`listen_tcp`](Self::listen_tcp)
    /// (or [`run_tcp_listener`](Self::run_tcp_listener)) so the server shares
    /// your existing runtime rather than spinning up a second one.
    ///
    /// - Localhost only: `server.run_tcp("127.0.0.1:3000")?`
    /// - External / Docker: `server.run_tcp("0.0.0.0:8080")?`
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime; use
    /// [`listen_tcp`](Self::listen_tcp) in that case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or binding to the
    /// TCP address fails.
    pub fn run_tcp(&self, addr: impl tokio::net::ToSocketAddrs) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.listen_tcp(addr))
    }

    /// Serve MCP over a Unix domain socket, blocking forever (until a fatal
    /// accept error).
    ///
    /// Blocking convenience mirroring [`run_stdio`](Self::run_stdio) for the
    /// Unix-socket transport: it builds a **multi-threaded** tokio runtime
    /// internally (each accepted connection is served on its own task) and
    /// blocks on [`listen_unix`](Self::listen_unix). Use it for simple
    /// standalone IPC binaries.
    ///
    /// For tokio applications, prefer the async
    /// [`listen_unix`](Self::listen_unix) so the server shares your existing
    /// runtime.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime; use
    /// [`listen_unix`](Self::listen_unix) in that case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or binding to the
    /// domain socket path fails.
    #[cfg(unix)]
    pub fn run_unix(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.listen_unix(path))
    }

    /// Serve MCP over the given [`Transport`], blocking until it finishes.
    ///
    /// A one-call blocking dispatcher that collapses a stdio-vs-TCP-vs-Unix
    /// selection into a single line — ideal for a binary that picks its
    /// transport from CLI args or config:
    ///
    /// ```no_run
    /// # use llm_tool::ToolRegistry;
    /// # use llm_tool_mcp::{McpServer, Transport};
    /// let server = McpServer::new("srv", "0.1.0", ToolRegistry::new());
    /// let transport = Transport::Tcp("127.0.0.1:3000".parse().unwrap());
    /// server.serve(transport).expect("server failed");
    /// ```
    ///
    /// This is a blocking convenience built on [`run_stdio`](Self::run_stdio),
    /// [`run_tcp`](Self::run_tcp) and [`run_unix`](Self::run_unix). Tokio
    /// applications should instead drive the async
    /// [`run_async`](Self::run_async) / [`listen_tcp`](Self::listen_tcp) /
    /// [`listen_unix`](Self::listen_unix) methods directly.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or the underlying
    /// transport fails to bind / serve.
    pub fn serve(&self, transport: Transport) -> io::Result<()> {
        match transport {
            Transport::Stdio => self.run_stdio(),
            Transport::Tcp(addr) => self.run_tcp(addr),
            #[cfg(unix)]
            Transport::Unix(path) => self.run_unix(path),
        }
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
    /// Use [`run_async`](Self::run_async) instead for async contexts, or use
    /// [`run`](Self::run) inside [`tokio::task::spawn_blocking`].
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
        // Reused across messages so response serialization amortizes to zero
        // allocations on the hot path.
        let mut out_buf: Vec<u8> = Vec::new();
        for line_result in reader.lines() {
            let line = line_result?;

            if line.trim().is_empty() {
                continue;
            }

            debug!(request = %line, "mcp request");

            let Some(outcome) = rt.block_on(self.handle_message(&line)) else {
                debug!("dropping notification response");
                continue;
            };

            out_buf.clear();
            outcome.write_json(&mut out_buf);
            debug!(response = %String::from_utf8_lossy(&out_buf), "mcp response");
            out_buf.push(b'\n');

            writer.write_all(&out_buf)?;
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
        // Reused across messages so response serialization amortizes to zero
        // allocations on the hot path.
        let mut out_buf: Vec<u8> = Vec::new();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            debug!(request = %line, "mcp request");

            let Some(outcome) = self.handle_message(&line).await else {
                debug!("dropping notification response");
                continue;
            };

            out_buf.clear();
            outcome.write_json(&mut out_buf);
            debug!(response = %String::from_utf8_lossy(&out_buf), "mcp response");
            out_buf.push(b'\n');

            writer.write_all(&out_buf).await?;
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
            if let Err(e) = std::fs::remove_file(path) {
                tracing::warn!(path = ?path, error = %e, "failed to remove stale socket file");
            }
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

    /// Handle exactly one JSON-RPC request string, always producing a response.
    ///
    /// Internal single-request core shared by [`handle_message`](Self::handle_message)
    /// and the crate's own tests. Public callers should use `handle_message`,
    /// which additionally understands batches and notification-only input.
    ///
    /// Safe to call from within an existing tokio runtime.
    pub(crate) async fn handle_request(&self, line: &str) -> JsonRpcResponse {
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

    /// Handle one JSON-RPC *message* and return the response to send back.
    ///
    /// This is **the** entry point for building a custom transport (Axum HTTP,
    /// `WebSockets`, a message queue, etc.). The [`run`](Self::run) family is
    /// built on top of it. A JSON-RPC message is either:
    ///
    /// - a single request/notification object (`{ ... }`), or
    /// - a batch array of them (`[ { ... }, ... ]`).
    ///
    /// Each request's `method` is dispatched against this server's
    /// [`ToolRegistry`] (`tools/list`, `tools/call`) and any registered prompts
    /// (`prompts/*`) and resources (`resources/*`); see the crate docs for the
    /// full method table.
    ///
    /// The result is a structured [`RpcOutcome`] you inspect or render to the
    /// wire in a single pass via [`to_wire`](RpcOutcome::to_wire) /
    /// [`Display`](core::fmt::Display) / [`write_json`](RpcOutcome::write_json):
    ///
    /// - <code>Some([Single](RpcOutcome::Single))</code> — one request → one response.
    /// - <code>Some([Batch](RpcOutcome::Batch))</code> — a batch → one response each,
    ///   for the non-notification members.
    /// - `None` — the input was purely notification(s); send nothing back
    ///   (e.g. reply `202 Accepted` with no body over HTTP).
    pub async fn handle_message(&self, line: &str) -> Option<RpcOutcome> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }

        if trimmed.as_bytes().first() == Some(&b'[') {
            return self.dispatch_batch(trimmed).await;
        }

        let response = self.handle_request(trimmed).await;
        if response.id.is_none() {
            None
        } else {
            Some(RpcOutcome::Single(response))
        }
    }

    /// Handle a JSON-RPC 2.0 batch request array, collecting one response per
    /// non-notification member.
    async fn dispatch_batch(&self, line: &str) -> Option<RpcOutcome> {
        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                return Some(RpcOutcome::Single(JsonRpcResponse::error(
                    None,
                    protocol::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                )));
            }
        };

        let Some(arr) = val.as_array() else {
            return Some(RpcOutcome::Single(JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "expected JSON array for batch request",
            )));
        };

        if arr.is_empty() {
            return Some(RpcOutcome::Single(JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "batch request array cannot be empty",
            )));
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
                responses.push(resp);
            }
        }

        if responses.is_empty() {
            None
        } else {
            Some(RpcOutcome::Batch(responses))
        }
    }

    /// Dispatch a validated JSON-RPC request to the appropriate method handler.
    async fn dispatch_method(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();

        match request.method.as_str() {
            protocol::METHOD_INITIALIZE => self.handle_initialize(id, request.params.as_ref()),
            // No-op acknowledgements: liveness/log-level, plus the `initialized`
            // notification some clients send (both namespaced and bare forms).
            protocol::METHOD_PING
            | protocol::METHOD_LOGGING_SET_LEVEL
            | protocol::METHOD_NOTIFICATIONS_INITIALIZED
            | protocol::METHOD_INITIALIZED => {
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            // Cancellation notifications — acknowledge silently.
            protocol::METHOD_NOTIFICATIONS_CANCELLED => {
                debug!("received cancellation notification");
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            protocol::METHOD_TOOLS_LIST => self.handle_tools_list(id),
            protocol::METHOD_TOOLS_CALL => self.handle_tools_call(id, request.params).await,
            protocol::METHOD_RESOURCES_LIST => {
                let list = protocol::ResourcesListResult {
                    resources: self
                        .resources
                        .definitions()
                        .into_iter()
                        .map(|def| protocol::Resource {
                            uri: def.uri_template,
                            name: def.name,
                            description: def.description,
                            mime_type: def.mime_type,
                        })
                        .collect(),
                };
                JsonRpcResponse::success(id, list)
            }
            protocol::METHOD_RESOURCES_TEMPLATES_LIST => {
                let list = protocol::ResourceTemplatesListResult {
                    resource_templates: self.resources.definitions(),
                };
                JsonRpcResponse::success(id, list)
            }
            protocol::METHOD_RESOURCES_READ => self.handle_resources_read(id, request.params).await,
            protocol::METHOD_PROMPTS_LIST => {
                let list = protocol::PromptsListResult {
                    prompts: self.prompts.definitions(),
                };
                JsonRpcResponse::success(id, list)
            }
            protocol::METHOD_PROMPTS_GET => self.handle_prompts_get(id, request.params).await,
            protocol::METHOD_COMPLETION_COMPLETE => {
                JsonRpcResponse::success(id, protocol::CompletionCompleteResult::default())
            }
            protocol::METHOD_NOTIFICATIONS_PROGRESS | protocol::METHOD_NOTIFICATIONS_MESSAGE => {
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
        // The tools/list body is pre-serialized at construction; clone the
        // cached JSON value directly instead of re-serializing the schema tree.
        JsonRpcResponse {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some((*self.cached_tools_list).clone()),
            error: None,
        }
    }

    /// Dispatch a `tools/call` programmatically and return the typed result.
    ///
    /// This runs the tool through the **exact same** code path as the JSON-RPC
    /// wire handler (`tools/call`), so the not-found and error mapping is
    /// identical: an unknown tool or a failing handler both yield a
    /// [`ToolCallResult`] with [`is_error`](ToolCallResult::is_error) set to
    /// `true` and the message surfaced in the content — never a panic or a
    /// JSON-RPC-level error.
    ///
    /// Prefer this over hand-building JSON-RPC frames when calling a tool from
    /// Rust (e.g. in tests): you get the typed [`ToolCallResult`] directly and
    /// can read [`ToolCallResult::text`] instead of indexing into JSON.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use llm_tool::{ToolContext, ToolError, ToolRegistry, llm_tool};
    /// # use llm_tool_mcp::McpServer;
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
    /// # tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
    /// let server = McpServer::new("srv", "0.1.0", ToolRegistry::new().with_tool(Add));
    /// let result = server.dispatch_tool("add", serde_json::json!({"a": 2, "b": 3})).await;
    /// assert!(!result.is_error);
    /// assert_eq!(result.text(), Some("5"));
    /// # })
    /// ```
    pub async fn dispatch_tool(&self, name: &str, arguments: serde_json::Value) -> ToolCallResult {
        self.call_tool(name, arguments).await
    }

    /// Shared tool-call core: dispatch against the registry and map the outcome
    /// to a [`ToolCallResult`].
    ///
    /// Both the wire handler ([`handle_tools_call`](Self::handle_tools_call))
    /// and the programmatic entry point ([`dispatch_tool`](Self::dispatch_tool))
    /// funnel through here so the success / error / not-found mapping lives in
    /// exactly one place.
    ///
    /// Per the MCP spec, tool execution errors — and unknown tools — are
    /// reported as a result with `isError=true`, not as JSON-RPC errors, so the
    /// model can recover within the turn. JSON-RPC errors are reserved for
    /// protocol-level failures (handled by the caller).
    async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> ToolCallResult {
        debug!(tool = %name, "tools/call");
        match self.registry.dispatch(name, arguments, &self.context).await {
            Ok(output) => ToolCallResult {
                content: vec![ContentItem::text(output.into_content())],
                is_error: false,
            },
            Err(e) => ToolCallResult {
                content: vec![ContentItem::text(e.to_string())],
                is_error: true,
            },
        }
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

        let result = self
            .call_tool(&call_params.name, call_params.arguments)
            .await;
        JsonRpcResponse::success(id, result)
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
        match self
            .prompts
            .render(&get_params.name, get_params.arguments)
            .await
        {
            Ok(output) => {
                let messages = output
                    .messages
                    .into_iter()
                    .map(|m| protocol::PromptMessage {
                        role: m.role.to_string(),
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
        match self.resources.read(&read_params.uri).await {
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

// ── Response serialization ──────────────────────────────────────────

/// A dispatched JSON-RPC result, ready to be inspected or rendered.
///
/// Returned by [`McpServer::handle_message`]. A `None` from that method means
/// there is nothing to send (a notification, or a batch of only
/// notifications); a `Some` is either a [`Single`](Self::Single) response
/// object or a [`Batch`](Self::Batch) array.
///
/// `RpcOutcome` implements [`Serialize`](serde::Serialize) and
/// [`Display`](core::fmt::Display), both rendering a `Single` as a JSON object
/// and a `Batch` as a JSON array. Use
/// [`to_wire`](Self::to_wire) (or `.to_string()`) for a `String`, or
/// [`write_json`](Self::write_json) to append directly to a byte buffer — each
/// serializes in a single pass with no intermediate [`serde_json::Value`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcOutcome {
    /// A single JSON-RPC response object.
    Single(JsonRpcResponse),
    /// A JSON-RPC batch response array (always non-empty).
    Batch(Vec<JsonRpcResponse>),
}

impl serde::Serialize for RpcOutcome {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Single(response) => response.serialize(serializer),
            Self::Batch(responses) => responses.serialize(serializer),
        }
    }
}

impl RpcOutcome {
    /// Render this outcome to a single JSON wire string.
    ///
    /// # Panics
    ///
    /// Panics only if a well-formed MCP response fails to serialize, which
    /// would indicate a bug in this crate.
    #[must_use]
    pub fn to_wire(&self) -> String {
        serde_json::to_string(self).expect("MCP response must be JSON-serializable")
    }

    /// Render this outcome's JSON wire form by appending it to `buf`.
    ///
    /// Lets callers reuse a single buffer across many responses, avoiding a
    /// fresh allocation per message.
    ///
    /// # Panics
    ///
    /// Panics only if a well-formed MCP response fails to serialize, which
    /// would indicate a bug in this crate.
    pub fn write_json(&self, buf: &mut Vec<u8>) {
        serde_json::to_writer(buf, self).expect("MCP response must be JSON-serializable");
    }

    /// Returns `true` if this is a [`Batch`](Self::Batch) of responses.
    #[must_use]
    pub const fn is_batch(&self) -> bool {
        matches!(self, Self::Batch(_))
    }

    /// Consume the outcome into a flat list of responses.
    ///
    /// A [`Single`](Self::Single) yields a one-element `Vec`; a
    /// [`Batch`](Self::Batch) yields its responses unchanged.
    #[must_use]
    pub fn into_responses(self) -> Vec<JsonRpcResponse> {
        match self {
            Self::Single(response) => vec![response],
            Self::Batch(responses) => responses,
        }
    }
}

impl fmt::Display for RpcOutcome {
    /// Renders the JSON wire form (object for `Single`, array for `Batch`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_wire())
    }
}

// ── Schema helpers ──────────────────────────────────────────────────

/// Build and pre-serialize the cached `tools/list` response body.
///
/// Called once at [`McpServerBuilder::build`] — the resulting JSON value is
/// wrapped in an `Arc` so `tools/list` clones a pointer plus one value rather
/// than re-serializing the schema tree on every request.
fn build_tools_list_value(registry: &ToolRegistry) -> serde_json::Value {
    let tools = registry
        .definitions()
        .iter()
        .map(definition_to_mcp_schema)
        .collect();
    let list = ToolsListResult { tools };
    serde_json::to_value(list).expect("tools/list schema must be JSON-serializable")
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
