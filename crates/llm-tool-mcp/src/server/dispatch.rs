//! JSON-RPC request dispatch and the [`RpcOutcome`] wire type.

use std::fmt;

use llm_tool::{ToolContext, ToolRegistry};
use tracing::{debug, info};

use super::{Connection, McpServer};
use crate::protocol::{
    self, Capabilities, ContentItem, InitializeResult, JSONRPC_VERSION, JsonRpcRequest,
    JsonRpcResponse, PromptCapabilities, ResourceCapabilities, ServerInfo, ToolCallParams,
    ToolCallResult, ToolCapabilities,
};

impl McpServer {
    /// Handle exactly one JSON-RPC request string, always producing a response.
    ///
    /// A **test-only** convenience over
    /// [`handle_request_conn`](Self::handle_request_conn) that uses the shared
    /// server identity (no per-connection caller). Production transports drive
    /// `handle_request_conn` so each connection keeps its own negotiated
    /// identity. Public callers should use [`handle_message`](Self::handle_message),
    /// which additionally understands batches and notification-only input.
    ///
    /// Safe to call from within an existing tokio runtime.
    #[cfg(test)]
    pub(crate) async fn handle_request(&self, line: &str) -> JsonRpcResponse {
        self.handle_request_conn(line, &mut Connection::default())
            .await
    }

    /// Connection-aware variant of [`handle_request`](Self::handle_request).
    ///
    /// `conn` carries the caller identity and per-caller registry view
    /// negotiated for this connection's `initialize` handshake; both are threaded
    /// into dispatch so per-connection identity and per-caller tool sets (when
    /// enabled) apply to `tools/list` and `tools/call`.
    async fn handle_request_conn(&self, line: &str, conn: &mut Connection) -> JsonRpcResponse {
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

        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                return JsonRpcResponse::error(
                    None,
                    protocol::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                );
            }
        };

        let Some(obj) = val.as_object() else {
            return JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "expected JSON-RPC request object",
            );
        };

        let id = obj.get("id").cloned();
        let request: JsonRpcRequest = match serde_json::from_value(val) {
            Ok(r) => r,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    protocol::INVALID_REQUEST,
                    format!("invalid JSON-RPC request: {e}"),
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

        self.dispatch_method(request, conn).await
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
        self.handle_message_conn(line, &mut Connection::default())
            .await
    }

    /// Connection-aware variant of [`handle_message`](Self::handle_message).
    ///
    /// `conn` is this connection's negotiated state: its caller identity slot
    /// and per-caller registry view are updated by the `initialize` handshake
    /// (when per-connection identity / a registry factory are enabled) and read
    /// on subsequent `tools/list` and `tools/call`s. The transport run loops own
    /// one such [`Connection`] per connection.
    pub async fn handle_message_conn(
        &self,
        line: &str,
        conn: &mut Connection,
    ) -> Option<RpcOutcome> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }

        if trimmed.as_bytes().first() == Some(&b'[') {
            return self.dispatch_batch(trimmed, conn).await;
        }

        let response = self.handle_request_conn(trimmed, conn).await;
        // JSON-RPC 2.0 §4.1: notifications (requests without an ID that succeeded)
        // must not produce a response. Protocol-level errors with null IDs must be sent.
        if response.id.is_none() && response.error.is_none() {
            None
        } else {
            Some(RpcOutcome::Single(response))
        }
    }

    /// Handle a JSON-RPC 2.0 batch request array, collecting one response per
    /// non-notification member.
    async fn dispatch_batch(&self, line: &str, conn: &mut Connection) -> Option<RpcOutcome> {
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
                        let resp = self.dispatch_method(request, conn).await;
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
    async fn dispatch_method(
        &self,
        request: JsonRpcRequest,
        conn: &mut Connection,
    ) -> JsonRpcResponse {
        let id = request.id.clone();

        match request.method.as_str() {
            protocol::METHOD_INITIALIZE => {
                self.handle_initialize(id, request.params.as_ref(), conn)
            }
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
            protocol::METHOD_TOOLS_LIST => self.handle_tools_list(id, conn),
            protocol::METHOD_TOOLS_CALL => {
                let ctx = conn.ctx.as_ref().unwrap_or(&self.context);
                let registry = conn
                    .view
                    .as_ref()
                    .map_or_else(|| self.registry.as_ref(), |v| v.registry.as_ref());
                self.handle_tools_call(id, request.params, ctx, registry)
                    .await
            }
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
        conn: &mut Connection,
    ) -> JsonRpcResponse {
        info!(server = %self.name, version = %self.version, "MCP initialize");

        // Protocol version negotiation: if the client sends a
        // `protocolVersion` in params, we report our supported version.
        // The server always responds with the version it actually supports.
        let mut caller: Option<&str> = None;
        if let Some(p) = params {
            if let Some(client_ver) = p.get("protocolVersion").and_then(|v| v.as_str()) {
                debug!(client_version = %client_ver, server_version = Self::PROTOCOL_VERSION, "protocol version negotiation");
            }

            // When per-connection identity is enabled, adopt the caller name the
            // client announced in `clientInfo.name` for this connection. The
            // derived context shares the server's state and typed extensions, so
            // every connection sees the same session while acting as itself.
            if self.per_connection_identity {
                if let Some(name) = p
                    .get("clientInfo")
                    .and_then(|c| c.get("name"))
                    .and_then(|n| n.as_str())
                    .map(str::trim)
                    .filter(|n| !n.is_empty())
                {
                    info!(caller = %name, "adopting per-connection caller identity");
                    conn.ctx = Some(self.context.with_caller(name));
                    caller = Some(name);
                } else {
                    debug!(
                        "per-connection identity enabled but client sent no \
                         usable clientInfo.name; using shared server identity"
                    );
                }
            }
        }

        // Resolve this caller's registry view. A no-op (leaves the shared
        // registry in play) when no [`RegistryFactory`] is configured.
        conn.view = self.resolve_view(caller);

        let tools_cap = Some(ToolCapabilities {});
        let prompts_cap = if self.prompts.is_empty() {
            None
        } else {
            Some(PromptCapabilities {})
        };
        let resources_cap = if self.resources.is_empty() {
            None
        } else {
            Some(ResourceCapabilities {})
        };

        JsonRpcResponse::success(
            id,
            InitializeResult {
                protocol_version: Self::PROTOCOL_VERSION,
                server_info: ServerInfo {
                    name: self.name.clone(),
                    version: self.version.clone(),
                },
                instructions: self.instructions.clone(),
                capabilities: Capabilities {
                    tools: tools_cap,
                    resources: resources_cap,
                    prompts: prompts_cap,
                },
            },
        )
    }

    fn handle_tools_list(
        &self,
        id: Option<serde_json::Value>,
        conn: &Connection,
    ) -> JsonRpcResponse {
        // Per-caller registry view when negotiated; otherwise the shared list.
        let (count, tools_list) = conn.view.as_ref().map_or_else(
            || (self.registry.len(), &self.cached_tools_list),
            |v| (v.registry.len(), &v.tools_list),
        );
        info!(count, "tools/list");
        // The tools/list body is pre-serialized (per caller, at first use);
        // clone the cached JSON value instead of re-serializing the schema tree.
        JsonRpcResponse {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some((**tools_list).clone()),
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
        self.call_tool(name, arguments, &self.context, &self.registry)
            .await
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
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
        ctx: &ToolContext,
        registry: &ToolRegistry,
    ) -> ToolCallResult {
        debug!(tool = %name, caller = ?ctx.conversation_id(), "tools/call");
        match registry.dispatch(name, arguments, ctx).await {
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
        ctx: &ToolContext,
        registry: &ToolRegistry,
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
            .call_tool(&call_params.name, call_params.arguments, ctx, registry)
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
