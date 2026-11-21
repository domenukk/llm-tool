//! JSON-RPC request parsing, validation, dispatch, and the [`RpcOutcome`] wire type.

use std::{collections::HashSet, fmt};

use llm_tool::{ToolContext, ToolRegistry};
use serde::de::{DeserializeSeed, IgnoredAny, SeqAccess, Visitor};
use tracing::{debug, info, warn};

use super::{Connection, McpServer};
use crate::protocol::{
    self, Capabilities, ContentItem, InitializeParams, InitializeResult, JSONRPC_VERSION,
    JsonRpcRequest, JsonRpcResponse, PromptCapabilities, ResourceCapabilities, ServerInfo,
    ToolCallParams, ToolCallResult, ToolCapabilities, deserialize_explicit_id,
};

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum WireString {
    Valid(String),
    Invalid(IgnoredAny),
}

#[derive(Debug, serde::Deserialize)]
struct RawRpcObject {
    #[serde(default)]
    jsonrpc: Option<WireString>,
    #[serde(default, deserialize_with = "deserialize_explicit_id")]
    id: Option<serde_json::Value>,
    #[serde(default)]
    method: Option<WireString>,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum RawRpcEnvelope {
    Object(RawRpcObject),
    NonObject(IgnoredAny),
}

impl RawRpcEnvelope {
    fn is_initialize(&self) -> bool {
        matches!(
            self,
            Self::Object(obj)
                if matches!(&obj.method, Some(WireString::Valid(m)) if m == protocol::METHOD_INITIALIZE)
        )
    }

    fn is_valid_notification(&self) -> bool {
        matches!(
            self,
            Self::Object(obj)
                if obj.id.is_none()
                    && matches!(&obj.jsonrpc, Some(WireString::Valid(v)) if v == JSONRPC_VERSION)
                    && matches!(&obj.method, Some(WireString::Valid(_)))
        )
    }
}

enum BoundedBatchOutcome {
    Valid(Vec<RawRpcEnvelope>),
    TooLarge { actual: usize, max: usize },
}

struct BoundedBatchSeed {
    max_len: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedBatchSeed {
    type Value = BoundedBatchOutcome;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BatchVisitor {
            max_len: usize,
        }

        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = BoundedBatchOutcome;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON-RPC batch array")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let cap = seq.size_hint().map_or(0, |h| h.min(self.max_len));
                let mut items = Vec::with_capacity(cap);
                let mut count = 0usize;
                while items.len() < self.max_len {
                    match seq.next_element::<RawRpcEnvelope>()? {
                        Some(elem) => {
                            items.push(elem);
                            count += 1;
                        }
                        None => return Ok(BoundedBatchOutcome::Valid(items)),
                    }
                }
                while let Some(_ignored) = seq.next_element::<IgnoredAny>()? {
                    count += 1;
                }
                if count > self.max_len {
                    Ok(BoundedBatchOutcome::TooLarge {
                        actual: count,
                        max: self.max_len,
                    })
                } else {
                    Ok(BoundedBatchOutcome::Valid(items))
                }
            }
        }

        deserializer.deserialize_seq(BatchVisitor {
            max_len: self.max_len,
        })
    }
}

enum ParsedRpcKind {
    Empty,
    Oversized { max_bytes: usize },
    ParseError(String),
    BatchEmpty,
    BatchTooLarge { actual: usize, max: usize },
    Single(RawRpcEnvelope),
    Batch(Vec<RawRpcEnvelope>),
}

/// Pre-parsed JSON-RPC message so transports parse every wire line strictly once.
pub(super) struct ParsedRpcMessage(ParsedRpcKind);

impl ParsedRpcMessage {
    pub(super) fn contains_initialize(&self) -> bool {
        match &self.0 {
            ParsedRpcKind::Single(env) => env.is_initialize(),
            ParsedRpcKind::Batch(arr) => arr.iter().any(RawRpcEnvelope::is_initialize),
            ParsedRpcKind::Empty
            | ParsedRpcKind::Oversized { .. }
            | ParsedRpcKind::ParseError(_)
            | ParsedRpcKind::BatchEmpty
            | ParsedRpcKind::BatchTooLarge { .. } => false,
        }
    }

    pub(super) fn is_single_notification(&self) -> bool {
        matches!(&self.0, ParsedRpcKind::Single(env) if env.is_valid_notification())
    }
}

fn is_valid_jsonrpc_id_value(val: &serde_json::Value) -> bool {
    match val {
        serde_json::Value::Null | serde_json::Value::String(_) => true,
        serde_json::Value::Number(n) => n.is_i64() || n.is_u64(),
        serde_json::Value::Bool(_) | serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            false
        }
    }
}

impl McpServer {
    fn sanitized_rpc_error(
        &self,
        id: Option<serde_json::Value>,
        code: protocol::RpcErrorCode,
        message: impl Into<String>,
    ) -> JsonRpcResponse {
        let raw = message.into();
        let floor = self.registry.output_floor();
        let clean = floor.apply_error(llm_tool::ToolError::new(raw)).message;
        JsonRpcResponse::error(id, code, clean)
    }

    /// Parse a raw JSON-RPC wire line once, enforcing line-byte and batch-length bounds.
    pub(super) fn parse_message(&self, line: &str) -> ParsedRpcMessage {
        if line.len() > self.max_line_bytes {
            return ParsedRpcMessage(ParsedRpcKind::Oversized {
                max_bytes: self.max_line_bytes,
            });
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return ParsedRpcMessage(ParsedRpcKind::Empty);
        }

        if trimmed.as_bytes().first() == Some(&b'[') {
            let mut de = serde_json::Deserializer::from_str(trimmed);
            let seed = BoundedBatchSeed {
                max_len: self.max_batch_len,
            };
            let batch_outcome = match seed.deserialize(&mut de) {
                Ok(outcome) => match de.end() {
                    Ok(()) => outcome,
                    Err(e) => {
                        return ParsedRpcMessage(ParsedRpcKind::ParseError(format!(
                            "invalid JSON: {e}"
                        )));
                    }
                },
                Err(e) => {
                    return ParsedRpcMessage(ParsedRpcKind::ParseError(format!(
                        "invalid JSON: {e}"
                    )));
                }
            };
            return ParsedRpcMessage(match batch_outcome {
                BoundedBatchOutcome::Valid(arr) if arr.is_empty() => ParsedRpcKind::BatchEmpty,
                BoundedBatchOutcome::Valid(arr) => ParsedRpcKind::Batch(arr),
                BoundedBatchOutcome::TooLarge { actual, max } => {
                    ParsedRpcKind::BatchTooLarge { actual, max }
                }
            });
        }

        ParsedRpcMessage(match serde_json::from_str::<RawRpcEnvelope>(trimmed) {
            Ok(env) => ParsedRpcKind::Single(env),
            Err(e) => ParsedRpcKind::ParseError(format!("invalid JSON: {e}")),
        })
    }

    pub(super) async fn dispatch_parsed_conn(
        &self,
        parsed: ParsedRpcMessage,
        conn: &Connection,
    ) -> Option<RpcOutcome> {
        match parsed.0 {
            ParsedRpcKind::Empty => None,
            ParsedRpcKind::Oversized { max_bytes } => {
                Some(RpcOutcome::Single(JsonRpcResponse::error(
                    None,
                    protocol::INVALID_REQUEST,
                    format!("JSON-RPC message exceeds maximum length of {max_bytes} bytes"),
                )))
            }
            ParsedRpcKind::ParseError(msg) => Some(RpcOutcome::Single(self.sanitized_rpc_error(
                None,
                protocol::PARSE_ERROR,
                msg,
            ))),
            ParsedRpcKind::BatchEmpty => Some(RpcOutcome::Single(JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "batch request array cannot be empty",
            ))),
            ParsedRpcKind::BatchTooLarge { actual, max } => {
                Some(RpcOutcome::Single(JsonRpcResponse::error(
                    None,
                    protocol::INVALID_REQUEST,
                    format!("batch request length ({actual}) exceeds maximum of {max}"),
                )))
            }
            ParsedRpcKind::Single(envelope) => {
                let (is_notification, response) = self.dispatch_envelope(envelope, conn).await;
                if is_notification {
                    None
                } else {
                    Some(RpcOutcome::Single(response))
                }
            }
            ParsedRpcKind::Batch(arr) => self.dispatch_batch_items(arr, conn).await,
        }
    }

    #[cfg(test)]
    pub(super) async fn handle_request(&self, line: &str) -> JsonRpcResponse {
        self.handle_request_conn(line, &Connection::default()).await
    }

    #[cfg(test)]
    async fn handle_request_conn(&self, line: &str, conn: &Connection) -> JsonRpcResponse {
        if line.len() > self.max_line_bytes {
            return JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                format!(
                    "JSON-RPC message exceeds maximum length of {} bytes",
                    self.max_line_bytes
                ),
            );
        }
        if let Some(first_non_ws) = line.trim_start().as_bytes().first()
            && *first_non_ws == b'['
        {
            return JsonRpcResponse::error(
                None,
                protocol::INVALID_REQUEST,
                "batch requests must be processed via handle_message or run/run_async",
            );
        }

        let envelope: RawRpcEnvelope = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                return self.sanitized_rpc_error(
                    None,
                    protocol::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                );
            }
        };

        let (_is_notification, response) = self.dispatch_envelope(envelope, conn).await;
        response
    }

    /// Validate a parsed JSON-RPC 2.0 envelope and dispatch it.
    async fn dispatch_envelope(
        &self,
        envelope: RawRpcEnvelope,
        conn: &Connection,
    ) -> (bool, JsonRpcResponse) {
        let RawRpcEnvelope::Object(raw) = envelope else {
            return (
                false,
                JsonRpcResponse::error(
                    None,
                    protocol::INVALID_REQUEST,
                    "expected JSON-RPC request object",
                ),
            );
        };

        if let Some(ref id_val) = raw.id
            && !is_valid_jsonrpc_id_value(id_val)
        {
            return (
                false,
                JsonRpcResponse::error(
                    None,
                    protocol::INVALID_REQUEST,
                    "invalid JSON-RPC request: 'id' must be a string, integer, or null",
                ),
            );
        }

        let id = raw.id;
        let version = match raw.jsonrpc {
            Some(WireString::Valid(v)) => v,
            Some(WireString::Invalid(_)) => {
                return (
                    false,
                    JsonRpcResponse::error(
                        id,
                        protocol::INVALID_REQUEST,
                        "invalid JSON-RPC request: 'jsonrpc' field must be a string",
                    ),
                );
            }
            None => {
                return (
                    false,
                    JsonRpcResponse::error(
                        id,
                        protocol::INVALID_REQUEST,
                        "invalid JSON-RPC request: missing 'jsonrpc' field",
                    ),
                );
            }
        };

        let method = match raw.method {
            Some(WireString::Valid(m)) => m,
            Some(WireString::Invalid(_)) => {
                return (
                    false,
                    JsonRpcResponse::error(
                        id,
                        protocol::INVALID_REQUEST,
                        "invalid JSON-RPC request: 'method' field must be a string",
                    ),
                );
            }
            None => {
                return (
                    false,
                    JsonRpcResponse::error(
                        id,
                        protocol::INVALID_REQUEST,
                        "invalid JSON-RPC request: missing 'method' field",
                    ),
                );
            }
        };

        if version != JSONRPC_VERSION {
            return (
                false,
                self.sanitized_rpc_error(
                    id,
                    protocol::INVALID_REQUEST,
                    format!("invalid jsonrpc version: expected \"2.0\", got \"{version}\""),
                ),
            );
        }

        let is_notification = id.is_none();
        let request = JsonRpcRequest {
            version,
            id,
            method,
            params: raw.params,
        };
        let response = self.dispatch_method(request, conn).await;
        (is_notification, response)
    }

    /// Handle one JSON-RPC *message* and return the response to send back.
    pub async fn handle_message(&self, line: &str) -> Option<RpcOutcome> {
        self.handle_message_conn(line, &mut Connection::default())
            .await
    }

    /// Connection-aware variant of [`handle_message`](Self::handle_message).
    pub async fn handle_message_conn(
        &self,
        line: &str,
        conn: &mut Connection,
    ) -> Option<RpcOutcome> {
        let parsed = self.parse_message(line);
        self.dispatch_parsed_conn(parsed, conn).await
    }

    async fn dispatch_batch_items(
        &self,
        arr: Vec<RawRpcEnvelope>,
        conn: &Connection,
    ) -> Option<RpcOutcome> {
        let mut seen_ids: HashSet<protocol::RequestId> = HashSet::with_capacity(arr.len());
        let mut responses = Vec::with_capacity(arr.len());
        for item in arr {
            if let RawRpcEnvelope::Object(ref obj) = item
                && let Some(ref id_val) = obj.id
                && let Some(req_id) = protocol::RequestId::from_json_value(id_val)
                && !seen_ids.insert(req_id.clone())
            {
                responses.push(self.sanitized_rpc_error(
                    Some(id_val.clone()),
                    protocol::INVALID_REQUEST,
                    format!("duplicate request id in batch: {req_id:?}"),
                ));
                continue;
            }
            let (is_notification, resp) = self.dispatch_envelope(item, conn).await;
            if !is_notification {
                responses.push(resp);
            }
        }

        if responses.is_empty() {
            None
        } else {
            Some(RpcOutcome::Batch(responses))
        }
    }

    fn handle_cancelled_notification(params: Option<serde_json::Value>, conn: &Connection) {
        if let Some(raw_params) = params {
            match serde_json::from_value::<protocol::CancelledNotificationParams>(raw_params) {
                Ok(cancelled_params) => {
                    let cancelled = conn.cancel_request(&cancelled_params.request_id);
                    debug!(
                        request_id = ?cancelled_params.request_id,
                        reason = ?cancelled_params.reason,
                        cancelled,
                        "processed cancellation notification"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "invalid params for notifications/cancelled");
                }
            }
        } else {
            debug!("received cancellation notification without params");
        }
    }

    /// Dispatch a validated JSON-RPC request to the appropriate method handler,
    /// rejecting duplicate in-flight request identifiers with `INVALID_REQUEST`.
    async fn dispatch_method(&self, request: JsonRpcRequest, conn: &Connection) -> JsonRpcResponse {
        let id = request.id.clone();
        let typed_id = id.as_ref().and_then(protocol::RequestId::from_json_value);

        let mut cancel_rx_and_guard = match typed_id.as_ref() {
            Some(req_id) => match conn.try_register_in_flight(req_id) {
                Ok(pair) => Some(pair),
                Err(super::DuplicateInFlightIdError) => {
                    return self.sanitized_rpc_error(
                        id,
                        protocol::INVALID_REQUEST,
                        format!("duplicate in-flight request id: {req_id:?}"),
                    );
                }
            },
            None => None,
        };

        let cancel_rx = cancel_rx_and_guard.as_mut().map(|(rx, _)| {
            // Take ownership of `rx` while keeping the RAII `InFlightGuard` alive across `.await`.
            let (_, dummy_rx) = tokio::sync::oneshot::channel();
            core::mem::replace(rx, dummy_rx)
        });

        match request.method.as_str() {
            protocol::METHOD_INITIALIZE => self.handle_initialize(id, request.params, conn),
            protocol::METHOD_PING
            | protocol::METHOD_LOGGING_SET_LEVEL
            | protocol::METHOD_NOTIFICATIONS_INITIALIZED
            | protocol::METHOD_INITIALIZED => {
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            protocol::METHOD_NOTIFICATIONS_CANCELLED => {
                Self::handle_cancelled_notification(request.params, conn);
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            protocol::METHOD_TOOLS_LIST => self.handle_tools_list(id, conn),
            protocol::METHOD_TOOLS_CALL => {
                let ctx = conn.get_or_init_ctx(&self.context);
                let view_owned = conn.get_view();
                let registry = view_owned
                    .as_ref()
                    .map_or_else(|| self.registry.as_ref(), |v| v.registry.as_ref());
                let id_clone = id.clone();
                Self::await_cancellable(
                    id,
                    cancel_rx,
                    self.handle_tools_call(id_clone, request.params, &ctx, registry),
                )
                .await
            }
            protocol::METHOD_RESOURCES_LIST => JsonRpcResponse {
                jsonrpc: JSONRPC_VERSION,
                id,
                result: Some((*self.cached_resources_list).clone()),
                error: None,
            },
            protocol::METHOD_RESOURCES_TEMPLATES_LIST => JsonRpcResponse {
                jsonrpc: JSONRPC_VERSION,
                id,
                result: Some((*self.cached_resource_templates_list).clone()),
                error: None,
            },
            protocol::METHOD_RESOURCES_READ => {
                let ctx = conn.get_or_init_ctx(&self.context);
                let id_clone = id.clone();
                Self::await_cancellable(
                    id,
                    cancel_rx,
                    self.handle_resources_read(id_clone, request.params, &ctx),
                )
                .await
            }
            protocol::METHOD_PROMPTS_LIST => JsonRpcResponse {
                jsonrpc: JSONRPC_VERSION,
                id,
                result: Some((*self.cached_prompts_list).clone()),
                error: None,
            },
            protocol::METHOD_PROMPTS_GET => {
                let ctx = conn.get_or_init_ctx(&self.context);
                let id_clone = id.clone();
                Self::await_cancellable(
                    id,
                    cancel_rx,
                    self.handle_prompts_get(id_clone, request.params, &ctx),
                )
                .await
            }
            protocol::METHOD_COMPLETION_COMPLETE => {
                JsonRpcResponse::success(id, protocol::CompletionCompleteResult::default())
            }
            protocol::METHOD_NOTIFICATIONS_PROGRESS | protocol::METHOD_NOTIFICATIONS_MESSAGE => {
                debug!("received progress/message notification");
                JsonRpcResponse::success(id, protocol::EmptyResult {})
            }
            other => self.sanitized_rpc_error(
                id,
                protocol::METHOD_NOT_FOUND,
                format!("unknown method: {other}"),
            ),
        }
    }

    async fn await_cancellable<F>(
        id: Option<serde_json::Value>,
        cancel_rx: Option<tokio::sync::oneshot::Receiver<()>>,
        fut: F,
    ) -> JsonRpcResponse
    where
        F: std::future::Future<Output = JsonRpcResponse>,
    {
        if let Some(rx) = cancel_rx {
            tokio::select! {
                res = fut => res,
                Ok(()) = rx => {
                    debug!(?id, "request cancelled by client notification");
                    JsonRpcResponse::error(
                        id,
                        protocol::REQUEST_CANCELLED,
                        "request cancelled by client",
                    )
                }
            }
        } else {
            fut.await
        }
    }

    // ── Method handlers ─────────────────────────────────────────────

    const PROTOCOL_VERSION: &str = "2024-11-05";

    /// Resolve the single effective caller identity for both [`ToolContext::conversation_id`]
    /// and [`RegistryFactory`]:
    /// 1. `conn.authenticated_caller()` (transport-bound identity)
    /// 2. `self.caller_resolver` (custom handshake authenticator)
    /// 3. `conn.client_info_name()` **iff** `self.per_connection_identity || self.trust_client_info_caller`
    fn resolve_effective_caller(
        &self,
        conn: &Connection,
        init_params: Option<&InitializeParams>,
    ) -> Option<String> {
        if let Some(auth_caller) = conn.authenticated_caller() {
            return Some(auth_caller.to_owned());
        }
        if let Some(ref resolver) = self.caller_resolver {
            return resolver
                .resolve_caller(init_params)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty());
        }
        let client_info_name = conn.client_info_name();
        if self.per_connection_identity || self.trust_client_info_caller {
            return client_info_name;
        }
        if self.registry_factory.is_some() && client_info_name.is_some() {
            warn!(
                unauthenticated_client = ?client_info_name,
                "ignoring unauthenticated clientInfo.name for caller identity and RegistryFactory (enable .with_per_connection_identity(true) / .trust_client_info_caller(true) or configure Connection::with_authenticated_caller / CallerResolver)"
            );
        }
        None
    }

    fn handle_initialize(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
        conn: &Connection,
    ) -> JsonRpcResponse {
        if !conn.try_mark_initialized() {
            return JsonRpcResponse::error(
                id,
                protocol::INVALID_REQUEST,
                "MCP connection is already initialized",
            );
        }

        info!(server = %self.name, version = %self.version, "MCP initialize");

        let init_params =
            params.and_then(
                |raw| match serde_json::from_value::<InitializeParams>(raw) {
                    Ok(parsed) => Some(parsed),
                    Err(err) => {
                        debug!(error = %err, "ignoring malformed MCP initialize params");
                        None
                    }
                },
            );

        if let Some(ref p) = init_params
            && let Some(ref client_ver) = p.protocol_version
        {
            debug!(
                client_version = %client_ver,
                server_version = Self::PROTOCOL_VERSION,
                "protocol version negotiation"
            );
        }

        let client_info_name = init_params
            .as_ref()
            .and_then(|p| p.client_info.as_ref())
            .and_then(|c| c.name.as_deref())
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(ToOwned::to_owned);
        conn.set_client_info_name(client_info_name);

        let effective_caller = self.resolve_effective_caller(conn, init_params.as_ref());
        conn.set_ctx(self.context.fork_connection(effective_caller.as_deref()));
        conn.set_view(self.resolve_view(effective_caller.as_deref()));

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
        let view_owned = conn.get_view();
        let (count, tools_list) = view_owned.as_ref().map_or_else(
            || (self.registry.len(), &self.cached_tools_list),
            |v| (v.registry.len(), &v.tools_list),
        );
        info!(count, "tools/list");
        JsonRpcResponse {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some((**tools_list).clone()),
            error: None,
        }
    }

    /// Dispatch a `tools/call` programmatically and return the typed result.
    pub async fn dispatch_tool(&self, name: &str, arguments: serde_json::Value) -> ToolCallResult {
        self.call_tool(name, arguments, &self.context, &self.registry)
            .await
    }

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
            Err(err) => {
                let effective = registry
                    .effective_output_sanitization(name)
                    .unwrap_or_else(|| registry.output_floor());
                let sanitized_err = effective.apply_error(err);
                ToolCallResult {
                    content: vec![ContentItem::text(sanitized_err.to_string())],
                    is_error: true,
                }
            }
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
                return self.sanitized_rpc_error(
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
        ctx: &ToolContext,
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
                return self.sanitized_rpc_error(
                    id,
                    protocol::INVALID_PARAMS,
                    format!("invalid params: {e}"),
                );
            }
        };
        let server_floor = self.registry.output_floor();
        let effective = self
            .prompts
            .effective_output_sanitization(&get_params.name)
            .unwrap_or_else(|| self.prompts.output_floor())
            .at_least(server_floor);
        let description = self
            .prompts
            .definition(&get_params.name)
            .and_then(|d| (!d.description.is_empty()).then(|| d.description.clone()));

        match self
            .prompts
            .render_with_context(&get_params.name, get_params.arguments, ctx)
            .await
        {
            Ok(output) => {
                let sanitized_output = effective.apply_prompt(output);
                let messages = sanitized_output
                    .messages
                    .into_iter()
                    .map(|m| protocol::PromptMessage {
                        role: m.role.to_string(),
                        content: protocol::PromptMessageContent::Text { text: m.content },
                    })
                    .collect();
                let res = protocol::GetPromptResult {
                    description,
                    messages,
                };
                JsonRpcResponse::success(id, res)
            }
            Err(err) => {
                let clean_err = effective.apply_error(err);
                JsonRpcResponse::error(id, protocol::INVALID_PARAMS, clean_err.message)
            }
        }
    }

    async fn handle_resources_read(
        &self,
        id: Option<serde_json::Value>,
        params: Option<serde_json::Value>,
        ctx: &ToolContext,
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
                return self.sanitized_rpc_error(
                    id,
                    protocol::INVALID_PARAMS,
                    format!("invalid params: {e}"),
                );
            }
        };
        let server_floor = self.registry.output_floor();
        match self
            .resources
            .read_with_context(&read_params.uri, ctx)
            .await
        {
            Ok(output) => {
                let sanitized_output = server_floor.apply_resource(output);
                let res = protocol::ReadResourceResult {
                    contents: sanitized_output.contents,
                };
                JsonRpcResponse::success(id, res)
            }
            Err(err) => {
                let clean_err = server_floor.apply_error(err);
                JsonRpcResponse::error(id, protocol::INVALID_PARAMS, clean_err.message)
            }
        }
    }
}

// ── Response serialization ──────────────────────────────────────────

/// A dispatched JSON-RPC result, ready to be inspected or rendered.
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
    /// Panics only if a well-formed MCP response fails to serialize.
    #[must_use]
    pub fn to_wire(&self) -> String {
        serde_json::to_string(self).expect("MCP response must be JSON-serializable")
    }

    /// Render this outcome's JSON wire form by appending it to `buf`.
    ///
    /// # Panics
    ///
    /// Panics only if a well-formed MCP response fails to serialize.
    pub fn write_json(&self, buf: &mut Vec<u8>) {
        serde_json::to_writer(buf, self).expect("MCP response must be JSON-serializable");
    }

    /// Returns `true` if this is a [`Batch`](Self::Batch) of responses.
    #[must_use]
    pub const fn is_batch(&self) -> bool {
        matches!(self, Self::Batch(_))
    }

    /// Consume the outcome into a flat list of responses.
    #[must_use]
    pub fn into_responses(self) -> Vec<JsonRpcResponse> {
        match self {
            Self::Single(response) => vec![response],
            Self::Batch(responses) => responses,
        }
    }
}

impl fmt::Display for RpcOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_wire())
    }
}
