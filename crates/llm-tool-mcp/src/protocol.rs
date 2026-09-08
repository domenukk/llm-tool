//! JSON-RPC 2.0 protocol types for MCP communication.
//!
//! These types model the wire format used by MCP's JSON-RPC transport.
//! Each request/response is a single JSON line on the stream.

use serde::{Deserialize, Serialize};

// ── JSON-RPC 2.0 constants ──────────────────────────────────────────

/// The only valid JSON-RPC protocol version.
pub const JSONRPC_VERSION: &str = "2.0";

// ── Strongly-typed JSON-RPC 2.0 / MCP error codes ───────────────────

/// Strongly-typed JSON-RPC 2.0 / MCP error code.
///
/// Serializes transparently as an integer on the wire (`-32700`, `-32800`, etc.)
/// while providing exhaustive pattern matching for standard protocol error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RpcErrorCode {
    /// Malformed JSON (`-32700`).
    ParseError,
    /// Valid JSON but not a valid JSON-RPC request (`-32600`).
    InvalidRequest,
    /// The requested method does not exist (`-32601`).
    MethodNotFound,
    /// Invalid method parameters (`-32602`).
    InvalidParams,
    /// Internal server error (`-32603`).
    InternalError,
    /// Request cancelled by client notification (`-32800`).
    RequestCancelled,
    /// Custom or application-specific error code.
    Custom(i64),
}

impl RpcErrorCode {
    const CODE_PARSE_ERROR: i64 = -32700;
    const CODE_INVALID_REQUEST: i64 = -32600;
    const CODE_METHOD_NOT_FOUND: i64 = -32601;
    const CODE_INVALID_PARAMS: i64 = -32602;
    const CODE_INTERNAL_ERROR: i64 = -32603;
    const CODE_REQUEST_CANCELLED: i64 = -32800;

    /// Return the underlying `i64` JSON-RPC wire code.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        match self {
            Self::ParseError => Self::CODE_PARSE_ERROR,
            Self::InvalidRequest => Self::CODE_INVALID_REQUEST,
            Self::MethodNotFound => Self::CODE_METHOD_NOT_FOUND,
            Self::InvalidParams => Self::CODE_INVALID_PARAMS,
            Self::InternalError => Self::CODE_INTERNAL_ERROR,
            Self::RequestCancelled => Self::CODE_REQUEST_CANCELLED,
            Self::Custom(code) => code,
        }
    }

    /// Construct an [`RpcErrorCode`] from its wire integer representation.
    #[must_use]
    pub const fn from_i64(code: i64) -> Self {
        match code {
            Self::CODE_PARSE_ERROR => Self::ParseError,
            Self::CODE_INVALID_REQUEST => Self::InvalidRequest,
            Self::CODE_METHOD_NOT_FOUND => Self::MethodNotFound,
            Self::CODE_INVALID_PARAMS => Self::InvalidParams,
            Self::CODE_INTERNAL_ERROR => Self::InternalError,
            Self::CODE_REQUEST_CANCELLED => Self::RequestCancelled,
            other => Self::Custom(other),
        }
    }
}

impl Serialize for RpcErrorCode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i64(self.as_i64())
    }
}

impl<'de> Deserialize<'de> for RpcErrorCode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let code = i64::deserialize(deserializer)?;
        Ok(Self::from_i64(code))
    }
}

impl core::fmt::Display for RpcErrorCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_i64())
    }
}

impl PartialEq<i64> for RpcErrorCode {
    fn eq(&self, other: &i64) -> bool {
        self.as_i64() == *other
    }
}

impl From<RpcErrorCode> for i64 {
    fn from(code: RpcErrorCode) -> Self {
        code.as_i64()
    }
}

impl From<i64> for RpcErrorCode {
    fn from(code: i64) -> Self {
        Self::from_i64(code)
    }
}

/// Malformed JSON (`-32700`).
pub const PARSE_ERROR: RpcErrorCode = RpcErrorCode::ParseError;

/// Valid JSON but not a valid JSON-RPC request (`-32600`).
pub const INVALID_REQUEST: RpcErrorCode = RpcErrorCode::InvalidRequest;

/// The requested method does not exist (`-32601`).
pub const METHOD_NOT_FOUND: RpcErrorCode = RpcErrorCode::MethodNotFound;

/// Invalid method parameters (`-32602`).
pub const INVALID_PARAMS: RpcErrorCode = RpcErrorCode::InvalidParams;

/// Internal server error (`-32603`).
pub const INTERNAL_ERROR: RpcErrorCode = RpcErrorCode::InternalError;

/// Request cancelled by client notification (`notifications/cancelled`).
pub const REQUEST_CANCELLED: RpcErrorCode = RpcErrorCode::RequestCancelled;

// ── Strongly-typed RequestId ────────────────────────────────────────

/// Strongly-typed JSON-RPC 2.0 request identifier for zero-allocation
/// hash-map indexing of in-flight requests.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// Numeric request ID (e.g. `1`, `42`).
    Number(i64),
    /// String request ID (e.g. `"req-1"`).
    String(String),
}

impl RequestId {
    /// Extract a strongly-typed [`RequestId`] from a JSON value if it is a
    /// valid JSON-RPC 2.0 identifier (`Number` or `String`).
    #[must_use]
    pub fn from_json_value(val: &serde_json::Value) -> Option<Self> {
        match val {
            serde_json::Value::Number(n) => n.as_i64().map(Self::Number),
            serde_json::Value::String(s) => Some(Self::String(s.clone())),
            _ => None,
        }
    }
}

/// Error returned when converting a non-identifier JSON value into a [`RequestId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidRequestIdError {
    /// Description of the invalid JSON type encountered.
    pub actual_type: &'static str,
}

impl core::fmt::Display for InvalidRequestIdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "invalid JSON-RPC request id: expected integer or string, got {}",
            self.actual_type
        )
    }
}

impl std::error::Error for InvalidRequestIdError {}

impl TryFrom<&serde_json::Value> for RequestId {
    type Error = InvalidRequestIdError;

    fn try_from(val: &serde_json::Value) -> Result<Self, Self::Error> {
        match val {
            serde_json::Value::Number(n) => {
                n.as_i64().map(Self::Number).ok_or(InvalidRequestIdError {
                    actual_type: "non-i64 number",
                })
            }
            serde_json::Value::String(s) => Ok(Self::String(s.clone())),
            serde_json::Value::Null => Err(InvalidRequestIdError {
                actual_type: "null",
            }),
            serde_json::Value::Bool(_) => Err(InvalidRequestIdError {
                actual_type: "boolean",
            }),
            serde_json::Value::Array(_) => Err(InvalidRequestIdError {
                actual_type: "array",
            }),
            serde_json::Value::Object(_) => Err(InvalidRequestIdError {
                actual_type: "object",
            }),
        }
    }
}

// ── MCP JSON-RPC method names ───────────────────────────────────────
//
// Every JSON-RPC `method` string the server dispatches on has a named
// constant here, so the wire protocol is defined in exactly one place and
// the server match arms never repeat a magic string literal.

/// `initialize` — capability negotiation handshake.
pub const METHOD_INITIALIZE: &str = "initialize";

/// `ping` — liveness check; server replies with an empty result.
pub const METHOD_PING: &str = "ping";

/// `logging/setLevel` — client sets the server log level.
pub const METHOD_LOGGING_SET_LEVEL: &str = "logging/setLevel";

/// `notifications/initialized` — client signals it finished initializing.
pub const METHOD_NOTIFICATIONS_INITIALIZED: &str = "notifications/initialized";

/// `initialized` — bare alias some clients send instead of the namespaced form.
pub const METHOD_INITIALIZED: &str = "initialized";

/// `notifications/cancelled` — client cancels an in-flight request.
pub const METHOD_NOTIFICATIONS_CANCELLED: &str = "notifications/cancelled";

/// `tools/list` — enumerate available tools and their schemas.
pub const METHOD_TOOLS_LIST: &str = "tools/list";

/// `tools/call` — invoke a named tool with arguments.
pub const METHOD_TOOLS_CALL: &str = "tools/call";

/// `resources/list` — enumerate concrete resources.
pub const METHOD_RESOURCES_LIST: &str = "resources/list";

/// `resources/templates/list` — enumerate resource URI templates.
pub const METHOD_RESOURCES_TEMPLATES_LIST: &str = "resources/templates/list";

/// `resources/read` — read a resource by URI.
pub const METHOD_RESOURCES_READ: &str = "resources/read";

/// `prompts/list` — enumerate registered prompts.
pub const METHOD_PROMPTS_LIST: &str = "prompts/list";

/// `prompts/get` — render a prompt with arguments.
pub const METHOD_PROMPTS_GET: &str = "prompts/get";

/// `completion/complete` — argument-completion request.
pub const METHOD_COMPLETION_COMPLETE: &str = "completion/complete";

/// `notifications/progress` — progress update notification.
pub const METHOD_NOTIFICATIONS_PROGRESS: &str = "notifications/progress";

/// `notifications/message` — log-message notification.
pub const METHOD_NOTIFICATIONS_MESSAGE: &str = "notifications/message";

// ── Request ─────────────────────────────────────────────────────────

/// A JSON-RPC 2.0 request.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version — must be `"2.0"`.
    #[serde(rename = "jsonrpc")]
    pub version: String,

    /// Request identifier (number or string). `None` for notifications.
    pub id: Option<serde_json::Value>,

    /// Method name (e.g. `"initialize"`, `"tools/list"`, `"tools/call"`).
    pub method: String,

    /// Optional parameters.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

// ── Response ────────────────────────────────────────────────────────

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JsonRpcResponse {
    /// Protocol version — always `"2.0"`.
    pub jsonrpc: &'static str,

    /// Echoed request identifier.
    pub id: Option<serde_json::Value>,

    /// Present on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,

    /// Present on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code.
    pub code: RpcErrorCode,
    /// Human-readable description.
    pub message: String,
    /// Optional additional data about the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    /// Build a success response from any serializable result type.
    ///
    /// # Panics
    ///
    /// Panics if `result` cannot be serialized to JSON. This should never
    /// happen for the well-formed MCP structs in this module.
    #[must_use]
    pub fn success(id: Option<serde_json::Value>, result: impl Serialize) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some(
                serde_json::to_value(result).expect("MCP result type must be JSON-serializable"),
            ),
            error: None,
        }
    }

    /// Build an error response.
    #[must_use]
    pub fn error(
        id: Option<serde_json::Value>,
        code: impl Into<RpcErrorCode>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(JsonRpcError {
                code: code.into(),
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Build an error response with additional structured data.
    #[must_use]
    pub fn error_with_data(
        id: Option<serde_json::Value>,
        code: impl Into<RpcErrorCode>,
        message: impl Into<String>,
        data: serde_json::Value,
    ) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(JsonRpcError {
                code: code.into(),
                message: message.into(),
                data: Some(data),
            }),
        }
    }
}

/// Parameters for `notifications/cancelled`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelledNotificationParams {
    /// The strongly-typed ID of the request to cancel.
    pub request_id: RequestId,
    /// Optional human-readable reason for cancellation.
    #[serde(default)]
    pub reason: Option<String>,
}

// ── MCP-specific types ──────────────────────────────────────────────

/// Result body for `initialize`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// MCP protocol version (e.g. `"2024-11-05"`).
    pub protocol_version: &'static str,
    /// Server name and version.
    pub server_info: ServerInfo,
    /// Optional instructions describing how to use the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Advertised capabilities.
    pub capabilities: Capabilities,
}

/// Server identification returned in `initialize`.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    /// Human-readable server name.
    pub name: String,
    /// Server version string.
    pub version: String,
}

/// Server capabilities advertised during `initialize`.
#[derive(Debug, Default, Serialize)]
pub struct Capabilities {
    /// Tool support — presence signals that `tools/list` and `tools/call`
    /// are available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCapabilities>,
    /// Resource support — presence signals that `resources/list` is available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceCapabilities>,
    /// Prompt support — presence signals that `prompts/list` is available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptCapabilities>,
}

/// Tool-specific capabilities (currently empty per MCP spec).
#[derive(Debug, Default, Serialize)]
pub struct ToolCapabilities {}

/// Resource-specific capabilities (currently empty per MCP spec).
#[derive(Debug, Default, Serialize)]
pub struct ResourceCapabilities {}

/// Prompt-specific capabilities (currently empty per MCP spec).
#[derive(Debug, Default, Serialize)]
pub struct PromptCapabilities {}

/// Result body for `tools/list`.
#[derive(Clone, Debug, Serialize)]
pub struct ToolsListResult {
    /// Available tools.
    pub tools: Vec<McpToolSchema>,
}

/// A single tool's schema in the `tools/list` response.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolSchema {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// Deserialized `tools/call` request parameters.
#[derive(Debug, Deserialize)]
pub struct ToolCallParams {
    /// Name of the tool to invoke.
    pub name: String,
    /// Tool arguments (defaults to `{}` if absent).
    #[serde(default = "empty_object")]
    pub arguments: serde_json::Value,
}

/// Returns an empty JSON object — used as the serde default for
/// `ToolCallParams::arguments`.
fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

/// Result body for a successful `tools/call`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    /// Response content blocks.
    pub content: Vec<ContentItem>,
    /// `true` when the tool returned an error (MCP-level, not JSON-RPC).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolCallResult {
    /// The text of the first content block, if any.
    ///
    /// Convenience for callers of [`McpServer::dispatch_tool`] who want the
    /// tool's textual output without indexing into [`content`](Self::content).
    ///
    /// [`McpServer::dispatch_tool`]: crate::McpServer::dispatch_tool
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.content.first().map(|item| item.text.as_str())
    }
}

/// The `type` value of a text [`ContentItem`] — currently the only content type
/// this server emits.
pub const CONTENT_TYPE_TEXT: &str = "text";

/// A single content block in a `tools/call` response.
#[derive(Debug, Serialize)]
pub struct ContentItem {
    /// Content type — currently always [`CONTENT_TYPE_TEXT`].
    #[serde(rename = "type")]
    pub content_type: &'static str,
    /// The text content.
    pub text: String,
}

impl ContentItem {
    /// Build a text content block, tagging it with [`CONTENT_TYPE_TEXT`].
    ///
    /// Prefer this over constructing [`ContentItem`] literally so the content
    /// type is set consistently in one place.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content_type: CONTENT_TYPE_TEXT,
            text: text.into(),
        }
    }
}

// ── Prompts ─────────────────────────────────────────────────────────

/// Result body for `prompts/list`.
#[derive(Clone, Debug, Serialize)]
pub struct PromptsListResult {
    /// Available prompts.
    pub prompts: Vec<PromptDefinition>,
}

pub use llm_tool::{PromptArgumentDefinition, PromptDefinition};

/// Parameters for `prompts/get`.
#[derive(Debug, Deserialize)]
pub struct GetPromptParams {
    /// Name of the prompt to retrieve.
    pub name: String,
    /// Arguments to substitute into the template.
    #[serde(default = "empty_object")]
    pub arguments: serde_json::Value,
}

/// Result body for `prompts/get`.
#[derive(Debug, Serialize)]
pub struct GetPromptResult {
    /// Optional description of the rendered prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Rendered messages.
    pub messages: Vec<PromptMessage>,
}

/// A rendered message inside `GetPromptResult`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptMessage {
    /// Role (`"user"` or `"assistant"`).
    pub role: String,
    /// Content block.
    pub content: PromptMessageContent,
}

/// Content inside a `PromptMessage`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum PromptMessageContent {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// Text string.
        text: String,
    },
    /// Embedded resource content.
    #[serde(rename = "resource")]
    Resource {
        /// Resource payload.
        resource: ResourceContent,
    },
}

// ── Resources ───────────────────────────────────────────────────────

/// Wire format for a concrete resource in `resources/list`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    /// Resource URI.
    pub uri: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Optional MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

pub use McpResource as Resource;

/// Result body for `resources/list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourcesListResult {
    /// Available resources.
    pub resources: Vec<McpResource>,
}

pub use llm_tool::ResourceDefinition;

/// Parameters for `resources/read`.
#[derive(Debug, Deserialize)]
pub struct ReadResourceParams {
    /// URI of the resource to read.
    pub uri: String,
}

/// Result body for `resources/read`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ReadResourceResult {
    /// Resource content blocks.
    pub contents: Vec<ResourceContent>,
}

pub use llm_tool::ResourceOutputContent as ResourceContent;

/// An empty JSON object used for responses like ping or logging/setLevel.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmptyResult {}

/// Result body for `resources/templates/list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceTemplatesListResult {
    /// Available resource templates.
    pub resource_templates: Vec<ResourceDefinition>,
}

/// Result body for `completion/complete`.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CompletionCompleteResult {
    /// Completion values and pagination.
    pub completion: CompletionResultData,
}

/// Data inside `CompletionCompleteResult`.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CompletionResultData {
    /// Recommended completion values.
    pub values: Vec<String>,
    /// Total number of available completions.
    pub total: usize,
    /// Whether more completions are available.
    pub has_more: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_request_with_params() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add"}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.version, "2.0");
        assert_eq!(req.id, Some(serde_json::json!(1)));
        assert_eq!(req.method, "tools/call");
        assert!(req.params.is_some());
    }

    #[test]
    fn deserialize_request_without_params() {
        let json = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.params.is_none());
    }

    #[test]
    fn deserialize_notification_without_id() {
        let json = r#"{"jsonrpc":"2.0","method":"initialized"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.id.is_none());
    }

    #[test]
    fn serialize_success_response() {
        let resp =
            JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!({"ok": true}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""jsonrpc":"2.0""#));
        assert!(json.contains(r#""result":{""#));
        assert!(!json.contains("error"));
    }

    #[test]
    fn serialize_error_response() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(1)), PARSE_ERROR, "bad json");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""code":-32700"#));
        assert!(json.contains(r#""message":"bad json""#));
        assert!(!json.contains("result"));
    }

    #[test]
    fn serialize_error_omits_null_id() {
        let resp = JsonRpcResponse::error(None, METHOD_NOT_FOUND, "no such method");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""id":null"#));
    }

    #[test]
    fn response_jsonrpc_field_is_static() {
        let resp = JsonRpcResponse::success(None, serde_json::json!(null));
        // &'static str avoids allocation for every response.
        assert_eq!(resp.jsonrpc, "2.0");
    }

    #[test]
    fn error_without_data_omits_data_field() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(1)), PARSE_ERROR, "bad");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("data"));
    }

    #[test]
    fn error_with_data_includes_data_field() {
        let resp = JsonRpcResponse::error_with_data(
            Some(serde_json::json!(1)),
            INTERNAL_ERROR,
            "boom",
            serde_json::json!({"detail": "stack trace"}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains(r#""data":{"detail":"stack trace"}"#));
    }

    #[test]
    fn jsonrpc_version_constant() {
        assert_eq!(JSONRPC_VERSION, "2.0");
    }

    #[test]
    fn method_consts_match_wire_strings() {
        assert_eq!(METHOD_INITIALIZE, "initialize");
        assert_eq!(METHOD_PING, "ping");
        assert_eq!(METHOD_LOGGING_SET_LEVEL, "logging/setLevel");
        assert_eq!(
            METHOD_NOTIFICATIONS_INITIALIZED,
            "notifications/initialized"
        );
        assert_eq!(METHOD_INITIALIZED, "initialized");
        assert_eq!(METHOD_NOTIFICATIONS_CANCELLED, "notifications/cancelled");
        assert_eq!(METHOD_TOOLS_LIST, "tools/list");
        assert_eq!(METHOD_TOOLS_CALL, "tools/call");
        assert_eq!(METHOD_RESOURCES_LIST, "resources/list");
        assert_eq!(METHOD_RESOURCES_TEMPLATES_LIST, "resources/templates/list");
        assert_eq!(METHOD_RESOURCES_READ, "resources/read");
        assert_eq!(METHOD_PROMPTS_LIST, "prompts/list");
        assert_eq!(METHOD_PROMPTS_GET, "prompts/get");
        assert_eq!(METHOD_COMPLETION_COMPLETE, "completion/complete");
        assert_eq!(METHOD_NOTIFICATIONS_PROGRESS, "notifications/progress");
        assert_eq!(METHOD_NOTIFICATIONS_MESSAGE, "notifications/message");
    }

    #[test]
    fn content_item_text_constructor_sets_type() {
        let item = ContentItem::text("hello");
        assert_eq!(item.content_type, CONTENT_TYPE_TEXT);
        assert_eq!(item.content_type, "text");
        assert_eq!(item.text, "hello");
    }

    #[test]
    fn tool_call_result_text_returns_first_block() {
        let result = ToolCallResult {
            content: vec![ContentItem::text("first"), ContentItem::text("second")],
            is_error: false,
        };
        assert_eq!(result.text(), Some("first"));

        let empty = ToolCallResult {
            content: vec![],
            is_error: true,
        };
        assert_eq!(empty.text(), None);
    }
}
