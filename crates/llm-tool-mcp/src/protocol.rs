//! JSON-RPC 2.0 protocol types for MCP communication.
//!
//! These types model the wire format used by MCP's JSON-RPC transport.
//! Each request/response is a single JSON line on the stream.

use serde::{Deserialize, Serialize};

// ── Standard JSON-RPC 2.0 error codes ───────────────────────────────

/// Malformed JSON.
pub const PARSE_ERROR: i64 = -32700;

/// Valid JSON but not a valid JSON-RPC request.
pub const INVALID_REQUEST: i64 = -32600;

/// The requested method does not exist.
pub const METHOD_NOT_FOUND: i64 = -32601;

/// Invalid method parameters.
pub const INVALID_PARAMS: i64 = -32602;

/// Internal server error.
pub const INTERNAL_ERROR: i64 = -32603;

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
#[derive(Debug, Serialize)]
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
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code.
    pub code: i64,
    /// Human-readable description.
    pub message: String,
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
            jsonrpc: "2.0",
            id,
            result: Some(
                serde_json::to_value(result).expect("MCP result type must be JSON-serializable"),
            ),
            error: None,
        }
    }

    /// Build an error response.
    #[must_use]
    pub fn error(id: Option<serde_json::Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
        }
    }
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
#[derive(Debug, Serialize)]
pub struct Capabilities {
    /// Tool support — presence signals that `tools/list` and `tools/call`
    /// are available. The inner struct is currently empty (no pagination).
    pub tools: ToolCapabilities,
}

/// Tool-specific capabilities (currently empty per MCP spec).
#[derive(Debug, Default, Serialize)]
pub struct ToolCapabilities {}

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

/// A single content block in a `tools/call` response.
#[derive(Debug, Serialize)]
pub struct ContentItem {
    /// Content type — currently always `"text"`.
    #[serde(rename = "type")]
    pub content_type: &'static str,
    /// The text content.
    pub text: String,
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
}
