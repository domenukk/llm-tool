use super::*;

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
        .handle_request(r#"{"jsonrpc":"2.0","id":9,"method":"foo/bar/unknown"}"#)
        .await;

    let err = resp.error.unwrap();
    assert_eq!(err.code, protocol::METHOD_NOT_FOUND);
    assert!(err.message.contains("foo/bar/unknown"));
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

// ── run_async test ──────────────────────────────────────────────

#[tokio::test]
async fn run_async_processes_requests() {
    let server = test_server();
    let input = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":100,"b":200}}}"#;
    let input_str = format!("{input}\n");
    let reader = input_str.as_bytes();
    let mut output = Vec::new();

    server.run_async(reader, &mut output).await.unwrap();

    let resp: serde_json::Value =
        serde_json::from_str(String::from_utf8(output).unwrap().trim()).unwrap();
    assert_eq!(resp["result"]["content"][0]["text"], "300");
}

// ── TCP listener test ───────────────────────────────────────────

#[tokio::test]
async fn tcp_listener_serves_requests() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let server = test_server();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_tcp_listener(listener).await;
    });

    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":15,"b":25}}}"#;
    stream
        .write_all(format!("{req}\n").as_bytes())
        .await
        .unwrap();
    stream.flush().await.unwrap();

    let mut reader = tokio::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();

    let resp: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(resp["result"]["content"][0]["text"], "40");
}

#[cfg(unix)]
#[tokio::test]
async fn unix_listener_serves_requests() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let server = test_server();
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("test_mcp.sock");
    let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_unix_listener(listener).await;
    });

    let mut stream = tokio::net::UnixStream::connect(&sock_path).await.unwrap();
    let req = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":30,"b":50}}}"#;
    stream
        .write_all(format!("{req}\n").as_bytes())
        .await
        .unwrap();
    stream.flush().await.unwrap();

    let mut reader = tokio::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();

    let resp: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(resp["result"]["content"][0]["text"], "80");
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

// ── Notification filtering tests ────────────────────────────────

#[test]
fn run_drops_notification_responses() {
    let server = test_server();

    // Mix a notification (no id) among regular requests.
    let input = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
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

    // Only 2 responses — the notification produced no output.
    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0]["id"], 1);
    assert_eq!(responses[1]["id"], 2);
}

#[tokio::test]
async fn run_async_drops_notification_responses() {
    let server = test_server();

    let input = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    ];
    let input_str = input.join("\n") + "\n";
    let mut output = Vec::new();

    server
        .run_async(input_str.as_bytes(), &mut output)
        .await
        .unwrap();

    let responses: Vec<serde_json::Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0]["id"], 1);
    assert_eq!(responses[1]["id"], 2);
}

// ── JSON-RPC version validation ─────────────────────────────────

#[tokio::test]
async fn invalid_jsonrpc_version_returns_invalid_request() {
    let server = test_server();
    let resp = server
        .handle_request(r#"{"jsonrpc":"1.0","id":1,"method":"initialize"}"#)
        .await;

    let err = resp.error.unwrap();
    assert_eq!(err.code, protocol::INVALID_REQUEST);
    assert!(err.message.contains("1.0"));
}

#[tokio::test]
async fn missing_jsonrpc_version_returns_parse_error() {
    let server = test_server();
    // Missing "jsonrpc" field entirely — serde will fail to deserialize.
    let resp = server
        .handle_request(r#"{"id":1,"method":"initialize"}"#)
        .await;

    let err = resp.error.unwrap();
    assert_eq!(err.code, protocol::PARSE_ERROR);
}

// ── Batch request handling ───────────────────────────────────────

#[tokio::test]
async fn batch_request_returns_invalid_request_on_handle_request() {
    let server = test_server();
    let resp = server
        .handle_request(r#"[{"jsonrpc":"2.0","id":1,"method":"initialize"}]"#)
        .await;

    let err = resp.error.unwrap();
    assert_eq!(err.code, protocol::INVALID_REQUEST);
    assert!(err.message.contains("batch"));
}

#[tokio::test]
async fn batch_request_via_handle_message_success() {
    let server = test_server();
    let req = r#"[{"jsonrpc":"2.0","id":1,"method":"initialize"},{"jsonrpc":"2.0","id":2,"method":"tools/list"}]"#;
    let resp_val = server
        .handle_message(req)
        .await
        .expect("batch response expected");
    let arr = resp_val.as_array().expect("expected array");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["id"], 1);
    assert_eq!(arr[1]["id"], 2);
    assert_eq!(arr[1]["result"]["tools"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn batch_request_with_notifications_omits_notifications() {
    let server = test_server();
    let req = r#"[{"jsonrpc":"2.0","id":1,"method":"initialize"},{"jsonrpc":"2.0","method":"initialized"},{"jsonrpc":"2.0","id":2,"method":"tools/list"}]"#;
    let resp_val = server
        .handle_message(req)
        .await
        .expect("batch response expected");
    let arr = resp_val.as_array().expect("expected array");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["id"], 1);
    assert_eq!(arr[1]["id"], 2);
}

#[tokio::test]
async fn batch_request_only_notifications_returns_none() {
    let server = test_server();
    let req = r#"[{"jsonrpc":"2.0","method":"initialized"},{"jsonrpc":"2.0","method":"notifications/cancelled"}]"#;
    assert!(server.handle_message(req).await.is_none());
}

#[tokio::test]
async fn empty_array_returns_invalid_request() {
    let server = test_server();
    let resp = server.handle_request("[]").await;

    let err = resp.error.unwrap();
    assert_eq!(err.code, protocol::INVALID_REQUEST);

    let batch_resp = server.handle_message("[]").await.expect("error response");
    assert_eq!(batch_resp["error"]["code"], protocol::INVALID_REQUEST);
}

#[tokio::test]
async fn resources_and_prompts_list_return_empty() {
    let server = test_server();
    let res_resp = server
        .handle_request(r#"{"jsonrpc":"2.0","id":1,"method":"resources/list"}"#)
        .await;
    assert_eq!(
        res_resp.result.unwrap()["resources"]
            .as_array()
            .unwrap()
            .len(),
        0
    );

    let prompt_resp = server
        .handle_request(r#"{"jsonrpc":"2.0","id":2,"method":"prompts/list"}"#)
        .await;
    assert_eq!(
        prompt_resp.result.unwrap()["prompts"]
            .as_array()
            .unwrap()
            .len(),
        0
    );

    let log_resp = server
        .handle_request(
            r#"{"jsonrpc":"2.0","id":3,"method":"logging/setLevel","params":{"level":"info"}}"#,
        )
        .await;
    assert!(log_resp.error.is_none());

    let comp_resp = server
        .handle_request(r#"{"jsonrpc":"2.0","id":4,"method":"completion/complete"}"#)
        .await;
    assert_eq!(
        comp_resp.result.unwrap()["completion"]["values"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
}

// ── Empty / whitespace input ────────────────────────────────────

#[test]
fn run_skips_empty_and_whitespace_lines() {
    let server = test_server();

    let input = "\n  \n\t\n";
    let reader = Cursor::new(input.as_bytes());

    let mut output = Vec::new();
    server.run(reader, &mut output).unwrap();

    // No output for blank lines.
    assert!(output.is_empty());
}

// ── String IDs ──────────────────────────────────────────────────

#[tokio::test]
async fn string_id_is_echoed_back() {
    let server = test_server();
    let resp = server
        .handle_request(r#"{"jsonrpc":"2.0","id":"abc-123","method":"initialize"}"#)
        .await;

    assert!(resp.error.is_none());
    assert_eq!(resp.id, Some(serde_json::json!("abc-123")));
}

#[tokio::test]
async fn null_id_is_treated_as_present() {
    let server = test_server();
    // JSON-RPC spec: null is a valid id value (though unusual).
    let resp = server
        .handle_request(r#"{"jsonrpc":"2.0","id":null,"method":"initialize"}"#)
        .await;

    // null id means this will be serialized with "id":null.
    // The response should still be produced (not dropped as a notification)
    // because the JSON had an explicit "id" key.
    // Note: serde deserializes `"id": null` as `Some(Value::Null)`.
    assert!(resp.error.is_none());
}

// ── Ping ────────────────────────────────────────────────────────

#[tokio::test]
async fn ping_returns_empty_object() {
    let server = test_server();
    let resp = server
        .handle_request(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#)
        .await;

    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result, serde_json::json!({}));
}

// ── notifications/cancelled ─────────────────────────────────────

#[tokio::test]
async fn notifications_cancelled_is_accepted() {
    let server = test_server();
    let resp = server
.handle_request(
r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"req-1","reason":"timeout"}}"#,
)
.await;

    // It's a notification (no id), so handle_request returns id=None.
    assert!(resp.error.is_none());
    assert!(resp.id.is_none());
}

// ── Large tool arguments ────────────────────────────────────────

#[tokio::test]
async fn large_nested_tool_arguments() {
    let server = test_server();

    // Build deeply nested JSON: {"a": 1, "b": 2} but with a large wrapper.
    let args = serde_json::json!({
    "a": 1,
    "b": 2,
    "metadata": {
    "level1": {
    "level2": {
    "level3": {
    "level4": {
    "level5": "deep"
    }
    }
    }
    }
    }
    });

    let req = serde_json::json!({
    "jsonrpc": "2.0",
    "id": 99,
    "method": "tools/call",
    "params": {
    "name": "add",
    "arguments": args
    }
    });

    let resp = server
        .handle_request(&serde_json::to_string(&req).unwrap())
        .await;

    // add tool only looks at a + b, extra fields are ignored by serde.
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["content"][0]["text"], "3");
}

// ── Tool returning empty string ─────────────────────────────────

struct EmptyResultTool;
impl RustTool for EmptyResultTool {
    type Params = EmptyParams;
    const NAME: &'static str = "empty_result";
    const DESCRIPTION: &'static str = "Returns empty string.";
    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::new(String::new()))
    }
}

#[tokio::test]
async fn tool_returning_empty_string() {
    let registry = ToolRegistry::new().with_tool(EmptyResultTool);
    let server = McpServer::new("test", "0.0.1", registry);

    let resp = server
.handle_request(
r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"empty_result","arguments":{}}}"#,
)
.await;

    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["content"][0]["text"], "");
    assert!(result.get("isError").is_none());
}

// ── Concurrent TCP clients ──────────────────────────────────────

#[tokio::test]
async fn tcp_multiple_concurrent_clients() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let server = test_server();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_tcp_listener(listener).await;
    });

    // Spawn 3 concurrent clients.
    let mut handles = Vec::new();
    for i in 0..3 {
        let client_addr = addr;
        handles.push(tokio::spawn(async move {
let mut stream = tokio::net::TcpStream::connect(client_addr).await.unwrap();
let req = format!(
r#"{{"jsonrpc":"2.0","id":{},"method":"tools/call","params":{{"name":"add","arguments":{{"a":{},"b":{}}}}}}}"#,
i, i, 100
);
stream
.write_all(format!("{req}\n").as_bytes())
.await
.unwrap();
stream.flush().await.unwrap();

let mut reader = tokio::io::BufReader::new(stream);
let mut line = String::new();
reader.read_line(&mut line).await.unwrap();

let resp: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
let text = resp["result"]["content"][0]["text"]
.as_str()
.unwrap()
.to_owned();
let expected: i64 = i + 100;
assert_eq!(text, expected.to_string());
}));
    }

    for h in handles {
        h.await.unwrap();
    }
}

#[cfg(unix)]
#[tokio::test]
async fn unix_multiple_concurrent_clients() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let server = test_server();
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("concurrent.sock");
    let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_unix_listener(listener).await;
    });

    let mut handles = Vec::new();
    for i in 0..3 {
        let path = sock_path.clone();
        handles.push(tokio::spawn(async move {
let mut stream = tokio::net::UnixStream::connect(&path).await.unwrap();
let req = format!(
r#"{{"jsonrpc":"2.0","id":{},"method":"tools/call","params":{{"name":"add","arguments":{{"a":{},"b":{}}}}}}}"#,
i, i, 200
);
stream
.write_all(format!("{req}\n").as_bytes())
.await
.unwrap();
stream.flush().await.unwrap();

let mut reader = tokio::io::BufReader::new(stream);
let mut line = String::new();
reader.read_line(&mut line).await.unwrap();

let resp: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
let text = resp["result"]["content"][0]["text"]
.as_str()
.unwrap()
.to_owned();
let expected: i64 = i + 200;
assert_eq!(text, expected.to_string());
}));
    }

    for h in handles {
        h.await.unwrap();
    }
}

// ── Protocol version negotiation ────────────────────────────────

#[tokio::test]
async fn initialize_with_client_protocol_version() {
    let server = test_server();
    let resp = server
.handle_request(
r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
)
.await;

    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
}

#[tokio::test]
async fn initialize_with_unknown_client_version_returns_server_version() {
    let server = test_server();
    let resp = server
.handle_request(
r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"9999-01-01"}}"#,
)
.await;

    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    // Server always returns its own supported version.
    assert_eq!(result["protocolVersion"], "2024-11-05");
}

// ── Notification in run() produces no output ────────────────────

#[test]
fn notification_initialized_no_output_in_run() {
    let server = test_server();
    let input = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let input_str = format!("{input}\n");
    let reader = Cursor::new(input_str.as_bytes());

    let mut output = Vec::new();
    server.run(reader, &mut output).unwrap();

    // Notification should produce zero bytes of output.
    assert!(output.is_empty());
}
