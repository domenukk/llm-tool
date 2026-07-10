use super::*;

// ── handle_request tests ────────────────────────────────────────

#[tokio::test]
async fn rmcp_client_integration_test() {
    let server = test_server();
    let (client_io, server_io) = tokio::io::duplex(8192);
    let (server_r, server_w) = tokio::io::split(server_io);

    tokio::spawn(async move {
        let mut reader = tokio::io::BufReader::new(server_r);
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_async(&mut reader, server_w).await;
    });

    let mut client = rmcp::service::serve_client((), client_io)
        .await
        .expect("rmcp client handshake failed");

    // 1. Verify list_all_tools
    let tools = client.list_all_tools().await.expect("list tools failed");
    assert_eq!(tools.len(), 3);

    // 2. Verify calling a successful tool ("add")
    let call_res = client
        .call_tool(
            rmcp::model::CallToolRequestParams::new("add").with_arguments(
                serde_json::json!({"a": 15, "b": 25})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        )
        .await
        .expect("call tool failed");

    // NOLINT: test assertion — false default is correct for missing is_error flag
    assert!(!call_res.is_error.unwrap_or(false));
    assert_eq!(call_res.content.len(), 1);
    let text = match &call_res.content[0] {
        rmcp::model::ContentBlock::Text(t) => &t.text,
        other => panic!("expected text content, got {other:?}"),
    };
    assert_eq!(text, "40");

    // 3. Verify calling a failing tool ("fail") returns is_error=true per MCP spec
    let fail_res = client
        .call_tool(rmcp::model::CallToolRequestParams::new("fail"))
        .await
        .expect("call tool response expected");
    // NOLINT: test assertion — false default is correct for missing is_error flag
    assert!(fail_res.is_error.unwrap_or(false));
    let fail_text = match &fail_res.content[0] {
        rmcp::model::ContentBlock::Text(t) => &t.text,
        other => panic!("expected text content, got {other:?}"),
    };
    assert!(fail_text.contains("intentional failure"));

    // 4. Verify calling an unknown tool returns is_error=true
    let unknown_res = client
        .call_tool(rmcp::model::CallToolRequestParams::new("non_existent"))
        .await
        .expect("call tool response expected");
    // NOLINT: test assertion — false default is correct for missing is_error flag
    assert!(unknown_res.is_error.unwrap_or(false));
    let unknown_text = match &unknown_res.content[0] {
        rmcp::model::ContentBlock::Text(t) => &t.text,
        other => panic!("expected text content, got {other:?}"),
    };
    assert!(unknown_text.contains("Unknown tool: non_existent"));

    // 5. Verify list_all_prompts returns empty list
    let prompts = client
        .list_all_prompts()
        .await
        .expect("list prompts failed");
    assert!(prompts.is_empty());

    // 6. Verify list_all_resources returns empty list
    let resources = client
        .list_all_resources()
        .await
        .expect("list resources failed");
    assert!(resources.is_empty());

    // 7. Verify reading a non-existent resource returns -32602 INVALID_PARAMS (resource not found)
    let read_err = client
        .read_resource(rmcp::model::ReadResourceRequestParams::new(
            "file:///nonexistent",
        ))
        .await
        .unwrap_err();
    match read_err {
        rmcp::ServiceError::McpError(err) => {
            assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }
        other => panic!("expected McpError with INVALID_PARAMS, got {other:?}"),
    }

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
}

#[tokio::test]
async fn rmcp_client_tcp_test() {
    let server = test_server();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_tcp_listener(listener).await;
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut client = rmcp::service::serve_client((), stream)
        .await
        .expect("rmcp client handshake failed over TCP");

    let tools = client
        .list_all_tools()
        .await
        .expect("list tools failed over TCP");
    assert_eq!(tools.len(), 3);

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn rmcp_client_unix_test() {
    let server = test_server();
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("rmcp.sock");
    let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_unix_listener(listener).await;
    });

    let stream = tokio::net::UnixStream::connect(&sock_path).await.unwrap();
    let mut client = rmcp::service::serve_client((), stream)
        .await
        .expect("rmcp client handshake failed over Unix socket");

    let tools = client
        .list_all_tools()
        .await
        .expect("list tools failed over Unix socket");
    assert_eq!(tools.len(), 3);

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
}
