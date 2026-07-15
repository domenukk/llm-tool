//! Tests for the programmatic [`McpServer::dispatch_tool`] entry point and the
//! blocking convenience runners (`run_tcp`, `run_unix`, `serve`).
//!
//! The blocking runners never return on success (they serve forever), so each
//! is launched on a dedicated `std::thread` while the test drives a real
//! `rmcp` client from the async test runtime. The background thread is detached
//! and reaped at process exit — the test controls its own lifecycle.

use std::time::Duration;

use super::*;

// ── dispatch_tool ───────────────────────────────────────────────────

#[tokio::test]
async fn dispatch_tool_success_returns_output() {
    let server = test_server();
    let result = server
        .dispatch_tool("add", serde_json::json!({"a": 2, "b": 40}))
        .await;
    assert!(!result.is_error);
    assert_eq!(result.text(), Some("42"));
}

#[tokio::test]
async fn dispatch_tool_failing_tool_sets_error() {
    let server = test_server();
    let result = server.dispatch_tool("fail", serde_json::json!({})).await;
    assert!(result.is_error);
    assert!(result.text().unwrap().contains("intentional failure"));
}

#[tokio::test]
async fn dispatch_tool_unknown_tool_is_not_found() {
    let server = test_server();
    let result = server
        .dispatch_tool("non_existent", serde_json::json!({}))
        .await;
    assert!(result.is_error);
    let text = result.text().unwrap();
    assert!(text.contains("non_existent"));
    assert!(text.contains("no tool named"));
}

#[tokio::test]
async fn dispatch_tool_matches_wire_tools_call() {
    // The programmatic path and the JSON-RPC wire path must agree.
    let server = test_server();
    let programmatic = server
        .dispatch_tool("add", serde_json::json!({"a": 1, "b": 2}))
        .await;

    let line = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":1,"b":2}}}"#;
    let resp = server.handle_request(line).await;
    let wire = serde_json::to_value(resp.result.unwrap()).unwrap();

    assert_eq!(wire["content"][0]["text"], programmatic.text().unwrap());
    assert_eq!(programmatic.text(), Some("3"));
}

// ── blocking runners ────────────────────────────────────────────────

/// Reserve an ephemeral localhost port by binding then dropping a std
/// listener, so the blocking `run_tcp` runner can rebind it.
fn reserve_local_port() -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr")
}

/// Connect an `rmcp` client to `addr`, retrying while the background server
/// thread is still binding.
async fn connect_client_tcp(
    addr: std::net::SocketAddr,
) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    for _ in 0..100 {
        let Ok(stream) = tokio::net::TcpStream::connect(addr).await else {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        };
        let Ok(client) = rmcp::service::serve_client((), stream).await else {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        };
        return client;
    }
    panic!("could not connect rmcp client to {addr}");
}

/// Connect an `rmcp` client over a Unix socket, retrying while the
/// background server thread is still binding.
#[cfg(unix)]
async fn connect_client_unix(
    path: &std::path::Path,
) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    for _ in 0..100 {
        let Ok(stream) = tokio::net::UnixStream::connect(path).await else {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        };
        let Ok(client) = rmcp::service::serve_client((), stream).await else {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        };
        return client;
    }
    panic!(
        "could not connect rmcp client over Unix socket at {}",
        path.display()
    );
}

#[tokio::test]
async fn run_tcp_blocking_serves_rmcp_client() {
    let addr = reserve_local_port();
    let server = test_server();
    // Blocking runner serves forever; detach onto its own thread.
    let handle = std::thread::spawn(move || {
        // NOLINT: background serve thread — result unused; test controls lifecycle
        let _ = server.run_tcp(addr);
    });

    let mut client = connect_client_tcp(addr).await;
    let tools = client.list_all_tools().await.expect("list tools failed");
    assert_eq!(tools.len(), 3);

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
    // Detach the forever-serving thread; it is reaped at process exit.
    drop(handle);
}

#[tokio::test]
async fn serve_transport_tcp_serves_rmcp_client() {
    let addr = reserve_local_port();
    let server = test_server();
    let transport = Transport::Tcp(addr);
    let handle = std::thread::spawn(move || {
        // NOLINT: background serve thread — result unused; test controls lifecycle
        let _ = server.serve(transport);
    });

    let mut client = connect_client_tcp(addr).await;
    let tools = client.list_all_tools().await.expect("list tools failed");
    assert_eq!(tools.len(), 3);

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
    drop(handle);
}

#[cfg(unix)]
#[tokio::test]
async fn run_unix_blocking_serves_rmcp_client() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("blocking.sock");
    let server = test_server();
    let thread_path = sock_path.clone();
    let handle = std::thread::spawn(move || {
        // NOLINT: background serve thread — result unused; test controls lifecycle
        let _ = server.run_unix(thread_path);
    });

    // Retry-connect while the background thread binds the socket.
    let mut client = connect_client_unix(&sock_path).await;

    let tools = client.list_all_tools().await.expect("list tools failed");
    assert_eq!(tools.len(), 3);

    // NOLINT: test cleanup — close errors are non-fatal
    let _ = client.close().await;
    drop(handle);
}
