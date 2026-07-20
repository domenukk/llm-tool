//! Tests for **per-connection caller identity**.
//!
//! A single [`McpServer`] can serve many callers when
//! [`with_per_connection_identity`](McpServer::with_per_connection_identity) is
//! enabled: each connection adopts the `clientInfo.name` from its `initialize`
//! handshake while sharing the base context's state and typed extensions. When
//! the flag is off, every dispatch uses the one shared identity.

use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use super::*;

/// A tool that returns `"<caller>:<session-label>"`, where the label is read
/// from a shared typed extension. Proves both that the caller identity is
/// per-connection *and* that the extension store is shared across connections.
struct EchoExtTool;
impl RustTool for EchoExtTool {
    type Params = EmptyParams;
    const NAME: &'static str = "echo_ext";
    const DESCRIPTION: &'static str = "Returns '<caller>:<shared-label>'.";
    async fn call(
        &self,
        _params: Self::Params,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let caller = ctx.conversation_id().unwrap_or("anonymous").to_owned();
        let label = ctx
            .get_ext::<Arc<String>>()
            .map_or_else(|| "no-label".to_owned(), |l| (*l).clone());
        Ok(ToolOutput::new(format!("{caller}:{label}")))
    }
}

/// Build a server whose base context is `caller = "server"` and carries a
/// shared `Arc<String>` session label extension.
fn identity_server(per_connection: bool) -> McpServer {
    let registry = ToolRegistry::new()
        .with_tool(ContextTool)
        .with_tool(EchoExtTool);
    let ctx = ToolContext::new().with_conversation_id("server");
    ctx.set_ext(Arc::new("sess42".to_owned()))
        .expect("seed shared extension");
    McpServer::new("identity-test", "0.0.1", registry)
        .with_context(ctx)
        .with_per_connection_identity(per_connection)
}

const INIT_ALICE: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"alice","version":"1"}}}"#;
const INIT_BOB: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"bob","version":"1"}}}"#;
const INIT_NO_NAME: &str =
    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#;
const WHOAMI: &str =
    r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"whoami","arguments":{}}}"#;
const ECHO_EXT: &str =
    r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo_ext","arguments":{}}}"#;

/// Perform the `initialize` handshake on `conn`, asserting the server responds.
async fn init_conn(server: &McpServer, conn: &mut Connection, init: &str) {
    assert!(
        server.handle_message_conn(init, conn).await.is_some(),
        "initialize handshake must produce a response",
    );
}

/// Drive one message on a connection slot and return the tool-call text.
async fn call_text(server: &McpServer, conn: &mut Connection, msg: &str) -> String {
    let outcome = server
        .handle_message_conn(msg, conn)
        .await
        .expect("a response is expected");
    let responses = outcome.into_responses();
    responses[0].result.as_ref().expect("result present")["content"][0]["text"]
        .as_str()
        .expect("text content")
        .to_owned()
}

#[tokio::test]
async fn distinct_connections_adopt_their_own_caller() {
    let server = identity_server(true);

    // Connection A → alice.
    let mut conn_a = Connection::new();
    init_conn(&server, &mut conn_a, INIT_ALICE).await;
    assert_eq!(call_text(&server, &mut conn_a, WHOAMI).await, "alice");

    // Connection B → bob, concurrently distinct from A.
    let mut conn_b = Connection::new();
    init_conn(&server, &mut conn_b, INIT_BOB).await;
    assert_eq!(call_text(&server, &mut conn_b, WHOAMI).await, "bob");

    // A is unaffected by B initializing.
    assert_eq!(call_text(&server, &mut conn_a, WHOAMI).await, "alice");
}

#[tokio::test]
async fn per_connection_identity_shares_extensions() {
    let server = identity_server(true);

    let mut conn = Connection::new();
    init_conn(&server, &mut conn, INIT_ALICE).await;

    // Identity is alice, but the shared session label extension is still visible
    // — proving the derived context shares the base context's extensions.
    assert_eq!(
        call_text(&server, &mut conn, ECHO_EXT).await,
        "alice:sess42"
    );
}

#[tokio::test]
async fn missing_client_info_falls_back_to_shared_identity() {
    let server = identity_server(true);

    let mut conn = Connection::new();
    init_conn(&server, &mut conn, INIT_NO_NAME).await;

    // No usable clientInfo.name → the shared server identity is used.
    assert_eq!(call_text(&server, &mut conn, WHOAMI).await, "server");
}

#[tokio::test]
async fn disabled_flag_ignores_client_info() {
    let server = identity_server(false);

    let mut conn = Connection::new();
    init_conn(&server, &mut conn, INIT_ALICE).await;

    // Flag off → clientInfo.name is ignored, shared identity always wins.
    assert_eq!(call_text(&server, &mut conn, WHOAMI).await, "server");
}

/// Connect to `addr`, send `init` then a `whoami` call, and return the caller
/// name the server reported — the identity that connection was granted.
async fn tcp_whoami(addr: std::net::SocketAddr, init: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(format!("{init}\n{WHOAMI}\n").as_bytes())
        .await
        .unwrap();
    stream.flush().await.unwrap();

    let mut reader = tokio::io::BufReader::new(stream);
    // First line: initialize result (ignored). Second line: whoami result.
    let mut init_line = String::new();
    reader.read_line(&mut init_line).await.unwrap();
    let mut who_line = String::new();
    reader.read_line(&mut who_line).await.unwrap();

    let resp: serde_json::Value = serde_json::from_str(who_line.trim()).unwrap();
    resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_owned()
}

/// End-to-end over real TCP sockets: two concurrent clients that announce
/// different `clientInfo.name`s must each see their own identity on the same
/// long-lived server — the exact "one MCP, many agents" scenario.
#[tokio::test]
async fn tcp_connections_keep_independent_identities() {
    let server = identity_server(true);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        // NOLINT: test background task — server result unused; test controls lifecycle
        let _ = server.run_tcp_listener(listener).await;
    });

    let (alice, bob) = tokio::join!(tcp_whoami(addr, INIT_ALICE), tcp_whoami(addr, INIT_BOB));
    assert_eq!(alice, "alice");
    assert_eq!(bob, "bob");
}
