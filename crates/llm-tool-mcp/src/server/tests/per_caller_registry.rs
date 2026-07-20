//! Tests for **per-caller tool registries** via [`RegistryFactory`].
//!
//! A single [`McpServer`] can present a *different* tool set — and different
//! tool descriptions — to each connection, based on the caller negotiated at
//! `initialize`. This is the "one MCP, many agents" path: the factory builds a
//! registry per caller, which is then cached and reused for that caller's
//! `tools/list` and `tools/call`.

use std::{
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::*;

/// A tool whose *description* is personalised per caller, proving the factory
/// can tailor not just which tools exist but their rendered schemas.
struct GreetTool {
    audience: String,
}
impl RustTool for GreetTool {
    type Params = EmptyParams;
    const NAME: &'static str = "greet";
    const DESCRIPTION: &'static str = "Greets the caller.";
    fn description(&self) -> Cow<'static, str> {
        Cow::Owned(format!("Greets {}.", self.audience))
    }
    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::new(format!("hello {}", self.audience)))
    }
}

/// A privileged tool that only some callers are granted.
struct AdminOnlyTool;
impl RustTool for AdminOnlyTool {
    type Params = EmptyParams;
    const NAME: &'static str = "admin_only";
    const DESCRIPTION: &'static str = "Privileged operation.";
    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::new("admin-ok"))
    }
}

/// Build the registry a given `caller` should see: everyone gets `add` and a
/// personalised `greet`; only `alice` is granted the privileged `admin_only`.
fn build_caller_registry(caller: Option<&str>) -> ToolRegistry {
    let mut registry = ToolRegistry::new().with_tool(AddTool).with_tool(GreetTool {
        audience: caller.unwrap_or("world").to_owned(),
    });
    if caller == Some("alice") {
        registry = registry.with_tool(AdminOnlyTool);
    }
    registry
}

/// A server that builds per-caller registries, counting factory invocations so
/// tests can assert the per-caller memoization.
fn factory_server(counter: Arc<AtomicUsize>) -> McpServer {
    McpServer::builder("factory-test", "0.0.1", ToolRegistry::new())
        .with_per_connection_identity(true)
        .with_registry_factory(move |caller: Option<&str>| {
            counter.fetch_add(1, Ordering::SeqCst);
            build_caller_registry(caller)
        })
        .build()
}

fn init_msg(name: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2024-11-05","clientInfo":{{"name":"{name}","version":"1"}}}}}}"#
    )
}

const INIT_NO_NAME: &str =
    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#;
const LIST: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
const CALL_ADMIN: &str = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_only","arguments":{}}}"#;

/// Send one message and return the single response.
async fn send(server: &McpServer, conn: &mut Connection, msg: &str) -> JsonRpcResponse {
    let outcome = server
        .handle_message_conn(msg, conn)
        .await
        .expect("a response is expected");
    outcome.into_responses().remove(0)
}

/// Perform the `initialize` handshake on `conn`, asserting the server responds.
async fn init_conn(server: &McpServer, conn: &mut Connection, init: &str) {
    assert!(
        server.handle_message_conn(init, conn).await.is_some(),
        "initialize handshake must produce a response",
    );
}

/// Sorted tool names from a `tools/list` response on `conn`.
async fn list_tool_names(server: &McpServer, conn: &mut Connection) -> Vec<String> {
    let resp = send(server, conn, LIST).await;
    let tools = resp.result.as_ref().expect("result present")["tools"]
        .as_array()
        .expect("tools array")
        .clone();
    let mut names: Vec<String> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("tool name").to_owned())
        .collect();
    names.sort();
    names
}

/// The `description` of a named tool in a `tools/list` response on `conn`.
async fn tool_description(server: &McpServer, conn: &mut Connection, tool: &str) -> String {
    let resp = send(server, conn, LIST).await;
    let tools = resp.result.as_ref().expect("result present")["tools"]
        .as_array()
        .expect("tools array")
        .clone();
    tools
        .iter()
        .find(|t| t["name"].as_str() == Some(tool))
        .map_or_else(
            || panic!("tool {tool} not present"),
            |t| {
                t["description"]
                    .as_str()
                    .expect("tool description")
                    .to_owned()
            },
        )
}

#[tokio::test]
async fn factory_serves_per_caller_tool_sets() {
    let counter = Arc::new(AtomicUsize::new(0));
    let server = factory_server(Arc::clone(&counter));

    // Alice is privileged: she sees `admin_only`.
    let mut alice = Connection::new();
    init_conn(&server, &mut alice, &init_msg("alice")).await;
    assert_eq!(
        list_tool_names(&server, &mut alice).await,
        vec![
            "add".to_owned(),
            "admin_only".to_owned(),
            "greet".to_owned()
        ]
    );

    // Bob is not: he sees the same base tools but *not* `admin_only`.
    let mut bob = Connection::new();
    init_conn(&server, &mut bob, &init_msg("bob")).await;
    assert_eq!(
        list_tool_names(&server, &mut bob).await,
        vec!["add".to_owned(), "greet".to_owned()]
    );

    // Alice's view is unaffected by Bob connecting.
    assert!(
        list_tool_names(&server, &mut alice)
            .await
            .contains(&"admin_only".to_owned())
    );
}

#[tokio::test]
async fn factory_personalises_tool_descriptions() {
    let counter = Arc::new(AtomicUsize::new(0));
    let server = factory_server(counter);

    let mut alice = Connection::new();
    init_conn(&server, &mut alice, &init_msg("alice")).await;
    assert_eq!(
        tool_description(&server, &mut alice, "greet").await,
        "Greets alice."
    );

    let mut bob = Connection::new();
    init_conn(&server, &mut bob, &init_msg("bob")).await;
    assert_eq!(
        tool_description(&server, &mut bob, "greet").await,
        "Greets bob."
    );
}

#[tokio::test]
async fn factory_tool_call_uses_caller_registry() {
    let counter = Arc::new(AtomicUsize::new(0));
    let server = factory_server(counter);

    // Alice can call the privileged tool.
    let mut alice = Connection::new();
    init_conn(&server, &mut alice, &init_msg("alice")).await;
    let resp = send(&server, &mut alice, CALL_ADMIN).await;
    let result = resp.result.as_ref().expect("result present");
    // `isError` is omitted from the wire form on success.
    assert!(result.get("isError").is_none());
    assert_eq!(result["content"][0]["text"].as_str(), Some("admin-ok"));

    // Bob's registry has no such tool: the call reports an error result.
    let mut bob = Connection::new();
    init_conn(&server, &mut bob, &init_msg("bob")).await;
    let resp = send(&server, &mut bob, CALL_ADMIN).await;
    let result = resp.result.as_ref().expect("result present");
    assert_eq!(result["isError"].as_bool(), Some(true));
}

#[tokio::test]
async fn factory_memoizes_registry_per_caller() {
    let counter = Arc::new(AtomicUsize::new(0));
    let server = factory_server(Arc::clone(&counter));

    // `build()` invokes the factory once for the default (None) view.
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    // First alice connection builds alice's registry (2 total).
    let mut alice1 = Connection::new();
    init_conn(&server, &mut alice1, &init_msg("alice")).await;
    list_tool_names(&server, &mut alice1).await;
    list_tool_names(&server, &mut alice1).await;
    assert_eq!(counter.load(Ordering::SeqCst), 2);

    // A *second* alice connection reuses the cached registry — no rebuild.
    let mut alice2 = Connection::new();
    init_conn(&server, &mut alice2, &init_msg("alice")).await;
    list_tool_names(&server, &mut alice2).await;
    assert_eq!(counter.load(Ordering::SeqCst), 2);

    // A distinct caller triggers exactly one more build (3 total).
    let mut bob = Connection::new();
    init_conn(&server, &mut bob, &init_msg("bob")).await;
    list_tool_names(&server, &mut bob).await;
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn factory_default_view_without_identity() {
    let counter = Arc::new(AtomicUsize::new(0));
    let server = factory_server(counter);

    // A client that announces no name gets the default (None) view: base tools,
    // greeting the generic "world", and no privileged tool.
    let mut conn = Connection::new();
    init_conn(&server, &mut conn, INIT_NO_NAME).await;
    assert_eq!(
        list_tool_names(&server, &mut conn).await,
        vec!["add".to_owned(), "greet".to_owned()]
    );
    assert_eq!(
        tool_description(&server, &mut conn, "greet").await,
        "Greets world."
    );
}

#[tokio::test]
async fn dispatch_tool_uses_shared_default_registry() {
    // `dispatch_tool` is the programmatic entry point; it uses the server's
    // shared/default registry (the factory's None view), independent of any
    // per-connection negotiation.
    let counter = Arc::new(AtomicUsize::new(0));
    let server = factory_server(counter);

    let result = server
        .dispatch_tool("add", serde_json::json!({"a": 2, "b": 3}))
        .await;
    assert!(!result.is_error);
    assert_eq!(result.text(), Some("5"));

    // The default view does not include the privileged tool.
    let result = server
        .dispatch_tool("admin_only", serde_json::json!({}))
        .await;
    assert!(result.is_error);
}

#[tokio::test]
async fn post_construction_registry_factory_matches_builder() {
    // `McpServer::with_registry_factory` (post-construction) must behave like the
    // builder path: default view has no privileged tool, alice's view does.
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_factory = Arc::clone(&counter);
    let server = McpServer::new("post", "0.0.1", ToolRegistry::new())
        .with_per_connection_identity(true)
        .with_registry_factory(move |caller: Option<&str>| {
            counter_for_factory.fetch_add(1, Ordering::SeqCst);
            build_caller_registry(caller)
        });

    // Default view built eagerly once.
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    let mut alice = Connection::new();
    init_conn(&server, &mut alice, &init_msg("alice")).await;
    assert!(
        list_tool_names(&server, &mut alice)
            .await
            .contains(&"admin_only".to_owned())
    );
}
