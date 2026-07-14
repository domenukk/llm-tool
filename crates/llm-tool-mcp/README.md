# llm-tool-mcp

MCP ([Model Context Protocol](https://modelcontextprotocol.io/)) stdio server
for [`llm-tool`](https://crates.io/crates/llm-tool) registries.

Register your tools in a `ToolRegistry`, hand it to `McpServer`, and get a
fully compliant MCP server — no boilerplate.

## Quick start

```rust
use llm_tool::{llm_prompt, llm_resource, llm_tool, ToolContext, ToolError, ToolRegistry};
use llm_tool_mcp::McpServer;

/// Adds two numbers.
#[llm_tool]
fn add(
    /// First operand.
    a: i64,
    /// Second operand.
    b: i64,
) -> Result<String, ToolError> {
    Ok(format!("{}", a + b))
}

/// Code review instruction template.
#[llm_prompt]
fn review_prompt(
    /// Programming language.
    lang: String,
) -> String {
    format!("Please review this {lang} code for security bugs.")
}

/// Dynamic application config resource.
#[llm_resource(uri = "file:///config/{app}.json")]
fn get_config(app: String) -> String {
    format!(r#"{{"app":"{app}","enabled":true}}"#)
}

let registry = ToolRegistry::new().with_tool(Add);

let server = McpServer::builder("my-server", "0.1.0", registry)
    .with_prompt(ReviewPrompt)
    .with_resource(GetConfig)
    .with_context(ToolContext::new().with_conversation_id("caller-id"))
    .build();

// In production: server.run_stdio().expect("server failed");
// Here we feed a request via an in-memory buffer:
let input = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":17,"b":25}}}"#;
let reader = std::io::Cursor::new(format!("{input}\n"));
let mut output = Vec::new();
server.run(reader, &mut output).unwrap();

let resp: serde_json::Value = serde_json::from_slice(&output).unwrap();
assert_eq!(resp["result"]["content"][0]["text"], "42");
```

## Transports: Stdio vs TCP vs Unix Sockets

`McpServer` is builder-style and supports all common execution models out-of-the-box:

```rust
# use llm_tool::ToolRegistry;
# use llm_tool_mcp::McpServer;
# tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
let server = McpServer::new("my-server", "0.1.0", ToolRegistry::new());

// 1. Standard MCP desktop client transport (stdio subprocess):
// server.run_stdio().expect("stdio server failed");

// 2. TCP network server (localhost only):
// server.listen_tcp("127.0.0.1:3000").await.expect("tcp bind failed");

// 3. TCP network server (external / container / docker):
// server.listen_tcp("0.0.0.0:8080").await.expect("tcp bind failed");

// 4. Unix Domain Socket (local IPC):
// server.listen_unix("/tmp/my-agent.sock").await.expect("unix bind failed");
# })
```

## What it handles

| MCP method                  | Behavior                                                       |
| --------------------------- | -------------------------------------------------------------- |
| `initialize`                | Returns server info and capabilities for registered primitives |
| `notifications/initialized` | Acknowledged silently                                          |
| `tools/list`                | Derives schemas from `ToolRegistry::definitions()`             |
| `tools/call`                | Dispatches via `ToolRegistry::dispatch()`, returns content     |
| `prompts/list`              | Lists all registered prompts and their argument schemas        |
| `prompts/get`               | Renders prompt messages with argument substitution             |
| `resources/list`            | Lists all static resources registered on the server            |
| `resources/templates/list`  | Lists all URI templates (e.g. `"file:///config/{app}.json"`)   |
| `resources/read`            | Matches URIs against resources/templates and returns content   |

Tool errors are returned as MCP content with `isError: true` (spec-compliant),
not as JSON-RPC errors.

## Async & custom transports

If running inside an existing Tokio application or network server, use `run_async`:

```rust
# use llm_tool::ToolRegistry;
# use llm_tool_mcp::McpServer;
# tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
let server = McpServer::new("s", "1", ToolRegistry::new());

// Runs over any tokio::io::AsyncBufRead + AsyncWrite streams:
// server.run_async(tokio::io::stdin(), tokio::io::stdout()).await.unwrap();
# })
```

For custom request/response routing (e.g. Axum HTTP POST or `WebSockets`), call
`handle_message`. It accepts a single request **or** a JSON-RPC batch array and
returns a structured [`RpcOutcome`] — a `Single` response object or a `Batch`
array — which you can inspect or render to the wire in a single pass with
`.to_wire()`. `None` means the input was purely a notification, so there is
nothing to send back:

```rust
# use llm_tool::ToolRegistry;
# use llm_tool_mcp::{McpServer, RpcOutcome};
# tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
let server = McpServer::new("s", "1", ToolRegistry::new());

let request = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
match server.handle_message(request).await {
    // `outcome` is a `Single` object or a `Batch` array — render it directly.
    Some(outcome) => {
        let body = outcome.to_wire();
        // ...write `body` to your HTTP/WebSocket response...
        assert!(body.contains("\"result\""));
    }
    // Notification-only input: reply 202/204 with no body.
    None => {}
}
# })
```

## License

Dual-licensed under Apache-2.0 OR MIT.
