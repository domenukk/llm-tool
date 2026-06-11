# llm-tool-mcp

MCP ([Model Context Protocol](https://modelcontextprotocol.io/)) stdio server
for [`llm-tool`](https://crates.io/crates/llm-tool) registries.

Register your tools in a `ToolRegistry`, hand it to `McpServer`, and get a
fully compliant MCP server — no boilerplate.

## Quick start

```rust
use llm_tool::{llm_tool, ToolError, ToolContext, ToolRegistry};
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

let registry = ToolRegistry::new().with_tool(Add);

let server = McpServer::new("my-server", "0.1.0", registry)
    .with_context(ToolContext::new(Some("caller-id".into())));

// In production: server.run_stdio().expect("server failed");
// Here we feed a request via an in-memory buffer:
let input = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":17,"b":25}}}"#;
let reader = std::io::Cursor::new(format!("{input}\n"));
let mut output = Vec::new();
server.run(reader, &mut output).unwrap();

let resp: serde_json::Value = serde_json::from_slice(&output).unwrap();
assert_eq!(resp["result"]["content"][0]["text"], "42");
```

## What it handles

| MCP method                  | Behavior                                                   |
| --------------------------- | ---------------------------------------------------------- |
| `initialize`                | Returns server info and `{"tools": {}}` capabilities       |
| `notifications/initialized` | Acknowledged silently                                      |
| `tools/list`                | Derives schemas from `ToolRegistry::definitions()`         |
| `tools/call`                | Dispatches via `ToolRegistry::dispatch()`, returns content |

Tool errors are returned as MCP content with `isError: true` (spec-compliant),
not as JSON-RPC errors.

## Custom transports

For non-stdio transports, use `handle_request` directly:

```rust
# use llm_tool::ToolRegistry;
# use llm_tool_mcp::McpServer;
# tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
let server = McpServer::new("s", "1", ToolRegistry::new());

let response = server
    .handle_request(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
    .await;
# })
```

## License

Dual-licensed under Apache-2.0 OR MIT.
