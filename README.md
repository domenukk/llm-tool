# llm-tool

> **Framework-agnostic Rust tool definitions for LLM agents.**

Write standard Rust functions. Get perfectly typed LLM tools, Prompts, and Resources, complete with JSON Schemas and instant [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server support.

`llm-tool` eliminates the boilerplate of writing JSON schemas and deserialization logic for your AI agents. You just write documented Rust functions, and we handle the rest.

## Why `llm-tool`?

*   **Zero Boilerplate:** Use `#[llm_tool]`, `#[llm_prompt]`, and `#[llm_resource]` on plain functions.
*   **Strongly Typed:** Parameters are automatically typed and validated. Missing or extra arguments are caught instantly.
*   **Framework Agnostic:** Use the raw `ToolRegistry` to get JSON Schemas for *any* LLM SDK (OpenAI, Anthropic, Gemini, etc).
*   **Batteries Included:** Spin up a fully compliant MCP Server in 3 lines of code using `llm-tool-mcp`.
*   **Markdown Prompts:** (Optional) Keep your codebase clean by defining tool descriptions in `.tmpl.md` files with the `md-tmpl` feature.

---

## ⚡ Quick Start

Add the crates to your `Cargo.toml`:

```toml
[dependencies]
llm-tool = "0.5"
llm-tool-mcp = "0.5" # Optional: for MCP server support
```

### 1. Define a Tool

Just write a function. The doc comments automatically become the tool and parameter descriptions!

```rust
use llm_tool::{llm_tool, ToolError, ToolRegistry};

/// Fetches the current weather for a given location.
#[llm_tool]
async fn get_weather(
    /// The city to look up (e.g., "San Francisco, CA").
    location: String,
    /// Whether to use Celsius or Fahrenheit.
    celsius: Option<bool>,
) -> Result<String, ToolError> {
    let temp = if celsius.unwrap_or(true) { "22°C" } else { "72°F" };
    Ok(format!("The weather in {location} is sunny and {temp}."))
}

// Register it! The macro generated a `GetWeather` struct for us.
let registry = ToolRegistry::new().with_tool(GetWeather);

// You can now extract the JSON schema for any LLM SDK...
let definitions = registry.definitions();
assert_eq!(definitions[0].name, "get_weather");

// ...or execute calls directly from JSON arguments!
# futures::executor::block_on(async {
let ctx = llm_tool::ToolContext::new(None);
let result = registry.dispatch(
    "get_weather",
    serde_json::json!({"location": "London"}),
    &ctx
).await.unwrap();
# });
```

---

## 🚀 The 3 Pillars of MCP

If you want to expose your tools, prompts, or resources over the standard **Model Context Protocol**, `llm-tool` has you covered. Use the `llm-tool-mcp` companion crate to spin up a server in seconds.

### Tools, Prompts, and Resources

```rust, ignore
use llm_tool::{llm_tool, llm_prompt, llm_resource, ToolRegistry};
use llm_tool_mcp::McpServer;

/// 1. A Tool for the LLM to execute
#[llm_tool]
fn restart_server(force: bool) -> String {
    "Server restarted.".into()
}

/// 2. A Prompt template for the LLM to use
#[llm_prompt]
fn code_review(lang: String) -> String {
    format!("Please review this {lang} code for security bugs.")
}

/// 3. A Resource for the LLM to read
#[llm_resource(uri = "file:///config/{app}.json")]
fn get_config(app: String) -> String {
    format!(r#"{{"app":"{app}","enabled":true}}"#)
}

// Spin up the MCP Server!
let registry = ToolRegistry::new().with_tool(RestartServer);
let server = McpServer::new("my-mcp-server", "1.0.0", registry)
    .with_prompt(CodeReview)
    .with_resource(GetConfig);

// Start serving over stdio or TCP (see llm-tool-mcp docs)
```

---

## 🧠 Advanced Features made Simple

### Return Types & Error Handling
Tools can return `Result<T, E>` or just `T` (for infallible tools).
*   **Zero-boilerplate Errors**: The `?` operator works flawlessly. `ToolError` implements `From<std::io::Error>`, `serde_json::Error`, etc.
*   **Auto-Serialization**: Return any `T: Serialize` and it will automatically be converted to a JSON response.
*   **Structured Metadata**: Return `ToolOutput` or `ToolError` to attach hidden metadata (like execution times or hidden error traces) that is logged but *not* sent to the LLM.

### Context
Need access to shared state or the current conversation ID? Just add `ctx: &ToolContext` to your function parameters. The macro automatically wires it up without exposing it in the JSON Schema!

### Markdown Template Descriptions (`md-tmpl` feature)
Don't want massive doc comments cluttering your Rust code? Use the `md-tmpl` feature to store descriptions in Markdown files.

```rust, ignore
#[llm_tool(prompt_file = "prompts/tools/database_query.tmpl.md")]
async fn query_db(query: String) -> Result<String, ToolError> { /* ... */ }
```
*Templates are parsed and validated at compile time!*

---

## Documentation

For a deep dive into all available features, check out the documentation:

*   [`llm-tool` Documentation](https://docs.rs/llm-tool) - Advanced return types, tool context, metadata, and `md-tmpl` integration.
*   [`llm-tool-mcp` Documentation](crates/llm-tool-mcp/README.md) - Async transports, stdio vs tcp, custom routing, and the MCP protocol.

## License

Dual-licensed under Apache-2.0 OR MIT.
