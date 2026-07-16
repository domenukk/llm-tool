# llm-tool

> **Framework-agnostic Rust tool definitions for LLM agents.**

Write plain Rust functions. Get typed LLM tools with
JSON Schemas, automatic deserialization, and instant
[MCP](https://modelcontextprotocol.io/) server support.

## Why `llm-tool`?

- **Zero Boilerplate:** `#[llm_tool]` on a function → typed tool with JSON Schema.
- **Strongly Typed:** Parameters are validated.
  Missing or extra arguments are caught instantly.
- **Framework Agnostic:** Use the `ToolRegistry` to get
  JSON Schemas for _any_ LLM SDK
  (`OpenAI`, Anthropic, Gemini, …).
- **MCP Ready:** Spin up a fully compliant MCP server in 3 lines with `llm-tool-mcp`.
- **`no_std` Compatible:** Core types work in embedded and WASM targets.

---

## ⚡ Quick Start

```toml
[dependencies]
llm-tool = "0.9"
llm-tool-mcp = "0.9" # Optional: for MCP server support
```

### Define a Tool

Doc comments become tool and parameter descriptions automatically.

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
let ctx = llm_tool::ToolContext::new();
let output = registry
    .dispatch("get_weather", serde_json::json!({"location": "London"}), &ctx)
    .await
    // `dispatch` returns `Err(ToolError::not_found(..))` for unknown tools.
    .expect("dispatch succeeds");
# let _ = output;
# });
```

---

## 🚀 MCP: Tools, Prompts, and Resources

Use `llm-tool-mcp` to expose everything over the **Model Context Protocol**.

```rust
# use llm_tool::{llm_tool, llm_prompt, llm_resource, ToolError, ToolRegistry};
/// A Tool for the LLM to execute.
#[llm_tool]
fn restart_server(
    /// Whether to force-restart even if requests are in-flight.
    force: bool,
) -> String {
    format!("Server restarted (force={force}).")
}

/// A Prompt template for the LLM to use.
#[llm_prompt]
fn code_review(
    /// Programming language of the code to review.
    lang: String,
) -> String {
    format!("Please review this {lang} code for security bugs.")
}

/// A Resource for the LLM to read.
#[llm_resource(uri = "file:///config/{app}.json")]
fn get_config(
    /// Application name whose config to retrieve.
    app: String,
) -> String {
    format!(r#"{{"app":"{app}","enabled":true}}"#)
}

// Register tools in the ToolRegistry.
let registry = ToolRegistry::new().with_tool(RestartServer);
assert_eq!(registry.definitions()[0].name, "restart_server");

// Prompts and Resources are registered via llm-tool-mcp's builder:
//   McpServer::builder("my-server", "1.0", registry)
//       .with_prompt(CodeReview)
//       .with_resource(GetConfig)
//       .build();
```

---

## 🧠 Features

### Return Types & Error Handling

Return `Result<T, E>` or just `T`. The `?` operator works out of the box.

- **Auto-Serialization**: Return any `T: Serialize` → automatic JSON response.
- **Structured Metadata**: Attach hidden metadata to
  `ToolOutput` or `ToolError`
  (logged but _not_ sent to the LLM).

### Context

Add `ctx: &ToolContext` to any tool function to access
shared state, conversation IDs, or typed extensions —
automatically hidden from the JSON Schema.

### Custom Descriptions

Override doc comments with an inline string — no extra features needed:

```rust
# use llm_tool::{llm_tool, ToolError};
#[llm_tool(description = "Query the database and return structured results.")]
async fn query_db(
    /// The SQL query to execute.
    query: String,
) -> Result<String, ToolError> {
    Ok(format!("Results for: {query}"))
}
```

With the `md-tmpl` feature, you can also load
descriptions from `.tmpl.md` template files
(`description_file = "..."`), with compile-time variable
substitution and validation.
See the [`md-tmpl` docs](https://docs.rs/md-tmpl)
for details.

---

## Documentation

- [`llm-tool`](https://docs.rs/llm-tool) — Return types,
  tool context, metadata, template descriptions.
- [`llm-tool-mcp`](https://docs.rs/llm-tool-mcp) — MCP
  transports, stdio/TCP, routing.
- [`md-tmpl`](https://docs.rs/md-tmpl) — Template syntax,
  env variables, response templates.

## License

Dual-licensed under Apache-2.0 OR MIT.
