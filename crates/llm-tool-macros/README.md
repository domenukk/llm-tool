# llm-tool-macros

Macros for [`llm-tool`](https://crates.io/crates/llm-tool).

Provides `#[llm_tool]` to turn a Rust function into a
[`RustTool`](https://docs.rs/llm-tool/latest/llm_tool/trait.RustTool.html). Use
`llm_tool::llm_tool` rather than depending on this crate directly.

## What the macro generates

Given a function like:

```rust
/// Adds two numbers together.
#[llm_tool::llm_tool]
fn add(
    /// First number.
    a: i64,
    /// Second number.
    b: i64,
) -> Result<String, ToolError> {
    Ok(format!("{}", a + b))
}
```

The macro produces:

1. **`AddParams`** — a struct deriving `Deserialize` and `JsonSchema` with
   fields `a: i64` and `b: i64`. Doc comments on parameters become
   `#[schemars(description = "...")]` attributes, so they appear in the JSON
   Schema sent to the model.

2. **`Add`** — a unit struct implementing `RustTool<Params = AddParams>`.
   - `NAME` = `"add"` (the original function name, snake_case).
   - `DESCRIPTION` = `"Adds two numbers together."` (from the function's doc
     comment).
   - `call()` contains the original function body.

## Rules

| Requirement                        | Detail                                                                                                                                                                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Doc comment on function**        | Required — becomes the tool description.                                                                                                                                                                                           |
| **Doc comment on every parameter** | Required — becomes the JSON Schema field description.                                                                                                                                                                              |
| **Return type**                    | `Result<T, E>` or bare `T` (infallible).                                                                                                                                                                                           |
| **`T` (Ok type)**                  | `String` (auto-wrapped into `ToolOutput`), `ToolOutput` (passed through), any `T: Serialize` (auto-serialized to JSON), or `Json<T>`.                                                                                              |
| **`E` (Err type)**                 | Any `E: std::fmt::Display` (preserving structured metadata for `ToolError`, `std::io::Error`, and `serde_json::Error`).                                                                                                            |
| **`async fn`**                     | Supported — the generated `call()` is always async.                                                                                                                                                                                |
| **`&str` / `Option<&str>` params** | Accepted — the struct stores `String` / `Option<String>`, macro auto-borrows.                                                                                                                                                      |
| **`Option<T>` / `default = expr`** | `Option<T>` gets `#[serde(default)]`; `#[llm_tool(default = expr)]` sets custom default and omits from `required`. `#[serde(...)]` and `#[schemars(...)]` on parameters are forwarded.                                             |
| **`&ToolContext` / `ext` params**  | `&ToolContext` forwards execution context; `#[llm_tool(ext)] state: T` extracts `Clone + Send + Sync + 'static` state via `ctx.get_ext::<T>()`; both are excluded from JSON Schema.                                                |
| **`name = "..."` / `r#ident`**     | Override the wire tool/prompt/resource name (`r#move` automatically unraws to `"move"`).                                                                                                                                           |
| **`description = "..."`**          | Inline description string (or template with `env(...)` / `params(...)` / `context`). Overrides the doc comment.                                                                                                                    |
| **`description_file = "..."`**     | Load description from a `.tmpl.md` template file. Compiled into a static string when static.                                                                                                                                       |
| **`response = "..."`**             | Format the tool's struct return value using an inline `md-tmpl` template while attaching the raw struct to `ToolOutput::metadata`.                                                                                                 |
| **`response_file = "..."`**        | Format the tool's struct return value using a `.tmpl.md` file while attaching the raw struct to `ToolOutput::metadata`.                                                                                                            |
| **`effect = "..."`**               | Side-effect classification: `"read_only"`, `"mutating"` (default), or `"destructive"` (`idempotent = true` is a shorthand for `"read_only"`).                                                                                      |
| **`provenance = "..."`**           | Data provenance tier: `"trusted"` (default), `"workspace_local"`, `"private_data"`, `"external_mcp"`, or `"untrusted_network"` (`external_mcp` and `untrusted_network` auto-default to `<tool_output_quarantine>` output framing). |
| **`sanitize` / `quarantine`**      | Output sanitization mode: `sanitize` (default token neutralization), `sanitize = "tag"`, `sanitize(tag = "...", notice = "...")`, `quarantine`, or `sanitization = "none"`.                                                        |
| **`params(k = "v", ...)`**         | Compile-time key-value pairs for template variables. Requires `description_file` or `description`.                                                                                                                                 |
| **`env(K = "v", ...)`**            | Compile-time values for `env:` frontmatter declarations. Requires `description_file` or `description`. Combinable with `params` or `context`.                                                                                      |
| **`context = fn`**                 | Runtime template context function. Signature: `fn(&Tool) -> Context`. Requires `description_file` or `description`.                                                                                                                |
| **Stateful `impl MyType`**         | Annotate `#[llm_tool] impl MyType` and `#[llm_tool(...)]` methods taking `&self` to generate `{MyType}{Method}Tool(pub MyType)`, `register_tools(&self, &mut ToolRegistry)`, and `tool_registry(&self) -> ToolRegistry`.           |

## License

Dual-licensed under Apache-2.0 OR MIT.
