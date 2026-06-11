# llm-tool-macros

Proc-macro crate for [`llm-tool`](https://crates.io/crates/llm-tool).

Provides the `#[llm_tool]` attribute macro that transforms a plain Rust function
into a strongly-typed
[`RustTool`](https://docs.rs/llm-tool/latest/llm_tool/trait.RustTool.html)
implementation. **You don't need to depend on this crate directly** — use
`llm_tool::llm_tool` instead.

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

| Requirement                        | Detail                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Doc comment on function**        | Required — becomes the tool description.                                                                                              |
| **Doc comment on every parameter** | Required — becomes the JSON Schema field description.                                                                                 |
| **Return type**                    | `Result<T, E>` or bare `T` (infallible).                                                                                              |
| **`T` (Ok type)**                  | `String` (auto-wrapped into `ToolOutput`), `ToolOutput` (passed through), any `T: Serialize` (auto-serialized to JSON), or `Json<T>`. |
| **`E` (Err type)**                 | Any `E: Into<ToolError>` — built-in for `ToolError`, `String`, `std::io::Error`, `serde_json::Error`, `Box<dyn Error + Send + Sync>`. |
| **`async fn`**                     | Supported — the generated `call()` is always async.                                                                                   |
| **`&str` params**                  | Accepted — the struct stores `String`, macro auto-borrows.                                                                            |
| **`Option<T>` params**             | Auto-annotated with `#[serde(default)]` → not in `required`.                                                                          |
| **`&ToolContext` param**           | Recognized as execution context, forwarded from registry, excluded from params struct.                                                |
| **`self` receiver**                | Not allowed — must be a free function.                                                                                                |

## License

Dual-licensed under Apache-2.0 OR MIT.
