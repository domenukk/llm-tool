use super::*;

// ── R7: #[llm_tool] on async fn ─────────────────────────────────────

/// Async tool defined with the `#[llm_tool]` proc macro. The body uses
/// `.await` to prove it runs in an async context.
#[llm_tool]
async fn async_delayed_echo(
    /// The message to echo back.
    message: String,
) -> Result<String, ToolError> {
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    Ok(format!("echo: {message}"))
}

#[tokio::test]
async fn tool_macro_async_fn_dispatches_with_await() {
    let mut d = ToolRegistry::new();
    d.register(AsyncDelayedEcho);

    let result = d
        .dispatch(
            "async_delayed_echo",
            serde_json::json!({"message": "hello async"}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "echo: hello async");
}

/// Async tool that reads a file via `tokio::fs`, proving real I/O works.
#[llm_tool]
async fn async_file_reader(
    /// Path to read.
    path: String,
) -> Result<String, ToolError> {
    tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| ToolError::new(format!("IO error: {e}")))
}

#[tokio::test]
async fn tool_macro_async_fn_reads_file() {
    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    std::fs::write(tmp.path(), "async macro content").expect("write");

    let mut d = ToolRegistry::new();
    d.register(AsyncFileReader);

    let path_str = tmp.path().to_str().expect("path").to_owned();
    let result = d
        .dispatch(
            "async_file_reader",
            serde_json::json!({"path": path_str}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "async macro content");
}

// ── R8: Option<T> auto-default via #[llm_tool] ────────────────────

/// Tool with an optional greeting parameter.
#[llm_tool]
fn greet_optional(
    /// Name to greet.
    name: String,
    /// Custom greeting (defaults to None if omitted).
    greeting: Option<String>,
) -> Result<String, ToolError> {
    let g = greeting.unwrap_or_else(|| "Hello".to_string());
    Ok(format!("{g}, {name}!"))
}

#[test]
fn tool_macro_option_param_not_in_required() {
    let def = definition_of(&GreetOptional).expect("schema");
    let schema = &def.parameter_schema;

    let required = schema["required"]
        .as_array()
        .expect("required should be an array");

    // `name` is required, `greeting` is Option<T> → not required.
    assert!(
        required.iter().any(|v| v == "name"),
        "'name' should be required, got: {required:?}"
    );
    assert!(
        !required.iter().any(|v| v == "greeting"),
        "'greeting' (Option<String>) should NOT be required, got: {required:?}"
    );
}

#[tokio::test]
async fn tool_macro_option_param_missing_from_json() {
    let mut d = ToolRegistry::new();
    d.register(GreetOptional);

    // Dispatch without the optional `greeting` field.
    let result = d
        .dispatch(
            "greet_optional",
            serde_json::json!({"name": "World"}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "Hello, World!");
}

#[tokio::test]
async fn tool_macro_option_param_provided_in_json() {
    let mut d = ToolRegistry::new();
    d.register(GreetOptional);

    // Dispatch with the optional `greeting` field present.
    let result = d
        .dispatch(
            "greet_optional",
            serde_json::json!({"name": "World", "greeting": "Hi"}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "Hi, World!");
}

/// Tool combining async + Option<T> to verify both features work together.
#[llm_tool]
async fn async_optional_tool(
    /// Required input.
    input: String,
    /// Optional suffix.
    suffix: Option<String>,
) -> Result<String, ToolError> {
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    // NOLINT: test helper — empty string default for optional param is intentional
    let s = suffix.unwrap_or_default();
    Ok(format!("{input}{s}"))
}

#[tokio::test]
async fn tool_macro_async_with_optional_param() {
    let mut d = ToolRegistry::new();
    d.register(AsyncOptionalTool);

    // Without optional param.
    let r1 = d
        .dispatch(
            "async_optional_tool",
            serde_json::json!({"input": "base"}),
            &test_ctx(),
        )
        .await;
    assert_eq!(r1.unwrap().content(), "base");

    // With optional param.
    let r2 = d
        .dispatch(
            "async_optional_tool",
            serde_json::json!({"input": "base", "suffix": "_ext"}),
            &test_ctx(),
        )
        .await;
    assert_eq!(r2.unwrap().content(), "base_ext");
}

#[test]
fn tool_macro_async_optional_schema_correctness() {
    let def = definition_of(&AsyncOptionalTool).expect("schema");
    let schema = &def.parameter_schema;

    let required = schema["required"].as_array().expect("required array");
    assert!(required.iter().any(|v| v == "input"), "'input' required");
    assert!(
        !required.iter().any(|v| v == "suffix"),
        "'suffix' (Option) should NOT be required"
    );
}

// ── IntoIterator tests ──────────────────────────────────────────

#[test]
fn into_iter_yields_all_tool_name_definition_pairs() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    d.register(RunCommandTool);

    let mut pairs: Vec<(&str, String)> = (&d)
        .into_iter()
        .map(|(name, def)| (name, def.name))
        .collect();
    pairs.sort();

    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, "run_command");
    assert_eq!(pairs[0].1, "run_command");
    assert_eq!(pairs[1].0, "sample");
    assert_eq!(pairs[1].1, "sample");
}

#[test]
fn into_iter_empty_registry_yields_nothing() {
    let d = ToolRegistry::new();
    let count = (&d).into_iter().count();
    assert_eq!(count, 0);
}

#[test]
fn into_iter_for_loop_syntax() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);

    let mut found = false;
    for (name, def) in &d {
        if name == "sample" {
            assert_eq!(def.description, "A sample tool");
            found = true;
        }
    }
    assert!(found, "Expected to find 'sample' tool via for-in loop");
}

// ── ToolContext tests ───────────────────────────────────────────

#[test]
fn tool_context_conversation_id_none_by_default() {
    let ctx = ToolContext::new();
    assert!(ctx.conversation_id().is_none());
}

#[test]
fn tool_context_conversation_id_returns_value() {
    let ctx = ToolContext::new().with_conversation_id("conv-123");
    assert_eq!(ctx.conversation_id(), Some("conv-123"));
}

#[test]
fn tool_context_get_set_state_roundtrip() {
    let ctx = ToolContext::new();

    // Default for missing key.
    let val = ctx.get_state("missing", serde_json::json!("fallback"));
    assert_eq!(val, serde_json::json!("fallback"));

    // Set and retrieve.
    ctx.set_state("counter", serde_json::json!(42))
        .expect("set_state");
    let val = ctx.get_state("counter", serde_json::json!(0));
    assert_eq!(val, serde_json::json!(42));

    // Overwrite.
    ctx.set_state("counter", serde_json::json!(99))
        .expect("set_state");
    let val = ctx.get_state("counter", serde_json::json!(0));
    assert_eq!(val, serde_json::json!(99));
}

#[test]
fn tool_context_state_persists_across_reads() {
    let ctx = ToolContext::new();
    ctx.set_state("key", serde_json::json!({"nested": true}))
        .expect("set_state");

    // Multiple reads return the same value.
    let v1 = ctx.get_state("key", serde_json::json!(null));
    let v2 = ctx.get_state("key", serde_json::json!(null));
    assert_eq!(v1, v2);
    assert_eq!(v1, serde_json::json!({"nested": true}));
}

#[tokio::test]
async fn dispatch_passes_context_to_tool() {
    /// A tool that reads from the `ToolContext` state.
    struct ContextAwareTool;

    impl RustTool for ContextAwareTool {
        type Params = EmptyParams;
        const NAME: &'static str = "ctx_tool";
        const DESCRIPTION: &'static str = "Reads conversation_id from context.";

        // NOLINT: required for backward-compatible async trait impl in tests
        #[allow(unknown_lints)]
        // NOLINT: forward-compat guard for clippy::unused_async_trait_impl
        #[expect(clippy::unused_async_trait_impl)]
        async fn call(
            &self,
            _params: Self::Params,
            ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let conv = ctx.conversation_id().unwrap_or("none");
            let count = ctx.get_state("call_count", serde_json::json!(0));
            // NOLINT: test assertion — 0 default is intentional for missing count
            let n = count.as_i64().unwrap_or(0);
            ctx.set_state("call_count", serde_json::json!(n + 1))
                .map_err(|e| ToolError::new(format!("set_state failed: {e}")))?;
            Ok(format!("conv={conv}, call={n}").into())
        }
    }

    let mut d = ToolRegistry::new();
    d.register(ContextAwareTool);

    let ctx = ToolContext::new().with_conversation_id("test-conv");

    // First call.
    let r1 = d.dispatch("ctx_tool", serde_json::json!({}), &ctx).await;
    assert_eq!(r1.unwrap().content(), "conv=test-conv, call=0");

    // Second call — state persists.
    let r2 = d.dispatch("ctx_tool", serde_json::json!({}), &ctx).await;
    assert_eq!(r2.unwrap().content(), "conv=test-conv, call=1");
}

// ── ToolOutput metadata tests ───────────────────────────────────

#[derive(serde::Serialize)]
struct ProcessMeta {
    bytes_read: usize,
    source: String,
}

/// A tool that attaches typed metadata to its output.
struct MetadataTool;

impl RustTool for MetadataTool {
    type Params = PathParams;
    const NAME: &'static str = "metadata_tool";
    const DESCRIPTION: &'static str = "Returns output with metadata.";

    // NOLINT: required for backward-compatible async trait impl in tests
    #[allow(unknown_lints)]
    // NOLINT: forward-compat guard for clippy::unused_async_trait_impl
    #[expect(clippy::unused_async_trait_impl)]
    async fn call(
        &self,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        ToolOutput::new(format!("processed: {}", params.path)).with_metadata(&ProcessMeta {
            bytes_read: 1024,
            source: params.path,
        })
    }
}

#[tokio::test]
async fn dispatch_preserves_tool_output_metadata() {
    let mut d = ToolRegistry::new();
    d.register(MetadataTool);

    let result = d
        .dispatch(
            "metadata_tool",
            serde_json::json!({"path": "/etc/hosts"}),
            &test_ctx(),
        )
        .await
        .unwrap();

    assert_eq!(result.content(), "processed: /etc/hosts");
    assert_eq!(result.metadata()["bytes_read"], 1024);
    assert_eq!(result.metadata()["source"], "/etc/hosts");
    assert_eq!(result.metadata().len(), 2);
}

#[tokio::test]
async fn dispatch_tool_output_display_uses_content() {
    let output = ToolOutput::new("hello world").with_meta("ignored", serde_json::json!(true));
    assert_eq!(output.to_string(), "hello world");
}

#[tokio::test]
async fn dispatch_tool_output_into_content_consumes() {
    let output = ToolOutput::new("owned").with_meta("key", serde_json::json!("val"));
    let content: String = output.into_content();
    assert_eq!(content, "owned");
}

#[test]
fn tool_output_from_str_has_empty_metadata() {
    let output: ToolOutput = "plain".into();
    assert_eq!(output.content(), "plain");
    assert!(output.metadata().is_empty());
}

#[test]
fn tool_output_from_string_has_empty_metadata() {
    let output: ToolOutput = "owned".to_string().into();
    assert_eq!(output.content(), "owned");
    assert!(output.metadata().is_empty());
}

// ── ToolError metadata tests ────────────────────────────────────

#[test]
fn tool_error_with_metadata() {
    let err = ToolError::new("HTTP request failed")
        .with_meta("status_code", serde_json::json!(503))
        .with_meta("url", serde_json::json!("https://example.com"));

    assert_eq!(err.message, "HTTP request failed");
    assert_eq!(err.metadata()["status_code"], 503);
    assert_eq!(err.metadata()["url"], "https://example.com");
    assert_eq!(err.metadata().len(), 2);
}

#[test]
fn tool_error_without_metadata_is_empty() {
    let err = ToolError::new("simple error");
    assert!(err.metadata().is_empty());
}

#[test]
fn tool_error_display_ignores_metadata() {
    let err = ToolError::new("visible").with_meta("hidden", serde_json::json!(true));
    assert_eq!(err.to_string(), "visible");
}

#[test]
fn tool_error_equality_includes_metadata() {
    let a = ToolError::new("err").with_meta("k", serde_json::json!(1));
    let b = ToolError::new("err").with_meta("k", serde_json::json!(1));
    let c = ToolError::new("err").with_meta("k", serde_json::json!(2));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// A tool that returns `ToolError` with metadata.
struct MetadataErrorTool;

impl RustTool for MetadataErrorTool {
    type Params = EmptyParams;
    const NAME: &'static str = "metadata_error_tool";
    const DESCRIPTION: &'static str = "Always fails with metadata.";

    // NOLINT: required for backward-compatible async trait impl in tests
    #[allow(unknown_lints)]
    // NOLINT: forward-compat guard for clippy::unused_async_trait_impl
    #[expect(clippy::unused_async_trait_impl)]
    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Err(ToolError::new("service unavailable")
            .with_meta("retry_after_secs", serde_json::json!(30)))
    }
}

#[tokio::test]
async fn dispatch_preserves_tool_error_metadata() {
    let mut d = ToolRegistry::new();
    d.register(MetadataErrorTool);

    let err = d
        .dispatch("metadata_error_tool", serde_json::json!({}), &test_ctx())
        .await
        .unwrap_err();

    assert_eq!(err.message, "service unavailable");
    assert_eq!(err.metadata()["retry_after_secs"], 30);
}

// ── #[llm_tool] macro returning ToolOutput directly ─────────────────

/// A tool that returns ToolOutput with metadata via the macro.
#[llm_tool]
fn tool_with_metadata(
    /// Input value.
    input: String,
) -> Result<ToolOutput, ToolError> {
    Ok(ToolOutput::new(format!("echoed: {input}"))
        .with_meta("input_len", serde_json::json!(input.len())))
}

#[tokio::test]
async fn macro_tool_returning_tool_output_preserves_metadata() {
    let mut d = ToolRegistry::new();
    d.register(ToolWithMetadata);

    let result = d
        .dispatch(
            "tool_with_metadata",
            serde_json::json!({"input": "hello"}),
            &test_ctx(),
        )
        .await
        .unwrap();

    assert_eq!(result.content(), "echoed: hello");
    assert_eq!(result.metadata()["input_len"], 5);
}

// ── with_metadata struct-based tests ─────────────────────────────

#[test]
fn tool_output_with_metadata_struct() {
    #[derive(serde::Serialize)]
    struct Meta {
        status: String,
        count: u32,
    }

    let out = ToolOutput::new("done")
        .with_metadata(&Meta {
            status: "ok".into(),
            count: 42,
        })
        .unwrap();

    assert_eq!(out.metadata()["status"], "ok");
    assert_eq!(out.metadata()["count"], 42);
    assert_eq!(out.metadata().len(), 2);
}

#[test]
fn tool_output_with_metadata_merges_with_existing() {
    #[derive(serde::Serialize)]
    struct Extra {
        source: String,
    }

    let out = ToolOutput::new("data")
        .with_meta("version", serde_json::json!(1))
        .with_metadata(&Extra {
            source: "cache".into(),
        })
        .unwrap();

    assert_eq!(out.metadata()["version"], 1);
    assert_eq!(out.metadata()["source"], "cache");
    assert_eq!(out.metadata().len(), 2);
}

#[test]
fn tool_output_with_metadata_rejects_non_object() {
    let err = ToolOutput::new("x").with_metadata(&42_i32).unwrap_err();

    assert!(
        err.message.contains("JSON object"),
        "Expected object error, got: {err}"
    );
}

#[test]
fn tool_error_with_metadata_struct() {
    #[derive(serde::Serialize)]
    struct ErrorMeta {
        status_code: u16,
        url: String,
    }

    let err = ToolError::new("HTTP request failed")
        .with_metadata(&ErrorMeta {
            status_code: 503,
            url: "https://example.com".into(),
        })
        .unwrap();

    assert_eq!(err.message, "HTTP request failed");
    assert_eq!(err.metadata()["status_code"], 503);
    assert_eq!(err.metadata()["url"], "https://example.com");
    assert_eq!(err.metadata().len(), 2);
}

// ── from_metadata tests ─────────────────────────────────────────

#[test]
fn tool_output_from_metadata_populates_both() {
    #[derive(serde::Serialize)]
    struct Weather {
        location: String,
        temp_f: i32,
    }

    let out = ToolOutput::from_metadata(&Weather {
        location: "Seattle".into(),
        temp_f: 72,
    })
    .unwrap();

    // Content is the JSON string sent to the model.
    assert!(out.content().contains("Seattle"));
    assert!(out.content().contains("72"));

    // Metadata has typed fields for hooks.
    assert_eq!(out.metadata()["location"], "Seattle");
    assert_eq!(out.metadata()["temp_f"], 72);
    assert_eq!(out.metadata().len(), 2);
}

#[test]
fn tool_output_from_metadata_rejects_non_object() {
    let err = ToolOutput::from_metadata(&"just a string").unwrap_err();
    assert!(
        err.message.contains("JSON object"),
        "Expected object error, got: {err}"
    );
}

#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct StructuredResult {
    success: bool,
    code: i32,
}

/// Returns structured struct.
#[llm_tool]
fn return_structured_struct() -> Result<StructuredResult, ToolError> {
    Ok(StructuredResult {
        success: true,
        code: 200,
    })
}

#[tokio::test]
async fn macro_tool_returning_struct_populates_metadata_automatically() {
    let mut d = ToolRegistry::new();
    d.register(ReturnStructuredStruct);

    let result = d
        .dispatch(
            "return_structured_struct",
            serde_json::json!({}),
            &test_ctx(),
        )
        .await
        .unwrap();

    assert!(result.content().contains("success"));
    assert!(result.content().contains("true"));
    assert!(result.content().contains("code"));
    assert!(result.content().contains("200"));

    assert_eq!(result.metadata()["success"], true);
    assert_eq!(result.metadata()["code"], 200);
    assert_eq!(result.metadata().len(), 2);
}

/// Returns a primitive.
#[llm_tool]
fn return_primitive() -> Result<i32, ToolError> {
    Ok(42)
}

#[tokio::test]
async fn macro_tool_returning_primitive_leaves_metadata_empty() {
    let mut d = ToolRegistry::new();
    d.register(ReturnPrimitive);

    let result = d
        .dispatch("return_primitive", serde_json::json!({}), &test_ctx())
        .await
        .unwrap();

    assert_eq!(result.content(), "42");
    assert!(result.metadata().is_empty());
}

/// Returns JSON wrapper.
#[llm_tool]
fn return_json_wrapper() -> Result<crate::Json<StructuredResult>, ToolError> {
    Ok(crate::Json(StructuredResult {
        success: false,
        code: 500,
    }))
}

#[tokio::test]
async fn macro_tool_returning_json_wrapper_populates_metadata() {
    let mut d = ToolRegistry::new();
    d.register(ReturnJsonWrapper);

    let result = d
        .dispatch("return_json_wrapper", serde_json::json!({}), &test_ctx())
        .await
        .unwrap();

    assert!(result.content().contains("success"));
    assert_eq!(result.metadata()["success"], false);
    assert_eq!(result.metadata()["code"], 500);
}
