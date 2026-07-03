use super::*;

#[test]
fn tool_definition_serde_roundtrip() {
    let def = definition_of(&SampleTool).expect("schema");
    let json = serde_json::to_string(&def).expect("serialize");
    let parsed: ToolDefinition = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.name, def.name);
    assert_eq!(parsed.description, def.description);
    assert_eq!(parsed.parameter_schema, def.parameter_schema);
}

struct EmptyParamTool;
impl RustTool for EmptyParamTool {
    type Params = EmptyParams;
    const NAME: &'static str = "empty";
    const DESCRIPTION: &'static str = "No params";
    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        Ok("ok".into())
    }
}

#[test]
fn tool_definition_with_empty_schema() {
    let tool = definition_of(&EmptyParamTool).expect("schema");
    let json = serde_json::to_string(&tool).expect("serialize");
    let parsed: ToolDefinition = serde_json::from_str(&json).expect("deserialize");
    // Compare via JSON to handle serde normalization (None vs empty struct).
    let orig_json = serde_json::to_value(&tool.parameter_schema).unwrap();
    let parsed_json = serde_json::to_value(&parsed.parameter_schema).unwrap();
    assert_eq!(orig_json, parsed_json);
}

#[test]
fn tool_definition_with_complex_schema() {
    let tool = definition_of(&RunCommandTool).expect("schema");
    let schema_json = serde_json::to_value(&tool.parameter_schema).expect("schema to json");
    // The schema should have 'command' as a required field.
    let required = schema_json["required"]
        .as_array()
        .expect("required should be an array");
    assert!(
        required.iter().any(|v| v == "command"),
        "'command' should be required, got: {required:?}"
    );
}

// ── ToolRegistry tests ────────────────────────────────────────

#[tokio::test]
async fn registry_dispatch_valid_tool() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    let result = d
        .dispatch(
            "sample",
            serde_json::json!({"path": "/tmp/foo"}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "/tmp/foo");
}

#[tokio::test]
async fn registry_dispatch_unknown_tool() {
    let d = ToolRegistry::new();
    let result = d
        .dispatch("nonexistent", serde_json::json!({}), &test_ctx())
        .await;
    assert_eq!(
        result.unwrap_err(),
        ToolError::new("Unknown tool: nonexistent")
    );
}

#[tokio::test]
async fn registry_dispatch_invalid_args() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    // SampleTool expects {"path": String}, not an integer.
    let result = d
        .dispatch("sample", serde_json::json!({"path": 42}), &test_ctx())
        .await;
    let err = result.unwrap_err();
    assert!(
        err.message.contains("deserialize"),
        "Error should mention deserialization, got: {err}"
    );
}

#[tokio::test]
async fn registry_dispatch_missing_required_field() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    // Missing the required "path" field entirely.
    let err = d
        .dispatch("sample", serde_json::json!({}), &test_ctx())
        .await
        .expect_err("Expected error for missing required field");
    assert!(
        err.message.contains("missing field"),
        "Error should mention missing field, got: {err}"
    );
}

#[test]
fn registry_definitions_returns_all() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    d.register(RunCommandTool);

    let defs = d.definitions();
    assert_eq!(defs.len(), 2);

    let mut names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["run_command", "sample"]);
}

#[test]
fn registry_register_chaining() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool).register(RunCommandTool);
    assert_eq!(d.len(), 2);
    assert!(!d.is_empty());
}

#[test]
fn registry_with_tool_owned_chaining() {
    let d = ToolRegistry::new()
        .with_tool(SampleTool)
        .with_tool(RunCommandTool);
    assert_eq!(d.len(), 2);
    assert!(!d.is_empty());

    let defs = d.definitions();
    let mut names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["run_command", "sample"]);
}

#[test]
fn registry_default_is_empty() {
    let d = ToolRegistry::default();
    assert!(d.is_empty());
    assert_eq!(d.len(), 0);
}

#[tokio::test]
async fn registry_replaces_on_duplicate_name() {
    struct AlternateSample;
    impl RustTool for AlternateSample {
        type Params = PathParams;
        const NAME: &'static str = "sample";
        const DESCRIPTION: &'static str = "Alternate sample";
        async fn call(
            &self,
            params: Self::Params,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(format!("alt: {}", params.path).into())
        }
    }

    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    d.register(AlternateSample);
    assert_eq!(d.len(), 1);

    let result = d
        .dispatch("sample", serde_json::json!({"path": "x"}), &test_ctx())
        .await;
    assert_eq!(result.unwrap().content(), "alt: x");
}

#[tokio::test]
async fn registry_tool_returning_error() {
    struct FailingTool;
    impl RustTool for FailingTool {
        type Params = EmptyParams;
        const NAME: &'static str = "fail";
        const DESCRIPTION: &'static str = "Always fails";
        async fn call(
            &self,
            _params: Self::Params,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Err(ToolError::new("intentional failure"))
        }
    }

    let mut d = ToolRegistry::new();
    d.register(FailingTool);
    let result = d.dispatch("fail", serde_json::json!({}), &test_ctx()).await;
    assert_eq!(result.unwrap_err(), ToolError::new("intentional failure"));
}

#[test]
fn registry_debug_shows_tool_names() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    let dbg = format!("{d:?}");
    assert!(dbg.contains("ToolRegistry"));
    assert!(dbg.contains("sample"));
    assert!(dbg.contains("tool_count: 1"));
}

// ── Async-specific tests ────────────────────────────────────────

/// A tool that actually awaits a tokio sleep, proving async dispatch works.
struct AsyncSleepTool;

impl RustTool for AsyncSleepTool {
    type Params = EmptyParams;
    const NAME: &'static str = "async_sleep";
    const DESCRIPTION: &'static str = "Sleeps briefly then returns.";

    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        Ok("slept".into())
    }
}

#[tokio::test]
async fn async_tool_with_tokio_sleep() {
    let mut d = ToolRegistry::new();
    d.register(AsyncSleepTool);
    let result = d
        .dispatch("async_sleep", serde_json::json!({}), &test_ctx())
        .await;
    assert_eq!(result.unwrap().content(), "slept");
}

/// A tool that reads a file using `tokio::fs`.
struct AsyncReadFileTool;

#[derive(Deserialize, schemars::JsonSchema)]
struct ReadFileParams {
    /// Path to the file to read.
    path: String,
}

impl RustTool for AsyncReadFileTool {
    type Params = ReadFileParams;
    const NAME: &'static str = "read_file";
    const DESCRIPTION: &'static str = "Reads a file asynchronously.";

    async fn call(
        &self,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        tokio::fs::read_to_string(&params.path)
            .await
            .map(ToolOutput::from)
            .map_err(|e| ToolError::new(format!("IO error: {e}")))
    }
}

#[tokio::test]
async fn async_tool_with_tokio_fs() {
    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    std::fs::write(tmp.path(), "hello async").expect("write tempfile");

    let mut d = ToolRegistry::new();
    d.register(AsyncReadFileTool);

    let path_str = tmp.path().to_str().expect("path to str").to_owned();
    let result = d
        .dispatch(
            "read_file",
            serde_json::json!({"path": path_str}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "hello async");
}

#[tokio::test]
async fn async_tool_tokio_fs_missing_file() {
    let mut d = ToolRegistry::new();
    d.register(AsyncReadFileTool);
    let result = d
        .dispatch(
            "read_file",
            serde_json::json!({"path": "/nonexistent/file.txt"}),
            &test_ctx(),
        )
        .await;
    let err = result.unwrap_err();
    assert!(
        err.message.contains("IO error"),
        "Expected IO error, got: {err}"
    );
}

/// A tool that uses a tokio channel to receive its result, proving
/// the full async machinery works end-to-end.
struct ChannelTool {
    tx: tokio::sync::mpsc::Sender<String>,
    rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<String>>>,
}

impl ChannelTool {
    fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        Self {
            tx,
            rx: std::sync::Mutex::new(Some(rx)),
        }
    }
}

impl RustTool for ChannelTool {
    type Params = EmptyParams;
    const NAME: &'static str = "channel_tool";
    const DESCRIPTION: &'static str = "Awaits a value from a channel.";

    async fn call(
        &self,
        _params: Self::Params,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let mut rx = self
            .rx
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| ToolError::new("channel already consumed"))?;
        rx.recv()
            .await
            .map(ToolOutput::from)
            .ok_or_else(|| ToolError::new("channel closed"))
    }
}

#[tokio::test]
async fn async_tool_awaits_channel() {
    let tool = ChannelTool::new();
    let tx = tool.tx.clone();

    let mut d = ToolRegistry::new();
    d.register(tool);

    // Send the value from another task.
    let ctx = test_ctx();
    let dispatch_future = d.dispatch("channel_tool", serde_json::json!({}), &ctx);
    let send_future = async move {
        tx.send("from_channel".to_string()).await.unwrap();
    };

    let (result, ()) = tokio::join!(dispatch_future, send_future);
    assert_eq!(result.unwrap().content(), "from_channel");
}

// ── Concurrent dispatch tests ───────────────────────────────────

#[tokio::test]
async fn concurrent_dispatches_to_different_tools() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);
    d.register(AsyncSleepTool);
    d.register(RunCommandTool);

    let ctx = test_ctx();
    let (r1, r2, r3) = tokio::join!(
        d.dispatch("sample", serde_json::json!({"path": "a"}), &ctx),
        d.dispatch("async_sleep", serde_json::json!({}), &ctx),
        d.dispatch("run_command", serde_json::json!({"command": "ls"}), &ctx),
    );

    assert_eq!(r1.unwrap().content(), "a");
    assert_eq!(r2.unwrap().content(), "slept");
    assert_eq!(r3.unwrap().content(), "Ran: ls");
}

#[tokio::test]
async fn concurrent_dispatches_to_same_tool() {
    let mut d = ToolRegistry::new();
    d.register(SampleTool);

    let ctx = test_ctx();
    let futs: Vec<_> = (0..10)
        .map(|i| d.dispatch("sample", serde_json::json!({"path": format!("p{i}")}), &ctx))
        .collect();

    let results = futures::future::join_all(futs).await;
    for (i, r) in results.into_iter().enumerate() {
        assert_eq!(r.unwrap().content(), format!("p{i}"));
    }
}

// ── Schema / doc comment tests ──────────────────────────────────

#[derive(Deserialize, schemars::JsonSchema)]
struct DocumentedParams {
    /// The target hostname to connect to.
    hostname: String,
    /// Port number (1-65535).
    port: u16,
    /// Optional timeout in seconds.
    #[serde(default)]
    timeout: Option<f64>,
}

struct DocumentedTool;
impl RustTool for DocumentedTool {
    type Params = DocumentedParams;
    const NAME: &'static str = "connect";
    const DESCRIPTION: &'static str = "Connects to a remote host.";
    async fn call(&self, p: Self::Params, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(format!("{}:{}:{:?}", p.hostname, p.port, p.timeout).into())
    }
}

#[test]
fn schema_contains_field_descriptions() {
    let def = definition_of(&DocumentedTool).expect("schema");
    let schema = &def.parameter_schema;

    // Check the properties contain our fields.
    let props = schema["properties"].as_object().expect("properties object");
    assert!(props.contains_key("hostname"), "missing hostname");
    assert!(props.contains_key("port"), "missing port");
    assert!(props.contains_key("timeout"), "missing timeout");

    // Check the descriptions from doc comments made it through.
    let hostname_desc = props["hostname"]["description"]
        .as_str()
        .expect("hostname description");
    assert!(
        hostname_desc.contains("hostname"),
        "hostname description should mention 'hostname', got: {hostname_desc}"
    );

    let port_desc = props["port"]["description"]
        .as_str()
        .expect("port description");
    assert!(
        port_desc.contains("1-65535"),
        "port description should mention range, got: {port_desc}"
    );
}

#[test]
fn schema_required_vs_optional_fields() {
    let def = definition_of(&DocumentedTool).expect("schema");
    let schema = &def.parameter_schema;

    let required = schema["required"]
        .as_array()
        .expect("required should be an array");

    // hostname and port are required, timeout is Option → not required.
    assert!(
        required.iter().any(|v| v == "hostname"),
        "hostname required"
    );
    assert!(required.iter().any(|v| v == "port"), "port required");
    assert!(
        !required.iter().any(|v| v == "timeout"),
        "timeout should NOT be required"
    );
}

#[tokio::test]
async fn dispatch_with_optional_field_missing() {
    let mut d = ToolRegistry::new();
    d.register(DocumentedTool);

    // Dispatch without `timeout` (it has serde(default)).
    let result = d
        .dispatch(
            "connect",
            serde_json::json!({"hostname": "example.com", "port": 443}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "example.com:443:None");
}

#[tokio::test]
async fn dispatch_with_optional_field_present() {
    let mut d = ToolRegistry::new();
    d.register(DocumentedTool);

    let result = d
        .dispatch(
            "connect",
            serde_json::json!({"hostname": "localhost", "port": 8080, "timeout": 30.0}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "localhost:8080:Some(30.0)");
}

#[tokio::test]
async fn dispatch_with_extra_fields_ignored() {
    // serde's default behavior ignores unknown fields.
    let mut d = ToolRegistry::new();
    d.register(SampleTool);

    let result = d
        .dispatch(
            "sample",
            serde_json::json!({"path": "/tmp/x", "unknown_field": 42}),
            &test_ctx(),
        )
        .await;
    assert_eq!(result.unwrap().content(), "/tmp/x");
}

// ── BoxFuture / ErasedTool edge case tests ──────────────────────

#[tokio::test]
async fn erased_dispatch_preserves_borrow_lifetime() {
    // Ensures the BoxToolFuture lifetime is tied to &self correctly,
    // i.e. the registry can be borrowed immutably while the future runs.
    let mut d = ToolRegistry::new();
    d.register(AsyncSleepTool);
    d.register(SampleTool);

    // Dispatch two calls on the same registry reference.
    let r1 = d
        .dispatch("async_sleep", serde_json::json!({}), &test_ctx())
        .await;
    let r2 = d
        .dispatch("sample", serde_json::json!({"path": "test"}), &test_ctx())
        .await;

    assert_eq!(r1.unwrap().content(), "slept");
    assert_eq!(r2.unwrap().content(), "test");
}

#[tokio::test]
async fn dispatch_returns_meaningful_error_for_wrong_type() {
    let mut d = ToolRegistry::new();
    d.register(RunCommandTool);

    // `command` expects a String, pass an object instead.
    let result = d
        .dispatch(
            "run_command",
            serde_json::json!({"command": {"nested": "object"}}),
            &test_ctx(),
        )
        .await;
    let err = result.unwrap_err();
    assert!(
        err.message
            .contains("Failed to deserialize tool parameters"),
        "Error should mention deserialization failure, got: {err}"
    );
}
