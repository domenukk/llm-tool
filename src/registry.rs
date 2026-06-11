//! Tool registry: registry and concurrent dispatch of named tools.

use std::collections::HashMap;

use super::{
    rust_tool::{ErasedTool, RustTool, definition_of},
    types::{ToolContext, ToolDefinition, ToolError, ToolOutput},
};

/// Entry holding a cached [`ToolDefinition`] alongside the type-erased tool.
///
/// The definition is computed once at registration time so that
/// [`ToolRegistry::definitions`] and [`ToolRegistry::iter`] never
/// regenerate JSON schemas.
struct RegisteredTool {
    definition: ToolDefinition,
    erased: Box<dyn ErasedTool>,
}

pub struct ToolRegistry {
    tools: HashMap<&'static str, RegisteredTool>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self
            .tools
            .values()
            .map(|r| r.definition.name.as_str())
            .collect();
        f.debug_struct("ToolRegistry")
            .field("tool_count", &self.tools.len())
            .field("tool_names", &names)
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a [`RustTool`]. Returns `&mut Self` for chaining.
    ///
    /// The tool's [`ToolDefinition`] (including JSON schema) is computed once
    /// here and cached for the lifetime of the registration.
    ///
    /// If a tool with the same name was already registered, it is replaced.
    ///
    /// # Panics
    ///
    /// Panics if the tool's JSON schema cannot be serialized. This indicates a
    /// bug in the tool's `Params` type (e.g. a broken `JsonSchema` impl).
    pub fn register<T: RustTool + 'static>(&mut self, tool: T) -> &mut Self {
        let definition = definition_of(&tool)
            .unwrap_or_else(|e| panic!("Failed to build definition for tool '{}': {e}", T::NAME));
        self.tools.insert(
            T::NAME,
            RegisteredTool {
                definition,
                erased: Box::new(tool),
            },
        );
        self
    }

    /// Register a [`RustTool`], consuming and returning `Self` for owned chaining.
    ///
    /// This is the owned counterpart of [`register`](Self::register), enabling
    /// patterns like:
    /// ```
    /// use llm_tool::{RustTool, ToolContext, ToolError, ToolOutput, ToolRegistry};
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct NoParams {}
    ///
    /// struct ToolA;
    /// impl RustTool for ToolA {
    ///     type Params = NoParams;
    ///     const NAME: &'static str = "tool_a";
    ///     const DESCRIPTION: &'static str = "Tool A";
    ///     async fn call(&self, _: NoParams, _: &ToolContext) -> Result<ToolOutput, ToolError> {
    ///         Ok("a".into())
    ///     }
    /// }
    ///
    /// struct ToolB;
    /// impl RustTool for ToolB {
    ///     type Params = NoParams;
    ///     const NAME: &'static str = "tool_b";
    ///     const DESCRIPTION: &'static str = "Tool B";
    ///     async fn call(&self, _: NoParams, _: &ToolContext) -> Result<ToolOutput, ToolError> {
    ///         Ok("b".into())
    ///     }
    /// }
    ///
    /// let registry = ToolRegistry::new().with_tool(ToolA).with_tool(ToolB);
    ///
    /// assert_eq!(registry.definitions().len(), 2);
    /// ```
    #[must_use]
    pub fn with_tool<T: RustTool + 'static>(mut self, tool: T) -> Self {
        self.register(tool);
        self
    }

    /// Collect [`ToolDefinition`]s for all registered tools.
    ///
    /// Returns clones of the cached definitions computed at registration time.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    /// Dispatch a tool call by name with raw JSON arguments and a context.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tool name is unknown or the handler returns an error.
    pub async fn dispatch(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let entry = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::new(format!("Unknown tool: {name}")))?;
        entry.erased.call_erased(args, ctx).await
    }

    /// Number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry has no registered tools.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Iterate over `(name, definition)` pairs for every registered tool.
    ///
    /// Returns clones of the cached definitions computed at registration time.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, ToolDefinition)> + '_ {
        self.tools
            .iter()
            .map(|(name, entry)| (*name, entry.definition.clone()))
    }
}

/// Iterate over `(name, definition)` pairs for every registered tool.
///
/// Yields `(&'static str, ToolDefinition)` for each tool in the registry.
impl<'a> IntoIterator for &'a ToolRegistry {
    type Item = (&'static str, ToolDefinition);
    type IntoIter = Box<dyn Iterator<Item = (&'static str, ToolDefinition)> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(
            self.tools
                .iter()
                .map(|(name, entry)| (*name, entry.definition.clone())),
        )
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{
        super::{EmptyParams, definition_of},
        *,
    };
    use crate::llm_tool;

    /// Create a default `ToolContext` for tests.
    fn test_ctx() -> ToolContext {
        ToolContext::new(None)
    }

    // ── Sample tool structs for tests ────────────────────────────────

    #[derive(Deserialize, schemars::JsonSchema)]
    struct PathParams {
        /// Filesystem path.
        path: String,
    }

    struct SampleTool;

    impl RustTool for SampleTool {
        type Params = PathParams;
        const NAME: &'static str = "sample";
        const DESCRIPTION: &'static str = "A sample tool";
        async fn call(
            &self,
            params: Self::Params,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(params.path.into())
        }
    }

    #[derive(Deserialize, schemars::JsonSchema)]
    struct RunCommandParams {
        /// Command to run.
        command: String,
        /// Timeout in seconds.
        #[serde(default)]
        timeout: Option<i64>,
        /// Environment variables.
        #[serde(default)]
        env: Option<std::collections::HashMap<String, String>>,
    }

    struct RunCommandTool;

    impl RustTool for RunCommandTool {
        type Params = RunCommandParams;
        const NAME: &'static str = "run_command";
        const DESCRIPTION: &'static str = "Runs a command.";
        async fn call(
            &self,
            params: Self::Params,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            assert!(params.timeout.is_none());
            assert!(params.env.is_none());
            Ok(format!("Ran: {}", params.command).into())
        }
    }

    // ── ToolDefinition tests ─────────────────────────────────────────

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
        let ctx = ToolContext::new(None);
        assert!(ctx.conversation_id().is_none());
        assert!(!ctx.is_idle());
    }

    #[test]
    fn tool_context_conversation_id_returns_value() {
        let ctx = ToolContext::new(Some("conv-123".to_owned()));
        assert_eq!(ctx.conversation_id(), Some("conv-123"));
    }

    #[test]
    fn tool_context_get_set_state_roundtrip() {
        let ctx = ToolContext::new(None);

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
        let ctx = ToolContext::new(None);
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

            async fn call(
                &self,
                _params: Self::Params,
                ctx: &ToolContext,
            ) -> Result<ToolOutput, ToolError> {
                let conv = ctx.conversation_id().unwrap_or("none");
                let count = ctx.get_state("call_count", serde_json::json!(0));
                let n = count.as_i64().unwrap_or(0);
                ctx.set_state("call_count", serde_json::json!(n + 1))
                    .map_err(|e| ToolError::new(format!("set_state failed: {e}")))?;
                Ok(format!("conv={conv}, call={n}").into())
            }
        }

        let mut d = ToolRegistry::new();
        d.register(ContextAwareTool);

        let ctx = ToolContext::new(Some("test-conv".to_owned()));

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
}
