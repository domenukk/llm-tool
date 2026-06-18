//! Strongly-typed Rust tool trait and type-erasure machinery.

use std::{borrow::Cow, future::Future, pin::Pin};

use super::types::{ToolContext, ToolDefinition, ToolError, ToolOutput};

/// Convenience type for tools that take no parameters.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
pub struct EmptyParams {}

/// A custom tool implemented entirely in Rust with strongly-typed parameters.
///
/// Define your parameters as a struct deriving [`serde::Deserialize`] and
/// `JsonSchema`, then implement this trait to provide the tool's logic.
/// The JSON Schema sent to the model is derived automatically from the
/// params struct — doc comments on fields become parameter descriptions.
///
/// Tools are async: for I/O-bound work (HTTP, filesystem, subprocess) the
/// runtime stays unblocked. Sync tools just don't `.await` anything — the
/// compiler optimizes the state machine to an immediate return.
///
/// # Example
///
/// ```rust
/// use llm_tool::{JsonSchema, RustTool, ToolContext, ToolError, ToolOutput};
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct FlashParams {
///     /// Target device identifier.
///     device_id: String,
///     /// Path to the firmware image.
///     image_path: String,
/// }
///
/// struct FlashDevice;
///
/// impl RustTool for FlashDevice {
///     type Params = FlashParams;
///     const NAME: &'static str = "flash_device";
///     const DESCRIPTION: &'static str = "Flashes firmware to a connected device.";
///
///     async fn call(
///         &self,
///         params: Self::Params,
///         _ctx: &ToolContext,
///     ) -> Result<ToolOutput, ToolError> {
///         Ok(format!("Flashed {} to {}", params.image_path, params.device_id).into())
///     }
/// }
/// ```
pub trait RustTool: Send + Sync {
    /// The strongly-typed parameters struct.
    ///
    /// Derive [`serde::Deserialize`] and `JsonSchema` on your params struct.
    /// `JsonSchema` auto-generates the parameter schema sent to the model;
    /// `Deserialize` parses the model's JSON arguments into your struct.
    type Params: serde::de::DeserializeOwned + schemars::JsonSchema + Send;

    /// Unique tool name (e.g. `"flash_device"`).
    const NAME: &'static str;

    /// Human-readable description shown to the model.
    const DESCRIPTION: &'static str;

    /// Return the tool description used in [`ToolDefinition`].
    ///
    /// The default returns [`Self::DESCRIPTION`] (the static string from a
    /// doc comment or template body). When using
    /// `#[llm_tool(template = "...", context = ...)]`, the generated
    /// implementation overrides this to render the template with runtime
    /// variables on each call. Templates are parsed once via `LazyLock`.
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::DESCRIPTION)
    }

    /// Execute the tool with typed parameters and an execution context.
    ///
    /// Async to support I/O-bound tools (HTTP, filesystem, subprocess).
    /// Sync tools just compute and return — the async wrapper is zero-cost.
    ///
    /// The `ctx` parameter provides access to conversation metadata and a
    /// shared key-value state store. Tools that don't need context can simply
    /// ignore it with `_ctx`.
    ///
    /// # Errors
    ///
    /// Returns `Err(ToolError)` if the tool execution fails.
    fn call(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl std::future::Future<Output = Result<ToolOutput, ToolError>> + Send;
}

/// Build a [`ToolDefinition`] from any [`RustTool`] implementor.
///
/// The generated schema is sanitized to be compatible with the Go-based
/// localharness, which expects `"type"` to always be a single string
/// (not the array form `["string", "null"]` that schemars emits for
/// `Option<T>` fields).
///
/// # Errors
///
/// Returns `Err` if the JSON schema serialization fails.
pub fn definition_of<T: RustTool>(tool: &T) -> Result<ToolDefinition, ToolError> {
    let schema = schemars::schema_for!(T::Params);
    let mut parameter_schema = serde_json::to_value(schema).map_err(|e| {
        ToolError::new(format!(
            "Failed to serialize schema for tool '{}': {e}",
            T::NAME
        ))
    })?;
    sanitize_schema_types(&mut parameter_schema);
    Ok(ToolDefinition {
        name: T::NAME.to_string(),
        description: tool.description().into_owned(),
        parameter_schema,
    })
}

/// Recursively sanitize JSON Schema `"type"` fields for Go genai compatibility.
///
/// `schemars` emits `"type": ["string", "null"]` for `Option<String>` fields
/// (the nullable type-array form). The Go genai SDK's `Schema.Type`
/// is a single `genai.Type` enum, so it can't unmarshal an array.
///
/// This function walks the schema tree and replaces any array `type` with the
/// first non-`"null"` element. For example:
/// - `["string", "null"]` → `"string"`
/// - `["integer", "null"]` → `"integer"`
fn sanitize_schema_types(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            // If "type" is an array (e.g. ["string", "null"]), pick the first
            // non-"null" element and replace with it as a scalar type.
            let replacement = match map.get("type") {
                Some(serde_json::Value::Array(arr)) => {
                    let non_null = arr.iter().find(|v| v.as_str() != Some("null")).cloned();
                    non_null.or_else(|| arr.first().cloned())
                }
                _ => None,
            };
            if let Some(val) = replacement {
                map.insert("type".to_string(), val);
            }
            for val in map.values_mut() {
                sanitize_schema_types(val);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                sanitize_schema_types(item);
            }
        }
        _ => {}
    }
}

/// Type-erased future returned by [`ErasedTool::call_erased`].
type BoxToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous tool storage.
///
/// Boxes the future from [`RustTool::call`] so we can store different tool
/// types in the same `HashMap<String, Box<dyn ErasedTool>>`.
pub(crate) trait ErasedTool: Send + Sync {
    /// Deserialize `args` and call the handler, returning a boxed future.
    fn call_erased<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a>;
}

impl<T: RustTool> ErasedTool for T {
    fn call_erased<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let params: T::Params = serde_json::from_value(args).map_err(|e| {
                ToolError::new(format!("Failed to deserialize tool parameters: {e}"))
            })?;
            self.call(params, ctx).await
        })
    }
}
