//! Strongly-typed Rust tool trait and type-erasure machinery.

use alloc::{borrow::Cow, boxed::Box, format};
use core::{future::Future, pin::Pin};

use super::types::{
    OutputSanitization, ToolContext, ToolDefinition, ToolEffect, ToolError, ToolOutput,
    ToolProvenance,
};

/// Convenience type for tools that take no parameters.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
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
    type Params: serde::de::DeserializeOwned + schemars::JsonSchema + Send;

    /// Unique tool name (e.g. `"flash_device"`).
    const NAME: &'static str;

    /// Human-readable description shown to the model.
    const DESCRIPTION: &'static str;

    /// Execution effect of the tool.
    const EFFECT: ToolEffect = if Self::IDEMPOTENT {
        ToolEffect::ReadOnly
    } else {
        ToolEffect::Mutating
    };

    /// Whether the tool is idempotent / read-only safe for concurrent wave execution.
    const IDEMPOTENT: bool = false;

    /// Data-provenance and trust classification of the tool.
    ///
    /// Defaults to [`ToolProvenance::Trusted`]. Tools with [`ToolProvenance::ExternalMcp`]
    /// or [`ToolProvenance::UntrustedNetwork`] automatically default [`Self::OUTPUT_SANITIZATION`]
    /// to [`OutputSanitization::Quarantine`] unless overridden.
    const PROVENANCE: ToolProvenance = ToolProvenance::Trusted;

    /// Shorthand: neutralize control tokens in this tool's output.
    ///
    /// Defaults to `true` (secure by default). Set to `false` to opt out, or prefer
    /// setting [`Self::OUTPUT_SANITIZATION`] directly.
    const SANITIZE_OUTPUT: bool = true;

    /// Shorthand: wrap this tool's output in a `<tool_output_quarantine>` envelope.
    ///
    /// Takes precedence over [`Self::SANITIZE_OUTPUT`]. Prefer setting
    /// [`Self::OUTPUT_SANITIZATION`] directly.
    const QUARANTINE_OUTPUT: bool = false;

    /// Automatic prompt-injection sanitization policy applied to this tool's output.
    const OUTPUT_SANITIZATION: OutputSanitization = if Self::QUARANTINE_OUTPUT
        || (Self::SANITIZE_OUTPUT && Self::PROVENANCE.is_untrusted_source())
    {
        OutputSanitization::Quarantine
    } else if Self::SANITIZE_OUTPUT {
        OutputSanitization::SanitizeTokens
    } else {
        OutputSanitization::None
    };

    /// Return the execution effect of the tool.
    fn effect(&self) -> ToolEffect {
        Self::EFFECT
    }

    /// Return the data-provenance tier of the tool.
    fn provenance(&self) -> ToolProvenance {
        Self::PROVENANCE
    }

    /// Return whether the tool is idempotent (read-only).
    fn is_idempotent(&self) -> bool {
        matches!(self.effect(), ToolEffect::ReadOnly)
    }

    /// Return the automatic prompt-injection sanitization policy for this tool's output,
    /// incorporating [`provenance`](Self::provenance) unless sanitization was explicitly disabled.
    fn output_sanitization(&self) -> OutputSanitization {
        let base = Self::OUTPUT_SANITIZATION;
        if !Self::SANITIZE_OUTPUT || base.is_none() {
            base
        } else {
            base.at_least(self.provenance().default_sanitization())
        }
    }

    /// Return the tool description used in [`ToolDefinition`].
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::DESCRIPTION)
    }

    /// Execute the tool with typed parameters and an execution context.
    ///
    /// # Errors
    ///
    /// Returns `Err(ToolError)` if the tool execution fails.
    fn call(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send;

    /// Internal unsanitized invocation hook overridden by `#[llm_tool]` so type-erased
    /// and registry dispatch apply output/error sanitization in a single pass.
    #[doc(hidden)]
    fn __call_raw(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        self.call(params, ctx)
    }
}

/// Shared [`alloc::sync::Arc`]-backed wrapper for any [`RustTool`] implementation,
/// enabling cheap cloning across registries and subagent scopes.
#[derive(Debug)]
pub struct SharedRustTool<T>(pub alloc::sync::Arc<T>);

impl<T> Clone for SharedRustTool<T> {
    fn clone(&self) -> Self {
        Self(alloc::sync::Arc::clone(&self.0))
    }
}

impl<T> SharedRustTool<T> {
    /// Wrap `tool` in a new [`SharedRustTool`].
    #[must_use]
    pub fn new(tool: T) -> Self {
        Self(alloc::sync::Arc::new(tool))
    }

    /// Borrow the underlying tool.
    #[must_use]
    pub fn as_inner(&self) -> &T {
        &self.0
    }
}

impl<T> From<alloc::sync::Arc<T>> for SharedRustTool<T> {
    fn from(arc: alloc::sync::Arc<T>) -> Self {
        Self(arc)
    }
}

impl<T: RustTool> RustTool for alloc::sync::Arc<T> {
    type Params = T::Params;
    const NAME: &'static str = T::NAME;
    const DESCRIPTION: &'static str = T::DESCRIPTION;
    const EFFECT: ToolEffect = T::EFFECT;
    const IDEMPOTENT: bool = T::IDEMPOTENT;
    const PROVENANCE: ToolProvenance = T::PROVENANCE;
    const SANITIZE_OUTPUT: bool = T::SANITIZE_OUTPUT;
    const QUARANTINE_OUTPUT: bool = T::QUARANTINE_OUTPUT;
    const OUTPUT_SANITIZATION: OutputSanitization = T::OUTPUT_SANITIZATION;

    fn effect(&self) -> ToolEffect {
        (**self).effect()
    }

    fn provenance(&self) -> ToolProvenance {
        (**self).provenance()
    }

    fn is_idempotent(&self) -> bool {
        (**self).is_idempotent()
    }

    fn output_sanitization(&self) -> OutputSanitization {
        (**self).output_sanitization()
    }

    fn description(&self) -> Cow<'static, str> {
        (**self).description()
    }

    fn call(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        (**self).call(params, ctx)
    }

    fn __call_raw(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        (**self).__call_raw(params, ctx)
    }
}

impl<T: RustTool> RustTool for SharedRustTool<T> {
    type Params = T::Params;
    const NAME: &'static str = T::NAME;
    const DESCRIPTION: &'static str = T::DESCRIPTION;
    const EFFECT: ToolEffect = T::EFFECT;
    const IDEMPOTENT: bool = T::IDEMPOTENT;
    const PROVENANCE: ToolProvenance = T::PROVENANCE;
    const SANITIZE_OUTPUT: bool = T::SANITIZE_OUTPUT;
    const QUARANTINE_OUTPUT: bool = T::QUARANTINE_OUTPUT;
    const OUTPUT_SANITIZATION: OutputSanitization = T::OUTPUT_SANITIZATION;

    fn effect(&self) -> ToolEffect {
        self.0.effect()
    }

    fn provenance(&self) -> ToolProvenance {
        self.0.provenance()
    }

    fn is_idempotent(&self) -> bool {
        self.0.is_idempotent()
    }

    fn output_sanitization(&self) -> OutputSanitization {
        self.0.output_sanitization()
    }

    fn description(&self) -> Cow<'static, str> {
        self.0.description()
    }

    fn call(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        self.0.call(params, ctx)
    }

    fn __call_raw(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        self.0.__call_raw(params, ctx)
    }
}

/// Build a [`ToolDefinition`] from any [`RustTool`] implementor.
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
    crate::types::quarantine::normalize_parameter_schema(&mut parameter_schema);
    Ok(ToolDefinition::with_effect(
        T::NAME,
        tool.description().into_owned(),
        parameter_schema,
        tool.effect(),
    )
    .with_provenance(tool.provenance())
    .with_output_sanitization(tool.output_sanitization()))
}

/// Type-erased future returned when dispatching a boxed tool.
pub type BoxToolFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous tool storage.
pub trait ErasedTool: Send + Sync {
    /// Deserialize `args` and call the handler with the tool's own sanitization policy.
    fn call_erased<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a>;

    /// Deserialize `args_json` directly from a JSON string and call the handler.
    fn call_erased_str<'a>(&'a self, args_json: &'a str, ctx: &'a ToolContext)
    -> BoxToolFuture<'a>;

    /// Deserialize `args` and call the handler under an explicit `effective_policy` in one pass.
    fn call_erased_with_policy<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        let fut = self.call_erased(args, ctx);
        Box::pin(async move { effective_policy.apply_result(fut.await) })
    }

    /// Deserialize `args_json` and call the handler under an explicit `effective_policy` in one pass.
    fn call_erased_str_with_policy<'a>(
        &'a self,
        args_json: &'a str,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        let fut = self.call_erased_str(args_json, ctx);
        Box::pin(async move { effective_policy.apply_result(fut.await) })
    }
}

impl<T: RustTool> ErasedTool for T {
    fn call_erased<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a> {
        let policy = self.output_sanitization();
        self.call_erased_with_policy(args, ctx, policy)
    }

    fn call_erased_str<'a>(
        &'a self,
        args_json: &'a str,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a> {
        let policy = self.output_sanitization();
        self.call_erased_str_with_policy(args_json, ctx, policy)
    }

    fn call_erased_with_policy<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let params: T::Params = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => {
                    let err = ToolError::new(format!("Failed to deserialize tool parameters: {e}"));
                    return Err(effective_policy.apply_error(err));
                }
            };
            let res = self.__call_raw(params, ctx).await;
            effective_policy.apply_result(res)
        })
    }

    fn call_erased_str_with_policy<'a>(
        &'a self,
        args_json: &'a str,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let params: T::Params = match serde_json::from_str(args_json) {
                Ok(p) => p,
                Err(e) => {
                    let err = ToolError::new(format!("Failed to deserialize tool parameters: {e}"));
                    return Err(effective_policy.apply_error(err));
                }
            };
            let res = self.__call_raw(params, ctx).await;
            effective_policy.apply_result(res)
        })
    }
}
