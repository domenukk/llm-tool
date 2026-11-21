//! Tool registry: registry, safety pipeline, projections, and concurrent dispatch of named tools.

mod batch;
mod projection;

use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};

pub use batch::{
    AllowDestructiveBatch, BatchExecutionMode, BatchToolCall, BatchToolCallOutcome,
    BatchToolCallStr,
};
pub use projection::DynamicToolHandler;

use super::{
    rust_tool::{ErasedTool, RustTool, definition_of},
    types::{
        OutputSanitization, RegistryItem, SpillContext, SurrogateVault, ToolContext,
        ToolDefinition, ToolEffect, ToolError, ToolOutput, ToolProvenance, TruncationPolicy,
    },
};
use crate::{
    compat::{HashMap, RwLock, read_lock_recover, write_lock_recover},
    rust_tool::BoxToolFuture,
};

/// Entry holding a cached [`ToolDefinition`] alongside a cheaply-cloneable type-erased tool.
#[derive(Clone)]
struct RegisteredTool {
    definition: ToolDefinition,
    erased: Arc<dyn ErasedTool>,
}

/// A registry of named tools available for dynamic dispatch.
///
/// Holds type-erased tool implementations and cached [`ToolDefinition`] schemas
/// behind [`Arc`] handles so cloning, subsetting, role-filtering, and merging
/// registries across subagent scopes is cheap (`O(tools)` pointer clones, zero
/// JSON Schema regeneration).
///
/// When configured with [`with_surrogate_redaction`](Self::with_surrogate_redaction)
/// and/or [`with_truncation_policy`](Self::with_truncation_policy), every dispatched
/// call executes the defense pipeline in strict order:
/// `1. SurrogateVault::redact -> 2. OutputSanitization -> 3. TruncationPolicy`.
pub struct ToolRegistry {
    tools: HashMap<Cow<'static, str>, RegisteredTool>,
    cached_definitions: RwLock<Option<Arc<[ToolDefinition]>>>,
    pub(super) output_floor: OutputSanitization,
    pub(super) truncation_policy: Option<TruncationPolicy>,
    pub(super) surrogate_redaction: bool,
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        let cached = read_lock_recover(&self.cached_definitions).clone();
        Self {
            tools: self.tools.clone(),
            cached_definitions: RwLock::new(cached),
            output_floor: self.output_floor,
            truncation_policy: self.truncation_policy.clone(),
            surrogate_redaction: self.surrogate_redaction,
        }
    }
}

impl core::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names: Vec<&str> = self
            .tools
            .values()
            .map(|r| r.definition.name.as_str())
            .collect();
        f.debug_struct("ToolRegistry")
            .field("tool_count", &self.tools.len())
            .field("tool_names", &names)
            .field("output_floor", &self.output_floor)
            .field("has_truncation_policy", &self.truncation_policy.is_some())
            .field("surrogate_redaction", &self.surrogate_redaction)
            .finish_non_exhaustive()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn redact_json_value_in_place(
    vault: &SurrogateVault,
    val: &mut serde_json::Value,
) -> Result<(), ToolError> {
    match val {
        serde_json::Value::String(s) => {
            *s = vault.redact_str(s)?;
        }
        serde_json::Value::Array(items) => {
            for item in items {
                redact_json_value_in_place(vault, item)?;
            }
        }
        serde_json::Value::Object(map) => {
            let entries = core::mem::take(map);
            let mut rebuilt = serde_json::Map::with_capacity(entries.len());
            for (k, mut v) in entries {
                let clean_key = vault.redact_str(&k)?;
                redact_json_value_in_place(vault, &mut v)?;
                rebuilt.insert(clean_key, v);
            }
            *map = rebuilt;
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
    Ok(())
}

fn redact_tool_output(
    vault: &SurrogateVault,
    mut output: ToolOutput,
) -> Result<ToolOutput, ToolError> {
    output.content = vault.redact_str(&output.content)?;
    if !output.metadata.is_empty() {
        let entries = core::mem::take(&mut output.metadata);
        let mut rebuilt = HashMap::with_capacity(entries.len());
        for (k, mut v) in entries {
            let clean_key = vault.redact_str(&k)?;
            redact_json_value_in_place(vault, &mut v)?;
            rebuilt.insert(clean_key, v);
        }
        output.metadata = rebuilt;
    }
    Ok(output)
}

fn redact_tool_error(vault: &SurrogateVault, mut err: ToolError) -> ToolError {
    match vault.redact_str(&err.message) {
        Ok(redacted) => err.message = redacted,
        Err(redact_err) => {
            tracing::warn!(error = %redact_err, "ToolRegistry: failed to redact surrogates in ToolError");
        }
    }
    err
}

impl ToolRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            cached_definitions: RwLock::new(None),
            output_floor: OutputSanitization::default(),
            truncation_policy: None,
            surrogate_redaction: false,
        }
    }

    /// Set the minimum [`OutputSanitization`] enforced on every dispatched tool output.
    #[must_use]
    pub const fn with_output_floor(mut self, floor: OutputSanitization) -> Self {
        self.output_floor = floor;
        self
    }

    /// In-place variant of [`with_output_floor`](Self::with_output_floor).
    pub const fn set_output_floor(&mut self, floor: OutputSanitization) -> &mut Self {
        self.output_floor = floor;
        self
    }

    /// The minimum [`OutputSanitization`] enforced on every dispatched tool output.
    #[must_use]
    pub const fn output_floor(&self) -> OutputSanitization {
        self.output_floor
    }

    /// Configure a registry-wide [`TruncationPolicy`] applied automatically after sanitization
    /// (and after optional surrogate redaction) on every dispatched tool output.
    #[must_use]
    pub fn with_truncation_policy(mut self, policy: TruncationPolicy) -> Self {
        self.truncation_policy = Some(policy);
        self
    }

    /// In-place variant of [`with_truncation_policy`](Self::with_truncation_policy).
    pub fn set_truncation_policy(&mut self, policy: TruncationPolicy) -> &mut Self {
        self.truncation_policy = Some(policy);
        self
    }

    /// Remove any configured registry-wide [`TruncationPolicy`].
    #[must_use]
    pub fn without_truncation_policy(mut self) -> Self {
        self.truncation_policy = None;
        self
    }

    /// Borrow the configured [`TruncationPolicy`], if any.
    #[must_use]
    pub const fn truncation_policy(&self) -> Option<&TruncationPolicy> {
        self.truncation_policy.as_ref()
    }

    /// Enable or disable automatic [`SurrogateVault::redact_str`] execution prior to
    /// output sanitization and truncation when a [`SurrogateVault`] is present in [`ToolContext`].
    ///
    /// Defaults to `false` so callers manually managing secret resolution/redaction keep
    /// exact control unless opted in.
    #[must_use]
    pub const fn with_surrogate_redaction(mut self, enabled: bool) -> Self {
        self.surrogate_redaction = enabled;
        self
    }

    /// In-place variant of [`with_surrogate_redaction`](Self::with_surrogate_redaction).
    pub const fn set_surrogate_redaction(&mut self, enabled: bool) -> &mut Self {
        self.surrogate_redaction = enabled;
        self
    }

    /// Return whether automatic surrogate redaction is enabled on this registry.
    #[must_use]
    pub const fn surrogate_redaction(&self) -> bool {
        self.surrogate_redaction
    }

    /// Compute the effective [`OutputSanitization`] for a registered [`ToolDefinition`].
    #[must_use]
    pub(crate) const fn effective_policy_for_def(
        &self,
        def: &ToolDefinition,
    ) -> OutputSanitization {
        let tool_policy = if def.output_sanitization.is_none() {
            OutputSanitization::None
        } else {
            def.output_sanitization
                .at_least(def.provenance.default_sanitization())
        };
        tool_policy.at_least(self.output_floor)
    }

    /// The policy actually applied to `name`'s output: its own policy raised to the registry floor.
    #[must_use]
    pub fn effective_output_sanitization(&self, name: &str) -> Option<OutputSanitization> {
        self.definition(name)
            .map(|def| self.effective_policy_for_def(def))
    }

    fn not_found_error(&self, name: &str) -> ToolError {
        let err = ToolError::not_found(RegistryItem::Tool, name);
        self.output_floor.apply_error(err)
    }

    pub(super) fn invalidate_cache(&self) {
        let mut guard = write_lock_recover(&self.cached_definitions);
        *guard = None;
    }

    /// Register a [`RustTool`]. Returns `&mut Self` for chaining.
    ///
    /// # Panics
    ///
    /// Panics if the tool's JSON schema cannot be serialized. Use
    /// [`try_register`](Self::try_register) for the non-panicking variant.
    pub fn register<T: RustTool + 'static>(&mut self, tool: T) -> &mut Self {
        if let Err(e) = self.try_register(tool) {
            panic!("Failed to build definition for tool '{}': {e}", T::NAME);
        }
        self
    }

    /// Register a [`RustTool`], returning an error instead of panicking if the
    /// tool's JSON schema cannot be built.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the tool's `Params` type fails to produce a JSON schema.
    pub fn try_register<T: RustTool + 'static>(&mut self, tool: T) -> Result<&mut Self, ToolError> {
        let definition = definition_of(&tool)?;
        self.tools.insert(
            Cow::Borrowed(T::NAME),
            RegisteredTool {
                definition,
                erased: Arc::new(tool),
            },
        );
        self.invalidate_cache();
        Ok(self)
    }

    /// Register a [`RustTool`], consuming and returning `Self` for owned chaining.
    #[must_use]
    pub fn with_tool<T: RustTool + 'static>(mut self, tool: T) -> Self {
        self.register(tool);
        self
    }

    /// Return a shared, deterministically sorted slice of all [`ToolDefinition`]s,
    /// computed once and cached until the registry is mutated.
    #[must_use]
    pub fn definitions_slice(&self) -> Arc<[ToolDefinition]> {
        {
            let guard = read_lock_recover(&self.cached_definitions);
            if let Some(ref slice) = *guard {
                return Arc::clone(slice);
            }
        }
        let mut defs: Vec<ToolDefinition> = self
            .tools
            .values()
            .map(|entry| entry.definition.clone())
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        let arc_slice: Arc<[ToolDefinition]> = Arc::from(defs);
        {
            let mut guard = write_lock_recover(&self.cached_definitions);
            *guard = Some(Arc::clone(&arc_slice));
        }
        arc_slice
    }

    /// Collect [`ToolDefinition`]s for all registered tools.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions_slice().to_vec()
    }

    fn execute_pipeline<'a>(
        &'a self,
        tool_name: &'a str,
        raw_fut: BoxToolFuture<'a>,
        ctx: &'a ToolContext,
        effective: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        Box::pin(async move {
            let vault_opt = if self.surrogate_redaction {
                ctx.get_ext::<SurrogateVault>()
            } else {
                None
            };

            let mut output = match raw_fut.await {
                Ok(out) => out,
                Err(mut err) => {
                    if let Some(ref vault) = vault_opt {
                        err = redact_tool_error(vault, err);
                    }
                    return Err(effective.apply_error(err));
                }
            };

            // Stage 1: SurrogateVault::redact
            if let Some(ref vault) = vault_opt {
                output = redact_tool_output(vault, output).map_err(|e| effective.apply_error(e))?;
            }

            // Stage 2 + Stage 3: OutputSanitization -> TruncationPolicy
            if let Some(ref trunc_policy) = self.truncation_policy {
                let mut spill_ctx = SpillContext::new(tool_name);
                if let Some(conv_id) = ctx.conversation_id() {
                    spill_ctx = spill_ctx.with_tag(conv_id);
                }
                Ok(effective.apply_truncated_with_context(output, trunc_policy, &spill_ctx))
            } else {
                Ok(effective.apply(output))
            }
        })
    }

    /// Dispatch a tool call by name with raw JSON arguments and a context,
    /// returning a boxed future.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no tool named `name` is registered.
    pub fn dispatch_boxed<'a>(
        &'a self,
        name: &str,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> Result<BoxToolFuture<'a>, ToolError> {
        let Some((stored_key, entry)) = self.tools.get_key_value(name) else {
            return Err(self.not_found_error(name));
        };
        let effective = self.effective_policy_for_def(&entry.definition);
        if !self.surrogate_redaction && self.truncation_policy.is_none() {
            return Ok(entry.erased.call_erased_with_policy(args, ctx, effective));
        }
        let raw_fut = entry
            .erased
            .call_erased_with_policy(args, ctx, OutputSanitization::None);
        Ok(self.execute_pipeline(stored_key.as_ref(), raw_fut, ctx, effective))
    }

    /// Dispatch a tool call by name with a raw JSON arguments string,
    /// returning a boxed future without intermediate DOM allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no tool named `name` is registered.
    pub fn dispatch_boxed_str<'a>(
        &'a self,
        name: &str,
        args_json: &'a str,
        ctx: &'a ToolContext,
    ) -> Result<BoxToolFuture<'a>, ToolError> {
        let Some((stored_key, entry)) = self.tools.get_key_value(name) else {
            return Err(self.not_found_error(name));
        };
        let effective = self.effective_policy_for_def(&entry.definition);
        if !self.surrogate_redaction && self.truncation_policy.is_none() {
            return Ok(entry
                .erased
                .call_erased_str_with_policy(args_json, ctx, effective));
        }
        let raw_fut =
            entry
                .erased
                .call_erased_str_with_policy(args_json, ctx, OutputSanitization::None);
        Ok(self.execute_pipeline(stored_key.as_ref(), raw_fut, ctx, effective))
    }

    /// Dispatch a tool call by name with raw JSON arguments and a context.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no tool named `name` is registered,
    /// or the tool's own error if argument deserialization or execution fails.
    pub async fn dispatch(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        self.dispatch_boxed(name, args, ctx)?.await
    }

    /// Dispatch a tool call by name with a raw JSON string argument.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no tool named `name` is registered,
    /// or the tool's own error if JSON parsing or the handler fails.
    pub async fn dispatch_str(
        &self,
        name: &str,
        args_json: &str,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        self.dispatch_boxed_str(name, args_json, ctx)?.await
    }

    /// Return `true` if the registered tool has [`ToolEffect::Destructive`].
    #[must_use]
    pub fn is_destructive(&self, name: &str) -> bool {
        self.definition(name)
            .is_some_and(|def| matches!(def.effect, ToolEffect::Destructive))
    }

    /// Return the [`ToolProvenance`] of the registered tool named `name`, or `None` if not registered.
    #[must_use]
    pub fn provenance(&self, name: &str) -> Option<ToolProvenance> {
        self.definition(name).map(|def| def.provenance)
    }

    /// Remove a tool by name, returning `true` if it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        let removed = self.tools.remove(name).is_some();
        if removed {
            self.invalidate_cache();
        }
        removed
    }

    /// Clear all registered tools.
    pub fn clear(&mut self) {
        self.tools.clear();
        self.invalidate_cache();
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

    /// Whether a tool with the given name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Borrow the cached [`ToolDefinition`] for a registered tool by name.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name).map(|entry| &entry.definition)
    }

    /// Iterate over `(name, definition)` pairs for every registered tool.
    #[must_use]
    pub fn iter(&self) -> ToolDefinitions<'_> {
        ToolDefinitions {
            inner: self.tools.iter(),
        }
    }
}

/// Borrowing iterator over `(name, definition)` pairs, yielded by [`ToolRegistry::iter`].
pub struct ToolDefinitions<'a> {
    inner: crate::compat::HashMapIter<'a, Cow<'static, str>, RegisteredTool>,
}

impl<'a> Iterator for ToolDefinitions<'a> {
    type Item = (&'a str, ToolDefinition);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(name, entry)| (name.as_ref(), entry.definition.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for ToolDefinitions<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> IntoIterator for &'a ToolRegistry {
    type Item = (&'a str, ToolDefinition);
    type IntoIter = ToolDefinitions<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
