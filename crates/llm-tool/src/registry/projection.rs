//! Registry role projections, subsetting, merging, dynamic tool registration,
//! and multi-provider wire schema helpers for [`ToolRegistry`].

use alloc::{borrow::Cow, boxed::Box, collections::BTreeSet, format, sync::Arc, vec::Vec};

use super::{RegisteredTool, ToolRegistry};
use crate::{
    rust_tool::{BoxToolFuture, ErasedTool},
    types::{
        AnthropicTool, GeminiFunctionDeclaration, OpenAiTool, OutputSanitization, ToolContext,
        ToolDefinition, ToolError, ToolProvenance,
    },
};

/// Dynamic handler trait for runtime-constructed or MCP-bridged tools registered via
/// [`ToolRegistry::register_dynamic`].
///
/// Blanket-implemented for any `Send + Sync` closure or function of signature
/// `Fn(serde_json::Value, &ToolContext) -> BoxToolFuture<'_>`.
pub trait DynamicToolHandler: Send + Sync {
    /// Invoke the dynamic tool with JSON `args` and execution `ctx`.
    fn call_dynamic<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a>;
}

impl<F> DynamicToolHandler for F
where
    F: for<'a> Fn(serde_json::Value, &'a ToolContext) -> BoxToolFuture<'a> + Send + Sync,
{
    fn call_dynamic<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a> {
        (self)(args, ctx)
    }
}

struct DynamicErasedAdapter<H> {
    default_policy: OutputSanitization,
    handler: H,
}

impl<H: DynamicToolHandler> ErasedTool for DynamicErasedAdapter<H> {
    fn call_erased<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a> {
        self.call_erased_with_policy(args, ctx, self.default_policy)
    }

    fn call_erased_str<'a>(
        &'a self,
        args_json: &'a str,
        ctx: &'a ToolContext,
    ) -> BoxToolFuture<'a> {
        self.call_erased_str_with_policy(args_json, ctx, self.default_policy)
    }

    fn call_erased_with_policy<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        let fut = self.handler.call_dynamic(args, ctx);
        Box::pin(async move { effective_policy.apply_result(fut.await) })
    }

    fn call_erased_str_with_policy<'a>(
        &'a self,
        args_json: &'a str,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxToolFuture<'a> {
        let args: serde_json::Value = match serde_json::from_str(args_json) {
            Ok(v) => v,
            Err(e) => {
                let err = ToolError::new(format!("Failed to deserialize tool parameters: {e}"));
                return Box::pin(core::future::ready(Err(effective_policy.apply_error(err))));
            }
        };
        self.call_erased_with_policy(args, ctx, effective_policy)
    }
}

impl ToolRegistry {
    /// Retain in-place only tools for which `pred(name, definition)` returns `true`.
    pub fn retain<F>(&mut self, mut pred: F) -> &mut Self
    where
        F: FnMut(&str, &ToolDefinition) -> bool,
    {
        let before = self.tools.len();
        self.tools
            .retain(|name, entry| pred(name.as_ref(), &entry.definition));
        if self.tools.len() != before {
            self.invalidate_cache();
        }
        self
    }

    /// Return a cloned registry containing only tools satisfying `pred(name, definition)`.
    #[must_use]
    pub fn filtered<F>(&self, pred: F) -> Self
    where
        F: FnMut(&str, &ToolDefinition) -> bool,
    {
        let mut copy = self.clone();
        copy.retain(pred);
        copy
    }

    /// Return a cloned registry narrowed to read-only (`ToolEffect::ReadOnly`) tools
    /// (e.g. for `/plan` mode or read-only verifier subagents).
    #[must_use]
    pub fn read_only(&self) -> Self {
        self.filtered(|_, def| def.effect.is_read_only())
    }

    /// Return a cloned registry with all `ToolEffect::Destructive` tools removed.
    #[must_use]
    pub fn without_destructive(&self) -> Self {
        self.filtered(|_, def| !def.effect.is_destructive())
    }

    /// Return a cloned registry narrowed to tools whose [`ToolProvenance`] is at most `max_provenance`.
    #[must_use]
    pub fn with_max_provenance(&self, max_provenance: ToolProvenance) -> Self {
        self.filtered(|_, def| def.provenance <= max_provenance)
    }

    /// Return a cloned registry containing only the specified tool `names`.
    #[must_use]
    pub fn subset<'a, I>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        let allowed: BTreeSet<&'a str> = names.into_iter().collect();
        self.filtered(|name, _| allowed.contains(name))
    }

    /// Return a cloned registry excluding all specified tool `names`.
    #[must_use]
    pub fn without<'a, I>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        let denied: BTreeSet<&'a str> = names.into_iter().collect();
        self.filtered(|name, _| !denied.contains(name))
    }

    /// Copy all tools from `other` into `self` (replacing colliding names), combining
    /// `output_floor` via [`OutputSanitization::at_least`], preserving `surrogate_redaction`
    /// if enabled on either registry, and adopting `other`'s `truncation_policy` if `self` has none.
    pub fn extend_from(&mut self, other: &Self) -> &mut Self {
        self.output_floor = self.output_floor.at_least(other.output_floor);
        self.surrogate_redaction |= other.surrogate_redaction;
        if self.truncation_policy.is_none() {
            self.truncation_policy.clone_from(&other.truncation_policy);
        }
        if !other.tools.is_empty() {
            for (name, entry) in &other.tools {
                self.tools.insert(name.clone(), entry.clone());
            }
            self.invalidate_cache();
        }
        self
    }

    /// Consume and merge `other` into `self`, combining security floors monotonically.
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.output_floor = self.output_floor.at_least(other.output_floor);
        self.surrogate_redaction |= other.surrogate_redaction;
        if self.truncation_policy.is_none() {
            self.truncation_policy = other.truncation_policy;
        }
        if !other.tools.is_empty() {
            self.tools.extend(other.tools);
            self.invalidate_cache();
        }
        self
    }

    /// Register a dynamic or MCP-bridged tool from a runtime [`ToolDefinition`] and async closure.
    ///
    /// For struct-based handlers implementing [`DynamicToolHandler`], see
    /// [`register_dynamic_handler`](Self::register_dynamic_handler).
    pub fn register_dynamic<F>(&mut self, definition: ToolDefinition, handler: F) -> &mut Self
    where
        F: for<'a> Fn(serde_json::Value, &'a ToolContext) -> BoxToolFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.register_dynamic_handler(definition, handler)
    }

    /// Owned builder variant of [`register_dynamic`](Self::register_dynamic).
    #[must_use]
    pub fn with_dynamic_tool<F>(mut self, definition: ToolDefinition, handler: F) -> Self
    where
        F: for<'a> Fn(serde_json::Value, &'a ToolContext) -> BoxToolFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.register_dynamic(definition, handler);
        self
    }

    /// Register a dynamic tool backed by any [`DynamicToolHandler`] implementation.
    pub fn register_dynamic_handler<H>(
        &mut self,
        definition: ToolDefinition,
        handler: H,
    ) -> &mut Self
    where
        H: DynamicToolHandler + 'static,
    {
        let default_policy = if definition.output_sanitization.is_none() {
            OutputSanitization::None
        } else {
            definition
                .output_sanitization
                .at_least(definition.provenance.default_sanitization())
        };
        let key = Cow::Owned(definition.name.clone());
        let adapter = DynamicErasedAdapter {
            default_policy,
            handler,
        };
        self.tools.insert(
            key,
            RegisteredTool {
                definition,
                erased: Arc::new(adapter),
            },
        );
        self.invalidate_cache();
        self
    }

    /// Project all registered tools into sorted [`OpenAiTool`] wire schemas.
    #[must_use]
    pub fn to_openai_tools(&self) -> Vec<OpenAiTool> {
        self.definitions_slice()
            .iter()
            .map(ToolDefinition::to_openai_tool)
            .collect()
    }

    /// Project all registered tools into sorted [`AnthropicTool`] wire schemas.
    #[must_use]
    pub fn to_anthropic_tools(&self) -> Vec<AnthropicTool> {
        self.definitions_slice()
            .iter()
            .map(ToolDefinition::to_anthropic_tool)
            .collect()
    }

    /// Project all registered tools into sorted [`GeminiFunctionDeclaration`] wire schemas.
    #[must_use]
    pub fn to_gemini_declarations(&self) -> Vec<GeminiFunctionDeclaration> {
        self.definitions_slice()
            .iter()
            .map(ToolDefinition::to_gemini_declaration)
            .collect()
    }
}
