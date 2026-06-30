//! Tool registry: registry and concurrent dispatch of named tools.

use alloc::{boxed::Box, format, vec::Vec};

use super::{
    rust_tool::{ErasedTool, RustTool, definition_of},
    types::{ToolContext, ToolDefinition, ToolError, ToolOutput},
};
use crate::compat::HashMap;

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

    /// Dispatch a tool call by name with a raw JSON string argument.
    ///
    /// # Errors
    ///
    /// Returns `Err` if JSON parsing fails, the tool name is unknown, or the handler fails.
    pub async fn dispatch_str(
        &self,
        name: &str,
        args_json: &str,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let args = serde_json::from_str(args_json)
            .map_err(|e| ToolError::new(format!("Malformed JSON arguments: {e}")))?;
        self.dispatch(name, args, ctx).await
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

#[cfg(all(test, feature = "std"))]
mod tests;
