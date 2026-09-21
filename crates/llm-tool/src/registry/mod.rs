//! Tool registry: registry and concurrent dispatch of named tools.

mod batch;

use alloc::{boxed::Box, sync::Arc, vec::Vec};

pub use batch::{
    AllowDestructiveBatch, BatchExecutionMode, BatchToolCall, BatchToolCallOutcome,
    BatchToolCallStr,
};

use super::{
    rust_tool::{ErasedTool, RustTool, definition_of},
    types::{RegistryItem, ToolContext, ToolDefinition, ToolError, ToolOutput},
};
use crate::{
    compat::{HashMap, RwLock, read_lock_recover, write_lock_recover},
    rust_tool::BoxToolFuture,
    types::ToolEffect,
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

/// A registry of named tools available for dynamic dispatch.
///
/// Holds type-erased tool implementations and cached [`ToolDefinition`](super::types::ToolDefinition)
/// schemas for fast lookup and execution.
pub struct ToolRegistry {
    tools: HashMap<&'static str, RegisteredTool>,
    cached_definitions: RwLock<Option<Arc<[ToolDefinition]>>>,
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
            .finish_non_exhaustive()
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
            cached_definitions: RwLock::new(None),
        }
    }

    fn invalidate_cache(&self) {
        let mut guard = write_lock_recover(&self.cached_definitions);
        *guard = None;
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
    /// bug in the tool's `Params` type (e.g. a broken `JsonSchema` impl). Use
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
    /// This is the fallible counterpart to [`register`](Self::register); prefer
    /// it when tool types are supplied dynamically and a broken `JsonSchema`
    /// impl should not abort the process. If a tool with the same name was
    /// already registered, it is replaced.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the tool's `Params` type fails to produce a
    /// JSON schema.
    pub fn try_register<T: RustTool + 'static>(&mut self, tool: T) -> Result<&mut Self, ToolError> {
        let definition = definition_of(&tool)?;
        self.tools.insert(
            T::NAME,
            RegisteredTool {
                definition,
                erased: Box::new(tool),
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
    ///
    /// Returns clones of the cached definitions computed at registration time.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions_slice().to_vec()
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
        let Some(entry) = self.tools.get(name) else {
            return Err(ToolError::not_found(RegistryItem::Tool, name));
        };
        Ok(entry.erased.call_erased(args, ctx))
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
        let Some(entry) = self.tools.get(name) else {
            return Err(ToolError::not_found(RegistryItem::Tool, name));
        };
        Ok(entry.erased.call_erased_str(args_json, ctx))
    }

    /// Dispatch a tool call by name with raw JSON arguments and a context.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no tool named `name` is registered
    /// (carrying `error_kind = "not_registered"` metadata), or the tool's own
    /// error if argument deserialization fails or the handler returns an error.
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
    ///
    /// Returns `None` if no tool named `name` is registered. Unlike
    /// [`definitions`](Self::definitions), this clones nothing.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name).map(|entry| &entry.definition)
    }

    /// Iterate over `(name, definition)` pairs for every registered tool.
    ///
    /// Yields clones of the cached definitions computed at registration time.
    #[must_use]
    pub fn iter(&self) -> ToolDefinitions<'_> {
        ToolDefinitions {
            inner: self.tools.iter(),
        }
    }
}

/// Borrowing iterator over `(name, definition)` pairs, yielded by
/// [`ToolRegistry::iter`] and by `&ToolRegistry`'s [`IntoIterator`] impl.
///
/// Unlike a boxed trait object, this named iterator allocates nothing to
/// construct and forwards `size_hint`/`len` from the underlying map iterator.
/// Each cached [`ToolDefinition`] is cloned lazily as it is yielded.
pub struct ToolDefinitions<'a> {
    inner: crate::compat::HashMapIter<'a, &'static str, RegisteredTool>,
}

impl Iterator for ToolDefinitions<'_> {
    type Item = (&'static str, ToolDefinition);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(name, entry)| (*name, entry.definition.clone()))
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

/// Iterate over `(name, definition)` pairs for every registered tool.
///
/// Yields `(&'static str, ToolDefinition)` for each tool in the registry.
impl<'a> IntoIterator for &'a ToolRegistry {
    type Item = (&'static str, ToolDefinition);
    type IntoIter = ToolDefinitions<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
