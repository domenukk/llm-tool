//! Execution context and shared state store passed to Rust tools during dispatch.

use alloc::{borrow::ToOwned, boxed::Box, format, string::String, sync::Arc};
use core::any::{Any, TypeId};

use super::ToolError;
use crate::compat::{HashMap, RwLock, read_lock, write_lock};

/// A cheaply-cloneable handle to a tool's shared key-value state store.
///
/// [`ToolContext`] keeps its state behind this handle so multiple contexts
/// (e.g. successive tool calls within the same agent turn) can read and write
/// the **same** underlying store. Clone it and hand it to another context via
/// [`ToolContext::with_shared_state`].
///
/// The concrete lock and map types are an implementation detail — they differ
/// between `std` and `no_std` builds — so they are deliberately not exposed.
#[derive(Clone)]
pub struct SharedState(pub(super) Arc<RwLock<HashMap<String, serde_json::Value>>>);

impl SharedState {
    /// Create a new, empty shared state store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve a value from the shared state, returning `default` if the key
    /// is absent or the lock is poisoned.
    #[must_use]
    pub fn get_state(&self, key: &str, default: serde_json::Value) -> serde_json::Value {
        match read_lock(&self.0) {
            Ok(guard) => guard.get(key).cloned().unwrap_or(default),
            Err(e) => {
                tracing::warn!(key, error = %e, "SharedState::get_state: lock poisoned, returning default");
                default
            }
        }
    }

    /// Insert or update a value in the shared state.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the lock is poisoned.
    pub fn set_state(&self, key: &str, value: serde_json::Value) -> Result<(), ToolError> {
        match write_lock(&self.0) {
            Ok(mut guard) => {
                guard.insert(key.to_owned(), value);
                Ok(())
            }
            Err(e) => {
                let msg = format!("SharedState::set_state: lock poisoned for key '{key}': {e}");
                tracing::warn!("{msg}");
                Err(ToolError::new(msg))
            }
        }
    }

    /// Remove a value from the shared state by key, returning `true` if it was present.
    #[must_use]
    pub fn remove_state(&self, key: &str) -> bool {
        match write_lock(&self.0) {
            Ok(mut guard) => guard.remove(key).is_some(),
            Err(e) => {
                tracing::warn!(key, error = %e, "SharedState::remove_state: lock poisoned");
                false
            }
        }
    }

    /// Clear all entries from the shared state.
    pub fn clear_state(&self) {
        match write_lock(&self.0) {
            Ok(mut guard) => guard.clear(),
            Err(e) => {
                tracing::warn!(error = %e, "SharedState::clear_state: lock poisoned");
            }
        }
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}

impl core::fmt::Debug for SharedState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SharedState").finish_non_exhaustive()
    }
}

/// Context passed to Rust tools during dispatch.
///
/// Provides access to the current conversation ID, a shared key-value state
/// store ([`SharedState`]) that persists across tool calls within the same
/// agent turn, and typed extensions via [`set_ext`](Self::set_ext) / [`get_ext`](Self::get_ext).
pub struct ToolContext {
    conversation_id: Option<String>,
    pub(super) state: SharedState,
    pub(super) extensions: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

impl Clone for ToolContext {
    fn clone(&self) -> Self {
        Self {
            conversation_id: self.conversation_id.clone(),
            state: self.state.clone(),
            extensions: Arc::clone(&self.extensions),
        }
    }
}

impl core::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let count = match read_lock(&self.extensions) {
            Ok(g) => g.len(),
            Err(e) => {
                tracing::warn!(error = %e, "ToolContext::fmt: extensions lock poisoned");
                0
            }
        };
        f.debug_struct("ToolContext")
            .field("conversation_id", &self.conversation_id)
            .field("state", &self.state)
            .field("extensions_count", &count)
            .finish()
    }
}

impl Default for ToolContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolContext {
    /// Create a new, empty context: no conversation ID and a fresh state store.
    ///
    /// Customize it with [`with_conversation_id`](Self::with_conversation_id)
    /// and [`with_shared_state`](Self::with_shared_state).
    #[must_use]
    pub fn new() -> Self {
        Self {
            conversation_id: None,
            state: SharedState::new(),
            extensions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set the conversation ID. Chainable.
    #[must_use]
    pub fn with_conversation_id(mut self, conversation_id: impl Into<String>) -> Self {
        self.conversation_id = Some(conversation_id.into());
        self
    }

    /// Derive a new context that carries a different conversation ID while
    /// **sharing** this context's state store and typed extensions.
    ///
    /// Both the shared state ([`SharedState`]) and the typed extension map are
    /// held behind `Arc`, so the returned context reads and writes the *same*
    /// underlying stores — only the conversation identity differs. This is the
    /// primitive a single MCP server uses to serve **many** callers: each
    /// connection derives its own identity from the shared, session-wide
    /// context without duplicating extensions like injected session state.
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use llm_tool::ToolContext;
    ///
    /// let session = ToolContext::new().with_conversation_id("server");
    /// session.set_ext(Arc::new(42u64)).unwrap();
    ///
    /// let alice = session.with_caller("alice");
    /// assert_eq!(alice.conversation_id(), Some("alice"));
    /// // Extensions are shared, not copied.
    /// assert_eq!(alice.get_ext::<Arc<u64>>().as_deref(), Some(&42));
    /// ```
    #[must_use]
    pub fn with_caller(&self, conversation_id: impl Into<String>) -> Self {
        Self {
            conversation_id: Some(conversation_id.into()),
            state: self.state.clone(),
            extensions: Arc::clone(&self.extensions),
        }
    }

    /// Use an externally-provided [`SharedState`] as this context's state store.
    ///
    /// Use this when multiple `ToolContext` instances (e.g. successive tool
    /// calls within the same agent) must read/write the **same** state store.
    /// Obtain a handle from an existing context via
    /// [`shared_state`](Self::shared_state).
    #[must_use]
    pub fn with_shared_state(mut self, state: SharedState) -> Self {
        self.state = state;
        self
    }

    /// Return a cloneable handle to this context's shared state store.
    ///
    /// Pass the returned handle to [`with_shared_state`](Self::with_shared_state)
    /// on another context to share the same underlying store.
    #[must_use]
    pub fn shared_state(&self) -> SharedState {
        self.state.clone()
    }

    /// Return the conversation ID, if one has been set.
    #[must_use]
    pub fn conversation_id(&self) -> Option<&str> {
        self.conversation_id.as_deref()
    }

    /// Retrieve a value from the shared state, returning `default` if the key
    /// is absent or the lock is poisoned.
    ///
    /// This method never fails — on a poisoned lock it logs a warning and
    /// returns `default`. See the [struct-level docs](Self) for rationale.
    #[must_use]
    pub fn get_state(&self, key: &str, default: serde_json::Value) -> serde_json::Value {
        match read_lock(&self.state.0) {
            Ok(guard) => guard.get(key).cloned().unwrap_or(default),
            Err(e) => {
                tracing::warn!(key, error = %e, "ToolContext::get_state: lock poisoned, returning default");
                default
            }
        }
    }

    /// Insert or update a value in the shared state.
    ///
    /// Unlike [`get_state`](Self::get_state), this method returns `Err` on a
    /// poisoned lock because silently dropping a write can cause subtle bugs.
    /// See the [struct-level docs](Self) for rationale.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`]
    /// if the lock is poisoned.
    pub fn set_state(&self, key: &str, value: serde_json::Value) -> Result<(), ToolError> {
        match write_lock(&self.state.0) {
            Ok(mut guard) => {
                guard.insert(key.to_owned(), value);
                Ok(())
            }
            Err(e) => {
                let msg = format!("ToolContext::set_state: lock poisoned for key '{key}': {e}");
                tracing::warn!("{msg}");
                Err(ToolError::new(msg))
            }
        }
    }

    /// Store a typed value in the extensions map.
    ///
    /// Values are keyed by `TypeId`, so each concrete type can only appear
    /// once. Typically used to store `Arc<T>` for shared, cloneable access.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the lock is poisoned.
    pub fn set_ext<T: Send + Sync + 'static>(&self, value: T) -> Result<(), ToolError> {
        match write_lock(&self.extensions) {
            Ok(mut exts) => {
                exts.insert(TypeId::of::<T>(), Box::new(value));
                Ok(())
            }
            Err(e) => {
                let msg = format!(
                    "ToolContext::set_ext: lock poisoned for type '{}': {e}",
                    core::any::type_name::<T>()
                );
                tracing::warn!("{msg}");
                Err(ToolError::new(msg))
            }
        }
    }

    /// Retrieve a clone of a typed value from the extensions map.
    ///
    /// Returns `None` if no value of type `T` has been stored via
    /// [`set_ext`](Self::set_ext).
    #[must_use]
    pub fn get_ext<T: Clone + Send + Sync + 'static>(&self) -> Option<T> {
        match read_lock(&self.extensions) {
            Ok(exts) => exts
                .get(&TypeId::of::<T>())
                .and_then(|v| v.downcast_ref::<T>())
                .cloned(),
            Err(e) => {
                tracing::warn!(error = %e, "ToolContext::get_ext: lock poisoned, returning None");
                None
            }
        }
    }
}
