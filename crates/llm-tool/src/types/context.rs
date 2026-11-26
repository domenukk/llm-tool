//! Execution context and shared state store passed to Rust tools during dispatch.

use alloc::{borrow::ToOwned, boxed::Box, format, string::String, sync::Arc, vec::Vec};
use core::any::{Any, TypeId};

use super::{SurrogateHandle, SurrogateVault, ToolError};
use crate::{
    compat::{HashMap, RwLock, read_lock, write_lock},
    registry::AllowDestructiveBatch,
};

type ExtensionMap = Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>;

/// Returns `true` for security-sensitive mutable extensions that must never be inherited
/// across isolated connection forks ([`ToolContext::fork_connection`]).
fn is_connection_isolated_extension<T: 'static>() -> bool {
    let tid = TypeId::of::<T>();
    tid == TypeId::of::<SurrogateVault>() || tid == TypeId::of::<AllowDestructiveBatch>()
}

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

    /// Create an independent [`SharedState`] initialized with a snapshot clone of this
    /// store's current entries.
    ///
    /// Initial server-configured defaults are preserved in the returned store, while
    /// subsequent [`set_state`](Self::set_state), [`remove_state`](Self::remove_state),
    /// or [`clear_state`](Self::clear_state) mutations on either store remain isolated.
    #[must_use]
    pub fn fork_snapshot(&self) -> Self {
        let snapshot = match read_lock(&self.0) {
            Ok(guard) => guard.clone(),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "SharedState::fork_snapshot: lock poisoned, initializing empty snapshot"
                );
                HashMap::new()
            }
        };
        Self(Arc::new(RwLock::new(snapshot)))
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
    pub(super) extensions: ExtensionMap,
    /// Read-only ancestor extension layers inherited on [`fork_connection`](Self::fork_connection)
    /// (excluding connection-isolated extensions such as [`SurrogateVault`] and [`AllowDestructiveBatch`]).
    pub(super) parent_extensions: Vec<ExtensionMap>,
}

impl Clone for ToolContext {
    fn clone(&self) -> Self {
        Self {
            conversation_id: self.conversation_id.clone(),
            state: self.state.clone(),
            extensions: Arc::clone(&self.extensions),
            parent_extensions: self.parent_extensions.clone(),
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
            .field("parent_layers", &self.parent_extensions.len())
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
            parent_extensions: Vec::new(),
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
    /// For multi-tenant servers accepting untrusted peer connections where
    /// mutable extensions ([`SurrogateVault`], [`AllowDestructiveBatch`]) must
    /// remain isolated per connection, prefer [`fork_connection`](Self::fork_connection).
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
            parent_extensions: self.parent_extensions.clone(),
        }
    }

    /// Fork an isolated per-connection context that inherits read-only server extensions
    /// (such as injected `Arc<Config>` or session handles) and a snapshot of initial
    /// [`SharedState`] defaults while isolating connection-local [`set_state`](Self::set_state)
    /// and [`set_ext`](Self::set_ext) writes, [`SurrogateVault`], and [`AllowDestructiveBatch`].
    #[must_use]
    pub fn fork_connection(&self, caller_override: Option<&str>) -> Self {
        let mut parents = Vec::with_capacity(self.parent_extensions.len() + 1);
        parents.push(Arc::clone(&self.extensions));
        parents.extend(self.parent_extensions.iter().cloned());
        let conversation_id = caller_override
            .map(ToOwned::to_owned)
            .or_else(|| self.conversation_id.clone());
        Self {
            conversation_id,
            state: self.state.fork_snapshot(),
            extensions: Arc::new(RwLock::new(HashMap::new())),
            parent_extensions: parents,
        }
    }

    /// Use an externally-provided [`SharedState`] as this context's state store.
    #[must_use]
    pub fn with_shared_state(mut self, state: SharedState) -> Self {
        self.state = state;
        self
    }

    /// Return a cloneable handle to this context's shared state store.
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
    /// # Errors
    ///
    /// Returns [`ToolError`] if the lock is poisoned.
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

    /// Store a typed value in this context's local extensions map.
    ///
    /// Values are keyed by `TypeId`, so each concrete type can only appear
    /// once per layer. Typically used to store `Arc<T>` for shared, cloneable access.
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
    /// Checks this context's own extension map first, then any inherited parent maps
    /// (unless `T` is a connection-isolated type such as [`SurrogateVault`] or
    /// [`AllowDestructiveBatch`]).
    #[must_use]
    pub fn get_ext<T: Clone + Send + Sync + 'static>(&self) -> Option<T> {
        let tid = TypeId::of::<T>();
        match read_lock(&self.extensions) {
            Ok(exts) => {
                if let Some(found) = exts.get(&tid).and_then(|v| v.downcast_ref::<T>()).cloned() {
                    return Some(found);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "ToolContext::get_ext: lock poisoned, returning None");
                return None;
            }
        }
        if is_connection_isolated_extension::<T>() {
            return None;
        }
        for parent in &self.parent_extensions {
            match read_lock(parent) {
                Ok(exts) => {
                    if let Some(found) = exts.get(&tid).and_then(|v| v.downcast_ref::<T>()).cloned()
                    {
                        return Some(found);
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "ToolContext::get_ext: parent lock poisoned");
                }
            }
        }
        None
    }

    /// Attach a pre-constructed [`SurrogateVault`] to this context. Chainable.
    #[must_use]
    pub fn with_surrogate_vault(self, vault: SurrogateVault) -> Self {
        match write_lock(&self.extensions) {
            Ok(mut exts) => {
                exts.insert(TypeId::of::<SurrogateVault>(), Box::new(vault));
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "ToolContext::with_surrogate_vault: extensions lock poisoned"
                );
            }
        }
        self
    }

    /// Obtain the session's [`SurrogateVault`], lazily initializing and storing one in
    /// [`extensions`](Self::get_ext) if not yet present.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the extensions lock is poisoned.
    pub fn surrogate_vault(&self) -> Result<SurrogateVault, ToolError> {
        if let Some(existing) = self.get_ext::<SurrogateVault>() {
            return Ok(existing);
        }
        let mut exts = write_lock(&self.extensions).map_err(|e| {
            let msg = format!("ToolContext::surrogate_vault: extensions lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;
        if let Some(existing) = exts
            .get(&TypeId::of::<SurrogateVault>())
            .and_then(|v| v.downcast_ref::<SurrogateVault>())
        {
            return Ok(existing.clone());
        }
        let vault = SurrogateVault::new();
        exts.insert(TypeId::of::<SurrogateVault>(), Box::new(vault.clone()));
        drop(exts);
        Ok(vault)
    }

    /// Mint an opaque [`SurrogateHandle`] (`<surrogate:kind:nonce>`) in this context's [`SurrogateVault`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `kind` or `raw_value` is invalid, capacity is exceeded, the vault
    /// has no secure seed (see [`SurrogateVault::new`]), or a lock is poisoned.
    pub fn mint_surrogate(
        &self,
        kind: &str,
        raw_value: impl Into<String>,
    ) -> Result<SurrogateHandle, ToolError> {
        self.surrogate_vault()?.mint(kind, raw_value)
    }

    /// Resolve an exact `<surrogate:kind:nonce>` token bound to `expected_kind` using this context's [`SurrogateVault`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `token` is malformed, unknown, kind-mismatched, or a lock is poisoned.
    pub fn resolve_surrogate_of_kind(
        &self,
        token: &str,
        expected_kind: &str,
    ) -> Result<String, ToolError> {
        self.surrogate_vault()?
            .resolve_token_of_kind(token, expected_kind)
    }

    /// Resolve all `<surrogate:kind:nonce>` tokens in `input` using this context's [`SurrogateVault`].
    ///
    /// Fails closed with [`ToolError`] if any surrogate token in `input` was not minted in this session.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if an unknown/malformed surrogate token is present or a lock is poisoned.
    pub fn resolve_surrogates(&self, input: &str) -> Result<String, ToolError> {
        self.surrogate_vault()?.resolve_in_str(input)
    }

    /// Redact any registered raw secret/handle values in `output` back to their `<surrogate:kind:nonce>` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if a lock is poisoned.
    pub fn redact_surrogates(&self, output: &str) -> Result<String, ToolError> {
        self.surrogate_vault()?.redact_str(output)
    }
}
