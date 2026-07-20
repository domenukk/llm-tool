//! Core types for `llm-tool`: tool output, errors, context, and definitions.
//!
//! Defines the types that tools produce ([`ToolOutput`], [`ToolError`]),
//! the execution context ([`ToolContext`]) shared across tool calls, and
//! the [`ToolDefinition`] metadata sent to models.

use alloc::{
    borrow::ToOwned,
    boxed::Box,
    format,
    string::{String, ToString},
    sync::Arc,
};
use core::any::{Any, TypeId};

use serde::{Deserialize, Serialize};

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
pub struct SharedState(Arc<RwLock<HashMap<String, serde_json::Value>>>);

impl SharedState {
    /// Create a new, empty shared state store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
/// Provides access to the current conversation ID and a shared key-value state
/// store that persists across tool calls within the same agent turn.
///
/// The state is backed by `Arc<RwLock<HashMap>>` so it can be cheaply cloned
/// and shared across concurrent tool invocations. Reads acquire a shared
/// lock; only writes take an exclusive lock.
///
/// # `get_state` vs `set_state` error handling
///
/// These two methods intentionally handle mutex poisoning differently:
///
/// - **[`get_state`](Self::get_state)** acquires a **read** lock and returns
///   the caller-supplied `default` when the lock is poisoned. Reads are
///   best-effort — a missing value is indistinguishable from a default, so
///   returning `default` keeps the tool running without surfacing
///   infrastructure errors to the model.
///
/// - **[`set_state`](Self::set_state)** acquires a **write** lock and returns
///   `Err` when the lock is poisoned. Writes that silently vanish can cause
///   subtle logic bugs, so callers must handle the failure explicitly.
/// # Typed extensions
///
/// In addition to the string-keyed JSON state, `ToolContext` supports
/// **typed extensions** via [`set_ext`](Self::set_ext) /
/// [`get_ext`](Self::get_ext). These use `std::any::Any` under the hood
/// and are keyed by `TypeId`, so callers store and retrieve strongly-typed
/// values (typically `Arc<T>`) without serialization.
///
/// ```rust
/// use std::sync::Arc;
///
/// use llm_tool::ToolContext;
///
/// struct MyState {
///     session_dir: String,
/// }
///
/// let ctx = ToolContext::new();
/// ctx.set_ext(Arc::new(MyState {
///     session_dir: "/tmp".into(),
/// }))
/// .unwrap();
///
/// let state: Arc<MyState> = ctx.get_ext::<Arc<MyState>>().unwrap();
/// assert_eq!(state.session_dir, "/tmp");
/// ```
pub struct ToolContext {
    conversation_id: Option<String>,
    state: SharedState,
    extensions: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
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

/// Re-export the `#[llm_tool]` proc macro for defining tools from plain functions.
///
/// # Usage
///
/// ```
/// use llm_tool::{RustTool, ToolContext, ToolRegistry, llm_tool};
///
/// /// Adds two numbers together (with a twist).
/// #[llm_tool]
/// fn wonky_add(
///     /// First number.
///     a: i64,
///     /// Second number.
///     b: i64,
/// ) -> Result<String, String> {
///     Ok(format!("{}", a + b + 1))
/// }
///
/// let mut registry = ToolRegistry::new();
/// registry.register(WonkyAdd);
/// assert_eq!(registry.definitions().len(), 1);
/// ```
pub use llm_tool_macros::{llm_prompt, llm_resource, llm_tool};
// Re-export `JsonSchema` derive so tool authors can write `use llm_tool::JsonSchema;`
// without adding `schemars` to their own `Cargo.toml`.
pub use schemars::JsonSchema;

/// Human-readable JSON type name for error messages.
const fn other_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// The return value of a Rust tool execution.
///
/// Every tool produces a `ToolOutput` containing:
/// - **`content`**: the text sent back to the model.
/// - **`metadata`**: an optional structured key-value map available to hooks,
///   policies, and logging pipelines — but **never** sent to the model.
///
/// # Ergonomics
///
/// `ToolOutput` implements `From<String>`, `From<&str>`, and `Display`, so
/// simple tools can return plain strings without ceremony:
///
/// ```rust
/// use llm_tool::ToolOutput;
/// use serde::Serialize;
///
/// // From a String
/// let out: ToolOutput = "hello".to_string().into();
/// assert_eq!(out.content(), "hello");
/// assert!(out.metadata().is_empty());
///
/// // From &str
/// let out: ToolOutput = "world".into();
/// assert_eq!(out.content(), "world");
///
/// // Structured metadata from a typed struct (preferred)
/// #[derive(Serialize)]
/// struct ReadMeta {
///     bytes_read: usize,
///     cached: bool,
/// }
///
/// let out = ToolOutput::new("file contents…")
///     .with_metadata(&ReadMeta {
///         bytes_read: 1024,
///         cached: true,
///     })
///     .unwrap();
/// assert_eq!(out.metadata()["bytes_read"], 1024);
/// assert_eq!(out.metadata()["cached"], true);
///
/// // Single ad-hoc entry
/// let out = ToolOutput::new("done").with_meta("exit_code", serde_json::json!(0));
/// assert_eq!(out.metadata()["exit_code"], 0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text content sent to the LLM.
    pub(crate) content: String,
    /// Structured metadata attached to the output.
    /// NOT sent to the LLM.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub(crate) metadata: HashMap<String, serde_json::Value>,
}

impl ToolOutput {
    /// Create a new `ToolOutput` with the given content and no metadata.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Serialize a value to JSON and wrap it as tool output.
    ///
    /// The JSON string becomes the content sent to the model, but no
    /// metadata is attached. For the zero-redundancy path that populates
    /// **both** content and metadata from the same struct, use
    /// [`from_metadata`](Self::from_metadata).
    ///
    /// ```rust
    /// use llm_tool::{ToolOutput, ToolError};
    ///
    /// let data = serde_json::json!({"temp": 72, "unit": "F"});
    /// let output = ToolOutput::json(&data).unwrap();
    /// assert!(output.content().contains("72"));
    /// assert!(output.metadata().is_empty()); // no metadata attached
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Err(ToolError)` if serialization fails.
    pub fn json<T: serde::Serialize>(value: &T) -> Result<Self, ToolError> {
        serde_json::to_string(value)
            .map(Self::new)
            .map_err(|e| ToolError::new(format!("serialization failed: {e}")))
    }

    /// Create a `ToolOutput` where **both** the content and metadata come
    /// from the same serializable value.
    ///
    /// - **Content** (sent to the model): the JSON representation of `value`.
    /// - **Metadata** (hooks / policies / logging): the flattened object fields.
    ///
    /// This is the zero-redundancy path: define one struct, derive
    /// `Serialize`, and everything is populated automatically.
    ///
    /// # Errors
    ///
    /// Returns `Err(ToolError)` if `value` doesn't serialize to a JSON object.
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_tool::ToolOutput;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct Weather {
    ///     location: String,
    ///     temp_f: i32,
    ///     condition: String,
    /// }
    ///
    /// let out = ToolOutput::from_metadata(&Weather {
    ///     location: "Seattle".into(),
    ///     temp_f: 72,
    ///     condition: "Sunny".into(),
    /// })
    /// .unwrap();
    ///
    /// // Model sees the JSON string
    /// assert!(out.content().contains("Seattle"));
    /// assert!(out.content().contains("72"));
    ///
    /// // Hooks see typed fields
    /// assert_eq!(out.metadata()["location"], "Seattle");
    /// assert_eq!(out.metadata()["temp_f"], 72);
    /// ```
    pub fn from_metadata<T: serde::Serialize>(value: &T) -> Result<Self, ToolError> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| ToolError::new(format!("metadata serialization failed: {e}")))?;
        // Serialize to string *before* destructuring so we borrow the Value
        // instead of cloning the inner Map.
        let content = json_value.to_string();
        match json_value {
            serde_json::Value::Object(map) => Ok(Self {
                content,
                metadata: map.into_iter().collect(),
            }),
            other => Err(ToolError::new(format!(
                "metadata must serialize to a JSON object, got {}",
                other_type_name(&other),
            ))),
        }
    }

    /// Attach a single metadata key-value pair. Chainable.
    ///
    /// For attaching multiple fields at once, prefer
    /// [`with_metadata`](Self::with_metadata) with a typed struct.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Attach structured metadata from a serializable value.
    ///
    /// The value is serialized to a JSON object and its fields are **merged**
    /// into the metadata map. This is the preferred way to attach metadata
    /// because it avoids stringly-typed keys and data duplication.
    ///
    /// # Errors
    ///
    /// Returns `Err(ToolError)` if `value` doesn't serialize to a JSON object
    /// (e.g. it serializes to a scalar or array).
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm_tool::ToolOutput;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct FileMeta {
    ///     bytes_read: usize,
    ///     source: String,
    /// }
    ///
    /// let out = ToolOutput::new("file contents")
    ///     .with_metadata(&FileMeta {
    ///         bytes_read: 1024,
    ///         source: "/etc/hosts".into(),
    ///     })
    ///     .unwrap();
    /// assert_eq!(out.metadata()["bytes_read"], 1024);
    /// assert_eq!(out.metadata()["source"], "/etc/hosts");
    /// ```
    pub fn with_metadata<T: serde::Serialize>(mut self, value: &T) -> Result<Self, ToolError> {
        let json = serde_json::to_value(value)
            .map_err(|e| ToolError::new(format!("metadata serialization failed: {e}")))?;
        match json {
            serde_json::Value::Object(map) => {
                self.metadata.extend(map);
                Ok(self)
            }
            other => Err(ToolError::new(format!(
                "metadata must serialize to a JSON object, got {}",
                other_type_name(&other),
            ))),
        }
    }

    /// The text content sent back to the model.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Consume self and return the owned content string.
    #[must_use]
    pub fn into_content(self) -> String {
        self.content
    }

    /// The structured metadata map.
    #[must_use]
    pub const fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }
}

impl core::fmt::Display for ToolOutput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.content)
    }
}

impl From<String> for ToolOutput {
    fn from(content: String) -> Self {
        Self::new(content)
    }
}

impl From<&str> for ToolOutput {
    fn from(content: &str) -> Self {
        Self::new(content)
    }
}

impl From<i64> for ToolOutput {
    fn from(value: i64) -> Self {
        Self::new(value.to_string())
    }
}

impl From<f64> for ToolOutput {
    fn from(value: f64) -> Self {
        Self::new(value.to_string())
    }
}

impl From<bool> for ToolOutput {
    fn from(value: bool) -> Self {
        Self::new(value.to_string())
    }
}

impl From<serde_json::Value> for ToolOutput {
    fn from(value: serde_json::Value) -> Self {
        // serde_json::Value::to_string() never fails.
        Self::new(value.to_string())
    }
}

/// Wrapper for returning serializable values as JSON tool output.
///
/// Implements `From<Json<T>> for ToolOutput` so it works with the
/// `#[llm_tool]` macro's `.into()` conversion — no `Result` wrapper needed
/// for infallible serialization.
///
/// # Panics
///
/// The `From` conversion panics if `serde_json::to_string` fails.
/// This only happens with broken `Serialize` implementations (e.g.,
/// maps with non-string keys). For explicit error handling, use
/// [`ToolOutput::json()`] instead.
///
/// # Example
///
/// ```rust
/// use llm_tool::{Json, ToolOutput};
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Weather {
///     temp: f64,
///     city: String,
/// }
///
/// let output: ToolOutput = Json(Weather {
///     temp: 72.0,
///     city: "NYC".into(),
/// })
/// .into();
/// assert!(output.content().contains("72"));
/// ```
pub struct Json<T>(pub T);

impl<T: serde::Serialize> From<Json<T>> for ToolOutput {
    fn from(json: Json<T>) -> Self {
        let json_value = serde_json::to_value(&json.0)
            .expect("Json<T> serialization failed — this is a bug in the Serialize impl");
        let content = json_value.to_string();
        match json_value {
            serde_json::Value::Object(map) => Self {
                content,
                metadata: map.into_iter().collect(),
            },
            _ => Self::new(content),
        }
    }
}

/// The category of a registered entity (tool, prompt, or resource).
///
/// Used by [`ToolError::not_found`] and the registry dispatchers to
/// provide clear, type-safe identification of which kind of item was
/// looked up.
///
/// # Example
///
/// ```
/// use llm_tool::RegistryItem;
///
/// assert_eq!(RegistryItem::Tool.to_string(), "tool");
/// assert_eq!(RegistryItem::Prompt.to_string(), "prompt");
/// assert_eq!(RegistryItem::Resource.to_string(), "resource");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegistryItem {
    /// An executable tool.
    Tool,
    /// A prompt template.
    Prompt,
    /// A readable resource.
    Resource,
}

impl core::fmt::Display for RegistryItem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Tool => write!(f, "tool"),
            Self::Prompt => write!(f, "prompt"),
            Self::Resource => write!(f, "resource"),
        }
    }
}

/// An error returned from a tool execution.
///
/// The error message is sent back to the model as the tool's error response.
/// Structured metadata can be attached for hooks and logging — it is **not**
/// sent to the model.
///
/// Implements `From<String>` and `From<&str>` for ergonomic construction.
///
/// # Example
///
/// ```
/// use llm_tool::ToolError;
/// use serde::Serialize;
///
/// let err: ToolError = "something went wrong".into();
/// assert_eq!(err.to_string(), "something went wrong");
///
/// let err = ToolError::new(format!("failed to read {}", "file.txt"));
/// assert!(err.to_string().contains("file.txt"));
///
/// // Structured metadata from a typed struct (preferred)
/// #[derive(Serialize)]
/// struct HttpErrorMeta {
///     status_code: u16,
///     url: String,
/// }
///
/// let err = ToolError::new("HTTP request failed")
///     .with_metadata(&HttpErrorMeta {
///         status_code: 503,
///         url: "https://example.com".into(),
///     })
///     .unwrap();
/// assert_eq!(err.metadata()["status_code"], 503);
///
/// // Single ad-hoc entry
/// let err = ToolError::new("timeout").with_meta("retry_after_secs", serde_json::json!(30));
/// assert_eq!(err.metadata()["retry_after_secs"], 30);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolError {
    /// Human-readable error message sent to the model.
    pub message: String,
    /// Structured metadata for hooks / policies / logging.
    /// NOT sent to the model.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, serde_json::Value>,
}

impl ToolError {
    /// Create a new tool error with no metadata.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            metadata: HashMap::new(),
        }
    }

    /// Metadata key identifying the structured category of an error.
    pub const ERROR_KIND_KEY: &'static str = "error_kind";
    /// [`ERROR_KIND_KEY`](Self::ERROR_KIND_KEY) value used by
    /// [`not_found`](Self::not_found) to mark a registry-lookup miss.
    pub const KIND_NOT_REGISTERED: &'static str = "not_registered";

    /// Construct an error for a registry lookup that found no entry named
    /// `name` of the given [`RegistryItem`].
    ///
    /// The human-readable message is suitable to hand back to the model so it
    /// can self-correct (e.g. retry with a valid name). An
    /// [`ERROR_KIND_KEY`](Self::ERROR_KIND_KEY) metadata field is attached —
    /// never sent to the model — so hosts can still distinguish a routing miss
    /// from a genuine execution failure for telemetry or policy decisions.
    /// Prefer [`is_not_found`](Self::is_not_found) over inspecting the metadata
    /// directly.
    ///
    /// # Example
    ///
    /// ```
    /// use llm_tool::{RegistryItem, ToolError};
    ///
    /// let err = ToolError::not_found(RegistryItem::Tool, "add_nummbers");
    /// assert!(err.to_string().contains("add_nummbers"));
    /// assert!(err.is_not_found());
    /// ```
    #[must_use]
    pub fn not_found(kind: RegistryItem, name: &str) -> Self {
        Self::new(format!("no {kind} named '{name}' is registered")).with_meta(
            Self::ERROR_KIND_KEY,
            serde_json::json!(Self::KIND_NOT_REGISTERED),
        )
    }

    /// Attach a single metadata key-value pair. Chainable.
    ///
    /// For attaching multiple fields at once, prefer
    /// [`with_metadata`](Self::with_metadata) with a typed struct.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Attach structured metadata from a serializable value.
    ///
    /// The value is serialized to a JSON object and its fields are **merged**
    /// into the metadata map. See [`ToolOutput::with_metadata`] for details.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if `value` doesn't serialize to a JSON object.
    pub fn with_metadata<T: serde::Serialize>(mut self, value: &T) -> Result<Self, Self> {
        let json = serde_json::to_value(value).map_err(|e| {
            Self::new(format!(
                "{} (metadata serialization also failed: {e})",
                self.message
            ))
        })?;
        match json {
            serde_json::Value::Object(map) => {
                self.metadata.extend(map);
                Ok(self)
            }
            other => Err(Self::new(format!(
                "{} (metadata must serialize to a JSON object, got {})",
                self.message,
                other_type_name(&other),
            ))),
        }
    }

    /// The structured metadata map.
    #[must_use]
    pub const fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }

    /// Whether this error denotes a registry-lookup miss, i.e. it was produced
    /// by [`not_found`](Self::not_found).
    ///
    /// Lets hosts branch on "the model asked for something that isn't
    /// registered" versus "a registered handler failed" without matching on
    /// message strings.
    ///
    /// # Example
    ///
    /// ```
    /// use llm_tool::{RegistryItem, ToolError};
    ///
    /// assert!(ToolError::not_found(RegistryItem::Prompt, "summarize").is_not_found());
    /// assert!(!ToolError::new("handler blew up").is_not_found());
    /// ```
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        self.metadata
            .get(Self::ERROR_KIND_KEY)
            .and_then(serde_json::Value::as_str)
            == Some(Self::KIND_NOT_REGISTERED)
    }
}

impl core::fmt::Display for ToolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl core::error::Error for ToolError {}

impl From<String> for ToolError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for ToolError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for ToolError {
    fn from(e: std::io::Error) -> Self {
        Self::new(e.to_string()).with_meta(
            Self::ERROR_KIND_KEY,
            serde_json::json!(format!("{:?}", e.kind())),
        )
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(e: serde_json::Error) -> Self {
        Self::new(e.to_string())
            .with_meta("category", serde_json::json!(format!("{:?}", e.classify())))
    }
}

impl From<Box<dyn core::error::Error + Send + Sync>> for ToolError {
    fn from(e: Box<dyn core::error::Error + Send + Sync>) -> Self {
        Self::new(e.to_string())
    }
}

impl From<core::convert::Infallible> for ToolError {
    fn from(never: core::convert::Infallible) -> Self {
        match never {}
    }
}

/// Describes a custom tool that can be registered with an agent.
///
/// This struct holds the metadata the SDK needs to expose the tool to the
/// model. The actual handler function is registered separately via
/// [`ToolRegistry::register`](super::ToolRegistry::register).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name (e.g. `"flash_device"`).
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameter_schema: serde_json::Value,
}

// ── Prompt types ────────────────────────────────────────────────────

/// Describes a prompt template available in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptDefinition {
    /// Prompt name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Arguments accepted by this prompt.
    #[serde(default, skip_serializing_if = "alloc::vec::Vec::is_empty")]
    pub arguments: alloc::vec::Vec<PromptArgumentDefinition>,
}

/// An argument accepted by a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptArgumentDefinition {
    /// Argument name.
    pub name: String,
    /// Argument description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Whether this argument is required.
    #[serde(default)]
    pub required: bool,
}

/// The role of a message in a prompt output (`user`, `assistant`, or `system`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptRole {
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// System message.
    System,
}

impl PromptRole {
    /// The string slice representation of the role (`"user"`, `"assistant"`, or `"system"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

impl core::fmt::Display for PromptRole {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A rendered message inside a prompt output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptOutputMessage {
    /// Message role.
    pub role: PromptRole,
    /// Text content.
    pub content: String,
}

impl PromptOutputMessage {
    /// Create a user role message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant role message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Assistant,
            content: content.into(),
        }
    }

    /// Create a system role message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::System,
            content: content.into(),
        }
    }
}

/// The output returned by rendering a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptOutput {
    /// Rendered messages.
    pub messages: alloc::vec::Vec<PromptOutputMessage>,
}

impl PromptOutput {
    /// Create a new prompt output with a single user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            messages: alloc::vec![PromptOutputMessage::user(content)],
        }
    }

    /// Create a new prompt output with a single assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            messages: alloc::vec![PromptOutputMessage::assistant(content)],
        }
    }

    /// Create a new prompt output with a single system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            messages: alloc::vec![PromptOutputMessage::system(content)],
        }
    }
}

impl From<String> for PromptOutput {
    fn from(content: String) -> Self {
        Self::user(content)
    }
}

impl From<&str> for PromptOutput {
    fn from(content: &str) -> Self {
        Self::user(content)
    }
}

// ── Resource types ──────────────────────────────────────────────────

/// Describes a resource or resource template available in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceDefinition {
    /// Resource URI (e.g. `file:///path` or `config://app`) or template pattern.
    #[serde(rename = "uriTemplate")]
    pub uri_template: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Optional MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A content block inside a resource read output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceOutputContent {
    /// UTF-8 text content.
    #[serde(rename_all = "camelCase")]
    Text {
        /// Resource URI.
        uri: String,
        /// Optional MIME type.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Text string.
        text: String,
    },
    /// Base64 binary content.
    #[serde(rename_all = "camelCase")]
    Blob {
        /// Resource URI.
        uri: String,
        /// Optional MIME type.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Base64 blob data.
        blob: String,
    },
}

/// The output returned by reading a resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceOutput {
    /// Returned content blocks.
    pub contents: alloc::vec::Vec<ResourceOutputContent>,
}

impl ResourceOutput {
    /// Create a text resource output.
    pub fn text(uri: impl Into<String>, mime_type: Option<&str>, text: impl Into<String>) -> Self {
        Self {
            contents: alloc::vec![ResourceOutputContent::Text {
                uri: uri.into(),
                mime_type: mime_type.map(ToString::to_string),
                text: text.into(),
            }],
        }
    }

    /// Create a binary blob resource output.
    pub fn blob(uri: impl Into<String>, mime_type: Option<&str>, blob: impl Into<String>) -> Self {
        Self {
            contents: alloc::vec![ResourceOutputContent::Blob {
                uri: uri.into(),
                mime_type: mime_type.map(ToString::to_string),
                blob: blob.into(),
            }],
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
