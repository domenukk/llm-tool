//! Core types for `llm-tool`: tool output, errors, context, and definitions.
//!
//! Defines the types that tools produce ([`ToolOutput`], [`ToolError`]),
//! the execution context ([`ToolContext`]) shared across tool calls, and
//! the [`ToolDefinition`] metadata sent to models.

use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
};

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
use serde::{Deserialize, Serialize};

use crate::compat::HashMap;

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

    /// Truncate the output text content according to the given [`TruncationPolicy`].
    #[must_use]
    pub fn truncated(mut self, policy: &TruncationPolicy) -> Self {
        truncation::truncate_output(&mut self, policy);
        self
    }

    /// Truncate the output text content according to the given [`TruncationPolicy`],
    /// returning a detailed [`TruncationOutcome`].
    ///
    /// # Errors
    ///
    /// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
    pub fn truncate_detailed(
        &mut self,
        policy: &TruncationPolicy,
    ) -> Result<TruncationOutcome<'static>, SpillError> {
        truncation::truncate_output_detailed(self, policy)
    }

    /// Truncate the output text content with caller context according to [`TruncationPolicy`].
    ///
    /// # Errors
    ///
    /// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
    pub fn truncate_detailed_with_context(
        &mut self,
        policy: &TruncationPolicy,
        ctx: &SpillContext<'_>,
    ) -> Result<TruncationOutcome<'static>, SpillError> {
        truncation::truncate_output_detailed_with_context(self, policy, ctx)
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
    /// [`ERROR_KIND_KEY`](Self::ERROR_KIND_KEY) value used by
    /// [`destructive_in_parallel_batch`](Self::destructive_in_parallel_batch) when a
    /// destructive tool is rejected during parallel batch execution.
    pub const KIND_DESTRUCTIVE_BATCH: &'static str = "destructive_in_parallel_batch";

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

    /// Construct a typed error when a tool with [`ToolEffect::Destructive`] is rejected
    /// in unconfirmed parallel batch dispatch.
    #[must_use]
    pub fn destructive_in_parallel_batch(tool_name: &str) -> Self {
        Self::new(format!(
            "Tool '{tool_name}' has Destructive effect and cannot be executed in Parallel batch mode without explicit confirmation"
        ))
        .with_meta(
            Self::ERROR_KIND_KEY,
            serde_json::json!(Self::KIND_DESTRUCTIVE_BATCH),
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
        let json = match serde_json::to_value(value) {
            Ok(j) => j,
            Err(e) => {
                self.message =
                    format!("{} (metadata serialization also failed: {e})", self.message);
                return Err(self);
            }
        };
        match json {
            serde_json::Value::Object(map) => {
                self.metadata.extend(map);
                Ok(self)
            }
            other => {
                self.message = format!(
                    "{} (metadata must serialize to a JSON object, got {})",
                    self.message,
                    other_type_name(&other),
                );
                Err(self)
            }
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

    /// Whether this error denotes a [`ToolEffect::Destructive`] tool rejected
    /// during parallel batch dispatch.
    #[must_use]
    pub fn is_destructive_batch_rejected(&self) -> bool {
        self.metadata
            .get(Self::ERROR_KIND_KEY)
            .and_then(serde_json::Value::as_str)
            == Some(Self::KIND_DESTRUCTIVE_BATCH)
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

pub mod context;
pub mod definition;
pub mod prompt;
pub mod quarantine;
pub mod resource;
pub mod sanitized;
pub mod truncation;

pub use context::*;
pub use definition::*;
pub use prompt::*;
pub use quarantine::*;
pub use resource::*;
pub use sanitized::*;
pub use truncation::*;

#[cfg(all(test, feature = "std"))]
mod tests;
