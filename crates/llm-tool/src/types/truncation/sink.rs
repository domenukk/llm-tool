//! Pluggable observation spill sink and contextual handle types.

use alloc::string::String;

/// Typed identifier/handle for full tool output spilled to an external observation store.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct SpillRef(String);

impl SpillRef {
    /// Create a new typed spill reference from a string or identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the string slice of the spill reference.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for SpillRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::ops::Deref for SpillRef {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for SpillRef {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<String> for SpillRef {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SpillRef {
    fn from(s: &str) -> Self {
        Self(s.into())
    }
}

/// Contextual metadata provided to an [`OutputSpillSink`] during a spill operation.
///
/// Keeps `llm-tool` transport-agnostic and free from domain concepts like
/// conversations, provenance, or file systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillContext<'a> {
    /// Name of the tool whose output is being truncated.
    pub tool_name: &'a str,
    /// Optional caller-supplied opaque tag or provenance token.
    pub opaque_tag: Option<&'a str>,
}

impl<'a> SpillContext<'a> {
    /// Anonymous context with default tool name.
    pub const ANONYMOUS: Self = Self {
        tool_name: "anonymous",
        opaque_tag: None,
    };

    /// Create a new spill context for a specific tool.
    #[must_use]
    pub const fn new(tool_name: &'a str) -> Self {
        Self {
            tool_name,
            opaque_tag: None,
        }
    }

    /// Attach an opaque provenance or session tag.
    #[must_use]
    pub const fn with_tag(mut self, opaque_tag: &'a str) -> Self {
        self.opaque_tag = Some(opaque_tag);
        self
    }
}

/// Strongly typed error returned when persisting tool output to an [`OutputSpillSink`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpillError {
    /// The storage backend or buffer capacity was exceeded.
    CapacityExceeded {
        /// Configured limit in bytes.
        limit: usize,
        /// Attempted spill size in bytes.
        attempted: usize,
    },
    /// Storage I/O or persistence failure.
    StorageFailed(String),
    /// The spill operation timed out.
    Timeout,
    /// The spill operation was rejected by policy or authorization.
    Rejected(String),
    /// Other backend-specific error.
    Other(String),
}

impl core::fmt::Display for SpillError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CapacityExceeded { limit, attempted } => {
                write!(
                    f,
                    "spill capacity exceeded: attempted {attempted} bytes with limit of {limit} bytes"
                )
            }
            Self::StorageFailed(msg) => write!(f, "spill storage failed: {msg}"),
            Self::Timeout => write!(f, "spill operation timed out"),
            Self::Rejected(msg) => write!(f, "spill rejected: {msg}"),
            Self::Other(msg) => write!(f, "spill error: {msg}"),
        }
    }
}

impl core::error::Error for SpillError {}

/// Resolved outcome of attempting to spill an oversized observation.
///
/// Truncation must never silently lose the fact that full-output retention failed.
/// This enum makes the three possible states explicit so the marker renderer can
/// distinguish "no sink configured" (nothing was promised) from "sink failed"
/// (retention was promised and did not happen), and surface the latter in-band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpillDisposition {
    /// No [`OutputSpillSink`] was configured; omitted bytes were never retained.
    NotConfigured,
    /// The full output was retained and is recallable via the returned handle.
    Spilled(SpillRef),
    /// A sink was configured but failed; the omitted bytes are unrecoverable.
    Failed(SpillError),
}

impl SpillDisposition {
    /// Return the spill handle when retention succeeded.
    #[must_use]
    pub const fn spill_ref(&self) -> Option<&SpillRef> {
        match self {
            Self::Spilled(handle) => Some(handle),
            Self::NotConfigured | Self::Failed(_) => None,
        }
    }

    /// Return the spill error when a configured sink failed.
    #[must_use]
    pub const fn error(&self) -> Option<&SpillError> {
        match self {
            Self::Failed(error) => Some(error),
            Self::NotConfigured | Self::Spilled(_) => None,
        }
    }
}

/// Pluggable sink for retaining full un-truncated tool output observations.
///
/// Implementations may store output in memory, write to an artifact store,
/// or upload to a remote recall cache. When configured on a [`crate::TruncationPolicy`],
/// output exceeding `max_bytes` is first saved via [`OutputSpillSink::spill`]
/// before middle-out truncation occurs.
pub trait OutputSpillSink: Send + Sync + core::fmt::Debug {
    /// Spill the full tool return content, returning a recallable [`SpillRef`].
    ///
    /// # Errors
    ///
    /// Returns [`SpillError`] if storage fails, capacity is exceeded, or the
    /// operation is rejected.
    fn spill(&self, ctx: &SpillContext<'_>, full: &str) -> Result<SpillRef, SpillError>;
}
