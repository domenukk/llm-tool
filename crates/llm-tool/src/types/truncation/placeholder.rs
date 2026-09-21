//! Placeholder definitions and substitution tokens for marker templates.

/// Named placeholders supported in custom marker templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerPlaceholder {
    /// Number of bytes omitted from the middle (`"{omitted_bytes}"`).
    OmittedBytes,
    /// Total byte length of the original un-truncated input (`"{original_bytes}"`).
    OriginalBytes,
    /// Starting byte offset in original input where truncation began (`"{offset_start}"`).
    OffsetStart,
    /// Ending byte offset in original input where truncation ended (`"{offset_end}"`).
    OffsetEnd,
    /// Typed reference handle for recall (`"{ref}"`).
    SpillRef,
}

impl MarkerPlaceholder {
    /// Canonical token string for [`MarkerPlaceholder::OmittedBytes`].
    pub const TOKEN_OMITTED_BYTES: &'static str = "{omitted_bytes}";
    /// Canonical token string for [`MarkerPlaceholder::OriginalBytes`].
    pub const TOKEN_ORIGINAL_BYTES: &'static str = "{original_bytes}";
    /// Canonical token string for [`MarkerPlaceholder::OffsetStart`].
    pub const TOKEN_OFFSET_START: &'static str = "{offset_start}";
    /// Canonical token string for [`MarkerPlaceholder::OffsetEnd`].
    pub const TOKEN_OFFSET_END: &'static str = "{offset_end}";
    /// Canonical token string for [`MarkerPlaceholder::SpillRef`].
    pub const TOKEN_SPILL_REF: &'static str = "{ref}";

    /// Legacy deprecated placeholder `{count}` (alias of `{omitted_bytes}`).
    pub const DEPRECATED_COUNT: &'static str = "{count}";
    /// Legacy deprecated placeholder `{bytes}` (alias of `{omitted_bytes}`).
    pub const DEPRECATED_BYTES: &'static str = "{bytes}";
    /// Legacy deprecated placeholder `<N>` (alias of `{omitted_bytes}`).
    pub const DEPRECATED_N: &'static str = "<N>";

    /// Return the canonical placeholder token string.
    #[must_use]
    pub const fn token(&self) -> &'static str {
        match self {
            Self::OmittedBytes => Self::TOKEN_OMITTED_BYTES,
            Self::OriginalBytes => Self::TOKEN_ORIGINAL_BYTES,
            Self::OffsetStart => Self::TOKEN_OFFSET_START,
            Self::OffsetEnd => Self::TOKEN_OFFSET_END,
            Self::SpillRef => Self::TOKEN_SPILL_REF,
        }
    }
}
