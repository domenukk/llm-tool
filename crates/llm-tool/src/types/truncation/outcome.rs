//! Detailed truncation outcome capturing metadata and recall references.

use alloc::borrow::Cow;

use super::sink::SpillRef;

/// Detailed outcome of a middle-out string truncation operation.
///
/// Preserves metadata including original length, omitted byte range, and
/// an optional [`SpillRef`] handle if an [`crate::OutputSpillSink`] was configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationOutcome<'a> {
    /// The truncated or original text.
    pub text: Cow<'a, str>,
    /// Byte length of the original input before truncation.
    pub original_len: usize,
    /// Byte range of the ORIGINAL input that was omitted.
    ///
    /// `None` when untruncated.
    pub omitted: Option<core::ops::Range<usize>>,
    /// Pluggable spill reference handle if full output was retained by an [`crate::OutputSpillSink`].
    pub spill_ref: Option<SpillRef>,
}

impl<'a> TruncationOutcome<'a> {
    /// Returns `true` if content was omitted during truncation.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.omitted.is_some()
    }

    /// Number of bytes omitted from the original input.
    #[must_use]
    pub const fn omitted_bytes(&self) -> usize {
        match &self.omitted {
            Some(range) => range.end.saturating_sub(range.start),
            None => 0,
        }
    }

    /// Consume this outcome and return the resulting text.
    #[must_use]
    pub fn into_text(self) -> Cow<'a, str> {
        self.text
    }
}
