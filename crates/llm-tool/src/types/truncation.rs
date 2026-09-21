//! Middle-out truncation defense and observation spill for oversized tool return values.
//!
//! Preserves both head (initial context) and tail (latest context/error summary)
//! while splitting out the middle with an informative marker. When an [`OutputSpillSink`]
//! is configured, the full un-truncated content is preserved in an external observation
//! store and referenced via a [`SpillRef`] for downstream recall.
//! Safely respects UTF-8 character boundaries and optional line boundaries without
//! splitting multi-byte code points or emitting partial lines.

pub mod alignment;
pub mod outcome;
pub mod placeholder;
pub mod policy;
pub mod ratio;
pub mod sink;

pub use alignment::*;
pub use outcome::*;
pub use placeholder::*;
pub use policy::*;
pub use ratio::*;
pub use sink::*;

#[cfg(test)]
mod tests;
