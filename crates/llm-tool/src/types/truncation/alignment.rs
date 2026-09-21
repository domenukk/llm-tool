//! Boundary alignment policies for truncation cuts.

/// Maximum window (in bytes) to scan backward or forward for a newline boundary.
///
/// Prevents pathological single-line inputs from collapsing the available budget.
pub const MAX_LINE_SCAN_WINDOW: usize = 4096;

/// Boundary snapping strategy for middle-out truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationAlignment {
    /// Snap strictly to valid UTF-8 character boundaries (default).
    #[default]
    CharBoundary,
    /// Snap head back to preceding newline and tail forward to following newline,
    /// degrading gracefully to character boundary if no newline is found within
    /// [`MAX_LINE_SCAN_WINDOW`].
    LineBoundary,
}
