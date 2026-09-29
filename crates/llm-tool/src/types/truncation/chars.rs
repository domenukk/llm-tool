//! Character-boundary truncation helpers, token-adaptive budget sizing, and canonical JSON ordering.
//!
//! Rust string slicing (`&s[..n]`) indexes by *byte* and panics when `n` splits a multi-byte
//! UTF-8 sequence. These helpers centralize character-counted head, tail, and middle truncation
//! alongside token-adaptive byte budget sizing for [`TruncationPolicy`].

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};

use super::policy::TruncationPolicy;

/// Heuristic UTF-8 byte-per-token multiplier (`4` bytes/token).
pub const DEFAULT_BYTES_PER_TOKEN: usize = 4;

/// Default minimum byte budget (`4 KiB`) for token-adaptive tool output truncation.
pub const MIN_ADAPTIVE_TOOL_OUTPUT_BYTES: usize = 4 * 1024;

/// Default maximum byte budget (`64 KiB`) for token-adaptive tool output truncation.
pub const MAX_ADAPTIVE_TOOL_OUTPUT_BYTES: usize = 64 * 1024;

/// Default numerator (`3`) of the remaining context token fraction (`3/10 = 30%`) assigned to a single tool output.
pub const DEFAULT_TOOL_OUTPUT_CONTEXT_NUMERATOR: usize = 3;

/// Default denominator (`10`) of the remaining context token fraction (`3/10 = 30%`) assigned to a single tool output.
pub const DEFAULT_TOOL_OUTPUT_CONTEXT_DENOMINATOR: usize = 10;

/// Compute a token-adaptive byte budget from `remaining_context_tokens`, allocating `30%`
/// (`3/10`) of remaining tokens at `4` bytes/token and clamping to `[min_bytes, max_bytes]`.
///
/// ```
/// use llm_tool::{
///     MAX_ADAPTIVE_TOOL_OUTPUT_BYTES, MIN_ADAPTIVE_TOOL_OUTPUT_BYTES,
///     compute_token_adaptive_budget,
/// };
///
/// // Small remaining window clamps to MIN_ADAPTIVE_TOOL_OUTPUT_BYTES (4 KiB).
/// assert_eq!(
///     compute_token_adaptive_budget(
///         500,
///         MIN_ADAPTIVE_TOOL_OUTPUT_BYTES,
///         MAX_ADAPTIVE_TOOL_OUTPUT_BYTES
///     ),
///     MIN_ADAPTIVE_TOOL_OUTPUT_BYTES
/// );
/// // 20,000 remaining tokens -> 30% = 6,000 tokens -> 24,000 bytes.
/// assert_eq!(
///     compute_token_adaptive_budget(
///         20_000,
///         MIN_ADAPTIVE_TOOL_OUTPUT_BYTES,
///         MAX_ADAPTIVE_TOOL_OUTPUT_BYTES
///     ),
///     24_000
/// );
/// // Huge remaining window clamps to MAX_ADAPTIVE_TOOL_OUTPUT_BYTES (64 KiB).
/// assert_eq!(
///     compute_token_adaptive_budget(
///         200_000,
///         MIN_ADAPTIVE_TOOL_OUTPUT_BYTES,
///         MAX_ADAPTIVE_TOOL_OUTPUT_BYTES
///     ),
///     MAX_ADAPTIVE_TOOL_OUTPUT_BYTES
/// );
/// ```
#[must_use]
pub const fn compute_token_adaptive_budget(
    remaining_context_tokens: usize,
    min_bytes: usize,
    max_bytes: usize,
) -> usize {
    let ceiling = if max_bytes < min_bytes {
        min_bytes
    } else {
        max_bytes
    };
    let allocated_tokens = remaining_context_tokens
        .saturating_mul(DEFAULT_TOOL_OUTPUT_CONTEXT_NUMERATOR)
        / DEFAULT_TOOL_OUTPUT_CONTEXT_DENOMINATOR;
    let raw_bytes = allocated_tokens.saturating_mul(DEFAULT_BYTES_PER_TOKEN);
    if raw_bytes < min_bytes {
        min_bytes
    } else if raw_bytes > ceiling {
        ceiling
    } else {
        raw_bytes
    }
}

impl TruncationPolicy {
    /// Construct a [`TruncationPolicy`] whose `max_bytes` scales with `remaining_context_tokens`
    /// (`30%` of remaining tokens at `4` bytes/token, clamped between [`MIN_ADAPTIVE_TOOL_OUTPUT_BYTES`]
    /// and [`MAX_ADAPTIVE_TOOL_OUTPUT_BYTES`]).
    #[must_use]
    pub const fn for_context_window(remaining_context_tokens: usize) -> Self {
        Self::new(compute_token_adaptive_budget(
            remaining_context_tokens,
            MIN_ADAPTIVE_TOOL_OUTPUT_BYTES,
            MAX_ADAPTIVE_TOOL_OUTPUT_BYTES,
        ))
    }

    /// Construct a [`TruncationPolicy`] scaled to `remaining_context_tokens` with custom
    /// `[min_bytes, max_bytes]` bounds.
    #[must_use]
    pub const fn for_context_window_clamped(
        remaining_context_tokens: usize,
        min_bytes: usize,
        max_bytes: usize,
    ) -> Self {
        Self::new(compute_token_adaptive_budget(
            remaining_context_tokens,
            min_bytes,
            max_bytes,
        ))
    }
}

/// Truncate `text` to at most `max_chars` Unicode scalar values, keeping the **start** and
/// appending `marker` when truncation occurs.
///
/// The returned string (including `marker`) never exceeds `max_chars` characters when
/// `marker.chars().count() <= max_chars`. Truncation always lands on a UTF-8 character boundary.
#[must_use]
pub fn truncate_head_chars(text: &str, max_chars: usize, marker: &str) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let keep = max_chars.saturating_sub(marker.chars().count());
    let end = text
        .char_indices()
        .nth(keep)
        .map_or(text.len(), |(idx, _)| idx);
    format!("{}{marker}", &text[..end])
}

/// Truncate `text` to at most `max_chars` Unicode scalar values, keeping the **end** and
/// prepending `prefix` when truncation occurs.
///
/// Mirrors [`truncate_head_chars`] for tail-retaining truncation (e.g. showing only the most
/// recent portion of a log or thread). Always lands on a UTF-8 character boundary.
#[must_use]
pub fn truncate_tail_chars(text: &str, max_chars: usize, prefix: &str) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let keep = max_chars.saturating_sub(prefix.chars().count());
    let skip = total - keep;
    let start = text
        .char_indices()
        .nth(skip)
        .map_or(text.len(), |(idx, _)| idx);
    format!("{prefix}{}", &text[start..])
}

/// Truncate `text` by keeping its **head and tail** Unicode scalar counts (`head_chars` and
/// `tail_chars`), eliding the middle with `marker` when `text.chars().count() > head_chars + tail_chars`.
///
/// Both cut points land on UTF-8 character boundaries. Returns `text` unchanged when it fits.
#[must_use]
pub fn truncate_middle_chars(
    text: &str,
    head_chars: usize,
    tail_chars: usize,
    marker: &str,
) -> String {
    let total = text.chars().count();
    if total <= head_chars.saturating_add(tail_chars) {
        return text.to_string();
    }
    let head_end = text
        .char_indices()
        .nth(head_chars)
        .map_or(text.len(), |(idx, _)| idx);
    let tail_start = text
        .char_indices()
        .nth(total - tail_chars)
        .map_or(text.len(), |(idx, _)| idx);
    format!("{}{marker}{}", &text[..head_end], &text[tail_start..])
}

/// Recursively sorts JSON object keys lexicographically and canonicalizes nested values.
///
/// Useful for deterministic tool-call argument hashing, loop detection, and cache keys.
#[must_use]
pub fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut btree = BTreeMap::new();
            for (k, v) in map {
                btree.insert(k.clone(), canonical_json(v));
            }
            let mut sorted_map = serde_json::Map::with_capacity(btree.len());
            for (k, v) in btree {
                sorted_map.insert(k, v);
            }
            serde_json::Value::Object(sorted_map)
        }
        serde_json::Value::Array(arr) => {
            let canonical_arr: Vec<serde_json::Value> = arr.iter().map(canonical_json).collect();
            serde_json::Value::Array(canonical_arr)
        }
        _ => value.clone(),
    }
}
