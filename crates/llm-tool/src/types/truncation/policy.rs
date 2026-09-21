//! Truncation policy definition and core middle-out execution logic.

use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    sync::Arc,
};

use super::{
    alignment::{MAX_LINE_SCAN_WINDOW, TruncationAlignment},
    outcome::TruncationOutcome,
    placeholder::MarkerPlaceholder,
    ratio::HeadRatio,
    sink::{OutputSpillSink, SpillContext, SpillDisposition, SpillError, SpillRef},
};
use crate::types::ToolOutput;

/// Prefix of the in-band warning emitted when a configured [`OutputSpillSink`] fails.
pub const SPILL_FAILURE_NOTICE_PREFIX: &str = " WARNING: full output could NOT be retained (";

/// Suffix of the in-band warning emitted when a configured [`OutputSpillSink`] fails.
pub const SPILL_FAILURE_NOTICE_SUFFIX: &str = "); the omitted bytes are unrecoverable.";

/// Marker used when `max_bytes` is too small to hold the fully informative marker.
///
/// Truncation is never allowed to be silent, so content is sacrificed before the
/// notice is: a reader must always be able to tell that bytes are missing.
pub const MINIMAL_TRUNCATION_MARKER: &str = "\n[...truncated...]\n";

/// Minimal marker variant used when a configured [`OutputSpillSink`] failed.
pub const MINIMAL_TRUNCATION_MARKER_SPILL_FAILED: &str = "\n[...truncated; NOT retained...]\n";

/// Render the human- and model-readable notice describing a failed spill.
fn spill_failure_notice(error: &SpillError) -> String {
    format!("{SPILL_FAILURE_NOTICE_PREFIX}{error}{SPILL_FAILURE_NOTICE_SUFFIX}")
}

/// Configuration policy for middle-out string truncation.
#[derive(Debug, Clone)]
pub struct TruncationPolicy {
    /// Maximum allowed bytes in the truncated output.
    pub max_bytes: usize,
    /// Ratio of available content budget assigned to the head portion.
    /// Default is 50/50 balance between head and tail.
    pub head_ratio: HeadRatio,
    /// Alignment boundary strategy.
    pub alignment: TruncationAlignment,
    /// Optional custom marker template.
    pub marker_template: Option<String>,
    /// Optional spill sink for retaining full un-truncated observations.
    pub spill_sink: Option<Arc<dyn OutputSpillSink>>,
}

impl PartialEq for TruncationPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.max_bytes == other.max_bytes
            && self.head_ratio == other.head_ratio
            && self.alignment == other.alignment
            && self.marker_template == other.marker_template
            && match (&self.spill_sink, &other.spill_sink) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

impl Eq for TruncationPolicy {}

impl TruncationPolicy {
    /// Create a new truncation policy with a specified `max_bytes` ceiling,
    /// balanced 50/50 head-tail ratio, and character boundary alignment.
    #[must_use]
    pub const fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            head_ratio: HeadRatio::BALANCED,
            alignment: TruncationAlignment::CharBoundary,
            marker_template: None,
            spill_sink: None,
        }
    }

    /// Set the head budget ratio.
    #[must_use]
    pub fn with_head_ratio(mut self, ratio: impl Into<HeadRatio>) -> Self {
        self.head_ratio = ratio.into();
        self
    }

    /// Set the boundary alignment strategy.
    #[must_use]
    pub const fn with_alignment(mut self, alignment: TruncationAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set a custom marker template.
    ///
    /// Placeholders `{omitted_bytes}`, `{original_bytes}`, `{offset_start}`,
    /// `{offset_end}`, and `{ref}` will be substituted. Legacy aliases `{count}`,
    /// `{bytes}`, and `<N>` are also supported.
    #[must_use]
    pub fn with_marker(mut self, marker: impl Into<String>) -> Self {
        self.marker_template = Some(marker.into());
        self
    }

    /// Set a spill sink for retaining un-truncated observations.
    #[must_use]
    pub fn with_spill_sink(mut self, sink: Arc<dyn OutputSpillSink>) -> Self {
        self.spill_sink = Some(sink);
        self
    }

    /// Format the marker string for the given omitted bytes without sink reference.
    ///
    /// Preserved for backwards compatibility with existing callers.
    #[must_use]
    pub fn format_marker(&self, omitted_bytes: usize) -> String {
        self.format_marker_detailed(
            omitted_bytes,
            omitted_bytes,
            0,
            omitted_bytes,
            &SpillDisposition::NotConfigured,
        )
    }

    /// Format marker with full metadata substitutions including offsets and spill disposition.
    ///
    /// When `spill` is [`SpillDisposition::Failed`], the rendered marker always carries
    /// an in-band warning that the omitted bytes are unrecoverable — including for custom
    /// templates that do not mention the failure themselves — so retention loss can never
    /// pass unnoticed by the model reading the output.
    #[must_use]
    pub fn format_marker_detailed(
        &self,
        omitted_bytes: usize,
        original_bytes: usize,
        offset_start: usize,
        offset_end: usize,
        spill: &SpillDisposition,
    ) -> String {
        let omitted_str = omitted_bytes.to_string();
        let mut marker = match &self.marker_template {
            Some(template) => {
                let orig_str = original_bytes.to_string();
                let start_str = offset_start.to_string();
                let end_str = offset_end.to_string();
                let ref_str = spill.spill_ref().map_or("", SpillRef::as_str);

                template
                    .replace(MarkerPlaceholder::TOKEN_OMITTED_BYTES, &omitted_str)
                    .replace(MarkerPlaceholder::TOKEN_ORIGINAL_BYTES, &orig_str)
                    .replace(MarkerPlaceholder::TOKEN_OFFSET_START, &start_str)
                    .replace(MarkerPlaceholder::TOKEN_OFFSET_END, &end_str)
                    .replace(MarkerPlaceholder::TOKEN_SPILL_REF, ref_str)
                    .replace(MarkerPlaceholder::DEPRECATED_COUNT, &omitted_str)
                    .replace(MarkerPlaceholder::DEPRECATED_BYTES, &omitted_str)
                    .replace(MarkerPlaceholder::DEPRECATED_N, &omitted_str)
            }
            None => match spill {
                SpillDisposition::Spilled(handle) => format!(
                    "\n[... {omitted_str} bytes omitted (bytes {offset_start}..{offset_end} of {original_bytes}). Full output retained as ref=\"{handle}\" — recall with read_observation. ...]\n"
                ),
                SpillDisposition::Failed(error) => format!(
                    "\n[... {omitted_str} bytes omitted (bytes {offset_start}..{offset_end} of {original_bytes}).{notice} ...]\n",
                    notice = spill_failure_notice(error)
                ),
                SpillDisposition::NotConfigured => {
                    format!("\n[... {omitted_str} bytes truncated ...]\n")
                }
            },
        };

        // A custom template cannot be trusted to mention the failure, so append it.
        if let Some(error) = spill.error()
            && self.marker_template.is_some()
        {
            marker.push_str(&spill_failure_notice(error));
        }

        marker
    }

    /// Truncate `input` using this policy, returning a borrowed `Cow` when no truncation
    /// is needed, or an owned string when middle-out truncation occurs.
    ///
    /// If a configured [`OutputSpillSink`] fails, this method does **not** fail: the
    /// error is logged and rendered into the truncation marker, so the loss of the
    /// omitted bytes is visible to both the operator and the model. Callers that need
    /// to handle spill failure programmatically should use
    /// [`TruncationPolicy::truncate_detailed`], which returns the [`SpillError`].
    #[must_use]
    pub fn truncate<'a>(&self, input: &'a str) -> Cow<'a, str> {
        self.truncate_lossy_with_context(input, &SpillContext::ANONYMOUS)
            .text
    }

    /// Infallible truncation that degrades visibly — rather than failing — when a
    /// configured [`OutputSpillSink`] returns an error.
    #[must_use]
    pub fn truncate_lossy_with_context<'a>(
        &self,
        input: &'a str,
        ctx: &SpillContext<'_>,
    ) -> TruncationOutcome<'a> {
        if let Some(outcome) = Self::passthrough(input, self.max_bytes) {
            return outcome;
        }

        let spill = self.resolve_spill(ctx, input);
        if let Some(error) = spill.error() {
            tracing::error!(
                error = %error,
                tool = ctx.tool_name,
                "spill sink failed; omitted bytes are unrecoverable and the truncation marker will say so"
            );
        }
        self.truncate_oversized(input, &spill)
    }

    /// Truncate `input` using an anonymous spill context, returning a detailed [`TruncationOutcome`].
    ///
    /// # Errors
    ///
    /// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
    pub fn truncate_detailed<'a>(
        &self,
        input: &'a str,
    ) -> Result<TruncationOutcome<'a>, SpillError> {
        self.truncate_detailed_with_context(input, &SpillContext::ANONYMOUS)
    }

    /// Truncate `input` with caller-provided [`SpillContext`], returning a detailed [`TruncationOutcome`].
    ///
    /// # Errors
    ///
    /// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
    pub fn truncate_detailed_with_context<'a>(
        &self,
        input: &'a str,
        ctx: &SpillContext<'_>,
    ) -> Result<TruncationOutcome<'a>, SpillError> {
        match Self::passthrough(input, self.max_bytes) {
            Some(outcome) => Ok(outcome),
            None => match self.resolve_spill(ctx, input) {
                SpillDisposition::Failed(error) => Err(error),
                spill => Ok(self.truncate_oversized(input, &spill)),
            },
        }
    }

    /// Return the untouched input when it already fits within `max_bytes`.
    ///
    /// Checked before any spill is attempted: content that is not truncated loses
    /// nothing, so it must not consume observation-store capacity.
    fn passthrough(input: &str, max_bytes: usize) -> Option<TruncationOutcome<'_>> {
        let total_bytes = input.len();
        (total_bytes <= max_bytes).then_some(TruncationOutcome {
            text: Cow::Borrowed(input),
            original_len: total_bytes,
            omitted: None,
            spill_ref: None,
        })
    }

    /// Attempt to retain the full output, classifying the result.
    fn resolve_spill(&self, ctx: &SpillContext<'_>, input: &str) -> SpillDisposition {
        match &self.spill_sink {
            Some(sink) => match sink.spill(ctx, input) {
                Ok(handle) => SpillDisposition::Spilled(handle),
                Err(error) => SpillDisposition::Failed(error),
            },
            None => SpillDisposition::NotConfigured,
        }
    }

    /// Perform middle-out elision on input already known to exceed `max_bytes`,
    /// with the spill attempt already resolved.
    fn truncate_oversized<'a>(
        &self,
        input: &'a str,
        spill: &SpillDisposition,
    ) -> TruncationOutcome<'a> {
        let total_bytes = input.len();
        let spill_ref = spill.spill_ref().cloned();

        if self.max_bytes == 0 {
            return TruncationOutcome {
                text: Cow::Borrowed(""),
                original_len: total_bytes,
                omitted: Some(0..total_bytes),
                spill_ref,
            };
        }

        // Estimate marker length using approximate omitted byte count.
        let est_omitted = total_bytes.saturating_sub(self.max_bytes);
        let sample_marker =
            self.format_marker_detailed(est_omitted, total_bytes, 0, est_omitted, spill);
        let marker_len = sample_marker.len();

        if self.max_bytes <= marker_len {
            return self.truncate_with_fallback_marker(input, spill, spill_ref);
        }

        let content_budget = self.max_bytes.saturating_sub(marker_len);
        let (head_end, tail_start) =
            self.compute_initial_boundaries(input, total_bytes, content_budget);

        // If boundaries overlap or touch, no truncation is necessary
        if head_end >= tail_start {
            return TruncationOutcome {
                text: Cow::Borrowed(input),
                original_len: total_bytes,
                omitted: None,
                spill_ref,
            };
        }

        let (head_end, tail_start, marker, current_total) =
            self.converge_boundaries(input, total_bytes, head_end, tail_start, spill);

        let mut output = String::with_capacity(current_total);
        output.push_str(&input[..head_end]);
        output.push_str(&marker);
        output.push_str(&input[tail_start..]);

        if output.len() > self.max_bytes {
            let mut cut = self.max_bytes;
            while cut > 0 && !output.is_char_boundary(cut) {
                cut -= 1;
            }
            output.truncate(cut);
        }

        TruncationOutcome {
            text: Cow::Owned(output),
            original_len: total_bytes,
            omitted: Some(head_end..tail_start),
            spill_ref,
        }
    }

    /// Choose the most informative marker that fits inside `max_bytes`.
    fn fallback_marker(spill: &SpillDisposition, max_bytes: usize) -> String {
        if let SpillDisposition::Spilled(handle) = spill {
            let with_ref = format!("\n[...truncated; ref=\"{handle}\"...]\n");
            if with_ref.len() <= max_bytes {
                return with_ref;
            }
        }
        match spill {
            SpillDisposition::Failed(_) => MINIMAL_TRUNCATION_MARKER_SPILL_FAILED.into(),
            SpillDisposition::NotConfigured | SpillDisposition::Spilled(_) => {
                MINIMAL_TRUNCATION_MARKER.into()
            }
        }
    }

    /// Truncate when `max_bytes` cannot hold the fully informative marker.
    ///
    /// Content is sacrificed before the marker is, so the reader can always tell
    /// that bytes are missing; a budget too small even for the minimal marker
    /// yields a truncated marker rather than a bare, deceptively complete-looking
    /// slice of the input.
    fn truncate_with_fallback_marker<'a>(
        &self,
        input: &str,
        spill: &SpillDisposition,
        spill_ref: Option<SpillRef>,
    ) -> TruncationOutcome<'a> {
        let total_bytes = input.len();
        let marker = Self::fallback_marker(spill, self.max_bytes);

        if marker.len() >= self.max_bytes {
            let mut cut = self.max_bytes;
            while cut > 0 && !marker.is_char_boundary(cut) {
                cut -= 1;
            }
            return TruncationOutcome {
                text: Cow::Owned(marker[..cut].into()),
                original_len: total_bytes,
                omitted: Some(0..total_bytes),
                spill_ref,
            };
        }

        let mut head_end = self.max_bytes - marker.len();
        while head_end > 0 && !input.is_char_boundary(head_end) {
            head_end -= 1;
        }

        let mut text = String::with_capacity(head_end + marker.len());
        text.push_str(&input[..head_end]);
        text.push_str(&marker);

        TruncationOutcome {
            text: Cow::Owned(text),
            original_len: total_bytes,
            omitted: Some(head_end..total_bytes),
            spill_ref,
        }
    }

    fn compute_initial_boundaries(
        &self,
        input: &str,
        total_bytes: usize,
        content_budget: usize,
    ) -> (usize, usize) {
        let head_budget = self.head_ratio.compute_head_budget(content_budget);
        let tail_budget = content_budget.saturating_sub(head_budget);

        // Floor to valid char boundary for head
        let mut head_end = head_budget.min(total_bytes);
        while head_end > 0 && !input.is_char_boundary(head_end) {
            head_end -= 1;
        }

        // Ceil to valid char boundary for tail
        let tail_target = total_bytes.saturating_sub(tail_budget);
        let mut tail_start = tail_target.min(total_bytes);
        while tail_start < total_bytes && !input.is_char_boundary(tail_start) {
            tail_start += 1;
        }

        // Apply line boundary alignment if configured
        if self.alignment == TruncationAlignment::LineBoundary {
            let head_scan_start = head_end.saturating_sub(MAX_LINE_SCAN_WINDOW);
            if let Some(rel_nl) = input[head_scan_start..head_end].rfind('\n') {
                head_end = head_scan_start + rel_nl + 1;
            }

            let tail_scan_end = (tail_start + MAX_LINE_SCAN_WINDOW).min(total_bytes);
            if let Some(rel_nl) = input[tail_start..tail_scan_end].find('\n') {
                tail_start = tail_start + rel_nl + 1;
            }
        }

        (head_end, tail_start)
    }

    fn shrink_head_boundary(&self, input: &str, mut head_end: usize) -> usize {
        if self.alignment == TruncationAlignment::LineBoundary {
            let scan_start = head_end.saturating_sub(MAX_LINE_SCAN_WINDOW);
            if let Some(rel_nl) = input[scan_start..head_end.saturating_sub(1)].rfind('\n') {
                return scan_start + rel_nl + 1;
            }
        }
        head_end = head_end.saturating_sub(1);
        while head_end > 0 && !input.is_char_boundary(head_end) {
            head_end -= 1;
        }
        head_end
    }

    fn expand_tail_boundary(
        &self,
        input: &str,
        total_bytes: usize,
        mut tail_start: usize,
    ) -> usize {
        if self.alignment == TruncationAlignment::LineBoundary {
            let scan_start = tail_start.saturating_add(1).min(total_bytes);
            let scan_end = (tail_start + MAX_LINE_SCAN_WINDOW).min(total_bytes);
            if scan_start < scan_end
                && let Some(rel_nl) = input[scan_start..scan_end].find('\n')
            {
                return scan_start + rel_nl + 1;
            }
        }
        tail_start = tail_start.saturating_add(1);
        while tail_start < total_bytes && !input.is_char_boundary(tail_start) {
            tail_start += 1;
        }
        tail_start
    }

    fn converge_boundaries(
        &self,
        input: &str,
        total_bytes: usize,
        mut head_end: usize,
        mut tail_start: usize,
        spill: &SpillDisposition,
    ) -> (usize, usize, String, usize) {
        let mut omitted_bytes = tail_start.saturating_sub(head_end);
        let mut marker =
            self.format_marker_detailed(omitted_bytes, total_bytes, head_end, tail_start, spill);
        let mut current_total = head_end + marker.len() + (total_bytes - tail_start);

        while current_total > self.max_bytes && (head_end > 0 || tail_start < total_bytes) {
            if head_end > 0 && (tail_start == total_bytes || head_end >= (total_bytes - tail_start))
            {
                head_end = self.shrink_head_boundary(input, head_end);
            } else if tail_start < total_bytes {
                tail_start = self.expand_tail_boundary(input, total_bytes, tail_start);
            } else {
                break;
            }

            omitted_bytes = tail_start.saturating_sub(head_end);
            marker = self.format_marker_detailed(
                omitted_bytes,
                total_bytes,
                head_end,
                tail_start,
                spill,
            );
            current_total = head_end + marker.len() + (total_bytes - tail_start);
        }

        (head_end, tail_start, marker, current_total)
    }
}

/// Helper function to perform middle-out truncation on a string slice.
#[must_use]
pub fn truncate_middle<'a>(input: &'a str, policy: &TruncationPolicy) -> Cow<'a, str> {
    policy.truncate(input)
}

/// Detailed middle-out truncation returning metadata and recall handle.
///
/// # Errors
///
/// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
pub fn truncate_middle_detailed<'a>(
    input: &'a str,
    policy: &TruncationPolicy,
) -> Result<TruncationOutcome<'a>, SpillError> {
    policy.truncate_detailed(input)
}

/// Detailed middle-out truncation with [`SpillContext`].
///
/// # Errors
///
/// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
pub fn truncate_middle_detailed_with_context<'a>(
    input: &'a str,
    policy: &TruncationPolicy,
    ctx: &SpillContext<'_>,
) -> Result<TruncationOutcome<'a>, SpillError> {
    policy.truncate_detailed_with_context(input, ctx)
}

/// Helper function to truncate a [`ToolOutput`] in-place according to `policy`.
pub fn truncate_output(output: &mut ToolOutput, policy: &TruncationPolicy) {
    output.content = policy.truncate(&output.content).into_owned();
}

/// Detailed in-place truncation of [`ToolOutput`].
///
/// # Errors
///
/// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
pub fn truncate_output_detailed(
    output: &mut ToolOutput,
    policy: &TruncationPolicy,
) -> Result<TruncationOutcome<'static>, SpillError> {
    truncate_output_detailed_with_context(output, policy, &SpillContext::ANONYMOUS)
}

/// Detailed in-place truncation of [`ToolOutput`] with [`SpillContext`].
///
/// # Errors
///
/// Returns [`SpillError`] if an [`OutputSpillSink`] is configured and fails.
pub fn truncate_output_detailed_with_context(
    output: &mut ToolOutput,
    policy: &TruncationPolicy,
    ctx: &SpillContext<'_>,
) -> Result<TruncationOutcome<'static>, SpillError> {
    let outcome = policy.truncate_detailed_with_context(&output.content, ctx)?;
    let text = outcome.text.into_owned();
    let original_len = outcome.original_len;
    let omitted = outcome.omitted;
    let spill_ref = outcome.spill_ref;
    output.content.clone_from(&text);
    Ok(TruncationOutcome {
        text: Cow::Owned(text),
        original_len,
        omitted,
        spill_ref,
    })
}
