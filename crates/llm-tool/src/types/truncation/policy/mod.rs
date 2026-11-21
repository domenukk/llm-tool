//! Truncation policy definition and core middle-out execution logic.

mod envelope;

use alloc::{
    borrow::Cow,
    format,
    string::{String, ToString},
    sync::Arc,
};

use envelope::{
    DetectedEnvelope, detect_verified_envelope, snap_head_outside_surrogate,
    snap_tail_outside_surrogate,
};

use super::{
    alignment::{MAX_LINE_SCAN_WINDOW, TruncationAlignment},
    outcome::TruncationOutcome,
    placeholder::MarkerPlaceholder,
    ratio::HeadRatio,
    sink::{OutputSpillSink, SpillContext, SpillDisposition, SpillError, SpillRef},
};
use crate::types::{
    ToolOutput,
    quarantine::{
        CLOSE_FRAME, FRAME_OVERHEAD, OPEN_FRAME, TOOL_OUTPUT_QUARANTINE_TAG,
        quarantine_tool_output, unquarantine_tool_output,
    },
};

/// Prefix of the in-band warning emitted when a configured [`OutputSpillSink`] fails.
pub const SPILL_FAILURE_NOTICE_PREFIX: &str = " WARNING: full output could NOT be retained (";

/// Suffix of the in-band warning emitted when a configured [`OutputSpillSink`] fails.
pub const SPILL_FAILURE_NOTICE_SUFFIX: &str = "); the omitted bytes are unrecoverable.";

/// Marker used when `max_bytes` is too small to hold the fully informative marker.
pub const MINIMAL_TRUNCATION_MARKER: &str = "\n[TRUNCATED: bytes omitted]\n";

/// Minimal marker variant used when a configured [`OutputSpillSink`] failed.
pub const MINIMAL_TRUNCATION_MARKER_SPILL_FAILED: &str =
    "\n[TRUNCATED: bytes omitted; NOT retained]\n";

/// Compact single-line truncation notice used inside XML envelopes when
/// the inner budget is smaller than [`MINIMAL_TRUNCATION_MARKER`].
pub(super) const COMPACT_TRUNCATION_MARKER: &str = "[TRUNCATED: bytes omitted]";

/// Compact single-line spill-failed truncation notice used inside XML envelopes.
pub(super) const COMPACT_TRUNCATION_MARKER_SPILL_FAILED: &str =
    "[TRUNCATED: bytes omitted; NOT retained]";

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
    /// Escapes any XML/attribute special characters in [`SpillRef`] handles via
    /// [`md_tmpl_core::escape_xml_str`].
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
        let mut marker = self.marker_template.as_ref().map_or_else(
            || match spill {
                SpillDisposition::Spilled(handle) => {
                    let escaped_ref = md_tmpl_core::escape_xml_str(handle.as_str());
                    format!(
                        "\n[TRUNCATED: {omitted_str} bytes omitted (bytes {offset_start}..{offset_end} of {original_bytes}). Full output retained as ref=\"{escaped_ref}\" — inspect with read_observation(ref, pattern, offset, length)]\n"
                    )
                }
                SpillDisposition::Failed(error) => format!(
                    "\n[TRUNCATED: {omitted_str} bytes omitted (bytes {offset_start}..{offset_end} of {original_bytes}).{notice}]\n",
                    notice = spill_failure_notice(error)
                ),
                SpillDisposition::NotConfigured => {
                    format!("\n[TRUNCATED: {omitted_str} bytes omitted]\n")
                }
            },
            |template| {
                let orig_str = original_bytes.to_string();
                let start_str = offset_start.to_string();
                let end_str = offset_end.to_string();
                let escaped_ref: Cow<'_, str> = spill
                    .spill_ref()
                    .map_or(Cow::Borrowed(""), |r| md_tmpl_core::escape_xml_str(r.as_str()));

                template
                    .replace(MarkerPlaceholder::TOKEN_OMITTED_BYTES, &omitted_str)
                    .replace(MarkerPlaceholder::TOKEN_ORIGINAL_BYTES, &orig_str)
                    .replace(MarkerPlaceholder::TOKEN_OFFSET_START, &start_str)
                    .replace(MarkerPlaceholder::TOKEN_OFFSET_END, &end_str)
                    .replace(MarkerPlaceholder::TOKEN_SPILL_REF, &escaped_ref)
                    .replace(MarkerPlaceholder::DEPRECATED_COUNT, &omitted_str)
                    .replace(MarkerPlaceholder::DEPRECATED_BYTES, &omitted_str)
                    .replace(MarkerPlaceholder::DEPRECATED_N, &omitted_str)
            },
        );

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
    /// Preserves verified `<tool_output_quarantine>`, `SanitizeTagged`, and `SanitizeWithNotice`
    /// envelopes across middle-out truncation without dropping closing tags or allowing splice
    /// breakouts, and avoids bisecting `<surrogate:kind:nonce>` handles.
    #[must_use]
    pub fn truncate<'a>(&self, input: &'a str) -> Cow<'a, str> {
        self.truncate_lossy_with_context(input, &SpillContext::ANONYMOUS)
            .text
    }

    /// Sanitize and truncate `input` in a single pass, neutralizing all known LLM
    /// control tokens and breakout tags (including before spilling to any configured
    /// [`OutputSpillSink`]) while fitting the result within `self.max_bytes`.
    #[must_use]
    pub fn truncate_sanitized<'a>(&self, input: &'a str) -> Cow<'a, str> {
        self.truncate_sanitized_lossy_with_context(input, &SpillContext::ANONYMOUS)
            .text
    }

    /// Quarantine and truncate `input`, neutralizing all known LLM control tokens and
    /// wrapping the output in `<tool_output_quarantine>` tags while fitting the total
    /// envelope inside `self.max_bytes` (when `self.max_bytes >= 70`).
    #[must_use]
    pub fn truncate_quarantined(&self, input: &str) -> String {
        self.truncate_quarantined_lossy_with_context(input, &SpillContext::ANONYMOUS)
            .text
            .into_owned()
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

        let spill = self.resolve_spill_logged(ctx, input);
        self.truncate_oversized_auto(input, &spill)
    }

    /// Infallible sanitize + truncate with caller-provided [`SpillContext`].
    ///
    /// Sanitizes `input` **before** spilling to any configured [`OutputSpillSink`] so raw
    /// prompt-injection delimiters are never persisted into external observation stores.
    #[must_use]
    pub fn truncate_sanitized_lossy_with_context<'a>(
        &self,
        input: &'a str,
        ctx: &SpillContext<'_>,
    ) -> TruncationOutcome<'a> {
        if detect_verified_envelope(input).is_some() {
            return self.truncate_lossy_with_context(input, ctx);
        }

        let sanitized = md_tmpl_core::sanitize_untrusted_str(input, TOOL_OUTPUT_QUARANTINE_TAG);
        if sanitized.len() <= self.max_bytes {
            return TruncationOutcome {
                text: sanitized,
                original_len: input.len(),
                omitted: None,
                spill_ref: None,
            };
        }

        let spill = self.resolve_spill_logged(ctx, &sanitized);
        let outcome = self.truncate_oversized_sanitized_with_tags(
            &sanitized,
            &spill,
            TOOL_OUTPUT_QUARANTINE_TAG,
        );
        TruncationOutcome {
            text: Cow::Owned(outcome.text.into_owned()),
            original_len: input.len(),
            omitted: outcome.omitted,
            spill_ref: outcome.spill_ref,
        }
    }

    /// Infallible quarantine + truncate with caller-provided [`SpillContext`].
    ///
    /// Quarantines and neutralizes `input` **before** spilling to any configured
    /// [`OutputSpillSink`].
    #[must_use]
    pub fn truncate_quarantined_lossy_with_context<'a>(
        &self,
        input: &'a str,
        ctx: &SpillContext<'_>,
    ) -> TruncationOutcome<'a> {
        if unquarantine_tool_output(input).is_some() {
            return self.truncate_lossy_with_context(input, ctx);
        }

        let inner_budget = self.max_bytes.saturating_sub(FRAME_OVERHEAD);
        if input.len() <= inner_budget && self.max_bytes >= FRAME_OVERHEAD {
            let sanitized = md_tmpl_core::sanitize_untrusted_str(input, TOOL_OUTPUT_QUARANTINE_TAG);
            if sanitized.len() <= inner_budget {
                return TruncationOutcome {
                    text: Cow::Owned(format!("{OPEN_FRAME}{sanitized}{CLOSE_FRAME}")),
                    original_len: input.len(),
                    omitted: None,
                    spill_ref: None,
                };
            }
        }

        let quarantined_full = quarantine_tool_output(input);
        let spill = self.resolve_spill_logged(ctx, &quarantined_full);
        let inner_sanitized =
            unquarantine_tool_output(&quarantined_full).unwrap_or(quarantined_full.as_str());
        let env = DetectedEnvelope {
            tag: TOOL_OUTPUT_QUARANTINE_TAG,
            notice: None,
            payload: inner_sanitized,
            offset_bias: 0,
        };
        self.truncate_and_wrap_envelope(&env, input.len(), &spill)
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
        Self::passthrough(input, self.max_bytes).map_or_else(
            || match self.resolve_spill(ctx, input) {
                SpillDisposition::Failed(error) => Err(error),
                spill => Ok(self.truncate_oversized_auto(input, &spill)),
            },
            Ok,
        )
    }

    fn passthrough(input: &str, max_bytes: usize) -> Option<TruncationOutcome<'_>> {
        let total_bytes = input.len();
        (total_bytes <= max_bytes).then_some(TruncationOutcome {
            text: Cow::Borrowed(input),
            original_len: total_bytes,
            omitted: None,
            spill_ref: None,
        })
    }

    fn resolve_spill(&self, ctx: &SpillContext<'_>, input: &str) -> SpillDisposition {
        self.spill_sink
            .as_ref()
            .map_or(SpillDisposition::NotConfigured, |sink| {
                match sink.spill(ctx, input) {
                    Ok(handle) => SpillDisposition::Spilled(handle),
                    Err(error) => SpillDisposition::Failed(error),
                }
            })
    }

    fn resolve_spill_logged(&self, ctx: &SpillContext<'_>, input: &str) -> SpillDisposition {
        let spill = self.resolve_spill(ctx, input);
        if let Some(error) = spill.error() {
            tracing::error!(
                error = %error,
                tool = ctx.tool_name,
                "spill sink failed; omitted bytes are unrecoverable and the truncation marker will say so"
            );
        }
        spill
    }

    /// Dispatch oversized truncation while preserving `<tool_output_quarantine>`,
    /// `SanitizeTagged`, and `SanitizeWithNotice` envelopes and sanitization invariants.
    fn truncate_oversized_auto<'a>(
        &self,
        input: &'a str,
        spill: &SpillDisposition,
    ) -> TruncationOutcome<'a> {
        if let Some(env) = detect_verified_envelope(input) {
            return self.truncate_and_wrap_envelope(&env, input.len(), spill);
        }
        if !md_tmpl_core::has_untrusted_breakout(input, TOOL_OUTPUT_QUARANTINE_TAG) {
            return self.truncate_oversized_sanitized_with_tags(
                input,
                spill,
                TOOL_OUTPUT_QUARANTINE_TAG,
            );
        }
        self.truncate_oversized(input, spill)
    }

    /// Perform middle-out elision on `input` while neutralizing any control tokens
    /// or breakout tags matching `tag_spec` across head, marker, tail, and their splice
    /// boundaries, using proportional step-back (`O(n log max_bytes)`) to converge.
    pub(super) fn truncate_oversized_sanitized_with_tags<'a>(
        &self,
        input: &'a str,
        spill: &SpillDisposition,
        tag_spec: &str,
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

        let est_omitted = total_bytes.saturating_sub(self.max_bytes);
        let sample_marker =
            self.format_marker_detailed(est_omitted, total_bytes, 0, est_omitted, spill);
        let sample_marker_sanitized =
            md_tmpl_core::sanitize_untrusted_str(&sample_marker, tag_spec);
        let marker_len = sample_marker_sanitized.len();

        if self.max_bytes <= marker_len {
            return self.truncate_with_fallback_marker_sanitized(input, spill, spill_ref, tag_spec);
        }

        let content_budget = self.max_bytes.saturating_sub(marker_len);
        let (mut head_end, mut tail_start) =
            self.compute_initial_boundaries(input, total_bytes, content_budget);

        if head_end >= tail_start {
            let sanitized_full = md_tmpl_core::sanitize_untrusted_str(input, tag_spec);
            if sanitized_full.len() <= self.max_bytes {
                return TruncationOutcome {
                    text: sanitized_full,
                    original_len: total_bytes,
                    omitted: None,
                    spill_ref,
                };
            }
            let mut mid = total_bytes / 2;
            while mid > 0 && !input.is_char_boundary(mid) {
                mid -= 1;
            }
            head_end = snap_head_outside_surrogate(input, mid);
            tail_start = snap_tail_outside_surrogate(input, mid);
        }

        let (mut head_end, mut tail_start, mut marker, _) =
            self.converge_boundaries(input, total_bytes, head_end, tail_start, spill);

        loop {
            let tail_len = total_bytes - tail_start;
            let mut candidate = String::with_capacity(head_end + marker.len() + tail_len);
            candidate.push_str(&input[..head_end]);
            candidate.push_str(&marker);
            candidate.push_str(&input[tail_start..]);

            let sanitized = md_tmpl_core::sanitize_untrusted_str(&candidate, tag_spec);
            if sanitized.len() <= self.max_bytes {
                return TruncationOutcome {
                    text: Cow::Owned(sanitized.into_owned()),
                    original_len: total_bytes,
                    omitted: Some(head_end..tail_start),
                    spill_ref,
                };
            }

            let kept = head_end + tail_len;
            if kept == 0 {
                return self
                    .truncate_with_fallback_marker_sanitized(input, spill, spill_ref, tag_spec);
            }

            let excess = sanitized.len() - self.max_bytes;
            let drop_total =
                usize::try_from((kept as u128 * excess as u128).div_ceil(sanitized.len() as u128))
                    .unwrap_or(excess)
                    .max(excess / 2)
                    .max(1);

            self.step_back_boundaries_by(
                input,
                total_bytes,
                &mut head_end,
                &mut tail_start,
                drop_total,
            );

            let omitted_bytes = tail_start.saturating_sub(head_end);
            marker = self.format_marker_detailed(
                omitted_bytes,
                total_bytes,
                head_end,
                tail_start,
                spill,
            );
        }
    }

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

    fn fallback_marker(spill: &SpillDisposition, max_bytes: usize) -> String {
        if let SpillDisposition::Spilled(handle) = spill {
            let escaped_ref = md_tmpl_core::escape_xml_str(handle.as_str());
            let with_ref = format!("\n[TRUNCATED: bytes omitted; ref=\"{escaped_ref}\"]\n");
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

    fn truncate_with_fallback_marker_sanitized<'a>(
        &self,
        input: &str,
        spill: &SpillDisposition,
        spill_ref: Option<SpillRef>,
        tag_spec: &str,
    ) -> TruncationOutcome<'a> {
        let total_bytes = input.len();
        let raw_marker = Self::fallback_marker(spill, self.max_bytes);
        let marker = md_tmpl_core::sanitize_untrusted_str(&raw_marker, tag_spec);

        if marker.len() >= self.max_bytes {
            let mut cut = self.max_bytes;
            while cut > 0 && !marker.is_char_boundary(cut) {
                cut -= 1;
            }
            let slice = &marker[..cut];
            let clean = md_tmpl_core::sanitize_untrusted_str(slice, tag_spec);
            return TruncationOutcome {
                text: Cow::Owned(clean.into_owned()),
                original_len: total_bytes,
                omitted: Some(0..total_bytes),
                spill_ref,
            };
        }

        let mut head_end = (self.max_bytes - marker.len()).min(total_bytes);
        while head_end > 0 && !input.is_char_boundary(head_end) {
            head_end -= 1;
        }
        head_end = snap_head_outside_surrogate(input, head_end);

        loop {
            let mut text = String::with_capacity(head_end + marker.len());
            text.push_str(&input[..head_end]);
            text.push_str(&marker);
            let sanitized = md_tmpl_core::sanitize_untrusted_str(&text, tag_spec);
            if sanitized.len() <= self.max_bytes {
                return TruncationOutcome {
                    text: Cow::Owned(sanitized.into_owned()),
                    original_len: total_bytes,
                    omitted: Some(head_end..total_bytes),
                    spill_ref,
                };
            }
            if head_end == 0 {
                let mut cut = self.max_bytes.min(marker.len());
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
            let excess = sanitized.len() - self.max_bytes;
            let drop_bytes = usize::try_from(
                (head_end as u128 * excess as u128).div_ceil(sanitized.len() as u128),
            )
            .unwrap_or(excess)
            .max(excess / 2)
            .max(1);
            head_end = head_end.saturating_sub(drop_bytes);
            while head_end > 0 && !input.is_char_boundary(head_end) {
                head_end -= 1;
            }
            head_end = snap_head_outside_surrogate(input, head_end);
        }
    }

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

        let mut head_end = (self.max_bytes - marker.len()).min(total_bytes);
        while head_end > 0 && !input.is_char_boundary(head_end) {
            head_end -= 1;
        }
        head_end = snap_head_outside_surrogate(input, head_end);

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

        let mut head_end = head_budget.min(total_bytes);
        while head_end > 0 && !input.is_char_boundary(head_end) {
            head_end -= 1;
        }

        let tail_target = total_bytes.saturating_sub(tail_budget);
        let mut tail_start = tail_target.min(total_bytes);
        while tail_start < total_bytes && !input.is_char_boundary(tail_start) {
            tail_start += 1;
        }

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

        head_end = snap_head_outside_surrogate(input, head_end);
        tail_start = snap_tail_outside_surrogate(input, tail_start);
        (head_end, tail_start)
    }

    fn shrink_head_by(&self, input: &str, head_end: usize, drop_bytes: usize) -> usize {
        if head_end == 0 {
            return 0;
        }
        let target = head_end.saturating_sub(drop_bytes.max(1));
        if self.alignment == TruncationAlignment::LineBoundary {
            let scan_start = target.saturating_sub(MAX_LINE_SCAN_WINDOW);
            if let Some(rel_nl) =
                input[scan_start..target.min(head_end.saturating_sub(1))].rfind('\n')
            {
                return snap_head_outside_surrogate(input, scan_start + rel_nl + 1);
            }
        }
        let mut next = target;
        while next > 0 && !input.is_char_boundary(next) {
            next -= 1;
        }
        snap_head_outside_surrogate(input, next)
    }

    fn expand_tail_by(
        &self,
        input: &str,
        total_bytes: usize,
        tail_start: usize,
        drop_bytes: usize,
    ) -> usize {
        if tail_start >= total_bytes {
            return total_bytes;
        }
        let target = tail_start
            .saturating_add(drop_bytes.max(1))
            .min(total_bytes);
        if self.alignment == TruncationAlignment::LineBoundary {
            let scan_end = target.saturating_add(MAX_LINE_SCAN_WINDOW).min(total_bytes);
            if target < scan_end
                && let Some(rel_nl) = input[target..scan_end].find('\n')
            {
                return snap_tail_outside_surrogate(input, target + rel_nl + 1);
            }
        }
        let mut next = target;
        while next < total_bytes && !input.is_char_boundary(next) {
            next += 1;
        }
        snap_tail_outside_surrogate(input, next)
    }

    fn step_back_boundaries_by(
        &self,
        input: &str,
        total_bytes: usize,
        head_end: &mut usize,
        tail_start: &mut usize,
        drop_total: usize,
    ) {
        let tail_len = total_bytes.saturating_sub(*tail_start);
        if drop_total <= 1 {
            if *head_end > 0 && (tail_len == 0 || *head_end >= tail_len) {
                *head_end = self.shrink_head_by(input, *head_end, 1);
            } else if *tail_start < total_bytes {
                *tail_start = self.expand_tail_by(input, total_bytes, *tail_start, 1);
            }
            return;
        }

        let kept = *head_end + tail_len;
        if kept == 0 {
            return;
        }
        let head_drop = usize::try_from((*head_end as u128 * drop_total as u128) / kept as u128)
            .unwrap_or(drop_total / 2);
        let tail_drop = drop_total.saturating_sub(head_drop);

        let prev_head = *head_end;
        let prev_tail = *tail_start;
        if head_drop > 0 && *head_end > 0 {
            *head_end = self.shrink_head_by(input, *head_end, head_drop);
        }
        if tail_drop > 0 && *tail_start < total_bytes {
            *tail_start = self.expand_tail_by(input, total_bytes, *tail_start, tail_drop);
        }
        // Guarantee strict progress even under extreme head/tail imbalance.
        if *head_end == prev_head && *tail_start == prev_tail {
            if *head_end > 0 {
                *head_end = self.shrink_head_by(input, *head_end, 1);
            } else if *tail_start < total_bytes {
                *tail_start = self.expand_tail_by(input, total_bytes, *tail_start, 1);
            }
        }
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
            let excess = current_total - self.max_bytes;
            self.step_back_boundaries_by(
                input,
                total_bytes,
                &mut head_end,
                &mut tail_start,
                excess,
            );

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
///
/// Zero heap allocations when `output.content.len() <= policy.max_bytes`.
pub fn truncate_output(output: &mut ToolOutput, policy: &TruncationPolicy) {
    if let Cow::Owned(truncated) = policy.truncate(&output.content) {
        output.content = truncated;
    }
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
    let original_len = outcome.original_len;
    let omitted = outcome.omitted;
    let spill_ref = outcome.spill_ref;
    let text = match outcome.text {
        Cow::Borrowed(_) => output.content.clone(),
        Cow::Owned(truncated) => {
            output.content.clone_from(&truncated);
            truncated
        }
    };
    Ok(TruncationOutcome {
        text: Cow::Owned(text),
        original_len,
        omitted,
        spill_ref,
    })
}
