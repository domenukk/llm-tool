//! Envelope detection (`<tool_output_quarantine>`, `SanitizeTagged`, `SanitizeWithNotice`)
//! and `<surrogate:kind:nonce>` token boundary snapping for middle-out truncation.

use alloc::{borrow::Cow, format, string::String};

use super::{
    COMPACT_TRUNCATION_MARKER, COMPACT_TRUNCATION_MARKER_SPILL_FAILED, MINIMAL_TRUNCATION_MARKER,
    TruncationPolicy,
};
use crate::types::{
    quarantine::{
        TOOL_OUTPUT_QUARANTINE_TAG, is_bracketed_notice_header, is_valid_bounded_sanitize_tag,
        unquarantine_tool_output,
    },
    surrogate::{MAX_SURROGATE_TOKEN_LEN, SURROGATE_TOKEN_PREFIX, match_surrogate_span_at},
    truncation::{outcome::TruncationOutcome, sink::SpillDisposition},
};

#[inline]
fn is_control_token_xml_tag(tag: &str) -> bool {
    matches!(
        tag,
        "tool_call"
            | "tool_response"
            | "function"
            | "parameter"
            | "tools"
            | "function_calls"
            | "function_results"
            | "think"
            | "role"
            | "bos"
            | "eos"
            | "sop"
            | "eop"
            | "start_of_turn"
            | "end_of_turn"
            | "beginning_of_sentence"
            | "end_of_sentence"
    )
}

/// Parsed structure of an already-verified XML quarantine or sanitized boundary envelope.
pub(super) struct DetectedEnvelope<'a> {
    pub tag: &'a str,
    pub notice: Option<&'a str>,
    pub payload: &'a str,
    pub offset_bias: usize,
}

#[inline]
fn subslice_offset(outer: &str, inner: &str) -> usize {
    (inner.as_ptr() as usize).saturating_sub(outer.as_ptr() as usize)
}

/// Detect whether `input` is a verified `<tool_output_quarantine>`, `SanitizeTagged`, or
/// `SanitizeWithNotice` `<tag>...</tag>` envelope with zero unescaped breakouts.
pub(super) fn detect_verified_envelope(input: &str) -> Option<DetectedEnvelope<'_>> {
    if !input.starts_with('<') || !input.ends_with('>') {
        return None;
    }
    if let Some(inner) = unquarantine_tool_output(input) {
        return Some(DetectedEnvelope {
            tag: TOOL_OUTPUT_QUARANTINE_TAG,
            notice: None,
            payload: inner,
            offset_bias: subslice_offset(input, inner),
        });
    }

    let gt_rel = input[1..].find('>')?;
    let tag = &input[1..=gt_rel];
    if !is_valid_bounded_sanitize_tag(tag) || is_control_token_xml_tag(tag) {
        return None;
    }

    let rest = &input[gt_rel + 2..];
    let body = rest
        .strip_suffix('>')
        .and_then(|r| r.strip_suffix(tag))
        .and_then(|r| r.strip_suffix("</"))?;
    if md_tmpl_core::has_untrusted_breakout(body, tag) {
        return None;
    }

    let inner = body.strip_prefix('\n').unwrap_or(body);
    let inner = inner.strip_suffix('\n').unwrap_or(inner);
    if let Some((first_line, after)) = inner.split_once('\n')
        && is_bracketed_notice_header(first_line)
    {
        return Some(DetectedEnvelope {
            tag,
            notice: Some(first_line),
            payload: after,
            offset_bias: subslice_offset(input, after),
        });
    }

    Some(DetectedEnvelope {
        tag,
        notice: None,
        payload: inner,
        offset_bias: subslice_offset(input, inner),
    })
}

/// Snap `head_end` backwards before any `<surrogate:kind:nonce>` token it would otherwise bisect.
#[must_use]
pub(super) fn snap_head_outside_surrogate(input: &str, head_end: usize) -> usize {
    if head_end == 0 || head_end >= input.len() {
        return head_end;
    }
    let mut window_start = head_end.saturating_sub(MAX_SURROGATE_TOKEN_LEN);
    while window_start < head_end && !input.is_char_boundary(window_start) {
        window_start += 1;
    }
    if let Some(rel) = input[window_start..head_end].rfind(SURROGATE_TOKEN_PREFIX) {
        let tok_start = window_start + rel;
        if let Some(tok_end) = match_surrogate_span_at(input, tok_start)
            && tok_end > head_end
        {
            return tok_start;
        }
    }
    head_end
}

/// Snap `tail_start` forwards past any `<surrogate:kind:nonce>` token it would otherwise bisect.
#[must_use]
pub(super) fn snap_tail_outside_surrogate(input: &str, tail_start: usize) -> usize {
    if tail_start == 0 || tail_start >= input.len() {
        return tail_start;
    }
    let mut window_start = tail_start.saturating_sub(MAX_SURROGATE_TOKEN_LEN);
    while window_start < tail_start && !input.is_char_boundary(window_start) {
        window_start += 1;
    }
    if let Some(rel) = input[window_start..tail_start].rfind(SURROGATE_TOKEN_PREFIX) {
        let tok_start = window_start + rel;
        if let Some(tok_end) = match_surrogate_span_at(input, tok_start)
            && tok_end > tail_start
        {
            return tok_end;
        }
    }
    tail_start
}

impl TruncationPolicy {
    /// Truncate `payload` inside `<tag>...</tag>` (with optional pinned `notice`), preserving
    /// the closing `</tag>` boundary and neutralizing any splice breakouts for both `tag`
    /// and `<tool_output_quarantine>`.
    pub(super) fn truncate_and_wrap_envelope<'a>(
        &self,
        env: &DetectedEnvelope<'_>,
        outer_original_len: usize,
        spill: &SpillDisposition,
    ) -> TruncationOutcome<'a> {
        let tag = env.tag;
        let payload = env.payload;
        let payload_len = payload.len();
        let spill_ref = spill.spill_ref().cloned();

        let tag_spec: Cow<'_, str> = if tag == TOOL_OUTPUT_QUARANTINE_TAG {
            Cow::Borrowed(TOOL_OUTPUT_QUARANTINE_TAG)
        } else {
            Cow::Owned(format!("{tag},{TOOL_OUTPUT_QUARANTINE_TAG}"))
        };

        // `<{tag}>\n` + `\n</{tag}>` = tag.len() * 2 + 7 bytes.
        let frame_overhead = tag.len() * 2 + 7;
        let compact_notice = match spill {
            SpillDisposition::Failed(_) => COMPACT_TRUNCATION_MARKER_SPILL_FAILED,
            SpillDisposition::NotConfigured | SpillDisposition::Spilled(_) => {
                COMPACT_TRUNCATION_MARKER
            }
        };

        // Keep the bracketed notice line pinned inside `<tag>` whenever the budget can fit it
        // alongside the outer tags and at least a compact truncation marker.
        let pinned_notice = env.notice.filter(|n| {
            let notice_overhead = n.len() + 1;
            self.max_bytes
                >= frame_overhead
                    .saturating_add(notice_overhead)
                    .saturating_add(compact_notice.len())
        });
        let notice_overhead = pinned_notice.map_or(0, |n| n.len() + 1);
        let inner_budget = self
            .max_bytes
            .saturating_sub(frame_overhead.saturating_add(notice_overhead));

        if inner_budget < MINIMAL_TRUNCATION_MARKER.len() {
            let mut head_end = inner_budget
                .saturating_sub(compact_notice.len())
                .min(payload_len);
            while head_end > 0 && !payload.is_char_boundary(head_end) {
                head_end -= 1;
            }
            head_end = snap_head_outside_surrogate(payload, head_end);

            let mut body = String::with_capacity(head_end + compact_notice.len());
            body.push_str(&payload[..head_end]);
            body.push_str(compact_notice);
            let sanitized_body = md_tmpl_core::sanitize_untrusted_str(&body, &tag_spec);
            let final_body = if head_end > 0 && sanitized_body.len() > inner_budget {
                compact_notice
            } else {
                sanitized_body.as_ref()
            };
            let text = pinned_notice.map_or_else(
                || format!("<{tag}>\n{final_body}\n</{tag}>"),
                |n| format!("<{tag}>\n{n}\n{final_body}\n</{tag}>"),
            );
            return TruncationOutcome {
                text: Cow::Owned(text),
                original_len: outer_original_len,
                omitted: Some(env.offset_bias..(env.offset_bias + payload_len)),
                spill_ref,
            };
        }

        let inner_policy = Self {
            max_bytes: inner_budget,
            head_ratio: self.head_ratio,
            alignment: self.alignment,
            marker_template: self.marker_template.clone(),
            spill_sink: None,
        };
        let inner_outcome =
            inner_policy.truncate_oversized_sanitized_with_tags(payload, spill, &tag_spec);
        let omitted = inner_outcome
            .omitted
            .map(|r| (r.start + env.offset_bias)..(r.end + env.offset_bias))
            .or(Some(env.offset_bias..(env.offset_bias + payload_len)));

        let text = pinned_notice.map_or_else(
            || format!("<{tag}>\n{}\n</{tag}>", inner_outcome.text),
            |n| format!("<{tag}>\n{n}\n{}\n</{tag}>", inner_outcome.text),
        );

        TruncationOutcome {
            text: Cow::Owned(text),
            original_len: outer_original_len,
            omitted,
            spill_ref,
        }
    }
}
