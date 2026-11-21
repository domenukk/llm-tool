//! Quarantine framing and delimiter neutralization for untrusted tool outputs and errors.
//!
//! Provides idempotent XML boundary wrapping (`<tool_output_quarantine>`) and
//! control-token neutralization (including JSON-escaped `\u003c`, `\n\nHuman:`, `\u202e`,
//! and `\u0000` payloads) so untrusted tool responses and error strings cannot inject
//! turn/tool delimiters into the model's token stream.

mod json_sanitize;

use alloc::{
    borrow::Cow,
    boxed::Box,
    format,
    string::{String, ToString},
};

pub(crate) use json_sanitize::{
    normalize_parameter_schema, pre_sanitize_json_escapes_for_envelope,
    sanitize_metadata_map_in_place, sanitize_string_if_needed, sanitize_tokens_json_aware,
};

use crate::{
    compat::{HashMap, RwLock, read_lock_recover, write_lock_recover},
    private::__private::Lazy,
    types::{
        PromptOutput, ResourceOutput, ResourceOutputContent, SpillContext, ToolError, ToolOutput,
        TruncationPolicy,
    },
};

/// Default XML tag name used for quarantined tool output envelopes.
pub const TOOL_OUTPUT_QUARANTINE_TAG: &str = md_tmpl_core::TOOL_OUTPUT_QUARANTINE_TAG;

/// Opening tag of a quarantined tool output envelope.
pub const TOOL_OUTPUT_QUARANTINE_OPEN: &str = md_tmpl_core::TOOL_OUTPUT_QUARANTINE_OPEN_TAG;

/// Closing tag of a quarantined tool output envelope.
pub const TOOL_OUTPUT_QUARANTINE_CLOSE: &str = md_tmpl_core::TOOL_OUTPUT_QUARANTINE_CLOSE_TAG;

/// Opening prefix including trailing newline.
pub(crate) const OPEN_FRAME: &str = "<tool_output_quarantine>\n";

/// Closing suffix including leading newline.
pub(crate) const CLOSE_FRAME: &str = "\n</tool_output_quarantine>";

/// Total byte overhead of the `<tool_output_quarantine>` outer frame (`OPEN_FRAME + CLOSE_FRAME`).
pub(crate) const FRAME_OVERHEAD: usize = OPEN_FRAME.len() + CLOSE_FRAME.len();

/// Maximum byte length allowed for an XML boundary tag name in [`OutputSanitization`].
pub const MAX_SANITIZE_TAG_LEN: usize = 128;

/// Maximum byte length allowed for a custom boundary notice in [`OutputSanitization`].
pub const MAX_SANITIZE_NOTICE_LEN: usize = 2048;

/// Maximum number of distinct dynamic `tag` / `notice` strings interned during
/// [`OutputSanitization`] deserialization.
pub const MAX_INTERNED_SANITIZE_STRINGS: usize = 256;

/// Reserved tag name forbidden for custom [`OutputSanitization`] tags so they never
/// collide with `<surrogate:kind:nonce>` vault handles.
pub const RESERVED_SURROGATE_TAG: &str = "surrogate";

/// Well-known LLM control token and role/tool delimiter replacement rules
/// (re-exported from [`md_tmpl_core::TOKEN_DELIMITERS`], Single Source of Truth).
pub const DELIMITER_RULES: &[(&str, &str)] = md_tmpl_core::TOKEN_DELIMITERS;

/// Canonical default XML tag name used for sanitized untrusted data envelopes.
pub const DEFAULT_SANITIZE_TAG: &str = md_tmpl_core::DEFAULT_SANITIZE_TAG;

/// Canonical default notice placed inside sanitized untrusted boundary envelopes.
pub const DEFAULT_SANITIZE_NOTICE: &str = md_tmpl_core::DEFAULT_SANITIZE_NOTICE;

/// Placeholder substring substituted with the actual tag name in notice templates.
pub const SANITIZE_NOTICE_TAG_PLACEHOLDER: &str = md_tmpl_core::SANITIZE_NOTICE_TAG_PLACEHOLDER;

pub use md_tmpl_core::{
    format_sanitize_notice, is_sanitized_block_str, sanitize_block_idempotent_str,
    sanitize_block_str, sanitize_str, unsanitize_block_str,
};

static INTERNED_SANITIZE_STRINGS: Lazy<RwLock<HashMap<String, &'static str>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Returns `true` if `tag` ASCII case-insensitively equals `"surrogate"`.
#[inline]
#[must_use]
pub const fn is_reserved_surrogate_tag(tag: &str) -> bool {
    let b = tag.as_bytes();
    let expected = RESERVED_SURROGATE_TAG.as_bytes();
    if b.len() != expected.len() {
        return false;
    }
    let mut i = 0;
    while i < expected.len() {
        if (b[i] | 0x20) != expected[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Returns `true` if `name` is a valid non-empty ASCII XML 1.0 `NCName` (`[A-Za-z_][A-Za-z0-9_.-]*`)
/// and is not the reserved `"surrogate"` prefix.
#[must_use]
pub const fn is_valid_xml_ncname(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || is_reserved_surrogate_tag(name) {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return false;
    }
    let mut idx = 1;
    while idx < bytes.len() {
        let b = bytes[idx];
        if !(b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.') {
            return false;
        }
        idx += 1;
    }
    true
}

#[inline]
pub(crate) const fn is_valid_bounded_sanitize_tag(tag: &str) -> bool {
    tag.len() <= MAX_SANITIZE_TAG_LEN && is_valid_xml_ncname(tag)
}

#[inline]
pub(crate) fn is_bracketed_notice_header(line: &str) -> bool {
    let trimmed = line.trim_end_matches('\r');
    trimmed.len() >= 2 && trimmed.starts_with('[') && trimmed.ends_with(']')
}

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

/// Returns `true` if `raw` is already a verified, breakout-free quarantine or sanitized
/// boundary envelope (including free of JSON-escaped breakouts inside JSON bodies) whose
/// outer XML framing should be preserved without re-escaping.
#[must_use]
pub(crate) fn is_preserved_sanitized_envelope(raw: &str) -> bool {
    if !raw.starts_with('<') || !raw.ends_with('>') {
        return false;
    }
    if let Some(inner) = unquarantine_tool_output(raw) {
        return matches!(
            pre_sanitize_json_escapes_for_envelope(inner, TOOL_OUTPUT_QUARANTINE_TAG),
            Cow::Borrowed(_)
        );
    }
    if let Some(inner) = md_tmpl_core::unquarantine_str(raw, Some(DEFAULT_SANITIZE_TAG)) {
        return matches!(
            pre_sanitize_json_escapes_for_envelope(inner, DEFAULT_SANITIZE_TAG),
            Cow::Borrowed(_)
        );
    }
    if let Some(inner) =
        md_tmpl_core::unquarantine_str(raw, Some(md_tmpl_core::UNTRUSTED_TOOL_OUTPUT_TAG))
    {
        return matches!(
            pre_sanitize_json_escapes_for_envelope(inner, md_tmpl_core::UNTRUSTED_TOOL_OUTPUT_TAG),
            Cow::Borrowed(_)
        );
    }
    let Some(gt_idx) = raw[1..].find('>') else {
        return false;
    };
    let candidate_tag = &raw[1..=gt_idx];
    if !is_valid_bounded_sanitize_tag(candidate_tag) || is_control_token_xml_tag(candidate_tag) {
        return false;
    }
    let Some(inner) = unsanitize_block_str(raw, Some(candidate_tag), None) else {
        return false;
    };
    matches!(
        pre_sanitize_json_escapes_for_envelope(inner, candidate_tag),
        Cow::Borrowed(_)
    )
}

fn intern_bounded_sanitize_str(s: String) -> Result<&'static str, &'static str> {
    match s.as_str() {
        "" => return Ok(""),
        DEFAULT_SANITIZE_TAG => return Ok(DEFAULT_SANITIZE_TAG),
        TOOL_OUTPUT_QUARANTINE_TAG => return Ok(TOOL_OUTPUT_QUARANTINE_TAG),
        md_tmpl_core::UNTRUSTED_TOOL_OUTPUT_TAG => {
            return Ok(md_tmpl_core::UNTRUSTED_TOOL_OUTPUT_TAG);
        }
        DEFAULT_SANITIZE_NOTICE => return Ok(DEFAULT_SANITIZE_NOTICE),
        _ => {}
    }

    {
        let guard = read_lock_recover(&INTERNED_SANITIZE_STRINGS);
        if let Some(&existing) = guard.get(s.as_str()) {
            return Ok(existing);
        }
    }

    let mut guard = write_lock_recover(&INTERNED_SANITIZE_STRINGS);
    if let Some(&existing) = guard.get(s.as_str()) {
        return Ok(existing);
    }
    if guard.len() >= MAX_INTERNED_SANITIZE_STRINGS {
        return Err("OutputSanitization intern pool capacity exceeded (max 256 distinct strings)");
    }
    let leaked: &'static str = Box::leak(s.clone().into_boxed_str());
    guard.insert(s, leaked);
    drop(guard);
    Ok(leaked)
}

fn quarantine_after_check(raw: &str) -> String {
    let inner = md_tmpl_core::strip_outer_quarantine_tag(raw.trim(), TOOL_OUTPUT_QUARANTINE_TAG)
        .unwrap_or(raw);
    let prepared = pre_sanitize_json_escapes_for_envelope(inner, TOOL_OUTPUT_QUARANTINE_TAG);
    md_tmpl_core::quarantine_untrusted_str(&prepared, Some(TOOL_OUTPUT_QUARANTINE_TAG))
        .expect("TOOL_OUTPUT_QUARANTINE_TAG is a valid XML tag")
}

fn is_json_safe_sanitized_block(raw: &str, tag: &str, notice: Option<&str>) -> bool {
    let Some(inner) = unsanitize_block_str(raw, Some(tag), notice) else {
        return false;
    };
    let spec = format!("{tag},{TOOL_OUTPUT_QUARANTINE_TAG}");
    matches!(
        pre_sanitize_json_escapes_for_envelope(inner, &spec),
        Cow::Borrowed(_)
    )
}

fn apply_tagged_sanitize_safe(raw: &str, tag: &str, notice: Option<&str>) -> String {
    if !is_valid_bounded_sanitize_tag(tag) {
        tracing::warn!(
            tag,
            "invalid, reserved, or oversized XML tag in OutputSanitization; falling back to quarantine_tool_output"
        );
        return quarantine_tool_output(raw);
    }
    if let Some(n) = notice
        && n.len() > MAX_SANITIZE_NOTICE_LEN
    {
        tracing::warn!(
            tag,
            notice_len = n.len(),
            "oversized notice in OutputSanitization; falling back to quarantine_tool_output"
        );
        return quarantine_tool_output(raw);
    }
    let inner = md_tmpl_core::strip_outer_quarantine_tag(raw.trim(), tag).unwrap_or(raw);
    let spec = format!("{tag},{TOOL_OUTPUT_QUARANTINE_TAG}");
    let prepared = pre_sanitize_json_escapes_for_envelope(inner, &spec);
    match sanitize_block_str(&prepared, Some(tag), notice) {
        Ok(wrapped) => wrapped,
        Err(err) => {
            tracing::warn!(
                tag,
                error = %err,
                "sanitize_block_str failed; falling back to quarantine_tool_output"
            );
            quarantine_tool_output(raw)
        }
    }
}

/// Automatic prompt-injection sanitization policy applied to tool, prompt, and resource outputs.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OutputSanitization {
    /// Do not modify the output. Must be chosen explicitly.
    None,
    /// Neutralize all known LLM control/turn/tool delimiters ([`DELIMITER_RULES`])
    /// and breakout tags in-place without adding outer XML boundary tags (already-valid
    /// boundary envelopes are preserved unchanged). This is the default.
    #[default]
    SanitizeTokens,
    /// Neutralize all known LLM control/turn/tool delimiters ([`DELIMITER_RULES`])
    /// and idempotently wrap the output in a `<tool_output_quarantine>` XML envelope
    /// via [`quarantine_tool_output`].
    Quarantine,
    /// Neutralize control delimiters and wrap output in a `<tag>` boundary envelope
    /// with the default boundary notice.
    SanitizeTagged(&'static str),
    /// Neutralize control delimiters and wrap output in a `<tag>` boundary envelope
    /// with a custom boundary notice.
    SanitizeWithNotice {
        tag: &'static str,
        notice: &'static str,
    },
}

impl<'de> serde::Deserialize<'de> for OutputSanitization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Helper {
            None,
            SanitizeTokens,
            Quarantine,
            SanitizeTagged(String),
            SanitizeWithNotice { tag: String, notice: String },
        }

        fn validate_deserialized_tag<E: serde::de::Error>(tag: &str) -> Result<(), E> {
            if tag.len() > MAX_SANITIZE_TAG_LEN {
                return Err(E::custom(format!(
                    "sanitize tag length ({} bytes) exceeds maximum of {MAX_SANITIZE_TAG_LEN} bytes",
                    tag.len()
                )));
            }
            if is_reserved_surrogate_tag(tag) {
                return Err(E::custom(format!(
                    "sanitize tag '{tag}' is reserved by SurrogateVault"
                )));
            }
            if !is_valid_xml_ncname(tag) {
                return Err(E::custom(format!(
                    "sanitize tag '{tag}' is not a valid ASCII XML NCName"
                )));
            }
            Ok(())
        }

        match Helper::deserialize(deserializer)? {
            Helper::None => Ok(Self::None),
            Helper::SanitizeTokens => Ok(Self::SanitizeTokens),
            Helper::Quarantine => Ok(Self::Quarantine),
            Helper::SanitizeTagged(tag) => {
                validate_deserialized_tag(&tag)?;
                let interned_tag =
                    intern_bounded_sanitize_str(tag).map_err(serde::de::Error::custom)?;
                Ok(Self::SanitizeTagged(interned_tag))
            }
            Helper::SanitizeWithNotice { tag, notice } => {
                validate_deserialized_tag(&tag)?;
                if notice.len() > MAX_SANITIZE_NOTICE_LEN {
                    return Err(serde::de::Error::custom(format!(
                        "sanitize notice length ({} bytes) exceeds maximum of {MAX_SANITIZE_NOTICE_LEN} bytes",
                        notice.len()
                    )));
                }
                let interned_tag =
                    intern_bounded_sanitize_str(tag).map_err(serde::de::Error::custom)?;
                let interned_notice =
                    intern_bounded_sanitize_str(notice).map_err(serde::de::Error::custom)?;
                Ok(Self::SanitizeWithNotice {
                    tag: interned_tag,
                    notice: interned_notice,
                })
            }
        }
    }
}

impl OutputSanitization {
    const fn protection_level(self) -> u8 {
        match self {
            Self::None => 0,
            Self::SanitizeTokens => 1,
            Self::Quarantine | Self::SanitizeTagged(_) | Self::SanitizeWithNotice { .. } => 2,
        }
    }

    /// Return the more protective of `self` and `floor`.
    #[must_use]
    pub const fn at_least(self, floor: Self) -> Self {
        if floor.protection_level() > self.protection_level() {
            floor
        } else {
            self
        }
    }

    /// Returns `true` if the policy is [`OutputSanitization::None`].
    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns `true` if this policy neutralizes control tokens.
    #[must_use]
    pub const fn sanitizes_tokens(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Returns `true` if this policy wraps output in an XML boundary envelope.
    #[must_use]
    pub const fn is_quarantined(self) -> bool {
        matches!(
            self,
            Self::Quarantine | Self::SanitizeTagged(_) | Self::SanitizeWithNotice { .. }
        )
    }

    /// Apply this sanitization policy to a string slice, returning a borrowed [`Cow`]
    /// when no modification is needed.
    #[must_use]
    pub fn apply_str(self, raw: &str) -> Cow<'_, str> {
        match self {
            Self::None => Cow::Borrowed(raw),
            Self::SanitizeTokens => sanitize_tokens_json_aware(raw),
            Self::Quarantine => {
                if is_quarantined_output(raw) {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(quarantine_after_check(raw))
                }
            }
            Self::SanitizeTagged(tag) => {
                if is_valid_bounded_sanitize_tag(tag)
                    && is_json_safe_sanitized_block(raw, tag, None)
                {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(apply_tagged_sanitize_safe(raw, tag, None))
                }
            }
            Self::SanitizeWithNotice { tag, notice } => {
                if is_valid_bounded_sanitize_tag(tag)
                    && notice.len() <= MAX_SANITIZE_NOTICE_LEN
                    && is_json_safe_sanitized_block(raw, tag, Some(notice))
                {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(apply_tagged_sanitize_safe(raw, tag, Some(notice)))
                }
            }
        }
    }

    /// Apply this sanitization policy to a [`ToolOutput`].
    #[must_use]
    pub fn apply(self, output: ToolOutput) -> ToolOutput {
        match self {
            Self::None => output,
            Self::SanitizeTokens => output.sanitized(),
            Self::Quarantine => output.quarantined(),
            Self::SanitizeTagged(tag) => output.sanitized_block(tag),
            Self::SanitizeWithNotice { tag, notice } => output.sanitized_with(tag, notice),
        }
    }

    /// Sanitize `output` according to `self` and truncate it according to `policy` with
    /// caller `spill_ctx`, ensuring full observations spilled to any [`OutputSpillSink`](super::OutputSpillSink)
    /// are already sanitized/quarantined and outer envelopes remain intact within `policy.max_bytes`.
    #[must_use]
    pub fn apply_truncated_with_context(
        self,
        mut output: ToolOutput,
        policy: &TruncationPolicy,
        spill_ctx: &SpillContext<'_>,
    ) -> ToolOutput {
        match self {
            Self::None => {
                let outcome = policy.truncate_lossy_with_context(&output.content, spill_ctx);
                if let Cow::Owned(truncated) = outcome.text {
                    output.content = truncated;
                }
                output
            }
            Self::SanitizeTokens => {
                // First apply JSON-escape-aware token sanitization (preserving Untrusted<T> fields).
                if let Cow::Owned(clean) = sanitize_tokens_json_aware(&output.content) {
                    output.content = clean;
                }
                output.sanitize_metadata_in_place();
                let outcome =
                    policy.truncate_sanitized_lossy_with_context(&output.content, spill_ctx);
                if let Cow::Owned(truncated) = outcome.text {
                    output.content = truncated;
                }
                output
            }
            Self::Quarantine => {
                output.sanitize_metadata_in_place();
                if !is_quarantined_output(&output.content) {
                    output.content = quarantine_after_check(&output.content);
                }
                let outcome = policy.truncate_lossy_with_context(&output.content, spill_ctx);
                if let Cow::Owned(truncated) = outcome.text {
                    output.content = truncated;
                }
                output
            }
            Self::SanitizeTagged(_) | Self::SanitizeWithNotice { .. } => {
                let mut wrapped = self.apply(output);
                let outcome = policy.truncate_lossy_with_context(&wrapped.content, spill_ctx);
                if let Cow::Owned(truncated) = outcome.text {
                    wrapped.content = truncated;
                }
                wrapped
            }
        }
    }

    /// Apply token/delimiter neutralization to a [`ToolError`] when this policy is not [`Self::None`].
    #[must_use]
    pub fn apply_error(self, mut err: ToolError) -> ToolError {
        match self {
            Self::None => err,
            Self::SanitizeTokens | Self::Quarantine => err.sanitized(),
            Self::SanitizeTagged(tag) | Self::SanitizeWithNotice { tag, .. } => {
                if is_valid_bounded_sanitize_tag(tag) && tag != TOOL_OUTPUT_QUARANTINE_TAG {
                    let spec = format!("{TOOL_OUTPUT_QUARANTINE_TAG},{tag}");
                    if !is_preserved_sanitized_envelope(&err.message) {
                        let prepared = pre_sanitize_json_escapes_for_envelope(&err.message, &spec);
                        match md_tmpl_core::sanitize_untrusted_str(&prepared, &spec) {
                            Cow::Owned(clean) => err.message = clean,
                            Cow::Borrowed(_) => {
                                if let Cow::Owned(clean_json) = prepared {
                                    err.message = clean_json;
                                }
                            }
                        }
                    }
                    sanitize_metadata_map_in_place(&mut err.metadata);
                    err
                } else {
                    err.sanitized()
                }
            }
        }
    }

    /// Apply this sanitization policy to both `Ok(ToolOutput)` and `Err(ToolError)`.
    ///
    /// # Errors
    ///
    /// Returns the (sanitized when `self != OutputSanitization::None`) [`ToolError`] when `res` is `Err`.
    pub fn apply_result(self, res: Result<ToolOutput, ToolError>) -> Result<ToolOutput, ToolError> {
        match res {
            Ok(out) => Ok(self.apply(out)),
            Err(err) => Err(self.apply_error(err)),
        }
    }

    /// Apply this sanitization policy across all messages of a [`PromptOutput`].
    #[must_use]
    pub fn apply_prompt(self, mut output: PromptOutput) -> PromptOutput {
        if self.is_none() {
            return output;
        }
        for msg in &mut output.messages {
            if let Cow::Owned(clean) = self.apply_str(&msg.content) {
                msg.content = clean;
            }
        }
        output
    }

    /// Apply this sanitization policy across all content blocks of a [`ResourceOutput`].
    #[must_use]
    pub fn apply_resource(self, mut output: ResourceOutput) -> ResourceOutput {
        if self.is_none() {
            return output;
        }
        for item in &mut output.contents {
            match item {
                ResourceOutputContent::Text {
                    uri,
                    mime_type,
                    text,
                } => {
                    if let Cow::Owned(clean_uri) = Self::SanitizeTokens.apply_str(uri) {
                        *uri = clean_uri;
                    }
                    if let Some(mt) = mime_type
                        && let Cow::Owned(clean_mt) = Self::SanitizeTokens.apply_str(mt)
                    {
                        *mt = clean_mt;
                    }
                    if let Cow::Owned(clean_text) = self.apply_str(text) {
                        *text = clean_text;
                    }
                }
                ResourceOutputContent::Blob {
                    uri,
                    mime_type,
                    blob: _,
                } => {
                    if let Cow::Owned(clean_uri) = Self::SanitizeTokens.apply_str(uri) {
                        *uri = clean_uri;
                    }
                    if let Some(mt) = mime_type
                        && let Cow::Owned(clean_mt) = Self::SanitizeTokens.apply_str(mt)
                    {
                        *mt = clean_mt;
                    }
                }
            }
        }
        output
    }
}

/// Neutralize all known control tokens and quarantine breakout tags in `raw` in a single pass.
#[must_use]
pub fn neutralize_delimiters(raw: &str) -> String {
    sanitize_tokens_json_aware(raw).into_owned()
}

/// Check whether `raw` is free of un-neutralized delimiters and breakout tags.
///
/// Also inspects JSON-escaped sequences inside valid JSON strings and returns
/// `true` when `raw` is already a verified quarantine or sanitized envelope.
#[must_use]
pub fn is_sanitized_output(raw: &str) -> bool {
    is_preserved_sanitized_envelope(raw)
        || matches!(sanitize_tokens_json_aware(raw), Cow::Borrowed(_))
}

/// Check whether `raw` is already a valid, breakout-free `<tool_output_quarantine>` envelope.
///
/// Verifies both raw inner delimiters and JSON-escaped breakout sequences.
#[must_use]
pub fn is_quarantined_output(raw: &str) -> bool {
    let Some(inner) = md_tmpl_core::unquarantine_str(raw, Some(TOOL_OUTPUT_QUARANTINE_TAG)) else {
        return false;
    };
    matches!(
        pre_sanitize_json_escapes_for_envelope(inner, TOOL_OUTPUT_QUARANTINE_TAG),
        Cow::Borrowed(_)
    )
}

/// Extract the inner payload slice from a quarantined tool output string if and
/// only if `raw` is a valid quarantined envelope (as verified by [`is_quarantined_output`]).
#[must_use]
pub fn unquarantine_tool_output(raw: &str) -> Option<&str> {
    let inner = md_tmpl_core::unquarantine_str(raw, Some(TOOL_OUTPUT_QUARANTINE_TAG))?;
    matches!(
        pre_sanitize_json_escapes_for_envelope(inner, TOOL_OUTPUT_QUARANTINE_TAG),
        Cow::Borrowed(_)
    )
    .then_some(inner)
}

/// Wrap `raw` in a `<tool_output_quarantine>` envelope after neutralizing any
/// embedded [`DELIMITER_RULES`], JSON-escaped delimiters, and breakout tags.
#[must_use]
pub fn quarantine_tool_output(raw: &str) -> String {
    if is_quarantined_output(raw) {
        raw.to_string()
    } else {
        quarantine_after_check(raw)
    }
}

impl ToolError {
    /// Return a sanitized copy of this [`ToolError`], neutralizing LLM control tokens,
    /// closing quarantine tags, Bidi overrides, and invisible smuggling characters in
    /// `self.message` (and structured `metadata`) while leaving clean ASCII error strings
    /// 100% byte-for-byte unchanged (`0` allocations).
    #[must_use]
    pub fn sanitized(mut self) -> Self {
        if let Cow::Owned(clean) = sanitize_tokens_json_aware(&self.message) {
            self.message = clean;
        }
        sanitize_metadata_map_in_place(&mut self.metadata);
        self
    }
}

impl PromptOutput {
    /// Return a sanitized copy of this [`PromptOutput`] with LLM control tokens neutralized
    /// across all rendered messages.
    #[must_use]
    pub fn sanitized(self) -> Self {
        OutputSanitization::SanitizeTokens.apply_prompt(self)
    }

    /// Return a quarantined copy of this [`PromptOutput`] with every rendered message wrapped
    /// in `<tool_output_quarantine>`.
    #[must_use]
    pub fn quarantined(self) -> Self {
        OutputSanitization::Quarantine.apply_prompt(self)
    }
}

impl ResourceOutput {
    /// Return a sanitized copy of this [`ResourceOutput`] with LLM control tokens neutralized
    /// across all text content blocks and URIs.
    #[must_use]
    pub fn sanitized(self) -> Self {
        OutputSanitization::SanitizeTokens.apply_resource(self)
    }

    /// Return a quarantined copy of this [`ResourceOutput`] with text content blocks wrapped
    /// in `<tool_output_quarantine>`.
    #[must_use]
    pub fn quarantined(self) -> Self {
        OutputSanitization::Quarantine.apply_resource(self)
    }
}

impl ToolOutput {
    pub(crate) fn sanitize_metadata_in_place(&mut self) {
        sanitize_metadata_map_in_place(&mut self.metadata);
    }

    /// Return a sanitized copy of this [`ToolOutput`], neutralizing control
    /// delimiters ([`DELIMITER_RULES`]), JSON-escaped smuggling sequences, and
    /// `<tool_output_quarantine>` breakout tags while preserving valid
    /// [`Untrusted<String>`](super::Untrusted) struct fields.
    #[must_use]
    pub fn sanitized(mut self) -> Self {
        if let Cow::Owned(clean) = sanitize_tokens_json_aware(&self.content) {
            self.content = clean;
        }
        self.sanitize_metadata_in_place();
        self
    }

    /// Returns `true` if this output's content contains no un-neutralized
    /// [`DELIMITER_RULES`] or `<tool_output_quarantine>` breakout tags (or is
    /// wrapped in a verified boundary envelope).
    #[must_use]
    pub fn is_sanitized(&self) -> bool {
        is_sanitized_output(&self.content)
    }

    /// Return a quarantined copy of this [`ToolOutput`], neutralizing control
    /// delimiters and wrapping the content in `<tool_output_quarantine>` tags.
    #[must_use]
    pub fn quarantined(mut self) -> Self {
        if !is_quarantined_output(&self.content) {
            self.content = quarantine_after_check(&self.content);
        }
        self.sanitize_metadata_in_place();
        self
    }

    /// Returns `true` if this output's content is wrapped in a verified
    /// `<tool_output_quarantine>` envelope with no un-neutralized delimiters.
    #[must_use]
    pub fn is_quarantined(&self) -> bool {
        is_quarantined_output(&self.content)
    }

    /// Return a sanitized copy of this [`ToolOutput`], wrapped in `<tag>` with the default
    /// boundary notice and control tokens neutralized.
    #[must_use]
    pub fn sanitized_block(mut self, tag: &str) -> Self {
        if !(is_valid_bounded_sanitize_tag(tag)
            && is_json_safe_sanitized_block(&self.content, tag, None))
        {
            self.content = apply_tagged_sanitize_safe(&self.content, tag, None);
        }
        self.sanitize_metadata_in_place();
        self
    }

    /// Return a sanitized copy of this [`ToolOutput`], wrapped in `<tag>` with a custom
    /// `notice` and control tokens neutralized.
    #[must_use]
    pub fn sanitized_with(mut self, tag: &str, notice: &str) -> Self {
        if !(is_valid_bounded_sanitize_tag(tag)
            && notice.len() <= MAX_SANITIZE_NOTICE_LEN
            && is_json_safe_sanitized_block(&self.content, tag, Some(notice)))
        {
            self.content = apply_tagged_sanitize_safe(&self.content, tag, Some(notice));
        }
        self.sanitize_metadata_in_place();
        self
    }

    /// Returns `true` if this output is wrapped in a verified `<tag>` boundary envelope.
    #[must_use]
    pub fn is_sanitized_with(&self, tag: &str) -> bool {
        if !is_valid_bounded_sanitize_tag(tag) {
            return false;
        }
        if is_json_safe_sanitized_block(&self.content, tag, None) {
            return true;
        }
        let Some(rest) = self
            .content
            .strip_prefix('<')
            .and_then(|r| r.strip_prefix(tag))
            .and_then(|r| r.strip_prefix('>'))
        else {
            return false;
        };
        let Some(inner) = rest
            .strip_suffix('>')
            .and_then(|r| r.strip_suffix(tag))
            .and_then(|r| r.strip_suffix("</"))
        else {
            return false;
        };
        !md_tmpl_core::has_untrusted_breakout(inner, tag)
            && matches!(
                pre_sanitize_json_escapes_for_envelope(inner, tag),
                Cow::Borrowed(_)
            )
    }

    /// Extract the inner payload if this output is a verified `<tag>` boundary block.
    #[must_use]
    pub fn unsanitized_with<'a>(&'a self, tag: &str) -> Option<&'a str> {
        if !is_valid_bounded_sanitize_tag(tag) {
            return None;
        }
        if let Some(payload) = unsanitize_block_str(&self.content, Some(tag), None) {
            return matches!(
                pre_sanitize_json_escapes_for_envelope(payload, tag),
                Cow::Borrowed(_)
            )
            .then_some(payload);
        }
        let rest = self
            .content
            .strip_prefix('<')
            .and_then(|r| r.strip_prefix(tag))
            .and_then(|r| r.strip_prefix('>'))?;
        let body = rest
            .strip_suffix('>')
            .and_then(|r| r.strip_suffix(tag))
            .and_then(|r| r.strip_suffix("</"))?;
        if md_tmpl_core::has_untrusted_breakout(body, tag) {
            return None;
        }
        let inner = body.strip_prefix('\n').unwrap_or(body);
        let inner = inner.strip_suffix('\n').unwrap_or(inner);
        let payload = match inner.split_once('\n') {
            Some((first_line, after)) if is_bracketed_notice_header(first_line) => after,
            _ => inner,
        };
        matches!(
            pre_sanitize_json_escapes_for_envelope(payload, tag),
            Cow::Borrowed(_)
        )
        .then_some(payload)
    }

    /// Returns `true` if this output is wrapped in `<tag>` with the exact specified `notice`.
    #[must_use]
    pub fn is_sanitized_with_notice(&self, tag: &str, notice: &str) -> bool {
        is_valid_bounded_sanitize_tag(tag)
            && is_json_safe_sanitized_block(&self.content, tag, Some(notice))
    }

    /// Extract the inner payload if this output is wrapped in `<tag>` with the exact specified `notice`.
    #[must_use]
    pub fn unsanitized_with_notice<'a>(&'a self, tag: &str, notice: &str) -> Option<&'a str> {
        if !is_valid_bounded_sanitize_tag(tag) {
            return None;
        }
        let payload = unsanitize_block_str(&self.content, Some(tag), Some(notice))?;
        matches!(
            pre_sanitize_json_escapes_for_envelope(payload, tag),
            Cow::Borrowed(_)
        )
        .then_some(payload)
    }

    /// Sanitize and truncate this [`ToolOutput`] according to `policy`.
    #[must_use]
    pub fn sanitized_truncated(mut self, policy: &TruncationPolicy) -> Self {
        if let Cow::Owned(clean) = sanitize_tokens_json_aware(&self.content) {
            self.content = clean;
        }
        if let Cow::Owned(truncated) = policy.truncate_sanitized(&self.content) {
            self.content = truncated;
        }
        self.sanitize_metadata_in_place();
        self
    }

    /// Quarantine and truncate this [`ToolOutput`] according to `policy`.
    #[must_use]
    pub fn quarantined_truncated(mut self, policy: &TruncationPolicy) -> Self {
        self.content = policy.truncate_quarantined(&self.content);
        self.sanitize_metadata_in_place();
        self
    }
}
