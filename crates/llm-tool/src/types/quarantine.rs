//! Quarantine framing and delimiter neutralization for untrusted tool outputs.
//!
//! Provides idempotent XML boundary wrapping (`<tool_output_quarantine>`) and
//! control-token neutralization so untrusted tool responses cannot inject `ChatML`,
//! Harmony, Llama, or `DeepSeek` turn/tool delimiters into the model's token stream.

use alloc::{borrow::Cow, string::String};

use crate::types::{ToolOutput, TruncationPolicy};

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

/// Automatic prompt-injection sanitization policy applied to tool outputs.
///
/// **Secure by default:** the default is [`OutputSanitization::SanitizeTokens`].
/// Raw, unmodified output requires an explicit opt-out.
///
/// Configure it on a tool via:
/// - `#[llm_tool(quarantine)]` / `#[llm_tool(sanitize = "tag")]` for stronger envelopes
/// - `#[llm_tool(sanitization = "none")]` (or `sanitize = false`) to opt out
/// - [`RustTool::OUTPUT_SANITIZATION`](crate::RustTool::OUTPUT_SANITIZATION) on manual impls
///
/// [`ToolRegistry::with_output_floor`](crate::ToolRegistry::with_output_floor) enforces a
/// registry-wide minimum (also `SanitizeTokens` by default), so a single tool that opts out
/// still gets the floor unless the registry floor is lowered too.
///
/// Or on individual return values / struct fields via [`Sanitized<T>`](super::Sanitized)
/// and [`Untrusted<T>`](super::Untrusted) ([`Quarantined<T>`](super::Quarantined)).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OutputSanitization {
    /// Do not modify the tool output. Must be chosen explicitly.
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

        match Helper::deserialize(deserializer)? {
            Helper::None => Ok(Self::None),
            Helper::SanitizeTokens => Ok(Self::SanitizeTokens),
            Helper::Quarantine => Ok(Self::Quarantine),
            Helper::SanitizeTagged(tag) => Ok(Self::SanitizeTagged(alloc::boxed::Box::leak(
                tag.into_boxed_str(),
            ))),
            Helper::SanitizeWithNotice { tag, notice } => Ok(Self::SanitizeWithNotice {
                tag: alloc::boxed::Box::leak(tag.into_boxed_str()),
                notice: alloc::boxed::Box::leak(notice.into_boxed_str()),
            }),
        }
    }
}

impl OutputSanitization {
    /// Protection level used to compare policies: raw < token-neutralized < enveloped.
    const fn protection_level(self) -> u8 {
        match self {
            Self::None => 0,
            Self::SanitizeTokens => 1,
            Self::Quarantine | Self::SanitizeTagged(_) | Self::SanitizeWithNotice { .. } => 2,
        }
    }

    /// Return the more protective of `self` and `floor`.
    ///
    /// Ties keep `self`, so a tool's own envelope (e.g. a custom tag) is never
    /// replaced by an equally strong floor.
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
    ///
    /// # Panics
    ///
    /// Panics if a tagged sanitization policy was configured with a tag that is not a valid ASCII XML `NCName`.
    #[must_use]
    pub fn apply_str(self, raw: &str) -> Cow<'_, str> {
        match self {
            Self::None => Cow::Borrowed(raw),
            Self::SanitizeTokens => {
                if is_quarantined_output(raw) {
                    Cow::Borrowed(raw)
                } else {
                    md_tmpl_core::sanitize_untrusted_str(raw, TOOL_OUTPUT_QUARANTINE_TAG)
                }
            }
            Self::Quarantine => {
                if is_quarantined_output(raw) {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(quarantine_tool_output(raw))
                }
            }
            Self::SanitizeTagged(tag) => {
                if is_sanitized_block_str(raw, Some(tag), None) {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(
                        sanitize_block_str(raw, Some(tag), None).expect("tag is a valid XML tag"),
                    )
                }
            }
            Self::SanitizeWithNotice { tag, notice } => {
                if is_sanitized_block_str(raw, Some(tag), Some(notice)) {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(
                        sanitize_block_str(raw, Some(tag), Some(notice))
                            .expect("tag is a valid XML tag"),
                    )
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
}

/// Neutralize all known control tokens and quarantine breakout tags in `raw` in a single pass.
#[must_use]
pub fn neutralize_delimiters(raw: &str) -> String {
    md_tmpl_core::sanitize_untrusted_str(raw, TOOL_OUTPUT_QUARANTINE_TAG).into_owned()
}

/// Check whether `raw` is free of un-neutralized [`DELIMITER_RULES`] and
/// `<tool_output_quarantine>` breakout tags, or is already a verified
/// `<tool_output_quarantine>` envelope.
#[must_use]
pub fn is_sanitized_output(raw: &str) -> bool {
    is_quarantined_output(raw)
        || !md_tmpl_core::has_untrusted_breakout(raw, TOOL_OUTPUT_QUARANTINE_TAG)
}

/// Check whether `raw` is already a valid, un-escaped `<tool_output_quarantine>`
/// envelope whose inner body contains zero un-neutralized [`DELIMITER_RULES`] or
/// boundary breakout tags.
///
/// A naive `starts_with` / `ends_with` check is unsound on its own because an
/// attacker payload like `<tool_output_quarantine>\nevil\n</tool_output_quarantine>\n<tool_call>...`
/// or `<tool_output_quarantine>\n</tool_output_quarantine><tool_call>\n</tool_output_quarantine>`
/// would pass the outer check while smuggling raw delimiters inside the body.
#[must_use]
pub fn is_quarantined_output(raw: &str) -> bool {
    md_tmpl_core::is_quarantined_str(raw, Some(TOOL_OUTPUT_QUARANTINE_TAG))
}

/// Extract the inner payload slice from a quarantined tool output string if and
/// only if `raw` is a valid quarantined envelope (as verified by [`is_quarantined_output`]).
#[must_use]
pub fn unquarantine_tool_output(raw: &str) -> Option<&str> {
    md_tmpl_core::unquarantine_str(raw, Some(TOOL_OUTPUT_QUARANTINE_TAG))
}

/// Wrap `raw` in a `<tool_output_quarantine>` envelope after neutralizing any
/// embedded [`DELIMITER_RULES`] and breakout tags in a single pass.
///
/// This function is **idempotent**: if `raw` already satisfies [`is_quarantined_output`],
/// it is returned unchanged without double-wrapping.
///
/// # Panics
///
/// Panics if the internal quarantine tag specification is invalid XML.
#[must_use]
pub fn quarantine_tool_output(raw: &str) -> String {
    md_tmpl_core::quarantine_untrusted_idempotent_str(raw, Some(TOOL_OUTPUT_QUARANTINE_TAG))
        .expect("TOOL_OUTPUT_QUARANTINE_TAG is a valid XML tag")
}

impl ToolOutput {
    /// Return a sanitized copy of this [`ToolOutput`], neutralizing control
    /// delimiters ([`DELIMITER_RULES`]) and `<tool_output_quarantine>` breakout tags
    /// in a single pass without wrapping in outer XML tags.
    ///
    /// Zero-allocation when the content contains no delimiters, and idempotent when
    /// called on an already-sanitized or already-quarantined output.
    #[must_use]
    pub fn sanitized(mut self) -> Self {
        if !is_quarantined_output(&self.content)
            && let Cow::Owned(clean) =
                md_tmpl_core::sanitize_untrusted_str(&self.content, TOOL_OUTPUT_QUARANTINE_TAG)
        {
            self.content = clean;
        }
        self
    }

    /// Returns `true` if this output's content contains no un-neutralized
    /// [`DELIMITER_RULES`] or `<tool_output_quarantine>` breakout tags (or is
    /// wrapped in a verified `<tool_output_quarantine>` envelope).
    #[must_use]
    pub fn is_sanitized(&self) -> bool {
        is_sanitized_output(&self.content)
    }

    /// Return a quarantined copy of this [`ToolOutput`], neutralizing control
    /// delimiters and wrapping the content in `<tool_output_quarantine>` tags.
    ///
    /// Idempotent when called on an already-quarantined output.
    #[must_use]
    pub fn quarantined(mut self) -> Self {
        self.content = quarantine_tool_output(&self.content);
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
    ///
    /// Idempotent when called on an already-sanitized block with matching `tag`.
    ///
    /// # Panics
    ///
    /// Panics if `tag` is not a valid ASCII XML `NCName`.
    #[must_use]
    pub fn sanitized_block(mut self, tag: &str) -> Self {
        if !is_sanitized_block_str(&self.content, Some(tag), None) {
            self.content =
                sanitize_block_str(&self.content, Some(tag), None).expect("tag is a valid XML tag");
        }
        self
    }

    /// Return a sanitized copy of this [`ToolOutput`], wrapped in `<tag>` with a custom
    /// `notice` and control tokens neutralized.
    ///
    /// Idempotent when called on an already-sanitized block with matching `tag`.
    ///
    /// # Panics
    ///
    /// Panics if `tag` is not a valid ASCII XML `NCName`.
    #[must_use]
    pub fn sanitized_with(mut self, tag: &str, notice: &str) -> Self {
        if !is_sanitized_block_str(&self.content, Some(tag), Some(notice)) {
            self.content = sanitize_block_str(&self.content, Some(tag), Some(notice))
                .expect("tag is a valid XML tag");
        }
        self
    }

    /// Returns `true` if this output is wrapped in a verified `<tag>` boundary envelope.
    #[must_use]
    pub fn is_sanitized_with(&self, tag: &str) -> bool {
        if is_sanitized_block_str(&self.content, Some(tag), None) {
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
    }

    /// Extract the inner payload if this output is a verified `<tag>` boundary block.
    #[must_use]
    pub fn unsanitized_with<'a>(&'a self, tag: &str) -> Option<&'a str> {
        if let Some(payload) = unsanitize_block_str(&self.content, Some(tag), None) {
            return Some(payload);
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
        let inner = body.strip_prefix('\n').unwrap_or(body);
        let inner = inner.strip_suffix('\n').unwrap_or(inner);
        let payload = if let Some((_notice_line, after)) = inner.split_once('\n') {
            after.strip_prefix('\n').unwrap_or(after)
        } else {
            inner
        };
        if md_tmpl_core::has_untrusted_breakout(payload, tag) {
            return None;
        }
        Some(payload)
    }

    /// Returns `true` if this output is wrapped in `<tag>` with the exact specified `notice`.
    #[must_use]
    pub fn is_sanitized_with_notice(&self, tag: &str, notice: &str) -> bool {
        is_sanitized_block_str(&self.content, Some(tag), Some(notice))
    }

    /// Extract the inner payload if this output is wrapped in `<tag>` with the exact specified `notice`.
    #[must_use]
    pub fn unsanitized_with_notice<'a>(&'a self, tag: &str, notice: &str) -> Option<&'a str> {
        unsanitize_block_str(&self.content, Some(tag), Some(notice))
    }

    /// Sanitize and truncate this [`ToolOutput`] according to `policy`, ensuring
    /// that control tokens and any split tokens formed across the truncation cut
    /// boundary are neutralized while respecting `policy.max_bytes`.
    #[must_use]
    pub fn sanitized_truncated(mut self, policy: &TruncationPolicy) -> Self {
        self.content = policy.truncate_sanitized(&self.content).into_owned();
        self
    }

    /// Quarantine and truncate this [`ToolOutput`] according to `policy`, preserving
    /// the outer `<tool_output_quarantine>` envelope and neutralizing any split
    /// tokens formed across the truncation cut boundary while fitting the total
    /// envelope inside `policy.max_bytes`.
    #[must_use]
    pub fn quarantined_truncated(mut self, policy: &TruncationPolicy) -> Self {
        self.content = policy.truncate_quarantined(&self.content);
        self
    }
}
