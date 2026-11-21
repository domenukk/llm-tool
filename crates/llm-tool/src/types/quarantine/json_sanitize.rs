//! Recursive JSON string/key sanitization, JSON escape smuggling neutralization
//! (`\u003c`, `\n\nHuman:`, `\u202e`, `\u0000`), and schema normalization.

use alloc::{
    borrow::Cow,
    string::{String, ToString},
};

use super::{TOOL_OUTPUT_QUARANTINE_TAG, is_preserved_sanitized_envelope};
use crate::compat::HashMap;

const MAX_NESTED_JSON_DEPTH: u8 = 4;
const SCHEMA_META_KEY: &str = "$schema";
const SCHEMA_TITLE_KEY: &str = "title";
const SCHEMA_TYPE_KEY: &str = "type";
const SCHEMA_OBJECT_TYPE: &str = "object";
const SCHEMA_NULL_TYPE: &str = "null";
const SCHEMA_PROPERTIES_KEY: &str = "properties";

/// Returns `true` when `s` (ignoring leading ASCII whitespace) begins with `{`, `[`, or `"`.
#[inline]
pub fn looks_like_json_value(s: &str) -> bool {
    matches!(s.trim_start().as_bytes().first(), Some(b'{' | b'[' | b'"'))
}

fn sanitize_string_with_spec_depth(
    s: &str,
    tag_spec: &str,
    preserve_envelopes: bool,
    depth: u8,
) -> Option<String> {
    if preserve_envelopes && is_preserved_sanitized_envelope(s) {
        return None;
    }
    if depth < MAX_NESTED_JSON_DEPTH && looks_like_json_value(s) {
        let trimmed = s.trim();
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(mut nested) => {
                let changed = sanitize_json_value_with_spec_depth(
                    &mut nested,
                    tag_spec,
                    preserve_envelopes,
                    depth + 1,
                );
                if changed {
                    let reserialized = nested.to_string();
                    return Some(
                        match md_tmpl_core::sanitize_untrusted_str(&reserialized, tag_spec) {
                            Cow::Borrowed(_) => reserialized,
                            Cow::Owned(clean) => clean,
                        },
                    );
                }
                // Valid JSON value whose decoded strings/keys needed zero changes (e.g. a
                // struct field already wrapped in `<tool_output_quarantine>` when `preserve_envelopes` is true).
                return None;
            }
            Err(parse_err) => {
                tracing::trace!(
                    error = %parse_err,
                    "nested string has JSON prefix but is not valid JSON; falling back to string sanitizer"
                );
            }
        }
    }
    match md_tmpl_core::sanitize_untrusted_str(s, tag_spec) {
        Cow::Borrowed(_) => None,
        Cow::Owned(clean) => Some(clean),
    }
}

pub fn sanitize_string_if_needed(s: &str) -> Option<String> {
    sanitize_string_with_spec_depth(s, TOOL_OUTPUT_QUARANTINE_TAG, true, 0)
}

/// Sanitize a raw string slice under `SanitizeTokens` rules, decoding JSON escape sequences
/// (`\u003c`, `\n\nHuman:`, `\u202e`, `\u0000`) when `raw` is valid JSON while preserving
/// already-valid `Untrusted<String>` / `<tool_output_quarantine>` struct fields and returning
/// [`Cow::Borrowed`] when clean.
#[must_use]
pub fn sanitize_tokens_json_aware(raw: &str) -> Cow<'_, str> {
    if is_preserved_sanitized_envelope(raw) {
        return Cow::Borrowed(raw);
    }
    if looks_like_json_value(raw) {
        let trimmed = raw.trim();
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(mut json_val) => {
                let changed = sanitize_json_value_with_spec_depth(
                    &mut json_val,
                    TOOL_OUTPUT_QUARANTINE_TAG,
                    true,
                    1,
                );
                if changed {
                    let serialized = json_val.to_string();
                    return match md_tmpl_core::sanitize_untrusted_str(
                        &serialized,
                        TOOL_OUTPUT_QUARANTINE_TAG,
                    ) {
                        Cow::Borrowed(_) => Cow::Owned(serialized),
                        Cow::Owned(clean) => Cow::Owned(clean),
                    };
                }
                return Cow::Borrowed(raw);
            }
            Err(parse_err) => {
                tracing::trace!(
                    error = %parse_err,
                    "input has JSON prefix but is not valid JSON; falling back to string sanitizer"
                );
            }
        }
    }
    md_tmpl_core::sanitize_untrusted_str(raw, TOOL_OUTPUT_QUARANTINE_TAG)
}

/// Prepare a payload string before wrapping in `<tag_spec>` (quarantine or tagged block),
/// neutralizing JSON-escaped breakout tags (`\u003c/tag\u003e`) without preserving inner
/// matching `<tag>` / `<tool_output_quarantine>` frames.
#[must_use]
pub fn pre_sanitize_json_escapes_for_envelope<'a>(inner: &'a str, tag_spec: &str) -> Cow<'a, str> {
    if looks_like_json_value(inner) {
        let trimmed = inner.trim();
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(mut json_val) => {
                let changed =
                    sanitize_json_value_with_spec_depth(&mut json_val, tag_spec, false, 1);
                if changed {
                    return Cow::Owned(json_val.to_string());
                }
            }
            Err(parse_err) => {
                tracing::trace!(
                    error = %parse_err,
                    "envelope payload has JSON prefix but is not valid JSON; falling back to string sanitizer"
                );
            }
        }
    }
    Cow::Borrowed(inner)
}

fn sanitize_json_value_with_spec_depth(
    val: &mut serde_json::Value,
    tag_spec: &str,
    preserve_envelopes: bool,
    depth: u8,
) -> bool {
    match val {
        serde_json::Value::String(s) => {
            sanitize_string_with_spec_depth(s, tag_spec, preserve_envelopes, depth).is_some_and(
                |clean| {
                    *s = clean;
                    true
                },
            )
        }
        serde_json::Value::Array(arr) => {
            let mut changed = false;
            for item in arr {
                changed |=
                    sanitize_json_value_with_spec_depth(item, tag_spec, preserve_envelopes, depth);
            }
            changed
        }
        serde_json::Value::Object(map) => {
            let mut changed = false;
            for v in map.values_mut() {
                changed |=
                    sanitize_json_value_with_spec_depth(v, tag_spec, preserve_envelopes, depth);
            }
            let has_dirty_key = map
                .keys()
                .any(|k| sanitize_string_with_spec_depth(k, tag_spec, false, depth).is_some());
            if has_dirty_key {
                let entries = core::mem::take(map);
                let mut rebuilt = serde_json::Map::with_capacity(entries.len());
                for (k, v) in entries {
                    let clean_key =
                        sanitize_string_with_spec_depth(&k, tag_spec, false, depth).unwrap_or(k);
                    rebuilt.insert(clean_key, v);
                }
                *map = rebuilt;
                changed = true;
            }
            changed
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            false
        }
    }
}

/// Recursively sanitize all string values and object keys inside `val`, returning `true` if modified.
fn sanitize_json_value_in_place(val: &mut serde_json::Value) -> bool {
    sanitize_json_value_with_spec_depth(val, TOOL_OUTPUT_QUARANTINE_TAG, true, 0)
}

pub fn sanitize_metadata_map_in_place(map: &mut HashMap<String, serde_json::Value>) {
    for val in map.values_mut() {
        sanitize_json_value_in_place(val);
    }
    let has_dirty_key = map.keys().any(|k| {
        sanitize_string_with_spec_depth(k, TOOL_OUTPUT_QUARANTINE_TAG, false, 0).is_some()
    });
    if has_dirty_key {
        let entries = core::mem::take(map);
        let mut rebuilt = HashMap::with_capacity(entries.len());
        for (k, v) in entries {
            let clean_key =
                sanitize_string_with_spec_depth(&k, TOOL_OUTPUT_QUARANTINE_TAG, false, 0)
                    .unwrap_or(k);
            rebuilt.insert(clean_key, v);
        }
        *map = rebuilt;
    }
}

/// Normalize and sanitize a tool parameter JSON Schema for cross-provider compatibility:
/// - Strips top-level `"$schema"` and auto-generated root `"title"`.
/// - Flattens nullable `"type": ["T", "null"]` arrays to scalar `"type": "T"`.
/// - Ensures `"properties": {}` is present on root `"type": "object"` schemas.
/// - Neutralizes any LLM control tokens or invisible characters across schema strings/keys.
pub fn normalize_parameter_schema(schema: &mut serde_json::Value) {
    normalize_schema_node(schema, true, false);
    sanitize_json_value_in_place(schema);
}

fn normalize_schema_node(value: &mut serde_json::Value, is_root: bool, in_properties_map: bool) {
    match value {
        serde_json::Value::Object(map) => {
            if !in_properties_map {
                if is_root {
                    map.remove(SCHEMA_META_KEY);
                    map.remove(SCHEMA_TITLE_KEY);
                }
                let replacement = match map.get(SCHEMA_TYPE_KEY) {
                    Some(serde_json::Value::Array(arr)) => {
                        let non_null = arr
                            .iter()
                            .find(|v| {
                                !matches!(v, serde_json::Value::String(s) if s == SCHEMA_NULL_TYPE)
                            })
                            .cloned();
                        non_null.or_else(|| arr.first().cloned())
                    }
                    _ => None,
                };
                if let Some(val) = replacement {
                    map.insert(String::from(SCHEMA_TYPE_KEY), val);
                }
                if is_root
                    && matches!(
                        map.get(SCHEMA_TYPE_KEY),
                        Some(serde_json::Value::String(t)) if t == SCHEMA_OBJECT_TYPE
                    )
                    && !map.contains_key(SCHEMA_PROPERTIES_KEY)
                {
                    map.insert(
                        String::from(SCHEMA_PROPERTIES_KEY),
                        serde_json::Value::Object(serde_json::Map::new()),
                    );
                }
            }
            for (key, child) in map.iter_mut() {
                let child_is_props = !in_properties_map && key == SCHEMA_PROPERTIES_KEY;
                normalize_schema_node(child, false, child_is_props);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                normalize_schema_node(item, false, false);
            }
        }
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => {}
    }
}
