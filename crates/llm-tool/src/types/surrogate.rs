//! Host-pinned surrogate token vault (`<surrogate:kind:id>`) for prompt-invisible
//! secret/capability binding and multi-turn continuity verification.
//!
//! Inspired by Meta AI's Muse multi-turn continuity & capability-binding architecture:
//! sensitive credentials, session handles, or capability tokens are replaced with
//! opaque `<surrogate:kind:id>` handles before tool outputs enter the LLM context
//! window, and resolved back to raw values on the host boundary before tool execution.
//! Any forged or unknown `<surrogate:...>` token in model-supplied arguments fails
//! closed with a [`ToolError`].

use alloc::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use super::ToolError;
use crate::compat::{HashMap, RwLock, read_lock, write_lock};

/// Prefix of every serialized surrogate token.
pub const SURROGATE_TOKEN_PREFIX: &str = "<surrogate:";

/// Suffix of every serialized surrogate token.
pub const SURROGATE_TOKEN_SUFFIX: &str = ">";

/// Strongly-typed opaque surrogate handle minted by a [`SurrogateVault`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SurrogateHandle {
    kind: String,
    id: u64,
    token: String,
}

impl SurrogateHandle {
    /// Construct a handle for `(kind, id)`.
    #[must_use]
    fn new(kind: String, id: u64) -> Self {
        let token = format!("{SURROGATE_TOKEN_PREFIX}{kind}:{id}{SURROGATE_TOKEN_SUFFIX}");
        Self { kind, id, token }
    }

    /// Return the namespace/kind of this surrogate handle (e.g. `"api_key"`, `"file_cap"`).
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Return the numeric ID assigned within the vault.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Return the wire token string (`"<surrogate:kind:id>"`) shown to the model.
    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }
}

impl core::fmt::Display for SurrogateHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.token)
    }
}

#[derive(Default)]
struct SurrogateVaultInner {
    next_id: u64,
    /// Maps `"<surrogate:kind:id>"` -> raw value.
    token_to_raw: HashMap<String, String>,
    /// Maps `(kind, raw_value)` -> `SurrogateHandle` for deterministic deduplication.
    raw_to_handle: HashMap<(String, String), SurrogateHandle>,
    /// Pre-sorted `(raw_value, token)` pairs (longest `raw_value` first) for `redact_str`.
    sorted_pairs: Vec<(String, String)>,
}

/// Thread-safe, host-pinned vault mapping opaque `<surrogate:kind:id>` tokens to raw values.
#[derive(Clone, Default)]
pub struct SurrogateVault(Arc<RwLock<SurrogateVaultInner>>);

impl core::fmt::Debug for SurrogateVault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let count = match read_lock(&self.0) {
            Ok(g) => g.token_to_raw.len(),
            Err(e) => {
                tracing::warn!(error = %e, "SurrogateVault::fmt: lock poisoned");
                0
            }
        };
        f.debug_struct("SurrogateVault")
            .field("entry_count", &count)
            .finish()
    }
}

fn is_valid_surrogate_kind(kind: &str) -> bool {
    !kind.is_empty()
        && kind
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl SurrogateVault {
    /// Create a new, empty [`SurrogateVault`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint or retrieve an existing [`SurrogateHandle`] for `(kind, raw_value)`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `kind` is empty or contains characters outside `[a-zA-Z0-9_-]`,
    /// `raw_value` is empty, or the vault lock is poisoned.
    pub fn mint(
        &self,
        kind: &str,
        raw_value: impl Into<String>,
    ) -> Result<SurrogateHandle, ToolError> {
        if !is_valid_surrogate_kind(kind) {
            return Err(ToolError::new(format!(
                "SurrogateVault::mint: invalid kind '{kind}' (must match [a-zA-Z0-9_-]+)"
            )));
        }
        let raw = raw_value.into();
        if raw.is_empty() {
            return Err(ToolError::new(
                "SurrogateVault::mint: raw_value must not be empty",
            ));
        }
        let mut guard = write_lock(&self.0).map_err(|e| {
            let msg = format!("SurrogateVault::mint: lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;
        let key = (kind.to_string(), raw.clone());
        if let Some(existing) = guard.raw_to_handle.get(&key) {
            return Ok(existing.clone());
        }
        guard.next_id += 1;
        let id = guard.next_id;
        let handle = SurrogateHandle::new(kind.to_string(), id);
        let insert_idx = guard
            .sorted_pairs
            .binary_search_by(|(probe_raw, _)| {
                raw.len()
                    .cmp(&probe_raw.len())
                    .then_with(|| probe_raw.as_str().cmp(raw.as_str()))
            })
            .unwrap_or_else(|idx| idx);
        guard
            .sorted_pairs
            .insert(insert_idx, (raw.clone(), handle.token.clone()));
        guard.token_to_raw.insert(handle.token.clone(), raw);
        guard.raw_to_handle.insert(key, handle.clone());
        drop(guard);
        Ok(handle)
    }

    /// Resolve an exact `<surrogate:kind:id>` token to its bound raw value.
    ///
    /// Fails closed with [`ToolError`] if the token was not minted in this vault.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `token` is unknown or the vault lock is poisoned.
    pub fn resolve_token(&self, token: &str) -> Result<String, ToolError> {
        let guard = read_lock(&self.0).map_err(|e| {
            let msg = format!("SurrogateVault::resolve_token: lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;
        guard.token_to_raw.get(token).cloned().ok_or_else(|| {
            ToolError::new(format!(
                "unknown or forged surrogate token '{token}' rejected by continuity check"
            ))
        })
    }

    /// Resolve all `<surrogate:kind:id>` tokens embedded in `input`.
    ///
    /// Returns `Ok((resolved_string, replacement_count))` or fails closed with
    /// [`ToolError`] if any `<surrogate:...>` token in `input` is malformed or unknown.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if an unknown/malformed surrogate token is encountered
    /// or the vault lock is poisoned.
    pub fn resolve_in_str_counted(&self, input: &str) -> Result<(String, usize), ToolError> {
        if !input.contains(SURROGATE_TOKEN_PREFIX) {
            return Ok((input.to_string(), 0));
        }
        let guard = read_lock(&self.0).map_err(|e| {
            let msg = format!("SurrogateVault::resolve_in_str: lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;

        let mut out = String::with_capacity(input.len());
        let mut rest = input;
        let mut count = 0usize;

        while let Some(start) = rest.find(SURROGATE_TOKEN_PREFIX) {
            out.push_str(&rest[..start]);
            let after_prefix = &rest[start..];
            let Some(end_rel) = after_prefix.find(SURROGATE_TOKEN_SUFFIX) else {
                return Err(ToolError::new(
                    "unclosed '<surrogate:' token in tool input rejected by continuity check",
                ));
            };
            let candidate = &after_prefix[..=end_rel];
            let Some(raw) = guard.token_to_raw.get(candidate) else {
                return Err(ToolError::new(format!(
                    "unknown or forged surrogate token '{candidate}' rejected by continuity check"
                )));
            };
            out.push_str(raw);
            count += 1;
            rest = &after_prefix[end_rel + 1..];
        }
        drop(guard);
        out.push_str(rest);
        Ok((out, count))
    }

    /// Resolve all `<surrogate:kind:id>` tokens embedded in `input`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if an unknown/malformed surrogate token is encountered
    /// or the vault lock is poisoned.
    pub fn resolve_in_str(&self, input: &str) -> Result<String, ToolError> {
        self.resolve_in_str_counted(input).map(|(s, _)| s)
    }

    /// Recursively resolve all `<surrogate:kind:id>` tokens across string values inside `value`.
    ///
    /// Returns the total number of resolved tokens. Fails closed with [`ToolError`] if any
    /// surrogate token is unknown or malformed.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if an unknown/malformed surrogate token is encountered
    /// or the vault lock is poisoned.
    pub fn resolve_in_json(&self, value: &mut serde_json::Value) -> Result<usize, ToolError> {
        match value {
            serde_json::Value::String(s) => {
                let (resolved, count) = self.resolve_in_str_counted(s)?;
                if count > 0 {
                    *s = resolved;
                }
                Ok(count)
            }
            serde_json::Value::Array(items) => {
                let mut total = 0usize;
                for item in items {
                    total += self.resolve_in_json(item)?;
                }
                Ok(total)
            }
            serde_json::Value::Object(map) => {
                let mut total = 0usize;
                for val in map.values_mut() {
                    total += self.resolve_in_json(val)?;
                }
                Ok(total)
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                Ok(0)
            }
        }
    }

    /// Replace any registered raw values occurring in `output` with their `<surrogate:kind:id>`
    /// tokens (longest raw values first so overlapping prefixes cannot leak partial secrets).
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the vault lock is poisoned.
    pub fn redact_str(&self, output: &str) -> Result<String, ToolError> {
        if output.is_empty() {
            return Ok(String::new());
        }
        let guard = read_lock(&self.0).map_err(|e| {
            let msg = format!("SurrogateVault::redact_str: lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;
        if guard.sorted_pairs.is_empty() {
            return Ok(output.to_string());
        }
        let matching_pairs: Vec<(String, String)> = guard
            .sorted_pairs
            .iter()
            .filter(|(raw, _)| output.contains(raw.as_str()))
            .cloned()
            .collect();
        drop(guard);
        if matching_pairs.is_empty() {
            return Ok(output.to_string());
        }

        let mut redacted = output.to_string();
        for (raw, token) in &matching_pairs {
            if redacted.contains(raw.as_str()) {
                redacted = redacted.replace(raw.as_str(), token.as_str());
            }
        }
        Ok(redacted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolContext;

    #[test]
    fn surrogate_vault_mint_deduplicate_resolve_and_redact() {
        let ctx = ToolContext::new();
        let h1 = ctx.mint_surrogate("api_key", "sk-live-secret-123").unwrap();
        let h2 = ctx.mint_surrogate("api_key", "sk-live-secret-123").unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.token(), "<surrogate:api_key:1>");
        assert_eq!(h1.kind(), "api_key");
        assert_eq!(h1.id(), 1);

        let redacted = ctx
            .redact_surrogates("Authorization: Bearer sk-live-secret-123")
            .unwrap();
        assert_eq!(redacted, "Authorization: Bearer <surrogate:api_key:1>");

        let resolved = ctx.resolve_surrogates(&redacted).unwrap();
        assert_eq!(resolved, "Authorization: Bearer sk-live-secret-123");

        let mut json_val = serde_json::json!({
            "headers": ["Bearer <surrogate:api_key:1>"],
            "count": 1
        });
        let count = ctx
            .surrogate_vault()
            .unwrap()
            .resolve_in_json(&mut json_val)
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(json_val["headers"][0], "Bearer sk-live-secret-123");
    }

    #[test]
    fn surrogate_vault_rejects_forged_or_unclosed_tokens() {
        let ctx = ToolContext::new();
        ctx.mint_surrogate("cap", "real-cap").unwrap();

        let forged = ctx
            .resolve_surrogates("use <surrogate:cap:999>")
            .unwrap_err();
        assert!(
            forged
                .to_string()
                .contains("unknown or forged surrogate token")
        );

        let unclosed = ctx.resolve_surrogates("use <surrogate:cap:1").unwrap_err();
        assert!(unclosed.to_string().contains("unclosed '<surrogate:'"));
    }

    #[test]
    fn shared_surrogate_vault_across_contexts_and_validation_rules() {
        let vault = SurrogateVault::new();
        assert!(vault.mint("", "secret").is_err());
        assert!(vault.mint("bad:kind", "secret").is_err());
        assert!(vault.mint("api_key", "").is_err());

        // Mint overlapping prefix secrets to verify longest-match-first redaction.
        let short = vault.mint("api_key", "sk-live").unwrap();
        let long = vault.mint("api_key", "sk-live-extended-999").unwrap();

        let parent_ctx = ToolContext::new().with_surrogate_vault(vault.clone());
        let child_ctx = ToolContext::new().with_surrogate_vault(vault);

        let redacted = parent_ctx
            .redact_surrogates("keys: sk-live-extended-999 and sk-live")
            .unwrap();
        assert_eq!(
            redacted,
            format!("keys: {} and {}", long.token(), short.token())
        );

        let resolved = child_ctx.resolve_surrogates(&redacted).unwrap();
        assert_eq!(resolved, "keys: sk-live-extended-999 and sk-live");
    }
}
