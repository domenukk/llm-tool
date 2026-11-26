//! Host-pinned surrogate token vault (`<surrogate:kind:nonce>`) for prompt-invisible
//! secret/capability binding and multi-turn continuity verification.
//!
//! Sensitive credentials, session handles, or capability tokens are replaced with
//! cryptographically unpredictable 128-bit `<surrogate:kind:nonce>` handles before tool
//! outputs enter the LLM context window, and resolved back to raw values on the host
//! boundary before tool execution. Any forged, malformed, kind-mismatched, or unknown
//! `<surrogate:...>` token in model-supplied arguments fails closed with a [`ToolError`].

use alloc::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::hash::{Hash, Hasher};

use serde::Serialize;

use super::ToolError;
use crate::compat::{HashMap, RwLock, read_lock, write_lock, write_lock_recover};

/// Prefix of every serialized surrogate token.
pub const SURROGATE_TOKEN_PREFIX: &str = "<surrogate:";

/// Suffix of every serialized surrogate token.
pub const SURROGATE_TOKEN_SUFFIX: &str = ">";

/// Separator between `kind` and `nonce` inside `<surrogate:kind:nonce>`.
const SURROGATE_FIELD_SEP: char = ':';

/// Maximum number of distinct `(kind, raw_value)` entries a [`SurrogateVault`] will store.
pub const MAX_SURROGATE_VAULT_ENTRIES: usize = 4096;

/// Maximum byte length of a surrogate `kind` namespace identifier.
pub const MAX_SURROGATE_KIND_LEN: usize = 64;

/// Exact lowercase hexadecimal character count of a 128-bit surrogate nonce (`16` bytes = `32` hex chars).
pub const SURROGATE_NONCE_HEX_LEN: usize = 32;

/// Maximum byte length of a valid `<surrogate:kind:nonce>` wire token.
pub(crate) const MAX_SURROGATE_TOKEN_LEN: usize = SURROGATE_TOKEN_PREFIX.len()
    + MAX_SURROGATE_KIND_LEN
    + 1
    + SURROGATE_NONCE_HEX_LEN
    + SURROGATE_TOKEN_SUFFIX.len();

const CHACHA_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Zeroize a byte vector in-place without `unsafe` while preventing dead-store elimination.
fn zeroize_byte_vec(buf: &mut Vec<u8>) {
    let cap = buf.capacity();
    if cap > 0 {
        buf.resize(cap, 0);
        for byte in buf.iter_mut() {
            *byte = core::hint::black_box(0u8);
        }
        core::hint::black_box(buf.as_mut_ptr());
        buf.clear();
    }
}

/// Zeroize a fixed 32-byte seed buffer in-place without `unsafe`.
fn zeroize_seed(seed: &mut [u8; 32]) {
    for byte in seed.iter_mut() {
        *byte = core::hint::black_box(0u8);
    }
    core::hint::black_box(seed.as_mut_ptr());
}

/// Heap-allocated UTF-8 secret buffer that overwrites its contents with zeros on [`Drop`].
struct SecretString {
    bytes: Vec<u8>,
}

impl SecretString {
    const fn from_string(s: String) -> Self {
        Self {
            bytes: s.into_bytes(),
        }
    }

    fn as_str(&self) -> &str {
        // Invariant: `SecretString` is constructed exclusively from valid `String`s.
        core::str::from_utf8(&self.bytes).expect("SecretString holds valid UTF-8")
    }

    const fn len(&self) -> usize {
        self.bytes.len()
    }

    const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl Clone for SecretString {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
        }
    }
}

impl PartialEq for SecretString {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl Eq for SecretString {}

impl Hash for SecretString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        zeroize_byte_vec(&mut self.bytes);
    }
}

/// Strongly-typed opaque surrogate handle minted by a [`SurrogateVault`].
///
/// Deliberately omits `Deserialize` so untrusted JSON arguments can never forge a
/// [`SurrogateHandle`] without minting through the host vault.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, schemars::JsonSchema)]
pub struct SurrogateHandle {
    kind: String,
    id: u64,
    token: String,
}

impl SurrogateHandle {
    #[must_use]
    fn new(kind: String, id: u64, nonce_hex: &str) -> Self {
        let token = format!(
            "{SURROGATE_TOKEN_PREFIX}{kind}{SURROGATE_FIELD_SEP}{nonce_hex}{SURROGATE_TOKEN_SUFFIX}"
        );
        Self { kind, id, token }
    }

    /// Return the namespace/kind of this surrogate handle (e.g. `"api_key"`, `"file_cap"`).
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Return the monotonic sequence ID assigned within the vault.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Return the 128-bit (`32`-char lowercase hex) cryptographic nonce of this handle.
    #[must_use]
    pub fn nonce(&self) -> &str {
        let prefix_len = SURROGATE_TOKEN_PREFIX.len() + self.kind.len() + 1;
        let end = self.token.len() - SURROGATE_TOKEN_SUFFIX.len();
        &self.token[prefix_len..end]
    }

    /// Return the wire token string (`"<surrogate:kind:nonce>"`) shown to the model.
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

#[inline]
const fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]);
    state[d] = (state[d] ^ state[a]).rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_left(7);
}

fn chacha20_block(key: &[u8; 32], counter: u32, nonce: [u32; 3]) -> [u32; 16] {
    let mut initial = [0u32; 16];
    initial[0] = CHACHA_CONSTANTS[0];
    initial[1] = CHACHA_CONSTANTS[1];
    initial[2] = CHACHA_CONSTANTS[2];
    initial[3] = CHACHA_CONSTANTS[3];
    for i in 0..8 {
        let off = i * 4;
        initial[4 + i] = u32::from_le_bytes([key[off], key[off + 1], key[off + 2], key[off + 3]]);
    }
    initial[12] = counter;
    initial[13] = nonce[0];
    initial[14] = nonce[1];
    initial[15] = nonce[2];

    let mut working = initial;
    for _ in 0..10 {
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);
        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }
    for i in 0..16 {
        working[i] = working[i].wrapping_add(initial[i]);
    }
    working
}

fn fnv1a_u64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

const fn split_u64_le(val: u64) -> (u32, u32) {
    let b = val.to_le_bytes();
    (
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
    )
}

fn derive_nonce_hex(seed: &[u8; 32], seq_id: u64, kind: &str) -> String {
    let kind_digest = fnv1a_u64(kind.as_bytes());
    let (counter, seq_hi) = split_u64_le(seq_id);
    let (kind_lo, kind_hi) = split_u64_le(kind_digest);

    let block = chacha20_block(seed, counter, [seq_hi, kind_lo, kind_hi]);
    let mut raw_nonce = [0u8; 16];
    for i in 0..4 {
        let word_bytes = block[i].to_le_bytes();
        raw_nonce[i * 4..i * 4 + 4].copy_from_slice(&word_bytes);
    }

    let mut hex = String::with_capacity(SURROGATE_NONCE_HEX_LEN);
    for &byte in &raw_nonce {
        hex.push(char::from(HEX_LOWER[usize::from(byte >> 4)]));
        hex.push(char::from(HEX_LOWER[usize::from(byte & 0x0f)]));
    }
    hex
}

#[cfg(all(feature = "std", unix))]
const DEV_URANDOM_PATH: &str = "/dev/urandom";

/// Reason recorded when the target has no operating-system entropy source.
#[cfg(not(all(feature = "std", any(unix, windows))))]
const NO_OS_ENTROPY_SOURCE: &str = "this target has no operating-system entropy source";

/// Seed state of a vault: 256 secret bits, or the reason none could be obtained.
///
/// A vault without a seed refuses to mint instead of falling back to guessable nonces.
enum VaultSeed {
    Ready([u8; 32]),
    Unavailable(String),
}

impl VaultSeed {
    /// Derive the nonce for `(seq_id, kind)`, failing closed when no secret seed exists.
    fn nonce_hex(&self, seq_id: u64, kind: &str) -> Result<String, ToolError> {
        match self {
            Self::Ready(seed) => Ok(derive_nonce_hex(seed, seq_id, kind)),
            Self::Unavailable(reason) => Err(ToolError::new(format!(
                "SurrogateVault::mint: no secure seed available ({reason}); \
                 create the vault with SurrogateVault::with_seed and 32 bytes from a secure RNG"
            ))),
        }
    }
}

#[cfg(all(feature = "std", unix))]
fn os_seed() -> VaultSeed {
    use std::io::Read;

    let mut seed = [0u8; 32];
    match std::fs::File::open(DEV_URANDOM_PATH).and_then(|mut f| f.read_exact(&mut seed)) {
        Ok(()) => VaultSeed::Ready(seed),
        Err(err) => {
            tracing::warn!(
                error = %err,
                "SurrogateVault: failed to read {DEV_URANDOM_PATH}; minting is disabled for this vault"
            );
            VaultSeed::Unavailable(format!("failed to read {DEV_URANDOM_PATH}: {err}"))
        }
    }
}

#[cfg(all(feature = "std", windows))]
fn os_seed() -> VaultSeed {
    let mut seed = [0u8; 32];
    match getrandom::fill(&mut seed) {
        Ok(()) => VaultSeed::Ready(seed),
        Err(err) => {
            tracing::warn!(
                error = %err,
                "SurrogateVault: OS entropy source failed; minting is disabled for this vault"
            );
            VaultSeed::Unavailable(format!("OS entropy source failed: {err}"))
        }
    }
}

#[cfg(not(all(feature = "std", any(unix, windows))))]
fn os_seed() -> VaultSeed {
    VaultSeed::Unavailable(String::from(NO_OS_ENTROPY_SOURCE))
}

struct VaultEntry {
    kind: String,
    raw: SecretString,
}

struct SurrogateVaultInner {
    seed: VaultSeed,
    next_id: u64,
    /// Maps `"<surrogate:kind:nonce>"` -> `(kind, raw_secret)`.
    token_to_entry: HashMap<String, VaultEntry>,
    /// Maps `(kind, raw_secret)` -> `SurrogateHandle` for deterministic deduplication.
    raw_to_handle: HashMap<(String, SecretString), SurrogateHandle>,
    /// Pre-sorted `(raw_secret, token)` pairs (longest `raw_secret` first) for `redact_str`.
    sorted_pairs: Vec<(SecretString, String)>,
}

impl SurrogateVaultInner {
    fn with_seed(seed: VaultSeed) -> Self {
        Self {
            seed,
            next_id: 0,
            token_to_entry: HashMap::new(),
            raw_to_handle: HashMap::new(),
            sorted_pairs: Vec::new(),
        }
    }

    fn clear_entries(&mut self) {
        self.token_to_entry.clear();
        self.raw_to_handle.clear();
        self.sorted_pairs.clear();
    }
}

impl Default for SurrogateVaultInner {
    fn default() -> Self {
        Self::with_seed(os_seed())
    }
}

impl Drop for SurrogateVaultInner {
    fn drop(&mut self) {
        self.clear_entries();
        if let VaultSeed::Ready(seed) = &mut self.seed {
            zeroize_seed(seed);
        }
    }
}

/// Thread-safe, host-pinned vault mapping opaque `<surrogate:kind:nonce>` tokens to raw values.
#[derive(Clone, Default)]
pub struct SurrogateVault(Arc<RwLock<SurrogateVaultInner>>);

impl core::fmt::Debug for SurrogateVault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let count = match read_lock(&self.0) {
            Ok(g) => g.token_to_entry.len(),
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

/// Validate whether `kind` is a well-formed surrogate namespace (`1..=64` ASCII `[a-zA-Z0-9_-]` bytes).
#[must_use]
pub fn is_valid_surrogate_kind(kind: &str) -> bool {
    !kind.is_empty()
        && kind.len() <= MAX_SURROGATE_KIND_LEN
        && kind
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn is_valid_surrogate_nonce(nonce: &str) -> bool {
    nonce.len() == SURROGATE_NONCE_HEX_LEN
        && nonce
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Strictly parse and validate a `<surrogate:kind:nonce>` token into `(kind, nonce)`.
pub(crate) fn parse_surrogate_token(token: &str) -> Result<(&str, &str), ToolError> {
    let Some(inner) = token
        .strip_prefix(SURROGATE_TOKEN_PREFIX)
        .and_then(|s| s.strip_suffix(SURROGATE_TOKEN_SUFFIX))
    else {
        return Err(ToolError::new(format!(
            "malformed surrogate token '{token}' rejected by continuity check"
        )));
    };
    let Some((kind, nonce)) = inner.split_once(SURROGATE_FIELD_SEP) else {
        return Err(ToolError::new(format!(
            "malformed surrogate token '{token}' (missing kind:nonce separator) rejected by continuity check"
        )));
    };
    if !is_valid_surrogate_kind(kind) || !is_valid_surrogate_nonce(nonce) {
        return Err(ToolError::new(format!(
            "malformed surrogate token '{token}' (invalid kind or 128-bit hex nonce) rejected by continuity check"
        )));
    }
    Ok((kind, nonce))
}

/// If `input[start..]` begins with a syntactically valid `<surrogate:kind:nonce>` token,
/// return the exclusive end byte offset `end` (`start < end <= input.len()`).
pub(crate) fn match_surrogate_span_at(input: &str, start: usize) -> Option<usize> {
    let rest = input.get(start..)?;
    if !rest.starts_with(SURROGATE_TOKEN_PREFIX) {
        return None;
    }
    let window_len = rest.len().min(MAX_SURROGATE_TOKEN_LEN);
    let window = &rest.as_bytes()[..window_len];
    let rel_end = window
        .iter()
        .position(|&b| b == SURROGATE_TOKEN_SUFFIX.as_bytes()[0])?;
    let candidate = rest.get(..=rel_end)?;
    if parse_surrogate_token(candidate).is_ok() {
        Some(start + rel_end + 1)
    } else {
        None
    }
}

#[inline]
fn set_first_byte_bit(bitmap: &mut [u64; 4], byte: u8) {
    let idx = usize::from(byte >> 6);
    let bit = byte & 63;
    bitmap[idx] |= 1u64 << bit;
}

#[inline]
const fn has_first_byte_bit(bitmap: &[u64; 4], byte: u8) -> bool {
    let idx = (byte >> 6) as usize;
    let bit = byte & 63;
    (bitmap[idx] & (1u64 << bit)) != 0
}

impl SurrogateVault {
    /// Create a new, empty [`SurrogateVault`] seeded with 256 bits from the operating system's
    /// secure random source (`/dev/urandom` on Unix, `ProcessPrng` on Windows).
    ///
    /// If no such source exists (for example `no_std` builds or `wasm32-unknown-unknown`) or it
    /// fails, the vault fails closed: [`mint`](Self::mint) returns an error instead of producing
    /// guessable tokens. Use [`with_seed`](Self::with_seed) on those targets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new [`SurrogateVault`] from an explicit 256-bit secret `seed`.
    ///
    /// The seed must come from a secure RNG. Intended for targets without an OS entropy source
    /// (such as `no_std` callers with a hardware RNG) and for deterministic unit tests.
    #[must_use]
    pub fn with_seed(seed: [u8; 32]) -> Self {
        Self(Arc::new(RwLock::new(SurrogateVaultInner::with_seed(
            VaultSeed::Ready(seed),
        ))))
    }

    /// Return the number of distinct surrogate entries currently stored in this vault.
    #[must_use]
    pub fn len(&self) -> usize {
        match read_lock(&self.0) {
            Ok(guard) => guard.token_to_entry.len(),
            Err(err) => {
                tracing::warn!(error = %err, "SurrogateVault::len: lock poisoned");
                0
            }
        }
    }

    /// Return `true` if no surrogate entries are stored in this vault.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wipe and zeroize all stored secret values in this vault while advancing the internal sequence counter.
    pub fn clear(&self) {
        let mut guard = write_lock_recover(&self.0);
        guard.clear_entries();
    }

    /// Mint or retrieve an existing [`SurrogateHandle`] for `(kind, raw_value)`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `kind` is invalid, `raw_value` is empty, the vault has reached
    /// [`MAX_SURROGATE_VAULT_ENTRIES`], the vault has no secure seed (see [`new`](Self::new)),
    /// or the vault lock is poisoned.
    pub fn mint(
        &self,
        kind: &str,
        raw_value: impl Into<String>,
    ) -> Result<SurrogateHandle, ToolError> {
        // Wrap immediately in `SecretString` so early-return validation errors zeroize the buffer.
        let secret = SecretString::from_string(raw_value.into());
        if !is_valid_surrogate_kind(kind) {
            return Err(ToolError::new(format!(
                "SurrogateVault::mint: invalid kind '{kind}' (must be 1..={MAX_SURROGATE_KIND_LEN} bytes matching [a-zA-Z0-9_-]+)"
            )));
        }
        if secret.is_empty() {
            return Err(ToolError::new(
                "SurrogateVault::mint: raw_value must not be empty",
            ));
        }
        let mut guard = write_lock(&self.0).map_err(|e| {
            let msg = format!("SurrogateVault::mint: lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;
        let key = (kind.to_string(), secret.clone());
        if let Some(existing) = guard.raw_to_handle.get(&key) {
            return Ok(existing.clone());
        }
        if guard.token_to_entry.len() >= MAX_SURROGATE_VAULT_ENTRIES {
            return Err(ToolError::new(format!(
                "SurrogateVault::mint: vault capacity exceeded (max {MAX_SURROGATE_VAULT_ENTRIES} entries)"
            )));
        }
        let id = guard
            .next_id
            .checked_add(1)
            .ok_or_else(|| ToolError::new("SurrogateVault::mint: sequence counter exhausted"))?;
        let nonce_hex = guard.seed.nonce_hex(id, kind)?;
        guard.next_id = id;
        let handle = SurrogateHandle::new(kind.to_string(), id, &nonce_hex);
        let insert_idx = guard
            .sorted_pairs
            .binary_search_by(|(probe_raw, _)| {
                secret
                    .len()
                    .cmp(&probe_raw.len())
                    .then_with(|| probe_raw.as_str().cmp(secret.as_str()))
            })
            .unwrap_or_else(|idx| idx);
        guard
            .sorted_pairs
            .insert(insert_idx, (secret.clone(), handle.token.clone()));
        guard.token_to_entry.insert(
            handle.token.clone(),
            VaultEntry {
                kind: kind.to_string(),
                raw: secret,
            },
        );
        guard.raw_to_handle.insert(key, handle.clone());
        drop(guard);
        Ok(handle)
    }

    /// Resolve an exact `<surrogate:kind:nonce>` token to its bound raw value.
    ///
    /// Fails closed with [`ToolError`] if `token` is syntactically invalid or was not minted in this vault.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `token` is malformed, unknown, or the vault lock is poisoned.
    pub fn resolve_token(&self, token: &str) -> Result<String, ToolError> {
        let (_kind, _nonce) = parse_surrogate_token(token)?;
        let guard = read_lock(&self.0).map_err(|e| {
            let msg = format!("SurrogateVault::resolve_token: lock poisoned: {e}");
            tracing::warn!("{msg}");
            ToolError::new(msg)
        })?;
        guard
            .token_to_entry
            .get(token)
            .map(|entry| entry.raw.as_str().to_string())
            .ok_or_else(|| {
                ToolError::new(format!(
                    "unknown or forged surrogate token '{token}' rejected by continuity check"
                ))
            })
    }

    /// Resolve an exact `<surrogate:kind:nonce>` token while requiring its `kind` namespace
    /// to equal `expected_kind`.
    ///
    /// Prevents cross-domain capability confusion (e.g. substituting an `"api_key"` surrogate
    /// where a `"file_cap"` surrogate was required).
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if `expected_kind` is invalid, `token` is malformed, `token`'s kind
    /// does not match `expected_kind`, `token` is unknown, or the vault lock is poisoned.
    pub fn resolve_token_of_kind(
        &self,
        token: &str,
        expected_kind: &str,
    ) -> Result<String, ToolError> {
        if !is_valid_surrogate_kind(expected_kind) {
            return Err(ToolError::new(format!(
                "SurrogateVault::resolve_token_of_kind: invalid expected_kind '{expected_kind}'"
            )));
        }
        let (actual_kind, _nonce) = parse_surrogate_token(token)?;
        if actual_kind != expected_kind {
            return Err(ToolError::new(format!(
                "surrogate token kind mismatch: expected '{expected_kind}', got '{actual_kind}'"
            )));
        }
        let lookup = read_lock(&self.0)
            .map_err(|e| {
                let msg = format!("SurrogateVault::resolve_token_of_kind: lock poisoned: {e}");
                tracing::warn!("{msg}");
                ToolError::new(msg)
            })?
            .token_to_entry
            .get(token)
            .map(|entry| (entry.kind.clone(), entry.raw.as_str().to_string()));
        let Some((entry_kind, raw_val)) = lookup else {
            return Err(ToolError::new(format!(
                "unknown or forged surrogate token '{token}' rejected by continuity check"
            )));
        };
        if entry_kind != expected_kind {
            return Err(ToolError::new(format!(
                "surrogate token kind mismatch: expected '{expected_kind}', got '{entry_kind}'"
            )));
        }
        Ok(raw_val)
    }

    /// Resolve all `<surrogate:kind:nonce>` tokens embedded in `input`.
    ///
    /// Returns `Ok((resolved_string, replacement_count))` or fails closed with
    /// [`ToolError`] if any `<surrogate:...>` token in `input` is unclosed, malformed, or unknown.
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
            let (_kind, _nonce) = parse_surrogate_token(candidate)?;
            let Some(entry) = guard.token_to_entry.get(candidate) else {
                return Err(ToolError::new(format!(
                    "unknown or forged surrogate token '{candidate}' rejected by continuity check"
                )));
            };
            out.push_str(entry.raw.as_str());
            count += 1;
            rest = &after_prefix[end_rel + 1..];
        }
        drop(guard);
        out.push_str(rest);
        Ok((out, count))
    }

    /// Resolve all `<surrogate:kind:nonce>` tokens embedded in `input`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if an unknown/malformed surrogate token is encountered
    /// or the vault lock is poisoned.
    pub fn resolve_in_str(&self, input: &str) -> Result<String, ToolError> {
        self.resolve_in_str_counted(input).map(|(s, _)| s)
    }

    /// Recursively resolve all `<surrogate:kind:nonce>` tokens across object keys and string
    /// values inside `value`.
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
                let has_surrogate_key = map.keys().any(|k| k.contains(SURROGATE_TOKEN_PREFIX));
                if !has_surrogate_key {
                    let mut total = 0usize;
                    for val in map.values_mut() {
                        total += self.resolve_in_json(val)?;
                    }
                    return Ok(total);
                }
                let entries = core::mem::take(map);
                let mut rebuilt = serde_json::Map::with_capacity(entries.len());
                let mut total = 0usize;
                for (key, mut val) in entries {
                    let (resolved_key, key_count) = self.resolve_in_str_counted(&key)?;
                    total += key_count;
                    total += self.resolve_in_json(&mut val)?;
                    rebuilt.insert(resolved_key, val);
                }
                *map = rebuilt;
                Ok(total)
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                Ok(0)
            }
        }
    }

    /// Replace any registered raw values occurring in `output` with their `<surrogate:kind:nonce>`
    /// tokens using a single-pass leftmost-longest non-overlapping scanner that skips existing
    /// `<surrogate:kind:nonce>` spans.
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
        let mut first_bytes = [0u64; 4];
        let mut active_indices: Vec<usize> = Vec::new();
        for (idx, (secret, _)) in guard.sorted_pairs.iter().enumerate() {
            let raw = secret.as_str();
            if output.contains(raw) {
                set_first_byte_bit(&mut first_bytes, raw.as_bytes()[0]);
                active_indices.push(idx);
            }
        }
        if active_indices.is_empty() {
            return Ok(output.to_string());
        }

        let has_existing_surrogate = output.contains(SURROGATE_TOKEN_PREFIX);
        if has_existing_surrogate {
            set_first_byte_bit(&mut first_bytes, b'<');
        }

        let bytes = output.as_bytes();
        let mut out: Option<String> = None;
        let mut last_copied = 0usize;
        let mut pos = 0usize;

        while pos < bytes.len() {
            let b = bytes[pos];
            if !has_first_byte_bit(&first_bytes, b) || !output.is_char_boundary(pos) {
                pos += 1;
                continue;
            }

            // Never split or corrupt an already-minted `<surrogate:kind:nonce>` span.
            if has_existing_surrogate
                && b == b'<'
                && let Some(span_end) = match_surrogate_span_at(output, pos)
            {
                pos = span_end;
                continue;
            }

            let tail = &output[pos..];
            let mut matched: Option<(usize, &str)> = None;
            for &pair_idx in &active_indices {
                let (ref secret, ref token) = guard.sorted_pairs[pair_idx];
                let raw = secret.as_str();
                if raw.as_bytes()[0] == b && tail.starts_with(raw) {
                    matched = Some((raw.len(), token.as_str()));
                    break;
                }
            }

            if let Some((raw_len, token)) = matched {
                let buf = out.get_or_insert_with(|| String::with_capacity(output.len() + 32));
                buf.push_str(&output[last_copied..pos]);
                buf.push_str(token);
                pos += raw_len;
                last_copied = pos;
            } else {
                pos += 1;
            }
        }

        drop(guard);
        let result = out.map_or_else(
            || output.to_string(),
            |mut buf| {
                buf.push_str(&output[last_copied..]);
                buf
            },
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolContext;

    fn unseeded_vault(reason: &str) -> SurrogateVault {
        SurrogateVault(Arc::new(RwLock::new(SurrogateVaultInner::with_seed(
            VaultSeed::Unavailable(String::from(reason)),
        ))))
    }

    #[test]
    fn unseeded_vault_fails_closed_on_mint_but_still_resolves_and_redacts() {
        let vault = unseeded_vault("test: no entropy");

        let err = vault
            .mint("api_key", "sk-live-secret-123")
            .expect_err("a vault without a secure seed must refuse to mint");
        let msg = err.to_string();
        assert!(msg.contains("no secure seed available"), "{msg}");
        assert!(msg.contains("test: no entropy"), "{msg}");
        assert!(msg.contains("SurrogateVault::with_seed"), "{msg}");
        assert!(vault.is_empty(), "a failed mint must not store the secret");

        // Repeated attempts keep failing closed and never consume sequence numbers.
        assert!(vault.mint("api_key", "sk-live-secret-123").is_err());
        let seeded = SurrogateVault::with_seed([1u8; 32]);
        let h = seeded.mint("api_key", "sk-live-secret-123").unwrap();
        assert_eq!(h.id(), 1);

        // Non-minting operations are unaffected and keep failing closed on unknown tokens.
        assert_eq!(
            vault.redact_str("nothing to hide").unwrap(),
            "nothing to hide"
        );
        assert!(vault.resolve_token(h.token()).is_err());

        let ctx = ToolContext::new().with_surrogate_vault(vault);
        assert!(ctx.mint_surrogate("api_key", "sk-live-secret-123").is_err());
    }

    #[cfg(all(feature = "std", any(unix, windows)))]
    #[test]
    fn new_vault_is_seeded_from_os_entropy_source() {
        let vault = SurrogateVault::new();
        let guard = read_lock(&vault.0).expect("fresh vault lock is not poisoned");
        match &guard.seed {
            VaultSeed::Ready(seed) => assert_ne!(seed, &[0u8; 32]),
            VaultSeed::Unavailable(reason) => {
                panic!("OS entropy source must be available on this platform: {reason}")
            }
        }
    }

    #[test]
    fn surrogate_vault_unpredictable_nonces_kind_binding_and_no_deserialize() {
        let seed = [0x42u8; 32];
        let vault = SurrogateVault::with_seed(seed);
        let ctx = ToolContext::new().with_surrogate_vault(vault.clone());

        let h1 = ctx.mint_surrogate("api_key", "sk-live-secret-123").unwrap();
        let h2 = ctx.mint_surrogate("api_key", "sk-live-secret-123").unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.kind(), "api_key");
        assert_eq!(h1.id(), 1);
        assert_eq!(h1.nonce().len(), SURROGATE_NONCE_HEX_LEN);
        assert_ne!(h1.token(), "<surrogate:api_key:1>");
        assert!(h1.token().starts_with("<surrogate:api_key:"));
        assert!(h1.token().ends_with('>'));

        // Same seed + sequence + kind reproduces identical deterministic nonce.
        let deterministic_peer = SurrogateVault::with_seed(seed);
        let peer_h1 = deterministic_peer
            .mint("api_key", "sk-live-secret-123")
            .unwrap();
        assert_eq!(h1.token(), peer_h1.token());

        #[cfg(feature = "std")]
        {
            // Different OS-seeded vaults produce distinct unpredictable nonces.
            let random_vault = SurrogateVault::new();
            let rand_h1 = random_vault.mint("api_key", "sk-live-secret-123").unwrap();
            assert_ne!(h1.token(), rand_h1.token());
        }
        #[cfg(not(feature = "std"))]
        {
            // Without `std`, default vaults fail closed because no OS RNG is available.
            let unseeded = SurrogateVault::new();
            assert!(unseeded.mint("api_key", "sk-live-secret-123").is_err());
        }

        // Kind-bound resolution succeeds on matching kind and fails closed on wrong kind.
        assert_eq!(
            vault
                .resolve_token_of_kind(h1.token(), "api_key")
                .expect("matching kind resolves"),
            "sk-live-secret-123"
        );
        let mismatch = vault
            .resolve_token_of_kind(h1.token(), "file_cap")
            .expect_err("wrong kind must fail closed");
        assert!(mismatch.to_string().contains("kind mismatch"));
    }

    #[test]
    fn redact_str_single_pass_prevents_self_corruption_and_skips_existing_tokens() {
        let vault = SurrogateVault::with_seed([7u8; 32]);
        // Secret 1 mints a token containing kind `mykind`; Secret 2 is literally `"mykind"` and Secret 3 is `"surrogate"`.
        let outer = vault.mint("mykind", "outer_secret_mykind").unwrap();
        let short_k = vault.mint("x", "mykind").unwrap();
        let word_surrogate = vault.mint("y", "surrogate").unwrap();

        let input = " item1=outer_secret_mykind item2=mykind item3=surrogate ";
        let redacted = vault.redact_str(input).unwrap();
        let expected = format!(
            " item1={} item2={} item3={} ",
            outer.token(),
            short_k.token(),
            word_surrogate.token()
        );
        assert_eq!(redacted, expected);

        // Calling redact_str a second time must be 100% idempotent (never corrupting `<surrogate:...>`).
        let redacted_twice = vault.redact_str(&redacted).unwrap();
        assert_eq!(redacted_twice, expected);

        // Round-trip resolution recovers the original input verbatim.
        assert_eq!(vault.resolve_in_str(&redacted_twice).unwrap(), input);
    }

    #[test]
    fn surrogate_vault_resolves_json_object_keys_and_rejects_malformed_tokens() {
        let ctx = ToolContext::new().with_surrogate_vault(SurrogateVault::with_seed([9u8; 32]));
        let hk = ctx.mint_surrogate("hdr", "X-Api-Key").unwrap();
        let hv = ctx.mint_surrogate("api_key", "sk-live-secret-123").unwrap();

        let mut json_val = serde_json::json!({
            hk.token(): format!("Bearer {}", hv.token()),
            "nested": {
                format!("prefix-{}", hk.token()): [hv.token()]
            }
        });
        let count = ctx
            .surrogate_vault()
            .unwrap()
            .resolve_in_json(&mut json_val)
            .unwrap();
        assert_eq!(count, 4);
        assert_eq!(json_val["X-Api-Key"], "Bearer sk-live-secret-123");
        assert_eq!(
            json_val["nested"]["prefix-X-Api-Key"][0],
            "sk-live-secret-123"
        );

        // Sequential integer guess `<surrogate:cap:1>` and uppercase hex are rejected strictly.
        let seq_guess = ctx.resolve_surrogates("use <surrogate:cap:1>").unwrap_err();
        assert!(seq_guess.to_string().contains("malformed surrogate token"));

        let forged_hex = ctx
            .resolve_surrogates("use <surrogate:cap:00000000000000000000000000000000>")
            .unwrap_err();
        assert!(
            forged_hex
                .to_string()
                .contains("unknown or forged surrogate token")
        );

        let mut forged_key_json = serde_json::json!({
            "<surrogate:hdr:00000000000000000000000000000000>": "ok"
        });
        ctx.surrogate_vault()
            .unwrap()
            .resolve_in_json(&mut forged_key_json)
            .expect_err("forged surrogate in JSON object key must fail closed");

        let unclosed = ctx.resolve_surrogates("use <surrogate:cap:1").unwrap_err();
        assert!(unclosed.to_string().contains("unclosed '<surrogate:'"));
    }

    #[test]
    fn shared_surrogate_vault_capacity_bound_and_clear() {
        let vault = SurrogateVault::with_seed([9u8; 32]);
        vault.mint("", "secret").expect_err("empty kind must fail");
        vault
            .mint("bad:kind", "secret")
            .expect_err("colon in kind must fail");
        vault
            .mint(&"k".repeat(MAX_SURROGATE_KIND_LEN + 1), "secret")
            .expect_err("oversized kind must fail");
        vault
            .mint("api_key", "")
            .expect_err("empty raw_value must fail");

        for idx in 0..MAX_SURROGATE_VAULT_ENTRIES {
            vault.mint("cap", format!("secret-{idx}")).unwrap();
        }
        assert_eq!(vault.len(), MAX_SURROGATE_VAULT_ENTRIES);
        let overflow = vault.mint("cap", "one-too-many").unwrap_err();
        assert!(overflow.to_string().contains("capacity exceeded"));

        // Clearing wipes entries and rejects previously minted tokens, while advancing nonces.
        let old_handle = vault.mint("cap", "secret-0").unwrap();
        vault.clear();
        assert!(vault.is_empty());
        assert!(vault.resolve_token(old_handle.token()).is_err());

        let new_handle = vault.mint("cap", "secret-0").unwrap();
        assert_ne!(old_handle.token(), new_handle.token());
    }
}
