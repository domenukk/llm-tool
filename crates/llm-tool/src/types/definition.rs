//! Tool execution effect classification and [`ToolDefinition`] metadata.

use alloc::{string::String, sync::Arc};

use serde::{Deserialize, Serialize};

use super::{
    OutputSanitization,
    quarantine::{normalize_parameter_schema, sanitize_string_if_needed},
};
use crate::compat::OnceLock;

/// Effect classification for a tool.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffect {
    /// Read-only, side-effect free invocation. Safe for concurrent wave execution.
    ReadOnly,
    /// Mutates system state (default). Safe for normal execution.
    #[default]
    Mutating,
    /// Destructive operation (e.g. deleting files, wiping data, dropping resources).
    /// Requires explicit confirmation before parallel batch dispatch.
    Destructive,
}

impl ToolEffect {
    /// Returns `true` if the tool effect is read-only / idempotent.
    #[must_use]
    pub const fn is_read_only(self) -> bool {
        matches!(self, Self::ReadOnly)
    }

    /// Returns `true` if the tool effect is destructive.
    #[must_use]
    pub const fn is_destructive(self) -> bool {
        matches!(self, Self::Destructive)
    }
}

/// Trust and data-provenance tier for a tool.
///
/// Ordered monotonically by untrustedness (`Trusted < WorkspaceLocal < PrivateData < ExternalMcp < UntrustedNetwork`)
/// so agent runtimes can compute turn taint monotonically via `max()`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ToolProvenance {
    /// Deterministic internal or trusted host computation.
    #[default]
    Trusted,
    /// Local workspace files or developer tree state.
    WorkspaceLocal,
    /// Authenticated private user/corp data (mail, drive, calendar, secrets).
    PrivateData,
    /// External MCP server or third-party plugin tool.
    ExternalMcp,
    /// Open internet or arbitrary untrusted network content (web fetch, search, browser).
    UntrustedNetwork,
}

impl ToolProvenance {
    /// Returns `true` if the provenance is [`ToolProvenance::Trusted`] (the default).
    #[must_use]
    pub const fn is_trusted(&self) -> bool {
        matches!(self, Self::Trusted)
    }

    /// Returns `true` if the tool ingests untrusted external content (`ExternalMcp` or `UntrustedNetwork`).
    #[must_use]
    pub const fn is_untrusted_source(self) -> bool {
        matches!(self, Self::ExternalMcp | Self::UntrustedNetwork)
    }

    /// Returns `true` if the tool accesses authenticated private data (`PrivateData`).
    #[must_use]
    pub const fn is_private_data(self) -> bool {
        matches!(self, Self::PrivateData)
    }

    /// Recommended default [`OutputSanitization`] for this provenance tier.
    #[must_use]
    pub const fn default_sanitization(self) -> OutputSanitization {
        match self {
            Self::ExternalMcp | Self::UntrustedNetwork => OutputSanitization::Quarantine,
            Self::Trusted | Self::WorkspaceLocal | Self::PrivateData => {
                OutputSanitization::SanitizeTokens
            }
        }
    }

    /// Alias for [`default_sanitization`](Self::default_sanitization) preserved for compatibility.
    #[must_use]
    pub const fn default_output_sanitization(self) -> OutputSanitization {
        self.default_sanitization()
    }
}

/// Describes a custom tool that can be registered with an agent.
#[non_exhaustive]
#[derive(Debug, Serialize)]
pub struct ToolDefinition {
    /// Unique tool name (e.g. `"flash_device"`).
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema describing the tool's parameters.
    pub parameter_schema: serde_json::Value,
    /// Execution effect classification of the tool.
    pub effect: ToolEffect,
    /// Data-provenance and trust classification of the tool.
    #[serde(default, skip_serializing_if = "ToolProvenance::is_trusted")]
    pub provenance: ToolProvenance,
    /// Automatic prompt-injection sanitization policy applied to the tool's output.
    #[serde(default, skip_serializing_if = "OutputSanitization::is_none")]
    pub output_sanitization: OutputSanitization,
    /// Pre-serialized compact JSON Schema string cached lazily on first access.
    #[serde(skip)]
    parameter_schema_json: OnceLock<Arc<str>>,
}

impl Clone for ToolDefinition {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            parameter_schema: self.parameter_schema.clone(),
            effect: self.effect,
            provenance: self.provenance,
            output_sanitization: self.output_sanitization,
            parameter_schema_json: OnceLock::new(),
        }
    }
}

impl PartialEq for ToolDefinition {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.description == other.description
            && self.parameter_schema == other.parameter_schema
            && self.effect == other.effect
            && self.provenance == other.provenance
            && self.output_sanitization == other.output_sanitization
    }
}

impl Eq for ToolDefinition {}

impl<'de> Deserialize<'de> for ToolDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDef {
            name: String,
            description: String,
            parameter_schema: serde_json::Value,
            #[serde(default)]
            effect: Option<ToolEffect>,
            #[serde(default)]
            idempotent: Option<bool>,
            #[serde(default)]
            provenance: ToolProvenance,
            #[serde(default)]
            output_sanitization: Option<OutputSanitization>,
        }

        let mut raw = RawDef::deserialize(deserializer)?;
        let effect = match (raw.effect, raw.idempotent) {
            (Some(eff), _) => eff,
            (None, Some(true)) => ToolEffect::ReadOnly,
            (None, Some(false) | None) => ToolEffect::Mutating,
        };
        let output_sanitization = match raw.output_sanitization {
            Some(OutputSanitization::None) => OutputSanitization::None,
            Some(explicit) => explicit.at_least(raw.provenance.default_sanitization()),
            None => OutputSanitization::default().at_least(raw.provenance.default_sanitization()),
        };

        let name = sanitize_string_if_needed(&raw.name).unwrap_or(raw.name);
        let description = sanitize_string_if_needed(&raw.description).unwrap_or(raw.description);
        normalize_parameter_schema(&mut raw.parameter_schema);

        Ok(Self {
            name,
            description,
            parameter_schema: raw.parameter_schema,
            effect,
            provenance: raw.provenance,
            output_sanitization,
            parameter_schema_json: OnceLock::new(),
        })
    }
}

impl ToolDefinition {
    /// Construct a [`ToolDefinition`] with default [`ToolEffect::Mutating`]
    /// and the secure default [`OutputSanitization::SanitizeTokens`].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameter_schema: serde_json::Value,
    ) -> Self {
        Self::with_effect(name, description, parameter_schema, ToolEffect::Mutating)
    }

    /// Construct a normalized [`ToolDefinition`] directly from a parameter struct `T`
    /// implementing [`schemars::JsonSchema`].
    ///
    /// # Errors
    ///
    /// Returns the [`serde_json::Error`] if the generated schema cannot be converted to a JSON value.
    pub fn from_schema<T: schemars::JsonSchema>(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Self, serde_json::Error> {
        let mut parameter_schema = serde_json::to_value(schemars::schema_for!(T))?;
        normalize_parameter_schema(&mut parameter_schema);
        Ok(Self::new(name, description, parameter_schema))
    }

    /// Construct a [`ToolDefinition`] with an explicit [`ToolEffect`], sanitizing
    /// `name`, `description`, and `parameter_schema` in-place.
    #[must_use]
    pub fn with_effect(
        name: impl Into<String>,
        description: impl Into<String>,
        mut parameter_schema: serde_json::Value,
        effect: ToolEffect,
    ) -> Self {
        let raw_name = name.into();
        let raw_desc = description.into();
        let clean_name = sanitize_string_if_needed(&raw_name).unwrap_or(raw_name);
        let clean_desc = sanitize_string_if_needed(&raw_desc).unwrap_or(raw_desc);
        normalize_parameter_schema(&mut parameter_schema);
        Self {
            name: clean_name,
            description: clean_desc,
            parameter_schema,
            effect,
            provenance: ToolProvenance::default(),
            output_sanitization: OutputSanitization::default(),
            parameter_schema_json: OnceLock::new(),
        }
    }

    /// Attach a [`ToolProvenance`] tier to this [`ToolDefinition`], upgrading
    /// `output_sanitization` via [`OutputSanitization::at_least`] to at least
    /// `provenance.default_sanitization()`.
    #[must_use]
    pub const fn with_provenance(mut self, provenance: ToolProvenance) -> Self {
        self.provenance = provenance;
        self.output_sanitization = self
            .output_sanitization
            .at_least(provenance.default_sanitization());
        self
    }

    /// Attach an [`OutputSanitization`] policy to this [`ToolDefinition`]. Chainable.
    #[must_use]
    pub const fn with_output_sanitization(mut self, sanitization: OutputSanitization) -> Self {
        self.output_sanitization = sanitization;
        self
    }

    /// Update `parameter_schema` (normalizing and sanitizing strings/keys) and invalidate
    /// any cached JSON serialization.
    pub fn set_parameter_schema(&mut self, mut parameter_schema: serde_json::Value) {
        normalize_parameter_schema(&mut parameter_schema);
        self.parameter_schema = parameter_schema;
        self.parameter_schema_json = OnceLock::new();
    }

    /// Borrow `parameter_schema` mutably and invalidate any cached JSON serialization.
    pub fn parameter_schema_mut(&mut self) -> &mut serde_json::Value {
        self.parameter_schema_json = OnceLock::new();
        &mut self.parameter_schema
    }

    /// Return the pre-serialized JSON schema string (`&str`), lazily derived
    /// from `parameter_schema`.
    #[must_use]
    pub fn parameter_schema_json(&self) -> &str {
        self.parameter_schema_json_arc()
    }

    /// Return the pre-serialized JSON schema string (`&Arc<str>`), lazily derived
    /// from `parameter_schema`.
    #[must_use]
    pub fn parameter_schema_json_arc(&self) -> &Arc<str> {
        self.parameter_schema_json.get_or_init(|| {
            match serde_json::to_string(&self.parameter_schema) {
                Ok(s) => Arc::from(s),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        tool = %self.name,
                        "Failed to serialize parameter_schema to JSON"
                    );
                    Arc::from("{}")
                }
            }
        })
    }

    /// Whether repeated invocations with identical arguments are side-effect free / idempotent.
    #[must_use]
    pub const fn is_idempotent(&self) -> bool {
        matches!(self.effect, ToolEffect::ReadOnly)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;

    #[derive(schemars::JsonSchema, serde::Deserialize)]
    struct LookupParams {
        path: String,
        max_lines: u32,
    }

    #[test]
    fn from_schema_derives_normalized_typed_parameter_schema() {
        let def = ToolDefinition::from_schema::<LookupParams>("lookup", "Look up a file")
            .expect("schema serializes");
        assert_eq!(def.name, "lookup");
        assert_eq!(def.description, "Look up a file");
        assert!(def.parameter_schema.get("$schema").is_none());
        assert!(def.parameter_schema.get("title").is_none());
        let props = def.parameter_schema["properties"]
            .as_object()
            .expect("object schema has properties");
        assert!(props.contains_key("path"));
        assert!(props.contains_key("max_lines"));
        assert!(def.parameter_schema_json().contains("\"max_lines\""));

        let parsed: LookupParams =
            serde_json::from_str(r#"{"path":"src/lib.rs","max_lines":40}"#).expect("valid args");
        assert_eq!(parsed.path, "src/lib.rs");
        assert_eq!(parsed.max_lines, 40);
    }

    #[test]
    fn tool_provenance_ordering_and_macro_auto_quarantine() {
        assert!(ToolProvenance::Trusted < ToolProvenance::WorkspaceLocal);
        assert!(ToolProvenance::WorkspaceLocal < ToolProvenance::PrivateData);
        assert!(ToolProvenance::PrivateData < ToolProvenance::ExternalMcp);
        assert!(ToolProvenance::ExternalMcp < ToolProvenance::UntrustedNetwork);
        assert!(!ToolProvenance::PrivateData.is_untrusted_source());
        assert!(ToolProvenance::PrivateData.is_private_data());
        assert!(ToolProvenance::ExternalMcp.is_untrusted_source());
        assert!(ToolProvenance::UntrustedNetwork.is_untrusted_source());

        let manual_def = ToolDefinition::new("web", "desc", serde_json::json!({}))
            .with_provenance(ToolProvenance::UntrustedNetwork);
        assert_eq!(
            manual_def.output_sanitization,
            OutputSanitization::Quarantine
        );

        let deser_def: ToolDefinition = serde_json::from_str(
            r#"{"name":"mcp_tool","description":"d","parameter_schema":{},"provenance":"external_mcp"}"#,
        )
        .expect("deserialize ToolDefinition");
        assert_eq!(deser_def.provenance, ToolProvenance::ExternalMcp);
        assert_eq!(
            deser_def.output_sanitization,
            OutputSanitization::Quarantine
        );

        let deser_opt_out: ToolDefinition = serde_json::from_str(
            r#"{"name":"mcp_raw","description":"d","parameter_schema":{},"provenance":"external_mcp","output_sanitization":"none"}"#,
        )
        .expect("deserialize explicit none");
        assert_eq!(deser_opt_out.output_sanitization, OutputSanitization::None);

        /// Fetches an untrusted web page.
        #[crate::llm_tool(provenance = "untrusted_network", effect = "read_only")]
        fn fetch_web(
            /// Target URL.
            url: String,
        ) -> String {
            format!("body from {url}: <|im_start|>evil")
        }

        let mut reg = crate::ToolRegistry::new();
        reg.register(FetchWeb);
        assert_eq!(
            reg.provenance("fetch_web"),
            Some(ToolProvenance::UntrustedNetwork)
        );
        let def = reg.definition("fetch_web").expect("registered");
        assert_eq!(def.provenance, ToolProvenance::UntrustedNetwork);
        assert_eq!(def.output_sanitization, OutputSanitization::Quarantine);
    }
}
