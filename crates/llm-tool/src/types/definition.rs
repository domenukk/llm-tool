//! Tool execution effect classification and [`ToolDefinition`] metadata.

use alloc::{string::String, sync::Arc};

use serde::{Deserialize, Serialize};

use super::OutputSanitization;
use crate::compat::OnceLock;

/// Effect classification for a tool.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, schemars::JsonSchema,
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
            output_sanitization: OutputSanitization,
        }

        let raw = RawDef::deserialize(deserializer)?;
        let effect = match (raw.effect, raw.idempotent) {
            (Some(eff), _) => eff,
            (None, Some(true)) => ToolEffect::ReadOnly,
            (None, Some(false) | None) => ToolEffect::Mutating,
        };

        Ok(Self {
            name: raw.name,
            description: raw.description,
            parameter_schema: raw.parameter_schema,
            effect,
            output_sanitization: raw.output_sanitization,
            parameter_schema_json: OnceLock::new(),
        })
    }
}

impl ToolDefinition {
    /// Construct a [`ToolDefinition`] with default [`ToolEffect::Mutating`]
    /// and [`OutputSanitization::None`].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameter_schema: serde_json::Value,
    ) -> Self {
        Self::with_effect(name, description, parameter_schema, ToolEffect::Mutating)
    }

    /// Construct a [`ToolDefinition`] with an explicit [`ToolEffect`].
    #[must_use]
    pub fn with_effect(
        name: impl Into<String>,
        description: impl Into<String>,
        parameter_schema: serde_json::Value,
        effect: ToolEffect,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameter_schema,
            effect,
            output_sanitization: OutputSanitization::None,
            parameter_schema_json: OnceLock::new(),
        }
    }

    /// Attach an [`OutputSanitization`] policy to this [`ToolDefinition`]. Chainable.
    #[must_use]
    pub const fn with_output_sanitization(mut self, sanitization: OutputSanitization) -> Self {
        self.output_sanitization = sanitization;
        self
    }

    /// Update `parameter_schema` and invalidate any cached JSON serialization.
    pub fn set_parameter_schema(&mut self, parameter_schema: serde_json::Value) {
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
                    tracing::warn!(error = %err, tool = %self.name, "Failed to serialize parameter_schema to JSON");
                    Arc::from("{}")
                }
            }
        })
    }

    /// Whether repeated invocations with identical arguments are side-effect free / idempotent.
    ///
    /// Preserved for backward compatibility.
    #[must_use]
    pub const fn is_idempotent(&self) -> bool {
        matches!(self.effect, ToolEffect::ReadOnly)
    }
}
