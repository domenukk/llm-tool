//! Provider-specific function-calling wire schema projections (`OpenAI`, `Anthropic`, `Gemini`).
//!
//! Converts normalized [`ToolDefinition`](super::ToolDefinition) instances directly into
//! strongly-typed provider wire schemas (`no_std + alloc` compatible).

use alloc::string::String;

use serde::{Deserialize, Serialize};

use super::ToolDefinition;

/// Canonical `"type"` discriminator for `OpenAI` function-calling tool definitions.
pub const OPENAI_TOOL_TYPE_FUNCTION: &str = "function";

/// Inner `function` specification inside an [`OpenAiTool`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiFunction {
    /// Tool / function name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Normalized JSON Schema describing the function's parameters.
    pub parameters: serde_json::Value,
}

/// `OpenAI` Chat Completions / Responses `tools` wire format (`{"type": "function", "function": {...}}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiTool {
    /// Tool discriminator (always `"function"`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function specification.
    pub function: OpenAiFunction,
}

/// Anthropic Messages API `tools` wire format (`{"name": ..., "description": ..., "input_schema": ...}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Normalized JSON Schema (`"type": "object"`) describing input parameters.
    pub input_schema: serde_json::Value,
}

/// Google Gemini `FunctionDeclaration` wire format (`{"name": ..., "description": ..., "parameters": ...}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeminiFunctionDeclaration {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Normalized JSON Schema describing function parameters (with `$schema` stripped and nullable types flattened).
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// Project this [`ToolDefinition`] into an [`OpenAiTool`] (`{"type": "function", "function": ...}`).
    #[must_use]
    pub fn to_openai_tool(&self) -> OpenAiTool {
        OpenAiTool {
            tool_type: String::from(OPENAI_TOOL_TYPE_FUNCTION),
            function: OpenAiFunction {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self.parameter_schema.clone(),
            },
        }
    }

    /// Project this [`ToolDefinition`] into an [`AnthropicTool`] (`{"name", "description", "input_schema"}`).
    #[must_use]
    pub fn to_anthropic_tool(&self) -> AnthropicTool {
        AnthropicTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.parameter_schema.clone(),
        }
    }

    /// Project this [`ToolDefinition`] into a [`GeminiFunctionDeclaration`] (`{"name", "description", "parameters"}`).
    #[must_use]
    pub fn to_gemini_declaration(&self) -> GeminiFunctionDeclaration {
        GeminiFunctionDeclaration {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameter_schema.clone(),
        }
    }
}
