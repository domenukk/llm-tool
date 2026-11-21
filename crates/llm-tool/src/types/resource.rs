//! Resource definitions and read output types.

use alloc::string::{String, ToString};

use serde::{Deserialize, Serialize};

use super::quarantine::sanitize_string_if_needed;

/// Describes a resource or resource template available in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceDefinition {
    /// Resource URI (e.g. `file:///path` or `config://app`) or template pattern.
    #[serde(rename = "uriTemplate")]
    pub uri_template: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Optional MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl ResourceDefinition {
    /// Construct a sanitized [`ResourceDefinition`].
    #[must_use]
    pub fn new(
        uri_template: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        mime_type: Option<String>,
    ) -> Self {
        let raw_uri = uri_template.into();
        let raw_name = name.into();
        let raw_desc = description.into();
        let clean_mime = mime_type.map(|m| sanitize_string_if_needed(&m).unwrap_or(m));
        Self {
            uri_template: sanitize_string_if_needed(&raw_uri).unwrap_or(raw_uri),
            name: sanitize_string_if_needed(&raw_name).unwrap_or(raw_name),
            description: sanitize_string_if_needed(&raw_desc).unwrap_or(raw_desc),
            mime_type: clean_mime,
        }
    }
}

impl<'de> Deserialize<'de> for ResourceDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct RawResourceDef {
            #[serde(rename = "uriTemplate")]
            uri_template: String,
            name: String,
            #[serde(default)]
            description: String,
            mime_type: Option<String>,
        }

        let raw = RawResourceDef::deserialize(deserializer)?;
        Ok(Self::new(
            raw.uri_template,
            raw.name,
            raw.description,
            raw.mime_type,
        ))
    }
}

/// Backwards-compatible type alias for [`ResourceDefinition`].
pub type ResourceTemplateDefinition = ResourceDefinition;

/// A content block inside a resource read output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceOutputContent {
    /// UTF-8 text content.
    #[serde(rename_all = "camelCase")]
    Text {
        /// Resource URI.
        uri: String,
        /// Optional MIME type.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Text string.
        text: String,
    },
    /// Base64 binary content.
    #[serde(rename_all = "camelCase")]
    Blob {
        /// Resource URI.
        uri: String,
        /// Optional MIME type.
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Base64 blob data.
        blob: String,
    },
}

/// The output returned by reading a resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceOutput {
    /// Returned content blocks.
    pub contents: alloc::vec::Vec<ResourceOutputContent>,
}

impl ResourceOutput {
    /// Create a text resource output.
    pub fn text(uri: impl Into<String>, mime_type: Option<&str>, text: impl Into<String>) -> Self {
        Self {
            contents: alloc::vec![ResourceOutputContent::Text {
                uri: uri.into(),
                mime_type: mime_type.map(ToString::to_string),
                text: text.into(),
            }],
        }
    }

    /// Create a binary blob resource output.
    pub fn blob(uri: impl Into<String>, mime_type: Option<&str>, blob: impl Into<String>) -> Self {
        Self {
            contents: alloc::vec![ResourceOutputContent::Blob {
                uri: uri.into(),
                mime_type: mime_type.map(ToString::to_string),
                blob: blob.into(),
            }],
        }
    }
}
