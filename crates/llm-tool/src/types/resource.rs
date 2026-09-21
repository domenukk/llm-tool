//! Resource definitions and read output types.

use alloc::string::{String, ToString};

use serde::{Deserialize, Serialize};

/// Describes a resource or resource template available in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
