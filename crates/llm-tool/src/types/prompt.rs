use alloc::string::String;

use serde::{Deserialize, Serialize};

/// Describes a prompt template available in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptDefinition {
    /// Prompt name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Arguments accepted by this prompt.
    #[serde(default, skip_serializing_if = "alloc::vec::Vec::is_empty")]
    pub arguments: alloc::vec::Vec<PromptArgumentDefinition>,
}

/// An argument accepted by a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptArgumentDefinition {
    /// Argument name.
    pub name: String,
    /// Argument description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Whether this argument is required.
    #[serde(default)]
    pub required: bool,
}

/// The role of a message in a prompt output (`user`, `assistant`, or `system`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptRole {
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// System message.
    System,
}

impl PromptRole {
    /// The string slice representation of the role (`"user"`, `"assistant"`, or `"system"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

impl core::fmt::Display for PromptRole {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A rendered message inside a prompt output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptOutputMessage {
    /// Message role.
    pub role: PromptRole,
    /// Text content.
    pub content: String,
}

impl PromptOutputMessage {
    /// Create a user role message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant role message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Assistant,
            content: content.into(),
        }
    }

    /// Create a system role message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::System,
            content: content.into(),
        }
    }
}

/// The output returned by rendering a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptOutput {
    /// Rendered messages.
    pub messages: alloc::vec::Vec<PromptOutputMessage>,
}

impl PromptOutput {
    /// Create a new prompt output with a single user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            messages: alloc::vec![PromptOutputMessage::user(content)],
        }
    }

    /// Create a new prompt output with a single assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            messages: alloc::vec![PromptOutputMessage::assistant(content)],
        }
    }

    /// Create a new prompt output with a single system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            messages: alloc::vec![PromptOutputMessage::system(content)],
        }
    }
}

impl From<String> for PromptOutput {
    fn from(content: String) -> Self {
        Self::user(content)
    }
}

impl From<&str> for PromptOutput {
    fn from(content: &str) -> Self {
        Self::user(content)
    }
}
