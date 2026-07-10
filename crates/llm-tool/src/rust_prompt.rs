//! Strongly-typed Rust prompt trait and type-erasure machinery.

use alloc::{borrow::Cow, boxed::Box, format, string::ToString, vec::Vec};
use core::{future::Future, pin::Pin};

use super::types::{PromptArgumentDefinition, PromptDefinition, PromptOutput, ToolError};

/// A custom prompt template implemented in Rust with strongly-typed parameters.
pub trait RustPrompt: Send + Sync {
    /// The strongly-typed parameters struct deriving `Deserialize` + `JsonSchema`.
    type Params: serde::de::DeserializeOwned + schemars::JsonSchema + Send;

    /// Unique prompt name.
    const NAME: &'static str;

    /// Human-readable description.
    const DESCRIPTION: &'static str;

    /// Return the prompt description.
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::DESCRIPTION)
    }

    /// Render the prompt template with typed parameters.
    fn render(
        &self,
        params: Self::Params,
    ) -> impl Future<Output = Result<PromptOutput, ToolError>> + Send;
}

/// Build a [`PromptDefinition`] from any [`RustPrompt`] implementor.
#[must_use]
pub fn definition_of_prompt<T: RustPrompt>(prompt: &T) -> PromptDefinition {
    let schema = schemars::schema_for!(T::Params);
    let val = match serde_json::to_value(&schema) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                prompt = T::NAME,
                error = %e,
                "failed to serialize prompt schema, producing empty definition"
            );
            return PromptDefinition {
                name: T::NAME.to_string(),
                description: prompt.description().into_owned(),
                arguments: Vec::new(),
            };
        }
    };
    let mut arguments = Vec::new();

    if let Some(obj) = val.as_object() {
        let required_fields: Vec<&str> = obj
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            // NOLINT: empty vec is correct when schema has no 'required' field
            .unwrap_or_default();
        if let Some(props) = obj.get("properties").and_then(|p| p.as_object()) {
            for (name, prop) in props {
                let description = prop
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                let required = required_fields.contains(&name.as_str());
                arguments.push(PromptArgumentDefinition {
                    name: name.clone(),
                    description,
                    required,
                });
            }
        }
    }

    PromptDefinition {
        name: T::NAME.to_string(),
        description: prompt.description().into_owned(),
        arguments,
    }
}

/// Type-erased future returned by [`ErasedPrompt::render_erased`].
pub type BoxPromptFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PromptOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous prompt storage.
pub trait ErasedPrompt: Send + Sync {
    /// Return the prompt definition.
    fn definition(&self) -> PromptDefinition;

    /// Deserialize arguments and render the prompt template.
    fn render_erased(&self, args: serde_json::Value) -> BoxPromptFuture<'_>;
}

impl<T: RustPrompt> ErasedPrompt for T {
    fn definition(&self) -> PromptDefinition {
        definition_of_prompt(self)
    }

    fn render_erased(&self, args: serde_json::Value) -> BoxPromptFuture<'_> {
        Box::pin(async move {
            let params: T::Params = serde_json::from_value(args).map_err(|e| {
                ToolError::new(format!("Failed to deserialize prompt parameters: {e}"))
            })?;
            self.render(params).await
        })
    }
}
