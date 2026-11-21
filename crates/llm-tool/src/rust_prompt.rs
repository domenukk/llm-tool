//! Strongly-typed Rust prompt trait and type-erasure machinery.

use alloc::{
    borrow::Cow, boxed::Box, collections::BTreeMap, format, string::String, sync::Arc, vec::Vec,
};
use core::{future::Future, pin::Pin};

use super::types::{
    OutputSanitization, PromptArgumentDefinition, PromptDefinition, PromptOutput, RegistryItem,
    ToolContext, ToolError, ToolProvenance,
};
use crate::compat::{HashMap, HashMapIter};

/// A custom prompt template implemented in Rust with strongly-typed parameters.
pub trait RustPrompt: Send + Sync {
    /// The strongly-typed parameters struct deriving `Deserialize` + `JsonSchema`.
    type Params: serde::de::DeserializeOwned + schemars::JsonSchema + Send;

    /// Unique prompt name.
    const NAME: &'static str;

    /// Human-readable description.
    const DESCRIPTION: &'static str;

    /// Data-provenance and trust classification of the prompt.
    const PROVENANCE: ToolProvenance = ToolProvenance::Trusted;

    /// Shorthand: neutralize control tokens in this prompt's output.
    const SANITIZE_OUTPUT: bool = true;

    /// Shorthand: wrap this prompt's output messages in `<tool_output_quarantine>`.
    const QUARANTINE_OUTPUT: bool = false;

    /// Automatic prompt-injection sanitization policy applied to this prompt's output.
    const OUTPUT_SANITIZATION: OutputSanitization = if Self::QUARANTINE_OUTPUT
        || (Self::SANITIZE_OUTPUT && Self::PROVENANCE.is_untrusted_source())
    {
        OutputSanitization::Quarantine
    } else if Self::SANITIZE_OUTPUT {
        OutputSanitization::SanitizeTokens
    } else {
        OutputSanitization::None
    };

    /// Return the data-provenance tier of the prompt.
    fn provenance(&self) -> ToolProvenance {
        Self::PROVENANCE
    }

    /// Return the automatic prompt-injection sanitization policy for this prompt's output.
    fn output_sanitization(&self) -> OutputSanitization {
        let base = Self::OUTPUT_SANITIZATION;
        if !Self::SANITIZE_OUTPUT || base.is_none() {
            base
        } else {
            base.at_least(self.provenance().default_sanitization())
        }
    }

    /// Return the prompt description.
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::DESCRIPTION)
    }

    /// Render the prompt template with typed parameters.
    fn render(
        &self,
        params: Self::Params,
    ) -> impl Future<Output = Result<PromptOutput, ToolError>> + Send;

    /// Render the prompt template with typed parameters and execution [`ToolContext`].
    fn render_with_context(
        &self,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> impl Future<Output = Result<PromptOutput, ToolError>> + Send {
        self.render(params)
    }

    /// Internal unsanitized invocation hook overridden by `#[llm_prompt]` so registry
    /// dispatch applies output/error sanitization in a single pass.
    #[doc(hidden)]
    fn __render_raw(
        &self,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<PromptOutput, ToolError>> + Send {
        self.render_with_context(params, ctx)
    }
}

/// Strongly-typed representation of a JSON Schema property for prompt arguments.
#[derive(serde::Deserialize)]
struct PromptPropertySchema {
    /// Optional human-readable description from field doc comments.
    description: Option<String>,
}

/// Strongly-typed representation of an object JSON Schema for [`RustPrompt::Params`].
#[derive(serde::Deserialize)]
struct PromptParamsSchema {
    /// JSON Schema `"type"` discriminator (must be `"object"` for prompt parameter structs).
    #[serde(rename = "type")]
    schema_type: Option<String>,
    /// Required parameter names.
    required: Option<Vec<String>>,
    /// Declared object properties in deterministic key order.
    properties: Option<BTreeMap<String, PromptPropertySchema>>,
}

/// Build a sanitized [`PromptDefinition`] from any [`RustPrompt`] implementor.
///
/// # Errors
///
/// Returns `Err` if the prompt's `Params` type fails to produce a valid
/// `"type": "object"` JSON schema.
pub fn definition_of_prompt<T: RustPrompt>(prompt: &T) -> Result<PromptDefinition, ToolError> {
    let schema = schemars::schema_for!(T::Params);
    let val = serde_json::to_value(&schema).map_err(|e| {
        ToolError::new(format!(
            "Failed to serialize schema for prompt '{}': {e}",
            T::NAME
        ))
    })?;
    let parsed: PromptParamsSchema = serde_json::from_value(val).map_err(|e| {
        ToolError::new(format!(
            "Failed to parse object schema for prompt '{}': {e}",
            T::NAME
        ))
    })?;
    if parsed.schema_type.as_deref() != Some("object") {
        return Err(ToolError::new(format!(
            "Prompt '{}' Params schema must have type \"object\", got {:?}",
            T::NAME,
            parsed.schema_type
        )));
    }

    let required_fields = parsed.required.as_deref().unwrap_or(&[]);
    let arguments = parsed
        .properties
        .into_iter()
        .flatten()
        .map(|(name, prop)| {
            let required = required_fields.iter().any(|req| req == &name);
            PromptArgumentDefinition::new(
                name,
                prop.description.unwrap_or_else(String::new),
                required,
            )
        })
        .collect();

    Ok(PromptDefinition::new(
        T::NAME,
        prompt.description().into_owned(),
        arguments,
    ))
}

/// Type-erased future returned by [`ErasedPrompt::render_erased`].
pub type BoxPromptFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PromptOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous prompt storage.
pub trait ErasedPrompt: Send + Sync {
    /// Deserialize arguments and render the prompt template.
    fn render_erased(&self, args: serde_json::Value) -> BoxPromptFuture<'_>;

    /// Deserialize arguments and render the prompt template with caller `ctx` under `effective_policy`.
    fn render_erased_with_policy<'a>(
        &'a self,
        args: serde_json::Value,
        _ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxPromptFuture<'a> {
        let fut = self.render_erased(args);
        Box::pin(async move {
            match fut.await {
                Ok(out) => Ok(effective_policy.apply_prompt(out)),
                Err(err) => Err(effective_policy.apply_error(err)),
            }
        })
    }
}

impl<T: RustPrompt> ErasedPrompt for T {
    fn render_erased(&self, args: serde_json::Value) -> BoxPromptFuture<'_> {
        let policy = self.output_sanitization();
        Box::pin(async move {
            let ctx = ToolContext::new();
            let params: T::Params = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => {
                    let err =
                        ToolError::new(format!("Failed to deserialize prompt parameters: {e}"));
                    return Err(policy.apply_error(err));
                }
            };
            match self.__render_raw(params, &ctx).await {
                Ok(out) => Ok(policy.apply_prompt(out)),
                Err(err) => Err(policy.apply_error(err)),
            }
        })
    }

    fn render_erased_with_policy<'a>(
        &'a self,
        args: serde_json::Value,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxPromptFuture<'a> {
        Box::pin(async move {
            let params: T::Params = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => {
                    let err =
                        ToolError::new(format!("Failed to deserialize prompt parameters: {e}"));
                    return Err(effective_policy.apply_error(err));
                }
            };
            match self.__render_raw(params, ctx).await {
                Ok(out) => Ok(effective_policy.apply_prompt(out)),
                Err(err) => Err(effective_policy.apply_error(err)),
            }
        })
    }
}

#[derive(Clone)]
struct RegisteredPrompt {
    definition: PromptDefinition,
    provenance: ToolProvenance,
    output_sanitization: OutputSanitization,
    erased: Arc<dyn ErasedPrompt>,
}

/// A registry of named prompt templates for dynamic dispatch.
#[derive(Clone)]
pub struct PromptRegistry {
    prompts: HashMap<Cow<'static, str>, RegisteredPrompt>,
    output_floor: OutputSanitization,
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for PromptRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names: Vec<&str> = self.prompts.keys().map(Cow::as_ref).collect();
        f.debug_struct("PromptRegistry")
            .field("prompt_count", &self.prompts.len())
            .field("prompt_names", &names)
            .field("output_floor", &self.output_floor)
            .finish()
    }
}

impl PromptRegistry {
    /// Create an empty prompt registry with secure default [`OutputSanitization::SanitizeTokens`] floor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
            output_floor: OutputSanitization::default(),
        }
    }

    /// Set the minimum [`OutputSanitization`] enforced on every rendered prompt output.
    #[must_use]
    pub const fn with_output_floor(mut self, floor: OutputSanitization) -> Self {
        self.output_floor = floor;
        self
    }

    /// In-place variant of [`with_output_floor`](Self::with_output_floor).
    pub const fn set_output_floor(&mut self, floor: OutputSanitization) -> &mut Self {
        self.output_floor = floor;
        self
    }

    /// Return the minimum [`OutputSanitization`] enforced on every rendered prompt output.
    #[must_use]
    pub const fn output_floor(&self) -> OutputSanitization {
        self.output_floor
    }

    const fn effective_policy_for_entry(&self, entry: &RegisteredPrompt) -> OutputSanitization {
        let prompt_policy = if entry.output_sanitization.is_none() {
            OutputSanitization::None
        } else {
            entry
                .output_sanitization
                .at_least(entry.provenance.default_sanitization())
        };
        prompt_policy.at_least(self.output_floor)
    }

    /// Return the effective [`OutputSanitization`] applied to `name`, or `None` if not registered.
    #[must_use]
    pub fn effective_output_sanitization(&self, name: &str) -> Option<OutputSanitization> {
        self.prompts
            .get(name)
            .map(|entry| self.effective_policy_for_entry(entry))
    }

    /// Return the [`ToolProvenance`] of the registered prompt named `name`, or `None` if not registered.
    #[must_use]
    pub fn provenance(&self, name: &str) -> Option<ToolProvenance> {
        self.prompts.get(name).map(|entry| entry.provenance)
    }

    /// Register a [`RustPrompt`], replacing any existing prompt of the same name.
    ///
    /// # Panics
    ///
    /// Panics if the prompt's JSON schema cannot be serialized. Use
    /// [`try_register`](Self::try_register) for the non-panicking variant.
    pub fn register<P: RustPrompt + 'static>(&mut self, prompt: P) -> &mut Self {
        if let Err(e) = self.try_register(prompt) {
            panic!("Failed to build definition for prompt '{}': {e}", P::NAME);
        }
        self
    }

    /// Register a [`RustPrompt`], returning an error instead of panicking if
    /// the prompt's JSON schema cannot be built.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the prompt's `Params` type fails to produce a JSON schema.
    pub fn try_register<P: RustPrompt + 'static>(
        &mut self,
        prompt: P,
    ) -> Result<&mut Self, ToolError> {
        let definition = definition_of_prompt(&prompt)?;
        let key = Cow::Borrowed(P::NAME);
        let provenance = prompt.provenance();
        let output_sanitization = prompt.output_sanitization();
        self.prompts.insert(
            key,
            RegisteredPrompt {
                definition,
                provenance,
                output_sanitization,
                erased: Arc::new(prompt),
            },
        );
        Ok(self)
    }

    /// Register a [`RustPrompt`], consuming and returning `Self` for chaining.
    #[must_use]
    pub fn with_prompt<P: RustPrompt + 'static>(mut self, prompt: P) -> Self {
        self.register(prompt);
        self
    }

    /// Collect [`PromptDefinition`]s for all registered prompts.
    #[must_use]
    pub fn definitions(&self) -> Vec<PromptDefinition> {
        self.prompts
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    /// Remove a prompt by name, returning `true` if it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        self.prompts.remove(name).is_some()
    }

    /// Clear all registered prompts.
    pub fn clear(&mut self) {
        self.prompts.clear();
    }

    /// Number of registered prompts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Whether the registry has no registered prompts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    /// Whether a prompt with the given name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.prompts.contains_key(name)
    }

    /// Borrow the cached [`PromptDefinition`] for a registered prompt by name.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<&PromptDefinition> {
        self.prompts.get(name).map(|entry| &entry.definition)
    }

    /// Iterate over `(name, definition)` pairs for every registered prompt.
    #[must_use]
    pub fn iter(&self) -> PromptDefinitions<'_> {
        PromptDefinitions {
            inner: self.prompts.iter(),
        }
    }

    /// Render a registered prompt by name with raw JSON arguments.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no prompt named `name` is registered,
    /// or a render error if argument deserialization or rendering fails.
    pub async fn render(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<PromptOutput, ToolError> {
        let ctx = ToolContext::new();
        self.render_with_context(name, args, &ctx).await
    }

    /// Render a registered prompt by name with raw JSON arguments and caller `ctx`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no prompt named `name` is registered,
    /// or a render error if argument deserialization, extension extraction, or rendering fails.
    pub async fn render_with_context(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<PromptOutput, ToolError> {
        let Some(entry) = self.prompts.get(name) else {
            let err = ToolError::not_found(RegistryItem::Prompt, name);
            return Err(self.output_floor.apply_error(err));
        };
        let effective = self.effective_policy_for_entry(entry);
        entry
            .erased
            .render_erased_with_policy(args, ctx, effective)
            .await
    }
}

/// Borrowing iterator over `(name, definition)` pairs, yielded by [`PromptRegistry::iter`].
pub struct PromptDefinitions<'a> {
    inner: HashMapIter<'a, Cow<'static, str>, RegisteredPrompt>,
}

impl<'a> Iterator for PromptDefinitions<'a> {
    type Item = (&'a str, PromptDefinition);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(name, entry)| (name.as_ref(), entry.definition.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for PromptDefinitions<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> IntoIterator for &'a PromptRegistry {
    type Item = (&'a str, PromptDefinition);
    type IntoIter = PromptDefinitions<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NonObjectPrompt;

    impl RustPrompt for NonObjectPrompt {
        type Params = String;
        const NAME: &'static str = "non_object_prompt";
        const DESCRIPTION: &'static str = "Prompt with non-object params";

        fn render(
            &self,
            params: Self::Params,
        ) -> impl Future<Output = Result<PromptOutput, ToolError>> + Send {
            core::future::ready(Ok(PromptOutput::user(params)))
        }
    }

    #[test]
    fn definition_of_prompt_rejects_non_object_schema() {
        let err = definition_of_prompt(&NonObjectPrompt)
            .expect_err("non-object prompt schema must fail loudly");
        assert!(
            err.message.contains("must have type \"object\""),
            "unexpected error: {}",
            err.message
        );
    }
}
