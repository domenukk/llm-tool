//! Strongly-typed Rust prompt trait and type-erasure machinery.

use alloc::{borrow::Cow, boxed::Box, format, string::ToString, vec::Vec};
use core::{future::Future, pin::Pin};

use super::types::{PromptArgumentDefinition, PromptDefinition, PromptOutput, ToolError};
use crate::compat::{HashMap, HashMapIter};

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
///
/// # Errors
///
/// Returns `Err` if the prompt's `Params` type fails to produce a JSON schema.
pub fn definition_of_prompt<T: RustPrompt>(prompt: &T) -> Result<PromptDefinition, ToolError> {
    let schema = schemars::schema_for!(T::Params);
    let val = serde_json::to_value(&schema).map_err(|e| {
        ToolError::new(format!(
            "Failed to serialize schema for prompt '{}': {e}",
            T::NAME
        ))
    })?;
    let mut arguments = Vec::new();

    if let Some(obj) = val.as_object() {
        let required_fields: Vec<&str> = match obj.get("required").and_then(|r| r.as_array()) {
            Some(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
            // An absent `required` field correctly yields no required arguments.
            None => Vec::new(),
        };
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

    Ok(PromptDefinition {
        name: T::NAME.to_string(),
        description: prompt.description().into_owned(),
        arguments,
    })
}

/// Type-erased future returned by [`ErasedPrompt::render_erased`].
pub(crate) type BoxPromptFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PromptOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous prompt storage.
///
/// This is an internal implementation detail of [`PromptRegistry`]; callers
/// interact with prompts through the registry rather than this trait.
pub(crate) trait ErasedPrompt: Send + Sync {
    /// Deserialize arguments and render the prompt template.
    fn render_erased(&self, args: serde_json::Value) -> BoxPromptFuture<'_>;
}

impl<T: RustPrompt> ErasedPrompt for T {
    fn render_erased(&self, args: serde_json::Value) -> BoxPromptFuture<'_> {
        Box::pin(async move {
            let params: T::Params = serde_json::from_value(args).map_err(|e| {
                ToolError::new(format!("Failed to deserialize prompt parameters: {e}"))
            })?;
            self.render(params).await
        })
    }
}

/// A registered prompt: its cached definition plus the type-erased handler.
struct RegisteredPrompt {
    definition: PromptDefinition,
    erased: Box<dyn ErasedPrompt>,
}

/// A registry of named prompt templates for dynamic dispatch.
///
/// Mirrors [`ToolRegistry`](crate::ToolRegistry) for prompts: it stores
/// type-erased [`RustPrompt`] implementations keyed by name, caches each
/// [`PromptDefinition`] at registration time, and renders them on demand,
/// keeping the type-erasure machinery a private implementation detail.
#[derive(Default)]
pub struct PromptRegistry {
    prompts: HashMap<&'static str, RegisteredPrompt>,
}

impl core::fmt::Debug for PromptRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names: Vec<&str> = self.prompts.keys().copied().collect();
        f.debug_struct("PromptRegistry")
            .field("prompt_count", &self.prompts.len())
            .field("prompt_names", &names)
            .finish()
    }
}

impl PromptRegistry {
    /// Create an empty prompt registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
        }
    }

    /// Register a [`RustPrompt`], replacing any existing prompt of the same name.
    ///
    /// Returns `&mut Self` for chaining.
    ///
    /// # Panics
    ///
    /// Panics if the prompt's JSON schema cannot be serialized. This indicates
    /// a bug in the prompt's `Params` type. Use
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
    /// This is the fallible counterpart to [`register`](Self::register). If a
    /// prompt with the same name was already registered, it is replaced.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] if the prompt's `Params` type fails to produce a
    /// JSON schema.
    pub fn try_register<P: RustPrompt + 'static>(
        &mut self,
        prompt: P,
    ) -> Result<&mut Self, ToolError> {
        let definition = definition_of_prompt(&prompt)?;
        self.prompts.insert(
            P::NAME,
            RegisteredPrompt {
                definition,
                erased: Box::new(prompt),
            },
        );
        Ok(self)
    }

    /// Register a [`RustPrompt`], consuming and returning `Self` for chaining.
    ///
    /// # Panics
    ///
    /// Panics if the prompt's JSON schema cannot be serialized; see
    /// [`register`](Self::register).
    #[must_use]
    pub fn with_prompt<P: RustPrompt + 'static>(mut self, prompt: P) -> Self {
        self.register(prompt);
        self
    }

    /// Collect [`PromptDefinition`]s for all registered prompts.
    ///
    /// Returns clones of the cached definitions computed at registration time.
    #[must_use]
    pub fn definitions(&self) -> Vec<PromptDefinition> {
        self.prompts
            .values()
            .map(|entry| entry.definition.clone())
            .collect()
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
    ///
    /// Returns `None` if no prompt named `name` is registered. Unlike
    /// [`definitions`](Self::definitions), this clones nothing.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<&PromptDefinition> {
        self.prompts.get(name).map(|entry| &entry.definition)
    }

    /// Iterate over `(name, definition)` pairs for every registered prompt.
    ///
    /// Yields clones of the cached definitions computed at registration time.
    #[must_use]
    pub fn iter(&self) -> PromptDefinitions<'_> {
        PromptDefinitions {
            inner: self.prompts.iter(),
        }
    }

    /// Render a registered prompt by name with raw JSON arguments.
    ///
    /// Returns `None` if no prompt named `name` is registered; otherwise the
    /// inner `Result` carries the rendered output or a render error.
    ///
    /// # Errors
    ///
    /// The inner `Result` is `Err` if argument deserialization or rendering
    /// fails.
    pub async fn render(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Option<Result<PromptOutput, ToolError>> {
        let entry = self.prompts.get(name)?;
        Some(entry.erased.render_erased(args).await)
    }
}

/// Borrowing iterator over `(name, definition)` pairs, yielded by
/// [`PromptRegistry::iter`] and by `&PromptRegistry`'s [`IntoIterator`] impl.
///
/// Each cached [`PromptDefinition`] is cloned lazily as it is yielded.
pub struct PromptDefinitions<'a> {
    inner: HashMapIter<'a, &'static str, RegisteredPrompt>,
}

impl Iterator for PromptDefinitions<'_> {
    type Item = (&'static str, PromptDefinition);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(name, entry)| (*name, entry.definition.clone()))
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

/// Iterate over `(name, definition)` pairs for every registered prompt.
impl<'a> IntoIterator for &'a PromptRegistry {
    type Item = (&'static str, PromptDefinition);
    type IntoIter = PromptDefinitions<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
