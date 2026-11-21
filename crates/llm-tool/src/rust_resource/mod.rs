//! Strongly-typed Rust resource trait and URI template matching.

mod uri_de;

use alloc::{
    borrow::Cow,
    boxed::Box,
    collections::BTreeMap,
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::{future::Future, pin::Pin};

use uri_de::UriVarDeserializer;

use super::types::{
    OutputSanitization, RegistryItem, ResourceDefinition, ResourceOutput, ToolContext, ToolError,
    ToolProvenance,
};

/// A custom resource or resource template implemented in Rust.
pub trait RustResource: Send + Sync {
    /// Strongly-typed parameters extracted from URI variables.
    type Params: serde::de::DeserializeOwned + Send;

    /// URI template pattern (e.g. `"file:///logs/{date}/{app}.log"` or `"config://app"`).
    const URI_TEMPLATE: &'static str;

    /// Unique resource name.
    const NAME: &'static str;

    /// Human-readable description.
    const DESCRIPTION: &'static str;

    /// Optional MIME type.
    const MIME_TYPE: Option<&'static str>;

    /// Data-provenance and trust classification of the resource.
    const PROVENANCE: ToolProvenance = ToolProvenance::Trusted;

    /// Shorthand: neutralize control tokens in this resource's output.
    const SANITIZE_OUTPUT: bool = true;

    /// Shorthand: wrap this resource's text content in `<tool_output_quarantine>`.
    const QUARANTINE_OUTPUT: bool = false;

    /// Automatic prompt-injection sanitization policy applied to this resource's output.
    const OUTPUT_SANITIZATION: OutputSanitization = if Self::QUARANTINE_OUTPUT
        || (Self::SANITIZE_OUTPUT && Self::PROVENANCE.is_untrusted_source())
    {
        OutputSanitization::Quarantine
    } else if Self::SANITIZE_OUTPUT {
        OutputSanitization::SanitizeTokens
    } else {
        OutputSanitization::None
    };

    /// Return the data-provenance tier of the resource.
    fn provenance(&self) -> ToolProvenance {
        Self::PROVENANCE
    }

    /// Return the automatic prompt-injection sanitization policy for this resource's output.
    fn output_sanitization(&self) -> OutputSanitization {
        let base = Self::OUTPUT_SANITIZATION;
        if !Self::SANITIZE_OUTPUT || base.is_none() {
            base
        } else {
            base.at_least(self.provenance().default_sanitization())
        }
    }

    /// Return the resource description.
    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(Self::DESCRIPTION)
    }

    /// Read the resource with URI and extracted parameters.
    fn read(
        &self,
        uri: &str,
        params: Self::Params,
    ) -> impl Future<Output = Result<ResourceOutput, ToolError>> + Send;

    /// Read the resource with URI, extracted parameters, and execution [`ToolContext`].
    fn read_with_context(
        &self,
        uri: &str,
        params: Self::Params,
        _ctx: &ToolContext,
    ) -> impl Future<Output = Result<ResourceOutput, ToolError>> + Send {
        self.read(uri, params)
    }

    /// Internal unsanitized invocation hook overridden by `#[llm_resource]` so registry
    /// dispatch applies output/error sanitization in a single pass.
    #[doc(hidden)]
    fn __read_raw(
        &self,
        uri: &str,
        params: Self::Params,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<ResourceOutput, ToolError>> + Send {
        self.read_with_context(uri, params, ctx)
    }
}

/// Build a sanitized [`ResourceDefinition`] from any [`RustResource`] implementor.
#[must_use]
pub fn definition_of_resource<T: RustResource>(resource: &T) -> ResourceDefinition {
    ResourceDefinition::new(
        T::URI_TEMPLATE,
        T::NAME,
        resource.description().into_owned(),
        T::MIME_TYPE.map(ToString::to_string),
    )
}

const TEMPLATE_VAR_START: char = '{';
const TEMPLATE_VAR_END: char = '}';
const EMPTY_STR: &str = "";

/// A single segment of a pre-compiled URI template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateSegment<'a> {
    /// Literal substring that must match identically.
    Literal(&'a str),
    /// Named placeholder variable (e.g. `date` from `{date}`).
    Variable(&'a str),
}

/// Pre-compiled representation of a URI template pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledUriTemplate<'a> {
    raw: &'a str,
    segments: Vec<TemplateSegment<'a>>,
    var_count: usize,
}

impl<'a> CompiledUriTemplate<'a> {
    /// Compile a URI template string into its constituent segments.
    #[must_use]
    pub fn new(template: &'a str) -> Self {
        if !template.contains(TEMPLATE_VAR_START) {
            return Self {
                raw: template,
                segments: alloc::vec![TemplateSegment::Literal(template)],
                var_count: 0,
            };
        }

        let mut segments = Vec::new();
        let mut rem = template;
        let mut var_count = 0;

        while let Some(start_idx) = rem.find(TEMPLATE_VAR_START) {
            if start_idx > 0 {
                segments.push(TemplateSegment::Literal(&rem[..start_idx]));
            }
            let after_start = &rem[start_idx + 1..];
            if let Some(end_idx) = after_start.find(TEMPLATE_VAR_END) {
                let var_name = &after_start[..end_idx];
                segments.push(TemplateSegment::Variable(var_name));
                var_count += 1;
                rem = &after_start[end_idx + 1..];
            } else {
                segments.push(TemplateSegment::Literal(&rem[start_idx..]));
                rem = EMPTY_STR;
                break;
            }
        }
        if !rem.is_empty() {
            segments.push(TemplateSegment::Literal(rem));
        }

        Self {
            raw: template,
            segments,
            var_count,
        }
    }

    /// Return the original unparsed template string.
    #[must_use]
    pub const fn raw(&self) -> &'a str {
        self.raw
    }

    /// Return the parsed template segments.
    #[must_use]
    pub fn segments(&self) -> &[TemplateSegment<'a>] {
        &self.segments
    }

    /// Return the number of placeholder variables in this template.
    #[must_use]
    pub const fn var_count(&self) -> usize {
        self.var_count
    }

    /// Check whether `uri` matches this template pattern without allocating heap strings.
    #[must_use]
    pub fn matches(&self, uri: &str) -> bool {
        if self.var_count == 0 {
            return self.raw == uri;
        }
        self.match_with_callback(uri, |_, _| {}).is_some()
    }

    /// Extract borrowed `(variable_name, value)` pairs from `uri`.
    #[must_use]
    pub fn extract_borrowed<'u>(&self, uri: &'u str) -> Option<Vec<(&'a str, &'u str)>> {
        if self.var_count == 0 {
            return if self.raw == uri {
                Some(Vec::new())
            } else {
                None
            };
        }
        let mut vars = Vec::with_capacity(self.var_count);
        self.match_with_callback(uri, |k, v| vars.push((k, v)))?;
        Some(vars)
    }

    fn match_with_callback<'u>(
        &self,
        uri: &'u str,
        mut on_var: impl FnMut(&'a str, &'u str),
    ) -> Option<()> {
        let mut u_rem = uri;
        for (i, segment) in self.segments.iter().enumerate() {
            match *segment {
                TemplateSegment::Literal(lit) => {
                    if !u_rem.starts_with(lit) {
                        return None;
                    }
                    u_rem = &u_rem[lit.len()..];
                }
                TemplateSegment::Variable(var_name) => {
                    let val_str = match self.segments.get(i + 1) {
                        None => {
                            let val = u_rem;
                            u_rem = EMPTY_STR;
                            val
                        }
                        Some(TemplateSegment::Literal(next_lit)) => {
                            let val_end = u_rem.find(next_lit)?;
                            let val = &u_rem[..val_end];
                            u_rem = &u_rem[val_end..];
                            val
                        }
                        Some(TemplateSegment::Variable(_)) => EMPTY_STR,
                    };
                    on_var(var_name, val_str);
                }
            }
        }
        if u_rem.is_empty() { Some(()) } else { None }
    }
}

/// Helper to match an incoming URI against a URI template pattern with `{variable}` placeholders.
#[must_use]
pub fn match_uri_template(template: &str, uri: &str) -> Option<BTreeMap<String, String>> {
    let compiled = CompiledUriTemplate::new(template);
    let mut map = BTreeMap::new();
    compiled.match_with_callback(uri, |k, v| {
        map.insert(k.to_string(), v.to_string());
    })?;
    Some(map)
}

/// Type-erased future returned by [`ErasedResource::read_with_vars`].
pub type BoxResourceFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ResourceOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous resource storage.
pub trait ErasedResource: Send + Sync {
    /// Execute `read` given pre-extracted borrowed `(key, value)` URI variables.
    fn read_with_vars<'a>(
        &'a self,
        uri: &'a str,
        vars: Vec<(&'static str, &'a str)>,
    ) -> BoxResourceFuture<'a>;

    /// Execute `read` with caller `ctx` under `effective_policy`.
    fn read_with_vars_and_policy<'a>(
        &'a self,
        uri: &'a str,
        vars: Vec<(&'static str, &'a str)>,
        _ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxResourceFuture<'a> {
        let fut = self.read_with_vars(uri, vars);
        Box::pin(async move {
            match fut.await {
                Ok(out) => Ok(effective_policy.apply_resource(out)),
                Err(err) => Err(effective_policy.apply_error(err)),
            }
        })
    }
}

impl<T: RustResource> ErasedResource for T {
    fn read_with_vars<'a>(
        &'a self,
        uri: &'a str,
        vars: Vec<(&'static str, &'a str)>,
    ) -> BoxResourceFuture<'a> {
        let policy = self.output_sanitization();
        Box::pin(async move {
            let ctx = ToolContext::new();
            let deserializer = serde::de::value::MapDeserializer::new(
                vars.into_iter()
                    .map(|(k, v)| (k, UriVarDeserializer::new(v))),
            );
            let params: T::Params = match serde::de::Deserialize::deserialize(deserializer) {
                Ok(p) => p,
                Err(e) => {
                    let err = ToolError::new(format!(
                        "Failed to deserialize resource parameters from URI variables: {e}"
                    ));
                    return Err(policy.apply_error(err));
                }
            };
            match self.__read_raw(uri, params, &ctx).await {
                Ok(out) => Ok(policy.apply_resource(out)),
                Err(err) => Err(policy.apply_error(err)),
            }
        })
    }

    fn read_with_vars_and_policy<'a>(
        &'a self,
        uri: &'a str,
        vars: Vec<(&'static str, &'a str)>,
        ctx: &'a ToolContext,
        effective_policy: OutputSanitization,
    ) -> BoxResourceFuture<'a> {
        Box::pin(async move {
            let deserializer = serde::de::value::MapDeserializer::new(
                vars.into_iter()
                    .map(|(k, v)| (k, UriVarDeserializer::new(v))),
            );
            let params: T::Params = match serde::de::Deserialize::deserialize(deserializer) {
                Ok(p) => p,
                Err(e) => {
                    let err = ToolError::new(format!(
                        "Failed to deserialize resource parameters from URI variables: {e}"
                    ));
                    return Err(effective_policy.apply_error(err));
                }
            };
            match self.__read_raw(uri, params, ctx).await {
                Ok(out) => Ok(effective_policy.apply_resource(out)),
                Err(err) => Err(effective_policy.apply_error(err)),
            }
        })
    }
}

#[derive(Clone)]
struct RegisteredResource {
    name: &'static str,
    definition: ResourceDefinition,
    provenance: ToolProvenance,
    output_sanitization: OutputSanitization,
    template: CompiledUriTemplate<'static>,
    erased: Arc<dyn ErasedResource>,
}

/// A registry of resources and resource templates for dynamic dispatch.
///
/// # Example
///
/// ```
/// use llm_tool::{ResourceOutput, ResourceRegistry, RustResource, ToolError};
///
/// struct ConfigResource;
///
/// impl RustResource for ConfigResource {
///     const NAME: &'static str = "config";
///     const URI_TEMPLATE: &'static str = "file:///etc/app.conf";
///     const DESCRIPTION: &'static str = "Application configuration";
///     const MIME_TYPE: Option<&'static str> = Some("text/plain");
///     type Params = ();
///
///     async fn read(
///         &self,
///         uri: &str,
///         _params: Self::Params,
///     ) -> Result<ResourceOutput, ToolError> {
///         Ok(ResourceOutput::text(uri, Some("text/plain"), "debug=true"))
///     }
/// }
///
/// let mut reg = ResourceRegistry::new();
/// reg.register(ConfigResource);
/// assert_eq!(reg.len(), 1);
/// assert!(reg.matches("file:///etc/app.conf"));
/// ```
#[derive(Clone)]
pub struct ResourceRegistry {
    resources: Vec<RegisteredResource>,
    output_floor: OutputSanitization,
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for ResourceRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names: Vec<&str> = self.resources.iter().map(|r| r.name).collect();
        f.debug_struct("ResourceRegistry")
            .field("resource_count", &self.resources.len())
            .field("resource_names", &names)
            .field("output_floor", &self.output_floor)
            .finish()
    }
}

impl ResourceRegistry {
    /// Create a new, empty resource registry with secure default [`OutputSanitization::SanitizeTokens`] floor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            resources: Vec::new(),
            output_floor: OutputSanitization::SanitizeTokens,
        }
    }

    /// Set the minimum [`OutputSanitization`] enforced on every read resource output.
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

    /// Return the minimum [`OutputSanitization`] enforced on every read resource output.
    #[must_use]
    pub const fn output_floor(&self) -> OutputSanitization {
        self.output_floor
    }

    const fn effective_policy_for_entry(&self, entry: &RegisteredResource) -> OutputSanitization {
        let res_policy = if entry.output_sanitization.is_none() {
            OutputSanitization::None
        } else {
            entry
                .output_sanitization
                .at_least(entry.provenance.default_sanitization())
        };
        res_policy.at_least(self.output_floor)
    }

    /// Return the effective [`OutputSanitization`] applied to resource `name`, or `None` if not registered.
    #[must_use]
    pub fn effective_output_sanitization(&self, name: &str) -> Option<OutputSanitization> {
        self.resources
            .iter()
            .find(|e| e.name == name)
            .map(|e| self.effective_policy_for_entry(e))
    }

    /// Return the [`ToolProvenance`] of resource `name`, or `None` if not registered.
    #[must_use]
    pub fn provenance(&self, name: &str) -> Option<ToolProvenance> {
        self.resources
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.provenance)
    }

    /// Register a [`RustResource`], replacing any existing registration with the same [`RustResource::NAME`].
    pub fn register<R: RustResource + 'static>(&mut self, resource: R) -> &mut Self {
        if let Some(pos) = self.resources.iter().position(|e| e.name == R::NAME) {
            self.resources.remove(pos);
        }
        let definition = definition_of_resource(&resource);
        let provenance = resource.provenance();
        let output_sanitization = resource.output_sanitization();
        self.resources.push(RegisteredResource {
            name: R::NAME,
            definition,
            provenance,
            output_sanitization,
            template: CompiledUriTemplate::new(R::URI_TEMPLATE),
            erased: Arc::new(resource),
        });
        self
    }

    /// Register a [`RustResource`], consuming and returning `Self` for chaining.
    #[must_use]
    pub fn with_resource<R: RustResource + 'static>(mut self, resource: R) -> Self {
        self.register(resource);
        self
    }

    /// Remove a resource by name, returning `true` if it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        if let Some(pos) = self.resources.iter().position(|e| e.name == name) {
            self.resources.remove(pos);
            true
        } else {
            false
        }
    }

    /// Clear all registered resources.
    pub fn clear(&mut self) {
        self.resources.clear();
    }

    /// Collect [`ResourceDefinition`]s for all registered resources.
    #[must_use]
    pub fn definitions(&self) -> Vec<ResourceDefinition> {
        self.resources
            .iter()
            .map(|entry| entry.definition.clone())
            .collect()
    }

    /// Number of registered resources.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.resources.len()
    }

    /// Whether the registry has no registered resources.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Whether a resource with the given name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.resources.iter().any(|entry| entry.name == name)
    }

    /// Whether any registered resource's URI template matches `uri`.
    #[must_use]
    pub fn matches(&self, uri: &str) -> bool {
        self.resources
            .iter()
            .any(|entry| entry.template.matches(uri))
    }

    /// Borrow the cached [`ResourceDefinition`] for a registered resource by name.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<&ResourceDefinition> {
        self.resources
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| &entry.definition)
    }

    /// Iterate over `(name, definition)` pairs for every registered resource.
    #[must_use]
    pub fn iter(&self) -> ResourceDefinitions<'_> {
        ResourceDefinitions {
            inner: self.resources.iter(),
        }
    }

    /// Read the first resource whose URI template matches `uri`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no registered resource's template matches `uri`,
    /// or a read error if URI-variable deserialization or reading fails.
    pub async fn read(&self, uri: &str) -> Result<ResourceOutput, ToolError> {
        let ctx = ToolContext::new();
        self.read_with_context(uri, &ctx).await
    }

    /// Read the first resource whose URI template matches `uri`, passing caller `ctx`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::not_found`] if no registered resource's template matches `uri`,
    /// or a read error if URI-variable deserialization, extension extraction, or reading fails.
    pub async fn read_with_context(
        &self,
        uri: &str,
        ctx: &ToolContext,
    ) -> Result<ResourceOutput, ToolError> {
        for resource in &self.resources {
            if let Some(vars) = resource.template.extract_borrowed(uri) {
                let effective = self.effective_policy_for_entry(resource);
                return resource
                    .erased
                    .read_with_vars_and_policy(uri, vars, ctx, effective)
                    .await;
            }
        }
        let err = ToolError::not_found(RegistryItem::Resource, uri);
        Err(self.output_floor.apply_error(err))
    }
}

/// Borrowing iterator over `(name, definition)` pairs, yielded by [`ResourceRegistry::iter`].
pub struct ResourceDefinitions<'a> {
    inner: core::slice::Iter<'a, RegisteredResource>,
}

impl Iterator for ResourceDefinitions<'_> {
    type Item = (&'static str, ResourceDefinition);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|entry| (entry.name, entry.definition.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for ResourceDefinitions<'_> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> IntoIterator for &'a ResourceRegistry {
    type Item = (&'static str, ResourceDefinition);
    type IntoIter = ResourceDefinitions<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::match_uri_template;

    #[test]
    fn exact_match_no_variables() {
        let m = match_uri_template("config://app", "config://app").expect("should match");
        assert!(m.is_empty());
    }

    #[test]
    fn no_match_different_literal() {
        assert!(match_uri_template("config://app", "config://other").is_none());
    }

    #[test]
    fn single_trailing_variable_captures_rest() {
        let m = match_uri_template("file:///{path}", "file:///etc/hosts").expect("should match");
        assert_eq!(m.get("path").map(String::as_str), Some("etc/hosts"));
    }

    #[test]
    fn multiple_variables() {
        let m = match_uri_template(
            "file:///logs/{date}/{app}.log",
            "file:///logs/2024-01-01/server.log",
        )
        .expect("should match");
        assert_eq!(m.get("date").map(String::as_str), Some("2024-01-01"));
        assert_eq!(m.get("app").map(String::as_str), Some("server"));
    }

    #[test]
    fn variable_stops_at_delimiter() {
        let m = match_uri_template("x://{a}/{b}", "x://one/two").expect("should match");
        assert_eq!(m.get("a").map(String::as_str), Some("one"));
        assert_eq!(m.get("b").map(String::as_str), Some("two"));
    }

    #[test]
    fn prefix_mismatch_returns_none() {
        assert!(match_uri_template("x://{a}", "y://foo").is_none());
    }

    #[test]
    fn unterminated_template_variable_returns_none() {
        assert!(match_uri_template("x://{a", "x://foo").is_none());
    }

    #[test]
    fn missing_delimiter_in_uri_returns_none() {
        assert!(match_uri_template("x://{a}/end", "x://noslash").is_none());
    }

    #[test]
    fn trailing_literal_must_match() {
        assert!(match_uri_template("f://{a}.log", "f://name.txt").is_none());
    }

    #[test]
    fn empty_variable_value_is_allowed() {
        let m = match_uri_template("a://{x}/b", "a:///b").expect("should match");
        assert_eq!(m.get("x").map(String::as_str), Some(""));
    }

    #[test]
    fn longer_uri_than_literal_template_returns_none() {
        assert!(match_uri_template("a://b", "a://bc").is_none());
    }

    #[test]
    fn adjacent_template_variables_match() {
        let m = match_uri_template("x://{a}{b}", "x://hello").expect("should match");
        assert_eq!(m.get("a").map(String::as_str), Some(""));
        assert_eq!(m.get("b").map(String::as_str), Some("hello"));
    }

    #[tokio::test]
    async fn erased_resource_deserializes_typed_uri_variables_and_rejects_malformed() {
        use super::*;

        #[derive(serde::Deserialize)]
        #[serde(rename_all = "lowercase")]
        enum Env {
            Prod,
            Staging,
        }

        struct TypedResource;
        #[derive(serde::Deserialize)]
        struct TypedParams {
            env: Env,
            count: u64,
            verbose: bool,
        }
        impl RustResource for TypedResource {
            const NAME: &'static str = "typed_res";
            const URI_TEMPLATE: &'static str = "res://{env}/{count}/{verbose}";
            const DESCRIPTION: &'static str = "Resource with typed URI params";
            const MIME_TYPE: Option<&'static str> = None;
            type Params = TypedParams;
            fn read(
                &self,
                uri: &str,
                params: Self::Params,
            ) -> impl Future<Output = Result<ResourceOutput, ToolError>> + Send {
                let env_name = match params.env {
                    Env::Prod => "prod",
                    Env::Staging => "staging",
                };
                core::future::ready(Ok(ResourceOutput::text(
                    uri,
                    None,
                    format!("{env_name}:{}:{}", params.count, params.verbose),
                )))
            }
        }

        let reg = ResourceRegistry::new().with_resource(TypedResource);
        let ok_out = reg
            .read("res://prod/42/true")
            .await
            .expect("valid URI vars");
        match &ok_out.contents[0] {
            crate::ResourceOutputContent::Text { text, .. } => {
                assert_eq!(text, "prod:42:true");
            }
            crate::ResourceOutputContent::Blob { .. } => panic!("expected text"),
        }

        let err = reg.read("res://prod/not_an_int/true").await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Failed to deserialize resource parameters from URI variables")
        );
    }

    #[test]
    fn compiled_uri_template_segments_and_accessors() {
        use super::{CompiledUriTemplate, TemplateSegment};

        let static_tmpl = CompiledUriTemplate::new("config://app");
        assert_eq!(static_tmpl.raw(), "config://app");
        assert_eq!(static_tmpl.var_count(), 0);
        assert_eq!(
            static_tmpl.segments(),
            &[TemplateSegment::Literal("config://app")]
        );
        assert!(static_tmpl.matches("config://app"));
        assert!(!static_tmpl.matches("config://other"));
        assert_eq!(
            static_tmpl.extract_borrowed("config://app"),
            Some(alloc::vec::Vec::new())
        );

        let dyn_tmpl = CompiledUriTemplate::new("file:///logs/{date}/{app}.log");
        assert_eq!(dyn_tmpl.raw(), "file:///logs/{date}/{app}.log");
        assert_eq!(dyn_tmpl.var_count(), 2);
        assert_eq!(
            dyn_tmpl.segments(),
            &[
                TemplateSegment::Literal("file:///logs/"),
                TemplateSegment::Variable("date"),
                TemplateSegment::Literal("/"),
                TemplateSegment::Variable("app"),
                TemplateSegment::Literal(".log"),
            ]
        );
    }

    #[test]
    fn compiled_uri_template_extract_borrowed_zero_copy() {
        use super::CompiledUriTemplate;

        let tmpl = CompiledUriTemplate::new("res://{host}:{port}/{path}");
        let uri = "res://localhost:8080/api/v1";
        let extracted = tmpl.extract_borrowed(uri).expect("should match");
        assert_eq!(
            extracted,
            alloc::vec![("host", "localhost"), ("port", "8080"), ("path", "api/v1"),]
        );
        let uri_start = uri.as_ptr() as usize;
        let uri_end = uri_start + uri.len();
        for (_, val) in extracted {
            let val_ptr = val.as_ptr() as usize;
            assert!(val_ptr >= uri_start && val_ptr + val.len() <= uri_end);
        }
    }
}
