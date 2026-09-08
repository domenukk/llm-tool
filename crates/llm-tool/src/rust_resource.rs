//! Strongly-typed Rust resource trait and URI template matching.

use alloc::{
    borrow::Cow,
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::{future::Future, pin::Pin};

use super::types::{RegistryItem, ResourceDefinition, ResourceOutput, ToolError};

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
}

/// Build a [`ResourceDefinition`] from any [`RustResource`] implementor.
///
/// Infallible: the definition is built purely from associated constants.
#[must_use]
pub fn definition_of_resource<T: RustResource>(resource: &T) -> ResourceDefinition {
    ResourceDefinition {
        uri_template: T::URI_TEMPLATE.to_string(),
        name: T::NAME.to_string(),
        description: resource.description().into_owned(),
        mime_type: T::MIME_TYPE.map(ToString::to_string),
    }
}

/// Opening delimiter for URI template variable placeholders.
const TEMPLATE_VAR_START: char = '{';
/// Closing delimiter for URI template variable placeholders.
const TEMPLATE_VAR_END: char = '}';
/// Empty string slice.
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
///
/// Parses `{variable}` placeholders once at registration time so that
/// [`matches`](Self::matches) and [`extract_borrowed`](Self::extract_borrowed)
/// run in pure stack memory without re-scanning the template string or
/// allocating heap `String`s.
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
    ///
    /// Returns `None` if `uri` does not match this template.
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
///
/// Returns `Some(map)` if the URI matches the pattern, mapping each `{variable}` name
/// to its extracted value string. Returns `None` if the URI does not match.
#[must_use]
pub fn match_uri_template(
    template: &str,
    uri: &str,
) -> Option<alloc::collections::BTreeMap<String, String>> {
    let compiled = CompiledUriTemplate::new(template);
    let mut map = alloc::collections::BTreeMap::new();
    compiled.match_with_callback(uri, |k, v| {
        map.insert(k.to_string(), v.to_string());
    })?;
    Some(map)
}

/// Type-erased future returned by [`ErasedResource::read_with_vars`].
pub(crate) type BoxResourceFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ResourceOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous resource storage.
///
/// This is an internal implementation detail of [`ResourceRegistry`]; callers
/// interact with resources through the registry rather than this trait.
pub(crate) trait ErasedResource: Send + Sync {
    /// Execute `read` given pre-extracted borrowed `(key, value)` URI variables.
    fn read_with_vars<'a>(
        &'a self,
        uri: &'a str,
        vars: Vec<(&'static str, &'a str)>,
    ) -> BoxResourceFuture<'a>;
}

impl<T: RustResource> ErasedResource for T {
    fn read_with_vars<'a>(
        &'a self,
        uri: &'a str,
        vars: Vec<(&'static str, &'a str)>,
    ) -> BoxResourceFuture<'a> {
        Box::pin(async move {
            let deserializer = serde::de::value::MapDeserializer::new(
                vars.into_iter()
                    .map(|(k, v)| (k, serde::de::value::BorrowedStrDeserializer::new(v))),
            );
            let params: T::Params = serde::de::Deserialize::deserialize(deserializer).map_err(
                |e: serde::de::value::Error| {
                    ToolError::new(format!(
                        "Failed to deserialize resource parameters from URI variables: {e}"
                    ))
                },
            )?;
            self.read(uri, params).await
        })
    }
}

/// A registered resource: its cached definition, pre-compiled URI template,
/// and type-erased handler.
struct RegisteredResource {
    name: &'static str,
    definition: ResourceDefinition,
    template: CompiledUriTemplate<'static>,
    erased: Box<dyn ErasedResource>,
}

/// A registry of resources and resource templates for dynamic dispatch.
///
/// Models MCP's `resources/list`, `resources/read`, and `resources/templates/list`
/// primitives. Tools provide actionable commands; resources provide readable
/// context (static documents, log files, configuration snapshots).
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
#[derive(Default)]
pub struct ResourceRegistry {
    resources: Vec<RegisteredResource>,
}

impl core::fmt::Debug for ResourceRegistry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let names: Vec<&str> = self.resources.iter().map(|r| r.name).collect();
        f.debug_struct("ResourceRegistry")
            .field("resource_count", &self.resources.len())
            .field("resource_names", &names)
            .finish()
    }
}

impl ResourceRegistry {
    /// Create a new, empty resource registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            resources: Vec::new(),
        }
    }

    /// Register a [`RustResource`].
    ///
    /// Replaces any existing registration with the same [`RustResource::NAME`].
    pub fn register<R: RustResource + 'static>(&mut self, resource: R) -> &mut Self {
        if let Some(pos) = self.resources.iter().position(|e| e.name == R::NAME) {
            self.resources.remove(pos);
        }
        self.resources.push(RegisteredResource {
            name: R::NAME,
            definition: definition_of_resource(&resource),
            template: CompiledUriTemplate::new(R::URI_TEMPLATE),
            erased: Box::new(resource),
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
    ///
    /// Returns clones of the cached definitions computed at registration time.
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
    ///
    /// Note that resources are *read* by URI (see [`matches`](Self::matches)),
    /// not by name; this checks the registered resource **name** for parity
    /// with [`ToolRegistry::contains`](crate::ToolRegistry::contains).
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.resources.iter().any(|entry| entry.name == name)
    }

    /// Whether any registered resource's URI template matches `uri`.
    ///
    /// This is the URI-keyed analog of [`contains`](Self::contains) and mirrors
    /// what [`read`](Self::read) uses to select a resource.
    #[must_use]
    pub fn matches(&self, uri: &str) -> bool {
        self.resources
            .iter()
            .any(|entry| entry.template.matches(uri))
    }

    /// Borrow the cached [`ResourceDefinition`] for a registered resource by name.
    ///
    /// Returns `None` if no resource named `name` is registered.
    #[must_use]
    pub fn definition(&self, name: &str) -> Option<&ResourceDefinition> {
        self.resources
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| &entry.definition)
    }

    /// Iterate over `(name, definition)` pairs for every registered resource.
    ///
    /// Yields clones of the cached definitions computed at registration time.
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
    /// Returns [`ToolError::not_found`] if no registered resource's template
    /// matches `uri` (carrying `error_kind = "not_registered"` metadata), or a
    /// read error if URI-variable deserialization or reading fails.
    pub async fn read(&self, uri: &str) -> Result<ResourceOutput, ToolError> {
        for resource in &self.resources {
            if let Some(vars) = resource.template.extract_borrowed(uri) {
                return resource.erased.read_with_vars(uri, vars).await;
            }
        }
        Err(ToolError::not_found(RegistryItem::Resource, uri))
    }
}

/// Borrowing iterator over `(name, definition)` pairs, yielded by
/// [`ResourceRegistry::iter`] and by `&ResourceRegistry`'s [`IntoIterator`] impl.
///
/// Each cached [`ResourceDefinition`] is cloned lazily as it is yielded.
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

/// Iterate over `(name, definition)` pairs for every registered resource.
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
        // '{' with no closing '}' cannot be parsed into a variable.
        assert!(match_uri_template("x://{a", "x://foo").is_none());
    }

    #[test]
    fn missing_delimiter_in_uri_returns_none() {
        // Template expects '/' after {a}, but the URI has none.
        assert!(match_uri_template("x://{a}/end", "x://noslash").is_none());
    }

    #[test]
    fn trailing_literal_must_match() {
        // {a} captures up to '.', then ".log" must match the remainder.
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
    async fn erased_resource_deserialization_failure_on_non_string_param() {
        use super::*;

        struct IntResource;
        #[derive(serde::Deserialize)]
        struct IntParams {
            count: u64,
        }
        impl RustResource for IntResource {
            const NAME: &'static str = "int_res";
            const URI_TEMPLATE: &'static str = "res://{count}";
            const DESCRIPTION: &'static str = "Resource with integer param";
            const MIME_TYPE: Option<&'static str> = None;
            type Params = IntParams;
            async fn read(
                &self,
                uri: &str,
                params: Self::Params,
            ) -> Result<ResourceOutput, ToolError> {
                Ok(ResourceOutput::text(uri, None, format!("{}", params.count)))
            }
        }

        let reg = ResourceRegistry::new().with_resource(IntResource);
        let err = reg.read("res://42").await.unwrap_err();
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
        // Verify that extracted values point directly into the input `uri` slice (zero copy)
        let uri_start = uri.as_ptr() as usize;
        let uri_end = uri_start + uri.len();
        for (_, val) in extracted {
            let val_ptr = val.as_ptr() as usize;
            assert!(val_ptr >= uri_start && val_ptr + val.len() <= uri_end);
        }
    }
}
