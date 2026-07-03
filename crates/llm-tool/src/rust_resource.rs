//! Strongly-typed Rust resource trait and URI template matching.

use alloc::{
    borrow::Cow,
    boxed::Box,
    format,
    string::{String, ToString},
};
use core::{future::Future, pin::Pin};

use super::types::{ResourceDefinition, ResourceOutput, ToolError};

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
pub fn definition_of_resource<T: RustResource>(resource: &T) -> ResourceDefinition {
    ResourceDefinition {
        uri_template: T::URI_TEMPLATE.to_string(),
        name: T::NAME.to_string(),
        description: resource.description().into_owned(),
        mime_type: T::MIME_TYPE.map(ToString::to_string),
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
    let mut map = alloc::collections::BTreeMap::new();
    let mut t_rem = template;
    let mut u_rem = uri;

    while let Some(start_idx) = t_rem.find('{') {
        let prefix = &t_rem[..start_idx];
        if !u_rem.starts_with(prefix) {
            return None;
        }
        u_rem = &u_rem[prefix.len()..];
        t_rem = &t_rem[start_idx + 1..];

        let end_idx = t_rem.find('}')?;
        let var_name = &t_rem[..end_idx];
        t_rem = &t_rem[end_idx + 1..];

        let val_str = if t_rem.is_empty() {
            let val = u_rem;
            u_rem = "";
            val
        } else {
            let next_char = t_rem.chars().next()?;
            let val_end = u_rem.find(next_char)?;
            let val = &u_rem[..val_end];
            u_rem = &u_rem[val_end..];
            val
        };

        map.insert(var_name.to_string(), val_str.to_string());
    }

    if t_rem == u_rem { Some(map) } else { None }
}

/// Type-erased future returned by [`ErasedResource::read_erased`].
pub type BoxResourceFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ResourceOutput, ToolError>> + Send + 'a>>;

/// Type-erased wrapper enabling heterogeneous resource storage.
pub trait ErasedResource: Send + Sync {
    /// Return the resource definition.
    fn definition(&self) -> ResourceDefinition;

    /// Check if the incoming URI matches this resource's pattern, extract variables, and execute `read`.
    fn read_erased<'a>(&'a self, uri: &'a str) -> Option<BoxResourceFuture<'a>>;
}

impl<T: RustResource> ErasedResource for T {
    fn definition(&self) -> ResourceDefinition {
        definition_of_resource(self)
    }

    fn read_erased<'a>(&'a self, uri: &'a str) -> Option<BoxResourceFuture<'a>> {
        let params_map = match_uri_template(T::URI_TEMPLATE, uri)?;
        Some(Box::pin(async move {
            let deserializer = serde::de::value::MapDeserializer::new(
                params_map
                    .into_iter()
                    .map(|(k, v)| (k, serde::de::value::StringDeserializer::new(v))),
            );
            let params: T::Params = serde::de::Deserialize::deserialize(deserializer).map_err(
                |e: serde::de::value::Error| {
                    ToolError::new(format!(
                        "Failed to deserialize resource parameters from URI variables: {e}"
                    ))
                },
            )?;
            self.read(uri, params).await
        }))
    }
}
