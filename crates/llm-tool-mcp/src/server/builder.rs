//! [`McpServerBuilder`] — register prompts, resources, and per-caller
//! registries before constructing an [`McpServer`].

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use llm_tool::{PromptRegistry, ResourceRegistry, ToolContext, ToolRegistry};

use super::{CallerView, McpServer, RegistryFactory, build_tools_list_value};

/// Builder for [`McpServer`] that registers prompts and resources up front.
///
/// Prefer this over constructing a server and mutating it: the builder owns
/// its [`PromptRegistry`] and [`ResourceRegistry`] outright, so registration
/// is infallible and never has to unwrap a shared `Arc`.
///
/// # Example
///
/// ```rust
/// use llm_tool::{ToolContext, ToolRegistry};
/// use llm_tool_mcp::McpServer;
///
/// let server = McpServer::builder("srv", "0.1.0", ToolRegistry::new())
///     .with_context(ToolContext::new().with_conversation_id("agent"))
///     .build();
/// # let _server = server;
/// ```
pub struct McpServerBuilder {
    name: String,
    version: String,
    registry: ToolRegistry,
    context: ToolContext,
    prompts: PromptRegistry,
    resources: ResourceRegistry,
    per_connection_identity: bool,
    registry_factory: Option<Arc<dyn RegistryFactory>>,
}

impl McpServerBuilder {
    /// Start building a server that serves tools from `registry`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            registry,
            context: ToolContext::new(),
            prompts: PromptRegistry::new(),
            resources: ResourceRegistry::new(),
            per_connection_identity: false,
            registry_factory: None,
        }
    }

    /// Set the [`ToolContext`] used for all tool dispatches.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = context;
        self
    }

    /// Enable per-connection caller identity.
    ///
    /// When enabled, each connection derives its own caller from the
    /// `clientInfo.name` in its MCP `initialize` handshake (sharing the base
    /// context's state and typed extensions via [`ToolContext::with_caller`]).
    /// This is what lets one server serve many agents, each acting under its
    /// own persona. Defaults to `false`.
    #[must_use]
    pub const fn with_per_connection_identity(mut self, enabled: bool) -> Self {
        self.per_connection_identity = enabled;
        self
    }

    /// Serve a **per-caller** tool registry via a [`RegistryFactory`].
    ///
    /// When set, the first time each negotiated caller is seen the factory
    /// builds that caller's [`ToolRegistry`]; its schema is serialized once and
    /// cached, then reused for that caller's `tools/list` and `tools/call`. This
    /// preserves per-caller tool gating and description tailoring on a single
    /// shared server. The `registry` passed to [`new`](Self::new) becomes a
    /// placeholder — the factory's `registry_for(None)` provides the default
    /// view. Typically combined with
    /// [`with_per_connection_identity`](Self::with_per_connection_identity).
    #[must_use]
    pub fn with_registry_factory<F: RegistryFactory + 'static>(mut self, factory: F) -> Self {
        self.registry_factory = Some(Arc::new(factory));
        self
    }

    /// Register a prompt template.
    #[must_use]
    pub fn with_prompt<P: llm_tool::RustPrompt + 'static>(mut self, prompt: P) -> Self {
        self.prompts.register(prompt);
        self
    }

    /// Register a resource (or resource template).
    #[must_use]
    pub fn with_resource<R: llm_tool::RustResource + 'static>(mut self, resource: R) -> Self {
        self.resources.register(resource);
        self
    }

    /// Finish building the [`McpServer`].
    ///
    /// Tool schemas are computed and serialized **once** here and cached for
    /// all subsequent `tools/list` requests.
    #[must_use]
    pub fn build(self) -> McpServer {
        // With a factory, the default (no-identity) view comes from
        // `registry_for(None)`; the `registry` passed to the builder is unused.
        let base_registry = match &self.registry_factory {
            Some(factory) => factory.registry_for(None),
            None => self.registry,
        };
        let registry = Arc::new(base_registry);
        let cached_tools_list = Arc::new(build_tools_list_value(&registry));

        // Seed the cache so the default caller reuses the same registry + list
        // Arcs already built above, avoiding a redundant rebuild on first use.
        let mut caller_views = HashMap::new();
        if self.registry_factory.is_some() {
            caller_views.insert(
                String::new(),
                CallerView {
                    registry: Arc::clone(&registry),
                    tools_list: Arc::clone(&cached_tools_list),
                },
            );
        }

        McpServer {
            name: self.name,
            version: self.version,
            registry,
            context: Arc::new(self.context),
            cached_tools_list,
            prompts: Arc::new(self.prompts),
            resources: Arc::new(self.resources),
            per_connection_identity: self.per_connection_identity,
            registry_factory: self.registry_factory,
            caller_views: Arc::new(Mutex::new(caller_views)),
        }
    }
}
