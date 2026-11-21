//! [`McpServerBuilder`] — register prompts, resources, limits, and per-caller
//! registries before constructing an [`McpServer`].

use std::sync::{Arc, Mutex};

use llm_tool::{PromptRegistry, ResourceRegistry, ToolContext, ToolRegistry};
use tokio::sync::Semaphore;

use super::{
    CallerResolver, CallerView, CallerViewCache, DEFAULT_MAX_CALLER_VIEWS, DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_IN_FLIGHT_PER_CONN, DEFAULT_MAX_JSONRPC_BATCH_LEN, DEFAULT_MAX_JSONRPC_LINE_BYTES,
    McpServer, RegistryFactory, build_prompts_list_value, build_resource_templates_list_value,
    build_resources_list_value, build_tools_list_value,
};

/// Builder for [`McpServer`] that registers prompts, resources, limits, and caller policies up front.
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
    instructions: Option<String>,
    registry: ToolRegistry,
    context: ToolContext,
    prompts: PromptRegistry,
    resources: ResourceRegistry,
    per_connection_identity: bool,
    trust_client_info_caller: bool,
    caller_resolver: Option<Arc<dyn CallerResolver>>,
    registry_factory: Option<Arc<dyn RegistryFactory>>,
    max_line_bytes: usize,
    max_batch_len: usize,
    max_in_flight_per_conn: usize,
    max_connections: usize,
    max_caller_views: usize,
}

impl McpServerBuilder {
    /// Start building a server that serves tools from `registry`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        let raw_name = name.into();
        let raw_ver = version.into();
        Self {
            name: llm_tool::OutputSanitization::SanitizeTokens
                .apply_str(&raw_name)
                .into_owned(),
            version: llm_tool::OutputSanitization::SanitizeTokens
                .apply_str(&raw_ver)
                .into_owned(),
            instructions: None,
            registry,
            context: ToolContext::new(),
            prompts: PromptRegistry::new(),
            resources: ResourceRegistry::new(),
            per_connection_identity: false,
            trust_client_info_caller: false,
            caller_resolver: None,
            registry_factory: None,
            max_line_bytes: DEFAULT_MAX_JSONRPC_LINE_BYTES,
            max_batch_len: DEFAULT_MAX_JSONRPC_BATCH_LEN,
            max_in_flight_per_conn: DEFAULT_MAX_IN_FLIGHT_PER_CONN,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            max_caller_views: DEFAULT_MAX_CALLER_VIEWS,
        }
    }

    /// Set the base [`ToolContext`] forked per connection.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = context;
        self
    }

    /// Provide instructions describing how to use the server.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        let raw = instructions.into();
        self.instructions = Some(
            llm_tool::OutputSanitization::SanitizeTokens
                .apply_str(&raw)
                .into_owned(),
        );
        self
    }

    /// Enable per-connection caller identity from `initialize` `clientInfo.name` for both
    /// [`ToolContext::conversation_id`] and [`RegistryFactory`].
    ///
    /// Defaults to `false` (unauthenticated `clientInfo.name` is ignored for both context
    /// identity and per-caller registries). Multi-tenant TCP/Unix servers accepting untrusted
    /// peers should pair per-caller privilege gating with transport authentication
    /// ([`super::Connection::with_authenticated_caller`]) or [`with_caller_resolver`](Self::with_caller_resolver).
    #[must_use]
    pub const fn with_per_connection_identity(mut self, enabled: bool) -> Self {
        self.per_connection_identity = enabled;
        self
    }

    /// Opt in to trusting unauthenticated `clientInfo.name` for switching per-caller
    /// [`RegistryFactory`] views. Defaults to `false`.
    #[must_use]
    pub const fn trust_client_info_caller(mut self, trust: bool) -> Self {
        self.trust_client_info_caller = trust;
        self
    }

    /// Attach an authenticated [`CallerResolver`] invoked during `initialize`.
    #[must_use]
    pub fn with_caller_resolver<R: CallerResolver + 'static>(mut self, resolver: R) -> Self {
        self.caller_resolver = Some(Arc::new(resolver));
        self
    }

    /// Serve a **per-caller** tool registry via a [`RegistryFactory`].
    #[must_use]
    pub fn with_registry_factory<F: RegistryFactory + 'static>(mut self, factory: F) -> Self {
        self.registry_factory = Some(Arc::new(factory));
        self
    }

    /// Configure the maximum allowed JSON-RPC wire line length in bytes (`default = 16 MiB`).
    #[must_use]
    pub const fn with_max_line_bytes(mut self, max_bytes: usize) -> Self {
        self.max_line_bytes = if max_bytes == 0 { 1 } else { max_bytes };
        self
    }

    /// Configure the maximum number of items in a JSON-RPC batch array (`default = 128`).
    #[must_use]
    pub const fn with_max_batch_len(mut self, max_len: usize) -> Self {
        self.max_batch_len = if max_len == 0 { 1 } else { max_len };
        self
    }

    /// Configure the maximum number of concurrent in-flight requests per connection (`default = 64`).
    #[must_use]
    pub const fn with_max_in_flight_requests(mut self, max_in_flight: usize) -> Self {
        self.max_in_flight_per_conn = if max_in_flight == 0 { 1 } else { max_in_flight };
        self
    }

    /// Configure the maximum number of concurrent accepted TCP/Unix connections (`default = 256`).
    #[must_use]
    pub const fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.max_connections = if max_connections == 0 {
            1
        } else {
            max_connections
        };
        self
    }

    /// Configure the maximum number of distinct per-caller views cached by [`RegistryFactory`] (`default = 256`).
    #[must_use]
    pub const fn with_max_caller_views(mut self, max_views: usize) -> Self {
        self.max_caller_views = if max_views == 0 { 1 } else { max_views };
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
    #[must_use]
    pub fn build(self) -> McpServer {
        let base_registry = match &self.registry_factory {
            Some(factory) => factory.registry_for(None),
            None => self.registry,
        };
        let registry = Arc::new(base_registry);
        let cached_tools_list = Arc::new(build_tools_list_value(&registry));
        let cached_prompts_list = Arc::new(build_prompts_list_value(&self.prompts));
        let cached_resources_list = Arc::new(build_resources_list_value(&self.resources));
        let cached_resource_templates_list =
            Arc::new(build_resource_templates_list_value(&self.resources));

        let default_view = if self.registry_factory.is_some() {
            Some(CallerView {
                registry: Arc::clone(&registry),
                tools_list: Arc::clone(&cached_tools_list),
            })
        } else {
            None
        };
        let caller_views = CallerViewCache::new(default_view, self.max_caller_views);

        McpServer {
            name: self.name,
            version: self.version,
            instructions: self.instructions,
            registry,
            context: Arc::new(self.context),
            cached_tools_list,
            cached_prompts_list,
            cached_resources_list,
            cached_resource_templates_list,
            prompts: Arc::new(self.prompts),
            resources: Arc::new(self.resources),
            per_connection_identity: self.per_connection_identity,
            trust_client_info_caller: self.trust_client_info_caller,
            caller_resolver: self.caller_resolver,
            registry_factory: self.registry_factory,
            caller_views: Arc::new(Mutex::new(caller_views)),
            max_line_bytes: self.max_line_bytes,
            max_batch_len: self.max_batch_len,
            max_in_flight_per_conn: self.max_in_flight_per_conn,
            max_caller_views: self.max_caller_views,
            connection_semaphore: Arc::new(Semaphore::new(self.max_connections)),
        }
    }
}
