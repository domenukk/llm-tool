//! MCP stdio/TCP/Unix server backed by a [`ToolRegistry`].

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use llm_tool::{PromptRegistry, ResourceRegistry, ToolContext, ToolDefinition, ToolRegistry};
use tokio::sync::Semaphore;
use tracing::warn;

#[cfg(test)]
use crate::protocol::{self, *};
use crate::protocol::{InitializeParams, McpToolAnnotations, McpToolSchema, ToolsListResult};

mod builder;
mod dispatch;
mod transport;

pub use builder::McpServerBuilder;
pub use dispatch::RpcOutcome;
pub use transport::Transport;

/// Default maximum JSON-RPC wire line size (`16 MiB`).
pub const DEFAULT_MAX_JSONRPC_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Default maximum number of requests inside a single JSON-RPC batch array (`128`).
pub const DEFAULT_MAX_JSONRPC_BATCH_LEN: usize = 128;

/// Default maximum number of concurrent in-flight requests per connection (`64`).
pub const DEFAULT_MAX_IN_FLIGHT_PER_CONN: usize = 64;

/// Default maximum number of concurrent accepted TCP / Unix connections (`256`).
pub const DEFAULT_MAX_CONNECTIONS: usize = 256;

/// Default maximum number of distinct per-caller views cached by [`RegistryFactory`] (`256`).
pub const DEFAULT_MAX_CALLER_VIEWS: usize = 256;

/// Authenticated caller resolver invoked during the MCP `initialize` handshake.
pub trait CallerResolver: Send + Sync {
    /// Return a verified caller name for this connection, or `None` to use the default caller.
    fn resolve_caller(&self, params: Option<&InitializeParams>) -> Option<String>;
}

impl<F> CallerResolver for F
where
    F: Fn(Option<&InitializeParams>) -> Option<String> + Send + Sync,
{
    fn resolve_caller(&self, params: Option<&InitializeParams>) -> Option<String> {
        self(params)
    }
}

/// Builds the [`ToolRegistry`] a given caller should see.
pub trait RegistryFactory: Send + Sync {
    /// Build the registry for `caller`.
    ///
    /// `caller` is `None` for the default view (before/without a negotiated
    /// identity), or `Some(name)` for a connection with an authenticated or
    /// explicitly trusted caller identity.
    fn registry_for(&self, caller: Option<&str>) -> ToolRegistry;
}

impl<F> RegistryFactory for F
where
    F: Fn(Option<&str>) -> ToolRegistry + Send + Sync,
{
    fn registry_for(&self, caller: Option<&str>) -> ToolRegistry {
        self(caller)
    }
}

/// A caller-specific view: the registry to dispatch against plus its
/// pre-serialized `tools/list` body. Cheap to clone (two `Arc`s).
#[derive(Clone)]
struct CallerView {
    pub registry: Arc<ToolRegistry>,
    pub tools_list: Arc<serde_json::Value>,
}

/// Bounded FIFO cache of per-caller views, keeping the default (`None`) view pinned.
struct CallerViewCache {
    pub default_view: Option<CallerView>,
    pub named: HashMap<String, CallerView>,
    order: VecDeque<String>,
    max_named: usize,
}

impl CallerViewCache {
    pub fn new(default_view: Option<CallerView>, max_named: usize) -> Self {
        Self {
            default_view,
            named: HashMap::new(),
            order: VecDeque::new(),
            max_named: max_named.max(1),
        }
    }

    fn get(&self, caller: Option<&str>) -> Option<CallerView> {
        match caller {
            None | Some("") => self.default_view.clone(),
            Some(name) => self.named.get(name).cloned(),
        }
    }

    fn get_or_insert(&mut self, caller: Option<&str>, view: CallerView) -> CallerView {
        let Some(name) = caller.filter(|s| !s.is_empty()) else {
            return self.default_view.get_or_insert(view).clone();
        };
        if let Some(existing) = self.named.get(name) {
            return existing.clone();
        }
        while self.named.len() >= self.max_named {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.named.remove(&oldest);
        }
        self.order.push_back(name.to_owned());
        self.named.insert(name.to_owned(), view.clone());
        view
    }
}

/// Error returned when a duplicate in-flight [`RequestId`](crate::protocol::RequestId) is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DuplicateInFlightIdError;

/// RAII guard removing an active [`RequestId`](crate::protocol::RequestId) from [`Connection`] on drop.
struct InFlightGuard<'a> {
    conn: &'a Connection,
    id: crate::protocol::RequestId,
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.conn.remove_in_flight(&self.id);
    }
}

/// An MCP server that serves tools from a [`ToolRegistry`] over JSON-RPC.
///
/// # Example
///
/// ```rust
/// use llm_tool::{ToolContext, ToolError, ToolRegistry, llm_tool};
/// use llm_tool_mcp::McpServer;
///
/// /// Adds two numbers.
/// #[llm_tool]
/// fn add(
///     /// First operand.
///     a: i64,
///     /// Second operand.
///     b: i64,
/// ) -> Result<String, ToolError> {
///     Ok(format!("{}", a + b))
/// }
///
/// let registry = ToolRegistry::new().with_tool(Add);
/// let ctx = ToolContext::new().with_conversation_id("my-agent");
///
/// let server = McpServer::new("my-server", "0.1.0", registry)
///     .with_context(ctx);
///
/// let input = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":1,"b":2}}}"#;
/// let reader = std::io::Cursor::new(format!("{input}\n"));
/// let mut output = Vec::new();
/// server.run(reader, &mut output).unwrap();
///
/// let resp: serde_json::Value = serde_json::from_slice(&output).unwrap();
/// assert_eq!(resp["result"]["content"][0]["text"], "3");
/// ```
#[derive(Clone)]
pub struct McpServer {
    name: String,
    version: String,
    instructions: Option<String>,
    registry: Arc<ToolRegistry>,
    context: Arc<ToolContext>,
    cached_tools_list: Arc<serde_json::Value>,
    cached_prompts_list: Arc<serde_json::Value>,
    cached_resources_list: Arc<serde_json::Value>,
    cached_resource_templates_list: Arc<serde_json::Value>,
    prompts: Arc<PromptRegistry>,
    resources: Arc<ResourceRegistry>,
    per_connection_identity: bool,
    trust_client_info_caller: bool,
    caller_resolver: Option<Arc<dyn CallerResolver>>,
    registry_factory: Option<Arc<dyn RegistryFactory>>,
    caller_views: Arc<Mutex<CallerViewCache>>,
    max_line_bytes: usize,
    max_batch_len: usize,
    max_in_flight_per_conn: usize,
    max_caller_views: usize,
    connection_semaphore: Arc<Semaphore>,
}

/// Per-connection negotiated state, owned by each transport run loop.
#[derive(Clone, Default)]
pub struct Connection {
    initialized: Arc<AtomicBool>,
    authenticated_caller: Option<String>,
    client_info_name: Arc<Mutex<Option<String>>>,
    ctx: Arc<Mutex<Option<ToolContext>>>,
    view: Arc<Mutex<Option<CallerView>>>,
    in_flight: Arc<Mutex<HashMap<crate::protocol::RequestId, tokio::sync::oneshot::Sender<()>>>>,
}

impl Connection {
    /// Create a fresh, un-negotiated connection state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a connection pre-bound to a transport-authenticated caller identity
    /// (e.g. verified via Unix `SO_PEERCRED`, mTLS client certificate, or HTTP Bearer token).
    #[must_use]
    pub fn with_authenticated_caller(caller: impl Into<String>) -> Self {
        Self {
            initialized: Arc::new(AtomicBool::new(false)),
            authenticated_caller: Some(caller.into()),
            client_info_name: Arc::new(Mutex::new(None)),
            ctx: Arc::new(Mutex::new(None)),
            view: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn authenticated_caller(&self) -> Option<&str> {
        self.authenticated_caller
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }

    fn set_client_info_name(&self, name: Option<String>) {
        match self.client_info_name.lock() {
            Ok(mut guard) => *guard = name,
            Err(e) => {
                warn!(error = %e, "Connection::set_client_info_name: mutex poisoned");
            }
        }
    }

    fn client_info_name(&self) -> Option<String> {
        match self.client_info_name.lock() {
            Ok(guard) => guard.clone(),
            Err(e) => {
                warn!(error = %e, "Connection::client_info_name: mutex poisoned");
                None
            }
        }
    }

    fn try_mark_initialized(&self) -> bool {
        matches!(
            self.initialized
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire),
            Ok(false)
        )
    }

    /// Return this connection's isolated [`ToolContext`], lazily forking from `server_base`
    /// on first access so `SurrogateVault` and `AllowDestructiveBatch` are never shared
    /// across connections.
    fn get_or_init_ctx(&self, server_base: &ToolContext) -> ToolContext {
        match self.ctx.lock() {
            Ok(mut guard) => guard
                .get_or_insert_with(|| server_base.fork_connection(self.authenticated_caller()))
                .clone(),
            Err(e) => {
                warn!(error = %e, "Connection::get_or_init_ctx: mutex poisoned; forking fresh context");
                server_base.fork_connection(self.authenticated_caller())
            }
        }
    }

    fn set_ctx(&self, ctx: ToolContext) {
        match self.ctx.lock() {
            Ok(mut guard) => *guard = Some(ctx),
            Err(e) => {
                warn!(error = %e, "Connection::set_ctx: mutex poisoned");
            }
        }
    }

    fn get_view(&self) -> Option<CallerView> {
        match self.view.lock() {
            Ok(guard) => guard.clone(),
            Err(e) => {
                warn!(error = %e, "Connection::get_view: mutex poisoned");
                None
            }
        }
    }

    fn set_view(&self, view: Option<CallerView>) {
        match self.view.lock() {
            Ok(mut guard) => *guard = view,
            Err(e) => {
                warn!(error = %e, "Connection::set_view: mutex poisoned");
            }
        }
    }

    fn try_register_in_flight(
        &self,
        id: &crate::protocol::RequestId,
    ) -> Result<(tokio::sync::oneshot::Receiver<()>, InFlightGuard<'_>), DuplicateInFlightIdError>
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        match self.in_flight.lock() {
            Ok(mut guard) => {
                if guard.contains_key(id) {
                    return Err(DuplicateInFlightIdError);
                }
                guard.insert(id.clone(), tx);
            }
            Err(e) => {
                warn!(error = %e, "Connection::try_register_in_flight: mutex poisoned");
            }
        }
        Ok((
            rx,
            InFlightGuard {
                conn: self,
                id: id.clone(),
            },
        ))
    }

    fn remove_in_flight(&self, id: &crate::protocol::RequestId) {
        match self.in_flight.lock() {
            Ok(mut guard) => {
                guard.remove(id);
            }
            Err(e) => {
                warn!(error = %e, "Connection::remove_in_flight: mutex poisoned");
            }
        }
    }

    fn cancel_request(&self, id: &crate::protocol::RequestId) -> bool {
        let tx_opt = match self.in_flight.lock() {
            Ok(mut guard) => guard.remove(id),
            Err(e) => {
                warn!(error = %e, "Connection::cancel_request: mutex poisoned");
                None
            }
        };
        tx_opt.is_some_and(|tx| match tx.send(()) {
            Ok(()) => true,
            Err(()) => false,
        })
    }
}

impl McpServer {
    /// Create a new MCP server serving tools from `registry`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        McpServerBuilder::new(name, version, registry).build()
    }

    /// Begin building a server, allowing prompts, resources, limits, and caller authentication to be configured.
    #[must_use]
    pub fn builder(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> McpServerBuilder {
        McpServerBuilder::new(name, version, registry)
    }

    /// Set the base [`ToolContext`] forked per connection.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = Arc::new(context);
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
    /// ([`Connection::with_authenticated_caller`]) or [`with_caller_resolver`](Self::with_caller_resolver).
    #[must_use]
    pub const fn with_per_connection_identity(mut self, enabled: bool) -> Self {
        self.per_connection_identity = enabled;
        self
    }

    /// Opt in to trusting unauthenticated `clientInfo.name` for switching per-caller
    /// [`RegistryFactory`] views.
    ///
    /// Defaults to `false` (secure by default): unless enabled or verified via
    /// [`with_caller_resolver`](Self::with_caller_resolver) /
    /// [`Connection::with_authenticated_caller`], `clientInfo.name` cannot select
    /// a privileged per-caller registry.
    #[must_use]
    pub const fn trust_client_info_caller(mut self, trust: bool) -> Self {
        self.trust_client_info_caller = trust;
        self
    }

    /// Attach an authenticated [`CallerResolver`] used during `initialize`.
    #[must_use]
    pub fn with_caller_resolver<R: CallerResolver + 'static>(mut self, resolver: R) -> Self {
        self.caller_resolver = Some(Arc::new(resolver));
        self
    }

    /// Configure the maximum allowed JSON-RPC wire line length in bytes.
    #[must_use]
    pub const fn with_max_line_bytes(mut self, max_bytes: usize) -> Self {
        self.max_line_bytes = if max_bytes == 0 { 1 } else { max_bytes };
        self
    }

    /// Configure the maximum number of items in a JSON-RPC batch array.
    #[must_use]
    pub const fn with_max_batch_len(mut self, max_len: usize) -> Self {
        self.max_batch_len = if max_len == 0 { 1 } else { max_len };
        self
    }

    /// Configure the maximum number of concurrent in-flight requests per connection.
    #[must_use]
    pub const fn with_max_in_flight_requests(mut self, max_in_flight: usize) -> Self {
        self.max_in_flight_per_conn = if max_in_flight == 0 { 1 } else { max_in_flight };
        self
    }

    /// Configure the maximum number of concurrent accepted TCP/Unix connections.
    #[must_use]
    pub fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.connection_semaphore = Arc::new(Semaphore::new(max_connections.max(1)));
        self
    }

    /// Configure the maximum number of distinct per-caller views cached by [`RegistryFactory`].
    #[must_use]
    pub fn with_max_caller_views(mut self, max_views: usize) -> Self {
        let clamped = max_views.max(1);
        self.max_caller_views = clamped;
        let default_view = self.lock_views().default_view.clone();
        self.caller_views = Arc::new(Mutex::new(CallerViewCache::new(default_view, clamped)));
        self
    }

    /// Serve a **per-caller** tool registry via a [`RegistryFactory`].
    #[must_use]
    pub fn with_registry_factory<F: RegistryFactory + 'static>(mut self, factory: F) -> Self {
        let factory: Arc<dyn RegistryFactory> = Arc::new(factory);
        let registry = Arc::new(factory.registry_for(None));
        let cached_tools_list = Arc::new(build_tools_list_value(&registry));
        let default_view = CallerView {
            registry: Arc::clone(&registry),
            tools_list: Arc::clone(&cached_tools_list),
        };
        self.registry = registry;
        self.cached_tools_list = cached_tools_list;
        self.registry_factory = Some(factory);
        self.caller_views = Arc::new(Mutex::new(CallerViewCache::new(
            Some(default_view),
            self.max_caller_views,
        )));
        self
    }

    fn lock_views(&self) -> std::sync::MutexGuard<'_, CallerViewCache> {
        self.caller_views.lock().unwrap_or_else(|poisoned| {
            warn!("caller-view cache mutex was poisoned; recovering guard");
            poisoned.into_inner()
        })
    }

    fn resolve_view(&self, caller: Option<&str>) -> Option<CallerView> {
        let factory = self.registry_factory.as_ref()?;
        let cached = self.lock_views().get(caller);
        if let Some(view) = cached {
            return Some(view);
        }

        let registry = Arc::new(factory.registry_for(caller));
        let tools_list = Arc::new(build_tools_list_value(&registry));
        let view = CallerView {
            registry,
            tools_list,
        };

        Some(self.lock_views().get_or_insert(caller, view))
    }

    /// Borrow the underlying [`ToolRegistry`].
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }
}

// ── Schema helpers ──────────────────────────────────────────────────

fn build_tools_list_value(registry: &ToolRegistry) -> serde_json::Value {
    let tools = registry
        .definitions()
        .iter()
        .map(definition_to_mcp_schema)
        .collect();
    let list = ToolsListResult { tools };
    serde_json::to_value(list).expect("tools/list schema must be JSON-serializable")
}

/// Convert a [`ToolDefinition`] to the MCP `tools/list` schema format with behavior annotations.
fn definition_to_mcp_schema(def: &ToolDefinition) -> McpToolSchema {
    McpToolSchema {
        name: def.name.clone(),
        description: def.description.clone(),
        input_schema: def.parameter_schema.clone(),
        annotations: Some(McpToolAnnotations::from_definition(def)),
    }
}

fn build_prompts_list_value(prompts: &PromptRegistry) -> serde_json::Value {
    let list = crate::protocol::PromptsListResult {
        prompts: prompts.definitions(),
    };
    serde_json::to_value(list).expect("prompts/list schema must be JSON-serializable")
}

fn build_resources_list_value(resources: &ResourceRegistry) -> serde_json::Value {
    let list = crate::protocol::ResourcesListResult {
        resources: resources
            .definitions()
            .into_iter()
            .map(|def| crate::protocol::Resource {
                uri: def.uri_template,
                name: def.name,
                description: def.description,
                mime_type: def.mime_type,
            })
            .collect(),
    };
    serde_json::to_value(list).expect("resources/list schema must be JSON-serializable")
}

fn build_resource_templates_list_value(resources: &ResourceRegistry) -> serde_json::Value {
    let list = crate::protocol::ResourceTemplatesListResult {
        resource_templates: resources.definitions(),
    };
    serde_json::to_value(list).expect("resources/templates/list schema must be JSON-serializable")
}

#[cfg(test)]
mod tests;
