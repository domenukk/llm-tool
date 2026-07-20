//! MCP stdio server backed by a [`ToolRegistry`].
//!
//! [`McpServer`] wraps a [`ToolRegistry`] and exposes it via the
//! [Model Context Protocol](https://modelcontextprotocol.io/) over any
//! `BufRead`/`Write` pair (typically stdin/stdout).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐   JSON-RPC    ┌───────────┐   dispatch   ┌──────────────┐
//! │  Client  │──────────────▶│ McpServer │─────────────▶│ ToolRegistry │
//! │ (stdin)  │◀──────────────│           │◀─────────────│              │
//! └─────────┘   JSON-RPC    └───────────┘   Result     └──────────────┘
//! ```
//!
//! # Performance
//!
//! - MCP tool schemas are computed **once** at construction and cached.
//! - [`run`](McpServer::run) creates a **single** tokio `current_thread`
//!   runtime, reused for all dispatches.
//! - The `"2.0"` JSON-RPC version is a `&'static str` to avoid allocation.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use llm_tool::{PromptRegistry, ResourceRegistry, ToolContext, ToolDefinition, ToolRegistry};
use tracing::warn;

// Re-exported so the `#[cfg(test)] mod tests` submodule keeps the same protocol
// names in scope (via its glob import of this module) as before the split into
// `builder` / `transport` / `dispatch`.
#[cfg(test)]
pub(crate) use crate::protocol::{self, *};
use crate::protocol::{McpToolSchema, ToolsListResult};

mod builder;
mod dispatch;
mod transport;

pub use builder::McpServerBuilder;
pub use dispatch::RpcOutcome;
pub use transport::Transport;

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
/// // In production: server.run_stdio().expect("MCP server failed");
/// // Here we prove it works with an in-memory request:
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
    registry: Arc<ToolRegistry>,
    context: Arc<ToolContext>,
    /// Pre-serialized `tools/list` result body — built once at construction and
    /// wrapped in `Arc` so `tools/list` clones a pointer plus one JSON value,
    /// never re-serializing the schema tree.
    cached_tools_list: Arc<serde_json::Value>,
    prompts: Arc<PromptRegistry>,
    resources: Arc<ResourceRegistry>,
    /// When `true`, each connection derives its own caller identity from the
    /// `clientInfo.name` sent in that connection's MCP `initialize` handshake,
    /// letting a single server serve many distinct callers. When `false` (the
    /// default) every dispatch uses the one shared [`ToolContext`] identity.
    per_connection_identity: bool,
    /// Optional factory that builds a **per-caller** [`ToolRegistry`] the first
    /// time each negotiated caller is seen. When set, `tools/list` and
    /// `tools/call` use the caller's own registry (see [`RegistryFactory`]);
    /// when `None`, every connection shares [`registry`](Self::registry).
    registry_factory: Option<Arc<dyn RegistryFactory>>,
    /// Memoized caller → view cache. Keyed by the negotiated caller name (empty
    /// string for the default/no-identity view). Shared across cloned servers
    /// and connections so each caller's registry is built and serialized once.
    caller_views: Arc<Mutex<HashMap<String, CallerView>>>,
}

/// Builds the [`ToolRegistry`] a given caller should see.
///
/// A single server can present a *different* tool set — and different tool
/// descriptions — to each connection based on the caller negotiated at
/// `initialize`. This is what lets "one MCP, many agents" preserve per-caller
/// tailoring (gating privileged tools, personalising descriptions, …) rather
/// than serving one fixed tool list to everyone.
///
/// Any `Fn(Option<&str>) -> ToolRegistry` implements this trait, so a closure
/// is usually all that's needed. Pair it with
/// [`with_per_connection_identity`](McpServerBuilder::with_per_connection_identity)
/// so a caller is actually negotiated:
///
/// ```rust
/// use llm_tool::ToolRegistry;
/// use llm_tool_mcp::McpServer;
///
/// let server = McpServer::builder("srv", "0.1.0", ToolRegistry::new())
///     .with_per_connection_identity(true)
///     .with_registry_factory(|_caller: Option<&str>| {
///         // Build a registry tailored to `_caller`.
///         ToolRegistry::new()
///     })
///     .build();
/// # let _server = server;
/// ```
pub trait RegistryFactory: Send + Sync {
    /// Build the registry for `caller`.
    ///
    /// `caller` is `None` for the default view (before/without a negotiated
    /// identity), or `Some(name)` for a connection that announced itself.
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
    registry: Arc<ToolRegistry>,
    tools_list: Arc<serde_json::Value>,
}

/// Per-connection negotiated state, owned by each transport run loop.
///
/// Custom transports built on [`handle_message_conn`](McpServer::handle_message_conn)
/// create one [`Connection`] per client (via [`Connection::new`] /
/// [`Default`]) and pass `&mut` to it for every message on that connection, so
/// the caller identity and per-caller registry view negotiated at `initialize`
/// persist for the connection's lifetime.
///
/// Defaults to "nothing negotiated yet": dispatch then falls back to the
/// server's shared identity, registry, and cached `tools/list`.
#[derive(Default)]
pub struct Connection {
    /// Caller identity adopted from this connection's `initialize` handshake
    /// (when per-connection identity is enabled); `None` uses the shared one.
    ctx: Option<ToolContext>,
    /// Caller-specific registry view (when a [`RegistryFactory`] is set);
    /// `None` uses the server's shared registry + cached `tools/list`.
    view: Option<CallerView>,
}

impl Connection {
    /// Create a fresh, un-negotiated connection state.
    ///
    /// Equivalent to [`Connection::default`]; provided for call-site clarity in
    /// transport loops.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl McpServer {
    /// Create a new MCP server serving tools from `registry`.
    ///
    /// The `name` and `version` are reported in the MCP `initialize` response.
    /// To also register prompts or resources, use [`builder`](Self::builder).
    ///
    /// Tool schemas are computed **once** here and cached for all subsequent
    /// `tools/list` requests.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        McpServerBuilder::new(name, version, registry).build()
    }

    /// Begin building a server, allowing prompts and resources to be registered
    /// before [`build`](McpServerBuilder::build).
    #[must_use]
    pub fn builder(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> McpServerBuilder {
        McpServerBuilder::new(name, version, registry)
    }

    /// Set the [`ToolContext`] used for all tool dispatches.
    ///
    /// The context provides the conversation ID and a shared state store
    /// that persists across tool calls.
    #[must_use]
    pub fn with_context(mut self, context: ToolContext) -> Self {
        self.context = Arc::new(context);
        self
    }

    /// Enable per-connection caller identity.
    ///
    /// When enabled, each connection derives its own caller from the
    /// `clientInfo.name` in its MCP `initialize` handshake (sharing the base
    /// context's state and typed extensions via [`ToolContext::with_caller`]),
    /// so a single long-lived server can serve many agents under distinct
    /// personas. Defaults to `false`, preserving the single-caller behaviour.
    #[must_use]
    pub const fn with_per_connection_identity(mut self, enabled: bool) -> Self {
        self.per_connection_identity = enabled;
        self
    }

    /// Serve a **per-caller** tool registry via a [`RegistryFactory`].
    ///
    /// Post-construction counterpart of
    /// [`McpServerBuilder::with_registry_factory`]: it recomputes the default
    /// view from `registry_for(None)` and reseeds the caller cache, so a server
    /// created with [`new`](Self::new) can still opt into per-caller registries.
    #[must_use]
    pub fn with_registry_factory<F: RegistryFactory + 'static>(mut self, factory: F) -> Self {
        let factory: Arc<dyn RegistryFactory> = Arc::new(factory);
        let registry = Arc::new(factory.registry_for(None));
        let cached_tools_list = Arc::new(build_tools_list_value(&registry));
        let mut views = HashMap::new();
        views.insert(
            String::new(),
            CallerView {
                registry: Arc::clone(&registry),
                tools_list: Arc::clone(&cached_tools_list),
            },
        );
        self.registry = registry;
        self.cached_tools_list = cached_tools_list;
        self.registry_factory = Some(factory);
        self.caller_views = Arc::new(Mutex::new(views));
        self
    }

    /// Lock the caller-view cache, recovering a poisoned guard rather than
    /// propagating the panic.
    ///
    /// The cache holds only derived, idempotently-rebuildable views, so reusing
    /// a poisoned guard is safe. The poisoning is logged so the original panic
    /// isn't silently swallowed.
    fn lock_views(&self) -> std::sync::MutexGuard<'_, HashMap<String, CallerView>> {
        self.caller_views.lock().unwrap_or_else(|poisoned| {
            warn!("caller-view cache mutex was poisoned; recovering guard");
            poisoned.into_inner()
        })
    }

    /// Resolve (and memoize) the caller-specific [`CallerView`] for `caller`.
    ///
    /// Returns `None` when no [`RegistryFactory`] is configured, so dispatch
    /// falls back to the shared registry and cached `tools/list`. The registry
    /// is built and its schema serialized only once per distinct caller.
    fn resolve_view(&self, caller: Option<&str>) -> Option<CallerView> {
        let factory = self.registry_factory.as_ref()?;
        // `None` (no negotiated caller) maps to the empty-string default-view key.
        let key = caller.unwrap_or("");

        // Fast path: hand back a cached view without holding the lock across the
        // (potentially expensive) build below.
        if let Some(view) = self.lock_views().get(key) {
            return Some(view.clone());
        }

        // Build outside the lock: registry construction and schema serialization
        // can be costly, and many agents may initialize concurrently.
        let registry = Arc::new(factory.registry_for(caller));
        let tools_list = Arc::new(build_tools_list_value(&registry));
        let view = CallerView {
            registry,
            tools_list,
        };

        // Insert, tolerating a concurrent build that beat us to this caller.
        Some(
            self.lock_views()
                .entry(key.to_owned())
                .or_insert(view)
                .clone(),
        )
    }

    /// Borrow the underlying [`ToolRegistry`].
    ///
    /// Useful for extracting definitions or dispatching outside MCP.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }
}

// ── Schema helpers ──────────────────────────────────────────────────

/// Build and pre-serialize the cached `tools/list` response body.
///
/// Called once at [`McpServerBuilder::build`] — the resulting JSON value is
/// wrapped in an `Arc` so `tools/list` clones a pointer plus one value rather
/// than re-serializing the schema tree on every request.
fn build_tools_list_value(registry: &ToolRegistry) -> serde_json::Value {
    let tools = registry
        .definitions()
        .iter()
        .map(definition_to_mcp_schema)
        .collect();
    let list = ToolsListResult { tools };
    serde_json::to_value(list).expect("tools/list schema must be JSON-serializable")
}

/// Convert a [`ToolDefinition`] to the MCP `tools/list` schema format.
fn definition_to_mcp_schema(def: &ToolDefinition) -> McpToolSchema {
    McpToolSchema {
        name: def.name.clone(),
        description: def.description.clone(),
        input_schema: def.parameter_schema.clone(),
    }
}

#[cfg(test)]
mod tests;
