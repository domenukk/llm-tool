//! Transport selection and the blocking / async run loops for [`McpServer`].

use std::io::{self, BufRead, Write};

use tracing::{debug, error, info};

use super::{Connection, McpServer};

/// A transport for the blocking [`McpServer::serve`] entry point.
///
/// Lets a binary select stdio, TCP, or a Unix domain socket at runtime (e.g.
/// from CLI args) and hand it to [`serve`](McpServer::serve) in a single call,
/// instead of matching over transports and calling the individual blocking
/// runners itself.
///
/// This drives the **blocking** convenience runners. Tokio applications should
/// prefer the async [`run_async`](McpServer::run_async),
/// [`listen_tcp`](McpServer::listen_tcp), and
/// [`listen_unix`](McpServer::listen_unix) methods directly.
///
/// # Example
///
/// ```rust
/// use llm_tool_mcp::Transport;
///
/// let tcp = Transport::Tcp("127.0.0.1:3000".parse().unwrap());
/// // A binary might build one of these from `--stdio` / `--tcp <addr>` flags,
/// // then call `server.serve(transport)`.
/// assert!(matches!(tcp, Transport::Tcp(_)));
/// ```
#[derive(Debug, Clone)]
pub enum Transport {
    /// Serve on stdin/stdout — the standard MCP subprocess transport.
    Stdio,
    /// Serve on the given TCP socket address.
    Tcp(std::net::SocketAddr),
    /// Serve on a Unix domain socket at the given filesystem path.
    #[cfg(unix)]
    Unix(std::path::PathBuf),
}

impl McpServer {
    // ── Public entry points ─────────────────────────────────────────

    /// Run the server on stdin/stdout, blocking until stdin is closed.
    ///
    /// This is a blocking convenience for the common "just serve" binary: it
    /// builds a tokio runtime internally so `main` needs no `async`. If you are
    /// already inside a tokio application, drive the server with the async
    /// [`run_async`](Self::run_async) instead.
    ///
    /// Reads JSON-RPC lines from stdin, dispatches them, and writes
    /// responses to stdout.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    /// Use [`run_async`](Self::run_async) instead for async contexts.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created.
    pub fn run_stdio(&self) -> io::Result<()> {
        self.run(io::stdin().lock(), io::stdout().lock())
    }

    /// Serve MCP over TCP, blocking forever (until a fatal accept error).
    ///
    /// Blocking convenience mirroring [`run_stdio`](Self::run_stdio) for the
    /// TCP transport: it builds a **multi-threaded** tokio runtime internally
    /// (each accepted connection is served on its own task) and blocks on
    /// [`listen_tcp`](Self::listen_tcp). Use it for simple standalone binaries
    /// that only need to serve TCP.
    ///
    /// For tokio applications, prefer the async [`listen_tcp`](Self::listen_tcp)
    /// (or [`run_tcp_listener`](Self::run_tcp_listener)) so the server shares
    /// your existing runtime rather than spinning up a second one.
    ///
    /// - Localhost only: `server.run_tcp("127.0.0.1:3000")?`
    /// - External / Docker: `server.run_tcp("0.0.0.0:8080")?`
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime; use
    /// [`listen_tcp`](Self::listen_tcp) in that case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or binding to the
    /// TCP address fails.
    pub fn run_tcp(&self, addr: impl tokio::net::ToSocketAddrs) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.listen_tcp(addr))
    }

    /// Serve MCP over a Unix domain socket, blocking forever (until a fatal
    /// accept error).
    ///
    /// Blocking convenience mirroring [`run_stdio`](Self::run_stdio) for the
    /// Unix-socket transport: it builds a **multi-threaded** tokio runtime
    /// internally (each accepted connection is served on its own task) and
    /// blocks on [`listen_unix`](Self::listen_unix). Use it for simple
    /// standalone IPC binaries.
    ///
    /// For tokio applications, prefer the async
    /// [`listen_unix`](Self::listen_unix) so the server shares your existing
    /// runtime.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime; use
    /// [`listen_unix`](Self::listen_unix) in that case.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or binding to the
    /// domain socket path fails.
    #[cfg(unix)]
    pub fn run_unix(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.listen_unix(path))
    }

    /// Serve MCP over the given [`Transport`], blocking until it finishes.
    ///
    /// A one-call blocking dispatcher that collapses a stdio-vs-TCP-vs-Unix
    /// selection into a single line — ideal for a binary that picks its
    /// transport from CLI args or config:
    ///
    /// ```no_run
    /// # use llm_tool::ToolRegistry;
    /// # use llm_tool_mcp::{McpServer, Transport};
    /// let server = McpServer::new("srv", "0.1.0", ToolRegistry::new());
    /// let transport = Transport::Tcp("127.0.0.1:3000".parse().unwrap());
    /// server.serve(transport).expect("server failed");
    /// ```
    ///
    /// This is a blocking convenience built on [`run_stdio`](Self::run_stdio),
    /// [`run_tcp`](Self::run_tcp) and [`run_unix`](Self::run_unix). Tokio
    /// applications should instead drive the async
    /// [`run_async`](Self::run_async) / [`listen_tcp`](Self::listen_tcp) /
    /// [`listen_unix`](Self::listen_unix) methods directly.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or the underlying
    /// transport fails to bind / serve.
    pub fn serve(&self, transport: Transport) -> io::Result<()> {
        match transport {
            Transport::Stdio => self.run_stdio(),
            Transport::Tcp(addr) => self.run_tcp(addr),
            #[cfg(unix)]
            Transport::Unix(path) => self.run_unix(path),
        }
    }

    /// Run the server on arbitrary reader/writer streams.
    ///
    /// Creates a single-threaded tokio runtime for async tool dispatch
    /// and reuses it for every request.  The runtime is dropped when the
    /// reader is exhausted.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    /// Use [`run_async`](Self::run_async) instead for async contexts, or use
    /// [`run`](Self::run) inside [`tokio::task::spawn_blocking`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or a fatal
    /// write error occurs.
    pub fn run(&self, reader: impl BufRead, mut writer: impl Write) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        self.run_with_runtime(&rt, reader, &mut writer)
    }

    /// Run the server using an externally-provided tokio runtime.
    ///
    /// Use this when you already have a runtime and want to avoid the
    /// nested-runtime panic.  For the common case (standalone binary),
    /// prefer [`run_stdio`](Self::run_stdio) or [`run`](Self::run).
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal write errors.
    pub fn run_with_runtime(
        &self,
        rt: &tokio::runtime::Runtime,
        reader: impl BufRead,
        writer: &mut impl Write,
    ) -> io::Result<()> {
        // Reused across messages so response serialization amortizes to zero
        // allocations on the hot path.
        let mut out_buf: Vec<u8> = Vec::new();
        // Per-connection caller identity, negotiated at `initialize`. `None`
        // until then (and always, unless per-connection identity is enabled),
        // in which case the shared server context is used.
        let mut conn = Connection::default();
        for line_result in reader.lines() {
            let line = line_result?;

            if line.trim().is_empty() {
                continue;
            }

            debug!(request = %line, "mcp request");

            let Some(outcome) = rt.block_on(self.handle_message_conn(&line, &mut conn)) else {
                debug!("dropping notification response");
                continue;
            };

            out_buf.clear();
            outcome.write_json(&mut out_buf);
            debug!(response = %String::from_utf8_lossy(&out_buf), "mcp response");
            out_buf.push(b'\n');

            writer.write_all(&out_buf)?;
            writer.flush()?;
        }

        info!("input stream closed — shutting down");
        Ok(())
    }

    /// Run the server asynchronously on Tokio reader/writer streams.
    ///
    /// Reads line-delimited JSON-RPC requests from an async reader, dispatches
    /// them asynchronously without blocking threads or spinning up nested runtimes,
    /// and writes serialized responses back to an async writer.
    ///
    /// Ideal for integrating into existing Tokio applications, network servers,
    /// or when running inside an existing async runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal I/O errors reading requests or writing responses.
    pub async fn run_async(
        &self,
        reader: impl tokio::io::AsyncBufRead + Unpin,
        mut writer: impl tokio::io::AsyncWrite + Unpin,
    ) -> io::Result<()> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let mut lines = reader.lines();
        // Reused across messages so response serialization amortizes to zero
        // allocations on the hot path.
        let mut out_buf: Vec<u8> = Vec::new();
        // Per-connection caller identity, negotiated at `initialize`.
        let mut conn = Connection::default();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            debug!(request = %line, "mcp request");

            let Some(outcome) = self.handle_message_conn(&line, &mut conn).await else {
                debug!("dropping notification response");
                continue;
            };

            out_buf.clear();
            outcome.write_json(&mut out_buf);
            debug!(response = %String::from_utf8_lossy(&out_buf), "mcp response");
            out_buf.push(b'\n');

            writer.write_all(&out_buf).await?;
            writer.flush().await?;
        }

        info!("input stream closed — shutting down");
        Ok(())
    }

    /// Listen for TCP connections and serve MCP requests asynchronously on each connection.
    ///
    /// This allows remote clients, IDEs, or multi-client agents to connect over TCP:
    /// - Localhost only: `server.listen_tcp("127.0.0.1:3000").await?`
    /// - External / Docker: `server.listen_tcp("0.0.0.0:8080").await?`
    ///
    /// For each incoming connection, a new Tokio task is spawned running [`run_async`](Self::run_async).
    ///
    /// # Errors
    ///
    /// Returns `Err` if binding to the TCP address fails.
    pub async fn listen_tcp(&self, addr: impl tokio::net::ToSocketAddrs) -> io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!(addr = ?listener.local_addr()?, "listening on TCP for MCP connections");
        self.run_tcp_listener(listener).await
    }

    /// Serve MCP requests asynchronously on an existing [`tokio::net::TcpListener`].
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal accept loop errors.
    pub async fn run_tcp_listener(&self, listener: tokio::net::TcpListener) -> io::Result<()> {
        loop {
            let (mut socket, peer_addr) = listener.accept().await?;
            info!(peer = %peer_addr, "accepted MCP TCP connection");

            let server = self.clone();
            tokio::spawn(async move {
                let (reader, writer) = socket.split();
                let reader = tokio::io::BufReader::new(reader);
                if let Err(e) = server.run_async(reader, writer).await {
                    error!(peer = %peer_addr, error = %e, "MCP TCP connection error");
                }
                info!(peer = %peer_addr, "MCP TCP connection closed");
            });
        }
    }

    #[cfg(unix)]
    /// Listen on a Unix domain socket and serve MCP requests asynchronously on each connection.
    ///
    /// This allows local IPC clients to connect over a filesystem domain socket
    /// (e.g. `/tmp/my-agent.sock`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if binding to the domain socket path fails.
    pub async fn listen_unix(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let path = path.as_ref();
        if path.exists() {
            if let Err(e) = std::fs::remove_file(path) {
                tracing::warn!(path = ?path, error = %e, "failed to remove stale socket file");
            }
        }
        let listener = tokio::net::UnixListener::bind(path)?;
        info!(path = ?path, "listening on Unix domain socket for MCP connections");
        self.run_unix_listener(listener).await
    }

    #[cfg(unix)]
    /// Serve MCP requests asynchronously on an existing [`tokio::net::UnixListener`].
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal accept loop errors.
    pub async fn run_unix_listener(&self, listener: tokio::net::UnixListener) -> io::Result<()> {
        loop {
            let (mut socket, _) = listener.accept().await?;
            info!("accepted MCP Unix domain socket connection");

            let server = self.clone();
            tokio::spawn(async move {
                let (reader, writer) = socket.split();
                let reader = tokio::io::BufReader::new(reader);
                if let Err(e) = server.run_async(reader, writer).await {
                    error!(error = %e, "MCP Unix connection error");
                }
                info!("MCP Unix connection closed");
            });
        }
    }
}
