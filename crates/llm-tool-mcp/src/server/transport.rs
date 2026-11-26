//! Transport selection and bounded blocking / async run loops for [`McpServer`].

use std::{
    io::{self, BufRead, Read, Write},
    sync::Arc,
};

use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt},
    sync::Semaphore,
};
use tracing::{debug, error, info};

use super::{Connection, McpServer, RpcOutcome};
use crate::protocol::{self, JsonRpcResponse};

/// A transport for the blocking [`McpServer::serve`] entry point.
///
/// # Example
///
/// ```rust
/// use llm_tool_mcp::Transport;
///
/// let tcp = Transport::Tcp("127.0.0.1:3000".parse().unwrap());
/// assert!(matches!(tcp, Transport::Tcp(_)));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(not(unix), derive(Copy))]
pub enum Transport {
    /// Serve on stdin/stdout — the standard MCP subprocess transport.
    Stdio,
    /// Serve on the given TCP socket address.
    Tcp(std::net::SocketAddr),
    /// Serve on a Unix domain socket at the given filesystem path.
    #[cfg(unix)]
    Unix(std::path::PathBuf),
}

enum BoundedLine<'a> {
    Eof,
    Line(&'a str),
    Oversized,
    InvalidUtf8(std::str::Utf8Error),
}

fn strip_newline_len(buf: &[u8]) -> usize {
    let without_lf = buf.strip_suffix(b"\n").unwrap_or(buf);
    without_lf.strip_suffix(b"\r").unwrap_or(without_lf).len()
}

fn drain_until_newline_sync(reader: &mut impl BufRead) -> io::Result<()> {
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(());
        }
        if let Some(nl_idx) = available.iter().position(|&b| b == b'\n') {
            reader.consume(nl_idx + 1);
            return Ok(());
        }
        let len = available.len();
        reader.consume(len);
    }
}

async fn drain_until_newline_async(reader: &mut (impl AsyncBufRead + Unpin)) -> io::Result<()> {
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(());
        }
        if let Some(nl_idx) = available.iter().position(|&b| b == b'\n') {
            reader.consume(nl_idx + 1);
            return Ok(());
        }
        let len = available.len();
        reader.consume(len);
    }
}

fn read_bounded_line_sync<'a>(
    reader: &mut impl BufRead,
    buf: &'a mut Vec<u8>,
    max_line_bytes: usize,
) -> io::Result<BoundedLine<'a>> {
    buf.clear();
    let take_limit = (max_line_bytes as u64).saturating_add(2);
    let n = reader.by_ref().take(take_limit).read_until(b'\n', buf)?;
    if n == 0 {
        return Ok(BoundedLine::Eof);
    }
    let ended_with_nl = buf.last() == Some(&b'\n');
    let content_len = strip_newline_len(buf);
    if !ended_with_nl && content_len > max_line_bytes {
        buf.clear();
        drain_until_newline_sync(reader)?;
        return Ok(BoundedLine::Oversized);
    }
    if content_len > max_line_bytes {
        buf.clear();
        return Ok(BoundedLine::Oversized);
    }
    match std::str::from_utf8(&buf[..content_len]) {
        Ok(line) => Ok(BoundedLine::Line(line)),
        Err(err) => Ok(BoundedLine::InvalidUtf8(err)),
    }
}

async fn read_bounded_line_async<'a>(
    reader: &mut (impl AsyncBufRead + Unpin),
    buf: &'a mut Vec<u8>,
    max_line_bytes: usize,
) -> io::Result<BoundedLine<'a>> {
    buf.clear();
    let take_limit = (max_line_bytes as u64).saturating_add(2);
    let n = (&mut *reader)
        .take(take_limit)
        .read_until(b'\n', buf)
        .await?;
    if n == 0 {
        return Ok(BoundedLine::Eof);
    }
    let ended_with_nl = buf.last() == Some(&b'\n');
    let content_len = strip_newline_len(buf);
    if !ended_with_nl && content_len > max_line_bytes {
        buf.clear();
        drain_until_newline_async(reader).await?;
        return Ok(BoundedLine::Oversized);
    }
    if content_len > max_line_bytes {
        buf.clear();
        return Ok(BoundedLine::Oversized);
    }
    match std::str::from_utf8(&buf[..content_len]) {
        Ok(line) => Ok(BoundedLine::Line(line)),
        Err(err) => Ok(BoundedLine::InvalidUtf8(err)),
    }
}

impl McpServer {
    /// Run the server on stdin/stdout, blocking until stdin is closed.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created.
    pub fn run_stdio(&self) -> io::Result<()> {
        self.run(io::stdin().lock(), io::stdout().lock())
    }

    /// Serve MCP over TCP, blocking forever (until a fatal accept error).
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or binding fails.
    pub fn run_tcp(&self, addr: impl tokio::net::ToSocketAddrs) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.listen_tcp(addr))
    }

    /// Serve MCP over a Unix domain socket, blocking forever (until a fatal accept error).
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or binding fails.
    #[cfg(unix)]
    pub fn run_unix(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.listen_unix(path))
    }

    /// Serve MCP over the given [`Transport`], blocking until it finishes.
    ///
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or the underlying transport fails.
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
    /// # Panics
    ///
    /// Panics if called from within an existing tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tokio runtime cannot be created or a fatal write error occurs.
    pub fn run(&self, reader: impl BufRead, mut writer: impl Write) -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        self.run_with_runtime(&rt, reader, &mut writer)
    }

    /// Run the server using an externally-provided tokio runtime.
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal read/write errors.
    pub fn run_with_runtime(
        &self,
        rt: &tokio::runtime::Runtime,
        mut reader: impl BufRead,
        writer: &mut impl Write,
    ) -> io::Result<()> {
        let mut line_buf: Vec<u8> = Vec::new();
        let mut out_buf: Vec<u8> = Vec::new();
        let conn = Connection::default();

        loop {
            let outcome_opt =
                match read_bounded_line_sync(&mut reader, &mut line_buf, self.max_line_bytes)? {
                    BoundedLine::Eof => break,
                    BoundedLine::Oversized => Some(RpcOutcome::Single(JsonRpcResponse::error(
                        None,
                        protocol::INVALID_REQUEST,
                        format!(
                            "JSON-RPC message exceeds maximum length of {} bytes",
                            self.max_line_bytes
                        ),
                    ))),
                    BoundedLine::InvalidUtf8(err) => {
                        Some(RpcOutcome::Single(JsonRpcResponse::error(
                            None,
                            protocol::PARSE_ERROR,
                            format!("invalid UTF-8 in JSON-RPC message: {err}"),
                        )))
                    }
                    BoundedLine::Line(line) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        debug!(request = %line, "mcp request");
                        let parsed = self.parse_message(line);
                        rt.block_on(self.dispatch_parsed_conn(parsed, &conn))
                    }
                };

            let Some(outcome) = outcome_opt else {
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

    /// Run the server asynchronously on Tokio reader/writer streams with bounded line framing
    /// and bounded per-connection concurrency.
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal I/O errors reading requests or writing responses.
    pub async fn run_async(
        &self,
        mut reader: impl AsyncBufRead + Unpin,
        mut writer: impl tokio::io::AsyncWrite + Unpin,
    ) -> io::Result<()> {
        let mut line_buf: Vec<u8> = Vec::new();
        let mut out_buf: Vec<u8> = Vec::new();
        let conn = Connection::default();
        let in_flight_sem = Arc::new(Semaphore::new(self.max_in_flight_per_conn));
        let mut in_flight_tasks: tokio::task::JoinSet<Option<RpcOutcome>> =
            tokio::task::JoinSet::new();

        loop {
            tokio::select! {
                read_res = read_bounded_line_async(&mut reader, &mut line_buf, self.max_line_bytes) => {
                    let parsed = match read_res? {
                        BoundedLine::Eof => break,
                        BoundedLine::Oversized => {
                            let err_out = RpcOutcome::Single(JsonRpcResponse::error(
                                None,
                                protocol::INVALID_REQUEST,
                                format!(
                                    "JSON-RPC message exceeds maximum length of {} bytes",
                                    self.max_line_bytes
                                ),
                            ));
                            write_outcome_async(&mut writer, &mut out_buf, &err_out).await?;
                            continue;
                        }
                        BoundedLine::InvalidUtf8(err) => {
                            let err_out = RpcOutcome::Single(JsonRpcResponse::error(
                                None,
                                protocol::PARSE_ERROR,
                                format!("invalid UTF-8 in JSON-RPC message: {err}"),
                            ));
                            write_outcome_async(&mut writer, &mut out_buf, &err_out).await?;
                            continue;
                        }
                        BoundedLine::Line(line) => {
                            if line.trim().is_empty() {
                                continue;
                            }
                            debug!(request = %line, "mcp request");
                            self.parse_message(line)
                        }
                    };

                    if parsed.contains_initialize() {
                        drain_in_flight(&mut in_flight_tasks, &mut writer, &mut out_buf).await?;
                        if let Some(outcome) = self.dispatch_parsed_conn(parsed, &conn).await {
                            write_outcome_async(&mut writer, &mut out_buf, &outcome).await?;
                        }
                    } else if parsed.is_single_notification() {
                        // Notifications (such as `notifications/cancelled`) run inline without
                        // acquiring an in-flight request slot so cancellation never deadlocks.
                        if let Some(unexpected) = self.dispatch_parsed_conn(parsed, &conn).await {
                            write_outcome_async(&mut writer, &mut out_buf, &unexpected).await?;
                        }
                    } else {
                        let permit = acquire_in_flight_permit(
                            &in_flight_sem,
                            &mut in_flight_tasks,
                            &mut writer,
                            &mut out_buf,
                        )
                        .await?;
                        let server = self.clone();
                        let conn_clone = conn.clone();
                        in_flight_tasks.spawn(async move {
                            let _permit = permit;
                            server.dispatch_parsed_conn(parsed, &conn_clone).await
                        });
                    }
                }
                Some(res) = in_flight_tasks.join_next(), if !in_flight_tasks.is_empty() => {
                    handle_joined_task(res, &mut writer, &mut out_buf).await?;
                }
            }
        }

        drain_in_flight(&mut in_flight_tasks, &mut writer, &mut out_buf).await?;
        info!("input stream closed — shutting down");
        Ok(())
    }

    /// Listen for TCP connections and serve MCP requests asynchronously on each connection.
    ///
    /// # Errors
    ///
    /// Returns `Err` if binding to the TCP address fails.
    pub async fn listen_tcp(&self, addr: impl tokio::net::ToSocketAddrs) -> io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!(addr = ?listener.local_addr()?, "listening on TCP for MCP connections");
        self.run_tcp_listener(listener).await
    }

    /// Serve MCP requests asynchronously on an existing [`tokio::net::TcpListener`],
    /// bounded by the server's concurrent connection semaphore.
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal accept loop errors.
    pub async fn run_tcp_listener(&self, listener: tokio::net::TcpListener) -> io::Result<()> {
        loop {
            let permit = Arc::clone(&self.connection_semaphore)
                .acquire_owned()
                .await
                .map_err(|e| io::Error::other(format!("connection semaphore closed: {e}")))?;
            let (mut socket, peer_addr) = listener.accept().await?;
            info!(peer = %peer_addr, "accepted MCP TCP connection");

            let server = self.clone();
            tokio::spawn(async move {
                let _permit = permit;
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
    /// Verifies via [`std::fs::symlink_metadata`] that any pre-existing path is a Unix domain
    /// socket before unlinking stale socket files (failing with [`io::ErrorKind::AlreadyExists`]
    /// if `path` is a regular file, directory, or symlink).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `path` is a non-socket filesystem entry, already actively bound, or binding fails.
    pub async fn listen_unix(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        use std::os::unix::fs::FileTypeExt;

        let path = path.as_ref();
        match std::fs::symlink_metadata(path) {
            Ok(meta) => {
                if !meta.file_type().is_socket() {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        format!("refusing to unlink non-socket path {}", path.display()),
                    ));
                }
                match tokio::net::UnixStream::connect(path).await {
                    Ok(_active_stream) => {
                        return Err(io::Error::new(
                            io::ErrorKind::AddrInUse,
                            format!(
                                "another process is already listening on Unix socket {}",
                                path.display()
                            ),
                        ));
                    }
                    Err(connect_err) => {
                        debug!(
                            path = ?path,
                            error = %connect_err,
                            "existing Unix socket is stale; unlinking before bind"
                        );
                    }
                }
                std::fs::remove_file(path)?;
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }

        let listener = tokio::net::UnixListener::bind(path)?;
        info!(path = ?path, "listening on Unix domain socket for MCP connections");
        self.run_unix_listener(listener).await
    }

    #[cfg(unix)]
    /// Serve MCP requests asynchronously on an existing [`tokio::net::UnixListener`],
    /// bounded by the server's concurrent connection semaphore.
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal accept loop errors.
    pub async fn run_unix_listener(&self, listener: tokio::net::UnixListener) -> io::Result<()> {
        loop {
            let permit = Arc::clone(&self.connection_semaphore)
                .acquire_owned()
                .await
                .map_err(|e| io::Error::other(format!("connection semaphore closed: {e}")))?;
            let (mut socket, _) = listener.accept().await?;
            info!("accepted MCP Unix domain socket connection");

            let server = self.clone();
            tokio::spawn(async move {
                let _permit = permit;
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

async fn acquire_in_flight_permit(
    sem: &Arc<Semaphore>,
    in_flight_tasks: &mut tokio::task::JoinSet<Option<RpcOutcome>>,
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    out_buf: &mut Vec<u8>,
) -> io::Result<tokio::sync::OwnedSemaphorePermit> {
    match Arc::clone(sem).try_acquire_owned() {
        Ok(permit) => return Ok(permit),
        Err(err) => {
            debug!(error = %err, "per-connection in-flight slots full; awaiting task completion");
        }
    }
    loop {
        tokio::select! {
            permit_res = Arc::clone(sem).acquire_owned() => {
                return permit_res.map_err(|e| io::Error::other(format!("in-flight semaphore closed: {e}")));
            }
            Some(res) = in_flight_tasks.join_next(), if !in_flight_tasks.is_empty() => {
                handle_joined_task(res, writer, out_buf).await?;
            }
        }
    }
}

async fn drain_in_flight(
    in_flight_tasks: &mut tokio::task::JoinSet<Option<RpcOutcome>>,
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    out_buf: &mut Vec<u8>,
) -> io::Result<()> {
    while let Some(res) = in_flight_tasks.join_next().await {
        handle_joined_task(res, writer, out_buf).await?;
    }
    Ok(())
}

async fn handle_joined_task(
    res: Result<Option<RpcOutcome>, tokio::task::JoinError>,
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    out_buf: &mut Vec<u8>,
) -> io::Result<()> {
    match res {
        Ok(Some(outcome)) => write_outcome_async(writer, out_buf, &outcome).await,
        Ok(None) => {
            debug!("dropping notification response");
            Ok(())
        }
        Err(e) => {
            error!(error = %e, "request handler task panicked");
            Ok(())
        }
    }
}

async fn write_outcome_async(
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    out_buf: &mut Vec<u8>,
    outcome: &RpcOutcome,
) -> io::Result<()> {
    out_buf.clear();
    outcome.write_json(out_buf);
    debug!(response = %String::from_utf8_lossy(out_buf), "mcp response");
    out_buf.push(b'\n');
    writer.write_all(out_buf).await?;
    writer.flush().await
}
