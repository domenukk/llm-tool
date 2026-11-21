#![doc = include_str!("../README.md")]

pub mod protocol;
mod server;

pub use server::{
    CallerResolver, Connection, DEFAULT_MAX_CALLER_VIEWS, DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_IN_FLIGHT_PER_CONN, DEFAULT_MAX_JSONRPC_BATCH_LEN, DEFAULT_MAX_JSONRPC_LINE_BYTES,
    McpServer, McpServerBuilder, RegistryFactory, RpcOutcome, Transport,
};
