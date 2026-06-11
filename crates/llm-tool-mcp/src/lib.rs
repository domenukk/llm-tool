#![doc = include_str!("../README.md")]

pub mod protocol;
mod server;

pub use server::McpServer;
