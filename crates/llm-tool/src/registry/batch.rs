//! Concurrent and sequential batch tool execution for [`super::ToolRegistry`].

use alloc::{boxed::Box, string::String, vec::Vec};
use core::future;

use serde::{Deserialize, Serialize};

use super::ToolRegistry;
use crate::{
    rust_tool::BoxToolFuture,
    types::{ToolContext, ToolError, ToolOutput},
};

/// Execution strategy for a batch of tool invocations.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum BatchExecutionMode {
    /// Execute all tool calls concurrently and collect outcomes in original call order.
    ///
    /// If any tool in the batch has [`crate::types::ToolEffect::Destructive`], its execution
    /// will fail with a [`ToolError`] unless [`AllowDestructiveBatch`] is present in context
    /// or [`BatchExecutionMode::ParallelAllowDestructive`] is used.
    #[default]
    Parallel,
    /// Execute all tool calls concurrently, explicitly confirming and allowing destructive tools.
    ParallelAllowDestructive,
    /// Execute tool calls sequentially in call order, halting immediately on the first error.
    SequentialStopOnError,
}

/// Marker context extension to explicitly allow destructive tools in parallel batch execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowDestructiveBatch(pub bool);

/// Owned tool invocation specification within a batch request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BatchToolCall {
    /// Caller-assigned tool call identifier.
    #[serde(default)]
    pub id: String,
    /// Target registered tool name.
    pub name: String,
    /// JSON arguments object passed to the tool.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Borrowed string-based tool invocation specification for zero-DOM-allocation batch dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchToolCallStr<'a> {
    /// Caller-assigned tool call identifier.
    pub id: &'a str,
    /// Target registered tool name.
    pub name: &'a str,
    /// Raw JSON arguments string.
    pub arguments_json: &'a str,
}

/// Outcome of an individual tool invocation within a batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchToolCallOutcome {
    /// Caller-assigned tool call identifier.
    pub id: String,
    /// Target tool name.
    pub name: String,
    /// Result produced by the tool handler.
    pub result: Result<ToolOutput, ToolError>,
}

impl ToolRegistry {
    /// Dispatches a batch of [`BatchToolCall`] items according to `mode`, preserving call order.
    ///
    /// - [`BatchExecutionMode::Parallel`]: polls all tool futures concurrently using
    ///   [`futures_util::future::join_all`], returning every outcome in exact input slice order.
    ///   Destructive tools require explicit confirmation via [`BatchExecutionMode::ParallelAllowDestructive`]
    ///   or [`AllowDestructiveBatch`] on `ctx`.
    /// - [`BatchExecutionMode::SequentialStopOnError`]: runs calls one by one and stops after
    ///   the first `Err(ToolError)`.
    pub async fn dispatch_batch(
        &self,
        calls: &[BatchToolCall],
        mode: BatchExecutionMode,
        ctx: &ToolContext,
    ) -> Vec<BatchToolCallOutcome> {
        match mode {
            BatchExecutionMode::Parallel | BatchExecutionMode::ParallelAllowDestructive => {
                let allow_destructive =
                    matches!(mode, BatchExecutionMode::ParallelAllowDestructive)
                        || ctx
                            .get_ext::<AllowDestructiveBatch>()
                            .is_some_and(|ext| ext.0);

                let futures: Vec<BoxToolFuture<'_>> = calls
                    .iter()
                    .map(|call| -> BoxToolFuture<'_> {
                        if !allow_destructive && self.is_destructive(&call.name) {
                            tracing::warn!(
                                tool = %call.name,
                                "Destructive tool rejected in parallel batch execution without explicit confirmation"
                            );
                            let fut: BoxToolFuture<'_> = Box::pin(future::ready(Err(
                                ToolError::destructive_in_parallel_batch(&call.name),
                            )));
                            return fut;
                        }
                        match self.dispatch_boxed(&call.name, call.arguments.clone(), ctx) {
                            Ok(fut) => fut,
                            Err(err) => {
                                let fut: BoxToolFuture<'_> = Box::pin(future::ready(Err(err)));
                                fut
                            }
                        }
                    })
                    .collect();

                let results = futures_util::future::join_all(futures).await;
                calls
                    .iter()
                    .zip(results)
                    .map(|(call, result)| BatchToolCallOutcome {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        result,
                    })
                    .collect()
            }
            BatchExecutionMode::SequentialStopOnError => {
                let mut outcomes = Vec::with_capacity(calls.len());
                for call in calls {
                    let result = self.dispatch(&call.name, call.arguments.clone(), ctx).await;
                    let should_stop = match &result {
                        Ok(_) => false,
                        Err(err) => {
                            tracing::debug!(
                                error = %err,
                                tool = %call.name,
                                "Sequential batch execution stopped due to error"
                            );
                            true
                        }
                    };
                    outcomes.push(BatchToolCallOutcome {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        result,
                    });
                    if should_stop {
                        break;
                    }
                }
                outcomes
            }
        }
    }

    /// Dispatches a batch of borrowed [`BatchToolCallStr`] items according to `mode`,
    /// deserializing directly from raw JSON strings without intermediate DOM allocation.
    pub async fn dispatch_batch_str(
        &self,
        calls: &[BatchToolCallStr<'_>],
        mode: BatchExecutionMode,
        ctx: &ToolContext,
    ) -> Vec<BatchToolCallOutcome> {
        match mode {
            BatchExecutionMode::Parallel | BatchExecutionMode::ParallelAllowDestructive => {
                let allow_destructive =
                    matches!(mode, BatchExecutionMode::ParallelAllowDestructive)
                        || ctx
                            .get_ext::<AllowDestructiveBatch>()
                            .is_some_and(|ext| ext.0);

                let futures: Vec<BoxToolFuture<'_>> = calls
                    .iter()
                    .map(|call| -> BoxToolFuture<'_> {
                        if !allow_destructive && self.is_destructive(call.name) {
                            tracing::warn!(
                                tool = %call.name,
                                "Destructive tool rejected in parallel batch execution without explicit confirmation"
                            );
                            let fut: BoxToolFuture<'_> = Box::pin(future::ready(Err(
                                ToolError::destructive_in_parallel_batch(call.name),
                            )));
                            return fut;
                        }
                        match self.dispatch_boxed_str(call.name, call.arguments_json, ctx) {
                            Ok(fut) => fut,
                            Err(err) => Box::pin(future::ready(Err(err))),
                        }
                    })
                    .collect();

                let results = futures_util::future::join_all(futures).await;
                calls
                    .iter()
                    .zip(results)
                    .map(|(call, result)| BatchToolCallOutcome {
                        id: String::from(call.id),
                        name: String::from(call.name),
                        result,
                    })
                    .collect()
            }
            BatchExecutionMode::SequentialStopOnError => {
                let mut outcomes = Vec::with_capacity(calls.len());
                for call in calls {
                    let result = self.dispatch_str(call.name, call.arguments_json, ctx).await;
                    let should_stop = match &result {
                        Ok(_) => false,
                        Err(err) => {
                            tracing::debug!(
                                error = %err,
                                tool = %call.name,
                                "Sequential batch execution stopped due to error"
                            );
                            true
                        }
                    };
                    outcomes.push(BatchToolCallOutcome {
                        id: String::from(call.id),
                        name: String::from(call.name),
                        result,
                    });
                    if should_stop {
                        break;
                    }
                }
                outcomes
            }
        }
    }
}
