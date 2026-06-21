//! Quick performance benchmark for template descriptions.
//!
//! Run with: `cargo test --features prompt-templates --release -- perf --nocapture`

#![cfg(feature = "prompt-templates")]
#![allow(clippy::cast_precision_loss)] // Benchmark display values; precision loss is fine.

use std::time::Instant;

use llm_tool::{RustTool, ToolRegistry, llm_tool};

// ── Static template description ──

#[llm_tool(prompt_file = "tools/static_desc.tmpl.md")]
fn static_tool(
    /// A value.
    x: i64,
) -> String {
    format!("{x}")
}

// ── Dynamic template description ──

fn bench_context(_tool: &DynamicTool) -> prompt_templates::Context {
    let mut ctx = prompt_templates::Context::new();
    ctx.set("api_version", "v3.1");
    ctx.set("env_name", "production");
    ctx
}

#[llm_tool(prompt_file = "tools/dynamic_desc.tmpl.md", context = bench_context)]
fn dynamic_tool(
    /// A value.
    x: i64,
) -> String {
    format!("{x}")
}

// ── Doc comment description (baseline) ──

/// Fetch the current weather for any city worldwide. Returns a JSON object
/// containing temperature, humidity, wind speed, and a human-readable
/// conditions summary.
#[llm_tool]
fn doc_comment_tool(
    /// A value.
    x: i64,
) -> String {
    format!("{x}")
}

const ITERATIONS: u32 = 100_000;

#[test]
fn perf_description_methods() {
    let static_tool = StaticTool;
    let dynamic_tool = DynamicTool;
    let doc_tool = DocCommentTool;

    // Warm up LazyLock
    std::hint::black_box(dynamic_tool.description());

    // Benchmark doc comment description (baseline)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(doc_tool.description());
    }
    let doc_elapsed = start.elapsed();

    // Benchmark static template description
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(static_tool.description());
    }
    let static_elapsed = start.elapsed();

    // Benchmark dynamic template description
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(dynamic_tool.description());
    }
    let dynamic_elapsed = start.elapsed();

    println!("\n=== description() performance ({ITERATIONS} iterations) ===");
    println!(
        "  Doc comment (baseline): {:>8.2?}  ({:.0} ns/call)",
        doc_elapsed,
        doc_elapsed.as_nanos() as f64 / f64::from(ITERATIONS)
    );
    println!(
        "  Static template:        {:>8.2?}  ({:.0} ns/call)",
        static_elapsed,
        static_elapsed.as_nanos() as f64 / f64::from(ITERATIONS)
    );
    println!(
        "  Dynamic template:       {:>8.2?}  ({:.0} ns/call)",
        dynamic_elapsed,
        dynamic_elapsed.as_nanos() as f64 / f64::from(ITERATIONS)
    );

    // Also benchmark registry definition generation
    let registry = ToolRegistry::new()
        .with_tool(StaticTool)
        .with_tool(DynamicTool)
        .with_tool(DocCommentTool);

    let start = Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(registry.definitions());
    }
    let defn_elapsed = start.elapsed();
    println!(
        "  definitions() x1000:    {:>8.2?}  ({:.0} µs/call)",
        defn_elapsed,
        defn_elapsed.as_micros() as f64 / 1000.0
    );
    println!();
}
