# Agent Coding Rules for `llm-tool`

Rules for AI coding agents working in this repository. They are enforced by CI,
`just lint`, `scripts/lint_hygiene.py`, and code review.

---

## 1. Crate Overview

`llm-tool` is a `no_std`-compatible Rust workspace for defining strongly-typed
LLM tools, prompts, resources, and MCP servers:

| Crate / Path              | Role                                                                                                                                                                                                                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/llm-tool/`        | Core traits (`RustTool`, `RustPrompt`, `RustResource`), `ToolRegistry`, batch dispatch, middle-out truncation (`TruncationPolicy`, `OutputSpillSink`), and tool output quarantine (`quarantine_tool_output`, delegating to `md-tmpl-core`). `no_std` + `alloc` compatible. |
| `crates/llm-tool-macros/` | Proc-macro crate providing `#[llm_tool]`, `#[llm_prompt]`, and `#[llm_resource]` with compile-time and runtime `md-tmpl` description and response template support.                                                                                                        |
| `crates/llm-tool-mcp/`    | Model Context Protocol (MCP) JSON-RPC 2.0 server (`McpServer`) over stdio, TCP, Unix sockets, and custom async streams.                                                                                                                                                    |

Before declaring any change complete, verify:

```sh
just fmt
just check
```

---

## 2. Core Principles

- **DRY (Don't Repeat Yourself)** — extract shared logic into functions, traits,
  or shared modules. Delegate control-token and XML quarantine sanitization to
  `md-tmpl-core` (Single Source of Truth) rather than duplicating delimiter
  tables.
- **YAGNI (You Aren't Gonna Need It)** — don't build speculative abstractions,
  unused feature flags, or dead code paths.
- **Strong types everywhere** — use enums, newtypes, and structs instead of
  stringly-typed APIs or raw primitives where domain meaning exists.
- **Constants over magic values** — use named `const` values instead of inline
  magic numbers or repeated string literals.
- **No inline strings** — never scatter inline magic strings, attribute keys,
  JSON-RPC method names, or delimiter literals across functions; define named
  constants in a central place and reference them.
- **Maintainable & readable** — meaningful names, small focused functions, clear
  module boundaries.
- **Beautiful code** — clean formatting, idiomatic patterns, zero warnings.

---

## 3. Git Rules

- **NEVER commit or push** unless explicitly asked by the user.
- **NEVER `git push`** under any circumstances unless explicitly asked.
- **NEVER change commit descriptions** when amending.
- Place one-off fix or migration scripts in `/tmp/jetski`, **not** in the repo.

---

## 4. Lint & Hygiene Rules

Run `python3 scripts/lint_hygiene.py` (or `just lint`) before finishing any
task. All checks must pass with **zero** violations and **zero** suppressions.

### 4.1 Never Suppress Lints

- **NEVER** add `#[allow(...)]` or `#[expect(...)]` in Rust — fix the underlying
  issue instead.
- **NEVER** add `// NOLINT` comments — fix the underlying issue properly
  instead.
- **NEVER** allow function-wide or file-wide lint suppressions.
- Particularly forbidden: `clippy::too_many_lines`,
  `clippy::cognitive_complexity`, `clippy::type_complexity` — refactor the
  function or type instead.

### 4.2 Never Ignore Errors or Return Values

- **NEVER** write `let _ = ...` in Rust — handle every `Result` and return value
  explicitly.
- **NEVER** write `if let Ok(...)` without handling the `Err` branch — use
  `match`, `map_err`, or `?`.
- **NEVER** use `.ok()` on a `Result` (silently discards the error).
- **NEVER** use `.unwrap_or_default()` where it hides an error or missing
  configuration.
- **NEVER** use `.unwrap_or(())`, `.unwrap_or(0)`, `.unwrap_or(false)`,
  `.unwrap_or(true)` to silently swallow errors.
- **NEVER** discard closure error arguments with `|_|` or match arms with
  `Err(_)` / `Err(_err)` or `.flatten()` on `Result` iterators — capture the
  error and log (`tracing::warn!` / `tracing::debug!`) or propagate it.
- **NEVER** leave `todo!()` or `unimplemented!()` in code.

### 4.3 Never Skip Tests

- **NEVER** mark tests `#[ignore]`.
- Always fix broken tests — never comment them out or delete assertions to make
  a suite pass.

### 4.4 File Length Limit

- Every source file must be **strictly below 1,200 lines**
  (`MAX_FILE_LINES = 1200` in `scripts/lint_hygiene.py`).
- There are **zero grandfathered exceptions**. Split large files into focused
  submodules (`foo/mod.rs`, `foo/bar.rs`, `foo/tests.rs`).

### 4.5 Template Strings

- **NEVER** write single-line `\n`-escaped frontmatter strings (e.g.
  `"---\nparams:\n..."`) in `.rs` or `.md` files. Always use proper multiline
  strings.

---

## 5. Documentation & README Style

- Write all `README.md` files in simple, non-technical, non-AI-sounding,
  straight-to-the-point language with zero marketing buzzwords, emojis, or
  grandiose slogans.
