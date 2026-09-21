# Read workspace version from root Cargo.toml
version := `grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'`

# Default recipe: format, lint, and test
default: fmt lint test

# ── Format ────────────────────────────────────────────────────────────

# Format all code (Rust, TOML, Markdown, Justfile)
fmt: fmt-rust fmt-toml fmt-markdown fmt-just

# Format Rust code (nightly required for import grouping)
fmt-rust:
    cargo +nightly fmt

# Format TOML files
fmt-toml:
    taplo fmt

# Format Markdown files with prettier
fmt-markdown:
    npx -y prettier@latest --prose-wrap always --write '**/*.md'

# Format the justfile itself
fmt-just:
    just --fmt --unstable

# ── Lint ──────────────────────────────────────────────────────────────

# Lint all code (Rust clippy, TOML, Markdown, Justfile)
lint: lint-rust lint-toml lint-markdown lint-just lint-hygiene

# Lint Rust with clippy (pedantic + all, deny warnings)
lint-rust:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --features md-tmpl -- -D warnings

# Lint TOML files
lint-toml:
    taplo check

# Lint Markdown files
lint-markdown:
    npx -y markdownlint-cli2@latest '**/*.md'

# Lint the justfile (check formatting)
lint-just:
    just --fmt --unstable --check

# Run hygiene linter (flags suppression patterns, discarded errors, etc.)
lint-hygiene:
    python3 scripts/lint_hygiene.py

# ── Test ──────────────────────────────────────────────────────────────

# Run all tests (both with and without md-tmpl feature)
test:
    cargo test
    cargo test --features md-tmpl

# ── Docs ──────────────────────────────────────────────────────────────

# Build documentation (checks for broken intra-doc links)
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features md-tmpl

# ── Other ─────────────────────────────────────────────────────────────

# Verify no_std support (core crate only, on a bare-metal target)
check-no-std:
    cargo build -p llm-tool --no-default-features --target thumbv7em-none-eabihf

# Run all checks (lint + test + doc + no_std)
check: lint test doc check-no-std

# Run the same checks as GitHub Actions CI
ci: fmt-rust lint-rust test doc

# ── Publish ───────────────────────────────────────────────────────────

# Bump version across workspace Cargo.toml and all crates/*/Cargo.toml manifests
bump new_version:
    sed -i 's/^version = "{{ version }}"/version = "{{ new_version }}"/' Cargo.toml
    sed -i 's/llm-tool = { version = "{{ version }}"/llm-tool = { version = "{{ new_version }}"/' crates/*/Cargo.toml
    sed -i 's/llm-tool-macros = { version = "{{ version }}"/llm-tool-macros = { version = "{{ new_version }}"/' crates/*/Cargo.toml
    cargo update -w

# Publish all workspace crates in dependency order (macros → llm-tool → llm-tool-mcp)
publish: lint test publish-rust

publish-rust:
    cargo publish -p llm-tool-macros
    @echo "Waiting for crates.io index..."
    sleep 30
    cargo publish -p llm-tool
    @echo "Waiting for crates.io index..."
    sleep 30
    cargo publish -p llm-tool-mcp
