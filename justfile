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
    npx -y prettier@latest --write '**/*.md'

# Format the justfile itself
fmt-just:
    just --fmt --unstable

# ── Lint ──────────────────────────────────────────────────────────────

# Lint all code (Rust clippy, TOML, Markdown, Justfile)
lint: lint-rust lint-toml lint-markdown lint-just

# Lint Rust with clippy (pedantic + all, deny warnings)
lint-rust:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --features prompt-templates -- -D warnings

# Lint TOML files
lint-toml:
    taplo check

# Lint Markdown files
lint-markdown:
    npx -y markdownlint-cli2@latest '**/*.md'

# Lint the justfile (check formatting)
lint-just:
    just --fmt --unstable --check

# ── Test ──────────────────────────────────────────────────────────────

# Run all tests (both with and without prompt-templates feature)
test:
    cargo test
    cargo test --features prompt-templates

# ── Docs ──────────────────────────────────────────────────────────────

# Build documentation (checks for broken intra-doc links)
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features prompt-templates

# ── Other ─────────────────────────────────────────────────────────────

# Run all checks (lint + test + doc)
check: lint test doc

# Run the same checks as GitHub Actions CI
ci: fmt-rust lint-rust test doc
