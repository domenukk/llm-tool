set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

python := if os_family() == "windows" { "python" } else { "python3" }

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
    cargo clippy --workspace --all-targets -- -D warnings

# Lint TOML files
lint-toml:
    taplo check

# Lint Markdown files
lint-markdown:
    npx -y markdownlint-cli2@latest '**/*.md'

# Lint the justfile (check formatting)
lint-just:
    just --fmt --unstable --check

# Run hygiene linter (flags suppression patterns, discarded errors, etc.) and its self-tests
lint-hygiene:
    {{ python }} scripts/lint_hygiene.py
    {{ python }} -m unittest discover -s tests -p "test_*.py"

# ── Test ──────────────────────────────────────────────────────────────

# Run all tests
test:
    cargo test --workspace

# ── Docs ──────────────────────────────────────────────────────────────

# Build documentation (checks for broken intra-doc links)
[unix]
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Build documentation (checks for broken intra-doc links)
[windows]
doc:
    $env:RUSTDOCFLAGS = '-D warnings'; cargo doc --workspace --no-deps

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
    {{ python }} scripts/bump_version.py {{ new_version }}
    cargo update -w

# Publish all workspace crates in dependency order (macros → llm-tool → llm-tool-mcp)
publish: lint test publish-rust

publish-rust:
    cargo publish -p llm-tool-macros
    @echo "Waiting for crates.io index..."
    {{ python }} -c "import time; time.sleep(30)"
    cargo publish -p llm-tool
    @echo "Waiting for crates.io index..."
    {{ python }} -c "import time; time.sleep(30)"
    cargo publish -p llm-tool-mcp
