#!/usr/bin/env python3
"""Hygiene linter — flags suppression patterns, untyped production json!(), and structural issues.

Run via `just lint-hygiene` or directly: `python3 scripts/lint_hygiene.py`

Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

MAX_FILE_LINES = 1200

RUST_DIRS = ("crates/",)
RUST_EXTS = frozenset({".rs"})
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "target",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

# Suppression markers.
NOLINT_WITH_REASON = re.compile(r"//\s*NOLINT:\s*\S")
NOLINT_BARE = re.compile(r"//\s*NOLINT\s*$")

# Test-code classification (Rust).
TEST_PATH = re.compile(
    r"(^|/)(tests|benches|examples)/|(^|/)tests\.rs$|(^|/)tests_[^/]*\.rs$|_tests?\.rs$"
)
CFG_TEST_ATTR = re.compile(
    r"^\s*#\s*\[\s*cfg\s*\(\s*(?:all\s*\([^)]*\btest\b[^)]*\)|test)\s*\)\s*\]\s*$"
)
CFG_TEST_MOD_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")
CFG_TEST_MOD_INLINE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")
MODULE_ROOT_FILES = frozenset({"mod.rs", "lib.rs", "main.rs"})


class Scope(Enum):
    """Which code a check applies to."""

    ALL = "all"
    PRODUCTION_ONLY = "production"
    TEST_ONLY = "test"


# ── Check definitions ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Check:
    """A single hygiene check definition."""

    name: str
    pattern: re.Pattern[str]
    dirs: tuple[str, ...]
    exts: frozenset[str]
    message: str
    exclude: re.Pattern[str] | None = None
    exclude_path: re.Pattern[str] | None = None
    no_nolint: bool = False
    scope: Scope = Scope.ALL


CHECKS: tuple[Check, ...] = (
    # ── Rust: lint suppression ────────────────────────────────────────────
    Check(
        name="Rust: #[allow(...)] / #![allow(...)]",
        pattern=re.compile(r"#!?\s*\[\s*allow\s*\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Fix the underlying issue instead of suppressing the lint.",
        no_nolint=True,
    ),
    Check(
        name="Rust: #[expect(...)] / #![expect(...)]",
        pattern=re.compile(r"#!?\s*\[\s*expect\s*\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Fix the underlying issue instead of suppressing the lint.",
        no_nolint=True,
    ),
    Check(
        name="Rust: allow(clippy::too_many_lines) (never acceptable)",
        pattern=re.compile(r"clippy::too_many_lines"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Refactor the function into smaller pieces instead of suppressing the lint.",
        no_nolint=True,
    ),
    Check(
        name="Rust: type_complexity suppression",
        pattern=re.compile(r"clippy::type_complexity"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Extract a type alias or struct instead of suppressing clippy::type_complexity.",
        no_nolint=True,
    ),
    Check(
        name="Rust: cognitive_complexity suppression",
        pattern=re.compile(r"clippy::cognitive_complexity"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Refactor the function into smaller pieces instead of suppressing clippy::cognitive_complexity.",
        no_nolint=True,
    ),
    Check(
        name="Rust: #[path = ...] module override",
        pattern=re.compile(r"#!?\s*\[\s*path\s*="),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Always use standard Rust <module>/mod.rs directory hierarchy instead of #[path = ...] overrides.",
        no_nolint=True,
    ),
    Check(
        name="Rust: serde_json::json! in production code",
        pattern=re.compile(r"\bjson!\s*[\(\[\{]"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Untyped JSON grab-bag in production code. Use #[derive(Serialize, Deserialize)] structs (json! is allowed only in tests).",
        no_nolint=True,
        scope=Scope.PRODUCTION_ONLY,
    ),
    # ── Rust: silently discarded values ───────────────────────────────────
    Check(
        name="Rust: let _ = (ignored Result/value)",
        pattern=re.compile(r"\blet _\s*="),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Handle the Result/value properly — don't silently discard it.",
        no_nolint=True,
    ),
    Check(
        name="Rust: if let Ok(...) (silent Err drop)",
        pattern=re.compile(r"\bif let Ok\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="The Err branch is silently ignored. Use match, map_err, or ? to handle errors.",
        no_nolint=True,
    ),
    Check(
        name="Rust: .unwrap_or_default() (hidden errors)",
        pattern=re.compile(r"\.unwrap_or_default\(\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Silently replaces errors/None with defaults. Log or propagate instead.",
        no_nolint=True,
    ),
    Check(
        name="Rust: .ok() (Result→Option, error discarded)",
        pattern=re.compile(r"\.ok\(\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Converts Result to Option, silently discarding the error.",
        no_nolint=True,
    ),
    Check(
        name="Rust: .is_ok() / .is_err() (value discarded)",
        pattern=re.compile(r"\.(is_ok|is_err)\(\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Checks the Result but discards the inner value. Use match or ? instead.",
        exclude=re.compile(r"assert|if |while |\|\||&&|return .*\.is_"),
        no_nolint=True,
    ),
    Check(
        name="Rust: discarded closure arg |_|",
        pattern=re.compile(r"\|_\|"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Closure discards its argument. Name it and use it (log, propagate, etc.).",
        exclude=re.compile(r"^\s*///|^\s*//[^/].*\|_\||" + r'".*\|_\|'),
        no_nolint=True,
    ),
    Check(
        name="Rust: Err(_) / Err(_err) (error value discarded in match)",
        pattern=re.compile(r"\bErr\(_[a-zA-Z0-9_]*\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Error value is discarded in pattern match. Capture and log/propagate it.",
        no_nolint=True,
    ),
    Check(
        name="Rust: .unwrap_or(()) / .unwrap_or(0) (silent swallow)",
        pattern=re.compile(r"\.unwrap_or\(\s*(\(\)|\b0\b|false|true)\s*\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Silently swallows errors with a trivial default. Handle the error explicitly.",
        no_nolint=True,
    ),
    # ── Rust: panic-at-runtime / skipped test markers ─────────────────────
    Check(
        name="Rust: todo!() / unimplemented!()",
        pattern=re.compile(r"\b(todo|unimplemented)!\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Will panic at runtime. Implement or return an error.",
        no_nolint=True,
    ),
    Check(
        name="Rust: #[ignore] (skipped test)",
        pattern=re.compile(r"#\[ignore\b"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Never ignore tests — fix or make deterministic.",
        no_nolint=True,
    ),
    Check(
        name="Template source: single-line \\n-escaped frontmatter",
        pattern=re.compile(r"""['"`]---\\n"""),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="NEVER use single-line \\n-escaped frontmatter strings; always use proper multiline strings.",
        no_nolint=True,
    ),
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def source_files(
    root_dir: Path, dirs: tuple[str, ...], exts: frozenset[str]
) -> list[Path]:
    """Collect all source files under `root_dir / d` matching `exts`."""
    files: list[Path] = []
    for d in dirs:
        sub = root_dir / d
        if not sub.is_dir():
            continue
        for path in sub.rglob("*"):
            rel_parts = path.relative_to(root_dir).parts
            if any(part in EXCLUDED_DIRS for part in rel_parts):
                continue
            if path.is_file() and path.suffix in exts:
                files.append(path)
    return sorted(files)


def module_dir(path: Path) -> Path:
    """Directory holding the child modules declared in `path`."""
    if path.name in MODULE_ROOT_FILES:
        return path.parent
    return path.parent / path.stem


def cfg_test_module_paths(rust_files: list[Path]) -> list[Path]:
    """Files/dirs of out-of-line modules declared under `#[cfg(test)]` or `#[cfg(all(test, ...))]`."""
    gated: list[Path] = []
    for path in rust_files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for attr_line, decl_line in pairwise(lines):
            if not CFG_TEST_ATTR.match(attr_line):
                continue
            decl = CFG_TEST_MOD_DECL.match(decl_line)
            if decl is None:
                continue
            child = module_dir(path) / decl.group(1)
            gated.extend([child.with_suffix(".rs"), child])
    return gated


def is_test_file(rel_posix: str, path: Path, gated_modules: list[Path]) -> bool:
    """Whether every line of `path` is test/example/bench code."""
    if TEST_PATH.search(rel_posix):
        return True
    return any(path == gated or gated in path.parents for gated in gated_modules)


def inline_test_line_numbers(lines: list[str]) -> set[int]:
    """Return 1-indexed line numbers covered by `#[cfg(test)]` / `#[cfg(all(test, ...))]` items."""
    test_lines: set[int] = set()
    i = 0
    n = len(lines)
    while i < n:
        if CFG_TEST_ATTR.match(lines[i]):
            start = i
            i += 1
            while i < n and lines[i].strip().startswith("#"):
                i += 1
            if i < n and CFG_TEST_MOD_INLINE.match(lines[i]):
                # Inline #[cfg(test)] mod tests { ... } through balanced closing brace.
                brace_depth = 0
                started = False
                while i < n:
                    clean = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[i])
                    clean = re.sub(r"//.*", "", clean)
                    for ch in clean:
                        if ch == "{":
                            brace_depth += 1
                            started = True
                        elif ch == "}":
                            brace_depth -= 1
                    i += 1
                    if started and brace_depth <= 0:
                        break
                for ln in range(start + 1, i + 1):
                    test_lines.add(ln)
                continue
            # Single #[cfg(test)] item (fn, struct, impl, use, mod decl).
            brace_depth = 0
            started = False
            while i < n:
                clean = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[i])
                clean = re.sub(r"//.*", "", clean)
                for ch in clean:
                    if ch == "{":
                        brace_depth += 1
                        started = True
                    elif ch == "}":
                        brace_depth -= 1
                i += 1
                if started and brace_depth <= 0:
                    break
                if not started and ";" in clean:
                    break
            for ln in range(start + 1, i + 1):
                test_lines.add(ln)
            continue
        i += 1
    return test_lines


def line_in_scope(scope: Scope, is_test_line: bool) -> bool:
    """Whether a check with `scope` applies to a (test / production) line."""
    match scope:
        case Scope.ALL:
            return True
        case Scope.PRODUCTION_ONLY:
            return not is_test_line
        case Scope.TEST_ONLY:
            return is_test_line


def is_nolinted(lines: list[str], lineno: int) -> bool:
    """Check if the line above `lineno` (1-indexed) has a valid NOLINT: reason."""
    if lineno < 2:
        return False
    prev_line = lines[lineno - 2]
    return bool(NOLINT_WITH_REASON.search(prev_line))


# ── Check runners ────────────────────────────────────────────────────────────


def run_pattern_checks(root_dir: Path | None = None) -> bool:
    """Run all pattern checks under `root_dir`. Returns True if any failed."""
    base = root_dir if root_dir is not None else Path()
    failed = False
    nolint_locations: dict[tuple[Path, int], bool] = {}

    rust_files = source_files(base, RUST_DIRS, RUST_EXTS)
    gated_modules = cfg_test_module_paths(rust_files)

    for path in rust_files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            if NOLINT_WITH_REASON.search(line):
                nolint_locations[(path, lineno)] = False

    for check in CHECKS:
        print(f"=== {check.name} ===")
        hits: list[str] = []
        suppressed: list[str] = []
        files = source_files(base, check.dirs, check.exts)
        for path in files:
            rel_posix = path.relative_to(base).as_posix()
            if check.exclude_path and check.exclude_path.search(rel_posix):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            whole_file_is_test = is_test_file(rel_posix, path, gated_modules)
            cfg_test_lines = set() if whole_file_is_test else inline_test_line_numbers(lines)
            for lineno, line in enumerate(lines, start=1):
                if not check.pattern.search(line):
                    continue
                is_test_line = whole_file_is_test or (lineno in cfg_test_lines)
                if not line_in_scope(check.scope, is_test_line):
                    continue
                if check.exclude and check.exclude.search(line):
                    continue
                if not check.no_nolint and is_nolinted(lines, lineno):
                    nolint_key = (path, lineno - 1)
                    if nolint_key in nolint_locations:
                        nolint_locations[nolint_key] = True
                    suppressed.append(f"  {rel_posix}:{lineno}: {line.strip()}")
                    continue
                hits.append(f"  {rel_posix}:{lineno}: {line.strip()}")
        if hits:
            for hit in hits:
                print(hit)
            print(f"^^^ {check.message}")
            failed = True
        else:
            print("  ✓ clean")
        for sup in suppressed:
            print(f"  ℹ  (suppressed) {sup.strip()}")

    print("=== Stale NOLINT comments ===")
    stale: list[str] = []
    for (path, lineno), consumed in sorted(nolint_locations.items()):
        if not consumed:
            rel_posix = path.relative_to(base).as_posix()
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                line_text = lines[lineno - 1].strip() if lineno <= len(lines) else "???"
            except OSError:
                line_text = "???"
            stale.append(f"  {rel_posix}:{lineno}: {line_text}")
    if stale:
        for s in stale:
            print(s)
        print("^^^ NOLINT comment doesn't suppress anything — remove it.")
        failed = True
    else:
        print("  ✓ clean")

    return failed


def run_bare_nolint_check(root_dir: Path | None = None) -> bool:
    """Flag NOLINT comments that don't include a reason."""
    base = root_dir if root_dir is not None else Path()
    print("=== Bare NOLINT (missing reason) ===")
    hits: list[str] = []
    for path in source_files(base, RUST_DIRS, RUST_EXTS):
        rel_posix = path.relative_to(base).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if NOLINT_BARE.search(line):
                hits.append(f"  {rel_posix}:{lineno}: {line.strip()}")
    if hits:
        for hit in hits:
            print(hit)
        print("^^^ NOLINT must include a reason: // NOLINT: <why this is acceptable>")
        return True
    print("  ✓ clean")
    return False


def run_long_file_check(
    root_dir: Path | None = None, max_lines: int = MAX_FILE_LINES
) -> bool:
    """Flag files reaching or exceeding `max_lines`."""
    base = root_dir if root_dir is not None else Path()
    print(f"=== Long files (>={max_lines} lines) ===")
    long_files: list[tuple[str, int]] = []

    for path in source_files(base, RUST_DIRS, RUST_EXTS):
        rel_posix = path.relative_to(base).as_posix()
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                line_count = sum(1 for _ in handle)
        except OSError:
            continue
        if line_count >= max_lines:
            long_files.append((rel_posix, line_count))

    if long_files:
        long_files.sort(key=lambda x: -x[1])
        for rel_posix, count in long_files:
            print(f"  ⚠  {rel_posix} ({count} lines)")
        print(
            "^^^ Long file detected. Should be refactored to be modular "
            "and testable (with good testing) and a good folder structure."
        )
        return True

    print("  ✓ clean")
    return False


def check_workspace(root_dir: Path) -> bool:
    """Run all hygiene checks on `root_dir`. Returns True if any violation is found."""
    failed = run_pattern_checks(root_dir)
    failed = run_bare_nolint_check(root_dir) or failed
    failed = run_long_file_check(root_dir) or failed
    return failed


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    failed = check_workspace(repo_root)

    print()
    if failed:
        print("❌ Hygiene check failed — see above")
        return 1
    print("✅ All hygiene checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
