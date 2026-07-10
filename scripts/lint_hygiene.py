#!/usr/bin/env python3
"""Hygiene linter — flags suppression patterns and structural issues.

Run via `just lint-hygiene` or directly: `python3 scripts/lint_hygiene.py`

Exit code 0 = clean, 1 = violations found.

Escape hatch: add `// NOLINT: <reason>` on the line ABOVE a flagged
pattern to suppress it.  The reason is mandatory.
Bare `// NOLINT` without a reason is itself flagged as a violation.
Stale NOLINTs (where the next line doesn't trigger any check) are also flagged.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

MAX_FILE_LINES = 1200

RUST_DIRS = ["crates/"]
RUST_EXTS = {".rs"}

# Suppression markers.
NOLINT_WITH_REASON = re.compile(r"//\s*NOLINT:\s*\S")
NOLINT_ANY = re.compile(r"//\s*NOLINT")
NOLINT_BARE = re.compile(r"//\s*NOLINT\s*$")


# ── Check definitions ────────────────────────────────────────────────────────


@dataclass
class Check:
    """A single hygiene check definition."""

    name: str
    pattern: re.Pattern[str]
    dirs: list[str]
    exts: set[str]
    message: str
    exclude: re.Pattern[str] | None = None  # lines matching this are skipped
    exclude_path: re.Pattern[str] | None = None  # file paths matching this are skipped
    hits: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)


CHECKS: list[Check] = [
    # ── Rust: lint suppression ────────────────────────────────────────────
    Check(
        name="Rust: #[allow(...)]",
        pattern=re.compile(r"#\[allow\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Fix the underlying issue instead of suppressing the lint.",
    ),
    Check(
        name="Rust: #[expect(...)]",
        pattern=re.compile(r"#\[expect\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Fix the underlying issue instead of suppressing the lint.",
    ),
    Check(
        name="Rust: allow(clippy::too_many_lines) (never acceptable)",
        pattern=re.compile(r"clippy::too_many_lines"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Refactor the function into smaller pieces instead of suppressing the lint.",
    ),
    # ── Rust: silently discarded values ───────────────────────────────────
    Check(
        name="Rust: let _ = (ignored Result/value)",
        pattern=re.compile(r"\blet _\s*="),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Handle the Result/value properly — don't silently discard it.",
    ),
    Check(
        name="Rust: if let Ok(...) (silent Err drop)",
        pattern=re.compile(r"\bif let Ok\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="The Err branch is silently ignored. Use match, map_err, or ? to handle errors.",
    ),
    Check(
        name="Rust: .unwrap_or_default() (hidden errors)",
        pattern=re.compile(r"\.unwrap_or_default\(\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Silently replaces errors/None with defaults. Log or propagate instead.",
    ),
    Check(
        name="Rust: .ok() (Result→Option, error discarded)",
        pattern=re.compile(r"\.ok\(\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Converts Result to Option, silently discarding the error.",
    ),
    Check(
        name="Rust: .is_ok() / .is_err() (value discarded)",
        pattern=re.compile(r"\.(is_ok|is_err)\(\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Checks the Result but discards the inner value. Use match or ? instead.",
        # Legitimate in assertions, conditions, and boolean expressions.
        exclude=re.compile(r"assert|if |while |\|\||&&|return .*\.is_"),
    ),
    Check(
        name="Rust: discarded closure arg |_|",
        pattern=re.compile(r"\|_\|"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Closure discards its argument. Name it and use it (log, propagate, etc.).",
        # Skip doc-comments and string literals that mention closures.
        exclude=re.compile(r"^\s*///|^\s*//[^/].*\|_\||" + r'".*\|_\|'),
    ),
    Check(
        name="Rust: Err(_) (error value discarded in match)",
        pattern=re.compile(r"\bErr\(_\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Error value is discarded in pattern match. Capture and log/propagate it.",
    ),
    Check(
        name="Rust: .unwrap_or(()) / .unwrap_or(0) (silent swallow)",
        pattern=re.compile(r"\.unwrap_or\(\s*(\(\)|\b0\b|false|true)\s*\)"),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Silently swallows errors with a trivial default. Handle the error explicitly.",
    ),
    # ── Rust: panic-at-runtime markers ────────────────────────────────────
    Check(
        name="Rust: todo!() / unimplemented!()",
        pattern=re.compile(r"\b(todo|unimplemented)!\("),
        dirs=RUST_DIRS,
        exts=RUST_EXTS,
        message="Will panic at runtime. Implement or return an error.",
    ),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def source_files(dirs: list[str], exts: set[str]) -> list[Path]:
    """Collect all source files under `dirs` matching `exts`."""
    files: list[Path] = []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in exts:
                files.append(path)
    return sorted(files)


def is_nolinted(lines: list[str], lineno: int) -> bool:
    """Check if the line above `lineno` (1-indexed) has a valid NOLINT: reason."""
    if lineno < 2:
        return False
    prev_line = lines[lineno - 2]  # lineno 1-indexed, list 0-indexed
    return bool(NOLINT_WITH_REASON.search(prev_line))


def line_matches_any_check(line: str, checks: list[Check]) -> bool:
    """Return True if `line` would trigger any check pattern (ignoring excludes)."""
    for check in checks:
        if check.pattern.search(line):
            return True
    return False


# ── Check runners ────────────────────────────────────────────────────────────


def run_pattern_checks() -> bool:
    """Run all pattern checks. Returns True if any failed.

    Tracks which NOLINT comments are consumed (suppress a real hit) so we can
    detect stale ones afterwards.
    """
    failed = False
    # Collect all NOLINT locations and whether they were consumed.
    # Key: (path, lineno of NOLINT line), value: consumed?
    nolint_locations: dict[tuple[Path, int], bool] = {}

    # First pass: find all NOLINT lines.
    all_dirs_exts: set[tuple[str, str]] = set()
    for check in CHECKS:
        for d in check.dirs:
            for ext in check.exts:
                all_dirs_exts.add((d, ext))
    all_dirs_set = {d for d, _ in all_dirs_exts}
    all_exts_set = {e for _, e in all_dirs_exts}
    for path in source_files(list(all_dirs_set), all_exts_set):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            if NOLINT_WITH_REASON.search(line):
                nolint_locations[(path, lineno)] = False  # not yet consumed

    # Second pass: run checks, marking consumed NOLINTs.
    for check in CHECKS:
        print(f"=== {check.name} ===")
        files = source_files(check.dirs, check.exts)
        for path in files:
            if check.exclude_path and check.exclude_path.search(str(path)):
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, start=1):
                if not check.pattern.search(line):
                    continue
                if check.exclude and check.exclude.search(line):
                    continue
                if is_nolinted(lines, lineno):
                    # Mark the NOLINT as consumed.
                    nolint_key = (path, lineno - 1)
                    if nolint_key in nolint_locations:
                        nolint_locations[nolint_key] = True
                    check.suppressed.append(
                        f"  {path}:{lineno}: {line.strip()}"
                    )
                    continue
                check.hits.append(f"  {path}:{lineno}: {line.strip()}")
        if check.hits:
            for hit in check.hits:
                print(hit)
            print(f"^^^ {check.message}")
            failed = True
        else:
            print("  ✓ clean")
        if check.suppressed:
            for sup in check.suppressed:
                print(f"  ℹ  (suppressed) {sup.strip()}")

    # Check for stale NOLINTs (not consumed by any check).
    print("=== Stale NOLINT comments ===")
    stale: list[str] = []
    for (path, lineno), consumed in sorted(nolint_locations.items()):
        if not consumed:
            try:
                lines = path.read_text(errors="replace").splitlines()
                line_text = lines[lineno - 1].strip() if lineno <= len(lines) else "???"
            except OSError:
                line_text = "???"
            stale.append(f"  {path}:{lineno}: {line_text}")
    if stale:
        for s in stale:
            print(s)
        print("^^^ NOLINT comment doesn't suppress anything — remove it.")
        failed = True
    else:
        print("  ✓ clean")

    return failed


def run_bare_nolint_check() -> bool:
    """Flag NOLINT comments that don't include a reason."""
    print("=== Bare NOLINT (missing reason) ===")
    hits: list[str] = []
    for path in source_files(RUST_DIRS, RUST_EXTS):
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if NOLINT_BARE.search(line):
                hits.append(f"  {path}:{lineno}: {line.strip()}")
    if hits:
        for hit in hits:
            print(hit)
        print("^^^ NOLINT must include a reason: // NOLINT: <why this is acceptable>")
        return True
    print("  ✓ clean")
    return False


def run_long_file_check() -> bool:
    """Flag files exceeding MAX_FILE_LINES."""
    print(f"=== Long files (>{MAX_FILE_LINES} lines) ===")
    long_files: list[tuple[Path, int]] = []

    for path in source_files(RUST_DIRS, RUST_EXTS):
        try:
            line_count = sum(1 for _ in path.open(errors="replace"))
        except OSError:
            continue
        if line_count > MAX_FILE_LINES:
            long_files.append((path, line_count))

    if long_files:
        long_files.sort(key=lambda x: -x[1])
        for path, count in long_files:
            print(f"  ⚠  {path} ({count} lines)")
        print(
            "^^^ Long file detected. Should be refactored to be modular "
            "and testable (with good testing) and a good folder structure."
        )
        return True

    print("  ✓ clean")
    return False


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    failed = run_pattern_checks()
    failed = run_bare_nolint_check() or failed
    failed = run_long_file_check() or failed

    print()
    if failed:
        print("❌ Hygiene check failed — see above")
        return 1
    print("✅ All hygiene checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
