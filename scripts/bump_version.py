#!/usr/bin/env python3
"""Bump the workspace version in Cargo.toml and internal crate dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKSPACE_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)

# READMEs whose `[dependencies]` snippets pin the current release version.
README_FILES: tuple[str, ...] = ("README.md", "crates/llm-tool/README.md")


def bump_workspace_version(repo_root: Path, new_version: str) -> str:
    """Update workspace version and internal dependency versions; return previous version."""
    root_toml = repo_root / "Cargo.toml"
    root_text = root_toml.read_text(encoding="utf-8")
    match = WORKSPACE_VERSION_RE.search(root_text)
    if match is None:
        raise ValueError(f"no `version = \"...\"` line found in {root_toml}")
    old_version = match.group(2)

    updated_root = WORKSPACE_VERSION_RE.sub(
        lambda m: f"{m.group(1)}{new_version}{m.group(3)}",
        root_text,
        count=1,
    )
    root_toml.write_text(updated_root.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    dep_re = re.compile(
        rf'(llm-tool(?:-macros)?\s*=\s*\{{[^}}]*?\bversion\s*=\s*")'
        rf"{re.escape(old_version)}"
        rf'(")'
    )
    for manifest in sorted((repo_root / "crates").glob("*/Cargo.toml")):
        text = manifest.read_text(encoding="utf-8")
        updated = dep_re.sub(lambda m: f"{m.group(1)}{new_version}{m.group(2)}", text)
        manifest.write_text(updated.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    readme_re = re.compile(
        rf'^(llm-tool(?:-mcp|-macros)?\s*=\s*"){re.escape(old_version)}(")',
        re.MULTILINE,
    )
    for readme in README_FILES:
        path = repo_root / readme
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        updated = readme_re.sub(lambda m: f"{m.group(1)}{new_version}{m.group(2)}", text)
        path.write_text(updated.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    return old_version


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].strip():
        print("usage: bump_version.py <new_version>", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parent.parent
    new_version = argv[1].strip()
    old_version = bump_workspace_version(repo_root, new_version)
    print(f"Bumped workspace version: {old_version} -> {new_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
