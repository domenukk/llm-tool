#!/usr/bin/env python3
"""Tests for `scripts/bump_version.py`."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUMP_SCRIPT = REPO_ROOT / "scripts" / "bump_version.py"
TARGET_VERSION = "9.9.9"
COPIED_FILES = (
    "Cargo.toml",
    "README.md",
    "crates/llm-tool/README.md",
    "crates/llm-tool/Cargo.toml",
    "crates/llm-tool-macros/Cargo.toml",
    "crates/llm-tool-mcp/Cargo.toml",
)


def _load_bump_module():
    spec = importlib.util.spec_from_file_location("llm_tool_bump_version", BUMP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bump module from {BUMP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_tool_bump_version"] = module
    spec.loader.exec_module(module)
    return module


BUMP = _load_bump_module()


def _current_version() -> str:
    text = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = BUMP.WORKSPACE_VERSION_RE.search(text)
    if match is None:
        raise RuntimeError("workspace Cargo.toml has no version line")
    return match.group(2)


class TestBumpVersion(unittest.TestCase):
    def test_bumps_manifests_deps_and_readmes_preserving_lf(self) -> None:
        current = _current_version()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in COPIED_FILES:
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / rel, dst)

            old = BUMP.bump_workspace_version(root, TARGET_VERSION)
            self.assertEqual(old, current)

            root_bytes = (root / "Cargo.toml").read_bytes()
            self.assertNotIn(b"\r\n", root_bytes)
            self.assertIn(f'version = "{TARGET_VERSION}"'.encode(), root_bytes)

            tool_text = (root / "crates/llm-tool/Cargo.toml").read_text(encoding="utf-8")
            self.assertIn(
                f'llm-tool-macros = {{ path = "../llm-tool-macros", version = "{TARGET_VERSION}" }}',
                tool_text,
            )

            mcp_text = (root / "crates/llm-tool-mcp/Cargo.toml").read_text(encoding="utf-8")
            self.assertIn(
                f'llm-tool = {{ path = "../llm-tool", version = "{TARGET_VERSION}" }}',
                mcp_text,
            )

            for readme in BUMP.README_FILES:
                text = (root / readme).read_text(encoding="utf-8")
                self.assertIn(f'llm-tool = "{TARGET_VERSION}"', text, readme)
                self.assertIn(f'llm-tool-mcp = "{TARGET_VERSION}"', text, readme)
                self.assertNotIn(f'"{current}"', text, readme)


if __name__ == "__main__":
    unittest.main()
