#!/usr/bin/env python3
"""Adversarial Gate R tests for `scripts/lint_hygiene.py` in `llm-tool`.

Verifies that:
1. Synthetic `serde_json::json!({...})` in production `crates/*/src/` fails `lint_hygiene.py` (Red).
2. `serde_json::json!(...)` inside `#[cfg(test)]` or `tests/` is allowed (Green).
3. Lint suppressions (`#[allow(...)]`, `#[expect(...)]`, `#[path = ...]`) and `>= 1,200` line files fail (Red).
4. The real `llm-tool` workspace passes `check_workspace(REPO_ROOT)` cleanly (`rc=0`, Green).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint_hygiene.py"

OUTER_ALLOW = "#" + "[allow"
OUTER_EXPECT = "#" + "[expect"
PATH_ATTR = "#" + '[path = "other.rs"]'
JSON_MACRO = "serde_json::" + "json!"


def _load_lint_module():
    spec = importlib.util.spec_from_file_location("llm_tool_lint_hygiene", LINT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load lint module from {LINT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_tool_lint_hygiene"] = module
    spec.loader.exec_module(module)
    return module


LINT = _load_lint_module()


class TestLintHygieneGateR(unittest.TestCase):
    def _run_check(self, root: Path) -> tuple[bool, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            failed = LINT.check_workspace(root)
        return failed, buf.getvalue()

    def test_gate_r_rejects_production_json_macro(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "crates" / "demo" / "src" / "lib.rs"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(
                f'pub fn payload() -> serde_json::Value {{\n    {JSON_MACRO}({{"ok": true}})\n}}\n',
                encoding="utf-8",
            )
            failed, output = self._run_check(root)
            self.assertTrue(failed, f"Expected Red failure for production json!, got:\n{output}")
            self.assertIn("crates/demo/src/lib.rs:2", output)

    def test_gate_r_allows_json_macro_in_cfg_test_and_integration_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "crates" / "demo" / "src" / "lib.rs"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(
                "pub const VERSION: u32 = 1;\n\n"
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    #[test]\n"
                "    fn allows_test_json() {\n"
                f'        let v = {JSON_MACRO}({{"v": super::VERSION}});\n'
                '        assert_eq!(v["v"], 1);\n'
                "    }\n"
                "}\n",
                encoding="utf-8",
            )
            integ = root / "crates" / "demo" / "tests" / "api.rs"
            integ.parent.mkdir(parents=True, exist_ok=True)
            integ.write_text(
                f'#[test]\nfn ok() {{ assert_eq!({JSON_MACRO}(1), 1); }}\n',
                encoding="utf-8",
            )
            failed, output = self._run_check(root)
            self.assertFalse(failed, f"Expected Green for #[cfg(test)] json!, got:\n{output}")

    def test_gate_r_rejects_suppressions_path_override_and_long_files(self) -> None:
        cases = [
            ("crates/demo/src/a.rs", f"{OUTER_ALLOW}(dead_code)]\npub fn f() {{}}\n"),
            ("crates/demo/src/b.rs", f"{OUTER_EXPECT}(unused)]\npub fn f() {{}}\n"),
            ("crates/demo/src/c.rs", f"{PATH_ATTR}\nmod other;\n"),
            ("crates/demo/src/big.rs", "pub const X: usize = 1;\n" * 1200),
        ]
        for rel_path, body in cases:
            with self.subTest(rel_path=rel_path), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                failed, output = self._run_check(root)
                self.assertTrue(failed, f"Expected Red failure on {rel_path}, got:\n{output}")
                self.assertIn(rel_path, output)

    def test_real_workspace_is_green(self) -> None:
        failed, output = self._run_check(REPO_ROOT)
        self.assertFalse(failed, f"Real workspace failed lint_hygiene:\n{output}")


if __name__ == "__main__":
    unittest.main()
