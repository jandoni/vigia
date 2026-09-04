#!/usr/bin/env python3
"""Guard: keep AGPL-licensed packages out of the VIGÍA runtime path.

VIGÍA ships under Apache-2.0. Ultralytics ships under AGPL-3.0, and its licence
would virally apply to this whole project if we imported it at runtime. Model
weights are fine — those come from distributors who state their own licence —
but the *package* must never be imported by anything under `vigia/`.

`.pt -> .onnx` conversion legitimately needs ultralytics, so `tools/` is
exempt: it is an offline build step, not part of the distributed runtime.

Run in CI. Exits non-zero on violation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PACKAGE = REPO_ROOT / "vigia"

# Packages whose licences are incompatible with distributing VIGÍA as Apache-2.0.
FORBIDDEN = {
    "ultralytics": "AGPL-3.0",
}


def imported_modules(path: Path) -> set[str]:
    """Top-level module names actually imported by a file.

    Parses the AST rather than grepping, so comments and docstrings that
    merely mention a package do not trip the check.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"  ! could not parse {path}: {exc}", file=sys.stderr)
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
    return modules


def main() -> int:
    if not RUNTIME_PACKAGE.is_dir():
        print(f"Runtime package not found at {RUNTIME_PACKAGE}", file=sys.stderr)
        return 2

    violations: list[tuple[Path, str, str]] = []
    checked = 0

    for path in sorted(RUNTIME_PACKAGE.rglob("*.py")):
        checked += 1
        for module in imported_modules(path):
            if module in FORBIDDEN:
                violations.append(
                    (path.relative_to(REPO_ROOT), module, FORBIDDEN[module])
                )

    if violations:
        print("LICENCE GUARD FAILED\n")
        for path, module, licence in violations:
            print(f"  {path} imports {module!r} ({licence})")
        print(
            "\nThese packages must not be imported from vigia/. Move the code to "
            "tools/ (offline build step) or replace it with an ONNX Runtime path.\n"
            "See PLAN.md section 4."
        )
        return 1

    print(f"LICENCE GUARD PASSED — {checked} file(s) under vigia/, "
          f"no forbidden imports ({', '.join(sorted(FORBIDDEN))}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
