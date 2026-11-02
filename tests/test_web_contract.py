"""Tests that the operator view's imports still resolve.

The web extra (FastAPI, uvicorn) is not installed in CI, so nothing used to
import vigia/web/server.py there. When the overlay module was rewritten,
`encode_jpeg` and `placeholder` disappeared from it while server.py still
imported them, and `make dev` failed on a fresh clone while CI stayed green.

These tests close that gap without installing the web extra: server.py is read
with the AST module rather than imported, and every name it takes from a
`vigia.*` module is checked to exist there.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SERVER = REPO_ROOT / "vigia" / "web" / "server.py"


def test_server_imports_from_vigia_resolve() -> None:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("vigia"):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            if not hasattr(module, alias.name):
                missing.append(f"line {node.lineno}: {node.module}.{alias.name}")
    assert not missing, "server.py imports names that do not exist:\n  " + "\n  ".join(missing)


def test_placeholder_encodes_as_jpeg() -> None:
    from vigia.web.overlay import encode_jpeg, placeholder

    frame = placeholder()
    assert frame.shape == (540, 960, 3) and frame.dtype == np.uint8
    data = encode_jpeg(frame)
    assert data[:2] == b"\xff\xd8", "not a JPEG stream"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
