#!/usr/bin/env python3
"""Offline build step: convert a `.pt` checkpoint to ONNX.

THIS IS THE ONLY PLACE IN THE PROJECT THAT MAY IMPORT `ultralytics`.

Ultralytics ships under AGPL-3.0. VIGÍA ships under Apache-2.0. Those cannot
be combined in a distributed work, so the AGPL tool is confined to a build step
that runs on a developer machine and produces an artefact — it is never
imported by anything under `vigia/`, and never installed in the runtime
environment. `tools/check_licence.py` enforces that automatically.

Run it with the separate export environment, not the runtime one:

    .venv-export/bin/python tools/export_onnx.py \\
        --weights .vigia_cache/traffic/epoch14.pt \\
        --out models/traffic/enos_traffic_accident_yolo11x.onnx \\
        --imgsz 640

Exports with dynamic axes by default (matching how PyroNear publish theirs), so
`tools/make_static.py` can then produce the CoreML-friendly static variant.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="source .pt file")
    parser.add_argument("--out", type=Path, required=True, help="destination .onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset. 17 is widely supported; the default "
                             "Ultralytics opset can exceed what onnxruntime "
                             "guarantees and then fails to load.")
    parser.add_argument("--dynamic", action="store_true", default=True)
    parser.add_argument("--inspect-only", action="store_true",
                        help="print model metadata and exit without exporting")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed in this interpreter — by design.\n"
            "Use the isolated export environment:\n"
            "    python3 -m venv .venv-export\n"
            "    .venv-export/bin/pip install ultralytics onnx\n"
            "    .venv-export/bin/python tools/export_onnx.py ...",
            file=sys.stderr,
        )
        return 2

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 1

    print(f"loading {args.weights} ...")
    model = YOLO(str(args.weights))

    names = model.names
    print(f"  task    : {getattr(model, 'task', 'unknown')}")
    print(f"  classes : {len(names)} -> {names}")

    if args.inspect_only:
        return 0

    print(f"exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}, "
          f"dynamic={args.dynamic}) ...")
    produced = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(args.out))

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")
    print(f"classes: {names}")
    print("\nRecord the class names in models/REGISTRY.yaml, then run "
          "`tools/make_static.py` for the CoreML variant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
