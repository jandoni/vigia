#!/usr/bin/env python3
"""Build static-input-shape ONNX variants so CoreML can accelerate them.

Ultralytics exports with dynamic spatial dimensions. CoreML's execution
provider cannot build a plan for that and fails at the first inference, so the
model silently runs on CPU. Pinning the input to a fixed shape restores CoreML
and, measured on an M4 Pro, roughly halves latency.

This is a pure ONNX graph transformation using onnxruntime's own tooling — it
does NOT require the ultralytics package, so it stays clear of AGPL entirely.
The resulting variant is a *demo* artefact. Published numbers come from the
REFERENCE backend (CPU, dynamic graph). See vigia/detectors/base.py::Backend.

Usage:
    python tools/make_static.py                 # all fetched models
    python tools/make_static.py --imgsz 1024    # override the pinned size
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "models" / "REGISTRY.yaml"


def dynamic_dim_params(model_path: Path) -> list[str]:
    """Names of the symbolic (dynamic) input dimensions, in order."""
    import onnx

    model = onnx.load(str(model_path))
    names: list[str] = []
    for graph_input in model.graph.input:
        for dim in graph_input.type.tensor_type.shape.dim:
            if dim.dim_param:
                names.append(dim.dim_param)
    return names


def fix_dimension(source: Path, destination: Path, name: str, value: int) -> None:
    """Pin one symbolic dimension to a concrete value."""
    result = subprocess.run(
        [
            sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--dim_param", name, "--dim_value", str(value),
            str(source), str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to pin {name}={value}:\n{result.stderr or result.stdout}"
        )


def make_static(model_path: Path, imgsz: int) -> Path | None:
    destination = model_path.with_name(f"{model_path.stem}_static{imgsz}.onnx")
    if destination.exists():
        print(f"[ok]   {destination.name} already exists")
        return destination

    dims = dynamic_dim_params(model_path)
    if not dims:
        print(f"[skip] {model_path.name} has no dynamic dimensions")
        return None

    # Batch is always 1 for a live stream; spatial dims take the model's
    # native training resolution.
    values = {"batch": 1, "height": imgsz, "width": imgsz}
    unknown = [d for d in dims if d not in values]
    if unknown:
        print(f"[skip] {model_path.name}: unrecognised dynamic dims {unknown}")
        return None

    print(f"[make] {model_path.name} -> {destination.name}")
    print(f"       pinning {', '.join(f'{d}={values[d]}' for d in dims)}")

    with tempfile.TemporaryDirectory() as tmp:
        current = model_path
        for index, dim in enumerate(dims):
            is_last = index == len(dims) - 1
            target = destination if is_last else Path(tmp) / f"step{index}.onnx"
            fix_dimension(current, target, dim, values[dim])
            current = target

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"       {size_mb:.1f} MB")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="input size to pin (default: each model's registry imgsz)")
    args = parser.parse_args()

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    built = 0

    for key, spec in registry.get("models", {}).items():
        model_path = REPO_ROOT / spec["file"]
        if not model_path.exists():
            continue
        imgsz = args.imgsz or spec.get("inference", {}).get("imgsz", 640)
        print(f"\n{key}")
        if make_static(model_path, imgsz):
            built += 1

    print(f"\n{built} static variant(s) ready.")
    print("Use them with Backend.FAST. Reported numbers still come from "
          "Backend.REFERENCE — see PLAN.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
