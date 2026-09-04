#!/usr/bin/env python3
"""Export the trained scene router to ONNX.

    .venv-export/bin/python scripts/export_scene_router_onnx.py

Offline build step — run with .venv-export. The output is what ships; torch
never enters the runtime environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "scene_router" / "scene_router.pt"
DEFAULT_OUT = REPO_ROOT / "models" / "scene_router" / "scene_router.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Use .venv-export.", file=sys.stderr)
        return 2

    if not args.weights.exists():
        print(f"No checkpoint at {args.weights}. Train it first with "
              f"scripts/train_scene_router.py", file=sys.stderr)
        return 1

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    imgsz = int(checkpoint.get("imgsz", 224))
    accuracy = checkpoint.get("accuracy")

    print(f"checkpoint : {args.weights.name}")
    print(f"classes    : {', '.join(classes)}")
    print(f"accuracy   : {accuracy:.4f}" if accuracy else "accuracy   : unknown")

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        reference = model(dummy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, (dummy,), str(args.out), dynamo=False,
                          input_names=["image"], output_names=["logits"],
                          opset_version=args.opset, do_constant_folding=True,
                          dynamic_axes={"image": {0: "batch"},
                                        "logits": {0: "batch"}})
    except TypeError:
        torch.onnx.export(model, (dummy,), str(args.out),
                          input_names=["image"], output_names=["logits"],
                          opset_version=args.opset, do_constant_folding=True)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.1f} MB)")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.out),
                                       providers=["CPUExecutionProvider"])
        produced = session.run(None, {"image": dummy.numpy()})[0]
        delta = float(np.abs(produced - reference.numpy()).max())
        agree = bool(produced.argmax(1)[0] == int(reference.argmax(1)[0]))
        print(f"parity     : max logit delta {delta:.6f}, same class {agree}")
        if not agree or delta > 1e-3:
            print("  WARNING: the exported graph disagrees with the source model.",
                  file=sys.stderr)
            return 1
    except ImportError:
        print("onnxruntime not available here; skipping the parity check")

    args.out.with_suffix(".meta.json").write_text(json.dumps({
        "classes": classes,
        "imgsz": imgsz,
        "heldout_accuracy": accuracy,
        "preprocessing": "RGB, 0-1, ImageNet mean/std, resized (not letterboxed)",
        "scope": ("Upload path only. Live cameras route by declaration; see "
                  "vigia/registry.py and the note in vigia/scene_router.py."),
        "excluded": {"traffic": "only four usable training images exist"},
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
