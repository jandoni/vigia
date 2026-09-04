#!/usr/bin/env python3
"""Export the trained flood segmentation model to ONNX.

Offline build step — run with .venv-export. The output is what ships; torch
never enters the runtime environment.

DeepLabV3 returns an OrderedDict with 'out' and 'aux' keys. The auxiliary head
exists only as a deep-supervision signal during training and is dead weight at
inference, so we wrap the model to return the primary logits alone. That both
shrinks the graph and keeps the ONNX output a plain tensor, which the runtime
segmenter expects.

Usage:
    .venv-export/bin/python scripts/export_flood_onnx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "flood" / "atlantis_water_deeplabv3.pt"
DEFAULT_OUT = REPO_ROOT / "models" / "flood" / "atlantis_water_deeplabv3.onnx"


def consolidate_external_data(path: Path) -> None:
    """Fold any external weight sidecar back into the .onnx file itself."""
    import onnx

    model = onnx.load(str(path))          # resolves external data if present
    sidecars = list(path.parent.glob(f"{path.name}.data")) + \
               list(path.parent.glob(f"{path.stem}*.onnx.data"))

    onnx.save_model(model, str(path), save_as_external_data=False)

    for sidecar in sidecars:
        if sidecar.exists():
            sidecar.unlink()
            print(f"  folded {sidecar.name} into {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="defaults to the size recorded in the checkpoint")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Use .venv-export.", file=sys.stderr)
        return 2

    if not args.weights.exists():
        print(f"No checkpoint at {args.weights}. Train it first with "
              f"scripts/train_flood.py", file=sys.stderr)
        return 1

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    imgsz = args.imgsz or checkpoint.get("imgsz", 512)
    water_values = checkpoint.get("water_values", [])
    reported_iou = checkpoint.get("val_water_iou")

    print(f"checkpoint : {args.weights.name}")
    print(f"imgsz      : {imgsz}")
    print(f"val IoU    : {reported_iou:.4f}" if reported_iou else "val IoU    : unknown")
    print(f"water vals : {water_values}")

    model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(10, 2, kernel_size=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class PrimaryLogitsOnly(nn.Module):
        """Return only the main head's logits, as a bare tensor."""

        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            return self.wrapped(x)["out"]

    wrapper = PrimaryLogitsOnly(model).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz)

    with torch.no_grad():
        reference = wrapper(dummy)
    print(f"output shape: {tuple(reference.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(args.out),
        input_names=["images"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        do_constant_folding=True,
    )

    # torch's exporter may split weights into a sidecar `.onnx.data` file. That
    # is a deployment hazard: copying only the .onnx yields a graph with no
    # weights, which loads and then produces garbage. VIGIA distributes one file
    # per model, so consolidate unconditionally.
    consolidate_external_data(args.out)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nwrote {args.out} ({size_mb:.1f} MB, single file)")

    # Verify the exported graph agrees with the PyTorch model before trusting it.
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.out),
                                       providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"images": dummy.numpy()})[0]
        difference = float(np.abs(onnx_output - reference.numpy()).max())
        agreement = float(
            (onnx_output.argmax(1) == reference.numpy().argmax(1)).mean()
        )
        print(f"parity check: max logit difference {difference:.6f}, "
              f"argmax agreement {agreement:.4%}")
        if agreement < 0.999:
            print("  WARNING: exported graph disagrees with the source model.",
                  file=sys.stderr)
            return 1
    except ImportError:
        print("onnxruntime not available here; skipping parity check")

    metadata = args.out.with_suffix(".meta.json")
    metadata.write_text(json.dumps({
        "imgsz": imgsz, "water_values": water_values,
        "val_water_iou": reported_iou, "opset": args.opset,
        "classes": ["background", "water"],
        "normalisation": "ImageNet mean/std, RGB, resized (not letterboxed)",
    }, indent=2))
    print(f"wrote {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
