#!/usr/bin/env python3
"""Export the SeaDronesSee RF-DETR detector to ONNX.

Offline build step — run with .venv-export.

Why RF-DETR rather than the YOLO variants published alongside it: licensing.
The same author released both RF-DETR and YOLO11 checkpoints for SeaDronesSee;
the RF-DETR ones are Apache-2.0 and the YOLO ones AGPL-3.0. RF-DETR keeps this
detector, like flood, outside the AGPL perimeter described in PLAN.md section 4.
It is also the model that led the RF100-VL domain-transfer benchmark, which is
the property that matters when a detector meets footage unlike its training set.

Model  : dronefreak/seadronessee-rfdetr-small
Base   : Roboflow/rf-detr-small
Licence: Apache-2.0
Classes: swimmer, boat, jetski, life_saving_appliances, buoy

Reported by the author on the SeaDronesSee val split (NOT measured by us):
mAP@50 0.7931, mAP@50-95 0.4525, precision 0.8986, recall 0.7789, F1 0.8289.

Per-class AP matters far more than the headline here:
    boat 0.711 | jetski 0.593 | buoy 0.489 | swimmer 0.282 | appliances 0.188
The class VIGÍA actually needs — a person in the water — is the second weakest.
Quoting the headline mAP for a "people in water" detector would be misleading.

Usage:
    .venv-export/bin/python scripts/export_person_in_water_onnx.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / ".vigia_cache" / "sds" / "rfdetr_small.pth"
DEFAULT_OUT = REPO_ROOT / "models" / "person_in_water" / "seadronessee_rfdetr_small.onnx"

# Order fixed by the dataset's data.yaml, not guessed.
CLASS_NAMES = ("swimmer", "boat", "jetski", "life_saving_appliances", "buoy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resolution", type=int, default=None,
                        help="defaults to the resolution stored in the checkpoint")
    args = parser.parse_args()

    try:
        from rfdetr import RFDETRSmall
    except ImportError:
        print("rfdetr is not installed here. Use the export environment:\n"
              "    .venv-export/bin/pip install rfdetr\n"
              "    .venv-export/bin/python scripts/export_person_in_water_onnx.py",
              file=sys.stderr)
        return 2

    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"checkpoint : {args.checkpoint.name} "
          f"({args.checkpoint.stat().st_size / 1e6:.0f} MB)")

    kwargs = {"pretrain_weights": str(args.checkpoint)}
    if args.resolution:
        kwargs["resolution"] = args.resolution
    model = RFDETRSmall(**kwargs)
    print(f"loaded RFDETRSmall, {len(CLASS_NAMES)} classes: {', '.join(CLASS_NAMES)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_dir = args.out.parent / "_rfdetr_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("exporting to ONNX (this rebuilds the graph; it takes a minute) ...")
    model.export(output_dir=str(export_dir))

    produced = sorted(export_dir.rglob("*.onnx"))
    if not produced:
        print(f"Export produced no .onnx under {export_dir}", file=sys.stderr)
        return 1

    # Take the largest artefact: some exporters emit auxiliary graphs alongside
    # the model, and the weights-bearing one is always the biggest.
    source = max(produced, key=lambda p: p.stat().st_size)
    shutil.move(str(source), str(args.out))

    # Fold any external weight sidecar in, for the same reason as the flood
    # export: a bare .onnx with weights in a neighbouring file loads and then
    # produces garbage if only the .onnx is copied.
    try:
        import onnx
        model_proto = onnx.load(str(args.out))
        onnx.save_model(model_proto, str(args.out), save_as_external_data=False)
        for sidecar in args.out.parent.glob(f"{args.out.name}.data"):
            sidecar.unlink()
            print(f"  folded {sidecar.name} into {args.out.name}")
    except Exception as exc:            # noqa: BLE001 - informational only
        print(f"  (could not consolidate external data: {exc})")

    shutil.rmtree(export_dir, ignore_errors=True)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nwrote {args.out} ({size_mb:.1f} MB)")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(args.out),
                                       providers=["CPUExecutionProvider"])
        print("graph loads under ONNX Runtime:")
        for tensor in session.get_inputs():
            print(f"  IN  {tensor.name:<12} {tensor.shape} {tensor.type}")
        for tensor in session.get_outputs():
            print(f"  OUT {tensor.name:<12} {tensor.shape} {tensor.type}")
    except Exception as exc:            # noqa: BLE001
        print(f"  WARNING: could not load the exported graph: {exc}",
              file=sys.stderr)
        return 1

    metadata = args.out.with_suffix(".meta.json")
    metadata.write_text(json.dumps({
        "classes": list(CLASS_NAMES),
        "hazard_class": "swimmer",
        "source": "https://huggingface.co/dronefreak/seadronessee-rfdetr-small",
        "licence": "Apache-2.0",
        "reported_by_author_val_split": {
            "mAP_50": 0.7931, "mAP_50_95": 0.4525,
            "precision": 0.8986, "recall": 0.7789, "f1": 0.8289,
            "per_class_AP": {"boat": 0.711, "jetski": 0.593, "buoy": 0.489,
                             "swimmer": 0.282, "life_saving_appliances": 0.188},
            "note": "The swimmer class — the one VIGÍA needs — is the second "
                    "weakest. Do not quote the headline mAP for this detector.",
        },
    }, indent=2))
    print(f"wrote {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
