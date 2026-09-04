#!/usr/bin/env python3
"""Export the trained building-access detector to ONNX.

Offline build step — run with .venv-export. The output is what ships; torch
never enters the runtime environment.

Two things make this export different from the flood one, and both are traps.

NORMALISATION LIVES INSIDE THE GRAPH. torchvision detection models carry a
`GeneralizedRCNNTransform` that resizes and applies ImageNet mean/std itself,
before the backbone ever sees the tensor. The training dataset therefore fed
plain 0-1 RGB, and so must the runtime. Normalising again outside the graph
would double-normalise and quietly wreck accuracy — no error, just worse
numbers. The transform's min/max size is pinned here so that resize is a no-op
at our native 640, which also keeps the exported shapes static.

THE OUTPUT IS ALREADY NMS-ED. Faster R-CNN returns a list of per-image dicts
with `boxes`, `labels` and `scores` after its own non-maximum suppression. The
wrapper below flattens that to three plain tensors for image 0, because a list
of dicts is not a thing ONNX can express. The runtime must NOT run NMS again.

Usage:
    .venv-export/bin/python scripts/export_building_access_onnx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "building_access" / "drespnet_fasterrcnn.pt"
DEFAULT_OUT = REPO_ROOT / "models" / "building_access" / "drespnet_fasterrcnn.onnx"


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


def load_sample(args, torch):
    """A real 640x640 frame as NCHW float 0-1, matching runtime preprocessing.

    Falls back to noise only if no dataset frame can be found, and says so —
    a noise sample makes the parity check nearly worthless (see caller).
    """
    import numpy as np

    candidates = []
    if args.sample:
        candidates.append(args.sample)
    else:
        for split in ("valid", "test", "train"):
            candidates.extend(sorted(
                (REPO_ROOT / "data" / "drespnet" / split).glob("*.jpg")
            )[:1])

    for path in candidates:
        if not path.exists():
            continue
        try:
            import cv2
        except ImportError:
            break
        image = cv2.imread(str(path))
        if image is None:
            continue
        resized = cv2.resize(image, (args.imgsz, args.imgsz),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]
        )
        return tensor, path.name

    return torch.rand(1, 3, args.imgsz, args.imgsz), "RANDOM NOISE (no frame found)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--score-threshold", type=float, default=0.05,
                        help="graph-level floor; the runtime filters higher")
    parser.add_argument("--sample", type=Path, default=None,
                        help="real image to trace and parity-check against; "
                             "defaults to the first DRespNeT valid frame")
    parser.add_argument("--random-weights", action="store_true",
                        help="export an untrained model to test the export path "
                             "itself, before training has finished")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Use .venv-export.", file=sys.stderr)
        return 2

    if args.random_weights:
        class_names = ["civilian", "rescue_team", "entry_accessible",
                       "entry_blocked", "building_collapsed"]
        class_merge = None
        val_loss = None
        print("EXPORT PATH TEST — untrained weights, output is not a usable model")
    else:
        if not args.weights.exists():
            print(f"No checkpoint at {args.weights}. Train it first with "
                  f"scripts/train_building_access.py", file=sys.stderr)
            return 1
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        class_names = list(checkpoint["class_names"])
        class_merge = checkpoint.get("class_merge")
        val_loss = checkpoint.get("val_loss")
        print(f"checkpoint : {args.weights.name}")
        print(f"val loss   : {val_loss:.4f}" if val_loss else "val loss   : unknown")

    print(f"classes    : {len(class_names)} -> {', '.join(class_names)}")
    print(f"imgsz      : {args.imgsz}")

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_names) + 1)
    if not args.random_weights:
        model.load_state_dict(checkpoint["state_dict"])

    # Pin the internal transform so the resize is an identity at our native
    # size. Without this the graph carries a dynamic resize whose output shape
    # depends on the input aspect ratio, which defeats a static export.
    model.transform.min_size = (args.imgsz,)
    model.transform.max_size = args.imgsz
    model.roi_heads.score_thresh = args.score_threshold
    model.eval()

    class DetectionGraph(nn.Module):
        """NCHW float tensor in, three flat tensors out.

        ONNX has no list-of-dicts, so the per-image result is unpacked here.
        Batch is fixed at 1: VIGÍA scores one frame at a time.
        """

        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, images):
            result = self.wrapped([images[0]])[0]
            return result["boxes"], result["scores"], result["labels"]

    wrapper = DetectionGraph(model).eval()

    # Trace and parity-check on a REAL frame, not on noise.
    #
    # This is not fussiness. A correctly trained detector finds nothing in
    # random noise, so a noise-based parity check compares zero detections
    # against zero detections and passes while verifying nothing at all — it is
    # weakest exactly when the model is good. Measured during this build: the
    # first trained checkpoint reported "torch 0 vs onnx 0 detections" and the
    # check was vacuous. On a real frame the same export compares real boxes.
    dummy, sample_source = load_sample(args, torch)

    with torch.no_grad():
        ref_boxes, ref_scores, ref_labels = wrapper(dummy)
    print(f"sample     : {sample_source}")
    print(f"reference  : {ref_boxes.shape[0]} detections above "
          f"{args.score_threshold}")
    if ref_boxes.shape[0] == 0:
        print("  WARNING: no detections on the parity sample, so the check "
              "below\n           compares nothing. Treat the export as "
              "UNVERIFIED.", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        opset_version=args.opset,
        dynamic_axes={"boxes": {0: "detections"},
                      "scores": {0: "detections"},
                      "labels": {0: "detections"}},
        do_constant_folding=True,
    )
    # torch 2.x may default to the dynamo exporter, which does not yet handle
    # Faster R-CNN's control flow. Ask for the TorchScript path explicitly when
    # this build supports the flag.
    try:
        torch.onnx.export(wrapper, (dummy,), str(args.out),
                          dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (dummy,), str(args.out), **export_kwargs)

    consolidate_external_data(args.out)

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"\nwrote {args.out} ({size_mb:.1f} MB, single file)")

    # Verify the exported graph agrees with the PyTorch model before trusting
    # it. The RF-DETR integration bug (see REGISTRY.yaml) is the reason this
    # check compares coordinates and not just counts.
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.out),
                                       providers=["CPUExecutionProvider"])
        boxes, scores, labels = session.run(None, {"images": dummy.numpy()})

        print(f"parity     : torch {ref_boxes.shape[0]} vs onnx {boxes.shape[0]} "
              f"detections")
        if boxes.shape[0] != ref_boxes.shape[0]:
            print("  WARNING: detection count differs.", file=sys.stderr)
            return 1
        if boxes.shape[0]:
            box_delta = float(np.abs(boxes - ref_boxes.numpy()).max())
            score_delta = float(np.abs(scores - ref_scores.numpy()).max())
            label_match = float((labels == ref_labels.numpy()).mean())
            print(f"             max box delta {box_delta:.4f} px, "
                  f"max score delta {score_delta:.6f}, "
                  f"label agreement {label_match:.2%}")

            # Box parity is only meaningful for a trained model. An untrained
            # head emits huge regression deltas, and the exp() in the box
            # decode amplifies a 1e-7 difference into hundreds of pixels — this
            # export measured 553 px on random weights while scores matched to
            # 0.000000 and labels to 100%. Verified against the stock COCO
            # checkpoint on a real DRespNeT frame, where the same code path
            # gives a max box delta of 3.05e-05 px. So scores and labels are
            # the trustworthy signal in test mode; boxes are not.
            if label_match < 1.0 or score_delta > 1e-4:
                print("  WARNING: exported graph disagrees with the source model.",
                      file=sys.stderr)
                return 1
            if box_delta > 1.0:
                if args.random_weights:
                    print("             (large box delta expected on untrained "
                          "weights — see comment; scores and labels agree)")
                else:
                    print("  WARNING: box coordinates disagree with the source "
                          "model.", file=sys.stderr)
                    return 1
    except ImportError:
        print("onnxruntime not available here; skipping parity check")

    metadata = args.out.with_suffix(".meta.json")
    metadata.write_text(json.dumps({
        "imgsz": args.imgsz,
        "classes": class_names,
        # Carried through from the checkpoint so the evaluation harness can
        # rebuild ground truth with the SAME 28 -> 5 merge the model was
        # trained on, without importing the training script or duplicating the
        # map. One source of truth, and it travels with the model file.
        "class_merge": class_merge,
        "label_offset": 1,
        "label_note": "ONNX labels are 1-based; 0 is torchvision's background "
                      "class and is never emitted. class_names[label - 1].",
        "opset": args.opset,
        "score_threshold": args.score_threshold,
        "val_loss": val_loss,
        "nms": "applied inside the graph by torchvision; do not repeat it",
        "preprocessing": "RGB, 0-1 float, resized to imgsz. ImageNet mean/std "
                         "normalisation happens INSIDE the graph.",
        "untrained_export_path_test": bool(args.random_weights),
    }, indent=2))
    print(f"wrote {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
