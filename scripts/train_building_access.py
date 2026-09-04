#!/usr/bin/env python3
"""Fine-tune a detector for post-disaster building access and trapped people.

OFFLINE TRAINING SCRIPT. Run with .venv-export. The product is an ONNX file.

Dataset: DRespNeT (Cranfield University), UAV imagery from the 2023 Türkiye
earthquakes, released under CC BY 4.0. The initial public release used here is
650 train / 50 valid / 25 test images with 18,881 training annotations across
28 classes. The paper describes a larger corpus; this is what is released.

Model: torchvision Faster R-CNN with a MobileNetV3-Large FPN backbone,
BSD-3-Clause. Chosen for the same reason as the flood model — it keeps this
detector outside the AGPL perimeter, and it is light enough to run alongside
four other detectors.

CLASS REDUCTION. DRespNeT's 28 classes serve a broader research agenda than
ours. VIGÍA needs two things from a collapsed building: are there people, and
where can a rescuer get in. So the classes are merged down to five that answer
those questions, which also flattens a severe class imbalance (entry_door_blocked
has 52 instances; entry_window_blocked has 3,622).

    civilian           <- civilian_visible, group_of_civilians
    rescue_team        <- rescue_team
    entry_accessible   <- entry_door/window/gap_accessible, entry_gap_block_accessible
    entry_blocked      <- entry_door_blocked, entry_window_blocked
    building_collapsed <- building_collapsed

Usage:
    .venv-export/bin/python scripts/train_building_access.py --epochs 20
    .venv-export/bin/python scripts/train_building_access.py --epochs 1 --limit 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "drespnet"
OUT_DIR = REPO_ROOT / "models" / "building_access"

#: Why this model trains on CPU while the flood model trains on MPS.
#: Measured on this machine, batch 4 at 640x640, identical data and weights:
#:
#:     CPU  ->    2.34 s/step
#:     MPS  ->  148.79 s/step
#:
#: A 64x penalty, not a small one. Faster R-CNN leans on RoIAlign and the
#: proposal machinery, which Apple's Metal backend either does not implement
#: or implements badly, so the graph thrashes between GPU and CPU. DeepLabV3
#: in train_flood.py is plain convolution end to end and MPS suits it fine.
#: The device choice therefore belongs to the model, not to the machine.
MPS_NOTE = "Faster R-CNN is 64x slower on MPS than CPU here; CPU is correct."

#: Target class -> the DRespNeT source classes folded into it.
#: Index 0 is reserved for background by torchvision detection models.
CLASS_MERGE: dict[str, tuple[str, ...]] = {
    "civilian": ("civilian_visible", "group_of_civilians"),
    "rescue_team": ("rescue_team",),
    "entry_accessible": ("entry_door_accessible", "entry_window_accessible",
                         "entry_gap_accessible", "entry_gap_block_accessible"),
    "entry_blocked": ("entry_door_blocked", "entry_window_blocked"),
    "building_collapsed": ("building_collapsed",),
}
CLASS_NAMES = tuple(CLASS_MERGE)


def build_source_map(annotations_path: Path) -> dict[int, int]:
    """COCO category id -> our 1-based class index (0 is background)."""
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    by_name = {c["name"]: c["id"] for c in data["categories"]}
    mapping: dict[int, int] = {}
    for index, (target, sources) in enumerate(CLASS_MERGE.items(), start=1):
        for source in sources:
            if source in by_name:
                mapping[by_name[source]] = index
    return mapping


class DRespNeTDetection:
    """COCO-format DRespNeT, reduced to VIGÍA's five classes.

    Module level so DataLoader workers can pickle it.
    """

    def __init__(self, split: str, augment: bool, limit: int = 0):
        root = DATA_ROOT / split
        data = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
        self.root = root
        self.augment = augment
        self.source_map = build_source_map(root / "_annotations.coco.json")

        by_image: dict[int, list] = defaultdict(list)
        for annotation in data["annotations"]:
            target = self.source_map.get(annotation["category_id"])
            if target is None:
                continue                       # class not in our reduced set
            x, y, w, h = annotation["bbox"]
            if w <= 1 or h <= 1:
                continue                       # degenerate box; torchvision rejects it
            by_image[annotation["image_id"]].append(([x, y, x + w, y + h], target))

        self.items = [
            (image["file_name"], by_image[image["id"]])
            for image in data["images"] if by_image[image["id"]]
        ]
        if limit:
            self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import cv2
        import torch

        file_name, records = self.items[index]
        image = cv2.imread(str(self.root / file_name))
        if image is None:
            image = np.zeros((640, 640, 3), np.uint8)
            records = []

        boxes = np.array([r[0] for r in records], dtype=np.float32).reshape(-1, 4)
        labels = np.array([r[1] for r in records], dtype=np.int64)

        if self.augment and random.random() < 0.5:
            width = image.shape[1]
            image = image[:, ::-1]
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        )
        target = {"boxes": torch.from_numpy(boxes),
                  "labels": torch.from_numpy(labels)}
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "drespnet_fasterrcnn.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="auto picks cuda if present, else cpu (see MPS_NOTE)")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Use .venv-export.", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_set = DRespNeTDetection("train", augment=True, limit=args.limit)
    val_set = DRespNeTDetection("valid", augment=False, limit=args.limit)

    # CPU by default, and that is deliberate — see MPS_NOTE. CUDA is still
    # preferred where it exists; only Apple's GPU is the wrong answer here.
    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"classes : {len(CLASS_NAMES)} -> {', '.join(CLASS_NAMES)}")
    print(f"train   : {len(train_set)} images   val: {len(val_set)}   device: {device}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=2, collate_fn=collate)

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    model.to(device)

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    def validation_loss() -> float:
        """Mean training-style loss on the validation split.

        torchvision detection models only return losses in train mode, so the
        model is put in train() but stepped under no_grad. This is a proxy for
        quality, not a detection metric — the real numbers come from
        eval/run_building_access_eval.py against held-out test annotations.
        """
        model.train()
        total, batches = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [i.to(device) for i in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                losses = model(images, targets)
                total += float(sum(losses.values()))
                batches += 1
        return total / max(1, batches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, started = 0.0, time.perf_counter()
        for step, (images, targets) in enumerate(train_loader, 1):
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
            if step % 40 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}", flush=True)

        scheduler.step()
        val = validation_loss()
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)),
                        "val_loss": val, "seconds": round(elapsed, 1)})
        payload = {"state_dict": model.state_dict(),
                   "class_names": list(CLASS_NAMES),
                   "class_merge": {k: list(v) for k, v in CLASS_MERGE.items()},
                   "epoch": epoch,
                   "val_loss": val}

        # EVERY epoch is kept, deliberately. `validation_loss` is the
        # train-mode RPN/ROI loss — a proxy, as its own docstring says, not a
        # detection metric. For detection models it routinely rises while real
        # detection quality is still improving, which is exactly what happened
        # on the first attempt at this run: train loss fell 1.108 -> 0.858
        # while the proxy rose 1.072 -> 1.175 from epoch 2 onward. Selecting on
        # that number AND discarding every other epoch would have thrown away
        # the whole run on the strength of a signal we do not trust.
        #
        # So the proxy still marks `best` for continuity, but every epoch is
        # written and the real choice is made afterwards by
        # eval/run_building_access_eval.py against held-out annotations — the
        # project's stated methodology: measure, do not assume.
        epoch_dir = args.out.parent / "epochs"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, epoch_dir / f"epoch_{epoch:02d}.pt")
        torch.save(payload, args.out.with_name(f"{args.out.stem}_last.pt"))

        marker = ""
        if val < best:
            best = val
            torch.save(payload, args.out)
            marker = "  <- best by proxy loss"
        print(f"epoch {epoch:>3}/{args.epochs}  train {running / max(1, len(train_loader)):.4f}  "
              f"val {val:.4f}  {elapsed:.0f}s{marker}", flush=True)

    args.out.with_suffix(".history.json").write_text(json.dumps({
        "best_val_loss": best, "epochs": args.epochs, "device": device,
        "classes": list(CLASS_NAMES),
        "train_images": len(train_set), "val_images": len(val_set),
        "history": history,
    }, indent=2))

    print(f"\nbest proxy validation loss: {best:.4f}  ({args.out})")
    print(f"last epoch: {args.out.with_name(f'{args.out.stem}_last.pt')}")
    print(f"all epochs: {args.out.parent / 'epochs'}")
    print("\nThe proxy loss does NOT decide which checkpoint ships. Measure the"
          "\ncandidates against held-out annotations and pick on that:")
    print("  .venv-export/bin/python scripts/export_building_access_onnx.py "
          "--weights <candidate>")
    print("  python eval/run_building_access_eval.py --model <exported.onnx>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
