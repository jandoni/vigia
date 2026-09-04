#!/usr/bin/env python3
"""Fine-tune a binary water-segmentation model on ATLANTIS.

OFFLINE TRAINING SCRIPT. Run with the export environment (.venv-export), which
carries torch. The runtime environment has neither torch nor ultralytics — the
product of this script is an ONNX file, and that is all that ships.

Model choice: torchvision's DeepLabV3 with a MobileNetV3-Large backbone.
Chosen for three reasons:

  1. Licence. torchvision is BSD-3-Clause, so the flood detector is free of
     AGPL entirely — unlike the fire and traffic detectors, which are YOLO
     derivatives and required the ONNX isolation described in PLAN.md section 4.
  2. Weight. MobileNetV3 is small enough to run alongside four other detectors
     on a laptop, which is the deployment VIGÍA targets.
  3. Pretraining. It ships with COCO-subset weights, so we fine-tune rather
     than train from scratch — 3,364 images is far too few for the latter.

Task: ATLANTIS carries 56 classes. We collapse them to binary water / not
water. VIGÍA does not need to know whether it is looking at a canal or a river;
it needs to know where the water is and whether it is rising.

Usage:
    .venv-export/bin/python scripts/train_flood.py --epochs 20
    .venv-export/bin/python scripts/train_flood.py --epochs 1 --limit 60   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ATLANTIS = REPO_ROOT / "data" / "atlantis_src" / "atlantis" / "atlantis"
LABELS_JSON = REPO_ROOT / "data" / "atlantis_src" / "atlantis" / "utils" / "labels_info.json"
OUT_DIR = REPO_ROOT / "models" / "flood"

# Liquid water surfaces only. Structures associated with water (dam, levee,
# pier, culvert, breakwater, water tower, water well) are excluded because they
# are not water; so are glaciers and snow, which are not liquid, and mangrove,
# which is vegetation standing in water and whose extent does not track a level.
WATER_LABEL_NAMES = [
    "flood", "canal", "ditch", "fjord", "hot_spring", "lake", "puddle",
    "rapids", "reservoir", "river", "river_delta", "sea", "spillway",
    "swimming_pool", "waterfall", "wetland", "marsh",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def water_mask_values() -> set[int]:
    """Mask pixel values that count as water.

    ATLANTIS mask values are label id + 1, with 0 reserved for unlabelled.
    Derived empirically from the masks and cross-checked against
    utils/labels_info.json rather than assumed.
    """
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    name_to_id = {entry["name"]: entry["id"] for entry in labels}
    missing = [n for n in WATER_LABEL_NAMES if n not in name_to_id]
    if missing:
        raise KeyError(f"labels_info.json is missing: {missing}")
    return {name_to_id[name] + 1 for name in WATER_LABEL_NAMES}


def build_index(split: str) -> list[tuple[Path, Path]]:
    """(image, mask) pairs for a split, across all label folders."""
    pairs: list[tuple[Path, Path]] = []
    images_root = ATLANTIS / "images" / split
    masks_root = ATLANTIS / "masks" / split
    for image_path in sorted(images_root.rglob("*.jpg")):
        mask_path = masks_root / image_path.parent.name / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


class AtlantisWater:
    """ATLANTIS collapsed to binary water segmentation.

    Defined at module level rather than inside main(): DataLoader worker
    processes pickle the dataset, and a class defined in a function scope
    cannot be pickled.
    """

    def __init__(self, split: str, augment: bool, imgsz: int,
                 water_values: list[int], limit: int = 0):
        self.pairs = build_index(split)
        if limit:
            self.pairs = self.pairs[:limit]
        self.augment = augment
        self.imgsz = imgsz
        self.water = np.array(water_values)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import cv2
        import torch

        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask_raw is None:
            # A corrupt file must not kill a long run; substitute a blank.
            image = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
            mask_raw = np.zeros((self.imgsz, self.imgsz), np.uint8)

        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]

        image = cv2.resize(image, (self.imgsz, self.imgsz),
                           interpolation=cv2.INTER_LINEAR)
        mask_raw = cv2.resize(mask_raw, (self.imgsz, self.imgsz),
                              interpolation=cv2.INTER_NEAREST)
        mask = np.isin(mask_raw, self.water).astype(np.int64)

        if self.augment:
            if random.random() < 0.5:                    # horizontal flip
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.3:                    # brightness / contrast
                alpha = random.uniform(0.75, 1.25)
                beta = random.uniform(-25, 25)
                image = np.clip(image.astype(np.float32) * alpha + beta,
                                0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(tensor, (2, 0, 1))
        return (torch.from_numpy(np.ascontiguousarray(tensor)),
                torch.from_numpy(np.ascontiguousarray(mask)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap images per split (for smoke tests)")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "atlantis_water_deeplabv3.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import cv2
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large,
        )
    except ImportError as exc:
        print(f"Missing dependency ({exc}).\n"
              f"Run with the export environment:\n"
              f"    .venv-export/bin/python scripts/train_flood.py",
              file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    water_values = sorted(water_mask_values())
    print(f"water mask values ({len(water_values)}): {water_values}")

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    train_set = AtlantisWater("train", True, args.imgsz, water_values, args.limit)
    val_set = AtlantisWater("val", False, args.imgsz, water_values, args.limit)
    print(f"train {len(train_set)}  val {len(val_set)}  device {device}  "
          f"imgsz {args.imgsz}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=4)

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=True)
    # Replace both heads: pretrained on 21 COCO classes, we need 2.
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(10, 2, kernel_size=1)
    model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    def evaluate() -> tuple[float, float]:
        """Water-class IoU and pixel accuracy on the validation split."""
        model.eval()
        intersection = union = correct = total = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)["out"]
                predicted = logits.argmax(1)
                pred_water, true_water = predicted == 1, masks == 1
                intersection += int((pred_water & true_water).sum())
                union += int((pred_water | true_water).sum())
                correct += int((predicted == masks).sum())
                total += int(masks.numel())
        iou = intersection / union if union else 0.0
        return iou, correct / total if total else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_iou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.perf_counter()
        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            # The auxiliary head is a deep-supervision signal; the standard
            # torchvision recipe weights it at 0.4.
            loss = criterion(outputs["out"], masks) + 0.4 * criterion(outputs["aux"], masks)
            loss.backward()
            optimiser.step()
            running += float(loss)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}", flush=True)

        scheduler.step()
        iou, accuracy = evaluate()
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "loss": running / max(1, len(train_loader)),
                        "val_water_iou": iou, "val_pixel_accuracy": accuracy,
                        "seconds": round(elapsed, 1)})
        marker = ""
        if iou > best_iou:
            best_iou = iou
            torch.save({"state_dict": model.state_dict(),
                        "water_values": water_values, "imgsz": args.imgsz,
                        "val_water_iou": iou}, args.out)
            marker = "  <- best, saved"
        print(f"epoch {epoch:>3}/{args.epochs}  loss {running / max(1, len(train_loader)):.4f}  "
              f"val water IoU {iou:.4f}  pixel acc {accuracy:.4f}  "
              f"{elapsed:.0f}s{marker}", flush=True)

    history_path = args.out.with_suffix(".history.json")
    history_path.write_text(json.dumps(
        {"best_val_water_iou": best_iou, "epochs": args.epochs,
         "imgsz": args.imgsz, "batch": args.batch, "lr": args.lr,
         "device": device, "train_images": len(train_set),
         "val_images": len(val_set), "water_values": water_values,
         "history": history}, indent=2))

    print(f"\nbest validation water IoU: {best_iou:.4f}")
    print(f"weights : {args.out}")
    print(f"history : {history_path}")
    print("\nNext: .venv-export/bin/python scripts/export_flood_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
