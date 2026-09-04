#!/usr/bin/env python3
"""Fine-tune a flood segmenter for NADIR AERIAL imagery.

OFFLINE TRAINING SCRIPT. Run with .venv-export. The product is an ONNX file.

    .venv-export/bin/python scripts/train_flood_aerial.py --epochs 20
    .venv-export/bin/python scripts/train_flood_aerial.py --epochs 1 --limit 40

WHY A SECOND FLOOD MODEL. The existing flood segmenter is trained on ATLANTIS,
which is ground-level and oblique photography of water bodies. Measured on
straight-down UAV frames it reports up to 0.397 water coverage on images whose
actual water is a narrow canal, tinting large areas of mown grass as water. It
does not fail loudly; it returns a confident wrong number.

That failure is already CONTAINED — `Viewpoint` is declared on every camera and
detector and the pipeline refuses to pair a nadir camera with the ground model.
Containment is not capability, though: a UAV flood camera is currently refused
rather than served. This script closes the gap by training a model that has
actually seen the viewpoint, so the gate has something to route to.

DATA. FloodNet Track 1 (Rahnemoonfar et al., arXiv 2012.02951), UAV imagery of
Hurricane Harvey with 10-class semantic masks. Obtained from the torchgeo
rehost of the authors' own release, under CDLA-Permissive-1.0 — which, unlike
the CC BY-SA on the Track 2 mirror, carries no share-alike obligation and so
does not constrain anything built from it.

CLASS MAPPING IS DERIVED FROM THE DATA, NOT ASSUMED. The same discipline as
ATLANTIS, where mask values turned out to be label_id + 1 and were established
empirically rather than taken from the documentation. `--inspect` prints the
observed mask values and their frequencies so the mapping below can be checked
against reality before a single epoch is run.

WHICH CLASSES COUNT AS WATER. Liquid water surfaces only, mirroring the ATLANTIS
decision so the two models mean the same thing by "water":

    water           -> a water body in the scene
    road-flooded    -> road surface under water. This is the flood signal
    pool            -> water, though not flooding; included for consistency
                       with ATLANTIS, which counted swimming_pool as water

    building-flooded -> EXCLUDED. The building is flooded but the pixels are
                        roof and structure, not a water surface — the same
                        reason ATLANTIS excluded dams, levees and piers.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "floodnet"
OUT_DIR = REPO_ROOT / "models" / "flood"

#: FloodNet Track 1 label ids, per the authors' documentation. Verified against
#: the masks by --inspect before use.
LABELS = {
    0: "background", 1: "building-flooded", 2: "building-non-flooded",
    3: "road-flooded", 4: "road-non-flooded", 5: "water", 6: "tree",
    7: "vehicle", 8: "pool", 9: "grass",
}
WATER_IDS = {3, 5, 8}


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """(image, mask) pairs, matched by stem.

    FloodNet ships several directory layouts depending on the mirror, so the
    pairing is done by filename stem rather than by assuming a fixed tree.

    WHAT THIS RELEASE ACTUALLY CONTAINS, because it shapes everything below:
    Track 1 is a SEMI-SUPERVISED challenge. Of 1,445 training images only 398
    carry masks, and of those only 51 are flooded — the other 347 are dry
    scenes. Validation and test masks were withheld for the competition and are
    not in the release, so a held-out split has to be carved from the 398.
    """
    masks: dict[str, Path] = {}
    for path in root.rglob("*.png"):
        lowered = str(path).lower()
        if "lab" in lowered or "mask" in lowered or "annot" in lowered:
            masks[path.stem.replace("_lab", "").replace("_mask", "")] = path

    pairs = []
    for path in root.rglob("*.jpg"):
        stem = path.stem
        if stem in masks:
            pairs.append((path, masks[stem]))
    return sorted(pairs)


def inspect(root: Path, sample: int = 40) -> int:
    """Print observed mask values, so the mapping is checked not assumed."""
    import cv2

    pairs = find_pairs(root)
    print(f"image/mask pairs found: {len(pairs)}")
    if not pairs:
        print(f"none under {root}. Extract FloodNet Track 1 there first.",
              file=sys.stderr)
        return 1

    counter: collections.Counter = collections.Counter()
    for _, mask_path in pairs[:sample]:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, counts):
            counter[int(value)] += int(count)

    total = sum(counter.values()) or 1
    print(f"\nmask values over {min(sample, len(pairs))} masks:")
    for value, count in sorted(counter.items()):
        name = LABELS.get(value, "UNKNOWN — mapping may be wrong")
        flag = "  <- counted as water" if value in WATER_IDS else ""
        print(f"  {value:>3}  {count / total:6.2%}  {name}{flag}")

    unknown = [v for v in counter if v not in LABELS]
    if unknown:
        print(f"\nWARNING: mask values not in the documented label set: "
              f"{unknown}.\nDo not train until the mapping is understood.",
              file=sys.stderr)
        return 1
    print("\nmapping consistent with the documented labels.")
    return 0


class FloodNetWater:
    """FloodNet Track 1 collapsed to binary water.

    Module level so DataLoader workers can pickle it.
    """

    def __init__(self, pairs, augment: bool, imgsz: int):
        self.pairs = pairs
        self.augment = augment
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import cv2
        import torch

        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            image = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
            mask = np.zeros((self.imgsz, self.imgsz), np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image = cv2.resize(image, (self.imgsz, self.imgsz),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz),
                          interpolation=cv2.INTER_NEAREST)

        binary = np.isin(mask, list(WATER_IDS)).astype(np.int64)

        if self.augment and random.random() < 0.5:
            image = image[:, ::-1]
            binary = binary[:, ::-1]

        rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / \
                 np.array([0.229, 0.224, 0.225], np.float32)
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1)))
        return torch.from_numpy(tensor), torch.from_numpy(
            np.ascontiguousarray(binary))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "floodnet_aerial_deeplabv3.pt")
    parser.add_argument("--inspect", action="store_true",
                        help="print observed mask values and exit")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the last checkpoint")
    parser.add_argument("--phase-epochs", type=int, default=0,
                        help="run at most this many epochs then exit cleanly, "
                             "so a long schedule can be done in bounded phases")
    args = parser.parse_args()

    if args.inspect:
        return inspect(args.data)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large,
        )
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Use .venv-export.", file=sys.stderr)
        return 2

    pairs = find_pairs(args.data)
    if not pairs:
        print(f"no image/mask pairs under {args.data}", file=sys.stderr)
        return 1

    # STRATIFIED SPLIT, because the class balance here is extreme: 51 flooded
    # against 347 dry. A random 15% validation split draws roughly eight
    # flooded images, so "water IoU" would largely measure whether the model
    # finds ponds in dry neighbourhoods rather than whether it finds flooding —
    # the one thing it exists to do. Splitting each group separately keeps the
    # proportion honest on both sides.
    flooded = [p for p in pairs if "flooded" in str(p[0]).lower()
               and "non-flooded" not in str(p[0]).lower()]
    dry = [p for p in pairs if p not in flooded]
    random.seed(args.seed)
    random.shuffle(flooded)
    random.shuffle(dry)
    if args.limit:
        keep = max(1, args.limit // 2)
        flooded, dry = flooded[:keep], dry[:keep]

    def cut(group):
        n = max(1, int(len(group) * args.val_fraction))
        return group[:n], group[n:]

    val_f, train_f = cut(flooded)
    val_d, train_d = cut(dry)
    val_pairs, train_pairs = val_f + val_d, train_f + train_d
    random.shuffle(train_pairs)

    print(f"labelled pairs: {len(pairs)}  "
          f"({len(flooded)} flooded, {len(dry)} dry)")
    print(f"  train {len(train_pairs)} ({len(train_f)} flooded)   "
          f"val {len(val_pairs)} ({len(val_f)} flooded)")
    if len(flooded) < 60:
        print(f"  NOTE: only {len(flooded)} flooded images exist in this "
              f"release. Every figure below\n        is bounded by that, and "
              f"the flooded-subset IoU is the one that matters.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_set = FloodNetWater(train_pairs, True, args.imgsz)
    val_set = FloodNetWater(val_pairs, False, args.imgsz)

    # DeepLabV3 is plain convolution end to end, which MPS handles well — the
    # opposite of the Faster R-CNN case, where MPS was 64x slower than CPU.
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"train {len(train_set)}  val {len(val_set)}  device {device}  "
          f"imgsz {args.imgsz}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=4)
    flooded_loader = DataLoader(FloodNetWater(val_f, False, args.imgsz),
                                batch_size=args.batch, num_workers=2)

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(10, 2, kernel_size=1)
    model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_iou, history = 0.0, []
    start_epoch = 1

    # PHASED TRAINING. Each invocation runs a bounded number of epochs and
    # exits, writing a checkpoint that carries the optimiser and scheduler
    # state as well as the weights. Resuming therefore continues the same
    # schedule rather than restarting it — a cosine learning rate that resets
    # every phase would train a different, worse model than one long run.
    last_path = args.out.with_name(f"{args.out.stem}_last.pt")
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_iou = state.get("best_iou", 0.0)
        history = state.get("history", [])
        print(f"resumed from epoch {state['epoch']} "
              f"(best flooded IoU so far {best_iou:.4f})")
        if start_epoch > args.epochs:
            print("schedule already complete")
            return 0

    stop_after = (start_epoch + args.phase_epochs - 1) if args.phase_epochs \
        else args.epochs

    for epoch in range(start_epoch, min(args.epochs, stop_after) + 1):
        model.train()
        running, started = 0.0, time.perf_counter()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss = criterion(out["out"], masks) + 0.4 * criterion(out["aux"], masks)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
        scheduler.step()

        # Water IoU on the held-out split — the metric that matters, and the
        # same one reported for the ground model so the two are comparable.
        model.eval()
        inter = union = correct = total = 0
        f_inter = f_union = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                pred = model(images)["out"].argmax(1)
                inter += int(((pred == 1) & (masks == 1)).sum())
                union += int(((pred == 1) | (masks == 1)).sum())
                correct += int((pred == masks).sum())
                total += int(masks.numel())
            # The flooded subset on its own. A model can score well overall by
            # handling dry scenes correctly while failing on the flooding it
            # exists to detect, so the two are reported separately.
            for images, masks in flooded_loader:
                images, masks = images.to(device), masks.to(device)
                pred = model(images)["out"].argmax(1)
                f_inter += int(((pred == 1) & (masks == 1)).sum())
                f_union += int(((pred == 1) | (masks == 1)).sum())
        iou = inter / union if union else 0.0
        flooded_iou = f_inter / f_union if f_union else 0.0
        accuracy = correct / total if total else 0.0
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "loss": running / max(1, len(train_loader)),
                        "water_iou": iou, "flooded_water_iou": flooded_iou,
                        "pixel_accuracy": accuracy, "seconds": round(elapsed, 1)})

        # Selection is on the FLOODED subset: that is the operational case.
        marker = ""
        if flooded_iou > best_iou:
            best_iou = flooded_iou
            torch.save({"state_dict": model.state_dict(), "imgsz": args.imgsz,
                        "water_ids": sorted(WATER_IDS), "labels": LABELS,
                        "val_water_iou": iou, "val_flooded_water_iou": flooded_iou,
                        "epoch": epoch}, args.out)
            marker = "  <- best, saved"
        # Written every epoch, so an interruption costs at most one epoch.
        torch.save({"state_dict": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best_iou": best_iou, "history": history,
                    "imgsz": args.imgsz, "water_ids": sorted(WATER_IDS)},
                   last_path)

        print(f"epoch {epoch:>3}/{args.epochs}  loss "
              f"{running / max(1, len(train_loader)):.4f}  IoU all {iou:.4f}  "
              f"IoU flooded {flooded_iou:.4f}  acc {accuracy:.4f}  "
              f"{elapsed:.0f}s{marker}", flush=True)

    args.out.with_suffix(".history.json").write_text(json.dumps({
        "best_val_flooded_water_iou": best_iou,
        "selection_metric": "water IoU on the flooded validation subset", "epochs": args.epochs, "device": device,
        "train_images": len(train_set), "val_images": len(val_set),
        "water_ids": sorted(WATER_IDS), "history": history,
    }, indent=2))
    done = history[-1]["epoch"] if history else 0
    if done < args.epochs:
        print(f"\nphase complete: {done} of {args.epochs} epochs. "
              f"Resume with --resume.")
    else:
        print(f"\nschedule complete: {args.epochs} epochs.")
    print(f"best flooded-subset water IoU: {best_iou:.4f}")
    print(f"weights: {args.out}")
    print("\nNext: export to ONNX, then register with viewpoint nadir_aerial so "
          "the gate\ncan route UAV flood cameras to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
