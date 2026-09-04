#!/usr/bin/env python3
"""Train the scene router: what kind of footage is this?

OFFLINE TRAINING SCRIPT. Run with .venv-export. The product is an ONNX file.

    .venv-export/bin/python scripts/train_scene_router.py --epochs 12
    .venv-export/bin/python scripts/train_scene_router.py --inspect

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.

VIGÍA routes LIVE cameras by declaration: an operator says once what a camera
overlooks and the gate obeys (vigia/registry.py). That decision stands, and this
model must never be used for it — a declared camera needs no classifier, and
adding a guessing step to the critical path would introduce a misrouting failure
where none exists today.

This model exists for a different problem the architecture never addressed: an
UPLOADED clip carries no declaration. Something has to choose a detector, and
the choice is between asking the person or inferring it. The first attempt
inferred it by running all five detectors and taking whichever claimed the
footage most strongly; it routed 2 of 4 test clips wrongly, because the
detectors' scores are not comparable quantities — flood's 1.9 and building
access's 0.7 measure different things, and on rubble the aerial flood model
genuinely reports water.

So this is a small classifier trained for exactly that job. It is confined to
the upload path, and the interface shows its answer with a one-click override.

THE SPLIT IS BY SOURCE, NOT RANDOM. Frames from one flight or one camera are
near-duplicates of each other; a random split puts almost-identical images on
both sides and reports an accuracy that means nothing. Each class is therefore
split so that the held-out material comes from a DIFFERENT dataset, camera or
part of the archive than the training material. Where that was not possible the
compromise is stated in the report rather than hidden.

Traffic is deliberately absent: four usable images exist, which is not a class.
Dashcam footage must be chosen by hand, and the interface says so.
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
OUT_DIR = REPO_ROOT / "models" / "scene_router"

#: class -> (train sources, held-out sources, how independent the split is)
SOURCES = {
    "fire": (
        ["data/pyro_sdis_val_wide/*.jpg"],
        ["data/figlib/*/*.jpg", "data/figlib_neg/*/*.jpg"],
        "Different dataset entirely: trained on pyro-sdis stills, tested on "
        "HPWREN FIgLib sequences from other cameras.",
    ),
    # BOTH VIEWPOINTS, deliberately. A first run trained this class on
    # FloodNet alone — all of it aerial — and scored 60% against ground-level
    # test cameras while every other class reached 100%. Water looks entirely
    # different looking down at it than looking across it, and a class taught
    # only one of those cannot recognise the other. ATLANTIS supplies the
    # ground-level half, restricted to the SAME seventeen liquid-water
    # categories the flood detector itself counts as water (see
    # models/REGISTRY.yaml) — structures like dams, piers and water towers are
    # excluded here for the same reason they are excluded there.
    "flood": (
        ["data/floodnet/**/*.jpg"]
        + [f"data/atlantis_src/**/images/**/{name}/*.jpg" for name in (
            "flood", "canal", "ditch", "fjord", "hot_spring", "lake", "puddle",
            "rapids", "reservoir", "river", "river_delta", "sea", "spillway",
            "swimming_pool", "waterfall", "wetland", "marsh")]
        + [f"data/atlantis_src/**/{name}/images/*.jpg" for name in (
            "flood", "canal", "ditch", "fjord", "hot_spring", "lake", "puddle",
            "rapids", "reservoir", "river", "river_delta", "sea", "spillway",
            "swimming_pool", "waterfall", "wetland", "marsh")],
        ["data/usgs_camera/*.jpg", "data/lsu_creeks/*/*.jpg"],
        "Different sources and both viewpoints: trained on FloodNet UAV "
        "imagery plus ATLANTIS ground-level water photography, tested on USGS "
        "gauge cameras and LSU creek cameras it has never seen.",
    ),
    "drowning": (
        ["data/seadronessee/val/*.jpg"],
        ["data/seadronessee/val_seq/*.jpg"],
        "Same dataset, different flight: the held-out set is one contiguous "
        "run that the sparse training sample does not overlap.",
    ),
    "building_access": (
        ["data/drespnet_raw/*.jpg"],
        [],                      # filled by an index split, see below
        "Same dataset, split by capture index so the two halves are different "
        "scenes — the archive contains 362 scene cuts across 615 frames. The "
        "weakest split of the four, and the figure should be read with that "
        "in mind.",
    ),
}
CLASSES = tuple(SOURCES)


def gather(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found += sorted(REPO_ROOT.glob(pattern))
    return found


def build_split(cap: int, seed: int = 0):
    """(train, test) as lists of (path, class index)."""
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []

    for index, name in enumerate(CLASSES):
        train_patterns, test_patterns, _ = SOURCES[name]
        train_files = gather(train_patterns)
        test_files = gather(test_patterns)

        if not test_files:
            # Index split: consecutive capture indices are different scenes
            # here, so the last fifth is held out as unseen material.
            cut = int(len(train_files) * 0.8)
            train_files, test_files = train_files[:cut], train_files[cut:]

        rng.shuffle(train_files)
        train += [(p, index) for p in train_files[:cap]]
        test += [(p, index) for p in test_files[: max(40, cap // 4)]]

    rng.shuffle(train)
    return train, test


class SceneSet:
    """Module level so DataLoader workers can pickle it."""

    def __init__(self, items, augment: bool, imgsz: int):
        self.items = items
        self.augment = augment
        self.imgsz = imgsz

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch

        path, label = self.items[i]
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        image = cv2.resize(image, (self.imgsz, self.imgsz),
                           interpolation=cv2.INTER_AREA)
        if self.augment and random.random() < 0.5:
            image = image[:, ::-1]
        rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / \
                 np.array([0.229, 0.224, 0.225], np.float32)
        return (torch.from_numpy(np.ascontiguousarray(tensor.transpose(2, 0, 1))),
                torch.tensor(label, dtype=torch.long))


def inspect(cap: int) -> int:
    train, test = build_split(cap)
    print(f"{'class':<18} {'train':>7} {'held out':>9}   split")
    print("-" * 92)
    for index, name in enumerate(CLASSES):
        n_train = sum(1 for _, l in train if l == index)
        n_test = sum(1 for _, l in test if l == index)
        note = SOURCES[name][2].split(":")[0]
        print(f"{name:<18} {n_train:>7} {n_test:>9}   {note}")
    print("-" * 92)
    print(f"{'total':<18} {len(train):>7} {len(test):>9}")
    if min(sum(1 for _, l in train if l == i) for i in range(len(CLASSES))) < 50:
        print("\nWARNING: a class has under 50 training images.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--phase-epochs", type=int, default=4,
                        help="run this many epochs then exit, so a long "
                             "schedule can be done in bounded phases")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--cap", type=int, default=420,
                        help="max training images per class, for balance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "scene_router.pt")
    args = parser.parse_args()

    if args.inspect:
        return inspect(args.cap)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision.models import (MobileNet_V3_Small_Weights,
                                        mobilenet_v3_small)
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Use .venv-export.", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_items, test_items = build_split(args.cap, args.seed)
    counts = {n: sum(1 for _, l in train_items if l == i)
              for i, n in enumerate(CLASSES)}
    print(f"train {len(train_items)}  held out {len(test_items)}  {counts}")

    train_set = SceneSet(train_items, True, args.imgsz)
    test_set = SceneSet(test_items, False, args.imgsz)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}  imgsz {args.imgsz}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, num_workers=4)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    last_path = args.out.with_name(f"{args.out.stem}_last.pt")
    start_epoch, best, history = 1, 0.0, []

    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best, history = state["best"], state["history"]
        print(f"resumed from epoch {state['epoch']} (best {best:.4f})")
        if start_epoch > args.epochs:
            print("schedule already complete")
            return 0

    stop_after = start_epoch + args.phase_epochs - 1 if args.phase_epochs \
        else args.epochs

    for epoch in range(start_epoch, min(args.epochs, stop_after) + 1):
        model.train()
        running, started = 0.0, time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss.detach())
        scheduler.step()

        model.eval()
        correct = total = 0
        per_class = {n: [0, 0] for n in CLASSES}
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predicted = model(images).argmax(1)
                correct += int((predicted == labels).sum())
                total += int(labels.numel())
                for label, prediction in zip(labels.tolist(), predicted.tolist()):
                    per_class[CLASSES[label]][1] += 1
                    per_class[CLASSES[label]][0] += int(label == prediction)
        accuracy = correct / total if total else 0.0
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "loss": running / max(1, len(train_loader)),
                        "accuracy": accuracy,
                        "per_class": {k: (v[0] / v[1] if v[1] else 0.0)
                                      for k, v in per_class.items()},
                        "seconds": round(elapsed, 1)})

        marker = ""
        if accuracy > best:
            best = accuracy
            torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES),
                        "imgsz": args.imgsz, "accuracy": accuracy,
                        "epoch": epoch}, args.out)
            marker = "  <- best, saved"
        torch.save({"state_dict": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best": best, "history": history}, last_path)
        print(f"epoch {epoch:>3}/{args.epochs}  loss "
              f"{running / max(1, len(train_loader)):.4f}  held-out accuracy "
              f"{accuracy:.4f}  {elapsed:.0f}s{marker}", flush=True)

    done = history[-1]["epoch"] if history else 0
    args.out.with_suffix(".history.json").write_text(json.dumps({
        "best_heldout_accuracy": best, "classes": list(CLASSES),
        "epochs": args.epochs, "device": device,
        "train_images": len(train_items), "heldout_images": len(test_items),
        "split_notes": {n: SOURCES[n][2] for n in CLASSES},
        "history": history,
    }, indent=2))
    if done < args.epochs:
        print(f"\nphase complete: {done} of {args.epochs} epochs. "
              f"Resume with --resume.")
    else:
        print(f"\nschedule complete. Best held-out accuracy: {best:.4f}")
        if history:
            print("per class:", {k: round(v, 3)
                                 for k, v in history[-1]["per_class"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
