#!/usr/bin/env python3
"""Measure the people-in-water detector ourselves, on the swimmer class alone.

The author reports mAP@50 of 0.793 across all five SeaDronesSee classes. That
number is carried by boats. VIGÍA only cares about `swimmer`, so we measure
that class on its own rather than inheriting a headline that flatters it.

Reported per-class AP by the author: boat 0.711, jetski 0.593, buoy 0.489,
swimmer 0.282, life_saving_appliances 0.188.

Two levels, as elsewhere in VIGÍA:
  box level   — did we localise the people?
  image level — did we raise an alarm on this frame at all, and should we have?

An IoU match threshold of 0.3 is used rather than the conventional 0.5. A
swimmer at these altitudes is a handful of pixels; at 0.5 the metric measures
annotation convention more than detection ability. The threshold is stated
rather than buried.

Usage:
    python eval/run_person_in_water_eval.py
    python eval/run_person_in_water_eval.py --json eval/results/person_in_water.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from eval.metrics import CountResult, match_boxes  # noqa: E402
from vigia.detectors.person_in_water import PersonInWaterDetector  # noqa: E402
from vigia.types import Box, Frame  # noqa: E402

logging.basicConfig(level=logging.ERROR)

DEFAULT_DATA = REPO_ROOT / "data" / "seadronessee" / "val"
SWIMMER_CLASS_ID = "0"          # per the dataset's data.yaml


def swimmer_boxes(label_text: str, width: int, height: int) -> list[Box]:
    """Ground-truth swimmer boxes only, from YOLO-format labels."""
    boxes: list[Box] = []
    for line in label_text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != SWIMMER_CLASS_ID:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        bw, bh = w * width, h * height
        px, py = cx * width, cy * height
        boxes.append(Box(px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2))
    return boxes


def evaluate(detector, paths: list[Path], iou_threshold: float) -> dict:
    box_level, image_level = CountResult(), CountResult()
    gt_total = pred_total = 0

    for index, image_path in enumerate(paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        label_path = image_path.with_suffix(".txt")
        truth = swimmer_boxes(
            label_path.read_text() if label_path.exists() else "", width, height
        )

        detections = detector.detect(
            Frame(image=image, index=index, timestamp=float(index),
                  camera_id=image_path.stem)
        )
        detections = [d for d in detections if d.label == "swimmer"]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        predicted = [d.box for d in detections]

        gt_total += len(truth)
        pred_total += len(predicted)

        tp, fp, fn = match_boxes(predicted, truth, iou_threshold)
        box_level.true_positives += tp
        box_level.false_positives += fp
        box_level.false_negatives += fn

        alarmed, should = bool(predicted), bool(truth)
        if should and alarmed:
            image_level.true_positives += 1
        elif should:
            image_level.false_negatives += 1
        elif alarmed:
            image_level.false_positives += 1
        else:
            image_level.true_negatives += 1

    return {
        "images": len(paths),
        "ground_truth_swimmers": gt_total,
        "predicted_swimmers": pred_total,
        "box_level": box_level.to_dict(),
        "image_level": image_level.to_dict(),
        "median_latency_ms": round(detector.median_latency_ms, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--conf", type=float, nargs="+",
                        default=[0.10, 0.20, 0.30, 0.50, 0.70])
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.data.glob("*.jpg"))
    if not paths:
        print(f"No images in {args.data}", file=sys.stderr)
        return 1

    with_swimmers = sum(
        1 for p in paths
        if p.with_suffix(".txt").exists()
        and any(line.startswith("0 ")
                for line in p.with_suffix(".txt").read_text().splitlines())
    )

    detector = PersonInWaterDetector()
    detector.warmup(rounds=2)

    print(f"dataset  : {args.data}")
    print(f"images   : {len(paths)}  with swimmers: {with_swimmers}  "
          f"without: {len(paths) - with_swimmers}")
    print(f"model    : {detector.model_path.name}")
    print(f"match    : IoU >= {args.iou} (see module docstring)\n")

    header = (f"{'conf':>6} | {'BOX  P':>7} {'R':>6} {'F1':>6} {'F2':>6} | "
              f"{'IMG  P':>7} {'R':>6} {'FAR':>6} | {'pred':>6} {'ms':>6}")
    print(header)
    print("-" * len(header))

    results = {}
    for conf in args.conf:
        detector.conf_threshold = conf
        outcome = evaluate(detector, paths, args.iou)
        results[f"conf_{conf}"] = outcome
        box, img = outcome["box_level"], outcome["image_level"]
        print(f"{conf:>6.2f} | {box['precision']:>7.3f} {box['recall']:>6.3f} "
              f"{box['f1']:>6.3f} {box['f2']:>6.3f} | "
              f"{img['precision']:>7.3f} {img['recall']:>6.3f} "
              f"{img['false_alarm_rate']:>6.3f} | "
              f"{outcome['predicted_swimmers']:>6} "
              f"{outcome['median_latency_ms']:>6.1f}")

    first = results[f"conf_{args.conf[0]}"]
    print(f"\nground-truth swimmer instances: {first['ground_truth_swimmers']}")
    print("Box recall is the number that matters: it is the fraction of people "
          "in the water\nthat the detector found at all. Everything downstream "
          "can only filter, never recover.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "dataset": str(args.data),
            "model": detector.model_path.name,
            "iou_match_threshold": args.iou,
            "images": len(paths),
            "images_with_swimmers": with_swimmers,
            "reported_by_author_all_classes": {
                "mAP_50": 0.7931,
                "per_class_AP_swimmer": 0.282,
                "note": "Author's headline mAP is across five classes and is "
                        "carried by boats. Our figures below are swimmer-only.",
            },
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
