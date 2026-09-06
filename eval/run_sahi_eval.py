#!/usr/bin/env python3
"""Sliced inference (SAHI) for small objects: measured, not adopted on faith.

The people-in-water detector's graph takes a fixed 512 px input, so a
1280x720 frame reaches the model at less than half its native resolution —
and a swimmer at altitude is a handful of pixels to begin with. Slicing
Aided Hyper Inference (Akyon et al., 2022) is the standard remedy: run the
detector on overlapping tiles at native resolution, map detections back,
merge duplicates with NMS. It is inference-side only, so the project's
thesis — published weights, no retraining — survives intact.

This script measures what slicing actually buys on the SeaDronesSee
validation split, against the exact baseline already published
(eval/run_person_in_water_eval.py): same images, same labels, same matching,
same operating point. Both the gain and the cost (latency multiplier) are
reported, and the decision to adopt or not follows the numbers.

Usage:
    python eval/run_sahi_eval.py --json eval/results/sahi.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from eval.metrics import CountResult, match_boxes  # noqa: E402
from eval.run_person_in_water_eval import swimmer_boxes  # noqa: E402
from vigia.detectors.person_in_water import PersonInWaterDetector  # noqa: E402
from vigia.types import Box, Detection, Frame  # noqa: E402

logging.basicConfig(level=logging.ERROR)

DATA = REPO_ROOT / "data" / "seadronessee" / "val"
TILE = 512          # the model's own input size: a tile arrives unshrunk
OVERLAP = 0.2       # standard SAHI default; a swimmer is never split by both
                    # neighbours of an overlap this wide


def tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Overlapping tile origins covering the frame, last tile edge-aligned."""
    stride = int(TILE * (1 - OVERLAP))
    xs = list(range(0, max(width - TILE, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - TILE, 0) + 1, stride)) or [0]
    if xs[-1] != max(width - TILE, 0):
        xs.append(max(width - TILE, 0))
    if ys[-1] != max(height - TILE, 0):
        ys.append(max(height - TILE, 0))
    return [(x, y, min(x + TILE, width), min(y + TILE, height))
            for y in ys for x in xs]


def nms(detections: list[Detection], iou: float = 0.5) -> list[Detection]:
    """Greedy NMS over merged tile + full-frame detections."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.confidence,
                            reverse=True):
        if all(detection.box.iou(other.box) < iou for other in kept):
            kept.append(detection)
    return kept


def sliced_detect(detector, image, index: int) -> list[Detection]:
    height, width = image.shape[:2]
    merged: list[Detection] = []
    # Full frame first: large objects and global context.
    merged.extend(detector.detect(
        Frame(image=image, index=index, timestamp=0.0, camera_id="sahi")))
    for x1, y1, x2, y2 in tiles(width, height):
        crop = image[y1:y2, x1:x2]
        for detection in detector.detect(
                Frame(image=crop, index=index, timestamp=0.0,
                      camera_id="sahi")):
            bx1, by1, bx2, by2 = detection.box.as_xyxy()
            detection.box = Box(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
            merged.append(detection)
    return nms(merged)


def collect(detector, mode: str) -> list[dict]:
    """One detection pass per mode; thresholds are applied afterwards.

    Re-running the detector once per threshold doubled a ten-minute job for
    no reason: the detections do not change, only the floor does.
    """
    rows = []
    images = sorted(DATA.glob("*.jpg"))
    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label = path.with_suffix(".txt")
        truth = swimmer_boxes(label.read_text() if label.exists() else "",
                              width, height)
        started = time.perf_counter()
        if mode == "sliced":
            detections = sliced_detect(detector, image, index)
        else:
            detections = detector.detect(
                Frame(image=image, index=index, timestamp=0.0, camera_id="b"))
        elapsed = (time.perf_counter() - started) * 1000
        rows.append({
            "truth": truth,
            "detections": [(d.box, d.confidence)
                           for d in detections if d.label == "swimmer"],
            "ms": elapsed,
        })
        if (index + 1) % 50 == 0:
            print(f"  {mode}: {index + 1}/{len(images)}", file=sys.stderr)
    return rows


def evaluate(rows: list[dict], floor: float) -> dict:
    box_level, image_level = CountResult(), CountResult()
    for row in rows:
        predicted = [box for box, confidence in row["detections"]
                     if confidence >= floor]
        tp, fp, fn = match_boxes(predicted, row["truth"], 0.3)
        box_level.true_positives += tp
        box_level.false_positives += fp
        box_level.false_negatives += fn
        alarmed, should = bool(predicted), bool(row["truth"])
        if should and alarmed:
            image_level.true_positives += 1
        elif should:
            image_level.false_negatives += 1
        elif alarmed:
            image_level.false_positives += 1
        else:
            image_level.true_negatives += 1
    latencies = sorted(row["ms"] for row in rows)
    return {
        "box_level": box_level.to_dict(),
        "image_level": image_level.to_dict(),
        "median_latency_ms": round(latencies[len(latencies) // 2], 1),
        "images": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    detector = PersonInWaterDetector()
    results = {}
    print(f"{'mode':<10} {'floor':>5} {'prec':>7} {'recall':>7} {'f1':>7} "
          f"{'img FAR':>8} {'ms':>7}")
    print("-" * 58)
    for mode in ("baseline", "sliced"):
        rows = collect(detector, mode)
        for floor in (0.3, 0.5):
            r = evaluate(rows, floor)
            results[f"{mode}_conf_{floor}"] = r
            b, i = r["box_level"], r["image_level"]
            print(f"{mode:<10} {floor:>5} {b['precision']:>7.4f} "
                  f"{b['recall']:>7.4f} {b['f1']:>7.4f} "
                  f"{i['false_alarm_rate']:>8.4f} "
                  f"{r['median_latency_ms']:>7.1f}")

    if args.json:
        args.json.write_text(json.dumps({
            "question": ("does sliced inference (SAHI, Akyon et al. 2022) "
                         "improve small-object recall enough to pay its "
                         "latency bill?"),
            "method": (f"tiles of {TILE}px at {OVERLAP:.0%} overlap plus one "
                       "full-frame pass, greedy NMS at IoU 0.5; identical "
                       "images, labels and matching as "
                       "eval/run_person_in_water_eval.py"),
            "dataset": "data/seadronessee/val, swimmer class only",
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
