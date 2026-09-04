#!/usr/bin/env python3
"""Flood evaluation: segmentation quality, and trend detection on real cameras.

Two parts, answering two different questions.

  A. SEGMENTATION — how well does the model find water?
     Measured on the ATLANTIS test split (1,296 images), which the model never
     saw during training. Reports water-class IoU, precision, recall and F1.

  B. TREND — does the rate-of-rise signal work on real footage?
     Measured on the LSU creek camera sequences from the V-FloodNet dataset:
     real fixed cameras with timestamps in the filenames, spanning minutes to
     hours. This is where the synthetic tests in tests/test_flood_level.py get
     confronted with real segmentation noise, changing light and moving water.

     Two trend mechanisms are compared on identical inputs: our sliding-window
     least-squares fit, and a Page-Hinkley change detector following Choi et al.
     (2026), who published this approach for exactly this problem. Neither is
     claimed as novel; the comparison is the useful output.

Usage:
    python eval/run_flood_eval.py
    python eval/run_flood_eval.py --limit 200 --json eval/results/flood.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vigia.detectors.flood import FloodDetector  # noqa: E402
from vigia.flood.level import PageHinkleyDetector, Trend  # noqa: E402
from vigia.types import Frame  # noqa: E402

logging.basicConfig(level=logging.ERROR)

ATLANTIS = REPO_ROOT / "data" / "atlantis_src" / "atlantis" / "atlantis"
LABELS_JSON = REPO_ROOT / "data" / "atlantis_src" / "atlantis" / "utils" / "labels_info.json"
CREEKS = REPO_ROOT / "data" / "lsu_creeks"

WATER_LABEL_NAMES = [
    "flood", "canal", "ditch", "fjord", "hot_spring", "lake", "puddle",
    "rapids", "reservoir", "river", "river_delta", "sea", "spillway",
    "swimming_pool", "waterfall", "wetland", "marsh",
]

FULL_TIMESTAMP = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})$")
CLOCK_TIMESTAMP = re.compile(r"^(\d{2})_(\d{2})$")


def water_values() -> list[int]:
    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    name_to_id = {entry["name"]: entry["id"] for entry in labels}
    return sorted(name_to_id[name] + 1 for name in WATER_LABEL_NAMES)


def parse_timestamp(stem: str) -> dt.datetime | None:
    match = FULL_TIMESTAMP.match(stem)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return dt.datetime(year, month, day, hour, minute, second)
    match = CLOCK_TIMESTAMP.match(stem)
    if match:
        hour, minute = map(int, match.groups())
        return dt.datetime(2020, 4, 23, hour, minute)
    return None


# --------------------------------------------------------------------------- #
# A. Segmentation quality
# --------------------------------------------------------------------------- #

def evaluate_segmentation(detector: FloodDetector, limit: int = 0) -> dict:
    values = np.array(water_values())
    images_root = ATLANTIS / "images" / "test"
    masks_root = ATLANTIS / "masks" / "test"

    pairs = []
    for image_path in sorted(images_root.rglob("*.jpg")):
        mask_path = masks_root / image_path.parent.name / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    if limit:
        # Stride rather than truncate: taking the first N would sample only the
        # alphabetically earliest label folders.
        stride = max(1, len(pairs) // limit)
        pairs = pairs[::stride][:limit]

    intersection = union = true_positive = false_positive = false_negative = 0
    correct = total = 0

    for index, (image_path, mask_path) in enumerate(pairs):
        image = cv2.imread(str(image_path))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask_raw is None:
            continue
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]

        truth = np.isin(mask_raw, values)
        result = detector.segment(
            Frame(image=image, index=index, timestamp=float(index),
                  camera_id=image_path.stem)
        )
        predicted = result.mask

        if predicted.shape != truth.shape:
            predicted = cv2.resize(predicted.astype(np.uint8),
                                   (truth.shape[1], truth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        intersection += int((predicted & truth).sum())
        union += int((predicted | truth).sum())
        true_positive += int((predicted & truth).sum())
        false_positive += int((predicted & ~truth).sum())
        false_negative += int((~predicted & truth).sum())
        correct += int((predicted == truth).sum())
        total += int(truth.size)

        if (index + 1) % 100 == 0:
            print(f"    {index + 1}/{len(pairs)}", flush=True)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "images": len(pairs),
        "water_iou": round(intersection / union, 4) if union else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pixel_accuracy": round(correct / total, 4) if total else 0.0,
        "median_latency_ms": round(detector.median_latency_ms, 1),
    }


# --------------------------------------------------------------------------- #
# B. Trend detection on real creek cameras
# --------------------------------------------------------------------------- #

def evaluate_trend(detector: FloodDetector) -> dict:
    sequences = sorted(d for d in CREEKS.iterdir() if d.is_dir()) if CREEKS.exists() else []
    results = {}

    for sequence in sequences:
        frames = []
        for path in sequence.glob("*.jpg"):
            when = parse_timestamp(path.stem)
            if when is not None:
                frames.append((when, path))
        if len(frames) < 4:
            continue
        frames.sort()
        origin = frames[0][0]

        detector.reset()
        page_hinkley = PageHinkleyDetector(delta=0.002, threshold=0.02, min_samples=4)

        series = []
        ph_alarm_seconds = None
        for index, (when, path) in enumerate(frames):
            image = cv2.imread(str(path))
            if image is None:
                continue
            seconds = (when - origin).total_seconds()
            observation = detector.observe(
                Frame(image=image, index=index, timestamp=seconds,
                      camera_id=sequence.name)
            )
            if page_hinkley.update(observation.reading.area_fraction, seconds):
                if ph_alarm_seconds is None:
                    ph_alarm_seconds = seconds
            series.append({
                "seconds": round(seconds, 1),
                "coverage": round(observation.coverage, 5),
                "trend": observation.rise.trend.value,
                "rate_per_min": round(observation.rise.area_rate_per_min, 6),
                "r_squared": round(observation.rise.r_squared, 3),
                "ph_statistic": round(page_hinkley.statistic, 5),
            })

        coverages = [s["coverage"] for s in series]
        trends = [s["trend"] for s in series]
        # First frame at which the least-squares fit called a confident trend.
        ls_first = next(
            (s["seconds"] for s in series if s["trend"] in ("rising", "falling")),
            None,
        )

        results[sequence.name] = {
            "frames": len(series),
            "duration_minutes": round(series[-1]["seconds"] / 60, 1),
            "coverage_start": round(coverages[0], 4),
            "coverage_end": round(coverages[-1], 4),
            "coverage_change": round(coverages[-1] - coverages[0], 4),
            "trend_counts": {t: trends.count(t) for t in set(trends)},
            "least_squares_first_call_seconds": ls_first,
            "page_hinkley_alarm_seconds": ph_alarm_seconds,
            "series": series,
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path,
                        default=REPO_ROOT / "models/flood/atlantis_water_deeplabv3.onnx")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap ATLANTIS test images (0 = all 1,296)")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"No model at {args.model}.\n"
              f"Train:  .venv-export/bin/python scripts/train_flood.py\n"
              f"Export: .venv-export/bin/python scripts/export_flood_onnx.py",
              file=sys.stderr)
        return 1

    detector = FloodDetector(args.model)
    detector.warmup(rounds=2)
    print(f"model    : {args.model.name}")
    print(f"backend  : {detector.backend.value} "
          f"({detector.session.get_providers()[0]})\n")

    payload: dict = {"model": args.model.name}

    if not args.skip_segmentation:
        print("A. SEGMENTATION — ATLANTIS test split (unseen during training)")
        segmentation = evaluate_segmentation(detector, args.limit)
        payload["segmentation"] = segmentation
        print(f"    images         {segmentation['images']}")
        print(f"    water IoU      {segmentation['water_iou']:.4f}")
        print(f"    precision      {segmentation['precision']:.4f}")
        print(f"    recall         {segmentation['recall']:.4f}")
        print(f"    F1             {segmentation['f1']:.4f}")
        print(f"    pixel accuracy {segmentation['pixel_accuracy']:.4f}")
        print(f"    latency        {segmentation['median_latency_ms']:.1f} ms\n")

    print("B. TREND — LSU creek cameras (real fixed cameras, V-FloodNet dataset)")
    trend = evaluate_trend(detector)
    payload["trend"] = trend

    if not trend:
        print("    no creek sequences found; skipped")
    else:
        header = (f"    {'sequence':<26} {'frames':>6} {'mins':>6} "
                  f"{'cover start->end':>18} {'LS call':>9} {'PH alarm':>9}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for name, entry in trend.items():
            ls = entry["least_squares_first_call_seconds"]
            ph = entry["page_hinkley_alarm_seconds"]
            print(f"    {name:<26} {entry['frames']:>6} "
                  f"{entry['duration_minutes']:>6.1f} "
                  f"{entry['coverage_start']:>8.3f} -> {entry['coverage_end']:<6.3f} "
                  f"{('%.0fs' % ls) if ls is not None else '-':>9} "
                  f"{('%.0fs' % ph) if ph is not None else '-':>9}")
        print("\n    LS call  = first frame the least-squares fit called a trend")
        print("    PH alarm = first frame Page-Hinkley flagged a change")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
