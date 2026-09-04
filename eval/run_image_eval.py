#!/usr/bin/env python3
"""Evaluate a detector on a directory of images with YOLO-format labels.

This is the still-image harness. It establishes the raw per-frame baseline —
the number the temporal validator then has to improve on. Video/event-level
evaluation (where the validator actually earns its keep) is a separate harness.

Reports both box level and image level, and sweeps confidence thresholds so we
can see the operating curve rather than a single point.

Usage:
    python eval/run_image_eval.py --data data/samples/pyro_sdis_val
    python eval/run_image_eval.py --data <dir> --conf 0.1 0.2 0.3 --json out.json
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

from eval.metrics import CountResult, match_boxes, parse_yolo_label  # noqa: E402
from vigia.detectors.fire import FireDetector  # noqa: E402
from vigia.types import Frame  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def evaluate(detector, image_paths: list[Path], iou_threshold: float) -> dict:
    box_level = CountResult()
    image_level = CountResult()
    latencies: list[float] = []

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        label_path = image_path.with_suffix(".txt")
        label_text = label_path.read_text() if label_path.exists() else ""
        truth = parse_yolo_label(label_text, width, height)

        started = time.perf_counter()
        detections = detector.detect(
            Frame(image=image, index=index, timestamp=float(index), camera_id=image_path.stem)
        )
        latencies.append((time.perf_counter() - started) * 1000)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        predicted = [d.box for d in detections]

        tp, fp, fn = match_boxes(predicted, truth, iou_threshold)
        box_level.true_positives += tp
        box_level.false_positives += fp
        box_level.false_negatives += fn

        # Image level: did we alarm, and should we have?
        alarmed, should_alarm = bool(predicted), bool(truth)
        if should_alarm and alarmed:
            image_level.true_positives += 1
        elif should_alarm and not alarmed:
            image_level.false_negatives += 1
        elif not should_alarm and alarmed:
            image_level.false_positives += 1
        else:
            image_level.true_negatives += 1

    latencies.sort()
    median = latencies[len(latencies) // 2] if latencies else 0.0

    return {
        "box_level": box_level.to_dict(),
        "image_level": image_level.to_dict(),
        "median_latency_ms": round(median, 2),
        "images": len(image_paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path,
                        help="directory of .jpg images with matching .txt YOLO labels")
    parser.add_argument("--conf", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
                        help="confidence thresholds to sweep")
    parser.add_argument("--iou", type=float, default=0.3,
                        help="IoU threshold for matching a prediction to ground truth")
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    args = parser.parse_args()

    image_paths = sorted(args.data.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg files in {args.data}", file=sys.stderr)
        return 1

    labelled = sum(1 for p in image_paths if p.with_suffix(".txt").exists())
    positives = sum(
        1 for p in image_paths
        if p.with_suffix(".txt").exists() and p.with_suffix(".txt").read_text().strip()
    )

    print(f"dataset : {args.data}")
    print(f"images  : {len(image_paths)}  labelled: {labelled}  "
          f"positive: {positives}  negative: {labelled - positives}")
    print(f"match   : IoU >= {args.iou}\n")

    detector = FireDetector()
    detector.warmup(rounds=2)
    print(f"model   : {detector.model_path.name}")
    print(f"provider: {detector.session.get_providers()[0]}")
    print(f"imgsz   : {detector.imgsz}\n")

    header = (f"{'conf':>6} | {'BOX  P':>7} {'R':>6} {'F1':>6} {'F2':>6} | "
              f"{'IMG  P':>7} {'R':>6} {'F2':>6} {'FAR':>6} | {'ms':>6}")
    print(header)
    print("-" * len(header))

    results = {}
    for conf in args.conf:
        detector.conf_threshold = conf
        outcome = evaluate(detector, image_paths, args.iou)
        results[f"conf_{conf}"] = outcome
        box, img = outcome["box_level"], outcome["image_level"]
        print(f"{conf:>6.2f} | {box['precision']:>7.3f} {box['recall']:>6.3f} "
              f"{box['f1']:>6.3f} {box['f2']:>6.3f} | "
              f"{img['precision']:>7.3f} {img['recall']:>6.3f} {img['f2']:>6.3f} "
              f"{img['false_alarm_rate']:>6.3f} | {outcome['median_latency_ms']:>6.1f}")

    print("\nFAR = false-alarm rate: fraction of no-smoke images that raised an alarm.")
    print("This is the number the temporal validator must reduce.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(args.data),
            "model": detector.model_path.name,
            "provider": detector.session.get_providers()[0],
            "imgsz": detector.imgsz,
            "iou_match_threshold": args.iou,
            "images": len(image_paths),
            "positives": positives,
            "negatives": labelled - positives,
            "results": results,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
