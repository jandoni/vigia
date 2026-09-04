#!/usr/bin/env python3
"""Run a detector once over a dataset and cache every raw detection to JSONL.

Detection is deterministic and expensive; validation is cheap and has many
configurations we want to compare. Re-running the detector for each validator
setting wastes minutes and, worse, tempts you into evaluating fewer
configurations than the question deserves.

So: detect once, cache, then sweep validator settings against the cache. This
is also what makes the per-level ablation honest — every configuration sees
byte-identical detector output, so any difference in the result is caused by
the validator and nothing else.

Cache format: one JSON object per frame, in temporal order within each sequence.

Usage:
    python eval/cache_detections.py --data data/figlib_neg --out eval/cache/figlib_neg.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from vigia.detectors.base import Backend  # noqa: E402
from vigia.detectors.fire import FireDetector  # noqa: E402
from vigia.detectors.traffic import TrafficDetector  # noqa: E402
from vigia.types import Frame  # noqa: E402

DETECTORS = {"fire": FireDetector, "traffic": TrafficDetector}

logging.basicConfig(level=logging.ERROR)

FIGLIB_NAME = re.compile(r"(\d+)_([+-]\d+)\.jpg")


def frames_in(directory: Path) -> list[tuple[Path, float]]:
    """(path, offset_seconds) in temporal order for a FIgLib-style directory."""
    items: list[tuple[Path, float]] = []
    for path in directory.glob("*.jpg"):
        match = FIGLIB_NAME.match(path.name)
        if match:
            items.append((path, float(match.group(2))))
    return sorted(items, key=lambda item: item[1])



def cache_video(detector, args) -> int:
    """Cache detections from a video file, keeping every `stride`-th frame.

    Timestamps come from the video's real frame rate, so the validator's
    persistence and cooldown windows are expressed in true seconds regardless
    of how aggressively we subsample.
    """
    capture = cv2.VideoCapture(str(args.data))
    if not capture.isOpened():
        print(f"Could not open {args.data}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video   : {args.data.name}  {total} frames @ {fps:.0f} fps "
          f"({total / fps / 60:.1f} min)")
    print(f"stride  : {args.stride} (effective {fps / args.stride:.1f} fps)\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = detections_total = 0
    source_index = 0
    started = time.perf_counter()

    with args.out.open("w", encoding="utf-8") as sink:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if source_index % args.stride == 0:
                timestamp = source_index / fps
                detections = detector.detect(
                    Frame(image=image, index=kept, timestamp=timestamp,
                          camera_id=args.data.stem)
                )
                sink.write(json.dumps({
                    "sequence": args.data.stem,
                    "file": f"frame_{source_index:06d}",
                    "frame_index": kept,
                    "offset": round(timestamp, 3),
                    "source_frame": source_index,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "detections": [
                        {"box": [round(v, 2) for v in d.box.as_xyxy()],
                         "confidence": round(d.confidence, 5),
                         "label": d.label}
                        for d in detections
                    ],
                }) + "\n")
                kept += 1
                detections_total += len(detections)
                if kept % 50 == 0:
                    print(f"  {kept} frames cached "
                          f"({source_index}/{total} source)", flush=True)
            source_index += 1

    capture.release()
    elapsed = time.perf_counter() - started
    print(f"\n{kept} frames, {detections_total} detections in {elapsed:.0f}s "
          f"({detector.median_latency_ms:.0f} ms/frame)")
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="directory of sequence folders, or a video file")
    parser.add_argument("--hazard", default="fire", choices=sorted(DETECTORS))
    parser.add_argument("--model", type=Path, default=None,
                        help="override the detector's default model path")
    parser.add_argument("--stride", type=int, default=1,
                        help="video only: keep every Nth frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05,
                        help="cache at a LOW threshold so the validator's own "
                             "confidence level can be swept without re-detecting")
    parser.add_argument("--backend", default="reference", choices=["reference", "fast"])
    args = parser.parse_args()

    is_video = args.data.is_file() and args.data.suffix.lower() in {".mp4", ".avi", ".mov"}
    directories: list[Path] = []
    if not is_video:
        directories = sorted(d for d in args.data.iterdir() if d.is_dir())
        if not directories:
            print(f"No sequence directories in {args.data}", file=sys.stderr)
            return 1

    detector_class = DETECTORS[args.hazard]
    kwargs = {"conf_threshold": args.conf, "backend": Backend(args.backend)}
    if args.model:
        detector = detector_class(args.model, **kwargs)
    else:
        detector = detector_class(**kwargs)
    detector.warmup(rounds=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_frames = total_detections = 0
    started = time.perf_counter()

    if is_video:
        return cache_video(detector, args)

    with args.out.open("w", encoding="utf-8") as sink:
        for seq_index, directory in enumerate(directories, 1):
            items = frames_in(directory)
            for index, (path, offset) in enumerate(items):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                height, width = image.shape[:2]
                detections = detector.detect(
                    Frame(image=image, index=index, timestamp=offset,
                          camera_id=directory.name)
                )
                sink.write(json.dumps({
                    "sequence": directory.name,
                    "file": path.name,
                    "frame_index": index,
                    "offset": offset,
                    "width": width,
                    "height": height,
                    "detections": [
                        {"box": [round(v, 2) for v in d.box.as_xyxy()],
                         "confidence": round(d.confidence, 5)}
                        for d in detections
                    ],
                }) + "\n")
                total_frames += 1
                total_detections += len(detections)

            print(f"  [{seq_index:>3}/{len(directories)}] {directory.name:<42} "
                  f"{len(items):>3} frames", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\n{total_frames} frames, {total_detections} detections "
          f"(conf >= {args.conf}) in {elapsed:.0f}s "
          f"({detector.median_latency_ms:.0f} ms/frame)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
