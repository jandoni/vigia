#!/usr/bin/env python3
"""R1 + per-level ablation: what does each validator level actually buy?

Replays cached detector output through validator configurations. Because every
configuration sees byte-identical detections, any difference in the result is
attributable to the validator alone.

Two datasets, measuring two different things:

  negatives — FIgLib pre-ignition frames from many cameras, dates and weather.
              No fire is present, so every alert is a false alarm. This is
              where the validator has to earn its keep.
  fires     — FIgLib ignition sequences. Measures what the validator costs:
              fires still found, and how many minutes later.

Reporting both together is the point. Any filter can drive false alarms to zero
by refusing to alert; the question is what it costs in detection.

Usage:
    python eval/run_ablation.py
    python eval/run_ablation.py --json eval/results/r1_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vigia.types import Box, Detection, Hazard  # noqa: E402
from vigia.validator.cascade import TemporalValidator, ValidatorConfig  # noqa: E402

CACHE_DIR = REPO_ROOT / "eval" / "cache"


def load_cache(path: Path) -> dict[str, list[dict]]:
    """Group cached frames by sequence, preserving temporal order."""
    sequences: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sequences.setdefault(row["sequence"], []).append(row)
    for frames in sequences.values():
        frames.sort(key=lambda r: r["offset"])
    return sequences


def to_detections(row: dict) -> list[Detection]:
    return [
        Detection(
            hazard=Hazard.FIRE,
            box=Box(*d["box"]),
            confidence=d["confidence"],
            frame_index=row["frame_index"],
            timestamp=row["offset"],
            camera_id=row["sequence"],
        )
        for d in row["detections"]
    ]


def replay(sequences: dict[str, list[dict]], config: ValidatorConfig | None) -> dict:
    """Play every sequence through a validator config (None = raw detector)."""
    frames = alarmed_pre = alarmed_post = 0
    pre_frames = post_frames = 0
    alerts = 0
    first_alert: dict[str, float] = {}

    for name, rows in sequences.items():
        validator = (
            TemporalValidator(Hazard.FIRE, config) if config is not None else None
        )
        for row in rows:
            detections = to_detections(row)

            if validator is None:
                # Raw path still honours the confidence floor, so the comparison
                # against a validator with the same floor is like-for-like.
                floor = config.min_confidence if config else 0.20
                kept = [d for d in detections if d.confidence >= floor]
                hit, count = bool(kept), len(kept)
            else:
                events = validator.process(
                    detections, row["frame_index"], row["offset"], None, name
                )
                hit, count = bool(events), len(events)

            frames += 1
            alerts += count
            if row["offset"] < 0:
                pre_frames += 1
                alarmed_pre += int(hit)
            else:
                post_frames += 1
                alarmed_post += int(hit)

            if hit and row["offset"] > 0 and name not in first_alert:
                first_alert[name] = row["offset"] / 60.0

    latencies = sorted(first_alert.values())
    return {
        "frames": frames,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "false_alarm_rate": round(alarmed_pre / pre_frames, 4) if pre_frames else None,
        "detection_rate": round(alarmed_post / post_frames, 4) if post_frames else None,
        "sequences_detected": len(first_alert),
        "total_sequences": len(sequences),
        "alerts": alerts,
        "median_minutes_to_detect": (
            round(latencies[len(latencies) // 2], 1) if latencies else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--negatives", type=Path, default=CACHE_DIR / "figlib_neg.jsonl")
    parser.add_argument("--fires", type=Path, default=CACHE_DIR / "figlib_fires.jsonl")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    for path in (args.negatives, args.fires):
        if not path.exists():
            print(f"Missing cache {path}. Run eval/cache_detections.py first.",
                  file=sys.stderr)
            return 1

    negatives = load_cache(args.negatives)
    fires = load_cache(args.fires)

    print(f"negatives : {len(negatives)} sequences, "
          f"{sum(len(v) for v in negatives.values())} frames (no fire present)")
    print(f"fires     : {len(fires)} sequences, "
          f"{sum(len(v) for v in fires.values())} frames")
    print(f"conf floor: {args.conf}\n")

    base = ValidatorConfig(
        min_confidence=args.conf,
        min_frames=3,
        enable_cooldown=False,   # confounds per-frame recall, measured separately
    )

    configurations: list[tuple[str, ValidatorConfig | None]] = [
        ("raw detector (no validator)", None),
    ]
    for min_frames in (2, 3, 4, 5):
        configurations.append(
            (f"full cascade, persistence={min_frames}",
             replace(base, min_frames=min_frames))
        )
    # Per-level ablation at the reference setting.
    configurations += [
        ("  minus confidence", replace(base, enable_confidence=False)),
        ("  minus persistence", replace(base, enable_persistence=False)),
    ]
    # ByteTrack-style second stage: below-floor detections may keep a track
    # alive but never alarm. Measured before being defaulted, like every
    # other change to the cascade.
    configurations.append(
        ("full cascade, persistence=3 + LC assoc",
         replace(base, min_frames=3, low_confidence_association=True)))

    header = (f"{'configuration':<32} {'FAR':>7} {'reduction':>10} | "
              f"{'fires':>7} {'recall':>7} {'min-to-detect':>14}")
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    baseline_far: float | None = None

    for label, config in configurations:
        negative_result = replay(negatives, config)
        fire_result = replay(fires, config)

        far = negative_result["false_alarm_rate"]
        if baseline_far is None:
            baseline_far = far
        reduction = (
            f"{100 * (1 - far / baseline_far):.0f}%"
            if baseline_far else "-"
        )

        results[label.strip()] = {"negatives": negative_result, "fires": fire_result}

        print(f"{label:<32} {far:>7.3f} {reduction:>10} | "
              f"{fire_result['sequences_detected']:>3}/{fire_result['total_sequences']:<3} "
              f"{fire_result['detection_rate']:>7.3f} "
              f"{str(fire_result['median_minutes_to_detect']):>14}")

    print("\nFAR       = fraction of no-fire frames that raised an alert")
    print("reduction = relative to the raw detector at the same confidence floor")
    print("fires     = ignition sequences detected at all")
    print("\nThe 'minus X' rows disable one level of the full cascade at "
          "persistence=3,\nso the gap between each and the full cascade is what "
          "that level contributes.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "conf_threshold": args.conf,
            "negative_sequences": len(negatives),
            "fire_sequences": len(fires),
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
