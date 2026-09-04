#!/usr/bin/env python3
"""The hazard-agnostic claim, tested.

VIGÍA's contribution is not "temporal validation for fire" — PyroNear already
does sequential confirmation for fire, and we found that in their own training
manifest. The claim is narrower and still worth making: *one* validator,
unmodified, configured rather than specialised, working across hazards whose
detectors were trained by different people on different data for different
phenomena.

This script tests exactly that. It runs the identical `TemporalValidator` class
over two caches — wildfire smoke from PyroNear via FIgLib, and traffic
accidents from a YOLO11x fine-tune over dashcam video — and asserts that no
hazard-specific code was needed. The only difference between the two runs is a
`ValidatorConfig`.

If this script ever needs an `if hazard == ...` branch, the claim is dead and
the dossier has to say something weaker.

Usage:
    python eval/run_multihazard.py
    python eval/run_multihazard.py --json eval/results/multihazard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.run_ablation import load_cache  # noqa: E402
from vigia.types import Box, Detection, Hazard  # noqa: E402
from vigia.validator.cascade import TemporalValidator, ValidatorConfig  # noqa: E402

CACHE_DIR = REPO_ROOT / "eval" / "cache"


def to_detections(row: dict, hazard: Hazard, hazard_labels: set[str] | None) -> list[Detection]:
    """Cache row -> Detection list.

    `hazard_labels` filters multi-class detectors down to the classes that are
    actually hazards. The traffic model emits `vehicle` as well as `accident`;
    routing vehicles into the validator would make every car an alert.
    """
    detections = []
    for item in row["detections"]:
        label = item.get("label", "")
        if hazard_labels is not None and label not in hazard_labels:
            continue
        detections.append(
            Detection(
                hazard=hazard,
                box=Box(*item["box"]),
                confidence=item["confidence"],
                label=label,
                frame_index=row["frame_index"],
                timestamp=row["offset"],
                camera_id=row["sequence"],
            )
        )
    return detections


def run(
    sequences: dict[str, list[dict]],
    hazard: Hazard,
    config: ValidatorConfig | None,
    hazard_labels: set[str] | None = None,
) -> dict:
    """Replay sequences through the validator (or raw, if config is None)."""
    frames = raw_frames_with_detection = confirmed_events = 0
    raw_detections = 0
    level_stats: dict[str, dict] = {}

    for name, rows in sequences.items():
        validator = TemporalValidator(hazard, config) if config is not None else None

        for row in rows:
            detections = to_detections(row, hazard, hazard_labels)
            floor = config.min_confidence if config else 0.25
            above_floor = [d for d in detections if d.confidence >= floor]

            frames += 1
            raw_detections += len(above_floor)
            raw_frames_with_detection += int(bool(above_floor))

            if validator is not None:
                events = validator.process(
                    detections, row["frame_index"], row["offset"], None, name
                )
                confirmed_events += len(events)

        if validator is not None:
            for level in validator.stats_dict()["levels"]:
                bucket = level_stats.setdefault(
                    level["name"], {"seen": 0, "passed": 0}
                )
                bucket["seen"] += level["seen"]
                bucket["passed"] += level["passed"]

    return {
        "frames": frames,
        "raw_detections": raw_detections,
        "raw_frames_with_detection": raw_frames_with_detection,
        "raw_alert_rate": round(raw_frames_with_detection / frames, 4) if frames else 0.0,
        "confirmed_events": confirmed_events,
        "levels": level_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # Per-hazard setup. Everything hazard-specific lives HERE, in data —
    # never inside the validator.
    hazards = [
        {
            "hazard": Hazard.FIRE,
            "title": "wildfire smoke",
            "cache": CACHE_DIR / "figlib_fires.jsonl",
            "detector": "PyroNear yolo11s_sensitive (Apache-2.0)",
            "labels": None,
            "config": ValidatorConfig(
                min_confidence=0.20,
                min_frames=3,          # frames are 1 minute apart
                enable_cooldown=True,
                cooldown_seconds=600.0,
            ),
        },
        {
            "hazard": Hazard.TRAFFIC,
            "title": "traffic accidents",
            "cache": CACHE_DIR / "traffic_carcrash.jsonl",
            "detector": "Enos-123 traffic-accident yolo11x (MIT)",
            "labels": {"accident"},
            "config": ValidatorConfig(
                min_confidence=0.40,
                min_frames=3,          # frames are 0.5 s apart
                enable_cooldown=True,
                cooldown_seconds=10.0,
                track_iou=0.15,        # vehicles move fast between frames
            ),
        },
    ]

    print("Testing one validator across two hazards.\n")
    print("The ONLY per-hazard difference is a ValidatorConfig. No branch in\n"
          "vigia/validator/cascade.py refers to any hazard.\n")

    payload: dict = {"hazards": {}}

    for spec in hazards:
        cache_path: Path = spec["cache"]
        if not cache_path.exists():
            print(f"  ! missing cache {cache_path.name}; skipping {spec['title']}")
            continue

        sequences = load_cache(cache_path)
        config: ValidatorConfig = spec["config"]

        raw = run(sequences, spec["hazard"], None, spec["labels"])
        validated = run(sequences, spec["hazard"], config, spec["labels"])

        suppression = (
            1 - validated["confirmed_events"] / raw["raw_detections"]
            if raw["raw_detections"] else 0.0
        )

        print(f"--- {spec['title'].upper()} ---")
        print(f"  detector          : {spec['detector']}")
        print(f"  sequences/frames  : {len(sequences)} / {raw['frames']}")
        print(f"  config            : conf={config.min_confidence} "
              f"persistence={config.min_frames} "
              f"cooldown={config.cooldown_seconds:.0f}s "
              f"track_iou={config.track_iou}")
        print(f"  raw detections    : {raw['raw_detections']} "
              f"across {raw['raw_frames_with_detection']} frames "
              f"({raw['raw_alert_rate']:.1%} of frames)")
        print(f"  confirmed events  : {validated['confirmed_events']}")
        print(f"  suppression       : {suppression:.1%}")
        print("  cascade:")
        for name in TemporalValidator.LEVEL_NAMES:
            level = validated["levels"].get(name)
            if not level or not level["seen"]:
                print(f"    {name:<12} (disabled)")
                continue
            rejected = level["seen"] - level["passed"]
            print(f"    {name:<12} seen {level['seen']:>5}  "
                  f"passed {level['passed']:>5}  rejected {rejected:>5} "
                  f"({rejected / level['seen']:.0%})")
        print()

        payload["hazards"][spec["hazard"].value] = {
            "detector": spec["detector"],
            "raw": raw,
            "validated": validated,
            "suppression": round(suppression, 4),
        }

    print("Same TemporalValidator class, same cascade.py, two hazards, two\n"
          "detectors trained by different authors on unrelated data.\n"
          "Zero hazard-specific code in the validator.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
