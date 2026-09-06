#!/usr/bin/env python3
"""The experiment the threshold-matched control demands: does a tuned floor
transfer across cameras, or does the cascade generalise better?

The threshold-matched control (eval/run_threshold_matched.py) showed that on
the FIgLib negatives an ORACLE confidence floor — swept on the very negatives
it is scored on — matches the cascade's false-alarm rate at higher recall. A
reviewer's correct objection is that this oracle cannot exist in deployment: a
floor must be chosen on some cameras and then run on OTHERS it will never have
seen. The cascade, by contrast, runs the detector authors' published default
floor unchanged and relies on persistence, which is camera-independent.

This script measures the transfer gap directly. Cameras are split into two
disjoint folds. On fold A the floor is tuned to hit a target false-alarm rate;
that FIXED floor is then applied to fold B, and vice versa. The tuned-threshold
transfer is compared against the cascade (persistence 3, published floor 0.20,
untuned) on the same held-out fold. Both directions are reported, plus the
oracle (tune and test on the same fold) so the transfer penalty is visible.

The honest outcomes are both worth publishing:
  * the transferred threshold's false-alarm rate drifts above the cascade's on
    held-out cameras -> the cascade's untuned stability is a real advantage;
  * it does not drift -> the false-alarm advantage narrows to deduplication
    and hazard-agnostic transfer, and the paper must say so.

Usage:
    python eval/run_threshold_transfer.py --json eval/results/threshold_transfer.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from run_ablation import load_cache, replay  # noqa: E402
from vigia.validator.cascade import ValidatorConfig  # noqa: E402

CACHE = REPO_ROOT / "eval" / "cache"
TARGET_FAR = 0.010          # the cascade's operating point (persistence 3)


def camera_of(sequence: str) -> str:
    match = re.match(r"\d+_FIRE_(.+)", sequence)
    return match.group(1) if match else sequence


def split_by_camera(sequences: dict, fold: set) -> dict:
    return {name: rows for name, rows in sequences.items()
            if camera_of(name) in fold}


def raw_far(negatives: dict, floor: float) -> float:
    config = ValidatorConfig(min_confidence=floor, enable_persistence=False,
                             enable_cooldown=False)
    return replay(negatives, config)["false_alarm_rate"]


def raw_recall(fires: dict, floor: float) -> float:
    config = ValidatorConfig(min_confidence=floor, enable_persistence=False,
                             enable_cooldown=False)
    return replay(fires, config)["detection_rate"]


def tune_floor(negatives: dict, target: float) -> float:
    """Smallest floor whose FAR on THIS fold is at or below the target."""
    for step in range(20, 96):
        floor = step / 100.0
        if raw_far(negatives, floor) <= target:
            return floor
    return 0.95


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    negatives = load_cache(CACHE / "figlib_neg.jsonl")
    fires = load_cache(CACHE / "figlib_fires.jsonl")

    # Deterministic camera split: sort, alternate into two folds. Alternating
    # rather than random keeps the two folds balanced in size and avoids a
    # seed that a reader would have to trust.
    cameras = sorted({camera_of(s) for s in negatives})
    fold_a = set(cameras[0::2])
    fold_b = set(cameras[1::2])
    print(f"{len(cameras)} cameras -> fold A {len(fold_a)}, fold B {len(fold_b)}\n")

    cascade = ValidatorConfig(min_confidence=0.20, min_frames=3,
                              enable_cooldown=False)

    directions = []
    for train, test, name in ((fold_a, fold_b, "A->B"), (fold_b, fold_a, "B->A")):
        neg_train = split_by_camera(negatives, train)
        neg_test = split_by_camera(negatives, test)
        fires_test = split_by_camera(fires, test)

        tuned = tune_floor(neg_train, TARGET_FAR)
        # oracle: the best this floor family could do if it HAD seen the test
        oracle = tune_floor(neg_test, TARGET_FAR)

        row = {
            "direction": name,
            "tuned_floor": tuned,
            "far_on_train": round(raw_far(neg_train, tuned), 4),
            "threshold_transfer_far": round(raw_far(neg_test, tuned), 4),
            "threshold_transfer_recall": round(raw_recall(fires_test, tuned), 4)
                                          if fires_test else None,
            "oracle_floor_on_test": oracle,
            "oracle_far": round(raw_far(neg_test, oracle), 4),
            "cascade_far": round(replay(neg_test, cascade)["false_alarm_rate"], 4),
            "cascade_recall": round(replay(fires_test, cascade)["detection_rate"], 4)
                              if fires_test else None,
            "test_cameras": len(test),
            "test_fire_sequences": len(fires_test),
        }
        directions.append(row)
        print(f"[{name}]  tuned floor {tuned:.2f} on train (FAR "
              f"{row['far_on_train']:.3f})")
        print(f"    threshold transferred:  FAR {row['threshold_transfer_far']:.3f}"
              f"   recall {row['threshold_transfer_recall']}")
        print(f"    cascade (untuned 0.20): FAR {row['cascade_far']:.3f}"
              f"   recall {row['cascade_recall']}")
        print(f"    (oracle floor on test would be {oracle:.2f} at FAR "
              f"{row['oracle_far']:.3f})\n")

    # Aggregate the transfer penalty: how far the tuned threshold's FAR drifts
    # above target on unseen cameras, versus the cascade's drift.
    thr_drift = sum(d["threshold_transfer_far"] for d in directions) / len(directions)
    cas_drift = sum(d["cascade_far"] for d in directions) / len(directions)
    verdict = ("the cascade holds its false-alarm rate on unseen cameras while "
               "the tuned threshold drifts above it"
               if thr_drift > cas_drift + 1e-9 else
               "the tuned threshold transfers as well as the cascade on this "
               "negative set; the cascade's advantage is deduplication and "
               "hazard-agnostic transfer, not per-frame false-alarm rate")
    print(f"mean transferred-threshold FAR {thr_drift:.3f}  "
          f"vs cascade FAR {cas_drift:.3f}\n=> {verdict}")

    if args.json:
        args.json.write_text(json.dumps({
            "question": ("does a confidence floor tuned on one set of cameras "
                         "hold its false-alarm rate on unseen cameras, or does "
                         "the cascade's untuned persistence generalise better?"),
            "target_far": TARGET_FAR,
            "split": "cameras sorted and alternated into two disjoint folds",
            "directions": directions,
            "mean_transferred_threshold_far": round(thr_drift, 4),
            "mean_cascade_far": round(cas_drift, 4),
            "verdict": verdict,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
