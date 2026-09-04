#!/usr/bin/env python3
"""The reviewer's question: why not just raise the confidence threshold?

The ablation (eval/run_ablation.py) compares the validator against the raw
detector AT THE SAME confidence floor. That leaves open the obvious
alternative: forget temporal validation, and simply raise the floor until the
false-alarm rate matches. If that reaches the same operating point, the
cascade is machinery for nothing; if it cannot, the cascade is earning its
keep. This script answers the question rather than leaving it to a reviewer.

Method: sweep the raw detector's confidence floor over a fine grid, replaying
the same cached detections the ablation uses — so both approaches see
byte-identical detector output — and record, at every floor, the false-alarm
rate on the negative sequences and the detection behaviour on the ignition
sequences. Then place the cascade's operating point (persistence 3, floor
0.20) on that frontier.

Usage:
    python eval/run_threshold_matched.py
    python eval/run_threshold_matched.py --json eval/results/threshold_matched.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from run_ablation import load_cache, replay  # noqa: E402
from vigia.validator.cascade import ValidatorConfig  # noqa: E402

CACHE_DIR = REPO_ROOT / "eval" / "cache"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--negatives", type=Path, default=CACHE_DIR / "figlib_neg.jsonl")
    parser.add_argument("--fires", type=Path, default=CACHE_DIR / "figlib_fires.jsonl")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    negatives = load_cache(args.negatives)
    fires = load_cache(args.fires)

    # --- the cascade's operating point, for reference ------------------- #
    cascade_config = ValidatorConfig(
        min_confidence=0.20, min_frames=3, enable_cooldown=False,
    )
    cascade = {
        "negatives": replay(negatives, cascade_config),
        "fires": replay(fires, cascade_config),
    }
    target_far = cascade["negatives"]["false_alarm_rate"]

    # --- the raw detector, swept --------------------------------------- #
    # A raw run is replay(config=None), which honours only the floor. The
    # replay signature takes the floor from a config's min_confidence when
    # one is passed alongside None-validator semantics, so sweep by giving
    # replay a bare-confidence cascade with every level disabled — identical
    # detections, no temporal machinery.
    sweep = []
    floors = [round(0.20 + 0.01 * i, 2) for i in range(76)]   # 0.20 .. 0.95
    for floor in floors:
        raw_config = ValidatorConfig(
            min_confidence=floor,
            enable_persistence=False, enable_cooldown=False,
        )
        entry = {
            "floor": floor,
            "negatives": replay(negatives, raw_config),
            "fires": replay(fires, raw_config),
        }
        sweep.append(entry)

    # The smallest floor whose FAR is no worse than the cascade's.
    matched = next((e for e in sweep
                    if e["negatives"]["false_alarm_rate"] <= target_far), None)

    header = (f"{'floor':>6} {'FAR':>7} | {'fires':>7} {'recall':>7} "
              f"{'min-to-detect':>14}")
    print(f"cascade reference: floor 0.20, persistence 3 -> "
          f"FAR {target_far:.3f}, "
          f"{cascade['fires']['sequences_detected']}/{cascade['fires']['total_sequences']} fires, "
          f"recall {cascade['fires']['detection_rate']:.3f}, "
          f"median {cascade['fires']['median_minutes_to_detect']} min\n")
    print(header)
    print("-" * len(header))
    for entry in sweep:
        marker = "  <- FAR matched" if entry is matched else ""
        print(f"{entry['floor']:>6.2f} {entry['negatives']['false_alarm_rate']:>7.3f} | "
              f"{entry['fires']['sequences_detected']:>3}/{entry['fires']['total_sequences']:<3} "
              f"{entry['fires']['detection_rate']:>7.3f} "
              f"{str(entry['fires']['median_minutes_to_detect']):>14}{marker}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "question": ("can raising the raw detector's confidence floor reach "
                         "the cascade's operating point without the cascade?"),
            "method": ("identical cached detections as eval/run_ablation.py; "
                       "raw = all temporal levels disabled, floor swept"),
            "cascade": {"floor": 0.20, "persistence": 3, **cascade},
            "matched_floor": matched["floor"] if matched else None,
            "matched": matched,
            "sweep": sweep,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
