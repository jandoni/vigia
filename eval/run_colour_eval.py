#!/usr/bin/env python3
"""Measure a colour prior for the cascade — the experiment that removed it.

Every configuration this project shipped disabled the colour level, which
meant the cascade definition described a level that had never run in a
reported experiment. The same standard that removed the size level applied:
measure it, then keep it or cut it on the evidence. THE EVIDENCE SAID CUT —
this script is the record of that measurement, and it is self-contained (the
prior is implemented locally) precisely because the level it measured no
longer exists in vigia/validator/cascade.py.

WHAT THE PRIOR IS. Wildfire smoke is distinguished by being achromatic: grey
against sky, vegetation or ground. The prior therefore demands that at least
a fraction of the pixels in a detection's box be LOW-SATURATION
(S <= cutoff, with a modest V floor so night-black does not count as grey).
Applying it as a stateless detection prefilter is exactly equivalent to the
level's old position in the cascade (after confidence, before persistence).

TWO REGIMES, because they answer different questions:

  stills  — pyro-sdis validation split: 1,085 images, 585 CURATED HARD
            negatives (cloud, fog, dust). Thresholding fails here, so this is
            where a per-frame appearance filter would matter if it worked.
  temporal— the FIgLib sequences of the ablation, frames loaded, prior
            applied inside the full cascade at persistence 3.

THE RESULT, for the reader arriving from the paper: fog and smoke are both
grey. On the hard stills the prior buys false-alarm rate only by discarding
real smoke (recall 0.978 -> 0.736 at S<=40); in the temporal cascade it
improves nothing at any cutoff and a strict one loses an ignition sequence
(5/6). The cutoff is SWEPT and every point reported, so the conclusion does
not rest on a cherry-picked prior.

Usage:
    python eval/run_colour_eval.py --json eval/results/colour_prior.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from run_ablation import load_cache, to_detections  # noqa: E402
from vigia.types import Detection, Frame, Hazard  # noqa: E402
from vigia.validator.cascade import TemporalValidator, ValidatorConfig  # noqa: E402

CACHE_DIR = REPO_ROOT / "eval" / "cache"
STILLS_DIR = REPO_ROOT / "data" / "pyro_sdis_val_wide"
FIRES_DIR = REPO_ROOT / "data" / "figlib"
NEG_DIR = REPO_ROOT / "data" / "figlib_neg"

#: Modest brightness floor: greyness only counts in daylight-visible pixels,
#: so a black night frame is not "achromatic smoke".
MIN_VALUE = 50


def passes_grey_prior(image, detection: Detection, max_saturation: int,
                      min_fraction: float = 0.10) -> bool:
    """The removed level's check, preserved here so the experiment reruns.

    True when at least `min_fraction` of the box's pixels are achromatic:
    S <= max_saturation and V >= MIN_VALUE, any hue.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in detection.box.as_xyxy())
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, MIN_VALUE], dtype=np.uint8)
    upper = np.array([180, max_saturation, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return (float(np.count_nonzero(mask)) / mask.size) >= min_fraction


def colour_filter(image, detections: list[Detection],
                  cutoff: int | None) -> list[Detection]:
    if cutoff is None or image is None:
        return detections
    return [d for d in detections if passes_grey_prior(image, d, cutoff)]


# --------------------------------------------------------------------------- #
# Regime 1: curated hard stills
# --------------------------------------------------------------------------- #

def cache_still_detections(floor: float) -> list[dict]:
    """One detector pass over the stills; the sweep re-filters this list."""
    from vigia.detectors.fire import FireDetector

    detector = FireDetector()
    rows = []
    images = sorted(STILLS_DIR.glob("*.jpg"))
    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            continue
        label = path.with_suffix(".txt")
        positive = bool(label.exists() and label.read_text().strip())
        detections = [d for d in detector.detect(
            Frame(image=image, index=index, timestamp=float(index),
                  camera_id=path.stem))
            if d.confidence >= floor]
        rows.append({
            "path": str(path),
            "positive": positive,
            "detections": detections,
        })
        if (index + 1) % 200 == 0:
            print(f"  stills: {index + 1}/{len(images)}", file=sys.stderr)
    return rows


def evaluate_stills(rows: list[dict], cutoff: int | None) -> dict:
    """Image-level FAR and recall, with the grey prior optionally applied."""
    tp = fp = fn = tn = 0
    for row in rows:
        detections = row["detections"]
        if cutoff is not None and detections:
            image = cv2.imread(row["path"])
            detections = colour_filter(image, detections, cutoff)
        alarmed = bool(detections)
        if row["positive"] and alarmed:
            tp += 1
        elif row["positive"]:
            fn += 1
        elif alarmed:
            fp += 1
        else:
            tn += 1
    return {
        "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else None,
    }


# --------------------------------------------------------------------------- #
# Regime 2: the temporal sequences, frames loaded
# --------------------------------------------------------------------------- #

def replay_with_frames(sequences: dict, root: Path, config: ValidatorConfig,
                       cutoff: int | None = None) -> dict:
    frames = alarmed_pre = alarmed_post = 0
    pre_frames = post_frames = 0
    first_alert: dict[str, float] = {}

    for name, rows in sequences.items():
        validator = TemporalValidator(Hazard.FIRE, config)
        for row in rows:
            image = cv2.imread(str(root / row["sequence"] / row["file"]))
            detections = colour_filter(image, to_detections(row), cutoff)
            events = validator.process(
                detections, row["frame_index"], row["offset"], image, name)
            hit = bool(events)
            frames += 1
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
        "false_alarm_rate": round(alarmed_pre / pre_frames, 4) if pre_frames else None,
        "detection_rate": round(alarmed_post / post_frames, 4) if post_frames else None,
        "sequences_detected": len(first_alert),
        "total_sequences": len(sequences),
        "median_minutes_to_detect": (
            round(latencies[len(latencies) // 2], 1) if latencies else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--floor", type=float, default=0.20)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cutoffs = [40, 60, 80, 100, 120]

    # ---- stills ------------------------------------------------------- #
    print("running the fire detector over the pyro-sdis stills once …")
    still_rows = cache_still_detections(args.floor)
    positives = sum(1 for r in still_rows if r["positive"])
    print(f"  {len(still_rows)} images, {positives} positive\n")

    print(f"{'prior':<28} {'FAR':>7} {'recall':>8}   (stills, hard negatives)")
    print("-" * 55)
    baseline = evaluate_stills(still_rows, None)
    print(f"{'colour off':<28} {baseline['false_alarm_rate']:>7.3f} "
          f"{baseline['recall']:>8.3f}")
    stills = {"colour off": baseline}
    for cutoff in cutoffs:
        result = evaluate_stills(still_rows, cutoff)
        stills[f"S<={cutoff}"] = result
        print(f"{f'grey prior, S<={cutoff}':<28} "
              f"{result['false_alarm_rate']:>7.3f} {result['recall']:>8.3f}")

    # ---- temporal ----------------------------------------------------- #
    negatives = load_cache(CACHE_DIR / "figlib_neg.jsonl")
    fires = load_cache(CACHE_DIR / "figlib_fires.jsonl")
    base = ValidatorConfig(min_confidence=args.floor, min_frames=3,
                           enable_cooldown=False)

    print(f"\n{'cascade config':<28} {'FAR':>7} {'fires':>7} {'recall':>8} "
          f"{'min':>5}   (temporal, persistence 3)")
    print("-" * 70)
    temporal = {}
    for label, cutoff in [("colour off", None)] + [
        (f"S<={c}", c) for c in cutoffs
    ]:
        neg = replay_with_frames(negatives, NEG_DIR, base, cutoff)
        fire = replay_with_frames(fires, FIRES_DIR, base, cutoff)
        temporal[label] = {"negatives": neg, "fires": fire}
        print(f"{label:<28} {neg['false_alarm_rate']:>7.3f} "
              f"{fire['sequences_detected']:>3}/{fire['total_sequences']:<3} "
              f"{fire['detection_rate']:>8.3f} "
              f"{str(fire['median_minutes_to_detect']):>5}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "question": ("does a colour prior earn a place in the cascade, "
                         "measured rather than assumed?"),
            "verdict": ("REMOVED. Fog and smoke are both grey: on hard stills "
                        "the prior trades ~5 points of recall per 10 points "
                        "of FAR; in the cascade it improves nothing and a "
                        "strict cutoff loses an ignition sequence."),
            "prior": ("achromatic (smoke): fraction >= 0.10 of box pixels "
                      f"with S <= cutoff, V >= {MIN_VALUE}"),
            "confidence_floor": args.floor,
            "stills": {"images": len(still_rows), "positives": positives,
                       "results": stills},
            "temporal": temporal,
        }, indent=2, default=lambda o: None))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
