#!/usr/bin/env python3
"""Measure the building-access detector on held-out DRespNeT annotations.

Ground truth is rebuilt from the COCO files with the SAME 28 -> 5 class merge
the model was trained on, read out of the model's own `.meta.json` rather than
duplicated here — so the two cannot drift apart.

WHAT IS REPORTED, AND WHY IT IS SPLIT

`civilian` is the only hazard class (see vigia/detectors/building_access.py),
so it gets the full two-level treatment VIGÍA uses everywhere:

  box level   — did we localise the people?
  image level — did we alarm on this frame, and should we have?

The other four classes are operator context. They are still measured, per class
at box level, because a responder reads them off the frame and a silently
broken `entry_accessible` would be just as useless as a broken alarm — but they
are never mixed into the headline.

SAMPLE SIZE — READ THIS BEFORE QUOTING ANYTHING. The public DRespNeT release
holds out 25 test images, of which 24 carry annotations in our five classes.
That is thinner than the 23 fire negatives already flagged in REGISTRY.yaml as
indicative rather than precise. Both splits are therefore evaluated and both
are reported: `valid` (50 images) was used for checkpoint selection and is NOT
held out, `test` (24) is held out but tiny. Neither is a publication-grade
sample on its own, and the JSON says so.

An IoU match threshold of 0.3 is used, consistent with the rest of VIGÍA and
for the same stated reason — at 0.5 the metric measures annotation convention
more than detection ability.

Usage:
    python eval/run_building_access_eval.py
    python eval/run_building_access_eval.py --json eval/results/building_access.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from eval.metrics import CountResult, match_boxes  # noqa: E402
from vigia.detectors.building_access import BuildingAccessDetector  # noqa: E402
from vigia.types import Box, Frame  # noqa: E402

logging.basicConfig(level=logging.ERROR)

DEFAULT_DATA = REPO_ROOT / "data" / "drespnet"
HAZARD_CLASS = "civilian"

#: Fallback only. The authoritative map ships in the model's .meta.json; this
#: exists so the harness still runs against a model exported before that field
#: was added.
FALLBACK_MERGE = {
    "civilian": ["civilian_visible", "group_of_civilians"],
    "rescue_team": ["rescue_team"],
    "entry_accessible": ["entry_door_accessible", "entry_window_accessible",
                         "entry_gap_accessible", "entry_gap_block_accessible"],
    "entry_blocked": ["entry_door_blocked", "entry_window_blocked"],
    "building_collapsed": ["building_collapsed"],
}


def load_merge(detector) -> dict[str, list[str]]:
    """The 28 -> 5 map the model was actually trained with."""
    metadata_path = detector.model_path.with_suffix(".meta.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        merge = metadata.get("class_merge")
        if merge:
            return {k: list(v) for k, v in merge.items()}
    print("  note: model has no class_merge in its metadata; using the "
          "fallback map", file=sys.stderr)
    return FALLBACK_MERGE


def load_ground_truth(split_dir: Path, merge: dict[str, list[str]]):
    """(file_name -> {class_name: [Box]}) for one COCO split, post-merge."""
    data = json.loads((split_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_name = {c["name"]: c["id"] for c in data["categories"]}

    source_to_target: dict[int, str] = {}
    for target, sources in merge.items():
        for source in sources:
            if source in by_name:
                source_to_target[by_name[source]] = target

    images = {image["id"]: image["file_name"] for image in data["images"]}
    truth: dict[str, dict[str, list[Box]]] = defaultdict(lambda: defaultdict(list))
    for annotation in data["annotations"]:
        target = source_to_target.get(annotation["category_id"])
        if target is None:
            continue
        x, y, w, h = annotation["bbox"]
        if w <= 1 or h <= 1:
            continue
        truth[images[annotation["image_id"]]][target].append(Box(x, y, x + w, y + h))

    # Only images carrying at least one label in our reduced set, matching the
    # training set's own filter.
    return {name: dict(classes) for name, classes in truth.items() if classes}


def evaluate(detector, split_dir: Path, truth: dict, iou_threshold: float,
             class_names) -> dict:
    box_level = CountResult()
    image_level = CountResult()
    per_class = {name: CountResult() for name in class_names}
    gt_total = pred_total = 0

    for index, (file_name, classes) in enumerate(sorted(truth.items())):
        image = cv2.imread(str(split_dir / file_name))
        if image is None:
            continue

        detections = detector.detect(
            Frame(image=image, index=index, timestamp=float(index),
                  camera_id=Path(file_name).stem)
        )
        detections.sort(key=lambda d: d.confidence, reverse=True)

        by_class: dict[str, list[Box]] = defaultdict(list)
        for detection in detections:
            by_class[detection.label].append(detection.box)

        for name in class_names:
            tp, fp, fn = match_boxes(by_class.get(name, []),
                                     classes.get(name, []), iou_threshold)
            per_class[name].true_positives += tp
            per_class[name].false_positives += fp
            per_class[name].false_negatives += fn

        # Headline: the hazard class alone.
        predicted = by_class.get(HAZARD_CLASS, [])
        ground = classes.get(HAZARD_CLASS, [])
        gt_total += len(ground)
        pred_total += len(predicted)

        tp, fp, fn = match_boxes(predicted, ground, iou_threshold)
        box_level.true_positives += tp
        box_level.false_positives += fp
        box_level.false_negatives += fn

        alarmed, should = bool(predicted), bool(ground)
        if should and alarmed:
            image_level.true_positives += 1
        elif should:
            image_level.false_negatives += 1
        elif alarmed:
            image_level.false_positives += 1
        else:
            image_level.true_negatives += 1

    return {
        "images": len(truth),
        "ground_truth_civilians": gt_total,
        "predicted_civilians": pred_total,
        "box_level": box_level.to_dict(),
        "image_level": image_level.to_dict(),
        "per_class_box_level": {n: r.to_dict() for n, r in per_class.items()},
        "median_latency_ms": round(detector.median_latency_ms, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", nargs="+", default=["test", "valid"])
    parser.add_argument("--conf", type=float, nargs="+",
                        default=[0.30, 0.50, 0.70])
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None,
                        help="override the model path (used to smoke-test the "
                             "harness against an untrained export)")
    args = parser.parse_args()

    detector = (BuildingAccessDetector(args.model, include_context=True)
                if args.model else BuildingAccessDetector(include_context=True))
    detector.warmup(rounds=2)
    merge = load_merge(detector)
    # What the detector EMITS, which is not the same as the classes its graph
    # predicts: this model has 28 and VIGÍA merges them to 5 at the boundary.
    class_names = list(getattr(detector, "emitted_class_names",
                               detector.class_names))

    print(f"model    : {detector.model_path.name}")
    print(f"classes  : {', '.join(class_names)}")
    print(f"hazard   : {HAZARD_CLASS} (the rest are operator context)")
    print(f"match    : IoU >= {args.iou}\n")

    results: dict = {}
    for split in args.splits:
        split_dir = args.data / split
        if not (split_dir / "_annotations.coco.json").exists():
            print(f"skipping {split}: no annotations", file=sys.stderr)
            continue
        truth = load_ground_truth(split_dir, merge)

        held_out = "held out" if split == "test" else "USED FOR CHECKPOINT SELECTION"
        print(f"--- {split} ({len(truth)} images, {held_out}) ---")
        header = (f"{'conf':>6} | {'BOX  P':>7} {'R':>6} {'F1':>6} {'F2':>6} | "
                  f"{'IMG  P':>7} {'R':>6} {'FAR':>6} | {'ms':>6}")
        print(header)
        print("-" * len(header))

        results[split] = {}
        for conf in args.conf:
            detector.conf_threshold = conf
            outcome = evaluate(detector, split_dir, truth, args.iou, class_names)
            results[split][f"conf_{conf}"] = outcome
            box, img = outcome["box_level"], outcome["image_level"]
            print(f"{conf:>6.2f} | {box['precision']:>7.3f} {box['recall']:>6.3f} "
                  f"{box['f1']:>6.3f} {box['f2']:>6.3f} | "
                  f"{img['precision']:>7.3f} {img['recall']:>6.3f} "
                  f"{img['false_alarm_rate']:>6.3f} | "
                  f"{outcome['median_latency_ms']:>6.1f}")

        middle = results[split][f"conf_{args.conf[len(args.conf) // 2]}"]
        print(f"\n  per-class box F1 @ conf {args.conf[len(args.conf) // 2]}:")
        for name, counts in middle["per_class_box_level"].items():
            marker = "  <- hazard" if name == HAZARD_CLASS else ""
            print(f"    {name:<20} P {counts['precision']:.3f}  "
                  f"R {counts['recall']:.3f}  F1 {counts['f1']:.3f}"
                  f"  (n={counts['true_positives'] + counts['false_negatives']})"
                  f"{marker}")
        print()

    print("Box recall on `civilian` is the number that matters: the fraction of "
          "people\nvisible in the rubble that the detector found at all.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "model": detector.model_path.name,
            "dataset": "DRespNeT (Cranfield University), CC BY 4.0",
            "iou_match_threshold": args.iou,
            "hazard_class": HAZARD_CLASS,
            "class_merge": merge,
            "sample_size_caveat":
                "The public DRespNeT release holds out 25 test images (24 with "
                "labels in our five classes). That is a very small sample and "
                "these figures are indicative, not precise — the same caveat "
                "already recorded for the 23 fire negatives. The `valid` split "
                "is larger but was used for checkpoint selection and is "
                "therefore NOT held out. Widen before publishing.",
            "reported_by_authors": {
                "model": "YOLOv8-DRN (DRespNeT paper, arXiv 2508.16016)",
                "mAP50": 0.927,
                "fps": 27,
                "note": "Theirs, not ours, and NOT comparable: they report over "
                        "28 classes, we merge to 5 and measure one hazard class.",
            },
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
