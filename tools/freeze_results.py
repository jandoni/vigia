#!/usr/bin/env python3
"""Guard: every published number must trace to a file in eval/results/.

    python tools/freeze_results.py
    python tools/freeze_results.py --write RESULTS.md

The acceptance criterion for the results freeze is that every published
figure traces back to a measurement on disk. models/REGISTRY.yaml quotes
figures and names the results file each came from, which is exactly the shape
of claim that rots: the harness is re-run, the JSON updates, and the summary
that a reader actually sees keeps the old number. Nothing about that failure is
visible — both files are well-formed and internally consistent.

So this walks every quoted figure in the registry, opens the results file it
cites, and compares. A mismatch fails the build. It is the same reasoning as
tools/check_licence.py and tools/check_privacy.py: a claim worth publishing is
worth enforcing in CI.

The `--write` mode emits a single traceable summary table, so there is
one source to quote from rather than a dozen JSON files to be transcribed by
hand — transcription being the other way these numbers drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "models" / "REGISTRY.yaml"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

TOLERANCE = 5e-4          # registry values are rounded to 4 decimal places


def dig(data, path: str):
    """Fetch a dotted path, treating every segment as a dict key.

    Written as an explicit walk rather than a split-on-dot lookup because the
    results files contain keys like `conf_0.2` that themselves contain a dot.
    """
    node = data
    for key in path.split("|"):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


#: (model key, registry path, results path) for every figure the registry
#: quotes from a results file. Registry paths are dotted; results paths use `|`
#: because some of their keys contain dots.
CHECKS: list[tuple[str, str, str]] = [
    # fire — baseline, single-frame, no validator
    ("fire.pyronear_yolo11s_sensitive", "box_level.precision",
     "results|conf_0.2|box_level|precision"),
    ("fire.pyronear_yolo11s_sensitive", "box_level.recall",
     "results|conf_0.2|box_level|recall"),
    ("fire.pyronear_yolo11s_sensitive", "image_level.false_alarm_rate",
     "results|conf_0.2|image_level|false_alarm_rate"),

    # flood — held-out ATLANTIS test split
    ("flood.atlantis_deeplabv3", "test_split.water_iou", "segmentation|water_iou"),
    ("flood.atlantis_deeplabv3", "test_split.precision", "segmentation|precision"),
    ("flood.atlantis_deeplabv3", "test_split.recall", "segmentation|recall"),
    ("flood.atlantis_deeplabv3", "test_split.f1", "segmentation|f1"),
    ("flood.atlantis_deeplabv3", "test_split.pixel_accuracy",
     "segmentation|pixel_accuracy"),
    ("flood.atlantis_deeplabv3", "test_split.median_latency_ms",
     "segmentation|median_latency_ms"),

    # people in water — swimmer class only, conf 0.30
    ("person_in_water.seadronessee_rfdetr", "box_level.precision",
     "results|conf_0.3|box_level|precision"),
    ("person_in_water.seadronessee_rfdetr", "box_level.recall",
     "results|conf_0.3|box_level|recall"),
    ("person_in_water.seadronessee_rfdetr", "box_level.f1",
     "results|conf_0.3|box_level|f1"),
    ("person_in_water.seadronessee_rfdetr", "image_level.false_alarm_rate",
     "results|conf_0.3|image_level|false_alarm_rate"),
    ("person_in_water.seadronessee_rfdetr", "ground_truth_swimmer_instances",
     "results|conf_0.3|ground_truth_swimmers"),

    # building access — HELD-OUT test split at the 0.30 operating point
    ("building_access.drespnet_yolov8_drn", "test_split.box_level.precision",
     "results|test|conf_0.3|box_level|precision"),
    ("building_access.drespnet_yolov8_drn", "test_split.box_level.recall",
     "results|test|conf_0.3|box_level|recall"),
    ("building_access.drespnet_yolov8_drn", "test_split.box_level.f1",
     "results|test|conf_0.3|box_level|f1"),
    ("building_access.drespnet_yolov8_drn", "test_split.image_level.recall",
     "results|test|conf_0.3|image_level|recall"),
    ("building_access.drespnet_yolov8_drn",
     "test_split.image_level.false_alarm_rate",
     "results|test|conf_0.3|image_level|false_alarm_rate"),
    ("building_access.drespnet_yolov8_drn",
     "test_split.ground_truth_civilian_instances",
     "results|test|conf_0.3|ground_truth_civilians"),
    ("building_access.drespnet_yolov8_drn", "valid_split.box_level.recall",
     "results|valid|conf_0.3|box_level|recall"),

    # aerial flood — training-time validation, labelled as such in the registry
    ("flood_aerial.floodnet_deeplabv3", "val_flooded_water_iou",
     "best|flooded_water_iou"),
    ("flood_aerial.floodnet_deeplabv3", "val_overall_water_iou",
     "best|overall_water_iou"),
    ("flood_aerial.floodnet_deeplabv3", "val_pixel_accuracy",
     "best|pixel_accuracy"),
    ("flood_aerial.floodnet_deeplabv3", "best_epoch", "best_epoch"),

    # scene router — held out on different sources
    ("scene_router.mobilenetv3", "heldout_accuracy", "best_heldout_accuracy"),
    ("scene_router.mobilenetv3", "heldout_images", "heldout_images"),

    # threshold-matched control — the oracle floor against the cascade
    ("validator.threshold_matched", "matched_floor", "matched_floor"),
    ("validator.threshold_matched", "matched_recall",
     "matched|fires|detection_rate"),
    ("validator.threshold_matched", "matched_minutes_to_detect",
     "matched|fires|median_minutes_to_detect"),
    ("validator.threshold_matched", "cascade_far",
     "cascade|negatives|false_alarm_rate"),
    ("validator.threshold_matched", "cascade_recall",
     "cascade|fires|detection_rate"),

    # sliced inference (SAHI) — measured and rejected
    ("detector.sliced_inference", "baseline_recall",
     "results|baseline_conf_0.3|box_level|recall"),
    ("detector.sliced_inference", "sliced_recall",
     "results|sliced_conf_0.3|box_level|recall"),
    ("detector.sliced_inference", "sliced_precision",
     "results|sliced_conf_0.3|box_level|precision"),
    ("detector.sliced_inference", "sliced_image_far",
     "results|sliced_conf_0.3|image_level|false_alarm_rate"),

    # colour prior — measured and removed
    ("validator.colour_prior", "stills_baseline_recall",
     "stills|results|colour off|recall"),
    ("validator.colour_prior", "stills_baseline_far",
     "stills|results|colour off|false_alarm_rate"),
    ("validator.colour_prior", "stills_strict_recall",
     "stills|results|S<=40|recall"),
    ("validator.colour_prior", "stills_strict_far",
     "stills|results|S<=40|false_alarm_rate"),
    ("validator.colour_prior", "temporal_strict_fires_detected",
     "temporal|S<=40|fires|sequences_detected"),
]


def registry_value(model: dict, path: str):
    node = model.get("measured", {})
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", type=Path, default=None,
                        help="write a traceable summary table")
    args = parser.parse_args()

    import yaml

    loaded = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    # Experiments are guarded exactly like models: a figure the paper quotes
    # is a claim, whether it describes a network or the system around one.
    registry = {**loaded["models"], **loaded.get("experiments", {})}

    cache: dict[str, dict] = {}
    failures: list[str] = []
    checked = 0
    rows: list[tuple[str, str, str, str]] = []

    for model_key, registry_path, results_path in CHECKS:
        model = registry.get(model_key)
        if model is None:
            failures.append(f"{model_key}: not in registry")
            continue

        cited = model.get("measured", {}).get("results_file")
        if not cited:
            failures.append(f"{model_key}: measured block cites no results_file")
            continue

        file_path = REPO_ROOT / cited
        if not file_path.exists():
            failures.append(f"{model_key}: cited {cited} does not exist")
            continue

        if cited not in cache:
            cache[cited] = json.loads(file_path.read_text(encoding="utf-8"))

        claimed = registry_value(model, registry_path)
        actual = dig(cache[cited], results_path)
        checked += 1

        if claimed is None:
            failures.append(f"{model_key}.{registry_path}: absent from registry")
            continue
        if actual is None:
            failures.append(
                f"{model_key}.{registry_path}: no value at {results_path} in {cited}")
            continue
        if abs(float(claimed) - float(actual)) > TOLERANCE:
            failures.append(
                f"{model_key}.{registry_path}: registry says {claimed}, "
                f"{cited} says {actual}")
            continue

        rows.append((model_key, registry_path, f"{claimed}", cited))

    # Every measured model must cite a file, even if no figure is checked yet.
    for model_key, model in registry.items():
        measured = model.get("measured", {})
        if measured.get("status") in (None, "not_yet_measured"):
            continue
        # `validation_measured` is a weaker evidence class but still has to
        # cite its source — the point of the guard is provenance, not strength.
        if not measured.get("results_file"):
            failures.append(f"{model_key}: measured but cites no results_file")
        if not measured.get("harness"):
            failures.append(f"{model_key}: measured but names no harness")

    if failures:
        print("RESULTS FREEZE FAILED\n")
        for failure in failures:
            print(f"  {failure}")
        print("\nA figure in the registry disagrees with the results file it "
              "cites, which means\nthe published summary and the measurement "
              "have drifted apart. Re-run the harness\nand update the registry, "
              "or correct the citation — do not adjust one to match the\nother "
              "without knowing which is right.")
        return 1

    print(f"RESULTS FREEZE PASSED — {checked} published figure(s) across "
          f"{len(cache)} results file(s) match their source.")

    if args.write:
        lines = [
            "# VIGÍA — frozen results",
            "",
            "Every figure below is checked against the results file named "
            "beside it by `tools/freeze_results.py`, which fails the build if "
            "the two disagree. Quote from this table rather than transcribing "
            "from JSON.",
            "",
            "| model | figure | value | source |",
            "|---|---|---|---|",
        ]
        for model_key, path, value, source in rows:
            lines.append(f"| `{model_key}` | {path} | **{value}** | `{source}` |")
        lines += [
            "",
            "## Not measured",
            "",
        ]
        for model_key, model in registry.items():
            measured = model.get("measured", {})
            if measured.get("status") in (None, "not_yet_measured"):
                note = model.get("status_note", "").strip().split("\n")[0]
                lines.append(f"- `{model_key}` — {measured.get('status', 'unknown')}"
                             + (f". {note}" if note else ""))
        args.write.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
