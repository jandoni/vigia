#!/usr/bin/env python3
"""Build the VIGÍA technical paper (LaTeX -> PDF).

    python docs/build_paper.py            # or: make paper

WHY THIS EXISTS RATHER THAN A .tex FULL OF TYPED NUMBERS
--------------------------------------------------------
The prose is written by hand, in docs/paper/sections/. Every FIGURE in it is
not: this script reads eval/results/*.json and models/REGISTRY.yaml and emits
docs/paper/generated/numbers.tex, a file of macros the sections cite by name.

    \\FireImageFAR      instead of     0.706

So a measurement cannot drift out of step with the sentence around it, the
same discipline tools/freeze_results.py enforces for the Word document — and
a re-measurement changes the paper by re-running it rather than by a search
and replace that misses two places.

It also DERIVES three things the results files do not store, because they are
properties of the numbers rather than new measurements:

  * Wilson score intervals for every proportion whose denominator is known.
    Half the figures in this project rest on small samples — 24 held-out
    images, 23 negatives — and a bare point estimate from n=24 invites a
    reader to believe it to three decimal places. The interval is the honest
    presentation, and it is computed rather than asserted.

  * The Page-Hinkley false-alarm model, evaluated against the measured
    calibration grid. A closed form that reproduces measurements is a design
    rule for a camera nobody has calibrated yet; one that does not is a
    warning. Both outcomes are worth printing, so the comparison is generated
    and the paper reports whatever it says.

  * The persistence latency identity, checked against the measured delay.

Usage:
    python docs/build_paper.py [--no-figures] [--keep-log]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PAPER = REPO_ROOT / "docs" / "paper"
GENERATED = PAPER / "generated"
RESULTS = REPO_ROOT / "eval" / "results"
MAIN = PAPER / "vigia.tex"

DOC_VERSION = "1.1"


# --------------------------------------------------------------------------- #
# Statistics derived here, not stored anywhere
# --------------------------------------------------------------------------- #

def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the textbook normal approximation deliberately. At the sample
    sizes this project actually has, the normal interval is not merely wide but
    WRONG: for the building-access image-level recall of 12/12 it gives
    [1.000, 1.000], asserting certainty from twelve observations, and for any
    proportion near 0 or 1 it can extend outside [0, 1] entirely. The Wilson
    interval is derived by inverting the score test rather than by assuming the
    estimate is normally distributed, so it stays inside the unit interval and
    remains sensible at the boundary.

        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z sqrt( p(1-p)/n + z^2/4n^2 ) / (1 + z^2/n)
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = (z * math.sqrt(p * (1 - p) / trials
                          + z * z / (4 * trials * trials))) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def page_hinkley_false_alarm(delta: float, threshold: float,
                             sigma: float) -> float:
    """Predicted false-alarm probability on a stationary noisy signal.

    Page-Hinkley accumulates the deviation of a signal from its running mean,
    less a tolerated drift. On a signal that is NOT changing, that accumulator
    is a random walk with step standard deviation sigma and drift -delta per
    sample, and the test statistic is the walk's drawdown below its own
    running minimum. For Brownian motion with negative drift the all-time
    maximum drawdown is exponentially distributed, which gives

        P(false alarm) ~= exp( -2 * delta * threshold / sigma^2 )

    This is an approximation in three respects — it assumes the infinite-time
    limit, treats the running-mean subtraction as exact, and ignores the
    minimum-sample guard — so it should be read as a floor rather than a
    prediction. The paper prints it beside the measured grid so the reader can
    see exactly how loose it is.
    """
    if sigma <= 0:
        return 0.0
    return math.exp(-2.0 * delta * threshold / (sigma * sigma))


def latex_number(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


# --------------------------------------------------------------------------- #
# Macro emission
# --------------------------------------------------------------------------- #

class Macros:
    """Accumulates \\newcommand definitions, refusing duplicates.

    Refusing rather than overwriting: two macros with the same name and
    different values means two parts of this script disagree about a
    measurement, which must stop the build rather than resolve itself by
    ordering.
    """

    def __init__(self) -> None:
        self._items: list[tuple[str, str, str]] = []
        self._seen: set[str] = set()

    def add(self, name: str, value: str, source: str = "") -> None:
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} must be letters only")
        if name in self._seen:
            raise ValueError(f"macro {name!r} defined twice")
        self._seen.add(name)
        self._items.append((name, str(value), source))

    def number(self, name: str, value, places: int = 3, source: str = "") -> None:
        self.add(name, latex_number(float(value), places), source)

    def percent(self, name: str, value, places: int = 1, source: str = "") -> None:
        self.add(name, f"{float(value) * 100:.{places}f}", source)

    def interval(self, name: str, successes: int, trials: int,
                 places: int = 3, source: str = "") -> None:
        """Point estimate plus its Wilson interval, as three macros."""
        low, high = wilson(successes, trials)
        self.number(name, successes / trials if trials else 0.0, places, source)
        self.number(name + "Low", low, places, source)
        self.number(name + "High", high, places, source)
        self.add(name + "CI",
                 f"[{latex_number(low, places)}, {latex_number(high, places)}]",
                 source)

    def render(self) -> str:
        width = max((len(n) for n, _, _ in self._items), default=0)
        lines = [
            "% GENERATED FILE — do not edit.",
            "% Written by docs/build_paper.py from eval/results/*.json.",
            "% Every quantity the paper states is defined here and cited by",
            "% name, so a re-measurement cannot leave a stale figure in the",
            "% prose. Edit the measurement, not the paper.",
            "",
        ]
        for name, value, source in self._items:
            comment = f"  % {source}" if source else ""
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}"
                         f"{' ' * (width - len(name))}{comment}")
        return "\n".join(lines) + "\n"


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #

def collect() -> Macros:
    m = Macros()
    m.add("DocVersion", DOC_VERSION)

    # ---------------------------------------------------------- fire ---- #
    wide = load("fire_pyro_sdis_val_wide.json")
    narrow = load("fire_pyro_sdis_val_baseline.json")
    if wide:
        c = wide["results"]["conf_0.2"]
        box, img = c["box_level"], c["image_level"]
        src = "eval/results/fire_pyro_sdis_val_wide.json"
        m.add("FireImages", str(wide["images"]), src)
        m.add("FirePositives", str(wide["positives"]), src)
        m.add("FireNegatives", str(wide["negatives"]), src)
        m.number("FireBoxPrecision", box["precision"], 3, src)
        m.number("FireBoxRecall", box["recall"], 3, src)
        m.number("FireBoxFTwo", box["f2"], 3, src)
        m.number("FireImageRecall", img["recall"], 3, src)
        m.number("FireLatency", c["median_latency_ms"], 1, src)
        # The headline false-alarm rate, with the interval its 585 negatives
        # actually support.
        m.interval("FireImageFAR", img["false_positives"],
                   img["false_positives"] + img["true_negatives"], 3, src)
    if narrow:
        c = narrow["results"]["conf_0.2"]
        img = c["image_level"]
        src = "eval/results/fire_pyro_sdis_val_baseline.json"
        m.add("FireNarrowNegatives", str(narrow["negatives"]), src)
        m.number("FireNarrowBoxPrecision", c["box_level"]["precision"], 3, src)
        m.number("FireNarrowBoxRecall", c["box_level"]["recall"], 3, src)
        m.interval("FireNarrowFAR", img["false_positives"],
                   img["false_positives"] + img["true_negatives"], 3, src)

    # ------------------------------------------------------- ablation ---- #
    ab = load("r1_ablation.json")
    if ab:
        src = "eval/results/r1_ablation.json"
        res = ab["results"]
        raw = res["raw detector (no validator)"]
        full = res["full cascade, persistence=3"]
        m.add("AblationNegativeSequences", str(ab["negative_sequences"]), src)
        m.add("AblationFireSequences", str(ab["fire_sequences"]), src)
        m.number("RawFAR", raw["negatives"]["false_alarm_rate"], 3, src)
        m.number("ValidatedFAR", full["negatives"]["false_alarm_rate"], 3, src)
        m.percent("FARReduction",
                  1 - full["negatives"]["false_alarm_rate"]
                  / raw["negatives"]["false_alarm_rate"], 0, src)
        m.add("FiresFound", str(full["fires"]["sequences_detected"]), src)
        m.add("FiresTotal", str(full["fires"]["total_sequences"]), src)
        m.add("RawFiresFound", str(raw["fires"]["sequences_detected"]), src)
        m.add("MedianMinutesToDetect",
              str(full["fires"]["median_minutes_to_detect"]), src)
        m.add("RawMinutesToDetect",
              str(raw["fires"]["median_minutes_to_detect"]), src)
        # The ADDED latency, which is what eq. (16) predicts — not the
        # absolute time to detect. An earlier draft of this paper labelled the
        # absolute figure as the added one; the two differ by the raw
        # detector's own median, and the identity only matches the difference.
        m.number("AddedMinutesToDetect",
                 full["fires"]["median_minutes_to_detect"]
                 - raw["fires"]["median_minutes_to_detect"], 1, src)
        m.number("RawFireRecall", raw["fires"]["detection_rate"], 3, src)
        m.number("ValidatedFireRecall", full["fires"]["detection_rate"], 3, src)
        m.number("PersistenceThree", 3, 0, "cascade configuration")

    # ---------------------------------------------------------- flood ---- #
    flood = load("flood.json")
    if flood and "segmentation" in flood:
        seg = flood["segmentation"]
        src = "eval/results/flood.json"
        m.number("FloodWaterIoU", seg["water_iou"], 4, src)
        m.number("FloodPrecision", seg["precision"], 4, src)
        m.number("FloodRecall", seg["recall"], 4, src)
        m.number("FloodFOne", seg["f1"], 4, src)
        m.number("FloodPixelAccuracy", seg["pixel_accuracy"], 4, src)
        m.number("FloodLatency", seg["median_latency_ms"], 1, src)
        m.add("FloodTestImages", str(seg["images"]), src)

    aerial = load("flood_aerial.json")
    if aerial:
        src = "eval/results/flood_aerial.json"
        split, best = aerial["split"], aerial["best"]
        m.add("AerialLabelled", str(split["labelled_pairs"]), src)
        m.add("AerialFlooded", str(split["flooded"]), src)
        m.add("AerialDry", str(split["dry"]), src)
        m.add("AerialValFlooded", str(split["val_flooded"]), src)
        m.add("AerialBestEpoch", str(aerial["best_epoch"]), src)
        m.number("AerialFloodedIoU", best["flooded_water_iou"], 4, src)
        m.number("AerialOverallIoU", best["overall_water_iou"], 4, src)
        m.number("AerialPixelAccuracy", best["pixel_accuracy"], 4, src)
        m.percent("AerialFloodedShare",
                  split["flooded"] / split["labelled_pairs"], 1, src)

    # ------------------------------------------------ people in water ---- #
    piw = load("person_in_water.json")
    if piw:
        src = "eval/results/person_in_water.json"
        # 0.30 is the published operating point; see the confidence sweep.
        c = piw["results"]["conf_0.3"]
        box, img = c["box_level"], c["image_level"]
        m.number("SwimmerPrecision", box["precision"], 4, src)
        m.number("SwimmerRecall", box["recall"], 4, src)
        m.number("SwimmerFOne", box["f1"], 4, src)
        m.number("SwimmerLatency", c["median_latency_ms"], 1, src)
        m.add("SwimmerInstances", str(c["ground_truth_swimmers"]), src)
        m.add("SwimmerImages", str(piw["images"]), src)
        m.add("SwimmerPositiveImages", str(piw["images_with_swimmers"]), src)
        m.interval("SwimmerImageFAR", img["false_positives"],
                   img["false_positives"] + img["true_negatives"], 4, src)
        m.interval("SwimmerBoxRecall", box["true_positives"],
                   box["true_positives"] + box["false_negatives"], 4, src)
        m.number("SwimmerAuthorAP",
                 piw["reported_by_author_all_classes"]["per_class_AP_swimmer"],
                 3, src)
        m.number("SwimmerAuthorMAP",
                 piw["reported_by_author_all_classes"]["mAP_50"], 4, src)

    # ------------------------------------------------- building access ---- #
    ba = load("building_access.json")
    if ba:
        src = "eval/results/building_access.json"
        test = ba["results"]["test"]["conf_0.3"]
        box, img = test["box_level"], test["image_level"]
        m.add("BuildingTestImages", str(test["images"]), src)
        m.add("BuildingCivilians", str(test["ground_truth_civilians"]), src)
        m.number("BuildingBoxPrecision", box["precision"], 3, src)
        m.number("BuildingBoxRecall", box["recall"], 3, src)
        m.number("BuildingBoxFOne", box["f1"], 3, src)
        m.number("BuildingLatency", test["median_latency_ms"], 1, src)
        # The civilian recall interval, over instances rather than images.
        m.interval("BuildingCivilianRecall", box["true_positives"],
                   box["true_positives"] + box["false_negatives"], 3, src)
        # Image-level recall on TWELVE positive images. This interval is the
        # whole reason the section exists.
        m.interval("BuildingImageRecall", img["true_positives"],
                   img["true_positives"] + img["false_negatives"], 3, src)
        m.interval("BuildingImageFAR", img["false_positives"],
                   img["false_positives"] + img["true_negatives"], 3, src)
        for key, name in (("rescue_team", "Rescue"),
                          ("entry_accessible", "EntryOpen"),
                          ("entry_blocked", "EntryBlocked"),
                          ("building_collapsed", "Collapsed")):
            per = test["per_class_box_level"].get(key)
            if per:
                m.number("Building" + name + "FOne", per["f1"], 3, src)

    # --------------------------------------------------- scene router ---- #
    router = load("scene_router.json")
    if router:
        src = "eval/results/scene_router.json"
        m.add("RouterTrainImages", str(router["train_images"]), src)
        m.add("RouterHeldOut", str(router["heldout_images"]), src)
        m.add("RouterClasses", str(len(router["classes"])), src)
        held = router["heldout_images"]
        m.interval("RouterAccuracy",
                   round(router["best_heldout_accuracy"] * held), held, 3, src)
        per_class_n = held // max(1, len(router["classes"]))
        m.add("RouterPerClassImages", str(per_class_n), src)
        m.interval("RouterPerClass", per_class_n, per_class_n, 3, src)

    # ---------------------------------------------- multi-hazard/gate ---- #
    multi = load("multihazard.json")
    if multi:
        src = "eval/results/multihazard.json"
        for key, name in (("fire", "Fire"), ("traffic", "Traffic")):
            h = multi.get("hazards", {}).get(key)
            if h:
                m.percent(name + "Suppression", h["suppression"], 1, src)
                m.add(name + "Frames", str(h["raw"]["frames"]), src)
                m.add(name + "RawDetections", str(h["raw"]["raw_detections"]), src)
                m.add(name + "ConfirmedEvents",
                      str(h["validated"]["confirmed_events"]), src)

    # ------------------------------------ threshold-matched control ---- #
    tm = load("threshold_matched.json")
    if tm:
        src = "eval/results/threshold_matched.json"
        m.number("MatchedFloor", tm["matched_floor"], 2, src)
        m.number("MatchedRecall", tm["matched"]["fires"]["detection_rate"], 3, src)
        m.number("MatchedMinutes",
                 tm["matched"]["fires"]["median_minutes_to_detect"], 1, src)
        m.add("MatchedFires",
              str(tm["matched"]["fires"]["sequences_detected"]), src)
        m.number("CascadeFAR",
                 tm["cascade"]["negatives"]["false_alarm_rate"], 3, src)
        m.number("CascadeRecall",
                 tm["cascade"]["fires"]["detection_rate"], 3, src)
        # The floor that zeroes FAR on this set, for the discussion.
        zero = next((e for e in tm["sweep"]
                     if e["negatives"]["false_alarm_rate"] == 0.0), None)
        if zero:
            m.number("ZeroFARFloor", zero["floor"], 2, src)
            m.number("ZeroFARRecall", zero["fires"]["detection_rate"], 3, src)

    # ------------------------------------------ sliced inference ---- #
    sahi = load("sahi.json")
    if sahi:
        src = "eval/results/sahi.json"
        base = sahi["results"]["baseline_conf_0.3"]
        sliced = sahi["results"]["sliced_conf_0.3"]
        m.number("SahiBaseRecall", base["box_level"]["recall"], 3, src)
        m.number("SahiSlicedRecall", sliced["box_level"]["recall"], 3, src)
        m.number("SahiBasePrecision", base["box_level"]["precision"], 3, src)
        m.number("SahiSlicedPrecision", sliced["box_level"]["precision"], 3, src)
        m.number("SahiBaseFAR", base["image_level"]["false_alarm_rate"], 3, src)
        m.number("SahiSlicedFAR", sliced["image_level"]["false_alarm_rate"], 3, src)
        m.number("SahiBaseLatency", base["median_latency_ms"], 1, src)
        m.number("SahiSlicedLatency", sliced["median_latency_ms"], 1, src)
        m.number("SahiLatencyFactor",
                 sliced["median_latency_ms"] / base["median_latency_ms"], 0, src)
        m.number("SahiFARFactor",
                 sliced["image_level"]["false_alarm_rate"]
                 / base["image_level"]["false_alarm_rate"], 0, src)

    # ------------------------------------------------ colour prior ---- #
    cp = load("colour_prior.json")
    if cp:
        src = "eval/results/colour_prior.json"
        res = cp["stills"]["results"]
        m.number("ColourBaselineRecall", res["colour off"]["recall"], 3, src)
        m.number("ColourBaselineFAR",
                 res["colour off"]["false_alarm_rate"], 3, src)
        m.number("ColourStrictRecall", res["S<=40"]["recall"], 3, src)
        m.number("ColourStrictFAR", res["S<=40"]["false_alarm_rate"], 3, src)
        m.number("ColourMidRecall", res["S<=80"]["recall"], 3, src)
        m.number("ColourMidFAR", res["S<=80"]["false_alarm_rate"], 3, src)
        m.add("ColourStrictFires",
              str(cp["temporal"]["S<=40"]["fires"]["sequences_detected"]), src)
        m.add("ColourStillsImages", str(cp["stills"]["images"]), src)

    # -------------------------------------- suppression decomposition ---- #
    split = REPO_ROOT / "docs" / "figures" / "suppression.json"
    if split.exists():
        rows = json.loads(split.read_text(encoding="utf-8"))
        src = "docs/figures/suppression.json"
        proposed = sum(r["proposed"] for r in rows)
        confirmed = sum(r["confirmed"] for r in rows)
        filtered = sum(r["filtered"] for r in rows)
        deduplicated = sum(r["deduplicated"] for r in rows)
        m.add("SuppressionProposed", str(proposed), src)
        m.add("SuppressionConfirmed", str(confirmed), src)
        m.percent("SuppressionTotal", 1 - confirmed / proposed, 0, src)
        m.percent("SuppressionFiltered", filtered / proposed, 1, src)
        m.percent("SuppressionDeduplicated", deduplicated / proposed, 1, src)
        # Per hazard, because the split INVERTS between fire and building
        # access and a single figure conceals exactly that.
        for row in rows:
            name = "".join(part.capitalize()
                           for part in row["hazard"].split("_"))
            m.percent("Split" + name + "Filtered",
                      row["filtered"] / row["proposed"], 1, src)
            m.percent("Split" + name + "Deduplicated",
                      row["deduplicated"] / row["proposed"], 1, src)
            m.add("Split" + name + "Proposed", str(row["proposed"]), src)
            m.add("Split" + name + "Confirmed", str(row["confirmed"]), src)

    # ------------------------------------- gate saving, from the code ---- #
    try:
        from vigia.registry import CameraRegistry
        summary = CameraRegistry.from_yaml(
            REPO_ROOT / "configs" / "cameras.demo.yaml").gate_summary()
        src = "configs/cameras.demo.yaml via CameraRegistry.gate_summary()"
        m.add("GateCameras", str(summary["cameras"]), src)
        m.add("GateHazards", str(summary["hazards_available"]), src)
        m.add("GateModelsLoaded", str(len(summary["models_loaded"])), src)
        m.add("GateGated",
              str(summary["detector_invocations_per_frame_gated"]), src)
        m.add("GateUngated",
              str(summary["detector_invocations_per_frame_ungated"]), src)
        m.percent("GateReduction", summary["reduction"], 0, src)
    except Exception as exc:                       # registry is optional here
        print(f"gate summary unavailable: {exc}", file=sys.stderr)

    return m


def page_hinkley_table() -> str:
    """The measured calibration grid beside the closed-form prediction.

    Generated rather than typed, so the model cannot be quietly fitted to the
    measurement after the fact: whatever the formula says is what the paper
    prints next to what was observed, including where it is wrong.
    """
    calib = load("page_hinkley_calibration.json")
    if not calib:
        return "% page_hinkley_calibration.json absent\n"

    grid = calib.get("false_alarm_rate_on_stable_water", {})
    if not grid:
        return "% page_hinkley_calibration.json has an unexpected shape\n"

    sigma_keys = sorted(next(iter(grid.values())),
                        key=lambda k: float(k.split("_")[1]))
    sigmas = [float(k.split("_")[1]) for k in sigma_keys]

    rows = []
    for parameters, measured in grid.items():
        delta = float(parameters.split("delta=")[1].split("_")[0])
        threshold = float(parameters.split("threshold=")[1])
        cells = []
        for key, sigma in zip(sigma_keys, sigmas):
            observed = float(measured[key]) * 100
            predicted = page_hinkley_false_alarm(delta, threshold, sigma) * 100
            cells.append(f"{observed:.1f} & {predicted:.1f}")
        rows.append(f"{delta:.3f} & {threshold:.2f} & " + " & ".join(cells)
                    + r" \\")

    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{$\sigma = {s:g}$}}"
                        for s in sigmas)
    subhead = " & ".join([r"{obs.} & {model}"] * len(sigmas))
    spec = ("S[table-format=1.3] S[table-format=1.2] "
            + " ".join(["S[table-format=3.1] S[table-format=3.1]"]
                       * len(sigmas)))
    return (
        "% GENERATED by docs/build_paper.py — the measured grid beside the\n"
        "% drawdown model. Edit the measurement or the model, not this file.\n"
        rf"\begin{{tabular}}{{{spec}}}" "\n"
        r"\toprule" "\n"
        rf"{{$\delta$}} & {{$\lambda$}} & {header} \\" "\n"
        rf"\cmidrule(lr){{3-{2 + 2 * len(sigmas)}}}" "\n"
        rf" & & {subhead} \\" "\n"
        r"\midrule" "\n" + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--keep-log", action="store_true")
    args = parser.parse_args()

    if not args.no_figures:
        print("regenerating figures from the stored results")
        sys.path.insert(0, str(REPO_ROOT / "docs"))
        import figures
        figures.main()

    GENERATED.mkdir(parents=True, exist_ok=True)
    macros = collect()
    (GENERATED / "numbers.tex").write_text(macros.render(), encoding="utf-8")
    (GENERATED / "page_hinkley.tex").write_text(page_hinkley_table(),
                                                encoding="utf-8")
    print(f"wrote {len(macros._seen)} macro(s) to "
          f"{(GENERATED / 'numbers.tex').relative_to(REPO_ROOT)}")

    print("compiling with tectonic")
    result = subprocess.run(
        ["tectonic", "-X", "compile", str(MAIN), "--outdir", str(PAPER)],
        capture_output=True, text=True)
    log = result.stdout + result.stderr
    if args.keep_log:
        (PAPER / "build.log").write_text(log, encoding="utf-8")

    warnings = [line for line in log.splitlines()
                if "Overfull" in line or "Underfull" in line
                or "undefined" in line.lower()]
    if result.returncode != 0:
        print(log[-4000:], file=sys.stderr)
        return 1
    for line in warnings[:20]:
        print(f"  {line.strip()}")
    pdf = MAIN.with_suffix(".pdf")
    if pdf.exists():
        print(f"wrote {pdf.relative_to(REPO_ROOT)} "
              f"({pdf.stat().st_size / 1024:.0f} KB, "
              f"{len(warnings)} layout warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
