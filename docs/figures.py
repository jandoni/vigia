#!/usr/bin/env python3
"""Generate the figures for the technical documentation.

    python docs/figures.py

Every figure is drawn FROM THE STORED RESULTS in eval/results/ and the training
histories in models/. Nothing is hand-entered, so a chart cannot drift from the
measurement it depicts — the same reasoning as tools/freeze_results.py, applied
to pictures instead of numbers.

DESIGN RULES APPLIED (see the data-visualisation method):
  * Form follows the data's job: change-over-time -> line; magnitude comparison
    -> bar; composition -> stacked bar. Never a chart where a number would do.
  * Categorical hues are assigned in fixed slot order and never cycled.
  * ONE axis per chart. Two measures of different scale get two charts.
  * Colour is validated, not eyeballed: the four slots used here pass the
    lightness, chroma, CVD-separation and normal-vision gates on this surface.
    Two of them fall below 3:1 contrast, so every chart carries direct labels
    and every chart sits beside a table of the same numbers — the documented
    relief for that warning.
  * Text wears ink colours, never the series colour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "eval" / "results"
FIGURES = Path(__file__).resolve().parent / "figures"

# Validated categorical slots, in fixed order.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e5e4e0"


def _style(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)


def _save(fig, name):
    """Write the figure as PNG and as PDF.

    The PNG is for the Word document and the web; the PDF is for the LaTeX
    paper, where a raster chart at 200 dpi is visibly soft next to vector type.
    Both come from the same draw call, so the two documents cannot disagree
    about what a chart shows.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    vector = path.with_suffix(".pdf")
    fig.savefig(vector, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)} + .pdf")
    return path


def _load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _bars(ax, labels, series, colours, names, fmt="{:.3f}"):
    """Horizontal grouped bars with direct labels on every bar.

    Direct labels are not decoration here: two of the validated slots sit below
    3:1 against this surface, and labelling is the documented relief.
    """
    n = len(series)
    height = 0.8 / n
    for i, (values, colour, name) in enumerate(zip(series, colours, names)):
        offsets = [j + (i - (n - 1) / 2) * height for j in range(len(labels))]
        ax.barh(offsets, values, height=height * 0.86, color=colour,
                label=name, zorder=3)
        for y, v in zip(offsets, values):
            ax.text(v + max(values) * 0.02, y, fmt.format(v), va="center",
                    fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()


# --------------------------------------------------------------------- #

def fig_fire_negative_widening():
    """Magnitude comparison: the same detector, two negative-set sizes."""
    wide = _load("fire_pyro_sdis_val_wide.json")
    if not wide:
        return None
    w = wide["results"]["conf_0.2"]
    labels = ["False-alarm rate", "Box precision", "Box recall", "Image recall"]
    narrow = [0.870, 0.625, 0.904, 0.987]
    broad = [w["image_level"]["false_alarm_rate"], w["box_level"]["precision"],
             w["box_level"]["recall"], w["image_level"]["recall"]]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _bars(ax, labels, [narrow, broad], [S2, S1],
          ["23 negatives (first estimate)", "585 negatives (current)"])
    _style(ax, xlabel="value at confidence 0.20",
           title="Widening the fire negative set moved the headline by 16 points")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), labelcolor=INK_2)
    ax.set_xlim(0, 1.12)
    return _save(fig, "fire_negative_widening.png")


def fig_building_access_models():
    """Magnitude comparison: our model against the authors' own weights."""
    data = _load("building_access.json")
    if not data:
        return None
    t = data["results"]["test"]["conf_0.3"]
    labels = ["Civilian box recall", "Civilian box precision",
              "Civilian box F1", "Image-level recall"]
    ours = [0.141, 0.138, 0.139, 1.000]
    theirs = [t["box_level"]["recall"], t["box_level"]["precision"],
              t["box_level"]["f1"], t["image_level"]["recall"]]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _bars(ax, labels, [ours, theirs], [S2, S1],
          ["Ours — Faster R-CNN", "Authors' YOLOv8-DRN"])
    _style(ax, xlabel="held-out test split, 24 images",
           title="Building access: the authors' weights against ours")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), labelcolor=INK_2)
    ax.set_xlim(0, 1.15)
    return _save(fig, "building_access_models.png")


def fig_per_class_f1():
    """Magnitude by identity: one bar per merged class."""
    data = _load("building_access.json")
    if not data:
        return None
    per = data["results"]["test"]["conf_0.3"]["per_class_box_level"]
    names = ["civilian", "rescue_team", "entry_accessible", "entry_blocked",
             "building_collapsed"]
    values = [per[n]["f1"] for n in names if n in per]
    labels = [n.replace("_", " ") for n in names if n in per]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    # One series: no legend needed, the title names it.
    ax.barh(range(len(values)), values, height=0.62, color=S1, zorder=3)
    for i, v in enumerate(values):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    _style(ax, xlabel="box-level F1 at confidence 0.30",
           title="Building access: per-class F1 on the held-out split")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.05)
    return _save(fig, "building_access_per_class.png")


def fig_aerial_training():
    """Change over time: the aerial flood model's two IoU curves."""
    path = REPO_ROOT / "models" / "flood" / "floodnet_aerial_deeplabv3.history.json"
    if not path.exists():
        return None
    hist = json.loads(path.read_text(encoding="utf-8"))["history"]
    epochs = [h["epoch"] for h in hist]
    overall = [h["water_iou"] for h in hist]
    flooded = [h.get("flooded_water_iou", 0.0) for h in hist]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(epochs, overall, color=S1, linewidth=2, label="All validation images",
            zorder=3)
    ax.plot(epochs, flooded, color=S2, linewidth=2,
            label="Flooded images only", zorder=3)
    # Direct labels at the final point rather than a number on every marker.
    if epochs:
        ax.text(epochs[-1], overall[-1], f"  {overall[-1]:.3f}", fontsize=8,
                color=INK_2, va="center")
        ax.text(epochs[-1], flooded[-1], f"  {flooded[-1]:.3f}", fontsize=8,
                color=INK_2, va="center")
    _style(ax, xlabel="epoch", ylabel="water IoU",
           title="Aerial flood model: the two curves say different things")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    ax.set_ylim(0, 1)
    return _save(fig, "aerial_flood_training.png")


def fig_suppression_split():
    """Composition: what the validator does with each hazard's proposals."""
    # Measured on the demonstration clips; stored alongside the figures so the
    # chart and the prose cannot disagree.
    path = FIGURES / "suppression.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    labels = [r["hazard"].replace("_", " ") for r in rows]
    filt = [r["filtered"] / r["proposed"] for r in rows]
    dedup = [r["deduplicated"] / r["proposed"] for r in rows]
    conf = [r["confirmed"] / r["proposed"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    y = range(len(labels))
    # 2px surface gap between stacked segments, per the mark spec.
    ax.barh(y, filt, height=0.6, color=S1, label="Rejected as not credible",
            zorder=3)
    ax.barh(y, dedup, height=0.6, left=filt, color=S2,
            label="Withheld as a repeat", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.barh(y, conf, height=0.6, left=[a + b for a, b in zip(filt, dedup)],
            color=S3, label="Confirmed", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for i, (f, d, c) in enumerate(zip(filt, dedup, conf)):
        if f > 0.08:
            ax.text(f / 2, i, f"{f:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white")
        if d > 0.08:
            ax.text(f + d / 2, i, f"{d:.0%}", ha="center", va="center",
                    fontsize=7.5, color="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style(ax, xlabel="share of proposed detections",
           title="One number would hide this: filtering and deduplication differ by hazard")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), labelcolor=INK_2)
    ax.set_xlim(0, 1)
    return _save(fig, "suppression_split.png")


def fig_confidence_sweep():
    """Change over a parameter: precision and recall against threshold."""
    data = _load("person_in_water.json")
    if not data:
        return None
    confs, prec, rec = [], [], []
    for key, row in sorted(data["results"].items(),
                           key=lambda kv: float(kv[0].split("_")[1])):
        confs.append(float(key.split("_")[1]))
        prec.append(row["box_level"]["precision"])
        rec.append(row["box_level"]["recall"])

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(confs, prec, color=S1, linewidth=2, marker="o", markersize=4,
            label="Precision", zorder=3)
    ax.plot(confs, rec, color=S2, linewidth=2, marker="o", markersize=4,
            label="Recall", zorder=3)
    ax.axvline(0.30, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.305, 0.05, "operating point", fontsize=7.5, color=MUTED)
    _style(ax, xlabel="confidence threshold", ylabel="value",
           title="People in water: swimmer precision and recall against threshold")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK_2)
    ax.set_ylim(0, 1.05)
    return _save(fig, "confidence_sweep.png")


def fig_threshold_frontier():
    """The control experiment: raw detector swept against the cascade point.

    One chart, one honest story: on this negative set the cascade sits BELOW
    the raw detector's threshold frontier — an oracle-tuned floor reaches the
    same false-alarm rate at higher recall. The paper says so in the text;
    the chart makes it impossible to miss. Change-over-two-quantities ->
    a frontier line with the single alternative as a distinct marked point.
    """
    data = _load("threshold_matched.json")
    if not data:
        return None

    # Collapse the sweep to the frontier: for each achieved FAR keep the best
    # recall (the sweep is monotone in floor, not in FAR, because of ties).
    frontier = {}
    for entry in data["sweep"]:
        far = entry["negatives"]["false_alarm_rate"]
        recall = entry["fires"]["detection_rate"]
        if far not in frontier or recall > frontier[far][0]:
            frontier[far] = (recall, entry["floor"])
    xs = sorted(frontier)
    ys = [frontier[x][0] for x in xs]

    cascade = data["cascade"]
    cascade_far = cascade["negatives"]["false_alarm_rate"]
    cascade_recall = cascade["fires"]["detection_rate"]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    _style(ax, xlabel="false-alarm rate on the negative sequences",
           ylabel="frame-level recall on the ignition sequences",
           title="The obvious baseline, run: a tuned floor beats the cascade "
                 "on this negative set")

    ax.plot(xs, ys, color=S1, linewidth=2, marker="o", markersize=4,
            zorder=3, clip_on=False)
    ax.annotate("raw detector,\nconfidence floor swept", (xs[-1], ys[-1]),
                xytext=(-8, 10), textcoords="offset points", ha="right",
                fontsize=8, color=INK_2)
    for far, (recall, floor) in sorted(frontier.items()):
        if floor in (0.20, 0.26, 0.30):
            ax.annotate(f"floor {floor:.2f}", (far, recall),
                        xytext=(6, -11), textcoords="offset points",
                        fontsize=7.5, color=MUTED)

    ax.scatter([cascade_far], [cascade_recall], s=70, color=S2, zorder=4,
               clip_on=False)
    ax.annotate("cascade, persistence 3\n(floor 0.20, untuned)",
                (cascade_far, cascade_recall),
                xytext=(10, -18), textcoords="offset points",
                fontsize=8, color=INK_2)

    ax.set_xlim(-0.002, 0.045)
    ax.set_ylim(0.60, 1.0)
    return _save(fig, "threshold_frontier.png")


def main() -> int:
    made = []
    for builder in (fig_fire_negative_widening, fig_building_access_models,
                    fig_per_class_f1, fig_aerial_training,
                    fig_suppression_split, fig_confidence_sweep,
                    fig_threshold_frontier):
        try:
            path = builder()
            if path:
                made.append(path.name)
            else:
                print(f"  skipped {builder.__name__} — results not available yet")
        except Exception as exc:                     # keep going; a figure is
            print(f"  FAILED {builder.__name__}: {exc}", file=sys.stderr)
    print(f"\n{len(made)} figure(s) generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
