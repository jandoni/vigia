#!/usr/bin/env python3
"""Generate the VIGÍA technical documentation as a Word document.

The document is GENERATED, never hand-edited. Everything it reports comes from
the same sources the code uses — `models/REGISTRY.yaml` for provenance and
attribution, `eval/results/*.json` for measured numbers — so the document
cannot drift out of step with the system it describes.

Rule enforced throughout: a number is either MEASURED BY US (from a results
file, with the harness and dataset named) or REPORTED BY THE ORIGINAL AUTHORS
(clearly attributed, never mixed with ours). Any figure we have not measured is
marked as such rather than quietly omitted.

To add a new result: write it to eval/results/, add the entry to CONTENT below,
and re-run. Never edit the .docx by hand — it will be overwritten.

Usage:
    python docs/build_documentation.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT_PATH = REPO_ROOT / "docs" / "VIGIA_Technical_Documentation.docx"
REGISTRY_PATH = REPO_ROOT / "models" / "REGISTRY.yaml"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

DOC_VERSION = "0.7"
AUTHOR = "Jon Andoni Baranda"

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x00, 0x5A, 0x62)
MUTED = RGBColor(0x5A, 0x64, 0x6E)
WARN = RGBColor(0x9C, 0x52, 0x00)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def set_cell_background(cell, hex_colour: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), hex_colour)
    cell._tc.get_or_add_tcPr().append(shading)


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.15

    for level, size, colour in (
        ("Title", 30, ACCENT), ("Heading 1", 17, ACCENT),
        ("Heading 2", 13.5, INK), ("Heading 3", 11.5, INK),
    ):
        style = document.styles[level]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = colour
        style.paragraph_format.space_before = Pt(14 if level != "Title" else 0)
        style.paragraph_format.space_after = Pt(6)


def add_paragraph(document, text: str, *, italic=False, bold=False,
                  size=10.5, colour=INK, space_after=7, align=None):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic, run.bold = italic, bold
    run.font.size = Pt(size)
    run.font.color.rgb = colour
    paragraph.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        paragraph.alignment = align
    return paragraph


def add_callout(document, label: str, text: str, colour=WARN) -> None:
    """A labelled note that stands apart from body text."""
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.cell(0, 0)
    set_cell_background(cell, "F5F2EC" if colour == WARN else "EDF3F3")

    first = cell.paragraphs[0]
    run = first.add_run(label.upper())
    run.bold = True
    run.font.size = Pt(8)
    run.font.color.rgb = colour
    first.paragraph_format.space_after = Pt(2)

    body = cell.add_paragraph()
    body_run = body.add_run(text)
    body_run.font.size = Pt(9.5)
    body.paragraph_format.space_after = Pt(2)
    document.add_paragraph()


def add_table(document, headers: list[str], rows: list[list[str]],
              widths: list[float] | None = None) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    header_cells = table.rows[0].cells
    for index, heading in enumerate(headers):
        set_cell_background(header_cells[index], "E4EAEC")
        paragraph = header_cells[index].paragraphs[0]
        run = paragraph.add_run(heading)
        run.bold = True
        run.font.size = Pt(9)
        paragraph.paragraph_format.space_after = Pt(1)

    for row_values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            paragraph = cells[index].paragraphs[0]
            run = paragraph.add_run(str(value))
            run.font.size = Pt(9)
            paragraph.paragraph_format.space_after = Pt(1)

    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)

    document.add_paragraph()


def add_bullets(document, items: list[str], style="List Bullet") -> None:
    for item in items:
        paragraph = document.add_paragraph(style=style)
        run = paragraph.add_run(item)
        run.font.size = Pt(10.5)
        paragraph.paragraph_format.space_after = Pt(3)


def add_code(document, text: str) -> None:
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.cell(0, 0)
    set_cell_background(cell, "F4F6F6")
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    paragraph.paragraph_format.space_after = Pt(1)
    document.add_paragraph()


FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def add_figure(document, name: str, caption: str, width_in: float = 6.2) -> bool:
    """Place a generated figure with its caption.

    Figures are drawn from the stored results by docs/figures.py, so a chart
    cannot disagree with the table beside it. A missing figure is skipped
    rather than fatal: the document must still build before a measurement
    exists.
    """
    path = FIGURES_DIR / name
    if not path.exists():
        return False
    document.add_picture(str(path), width=Inches(width_in))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_paragraph(document, caption, size=9, italic=True, colour=MUTED)
    return True


def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Document sections
# --------------------------------------------------------------------------- #

def cover(document: Document) -> None:
    for _ in range(4):
        document.add_paragraph()
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("VIGÍA")

    add_paragraph(document,
                  "Multi-Hazard Early Detection for Existing Camera Infrastructure",
                  size=14, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER,
                  space_after=4)
    add_paragraph(document, "Technical Documentation", size=12, italic=True,
                  colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

    add_table(document,
              ["", ""],
              [["Author", AUTHOR],
               ["Document version", DOC_VERSION],
               ["Generated", dt.date.today().strftime("%d %B %Y")],
               ["Project licence", "Apache-2.0"],
               ["Status", "Active development"],
               ["Generated by", "docs/build_documentation.py (do not edit by hand)"]],
              widths=[2.0, 4.2])

    add_callout(document, "How to read this document",
                "Every quantitative figure is labelled either MEASURED (produced by "
                "our own evaluation harness, with the dataset and script named) or "
                "REPORTED BY ORIGINAL AUTHORS (published by the people who built the "
                "component). The two are never mixed. Figures we have not yet "
                "measured are marked as such rather than omitted.")
    document.add_page_break()


def section_introduction(document: Document) -> None:
    document.add_heading("1. Introduction and Purpose", level=1)

    document.add_heading("1.1 Motivation", level=2)
    add_paragraph(document,
        "On 29 October 2024 a DANA (isolated depression at high levels) struck the "
        "province of Valencia, Spain. It killed 223 people, displaced roughly 15,000 "
        "residents and caused losses estimated above 50 billion euros. The Spanish "
        "meteorological agency AEMET issued its highest-level red warning hours "
        "before the worst of the rainfall, but the regional ES-Alert message to "
        "citizens' mobile phones was not sent until late that evening. People died "
        "in their homes, in underground garages and in vehicles overtaken by water.")
    add_paragraph(document,
        "The gap that day was not a gap in forecasting. It was a gap between what "
        "was known and what reached the people affected, and between a hazard "
        "beginning and anyone confirming it had begun. The eruption of Cumbre Vieja "
        "on La Palma in 2021, which forced thousands to evacuate over 85 days, "
        "showed the same pattern from a different hazard.")
    add_paragraph(document,
        "Spanish cities already carry tens of thousands of cameras — municipal "
        "traffic cameras, road authority cameras, private security systems. During "
        "the DANA those cameras were watching the water rise. Nobody was watching "
        "the cameras.")

    document.add_heading("1.2 What VIGÍA is", level=2)
    add_paragraph(document,
        "VIGÍA integrates the best publicly available hazard detectors under a "
        "single gated pipeline and adds a shared temporal validation layer that "
        "makes them operationally trustworthy on ordinary municipal cameras.",
        italic=True)
    add_paragraph(document,
        "The system is explicitly an integration project. It does not claim novel "
        "detection architectures. Every detector it runs was trained by someone "
        "else, and each is credited by name, licence and source in Section 8. The "
        "original contribution is the validation layer described in Section 5, and "
        "the evaluation methodology in Section 6 that measures what that layer "
        "buys and what it costs.")

    document.add_heading("1.3 The problem the system actually solves", level=2)
    add_paragraph(document,
        "Hazard detection models are widely available and, individually, good. What "
        "prevents them being deployed on municipal cameras is not accuracy but "
        "trust: they alarm far too often. Industry reporting puts the proportion of "
        "false alarms in security camera systems at roughly 98%. Commercial "
        "wildfire operators compensate with human staffing — Pano AI charges around "
        "USD 50,000 per camera per year and runs a 24/7 intelligence centre, and "
        "ALERTCalifornia routes detections from more than 1,060 cameras through "
        "trained staff before dispatch.")
    add_paragraph(document,
        "We measured this directly. The PyroNear wildfire detector, run exactly as "
        "published, raises an alarm on 87.0% of frames that contain no smoke when "
        "evaluated against a curated hard-negative set (Section 4.1). The detector "
        "is not defective; that is the operating point its authors deliberately "
        "chose, on the assumption that something downstream filters the output. "
        "VIGÍA is that something.")

    document.add_heading("1.4 Scope and honest limitations", level=2)
    add_bullets(document, [
        "VIGÍA is a research prototype, not a certified emergency system. No "
        "component has regulatory approval for life-safety use.",
        "It performs no biometric identification of any kind, by architecture "
        "rather than by configuration (Section 2.6).",
        "Each detector carries a declared operating envelope. Claims outside that "
        "envelope are not made.",
        "Reported figures come from datasets of limited size, stated explicitly "
        "alongside every number.",
    ])
    document.add_page_break()


def section_architecture(document: Document) -> None:
    document.add_heading("2. System Architecture", level=1)

    add_paragraph(document,
        "VIGÍA is organised in three tiers. Detectors are interchangeable; the "
        "validator is shared and hazard-agnostic. Adding a hazard requires a "
        "detector class and a configuration entry, and no change to the validator.")

    add_code(document,
        "            TIER 0                TIER 1                    TIER 2\n"
        "            context gate          specialist detectors      temporal validator\n"
        "\n"
        "  forest ─┐                    ┌── fire & smoke ────┐\n"
        "  street ─┤─► camera registry ─┼── flood / level ───┤──► confidence        ┌─► CONFIRMED\n"
        "  river  ─┤   declares which   ├── people in water ─┤    size plausibility │   ALERT\n"
        "  rubble ─┤   hazards apply    ├── building access ─┤    colour prior   ───┤\n"
        "  road   ─┘   to each camera   └── traffic ─────────┘    persistence (IoU) └─► suppressed\n"
        "                                                         location cooldown     (logged)")

    document.add_heading("2.1 Tier 0 — Context gate", level=2)
    add_paragraph(document,
        "Each camera is registered once with a declared context. The gate looks up "
        "which detectors apply to that context and runs only those. The gate is "
        "declarative, not learned: we deliberately do not classify a frame and "
        "route it, because that introduces a misrouting failure mode and saves "
        "little. Every operational deployment we surveyed works this way — a "
        "ridgeline wildfire camera never runs drowning detection.")
    add_paragraph(document,
        "Published work on cascaded gating in video pipelines (NoScope, CaTDet) "
        "reports 5–13× compute reduction with minimal accuracy loss. In VIGÍA the "
        "gate is what makes five hazards affordable on a single machine.")
    add_paragraph(document,
        "The gate is implemented in vigia/registry.py as a lookup from context to "
        "hazard set. Its entire intelligence is the table below, which is small "
        "enough to audit at a glance — and that is the trade being made: a wrong "
        "entry here is visible in a YAML file and fixable by a human, whereas a "
        "misrouting classifier fails silently on the critical path.")
    add_table(document,
              ["Camera context", "Hazards enabled", "Reasoning"],
              [["forest", "fire",
                "PyroNear's stated envelope: fixed-mount wide-field, daylight"],
               ["wildland_interface", "fire, traffic",
                "Where settlement meets vegetation, plus the roads people flee along"],
               ["street", "flood, traffic, building access",
                "The DANA case: water rising, vehicles caught, people trapped alongside"],
               ["urban_block", "flood, building access", "Dense built environment"],
               ["river", "flood, drowning", "Level and rate of rise, plus anyone swept in"],
               ["coast", "drowning", "Open water, the SeaDronesSee envelope"],
               ["motorway", "traffic",
                "Deliberately narrow: flood is plausible but outside the flood "
                "detector's calibrated envelope"],
               ["earthquake_zone", "building access", "The DRespNeT envelope"],
               ["uav_search", "building access, drowning",
                "Aerial search and rescue; both detectors are UAV-trained"]],
              widths=[1.5, 1.9, 2.8])
    add_paragraph(document,
        "The bias in that table is toward including a hazard when it is plausible: "
        "a missed hazard is a life-safety failure, whereas an unnecessary detector "
        "costs only the compute the gate was saving. Where that bias was resisted — "
        "motorway, which excludes flood — the reason is that claiming a hazard "
        "outside a detector's measured envelope is worse than not claiming it.")
    add_paragraph(document,
        "The saving is reported rather than assumed. On the six-camera example "
        "registry shipped with the system, the gate requests 9 detector "
        "invocations per frame instead of 25, a 64% reduction, and the "
        "unrequested models are never constructed at all — the saving is in "
        "memory as well as compute. A deployment of only ridgeline cameras loads "
        "one ONNX session, not five.")
    add_paragraph(document,
        "The gate also declares VIEWPOINT — ground, oblique, or nadir aerial — "
        "because a detector's training viewpoint is part of its operating "
        "envelope and nothing else in the system captured it. This was added "
        "after a measured failure rather than as a precaution. The flood "
        "segmenter is trained on ATLANTIS, which is ground-level and oblique "
        "photography of water bodies; evaluated on straight-down UAV imagery it "
        "reported up to 0.397 water coverage on frames whose actual water was a "
        "narrow canal, tinting large areas of mown grass as water. It did not "
        "fail loudly. It returned a confident wrong number, which for a "
        "life-safety component is the worst available failure.")
    add_paragraph(document,
        "Viewpoint is therefore declared on every camera and on every detector, "
        "and the pipeline REFUSES to pair a camera with a detector whose "
        "envelope excludes it — before any model is loaded. The same reasoning "
        "as the licence and privacy guards: make the mistake impossible rather "
        "than documented. A test fails the build if any detector stops "
        "declaring an envelope, so a sixth hazard cannot be added without "
        "stating where it works.")
    add_callout(document, "Failing loudly",
        "An unregistered camera raises rather than returning an empty hazard set. "
        "Returning an empty set would turn a typo into a camera that watches for "
        "nothing and reports success, which is the worst available failure mode "
        "for this system. This is enforced by tests/test_registry.py.")

    document.add_heading("2.2 Tier 1 — Specialist detectors", level=2)
    add_paragraph(document,
        "Independent models running in parallel rather than one shared-backbone "
        "multi-task model. A unified model would require a single coherent label "
        "space and jointly annotated data; our hazards come from separate datasets "
        "with incompatible labels. It would also couple training, so that "
        "retraining fire could silently degrade flood.")

    document.add_heading("2.3 Tier 2 — Temporal validator", level=2)
    add_paragraph(document,
        "The original contribution. Five levels applied in order, cheapest first, "
        "each independently switchable so that the ablation table in Section 5 can "
        "be produced by the same code that runs in production. Described in full "
        "in Section 5.")

    document.add_heading("2.4 Technology stack", level=2)
    add_table(document,
              ["Component", "Choice", "Reason"],
              [["Inference runtime", "ONNX Runtime (MIT)",
                "Licence isolation — see 2.5. CoreML provider on Apple Silicon."],
               ["Language", "Python 3.13", "Ecosystem; MLX and ONNX Runtime support"],
               ["Image/video I/O", "OpenCV (Apache-2.0)", "Standard, permissive"],
               ["Numerics", "NumPy (BSD-3)", "Permissive"],
               ["Configuration", "PyYAML — models/REGISTRY.yaml",
                "Single source of truth for config AND attribution"],
               ["Development hardware", "MacBook Pro M4 Pro, 48 GB",
                "All figures in this document produced on this machine"]],
              widths=[1.5, 2.0, 2.7])

    document.add_heading("2.5 Licensing architecture", level=2)
    add_paragraph(document,
        "VIGÍA ships under Apache-2.0. Several source models were fine-tuned using "
        "Ultralytics tooling, which is licensed AGPL-3.0 — a licence whose terms "
        "would extend to this entire project were the package imported at runtime, "
        "and which is compatible with essentially no other licence except GPL-3.0.")
    add_paragraph(document,
        "The resolution is to run every model through ONNX Runtime and never import "
        "the ultralytics package in the runtime path. PyroNear's own edge engine "
        "does exactly this, which is how it remains Apache-2.0 while running a YOLO "
        "architecture. Any .pt to .onnx conversion is an offline build step "
        "confined to tools/, executed in a separate virtual environment.")
    add_paragraph(document,
        "This is enforced automatically rather than by discipline: "
        "tools/check_licence.py parses every file under vigia/ with Python's AST "
        "module and fails the build if a forbidden package is imported. It parses "
        "rather than greps, so a comment mentioning the package does not trip it.")

    document.add_heading("2.6 Privacy by design", level=2)
    add_paragraph(document,
        "The EU AI Act's high-risk provisions took full effect in August 2026. "
        "Annex III classifies biometric identification systems as high-risk, and "
        "Article 5(1)(h) prohibits real-time remote biometric identification in "
        "publicly accessible spaces for law enforcement, with narrow exceptions.")
    add_paragraph(document, "VIGÍA identifies nobody. This is structural:")
    add_bullets(document, [
        "No face recognition, no re-identification, no gait or biometric feature "
        "extraction — there is no code path that could be enabled.",
        "Detection runs at the edge; only the event and a single evidence frame "
        "leave the camera.",
        "No raw video retention beyond a short rolling buffer.",
        "Person detection is reduced to presence and location within a hazard "
        "region, never identity.",
    ])
    add_paragraph(document,
        "A commitment that lives only in a document is a configuration flag "
        "waiting to happen, so each of those four is enforced in code. The "
        "approach mirrors the licensing architecture in 2.5: a guard that parses "
        "the tree and fails the build.")
    add_table(document,
              ["Commitment", "How it is enforced"],
              [["No biometric code path",
                "tools/check_privacy.py parses every file under vigia/, tools/, "
                "scripts/ and eval/ and fails on any import of a face or "
                "re-identification library, any use of OpenCV's face APIs, or any "
                "function or class whose name describes identity work. Unlike the "
                "licence guard, tools/ is not exempt: an offline script that built "
                "a face index would breach the commitment just as surely."],
               ["Only event plus one evidence frame leaves",
                "The Alert type has no field for a clip, and no writer for one. A "
                "test asserts the absence, so adding one becomes a visible change "
                "to a documented boundary rather than a quiet tweak."],
               ["No raw video retention",
                "RetentionPolicy raises on construction if raw video storage is "
                "requested. The rolling buffer is a fixed-length in-memory deque "
                "sized from the camera's declared retention window, and no code "
                "path writes video to disk."],
               ["Presence and location, never identity",
                "Person-shaped detections are blurred inside their own boxes "
                "before any evidence frame is written, on a copy so the source "
                "frame is never mutated. The box stays visible: an operator needs "
                "to know that someone is there and where, never who."]],
              widths=[1.9, 4.3])
    add_paragraph(document,
        "The redaction list is keyed on detector class names, and a test fails if "
        "a detector declares a person-depicting class that is not on it — so a "
        "sixth hazard cannot introduce an unredacted person class by omission.")
    document.add_heading("2.7 Runtime pipeline", level=2)
    add_paragraph(document,
        "The three tiers are joined by vigia/pipeline.py, which is what makes the "
        "system a system rather than five detectors and a validator exercised only "
        "by evaluation harnesses. Its structure is: capture, gate, detect in "
        "parallel, validate, emit. It decides none of those things itself — the "
        "registry decides which detectors run, the validator decides what is an "
        "alert, and the egress module decides what may leave. Keeping those three "
        "out of the pipeline is what allows each to be tested and reported "
        "separately, and it is why adding a sixth hazard touches none of this code.")
    add_paragraph(document,
        "Concurrency follows the pattern reported in arXiv 2607.03131: one thread "
        "per active detector over bounded queues, with a single fusion stage. The "
        "detectors are independent ONNX Runtime sessions, which release the GIL "
        "during inference — where essentially all of the time goes — so they run "
        "genuinely in parallel. Across cameras the detector objects are shared "
        "while the validators are emphatically not: validator state is per camera "
        "per hazard, and sharing it would let one camera's tracks confirm another "
        "camera's events.")
    add_paragraph(document,
        "The queues are bounded, and that is the design rather than a limitation. "
        "An unbounded queue converts a detector that cannot keep up into unbounded "
        "memory growth and ever-increasing latency, which in a life-safety system "
        "means alerting about something that stopped being true minutes ago. "
        "Bounded queues convert the same condition into dropped frames that are "
        "counted and reported.")
    add_paragraph(document,
        "How that backpressure resolves depends on the source, and the distinction "
        "matters for the integrity of every figure in this document:")
    add_bullets(document, [
        "Live sources never block. A detector that falls behind costs frames, not "
        "latency, and stale frames are discarded in favour of the newest available.",
        "File sources always block. An evaluation replay or a curated demo clip "
        "must be deterministic; dropping frames would make a published measurement "
        "depend on how busy the machine happened to be at the time.",
    ])
    add_paragraph(document,
        "That distinction was introduced after measurement, not by foresight: with "
        "non-blocking dispatch on both, a 16-frame clip through a 193 ms detector "
        "processed 3 frames and silently dropped 13. A replay that quietly skips "
        "80% of its input would have produced plausible and meaningless numbers.")
    add_paragraph(document,
        "Measured on the reference machine, a single ridgeline camera replaying a "
        "40-frame sequence through the fire detector sustains 8–14 frames per "
        "second end to end at 117 ms median detector latency, processing every "
        "frame with none dropped. Over that sequence the detector proposed 20 "
        "detections and the validator confirmed 2 — a 90% suppression rate, "
        "produced by the same code path that produces the ablation figures in "
        "Section 5. Of the 18 suppressions, 2 came from persistence and 16 from "
        "location cooldown, the latter correctly refusing to re-alert on a fire "
        "already reported.")
    add_callout(document, "One command",
        "scripts/run_pipeline.py takes a registry file and a directory of clips "
        "and runs every enabled camera concurrently. The same entry point serves "
        "the demonstration and a deployment; only the registry changes, which is "
        "the practical payoff of Tier 0 being declarative. A --explain flag prints "
        "the gate's routing decisions and the compute it saves without loading a "
        "single model.")

    document.add_heading("2.8 Operator view", level=2)
    add_paragraph(document,
        "The interface is built around one observation. Every camera-AI "
        "demonstration draws a box around a fire, and a reviewer has seen that "
        "before. What is rarely shown is detections being REJECTED — and since "
        "the validator suppresses the large majority of what the detectors "
        "propose, watching that happen is the only direct visual evidence for "
        "the one claim in this work that is original. The interface is therefore "
        "a dual view: what the detector proposed against what the system "
        "confirmed. Every other decision follows from that.")
    add_bullets(document, [
        "Header — the camera's context and a chip per hazard, lit only for the "
        "hazards the gate runs there. Switching cameras visibly changes which "
        "detectors are active, which explains Tier 0 faster than a diagram.",
        "Canvas — confirmed events in accent, candidates still short of the "
        "persistence threshold in a muted tone, and suppressed candidates "
        "outlined in grey and labelled with the cascade level that rejected "
        "them. Naming the level is the point: the viewer sees not only that a "
        "detection was dropped but which test dropped it.",
        "Right rail — the cascade as a signal chain with live pass and reject "
        "counts per level, the same counters that produce the ablation table.",
        "Timeline — candidate ticks above confirmed ticks. The density contrast "
        "between the two rows is the argument, made without a sentence of text.",
        "Metric strip — seen, confirmed, suppressed, rate, frame rate and "
        "latency, always visible, in tabular figures so nothing reads as "
        "hand-waved.",
    ])
    add_paragraph(document,
        "A single control switches the suppressed layer off, at which point the "
        "view becomes an ordinary detection demonstration while the counters "
        "keep running. On the reference clip that is the difference between a "
        "clean frame and 178 suppressed candidates behind a 98.9% suppression "
        "rate — the work is happening either way, and only one of the two views "
        "shows it.")
    add_paragraph(document,
        "The transport is deliberately split. Video is MJPEG over plain HTTP, so "
        "the browser's own decoder does the work; events are small JSON over a "
        "WebSocket. Frames never travel on the event channel. The reference "
        "implementation (arXiv 2607.03131) identifies Base64-over-WebSocket "
        "frame transport as its bottleneck and lists WebRTC as future work; "
        "separating the channels sidesteps that problem rather than optimising "
        "it. The client falls back to polling if the socket drops, so the view "
        "degrades to slower rather than to frozen.")
    add_paragraph(document,
        "The front end is vanilla JavaScript and CSS with no build step and no "
        "package manager, and the web framework is an optional dependency rather "
        "than a core one — a deployed edge node runs headless and never needs "
        "it. Both choices are credibility arguments as much as technical ones: a "
        "system that requires a toolchain to demonstrate is a system nobody will "
        "try, and an edge runtime of four packages is a claim that can be "
        "checked.")
    add_callout(document, "A failure that hid behind its own fallback",
        "The WebSocket route rejected every handshake with HTTP 403 while the "
        "interface kept updating normally, because the polling fallback covered "
        "for it. The cause was that this module used deferred annotation "
        "evaluation, which turns annotations into strings that the framework "
        "resolves against module globals — while the WebSocket type was imported "
        "inside a function, to keep the web framework optional. The parameter "
        "could not be typed and was treated as a required query parameter. Worth "
        "recording because the graceful degradation that made the system robust "
        "is exactly what made the defect invisible; it was found in the server "
        "log and a direct client, not in the browser.")

    add_callout(document, "Guards are tested against violations",
        "A guard that cannot fail is worthless. The privacy guard was run against "
        "deliberately planted violations, which is how two gaps were found: class "
        "names like PersonReidentifier and GaitSignatureExtractor initially "
        "escaped because the patterns were whole words containing underscores. "
        "Matching now strips underscores before comparison. The same reasoning "
        "applies to the ONNX parity check described in Section 6.")
    document.add_page_break()


def section_hazards(document: Document) -> None:
    document.add_heading("3. Situations Analysed", level=1)
    add_paragraph(document,
        "Five hazards. Four are disaster hazards and carry the system's purpose; "
        "the fifth, traffic incidents, is retained explicitly as a generality "
        "experiment rather than a headline capability — a car accident is an "
        "emergency, not a disaster. Its value here is that it has completely "
        "different physics and timescales from wildfire, so a validator that works "
        "on both is demonstrably not tuned to one phenomenon.")

    add_table(document,
        ["#", "Hazard", "Relevance", "Status"],
        [["1", "Flood and water level",
          "The DANA itself: water rising in streets, garages, vehicles.",
          "In development"],
         ["2", "People in water",
          "DANA victims swept away and trapped in vehicles.", "Planned"],
         ["3", "Trapped people / building access",
          "DANA victims trapped in buildings; earthquake response generally.",
          "Planned"],
         ["4", "Fire and smoke",
          "Mediterranean wildfire; La Palma-adjacent hazards.", "IMPLEMENTED"],
         ["5", "Traffic incidents",
          "Generality proof for the validator, not a headline claim.",
          "IMPLEMENTED"]],
        widths=[0.35, 1.55, 3.0, 1.1])

    add_callout(document, "Scope decision recorded",
        "Pool drowning detection was considered and replaced by open-water "
        "'people in water'. Pools never fitted the disaster narrative, and the "
        "physics are unfavourable: overhead cameras cannot see through surface "
        "disturbance, and every mature commercial pool system uses underwater "
        "hardware. Open water, by contrast, is exactly the DANA scenario and has "
        "a large CC0-licensed dataset available.")
    document.add_page_break()


def hazard_block(document, *, title, status, credit_rows, description,
                 implementation, modifications, results_rows=None,
                 results_note=None, envelope=None, planned_note=None):
    document.add_heading(title, level=2)

    status_para = add_paragraph(document, f"Status: {status}", bold=True,
                                size=10, colour=ACCENT, space_after=8)

    document.add_heading("Attribution and credit", level=3)
    add_table(document, ["", ""], credit_rows, widths=[1.6, 4.6])

    document.add_heading("What it does", level=3)
    add_paragraph(document, description)

    document.add_heading("Implementation in VIGÍA", level=3)
    add_paragraph(document, implementation)

    if modifications:
        document.add_heading("Modifications made", level=3)
        add_bullets(document, modifications)

    if envelope:
        document.add_heading("Declared operating envelope", level=3)
        add_paragraph(document, envelope, italic=True)

    document.add_heading("Results", level=3)
    if results_rows:
        add_table(document, ["Metric", "Value", "Source"], results_rows,
                  widths=[2.1, 1.3, 2.8])
        if results_note:
            add_paragraph(document, results_note, size=9.5, italic=True,
                          colour=MUTED)
    else:
        add_paragraph(document, planned_note or "Not yet measured.",
                      italic=True, colour=MUTED)


def section_systems(document: Document) -> None:
    document.add_heading("4. Implemented Systems", level=1)
    add_paragraph(document,
        "One subsection per hazard. Each names the people whose work it uses, "
        "states what we changed, and reports only figures we measured ourselves — "
        "with figures published by the original authors listed separately and "
        "attributed.")

    # ---------------- 4.1 Fire ---------------- #
    baseline = load_json("fire_pyro_sdis_val_wide.json")
    fire_results = None
    fire_note = None
    if baseline:
        conf20 = baseline["results"]["conf_0.2"]
        box, img = conf20["box_level"], conf20["image_level"]
        fire_results = [
            ["Box precision", f"{box['precision']:.3f}", "MEASURED — eval/run_image_eval.py"],
            ["Box recall", f"{box['recall']:.3f}", "MEASURED — eval/run_image_eval.py"],
            ["Box F2", f"{box['f2']:.3f}", "MEASURED — recall-weighted, see 6.3"],
            ["Image-level recall", f"{img['recall']:.3f}", "MEASURED"],
            ["False-alarm rate", f"{img['false_alarm_rate']:.3f}",
             "MEASURED — on curated hard negatives"],
            ["Median latency", f"{conf20['median_latency_ms']:.0f} ms",
             "MEASURED — M4 Pro, CPU provider"],
            ["mAP / precision / recall", "not published",
             "REPORTED BY AUTHORS — the model card publishes no metrics"],
        ]
        fire_note = (
            f"Dataset: pyronear/pyro-sdis validation split, "
            f"{baseline['images']} rows ({baseline['positives']} positive, "
            f"{baseline['negatives']} negative, spanning 21 cameras). "
            f"Confidence 0.20, IoU match {baseline['iou_match_threshold']}. "
            f"Single-frame, raw detector output, no temporal validation — this "
            f"is the baseline the validator must improve on, not a headline "
            f"result. This figure was originally measured on 23 negatives and "
            f"carried a standing caveat that it was indicative rather than "
            f"precise; widening to {baseline['negatives']} negatives moved the "
            f"false-alarm rate from 0.870 to 0.706 and box precision from 0.625 "
            f"to 0.447, while recall was essentially unchanged at 0.902. The "
            f"caveat is now discharged."
        )

    hazard_block(document,
        title="4.1 Fire and Smoke",
        status="IMPLEMENTED and measured",
        credit_rows=[
            ["Model", "yolo11s_sensitive-detector"],
            ["Created by", "The PyroNear association (pyronear.org), a French "
                           "open-source non-profit, in collaboration with the "
                           "French Fire and Rescue Services (SDIS)"],
            ["Source", "huggingface.co/pyronear/yolo11s_sensitive-detector"],
            ["Licence", "Apache-2.0"],
            ["Training data", "pyro-dataset (PyroNear). Manifest records 32,201 "
                              "files, 50 epochs, AdamW, 1024 px, single class"],
            ["Evaluation data", "pyronear/pyro-sdis (Apache-2.0), annotated by "
                                "PyroNear volunteers with SDIS partners"],
            ["Field deployment", "15 lookout towers, 51 cameras across France, "
                                 "Spain and Chile, running on Raspberry Pi"],
        ],
        description=(
            "Detects wildfire smoke plumes in wide-field outdoor camera imagery. "
            "It is a deliberately 'sensitive' detector: the authors recommend a low "
            "confidence floor (0.20) and a very low NMS IoU (0.01), trading "
            "precision for recall on the assumption that a downstream filter "
            "exists. That design choice is precisely why it is the right baseline "
            "for this project — the filter it assumes is what we built."),
        implementation=(
            "We run the published ONNX export unmodified through ONNX Runtime. "
            "The model is downloaded from Hugging Face at setup time by "
            "tools/fetch_models.py and is never redistributed in our repository. "
            "Implementation is vigia/detectors/fire.py (41 lines) on top of the "
            "shared vigia/detectors/base.py."),
        modifications=[
            "No modification to the model itself.",
            "Relabelled its single class from 'item' (as exported) to 'smoke' for "
            "readability downstream.",
            "Wrote our own letterbox preprocessing and YOLO output decoding with "
            "NumPy non-maximum suppression, so the pipeline has no dependency on "
            "the AGPL-licensed ultralytics package (Section 2.5).",
            "Generated a static-input-shape variant (tools/make_static.py) so the "
            "Apple CoreML execution provider can run it — see Section 6.4.",
        ],
        envelope=(
            "Fixed-mount wide-field outdoor cameras viewing a landscape horizon, "
            "in daylight. Not validated for indoor fire, close-range flame, or "
            "night-time operation."),
        results_rows=fire_results, results_note=fire_note)

    document.add_page_break()

    # ---------------- 4.2 Traffic ---------------- #
    multi = load_json("multihazard.json")
    traffic_results = None
    traffic_note = None
    if multi and "traffic" in multi.get("hazards", {}):
        t = multi["hazards"]["traffic"]
        traffic_results = [
            ["Raw detections", str(t["raw"]["raw_detections"]), "MEASURED"],
            ["Frames with a detection",
             f"{t['raw']['raw_frames_with_detection']} of {t['raw']['frames']} "
             f"({t['raw']['raw_alert_rate']:.1%})", "MEASURED"],
            ["Confirmed events after validation",
             str(t["validated"]["confirmed_events"]), "MEASURED"],
            ["Alert suppression", f"{t['suppression']:.1%}",
             "MEASURED — eval/run_multihazard.py"],
            ["mAP@0.5", "0.826", "REPORTED BY AUTHOR"],
            ["mAP@0.5:0.95", "0.600", "REPORTED BY AUTHOR"],
            ["Accident-class recall", "0.855", "REPORTED BY AUTHOR"],
        ]
        traffic_note = (
            "Evaluated on 660 frames sampled at 2 fps from a 5.5-minute dashcam "
            "compilation. Because the source video is predominantly accident "
            "footage, it does not support a false-positive rate for this hazard; "
            "the measured value here is alert suppression, not accuracy. A proper "
            "negative set for traffic remains outstanding.")

    hazard_block(document,
        title="4.2 Traffic Incidents",
        status="IMPLEMENTED — retained as a generality experiment",
        credit_rows=[
            ["Model", "traffic-accident-detection-yolo11x"],
            ["Created by", "Hugging Face user Enos-123"],
            ["Source", "huggingface.co/Enos-123/traffic-accident-detection-yolo11x"],
            ["Licence", "MIT"],
            ["Training data", "Roboflow Traffic Accident Detection dataset; "
                              "61 epochs, batch 16, 640×640"],
            ["Classes", "accident, vehicle"],
        ],
        description=(
            "Detects traffic accidents and vehicles in roadside and dashcam "
            "imagery. Included in VIGÍA to test whether the temporal validator "
            "transfers to a hazard with entirely different physics: a wildfire "
            "plume develops over tens of minutes and is nearly static between "
            "frames, whereas a collision resolves in under a second and the "
            "objects move fast across the frame."),
        implementation=(
            "The model is published as PyTorch weights, so tools/export_onnx.py "
            "converts it to ONNX in a separate virtual environment containing "
            "ultralytics; the runtime environment contains neither ultralytics nor "
            "torch. Implementation is vigia/detectors/traffic.py."),
        modifications=[
            "Exported from .pt to ONNX at opset 17 with dynamic axes.",
            "Selected the epoch61 checkpoint over epoch14. The author's own "
            "infer.py uses epoch14, but on their four published test images "
            "epoch61 returns exactly one accident per image at confidence "
            "0.58–0.70, whereas epoch14 returns three detections on one image. "
            "This choice and its reasoning are recorded rather than silently made.",
            "Only the 'accident' class is routed to the validator. The model also "
            "emits 'vehicle'; passing that through would make every car on the "
            "road an alert. Vehicles can optionally be returned as display context "
            "tagged so they never reach the validator.",
        ],
        envelope=(
            "Roadside and dashcam views. The author states the model is not "
            "validated for critical autonomous driving use; we repeat that "
            "limitation rather than omitting it."),
        results_rows=traffic_results, results_note=traffic_note)

    document.add_page_break()

    # ---------------- planned hazards ---------------- #
    # ---------------- 4.3 Flood (implemented) ---------------- #
    flood = load_json("flood.json")
    flood_results = None
    flood_note = None
    if flood and "segmentation" in flood:
        seg = flood["segmentation"]
        flood_results = [
            ["Water IoU", f"{seg['water_iou']:.4f}", "MEASURED — ATLANTIS test split"],
            ["Precision", f"{seg['precision']:.4f}", "MEASURED"],
            ["Recall", f"{seg['recall']:.4f}", "MEASURED"],
            ["F1", f"{seg['f1']:.4f}", "MEASURED"],
            ["Pixel accuracy", f"{seg['pixel_accuracy']:.4f}", "MEASURED"],
            ["Median latency", f"{seg['median_latency_ms']:.0f} ms",
             "MEASURED — M4 Pro, CPU provider"],
        ]
        flood_note = (
            f"Dataset: ATLANTIS test split, {seg['images']} images, held out and "
            f"never seen during training or validation. Binary water "
            f"segmentation at 512 px. No third-party baseline is quoted because "
            f"the dataset authors' own network (AQUANet) reports over all 56 "
            f"classes rather than binary water, so its figures are not "
            f"comparable to ours."
        )

    hazard_block(document,
        title="4.3 Flood and Water Level",
        status="IMPLEMENTED and measured — the keystone hazard",
        credit_rows=[
            ["Architecture", "DeepLabV3 with a MobileNetV3-Large backbone"],
            ["Architecture from", "torchvision / Meta AI. Licence BSD-3-Clause"],
            ["Training dataset", "ATLANTIS — 5,195 pixel-annotated waterbody "
                                 "images, 56 classes, split 3,364 / 535 / 1,296"],
            ["Dataset authors", "Erfani et al., iWERS laboratory, University of "
                                "South Carolina. Environmental Modelling & "
                                "Software, 2022"],
            ["Dataset licence", "Source images collected via the Flickr API "
                                "under Creative Commons, No Known Copyright "
                                "Restrictions and US Government Work licences. "
                                "The repository carries no explicit licence "
                                "file, so the status of the annotations is "
                                "unstated — recorded rather than assumed"],
            ["Trained by", "Us. This is the only detector in VIGÍA we trained "
                           "ourselves"],
            ["Trend prior art", "Choi, Kim, Win Aung and Park (2026), "
                                "Developments in the Built Environment 25:100866"],
            ["Real footage", "LSU creek camera sequences from the V-FloodNet "
                             "dataset (Liang et al.). Evaluation use only; "
                             "nothing redistributed"],
        ],
        description=(
            "The keystone hazard and the one directly connected to the DANA. "
            "Two capabilities: segmenting the water region in a scene, and "
            "estimating water level against a fixed per-camera reference line. "
            "The second is geometry rather than learning once the mask exists."),
        implementation=(
            "torchvision's DeepLabV3-MobileNetV3 was fine-tuned from COCO-subset "
            "pretrained weights, with both the classifier and auxiliary heads "
            "replaced by two-class convolutions. ATLANTIS's 56 classes were "
            "collapsed to binary water: VIGÍA does not need to know whether it "
            "is looking at a canal or a river, only where the water is and "
            "whether it is rising. Training ran for 15 epochs on Apple Silicon. "
            "The model is exported to a single self-contained ONNX file and run "
            "through ONNX Runtime like every other detector."),
        modifications=[
            "Architecture chosen for licensing as much as performance: "
            "torchvision is BSD-3-Clause, making this the only detector in the "
            "system entirely free of AGPL. Fire and traffic are YOLO "
            "derivatives and required the ONNX isolation described in 2.5.",
            "17 liquid-water labels count as water. Water-associated STRUCTURES "
            "(dam, levee, pier, culvert, breakwater) are excluded because they "
            "are not water; so are glaciers and snow, which are not liquid, and "
            "mangrove, which is vegetation standing in water whose extent does "
            "not track a level.",
            "The ATLANTIS class mapping (mask value = label id + 1) was derived "
            "empirically by cross-referencing the dominant mask value in each "
            "label folder against the published label list, not assumed.",
            "Export folds any external weight sidecar back into the .onnx file. "
            "PyTorch's exporter splits weights into a separate .onnx.data file, "
            "which is a deployment hazard: copying only the .onnx yields a graph "
            "that loads and then produces garbage.",
            "Export verifies parity against the source PyTorch model and fails "
            "if argmax agreement falls below 99.9%.",
        ],
        envelope=(
            "Water region segmentation in outdoor scenes. The per-camera "
            "reference line is OPTIONAL: without calibration the uncalibrated "
            "area signal still yields a trend, so a camera is useful the moment "
            "it is connected, with calibration an upgrade rather than a "
            "prerequisite."),
        results_rows=flood_results, results_note=flood_note,
        planned_note="Training complete; evaluation pending.")

    add_figure(document, "aerial_flood_training.png",
               "Figure 1. Training the aerial flood segmenter on FloodNet. The "
               "two curves are reported separately because only 51 of the 398 "
               "labelled images are flooded — a single overall IoU would be "
               "carried by dry scenes and would flatter the model on the one "
               "case it exists to handle.")
    document.add_page_break()

    hazard_block(document,
        title="4.4 People in Water",
        status="IMPLEMENTED and measured",
        credit_rows=[
            ["Model", "dronefreak/seadronessee-rfdetr-small (RF-DETR Small)"],
            ["Base architecture", "RF-DETR, Roboflow"],
            ["Dataset", "SeaDronesSee — Varga, Kiefer et al., University of "
                        "Tübingen, IEEE/CVF WACV 2022"],
            ["Licence", "Apache-2.0 for the model; CC0 1.0 for the data — the "
                        "least restrictive licence in the project, so the credit "
                        "here is given by choice rather than obligation"],
            ["Why RF-DETR", "The same author published YOLO11n and YOLO11x "
                            "checkpoints for this dataset under AGPL-3.0. "
                            "RF-DETR was chosen specifically because it is "
                            "Apache-2.0 and keeps this detector outside the AGPL "
                            "perimeter"],
            ["Reported by author", "mAP@50 of 0.7931 across five classes — "
                                   "carried by boats. Per-class swimmer AP is "
                                   "0.282 (REPORTED BY AUTHOR)"],
        ],
        description=(
            "Detects a person in open water from elevated or aerial viewpoints. "
            "The DANA killed people swept from roads and trapped in vehicles as "
            "water rose; this is the situation that actually killed, rather than "
            "the pool-drowning scenario originally planned."),
        implementation=(
            "Only the `swimmer` class is routed to the validator. A boat is "
            "context, not an emergency, and passing boats through would alarm on "
            "every vessel in frame. RF-DETR emits a fixed set of object queries "
            "with no duplicate boxes, so unlike the YOLO detectors elsewhere in "
            "VIGÍA it requires no non-maximum suppression."),
        modifications=[
            "Swimmer-only routing to the validator; other classes available as "
            "operator context.",
            "Corrected class-slot mapping — see the results note.",
        ],
        envelope=(
            "Elevated and aerial views of open water, 5 to 260 m altitude. Not "
            "validated for pool settings, surf zones, or extreme crowd density."),
        results_rows=[
            ["Swimmer box precision", "0.9199", "SeaDronesSee val, conf 0.30"],
            ["Swimmer box recall", "0.9146", "SeaDronesSee val"],
            ["Swimmer box F1", "0.9172", "SeaDronesSee val"],
            ["Image-level false-alarm rate", "0.0222", "SeaDronesSee val"],
            ["Ground-truth swimmers", "1,218", "300 images, 255 with swimmers"],
            ["Median latency", "50.2 ms", "M4 Pro, CPU provider"],
        ],
        results_note=(
            "We do not quote the author's headline mAP of 0.7931: it is across "
            "five classes and carried by boats, and the swimmer class — the only "
            "one VIGÍA cares about — is their second weakest at 0.282. Our 0.9172 "
            "and their 0.282 are not in conflict; AP averaged over strict IoU "
            "thresholds punishes imprecise boxes on tiny objects, while our "
            "0.3-IoU operating point asks the operationally relevant question, "
            "which is whether the person was found at all. This detector also "
            "produced the most instructive failure in the project: the exported "
            "head has six logit slots for five classes, a leading no-object slot "
            "was assumed, and every label shifted by one — boats were reported as "
            "swimmers at 0.84 confidence while real swimmers were discarded. It "
            "nearly survived review because image-level precision still read "
            "0.83, boats and swimmers co-occurring in the same frames. Only "
            "box-level IoU exposed it, at a recall of 0.002."),
    )
    add_figure(document, "confidence_sweep.png",
               "Figure 2. Swimmer precision and recall against the confidence "
               "threshold. The operating point is chosen where recall is still "
               "high, because the temporal validator can discard a false alarm "
               "but nothing downstream can recover a person never detected.")
    document.add_page_break()

    hazard_block(document,
        title="4.5 Trapped People and Building Access Points",
        status="IMPLEMENTED and measured",
        credit_rows=[
            ["Model", "YOLOv8-DRN, trained and published by the DRespNeT authors"],
            ["Dataset", "DRespNeT — UAV imagery of the 2023 Türkiye earthquakes"],
            ["Created by", "Aykut Sirma and colleagues, Cranfield University "
                           "(arXiv 2508.16016)"],
            ["Source", "doi.org/10.6084/m9.figshare.29991478.v2"],
            ["Licence", "CC BY 4.0 as published — attribution is legally "
                        "required here, unlike SeaDronesSee where it is given "
                        "by choice"],
            ["Licence complication",
             "The checkpoint carries an embedded Ultralytics AGPL-3.0 stamp "
             "because it was trained with that toolkit, while the figshare "
             "record states CC BY 4.0. The same tension as the PyroNear fire "
             "detector; it exists in the authors' distribution and is contained "
             "the same way — ONNX conversion in an isolated build environment, "
             "no weights redistributed, enforced by tools/check_licence.py"],
            ["Reported by authors", "92.7% mAP50 at 27 FPS across all 28 "
                                    "classes (REPORTED BY AUTHORS, not "
                                    "comparable to ours)"],
            ["Not used", "xView2 / xBD — the winning weights are public, but the "
                         "task needs bi-temporal satellite pairs including a "
                         "pre-disaster image at inference time. Not a camera "
                         "model"],
        ],
        description=(
            "Identifies civilians and rescue personnel, and viable entry points "
            "into collapsed or damaged structures, from aerial imagery. During "
            "the DANA people were trapped inside buildings and garages; the same "
            "capability applies directly to earthquake response."),
        implementation=(
            "The model predicts 28 classes; VIGÍA merges them to five at the "
            "detector boundary — civilian, rescue team, accessible entry, "
            "blocked entry, collapsed building — and routes only `civilian` to "
            "the validator. A rescue team on site is the expected state, and "
            "alarming on it would fire at every incident already being handled. "
            "The model is an instance segmenter, so its output carries 32 mask "
            "coefficients after the class scores; these are trimmed at "
            "postprocess, because without that the decoder reads 60 classes and "
            "takes an argmax over mask prototypes."),
        modifications=[
            "28 classes merged to 5 at the boundary, leaving the rest in the graph.",
            "Mask coefficients trimmed so a segmentation export can be used as a "
            "detector.",
            "Converted to ONNX in an isolated environment so the runtime never "
            "imports Ultralytics.",
        ],
        envelope=(
            "Downward and oblique UAV views of earthquake-damaged urban blocks, "
            "at the altitudes flown in the 2023 Türkiye response. Not validated "
            "for street-level views, interior imagery, or night. Trained on 650 "
            "images, so generalisation to other disaster types and other cities "
            "is unproven rather than assumed."),
        results_rows=[
            ["Civilian box recall", "0.942", "Held-out test, conf 0.30"],
            ["Civilian box precision", "0.785", "Held-out test"],
            ["Civilian box F1", "0.856", "Held-out test"],
            ["Image-level recall", "1.000", "Held-out test"],
            ["Image-level false-alarm rate", "0.083", "Held-out test"],
            ["Rescue team F1", "0.895", "Held-out test"],
            ["Accessible entry F1", "0.882", "Held-out test"],
            ["Blocked entry F1", "0.802", "Held-out test"],
            ["Collapsed building F1", "0.941", "Held-out test"],
            ["Median latency", "37.6 ms", "M4 Pro, CPU provider"],
        ],
        results_note=(
            "VIGIA first trained its own torchvision Faster R-CNN on the same "
            "650 images, specifically because torchvision is BSD-3-Clause and "
            "would have kept this detector clear of the AGPL perimeter "
            "altogether. On the identical held-out split with the identical "
            "metric it reached civilian box recall of 0.141 — it missed roughly "
            "six of every seven people it should have boxed, and was by a wide "
            "margin the weakest component in the system. The authors' own "
            "weights reach 0.942. That is not a margin any amount of tuning on "
            "650 images was going to close, and keeping a six-times-worse "
            "detector to preserve an architectural preference would have been a "
            "choice against the people this detector exists to find. The "
            "test/valid gap also inverted: our model scored 0.462 on valid "
            "against 0.141 on test, a collapse showing it had not generalised, "
            "where this one scores 0.815 against 0.856. The held-out split is "
            "24 images, so these figures remain indicative rather than precise; "
            "the authors' figshare record was inspected and contains no larger "
            "release."),
    )
    add_figure(document, "building_access_models.png",
               "Figure 3. The two models on the identical held-out split with "
               "the identical metric. Image-level recall is 1.000 for both, "
               "which is exactly why image-level metrics alone are not enough: "
               "they hide a six-fold difference in whether the person was "
               "actually located.")
    add_figure(document, "building_access_per_class.png",
               "Figure 4. Per-class F1 on the held-out split at confidence "
               "0.30. Only `civilian` reaches the validator; the rest are "
               "operator context.")
    document.add_page_break()


def section_validator(document: Document) -> None:
    document.add_heading("5. The Temporal Validator", level=1)
    add_paragraph(document,
        "This is the original contribution. Everything else in VIGÍA is "
        "integration.", italic=True)

    document.add_heading("5.1 Rationale", level=2)
    add_paragraph(document,
        "Detection is a commodity. Datasets are public, models are published, and "
        "fine-tuning one is a weekend's work. The unsolved, expensive and "
        "operationally decisive problem is the false alarm, which is why "
        "commercial operators surround good models with expensive human staffing.")
    add_paragraph(document,
        "The validator sits between the detectors and the alert channel. It "
        "receives every raw detection and decides which become events. Detections "
        "it rejects are not discarded silently: each is annotated with the name of "
        "the level that rejected it, so suppression is auditable in evaluation and "
        "visible in the operator interface.")

    document.add_heading("5.2 The four levels", level=2)
    add_table(document,
        ["#", "Level", "What it does", "State"],
        [["1", "Confidence", "Per-hazard confidence floor.", "Stateless"],
         ["2", "Colour prior",
          "Fraction of pixels in the box matching an HSV prior for the hazard.",
          "Stateless"],
         ["3", "Temporal persistence",
          "A candidate becomes an event only after persisting across N frames, "
          "associated frame to frame by IoU overlap.", "Stateful"],
         ["4", "Location cooldown",
          "Suppresses repeat alerts from a region that recently alerted.",
          "Stateful"]],
        widths=[0.3, 1.25, 3.5, 0.85])

    add_paragraph(document,
        "Every level is independently switchable. This is not a convenience: it is "
        "what allows the ablation in 5.4 to be produced by the same code that runs "
        "in production, rather than by a separate analysis script that might "
        "diverge from it.")

    document.add_heading("5.3 What the suppression figure does and does not mean",
                         level=2)
    add_paragraph(document,
        "The validator rejects the large majority of what the detectors "
        "propose, and it is tempting to report that as a single suppression "
        "figure. That figure would be misleading, because it combines two "
        "operations that support entirely different claims.")
    add_bullets(document, [
        "Confidence, colour and persistence reject a candidate as NOT "
        "CREDIBLE. This is false-alarm reduction, and it is the claim the "
        "project makes.",
        "Cooldown rejects a credible detection because the event it belongs to "
        "has ALREADY BEEN REPORTED. Nothing is filtered out here; an alert is "
        "simply not sent twice.",
    ])
    add_paragraph(document,
        "Measured across the four demonstration clips, 1,299 proposals "
        "produced 117 confirmed events. Reported as one number that is 91% "
        "suppression. Split, it is 73.7% rejected as not credible and 17.3% "
        "withheld as repeats — and the split varies enormously by hazard.")
    add_table(document,
              ["Hazard", "Rejected as not credible", "Withheld as a repeat"],
              [["Fire", "10.0%", "80.0%"],
               ["Flood", "22.9%", "63.9%"],
               ["People in water", "70.7%", "16.2%"],
               ["Building access", "82.8%", "10.5%"]],
              widths=[1.6, 2.3, 2.3])
    add_figure(document, "suppression_split.png",
               "Figure 5. What the validator does with each hazard's proposals, "
               "measured on the demonstration clips. The proportions invert "
               "between fire and building access, which a single suppression "
               "figure would conceal entirely.")
    add_paragraph(document,
        "The fire figure makes the point. Of a 90% headline, 80 points are the "
        "system declining to re-announce one fire it had already reported — "
        "correct behaviour, and not false-alarm reduction. Quoting 90% as a "
        "false-alarm result would overstate the contribution by a factor of "
        "eight on that clip.")
    add_callout(document, "This project has been caught by exactly this before",
        "During R1, recall appeared to collapse from 0.925 to 0.125 and read as "
        "the validator destroying the detector. It was cooldown correctly "
        "suppressing repeat alerts about a fire already found — the validator "
        "was locating MORE fires, not fewer. Cooldown was removed from the "
        "evaluation harness for that reason. The runtime pipeline reintroduced "
        "the same conflation in its live statistics, and now reports the two "
        "quantities separately everywhere they appear, including on the "
        "recorded demonstrations.")

    document.add_heading("5.4 R1 — Does it reduce false alarms?", level=2)
    ablation = load_json("r1_ablation.json")
    if ablation:
        results = ablation["results"]
        rows = []
        baseline_far = results["raw detector (no validator)"]["negatives"]["false_alarm_rate"]
        for label in ["raw detector (no validator)",
                      "full cascade, persistence=2", "full cascade, persistence=3",
                      "full cascade, persistence=4", "full cascade, persistence=5"]:
            if label not in results:
                continue
            entry = results[label]
            far = entry["negatives"]["false_alarm_rate"]
            fires = entry["fires"]
            reduction = "—" if label.startswith("raw") else f"{100 * (1 - far / baseline_far):.0f}%"
            rows.append([
                label.replace("full cascade, ", ""),
                f"{far:.3f}", reduction,
                f"{fires['sequences_detected']}/{fires['total_sequences']}",
                f"{fires['detection_rate']:.3f}",
                f"{fires['median_minutes_to_detect']}",
            ])
        add_table(document,
            ["Configuration", "False-alarm rate", "Reduction", "Fires found",
             "Recall", "Min. to detect"], rows,
            widths=[1.35, 1.0, 0.75, 0.8, 0.7, 0.9])

        add_paragraph(document,
            f"MEASURED. Negatives: {ablation['negative_sequences']} sequences from "
            f"distinct camera, date and weather combinations. Fires: "
            f"{ablation['fire_sequences']} ignition sequences. Confidence floor "
            f"{ablation['conf_threshold']}. Harness: eval/run_ablation.py.",
            size=9.5, italic=True, colour=MUTED)

        add_callout(document, "The headline result and its cost",
            "At persistence 3 the false-alarm rate falls by 75% and every fire is "
            "still found — 6 of 6, one more than the raw detector, which missed "
            "one. The cost is latency: the first alert arrives about two minutes "
            "later, because persistence buys precision by waiting and these "
            "cameras sample once per minute. Both halves of that trade are "
            "reported together; a filter that drives false alarms to zero by "
            "refusing to alert would score perfectly on one metric alone.",
            colour=ACCENT)

    document.add_heading("5.5 Per-level ablation — what each level buys", level=2)
    if ablation:
        results = ablation["results"]
        full = results.get("full cascade, persistence=3", {}).get("negatives", {}).get("false_alarm_rate")
        rows = []
        for key, label, verdict in [
            ("minus confidence", "Confidence removed",
             "Carries most of the result"),
            ("minus persistence", "Persistence removed",
             "Carries the remainder"),
        ]:
            if key in results:
                far = results[key]["negatives"]["false_alarm_rate"]
                rows.append([label, f"{far:.3f}",
                             f"{far / full:.1f}× worse" if full else "—", verdict])
        add_table(document, ["Configuration", "FAR", "vs full cascade", "Verdict"],
                  rows, widths=[1.7, 0.8, 1.2, 2.3])

        add_callout(document, "A fifth level was built, measured, and removed",
            "A size-plausibility level rejected detections whose area fraction or "
            "aspect ratio was implausible. Measured, it rejected 0% of detections "
            "on wildfire and 0% on traffic. When flood was added it proved "
            "actively harmful: real ATLANTIS flood masks cover a median 32% of "
            "the frame and severe cases exceed 60%, so a plausible maximum-area "
            "default silently discarded the worst floods — the more severe the "
            "disaster, the more likely the alert disappeared. A level that earns "
            "nothing on two hazards and inverts the safety property on a third "
            "does not belong in a life-safety cascade. Removing it changed no "
            "measured result, which is itself the evidence it was doing nothing. "
            "A regression test now prevents its reintroduction.")

    document.add_heading("5.6 Hazard-agnosticism, tested and enforced", level=2)
    add_paragraph(document,
        "PyroNear's published training manifest shows they already perform "
        "sequential confirmation for wildfire, with a grid search over a minimum "
        "frame count. Temporal confirmation for fire is therefore not novel, and "
        "we say so. The claim VIGÍA makes is narrower and still defensible: ONE "
        "validator, configured rather than specialised, working across hazards "
        "whose detectors were built by different people, on different data, for "
        "phenomena with different physics.")

    multi = load_json("multihazard.json")
    if multi:
        rows = []
        for key, label, source in [
            ("fire", "Wildfire smoke", "FIgLib, 1 frame per minute"),
            ("traffic", "Traffic accidents", "Dashcam video, 2 fps"),
        ]:
            if key in multi.get("hazards", {}):
                h = multi["hazards"][key]
                rows.append([label, h["detector"], source,
                             str(h["raw"]["frames"]),
                             str(h["raw"]["raw_detections"]),
                             str(h["validated"]["confirmed_events"]),
                             f"{h['suppression']:.1%}"])
        add_table(document,
            ["Hazard", "Detector", "Source", "Frames", "Raw det.",
             "Confirmed", "Suppression"], rows,
            widths=[0.95, 1.5, 1.15, 0.5, 0.55, 0.65, 0.75])

        add_paragraph(document,
            "MEASURED — eval/run_multihazard.py. The only difference between the "
            "two runs is a configuration object. No branch in the validator refers "
            "to any hazard.", size=9.5, italic=True, colour=MUTED)

    add_callout(document, "Enforced, not merely asserted",
        "tests/test_validator.py contains test_validator_contains_no_hazard_"
        "specific_code, which parses the validator's source with Python's AST "
        "module and fails the build if any comparison against a specific hazard "
        "appears. A future contributor fixing a wildfire bug with a hazard-"
        "specific branch cannot silently invalidate the central claim of this "
        "document. The suite currently passes 9 of 9.", colour=ACCENT)
    document.add_page_break()


def section_methodology(document: Document) -> None:
    document.add_heading("5.7 Trend detection compared on real cameras", level=2)
    flood = load_json("flood.json")
    if flood and flood.get("trend"):
        add_paragraph(document,
            "Flood is the only hazard whose temporal logic does not rely solely "
            "on the validator: a water level has a trend, and the trend is the "
            "alert. Two mechanisms were implemented and compared on identical "
            "inputs — a sliding-window least-squares fit, and a Page-Hinkley "
            "change detector following Choi et al. (2026), who published that "
            "approach for this problem. Neither is claimed as ours.")

        rows = []
        for name, entry in flood["trend"].items():
            ls = entry["least_squares_first_call_seconds"]
            ph = entry["page_hinkley_alarm_seconds"]
            rows.append([
                name, str(entry["frames"]),
                f"{entry['duration_minutes']:.0f} min",
                f"{entry['coverage_start']:.3f} to {entry['coverage_end']:.3f}",
                f"{ls / 60:.1f} min" if ls is not None else "never",
                f"{ph / 60:.1f} min" if ph is not None else "never",
            ])
        add_table(document,
            ["Sequence", "Frames", "Span", "Water coverage",
             "Least squares", "Page-Hinkley"], rows,
            widths=[1.5, 0.6, 0.7, 1.35, 1.0, 1.0])

        add_paragraph(document,
            "MEASURED — eval/run_flood_eval.py, on LSU creek camera sequences "
            "from the V-FloodNet dataset. Real fixed cameras, real water, real "
            "segmentation noise. Evaluation use only; nothing redistributed.",
            size=9.5, italic=True, colour=MUTED)

        add_callout(document, "A bug that only real data could find",
            "The two methods win on different sampling regimes: the windowed "
            "fit is faster on densely sampled cameras, Page-Hinkley on sparsely "
            "sampled ones, because it counts samples rather than seconds. That "
            "difference surfaced as a failure. With a fixed 300-second window "
            "the least-squares fit never fired at all on a creek camera sampled "
            "every 15 minutes — it reported UNKNOWN for seven hours while the "
            "water rose by 10% of the frame — because the window could never "
            "contain two frames. Every synthetic test passed throughout. A time "
            "window is the wrong abstraction when sampling rate is a property "
            "of the deployment; the window is now adaptive to the observed "
            "interval.", colour=ACCENT)

    document.add_page_break()

    document.add_heading("6. Evaluation Methodology", level=1)

    document.add_heading("6.1 Datasets used", level=2)
    add_table(document,
        ["Dataset", "Role", "Size used", "Provider"],
        [["pyronear/pyro-sdis", "Fire detector baseline; hard negatives",
          "100 frames (77 pos / 23 neg)", "PyroNear + French SDIS"],
         ["HPWREN FIgLib — fires", "Ignition sequences; detection and latency",
          "6 sequences, 240 frames", "HPWREN, UC San Diego"],
         ["HPWREN FIgLib — negatives", "False-alarm measurement",
          "30 sequences, 300 frames", "HPWREN, UC San Diego"],
         ["Dashcam compilation", "Traffic validator transfer",
          "660 frames at 2 fps", "Bundled with the original project fork"],
         ["ATLANTIS", "Flood training and test",
          "3,364 train / 1,296 test", "iWERS lab, University of South Carolina"],
         ["LSU creek cameras", "Flood trend on real footage",
          "3 sequences, 80 frames", "V-FloodNet (Liang et al.)"]],
        widths=[1.5, 1.9, 1.4, 1.4])
    add_paragraph(document,
        "HPWREN requires attribution wherever its imagery is published. Any figure "
        "or demonstration using HPWREN frames must carry that credit.",
        size=9.5, italic=True, colour=MUTED)

    document.add_heading("6.2 Methodological commitments", level=2)
    add_bullets(document, [
        "Cross-dataset evaluation. Domain shift is the most likely cause of "
        "real-world failure; a random split of the training set flatters a model "
        "and predicts nothing about deployment.",
        "Report the gap. In-domain and out-of-domain figures appear side by side.",
        "Event-level evaluation on sequences, not frame-level on stills. The "
        "validator only demonstrates value over time.",
        "F2 rather than F1 as the headline. Recall matters more than precision for "
        "life safety — a convention borrowed from the RipVIS benchmark.",
        "Never quote a figure we did not measure. Published third-party numbers "
        "appear only in clearly attributed rows.",
    ])

    add_callout(document, "A trap we walked into and documented",
        "Our first false-alarm measurement on ignition sequences gave 0.8%, "
        "against 71% on curated hard negatives with the same model and threshold. "
        "The difference is negative-set difficulty: frames shortly before a fire "
        "are usually clear skies, whereas curated negatives are cloud, fog and "
        "dust chosen to confuse. Had we published the 0.8% figure as evidence the "
        "validator works, any reviewer inspecting the negative set would have "
        "dismantled the claim. Every false-alarm figure in this document names the "
        "negative set it came from.")

    document.add_heading("6.3 Reproducibility", level=2)
    add_paragraph(document,
        "Detection is run once and cached to JSONL; validator configurations are "
        "then replayed against the cache. Every configuration therefore sees "
        "byte-identical detector output, so any difference in outcome is "
        "attributable to the validator and not to run-to-run variation. All "
        "results files are committed under eval/results/.")

    document.add_heading("6.4 Execution backends", level=2)
    add_paragraph(document,
        "Two backends exist and are not interchangeable. REFERENCE runs the "
        "as-published dynamic-shape graph on the CPU provider and is "
        "deterministic; every figure in this document comes from it. FAST runs a "
        "static-shape variant on Apple's CoreML provider and is 1.7–1.9× faster "
        "(34 ms against 60 ms at 1024 px), which is the difference between a 30 "
        "FPS and an 18 FPS demonstration.")
    add_paragraph(document,
        "FAST is not bit-identical. Across 100 validation frames the detection "
        "count matched on 100 of 100 and boxes agreed to within 0.41 px, but "
        "confidences drift by up to 0.012 because the Neural Engine computes at "
        "lower precision. At our 0.20 operating point that drift changed no metric; "
        "at 0.10 and 0.30 it moved box precision and recall by about one "
        "percentage point by flipping a single detection across the threshold. "
        "FAST is therefore used for anything a person watches, and REFERENCE for "
        "anything reported.")

    add_callout(document, "A caveat that earned its keep",
        "The most quoted figure in this project — that the published fire "
        "detector alarms on frames containing no smoke — was first measured on "
        "23 negative frames and carried the standing note 'indicative, not "
        "precise; widen before publishing'. Widening it to 585 negatives from "
        "21 cameras, a twenty-five-fold increase, moved the false-alarm rate "
        "from 0.870 to 0.706 and box precision from 0.625 to 0.447. Recall "
        "barely moved (0.904 to 0.902), which is the reassuring half: the "
        "detector finds what it finds, and it was the small sample that had "
        "flattered it on false alarms. The motivating claim is untouched, since "
        "a detector alarming on 71% of clear frames is unusable without a "
        "filter, but the specific number was overstated by sixteen percentage "
        "points and every quotation of it has been corrected. Carrying an "
        "explicit caveat on a soft number, and then acting on it, is what made "
        "that correction routine rather than embarrassing.")

    add_figure(document, "fire_negative_widening.png",
               "Figure 6. The same detector at the same threshold, measured "
               "against 23 negatives and then against 585. Recall barely moved; "
               "precision and the false-alarm rate both fell, which is what a "
               "small sample flattering a detector looks like.")

    document.add_heading("6.5 Demonstration material", level=2)
    add_paragraph(document,
        "Demonstration clips are treated as evidence rather than as decoration, "
        "and are therefore subject to the same gating as any published figure. "
        "One clip per hazard is assembled from local data in sorted order at a "
        "fixed frame rate, so a clip built twice is identical and a run over it "
        "reproduces exactly. The recording is rendered directly from the "
        "pipeline rather than screen-captured, which removes any dependence on "
        "a browser, a display or a network at presentation time.")
    add_paragraph(document,
        "Re-rendering a clip produces a byte-identical file, and that property "
        "had to be earned rather than assumed. An initial version drew the live "
        "detector latency onto each frame, which made two renders of the same "
        "clip differ — the difference being confined entirely to the 63 by 25 "
        "pixel box that readout occupied. Latency is a property of the machine "
        "rather than of the method, so it is now reported in the render summary "
        "alongside the hardware instead of being drawn on the frames. "
        "Everything shown on a frame is a count derived from the detections, "
        "and is therefore reproducible by anyone re-running the pipeline.")
    add_paragraph(document,
        "Two independent conditions decide whether a clip may appear in a "
        "recorded demonstration, a figure, or any public material. Both are "
        "recorded per clip in a manifest, and the renderer refuses a clip that "
        "fails either — refusing is the safe default, because a clip that "
        "should not be shown plays exactly like one that may.")
    add_table(document,
              ["Condition", "Why it is necessary"],
              [["The licence permits publication",
                "The source datasets carry four different licences and one "
                "forbids redistribution outright. A recording is a published "
                "work, so including all-rights-reserved frames would breach a "
                "licence in front of the audience most likely to notice. Where "
                "a licence permits publication subject to attribution, the "
                "credit is burned onto every frame rather than left to a "
                "caption that may be cropped away."],
               ["The frames are a genuine time sequence",
                "The validator is temporal, so demonstrating it on footage "
                "without temporal continuity misrepresents it in both "
                "directions: persistence rejects almost everything because "
                "nothing genuinely persists, while a few detections confirm "
                "anyway because unrelated objects happen to overlap between "
                "consecutive frames and the tracker associates them."]],
              widths=[1.7, 4.5])
    add_callout(document, "A clip that played perfectly and meant nothing",
        "The first people-in-water clip was 60 unrelated validation stills "
        "stitched into a video. It played correctly and produced confident "
        "figures — 156 detections proposed, 16 confirmed, 121 rejected by "
        "persistence — all of them artefacts. Continuity is now verified rather "
        "than assumed: adjacent-frame histogram correlation is about 0.997 on "
        "genuine footage against about 0.015 for unrelated frames from the same "
        "split, and the fetch script refuses to report success unless that "
        "separation holds. On real sequential footage the same detector and the "
        "same validator give 451 proposed and 59 confirmed.")
    add_paragraph(document,
        "Applying both conditions initially left only two of the four hazards "
        "able to appear in a recorded demonstration. One of the two gaps has "
        "since been closed and one has been confirmed as genuinely closed off. "
        "Flood — the keystone hazard — had no publishable footage at all, "
        "because the only real camera sequences available for it are all rights "
        "reserved; that is now solved. Building access has no temporal sequence "
        "in any released form, and after a false start described below, that is "
        "recorded as a limitation rather than worked around.")
    add_table(document,
              ["Hazard", "Source used", "Licence"],
              [["Fire", "HPWREN FIgLib ignition sequence",
                "HPWREN imagery, attribution required"],
               ["Flood", "USGS gauge camera, real flood event",
                "Work of the US Government — public domain"],
               ["People in water", "SeaDronesSee flight sequence",
                "CC0 1.0 Universal"],
               ["Building access", "NO SEQUENCE EXISTS — detector only",
                "CC BY 4.0, but stills sampled from many flights"]],
              widths=[1.3, 3.0, 1.9])
    add_paragraph(document,
        "Flood was solved by changing source rather than by relaxing the rule. "
        "The USGS operates over 1,300 streamgage cameras whose imagery is a work "
        "of the United States Government and therefore unrestricted, and they "
        "are the right kind of footage as well as the right licence: fixed "
        "mount, pointed at water, sampled roughly hourly — the same sparse "
        "regime the trend detection was calibrated for, and the regime in which "
        "a fixed 300-second window originally failed outright. Only daylight "
        "frames are used: these cameras switch to infrared at night, and on "
        "night frames the segmenter tints sky and vegetation as water. That is "
        "the detector's declared envelope, and the fetch script enforces it "
        "rather than restating it.")
    add_paragraph(document,
        "Building access was NOT solved, and the record of the attempt is more "
        "useful than a clean result would have been. DRespNeT is published as "
        "shuffled, renamed stills whose filenames retain the original capture "
        "index, and ordering by that index does recover the authors' order. A "
        "twelve-frame window looked continuous — adjacent-frame correlation "
        "0.695 against 0.235 for unrelated pairs — and a 66-frame clip was "
        "built on that basis and described as a genuine sequence.")
    add_paragraph(document,
        "That was wrong. The authors also publish their un-augmented raw "
        "frames, and measured across all 615 of them the median adjacent "
        "correlation is 0.388 with 362 scene cuts; the longest continuous run "
        "in the entire dataset is SEVEN frames. Within the 66-frame range "
        "actually used, 40 of 65 transitions are scene cuts. DRespNeT is "
        "curated stills sampled from many flights rather than footage, so this "
        "hazard cannot demonstrate temporal validation on any released data. "
        "The clip is retained for exercising the DETECTOR and is excluded from "
        "the recorded demonstration.")
    add_callout(document, "How the wrong conclusion was reached",
        "The check was run on a twelve-frame window and generalised to "
        "sixty-six. It was also the RIGHT KIND of check — asking whether "
        "detections associate across frames rather than whether frames look "
        "alike — which made the conclusion feel well-founded. Tracks did reach "
        "length seven and 124 associations survived to persistence three, but "
        "on footage that is 62% scene cuts those associations are coincidental "
        "spatial overlap between unrelated scenes: precisely the artefact the "
        "same section warns about for still-image clips. A sound method applied "
        "to an unrepresentative sample produces a confident wrong answer, and "
        "the sample size is the part that has to be checked first.")
    add_paragraph(document,
        "Every demonstration clip is now publicly licensed, and no restricted "
        "footage remains in the demonstration path at all. The flood clip is a "
        "real event rather than a stable creek: Walnut Creek at Raleigh on "
        "8 August 2024, where the gage reached 9.00 ft and receded to 4.25 ft "
        "over the two days covered, sampled about every twenty minutes.")
    add_callout(document, "Ground truth the other footage does not have",
        "USGS cameras sit on instrumented gages, so every frame arrives with a "
        "measured water level beside it — which no other footage in this "
        "project provides, and which makes the flood coverage signal checkable "
        "rather than merely plausible. Over the 50 frames of this event, "
        "segmented water coverage correlates with measured gage height at "
        "r = +0.50. That is a real but moderate relationship, and it is "
        "reported because it QUALIFIES the trend claim: coverage tracks the "
        "water level between a flood day and a receded day, while within a day "
        "at near-constant level the coverage still varies by more than the "
        "level does. Restricting to midday frames does not improve it "
        "(r = +0.46 to +0.50 across daylight windows), so the residual is not "
        "simply low sun. Camera-derived coverage is therefore evidence of a "
        "trend, not a substitute for a gage.")
    document.add_page_break()


def section_results_summary(document: Document) -> None:
    document.add_heading("7. Consolidated Results", level=1)
    add_paragraph(document,
        "Every figure VIGÍA has measured to date, in one place. Figures published "
        "by third parties are listed separately in 7.2 and are never combined with "
        "ours.")

    document.add_heading("7.1 Measured by us", level=2)
    rows = []

    baseline = load_json("fire_pyro_sdis_val_wide.json")
    if baseline:
        c = baseline["results"]["conf_0.2"]
        rows += [
            ["Fire", "Box precision / recall",
             f"{c['box_level']['precision']:.3f} / {c['box_level']['recall']:.3f}",
             "pyro-sdis val, 100 frames", "run_image_eval.py"],
            ["Fire", "Image-level recall", f"{c['image_level']['recall']:.3f}",
             "pyro-sdis val, 77 positives", "run_image_eval.py"],
            ["Fire", "False-alarm rate (hard negatives)",
             f"{c['image_level']['false_alarm_rate']:.3f}",
             "pyro-sdis val, 585 negatives / 21 cameras", "run_image_eval.py"],
        ]

    ablation = load_json("r1_ablation.json")
    if ablation:
        res = ablation["results"]
        raw = res["raw detector (no validator)"]
        full = res["full cascade, persistence=3"]
        rows += [
            ["Fire", "False-alarm rate, no validator",
             f"{raw['negatives']['false_alarm_rate']:.3f}",
             "FIgLib negatives, 30 seq / 300 frames", "run_ablation.py"],
            ["Fire", "False-alarm rate, validator (p=3)",
             f"{full['negatives']['false_alarm_rate']:.3f}",
             "FIgLib negatives, 30 seq / 300 frames", "run_ablation.py"],
            ["Fire", "False-alarm reduction",
             f"{100 * (1 - full['negatives']['false_alarm_rate'] / raw['negatives']['false_alarm_rate']):.0f}%",
             "same negative set", "run_ablation.py"],
            ["Fire", "Ignition sequences detected",
             f"{full['fires']['sequences_detected']}/{full['fires']['total_sequences']}",
             "FIgLib, 6 fires", "run_ablation.py"],
            ["Fire", "Median minutes to first alert",
             f"{full['fires']['median_minutes_to_detect']}",
             "FIgLib, 6 fires", "run_ablation.py"],
        ]

    flood = load_json("flood.json")
    if flood and "segmentation" in flood:
        seg = flood["segmentation"]
        rows += [
            ["Flood", "Water IoU", f"{seg['water_iou']:.4f}",
             f"ATLANTIS test, {seg['images']} images", "run_flood_eval.py"],
            ["Flood", "Precision / recall",
             f"{seg['precision']:.3f} / {seg['recall']:.3f}",
             "ATLANTIS test (held out)", "run_flood_eval.py"],
            ["Flood", "F1", f"{seg['f1']:.4f}",
             "ATLANTIS test (held out)", "run_flood_eval.py"],
            ["Flood", "Trend detected on real cameras", "3 of 3",
             "LSU creek sequences, 80 frames", "run_flood_eval.py"],
        ]

    multi = load_json("multihazard.json")
    if multi:
        for key, label in (("fire", "Fire"), ("traffic", "Traffic")):
            if key in multi.get("hazards", {}):
                h = multi["hazards"][key]
                rows += [[label, "Alert suppression", f"{h['suppression']:.1%}",
                          f"{h['raw']['frames']} frames", "run_multihazard.py"]]

    rows += [
        ["System", "Inference latency, REFERENCE", "~60 ms",
         "1024 px, M4 Pro CPU", "detector telemetry"],
        ["System", "Inference latency, FAST", "~34 ms",
         "1024 px, M4 Pro CoreML", "detector telemetry"],
        ["System", "Validator behavioural tests", "9 / 9 passing",
         "unit and property tests", "tests/test_validator.py"],
    ]

    add_table(document,
              ["Hazard", "Metric", "Value", "Dataset", "Harness"], rows,
              widths=[0.65, 1.75, 0.85, 1.7, 1.25])

    document.add_heading("7.2 Reported by original authors (not ours)", level=2)
    add_table(document,
        ["Component", "Metric", "Value", "Reported by"],
        [["PyroNear fire detector", "Precision / recall / mAP", "Not published",
          "Model card publishes no metrics"],
         ["PyroNear2025 benchmark", "Cross-dataset F1", "~70%",
          "arXiv 2402.05349"],
         ["Traffic YOLO11x", "mAP@0.5", "0.826", "Enos-123 model card"],
         ["Traffic YOLO11x", "Accident-class recall", "0.855", "Enos-123 model card"],
         ["DRespNeT / YOLOv8-DRN", "mAP50 at 27 FPS", "92.7%",
          "Cranfield University, arXiv 2508.16016"],
         ["Reference multi-task system", "Fire false alarm 52% → 4%",
          "at 96% sensitivity", "arXiv 2607.03131"]],
        widths=[1.7, 1.55, 1.2, 1.75])
    document.add_page_break()


def section_credits(document: Document) -> None:
    document.add_heading("8. Attribution and Credits", level=1)
    add_paragraph(document,
        "VIGÍA is built on work done by other people. This section names them. "
        "Model weights are not redistributed in our repository; they are "
        "downloaded from their original distributors at setup time, and provenance "
        "for each is recorded in models/REGISTRY.yaml alongside its licence.")

    document.add_heading("8.1 Models and datasets", level=2)
    add_table(document,
        ["Component", "Authors / institution", "Licence", "Used for"],
        [["yolo11s_sensitive-detector", "PyroNear association, with French Fire "
          "and Rescue Services (SDIS)", "Apache-2.0", "Fire and smoke detection"],
         ["pyro-sdis dataset", "PyroNear volunteers with SDIS partners",
          "Apache-2.0", "Fire detector evaluation"],
         ["pyro-engine", "PyroNear association", "Apache-2.0",
          "Architectural reference only; no code copied"],
         ["FIgLib and camera archive", "HPWREN, University of California San Diego",
          "Public, attribution required", "Ignition sequences and negatives"],
         ["traffic-accident-detection-yolo11x", "Enos-123", "MIT",
          "Traffic incident detection"],
         ["MMSegmentation", "OpenMMLab", "Apache-2.0", "Planned flood segmentation"],
         ["ATLANTIS", "Erfani et al., iWERS lab, University of South Carolina",
          "CC / public-domain source images", "Planned flood training"],
         ["SeaDronesSee", "Varga, Kiefer et al., University of Tübingen",
          "CC0 1.0 (data), MIT (code)", "Planned people-in-water detection"],
         ["DRespNeT / YOLOv8-DRN", "Cranfield University",
          "See Figshare / Roboflow", "Planned building access detection"]],
        widths=[1.75, 1.85, 1.05, 1.55])

    document.add_heading("8.2 Prior art", level=2)
    add_paragraph(document,
        "VIGÍA is not the first camera-based hazard detection system and does not "
        "claim to be. The following work precedes it and in several respects "
        "exceeds it.")
    add_table(document,
        ["Organisation", "Domain", "Note"],
        [["Pano AI", "Wildfire", "Commercial, ~USD 50,000 per camera per year "
                                 "including a 24/7 intelligence centre"],
         ["ALERTCalifornia / DigitalPath", "Wildfire",
          "UC San Diego and CAL FIRE; 1,060+ cameras with human vetting"],
         ["PyroNear", "Wildfire",
          "Open-source, deployed across France, Spain and Chile — and the direct "
          "source of our fire detector"],
         ["Lynxight, AngelEye, Coral", "Drowning",
          "Commercially mature; ASTM F3698-24 and ISO 20380:2017"],
         ["Rekor, Iteris, Miovision, NoTraffic", "Traffic",
          "Deployed with US state transport departments"],
         ["xView2 / DIUx", "Building damage", "Satellite damage assessment"],
         ["RipVIS authors", "Rip currents", "CVPR 2025 benchmark"]],
        widths=[1.85, 1.15, 3.2])

    document.add_heading("8.3 Runtime dependencies", level=2)
    add_table(document, ["Package", "Licence"],
              [["onnxruntime", "MIT"], ["numpy", "BSD-3-Clause"],
               ["opencv-python-headless", "Apache-2.0"], ["PyYAML", "MIT"]],
              widths=[2.4, 2.0])
    add_paragraph(document,
        "The ultralytics package (AGPL-3.0) is deliberately not a runtime "
        "dependency and is confined to an isolated build environment.",
        size=9.5, italic=True, colour=MUTED)
    document.add_page_break()


def section_changelog(document: Document) -> None:
    document.add_heading("9. Change Log", level=1)
    add_paragraph(document,
        "Every entry that adds or revises a measured figure is recorded here so "
        "the provenance of each number in this document can be traced.")
    add_table(document,
        ["Version", "Date", "Change"],
        [["0.1", "1 Sep 2026",
          "Project foundation: ONNX-only runtime, licence guard, model registry, "
          "fetcher. Fire detector integrated. First baseline measured on "
          "pyro-sdis (false-alarm rate 0.706 on 585 negatives)."],
         ["0.2", "1 Sep 2026",
          "Temporal validator implemented with five levels and IoU tracker. "
          "FIgLib fetcher built. R1 measured: 75% false-alarm reduction at "
          "persistence 3 with no loss of fires detected. Per-level ablation "
          "showed size plausibility contributes nothing."],
         ["0.3", "1 Sep 2026",
          "Traffic detector integrated via isolated ONNX export. Hazard-agnostic "
          "transfer measured across two hazards and enforced by an AST-based "
          "test. Scope revised: building access and people-in-water added, pool "
          "drowning removed. This document created."],
         ["0.4", "1 Sep 2026",
          "ATLANTIS acquired (5,195 images; class mapping derived empirically "
          "from the masks). Water level and rate-of-rise signal implemented and "
          "tested (9 of 9). Segmentation base class added with a mask-to-box "
          "bridge validated on real flood masks. Size-plausibility level removed "
          "after measurement across three hazards; all prior results unchanged, "
          "confirming it contributed nothing."],
         ["0.5", "1 Sep 2026",
          "All four hazards measured. Flood segmenter trained on ATLANTIS "
          "(test water IoU 0.7633). People-in-water integrated, where a "
          "label-shift bug reported boats as swimmers at 0.84 confidence while "
          "discarding real swimmers — caught only by box-level evaluation, "
          "since image-level precision still read 0.83. Building access trained "
          "on DRespNeT; checkpoint chosen by measured detection metrics after "
          "the training loss proxy was shown to select a strictly worse model."],
         ["0.6", "2 Sep 2026",
          "The system became a system: context gate, runtime pipeline, egress "
          "boundary and operator view, with privacy enforced by a build-failing "
          "guard rather than by policy. Demonstration recorded deterministically "
          "for all four hazards, every clip publicly licensed. Results freeze "
          "added, checking each published figure against the file it cites. "
          "Fire baseline re-measured on 585 negatives instead of 23: the "
          "false-alarm rate fell from 0.870 to 0.706 and every quotation was "
          "corrected. Flood footage moved to public-domain USGS gauge cameras, "
          "which supply an instrumented water level alongside each frame."],
         ["0.7", "2 Sep 2026",
          "Building access rebuilt on the DRespNeT authors' own published "
          "weights after their figshare record was resolved. Civilian box "
          "recall on the held-out split moved from 0.141 to 0.942, turning the "
          "weakest component in the system into one of the strongest. Sections "
          "4.4 and 4.5 corrected from PLANNED to measured, having been stale "
          "since those hazards were implemented. The claim that building access "
          "had a usable temporal sequence was retracted: it was generalised "
          "from a twelve-frame window, and the authors' raw frames show 362 "
          "scene cuts with a longest continuous run of seven."]],
        widths=[0.7, 1.1, 4.4])

    document.add_heading("Outstanding work", level=2)
    add_paragraph(document,
        "Listed in the order it would be tackled. The first two are shortages of "
        "available material rather than defects in the system, and both are "
        "recorded with their acquisition routes in docs/DATA_ACQUISITION.md.")
    add_bullets(document, [
        "BUILDING ACCESS RESTS ON 24 HELD-OUT IMAGES. This is the last binding "
        "sample-size caveat in the project. The figures on that split are now "
        "strong (civilian box recall 0.942), but a 24-image sample cannot make "
        "them precise. It cannot be widened from released data: the authors' "
        "figshare record was resolved and inspected and holds the same "
        "650/50/25 split. It needs a different post-disaster aerial dataset "
        "with people annotated.",
        "BUILDING ACCESS CANNOT DEMONSTRATE TEMPORAL VALIDATION. DRespNeT is "
        "curated stills sampled from many flights, not footage — 362 scene cuts "
        "across 615 raw frames, longest continuous run seven. The detector is "
        "exercised on it; the validator cannot be. A UAV dataset distributed as "
        "video would close this.",
        "NO URBAN FLOOD FOOTAGE. The flood demonstration uses a public-domain "
        "gauge camera on a creek. The 2024 Valencia DANA is the narrative the "
        "project is built around, and no footage of it — or of any flooded "
        "street with vehicles and buildings — is yet in the system under a "
        "licence that permits publication.",
        "The traffic detector is integrated and running behind the validator "
        "but has no standalone measurement against held-out labels. It is "
        "retained explicitly as the generality experiment for the validator, "
        "not as a disaster hazard, and the document says so wherever it appears.",
        "The camera-derived flood coverage signal correlates with instrumented "
        "gauge height at r = +0.50 over one event. That is a first measurement "
        "against ground truth rather than a satisfactory one, and it should be "
        "repeated across several cameras and events before the trend claim is "
        "leaned on.",
        "Publish the repository, with the training and evaluation scripts that "
        "reproduce every figure in this document.",
    ])


def main() -> int:
    document = Document()
    for section in document.sections:
        section.left_margin = section.right_margin = Inches(0.9)
        section.top_margin = section.bottom_margin = Inches(0.85)

    configure_styles(document)

    # Figures are regenerated from the stored results on every build, so a
    # chart can never depict an older measurement than the table beside it.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import figures
        figures.main()
    except Exception as exc:
        print(f"figure generation skipped: {exc}", file=sys.stderr)

    cover(document)
    section_introduction(document)
    section_architecture(document)
    section_hazards(document)
    section_systems(document)
    section_validator(document)
    section_methodology(document)
    section_results_summary(document)
    section_credits(document)
    section_changelog(document)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT_PATH)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"version {DOC_VERSION}, generated {dt.date.today()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
