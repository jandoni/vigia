#!/usr/bin/env python3
"""Render the demo video deterministically, straight from the pipeline.

    python scripts/record_demo.py
    python scripts/record_demo.py --clip fire_ridge --out runs/demo_fire.mp4

Live demonstrations fail, so the plan requires a recording made early rather
than the night before. This renders one rather than screen-capturing a browser,
for three reasons: it needs no browser, no network and no display; it produces
the same file every run because the pipeline replays file sources frame by
frame; and it can be regenerated the moment better footage arrives.

The composition mirrors the operator view — canvas, cascade rail, timeline,
metric strip — because the argument is the same one: what the detector proposed
against what the system confirmed.

LICENSING IS ENFORCED, NOT DOCUMENTED. A recording is a published work. This
script reads clips/MANIFEST.json and refuses to render any clip that is not
marked publishable, and it burns the required attribution onto every frame.
Refusing is the only safe default: the restricted clip in this project plays
exactly like the others and nothing about the footage itself would reveal the
problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from vigia.pipeline import CameraPipeline                     # noqa: E402
from vigia.registry import Camera, CameraContext              # noqa: E402
from vigia.types import Hazard                                # noqa: E402
from vigia.web.overlay import TrailBuffer, draw               # noqa: E402

W, H = 1280, 720
HEADER, RAIL, TIMELINE, METRICS, CREDIT = 44, 320, 42, 44, 22
CANVAS_W = W - RAIL
CANVAS_H = H - HEADER - TIMELINE - METRICS - CREDIT

GROUND = (15, 13, 13)
PANEL = (23, 20, 20)
LINE = (43, 38, 38)
TEXT = (232, 230, 230)
MUTED = (143, 133, 133)
FAINT = (99, 90, 90)
CONFIRMED = (86, 220, 255)
TRACKING = (166, 190, 120)
SUPPRESSED = (110, 110, 110)

FONT = cv2.FONT_HERSHEY_DUPLEX
MONO = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, x, y, colour=TEXT, scale=0.42, font=FONT, thick=1):
    cv2.putText(img, s, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def compose(annotated, state: dict, spec: dict) -> np.ndarray:
    frame = np.full((H, W, 3), GROUND, np.uint8)

    # ---- header: the gate made visible ---------------------------------
    cv2.rectangle(frame, (0, 0), (W, HEADER), PANEL, -1)
    cv2.line(frame, (0, HEADER), (W, HEADER), LINE, 1)
    # OpenCV's Hershey fonts have no accented glyphs, and the project is
    # VIGÍA, not VIGIA — getting a name wrong on the title card of a public
    # submission is not a detail. The acute accent is two strokes, which is a
    # smaller price than a font dependency for one glyph.
    brand = "V I G I A"
    text(frame, brand, 16, 28, TEXT, 0.5)
    second_i = 16 + int(cv2.getTextSize("V I G ", FONT, 0.5, 1)[0][0])
    cv2.line(frame, (second_i + 1, 12), (second_i + 6, 7), TEXT, 1, cv2.LINE_AA)
    text(frame, "CAMERA", 120, 26, MUTED, 0.34)
    text(frame, state["camera_id"], 176, 26, TEXT, 0.4)
    text(frame, "CONTEXT", 120, 38, MUTED, 0.34)
    text(frame, state["context"], 176, 38, TEXT, 0.4)

    x = 320
    for hazard in [h.value for h in Hazard]:
        on = hazard in state["hazards"]
        label = hazard.replace("_", " ")
        width = int(cv2.getTextSize(label, FONT, 0.34, 1)[0][0]) + 24
        cv2.rectangle(frame, (x, 13), (x + width, 32), LINE if on else PANEL, 1)
        cv2.circle(frame, (x + 10, 23), 3, TRACKING if on else FAINT, -1)
        text(frame, label, x + 18, 27, TEXT if on else FAINT, 0.34)
        x += width + 7

    # ---- canvas ---------------------------------------------------------
    top = HEADER
    scale = min(CANVAS_W / annotated.shape[1], CANVAS_H / annotated.shape[0])
    view = cv2.resize(annotated, (int(annotated.shape[1] * scale),
                                  int(annotated.shape[0] * scale)))
    ox = (CANVAS_W - view.shape[1]) // 2
    oy = top + (CANVAS_H - view.shape[0]) // 2
    frame[oy:oy + view.shape[0], ox:ox + view.shape[1]] = view

    # ---- rail: the cascade as a signal chain ----------------------------
    rx = CANVAS_W
    cv2.rectangle(frame, (rx, top), (W, top + CANVAS_H), PANEL, -1)
    cv2.line(frame, (rx, top), (rx, top + CANVAS_H), LINE, 1)
    text(frame, "CASCADE", rx + 16, top + 26, MUTED, 0.38)

    y = top + 52
    for index, name in enumerate(("confidence", "persistence", "cooldown"), 1):
        level = state["levels"].get(name, {"seen": 0, "passed": 0})
        seen, passed = level["seen"], level["passed"]
        off = seen == 0
        text(frame, f"{index} {name}", rx + 16, y, FAINT if off else TEXT, 0.38)
        counts = f"{passed} / {seen}"
        tw = int(cv2.getTextSize(counts, MONO, 0.36, 1)[0][0])
        text(frame, counts, W - 16 - tw, y, MUTED, 0.36, MONO)
        cv2.line(frame, (rx + 16, y + 8), (W - 16, y + 8), SUPPRESSED, 2)
        if seen:
            end = rx + 16 + int((W - 32 - rx) * passed / seen)
            cv2.line(frame, (rx + 16, y + 8), (end, y + 8), TRACKING, 2)
        y += 34

    y += 12
    cv2.rectangle(frame, (rx + 16, y - 18), (W - 16, y + 22), (30, 26, 26), -1)
    cv2.line(frame, (rx + 16, y - 18), (rx + 16, y + 22), CONFIRMED, 2)
    text(frame, "CONFIRMED", rx + 28, y + 6, MUTED, 0.38)
    big = str(state["confirmed"])
    tw = int(cv2.getTextSize(big, MONO, 0.95, 2)[0][0])
    text(frame, big, W - 24 - tw, y + 12, CONFIRMED, 0.95, MONO, 2)

    y += 58
    text(frame, "REJECTED BY", rx + 16, y, MUTED, 0.36)
    y += 20
    for name, count in sorted(state["by_level"].items(), key=lambda kv: -kv[1]):
        if not count:
            continue
        text(frame, f"{name}", rx + 16, y, SUPPRESSED, 0.36)
        tw = int(cv2.getTextSize(str(count), MONO, 0.36, 1)[0][0])
        text(frame, str(count), W - 16 - tw, y, SUPPRESSED, 0.36, MONO)
        y += 18

    # ---- timeline: density contrast is the argument ---------------------
    ty = top + CANVAS_H
    cv2.rectangle(frame, (0, ty), (W, ty + TIMELINE), PANEL, -1)
    cv2.line(frame, (0, ty), (W, ty), LINE, 1)
    text(frame, "CANDIDATES", 16, ty + 15, MUTED, 0.32)
    text(frame, "CONFIRMED", 16, ty + 33, MUTED, 0.32)

    # The density contrast between these two rows IS the argument, so both are
    # drawn as continuous tracks: every processed frame gets a mark, faint when
    # nothing happened. An earlier version drew a tick only where something was
    # proposed, which left the candidates row looking sparser than the
    # confirmed row and inverted the very comparison the panel exists to make.
    rows = state["timeline"][-240:]
    if rows:
        step = max(1, (W - 130) // max(len(rows), 1))
        for i, row in enumerate(rows):
            px = 110 + i * step
            if px > W - 12:
                break
            if row["proposed"]:
                height = min(12, 3 + row["proposed"])
                cv2.line(frame, (px, ty + 17), (px, ty + 17 - height), MUTED, 1)
            else:
                cv2.line(frame, (px, ty + 17), (px, ty + 16), FAINT, 1)

            if row["confirmed"]:
                cv2.line(frame, (px, ty + 36), (px, ty + 25), CONFIRMED, 2)
            else:
                cv2.line(frame, (px, ty + 36), (px, ty + 35), FAINT, 1)

    # ---- metric strip ---------------------------------------------------
    my = ty + TIMELINE
    cv2.rectangle(frame, (0, my), (W, my + METRICS), (28, 24, 24), -1)
    cv2.line(frame, (0, my), (W, my), LINE, 1)
    proposed = max(1, state["proposed"])
    filtered = sum(v for k, v in state["by_level"].items() if k != "cooldown")
    deduped = state["by_level"].get("cooldown", 0)
    # NO LATENCY FIGURE HERE, deliberately. Detector latency is wall-clock and
    # varies run to run, so burning it into the frame made the recording
    # non-reproducible: two renders of the same clip differed, and the
    # difference was confined entirely to the 63x25 pixel box this readout
    # occupied. Latency is a property of the machine rather than of the method,
    # so it is reported in the run summary and the sidecar instead, where it
    # can carry the hardware it was measured on. Everything shown on the frame
    # is a count derived from the detections.
    # Two numbers, not one. "Suppressed" conflated rejecting a candidate as not
    # credible with declining to re-report an event already sent, and those
    # support completely different claims — on the fire clip the single figure
    # read 90%, of which 80 points were repeat alerts about the same fire.
    metrics = [("SEEN", str(state["proposed"])),
               ("CONFIRMED", str(state["confirmed"])),
               ("FILTERED", f"{filtered}  ({filtered / proposed:.0%})"),
               ("DEDUPLICATED", f"{deduped}  ({deduped / proposed:.0%})"),
               ("FRAMES", str(len(state["timeline"])))]
    x = 16
    for label, value in metrics:
        text(frame, label, x, my + 16, MUTED, 0.32)
        text(frame, value, x, my + 34, TEXT, 0.46, MONO)
        x += 190

    # ---- attribution, burned in because a licence requires it -----------
    cy = my + METRICS
    cv2.rectangle(frame, (0, cy), (W, H), GROUND, -1)
    text(frame, spec["attribution"], 16, cy + 15, FAINT, 0.32)
    return frame


SPLIT_CANVAS_W = (W - 2) // 2


def split_height(source_w: int, source_h: int) -> int:
    """Output height for the split render: the footage decides, not a slot.

    A fixed-height slot letterboxed a 16:9 clip into acres of black. Half the
    width is fixed; scale the source to it and let the frame be exactly tall
    enough for header, footage, metrics and the credit line.
    """
    view_h = int(round(source_h * (SPLIT_CANVAS_W / source_w)))
    return HEADER + view_h + METRICS + CREDIT


def compose_split(raw_view, validated_view, state: dict, spec: dict,
                  out_h: int) -> np.ndarray:
    """The comparison nobody else shows: everyone's demo against ours.

    Left, every detection the raw detector proposes, drawn the way every
    showcase draws them. Right, only what survives validation. The point is
    not that the left side is wrong — it is exactly what the detector was
    built to do — but that nobody could sit in front of it, and something
    has to stand between it and a human.
    """
    frame = np.full((out_h, W, 3), GROUND, np.uint8)
    view_h = out_h - HEADER - METRICS - CREDIT

    cv2.rectangle(frame, (0, 0), (W, HEADER), PANEL, -1)
    cv2.line(frame, (0, HEADER), (W, HEADER), LINE, 1)
    brand = "V I G I A"
    text(frame, brand, 16, 28, TEXT, 0.5)
    second_i = 16 + int(cv2.getTextSize("V I G ", FONT, 0.5, 1)[0][0])
    cv2.line(frame, (second_i + 1, 12), (second_i + 6, 7), TEXT, 1, cv2.LINE_AA)
    text(frame, "RAW DETECTOR", 200, 26, MUTED, 0.36)
    text(frame, "every proposal, no validation", 200, 38, FAINT, 0.32)
    text(frame, "VALIDATED", W // 2 + 20, 26, CONFIRMED, 0.36)
    text(frame, "what actually reaches a person", W // 2 + 20, 38, FAINT, 0.32)

    for view, offset in ((raw_view, 0), (validated_view, W // 2 + 1)):
        sized = cv2.resize(view, (SPLIT_CANVAS_W, view_h))
        frame[HEADER:HEADER + view_h, offset:offset + SPLIT_CANVAS_W] = sized
    cv2.line(frame, (W // 2, HEADER), (W // 2, HEADER + view_h), LINE, 2)

    my = HEADER + view_h
    cv2.rectangle(frame, (0, my), (W, my + METRICS), (28, 24, 24), -1)
    cv2.line(frame, (0, my), (W, my), LINE, 1)
    proposed = max(1, state["proposed"])
    filtered = sum(v for k, v in state["by_level"].items() if k != "cooldown")
    deduped = state["by_level"].get("cooldown", 0)
    text(frame, "WOULD HAVE ALERTED", 16, my + 16, MUTED, 0.32)
    text(frame, str(state["proposed"]), 16, my + 36, TEXT, 0.55, MONO)
    text(frame, "FILTERED", 220, my + 16, MUTED, 0.32)
    text(frame, f"{filtered}", 220, my + 36, SUPPRESSED, 0.55, MONO)
    text(frame, "REPEATS WITHHELD", 340, my + 16, MUTED, 0.32)
    text(frame, f"{deduped}", 340, my + 36, SUPPRESSED, 0.55, MONO)
    text(frame, "ALERTS SENT", W // 2 + 20, my + 16, MUTED, 0.32)
    text(frame, str(state["confirmed"]), W // 2 + 20, my + 36, CONFIRMED,
         0.55, MONO)
    text(frame, "the difference is what a person is asked to look at",
         W // 2 + 150, my + 30, FAINT, 0.34)

    cy = my + METRICS
    cv2.rectangle(frame, (0, cy), (W, out_h), GROUND, -1)
    text(frame, spec["attribution"], 16, cy + 15, FAINT, 0.32)
    return frame


def replace_detection_state(detection):
    """A copy of the detection as the raw view sees it: alarmed, tracked.

    A shallow copy is not enough — mutating rejected_by on the original would
    corrupt the validated view drawn from the same objects.
    """
    import copy

    twin = copy.copy(detection)
    twin.rejected_by = None
    return twin


def render(spec: dict, clips_dir: Path, out: Path, fps: float,
           split: bool = False) -> dict:
    if not spec["publishable"]:
        raise PermissionError(
            f"{spec['clip']} is not publishable: {spec['licence']}. "
            f"{spec['why']}"
        )

    camera = Camera(spec["clip"].replace(".mp4", ""),
                    str(clips_dir / spec["clip"]),
                    CameraContext(spec["context"]),
                    hazards=frozenset({Hazard(spec["hazard"])}))

    if split:
        probe = cv2.VideoCapture(str(clips_dir / spec["clip"]))
        source_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        source_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        probe.release()
        out_h = split_height(source_w, source_h)
    else:
        out_h = H
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W, out_h))
    state = {
        "camera_id": camera.camera_id, "context": camera.context.value,
        "hazards": [spec["hazard"]], "proposed": 0, "confirmed": 0,
        "latency_ms": 0.0, "levels": {}, "by_level": {}, "timeline": [],
        "suppression_rate": 0.0, "latencies": [],
    }

    trails = TrailBuffer()

    raw_trails = TrailBuffer()

    def on_result(result, frame):
        suppressed = [d for d in result.detections if d.rejected_by]
        for detection in suppressed:
            state["by_level"][detection.rejected_by] = \
                state["by_level"].get(detection.rejected_by, 0) + 1
        state["proposed"] += len(result.detections)
        state["confirmed"] += len(result.events)
        state["latency_ms"] = result.latency_ms
        state["latencies"].append(result.latency_ms)
        state["levels"] = {
            level["name"]: level
            for level in pipeline.validators[result.hazard].stats_dict()["levels"]
        }
        state["timeline"].append({"proposed": len(result.detections),
                                  "confirmed": len(result.events)})
        state["suppression_rate"] = (
            1 - state["confirmed"] / state["proposed"] if state["proposed"] else 0.0
        )
        if split:
            # Left: the raw view — every detection drawn as if it alarmed,
            # which is how every other demo would draw it. The suppressed flag
            # is ignored on purpose: this is the counterfactual.
            raw = [replace_detection_state(d) for d in result.detections]
            raw_view = draw(frame.image, raw, events=[], show_suppressed=False,
                            trails=raw_trails)
            validated_view = draw(frame.image, result.detections,
                                  result.events, show_suppressed=False,
                                  trails=trails)
            writer.write(compose_split(raw_view, validated_view, state,
                                       spec, out_h))
        else:
            annotated = draw(frame.image, result.detections, result.events,
                             trails=trails)
            writer.write(compose(annotated, state, spec))

    pipeline = CameraPipeline(camera, on_result=on_result)
    pipeline.warmup()
    pipeline.run()
    writer.release()

    latencies = sorted(state["latencies"])
    median = latencies[len(latencies) // 2] if latencies else 0.0
    return {"clip": spec["clip"], "output": str(out),
            "frames": len(state["timeline"]),
            "proposed": state["proposed"], "confirmed": state["confirmed"],
            "suppression_rate": round(state["suppression_rate"], 4),
            "median_latency_ms": round(median, 1),
            "licence": spec["licence"], "attribution": spec["attribution"]}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=REPO_ROOT / "clips")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "demo")
    parser.add_argument("--clip", default=None, help="render only this clip")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--split", action="store_true",
                        help="also render the raw-vs-validated split screen")
    args = parser.parse_args()

    manifest_path = args.clips / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}. Build clips first:\n"
              f"    python scripts/make_demo_clips.py", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    args.out.mkdir(parents=True, exist_ok=True)
    results, refused = [], []

    for spec in manifest["clips"]:
        if args.clip and not spec["clip"].startswith(args.clip):
            continue
        if not spec["publishable"]:
            refused.append(spec)
            continue
        if not spec.get("temporal", True):
            refused.append(spec)
            continue
        target = args.out / f"demo_{spec['hazard']}.mp4"
        print(f"rendering {spec['clip']} -> {target.name}")
        results.append(render(spec, args.clips, target, args.fps))
        if args.split:
            side = args.out / f"split_{spec['hazard']}.mp4"
            print(f"rendering {spec['clip']} -> {side.name} (split-screen)")
            results.append(render(spec, args.clips, side, args.fps, split=True))

    for spec in refused:
        reason = ("not publishable — " + spec["licence"]) if not spec["publishable"] \
                 else "not a time sequence"
        print(f"REFUSED {spec['clip']}: {reason}", file=sys.stderr)

    print()
    for result in results:
        print(f"  {result['output']}")
        print(f"    {result['frames']} frames · {result['proposed']} proposed · "
              f"{result['confirmed']} confirmed · "
              f"{result['suppression_rate']:.1%} suppressed · "
              f"{result['median_latency_ms']:.0f} ms median")

    if results:
        sidecar = args.out / "MANIFEST.json"
        sidecar.write_text(json.dumps({
            "note": "Rendered by scripts/record_demo.py. The video frames are "
                    "reproducible: re-rendering the same clip produces a "
                    "byte-identical file. Latency is reported here rather than "
                    "drawn on the frames, because it is wall-clock and would "
                    "make the recording differ between runs and machines.",
            "renders": results,
        }, indent=2), encoding="utf-8")
        print(f"\n  wrote {sidecar}")

    if refused:
        print(f"\n{len(refused)} clip(s) refused. A recording is published "
              f"material; a clip needs a\nlicence that permits publication AND "
              f"genuine temporal continuity to appear in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
