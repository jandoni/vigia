#!/usr/bin/env python3
"""Qualitative results figure: one annotated frame per hazard.

    python scripts/make_qualitative_figure.py

A 38-page paper about detection contained no image of a detection — a gap a
reviewer notices before the abstract's second paragraph. This renders one,
under the same rules as every other published artefact:

  * frames come only from clips MANIFEST-marked publishable, with the
    required attribution in the caption;
  * the three temporal hazards show the VALIDATED view at a genuinely
    confirmed moment, drawn by the same overlay the system runs;
  * building access shows DETECTOR output only and is labelled as such —
    its footage has no temporal continuity, so showing a "confirmed" state
    would claim validation that cannot occur on stills.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2                                                    # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from vigia.pipeline import CameraPipeline                     # noqa: E402
from vigia.registry import Camera, CameraContext              # noqa: E402
from vigia.types import Frame, Hazard                         # noqa: E402
from vigia.web.overlay import TrailBuffer, draw               # noqa: E402

CLIPS = REPO_ROOT / "clips"
OUT = REPO_ROOT / "docs" / "figures"

PANEL_TITLES = {
    "fire": "(a) Wildfire smoke — confirmed at persistence 3",
    "flood": "(b) Flood — confirmed, water mask and level",
    "drowning": "(c) People in open water — confirmed",
    "building_access": "(d) Building access — detector output only "
                       "(no temporal validation possible on this footage)",
}


def best_confirmed_frame(spec: dict) -> tuple:
    """Replay the clip; keep the frame with the most confirmed events.

    For flood the detector's real product is the water mask, so the winning
    frame is re-segmented and the mask drawn under the confirmed boxes —
    showing only the derived boxes would sell the segmenter short.
    """
    camera = Camera(spec["clip"].replace(".mp4", ""),
                    str(CLIPS / spec["clip"]),
                    CameraContext(spec["context"]),
                    hazards=frozenset({Hazard(spec["hazard"])}))
    trails = TrailBuffer()
    best = {"count": -1, "image": None, "raw": None,
            "detections": None, "events": None}

    def on_result(result, frame):
        annotated = draw(frame.image, result.detections, result.events,
                         show_suppressed=False, trails=trails)
        if len(result.events) > best["count"]:
            best.update(count=len(result.events), image=annotated,
                        raw=frame.image.copy(),
                        detections=list(result.detections),
                        events=list(result.events))

    pipeline = CameraPipeline(camera, on_result=on_result)
    pipeline.warmup()
    pipeline.run()

    if spec["hazard"] == "flood" and best["raw"] is not None:
        detector = pipeline.detectors[Hazard.FLOOD]
        segmentation = detector.segment(
            Frame(image=best["raw"], index=0, timestamp=0.0, camera_id="fig"))
        best["image"] = draw(best["raw"], best["detections"], best["events"],
                             show_suppressed=False, mask=segmentation.mask)
    return best["image"], best["count"]


def detector_only_frame(spec: dict):
    """Building access: the detector's own output, honestly labelled.

    The clip is curated stills with 40 scene cuts in 66 frames, so no single
    frame index is safe to assume; scan and keep the frame with the most
    detections above the operating point.
    """
    from vigia.detectors.building_access import BuildingAccessDetector

    detector = BuildingAccessDetector()
    capture = cv2.VideoCapture(str(CLIPS / spec["clip"]))
    best = {"count": -1, "image": None}
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index % 3 == 0:
            # 0.45 rather than the 0.30 operating point, for legibility:
            # twenty overlapping chips bury the footage. Stated in the caption.
            detections = [d for d in detector.detect(
                Frame(image=image, index=index, timestamp=0.0,
                      camera_id="still"))
                if d.confidence >= 0.45]
            if len(detections) > best["count"]:
                for d in detections:
                    d.track_id = -1
                best["count"] = len(detections)
                best["image"] = draw(image, detections, events=(),
                                     show_suppressed=False)
        index += 1
    capture.release()
    return best["image"], best["count"]


def main() -> int:
    manifest = json.loads((CLIPS / "MANIFEST.json").read_text(encoding="utf-8"))
    specs = {s["hazard"]: s for s in manifest["clips"] if s["publishable"]}

    panels, credits = {}, []
    for hazard in ("fire", "flood", "drowning"):
        spec = specs[hazard]
        image, count = best_confirmed_frame(spec)
        print(f"{hazard}: best frame has {count} confirmed event(s)")
        panels[hazard] = image
        credits.append(spec["attribution"])
    spec = specs["building_access"]
    panels["building_access"], n = detector_only_frame(spec)
    print(f"building_access: {n} detections at conf 0.45 (detector only)")
    credits.append(spec["attribution"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    fig.patch.set_facecolor("white")
    for ax, hazard in zip(axes.flat,
                          ("fire", "flood", "drowning", "building_access")):
        ax.imshow(cv2.cvtColor(panels[hazard], cv2.COLOR_BGR2RGB))
        ax.set_title(PANEL_TITLES[hazard], fontsize=9, loc="left", pad=6)
        ax.axis("off")
    fig.tight_layout(pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 190}), (".pdf", {})):
        fig.savefig(OUT / f"qualitative{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {OUT / 'qualitative.png'} and .pdf")
    print("\ncredits for the caption:")
    for credit in credits:
        print(f"  - {credit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
