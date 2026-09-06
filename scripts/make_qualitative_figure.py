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
from vigia.pipeline import CameraPipeline                     # noqa: E402
from vigia.registry import Camera, CameraContext              # noqa: E402
from vigia.types import Frame, Hazard                         # noqa: E402
from vigia.web.overlay import TrailBuffer, draw               # noqa: E402

CLIPS = REPO_ROOT / "clips"
OUT = REPO_ROOT / "docs" / "figures"

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


def save_panel(image, name: str) -> None:
    """One annotated frame as a clean, high-quality image.

    No matplotlib chrome and no burned-in title: each frame goes into its own
    detector section with a formal LaTeX caption, so the panel must carry the
    footage and nothing else. Written as PNG because a camera frame is raster
    to begin with — wrapping a photograph in a vector container buys nothing.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{name}.png"), image,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"wrote {OUT / f'{name}.png'}")


def main() -> int:
    manifest = json.loads((CLIPS / "MANIFEST.json").read_text(encoding="utf-8"))
    specs = {s["hazard"]: s for s in manifest["clips"] if s["publishable"]}

    for hazard in ("fire", "flood", "drowning"):
        image, count = best_confirmed_frame(specs[hazard])
        print(f"{hazard}: best frame has {count} confirmed event(s)")
        save_panel(image, f"qual_{hazard}")

    image, n = detector_only_frame(specs["building_access"])
    print(f"building_access: {n} detections at conf 0.45 (detector only)")
    save_panel(image, "qual_building")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
