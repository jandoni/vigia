"""Frame annotation for the operator view.

THE DESIGN CONSTRAINT, restated because it drives every choice here: everyone's
demo draws a box around a fire. What nobody shows is detections being
*rejected*. The validator suppresses most of what the detectors propose, and
watching that happen is the only visual proof of the one claim that is ours. So
this module draws both — confirmed events in accent, suppressed candidates in a
muted outline labelled with the cascade level that stopped them.

Colour is reserved strictly for state and never used decoratively. Three states
exist and nothing else gets a hue:

    confirmed   the validator accepted it — this is an alert
    suppressed  a detector proposed it and a named level rejected it
    tracking    accepted so far, still short of the persistence threshold

Rendering happens on a copy. The source frame may still be in the rolling
buffer, and the same frame is what an evidence image would be written from —
mutating it here would put display furniture into the archived evidence.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from vigia.types import Detection, Event

# BGR, control-room palette. Deliberately few.
COLOUR_CONFIRMED = (86, 220, 255)     # amber — the alert
COLOUR_SUPPRESSED = (110, 110, 110)   # grey — proposed and rejected
COLOUR_TRACKING = (170, 190, 120)     # muted teal — accepted, not yet persistent
COLOUR_TEXT = (240, 240, 240)
COLOUR_PANEL = (24, 24, 26)

FONT = cv2.FONT_HERSHEY_DUPLEX


def _label(image: np.ndarray, text: str, x: int, y: int,
           colour: tuple[int, int, int], scale: float = 0.45) -> None:
    """Text on a filled plate, so it stays readable over any footage."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    top = max(0, y - th - baseline - 4)
    cv2.rectangle(image, (x, top), (x + tw + 8, top + th + baseline + 4),
                  COLOUR_PANEL, -1)
    cv2.putText(image, text, (x + 4, top + th + 2), FONT, scale, colour, 1,
                cv2.LINE_AA)


def draw(
    frame_image: np.ndarray,
    detections: Sequence[Detection],
    events: Sequence[Event] = (),
    *,
    show_suppressed: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Annotate one frame with proposed and confirmed detections.

    `show_suppressed` exists so the demo can toggle the suppressed layer off and
    on. Turning it off makes the view look like everyone else's demo, which is
    the most direct way to show an audience what the validator is doing.
    """
    canvas = frame_image.copy()

    # Flood is a mask, not boxes. Drawn first so boxes sit on top of it.
    if mask is not None and mask.size:
        if mask.shape[:2] != canvas.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (canvas.shape[1], canvas.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(canvas)
        tint[mask.astype(bool)] = (200, 140, 60)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)

    confirmed_tracks = {event.track_id for event in events if event.track_id >= 0}

    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box.as_xyxy())

        if detection.track_id in confirmed_tracks and detection.passed:
            colour, thickness = COLOUR_CONFIRMED, 2
            text = f"{detection.label or detection.hazard.value} {detection.confidence:.2f}"
        elif detection.passed:
            colour, thickness = COLOUR_TRACKING, 1
            text = f"{detection.label or detection.hazard.value} {detection.confidence:.2f}"
        else:
            if not show_suppressed:
                continue
            colour, thickness = COLOUR_SUPPRESSED, 1
            # Naming the level that rejected it is the whole point: the viewer
            # sees not just that it was dropped but which test dropped it.
            text = f"suppressed · {detection.rejected_by}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        _label(canvas, text, x1, y1, colour)

    return canvas


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def placeholder(width: int = 960, height: int = 540, text: str = "no signal") -> np.ndarray:
    """A frame to serve before the pipeline has produced one."""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 1)
    cv2.putText(canvas, text, ((width - tw) // 2, (height + th) // 2),
                FONT, 0.7, (90, 90, 90), 1, cv2.LINE_AA)
    return canvas
