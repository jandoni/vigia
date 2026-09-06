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


class TrailBuffer:
    """Recent centre points per track, for drawing motion trails.

    A trail is evidence, not decoration: it shows that a confirmed detection
    is the SAME object followed across frames, which is precisely what the
    persistence level asserts. Bounded per track and pruned when a track
    disappears, so a long replay cannot grow it without limit.
    """

    def __init__(self, length: int = 24) -> None:
        self.length = length
        self._points: dict[int, list[tuple[int, int]]] = {}

    def update(self, detections: Sequence[Detection]) -> None:
        seen = set()
        for detection in detections:
            if detection.track_id < 0 or not detection.passed:
                continue
            x1, y1, x2, y2 = detection.box.as_xyxy()
            centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            self._points.setdefault(detection.track_id, []).append(centre)
            if len(self._points[detection.track_id]) > self.length:
                self._points[detection.track_id].pop(0)
            seen.add(detection.track_id)
        for track_id in [k for k in self._points if k not in seen]:
            trail = self._points[track_id]
            trail.pop(0)                      # fade out rather than vanish
            if not trail:
                del self._points[track_id]

    def get(self, track_id: int) -> list[tuple[int, int]]:
        return self._points.get(track_id, [])


def _corners(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             colour: tuple[int, int, int], thickness: int) -> None:
    """Corner brackets instead of a full rectangle.

    Same information, less ink over the evidence: the footage stays readable
    inside the box, which matters when the thing inside is the point.
    """
    run = max(6, min(18, (x2 - x1) // 4, (y2 - y1) // 4))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(image, (cx, cy), (cx + dx * run, cy), colour, thickness,
                 cv2.LINE_AA)
        cv2.line(image, (cx, cy), (cx, cy + dy * run), colour, thickness,
                 cv2.LINE_AA)


def _chip(image: np.ndarray, label: str, value: str, x: int, y: int,
          colour: tuple[int, int, int]) -> None:
    """A label chip: name in state colour, value dimmed, on one dark plate."""
    scale = 0.42
    (lw, lh), baseline = cv2.getTextSize(label, FONT, scale, 1)
    (vw, _), _ = cv2.getTextSize(value, FONT, scale, 1)
    pad, gap = 5, 7 if value else 0
    width = lw + (vw + gap if value else 0) + 2 * pad
    x = max(0, min(x, image.shape[1] - width))    # keep the plate on-frame
    top = max(0, y - lh - baseline - 2 * pad)
    plate = np.array(COLOUR_PANEL, np.float64)
    x2 = x + width
    y2 = top + lh + baseline + 2 * pad
    roi = image[top:y2, x:min(x2, image.shape[1])]
    if roi.size:                                    # translucent, not opaque
        image[top:y2, x:min(x2, image.shape[1])] = (
            roi * 0.25 + plate * 0.75).astype(np.uint8)
    cv2.putText(image, label, (x + pad, top + lh + pad), FONT, scale, colour,
                1, cv2.LINE_AA)
    if value:
        cv2.putText(image, value, (x + pad + lw + gap, top + lh + pad), FONT,
                    scale, (185, 185, 185), 1, cv2.LINE_AA)


def draw(
    frame_image: np.ndarray,
    detections: Sequence[Detection],
    events: Sequence[Event] = (),
    *,
    show_suppressed: bool = True,
    mask: Optional[np.ndarray] = None,
    trails: Optional[TrailBuffer] = None,
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
        bool_mask = mask.astype(bool)
        tint = np.zeros_like(canvas)
        tint[bool_mask] = (200, 140, 60)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
        canvas[edges > 0] = (220, 170, 90)          # crisp water boundary

    if trails is not None:
        trails.update(detections)

    confirmed_tracks = {event.track_id for event in events if event.track_id >= 0}

    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box.as_xyxy())
        name = detection.label or detection.hazard.value

        if detection.track_id in confirmed_tracks and detection.passed:
            colour, thickness = COLOUR_CONFIRMED, 2
            label, value = name, f"{detection.confidence:.2f}"
        elif detection.passed:
            colour, thickness = COLOUR_TRACKING, 1
            label, value = name, f"{detection.confidence:.2f}"
        else:
            if not show_suppressed:
                continue
            colour, thickness = COLOUR_SUPPRESSED, 1
            # Naming the level that rejected it is the whole point: the viewer
            # sees not just that it was dropped but which test dropped it.
            label, value = "suppressed", detection.rejected_by or ""

        # The trail first, so the box reads on top of it.
        if trails is not None and detection.track_id >= 0 and detection.passed:
            points = trails.get(detection.track_id)
            for a, b in zip(points, points[1:]):
                cv2.line(canvas, a, b, colour, 1, cv2.LINE_AA)
            if points:
                cv2.circle(canvas, points[-1], 2, colour, -1, cv2.LINE_AA)

        _corners(canvas, x1, y1, x2, y2, colour, thickness)
        _chip(canvas, label, value, x1, y1, colour)

    return canvas
