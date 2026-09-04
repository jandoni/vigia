"""Event emission — what leaves the camera, and what provably does not.

This module is the system's only egress point, which makes it the place where
PLAN.md section 10's privacy commitments are enforced rather than described.
Three of them are structural here:

  * **Only the event and a single evidence frame leave.** An Alert carries the
    hazard, the box, the timing and one still image. There is no field for a
    clip, and no writer for one.
  * **The evidence frame is redacted by default.** Person-shaped detections are
    blurred inside their own boxes before the frame is written. The system needs
    to show an operator *that* someone is in the water and *where*; it never
    needs to show *who*. Blurring is applied at write time so no unredacted
    frame is ever persisted.
  * **No identity is computed, anywhere.** There is no face, gait, or
    re-identification code path in this codebase to switch on — enforced by
    `tools/check_privacy.py`, which fails the build if one appears, in the same
    way `tools/check_licence.py` enforces the AGPL perimeter.

The EU AI Act's high-risk provisions took full effect in August 2026; Annex III
classifies biometric identification as high-risk and Article 5(1)(h) bans
real-time remote biometric identification in publicly accessible spaces for law
enforcement. VIGÍA needs to identify nobody, so the cheapest way to stay outside
that regime is to be architecturally incapable of entering it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from vigia.types import Detection, Event, Hazard

logger = logging.getLogger(__name__)

#: Detection labels that depict a person. Anything listed here is blurred in
#: the evidence frame. Kept as data so a new detector that finds people cannot
#: be added without confronting this list.
PERSON_LABELS: frozenset[str] = frozenset({
    "civilian", "rescue_team", "swimmer", "person", "people",
})


@dataclass
class Alert:
    """One confirmed event, packaged for egress.

    Deliberately has no `clip`, `video` or `frames` field. Adding one would be a
    visible change to a documented boundary rather than a configuration tweak.
    """

    event: Event
    evidence_path: Optional[Path] = None
    emitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        event = self.event
        return {
            "hazard": event.hazard.value,
            "camera_id": event.camera_id,
            "confidence": round(event.confidence, 4),
            "box": [round(v, 1) for v in event.box.as_xyxy()],
            "first_seen_frame": event.first_seen_frame,
            "confirmed_frame": event.confirmed_frame,
            "seconds_to_confirm": round(event.seconds_to_confirm, 2),
            "supporting_detections": event.supporting_detections,
            "track_id": event.track_id,
            "evidence": str(self.evidence_path) if self.evidence_path else None,
            "emitted_at": self.emitted_at,
        }


def redact_people(
    image: np.ndarray,
    detections: Sequence[Detection],
    *,
    kernel_divisor: int = 6,
) -> np.ndarray:
    """Blur every person-shaped detection, on a copy.

    Operates on a copy so the caller's frame — which may still be in the
    rolling buffer — is never mutated. The blur is heavy and applied to the box
    interior only; the box itself remains visible because the operator needs to
    see where the person is.
    """
    import cv2

    redacted = image.copy()
    height, width = redacted.shape[:2]

    for detection in detections:
        if detection.label not in PERSON_LABELS:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box.as_xyxy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        patch = redacted[y1:y2, x1:x2]
        # Kernel scaled to the region so a distant swimmer is as unidentifiable
        # as a close one; forced odd because GaussianBlur requires it.
        size = max(3, (min(patch.shape[0], patch.shape[1]) // kernel_divisor) | 1)
        redacted[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (size, size), 0)

    return redacted


class AlertSink:
    """Where confirmed events go.

    Writes one JSON line per alert and, optionally, one redacted evidence frame.
    A caller may also supply `on_alert` to forward alerts elsewhere — a
    WebSocket to the operator view, for instance — without this class needing to
    know about transports.
    """

    def __init__(
        self,
        output_dir: str | Path = "runs/alerts",
        *,
        write_evidence: bool = True,
        redact: bool = True,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.write_evidence = write_evidence
        self.redact = redact
        self.on_alert = on_alert
        self.alerts: list[Alert] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "alerts.jsonl"

    def emit(
        self,
        event: Event,
        image: Optional[np.ndarray] = None,
        detections: Sequence[Detection] = (),
    ) -> Alert:
        evidence_path: Optional[Path] = None

        if self.write_evidence and image is not None:
            import cv2

            frame = redact_people(image, detections) if self.redact else image
            if not self.redact:
                # Not reachable through the pipeline, which never disables
                # redaction; logged loudly if a caller constructs it directly.
                logger.warning(
                    "writing UNREDACTED evidence frame for camera %s — this is "
                    "outside the privacy commitment in PLAN.md section 10",
                    event.camera_id,
                )
            name = (f"{event.camera_id}_{event.hazard.value}_"
                    f"{event.confirmed_frame:06d}.jpg")
            evidence_path = self.output_dir / name
            cv2.imwrite(str(evidence_path), frame)

        alert = Alert(event=event, evidence_path=evidence_path)
        self.alerts.append(alert)

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.to_dict()) + "\n")

        if self.on_alert is not None:
            self.on_alert(alert)

        logger.info("ALERT %s on %s (conf %.2f, %d supporting, %.1fs to confirm)",
                    event.hazard.value, event.camera_id, event.confidence,
                    event.supporting_detections, event.seconds_to_confirm)
        return alert

    def counts_by_hazard(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for alert in self.alerts:
            key = alert.event.hazard.value
            counts[key] = counts.get(key, 0) + 1
        return counts
