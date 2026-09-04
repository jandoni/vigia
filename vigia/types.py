"""Core data types shared across the VIGÍA pipeline.

Deliberately dependency-light: only numpy. Everything downstream — detectors,
validator, IO — speaks in these types so that any detector is interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Hazard(str, Enum):
    """Hazard classes VIGÍA can detect.

    String-valued so they serialise cleanly into REGISTRY.yaml and JSON events.
    """

    FIRE = "fire"
    FLOOD = "flood"
    TRAFFIC = "traffic"
    DROWNING = "drowning"
    BUILDING_ACCESS = "building_access"


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in original-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "Box") -> float:
        """Intersection over union. Used by the validator to associate
        detections across consecutive frames."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    """A single raw detection from one detector on one frame.

    This is a *candidate*, not an alert. It becomes an alert only if the
    temporal validator confirms it.
    """

    hazard: Hazard
    box: Box
    confidence: float
    label: str = ""
    frame_index: int = -1
    timestamp: float = 0.0
    camera_id: str = ""
    # Set by the validator when a detection is rejected, naming the cascade
    # level that rejected it. None means it passed every level.
    rejected_by: Optional[str] = None
    # Set by the validator once the tracker has associated this detection with
    # a track. -1 means not yet tracked.
    track_id: int = -1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.rejected_by is None


@dataclass
class Frame:
    """One decoded video frame plus its provenance."""

    image: np.ndarray  # HWC, BGR, uint8 — OpenCV convention
    index: int
    timestamp: float
    camera_id: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the original image."""
        return (self.image.shape[0], self.image.shape[1])


@dataclass
class Event:
    """A confirmed hazard event — a detection that survived the full cascade
    and persisted long enough to be trusted.

    This is what gets dispatched. Everything else is logged and suppressed.
    """

    hazard: Hazard
    box: Box
    confidence: float
    camera_id: str
    first_seen_frame: int
    confirmed_frame: int
    first_seen_time: float
    confirmed_time: float
    supporting_detections: int
    track_id: int = -1

    @property
    def frames_to_confirm(self) -> int:
        return self.confirmed_frame - self.first_seen_frame

    @property
    def seconds_to_confirm(self) -> float:
        return self.confirmed_time - self.first_seen_time
