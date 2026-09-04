"""Flood detector: water segmentation plus level and rate of rise.

The keystone hazard. The 2024 Valencia DANA killed 223 people as water rose
through streets, garages and vehicles; municipal cameras were watching it
happen.

Two things distinguish this detector from the others in VIGÍA:

  * It is trained by us rather than taken off the shelf — DeepLabV3 with a
    MobileNetV3 backbone (torchvision, BSD-3-Clause) fine-tuned on ATLANTIS.
    That makes it the only detector in the system entirely free of AGPL.
  * It produces a *trend*, not just a detection. Segmenting water is a solved
    problem; knowing the water is rising, and how fast, is the signal the flood
    literature says is missing and the one that buys evacuation time.

The temporal validator still applies, via the mask-to-box bridge in
`vigia.detectors.segmentation`. That bridge was validated on real ATLANTIS
flood masks before being adopted: masks are a single connected component
99.8% of the time, frame-to-frame box IoU under simulated rise with
segmentation noise held at a median 0.964 against a 0.20 tracker threshold,
and the conversion costs 0.31 ms per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vigia.detectors.base import Backend
from vigia.detectors.segmentation import ONNXSegmenter, SegmentationResult
from vigia.flood.level import (
    LevelReading, ReferenceLine, RiseEstimate, Trend, WaterLevelTracker,
)
from vigia.registry import Viewpoint
from vigia.types import Detection, Frame, Hazard

DEFAULT_MODEL_PATH = Path("models/flood/atlantis_water_deeplabv3.onnx")

#: A second set of weights for the viewpoint the first one cannot serve.
#: ATLANTIS is ground-level and oblique photography; this one is trained on
#: FloodNet, which is UAV imagery looking straight down. Same architecture,
#: same binary water task, different sensing problem — so the choice is made
#: by the camera's declared viewpoint rather than by a flag.
AERIAL_MODEL_PATH = Path("models/flood/floodnet_aerial_deeplabv3.onnx")

# Water has no reliable colour: it mirrors the sky, carries silt, and turns
# brown in a flood. The validator's colour level must stay off for this hazard.
FLOOD_IMGSZ = 512
FLOOD_THRESHOLD = 0.5


@dataclass
class FloodObservation:
    """Everything one frame tells us about the water."""

    segmentation: SegmentationResult
    reading: LevelReading
    rise: RiseEstimate

    @property
    def detections(self) -> list[Detection]:
        return self.segmentation.detections

    @property
    def coverage(self) -> float:
        return self.segmentation.coverage

    @property
    def is_rising(self) -> bool:
        return self.rise.trend is Trend.RISING and self.rise.confident


class FloodDetector(ONNXSegmenter):
    """Binary water segmentation with level tracking.

    Args:
        reference_line: per-camera calibration for reading water height. Without
            it the detector still works and still reports a trend, using the
            uncalibrated area signal — which means a camera can be added to the
            network and be useful immediately, with calibration as an upgrade
            rather than a prerequisite.
    """

    hazard = Hazard.FLOOD
    #: Trained on ATLANTIS, which is ground-level and oblique photography of
    #: water bodies. It has never seen a straight-down view and fails silently
    #: on one — measured at up to 0.397 coverage on FloodNet UAV frames whose
    #: water is a narrow canal, with mown grass tinted as water. Declaring the
    #: envelope turns that into a refusal instead of a confident wrong number.
    viewpoints: frozenset = frozenset({Viewpoint.GROUND, Viewpoint.OBLIQUE})

    @classmethod
    def model_for(cls, viewpoint: Viewpoint) -> Path:
        """Which weights serve this viewpoint."""
        return (AERIAL_MODEL_PATH if viewpoint is Viewpoint.NADIR_AERIAL
                else DEFAULT_MODEL_PATH)

    @classmethod
    def supported_viewpoints(cls) -> frozenset:
        """Viewpoints this detector can actually serve RIGHT NOW.

        Nadir is included only when the aerial weights are present on disk.
        Declaring the capability while the file is missing would turn a clear
        refusal into a FileNotFoundError halfway through a run, which is a
        worse failure and a later one.
        """
        supported = {Viewpoint.GROUND, Viewpoint.OBLIQUE}
        if AERIAL_MODEL_PATH.exists():
            supported.add(Viewpoint.NADIR_AERIAL)
        return frozenset(supported)
    class_name = "water"
    positive_class = 1

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        imgsz: int = FLOOD_IMGSZ,
        threshold: float = FLOOD_THRESHOLD,
        min_region_fraction: float = 0.002,
        reference_line: Optional[ReferenceLine] = None,
        window_seconds: float = 300.0,
        rising_threshold: float = 0.005,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
        static_model_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            model_path,
            imgsz=imgsz,
            threshold=threshold,
            min_region_fraction=min_region_fraction,
            providers=providers,
            backend=backend,
            static_model_path=static_model_path,
        )
        self.tracker = WaterLevelTracker(
            reference_line,
            window_seconds=window_seconds,
            rising_threshold=rising_threshold,
        )

    # ------------------------------------------------------------------ #

    def observe(self, frame: Frame) -> FloodObservation:
        """Segment, measure the level, and update the trend."""
        segmentation = self.segment(frame)
        reading = self.tracker.measure(segmentation.mask, frame.timestamp)
        rise = self.tracker.estimate_rise(frame.timestamp)

        # Attach the water-specific context to every detection so the operator
        # view and the alert payload carry it without a second lookup.
        for detection in segmentation.detections:
            detection.extra.update({
                "coverage": segmentation.coverage,
                "level_fraction": reading.level_fraction,
                "level_metres": reading.level_metres,
                "trend": rise.trend.value,
                "rate_per_min": (
                    rise.level_rate_per_min
                    if rise.level_rate_per_min is not None
                    else rise.area_rate_per_min
                ),
            })

        return FloodObservation(segmentation=segmentation, reading=reading, rise=rise)

    def detect(self, frame: Frame) -> list[Detection]:
        """Detector-compatible interface. Also advances the level tracker, so a
        caller that only wants boxes still gets a correct trend."""
        return self.observe(frame).detections

    def time_to_threshold(self, target_level: float = 0.5) -> Optional[float]:
        """Seconds until the water reaches `target_level` at the current rate.

        The operationally useful number: not "this street is flooding" but
        "this underpass is impassable in eleven minutes".
        """
        return self.tracker.time_to_threshold(target_level)

    def reset(self) -> None:
        """Clear level history. Required between videos in evaluation, or one
        clip's trend leaks into the next."""
        self.tracker.reset()
