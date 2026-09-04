"""Traffic incident detector.

Wraps a YOLO11x fine-tuned for traffic accident detection, published under MIT.

This replaces the approach in the project's original fork, which flagged a
collision whenever two vehicle bounding boxes came close and overlapped. That
heuristic fails constantly in real footage: vehicles in adjacent lanes overlap
in 2D image space all the time, and a monocular camera has no depth to tell an
overlap from a collision. A model trained on actual accident imagery does not
have that failure mode.

Source : https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x
Licence: MIT
Reported by the author (NOT measured by us): mAP@0.5 0.826, mAP@0.5:0.95 0.600,
precision 0.808, recall 0.759, F1 0.782, accident-class recall 0.855.
The author states it is not validated for critical autonomous driving use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from vigia.detectors.base import Backend, ONNXDetector
from vigia.registry import Viewpoint
from vigia.types import Hazard

TRAFFIC_IMGSZ = 640
TRAFFIC_CONF = 0.25
TRAFFIC_IOU = 0.45

#: epoch61, not epoch14. Both checkpoints are published and the upstream
#: infer.py points at epoch14, but compared on the author's own test
#: images epoch61 gave one accident per image on all four (conf
#: 0.58-0.70) where epoch14 gave three detections on fig1.
DEFAULT_MODEL_PATH = Path("models/traffic/enos_traffic_epoch61.onnx")

# Index order must match the exported model's `names` mapping. Verified at
# export time by tools/export_onnx.py, which prints the class dict.
TRAFFIC_CLASSES: tuple[str, ...] = ("accident", "vehicle")


class TrafficDetector(ONNXDetector):
    """Detects traffic accidents and vehicles.

    The model emits both classes, but only `accident` is a hazard. Vehicles are
    context: useful for the operator view and for future work on rate-of-change
    signals, but they must never reach the validator as candidate events or
    every car on the road becomes an alert.
    """

    hazard = Hazard.TRAFFIC
    viewpoints: frozenset = frozenset({Viewpoint.GROUND, Viewpoint.OBLIQUE})
    class_names: Sequence[str] = TRAFFIC_CLASSES

    #: Only these class names are treated as hazards.
    hazard_classes: frozenset[str] = frozenset({"accident"})

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        imgsz: int = TRAFFIC_IMGSZ,
        conf_threshold: float = TRAFFIC_CONF,
        iou_threshold: float = TRAFFIC_IOU,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
        static_model_path: Optional[str | Path] = None,
        include_context: bool = False,
    ) -> None:
        """
        Args:
            include_context: if True, also return `vehicle` detections. They are
                tagged `extra["context"] = True` so the pipeline can draw them
                without ever routing them to the validator.
        """
        self.include_context = include_context
        super().__init__(
            model_path,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            providers=providers,
            backend=backend,
            static_model_path=static_model_path,
        )

    def detect(self, frame):  # type: ignore[override]
        detections = super().detect(frame)

        hazards = []
        for detection in detections:
            is_hazard = detection.label in self.hazard_classes
            if is_hazard:
                hazards.append(detection)
            elif self.include_context:
                detection.extra["context"] = True
                hazards.append(detection)
        return hazards
