"""Fire and smoke detector.

Wraps the PyroNear "sensitive detector" — a YOLO11s trained by the PyroNear
association on their pyro-dataset corpus and deployed on a real network of
lookout-tower cameras across France, Spain and Chile.

We do not train this model. We run it as published and measure it. See NOTICE
for attribution and the licence position.

Source : https://huggingface.co/pyronear/yolo11s_sensitive-detector
Licence: Apache-2.0 (per the model repository)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from vigia.detectors.base import Backend, ONNXDetector
from vigia.registry import Viewpoint
from vigia.types import Hazard

# The upstream model card recommends these. It is a "sensitive" detector by
# design: a low confidence floor and a very low NMS IoU, deliberately trading
# precision for recall on the assumption that something downstream filters the
# result. That something is our temporal validator — which is exactly why this
# model is the right baseline for the experiment we want to run.
PYRONEAR_IMGSZ = 1024
PYRONEAR_CONF = 0.20
PYRONEAR_IOU = 0.01

DEFAULT_MODEL_PATH = Path("models/fire/pyronear_yolo11s_sensitive.onnx")


class FireDetector(ONNXDetector):
    """Smoke and fire plume detector.

    The upstream model is single-class; its one class is named `item` in the
    ONNX metadata, which is meaningless downstream, so we relabel it `smoke`.
    """

    hazard = Hazard.FIRE
    #: PyroNear's envelope: fixed-mount wide-field cameras viewing a landscape
    #: horizon.
    viewpoints: frozenset = frozenset({Viewpoint.GROUND, Viewpoint.OBLIQUE})
    class_names: Sequence[str] = ("smoke",)

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        imgsz: int = PYRONEAR_IMGSZ,
        conf_threshold: float = PYRONEAR_CONF,
        iou_threshold: float = PYRONEAR_IOU,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
        static_model_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            model_path,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            providers=providers,
            backend=backend,
            static_model_path=static_model_path,
        )
