"""People-in-water detector.

The DANA killed people who were swept from roads and trapped in vehicles as
water rose. This detector looks for a person in open water — the situation that
actually killed, rather than the pool-drowning scenario originally planned,
which never fitted the disaster narrative and faces an unfavourable sensing
problem besides.

Model  : dronefreak/seadronessee-rfdetr-small (RF-DETR Small, fine-tuned)
Base   : Roboflow RF-DETR Small
Data   : SeaDronesSee (Varga, Kiefer et al., University of Tübingen, WACV 2022)
Licence: Apache-2.0 for the model; the dataset is CC0 1.0 — the least
         restrictive licence in the whole project.

BE HONEST ABOUT WHAT THIS DETECTOR CAN DO. The author reports mAP@50 of 0.793
on the SeaDronesSee validation split, but that headline is carried almost
entirely by boats. Per-class AP:

    boat 0.711 | jetski 0.593 | buoy 0.489 | swimmer 0.282 | appliances 0.188

The `swimmer` class — the only one VIGÍA cares about — is the second weakest.
That is not a defect in the model so much as a statement about the problem: a
person in open water, viewed from 5 to 260 metres up, is a few pixels of head
against moving, glinting, textured water. Quoting 0.793 for a people-in-water
detector would be misleading, so we do not.

It also makes this the most interesting hazard for the temporal validator. A
person in the water persists frame after frame; a wave glint does not. The
weaker the detector, the more the validator has to do — and the more clearly
its contribution can be measured.

RF-DETR is a DETR-style detector: it emits a fixed set of 300 object queries
with no duplicate boxes, so unlike the YOLO detectors elsewhere in VIGÍA it
requires no non-maximum suppression.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from vigia.detectors.base import Backend
from vigia.registry import Viewpoint
from vigia.types import Box, Detection, Frame, Hazard

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/person_in_water/seadronessee_rfdetr_small.onnx")

#: Order fixed by the dataset's data.yaml. The exported head has one extra
#: logit slot ahead of these, which DETR-style models use for "no object".
CLASS_NAMES: tuple[str, ...] = (
    "swimmer", "boat", "jetski", "life_saving_appliances", "buoy",
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PersonInWaterDetector:
    """Detects people in open water from elevated or aerial viewpoints.

    Args:
        hazard_classes: which classes count as a hazard. Only `swimmer` by
            default — a boat is context, not an emergency, and routing boats to
            the validator would alarm on every vessel in the frame.
        include_context: also return non-hazard classes, tagged so they can be
            drawn in the operator view but never reach the validator.
    """

    hazard = Hazard.DROWNING
    #: SeaDronesSee is UAV imagery from 5 to 260 m, elevated and aerial.
    viewpoints: frozenset = frozenset({Viewpoint.OBLIQUE, Viewpoint.NADIR_AERIAL})
    class_names: Sequence[str] = CLASS_NAMES

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        conf_threshold: float = 0.30,
        max_detections: int = 100,
        hazard_classes: Optional[frozenset[str]] = None,
        include_context: bool = False,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
    ) -> None:
        self.backend = Backend(backend)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Export it with "
                f".venv-export/bin/python scripts/export_person_in_water_onnx.py"
            )

        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.hazard_classes = hazard_classes or frozenset({"swimmer"})
        self.include_context = include_context

        if providers is None:
            providers = (
                ["CPUExecutionProvider"] if self.backend is Backend.REFERENCE
                else [p for p in ("CoreMLExecutionProvider",)
                      if p in ort.get_available_providers()] + ["CPUExecutionProvider"]
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=list(providers)
        )
        self.input_name = self.session.get_inputs()[0].name
        # Input resolution is baked into the exported graph.
        shape = self.session.get_inputs()[0].shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else 512
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._latencies_ms: list[float] = []

        logger.info("Loaded %s backend=%s imgsz=%d providers=%s",
                    self.model_path.name, self.backend.value, self.imgsz,
                    self.session.get_providers())

    # ------------------------------------------------------------------ #

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> normalised NCHW float32 RGB, resized to the graph's
        fixed input. DETR-style models are trained on plain resizes rather than
        letterboxing, so the aspect ratio is not preserved — and the box decode
        below undoes exactly that same transform."""
        resized = cv2.resize(image, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(
            np.transpose(normalised, (2, 0, 1))[np.newaxis, ...]
        )

    def postprocess(
        self, boxes: np.ndarray, logits: np.ndarray,
        original_shape: tuple[int, int],
    ) -> list[tuple[Box, float, str]]:
        """Decode DETR queries into boxes in original-image coordinates.

        `boxes` is (1, queries, 4) in normalised cxcywh; `logits` is
        (1, queries, 1 + num_classes) where index 0 is the no-object slot.
        Scores are sigmoid rather than softmax: RF-DETR follows the
        focal-loss convention where classes are independent.
        """
        boxes = boxes[0] if boxes.ndim == 3 else boxes
        logits = logits[0] if logits.ndim == 3 else logits

        scores = 1.0 / (1.0 + np.exp(-logits))

        # Class index maps DIRECTLY onto logit slot: slot 0 is `swimmer`, not a
        # no-object slot. The exported head is one slot wider than the class
        # list and that trailing slot is unused (its maximum score across the
        # validation sample is 0.001).
        #
        # Getting this wrong is silent and expensive. An earlier version assumed
        # a leading no-object slot and took scores[:, 1:], which shifted every
        # label by one: boats were reported as swimmers with 0.84 confidence
        # while actual swimmers were discarded. Image-level metrics still looked
        # plausible (precision 0.83) because boats and swimmers co-occur, and
        # only box-level IoU exposed it — box recall was 0.002. Verified against
        # the author's own rfdetr.predict() on the same image before fixing.
        class_scores = scores[:, : len(self.class_names)]

        best_class = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(class_scores.shape[0]), best_class]

        keep = best_score >= self.conf_threshold
        if not np.any(keep):
            return []

        boxes, best_class, best_score = boxes[keep], best_class[keep], best_score[keep]

        height, width = original_shape
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2.0) * width
        y1 = (cy - h / 2.0) * height
        x2 = (cx + w / 2.0) * width
        y2 = (cy + h / 2.0) * height

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)

        order = np.argsort(-best_score)[: self.max_detections]
        results = []
        for index in order:
            class_id = int(best_class[index])
            name = (self.class_names[class_id]
                    if class_id < len(self.class_names) else f"class_{class_id}")
            results.append((Box(*(float(v) for v in xyxy[index])),
                            float(best_score[index]), name))
        return results

    def detect(self, frame: Frame) -> list[Detection]:
        tensor = self.preprocess(frame.image)

        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latencies_ms.append(elapsed_ms)
        if len(self._latencies_ms) > 200:
            self._latencies_ms.pop(0)

        decoded = self.postprocess(outputs[0], outputs[1], frame.shape)

        detections: list[Detection] = []
        for box, score, name in decoded:
            is_hazard = name in self.hazard_classes
            if not is_hazard and not self.include_context:
                continue
            detection = Detection(
                hazard=self.hazard, box=box, confidence=score, label=name,
                frame_index=frame.index, timestamp=frame.timestamp,
                camera_id=frame.camera_id,
            )
            if not is_hazard:
                detection.extra["context"] = True
            detections.append(detection)
        return detections

    # ------------------------------------------------------------------ #

    @property
    def median_latency_ms(self) -> float:
        return float(np.median(self._latencies_ms)) if self._latencies_ms else 0.0

    @property
    def fps(self) -> float:
        latency = self.median_latency_ms
        return 1000.0 / latency if latency > 0 else 0.0

    def warmup(self, rounds: int = 2) -> None:
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.detect(Frame(image=blank, index=-1, timestamp=0.0))
        self._latencies_ms.clear()
