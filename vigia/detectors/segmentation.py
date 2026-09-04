"""ONNX semantic segmentation base class.

Flood is the first VIGÍA hazard whose natural output is a mask rather than
boxes. That creates a design question: the temporal validator operates on
bounding boxes, and specialising it for masks would break the hazard-agnostic
property that is the project's central claim.

The resolution is to convert here rather than there. A segmenter produces a
mask; this class extracts connected components from that mask and emits them as
ordinary `Detection` objects. The validator then treats a flooded region
exactly as it treats a smoke plume — persistence, cooldown and all — with no
knowledge that it came from a mask.

The mask itself is not discarded. It is attached to each detection and returned
alongside, because the water level and rate-of-rise signal in `vigia.flood`
needs the mask, not the box.

Like `ONNXDetector`, this never imports `ultralytics`. Inference is ONNX Runtime
only; training and export are offline steps under `tools/` and `scripts/`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from vigia.detectors.base import Backend
from vigia.types import Box, Detection, Frame, Hazard

logger = logging.getLogger(__name__)

# ImageNet statistics — the normalisation torchvision segmentation backbones
# were pretrained with. Must match whatever scripts/train_flood.py used.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class SegmentationResult:
    """One frame's segmentation output."""

    mask: np.ndarray                 # bool, (H, W), at ORIGINAL frame resolution
    detections: list[Detection]      # connected components as boxes
    coverage: float                  # fraction of the frame covered by the mask
    latency_ms: float


class ONNXSegmenter:
    """Binary semantic segmentation over ONNX Runtime.

    Subclasses set `hazard` and `class_name`. The default implementation expects
    a model emitting logits of shape (1, C, H, W); with C == 1 it is treated as
    a single logit and passed through a sigmoid, otherwise argmax over channels
    picks the class and `positive_class` selects which index counts as the
    hazard.
    """

    hazard: Hazard = Hazard.FLOOD
    class_name: str = "water"
    positive_class: int = 1

    def __init__(
        self,
        model_path: str | Path,
        *,
        imgsz: int = 512,
        threshold: float = 0.5,
        min_region_fraction: float = 0.002,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
        static_model_path: Optional[str | Path] = None,
    ) -> None:
        """
        Args:
            min_region_fraction: connected components smaller than this fraction
                of the frame are dropped rather than emitted as detections.
                Water segmentation is speckly at boundaries; without this the
                validator would receive dozens of one-pixel "floods" per frame.
                Speckle is still counted in the mask used for level estimation —
                it is only excluded from the box representation.
        """
        self.backend = Backend(backend)

        resolved = Path(model_path)
        if self.backend is Backend.FAST:
            candidate = (
                Path(static_model_path) if static_model_path
                else resolved.with_name(f"{resolved.stem}_static{imgsz}.onnx")
            )
            if candidate.exists():
                resolved = candidate
            else:
                logger.warning(
                    "Backend FAST requested but %s is missing; using REFERENCE.",
                    candidate.name,
                )
                self.backend = Backend.REFERENCE

        self.model_path = resolved
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                f"Train it with scripts/train_flood.py or fetch it with "
                f"tools/fetch_models.py."
            )

        self.imgsz = imgsz
        self.threshold = threshold
        self.min_region_fraction = min_region_fraction

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
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._latencies_ms: list[float] = []

        logger.info("Loaded segmenter %s backend=%s providers=%s",
                    self.model_path.name, self.backend.value,
                    self.session.get_providers())

    # ------------------------------------------------------------------ #

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> normalised NCHW float32 RGB at model resolution.

        Segmentation is resized rather than letterboxed: padding would inject
        a large constant-colour region that the network has never seen, and the
        mask is resized straight back to the original shape afterwards.
        """
        resized = cv2.resize(image, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(
            np.transpose(normalised, (2, 0, 1))[np.newaxis, ...]
        )

    def postprocess(self, logits: np.ndarray,
                    original_shape: tuple[int, int]) -> np.ndarray:
        """Model logits -> boolean mask at the original frame resolution."""
        array = logits[0] if logits.ndim == 4 else logits

        if array.shape[0] == 1:
            probability = 1.0 / (1.0 + np.exp(-array[0]))
            low_res = probability >= self.threshold
        else:
            low_res = array.argmax(axis=0) == self.positive_class

        height, width = original_shape
        return cv2.resize(low_res.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def mask_to_detections(self, mask: np.ndarray, frame: Frame) -> list[Detection]:
        """Connected components of the mask, as boxes the validator understands.

        Confidence is the component's fill ratio — how much of its bounding box
        it actually occupies. A genuine flooded area fills its box; a thin
        diagonal reflection does not. This gives the validator's confidence
        level something meaningful to threshold on for a hazard that has no
        detector confidence of its own.
        """
        height, width = mask.shape[:2]
        frame_area = float(height * width)
        min_area = self.min_region_fraction * frame_area

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        detections: list[Detection] = []
        for index in range(1, count):           # 0 is background
            x, y, w, h, area = stats[index]
            if area < min_area:
                continue
            fill_ratio = area / float(w * h) if w * h else 0.0
            detections.append(
                Detection(
                    hazard=self.hazard,
                    box=Box(float(x), float(y), float(x + w), float(y + h)),
                    confidence=float(min(1.0, fill_ratio)),
                    label=self.class_name,
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    camera_id=frame.camera_id,
                    extra={"area_fraction": area / frame_area},
                )
            )

        detections.sort(key=lambda d: d.box.area, reverse=True)
        return detections

    def segment(self, frame: Frame) -> SegmentationResult:
        """Run segmentation on one frame."""
        tensor = self.preprocess(frame.image)

        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latencies_ms.append(elapsed_ms)
        if len(self._latencies_ms) > 200:
            self._latencies_ms.pop(0)

        mask = self.postprocess(outputs[0], frame.shape)
        detections = self.mask_to_detections(mask, frame)

        return SegmentationResult(
            mask=mask,
            detections=detections,
            coverage=float(mask.mean()) if mask.size else 0.0,
            latency_ms=elapsed_ms,
        )

    def detect(self, frame: Frame) -> list[Detection]:
        """Detector-compatible interface, so a segmenter can be dropped into
        the same pipeline slot as a box detector."""
        return self.segment(frame).detections

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
            self.segment(Frame(image=blank, index=-1, timestamp=0.0))
        self._latencies_ms.clear()
