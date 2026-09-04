"""ONNX Runtime detector base class.

Deliberately does NOT import `ultralytics`. Every model VIGÍA runs at inference
time is an ONNX graph executed through ONNX Runtime (MIT). Any `.pt -> .onnx`
conversion is a one-time offline build step and lives in `tools/export_onnx.py`,
never in the runtime path. See PLAN.md §4 for why this matters.

The postprocessing here targets the standard Ultralytics YOLO detection export
layout: output tensor of shape (batch, 4 + num_classes, num_anchors) with boxes
in (cx, cy, w, h) letterboxed-input pixel coordinates, and no built-in NMS.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from vigia.types import Box, Detection, Frame, Hazard

logger = logging.getLogger(__name__)

# Ultralytics letterbox pad colour.
_PAD_VALUE = 114


class Backend(str, Enum):
    """Which execution path to use.

    These are not interchangeable and the choice is deliberate.

    REFERENCE — CPU execution provider on the as-published dynamic-shape graph.
        Deterministic and bit-reproducible. **Every number we publish must come
        from this path.**

    FAST — CoreML execution provider on a static-shape graph. Measured at 1.7x
        to 1.9x faster on an M4 Pro (about 34 ms vs 60 ms at 1024px), which is
        the difference between a 30 FPS and a 17 FPS demo.

        It is not bit-identical. Across the 100-frame pyro-sdis validation set
        the detection count matched on 100/100 frames and boxes agreed to
        within 0.41 px, but confidences drift by up to ~0.012 because CoreML
        computes in lower precision on the Neural Engine. At our 0.20 operating
        point that drift changed no metric at all; at 0.10 and 0.30 it moved
        box precision/recall by about 1% by flipping a single detection across
        the threshold.

        So: FAST for anything a human watches, REFERENCE for anything we report.
    """

    REFERENCE = "reference"
    FAST = "fast"


def letterbox(
    image: np.ndarray, target: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio and pad to `target` (height, width).

    Returns the padded image, the scale factor applied, and the (left, top)
    padding in pixels — everything needed to map boxes back to original coords.
    """
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = target

    scale = min(dst_h / src_h, dst_w / src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))

    # INTER_AREA downsamples without aliasing; INTER_LINEAR is right for upscaling.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_w, pad_h = dst_w - new_w, dst_h - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
    )
    return padded, scale, (float(left), float(top))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression.

    `boxes` is (N, 4) in xyxy. Returns kept indices, highest score first.
    Written in numpy rather than pulled from cv2.dnn so the behaviour is
    explicit and testable — this is a component we report numbers about.
    """
    if boxes.size == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

        order = rest[iou <= iou_threshold]

    return keep


class ONNXDetector:
    """Base class for every VIGÍA hazard detector.

    Subclasses set `hazard` and may override `class_names`, thresholds, or
    postprocessing. The default implementation handles single- and multi-class
    Ultralytics-style YOLO detection exports.
    """

    hazard: Hazard = Hazard.FIRE
    class_names: Sequence[str] = ("item",)

    def __init__(
        self,
        model_path: str | Path,
        *,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
        static_model_path: Optional[str | Path] = None,
    ) -> None:
        self.backend = Backend(backend)

        # FAST needs the static-shape graph: CoreML cannot build an execution
        # plan for dynamic spatial dimensions. If that graph has not been
        # generated yet, say so plainly and fall back rather than dying.
        resolved = Path(model_path)
        if self.backend is Backend.FAST:
            candidate = (
                Path(static_model_path)
                if static_model_path
                else resolved.with_name(resolved.stem + "_static1024.onnx")
            )
            if candidate.exists():
                resolved = candidate
            else:
                logger.warning(
                    "Backend FAST requested but %s is missing. "
                    "Run `python tools/make_static.py` to build it. "
                    "Falling back to REFERENCE.",
                    candidate.name,
                )
                self.backend = Backend.REFERENCE

        self.model_path = resolved
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                f"Run `python tools/fetch_models.py` to download it."
            )

        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        if providers is None:
            providers = self._providers_for(self.backend)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.log_severity_level = 3  # warnings and above only

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=list(providers),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Rolling inference timings, so the demo can display honest latency.
        self._latencies_ms: list[float] = []

        self._session_options = session_options
        self._verify_providers()

        logger.info(
            "Loaded %s (%s) backend=%s providers=%s imgsz=%d",
            self.model_path.name,
            self.hazard.value,
            self.backend.value,
            self.session.get_providers(),
            self.imgsz,
        )

    def _verify_providers(self) -> None:
        """Run one throwaway inference and fall back to CPU if it fails.

        CoreML cannot build an execution plan for graphs with dynamic spatial
        dimensions, which is how Ultralytics exports by default. Rather than
        discover that mid-demo, we find out at construction time and degrade
        quietly to CPU. A static-shape re-export would let CoreML back in;
        that is a future optimisation, not a correctness issue.
        """
        active = self.session.get_providers()
        if active == ["CPUExecutionProvider"]:
            return

        probe = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        try:
            self.session.run(self.output_names, {self.input_name: probe})
        except Exception as exc:  # noqa: BLE001 - any EP failure means fall back
            logger.warning(
                "%s failed on %s (%s). Falling back to CPU.",
                active[0], self.model_path.name, type(exc).__name__,
            )
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=self._session_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def _providers_for(backend: Backend) -> list[str]:
        """Execution providers for a backend.

        REFERENCE pins CPU so results are reproducible on any machine —
        published numbers must not depend on which laptop produced them.
        FAST tries CoreML (Neural Engine / GPU on Apple Silicon) first.
        """
        if backend is Backend.REFERENCE:
            return ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        preferred = [p for p in ("CoreMLExecutionProvider",) if p in available]
        return preferred + ["CPUExecutionProvider"]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        """BGR HWC uint8 -> NCHW float32 RGB in [0, 1], letterboxed."""
        padded, scale, pad = letterbox(image, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(tensor), scale, pad

    def postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad: tuple[float, float],
        original_shape: tuple[int, int],
    ) -> list[tuple[Box, float, int]]:
        """Decode raw model output into boxes in original-image coordinates.

        Expects (batch, 4 + num_classes, anchors) — the Ultralytics detect
        export layout. Also accepts the already-transposed variant.
        """
        preds = raw[0] if raw.ndim == 3 else raw

        # Ultralytics emits (4 + nc, anchors); we want (anchors, 4 + nc).
        # Anchors vastly outnumber channels, so the longer axis is anchors.
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        num_classes = preds.shape[1] - 4
        if num_classes < 1:
            raise ValueError(
                f"Unexpected output layout {raw.shape}; got {num_classes} classes"
            )

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep_mask = confidences >= self.conf_threshold
        if not np.any(keep_mask):
            return []

        boxes_cxcywh = boxes_cxcywh[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask]

        # cxcywh -> xyxy, still in letterboxed input space
        cx, cy, w, h = (
            boxes_cxcywh[:, 0], boxes_cxcywh[:, 1],
            boxes_cxcywh[:, 2], boxes_cxcywh[:, 3],
        )
        xyxy = np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1
        )

        # Undo letterbox: remove padding, then undo the scale.
        pad_left, pad_top = pad
        xyxy[:, [0, 2]] -= pad_left
        xyxy[:, [1, 3]] -= pad_top
        xyxy /= max(scale, 1e-9)

        # Clip to the original frame.
        src_h, src_w = original_shape
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, src_w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, src_h)

        kept = nms(xyxy, confidences, self.iou_threshold)[: self.max_detections]

        return [
            (
                Box(*(float(v) for v in xyxy[i])),
                float(confidences[i]),
                int(class_ids[i]),
            )
            for i in kept
        ]

    def detect(self, frame: Frame) -> list[Detection]:
        """Run the detector on one frame. Returns raw candidates — the
        validator decides which of these become events."""
        tensor, scale, pad = self.preprocess(frame.image)

        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latencies_ms.append(elapsed_ms)
        if len(self._latencies_ms) > 200:
            self._latencies_ms.pop(0)

        decoded = self.postprocess(outputs[0], scale, pad, frame.shape)

        return [
            Detection(
                hazard=self.hazard,
                box=box,
                confidence=confidence,
                label=self._label_for(class_id),
                frame_index=frame.index,
                timestamp=frame.timestamp,
                camera_id=frame.camera_id,
            )
            for box, confidence, class_id in decoded
        ]

    def _label_for(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #

    @property
    def median_latency_ms(self) -> float:
        """Median inference latency. Median, not mean, because the first few
        calls include graph warm-up and would skew an average."""
        if not self._latencies_ms:
            return 0.0
        return float(np.median(self._latencies_ms))

    @property
    def fps(self) -> float:
        latency = self.median_latency_ms
        return 1000.0 / latency if latency > 0 else 0.0

    def warmup(self, rounds: int = 2) -> None:
        """Run the graph on a blank frame so the first real frame is not slow.

        Matters for the demo: without this the first detection reports a
        latency several times the steady-state figure.
        """
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.detect(Frame(image=blank, index=-1, timestamp=0.0))
        self._latencies_ms.clear()
