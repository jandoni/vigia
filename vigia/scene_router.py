"""Decide what KIND of footage a clip is, so it reaches the right detector.

STRICTLY FOR THE UPLOAD PATH. Live cameras route by declaration — an operator
states once what a camera overlooks and Tier 0 obeys (vigia/registry.py). That
decision stands and this module must never be used for it: a declared camera
needs no classifier, and adding a guess to the critical path would create a
misrouting failure where none exists.

An uploaded clip is a different problem. Nothing has been declared, so either
the person is asked or something infers it. The first attempt inferred it by
running all five detectors and taking whichever claimed the footage most
loudly; it was wrong on half the test clips, because the detectors' scores are
not comparable quantities — on rubble the aerial flood model genuinely reports
water, and says so with confidence.

So this is a small classifier trained for exactly that question, and the
interface still shows its answer with a one-click override.

Road incidents are absent by design: four usable training pictures exist for
that scene, which is not a class. Dashcam footage is chosen by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/scene_router/scene_router.onnx")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SceneRouter:
    """Classifies a still, and votes across several stills for a clip."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No scene router at {self.model_path}. Train and export it with "
                f"scripts/train_scene_router.py and "
                f"scripts/export_scene_router_onnx.py"
            )

        meta_path = self.model_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.classes: list[str] = meta.get("classes", [])
        self.imgsz: int = int(meta.get("imgsz", 224))
        self.heldout_accuracy = meta.get("heldout_accuracy")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------ #

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(
            np.transpose(normalised, (2, 0, 1))[np.newaxis, ...])

    def classify_frame(self, image: np.ndarray) -> dict[str, float]:
        logits = self.session.run(None, {self.input_name: self._prepare(image)})[0][0]
        shifted = logits - logits.max()
        weights = np.exp(shifted)
        probabilities = weights / weights.sum()
        return dict(zip(self.classes, (float(p) for p in probabilities)))

    def classify_video(self, video_path: str | Path,
                       samples: int = 6) -> list[tuple[str, float]]:
        """Mean probability per class across evenly spaced frames.

        Averaged rather than voted: a clip that changes scene halfway through
        should show both, and an average says so where a majority vote hides it.
        """
        capture = cv2.VideoCapture(str(video_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        picks = [int(total * (i + 0.5) / samples) for i in range(samples)]

        gathered: list[dict[str, float]] = []
        for index in picks:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok:
                gathered.append(self.classify_frame(image))
        capture.release()

        if not gathered:
            return []
        mean = {name: float(np.mean([g[name] for g in gathered]))
                for name in self.classes}
        return sorted(mean.items(), key=lambda kv: -kv[1])


def load(model_path: str | Path = DEFAULT_MODEL_PATH) -> Optional[SceneRouter]:
    """The router, or None when it has not been trained on this machine.

    Absence is normal and must not be fatal: the interface falls back to
    asking, which is what it did before this model existed.
    """
    try:
        return SceneRouter(model_path)
    except Exception as exc:
        logger.info("scene router unavailable (%s); the interface will ask", exc)
        return None
