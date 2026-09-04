"""Building-access detector: finding people in and around collapsed buildings.

The DANA trapped people in buildings and garages as water rose; earthquakes do
the same at far greater scale. This detector answers the two questions a rescue
team actually asks of an aerial frame: **is there anyone there**, and **where
can we get in**.

Model  : YOLOv8-DRN, trained by the DRespNeT authors and published with the
         dataset. Instance segmentation over 28 classes, 300 epochs.
Data   : DRespNeT (Cranfield University), UAV imagery of the 2023 Türkiye
         earthquakes
Source : https://doi.org/10.6084/m9.figshare.29991478.v2  (CC BY 4.0)

WHY THEIRS AND NOT OURS. VIGÍA first trained its own torchvision Faster R-CNN
on the same data, because torchvision is BSD-3-Clause and kept the detector
outside the AGPL perimeter. Measured on the held-out test split, that model was
by a wide margin the weakest thing in the system — civilian box recall 0.141,
meaning it missed roughly six of every seven people it should have boxed. The
authors' own weights, evaluated on the identical split with the identical
metric, reach **0.942**. That is not a margin any amount of tuning on 650
training images was going to close, and shipping a six-times-worse detector to
preserve an architectural preference would have been a choice against the people
this system exists to find.

LICENCE, STATED HONESTLY. Figshare publishes the record — dataset and weights —
under CC BY 4.0. The checkpoint itself carries an embedded
`license = AGPL-3.0 (https://ultralytics.com/license)`, stamped automatically
because it was trained with Ultralytics. This is the same tension already
recorded for the PyroNear fire detector, it exists in the authors' own
distribution, and we did not create it. We follow the licence the distributor
states, record the discrepancy in models/REGISTRY.yaml, redistribute no
weights, and convert to ONNX in an isolated build environment so that
`ultralytics` is never a runtime dependency — enforced by tools/check_licence.py.

TWENTY-EIGHT CLASSES REDUCED TO FIVE. DRespNeT labels a broad research agenda:
excavators, dump trucks, road, bridge, debris graded light/moderate/heavy. VIGÍA
needs the subset that answers the two questions above, so the model's 28 classes
are merged down to five at the boundary. The remaining 23 stay available in the
graph for anyone who wants them.

    civilian           <- civilian_visible, group_of_civilians
    rescue_team        <- rescue_team
    entry_accessible   <- entry_door/window/gap_accessible, entry_gap_block_accessible
    entry_blocked      <- entry_door_blocked, entry_window_blocked
    building_collapsed <- building_collapsed

ONLY `civilian` IS A HAZARD, following the precedent set by the people-in-water
detector where a boat is context rather than an emergency. A rescue team on site
is the *expected* state and alarming on it would fire at every incident already
being handled; access points and collapsed structures are context a responder
reads off the frame. Routing all five would bury the one signal that matters —
an unattended civilian — under every doorway in shot.

SEGMENTATION OUTPUT, DETECTION USE. This is a `-seg` model, so its output
carries 32 mask coefficients after the class scores. They are trimmed here: the
validator and the operator view both work in boxes, and the mask prototypes
would otherwise be misread as 32 extra classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from vigia.detectors.base import Backend, ONNXDetector
from vigia.registry import Viewpoint
from vigia.types import Detection, Frame, Hazard

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/building_access/drespnet_yolov8_drn.onnx")

#: The model's own 28 classes, in its training order.
MODEL_CLASSES: tuple[str, ...] = (
    "backhoe_loader", "bridge", "building_collapsed", "building_damaged",
    "building_undamaged", "bus", "car_damaged", "car_undamaged",
    "civilian_visible", "common_dump_truck", "crawler_crane", "crawler_loader",
    "debris_heavy", "debris_light", "debris_moderate", "drone_landing_zone_safe",
    "entry_door_accessible", "entry_door_blocked", "entry_gap_accessible",
    "entry_gap_block_accessible", "entry_window_accessible",
    "entry_window_blocked", "excavator", "group_of_civilians", "rescue_team",
    "road", "rubble", "truck",
)

#: VIGÍA's five, and the model classes folded into each.
CLASS_MERGE: dict[str, tuple[str, ...]] = {
    "civilian": ("civilian_visible", "group_of_civilians"),
    "rescue_team": ("rescue_team",),
    "entry_accessible": ("entry_door_accessible", "entry_window_accessible",
                         "entry_gap_accessible", "entry_gap_block_accessible"),
    "entry_blocked": ("entry_door_blocked", "entry_window_blocked"),
    "building_collapsed": ("building_collapsed",),
}
CLASS_NAMES: tuple[str, ...] = tuple(CLASS_MERGE)

_SOURCE_TO_TARGET = {
    source: target for target, sources in CLASS_MERGE.items() for source in sources
}


class BuildingAccessDetector(ONNXDetector):
    """Detects civilians, rescuers and building access points in aerial frames.

    Args:
        hazard_classes: which merged classes reach the validator. Only
            `civilian` by default — see the module docstring.
        include_context: also return the other merged classes, tagged
            `extra["context"] = True` so the operator view can draw them while
            the validator ignores them.
        keep_unmerged: emit the model's own class name for anything outside the
            five, instead of discarding it. Off by default.
    """

    hazard = Hazard.BUILDING_ACCESS
    #: DRespNeT is UAV imagery of earthquake-damaged blocks.
    viewpoints: frozenset = frozenset({Viewpoint.OBLIQUE, Viewpoint.NADIR_AERIAL})
    #: The base class maps a predicted id to a label through this, so it must
    #: stay the model's own 28 in training order.
    class_names: Sequence[str] = MODEL_CLASSES
    #: What this detector actually EMITS after merging, and what every
    #: consumer — validator, evaluation harness, operator view, redaction —
    #: sees. Kept separate from `class_names` because the two genuinely differ
    #: here, and conflating them silently mislabels every detection.
    emitted_class_names: Sequence[str] = CLASS_NAMES

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        imgsz: int = 640,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        hazard_classes: Optional[frozenset[str]] = None,
        include_context: bool = False,
        keep_unmerged: bool = False,
        providers: Optional[Sequence[str]] = None,
        backend: Backend | str = Backend.REFERENCE,
    ) -> None:
        super().__init__(
            model_path,
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            providers=providers,
            backend=backend,
        )
        self.hazard_classes = hazard_classes or frozenset({"civilian"})
        self.include_context = include_context
        self.keep_unmerged = keep_unmerged

    # ------------------------------------------------------------------ #

    def postprocess(self, raw, scale, pad, original_shape):
        """Trim the 32 mask coefficients, then decode as an ordinary detector.

        A `-seg` export emits 4 box values, then the class scores, then 32 mask
        coefficients. The base class infers the class count from the channel
        dimension, so without this trim it would read 60 classes instead of 28
        and take the argmax over mask coefficients.
        """
        preds = raw[0] if raw.ndim == 3 else raw
        keep = 4 + len(MODEL_CLASSES)
        if preds.shape[0] < preds.shape[1]:      # (channels, anchors)
            preds = preds[:keep]
        else:                                    # (anchors, channels)
            preds = preds[:, :keep].T
        return super().postprocess(preds[np.newaxis, ...], scale, pad,
                                   original_shape)

    def detect(self, frame: Frame) -> list[Detection]:
        detections = super().detect(frame)

        kept: list[Detection] = []
        for detection in detections:
            source = detection.label
            target = _SOURCE_TO_TARGET.get(source)

            if target is None:
                if not self.keep_unmerged:
                    continue
                detection.extra["context"] = True
                detection.extra["model_class"] = source
                kept.append(detection)
                continue

            detection.extra["model_class"] = source
            detection.label = target

            is_hazard = target in self.hazard_classes
            if not is_hazard and not self.include_context:
                continue
            if not is_hazard:
                detection.extra["context"] = True
            kept.append(detection)
        return kept
