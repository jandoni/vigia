"""The temporal event validator — VIGÍA's actual contribution.

Off-the-shelf hazard detectors are sensitive by design and consequently alarm
constantly. Measured on PyroNear's own validation split, their published fire
detector raises an alarm on 87% of frames containing no smoke. That is not a
defect in their model; it is the operating point they chose, on the assumption
that something downstream filters the output. This is that something.

Three levels, applied in order, cheapest first. Every level is independently
switchable so the ablation table — which level buys how much — can be produced
from the same code that runs in production. Confidence is a stateless
per-frame filter; persistence and cooldown are stateful and operate on tracks.

TWO FURTHER LEVELS WERE BUILT, MEASURED, AND REMOVED — the cascade earned its
shape rather than assuming it:

  * Size plausibility rejected implausible box geometry. Measured, it removed
    0% of detections on wildfire and road incidents, and on flood it was
    actively harmful: a plausible maximum-area default silently discarded the
    worst floods.
  * A colour prior demanded that a smoke box be mostly achromatic. Measured
    (eval/run_colour_eval.py, eval/results/colour_prior.json), it failed for
    a physical reason: fog and smoke are both grey, so on the curated hard
    negatives it bought false-alarm rate only by discarding real smoke
    (recall 0.978 -> 0.736 at the strictest cutoff), and inside the temporal
    cascade it improved nothing at any cutoff while a tight one lost an
    ignition sequence outright (5/6). A level that can lose a fire and never
    improves the operating point has no place in a life-safety cascade.

Regression tests keep both from returning.

Design rule: the validator knows nothing about any specific hazard. It is
configured per hazard, never specialised per hazard. Everything hazard-specific
lives in the detector or in a ValidatorConfig, so adding a fifth hazard costs
no changes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vigia.types import Box, Detection, Event, Hazard
from vigia.validator.tracker import IoUTracker, Track


@dataclass
class LevelStats:
    """Pass/reject counts for one cascade level.

    Feeds two things: the ablation table, and the live signal-chain display in
    the operator view. Same numbers, two audiences.
    """

    name: str
    seen: int = 0
    passed: int = 0

    @property
    def rejected(self) -> int:
        return self.seen - self.passed

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.seen if self.seen else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seen": self.seen,
            "passed": self.passed,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejection_rate, 4),
        }


@dataclass
class ValidatorConfig:
    """Per-hazard validator configuration.

    Defaults are tuned for wide-field fire cameras. Other hazards override.
    """

    # Level 1 — confidence
    enable_confidence: bool = True
    min_confidence: float = 0.20

    # NOTE: a size-plausibility level was implemented and then REMOVED.
    # Measured, it rejected 0% of detections on wildfire and 0% on traffic, and
    # on flood it was actively harmful: with a plausible 0.60 max-area default it
    # silently discarded any flood covering more than 60% of the frame — the
    # worse the disaster, the more likely the alert was dropped. A level that
    # earns nothing on two hazards and inverts the safety property on a third
    # does not belong in a life-safety cascade. See PLAN.md and section 5.4 of
    # the technical documentation.

    # NOTE: a COLOUR PRIOR level lived here and was REMOVED, like the size
    # level above, on measurement rather than taste. It checked the fraction
    # of box pixels inside an HSV region; for wildfire smoke the only
    # physically sensible prior is achromaticity, and fog — the curated hard
    # negative — is exactly as grey as smoke. Swept across five cutoffs it
    # traded ~5 points of recall per 10 points of false-alarm rate on stills,
    # and inside the cascade it contributed nothing while a strict cutoff
    # dropped a real ignition sequence. See eval/run_colour_eval.py and
    # eval/results/colour_prior.json; a regression test bars reintroduction.

    # Level 2 — temporal persistence
    enable_persistence: bool = True
    min_frames: int = 3                  # consecutive-ish frames before trusting
    track_iou: float = 0.2
    max_misses: int = 5
    #: ByteTrack-style second-stage association (Zhang et al., ECCV 2022):
    #: detections below the confidence floor may keep an existing track ALIVE
    #: (reset its misses) but never increment hits, start tracks, or alarm.
    #: MEASURED AND NULL on both regimes this project has: identical FAR,
    #: recall, latency and confirmed tracks on the sparse FIgLib ablation
    #: (eval/run_ablation.py, the "+ LC assoc" row) AND on the dense 8 fps
    #: SeaDronesSee sequence — max_misses already bridges the gaps that
    #: second-stage association exists to bridge. Stays default-off as a
    #: documented negative result with its reproduction path, not a feature.
    low_confidence_association: bool = False
    #: Floor for second-stage candidates, to keep pure noise out of stage 2.
    association_floor: float = 0.05

    # Level 3 — per-location cooldown
    enable_cooldown: bool = True
    cooldown_seconds: float = 60.0
    cooldown_iou: float = 0.3            # "same location" means this much overlap


class TemporalValidator:
    """Turns a stream of raw detections into a stream of confirmed events.

    Usage:
        validator = TemporalValidator(Hazard.FIRE, ValidatorConfig())
        events = validator.process(detections, frame_index, timestamp, image)

    `process` returns only confirmed events. Every detection is annotated
    in place with `rejected_by` naming the level that stopped it, so the caller
    can display or log suppressions — which is what makes the validator visible
    in the demo, and auditable in evaluation.
    """

    LEVEL_NAMES = ("confidence", "persistence", "cooldown")

    def __init__(self, hazard: Hazard, config: Optional[ValidatorConfig] = None) -> None:
        self.hazard = hazard
        self.config = config or ValidatorConfig()
        self.tracker = IoUTracker(
            iou_threshold=self.config.track_iou,
            max_misses=self.config.max_misses,
        )
        self.stats: dict[str, LevelStats] = {
            name: LevelStats(name) for name in self.LEVEL_NAMES
        }
        self.confirmed_count = 0
        # (box, time) of recently confirmed events, for the cooldown level.
        self._recent: list[tuple[Box, float]] = []

    # ------------------------------------------------------------------ #
    # Stateless levels
    # ------------------------------------------------------------------ #

    def _check_confidence(self, detection: Detection) -> bool:
        return detection.confidence >= self.config.min_confidence

    # ------------------------------------------------------------------ #
    # Stateful levels
    # ------------------------------------------------------------------ #

    def _check_persistence(self, track: Track) -> bool:
        return track.hits >= self.config.min_frames

    def _check_cooldown(self, detection: Detection, timestamp: float) -> bool:
        """Suppress a repeat alert from a location that recently alerted.

        Without this, one genuine fire produces an alert every frame for as
        long as it burns — technically correct and operationally useless.
        """
        self._recent = [
            (box, when) for box, when in self._recent
            if timestamp - when <= self.config.cooldown_seconds
        ]
        for box, _ in self._recent:
            if detection.box.iou(box) >= self.config.cooldown_iou:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def process(
        self,
        detections: list[Detection],
        frame_index: int,
        timestamp: float,
        image: Optional[np.ndarray] = None,
        camera_id: str = "",
    ) -> list[Event]:
        """Run the full cascade over one frame's detections."""
        survivors: list[Detection] = []
        below_floor: list[Detection] = []

        # --- Level 1: stateless, cheapest first ----------------------- #
        for detection in detections:
            detection.rejected_by = None

            self.stats["confidence"].seen += 1
            if self.config.enable_confidence and not self._check_confidence(detection):
                detection.rejected_by = "confidence"
                if detection.confidence >= self.config.association_floor:
                    below_floor.append(detection)
                continue
            self.stats["confidence"].passed += 1

            survivors.append(detection)

        # --- Level 2: persistence over tracks ------------------------- #
        pairs = self.tracker.update(survivors, frame_index)
        if self.config.low_confidence_association and below_floor:
            # Below-floor detections may keep an existing track alive but can
            # never alarm, add hits, or start tracks (see tracker.refresh).
            self.tracker.refresh(below_floor, frame_index)
        events: list[Event] = []

        for detection, track in pairs:
            self.stats["persistence"].seen += 1
            if self.config.enable_persistence and not self._check_persistence(track):
                detection.rejected_by = "persistence"
                continue
            self.stats["persistence"].passed += 1

            # --- Level 3: cooldown ----------------------------------- #
            self.stats["cooldown"].seen += 1
            if self.config.enable_cooldown and not self._check_cooldown(detection, timestamp):
                detection.rejected_by = "cooldown"
                continue
            self.stats["cooldown"].passed += 1

            detection.track_id = track.track_id
            events.append(
                Event(
                    hazard=self.hazard,
                    box=detection.box,
                    confidence=track.peak_confidence,
                    camera_id=camera_id or detection.camera_id,
                    first_seen_frame=track.first_frame,
                    confirmed_frame=frame_index,
                    first_seen_time=track.first_time,
                    confirmed_time=timestamp,
                    supporting_detections=track.hits,
                    track_id=track.track_id,
                )
            )
            track.confirmed = True
            self._recent.append((detection.box, timestamp))
            self.confirmed_count += 1

        return events

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear all state between videos. Failing to call this between clips
        would let one clip's tracks and cooldowns corrupt the next."""
        self.tracker.reset()
        self._recent.clear()
        self.confirmed_count = 0
        for stat in self.stats.values():
            stat.seen = stat.passed = 0

    def stats_dict(self) -> dict:
        return {
            "hazard": self.hazard.value,
            "confirmed": self.confirmed_count,
            "levels": [self.stats[name].to_dict() for name in self.LEVEL_NAMES],
        }

    @property
    def enabled_levels(self) -> list[str]:
        flags = {
            "confidence": self.config.enable_confidence,
            "persistence": self.config.enable_persistence,
            "cooldown": self.config.enable_cooldown,
        }
        return [name for name, on in flags.items() if on]
