"""Analyse an uploaded video: work out what it shows, then detect.

The rest of VIGÍA is built for FIXED cameras, where a human declares once what
the camera overlooks and the gate routes accordingly (see vigia/registry.py).
An uploaded clip has no such declaration, so something has to decide which
detector to run.

WHY THIS DOES NOT CONTRADICT TIER 0. The architecture's position is that
routing a LIVE camera by classifying its picture adds a misrouting failure mode
on the critical path in exchange for information an operator already has. That
argument holds for cameras and not for an arbitrary file: nobody has declared
anything, so the choice is between asking the user and inferring it. We infer
it, show the inference prominently, and let the user override — the routing is
visible and correctable rather than silent.

HOW IT INFERS. It does not add a scene classifier. It runs the detectors it
already has over a handful of frames and asks which of them actually finds
anything — the same models, used as their own evidence. That keeps the system
to five models rather than six, and means the routing can never disagree with
the detector that ultimately runs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vigia.registry import Viewpoint
from vigia.types import Detection, Event, Frame, Hazard
from vigia.validator.cascade import TemporalValidator, ValidatorConfig

logger = logging.getLogger(__name__)

#: Plain-language names, for an audience that does not know the codebase.
HAZARD_LABEL = {
    Hazard.FIRE: "wildfire smoke",
    Hazard.FLOOD: "flooding",
    Hazard.DROWNING: "people in open water",
    Hazard.BUILDING_ACCESS: "people in damaged buildings",
    Hazard.TRAFFIC: "road incidents",
}

#: What a scene of this kind looks like, said in a sentence.
SCENE_SENTENCE = {
    Hazard.FIRE: "a wide outdoor view that could show smoke",
    Hazard.FLOOD: "a scene with standing or rising water",
    Hazard.DROWNING: "open water seen from above",
    Hazard.BUILDING_ACCESS: "aerial footage over damaged buildings",
    Hazard.TRAFFIC: "a road seen from a fixed camera",
}


@dataclass
class Candidate:
    """One detector's showing on the sample frames."""

    hazard: Hazard
    hits: int = 0
    frames_with_hits: int = 0
    best_confidence: float = 0.0
    mean_confidence: float = 0.0

    @property
    def score(self) -> float:
        """How strongly this detector claims the footage.

        Frames-with-hits rather than raw hit count: a detector that fires
        fifty times on one frame and never again is describing an artefact,
        not a scene.
        """
        return self.frames_with_hits * self.mean_confidence


@dataclass
class Finding:
    """A confirmed event, in terms the interface can show directly."""

    label: str
    confidence: float
    seconds: float
    #: NORMALISED to 0-1 of the frame, not pixels. The interface scales the
    #: still it displays, so pixel coordinates would have to be re-derived
    #: against whatever size was served — a scaling bug waiting to happen.
    #: Fractions survive any resize.
    box: tuple[float, float, float, float]
    supporting: int


@dataclass
class Analysis:
    video: str
    hazard: Optional[Hazard] = None
    hazard_label: str = ""
    scene_sentence: str = ""
    #: True only when the scene router chose the detector. An explicit
    #: hazard from the caller is an override and says so.
    routed_automatically: bool = False
    #: True when no detector has been chosen yet and the interface should ask.
    needs_choice: bool = False
    considered: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    frames_examined: int = 0
    proposed: int = 0
    filtered: int = 0
    deduplicated: int = 0
    findings: list[Finding] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    note: str = ""

    @property
    def confirmed(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "hazard": self.hazard.value if self.hazard else None,
            "hazard_label": self.hazard_label,
            "scene_sentence": self.scene_sentence,
            "routed_automatically": self.routed_automatically,
            "needs_choice": self.needs_choice,
            "considered": self.considered,
            "duration_seconds": round(self.duration_seconds, 1),
            "frames_examined": self.frames_examined,
            "proposed": self.proposed,
            "filtered": self.filtered,
            "deduplicated": self.deduplicated,
            "confirmed": self.confirmed,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "note": self.note,
            "findings": [
                {
                    "label": f.label,
                    "confidence": round(f.confidence, 3),
                    "seconds": round(f.seconds, 2),
                    "box": [round(v, 5) for v in f.box],
                    "supporting": f.supporting,
                }
                for f in self.findings
            ],
        }


def _detector_for(hazard: Hazard, viewpoint: Viewpoint):
    from vigia.pipeline import build_detector
    return build_detector(hazard, viewpoint)


def probe(video_path: Path, hazards: list[Hazard], viewpoint: Viewpoint,
          sample: int = 6, detectors: Optional[dict] = None) -> list[Candidate]:
    """Run each detector over a few frames and see which one finds anything."""
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    picks = [int(total * (i + 0.5) / sample) for i in range(sample)]

    frames = []
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if ok:
            frames.append((index, image))
    capture.release()

    results = []
    for hazard in hazards:
        detector = (detectors or {}).get(hazard) or _detector_for(hazard, viewpoint)
        candidate = Candidate(hazard=hazard)
        confidences: list[float] = []

        for index, image in frames:
            frame = Frame(image=image, index=index, timestamp=float(index))

            if hazard is Hazard.FLOOD:
                # A SEGMENTER'S EVIDENCE IS NOT A DETECTOR'S. Flood emits one
                # box per connected region of water, so counting boxes rewards
                # it for being WRONG: a misfire scatters many fragments while
                # real water is one body. Measured on the demonstration clips —
                # rubble 0.376 coverage across 4 regions per frame, a real
                # flood 0.310 across 1. Coverage alone cannot separate them.
                #
                # So flood is scored on coverage divided by fragmentation,
                # which is the same property the registry already records for
                # the mask-to-box bridge: real flood masks are a single
                # connected component 99.8% of the time.
                observation = detector.observe(frame)
                regions = [d for d in observation.detections
                           if not d.extra.get("context")]
                if observation.coverage > 0.05 and regions:
                    candidate.frames_with_hits += 1
                    candidate.hits += len(regions)
                    confidences.append(observation.coverage / len(regions))
                continue

            detections = detector.detect(frame)
            hazard_only = [d for d in detections if not d.extra.get("context")]
            if hazard_only:
                candidate.frames_with_hits += 1
                candidate.hits += len(hazard_only)
                confidences += [d.confidence for d in hazard_only]

        if confidences:
            candidate.best_confidence = max(confidences)
            candidate.mean_confidence = float(np.mean(confidences))
        results.append(candidate)

    results.sort(key=lambda c: -c.score)
    return results


def analyse(video_path: str | Path, *, hazard: Optional[Hazard] = None,
            viewpoint: Viewpoint = Viewpoint.OBLIQUE,
            max_frames: int = 240,
            detectors: Optional[dict] = None,
            progress=None) -> Analysis:
    """Route the clip to a detector, then run the full cascade over it.

    `max_frames` caps the work: a long clip is sampled evenly rather than
    processed frame by frame, so a demonstration never turns into a job.
    """
    started = time.perf_counter()
    video_path = Path(video_path)
    result = Analysis(video=video_path.name)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        result.note = "That file could not be opened as a video."
        return result
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if not (0 < fps < 1000):
        fps = 25.0
    result.duration_seconds = total / fps if total else 0.0
    capture.release()

    # --- decide which detector ---------------------------------------- #
    #
    # THE SYSTEM DECIDES; THE PERSON CAN OVERRULE IT.
    #
    # The first version of this asked the detectors themselves: run all of
    # them over a few frames and take whichever claimed the footage most
    # strongly. That routed 2 of 4 test clips wrongly, and not for want of
    # tuning — the scores are not comparable quantities. Each detector has its
    # own confidence calibration and its own propensity to fire, so "flood
    # scored 1.9 and building access scored 0.7" says almost nothing about
    # which is right. On rubble the aerial flood model genuinely reports
    # water, the known consequence of training on 51 flooded images.
    #
    # The second version handed the choice to the user. Always correct, and
    # one click of homework for someone who came to watch a system work.
    #
    # What actually fixed it is a classifier trained for this exact question
    # (vigia/scene_router.py): 4 classes, held out on different sources,
    # accuracy 1.000 on 384 images. It routes all four demonstration clips
    # correctly. Read that figure carefully — four visually distinct scene
    # types are an easy discrimination and the model has memorised its
    # training set — which is why the answer is applied automatically but
    # every alternative stays one click away, and why this is confined to
    # UPLOADED footage. A live camera's hazards are still DECLARED in Tier 0,
    # never inferred, so nothing here weakens that decision.
    if hazard is None:
        from vigia import scene_router as _router

        router = _router.load()
        if router is not None:
            # The trained classifier answers the question it was trained on.
            ranked = router.classify_video(video_path)
            by_name = {
                "fire": Hazard.FIRE, "flood": Hazard.FLOOD,
                "drowning": Hazard.DROWNING,
                "building_access": Hazard.BUILDING_ACCESS,
            }
            result.considered = [
                {"hazard": by_name[name].value,
                 "label": HAZARD_LABEL[by_name[name]],
                 "sentence": SCENE_SENTENCE[by_name[name]],
                 "certainty": round(probability, 3),
                 "evidence": f"{round(probability * 100)}%",
                 "suggested": index == 0}
                for index, (name, probability) in enumerate(ranked)
                if name in by_name
            ]
            # Road incidents are not a class the router knows, so they are
            # offered last and unranked rather than silently dropped.
            result.considered.append(
                {"hazard": Hazard.TRAFFIC.value,
                 "label": HAZARD_LABEL[Hazard.TRAFFIC],
                 "sentence": SCENE_SENTENCE[Hazard.TRAFFIC],
                 "certainty": None, "evidence": "choose by hand",
                 "suggested": False})
            # Apply the reading and carry straight on into the run. Returning
            # here — which this used to do — meant the caller had to ask a
            # second time with the answer it had just been given, and put a
            # question on screen that the system could already answer.
            ranked_known = [name for name, _ in ranked if name in by_name]
            if ranked_known:
                hazard = by_name[ranked_known[0]]
                result.routed_automatically = True
                result.note = ("Read by the scene router. Change it below if "
                               "it has this wrong.")
            else:
                result.needs_choice = True
                result.note = "The scene router could not read this footage."
                result.elapsed_seconds = time.perf_counter() - started
                return result

    if hazard is None:
        # No router on this machine: fall back to asking, with the detectors'
        # own reactions as weak evidence. This is what the interface did
        # before the router existed and it stays correct, just less helpful.
        considered = probe(video_path, list(HAZARD_LABEL), viewpoint,
                           detectors=detectors)
        result.considered = [
            {"hazard": c.hazard.value, "label": HAZARD_LABEL[c.hazard],
             "sentence": SCENE_SENTENCE[c.hazard],
             "frames_with_hits": c.frames_with_hits,
             "frames_sampled": 6,
             "evidence": f"reacted {c.frames_with_hits}/6",
             "certainty": None,
             "score": round(c.score, 3),
             "suggested": False}
            for c in considered
        ]
        # NOTHING IS STARRED AS RECOMMENDED, on purpose. The ranking is
        # evidence — how many sampled frames each detector reacted to — and
        # the reader can weigh it. Promoting the top row to a recommendation
        # would put a wrong answer on screen in front of an audience: on
        # rubble, flood reacts to 6 frames of 6 and is wrong every time.
        result.needs_choice = True
        result.note = ("Pick what this footage shows. The figures beside each "
                       "option are how many sampled frames that detector "
                       "reacted to — useful evidence, not an answer.")
        result.elapsed_seconds = time.perf_counter() - started
        return result

    result.hazard = hazard
    result.hazard_label = HAZARD_LABEL[hazard]
    result.scene_sentence = SCENE_SENTENCE[hazard]

    # --- run the full cascade ------------------------------------------ #
    detector = (detectors or {}).get(hazard) or _detector_for(hazard, viewpoint)
    from vigia.pipeline import default_validator_configs
    validator = TemporalValidator(hazard, default_validator_configs().get(hazard))

    capture = cv2.VideoCapture(str(video_path))
    step = max(1, total // max_frames) if total > max_frames else 1
    index = 0
    examined = 0

    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index % step == 0:
            seconds = index / fps
            detections = detector.detect(
                Frame(image=image, index=index, timestamp=seconds))
            hazard_only = [d for d in detections if not d.extra.get("context")]
            events = validator.process(hazard_only, index, seconds, image=image)
            result.proposed += len(hazard_only)
            height, width = image.shape[:2]
            for event in events:
                x1, y1, x2, y2 = event.box.as_xyxy()
                result.findings.append(Finding(
                    label=HAZARD_LABEL[hazard],
                    confidence=event.confidence,
                    seconds=seconds,
                    box=(x1 / width, y1 / height, x2 / width, y2 / height),
                    supporting=event.supporting_detections,
                ))
            examined += 1
            if progress and examined % 10 == 0:
                progress(examined, max_frames)
        index += 1
    capture.release()

    for name, stat in validator.stats.items():
        if name == "cooldown":
            result.deduplicated += stat.rejected
        else:
            result.filtered += stat.rejected

    result.frames_examined = examined
    result.elapsed_seconds = time.perf_counter() - started
    if not result.findings:
        result.note = (f"Nothing was confirmed. The detector proposed "
                       f"{result.proposed} times, but nothing stayed put long "
                       f"enough to be trusted.")
    return result
