"""The pipeline: capture -> gate -> detect -> validate -> emit.

This is the module that makes VIGÍA a system rather than five detectors and a
validator that have only ever been run by evaluation harnesses. Everything it
does has been decided elsewhere in this codebase; its job is to hold those
decisions together and to be honest about the cost of each stage.

CONCURRENCY. One thread per active detector, plus one fusion thread, over
bounded queues — the pattern reported in arXiv 2607.03131, which reached ~10 FPS
across 6 concurrent streams with per-frame latency under 100 ms. Detectors are
independent ONNX sessions, so they genuinely run in parallel: onnxruntime
releases the GIL during inference, which is where essentially all of the time
goes.

THE QUEUES ARE BOUNDED, AND THAT IS THE POINT. An unbounded queue converts a
detector that cannot keep up into unbounded memory growth and ever-increasing
latency, which in a life-safety system means alerting about something that
stopped being true minutes ago. Bounded queues convert the same condition into
dropped frames — visible, counted, and reported in `stats()`. A system that
skips frames under load and says so is safer than one that silently falls
behind.

WHAT THE PIPELINE DOES NOT DO. It does not decide which detectors to run; the
registry does (Tier 0). It does not decide what is an alert; the validator does
(Tier 2). It does not decide what may leave the camera; `vigia.io.alerts` does.
Keeping those three out of here is what allows each to be tested and published
on its own, and it is why adding a sixth hazard touches none of this file.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from vigia.io.alerts import AlertSink
from vigia.io.sources import FrameSource, open_source
from vigia.registry import Camera, CameraRegistry, Viewpoint
from vigia.types import Detection, Event, Frame, Hazard
from vigia.validator.cascade import TemporalValidator, ValidatorConfig

logger = logging.getLogger(__name__)


#: Per-hazard validator tuning. The validator itself is hazard-agnostic — this
#: table is the only place hazards differ, which is precisely the property
#: `test_validator_contains_no_hazard_specific_code` enforces in cascade.py.
#:
#: Values follow the operating points recorded in models/REGISTRY.yaml.
def default_validator_configs() -> dict[Hazard, ValidatorConfig]:
    return {
        # Persistence 3 is the R1 result: 75% fewer false alarms at zero cost
        # in fires detected, for two extra minutes of latency.
        Hazard.FIRE: ValidatorConfig(min_confidence=0.20, min_frames=3),
        # Flood is a level trend, not an event that appears and vanishes; a
        # flood visible for three frames is not a false positive, so
        # persistence is short and the real signal is rate of rise.
        Hazard.FLOOD: ValidatorConfig(min_confidence=0.50, min_frames=2,
                                      cooldown_seconds=300.0),
        Hazard.TRAFFIC: ValidatorConfig(min_confidence=0.25, min_frames=3),
        # The weakest detector in the system per its own measurement, so the
        # validator carries more of the load here than anywhere else.
        Hazard.DROWNING: ValidatorConfig(min_confidence=0.30, min_frames=3),
        # Building access runs at a deliberately sensitive 0.20 (see the
        # detector docstring); persistence does the filtering instead.
        Hazard.BUILDING_ACCESS: ValidatorConfig(min_confidence=0.20, min_frames=3),
    }


def _detector_viewpoints(hazard: Hazard) -> frozenset:
    """The viewpoints a hazard's detector declares, without loading its model.

    Imported lazily and by class attribute so this costs nothing: the gate must
    be able to reject a bad pairing before any ONNX session is created.
    """
    if hazard is Hazard.FIRE:
        from vigia.detectors.fire import FireDetector
        return getattr(FireDetector, "viewpoints", frozenset())
    if hazard is Hazard.FLOOD:
        from vigia.detectors.flood import FloodDetector
        # Ask what it can serve now, not what the class declares in general:
        # the aerial capability exists only when its weights are on disk.
        return FloodDetector.supported_viewpoints()
    if hazard is Hazard.TRAFFIC:
        from vigia.detectors.traffic import TrafficDetector
        return getattr(TrafficDetector, "viewpoints", frozenset())
    if hazard is Hazard.DROWNING:
        from vigia.detectors.person_in_water import PersonInWaterDetector
        return getattr(PersonInWaterDetector, "viewpoints", frozenset())
    if hazard is Hazard.BUILDING_ACCESS:
        from vigia.detectors.building_access import BuildingAccessDetector
        return getattr(BuildingAccessDetector, "viewpoints", frozenset())
    return frozenset()


def build_detector(hazard: Hazard, viewpoint=None):
    """Construct the detector for one hazard.

    Imported lazily and per hazard so that a forest-only deployment loads one
    ONNX session rather than five — the gate saving is real only if the models
    are never loaded, not merely never called.
    """
    if hazard is Hazard.FIRE:
        from vigia.detectors.fire import FireDetector
        return FireDetector()
    if hazard is Hazard.FLOOD:
        from vigia.detectors.flood import FloodDetector
        if viewpoint is not None:
            return FloodDetector(FloodDetector.model_for(viewpoint))
        return FloodDetector()
    if hazard is Hazard.TRAFFIC:
        from vigia.detectors.traffic import TrafficDetector
        return TrafficDetector()
    if hazard is Hazard.DROWNING:
        from vigia.detectors.person_in_water import PersonInWaterDetector
        return PersonInWaterDetector()
    if hazard is Hazard.BUILDING_ACCESS:
        from vigia.detectors.building_access import BuildingAccessDetector
        return BuildingAccessDetector()
    raise ValueError(f"no detector registered for hazard {hazard}")


@dataclass
class StageResult:
    """One frame's worth of output from one detector, after validation.

    Carries the raw detections as well as the events, because the suppressed
    ones are what make the validator visible — the operator view's whole
    argument is the gap between proposed and confirmed.
    """

    hazard: Hazard
    frame_index: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    latency_ms: float = 0.0
    #: The frame this result came from. Carried explicitly because detection is
    #: asynchronous: by the time fusion sees this, capture has moved on, so
    #: there is no "current frame" that corresponds to it. An earlier version
    #: matched on frame index against whatever capture happened to be holding
    #: and therefore attached an evidence image to almost nothing — every alert
    #: was written with "evidence": null. The frame is already resident in the
    #: source's rolling buffer, so holding this reference costs nothing extra.
    frame: Optional[Frame] = None

    @property
    def proposed(self) -> int:
        return len(self.detections)

    @property
    def confirmed(self) -> int:
        return len(self.events)

    @property
    def suppressed(self) -> list[Detection]:
        return [d for d in self.detections if d.rejected_by is not None]


@dataclass
class PipelineStats:
    frames_in: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    detections_proposed: int = 0
    events_confirmed: int = 0
    #: Rejected as NOT CREDIBLE — confidence and persistence.
    filtered: int = 0
    #: Rejected as a DUPLICATE of an event already reported — cooldown.
    deduplicated: int = 0
    started_at: float = 0.0
    per_hazard: dict[str, dict] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at if self.started_at else 0.0

    @property
    def fps(self) -> float:
        return self.frames_processed / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def suppression_rate(self) -> float:
        """Fraction of proposed detections that did not become an alert.

        DO NOT QUOTE THIS AS A FALSE-ALARM REDUCTION. It combines two different
        things, and the project has already been caught by that conflation
        once: during R1, recall appeared to collapse from 0.925 to 0.125 until
        it turned out cooldown was correctly suppressing repeat alerts about a
        fire already reported, not rejecting fires. Cooldown was taken out of
        the evaluation harness for exactly this reason.

        Use `filtering_rate` for the false-alarm claim and
        `deduplication_rate` for the "one event, one alert" claim.
        """
        if not self.detections_proposed:
            return 0.0
        return 1.0 - (self.events_confirmed / self.detections_proposed)

    @property
    def filtering_rate(self) -> float:
        """Fraction rejected as not credible. THIS is the false-alarm claim."""
        if not self.detections_proposed:
            return 0.0
        return self.filtered / self.detections_proposed

    @property
    def deduplication_rate(self) -> float:
        """Fraction rejected as repeats of an already-reported event.

        Not a false alarm removed: an alert not sent twice.
        """
        if not self.detections_proposed:
            return 0.0
        return self.deduplicated / self.detections_proposed

    def to_dict(self) -> dict:
        return {
            "frames_in": self.frames_in,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "detections_proposed": self.detections_proposed,
            "events_confirmed": self.events_confirmed,
            "suppression_rate": round(self.suppression_rate, 4),
            "filtered_not_credible": self.filtered,
            "filtering_rate": round(self.filtering_rate, 4),
            "deduplicated_repeat_events": self.deduplicated,
            "deduplication_rate": round(self.deduplication_rate, 4),
            "fps": round(self.fps, 2),
            "elapsed_s": round(self.elapsed, 1),
            "per_hazard": self.per_hazard,
        }


class CameraPipeline:
    """Runs one camera: gate, parallel detectors, shared validator, egress.

    Usage:
        pipeline = CameraPipeline(camera, sink=AlertSink())
        pipeline.run(max_frames=500)
    """

    def __init__(
        self,
        camera: Camera,
        *,
        sink: Optional[AlertSink] = None,
        validator_configs: Optional[dict[Hazard, ValidatorConfig]] = None,
        queue_size: int = 2,
        detectors: Optional[dict[Hazard, object]] = None,
        on_result: Optional[Callable[[StageResult, Frame], None]] = None,
    ) -> None:
        self.camera = camera
        self.sink = sink
        self.on_result = on_result
        self.queue_size = queue_size
        self.stats = PipelineStats()
        #: The frame source, once run() has opened one. Exposed so the
        #: operator view can ask a live public camera how fresh its
        #: newest image is without reaching through the pipeline.
        self.source = None

        self.hazards = sorted(camera.active_hazards, key=lambda h: h.value)
        if not self.hazards:
            raise ValueError(
                f"camera {camera.camera_id!r} has no active hazards; the gate "
                f"would run nothing. Check its context or hazard override."
            )

        configs = validator_configs or default_validator_configs()

        # VIEWPOINT ENFORCEMENT. A detector's training viewpoint is part of its
        # operating envelope, and running outside it does not fail loudly — it
        # returns a confident wrong answer. The flood segmenter on straight-down
        # UAV imagery reports up to 0.397 water coverage while tinting mown
        # grass as water. Refusing here makes that impossible to do by accident,
        # in the same spirit as the licence and privacy guards.
        self._check_viewpoints()

        # THE GATE, APPLIED. Only these detectors are constructed at all.
        self.detectors = detectors if detectors is not None else {
            hazard: build_detector(hazard, camera.active_viewpoint)
            for hazard in self.hazards
        }
        self.validators = {
            hazard: TemporalValidator(hazard, configs.get(hazard))
            for hazard in self.hazards
        }

        # One inbound queue per detector, each bounded. maxsize=2 keeps at most
        # one frame waiting behind the one being processed.
        self._queues: dict[Hazard, queue.Queue] = {
            hazard: queue.Queue(maxsize=queue_size) for hazard in self.hazards
        }
        self._results: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        logger.info("camera %s: context=%s hazards=%s",
                    camera.camera_id, camera.context.value,
                    [h.value for h in self.hazards])

    # ------------------------------------------------------------------ #

    def _check_viewpoints(self) -> None:
        viewpoint = self.camera.active_viewpoint
        offenders = []
        for hazard in self.hazards:
            supported = getattr(build_detector, "_viewpoints_cache", {}).get(hazard)
            if supported is None:
                supported = _detector_viewpoints(hazard)
            if supported and viewpoint not in supported:
                offenders.append(
                    f"{hazard.value} (supports "
                    f"{', '.join(sorted(v.value for v in supported))})")
        if offenders:
            raise ValueError(
                f"camera {self.camera.camera_id!r} is declared "
                f"{viewpoint.value!r}, which is outside the operating envelope "
                f"of: {'; '.join(offenders)}.\n"
                f"Running a detector outside its trained viewpoint does not "
                f"fail loudly — it returns a confident wrong answer. Either "
                f"correct the camera's viewpoint, remove that hazard from it, "
                f"or register a model trained for this viewpoint."
            )

    def warmup(self) -> None:
        for detector in self.detectors.values():
            if hasattr(detector, "warmup"):
                detector.warmup(rounds=1)

    def _worker(self, hazard: Hazard) -> None:
        """One detector thread. Blocks on its own queue; exits on sentinel."""
        detector = self.detectors[hazard]
        validator = self.validators[hazard]

        while not self._stop.is_set():
            item = self._queues[hazard].get()
            if item is None:                    # sentinel
                self._queues[hazard].task_done()
                break

            frame: Frame = item
            started = time.perf_counter()
            try:
                detections = detector.detect(frame)
                events = validator.process(
                    detections, frame.index, frame.timestamp,
                    image=frame.image, camera_id=frame.camera_id,
                )
            except Exception:
                # One detector failing must not take the camera down: a fire
                # detector crash should not also stop flood monitoring.
                logger.exception("detector %s failed on frame %d",
                                 hazard.value, frame.index)
                detections, events = [], []

            self._results.put(StageResult(
                hazard=hazard,
                frame_index=frame.index,
                timestamp=frame.timestamp,
                detections=detections,
                events=events,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                frame=frame,
            ))
            self._queues[hazard].task_done()

    def _start_workers(self) -> None:
        self._stop.clear()
        self._threads = []
        for hazard in self.hazards:
            thread = threading.Thread(
                target=self._worker, args=(hazard,),
                name=f"detect-{hazard.value}", daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _stop_workers(self) -> None:
        for hazard in self.hazards:
            try:
                self._queues[hazard].put_nowait(None)
            except queue.Full:
                # Drain one slot so the sentinel can always be delivered.
                try:
                    self._queues[hazard].get_nowait()
                    self._queues[hazard].put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._stop.set()

    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        max_fps: Optional[float] = None,
        loop: bool = False,
    ) -> PipelineStats:
        """Process the camera's source until it ends or `max_frames` is hit."""
        # open_source, not FrameSource directly: a public gauge camera that
        # publishes stills over HTTP has to run the same path as a clip and an
        # RTSP feed, or the live demonstration proves nothing about the
        # deployment. The factory returns whichever kind the source calls for.
        source = open_source(
            self.camera.source, self.camera.camera_id,
            buffer_seconds=self.camera.retention.rolling_buffer_seconds,
            max_fps=max_fps, loop=loop,
        )
        self.source = source
        # A polled still camera spends nearly all its time waiting. Give it a
        # way to let fusion run while it does, or a five-minute camera shows
        # its first detection five minutes late.
        if hasattr(source, "on_idle"):
            source.on_idle = lambda: self._drain_results(None)

        self.stats = PipelineStats(started_at=time.perf_counter())
        self._start_workers()

        try:
            with source:
                for frame in source.frames():
                    self.stats.frames_in += 1

                    # BACKPRESSURE POLICY, and it differs by source type for
                    # the same reason FrameSource's does.
                    #
                    # Live: never block. A detector that cannot keep up must
                    # cost us frames, not latency — alerting about something
                    # that stopped being true is worse than missing a frame.
                    #
                    # File: always block. An evaluation replay or a curated
                    # demo clip must be deterministic, and dropping frames
                    # would make a published measurement depend on how busy the
                    # machine happened to be. Found by measurement: with
                    # put_nowait on both, a 16-frame clip through a 193 ms
                    # detector processed 3 frames and silently dropped 13.
                    dispatched = 0
                    for hazard in self.hazards:
                        if source.is_live:
                            try:
                                self._queues[hazard].put_nowait(frame)
                                dispatched += 1
                            except queue.Full:
                                self.stats.frames_dropped += 1
                        else:
                            self._queues[hazard].put(frame)
                            dispatched += 1

                    if dispatched:
                        self.stats.frames_processed += 1

                    self._drain_results(frame)

                    if max_frames is not None and self.stats.frames_in >= max_frames:
                        break
        finally:
            if not source.is_live:
                # Let every queued frame finish, so a replay's totals reflect
                # the whole clip rather than whatever happened to be done.
                for hazard in self.hazards:
                    self._queues[hazard].join()
                self._drain_results(None)
            self._stop_workers()
            self._drain_results(None)
            self._finalise_stats()

        return self.stats

    def _drain_results(self, frame: Optional[Frame] = None,
                       block: bool = False) -> None:
        """Fusion: collect finished work, emit alerts, notify observers.

        Takes its image from the result's own frame, never from whatever
        capture is currently holding — see StageResult.frame.
        """
        while True:
            try:
                result: StageResult = self._results.get_nowait()
            except queue.Empty:
                return

            self.stats.detections_proposed += result.proposed
            self.stats.events_confirmed += result.confirmed

            source_frame = result.frame
            if self.sink is not None:
                for event in result.events:
                    image = source_frame.image if source_frame is not None else None
                    self.sink.emit(event, image, result.detections)

            if self.on_result is not None and source_frame is not None:
                self.on_result(result, source_frame)

    def _finalise_stats(self) -> None:
        # Split the rejections by what they MEAN, not merely by which level
        # fired. Confidence and persistence reject a candidate as not
        # credible; cooldown rejects a credible one as already reported.
        for validator in self.validators.values():
            for name, stat in validator.stats.items():
                if name == "cooldown":
                    self.stats.deduplicated += stat.rejected
                else:
                    self.stats.filtered += stat.rejected

        for hazard in self.hazards:
            detector = self.detectors[hazard]
            self.stats.per_hazard[hazard.value] = {
                "validator": self.validators[hazard].stats_dict(),
                "median_latency_ms": round(
                    getattr(detector, "median_latency_ms", 0.0), 1
                ),
            }

    def reset(self) -> None:
        """Clear validator state between clips."""
        for validator in self.validators.values():
            validator.reset()


class MultiCameraPipeline:
    """Runs every enabled camera in a registry, one thread each.

    Detectors are built once per hazard and SHARED across cameras. onnxruntime
    sessions are thread-safe for concurrent `run` calls, and a 72 MB model held
    once instead of once per camera is the difference between a laptop demo and
    a swap storm. The validators are emphatically NOT shared: validator state is
    per camera per hazard, and sharing it would let one camera's tracks confirm
    another camera's events.
    """

    def __init__(
        self,
        registry: CameraRegistry,
        *,
        sink: Optional[AlertSink] = None,
        validator_configs: Optional[dict[Hazard, ValidatorConfig]] = None,
    ) -> None:
        self.registry = registry
        self.sink = sink

        needed = registry.required_hazards()
        logger.info("gate: loading %d of %d detectors (%s)",
                    len(needed), len(Hazard),
                    ", ".join(sorted(h.value for h in needed)))
        # Keyed by (hazard, viewpoint): two cameras watching the same hazard
        # from different viewpoints need different weights, and sharing on
        # hazard alone would silently hand a nadir camera the ground model.
        self._shared: dict = {}
        for camera in registry.enabled_cameras:
            for hazard in camera.active_hazards:
                key = (hazard, camera.active_viewpoint)
                if key not in self._shared:
                    self._shared[key] = build_detector(hazard,
                                                       camera.active_viewpoint)

        self.pipelines = {
            camera.camera_id: CameraPipeline(
                camera, sink=sink, validator_configs=validator_configs,
                detectors={h: self._shared[(h, camera.active_viewpoint)]
                           for h in camera.active_hazards},
            )
            for camera in registry.enabled_cameras
        }

    def run(self, *, max_frames: Optional[int] = None,
            max_fps: Optional[float] = None) -> dict[str, PipelineStats]:
        threads = []
        results: dict[str, PipelineStats] = {}

        def run_one(camera_id: str) -> None:
            results[camera_id] = self.pipelines[camera_id].run(
                max_frames=max_frames, max_fps=max_fps
            )

        for camera_id in self.pipelines:
            thread = threading.Thread(target=run_one, args=(camera_id,),
                                      name=f"camera-{camera_id}", daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return results
