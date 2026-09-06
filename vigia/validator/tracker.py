"""IoU tracker.

Associates detections across consecutive frames so the validator can ask
"has this thing been there for a while?" — which is the single most effective
filter available, because real hazards persist and most false positives do not.

Deliberately simple: greedy IoU association, no Kalman filter, no appearance
model. A smoke plume drifts slowly in a fixed-mount camera, and motion
prediction would add tuning surface and failure modes for no measurable gain
at the frame rates we run. If a hazard ever needs real motion modelling
(a vehicle at speed, say), that belongs in that detector's own logic, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vigia.types import Box, Detection


@dataclass
class Track:
    """A candidate hazard followed across frames."""

    track_id: int
    hazard: str
    box: Box
    confidence: float
    first_frame: int
    first_time: float
    last_frame: int
    last_time: float
    hits: int = 1               # frames on which this track was seen
    misses: int = 0             # consecutive frames missed since the last hit
    confirmed: bool = False     # has it already been promoted to an event?
    confidence_history: list[float] = field(default_factory=list)

    @property
    def age_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def age_seconds(self) -> float:
        return self.last_time - self.first_time

    @property
    def mean_confidence(self) -> float:
        if not self.confidence_history:
            return self.confidence
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def peak_confidence(self) -> float:
        return max(self.confidence_history, default=self.confidence)

    def update(self, detection: Detection) -> None:
        self.box = detection.box
        self.confidence = detection.confidence
        self.confidence_history.append(detection.confidence)
        self.last_frame = detection.frame_index
        self.last_time = detection.timestamp
        self.hits += 1
        self.misses = 0


class IoUTracker:
    """Greedy IoU tracker.

    Args:
        iou_threshold: minimum IoU to consider a detection a continuation of an
            existing track. Low by default (0.2) because smoke plumes grow and
            drift between frames, so demanding tight overlap would fragment a
            single real event into many short tracks — which would then all fail
            the persistence test. Fragmenting real events is the expensive
            failure mode here.
        max_misses: how many consecutive frames a track may go unmatched before
            it is dropped. Tolerating a few misses is what lets a flickering
            detector still accumulate persistence.
    """

    def __init__(self, iou_threshold: float = 0.2, max_misses: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame_index: int) -> list[tuple[Detection, Track]]:
        """Match detections to tracks. Returns (detection, track) pairs.

        Every detection gets a track: either an existing one it matched, or a
        newly created one. Nothing is dropped here — filtering is the cascade's
        job, not the tracker's.
        """
        pairs: list[tuple[Detection, Track]] = []
        unmatched_tracks = list(self.tracks)

        # Highest confidence first, so the strongest detection claims a track.
        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            best_track: Optional[Track] = None
            best_iou = self.iou_threshold

            for track in unmatched_tracks:
                if track.hazard != detection.hazard.value:
                    continue
                iou = detection.box.iou(track.box)
                if iou >= best_iou:
                    best_iou, best_track = iou, track

            if best_track is not None:
                best_track.update(detection)
                unmatched_tracks.remove(best_track)
                pairs.append((detection, best_track))
            else:
                track = Track(
                    track_id=self._next_id,
                    hazard=detection.hazard.value,
                    box=detection.box,
                    confidence=detection.confidence,
                    first_frame=detection.frame_index,
                    first_time=detection.timestamp,
                    last_frame=detection.frame_index,
                    last_time=detection.timestamp,
                    confidence_history=[detection.confidence],
                )
                self._next_id += 1
                self.tracks.append(track)
                pairs.append((detection, track))

        # Age out tracks that went unmatched this frame. refresh() may still
        # rescue them with low-confidence detections before pruning bites.
        for track in unmatched_tracks:
            track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return pairs

    def refresh(self, detections: list[Detection], frame_index: int) -> int:
        """Second-stage association with BELOW-FLOOR detections.

        The core idea of ByteTrack (Zhang et al., ECCV 2022), adapted to an
        alarm cascade rather than a benchmark tracker: a detection too weak to
        alarm on can still be evidence that an existing track's object is
        still there. Matched low-confidence detections therefore KEEP A TRACK
        ALIVE — reset its miss counter and move its box — but deliberately do
        not increment `hits`, cannot start new tracks, and never reach the
        caller as alarm candidates. The confidence floor still decides what
        may alarm; this only decides what may keep being watched.

        Returns the number of tracks refreshed, for the ablation's accounting.
        """
        refreshed = 0
        candidates = [t for t in self.tracks if t.misses > 0]
        for detection in sorted(detections, key=lambda d: d.confidence,
                                reverse=True):
            best_track: Optional[Track] = None
            best_iou = self.iou_threshold
            for track in candidates:
                if track.hazard != detection.hazard.value:
                    continue
                iou = detection.box.iou(track.box)
                if iou >= best_iou:
                    best_iou, best_track = iou, track
            if best_track is not None:
                best_track.box = detection.box
                best_track.last_frame = detection.frame_index
                best_track.last_time = detection.timestamp
                best_track.misses = 0
                candidates.remove(best_track)
                refreshed += 1
        return refreshed

    def reset(self) -> None:
        """Clear all state. Call between videos so one clip cannot leak
        tracks into the next — which would silently corrupt evaluation."""
        self.tracks.clear()
        self._next_id = 1
