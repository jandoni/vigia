"""Frame sources: video file, RTSP stream, webcam.

One interface over all three, because the pipeline must not care which it is
holding — a demo replaying a curated clip and a deployment reading a municipal
RTSP feed have to exercise the same code path, or the demo proves nothing.

TWO POLICIES LIVE HERE, and both are consequences of decisions made elsewhere.

LAST-AVAILABLE-FRAME, not every frame. For a live stream, falling behind is
normal and queueing is the wrong response: a five-second-old frame is not worth
processing when a current one exists. Live sources therefore drop stale frames
and hand over the newest, so latency stays bounded and the system degrades by
skipping rather than by lagging further and further behind. File sources do the
opposite and yield every frame, because an evaluation replay must be
deterministic and reproducible — dropping frames would make a measurement
depend on how busy the machine was.

THE ROLLING BUFFER IS BOUNDED AND NEVER WRITTEN. PLAN.md section 10 commits to
no raw video retention beyond a short rolling buffer. That buffer is a
fixed-length in-memory deque here, sized from the camera's RetentionPolicy, and
nothing in this module opens a file for writing. The commitment is structural:
there is no code path to disable it, because there is no code path that persists
video at all.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from vigia.types import Frame

logger = logging.getLogger(__name__)


@dataclass
class SourceStats:
    frames_yielded: int = 0
    frames_dropped: int = 0
    started_at: float = 0.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at if self.started_at else 0.0

    @property
    def fps(self) -> float:
        return self.frames_yielded / self.elapsed if self.elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "frames_yielded": self.frames_yielded,
            "frames_dropped": self.frames_dropped,
            "fps": round(self.fps, 2),
        }


class FrameSource:
    """Frames from a file, an RTSP URL, or a webcam index.

    `source` is interpreted the way a human would write it: a bare integer is a
    webcam, anything containing "://" is a network stream, everything else is a
    path.
    """

    def __init__(
        self,
        source: str,
        camera_id: str = "",
        *,
        buffer_seconds: float = 10.0,
        max_fps: Optional[float] = None,
        loop: bool = False,
    ) -> None:
        self.source = source
        self.camera_id = camera_id or str(source)
        self.max_fps = max_fps
        self.loop = loop
        self.stats = SourceStats()

        self.is_live = self._looks_live(source)
        self._handle: Optional[cv2.VideoCapture] = None

        # Rolling buffer: bounded, in memory, never written to disk. Sized from
        # the declared retention window at an assumed 25 fps when the source
        # cannot report its own rate.
        self._buffer_seconds = max(0.0, buffer_seconds)
        self._buffer: deque[Frame] = deque(maxlen=1)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _looks_live(source: str) -> bool:
        text = str(source)
        if text.isdigit():
            return True
        return "://" in text and not text.startswith("file://")

    def open(self) -> "FrameSource":
        target: object = int(self.source) if str(self.source).isdigit() else self.source
        handle = cv2.VideoCapture(target)
        if not handle.isOpened():
            raise RuntimeError(
                f"could not open source {self.source!r} for camera "
                f"{self.camera_id!r}"
            )
        self._handle = handle

        rate = handle.get(cv2.CAP_PROP_FPS) or 0.0
        self.native_fps = rate if rate and rate == rate and rate < 1000 else 25.0
        frames = max(1, int(round(self._buffer_seconds * self.native_fps)))
        self._buffer = deque(maxlen=frames)

        logger.info("opened %s (%s, %.1f fps, buffer %d frames)",
                    self.source, "live" if self.is_live else "file",
                    self.native_fps, frames)
        self.stats.started_at = time.perf_counter()
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.release()
            self._handle = None
        # Drop the buffer explicitly rather than waiting for garbage
        # collection: it is the only place decoded video lives.
        self._buffer.clear()

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ #

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted.

        Live sources drop to the newest available frame; file sources yield
        every frame so that replays are deterministic.
        """
        if self._handle is None:
            self.open()

        index = 0
        interval = 1.0 / self.max_fps if self.max_fps else 0.0
        next_due = time.perf_counter()

        while True:
            ok, image = self._handle.read()
            if not ok:
                if self.loop and not self.is_live:
                    self._handle.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, image = self._handle.read()
                    if not ok:
                        break
                else:
                    break

            if self.is_live:
                # Catch up to the newest frame. grab() decodes nothing, so
                # discarding backlog is cheap; only the survivor is retrieved.
                dropped = 0
                while self._handle.grab():
                    got, newer = self._handle.retrieve()
                    if not got:
                        break
                    image = newer
                    dropped += 1
                    if dropped > 300:      # pathological backlog; stop draining
                        break
                self.stats.frames_dropped += dropped

            timestamp = (
                time.time() if self.is_live
                else index / max(1e-6, self.native_fps)
            )
            frame = Frame(image=image, index=index, timestamp=timestamp,
                          camera_id=self.camera_id)

            self._buffer.append(frame)
            self.stats.frames_yielded += 1
            index += 1

            if interval:
                now = time.perf_counter()
                if now < next_due:
                    time.sleep(next_due - now)
                next_due = max(now, next_due) + interval

            yield frame

    # ------------------------------------------------------------------ #

    def recent(self) -> list[Frame]:
        """The rolling buffer, oldest first. In memory only."""
        return list(self._buffer)

    @property
    def buffered_frames(self) -> int:
        return len(self._buffer)


class HttpStillSource:
    """A public camera that publishes stills over HTTP rather than a stream.

    Deliberately the same shape as FrameSource — open/close/frames/recent, the
    context-manager protocol, the same SourceStats — so CameraPipeline holds
    one or the other without knowing which. A live municipal feed and a USGS
    gauge camera have to exercise the same code path, for the same reason a
    replayed clip does.

    ONLY GENUINELY NEW IMAGES ARE YIELDED, and this is a correctness
    requirement rather than an efficiency one. The validator confirms an event
    once a candidate PERSISTS across frames. Handing it the same JPEG three
    times would satisfy persistence=3 without the world having been observed
    three times — a confirmation manufactured out of one observation. So each
    fetch is compared against the last and dropped when unchanged, and the
    drops are counted like any other so the interface can show the camera is
    alive but quiet.

    Falling behind is impossible here (we poll; nothing queues), so the
    stale-frame policy FrameSource applies to RTSP does not arise.
    """

    def __init__(
        self,
        source: str,
        camera_id: str = "",
        *,
        interval_seconds: float = 60.0,
        buffer_frames: int = 12,
        timeout: float = 30.0,
        on_idle: Optional[Callable[[], None]] = None,
    ) -> None:
        self.source = source
        self.camera_id = camera_id or str(source)
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.timeout = timeout
        #: Called about twice a second while waiting for the next image.
        #: Without it the host does no work between polls, and on a camera
        #: that publishes every five minutes the first detection would not
        #: reach the screen for five minutes — the pipeline drains finished
        #: detector work when a frame arrives, and no frame arrives.
        self.on_idle = on_idle
        self.stats = SourceStats()
        self.is_live = True
        self.native_fps = 1.0 / self.interval_seconds

        self.last_url: Optional[str] = None
        self.last_capture_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0

        self._closed = False
        self._signature: Optional[bytes] = None
        self._buffer: deque[Frame] = deque(maxlen=max(1, buffer_frames))

    # ------------------------------------------------------------------ #

    def open(self) -> "HttpStillSource":
        self._closed = False
        self.stats.started_at = time.perf_counter()
        logger.info("polling %s every %.0fs for camera %s",
                    self.source, self.interval_seconds, self.camera_id)
        return self

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()

    def __enter__(self) -> "HttpStillSource":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _signature_of(image: np.ndarray) -> bytes:
        """Cheap content fingerprint, used only to spot a repeated image.

        A 16x16 grey thumbnail is enough to tell 'the camera has not updated'
        from 'the scene changed slightly', costs microseconds, and — unlike
        hashing the raw bytes — is not defeated by a re-encode of an identical
        scene.
        """
        small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).tobytes()

    def fetch_once(self) -> Optional[Frame]:
        """One poll. Returns a Frame only when the image is genuinely new."""
        from vigia.io import live

        url = live.resolve_image_url(self.source)
        if url is None:
            self.last_error = "camera published no recent image"
            self.consecutive_failures += 1
            return None

        image = live.fetch_image(url, timeout=self.timeout)
        if image is None:
            self.last_error = "image could not be fetched"
            self.consecutive_failures += 1
            return None

        self.last_error = None
        self.consecutive_failures = 0
        self.last_url = url
        captured = live.capture_time(url)
        self.last_capture_time = captured.timestamp() if captured else time.time()

        signature = self._signature_of(image)
        if signature == self._signature:
            self.stats.frames_dropped += 1      # alive, but nothing new
            return None
        self._signature = signature

        frame = Frame(image=image, index=self.stats.frames_yielded,
                      timestamp=self.last_capture_time,
                      camera_id=self.camera_id)
        self._buffer.append(frame)
        self.stats.frames_yielded += 1
        return frame

    def frames(self) -> Iterator[Frame]:
        """Poll until closed, yielding only images the camera has not shown."""
        if self.stats.started_at == 0.0:
            self.open()

        next_due = 0.0
        while not self._closed:
            now = time.perf_counter()
            if now < next_due:
                # Sleep in slices so close() is honoured promptly rather than
                # after a full polling interval, and hand control back to the
                # host on every slice so finished work reaches the screen.
                if self.on_idle is not None:
                    try:
                        self.on_idle()
                    except Exception:       # a display fault must not stop a camera
                        logger.exception("idle callback failed for %s",
                                         self.camera_id)
                time.sleep(min(0.5, next_due - now))
                continue
            next_due = now + self.interval_seconds

            frame = self.fetch_once()
            if frame is not None:
                yield frame

    # ------------------------------------------------------------------ #

    def recent(self) -> list[Frame]:
        return list(self._buffer)

    @property
    def buffered_frames(self) -> int:
        return len(self._buffer)

    def status(self) -> dict:
        """What the interface needs to show the camera as live and honest."""
        return {
            "source": self.source,
            "interval_seconds": self.interval_seconds,
            "images_seen": self.stats.frames_yielded,
            "repeats_skipped": self.stats.frames_dropped,
            "last_capture_time": self.last_capture_time,
            "last_error": self.last_error,
        }


def open_source(source: str, camera_id: str = "", **kwargs):
    """FrameSource or HttpStillSource, whichever the source calls for.

    The pipeline asks for a source and gets one; deciding here rather than at
    every call site is what keeps 'a clip, an RTSP feed and a public gauge
    camera all run the same way' true rather than aspirational.
    """
    from vigia.io import live

    if live.looks_live_source(source):
        catalogue_entry = live.BY_SOURCE.get(str(source))
        interval = kwargs.pop("interval_seconds", None)
        if interval is None:
            interval = catalogue_entry.interval_seconds if catalogue_entry else 60.0
        for unused in ("max_fps", "loop", "buffer_seconds"):
            kwargs.pop(unused, None)
        return HttpStillSource(source, camera_id,
                               interval_seconds=interval, **kwargs)
    return FrameSource(source, camera_id, **kwargs)
