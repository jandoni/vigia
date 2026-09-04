"""The operator view — FastAPI, MJPEG for video, WebSocket for events.

TWO CHANNELS, SEPARATE CONCERNS, and the split is deliberate. The reference
implementation (arXiv 2607.03131) identifies Base64-over-WebSocket frame
transport as its bottleneck and lists WebRTC as future work. Sending frames as
MJPEG over plain HTTP and keeping the event channel to small JSON messages
sidesteps that problem entirely rather than optimising it: the browser's own
image decoder handles the video, and the WebSocket carries only numbers.

NO BUILD STEP. Vanilla JS and modern CSS, served as static files. A judge or a
municipality should be able to clone the repository and run it, and a project
that needs a toolchain to demonstrate is a project nobody will try. This is a
credibility argument as much as a technical one.

WHAT THIS SERVER DOES NOT DO. It does not detect, gate, validate, or decide what
may leave the camera — it subscribes to a pipeline that does all four, and
renders. That separation is why the operator view can be absent entirely in a
deployment: a node running headless is the same code minus this package, which
is also why FastAPI is an optional dependency rather than a core one.
"""

# NOTE: deliberately NO `from __future__ import annotations` in this module.
#
# That import turns every annotation into a string, and FastAPI resolves those
# strings against the MODULE's globals. FastAPI is imported inside create_app()
# so that it stays an optional dependency — an edge node runs headless and must
# not need a web framework — which means `WebSocket` is a function-local name
# and invisible to that resolution. The result was a WebSocket route whose
# socket parameter FastAPI could not type, so it treated it as a required query
# parameter and rejected every handshake with 403.
#
# The failure was invisible from the browser: the interface falls back to
# polling on a dead socket, so the view kept updating and only the server log
# and a direct client showed the 403. Without the future import, annotations
# evaluate at definition time, inside create_app, where WebSocket is in scope.

import asyncio
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import tempfile

import numpy as np

from vigia.registry import CameraRegistry
from vigia.types import Hazard
from vigia.web.overlay import draw, encode_jpeg, placeholder

# `Path` and `Hazard` are used by the analysis endpoints below.

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ViewState:
    """The single mutable thing the server shows, written by pipeline threads.

    Kept behind a lock and deliberately small: one annotated frame plus counts.
    The pipeline must never block on the UI, so writers replace rather than
    queue, and a slow browser costs nothing but staleness.
    """

    def __init__(self, timeline_length: int = 240) -> None:
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._frame_id = 0

        self.camera_id = ""
        self.context = ""
        self.hazards: list[str] = []
        self.started = time.perf_counter()

        # Live-camera identity. A public camera is somebody else's
        # infrastructure, so the interface has to be able to say whose, under
        # what licence, and how long ago it last published — otherwise a
        # five-minute gap between images reads as a crashed demo.
        self.live = False
        self.name = ""
        self.source = ""
        self.attribution = ""
        self.licence = ""
        self.place = ""
        self.interval_seconds = 0
        self.home_url = ""
        #: Set by the launcher to a zero-argument callable returning the
        #: source's own status dict. A callable rather than a snapshot because
        #: the source is created when the pipeline starts, not before.
        self.source_status = None

        self.proposed = 0
        self.confirmed = 0
        self.frames = 0
        self.latency_ms = 0.0
        self.levels: dict[str, dict] = {}
        self.by_level: dict[str, int] = {}
        self.show_suppressed = True

        # Bottom timeline: one entry per processed frame. Bounded, because it
        # is a display buffer and must not grow for a demo left running.
        self.timeline: deque[dict] = deque(maxlen=timeline_length)
        self.recent_events: deque[dict] = deque(maxlen=25)

    # ------------------------------------------------------------------ #

    def publish(self, result, frame, validator_stats: dict) -> None:
        """Called from a pipeline detector thread, once per finished frame."""
        annotated = draw(frame.image, result.detections, result.events,
                         show_suppressed=self.show_suppressed)
        jpeg = encode_jpeg(annotated)

        suppressed = [d for d in result.detections if d.rejected_by is not None]

        with self._lock:
            self._jpeg = jpeg
            self._frame_id += 1
            self.frames += 1
            self.proposed += len(result.detections)
            self.confirmed += len(result.events)
            self.latency_ms = result.latency_ms
            self.levels = {
                level["name"]: level
                for level in validator_stats.get("levels", [])
            }
            for detection in suppressed:
                key = detection.rejected_by
                self.by_level[key] = self.by_level.get(key, 0) + 1
            self.timeline.append({
                "frame": result.frame_index,
                "proposed": len(result.detections),
                "confirmed": len(result.events),
                "suppressed": len(suppressed),
            })
            for event in result.events:
                self.recent_events.append({
                    "hazard": event.hazard.value,
                    "camera_id": event.camera_id,
                    "confidence": round(event.confidence, 3),
                    "frame": event.confirmed_frame,
                    "seconds_to_confirm": round(event.seconds_to_confirm, 2),
                    "supporting": event.supporting_detections,
                })

    def jpeg(self) -> bytes:
        with self._lock:
            if self._jpeg is not None:
                return self._jpeg
        return encode_jpeg(placeholder())

    @property
    def suppression_rate(self) -> float:
        """Kept for continuity; see filtering_rate for the false-alarm claim."""
        return 1.0 - (self.confirmed / self.proposed) if self.proposed else 0.0

    @property
    def filtered(self) -> int:
        """Rejected as not credible — confidence and persistence."""
        return sum(v for k, v in self.by_level.items() if k != "cooldown")

    @property
    def deduplicated(self) -> int:
        """Rejected as a repeat of an event already reported — cooldown."""
        return self.by_level.get("cooldown", 0)

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = time.perf_counter() - self.started
            return {
                "camera_id": self.camera_id,
                "context": self.context,
                "hazards": self.hazards,
                "all_hazards": [h.value for h in Hazard],
                "frames": self.frames,
                "proposed": self.proposed,
                "confirmed": self.confirmed,
                "suppressed": self.proposed - self.confirmed,
                "suppression_rate": round(self.suppression_rate, 4),
                "filtered": self.filtered,
                "deduplicated": self.deduplicated,
                "by_level": dict(self.by_level),
                "fps": round(self.frames / elapsed, 1) if elapsed > 0 else 0.0,
                "latency_ms": round(self.latency_ms, 1),
                "levels": list(self.levels.values()),
                "timeline": list(self.timeline),
                "events": list(self.recent_events)[::-1],
                "show_suppressed": self.show_suppressed,
                "frame_id": self._frame_id,
                "live": self.live,
                "name": self.name,
                "attribution": self.attribution,
                "licence": self.licence,
                "place": self.place,
                "interval_seconds": self.interval_seconds,
                "home_url": self.home_url,
                "source_status": self._source_status(),
            }

    def _source_status(self) -> Optional[dict]:
        """Whatever the frame source knows about itself, or None.

        Wrapped because it reaches into a live pipeline: a source that has not
        started yet, or one that has just been replaced, must degrade to 'no
        status' rather than take the whole view down.
        """
        if self.source_status is None:
            return None
        try:
            return self.source_status()
        except Exception:
            return None


def _viewpoint(name):
    from vigia.registry import Viewpoint
    try:
        return Viewpoint(name)
    except (ValueError, TypeError):
        return Viewpoint.OBLIQUE


def _resolve_clip(reference: str):
    """A sample clip name, or an uploaded file. Never an arbitrary path."""
    import tempfile as _tf

    if not reference:
        return None
    if reference.startswith("upload:"):
        candidate = Path(_tf.gettempdir()) / "vigia_uploads" / Path(
            reference.split(":", 1)[1]).name
    else:
        candidate = Path("clips") / Path(reference).name
    return candidate if candidate.exists() else None


def create_app(states, registry: Optional[CameraRegistry] = None):
    """Serve one or many cameras.

    `states` is either a single ViewState or a {camera_id: ViewState} mapping.
    Every endpoint takes an optional `camera` parameter and falls back to the
    first, so a single-camera run needs no query string and a multi-camera one
    switches with the header selector — which is what makes Tier 0 legible: the
    detector chips visibly change as you move between cameras.
    """
    if isinstance(states, ViewState):
        states = {states.camera_id or "camera": states}
    order = list(states)

    def pick(camera: Optional[str]) -> ViewState:
        return states.get(camera or "", states[order[0]])

    return _build_app(states, order, pick, registry)


def _build_app(states, order, pick, registry):
    from fastapi import (FastAPI, File, Request, UploadFile, WebSocket,
                         WebSocketDisconnect)
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="VIGÍA operator view",
        description=(
            "Gated multi-hazard detection with shared temporal validation. "
            "The dual view — proposed versus confirmed — is the point."
        ),
        version="0.1.0",
    )

    @app.get("/", include_in_schema=False)
    async def index():
        """The demonstration interface, for an audience that has not seen this."""
        return FileResponse(STATIC_DIR / "demo.html")

    # Every screen of the demonstration gets its own address. The interface is
    # one document that swaps sections, but a URL you cannot link to is a URL
    # you cannot put in a slide, an email or a bug report — so each section is
    # reachable directly and the client keeps the address bar in step as you
    # move. A path that is not a screen still 404s — masking that would hide
    # a genuine broken link behind a page that looks fine.
    DEMO_TABS = {
        "overview": "home",
        "analyse": "analyse",
        "live": "live",
        "how-it-works": "how",
        "teach": "train",
    }

    for _path in DEMO_TABS:
        @app.get(f"/{_path}", include_in_schema=False)
        async def demo_tab():
            return FileResponse(STATIC_DIR / "demo.html")

    @app.get("/operator", include_in_schema=False)
    async def operator():
        """The engineer's view — dense, unchanged, still the working tool."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/training")
    def api_training() -> dict:
        """Completed training runs, read from the histories on disk."""
        runs = []

        router = Path("models/scene_router/scene_router.history.json")
        if router.exists():
            data = json.loads(router.read_text(encoding="utf-8"))
            history = data.get("history", [])
            best = max((h["accuracy"] for h in history), default=0.0)
            runs.append({
                "title": "Scene router",
                "summary": (
                    "Learns what KIND of footage a clip is, so an uploaded video "
                    "can be sent to the right specialist. Held-out material comes "
                    "from different datasets and cameras than the training "
                    "material, so the score is not inflated by near-duplicate "
                    "frames."),
                "scores": [
                    {"label": "on held-out footage", "value": f"{best * 100:.0f}%",
                     "honest": True,
                     "note": "Different cameras and datasets from the training set"},
                    {"label": "classes it knows", "value": str(len(data.get("classes", []))),
                     "honest": False,
                     "note": "Road incidents excluded — only four usable pictures exist"},
                ],
                "curve": [{"epoch": h["epoch"], "value": h["accuracy"],
                           "best": h["accuracy"] >= best - 1e-9} for h in history],
                "curve_note": ("Each bar is one stage. Training runs in short "
                               "blocks so a laptop can finish it, and a stage "
                               "that is interrupted resumes where it stopped."),
            })

        aerial = Path("models/flood/floodnet_aerial_deeplabv3.history.json")
        if aerial.exists():
            data = json.loads(aerial.read_text(encoding="utf-8"))
            history = data.get("history", [])
            best = max((h.get("flooded_water_iou", 0) for h in history), default=0.0)
            overall = max((h.get("water_iou", 0) for h in history), default=0.0)
            runs.append({
                "title": "Flooding, seen from a drone",
                "summary": (
                    "Learned from 398 marked-up pictures, of which only 51 "
                    "actually show flooding. Two scores are reported because one "
                    "would flatter it: most of the pictures are dry, so a "
                    "detector that never spots anything still scores well "
                    "overall."),
                "scores": [
                    {"label": "on every picture", "value": f"{overall * 100:.0f}%",
                     "honest": False, "note": "Flattering — carried by the dry ones"},
                    {"label": "on the flooded pictures only",
                     "value": f"{best * 100:.0f}%", "honest": True,
                     "note": "The honest one. This is what we publish."},
                ],
                "curve": [{"epoch": h["epoch"],
                           "value": h.get("flooded_water_iou", 0),
                           "best": h.get("flooded_water_iou", 0) >= best - 1e-9}
                          for h in history],
                "curve_note": ("It stopped improving early: the score on flooded "
                               "pictures has not moved since stage 8 while the "
                               "overall one kept creeping up. More time will not "
                               "help — more flooded pictures would."),
            })

        return {"runs": runs}

    # ---- video analysis: the upload flow ---------------------------- #

    ANALYSIS_DIR = Path(tempfile.gettempdir()) / "vigia_uploads"

    @app.get("/api/clips")
    def api_clips() -> dict:
        """Sample footage, so a demonstration needs no file at hand.

        Each entry carries its own credit, read from clips/MANIFEST.json
        rather than restated here. The manifest is the same file that decides
        whether a clip may appear in anything published, so the interface
        cannot show a clip under a licence the build has not already cleared.
        """
        clips_dir = Path("clips")
        names = {
            "fire_ridge.mp4": ("Wildfire on a ridge",
                               "A mountain lookout camera, minutes after ignition"),
            "creek_gauge.mp4": ("A creek in flood",
                                "A gauge camera through a real flood and its recession"),
            "open_water.mp4": ("People in open water",
                               "A drone over the sea, searching"),
            "collapsed_building.mp4": ("A damaged city block",
                                       "A drone over earthquake damage"),
        }

        manifest = {}
        manifest_path = clips_dir / "MANIFEST.json"
        if manifest_path.exists():
            try:
                for entry in json.loads(
                        manifest_path.read_text(encoding="utf-8"))["clips"]:
                    manifest[entry["clip"]] = entry
            except Exception:
                manifest = {}

        clips = []
        for name, (title, blurb) in names.items():
            if not (clips_dir / name).exists():
                continue
            entry = manifest.get(name, {})
            clips.append({
                "file": name,
                "title": title,
                "blurb": blurb,
                "hazard": entry.get("hazard", ""),
                "frames": entry.get("frames"),
                "licence": entry.get("licence", ""),
                "attribution": entry.get("attribution", ""),
                "thumb": f"/api/thumb?file={name}",
            })
        return {"clips": clips}

    @app.get("/api/thumb")
    def api_thumb(file: str):
        """A small poster frame, so the sample list looks like footage.

        Taken a little way in rather than at frame zero: the first frame of a
        gauge camera clip is often the least representative one in it.
        """
        import cv2
        from fastapi.responses import Response

        path = _resolve_clip(file)
        if path is None:
            return Response(status_code=404)
        capture = cv2.VideoCapture(str(path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 4:
            capture.set(cv2.CAP_PROP_POS_FRAMES, total // 4)
        ok, image = capture.read()
        capture.release()
        if not ok:
            return Response(status_code=404)
        width = 320
        scale = width / image.shape[1]
        image = cv2.resize(image, (width, max(1, int(image.shape[0] * scale))))
        ok, buffer = cv2.imencode(".jpg", image,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return Response(content=buffer.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/api/analyse/suggest")
    async def api_suggest(request: Request) -> dict:
        """Sample a few frames and report what each detector reacted to."""
        from vigia.analysis import analyse

        body = await request.json()
        path = _resolve_clip(body.get("file", ""))
        if path is None:
            return {"error": "That clip could not be found."}
        result = await asyncio.to_thread(
            analyse, path, viewpoint=_viewpoint(body.get("viewpoint")))
        return result.to_dict()

    @app.post("/api/analyse/run")
    async def api_analyse(request: Request) -> dict:
        """Run the chosen detector over the whole clip."""
        from vigia.analysis import analyse

        body = await request.json()
        path = _resolve_clip(body.get("file", ""))
        if path is None:
            return {"error": "That clip could not be found."}
        # No hazard means "you work it out": the scene router runs and its
        # answer is used. An explicit hazard is an override from someone who
        # disagreed with it, and always wins.
        raw = body.get("hazard")
        try:
            hazard = Hazard(raw) if raw else None
        except ValueError:
            return {"error": f"{raw!r} is not a hazard this system knows."}
        # to_thread, not a direct call: analysing a clip takes seconds, and on
        # the event loop those seconds stop every live camera, every MJPEG
        # stream and every WebSocket in the process.
        result = await asyncio.to_thread(
            analyse, path, hazard=hazard,
            viewpoint=_viewpoint(body.get("viewpoint")),
            max_frames=int(body.get("max_frames", 200)))
        return result.to_dict()

    @app.post("/api/analyse/upload")
    async def api_upload(file: UploadFile = File(...)) -> dict:
        """Accept a video the viewer chose from their own machine."""
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        safe = Path(file.filename or "upload.mp4").name
        target = ANALYSIS_DIR / safe
        payload = await file.read()
        await asyncio.to_thread(target.write_bytes, payload)
        return {"file": f"upload:{safe}", "name": safe,
                "size_mb": round(target.stat().st_size / 1048576, 1)}

    @app.get("/api/frame")
    def api_frame(file: str, seconds: float = 0.0):
        """One still from a clip, for drawing findings onto."""
        import cv2
        from fastapi.responses import Response

        path = _resolve_clip(file)
        if path is None:
            return Response(status_code=404)
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
        ok, image = capture.read()
        capture.release()
        if not ok:
            return Response(status_code=404)
        height = 720
        if image.shape[0] > height:
            scale = height / image.shape[0]
            image = cv2.resize(image, (int(image.shape[1] * scale), height))
        ok, buffer = cv2.imencode(".jpg", image,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @app.get("/api/cameras")
    def api_cameras() -> dict:
        """The cameras this instance is running, for the header selector.

        Live public cameras are listed first and flagged, with the credit and
        update interval the interface is obliged to show beside them.
        """
        cameras = []
        for cid in order:
            state = states[cid]
            cameras.append({
                "camera_id": cid,
                "name": state.name or cid,
                "context": state.context,
                "hazards": state.hazards,
                "live": state.live,
                "place": state.place,
                "attribution": state.attribution,
                "licence": state.licence,
                "interval_seconds": state.interval_seconds,
                "home_url": state.home_url,
            })
        cameras.sort(key=lambda c: (not c["live"], c["camera_id"]))
        return {"cameras": cameras}

    @app.get("/api/state")
    async def api_state(camera: Optional[str] = None) -> dict:
        """Everything the interface needs for one camera, as one document."""
        return pick(camera).snapshot()

    @app.get("/api/gate")
    def api_gate() -> dict:
        """Tier 0's routing decisions, so the header can show the gate."""
        if registry is None:
            return {"cameras": [], "summary": {}}
        return {
            "cameras": [c.to_dict() for c in registry.cameras],
            "summary": registry.gate_summary(),
        }

    @app.post("/api/suppressed/{enabled}")
    async def api_toggle(enabled: str, camera: Optional[str] = None) -> dict:
        """Toggle the suppressed layer.

        Turning it off makes the view look like every other detection demo,
        which is the fastest way to show an audience what is being added.
        """
        # Applied to every camera: the toggle is a property of the view, not
        # of one feed, and a viewer switching cameras should not find it reset.
        on = enabled.lower() in ("1", "true", "on")
        for view in states.values():
            view.show_suppressed = on
        return {"show_suppressed": on}

    @app.get("/video")
    async def video(camera: Optional[str] = None):
        """MJPEG. The browser's own decoder does the work."""
        boundary = "vigiaframe"
        view = pick(camera)

        async def frames():
            last = -1
            while True:
                snapshot_id = view.snapshot()["frame_id"]
                if snapshot_id != last:
                    last = snapshot_id
                    payload = view.jpeg()
                    yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                           f"Content-Length: {len(payload)}\r\n\r\n").encode()
                    yield payload
                    yield b"\r\n"
                await asyncio.sleep(1 / 30)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        )

    @app.websocket("/events")
    async def events(socket: WebSocket, camera: Optional[str] = None):
        """Small JSON only — never frames. See the module docstring."""
        await socket.accept()
        view = pick(camera)
        try:
            while True:
                await socket.send_text(json.dumps(view.snapshot()))
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            return
        except Exception:
            logger.debug("event socket closed", exc_info=True)
            return

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
