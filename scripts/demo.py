#!/usr/bin/env python3
"""Start the whole demonstration with one command.

    python scripts/demo.py          # or: make dev

Brings up every camera in the demonstration registry, serves them all in one
operator view, and opens a browser. Nothing else to run and no flags to
remember — the point is that a judge or a municipality can clone this and see it
work.

It checks its own prerequisites rather than failing halfway: missing clips are
built, a missing model is reported by name with the command that fetches it, and
a camera whose viewpoint is outside its detector's envelope is skipped with the
reason rather than crashing the run.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REGISTRY = REPO_ROOT / "configs" / "cameras.demo.yaml"
DEFAULT_CLIPS = REPO_ROOT / "clips"


def ensure_clips(clips: Path, registry_path: Path) -> bool:
    """Build the demonstration clips if they are not already present."""
    import yaml

    from vigia.io.live import looks_live_source

    entries = yaml.safe_load(registry_path.read_text(encoding="utf-8"))["cameras"]
    wanted = [Path(e["source"]).name for e in entries
              if e.get("enabled", True) and not looks_live_source(e["source"])]
    missing = [w for w in wanted if not (clips / w).exists()]
    if not missing:
        return True

    print(f"building {len(missing)} missing clip(s): {', '.join(missing)}")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_demo_clips.py")],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-1500:], file=sys.stderr)
        print("could not build clips — see above", file=sys.stderr)
        return False
    still = [w for w in wanted if not (clips / w).exists()]
    if still:
        print(f"still missing after build: {', '.join(still)}\n"
              f"Those need their source data; see docs/DATA_ACQUISITION.md.",
              file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-fps", type=float, default=8.0)
    parser.add_argument("--no-browser", action="store_true")
    # Replay runs ONCE by default. Looping four clips through four detectors
    # at 8 fps for as long as the window is open is what made this laptop hot
    # enough to notice, and it buys nothing: the counters, the timeline and
    # the suppression figures all survive the end of a clip. The live public
    # cameras keep updating either way, so the demonstration does not go
    # still. Pass --loop when you want a screen that never stops moving.
    parser.add_argument("--loop", action="store_true",
                        help="replay the recorded clips continuously "
                             "(default: play each once)")
    parser.add_argument("--no-live", action="store_true",
                        help="skip the public live cameras and run clips only")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("The operator view needs the optional web extra:\n"
              "    pip install -e '.[web]'", file=sys.stderr)
        return 2

    from vigia.io import live as live_catalogue
    from vigia.io.alerts import AlertSink
    from vigia.io.live import looks_live_source
    from vigia.pipeline import CameraPipeline
    from vigia.registry import CameraRegistry
    from vigia.web.server import ViewState, create_app

    if not args.registry.exists():
        print(f"no registry at {args.registry}", file=sys.stderr)
        return 1

    if not ensure_clips(args.clips, args.registry):
        return 1

    registry = CameraRegistry.from_yaml(args.registry)
    for camera in registry.cameras:
        if "://" not in camera.source and not str(camera.source).isdigit():
            camera.source = str(args.clips / Path(camera.source).name)

    print(f"\nVIGÍA demonstration")
    print(f"  registry : {args.registry.name}")

    sink = AlertSink(REPO_ROOT / "runs" / "alerts")
    states, pipelines, skipped = {}, {}, []

    for camera in registry.enabled_cameras:
        live = looks_live_source(camera.source)
        if live and args.no_live:
            continue
        if not live and not Path(camera.source).exists():
            skipped.append((camera.camera_id, "clip not found"))
            continue

        state = ViewState()
        state.camera_id = camera.camera_id
        state.context = camera.context.value
        state.hazards = sorted(h.value for h in camera.active_hazards)
        state.name = camera.name or camera.camera_id
        state.source = camera.source

        # A public camera arrives with obligations attached — a credit, a
        # licence, and a publishing cadence the interface has to state. They
        # come from the catalogue rather than from this launcher, so there is
        # one place where "may we show this, and whose is it" is answered.
        entry = live_catalogue.BY_SOURCE.get(camera.source)
        if entry is not None:
            state.name = entry.name
            state.place = entry.place
            state.attribution = entry.attribution
            state.licence = entry.licence
            state.interval_seconds = entry.interval_seconds
            state.home_url = entry.home_url

        def publish(result, frame, _state=state, _cam=camera.camera_id):
            stats = pipelines[_cam].validators[result.hazard].stats_dict()
            _state.publish(result, frame, stats)

        try:
            pipeline = CameraPipeline(camera, sink=sink, on_result=publish)
        except ValueError as exc:
            # Viewpoint outside the detector's envelope, most likely. Skip the
            # camera and say why rather than taking the whole demo down.
            skipped.append((camera.camera_id, str(exc).split("\n")[0]))
            continue

        pipelines[camera.camera_id] = pipeline
        states[camera.camera_id] = state
        state.live = live
        # The source only exists once the pipeline is running, so the view asks
        # for its status through a callable rather than holding a snapshot that
        # would be stale the moment it was taken.
        state.source_status = (
            lambda _p=pipeline: _p.source.status()
            if getattr(_p, "source", None) is not None
            and hasattr(_p.source, "status") else None)
        print(f"  {'LIVE     ' if live else 'camera   '}: {camera.camera_id:<18} "
              f"{camera.context.value:<14} {', '.join(state.hazards)}")

    if not states:
        print("no runnable cameras", file=sys.stderr)
        for cid, why in skipped:
            print(f"  {cid}: {why}", file=sys.stderr)
        return 1

    for cid, why in skipped:
        print(f"  skipped  : {cid} — {why}")

    summary = registry.gate_summary()
    print(f"  gate     : {summary['detector_invocations_per_frame_gated']} of "
          f"{summary['detector_invocations_per_frame_ungated']} detector "
          f"invocations per frame ({summary['reduction']:.0%} fewer)")

    print("\n  loading models …")
    for pipeline in pipelines.values():
        pipeline.warmup()

    def run(pipeline):
        # A live camera's source never ends, so looping and frame-rate capping
        # are meaningless for it — it is already paced by whoever runs it.
        live = looks_live_source(pipeline.camera.source)
        if live:
            pipeline.run()
            return
        while True:
            pipeline.run(max_fps=args.max_fps, loop=args.loop)
            if not args.loop:
                return
            pipeline.reset()

    for pipeline in pipelines.values():
        threading.Thread(target=run, args=(pipeline,), daemon=True).start()

    url = f"http://{args.host}:{args.port}"
    print(f"\n  ready → {url}\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app = create_app(states, registry)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
