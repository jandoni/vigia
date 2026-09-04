#!/usr/bin/env python3
"""Start the operator view over a live pipeline.

    python scripts/run_operator_view.py --source clips/fire_ridge.mp4 --context forest
    python scripts/run_operator_view.py --registry configs/cameras.example.yaml \
                                        --clips clips/ --camera ridge_01

Then open http://127.0.0.1:8000.

The pipeline runs on a background thread and publishes into a small shared
state; the server only renders it. The pipeline never blocks on the UI — a slow
or absent browser costs staleness and nothing else, which is the property that
lets the same code run headless on an edge node.

STAGE PATH. `--loop` replays a clip continuously so a demo can be left running,
and file sources are processed deterministically frame by frame, so a given clip
produces the same output every run. Rehearsal and stage behave identically,
which is the only way a live demonstration is worth attempting at all.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vigia.io.alerts import AlertSink                     # noqa: E402
from vigia.pipeline import CameraPipeline                 # noqa: E402
from vigia.registry import Camera, CameraContext, CameraRegistry  # noqa: E402
from vigia.web.server import ViewState, create_app        # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="video file, RTSP URL, or webcam index")
    parser.add_argument("--context", default="forest",
                        choices=[c.value for c in CameraContext])
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--clips", type=Path, default=None)
    parser.add_argument("--camera", default=None,
                        help="which camera from the registry to display")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-fps", type=float, default=12.0,
                        help="cap replay rate so a clip plays at a watchable speed")
    parser.add_argument("--loop", action="store_true",
                        help="replay the clip continuously (stage path)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "alerts")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    try:
        import uvicorn
    except ImportError:
        print("The operator view needs the optional web extra:\n"
              "    pip install -e '.[web]'\n"
              "It is optional by design — an edge node runs headless.",
              file=sys.stderr)
        return 2

    # --- resolve which camera to show -------------------------------- #
    registry = None
    if args.registry:
        registry = CameraRegistry.from_yaml(args.registry)
        if args.camera:
            if args.camera not in registry:
                print(f"camera {args.camera!r} not in {args.registry}",
                      file=sys.stderr)
                return 1
            camera = registry.get(args.camera)
        else:
            enabled = registry.enabled_cameras
            if not enabled:
                print("registry has no enabled cameras", file=sys.stderr)
                return 1
            camera = enabled[0]
        if args.clips and "://" not in camera.source \
                and not str(camera.source).isdigit():
            camera.source = str(args.clips / Path(camera.source).name)
    elif args.source:
        camera = Camera("adhoc", args.source, CameraContext(args.context))
        registry = CameraRegistry([camera])
    else:
        print("give --source or --registry", file=sys.stderr)
        return 1

    # --- wire pipeline -> view state ---------------------------------- #
    state = ViewState()
    state.camera_id = camera.camera_id
    state.context = camera.context.value
    state.hazards = sorted(h.value for h in camera.active_hazards)

    sink = AlertSink(args.out)

    def publish(result, frame) -> None:
        stats = pipeline.validators[result.hazard].stats_dict()
        state.publish(result, frame, stats)

    pipeline = CameraPipeline(camera, sink=sink, on_result=publish)
    pipeline.warmup()

    def run_pipeline() -> None:
        while True:
            pipeline.run(max_fps=args.max_fps, loop=args.loop)
            if not args.loop:
                return
            pipeline.reset()

    threading.Thread(target=run_pipeline, name="pipeline", daemon=True).start()

    print(f"camera   : {camera.camera_id} ({camera.context.value})")
    print(f"hazards  : {', '.join(state.hazards)}")
    print(f"operator : http://{args.host}:{args.port}")

    app = create_app(state, registry)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
