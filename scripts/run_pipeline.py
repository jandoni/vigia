#!/usr/bin/env python3
"""Run VIGÍA over a camera registry. This is the demo entry point.

    python scripts/run_pipeline.py --registry configs/cameras.example.yaml \
                                   --clips clips/

One command, a registry file and a directory of clips — the acceptance
criterion for Phase 4. The same code path serves the stage demo and a
deployment; only the registry changes, which is the point of Tier 0.

    # gate only: what would run where, loading nothing
    python scripts/run_pipeline.py --registry configs/cameras.example.yaml --explain

    # one camera, ad hoc, without writing a registry file
    python scripts/run_pipeline.py --source clip.mp4 --context forest

The summary reports proposed vs confirmed detections per hazard, which is the
project's headline in miniature: the validator's contribution is the gap
between the two, produced by the same code that produces it in evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vigia.io.alerts import AlertSink                       # noqa: E402
from vigia.pipeline import CameraPipeline, MultiCameraPipeline  # noqa: E402
from vigia.registry import Camera, CameraContext, CameraRegistry  # noqa: E402


def resolve_sources(registry: CameraRegistry, clips: Path | None) -> list[str]:
    """Rewrite each camera's source relative to a clip directory.

    Returns the ids of cameras whose clip is missing, so the run reports them
    instead of failing halfway through with a stack trace.
    """
    missing = []
    for camera in registry.cameras:
        if clips is not None and "://" not in camera.source \
                and not str(camera.source).isdigit():
            camera.source = str(clips / Path(camera.source).name)
        if "://" not in camera.source and not str(camera.source).isdigit() \
                and not Path(camera.source).exists():
            missing.append(camera.camera_id)
            camera.enabled = False
    return missing


def explain(registry: CameraRegistry) -> int:
    """Print the gate's decisions without loading a single model."""
    print(f"registry: {len(registry)} cameras, "
          f"{len(registry.enabled_cameras)} enabled\n")
    width = max((len(c.camera_id) for c in registry.cameras), default=10)
    for camera in registry.cameras:
        hazards = sorted(h.value for h in camera.active_hazards)
        flag = "" if camera.enabled else "  [disabled]"
        override = "  (override)" if camera.overrides_context else ""
        print(f"  {camera.camera_id:<{width}}  {camera.context.value:<18} "
              f"{', '.join(hazards) or '-'}{override}{flag}")

    summary = registry.gate_summary()
    print(f"\ngate: {summary['detector_invocations_per_frame_gated']} detector "
          f"invocations per frame instead of "
          f"{summary['detector_invocations_per_frame_ungated']} "
          f"({summary['reduction']:.0%} fewer)")
    print(f"models loaded: {', '.join(summary['models_loaded'])}")
    return 0


def report(name: str, stats) -> None:
    data = stats.to_dict()
    print(f"\n--- {name} ---")
    print(f"  frames      : {data['frames_processed']} processed"
          f"  ({data['frames_dropped']} dropped)  {data['fps']} fps")
    print(f"  proposed    : {data['detections_proposed']}")
    print(f"  confirmed   : {data['events_confirmed']}")
    print(f"  filtered    : {data['filtered_not_credible']}"
          f"  ({data['filtering_rate']:.1%})  rejected as not credible")
    print(f"  deduplicated: {data['deduplicated_repeat_events']}"
          f"  ({data['deduplication_rate']:.1%})  repeats of a reported event")
    for hazard, detail in data["per_hazard"].items():
        levels = detail["validator"]["levels"]
        rejected = ", ".join(
            f"{lv['name']} {lv['rejected']}" for lv in levels if lv["rejected"]
        )
        print(f"    {hazard:<16} {detail['median_latency_ms']:>6.1f} ms"
              f"   rejected by: {rejected or 'nothing'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path,
                        default=REPO_ROOT / "configs" / "cameras.example.yaml")
    parser.add_argument("--clips", type=Path, default=None,
                        help="directory holding the clips named in the registry")
    parser.add_argument("--source", default=None,
                        help="run one ad-hoc camera instead of a registry")
    parser.add_argument("--context", default="forest",
                        choices=[c.value for c in CameraContext])
    parser.add_argument("--camera", default=None,
                        help="run only this camera from the registry")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "alerts")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-fps", type=float, default=None)
    parser.add_argument("--explain", action="store_true",
                        help="show the gate's decisions and exit")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- ad-hoc single camera ---------------------------------------- #
    if args.source:
        camera = Camera("adhoc", args.source, CameraContext(args.context))
        if args.explain:
            return explain(CameraRegistry([camera]))
        sink = AlertSink(args.out)
        pipeline = CameraPipeline(camera, sink=sink)
        pipeline.warmup()
        stats = pipeline.run(max_frames=args.max_frames, max_fps=args.max_fps)
        report("adhoc", stats)
        print(f"\nalerts: {sink.counts_by_hazard()} -> {args.out}")
        return 0

    # --- registry ----------------------------------------------------- #
    if not args.registry.exists():
        print(f"no registry at {args.registry}", file=sys.stderr)
        return 1
    registry = CameraRegistry.from_yaml(args.registry)

    if args.camera:
        if args.camera not in registry:
            print(f"camera {args.camera!r} is not in {args.registry}",
                  file=sys.stderr)
            return 1
        keep = registry.get(args.camera)
        registry = CameraRegistry([keep])

    # --explain describes the GATE, so it runs before sources are resolved:
    # whether a clip happens to be on disk says nothing about which hazards a
    # camera watches, and disabling cameras first made the gate summary read 0.
    if args.explain:
        return explain(registry)

    missing = resolve_sources(registry, args.clips)
    if missing:
        print(f"note: no clip found for {', '.join(missing)} — disabled for "
              f"this run", file=sys.stderr)

    if not registry.enabled_cameras:
        print("no enabled cameras with a readable source", file=sys.stderr)
        return 1

    sink = AlertSink(args.out)
    multi = MultiCameraPipeline(registry, sink=sink)
    results = multi.run(max_frames=args.max_frames, max_fps=args.max_fps)

    for camera_id, stats in results.items():
        report(camera_id, stats)

    print(f"\nalerts: {sink.counts_by_hazard()} -> {args.out}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "registry": str(args.registry),
            "gate": registry.gate_summary(),
            "cameras": {cid: s.to_dict() for cid, s in results.items()},
            "alerts": sink.counts_by_hazard(),
        }, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
