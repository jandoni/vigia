#!/usr/bin/env python3
"""R1 — does the temporal validator actually reduce false alarms?

Runs the fire detector over FIgLib ignition sequences in temporal order, with
the validator switched off and then on, and reports what changes.

Ground truth comes from the filename offset: frames before ignition (negative
offset) contain no fire, frames after it do. So:

    false-alarm rate = fraction of PRE-ignition frames that raised an alert
    detection rate   = fraction of POST-ignition frames that raised an alert
    time to detect   = minutes after ignition before the first alert

The third metric is why this cannot be measured on single images. Persistence
buys precision by waiting, and waiting costs detection latency. At one frame
per minute — how these cameras actually run — requiring 3 frames means
tolerating roughly 3 extra minutes. Any honest claim about the validator has
to show both sides of that trade, so this harness reports them together.

Usage:
    python eval/run_sequence_eval.py
    python eval/run_sequence_eval.py --min-frames 2 3 4 --json eval/results/r1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from vigia.detectors.base import Backend  # noqa: E402
from vigia.detectors.fire import FireDetector  # noqa: E402
from vigia.types import Frame, Hazard  # noqa: E402
from vigia.validator.cascade import TemporalValidator, ValidatorConfig  # noqa: E402

logging.basicConfig(level=logging.ERROR)

FILENAME = re.compile(r"(\d+)_([+-]\d+)\.jpg")


@dataclass
class SequenceResult:
    name: str
    pre_frames: int = 0
    post_frames: int = 0
    pre_alarmed: int = 0          # pre-ignition frames that raised an alert
    post_alarmed: int = 0         # post-ignition frames that raised an alert
    alerts_pre: int = 0           # distinct alerts before ignition
    alerts_post: int = 0
    first_alert_offset: float | None = None   # seconds after ignition

    @property
    def false_alarm_rate(self) -> float:
        return self.pre_alarmed / self.pre_frames if self.pre_frames else 0.0

    @property
    def detection_rate(self) -> float:
        return self.post_alarmed / self.post_frames if self.post_frames else 0.0

    @property
    def detected(self) -> bool:
        return self.first_alert_offset is not None and self.first_alert_offset > 0


def load_sequence(directory: Path) -> list[tuple[Path, int]]:
    """Return (path, offset_seconds) sorted in temporal order."""
    items = []
    for path in directory.glob("*.jpg"):
        match = FILENAME.match(path.name)
        if match:
            items.append((path, int(match.group(2))))
    return sorted(items, key=lambda item: item[1])


def run_sequence(
    detector: FireDetector,
    directory: Path,
    validator: TemporalValidator | None,
) -> SequenceResult:
    """Play one ignition sequence through the pipeline in temporal order."""
    result = SequenceResult(name=directory.name)
    if validator is not None:
        validator.reset()

    for index, (path, offset) in enumerate(load_sequence(directory)):
        image = cv2.imread(str(path))
        if image is None:
            continue

        # Timestamps are real: FIgLib samples one frame per minute, so the
        # validator's persistence and cooldown windows operate in real seconds.
        timestamp = float(offset)
        frame = Frame(image=image, index=index, timestamp=timestamp,
                      camera_id=directory.name)

        detections = detector.detect(frame)

        if validator is None:
            alarmed = bool(detections)
            alert_count = len(detections)
        else:
            events = validator.process(
                detections, index, timestamp, image, directory.name
            )
            alarmed = bool(events)
            alert_count = len(events)

        if offset < 0:
            result.pre_frames += 1
            result.pre_alarmed += int(alarmed)
            result.alerts_pre += alert_count
        else:
            result.post_frames += 1
            result.post_alarmed += int(alarmed)
            result.alerts_post += alert_count

        if alarmed and result.first_alert_offset is None:
            result.first_alert_offset = timestamp

    return result


def summarise(results: list[SequenceResult]) -> dict:
    pre_frames = sum(r.pre_frames for r in results)
    post_frames = sum(r.post_frames for r in results)
    pre_alarmed = sum(r.pre_alarmed for r in results)
    post_alarmed = sum(r.post_alarmed for r in results)

    detected = [r for r in results if r.detected]
    latencies = [r.first_alert_offset / 60.0 for r in detected]

    return {
        "sequences": len(results),
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "false_alarm_rate": round(pre_alarmed / pre_frames, 4) if pre_frames else 0.0,
        "detection_rate": round(post_alarmed / post_frames, 4) if post_frames else 0.0,
        "sequences_detected": len(detected),
        "total_alerts_pre": sum(r.alerts_pre for r in results),
        "total_alerts_post": sum(r.alerts_post for r in results),
        "median_minutes_to_detect": (
            round(sorted(latencies)[len(latencies) // 2], 1) if latencies else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "figlib")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--min-frames", type=int, nargs="+", default=[2, 3, 4],
                        help="persistence settings to sweep")
    parser.add_argument("--backend", default="reference", choices=["reference", "fast"])
    parser.add_argument("--cooldown", action="store_true",
                        help="enable the cooldown level (off by default: it "
                             "confounds per-frame recall, see note in source)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    directories = sorted(d for d in args.data.iterdir() if d.is_dir())
    if not directories:
        print(f"No sequences in {args.data}. Run eval/fetch_figlib.py first.",
              file=sys.stderr)
        return 1

    detector = FireDetector(conf_threshold=args.conf, backend=Backend(args.backend))
    detector.warmup(rounds=2)

    print(f"sequences : {len(directories)}")
    print(f"model     : {detector.model_path.name}")
    print(f"backend   : {detector.backend.value} "
          f"({detector.session.get_providers()[0]})")
    print(f"conf      : {args.conf}")
    print("frames    : 1 per minute (FIgLib sampling)\n")

    payload: dict = {
        "detector": detector.model_path.name,
        "backend": detector.backend.value,
        "conf_threshold": args.conf,
        "sequences": [d.name for d in directories],
        "configurations": {},
    }

    header = (f"{'configuration':<30} {'FAR':>7} {'recall':>7} {'fires found':>12} "
              f"{'false alerts':>13} {'min-to-detect':>14}")
    print(header)
    print("-" * len(header))

    # --- baseline: raw detector, no validation ------------------------ #
    raw = [run_sequence(detector, d, None) for d in directories]
    summary = summarise(raw)
    payload["configurations"]["no_validator"] = summary
    print(f"{'raw detector (no validator)':<30} "
          f"{summary['false_alarm_rate']:>7.3f} {summary['detection_rate']:>7.3f} "
          f"{summary['sequences_detected']:>4}/{len(raw):<2} "
          f"{summary['total_alerts_pre']:>12} "
          f"{str(summary['median_minutes_to_detect']):>14}")

    # --- validator, sweeping persistence ------------------------------ #
    # Cooldown is deliberately OFF here. It suppresses repeat alerts from a
    # location that has already alerted, which is right in production but makes
    # "fraction of post-ignition frames that alarmed" measure alert *frequency*
    # rather than detection *capability*. Leaving it on made the validator look
    # like it destroyed recall (0.925 -> 0.125) when in fact it found more fires
    # than the raw detector. Cooldown is evaluated separately.
    for min_frames in args.min_frames:
        config = ValidatorConfig(
            min_confidence=args.conf,
            min_frames=min_frames,
            enable_cooldown=args.cooldown,
            cooldown_seconds=600.0,
        )
        validator = TemporalValidator(Hazard.FIRE, config)
        validated = [run_sequence(detector, d, validator) for d in directories]
        summary = summarise(validated)
        summary["levels"] = validator.stats_dict()["levels"]
        payload["configurations"][f"validator_min_frames_{min_frames}"] = summary
        print(f"{'validator, persistence=' + str(min_frames):<30} "
              f"{summary['false_alarm_rate']:>7.3f} {summary['detection_rate']:>7.3f} "
              f"{summary['sequences_detected']:>4}/{len(validated):<2} "
              f"{summary['total_alerts_pre']:>12} "
              f"{str(summary['median_minutes_to_detect']):>14}")

    base = payload["configurations"]["no_validator"]["false_alarm_rate"]
    print(f"\nFAR         = fraction of PRE-ignition frames that alarmed "
          f"(baseline {base:.1%})")
    print("recall      = fraction of POST-ignition frames that alarmed")
    print("fires found = sequences detected at all — the capability metric")
    print("\nNOTE: FIgLib pre-ignition frames are EASY negatives — the same camera")
    print("shortly before ignition, usually a clear scene. The raw detector's FAR")
    print("here is under 1%, versus 87% on the pyro-sdis hard-negative set. The")
    print("validator therefore has almost nothing to remove on this data. Do not")
    print("quote this FAR as evidence the validator works; it is evidence that")
    print("this negative set is too easy to test it. See PLAN.md.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
