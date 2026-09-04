#!/usr/bin/env python3
"""Fetch a contiguous SeaDronesSee flight sequence for temporal evaluation.

    python eval/fetch_seadronessee_sequence.py
    python eval/fetch_seadronessee_sequence.py --frames 120 --out data/seadronessee/val_seq

WHY THIS EXISTS. The people-in-water detector was evaluated on 300 validation
stills, which is correct for measuring a detector and useless for exercising a
temporal validator: the longest run of consecutive frames in that sample is
four. A demo clip built from it plays perfectly well and means nothing, because
nothing in it genuinely persists — the validator's rejections are then artefacts
of unrelated images, and its confirmations are artefacts of unrelated objects
happening to overlap between frames.

SeaDronesSee's object-detection split is drawn from continuous flights and its
filenames are frame numbers, so a run of consecutive ids IS continuous footage.
That is verified here rather than assumed: adjacent-frame histogram correlation
across a fetched run is about 0.997, against about 0.015 for random pairs from
the same split. The script refuses to declare success if that separation does
not hold.

Licence: CC0 1.0 Universal — no restriction of any kind, and the least
encumbered footage in the project. Credit is given by choice, not obligation:
Varga, Kiefer et al., University of Tübingen, IEEE/CVF WACV 2022.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPO = "dronefreak/SeaDronesSee"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
DEFAULT_OUT = REPO_ROOT / "data" / "seadronessee" / "val_seq"


def listing() -> list[str]:
    with urllib.request.urlopen(API, timeout=30) as response:
        data = json.load(response)
    return [f["rfilename"] for f in data.get("siblings", [])]


def longest_run(ids: list[int]) -> tuple[int, int]:
    """(length, first_id) of the longest consecutive run."""
    best_length, best_start = 0, ids[0] if ids else 0
    start, run = (ids[0] if ids else 0), 1
    for index in range(1, len(ids)):
        if ids[index] == ids[index - 1] + 1:
            run += 1
            if run > best_length:
                best_length, best_start = run, start
        else:
            run, start = 1, ids[index]
    return best_length, best_start


def verify_continuity(paths: list[Path],
                      baseline_paths: list[Path]) -> tuple[float, float]:
    """(median adjacent correlation, median unrelated correlation).

    A sequence must look far more like itself frame to frame than two frames
    from DIFFERENT flights do. Without this check the script would happily
    produce a plausible clip out of unrelated stills, which is the exact
    failure it exists to prevent.

    THE BASELINE MUST COME FROM OUTSIDE THE SEQUENCE. A first version drew its
    "unrelated" pairs from the fetched frames themselves and so compared
    continuous footage against continuous footage: adjacent 0.997 against
    unrelated 0.989, and the check failed on footage that was in fact perfectly
    continuous. A comparison against itself measures nothing — the same mistake
    as a parity check run on noise.
    """
    import cv2
    import numpy as np

    def histogram(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        return cv2.normalize(hist, None)

    def similarity(a, b):
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        return cv2.compareHist(histogram(a), histogram(b), cv2.HISTCMP_CORREL)

    images = [cv2.imread(str(p)) for p in paths[:24]]
    images = [i for i in images if i is not None]
    adjacent = [similarity(a, b) for a, b in zip(images, images[1:])]

    others = [cv2.imread(str(p)) for p in baseline_paths[:12]]
    others = [i for i in others if i is not None]
    if not others:
        return float(np.median(adjacent)), float("nan")

    rng = np.random.default_rng(0)
    unrelated = [
        similarity(images[int(rng.integers(len(images)))],
                   others[int(rng.integers(len(others)))])
        for _ in range(12)
    ]
    return float(np.median(adjacent)), float(np.median(unrelated))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frames", type=int, default=69)
    parser.add_argument("--split", default="val", choices=["val", "train", "test"])
    args = parser.parse_args()

    print(f"listing {REPO} …")
    try:
        files = listing()
    except Exception as exc:
        print(f"could not reach the dataset mirror: {exc}", file=sys.stderr)
        return 1

    images = [f for f in files if f"/{args.split}/" in f and f.endswith(".jpg")]
    if not images:
        print(f"no {args.split} images in the mirror listing", file=sys.stderr)
        return 1

    by_id = {int(os.path.basename(f).split(".")[0]): f for f in images}
    length, first = longest_run(sorted(by_id))
    take = min(args.frames, length)
    print(f"{len(images)} {args.split} images; longest consecutive run "
          f"{length} frames from id {first}; taking {take}")

    args.out.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for frame_id in range(first, first + take):
        remote = by_id[frame_id]
        target = args.out / os.path.basename(remote)
        if target.exists():
            fetched += 1
            continue
        try:
            urllib.request.urlretrieve(RESOLVE + remote, target)
            fetched += 1
        except Exception as exc:
            print(f"  failed {remote}: {exc}", file=sys.stderr)
            target.unlink(missing_ok=True)

    paths = sorted(args.out.glob("*.jpg"),
                   key=lambda p: int(p.stem.split(".")[0]))
    print(f"fetched {fetched} frames into {args.out}")

    # Baseline: frames from OTHER flights. Prefer the sparse validation sample
    # already on disk; fall back to ids far outside the fetched run.
    baseline_dir = args.out.parent / args.split
    baseline = sorted(baseline_dir.glob("*.jpg"))[:12] if baseline_dir.is_dir() else []
    if not baseline:
        far = [i for i in sorted(by_id) if abs(i - first) > 2 * max(take, 100)][:12]
        baseline_dir = args.out / "_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in far:
            target = baseline_dir / os.path.basename(by_id[frame_id])
            if not target.exists():
                try:
                    urllib.request.urlretrieve(RESOLVE + by_id[frame_id], target)
                except Exception:
                    continue
        baseline = sorted(baseline_dir.glob("*.jpg"))

    adjacent, unrelated = verify_continuity(paths, baseline)
    print(f"continuity: adjacent {adjacent:.3f} vs unrelated {unrelated:.3f} "
          f"(baseline from {len(baseline)} frames of other flights)")
    if unrelated != unrelated:      # NaN — no baseline available
        print("  UNVERIFIED: no out-of-sequence frames to compare against.",
              file=sys.stderr)
        return 1
    if adjacent < unrelated + 0.15:
        print("  FAILED: these frames are not continuous footage. Do not use "
              "them to demonstrate\n  temporal validation.", file=sys.stderr)
        return 1
    print("  verified continuous — usable for temporal evaluation and for the "
          "recorded demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
