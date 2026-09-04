#!/usr/bin/env python3
"""Download fire ignition sequences from HPWREN's FIgLib.

FIgLib is the right dataset for evaluating the temporal validator, for three
reasons no single-frame dataset can match:

  1. It is a genuine time series from a *fixed* camera — the same viewpoint
     minute after minute, which is exactly the deployment VIGÍA targets.
  2. Ground truth comes from the filename. Each image is named
     `{epoch}_{offset}.jpg` where offset is seconds relative to ignition.
     Negative offsets are pre-ignition: whatever the camera saw then, it was
     not this fire. Those frames are our clean negatives.
  3. Sampling is one frame per minute, which is how PyroNear's deployed
     cameras actually run. Evaluating at 25 FPS would flatter the validator.

Attribution: HPWREN requires credit when its imagery is published. See NOTICE.
Source: https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/

Usage:
    python eval/fetch_figlib.py --list
    python eval/fetch_figlib.py --count 5
    python eval/fetch_figlib.py --sequence 20160604_FIRE_rm-n-mobo-c
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import re
import sys
import urllib.request
from urllib.parse import quote
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
DEFAULT_OUT = REPO_ROOT / "data" / "figlib"

IMAGE_PATTERN = re.compile(r"(\d+)_([+-]\d+)\.jpg")
SEQUENCE_PATTERN = re.compile(r'href=([0-9]{8}_FIRE_[A-Za-z0-9\-]+)/index\.html')


def fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "vigia/0.1 (research)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def list_sequences() -> list[str]:
    html = fetch(f"{BASE_URL}/index.html").decode("utf-8", "replace")
    # dict.fromkeys preserves order while de-duplicating
    return list(dict.fromkeys(SEQUENCE_PATTERN.findall(html)))


def sequence_images(name: str) -> list[tuple[str, int]]:
    """Return (filename, offset_seconds) for every image in a sequence."""
    html = fetch(f"{BASE_URL}/{name}/index.html").decode("utf-8", "replace")
    found = dict.fromkeys(IMAGE_PATTERN.findall(html))
    return [(f"{epoch}_{offset}.jpg", int(offset)) for epoch, offset in found]


def download_sequence(name: str, out_root: Path, limit: int | None = None,
                      negatives_only: bool = False) -> dict:
    """Download one sequence. Skips files already on disk."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(sequence_images(name), key=lambda item: item[1])

    if negatives_only:
        # Take the EARLIEST pre-ignition frames, furthest from the fire.
        #
        # This matters. FIgLib ignition timestamps are approximate, so frames
        # immediately before t=0 may already contain the first faint smoke.
        # Labelling those as negatives would manufacture false positives that
        # are actually correct detections — and would make the detector look
        # worse than it is while making the validator look better than it is.
        # Frames 30-40 minutes before ignition are safe.
        images = [i for i in images if i[1] < 0][: limit or None]
    elif limit:
        # Keep a balanced window either side of ignition rather than the first N.
        negatives = [i for i in images if i[1] < 0][-limit // 2:]
        positives = [i for i in images if i[1] > 0][: limit // 2]
        images = negatives + positives

    def grab(item: tuple[str, int]) -> bool:
        filename, _ = item
        target = out_dir / filename
        if target.exists() and target.stat().st_size > 1024:
            return True
        try:
            # Post-ignition filenames contain a literal '+', which a URL path
            # treats as a space — the CDN returns 403 for the un-encoded form.
            # Without this every positive frame silently vanishes and you are
            # left evaluating on negatives only.
            encoded = quote(filename, safe="")
            target.write_bytes(fetch(f"{BASE_URL}/{name}/{encoded}"))
            return True
        except Exception:
            return False

    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(grab, images))

    negatives = sum(1 for _, offset in images if offset < 0)
    positives = sum(1 for _, offset in images if offset > 0)
    size_mb = sum(p.stat().st_size for p in out_dir.glob("*.jpg")) / (1024 * 1024)

    return {
        "sequence": name,
        "requested": len(images),
        "downloaded": sum(results),
        "negatives": negatives,
        "positives": positives,
        "size_mb": round(size_mb, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list sequences and exit")
    parser.add_argument("--count", type=int, default=3, help="how many sequences to fetch")
    parser.add_argument("--sequence", action="append", default=None,
                        help="fetch a specific sequence (repeatable)")
    parser.add_argument("--limit", type=int, default=40,
                        help="max images per sequence, balanced around ignition (0 = all)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--negatives-only", action="store_true",
                        help="fetch only pre-ignition frames, taken from the "
                             "start of the window (furthest from the fire)")
    args = parser.parse_args()

    print("Fetching sequence index from HPWREN...")
    available = list_sequences()
    print(f"{len(available)} fire ignition sequences available\n")

    if args.list:
        for name in available:
            print(" ", name)
        return 0

    if args.sequence:
        chosen = [s for s in args.sequence if s in available]
        missing = set(args.sequence) - set(chosen)
        for name in missing:
            print(f"  ! unknown sequence: {name}", file=sys.stderr)
    else:
        # Spread across years rather than taking the oldest N — camera hardware
        # and image quality changed a lot over a decade, and a sample from only
        # 2016 would not represent the dataset.
        step = max(1, len(available) // args.count)
        chosen = available[::step][: args.count]

    print(f"Downloading {len(chosen)} sequence(s), "
          f"{args.limit or 'all'} images each:\n")

    total = {"downloaded": 0, "negatives": 0, "positives": 0, "size_mb": 0.0}
    for name in chosen:
        result = download_sequence(name, args.out, args.limit or None,
                                   negatives_only=args.negatives_only)
        print(f"  {result['sequence']:<40} {result['downloaded']:>3}/{result['requested']:<3} "
              f"images  ({result['negatives']} pre, {result['positives']} post)  "
              f"{result['size_mb']:>6.1f} MB")
        for key in ("downloaded", "negatives", "positives", "size_mb"):
            total[key] += result[key]

    print(f"\n{total['downloaded']} images  "
          f"({total['negatives']} pre-ignition, {total['positives']} post-ignition)  "
          f"{total['size_mb']:.1f} MB -> {args.out}")
    print("\nImagery courtesy of HPWREN (hpwren.ucsd.edu). Attribution required "
          "on publication — see NOTICE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
