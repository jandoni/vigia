#!/usr/bin/env python3
"""Fetch a daylight run from a USGS streamgage camera.

    python eval/fetch_usgs_camera.py
    python eval/fetch_usgs_camera.py --search "river" --list
    python eval/fetch_usgs_camera.py --camera IL_Lick_Creek_near_Woodside --frames 40

WHY THIS EXISTS. Flood is the keystone hazard and had no footage that could be
shown publicly. The only real camera sequences available for it came from
V-FloodNet, which is all rights reserved: usable to develop and evaluate
against, and impossible to put in a recorded demonstration or a dossier figure
without breaching the licence.

USGS operates over 1,300 streamgage cameras whose imagery is a work of the
United States Government and therefore in the public domain. They are also the
right *kind* of footage: fixed mount, pointed at water, sampled roughly hourly —
the same sparse-sampling regime the flood trend detection was calibrated for,
and the regime in which a fixed 300-second window originally failed outright.

DAYLIGHT ONLY, and that is a stated envelope rather than a convenience. These
cameras switch to infrared at night, and the segmenter — trained on ATLANTIS,
which is daylight photography — tints sky and vegetation as water on those
frames. Measured on a night run from this same camera: coverage swings between
0.25 and 0.61 with the mask covering most of the frame, against a steady 0.38 to
0.50 across daylight frames of the same scene. The detector's declared envelope
already excludes night; this script enforces it rather than restating it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CAMERAS_API = "https://api.waterdata.usgs.gov/nims/cameras"
BUCKET = "https://usgs-nims-images.s3.amazonaws.com/"
DEFAULT_CAMERA = "IL_Lick_Creek_near_Woodside"
DEFAULT_OUT = REPO_ROOT / "data" / "usgs_camera"

#: UTC hours to accept. The default suits US Central sites (UTC-5/6); shift it
#: for cameras in other time zones rather than widening it.
DAYLIGHT_UTC = range(14, 24)


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def list_keys(camera: str, limit: int = 1000, prefix_suffix: str = "") -> list[str]:
    """Image keys for one camera, oldest first.

    `prefix_suffix` narrows to a date. The keys embed the capture timestamp,
    which is what makes fetching one specific flood event possible without
    paging the whole bucket.
    """
    prefix = f"720/{camera}/"
    if prefix_suffix:
        prefix += f"{camera}___{prefix_suffix}"
    url = (f"{BUCKET}?list-type=2&prefix={prefix}&max-keys={limit}")
    with urllib.request.urlopen(url, timeout=60) as response:
        tree = ElementTree.fromstring(response.read())
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [node.text for node in tree.findall(".//s3:Contents/s3:Key", ns)
            if node.text and node.text.endswith(".jpg")]


def is_daylight(key: str) -> bool:
    match = re.search(r"___\d{4}-\d{2}-\d{2}T(\d{2})-", key)
    return bool(match) and int(match.group(1)) in DAYLIGHT_UTC


def validate_against_gage(args, paths, coverage):
    """Pearson r between segmented water coverage and measured gage height."""
    import datetime as dt

    import numpy as np

    days = sorted({m.group(1) for p in paths
                   if (m := re.search(r"___(\d{4}-\d{2}-\d{2})T", p.name))})
    if not days:
        return None
    url = ("https://waterservices.usgs.gov/nwis/iv/?format=json&sites="
           f"{args.gage_site}&parameterCd=00065"
           f"&startDT={days[0]}T00:00Z&endDT={days[-1]}T23:59Z")
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            series = json.load(response)["value"]["timeSeries"][0]["values"][0]["value"]
    except Exception:
        return None

    gage = {}
    for point in series:
        stamp = dt.datetime.fromisoformat(point["dateTime"]).astimezone(
            dt.timezone.utc).replace(tzinfo=None)
        gage[stamp] = float(point["value"])
    if not gage:
        return None

    pairs = []
    for path, cover in zip(paths, coverage):
        match = re.search(r"___(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z",
                          path.name)
        if not match:
            continue
        stamp = dt.datetime.fromisoformat(
            f"{match.group(1)}T{match.group(2)}:{match.group(3)}:{match.group(4)}")
        nearest = min(gage, key=lambda g: abs((g - stamp).total_seconds()))
        if abs((nearest - stamp).total_seconds()) <= 1800:
            pairs.append((cover, gage[nearest]))
    if len(pairs) < 4:
        return None
    return float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--search", default=None,
                        help="filter the camera list by name")
    parser.add_argument("--list", action="store_true",
                        help="list matching cameras and exit")
    parser.add_argument("--days", nargs="*", default=None,
                        help="specific YYYY-MM-DD days, e.g. a flood event")
    parser.add_argument("--gage-site", default=None,
                        help="USGS site number; validates coverage against "
                             "measured gage height")
    args = parser.parse_args()

    if args.list or args.search:
        print("listing USGS streamgage cameras …")
        cameras = fetch_json(CAMERAS_API)
        needle = (args.search or "").lower()
        matches = [c for c in cameras
                   if not needle or needle in c["camName"].lower()
                   or needle in c["camId"].lower()]
        print(f"{len(matches)} of {len(cameras)} cameras match")
        for camera in matches[:40]:
            print(f"  {camera['camId']:<44} {camera['camName']}")
        if args.list:
            return 0

    print(f"camera: {args.camera}")
    if args.days:
        keys = sorted(k for day in args.days
                      for k in list_keys(args.camera, prefix_suffix=day))
    else:
        keys = list_keys(args.camera)
    keys = [k for k in keys if is_daylight(k)]
    if not keys:
        print("no daylight imagery found for this camera", file=sys.stderr)
        return 1
    keys = keys[-args.frames:]
    print(f"{len(keys)} daylight frames (UTC hours "
          f"{DAYLIGHT_UTC.start}-{DAYLIGHT_UTC.stop - 1})")

    args.out.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for key in keys:
        target = args.out / Path(key).name
        if target.exists():
            fetched += 1
            continue
        try:
            urllib.request.urlretrieve(BUCKET + key, target)
            fetched += 1
        except Exception as exc:
            print(f"  failed {key}: {exc}", file=sys.stderr)

    print(f"fetched {fetched} frames into {args.out}")

    (args.out / "SOURCE.txt").write_text(
        f"USGS streamgage camera: {args.camera}\n"
        f"Source: {BUCKET}720/{args.camera}/\n"
        f"Camera index: {CAMERAS_API}\n"
        "Licence: work of the United States Government — public domain.\n"
        "Daylight frames only; these cameras switch to infrared at night and\n"
        "the flood segmenter's declared envelope is daylight outdoor scenes.\n",
        encoding="utf-8")

    # Report what the segmenter actually sees, so a bad camera is caught here
    # rather than in a rendered demo.
    try:
        import cv2
        import numpy as np

        from vigia.detectors.flood import FloodDetector
        from vigia.types import Frame

        detector = FloodDetector()
        paths = sorted(args.out.glob("*.jpg"))
        coverage = [
            detector.observe(Frame(image=cv2.imread(str(p)), index=i,
                                   timestamp=float(i) * 3600)).coverage
            for i, p in enumerate(paths)
        ]
        if coverage:
            print(f"water coverage: median {np.median(coverage):.3f}  "
                  f"min {min(coverage):.3f}  max {max(coverage):.3f}")
            if np.median(coverage) < 0.05:
                print("  WARNING: almost no water segmented — this camera may "
                      "not be pointed at\n  a waterbody, or may be outside the "
                      "detector's envelope.", file=sys.stderr)
                return 1
            print("  usable: the segmenter sees water throughout the run.")

            # GROUND TRUTH. These cameras sit on instrumented gages, so the
            # coverage signal can be checked against a measured water level —
            # something no other footage in this project offers. The number is
            # printed whatever it is: it qualifies the trend claim rather than
            # automatically supporting it.
            if args.gage_site:
                r = validate_against_gage(args, paths, coverage)
                if r is not None:
                    print(f"gage validation: coverage vs measured height "
                          f"r = {r:+.3f} over {len(paths)} frames")
    except ImportError:
        print("(skipping the coverage check — detector deps unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
