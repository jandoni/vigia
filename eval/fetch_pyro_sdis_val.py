#!/usr/bin/env python3
"""Widen the fire negative set — the binding caveat on the project's headline.

    python eval/fetch_pyro_sdis_val.py
    python eval/fetch_pyro_sdis_val.py --negatives 548 --positives 500

WHY THIS EXISTS. The single most quoted number in this project is that the
published fire detector alarms on 87% of frames containing no smoke. That is
the figure the temporal validator exists to destroy, and it was measured on
**23 negative frames**. The registry has carried the caveat "indicative, not
precise — widen the negative set before publishing" ever since, because a
false-alarm rate estimated from 23 samples has a confidence interval wide
enough to drive a truck through.

The pyro-sdis validation split holds 4,099 rows, of which roughly 550 are
negatives drawn from 21 different cameras — around 24 times the sample the
figure currently rests on, and spread across many more scenes. Rows are read
through the Hugging Face datasets server rather than by pulling the 390 MB
parquet, and a row counts as a negative when its annotation field is empty.

Licence: pyro-sdis is published by the PyroNear association, the same source as
the detector itself. Used here for evaluation, and no image is redistributed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=pyronear%2Fpyro-sdis&config=default&split=val")
DEFAULT_OUT = REPO_ROOT / "data" / "pyro_sdis_val_wide"
TOTAL_ROWS = 4099


def scan(limit: int) -> list[dict]:
    """Row metadata: name, annotations, image URL. No images fetched."""
    rows: list[dict] = []
    offset = 0
    while offset < limit:
        page = None
        for _ in range(4):
            try:
                with urllib.request.urlopen(
                        f"{ROWS_API}&offset={offset}&length=100", timeout=45) as r:
                    page = json.load(r)
                break
            except Exception:
                time.sleep(3)
        if page is None:
            print(f"  stopped scanning at offset {offset} (server unavailable)",
                  file=sys.stderr)
            break
        for item in page["rows"]:
            row = item["row"]
            rows.append({
                "name": row["image_name"],
                "annotations": (row["annotations"] or "").strip(),
                "src": row["image"]["src"],
                "camera": row["camera"],
            })
        offset += 100
    return rows


def save(rows: list[dict], out: Path) -> int:
    """Write image plus a YOLO label file, so the existing harness just works."""
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        image_path = out / row["name"]
        label_path = image_path.with_suffix(".txt")
        if not image_path.exists():
            try:
                urllib.request.urlretrieve(row["src"], image_path)
            except Exception as exc:
                print(f"  failed {row['name']}: {exc}", file=sys.stderr)
                continue
        # An empty label file is what marks a negative for the harness.
        label_path.write_text(
            (row["annotations"] + "\n") if row["annotations"] else "",
            encoding="utf-8")
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scan", type=int, default=TOTAL_ROWS)
    parser.add_argument("--negatives", type=int, default=600)
    parser.add_argument("--positives", type=int, default=500)
    args = parser.parse_args()

    print(f"scanning up to {args.scan} val rows …")
    rows = scan(args.scan)
    negatives = [r for r in rows if not r["annotations"]]
    positives = [r for r in rows if r["annotations"]]
    print(f"  {len(rows)} rows: {len(negatives)} negative, {len(positives)} positive")
    print(f"  negatives span {len({r['camera'] for r in negatives})} distinct cameras")

    take_neg = negatives[: args.negatives]
    take_pos = positives[: args.positives]
    print(f"fetching {len(take_neg)} negatives and {len(take_pos)} positives …")
    written = save(take_neg + take_pos, args.out)
    print(f"wrote {written} image/label pairs to {args.out}")

    on_disk = sorted(args.out.glob("*.jpg"))
    empty = [p for p in on_disk
             if not p.with_suffix(".txt").read_text(encoding="utf-8").strip()]
    print(f"on disk: {len(on_disk)} images, {len(empty)} negatives")
    if len(empty) < 100:
        print("  WARNING: fewer than 100 negatives — the false-alarm rate will "
              "still be\n  indicative rather than precise.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
