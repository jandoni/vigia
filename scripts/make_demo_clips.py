#!/usr/bin/env python3
"""Build one deterministic demo clip per hazard, with its licence attached.

    python scripts/make_demo_clips.py
    python scripts/make_demo_clips.py --only fire flood

The stage path depends on curated local clips: never rely on a network at
presentation time, and never rely on a model finding something interesting in
whatever footage happens to be to hand. Frames are taken in sorted order at a
fixed frame rate, so a clip built twice is byte-identical and a run over it
produces the same output every time.

LICENSING IS THE POINT OF THIS SCRIPT, not a footnote to it. The source
datasets carry four different licences, and one of them forbids redistribution
outright. A demo video is a published work: putting all-rights-reserved footage
into it would be a licence breach committed on stage, in front of the people
most likely to notice. Every clip is therefore written alongside a manifest
recording where its frames came from, under what terms, and whether it may
appear in anything public.

The clips themselves are never committed — they are derived from datasets we do
not redistribute. This script regenerates them from data already on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "clips"


@dataclass
class ClipSpec:
    """One curated clip: where the frames come from and what may be done with them."""

    name: str
    hazard: str
    context: str
    source_glob: str
    licence: str
    attribution: str
    publishable: bool
    why: str
    #: Whether the frames are a genuine time sequence from ONE camera.
    #: This decides whether a clip may be used to demonstrate the temporal
    #: validator at all. Stitching unrelated still images into a video makes a
    #: file that plays, and the validator's output over it is meaningless in
    #: both directions: persistence rejects almost everything because nothing
    #: genuinely persists, while a few detections confirm anyway because
    #: unrelated objects happen to overlap spatially between consecutive
    #: frames and the IoU tracker associates them. Measured: the SeaDronesSee
    #: still-image clip produced 121 persistence rejections and 16
    #: confirmations, none of which mean anything.
    temporal: bool = True
    temporal_note: str = ""
    #: When set, frames are selected by the ORIGINAL capture index embedded in
    #: the filename (e.g. "image111_...") rather than by sorted filename. Some
    #: datasets are published as shuffled, renamed stills that nonetheless came
    #: from continuous footage; recovering the capture order turns them back
    #: into a sequence. Sorting such a dataset by filename produces a video
    #: whose frames are in an arbitrary order — it plays, and means nothing.
    #: A clip kept for local rehearsal that is NOT the hazard's demo source.
    #: Excluded from the hazard-coverage check, which asks whether each hazard
    #: has something showable — not whether every file on disk is showable.
    reference_only: bool = False
    index_pattern: str = ""
    index_start: int = 0
    fps: float = 8.0
    max_frames: int = 60
    stride: int = 1
    sequences: list[str] = field(default_factory=list)


#: One per hazard. `publishable` is the field that decides whether a clip may
#: appear in the demo video, a figure, or a public repository.
SPECS: list[ClipSpec] = [
    ClipSpec(
        name="fire_ridge",
        hazard="fire", context="forest",
        source_glob="data/figlib/20160604_FIRE_rm-n-mobo-c/*.jpg",
        licence="HPWREN imagery — attribution required",
        attribution="HPWREN, University of California San Diego "
                    "(https://hpwren.ucsd.edu). FIgLib fire ignition sequence.",
        publishable=True,
        why="HPWREN requires attribution when its images are published. That is "
            "a condition we can meet, so this clip may be shown provided the "
            "credit appears on the frame or in the caption.",
    ),
    ClipSpec(
        name="creek_gauge",
        hazard="flood", context="river",
        source_glob="data/usgs_camera/*.jpg",
        licence="Public domain — work of the United States Government",
        attribution="USGS streamgage camera, Walnut Creek at South State "
                    "Street, Raleigh NC (station 0208735460). Public domain.",
        publishable=True,
        why="USGS imagery is a work of the United States Government and carries "
            "no restriction. It fully replaces V-FloodNet's LSU sequences, which "
            "are all rights reserved and could not appear in a recorded "
            "demonstration or a published figure — that restriction is why flood, "
            "the keystone hazard, previously had no showable footage at all. "
            "The USGS cameras are the right regime as well as the right "
            "licence: fixed mount, pointed at water, sampled sparsely, which is "
            "what the trend detection was calibrated for. They add something no "
            "other footage in this project has — an instrumented gage reading "
            "alongside every frame, so the segmented coverage can be checked "
            "against a measured water level instead of being trusted.",
        temporal=True,
        temporal_note="A real flood, not a stable creek: 8 August 2024 at "
                      "Raleigh, where the gage rose to 9.00 ft and receded to "
                      "4.25 ft over the two days covered, sampled about every "
                      "20 minutes in daylight. Daylight only, because these "
                      "cameras switch to infrared at night and the segmenter "
                      "then tints sky and vegetation as water; night is outside "
                      "the detector's declared envelope and the fetch script "
                      "enforces that rather than restating it.",
        fps=4.0,
        max_frames=40,
    ),
    ClipSpec(
        name="open_water",
        hazard="drowning", context="coast",
        source_glob="data/seadronessee/val_seq/*.jpg",
        licence="CC0 1.0 Universal",
        attribution="SeaDronesSee — Varga, Kiefer et al., University of "
                    "Tübingen, IEEE/CVF WACV 2022. CC0; credited by choice.",
        publishable=True,
        why="CC0 places no restriction of any kind. The least encumbered "
            "footage in the project and the safest thing to put on a screen.",
        temporal=True,
        temporal_note="A genuine 69-frame flight sequence, fetched by "
                      "eval/fetch_seadronessee_sequence.py. The first attempt "
                      "used the sparse validation sample already on disk, whose "
                      "longest run of consecutive frames is 4 — the resulting "
                      "clip played fine and meant nothing. Continuity was "
                      "verified rather than assumed: adjacent-frame histogram "
                      "correlation is 0.997 here against 0.015 for unrelated "
                      "pairs from the same split.",
        max_frames=69,
    ),
    ClipSpec(
        name="collapsed_building",
        hazard="building_access", context="uav_search",
        source_glob="data/drespnet/*/*.jpg",
        licence="CC BY 4.0",
        attribution="DRespNeT — Cranfield University, arXiv 2508.16016. "
                    "UAV imagery of the 2023 Türkiye earthquakes.",
        publishable=True,
        why="CC BY 4.0 permits redistribution with attribution, which is a "
            "condition we can meet. The credit is legally required here, unlike "
            "SeaDronesSee where it is given by choice.",
        temporal=False,
        temporal_note="NOT a sequence, and an earlier version of this file "
                      "wrongly claimed it was. Ordering by the capture index "
                      "embedded in the filenames does recover the original "
                      "order, and a 12-frame window from image111 looked "
                      "continuous (adjacent correlation 0.695 against 0.235 for "
                      "unrelated pairs), so a 66-frame clip was built on that "
                      "basis. Checking the authors' own un-augmented raw frames "
                      "later showed the generalisation was false: across all "
                      "615 raw frames the median adjacent correlation is 0.388 "
                      "with 362 scene cuts, and the longest genuinely "
                      "continuous run in the whole dataset is SEVEN frames. "
                      "Within the 66-frame range actually used, 40 of 65 "
                      "transitions are scene cuts. DRespNeT is curated stills "
                      "sampled from many flights, not footage, so this hazard "
                      "cannot demonstrate temporal validation on released data.",
        index_pattern=r"image(\d+)_",
        index_start=111,
        max_frames=66,
        fps=4.0,
    ),
]


def build(spec: ClipSpec, out_dir: Path) -> dict | None:
    import cv2

    if spec.index_pattern:
        # Recover capture order from the filename index. The first hash seen
        # for an index wins: Roboflow augmentation emits several variants of
        # the same source frame, and mixing them into a sequence would inject
        # flips and colour shifts between consecutive frames.
        import re
        by_index: dict[int, Path] = {}
        for path in sorted(REPO_ROOT.glob(spec.source_glob)):
            match = re.match(spec.index_pattern, path.name)
            if match:
                by_index.setdefault(int(match.group(1)), path)
        paths = [by_index[i] for i in
                 range(spec.index_start, spec.index_start + spec.max_frames)
                 if i in by_index]
    else:
        paths = sorted(REPO_ROOT.glob(spec.source_glob))[::spec.stride][: spec.max_frames]
    if not paths:
        print(f"  {spec.name:<20} SKIP — no frames at {spec.source_glob}")
        return None

    first = cv2.imread(str(paths[0]))
    if first is None:
        print(f"  {spec.name:<20} SKIP — could not read {paths[0].name}")
        return None
    height, width = first.shape[:2]

    # Cap the long edge: a 4K still sequence makes a clip nobody can play back
    # smoothly, and the detectors resize to their own input size anyway.
    scale = min(1.0, 1280 / max(width, height))
    size = (int(width * scale), int(height * scale))

    target = out_dir / f"{spec.name}.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             spec.fps, size)
    written = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if image.shape[1::-1] != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        writer.write(image)
        written += 1
    writer.release()

    flags = ["publishable" if spec.publishable else "LOCAL REHEARSAL ONLY"]
    if not spec.temporal:
        flags.append("NOT A SEQUENCE")
    print(f"  {spec.name:<20} {written:>3} frames  {size[0]}x{size[1]}  "
          f"{spec.fps:g} fps  [{', '.join(flags)}]")

    return {
        "clip": target.name,
        "hazard": spec.hazard,
        "context": spec.context,
        "frames": written,
        "fps": spec.fps,
        "resolution": f"{size[0]}x{size[1]}",
        "source": spec.source_glob,
        "licence": spec.licence,
        "attribution": spec.attribution,
        "publishable": spec.publishable,
        "why": spec.why,
        "reference_only": spec.reference_only,
        "temporal": spec.temporal,
        "temporal_note": spec.temporal_note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="+", default=None,
                        help="build only these hazards")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in SPECS
             if args.only is None or s.hazard in args.only or s.name in args.only]

    print(f"building {len(specs)} clip(s) into {args.out}\n")
    entries = [entry for entry in (build(spec, args.out) for spec in specs)
               if entry is not None]

    publishable = [e for e in entries if e["publishable"]]
    restricted = [e for e in entries if not e["publishable"]]

    manifest = args.out / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "note": "Clips are derived from datasets VIGÍA does not redistribute. "
                "They are generated locally by scripts/make_demo_clips.py and "
                "must not be committed. `publishable` decides whether a clip "
                "may appear in a recorded demo, a figure, or anything public.",
        "clips": entries,
    }, indent=2), encoding="utf-8")

    print(f"\npublishable ({len(publishable)}): "
          f"{', '.join(e['clip'] for e in publishable) or 'none'}")
    if restricted:
        print(f"RESTRICTED ({len(restricted)}): "
              f"{', '.join(e['clip'] for e in restricted)}")
        for entry in restricted:
            print(f"  {entry['clip']}: {entry['licence']}")
        print("  These may be used to rehearse and to develop against. They "
              "must NOT appear in\n  a recorded demo, a published figure, or any "
              "published material.")

    non_temporal = [e for e in entries if not e["temporal"]]
    if non_temporal:
        print(f"\nNOT TIME SEQUENCES ({len(non_temporal)}): "
              f"{', '.join(e['clip'] for e in non_temporal)}")
        print("  These are unrelated stills stitched into a video. They "
              "exercise the DETECTOR\n  correctly and must NOT be used to "
              "demonstrate the temporal validator: nothing\n  genuinely "
              "persists, so both the rejections and the confirmations are "
              "artefacts.")
        for entry in non_temporal:
            print(f"    {entry['clip']}: {entry['temporal_note']}")

    demo_ready = [e for e in entries
                  if e["publishable"] and e["temporal"] and not e["reference_only"]]
    print(f"\nUSABLE IN THE RECORDED DEMO ({len(demo_ready)}): "
          f"{', '.join(e['clip'] for e in demo_ready) or 'none'}")
    print("  A clip needs BOTH a licence that permits publication AND genuine "
          "temporal\n  continuity before it can carry the validator's argument "
          "on camera.")

    hazards_without_publishable = sorted(
        {e["hazard"] for e in entries if not e["reference_only"]}
        - {e["hazard"] for e in publishable}
    )
    if hazards_without_publishable:
        print(f"\nNO PUBLISHABLE FOOTAGE: {', '.join(hazards_without_publishable)}")
        print("  A hazard with no showable footage cannot appear in the demo "
              "video. Resolve by\n  sourcing permissively licensed footage — "
              "which is also the standing highest-value\n  action in the plan: "
              "get real Spanish footage.")

    print(f"\nwrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
