# VIGÍA — data acquisition plan

**Status: 2 September 2026.** Written offline, to be executed when bandwidth allows.

This is the standing shopping list for footage the project still needs, with the
licence test each candidate must pass and the exact command to run. It exists
because two gaps are material to the submission and both are shortages of
*available material* rather than defects in the system.

Nothing here is downloaded yet. Entries marked **UNVERIFIED** are leads whose
verification was interrupted; they must not be cited as findings until checked.

---

## What is already on disk — do not re-fetch

| dataset | size | status |
|---|---|---|
| `data/pyro_sdis_val_wide` | 1,085 imgs | Fire baseline, 585 negatives. **Complete** |
| `data/usgs_camera` | 50 imgs | Flood, Walnut Creek event, public domain. **Complete** |
| `data/seadronessee/val_seq` | 69 imgs | People-in-water flight sequence, CC0. **Complete** |
| `data/drespnet` | 725 imgs | Building access, CC BY 4.0. **Complete but too small** |
| `data/figlib`, `data/figlib_neg` | 540 imgs | Fire sequences. **Complete** |

---

## The two open gaps

### Gap 1 — Urban flood footage, ideally the 2024 Valencia DANA

The flood demo currently uses a USGS creek gauge camera. It is public domain,
temporal, and shows a real flood — but it is a **rural creek**, not a street
with vehicles and buildings. The DANA is the narrative the whole project is
built around, and no footage of it is in the system.

What would close it: a fixed camera looking at an urban street during a flood,
under a licence permitting publication.

### Gap 2 — Building access: a wider test set, and a sequence

**Updated 2 September 2026 — the DRespNeT route is now definitively closed.**
The authors' figshare record was resolved and inspected: the annotated corpus is
the same 725 images already held, and the raw frames are stills sampled from many
flights rather than video. Both halves of this gap therefore need a DIFFERENT
dataset, not a better extraction from this one.

`building_access` metrics rest on **24 held-out test images**. That is the last
binding sample-size caveat in the project, and unlike the fire negative set it
cannot be widened from what has been released. There is also no temporal clip:
recovering the capture order from the filenames restores the authors' ordering
but not continuity, because the frames were sampled across many flights.

What would close it: a DIFFERENT post-disaster aerial dataset with people
annotated — ideally one distributed as video. The "larger corpus" the paper
alludes to does not exist in released form; that has now been checked.

---

## Acceptance test — apply to every candidate

A source is only useful if it passes all four. Most fail on the first.

1. **Licence permits redistribution.** CC0, CC BY, CC BY-SA, public domain, or a
   government open-data licence. Record the exact string. "Free to view",
   "research use only", and "all rights reserved" all fail — a frame has to be
   publishable in a recorded demonstration.
2. **Camera perspective.** CCTV, dashcam, handheld or UAV. Satellite fails: this
   is a camera system, and that was the stated reason for rejecting xView2.
3. **Temporal.** Video, or a timestamped series from one continuous camera. This
   is the property that matters most and the one most often absent — see
   `docs/` §6.5 for the clip that played perfectly and meant nothing.
4. **A download route that has actually been exercised**, not merely believed to
   exist.

---

## Candidate leads

Verification was interrupted by connectivity. Status is recorded honestly.

### For Gap 1 — flood

| # | lead | why it might work | status | next action |
|---|---|---|---|---|
| 1 | **Wikimedia Commons**, 2024 Spain floods categories | CC-licensed and directly downloadable; the only realistic route to actual DANA imagery | **PARTIAL** — 24 candidate URLs returned HTTP 200, none licence-checked or inspected | Pull the category listing via the MediaWiki API, read `LicenseShortName` per file, keep only CC0/CC BY/CC BY-SA, then check which are video or burst sequences rather than single photos |
| 2 | **Flow Photo Explorer** | River-camera archive paired with streamflow — the same shape as the USGS route that already works | **UNVERIFIED** | Find the project's data/API page, confirm licence, check whether images are timestamped per site |
| 3 | **Tewkesbury river-camera dataset (UK), CC BY** | An actual CC BY river camera dataset with flood coverage | **UNVERIFIED** | Locate the record (Zenodo/figshare likely), confirm CC BY, check frame interval and whether it spans a flood event |
| 4 | **Taiwan WRA** flood/river camera data | Reported to exist as an open dataset | **UNVERIFIED** | Confirm the portal, licence terms in English, and whether historical frames are retrievable |
| 5 | **PNOA** (Spanish national aerial orthophoto), WMS | `AccessConstraints: CC BY 4.0` seen on the provisional WMS — genuine Spanish open data | **PARTIAL** — licence string seen, imagery not retrieved | Low priority: orthophoto is top-down and non-temporal, so it fails test 2 and 3. Useful for a dossier *figure* of DANA extent, not for the detector |
| 6 | **Rijkswaterstaat (NL)** river cameras | Considered as an EU equivalent of USGS | **CLOSED** | Cameras are operated by a third party (INMOVES); not open data. Do not pursue |

### Verified 2 September 2026 — direct checks

A structured sweep of the Hugging Face dataset index (14 broad terms, licence
tags read from the API rather than from prose) found **909 relevant datasets, of
which 339 carry a permissive licence**. The full list is in
`scratchpad/hf_good.json`. Two were run to ground:

| dataset | verdict |
|---|---|
| `ai-for-good-lab/ai4g-flood-dataset` (MIT, 116k downloads) | **RULED OUT.** GeoTIFF tiles on a lat/long grid — satellite, not camera. Fails the modality test for the same reason xView2 did |
| `takara-ai/FloodNet_2021-Track_2_Dataset_HF` (CC BY-SA 4.0) | **VIABLE DATA, WRONG VIEWPOINT — see below** |

**FloodNet is the most interesting negative result so far.** It is 2,348 UAV
images of Hurricane Harvey flooding, CC BY-SA 4.0, and genuinely temporal: the
longest consecutive-id run is 126 frames, and continuity was verified against an
out-of-sequence baseline at adjacent 0.954 versus unrelated 0.282. On paper it
closes the urban-flood gap.

It does not, because **our flood segmenter does not work on nadir aerial
imagery**. Measured on the fetched run, it reports up to 0.397 water coverage on
frames whose actual water is a narrow canal, tinting large areas of mown grass
as water. The segmenter was trained on ATLANTIS, which is ground-level and
oblique photography of water bodies; a straight-down view from a few hundred
metres is a different sensing problem and the model has never seen it.

That is an envelope finding worth more than the clip would have been: the flood
detector's declared envelope should say ground-level and oblique views, and a
nadir aerial capability would require fine-tuning on aerial data rather than
being assumed. FloodNet Track 1 ships segmentation masks and would allow that
gap to be measured properly rather than described.

**One licence nuance to carry:** FloodNet is CC BY-**SA** 4.0. Share-alike means
a demo video containing its frames would itself have to be released under CC
BY-SA. Every other source in the project is CC BY, CC0 or public domain, none of
which impose that. Usable, but it constrains the artefact rather than just
requiring a credit line.

**Fallback that is already proven:** the USGS route works and scales. There are
1,311 cameras and a gage-height API to find events; a scan of 500 gages for
warm-season rises with dense imagery was running when connectivity dropped.
Re-running it is the lowest-risk way to get more urban flood footage, since some
USGS cameras sit on urban creeks (Walnut Creek at Raleigh already does).

### For Gap 2 — building access

| # | lead | why it might work | status | next action |
|---|---|---|---|---|
| 1 | **DRespNeT on figshare** — `10.6084/m9.figshare.29991478.v2` | Held the raw full-resolution frames | **RESOLVED — CLOSED.** CC BY 4.0 confirmed. The annotated release is the SAME 650/50/25 split already on disk, so it does not widen the test set. The 615 raw frames are curated stills from many flights, not footage: 362 scene cuts, longest continuous run 7 frames | Do not re-download. The 1.3 GB 1920×1080 archive is the same 615 frames at higher resolution and would not change either conclusion |
#### Verified 2 September 2026 — licences read from source, downloads probed

**Passing all four tests.** Every download below was confirmed with a real HTTP
range request, and every licence string was read from the distributor.

| dataset | contents | people | sequential | licence | size |
|---|---|---|---|---|---|
| **WiSARD** (Univ. of Washington) | 26,862 visual + 29,989 thermal labelled, 15,453 synced pairs; raw video shipped | YOLO boxes, single class | **Yes** — 30 Hz video, every 6th frame kept | **MIT** | 43.5 GB |
| **Small-Object Aerial Person Detection** (Univ. of Cyprus) | 3,136 images / 61,708 person boxes; **test split 521 images / 10,432 boxes**; Civil Defence exercise footage | YOLO + COCO + VOC | not stated | **CC BY 4.0** (Zenodo 7740081) | 3.7 GB |
| **MOBDrone** | 126,170 frames from 66 Full HD clips, 113k+ person boxes, man-overboard scenario | COCO | **Yes** — 30 FPS | **CC BY 4.0** (Zenodo 5996890) | 5.4 GB video / 243 GB images |
| **HERIDAL** | 1,685 JPEG / 1,651 VOC XML wilderness stills | VOC boxes | No | **CC BY 3.0 Unported** | 8.3 GB (mirror) |
| **DeSARD** | 7,056 real UAV images / 3,420 human boxes, altitude 20–95 m | boxes | not stated | CC BY-SA 4.0 | 11 GB |

**A correction worth carrying: HERIDAL is NOT academic-use-only**, contrary to
how it is usually described. The IPSAR page grants use "for commercial,
scientific and educational purposes" under CC BY 3.0 Unported. Its official host
is dead (connection refused on both ports); the licence was recovered from the
Internet Archive and a Zenodo mirror serves the data (DOI 10.5281/zenodo.5662351).

**Ruled out, with the reason.** These are the useful negatives:

| dataset | why not |
|---|---|
| **NOMAD** | Technically the best match found — 42,825 sequential frames, 100 actors, boxes plus ten graded visibility levels. **No licence exists anywhere**: no LICENSE file, GitHub API returns null, README silent. Unstated = unusable. The one candidate where an email to the authors could change the ranking |
| **SARD** | Behind an IEEE DataPort subscription. No licence published |
| **VisDrone**, **Okutama-Action** | CC BY-**NC**-SA — non-commercial, therefore unusable |
| **UAVDT** | No authoritative licence from the original authors; CC BY 4.0 claims are re-uploader assertions |
| **ForestPersons** | 96,482 images and 377 sequences, but research-only licence AND gated |
| **C2A** | Kaggle metadata claims MIT and the download works — but it is synthetic composites of human crops pasted onto **AIDER** backgrounds, and AIDER was scraped from image search and news sites. The MIT grant is an uploader asserting rights over third-party press imagery. Do not publish frames from it |
| **AIDER** | Genuinely CC BY 4.0, but classification-only — no person boxes — and the same scraped provenance |
| **UAVs-TEBDE** (2023 Türkiye) | Building damage classes only, no people |
| Thermal SAR sets (Zenodo 4349220, 21515906) | CC BY 4.0 and well documented, but thermal/optical-sectioning imagery, not RGB camera frames |

**No AFAD or Turkish-university release of annotated post-earthquake UAV imagery
with people exists under any open licence.** DRespNeT remains the only such
corpus, and it is exhausted. That is now a searched-and-confirmed negative
rather than an assumption.

#### What this does and does not solve

None of the datasets above is post-earthquake imagery, so **none widens the
24-image DRespNeT test set in-domain**. What they offer instead is arguably more
useful: an OUT-OF-DOMAIN test of whether civilian box recall of 0.942 means
anything beyond the distribution it was trained on. The Cyprus set is the best
candidate — CC BY 4.0, a 521-image test split with 10,432 person boxes, and
Civil Defence exercise footage, which is the closest published thing to a
disaster-response drill.

WiSARD and MOBDrone are the sequential options, and would let the temporal
validator be exercised on aerial people footage — though on terrain and water
respectively, not rubble, so each would be a cross-domain demonstration and must
be labelled as one.

---

## Ready-to-run commands

When bandwidth returns, in this order. Each is bounded and re-runnable.

```bash
# 1. Widen flood coverage: find USGS flood events with dense imagery.
#    Metadata only — no images. Safe on a weak link.
python eval/fetch_usgs_camera.py --search "creek" --list
python eval/fetch_usgs_camera.py --search "river at" --list

# 2. Fetch one event once a camera is chosen (~40 images, a few MB).
python eval/fetch_usgs_camera.py --camera <CAM_ID> \
    --days YYYY-MM-DD YYYY-MM-DD --gage-site <NWIS_ID> --frames 60

# 3. Rebuild clips and re-render the demo from whatever is on disk.
python scripts/make_demo_clips.py
python scripts/record_demo.py

# 4. Re-verify the whole chain.
python tools/check_licence.py && python tools/check_privacy.py \
  && python tools/freeze_results.py
```

For the figshare DRespNeT lead, resolve and inspect the record **before**
downloading anything: the point is to learn whether a larger corpus exists, and
that is answerable from the file listing alone.

---

## A note on why this list is short

Most candidates fail on licence, and they fail quietly. A dataset that says
"freely available for research" is not publishable, and the difference only
surfaces when someone reads the terms rather than the download button. The two
sources that carried this project — USGS (public domain) and SeaDronesSee
(CC0) — were both found by checking the licence first and the content second.
That ordering is the main lesson worth carrying into the next search.
