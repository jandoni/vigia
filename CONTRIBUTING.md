# Contributing to VIGÍA

VIGÍA is looking for people to carry it forward. I built it between 2025 and
2026 as an independent project and I no longer have the time to develop it.
This file is the handover I would want if I were picking it up cold: how the
project thinks, how to get it running, how to make the common changes, and
what is still open.

If you want to take on something substantial, or to co-maintain, open an issue
and say what you have in mind. Small fixes can go straight to a pull request.

— Jon Andoni Baranda ([@jandoni](https://github.com/jandoni))

---

## Contents

- [The rules this project runs on](#the-rules-this-project-runs-on)
- [Getting it running](#getting-it-running)
- [Where things live](#where-things-live)
- [Common changes](#common-changes)
- [Open problems](#open-problems)
- [Known rough edges](#known-rough-edges)
- [Pull request checklist](#pull-request-checklist)

---

## The rules this project runs on

These are not style preferences. Each exists because breaking it caused a real
problem, and most are enforced by a check that fails the build.

**1. Measured and reported numbers never mix.** A figure is either *measured*
by this project (and the harness that produced it is named) or *reported by
the original authors*. `models/REGISTRY.yaml` keeps them in separate fields.
Only measured numbers are published.

**2. Numbers are generated, not typed.** Every published figure lives in a JSON
file under `eval/results/`. `RESULTS.md` is checked against those files by
`tools/freeze_results.py`, and the paper never contains a literal number: it
uses macros that `docs/build_paper.py` generates from the same files. This has
already caught two errors that prose had carried for weeks.

**3. The runtime is a licence perimeter.** The project is Apache-2.0. Several
source models were trained with Ultralytics tooling, which is AGPL-3.0. So
every model runs through ONNX Runtime, and `.pt -> .onnx` conversion is a build
step in a separate environment. `tools/check_licence.py` parses everything
under `vigia/` and fails on a forbidden import. Weights are never committed.

**4. Privacy is architectural.** No face, gait or re-identification code path
may exist anywhere in the repository. `tools/check_privacy.py` scans `vigia/`,
`tools/`, `scripts/` and `eval/` and fails if it finds one. Person boxes are
blurred inside themselves before an evidence frame is written, the alert type
has no video field (a test asserts it), and the retention policy raises if raw
video storage is requested.

**5. The validator stays hazard-agnostic.** `vigia/validator/cascade.py` is
configured per hazard, never specialised. A test parses it and fails if any
comparison against a hazard name appears. Anything hazard-specific belongs in
`ValidatorConfig` or in the detector.

**6. Failures get written up.** When something turns out wrong, record it
rather than quietly fixing it. The paper has a dozen "recorded" boxes
describing mistakes found and corrected, and they are its strongest evidence.
Small samples carry Wilson confidence intervals.

---

## Getting it running

Reference hardware for every published number: a MacBook Pro (M4 Pro, 48 GB),
CPU execution provider. Anything with Python 3.11+ will run the demo.

### Two environments, on purpose

| environment | contents | used for |
|---|---|---|
| `.venv` | onnxruntime, numpy, opencv, PyYAML (+ FastAPI for the web view) | everything at runtime |
| `.venv-export` | torch, torchvision, ultralytics | training and `.pt -> .onnx` export only |

```bash
# runtime
python -m venv .venv
.venv/bin/pip install -e '.[web]'

# build-only, if you will train or export models
python -m venv .venv-export
.venv-export/bin/pip install torch torchvision ultralytics
```

### Models, data and clips

None of these are in the repository.

```bash
.venv/bin/python tools/fetch_models.py           # all models, from their distributors
.venv/bin/python tools/fetch_models.py fire      # or one hazard
```

Evaluation datasets are fetched by the scripts in `eval/` (`fetch_figlib.py`,
`fetch_pyro_sdis_val.py`, `fetch_seadronessee_sequence.py`,
`fetch_usgs_camera.py`). `docs/DATA_ACQUISITION.md` lists what each harness
needs, the licence test every candidate dataset must pass, and the leads for
footage the project still lacks. Demo clips are built from local data by
`make clips`; `clips/MANIFEST.json` records each clip's licence and whether it
may be published.

### Check that it works

```bash
make check        # licence guard, privacy guard, results freeze, all tests
make dev          # demo UI at http://127.0.0.1:8000
```

`make check` should print three passing guards and every test file passing.
In the demo, if a live camera says it published no recent image, that is the
network or the camera being offline: `vigia/io/live.py` degrades rather than
raising.

---

## Where things live

The repository map is in the [README](README.md#repository). The files you
will touch most:

| file | what it is |
|---|---|
| `vigia/registry.py` | Tier 0. Contexts, viewpoints, which hazards each context enables (`CONTEXT_HAZARDS`) |
| `vigia/pipeline.py` | The three tiers joined. `build_detector()` is the detector factory |
| `vigia/detectors/base.py` | `ONNXDetector`, the shared base, and the REFERENCE vs FAST backends |
| `vigia/validator/cascade.py` | Tier 2. Read its docstring before changing anything |
| `vigia/io/live.py` | The curated catalogue of public live cameras, with the reasoning for each |
| `vigia/web/server.py` | FastAPI app; `static/demo.html` is the public UI, `static/index.html` the operator view |
| `models/REGISTRY.yaml` | Provenance, licence, reported and measured metrics for every model |
| `configs/cameras.*.yaml` | Camera registries: `demo` for `make dev`, `example` for documentation |
| `eval/results/*.json` | The only source of published numbers |

---

## Common changes

### Add a camera

Add an entry to a registry in `configs/`:

```yaml
- camera_id: river_02
  name: Riverside gauge, upstream
  source: rtsp://...           # a file path, RTSP URL, webcam index or still URL
  context: river               # decides which detectors run
  notes: why this camera, and anything unusual about it
```

The context decides the hazards (see the table at the top of
`configs/cameras.example.yaml`) and implies a viewpoint. If the viewpoint is
outside a detector's envelope, the pipeline refuses the pairing at start-up and
says why. That is intended. For a public live camera, add it to the catalogue
in `vigia/io/live.py` with its licence and polling interval, and check its
terms permit this use.

### Add or replace a detector

1. Put the detector in `vigia/detectors/<hazard>.py`, subclassing
   `ONNXDetector`. Declare `viewpoints` (the envelope you can defend) and merge
   the model's classes down to the single alarm class the validator should see.
2. Register it in `build_detector()` in `vigia/pipeline.py` and, for a new
   hazard, in `Hazard` (`vigia/types.py`) and `CONTEXT_HAZARDS`.
3. Add the model to `models/REGISTRY.yaml`: source, download URL, licence, any
   licence complication in `licence_note`, authors, training data, the authors'
   own metrics under `reported`, and `measured: null` until you measure it.
4. If the weights are PyTorch, export them in `.venv-export` (see
   `scripts/export_*_onnx.py` and `tools/export_onnx.py`). Nothing under
   `vigia/` may import torch or ultralytics.
5. Measure it on held-out data with a harness in `eval/`, write the result to
   `eval/results/`, then fill in `measured`. Run `make check`.

### Publish a new number

Write it to a JSON file in `eval/results/` from a harness that someone else
can rerun. Register it so `tools/freeze_results.py` checks it, add it to
`RESULTS.md`, then `make figures` and `make paper`. Never edit a figure in
`RESULTS.md` or the paper by hand.

### Rebuild the paper

`make paper` regenerates figures and number macros and compiles
`docs/paper/vigia.tex` with [tectonic](https://tectonic-typesetting.github.io)
(`brew install tectonic`). Prose lives in `docs/paper/sections/*.tex`, one
file per section. `make arxiv` produces `dist/vigia-arxiv.tar.gz`.
`make docs` is the older Word build, kept for reference only.

---

## Open problems

In the order I would tackle them. The first three would change what the
project can claim.

1. **No urban flood footage.** The demo uses a creek gauge camera. The story
   this project is built around is the 2024 Valencia DANA, and there is still
   no footage of a flooded street under a licence that permits publication.
   Leads and the licence test are in `docs/DATA_ACQUISITION.md`.
2. **Building access rests on 24 held-out images.** DRespNeT's own split is
   650/50/25 and cannot be widened. A different post-disaster aerial dataset
   with person annotations would fix this.
3. **Building access cannot show temporal validation.** DRespNeT is curated
   stills (the longest continuous run is 7 frames), so the detector is
   exercised but the validator is not. Needs post-disaster aerial *video*.
4. **The road-incident detector has no standalone measurement.** It runs
   behind the validator as a generality experiment, but precision and recall
   for the accident class on held-out labels have not been measured.
5. **The flood gauge correlation is one event on one camera** (r = +0.50).
   More gauge cameras with water-level records would test the trend signal.
6. **The aerial flood segmenter has no held-out split.** FloodNet withheld its
   validation and test masks, so its figures are training-time validation and
   are labelled that way.

Engineering work that would help without new data:

- An alert sink beyond the log (webhook, email, CAP message) behind the single
  egress point in `vigia/io/alerts.py`.
- Packaging for a small edge device, and a measured frames-per-second figure on
  it with the REFERENCE backend.
- A Spanish translation of the demo UI. The audience it was built for is
  Spanish municipalities.

---

## Known rough edges

- `docs/build_documentation.py` (the Word build) is superseded by the LaTeX
  paper and can probably be removed.
- In the operator view (`/operator`), the camera selector opens on the first
  live camera (Asheville), but in one test session the video panel showed a
  frame from a different camera (Red Mountain) under that label. Not yet
  investigated; selecting a camera explicitly shows the right stream.
- The scene router (upload path only) scores 1.000 on 384 held-out images
  because four visually distinct scenes are easy to separate. Treat it as a
  convenience, not a result. It never touches Tier 0.

---

## Pull request checklist

- [ ] `make check` passes locally.
- [ ] No new import of torch or ultralytics under `vigia/`.
- [ ] No face, gait or identity code anywhere.
- [ ] Every new number comes from a file in `eval/results/` and is checked by
      the freeze guard.
- [ ] New models and datasets are recorded in `models/REGISTRY.yaml` and
      credited in `NOTICE`, with their licences.
- [ ] No weights, datasets, clips or third-party imagery committed.
- [ ] If you found a mistake, the write-up says so.
