# VIGÍA

**Gated multi-hazard detection with shared temporal validation for municipal cameras.**

Off-the-shelf hazard detectors are sensitive by design and consequently alarm
constantly. Measured on PyroNear's own validation split, their published fire
detector raises an alarm on **71% of frames containing no smoke**. That is not a
defect in their model — it is the operating point they chose, on the assumption
that something downstream filters the output.

VIGÍA is that something. One temporal validator, configured rather than
specialised, sitting behind five detectors across four disaster hazards.

---

## Try it

Requires Python 3.11+. The runtime is ONNX Runtime only — no PyTorch, no
Ultralytics.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[web]'

make dev
```

That is the whole thing. `make dev` builds any missing clips, loads only the
models the gate actually needs, starts every demonstration camera, serves them
in one operator view and opens your browser at http://127.0.0.1:8000.

Switch cameras from the header selector: the detector chips light up and dim as
you move between them, which is Tier 0 made visible — a forest camera runs only
fire, a UAV camera runs building access and people in water.

Other commands, if you want them:

```bash
make check      # the three guards plus the test suite
make clips      # rebuild the demonstration clips from local data
make paper      # rebuild the technical paper (LaTeX -> docs/paper/vigia.pdf)
make docs       # the superseded Word build, kept for now

# the gate's routing decisions, loading no models at all
.venv/bin/python scripts/run_pipeline.py --explain

# a headless run that reports what the validator did, no browser
.venv/bin/python scripts/run_pipeline.py \
    --registry configs/cameras.demo.yaml --clips clips
```

The operator view is the point. Everyone's demo draws a box around a fire; this
one shows the detections being **rejected**, labelled with the cascade level that
rejected them. Switch off "show suppressed" and it becomes an ordinary detection
demo — the gap between the two views is the contribution.

> Model weights and clips are not in this repository. Fetch them with
> `tools/fetch_models.py` and build clips with `scripts/make_demo_clips.py`.
> Nothing here redistributes third-party weights or imagery.

---

## The three tiers

```
   TIER 0                TIER 1                     TIER 2
   context gate          specialist detectors       temporal validator

 forest ─┐                ┌── fire & smoke ────┐
 street ─┤─► registry ────┼── flood / level ───┤──► confidence      ┌─► CONFIRMED
 river  ─┤   declares     ├── people in water ─┤    colour prior    │   ALERT
 rubble ─┤   which        ├── building access ─┤    persistence  ───┤
 road   ─┘   hazards      └── traffic ─────────┘    cooldown        └─► suppressed
             apply                                                      (logged)
```

**Tier 0 is declarative, not learned.** Each camera declares a context and a
viewpoint; the gate looks up which detectors apply. On the example registry that
is 9 detector invocations per frame instead of 25 — and the unused models are
never loaded at all.

**Tier 2 is the contribution.** Four levels, each independently switchable, so
the ablation table is produced by the same code that runs in production.

---

## Measured results

Every figure below traces to a file in `eval/results/`, checked by
`tools/freeze_results.py`, which fails the build if the summary and the
measurement disagree. Full table in [`RESULTS.md`](RESULTS.md).

| hazard | headline | on |
|---|---|---|
| Fire | false-alarm rate **0.706** (raw detector, the number the validator must beat) | 585 negatives, 21 cameras |
| Flood | water IoU **0.7633** | held-out ATLANTIS test, 1,296 images |
| People in water | swimmer box F1 **0.9172** | SeaDronesSee val, 300 images |
| Building access | civilian box recall **0.942** | held-out DRespNeT test, 24 images |

**On suppression, two numbers rather than one.** Across the demonstration clips,
1,299 proposals became 117 confirmed events. Reported as a single figure that is
91% suppression — but 73.7% is *rejected as not credible* and 17.3% is *withheld
as a repeat of an event already reported*. Those support completely different
claims, and the split varies enormously: fire is 10% filtering and 80%
deduplication; building access is 83% and 11%. Quoting the combined number as a
false-alarm result would overstate the contribution eightfold on the fire clip.

---

## What this project does not claim

- **Temporal confirmation for fire is not novel.** PyroNear already run
  sequential evaluation. The claim is narrower: a *hazard-agnostic* validator,
  measured across four hazards, with a published per-level ablation.
- **The flood trend signal is not novel either.** Choi et al. published
  Page-Hinkley change detection for this problem. We implement both that and a
  sliding-window fit and compare them on identical inputs.
- **Traffic is a generality experiment, not a disaster hazard.** It is retained
  to show the validator transfers, and labelled that way everywhere.
- **Building access rests on 24 held-out images.** The figures are strong; the
  sample is small, and it cannot be widened from released data.
- **Detectors have viewpoints.** The ground flood model reported 0.397 water
  coverage on aerial frames whose water was a narrow canal, tinting mown grass.
  Viewpoint is now declared per camera and per detector, and the pipeline
  *refuses* a pairing outside a model's envelope rather than returning a
  confident wrong number.

---

## Repository

```
vigia/            runtime — ONNX only, no torch, no ultralytics
  registry.py       Tier 0: cameras, contexts, viewpoints, retention
  detectors/        one per hazard, plus the shared ONNX base
  validator/        the cascade and the IoU tracker
  pipeline.py       capture -> gate -> detect -> validate -> emit
  io/               sources and the single egress point
  web/              operator view (FastAPI, vanilla JS, no build step)
eval/             harnesses and results; every published number lives here
scripts/          training, export, clips, demo recording
tools/            guards: licence, privacy, results freeze
docs/paper/       the technical paper — LaTeX source, one file per section
docs/figures/     charts, generated from eval/results as PNG and PDF
models/REGISTRY.yaml   config AND attribution, so the two cannot drift
```

## Guards

Claims worth making are worth enforcing. All three run in CI:

```bash
.venv/bin/python tools/check_licence.py    # no AGPL package in the runtime path
.venv/bin/python tools/check_privacy.py    # no biometric identification, anywhere
.venv/bin/python tools/freeze_results.py   # every published figure matches its source
for t in tests/*.py; do .venv/bin/python "$t"; done
```

`check_privacy.py` is not a policy document: it parses the tree and fails the
build on any face, gait or re-identification code path — in `tools/` and
`scripts/` too, not just the runtime. VIGÍA needs to identify nobody, and being
architecturally incapable of it is cheaper and more credible than being
contractually unwilling.

---

## Licence

Apache-2.0. Model weights and datasets carry their own licences, recorded per
model in `models/REGISTRY.yaml` and credited in [`NOTICE`](NOTICE). No weights
and no third-party imagery are redistributed here.

## Paper and citation

The technical paper — every figure generated from `eval/results/`, built by
`make paper` — lives in `docs/paper/` (`make arxiv` packages it for arXiv).
It includes the control experiment that partially deflates our own headline
(section 6.6), two cascade levels removed on measurement, and a confidence
interval on every proportion. Cite via `CITATION.cff`.
