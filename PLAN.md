# VIGÍA — Build Plan

**Target:** UNESCO–Al Fozan International Prize for the Promotion of Young Scientists in STEM, 2027 edition (Ref. CL/4548)
**Submission deadline:** 30 November 2026, 23:59 UTC+1
**Nominator dossier due:** ~2 November 2026 (Instituto de la Ingeniería de España needs lead time for the Article 6.3 recommendation letter)
**Plan written:** 1 September 2026
**Author:** Jon Andoni Baranda

---

## Status — updated 1 September 2026

**Phase 0: COMPLETE.** **Phase 1: in progress.**

Done:
- Package skeleton, Apache-2.0, NOTICE with full attribution
- `ONNXDetector` base — ONNX Runtime only; AGPL guard (`tools/check_licence.py`) passing
- `tools/fetch_models.py` — working; PyroNear model downloaded and verified
- `tools/make_static.py` — static-shape variant so CoreML can accelerate
- Two-backend policy (`Backend.REFERENCE` / `Backend.FAST`) — see below
- `eval/metrics.py`, `eval/run_image_eval.py` — box + image level, F2 headline
- `vigia/validator/` — IoU tracker and the full five-level cascade
- `tests/test_validator.py` — 8/8 passing

**First measured result (R2 baseline) — REMEASURED on a wider negative set.**
PyroNear detector, as published, on the pyro-sdis val split, conf 0.20,
single-frame, no validation. Originally measured on 100 frames of which only 23
were negative; now on **1,085 frames (500 positive, 585 negative from 21
cameras)**:

| metric | 23 negatives | **585 negatives** |
|---|---|---|
| box precision / recall | 0.625 / 0.904 | **0.447 / 0.902** |
| image-level recall | 0.987 | **0.978** |
| **false-alarm rate** | **0.870** | **0.706** |
| median latency | 69 ms (CPU) | 73 ms (CPU) |

**The headline moved by 16 points, and downwards.** 71% of no-smoke frames still
raise an alarm, so the motivation is untouched — a detector like that is
unusable without a filter, and destroying that number is the validator's job.
But 87% was overstated, and it was overstated because it rested on 23 samples.
Recall barely moved (0.904 -> 0.902), which is the reassuring half: the detector
finds what it finds. Precision fell hard, meaning the small sample had flattered
it on false alarms. Every quotation of 0.870 in this repository has been
corrected. The caveat that sat on this number for the whole project — "indicative,
not precise, widen before publishing" — did precisely the job it existed to do.

**Backend policy (resolved).** CoreML could not run the published graph at all:
the Ultralytics export uses dynamic spatial dimensions and CoreML cannot build
a plan for those. Pinning the input to 1x3x1024x1024 fixes it and gives
**1.7-1.9x** speedup (60 ms -> 34 ms; 18 -> 30 FPS). Detection counts matched
on 100/100 frames and boxes to within 0.41 px, but confidences drift up to
~0.012 in lower precision, which moved box P/R by ~1% at conf 0.10 and 0.30
(zero change at our 0.20 operating point). Therefore:

- `Backend.REFERENCE` — CPU, dynamic graph. **Every published number.**
- `Backend.FAST` — CoreML, static graph. **Everything a human watches.**

**Correction to §1.** PyroNear's training manifest shows they already run
sequential evaluation (`predict_sequential.py`, `optimize_sequential.py`, a
grid search over `--min-frames 8`). Temporal confirmation for fire is therefore
**not** novel. Our claim must be the narrower and still-true one: a
*hazard-agnostic* validator, measured *across four hazards*, with a published
per-level ablation. Found this ourselves rather than having a juror find it.

**R1, first run — and a methodological trap avoided.** Built `eval/fetch_figlib.py`
(HPWREN FIgLib: 195 fire ignition sequences, one frame per minute, ground truth
in the filename offset) and `eval/run_sequence_eval.py`. Six fires, 120
pre-ignition and 120 post-ignition frames:

| configuration | FAR | recall | fires found | false alerts | min to detect |
|---|---|---|---|---|---|
| raw detector | 0.008 | 0.925 | 5/6 | 1 | 1.0 |
| validator, persistence=2 | 0.000 | 0.858 | **6/6** | 0 | 2.0 |
| validator, persistence=3 | 0.000 | 0.792 | **6/6** | 0 | 3.0 |
| validator, persistence=4 | 0.000 | 0.733 | **6/6** | 0 | 4.0 |

The validator finds **more** fires than the raw detector (6/6 vs 5/6) and removes
the only false alarm, at a cost of 1-3 extra minutes of latency and some
per-frame recall. That trade is the real finding and both halves must be
reported.

**But the headline false-alarm claim CANNOT be made on this data.** The raw
detector's FAR on FIgLib pre-ignition frames is **0.8%**, against **71%** on the
pyro-sdis negatives. The difference is negative-set difficulty: FIgLib
pre-ignition frames are the same camera shortly before a fire, usually a clear
scene, whereas pyro-sdis negatives are curated hard negatives - cloud, fog,
dust. On easy negatives the validator has nothing to remove.

Two traps avoided, both worth recording because a reviewer would have found them:

1. **Cooldown confounded the recall metric.** With cooldown on, recall appeared
   to collapse from 0.925 to 0.125 - which reads as the validator destroying the
   detector. In fact it was cooldown correctly suppressing repeat alerts about a
   fire already reported, while the validator found *more* fires. Cooldown is now
   off by default in the harness and evaluated separately.
2. **A URL-encoding bug silently dropped every positive frame.** FIgLib
   post-ignition filenames contain a literal `+`, which a URL path reads as a
   space; the CDN 403s. The first download produced 120 negatives and zero
   positives, and would have yielded a false-alarm measurement with nothing to
   detect. Fixed with `quote(filename, safe="")`.

**R1 COMPLETE — headline result.** The HPWREN per-minute archive is offline
(`c1.hpwren.ucsd.edu` does not resolve), so the negative set was built from
FIgLib instead: the 10 earliest pre-ignition frames from each of 30 different
fires, i.e. 30 distinct camera / date / weather combinations, each a 10-minute
consecutive run sampled 30-40 minutes before ignition. Taking the *earliest*
frames matters — FIgLib ignition timestamps are approximate, so frames just
before t=0 may already contain faint smoke, and labelling those as negatives
would manufacture false positives that are really correct detections.

Detections are cached once (`eval/cache_detections.py`) and replayed through
every validator configuration (`eval/run_ablation.py`), so each configuration
sees byte-identical detector output and any difference is caused by the
validator alone.

Negatives: 30 sequences / 300 frames. Fires: 6 sequences / 240 frames. conf 0.20.

| configuration | FAR | reduction | fires found | recall | min to detect |
|---|---|---|---|---|---|
| raw detector | 0.040 | — | 6/6 | 0.925 | 1.0 |
| cascade, persistence=2 | 0.020 | 50% | 6/6 | 0.858 | 2.0 |
| **cascade, persistence=3** | **0.010** | **75%** | **6/6** | 0.792 | 3.0 |
| cascade, persistence=4 | 0.007 | 83% | 6/6 | 0.733 | 4.0 |
| cascade, persistence=5 | 0.003 | 92% | 6/6 | 0.675 | 5.1 |

**75% fewer false alarms at zero cost in fires detected, for two extra minutes
of latency.** Every configuration still found all six fires. The trade is
latency, not capability, and both halves are reported.

**Per-level ablation (persistence=3, one level removed at a time):**

| removed | FAR | verdict |
|---|---|---|
| minus confidence | 0.137 | **Doing most of the work** — 13.7x worse without it |
| minus persistence | 0.040 | **Doing the rest** — removing it returns FAR to baseline |
| minus size | 0.010 | **Contributes nothing on this data** |

Two of five levels carry the result. Size plausibility earns its place on
exactly zero frames here and must either be tuned or dropped — reporting five
levels when two do the work would be padding. Colour remains disabled and
excluded from the claim.

**Caveats to carry into the dossier:** 300 negative frames is still a small
sample, and FIgLib pre-ignition frames are mid-difficulty negatives — the same
detector scores FAR 0.71 on pyro-sdis curated hard negatives. State which
negative set every number came from, every time.

**Traffic detector integrated — the hazard-agnostic claim now has evidence.**

`Enos-123/traffic-accident-detection-yolo11x` (MIT) ships as `.pt`, so
`tools/export_onnx.py` converts it in a **separate** `.venv-export` environment
that contains ultralytics; the runtime venv still has no ultralytics and no
torch. Both published checkpoints were exported and compared on the author's own
test images: `epoch61` is better behaved (one accident per image on all four,
conf 0.58-0.70) than `epoch14` (three detections on fig1). Note the upstream
`infer.py` points at epoch14 — we ship epoch61 and say why. Classes verified at
export: `{0: accident, 1: vehicle}`. Only `accident` reaches the validator;
routing `vehicle` would make every car an alert.

Ran the identical `TemporalValidator` over both hazards
(`eval/run_multihazard.py`):

| | wildfire smoke | traffic accidents |
|---|---|---|
| detector | PyroNear yolo11s (Apache-2.0) | Enos yolo11x (MIT) |
| source | FIgLib, 1 frame/min | dashcam video, 2 fps |
| frames | 240 | 660 |
| raw detections | 109 | 291 |
| confirmed events | 15 | 33 |
| **suppression** | **86.2%** | **88.7%** |

Same class, same `cascade.py`, two detectors trained by different authors on
unrelated data. The only per-hazard difference is a `ValidatorConfig`.

This is now enforced, not merely asserted:
`test_validator_contains_no_hazard_specific_code` parses `cascade.py` and fails
the build if any comparison against a specific hazard appears. Someone fixing a
fire bug with `if hazard == Hazard.FIRE` can no longer quietly invalidate the
dossier's central claim. **9/9 tests passing.**

**The size level is now confirmed dead.** It rejected 0% of detections on
wildfire AND 0% on traffic. Two hazards, zero contribution. Either tune it or
drop it — reporting a five-level cascade when two levels do the work is padding.

**Hazard 1 — flood (ATLANTIS) — MEASURED. The keystone, and it works.**

Trained by us: torchvision DeepLabV3 with a MobileNetV3-Large backbone,
fine-tuned on ATLANTIS (Erfani et al., University of South Carolina), 56 classes
collapsed to binary water. torchvision is BSD-3-Clause, so this was the first
detector in VIGÍA entirely free of the AGPL perimeter — that was the reason for
choosing it over YOLO-seg, not an accident.

Held-out ATLANTIS test split, 1,296 images never seen:

| metric | value |
|---|---|
| water IoU | **0.7633** |
| precision / recall | 0.8799 / 0.852 |
| F1 | 0.8657 |
| pixel accuracy | 0.9319 |
| median latency | 21.8 ms |

Validation IoU was 0.7769, so the test gap is about 1.4 points — small, and
evidence the model is not overfitted to the validation split.

**The trend signal is the point, and it is NOT novel.** Segmenting water is
solved; knowing the water is *rising* is the signal the flood literature says is
missing. Choi, Kim, Win Aung and Park published exactly this capability for this
problem (Page-Hinkley change detection on CCTV, *Developments in the Built
Environment* 25, 2026, article 100866). We implement both their Page-Hinkley
approach and a sliding-window least-squares fit and compare them on identical
inputs, rather than claiming the idea.

Three LSU creek sequences from V-FloodNet, 80 frames, 11 minutes to 7 hours. The
two methods win on different sampling regimes: the windowed fit is faster on
densely sampled cameras, Page-Hinkley on sparsely sampled ones because it is
sample-based rather than time-based. **A fixed 300 s window originally caused the
fit to fail completely on the 15-minute-interval camera** — reporting UNKNOWN for
seven hours while water rose 10% of the frame, because the window could never
hold two frames. Found on real data, not in synthetic tests; the window is now
adaptive to the sampling interval.

**Page-Hinkley needed calibrating and the naive defaults were dangerous.** Delta
0.002 / threshold 0.02 produced a **92% false-alarm rate on STABLE water** at
realistic segmentation noise. Calibrated defaults (delta 0.005, threshold 0.10)
give 0% at that noise level, costing 4-11 samples of detection latency.

**Mask-to-box bridge.** Flood is the first hazard whose natural output is a mask,
not boxes. Rather than specialise the validator — which would break the
hazard-agnostic property that is the whole claim — connected components of the
mask are emitted as ordinary Detection objects. Validated on real ATLANTIS masks
before adoption: single connected component 99.8% of the time, median
frame-to-frame box IoU 0.964 under simulated rise against a 0.20 tracker
threshold, 0.31 ms conversion cost.

Licence caveat: ATLANTIS carries **no explicit licence file**, so the status of
the annotations is unstated and whether weights trained on them may be
redistributed is genuinely unclear. Recorded rather than resolved in our favour.
V-FloodNet is all-rights-reserved — evaluation use only, nothing redistributed.

**Hazard 2 — people in water (SeaDronesSee) — MEASURED, and it found a bug that
nearly shipped.**

`dronefreak/seadronessee-rfdetr-small`, RF-DETR Small, Apache-2.0. The same
author published YOLO11n/x checkpoints for this dataset under AGPL-3.0; RF-DETR
was chosen specifically to stay outside the AGPL perimeter. SeaDronesSee itself
is **CC0 1.0** — the least restrictive licence in the project.

Only `swimmer` reaches the validator. A boat is context, not an emergency.

SeaDronesSee val, 300 images (255 with swimmers, 45 without), 1,218 ground-truth
swimmer instances, IoU 0.3, conf 0.30, swimmer-only:

| level | precision | recall | F1 |
|---|---|---|---|
| box | 0.9199 | 0.9146 | **0.9172** |
| image | 0.9961 | 0.9922 | FAR **0.0222** |

We do **not** quote the author's headline mAP@50 of 0.7931 — it is across five
classes and carried by boats. Their per-class swimmer AP is 0.282. Ours and
theirs are not in conflict: AP@50:0.05:95 averages over strict IoU thresholds and
punishes imprecise boxes on tiny objects, while our 0.3-IoU operating point asks
the operationally relevant question — did we find the person.

**The integration bug, and why it is the most instructive thing in this project.**
The exported head has six logit slots for five classes. We assumed a leading
no-object slot, the usual DETR convention, and took `scores[:, 1:]` — shifting
every label by one. **Boats were reported as swimmers at 0.84 confidence while
real swimmers were discarded.** Slot 0 is in fact `swimmer` and the unused slot
is trailing.

It nearly survived because the smoke test looked right: plausible counts, high
confidence, and image-level precision of 0.83 — because boats and swimmers
co-occur in the same frames, so "a hazard is present" was often right for the
wrong reason. Only box-level IoU exposed it: **box recall was 0.002.** Caught by
running the author's own `rfdetr.predict()` on the same image and comparing
coordinates against ground truth.

The lesson is now doctrine across every hazard here: **image-level metrics alone
would have reported a working detector.** Box-level evaluation is what made the
failure visible, and it is why the building-access harness below reports both.

**Hazard 4 — building access (DRespNeT) — MEASURED, and the result is weak.**

Fourth hazard, trained by us: torchvision Faster R-CNN MobileNetV3-FPN on
DRespNeT (Cranfield, UAV imagery of the 2023 Türkiye earthquakes, CC BY 4.0),
28 classes merged to five. Only `civilian` reaches the validator; rescue teams
and access points are operator context, on the same reasoning that keeps boats
out of the people-in-water alarm path.

Held-out test split, 24 images, conf 0.20:

| level | precision | recall | F2 |
|---|---|---|---|
| box (`civilian`) | 0.138 | **0.141** | 0.140 |
| image | 0.857 | **1.000** | — |

**Civilian box recall is 0.141. The detector misses roughly six of every seven
people it should box.** It is reported here rather than buried because the gap
to the selection split (0.462) is too large to blame on selection bias: the test
split averages 12 civilian instances per image against valid's 4.5, so it is
dominated by dense crowd scenes that 630 training images do not cover.

Two things survive that. Image-level recall on test is **1.000** — it flags every
frame containing a person, it simply localises them badly, and flagging the
frame is what the validator actually consumes. And the structural half of the
detector generalises far better than the human half (entry_accessible F1 0.311,
building_collapsed 0.261, against civilian 0.091), so "where can a rescuer get
in" is in better shape than "is anyone there".

**Checkpoint selection had to be rebuilt, and this is the transferable lesson.**
The training script selected on its own validation loss — which is the
train-mode RPN/ROI loss, a proxy its own docstring disclaims. That proxy rose
monotonically 1.0718 → 3.0806 while train loss fell 1.1080 → 0.6858, and it
selected epoch 1. All 20 epochs were therefore kept, exported and measured with
the real harness, and epoch 5 beats epoch 1 on **every** axis: box recall 0.390
vs 0.368, precision 0.133 vs 0.095, F2 0.282 vs 0.233, image recall 0.875 vs
0.792, false-alarm rate 0.038 vs 0.231. Selecting on the proxy would have
shipped a strictly worse model with no visible symptom. Two independent runs
gave identical losses to four decimals, so this is re-derivable.

**A parity check that verified nothing.** The ONNX export compared PyTorch
against onnxruntime on random noise. A correctly trained detector finds nothing
in noise, so it compared zero detections against zero and passed — weakest
exactly when the model is good. It now traces a real frame; all 20 exports
agree to ≤0.0002 px with 100% label agreement.

**Also measured:** Faster R-CNN trains 64x slower on MPS than CPU here (148.79
vs 2.34 s/step, same data and weights). The device belongs to the model, not the
machine — DeepLabV3 is fine on MPS, this is not.

**Caveat that binds every number above:** the public DRespNeT release holds out
25 test images. That is thinner than the 23 fire negatives already flagged as
indicative rather than precise. Do not publish this as a people-finding result.
It is honest evidence that the hazard-agnostic validator was measured across a
fourth hazard, with a detector whose weakness is stated rather than hidden.

**Record reconciled.** This status block, NOTICE and models/REGISTRY.yaml now
agree. NOTICE credits all five detectors as integrated (its "planned" section is
empty), including the two dataset credits it had been missing, and its preamble
no longer claims every detector was trained by someone else — two are ours.
The registry's traffic entry is corrected from `planned` to `implemented`, with
a note that its `measured` status stays open because the multihazard experiment
measured the VALIDATOR, not that detector.

---

---

## SCOPE REVISION — 1 September 2026

**Two hazards is too thin to call this a disaster detection system, and one of
the two (traffic accidents) is not a disaster.** Correct on both counts. A
research pass changed the hazard set, and reversed one of my earlier calls.

### Correction: building damage was wrongly excluded

I excluded it after evaluating **xView2** — bi-temporal *satellite* damage
assessment, wrong modality, correct to reject. But that is not the task that
matters here. The task is **finding people trapped in damaged buildings**, which
is search-and-rescue, not damage assessment, and it has strong public resources:

**DRespNeT** (arXiv 2508.16016, Cranfield University) — UAV imagery from the
**2023 Türkiye earthquakes**. 28 operationally critical classes covering
structurally compromised buildings, **access points (doors, windows, gaps)**,
debris levels, **rescue personnel** and **civilian visibility**. Polygon-level
instance segmentation from 1080p aerial footage. YOLOv8-DRN reports
**92.7% mAP50 at 27 FPS**. Released on **Figshare and Roboflow Universe**.

This is precisely what the original VIGÍA document proposed — UAVs identifying
viable access points in collapsed structures — but with a released dataset and
published metrics, and far stronger than the repo originally credited.

### Revised hazard set

Every one of these has public data, and three map directly onto the DANA.

| # | hazard | why it belongs to the story | resource | licence |
|---|---|---|---|---|
| 1 | **Flood + water level** | The DANA itself. 223 dead, water rising in streets. | ATLANTIS, Inundation2Depth, FloodVision | Apache-2.0 / CC |
| 2 | **People in water** | DANA victims swept away and trapped in vehicles. | **SeaDronesSee** — 8,930/1,547/3,750 images + 54,105 MOT frames, humans in open water | **CC0 1.0 + MIT code** |
| 3 | **Trapped people / building access** | DANA victims trapped in buildings; earthquakes generally. | **DRespNeT** — Türkiye 2023, 28 classes | Figshare / Roboflow |
| 4 | Fire & smoke | La Palma, Mediterranean wildfire | PyroNear | Apache-2.0 |
| 5 | Traffic incidents | **Generality proof, not a headline.** Different physics, different timescale — evidence the validator transfers. | Enos yolo11x | MIT |

**"People in water" replaces "pool drowning" entirely.** Pools never fitted the
disaster narrative. SeaDronesSee is open water, CC0 licensed, and the DANA
killed people in exactly that situation. It is also the best-licensed dataset in
the entire project — CC0 means no restrictions at all.

### Why this is now defensible as a *system*

Fire, flood, people-in-water and building-access are four genuine disaster
hazards. Traffic is retained and explicitly demoted in the writing to a
transfer experiment. The DANA narrative is now carried by three hazards rather
than none.

### Flood needs the most care

It is the keystone and must work well. Beyond ATLANTIS:

- **Inundation2Depth** (2025) — inundation extent *and depth* labels from aerial
  imagery plus LiDAR terrain models, 12 flood-affected areas.
- **FloodVision** (arXiv 2509.04772) — VLM-based depth estimation, MAE **8.17 cm**
  on 110 urban flood images, **no task-specific training**. Worth benchmarking
  against our fine-tune before assuming training is required.
- Published CNN segmentation of real surveillance-camera flood imagery reports
  **F1 > 0.9**, so the segmentation half is well-trodden. The open gap remains
  the anticipatory signal: rate-of-rise, not presence.

### Open-water caveat to state explicitly

Reported limitations of open-water systems: extreme crowd density, turbid or
churned water in surf zones, and blind spots from camera angle or lighting.
Declare that envelope rather than claiming beach-wide capability.

**Next:** all four hazards are now measured — flood, people-in-water and
DRespNeT are done; traffic remains the demoted transfer experiment. Still
outstanding: drop the dead size level, widen the negative set, reconcile NOTICE
and this status block with REGISTRY.yaml, and build the operator view.

---

## 0. Purpose of this document

This is the working build plan. It records the architecture, every component decision with its
source and licence, the phase-by-phase implementation sequence, and the acceptance criteria for
each phase. It is written so that implementation and validation can proceed without further
design decisions being needed.

Where a decision has been made, it is recorded as **DECIDED** with the reasoning. Where something
genuinely needs Jon's input or an external party, it is marked **BLOCKED** or **NEEDS JON**.

---

## 1. The claim we are making

> VIGÍA integrates the best publicly available hazard detectors under a single gated pipeline and
> adds a shared temporal validation layer that makes them operationally trustworthy on ordinary
> municipal cameras.

Every part of that sentence is defensible. The part that is ours is the **temporal validation
layer** and the **measurement** of what it buys. We are explicitly not claiming novel detection
architectures.

### Why this framing

- Detection is commodity. Datasets are public, fine-tuning is a weekend.
- The unsolved, expensive problem is the **false alarm**. Industry reference: ~98% of security
  camera alarms are false. Pano AI charges ~$50,000/camera/year and staffs a 24/7 human
  intelligence centre; ALERTCalifornia routes 1,060+ cameras through trained human vetting.
- Published evidence that the validator is where the value is: wrapping an off-the-shelf fire
  detector in a five-level validation cascade moved false-alarm rate from **52% → 4%** while
  retaining **96% sensitivity** (arXiv 2607.03131).

---

## 2. Architecture

Three tiers. Each detector is interchangeable; the validator is shared and hazard-agnostic.

```
                    TIER 0              TIER 1                    TIER 2
                    context gate        specialist detectors      temporal validator
  forest cam ──┐                    ┌── fire & smoke ───┐
  street cam ──┤──► camera registry ┼── flood / level ──┼──► confidence           ┌─► CONFIRMED
  river cam  ──┤    declares which  ├── traffic incident┤    size plausibility    │   ALERT
  pool cam   ──┤    hazards apply   └── pool drowning ──┘    colour prior     ────┤
  motorway   ──┘    to each camera                          persistence (IoU)     └─► suppressed
                                                            location cooldown         (logged)
```

### Tier 0 — Context gate  **DECIDED**

**Declarative, not learned.** Each camera is registered once with its context
(`forest`, `street`, `river`, `pool`, `motorway`). The gate looks up which detectors apply.

- We do NOT build a classifier that inspects a frame and routes it. That adds a misrouting
  failure mode and saves little.
- Every real deployment works this way. Pano AI does not run drowning detection on a ridgeline
  camera.
- Evidence for the efficiency gain: cascaded gating in video pipelines delivers 5–13× compute
  reduction with minimal accuracy loss (NoScope, arXiv 1703.02529; CaTDet).

### Tier 1 — Specialist detectors  **DECIDED**

Separate models running in parallel, **not** a shared-backbone multi-task model.

Rejected: unified multi-class model. It needs one coherent label space and jointly-annotated
data; our hazards come from separate datasets with incompatible labels. It also couples training —
retraining fire would risk silently degrading flood.

Reference implementation for the concurrency pattern (arXiv 2607.03131): five independent
detectors, seven daemon threads, bounded thread-safe queues, one fusion thread, last-available-frame
fusion policy. Achieved ~10 FPS across 6 concurrent streams on an RTX 4060 laptop, per-frame
latency under 100 ms.

### Tier 2 — Temporal event validator  **DECIDED — this is our contribution**

Hazard-agnostic. Five levels, applied to every detector's output:

1. **Per-class confidence threshold** — per-hazard, tuned on our own eval set.
2. **Size plausibility** — reject detections whose bounding box is implausible for the hazard at
   that image position.
3. **Colour / appearance prior** — e.g. HSV gate for fire.
4. **Temporal persistence** — a candidate becomes a confirmed event only after it persists across
   N consecutive frames, associated frame-to-frame by IoU overlap.
5. **Per-location cooldown** — suppress repeat alerts from the same region within a time window.

Every level must be individually switchable so we can publish an ablation showing each level's
contribution.

---

## 3. Component decisions

| Hazard | Component | Licence | Source | Status |
|---|---|---|---|---|
| Fire & smoke | PyroNear `yolo11s_sensitive-detector_v1.0.0`, ONNX export | Apache 2.0 | `huggingface.co/pyronear/yolo11s_sensitive-detector_v1.0.0` | **Core, off-the-shelf** |
| Traffic incident | `Enos-123/traffic-accident-detection-yolo11x`, exported to ONNX | MIT | `huggingface.co/Enos-123/traffic-accident-detection-yolo11x` | **Core, off-the-shelf** |
| Flood & water level | MMSegmentation checkpoint fine-tuned on ATLANTIS | Apache 2.0 + CC | `github.com/open-mmlab/mmsegmentation` + `github.com/smhassanerfani/atlantis` | **Core, we fine-tune** |
| Pool drowning | Fine-tune on Water Behavior Dataset + Figshare underwater set | verify per dataset | Figshare DOI `10.6084/m9.figshare.29497235`; IEEE Xplore 9618700 | **Core, we fine-tune** |
| Pipeline reference | `pyronear/pyro-engine` | Apache 2.0 | `github.com/pyronear/pyro-engine` | **Study, do not fork** |

### Excluded, with reasons

| Component | Reason |
|---|---|
| xView2 first-place ensemble | Weights ARE public. Excluded on **modality** — needs bi-temporal satellite pairs at 0.5 m GSD plus a pre-disaster image at inference time. Not a camera model. |
| V-FloodNet | **All rights reserved.** Best flood system found, but no licence to build on. Cite as prior art. Email sent in Phase 0 — a yes is a bonus, not a dependency. |
| SegFormer / Mask2Former | NVIDIA non-commercial licence. *Usable* for our purpose but must never be committed to the repo. Not needed — ATLANTIS route is permissive and better. |
| RipVIS (rip currents) | v2 candidate. Weight availability under a reuse-permitting licence not yet verified. Also detects a hazard *condition*, not a person in distress. |

### Reference performance figures (theirs, not ours)

- Traffic YOLO11x: mAP@0.5 **0.826**, mAP@0.5:0.95 **0.600**, precision 0.808, recall 0.759,
  F1 0.782, accident-class recall **0.855**.
- Figshare underwater drowning: YOLOv8n baseline **98.3% precision**, 22 ms/frame (45 FPS).
- PyroNear model card publishes **no** metrics. PyroNear2025 benchmark paper reports cross-dataset
  F1 ≈ **70%**. Treat that as the honest ballpark; measure our own.

> **Rule: never quote a number we did not measure.** Published third-party figures go in a clearly
> labelled "reported by original authors" column, never in our results table.

---

## 4. Licensing  **DECIDED**

**Project licence: Apache 2.0.**

### How we avoid AGPL

Ultralytics ships under AGPL-3.0, and their position is that using their code, models,
architectures, training pipelines or fine-tuned models requires releasing the complete
corresponding source of the entire derivative work under AGPL-3.0. AGPL-3.0 is compatible with
essentially nothing except GPL-3.0.

**The escape: run ONNX Runtime, never import `ultralytics` in the runtime path.**

- PyroNear publish `onnx_cpu.tar.gz` alongside the `.pt`, and `pyro-engine` does ONNX inference —
  which is exactly why pyro-engine can be Apache 2.0 while running a YOLO-architecture model.
- ONNX Runtime is MIT. Model weights are Apache 2.0 / MIT.
- Any `.pt → .onnx` export is a **one-time offline build step**, isolated in `tools/`, not part of
  the distributed runtime.

### Non-commercial components

The line is **use vs. redistribute**, not commercial vs. non-commercial.

- Using an NC-licensed model privately for a non-commercial prize submission with attribution
  fits that licence.
- What breaks is *bundling* NC weights in a repo published under an open licence — that grants
  downstream rights we do not hold.
- **Rule:** NC weights are fetched by `scripts/fetch_models.py` at setup time from the original
  source. Never committed. Never in a release archive.

### Attribution artefacts (all four required)

1. `NOTICE` — every third-party component: name, author, source URL, licence, version/commit, and
   what we changed.
2. `models/REGISTRY.yaml` — per model: provenance, licence, training dataset, **their** published
   metrics, **our** measured metrics, declared operating envelope. Doubles as system config so it
   cannot drift out of date.
3. Dossier prior-art section — naming PyroNear, Pano AI, ALERTCalifornia, Lynxight, AngelEye,
   Coral, Rekor, xView2 team, RipVIS authors, and saying what each does better than VIGÍA.
4. Academic citations for D-Fire, FIgLib, PyroNear2025, ATLANTIS, CADP, DoTA, the Figshare
   drowning set, and the Water Behavior Dataset. Cite papers, not download links.

---

## 5. Repository structure

Fresh repository. The current tree is a fork of `roihan12/traffic-accident-detection` — 4 of its 5
commits are the upstream author's. Keep the name VIGÍA, carry attribution forward explicitly, do
not carry the fork lineage.

```
vigia/
  LICENSE                  Apache-2.0
  NOTICE                   third-party attributions
  PLAN.md                  this document
  README.md                what it is, how to run it, what it is not
  pyproject.toml

  vigia/
    __init__.py
    registry.py            camera registry + context gate (Tier 0)
    detectors/
      base.py              ONNXDetector ABC — load, preprocess, infer, postprocess
      fire.py
      flood.py
      traffic.py
      drowning.py
    validator/
      __init__.py
      cascade.py           the five levels, each independently switchable
      tracker.py           IoU association across frames
      cooldown.py
    pipeline.py            capture → gate → parallel detect → validate → emit
    io/
      sources.py           file, RTSP, webcam
      alerts.py            event emission + evidence frame

  models/
    REGISTRY.yaml
    .gitignore             weights are never committed

  eval/
    harness.py             cross-dataset, event-level video evaluation
    metrics.py             precision, recall, F1, F2, false-alarm rate
    datasets/              loaders per dataset
    results/               committed JSON + generated tables

  tools/
    export_onnx.py         one-time .pt → .onnx (isolated; may import ultralytics)
    fetch_models.py        downloads weights from original sources

  scripts/
    train_flood.py         MMSeg fine-tune on ATLANTIS
    train_drowning.py
    train_fire_yolo26.py   for the YOLO11-vs-YOLO26 experiment

  docs/
    dossier/               English submission text
    figures/
```

---

## 6. Hardware & cost budget

**Development machine:** MacBook Pro 16", M4 Pro, 48 GB unified memory. No discrete GPU.

Measured references:

| Model | FPS on M4 Pro (MLX) |
|---|---|
| YOLO26n | 124.9 |
| YOLO26s | 57.7 |
| YOLO26m | 25.3 |
| YOLO26l | 19.8 |
| YOLO26x | 10.7 |

MLX runs 1.1–2.6× faster than PyTorch MPS for inference, ~1.7× for training, with accuracy parity
within 0.5% mAP. YOLOv8n-640 hits 92.6 FPS on M4 Pro (Roboflow).

**Inference: entirely local.** A two-to-four detector demo on one stream is far inside this machine's
limits.

**Training: mostly local.** Reference point — YOLO26n on COCO128 for 10 epochs takes 64.1 s under
MLX. Extrapolated (order-of-magnitude only; batch size, resolution and model scale all move it):

- ~5,000 images × 100 epochs, nano/small model → **overnight on the laptop**
- ~20,000 images → 1–2 nights locally, or **$10–30** rented

**Cloud budget ceiling: $200 for the whole project.** Spot pricing: A100 80GB ~$1.00–2.50/hr,
H100 $1.49–1.99/hr (Vast.ai, RunPod). Rent H100 over A100 when a job would exceed a day on the
cheaper card — the ~3× speedup usually makes it cheaper in absolute terms. Never keep a persistent
instance.

---

## 7. Evaluation strategy

**Build the harness before the second detector.** With commodity detectors, our only real result is
the comparison between raw and validated output. The harness *is* the contribution, measured.

### Non-negotiable methodology

1. **Cross-dataset, always.** Train on one dataset, report on another. Train on D-Fire, evaluate on
   FIgLib and PyroNear2025. Never headline a random split of the training set.
2. **Report the drop.** In-domain and out-of-domain side by side. The gap is the finding, not an
   embarrassment. Literature: domain shift causes sharp performance drops in real deployment;
   PyroNear2025 cross-dataset F1 ≈ 70% and that is described as the *stable* result.
3. **Event-level on video, not frame-level.** The validator only shows its value over time.
   Frame-level metrics make it look conservative.
4. **F2, not F1, as the headline.** Recall matters more than precision for life safety. Borrowed
   from RipVIS, which makes the same choice deliberately.
5. **Ablate the validator.** Report each cascade level's individual contribution to false-alarm
   reduction. This is the core experiment.

### The three headline results we are trying to produce

- **R1** — False-alarm rate with and without the temporal validator, per hazard, on negative video
  clips. *The main result.*
- **R2** — Cross-dataset generalisation gap per detector.
- **R3** — Overhead-only vs. underwater drowning detection recall. Nobody in the reference list has
  published this for a municipal-camera setting.

---

## 8. Phase plan

### Phase 0 — Foundation  ·  1–12 September  ·  ~20 h

Nothing new gets built until the ground is solid.

**Tasks**
- [ ] Create fresh `vigia/` repo with the structure in §5. Apache 2.0 LICENSE, NOTICE, README.
- [ ] `models/REGISTRY.yaml` schema defined and populated with the four planned components.
- [ ] `vigia/detectors/base.py` — `ONNXDetector` ABC over ONNX Runtime. No `ultralytics` import.
- [ ] Fetch PyroNear ONNX weights via `scripts/fetch_models.py`. Pin revision (v1.1.0 is newer
      than v1.0.0 — check which we want).
- [ ] Get PyroNear running on a test video end-to-end. Print detections. **Nothing else matters
      until this works.**
- [ ] Reproduce published M4 Pro FPS figures locally so we can quote our own numbers.
- [ ] Download D-Fire, FIgLib, PyroNear2025, ATLANTIS. Record licences in NOTICE as they arrive.
- [ ] Read `pyro-engine` source properly — it is the closest reference implementation for camera
      ingestion and the edge inference loop.
- [ ] **NEEDS JON:** send the V-FloodNet email (non-commercial, UNESCO submission, full credit).
      Longest lead time, zero cost. Build the ATLANTIS route regardless.

**Acceptance criteria**
- `python -m vigia.detectors.fire --video sample.mp4` prints per-frame detections.
- Runtime dependency tree contains no AGPL package.
- REGISTRY.yaml validates against its schema.

---

### Phase 1 — Fire vertical slice + the validator  ·  13 September – 4 October  ·  ~45 h

One hazard taken all the way through all three tiers, so the architecture is proven before it is
repeated. **This is the most important phase — give it the most time.**

**Tasks**
- [ ] `eval/harness.py` — cross-dataset, event-level video evaluation. Build this FIRST.
- [ ] `eval/metrics.py` — precision, recall, F1, F2, false-alarm rate on negative clips.
- [ ] Baseline: PyroNear detector, raw output, no validation. Record R2 numbers.
- [ ] `validator/cascade.py` — all five levels, each independently switchable.
- [ ] `validator/tracker.py` — IoU association across consecutive frames.
- [ ] **R1 experiment:** false-alarm rate with validator on / off, on negative fire clips.
- [ ] Ablation: each cascade level's individual contribution.
- [ ] **YOLO11-vs-YOLO26 experiment:** train YOLO26s on D-Fire, evaluate against PyroNear's YOLO11
      on the same held-out set. Export winner to ONNX. Whichever wins, the comparison is a result.

**Acceptance criteria**
- R1 produced: a measured before/after false-alarm number for fire.
- Ablation table generated into `eval/results/`.
- The validator is hazard-agnostic in code — no fire-specific logic outside `detectors/fire.py`.

---

### Phase 2 — Flood  ·  5–18 October  ·  ~30 h

If Phase 1 was built properly this is a new detector behind an unchanged validator.

**Tasks**
- [ ] MMSegmentation set up; pick an Apache-2.0 checkpoint from the model zoo.
- [ ] `scripts/train_flood.py` — fine-tune on ATLANTIS (5,195 images, 56 classes: sea, lake, river,
      marsh, wetland, dam, reservoir, canal, levee, pier). Overnight run.
- [ ] Water-region segmentation → mask.
- [ ] Level estimation: mask boundary against a fixed per-camera reference line. Geometry, not
      learning.
- [ ] **Rate-of-rise** over time. This is the anticipatory signal the literature says is missing —
      most vision work detects inundation only after it occurs, which "limits operational value for
      anticipatory early warning."
- [ ] Validator applied to flood events (persistence and cooldown adapt; colour prior likely off).

**Acceptance criteria**
- Water mask produced on held-out ATLANTIS test split with reported IoU.
- Rate-of-rise emits a graded alert on a synthetic or real rising-water clip.
- No changes required to `validator/cascade.py` beyond configuration.

---

### Phase 3 — Traffic, drowning, and the gate  ·  19 October – 1 November  ·  ~35 h

**Tasks**
- [ ] Traffic: export the MIT YOLO11x to ONNX via `tools/export_onnx.py`. Drop in behind the
      validator. Replaces the old bounding-box-overlap heuristic entirely.
- [ ] Note: YOLO11x is the largest variant — expect ~10–20 FPS on the M4 Pro. Acceptable because
      the gate means it only runs on road cameras. Consider a smaller variant if it hurts.
- [ ] Drowning: `scripts/train_drowning.py` on the Water Behavior Dataset (overhead + underwater)
      and the Figshare set (5,613 images, 3 balanced classes, already YOLO format).
- [ ] **R3 experiment:** train on both camera angles, then report overhead-only vs. underwater
      recall separately. Declare the operating envelope explicitly: clear pools, overhead view.
      Do NOT claim open-water or beach capability from these weights.
- [ ] `vigia/registry.py` — camera registry and context gate.
- [ ] `vigia/pipeline.py` — capture → gate → parallel detect → validate → emit. Bounded queues,
      one fusion thread, last-available-frame policy.

**Acceptance criteria**
- Four detectors registered; a camera declared `forest` runs only fire.
- R3 produced: a measured overhead-vs-underwater recall gap.
- Pipeline sustains real-time on one stream with two active detectors on the M4 Pro.

---

### Phase 4 — Integration, demo, results freeze  ·  2–12 November  ·  ~30 h

Dossier draft goes to the Instituto at the **start** of this phase, not the end.

**Tasks**
- [ ] Build the operator view to the design in §9. FastAPI + MJPEG + WebSocket, no build step.
      The dual view — proposed vs. confirmed — is the whole point; the suppressed-alert log is what
      visually proves the validator works.
- [ ] Curate one sample clip per hazard for deterministic replay. These are the stage demo.
- [ ] **Record the demo video and GIF early.** Live demos fail on stage; a recording is insurance,
      and this is also the lecture material if the prize is won.
- [ ] Freeze results. Regenerate all tables from `eval/results/`.
- [ ] **NEEDS JON:** send dossier draft + CV + supporting documents to the Instituto de la
      Ingeniería de España so they can draft the Article 6.3 recommendation.

**Acceptance criteria**
- One command starts the demo from a registry file and a video directory.
- Every number in the dossier traces to a file in `eval/results/`.

---

### Phase 5 — Dossier and submission  ·  13–30 November  ·  ~25 h

Reserve more time than feels necessary.

**Tasks**
- [ ] Rewrite the technical document in **English** (Article 6.3 requires English or French).
      Existing `VIGÍA.docx` is Spanish.
- [ ] Replace every projected figure with a measured one. The current deck's "85% detection,
      <10% false positives" are labelled projections in our own document — they go.
- [ ] Privacy-by-design section (see §10).
- [ ] Prior-art section naming the competition accurately.
- [ ] Publish the repository publicly with reproducible training and evaluation scripts.
- [ ] **NEEDS JON:** submit online at `unescoalfozanprize.org` alongside the nomination.

---

## 9. Presentation layer

### The insight that should drive the entire design

Everyone's demo draws a box around a fire. That is not distinctive and a jury has seen it before.
What nobody else shows is **detections being rejected**. Our validator suppresses most of what the
detector proposes, and watching that happen in real time *is* the demo — it is the visual proof of
the only claim that is genuinely ours.

So the interface is built around a **dual view: what the detector proposed vs. what the system
confirmed.** Every design decision below follows from that.

### Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  VIGÍA    [forest cam ▾]   fire ●  flood ○  traffic ○  drowning ○        │  ← Tier 0 made visible
├───────────────────────────────────────────┬──────────────────────────────┤
│                                           │  CASCADE                     │
│                                           │  1 confidence     412 │ 38   │
│           video canvas                    │  2 size           38  │ 12   │  ← the money shot:
│        with detection overlays            │  3 colour prior   12  │  9   │    live pass/reject
│                                           │  4 persistence     9  │  3   │    counts per level
│                                           │  5 cooldown        3  │  2   │
│                                           │                              │
│                                           │  CONFIRMED  2                │
├───────────────────────────────────────────┴──────────────────────────────┤
│ candidates ▏▎▏▏▎▏▏▏▎▏▎▏▏▏▎▏▏▎▏▏▏▎▏▏▎▏▏▏▎▏▏▎▏▏▏▎▏▏▎▏▏▏▎▏▏▎▏▏  (grey)     │  ← density contrast
│ confirmed  ▏              ▏                        ▏          (accent)   │    tells the story
├──────────────────────────────────────────────────────────────────────────┤
│ seen 412 · confirmed 2 · suppressed 410 · rate 99.5% · 31 FPS · 24 ms    │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Header** — camera context selector and active-detector chips. Makes the gate visible; switching
  from `forest cam` to `pool cam` visibly changes which detectors light up. That single interaction
  explains Tier 0 better than a slide would.
- **Main canvas** — video with overlays. Bounding boxes for detectors, translucent mask for flood,
  and a reference line with a live water-level readout.
- **Right rail** — the cascade as a signal chain, each level showing live pass/reject counts. This
  is the part that makes the contribution legible in five seconds.
- **Bottom timeline** — candidate ticks against confirmed events. The visual density contrast
  between the two rows is the argument.
- **Metric strip** — tabular figures, always visible, so nothing looks hand-waved.

### Input paths — build all four

1. **Sample scenarios** — curated clip per hazard, one click, deterministic replay.
   **This is the primary stage path.** Never rely on anything else in front of an audience.
2. **Drag-and-drop upload** — any video or image file.
3. **RTSP / HTTP stream URL** — for a live municipal camera. Impressive when it works, never
   depended upon.
4. **Local webcam** — the party trick: point a phone showing a fire video at the laptop camera and
   watch it detect. Cheap, memorable, works offline.

### Stack  **DECIDED**

- **FastAPI**, not Flask. Native async, first-class WebSocket support, auto-generated OpenAPI docs
  that make the system look finished. The existing `flaskapp.py` (40 KB, 30 routes, several dead or
  commented out) is not carried forward.
- **MJPEG over HTTP for video, WebSocket for events.** Two channels, separate concerns. The
  reference implementation (arXiv 2607.03131) flags Base64-over-WebSocket frame transport as a
  bottleneck and lists WebRTC as future work — we sidestep the whole problem by never sending
  frames down the event channel.
- **Vanilla JS + modern CSS. No build step, no `node_modules`.** A judge or a municipality should
  be able to clone and run it. This is also a credibility signal: a project that needs a toolchain
  to demo is a project nobody will try.
- **Deterministic replay mode** — fixed seed, fixed frame stepping, so a given clip produces byte-
  identical output every run. The demo behaves the same in rehearsal and on stage, and it makes the
  results reproducible for anyone checking our numbers.

### Visual direction

Control room, not consumer app. Dark ground with a restrained palette; semantic colour reserved
strictly for state (confirmed / suppressed / degraded) and never used decoratively. Tabular
numerals everywhere figures appear. Confidence scores shown, not hidden — displaying uncertainty
reads as rigour.

Avoid: neon-on-black, gratuitous animation, rounded-card soup, and any chart that does not encode a
real quantity. Restraint is what separates a system from a student project.

### Stage rules — non-negotiable

- Record the full demo video **early in Phase 4**, not the night before. Live demos fail.
- Every sample clip local on disk. Assume no network at presentation time.
- Keep the recording one keystroke away to cut to.
- Build it once and reuse: the same material serves the submission, the demo, and the Article 7.3
  lecture if the prize is won.

---

## 10. Privacy by design  **DECIDED — write this into the architecture, not the dossier**

EU AI Act high-risk provisions took full effect August 2026. Annex III classifies biometric
identification as high-risk; Article 5(1)(h) bans real-time remote biometric identification in
publicly accessible spaces for law enforcement, with narrow exceptions.

VIGÍA needs to identify nobody. Enforce this structurally:

- No face recognition, no re-identification, no gait or biometric feature extraction — **by
  architecture, not configuration**. There must be no code path that could be switched on.
- Detection at the edge; only the event plus a single evidence frame leaves the camera.
- No raw video retention beyond a short rolling buffer.
- Person detection reduced to presence and location within a hazard region, never identity.

UNESCO authored the Recommendation on the Ethics of AI. Designing the ethical constraint in rather
than bolting on a compliance paragraph speaks the institution's own language. Give it a named
section in the dossier.

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Domain shift — models fail on real Spanish cameras | **High** | High | Cross-dataset evaluation from Phase 1. Report the gap honestly. Get real Spanish footage if at all possible. |
| Scope creep back to 5+ hazards | Medium | High | Four is the cap. Adding a fifth requires cutting something else. |
| Fine-tuning turns into a research project | Medium | High | Integrate first, measure, fine-tune ONLY where evaluation shows a specific failure. |
| Traffic YOLO11x too slow for live demo | Medium | Low | Gate means it runs only on road cameras. Fall back to a smaller variant. |
| V-FloodNet permission never arrives | High | **Low** | ATLANTIS route is the plan; permission is a bonus. |
| Drowning overclaim challenged by a juror | Medium | High | R3 measures it and the envelope is declared. Turning the weakness into the experiment is the mitigation. |
| Dossier late to nominator | Low | **Critical** | Draft goes 2 November, three weeks before deadline. |
| Accidental AGPL contamination | Medium | Medium | CI check: fail the build if `ultralytics` appears in the runtime dependency tree. |

### Highest-value action not on the critical path

**Get real Spanish footage.** Municipal cameras, fire service archives, DANA material, a university
group with recordings. This affects final numbers more than every architectural decision in this
document combined. Routes worth pulling: the Instituto de la Ingeniería de España, UPV/EHU.

---

## 12. Sources

### Models and code
- PyroNear detector — `huggingface.co/pyronear/yolo11s_sensitive-detector_v1.0.0` (Apache 2.0)
- PyroNear engine — `github.com/pyronear/pyro-engine` (Apache 2.0)
- PyroNear vision — `github.com/pyronear/pyro-vision`
- Traffic detector — `huggingface.co/Enos-123/traffic-accident-detection-yolo11x` (MIT)
- MMSegmentation — `github.com/open-mmlab/mmsegmentation` (Apache 2.0)
- ATLANTIS — `github.com/smhassanerfani/atlantis` (iWERS lab, University of South Carolina)
- V-FloodNet — `github.com/xmlyqing00/V-FloodNet` (all rights reserved — cite only)
- xView2 first place — `github.com/DIUx-xView/xView2_first_place` (excluded on modality)

### Datasets
- D-Fire — 21,000+ images (5,867 smoke-only, 4,658 fire+smoke)
- FIgLib — 24,800 high-resolution images, HPWREN fixed cameras
- PyroNear2025 — ~50,000 images, ~150,000 annotations, 640 wildfires across France, **Spain**,
  Chile, USA (arXiv 2402.05349)
- ATLANTIS — 5,195 images, 56 classes, split 3,364/535/1,296; source images under Creative
  Commons / No Known Copyright / US Government Work
- Underwater Drowning Detection — Figshare DOI `10.6084/m9.figshare.29497235`, 5,613 images,
  balanced 1,871 per class, YOLO format, 640×640
- Water Behavior Dataset — IEEE Xplore 9618700, swim/drown/idle, overhead **and** underwater
- SeaDronesSee — 8.9k/1.5k/3.7k, UAV open-water swimmers, COCO format (v2 candidate)
- CADP (~2,000 videos), DoTA (4,677 dash-cam videos) — traffic, only if we replace the MIT model

### Key papers
- arXiv 2607.03131 — Multi-Task Deep Learning Framework for Real-Time Intelligent Video
  Surveillance with Temporal Event Validation. **The closest published analogue.** Source of the
  52%→4% figure and the concurrency pattern.
- arXiv 2402.05349 — PyroNear2025 real-world early wildfire detection benchmark
- arXiv 2112.08598 — FIgLib & SmokeyNet
- arXiv 1703.02529 — NoScope, cascaded filtering for video queries
- arXiv 2504.01128 — RipVIS (CVPR 2025), source of the F2-over-F1 argument
- ATLANTIS — Erfani et al., Environmental Modelling & Software, 2022

### Context and positioning
- Pano AI, ALERTCalifornia / DigitalPath, PyroNear — wildfire deployments
- Lynxight, AngelEye (ISO 20380:2017), Coral MYLO (ASTM F3698-24) — drowning
- Rekor, Iteris, Miovision, NoTraffic — traffic incident detection
- IntelliSee — 98% false alarm rate figure
- WMO / UNDRR — Early Warnings for All, universal multi-hazard coverage by 2027
- SSPH+ / Frontiers — DANA Valencia 2024: 223 fatalities, 15,000 displaced, >€50bn losses;
  AEMET red warning hours before the ES-Alert went out
- EU AI Act Annex III, Article 5(1)(h); August 2026 enforcement

### Model hubs — where to look
Hugging Face Hub (1M+ models, **licence is a search facet — filter first**), Roboflow Universe
(50k models, paired with training datasets), OpenMMLab model zoos (Apache 2.0), Kaggle Models.

Note: Papers With Code was sunset by Meta in July 2025. Successors: Hugging Face Trending Papers,
CodeSOTA, Wizwand, Hyper.ai. Frozen archive at the `pwc-archive` org on Hugging Face.

---

## 13. Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | Gated parallel specialists, not a unified model | Incompatible label spaces; coupled training; gate gives 5–13× compute reduction |
| D2 | Declarative gate, not a learned router | Avoids misrouting failure mode; matches every real deployment |
| D3 | Temporal validator is the contribution | Detection is commodity; false alarms are the unsolved problem (52%→4%) |
| D4 | Apache 2.0 via ONNX Runtime, never import `ultralytics` at runtime | AGPL is compatible with almost nothing; pyro-engine proves the pattern |
| D5 | NC licences usable, never redistributed | The line is use vs. redistribute, not commercial vs. non-commercial |
| D6 | Four hazards: fire, flood, traffic, drowning | All four have a credible sourced route; five was reckless, two was too few |
| D7 | Building damage excluded on modality | xView2 weights are public but need bi-temporal satellite pairs |
| D8 | Run models on their native architecture; don't migrate to YOLO26 wholesale | YOLO11 is ~2 years old and fully supported; on custom data the ranking can invert; PyroNear's value is its training data, not its architecture generation |
| D9 | YOLO11-vs-YOLO26 becomes an experiment, not an assumption | One overnight run converts a guess into a measurement |
| D10 | F2 over F1 as headline metric | Recall matters more than precision for life safety |
| D11 | Cross-dataset evaluation is mandatory | Domain shift is the single most likely cause of real-world failure |
| D12 | Demo is built around proposed-vs-confirmed, not detection | Boxes around fire are commodity; suppression is the only thing that is ours |
| D13 | FastAPI + MJPEG + WebSocket, vanilla frontend, no build step | Judges and municipalities must be able to clone and run it |
| D14 | Deterministic replay mode | Identical behaviour in rehearsal and on stage; also makes results reproducible |

---

## 14. Open items

**NEEDS JON**
- Send the V-FloodNet email (Phase 0)
- Chase real Spanish footage via the Instituto or UPV/EHU (any phase — highest value)
- Send dossier draft to the Instituto by 2 November (Phase 4)
- Final online submission at `unescoalfozanprize.org` (Phase 5)

**TO VERIFY DURING IMPLEMENTATION**
- PyroNear v1.0.0 vs v1.1.0 — which revision to pin
- Licence of each Roboflow project if any are used as fallbacks
- Water Behavior Dataset access terms and download route
- Whether RipVIS publishes reusable weights (v2 decision only)
