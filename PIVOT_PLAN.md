# PIVOT_PLAN — GlobLoc Backbone + Hypernetwork TTA for 3D Indoor Localization

Living document. Update in place as findings change; do not delete superseded
entries, mark them superseded instead.

## Project goal

Combine a collaborator's GlobLoc-style CNN pipeline (RSSI -> "virtual space"
image -> CNN + SPP -> (x,y) regression + floor classification) with the
hypernetwork-based test-time adaptation (TTA) mechanism from the completed
prior project (`../wifi_tta`, `../hypernetwork-tta-wifi-localization`), to
produce a foundation-model-style backbone that jointly predicts (x, y, floor)
with drift correction at deployment time, without ever using ground-truth
labels post-deployment.

Non-negotiable methodology carried over from the prior project (grouped
splits, naive-baseline reporting, oracle diagnostic before hypernetwork code,
validation-only hyperparameter calibration, multi-seed deployment evaluation,
honest negative results) — see original project brief. Not re-litigated here.

## Decisions log

- **2026-07-13** — New project directory created at
  `B:\Uni materials\WRC Internship\17th JUly\glob_tta_3d_localization`,
  sibling to `wifi_tta` and `hypernetwork-tta-wifi-localization`, kept
  separate until it produces an honest, validated result.
- **2026-07-13** — Primary dataset chosen: **HDLC** ("Hybrid-fingerprint Data
  with Layout Change"), Nor Hisham, Ng, Tan & Chieng, *Data* 2022, 7(11), 156,
  https://doi.org/10.3390/data7110156, data hosted at
  https://doi.org/10.5281/zenodo.7306455 (CC-BY 4.0, direct download, no
  auth). This is the primary source of the collaborator's "Layout 1/2/3"
  Kaggle CSVs (`yomnayasser888/localization-data` is a re-upload of this).
  Downloaded to `data/raw/` (7,478,725 bytes, matches Zenodo-reported size).
- **2026-07-13** — Scope decision: **WiFi-only** for the first pass (17 WAP
  columns), matching the collaborator's existing trained pipeline. The
  dataset also has 42 BLE beacon columns with published ground-truth
  positions (hybrid Wi-Fi+BLE fusion is the dataset's actual design intent)
  — deferred as a later fusion extension, not in scope for the initial
  backbone + TTA validation.
- **2026-07-13** — "3D" operationalized as **(x, y, floor)**, 3 floor
  classes. No building dimension: HDLC is a single site. If a
  cross-building dataset (UJIIndoorLoc) is added later for foundation-model
  pretraining diversity, a building head would need to be added at that
  point — not needed now.
- **2026-07-13** — TTA buffer/session structure (Item 4) confirmed:
  reference buffer = Layout 1 samples at/near a given RP; current/deployment
  buffer = Layout 2 or Layout 3 samples at that RP; buffers reset at layout
  boundaries and never span across layouts or across RPs.
- **2026-07-13** — UJIIndoorLoc confirmed as the cross-site diversity
  dataset, explicitly deferred to a later phase (Phase 6+). Not in scope
  for Phase 1-5.

## Phase 0 findings

### Item 1 — Is Layout 1 vs Layout 3 a different physical site?

**Resolved: same site.** Per the source paper: all three layouts are the
ground/1st/2nd floor corridors of Wing C, Faculty of Engineering, Multimedia
University, Cyberjaya, Malaysia. Same 17 WiFi APs, same 42 BLE beacons, same
384 reference points (RPs) on a 1m grid, same floor pitch (2.7m + 0.8m floor
thickness = 3.5m, which is exactly the collaborator code's assumed
`FLOOR_HEIGHT_M = 3.5` — validated, not just assumed). The only difference
between layouts is partition boards added as RF obstructions: Layout 1 = no
boards (baseline), Layout 2 = boards on ground + 2nd floor only, Layout 3 =
boards on every floor (most severe).

**Implication:** HDLC does NOT provide cross-site diversity for
foundation-model-style pretraining (Phase 6 goal). A separate dataset
(UJIIndoorLoc recommended — 3 real buildings, native (building, floor, x, y)
labels, reusable from the prior project's format) is still needed for that.

**New implication (upside):** Layout 1 -> Layout 2/3 is a *real*,
physically-caused RSSI drift, not a synthetic one. The source paper's own
baselines (KNN/DT/RF/MLP) show floor accuracy >=99% with no layout change,
degrading with real, physically-grounded errors (up to ~3m worse average
positioning error) under layout change. This makes HDLC a strong candidate
for Phase 3's drift simulator requirement — arguably better than a
synthetic drift injector, since the degradation is empirically real rather
than designed to look real.

### Item 2a — `resolve_ap_positions` prints an AP-count ratio >100% (59/17, 347%)

**Resolved: not a parsing bug, a scope decision.** The dataset genuinely
publishes ground-truth (x,y) positions for 59 signal sources: 42 BLE beacons
+ 17 WiFi APs (row 1 of the raw CSV correctly parses as an AP/beacon
position table for all 59). The collaborator's loader restricts
`rssi_cols` to `WAP`-prefixed columns only (17), so `ap_positions` (59
entries) legitimately exceeds `rssi_cols` (17) — hence 347%. All downstream
lookups are scoped by iterating `rssi_cols`, so the extra BLE entries in the
dict are inert, not a correctness bug. Given the WiFi-only decision above,
this print is expected and can be left as-is or annotated; no fix required
unless BLE fusion is added later (at which point `rssi_cols` should include
BLE columns and the ratio would read closer to 100%).

### Item 2b — First 5 test samples show identical GT (x,y) = (0,0) in both layouts

**Resolved via raw CSV inspection — confirmed file-ordering artifact, not a
label leak.** Checked `Layout 1 - Raw - Testing.csv` directly:
- Rows are ordered in bursts of exactly 10 samples per reference point,
  starting at point (Floor=0, x=0, y=0) — the first 10 rows are literally
  `(0,0,0)` repeated, then 10 rows of `(0,0,1)`, etc.
- 3840 data rows / 384 distinct points = 10 samples/point in the test file
  (matches the printed "[Layout 1] 3840/3840 rows").
- `Layout 1 - Raw - Training.csv` has 7950 data rows / 384 points ~= 20.7
  avg, consistent with 20 training + 10 testing = 30 samples/point (matches
  the source paper's stated 30 samples per RP).
- The eval loop uses `shuffle=False` (`DataLoader(..., shuffle=train_mode)`
  with `train_mode=False`), so it reads the file in this burst order — the
  first 5 printed samples are always the first 5 of the 10 repeated
  readings at point (0,0,0). This fully explains the symptom without
  invoking the historical leak the code comments describe (which is
  already fixed via `filter_locatable_rows` / the NaN-instead-of-(0,0)
  fallback in `build_virtual_space`).

### NEW — Official train/test split is row-level, not grouped by point

**Found while verifying Item 2b, not one of the original four items, but
directly load-bearing for every reported number so far.**

Checked distinct (Floor, x, y) points in `Layout 1 - Raw - Training.csv` vs
`Layout 1 - Raw - Testing.csv`:
- Training: 384 distinct points. Testing: 384 distinct points.
- Points in both: **384 (100%)**. Points only in testing: **0**.

Every physical reference point appears in both the "training" and "testing"
files — the official split just holds out a different subset of the 30
repeated readings per point (20 train / 10 test), not a subset of
*locations*. This means:
- The source paper's own KNN/DT/RF/MLP baseline numbers, and the
  collaborator's CNN numbers (median xy error 0.61m / 0.68m on Layout
  1/3 test), reflect **same-point interpolation** (the model has seen every
  location during training, just not that exact reading) — not spatial
  generalization to unseen locations.
- This directly violates the prior project's grouped-split discipline
  (splits must be grouped by physical point, never row-level).
- **Action for Phase 1:** build our own grouped (leave-points-out) split by
  physical RP for any headline number we report, and be explicit that the
  dataset's official "Testing" files and the paper's/collaborator's
  published numbers are same-point interpolation results, not comparable
  to a genuine spatial-generalization claim. Both regimes can still be
  reported side by side (labeled honestly) since the official split is
  useful for sanity-checking against the paper's baselines.

### Item 2a (sentinel value) — bonus finding, not one of the original items

The collaborator's code assumes a UJIIndoorLoc-style "+100 = not detected"
sentinel (`NO_SIGNAL_SENTINELS = {100, 100.0}`). HDLC's actual convention
(confirmed in the paper) is **-110 dBm for undetected/out-of-range signals**.
The printed run log confirms 0.0% sentinel matches — the sentinel-to-NaN
step never fires for this dataset. It currently works anyway *by
coincidence*: `RSSI_MIN = -100.0` sits between -110 and real readings, so
`rssi_row > RSSI_MIN` correctly excludes -110 as "not visible" everywhere it
matters (`select_k_strongest`, `additive_noise`). Fragile, not broken.
**Action for Phase 1:** make the -110 sentinel explicit rather than relying
on threshold overlap.

## Phase 0 status: closed

All four original open items, plus the row-level-split finding discovered
along the way, are resolved and confirmed (see Decisions log and Phase 0
findings above).

## Phase 1 status: done (data loading & grouped split)

Implemented in `src/data_loading.py`, `src/splits.py`, `src/naive_baseline.py`,
runnable end-to-end via `scripts/phase1_report.py`.

- **Loader** (`load_raw_layout_csv`): WiFi-only (17 WAP columns), robust to
  the raw CSVs' messy headers (`WAP 8 ` trailing space, `WAP13` no space,
  UTF-8 BOM on the Testing file) via a `^WAP\s*\d+$` regex match after
  stripping. Applies the explicit `-110.0` dBm sentinel fix (Phase 0 item
  2a bonus finding) instead of the original `+100` assumption. Confirms
  100% ground-truth WAP position coverage on load (asserts if it ever
  drops below that for a file HDLC hands us).
- **Grouped split** (`grouped_split`): `GroupShuffleSplit` by physical
  `(floor, x, y)` point, train/val/test 70/15/15, with
  `assert_disjoint_point_groups` run automatically and printed every call.
  Verified: 268/58/58 distinct points, zero overlap.
- **Official-split diagnostic** (`official_split_point_overlap`): now
  computed in code, not just by hand — reconfirms 384/384 (100%) point
  overlap between the dataset's own Training/Testing files every run.
- **Naive constant baseline** (`naive_constant_baseline`): predicts the
  training mean (x, y) and the training majority-class floor for every
  eval row.

### Naive baseline results (Layout 1, run 2026-07-13)

| Split | Median xy err | Floor accuracy |
|---|---|---|
| Official (row-level) | 12.03 m | 0.375 |
| Grouped (leave-points-out) | 11.02 m | 0.320 |

Reassuring: the grouped split's naive baseline is not meaningfully easier
than the official split's, so the row-level leakage in the official split
isn't doing the naive baseline's work for it. The real leakage risk is a
*trained* model memorizing per-point RSSI patterns — Phase 2 will need to
show a real gap between official-split and grouped-split performance if
the row-level leakage is actually inflating results, which is the whole
point of building the grouped split. **Any Phase 2 model must beat both of
these numbers, on both splits, loudly, or something is wrong.**

Also of note: ~49% of WAP-cell readings in Layout 1 are the -110 dBm
sentinel (not detected) — expected for a 17-AP deployment along a long,
partitioned corridor, not a data quality issue.

## Phase 2 status: done (base foundation network, honest baseline beaten)

Implemented in `src/virtual_space.py`, `src/dataset.py`, `src/model.py`,
`src/train.py`, run via `scripts/phase2_train.py`. Ported the collaborator's
GlobLoc CNN+SPP architecture as-is (5-block conv trunk, spatial pyramid
pooling, shared 128-d bottleneck, xy regression head + floor classification
head), WiFi-only, `torch.set_num_threads(1)` (measured ~3.5x speedup on
this workload — tiny per-sample images make multi-threaded op dispatch
overhead dominate over actual compute).

**Trained only on the Phase 1 grouped split** (train=8220, val=1790,
test=1780 rows across 268/58/58 disjoint physical points) — the official
row-level split is not used for training or model selection, per the
Phase 1 finding. 10 epochs, same hyperparameters as the collaborator's
original config (k_strongest=5, margin_m=2.0, max_extent_m=10.0, lr=1e-4,
pseudo_batch=16, lambda_floor=1.0), took 1370.7s (~23 min) on CPU.

### Result (2026-07-13), grouped test set (leakage-free spatial holdout)

| | Median xy err | Floor accuracy | Median 3D err |
|---|---|---|---|
| Naive constant baseline | 11.02 m | 0.320 | — |
| **Model** | **0.85 m** | **0.784** | 1.05 m |

Model clearly beats the naive baseline on both axes, by a wide margin — no
warning triggered. Full training curve and results in
`outputs/phase2/history.json` and `results.json`.

**Honest note on the grouped- vs official-split gap:** the collaborator's
original run (trained AND tested on the contaminated official/row-level
split) reported median xy error 0.61-0.68m. This model, trained and tested
on the leakage-free grouped split, gets 0.85m — worse, as expected, but by
a modest ~0.2-0.3m, not an order of magnitude. This suggests the backbone
is learning a genuinely transferable RSSI-to-location mapping rather than
mostly memorizing per-point patterns, though the row-level leakage in the
official split is still real and the grouped-split number (0.85m) is the
one that should be cited as this project's result, not the collaborator's
original 0.61-0.68m.

## Phase 3 status: done (drift sanity check — real, not synthetic)

Implemented in `scripts/phase3_drift_eval.py`. Froze the Phase 2 model
(trained only on Layout 1) and evaluated it, unmodified, on the exact same
58 held-out physical points across Layout 1 (reference), Layout 2 (partial
obstruction — boards on ground + 2nd floor), and Layout 3 (full obstruction
— boards on every floor). Using identical points across conditions isolates
layout-change effect from a "some points are just harder" confound.

### Result (2026-07-13)

| Condition | n | Median xy err | Floor acc | Median 3D err |
|---|---|---|---|---|
| Layout 1 (reference) | 1780 | 0.85 m | 0.784 | 1.05 m |
| Layout 2 (partial obstruction) | 1740 | 0.89 m | 0.728 | 1.23 m |
| Layout 3 (full obstruction) | 1740 | 0.81 m | 0.653 | 1.42 m |

**Honest, mixed result — not a clean monotonic story on every metric:**
- **Floor accuracy degrades monotonically and substantially** (78.4% ->
  72.8% -> 65.3%) with severity, matching the source paper's own finding
  that Layout 3 (all floors obstructed) hurts more than Layout 2 (2 of 3
  floors obstructed).
- **Combined 3D error degrades monotonically** (1.05m -> 1.23m -> 1.42m),
  driven mostly by the floor-accuracy component.
- **Pure xy error does NOT degrade monotonically** (0.85m -> 0.89m ->
  0.81m — Layout 3 is not worse than Layout 1 on xy alone). Plausible root
  cause, not yet independently verified: partition boards primarily
  attenuate cross-floor signal propagation (which floor classification
  depends on to tell floors apart), while within-floor relative AP
  geometry — what the xy regression head actually uses — is less
  perturbed by localized obstructions on the same floor.

**Verdict: usable as a drift testbed, with a caveat.** The floor-accuracy /
combined-3D-error degradation is real, substantial, and physically
explainable — good enough to drive Phase 4/5 hypernetwork TTA correction
and to make "does TTA help" a meaningful question. But claims should be
scoped to floor accuracy / combined 3D error, not to xy error specifically,
until/unless the xy-error flatness is investigated further.

Raw results: `outputs/phase3/drift_results.json`.

## Phase 4 prerequisite: oracle diagnostic — NEGATIVE RESULT, do not proceed

Implemented in `scripts/phase4_oracle_diagnostic.py`, directly porting the
prior project's design (`wifi_tta/src/hypernetwork.py`'s
`compute_buffer_stats`, and the CSI pipeline's three-way comparison
documented in `wifi_tta/CSI_PIVOT_PLAN.md`). Reference-relative buffer
stats (current vs. reference buffer at the same physical point, per Item
4's confirmed design): per-WAP detection-rate difference, mean
scaled-RSSI difference, and current absolute detection rate (51-dim = 3 x
17 WAPs). Reference buffer = that point's Layout 1 readings; current
buffer = that point's Layout 2 or Layout 3 readings. Target = frozen
Phase 2 model's per-buffer combined 3D error (median over a 10-sample
bootstrap buffer, 5 bootstrap draws x 2 layout conditions x 116 points =
1160 instances). Restricted to the 116 points held out from Phase 2
training (val+test) so the error target isn't contaminated by the base
model having memorized those points.

### Result (2026-07-13)

| Predictor | R^2 |
|---|---|
| Leave-points-out RandomForest (the real check) | **-0.070** |
| Random (non-grouped) RandomForest | 0.538 |
| Leave-points-out Ridge (linear) | -0.098 |
| Point-identity ALONE (oracle, deliberately unfair) | 0.538 |

**Clear negative result — worse than the prior project's CSI finding, not
just a repeat of it.** Leave-points-out R^2 is negative for both RF and
Ridge: no transferable signal, linear or nonlinear. The non-grouped RF's
R^2 (0.538) is essentially identical to the point-identity-alone oracle's
R^2 (0.538) — the entire apparent signal in the non-grouped number is
just the RF re-deriving point identity from near-duplicate bootstrap
buffer-stat vectors leaking across train/test folds. This is the exact
confound pattern the non-negotiable discipline exists to catch, and it is
unambiguous here (not a close call like the CSI pipeline's R^2 0.22 vs
0.375).

**Root cause (not hand-waved):** in the prior UJIIndoorLoc project, drift
was spatially generic (temporal/session drift that could occur similarly
at any physical location), so a location-agnostic "drift signature" was
at least a coherent thing to try to learn. Here, drift is partition
boards bolted at fixed corridor locations — a board's RSSI effect is
inherently local: points near a given board see a fundamentally different
signature than points far from it. The drift pattern is a function of
location by the physical structure of the experiment itself, not
separable from location identity the way calibration drift is. This is a
structurally harder case than the CSI pipeline's (which had real but
non-transferable signal); here there may be no location-agnostic drift
signature to find at all in this dataset.

**Decision: do not proceed to hypernetwork training on this data as-is.**
Per the non-negotiable discipline, this is a valid, reportable outcome,
not something to force past by tuning. Raw results:
`outputs/phase4/oracle_diagnostic.json`.

## Pivot: multi-site foundation-model expansion (2026-07-13)

Decided to build toward a genuine multi-site backbone (the actual
"foundation model" step) BEFORE building the hypernetwork, rather than
after, for two reasons: (1) the hypernetwork targets the specific final
layer of whatever backbone ships -- building it against a throwaway
single-site backbone risks having to redo it once the backbone changes;
(2) Phase 4's negative result looks like a property of HDLC's specific
drift mechanism (partition boards fixed at physical locations, so drift
is entangled with location by construction) rather than a property of
single-site data per se -- worth checking a spatially-generic drift
mechanism (matching what worked, partially, in the prior project) before
concluding TTA can't work here at all.

**Also corrected here, since it was stated imprecisely earlier in this
conversation:** the prior project's UJIIndoorLoc hypernetwork pipeline
does NOT use real longitudinal drift. It injects SYNTHETIC drift (random-
walk RSSI attenuation + ramping AP dropout, reset at session boundaries;
`wifi_tta/src/drift_simulator.py`) onto its own test split. A genuine
~3-month-later `validationData.csv` (confirmed: train timestamps May-June
2013, validation timestamps September 2013) exists in that dataset but is
unused by the current pipeline. The likely reason UJIIndoorLoc's oracle
check succeeded partially (ridge R^2~0.25, reference-relative stats, per
`wifi_tta/src/hypernetwork.py`'s docstring) is that the injected drift is
spatially generic by construction, not tied to this dataset specifically
-- the lesson to port forward is "use a location-independent drift
mechanism," not "use UJIIndoorLoc."

### Multi-site audit

Inventoried what's already downloaded from the prior project
(`wifi_tta/data*`) plus HDLC. Verified (not assumed) split integrity,
AP-position availability, and sentinel convention per site:

| Site | Floors | Official split | AP positions | Sentinel |
|---|---|---|---|---|
| hdlc | 3 | row-level (own split built, Phase 1) | ground-truth | -110 |
| sod_cetc331 | 3 | **row-level, 100% overlap** (own split needed, like HDLC) | ground-truth (xlsx) | +100 |
| sod_hcxy | 1 | genuine spatial holdout (0% overlap) | ground-truth (xlsx) | +100 |
| sod_syl | 1 | ~genuine (1% overlap) | ground-truth (xlsx) | +100 |
| uji_b0 (held out) | 4 | n/a, never trained on | estimated (not published) | +100 |
| uji_b1 | 4 | own grouped split (by session, validated by prior project) | estimated | +100 |
| uji_b2 | 5 | own grouped split (by session, validated by prior project) | estimated | +100 |

UJIIndoorLoc building richness (session count = how many independent
capture routes, i.e. how much grouped-split diversity exists): building 0
= 2 sessions (5249 rows, 4 floors) -- too thin for pretraining, which is
exactly why it's the cheap choice to hold out entirely for zero-shot
testing. Building 1 = 12 sessions, building 2 = 16 sessions (richest,
matches why the prior project picked it as their single-building choice).

**Site allocation decided:** pretrain on hdlc + sod_cetc331 + sod_hcxy +
sod_syl + uji_b1 + uji_b2 (6 sites); hold out uji_b0 entirely for
zero-shot transfer evaluation.

### Phase 5a: multi-site loading + grouped-split infrastructure -- done

Implemented in `src/sites.py` (per-site config/loaders, common schema:
`site_id, floor_id, x, y, _group, <wap_cols...>`), `src/splits.py`'s new
`grouped_split_by_column` (generic version of Phase 1's grouped_split, group
key varies per site: physical point for hdlc/sod_*, capture session for
uji_*), validated via `scripts/phase5a_multisite_split_report.py`.

**Key design point:** all 7 sites share ONE unified RSSI scale
(RSSI_MIN=-104, RSSI_MAX=0, matching SODIndoorLoc/UJIIndoorLoc's native
convention) so the shared virtual-space image encoding has one consistent
"detected vs not" threshold. HDLC's own single-site pipeline
(`data_loading.py`, Phase 1-4 results) is untouched and still uses its
local (-100,-30) scale -- `sites.py` re-derives HDLC independently on the
unified scale rather than reusing `load_raw_layout_csv`'s output, since
mixing scales would have silently misclassified HDLC's not-detected
readings as detected under a shared threshold. Heterogeneous AP counts
per site (17 to 520) are NOT a problem for the shared CNN: each row's
virtual-space image is built from just that row's k-strongest APs, so the
backbone never sees a fixed-width AP vector.

### Result (2026-07-13)

| site | rows | groups | floors | APs (positioned) | train/val/test | naive xy err | naive floor acc |
|---|---|---|---|---|---|---|---|
| hdlc | 11790 | 384 | 3 | 17/17 | 8220/1790/1780 | 11.02m | 0.320 |
| sod_cetc331 | 1795 | 955 | 3 | 52/52 | 1277/254/264 | 14.85m | 0.451 |
| sod_hcxy | 12230 | 465 | 1 | 56/56 | 8410/1940/1880 | 39.22m | 1.000 |
| sod_syl | 9900 | 397 | 1 | 46/46 | 6880/1500/1520 | 20.86m | 1.000 |
| uji_b1 | 5159 | 12 | 4 | 169/520 | 3264/980/915 | 59.53m | **0.000** |
| uji_b2 | 9454 | 16 | 5 | 153/520 | 5318/1690/2446 | 36.75m | 0.251 |

Disjointness assertions passed for all 6 sites. HDLC's naive numbers
exactly match Phase 1's (11.02m / 0.320) -- a good sanity check that the
unified-scale refactor didn't silently change anything for HDLC.

**Honest caveats, not smoothed over:**
- `sod_cetc331` has only ~1.9 samples per physical point on average (955
  rows for 955 groups seen in the raw audit) -- far sparser than HDLC's
  10-30/point bursts. Fine for backbone regression, but there's no natural
  multi-sample "buffer" per point here for TTA purposes later.
- `sod_hcxy`/`sod_syl` naive floor accuracy = 1.000 is trivial, not a real
  signal -- both sites have exactly one floor in this data, so "always
  guess the only floor" is tautologically perfect. Floor accuracy isn't a
  meaningful metric on single-floor sites.
- `uji_b1`'s naive floor accuracy = **0.000** is a real, honest finding,
  not a bug: with only 12 total capture sessions, a single grouped split
  can easily put every session of the train-majority floor into train and
  none into test by chance. This is the same small-group-count fragility
  the prior project's own `cv_hypernetwork.py` was built to average out
  (rotating all 12 sessions through K-fold CV rather than trusting one
  split). Point-grouped sites (hdlc, sod_*) have hundreds of groups and
  don't have this problem; session-grouped sites (uji_b1, uji_b2) do, and
  any headline number on them needs multi-seed averaging before it means
  anything -- consistent with the project's own >=10-seed discipline for
  deployment results, just showing up earlier than expected, at the
  backbone-training stage rather than only at TTA deployment evaluation.

### Phase 5b: multi-site model + training -- done

Implemented in `src/multisite_dataset.py`, `src/multisite_model.py`
(shared trunk + xy_head, per-site `floor_heads` ModuleDict), `src/multisite_train.py`,
run via `scripts/phase5b_multisite_train.py`. Headline result: **zero-shot
xy transfer to uji_b0 (never trained on) beat its naive baseline by ~27x
(1.51m vs 41.80m)** -- the core foundation-model premise held up. Per-site
xy regression beat naive everywhere. Floor classification was mixed: fine
for hdlc/sod_cetc331, trivial for the single-floor SOD sites, weak for
uji_b1 (14.9%, below chance) and uji_b2 (24.6%, failed to beat its own
25.1% naive baseline -- a real warning, not swept under the rug).

**Bug found and fixed during this phase:** `filter_locatable_rows` checked
whether the row's raw top-k-by-strength APs included a positioned one, but
training augmentation (noise, random AP-dropping) is applied AFTER that
check. For UJIIndoorLoc (only ~30% of APs have estimated positions, vs
100% for HDLC/SODIndoorLoc), augmentation could shift the top-k away from
any positioned AP, crashing `build_virtual_space`. Fixed at the root:
`build_virtual_space` now selects the k-strongest AMONG POSITIONED APs
directly (not "raw top-k, then filter"), `random_dropping` protects a
row's last remaining positioned-visible AP from being dropped, and
`MultiSiteDataset.__getitem__` falls back to the unaugmented row if a rare
edge case still slips through -- rather than crashing a multi-hour run.
No-op for 100%-coverage sites (HDLC, SODIndoorLoc); verified via a 2-epoch
stress test on the full UJI pool before relaunching the real run.

### Regression investigation + literature benchmarking (2026-07-14)

User asked to investigate the HDLC floor-accuracy regression, improve
results generally, and benchmark against published work before going
further.

**Diagnosis, corrected from the initial guess:** with plain
concatenation+shuffle, HDLC already saw its full 11,790 training rows
every epoch (not literally under-represented in absolute terms) --
`sod_cetc331` (1,795 rows) was the one structurally starved. The more
likely mechanism for HDLC's regression is shared-trunk interference from
5 other sites' gradients, combined with genuinely fewer total epochs (5)
than Phase 2's dedicated 10.

**Fix implemented:** `multisite_train.build_site_balanced_sampler` --
oversamples every site up to the LARGEST site's row count (target = max,
not mean), so no site's per-epoch exposure ever shrinks relative to plain
concatenation; small sites get boosted instead. Combined with 8 epochs
(vs. 5) in `phase5b_v2`.

**Literature benchmark** (Nature Sci. Reports comparative study, XGBoost,
identical hyperparameters across every UJIIndoorLoc/SODIndoorLoc building
-- https://pmc.ncbi.nlm.nih.gov/articles/PMC12494710/):

| site | lit. MAE | lit. floor acc | ours v1 xy/floor | ours v2 xy/floor |
|---|---|---|---|---|
| sod_cetc331 | 0.85m | 98.9% | 2.12m / 64.8% | 1.91m / **91.7%** |
| sod_hcxy | 1.60m | -- | 1.88m / 1.000 (trivial) | 1.90m / 1.000 |
| sod_syl | 2.20m | -- | 2.44m / 1.000 (trivial) | 2.19m / 1.000 |
| uji_b1 | 7.85m | 74.9% | 1.52m / 14.9% | 1.05m / 30.5% |
| uji_b2 | 8.15m | 93.0% | 1.02m / 24.6% (FAILED naive) | 0.74m / **33.7%** |
| uji_b0 (zero-shot) | 4.54m (trained!) | 96.8% | 1.51m | 1.14m |

**Important caveat on the xy comparison, not just a victory claim:** our
virtual-space method builds each sample's image from a small LOCAL window
anchored to nearby AP positions, then regresses position relative to that
window -- a strong locality prior blind global-regression methods (what
the literature benchmark uses) don't get. This is almost certainly why our
UJI xy numbers look so much better than literature's (1.0-1.5m vs 7.8-8.2m)
-- it likely isn't the same task. HDLC/SOD's dense, 100%-positioned AP
layouts make this effect much smaller, so those xy comparisons (roughly
competitive: 1.9-2.2m vs 0.85-2.2m) are more meaningful.

**On floor accuracy the gap is real, not a methodology artifact**, and
v2's balanced sampling closed most of it for `sod_cetc331` (91.7% vs
98.9%) and meaningfully improved `uji_b2` (33.7%, now clearing its own
naive baseline) and `uji_b1` (30.5%, no longer below chance) -- but a
large gap remains for both UJI buildings vs. literature (74.9%/93.0%),
and **HDLC barely moved (57.6% -> 59.3%), nowhere near Phase 2's dedicated
78.4%** despite now having comparable total training exposure. This
persisting gap despite fixing the exposure imbalance is itself informative:
it points to genuine shared-trunk interference across 6 diverse sites, not
an under-training artifact -- which is exactly the case for per-site
fine-tuning rather than more joint-training epochs.

### Phase 5c: per-site fine-tuning -- done, closes most of the gap

Implemented in `scripts/phase5c_finetune.py`. For each site: fresh copy of
the v2 joint checkpoint, `xy_head` frozen (`requires_grad=False`), trunk +
that site's own `floor_head` fine-tuned for 5 epochs at LR=3e-5 (v2's base
LR / ~3). The v2 checkpoint itself is untouched and remains the
"foundation model" artifact for new/zero-shot sites; each site gets its
own specialized `model_<site_id>.pt`.

### Result (2026-07-14): v2 (joint) -> v3 (fine-tuned) -> literature

| site | v2 xy/floor | v3 xy/floor | naive floor | literature MAE/floor |
|---|---|---|---|---|
| hdlc | 0.91m / 0.593 | **0.81m / 0.706** | 0.320 | (Phase 2 single-site: 0.85m/0.784) |
| sod_cetc331 | 1.91m / 0.917 | **1.59m / 0.932** | 0.451 | 0.85m / 98.9% |
| sod_hcxy | 1.90m / 1.000 | 1.79m / 1.000 | 1.000 (trivial) | 1.60m / -- |
| sod_syl | 2.19m / 1.000 | **1.91m** / 1.000 | 1.000 (trivial) | 2.20m / -- |
| uji_b1 | 1.05m / 0.305 | 0.70m / 0.370 | 0.000 | 7.85m / 74.9% |
| uji_b2 | 0.74m / 0.337 | 0.50m / 0.372 | 0.251 | 8.15m / 93.0% |

Fine-tuning improved BOTH xy and floor accuracy on every single site, with
zero exceptions -- a clean, unambiguous win from this stage.

**HDLC's regression substantially narrowed** (floor acc 59.3% -> 70.6%,
closing most of the gap to Phase 2's dedicated 78.4%; xy 0.91m -> 0.81m,
now close to Phase 2's 0.85m). Confirms the earlier diagnosis: the residual
gap was shared-trunk interference from joint multi-site training, and
site-specific fine-tuning recovers most (not quite all) of the
dedicated-model quality.

**sod_cetc331 and sod_syl are now genuinely competitive with the
literature benchmark** (93.2% floor acc vs. 98.9%; sod_syl's 1.91m median
xy is even below literature's 2.20m MAE, though median-vs-MAE isn't a
strictly fair comparison).

**UJI floor classification remains substantially behind literature even
after fine-tuning (37.0%/37.2% vs. 74.9%/93.0%), and this is a real,
likely-structural limitation, not something more training fixes.** Two
compounding, already-identified causes: (1) UJIIndoorLoc building floors
are known (documented by the prior project) to share near-identical (x,y)
footprints -- floors stack almost directly on top of each other, so a
purely spatial virtual-space encoding may structurally lack the
information to discriminate them the way raw-RSSI-vector methods
(literature's XGBoost) can; (2) only ~30% of UJI's 520 APs have estimated
(not ground-truth) positions, sharply limiting how much of the RSSI signal
even makes it into the virtual-space image at all, unlike HDLC/SOD's
100%-positioned, dense AP layouts. This reads as a genuine mismatch
between the virtual-space representation and UJI's sparse/large-scale
deployment characteristics, worth stating plainly as a limitation of this
approach for that class of site rather than something to keep tuning.

**Zero-shot uji_b0 result is unaffected by this stage** (v2 checkpoint,
not fine-tuned) and remains the headline foundation-model number: 1.14m
vs. 41.80m naive baseline, on a building never seen during training.

### Model-selection bugfix + longer fine-tuning (2026-07-14)

Found while reviewing per-epoch fine-tune logs at the user's request:
checkpoint selection during both joint training AND fine-tuning picked the
epoch with the best **xy error only**, ignoring floor accuracy entirely.
For HDLC's fine-tune specifically, floor accuracy peaked at epoch 2
(81.3%) but a later, xy-better/floor-worse epoch got selected instead,
reporting 70.6%. Fixed in `multisite_train.summarize_records` (now also
returns a pooled median combined-3D-error metric) and both
`train_model_multisite` and `phase5c_finetune.finetune_one_site` (now
select on that instead of xy alone). Re-ran fine-tuning for all 6 sites
with the fix, plus doubled epochs (5 -> 10) per the user's request to
give HDLC/sod_cetc331 more room to improve.

### Result (2026-07-14): v2 (joint) -> v3 5-epoch (xy-selected, prior) -> v3 10-epoch (3D-selected)

| site | v2 joint | v3 (5ep, xy-select) | v3 (10ep, 3D-select) | literature |
|---|---|---|---|---|
| hdlc | 0.91m / 59.3% | 0.81m / 70.6% | **0.89m / 74.5%** | (Phase 2: 0.85m/78.4%) |
| sod_cetc331 | 1.91m / 91.7% | 1.59m / 93.2% | 1.58m / 93.2% | 0.85m / 98.9% |
| sod_hcxy | 1.90m / 100% | 1.79m / 100% | 1.86m / 100% | 1.60m / -- |
| sod_syl | 2.19m / 100% | 1.91m / 100% | **1.53m** / 100% | 2.20m / -- |
| uji_b1 | 1.05m / 30.5% | 0.70m / 37.0% | **0.67m / 41.2%** | 7.85m / 74.9% |
| uji_b2 | 0.74m / 33.7% | 0.50m / 37.2% | **0.50m / 42.9%** | 8.15m / 93.0% |

HDLC's floor accuracy closed further (74.5% vs. Phase 2's dedicated
78.4% -- gap now 3.9pp, down from the original 19.1pp). sod_syl's xy beat
literature's MAE outright (1.53m vs. 2.20m).

**Correction to an earlier claim in this document, not glossed over:**
Phase 5c v1 called UJI's floor-accuracy gap "a real ceiling... likely
structural... more training unlikely to close it," based on a 5-epoch
curve that looked flat. That conclusion was premature and is now
contradicted by evidence: with 10 epochs, both `uji_b1` (30.5% -> 41.2%)
and `uji_b2` (33.7% -> 42.9%) kept climbing, and neither curve had
plateaued even at epoch 10 (`uji_b2`: 39.6, 40.4, 41.7, 41.9, 43.6, 44.2,
44.0, 44.9, 48.5, 48.8 -- still rising at the last epoch). More training
does keep helping here; the earlier "structural, not fixable" framing was
wrong. What's still true and worth keeping: UJI's ~30% AP-position
coverage and the near-identical cross-floor (x,y) footprint (both
documented, not new claims) plausibly explain why progress is SLOW and
why a large gap to literature (74.9%/93.0%) remains even after 10 epochs
-- but "slow" is not the same claim as "capped," and that distinction
matters for anyone deciding whether to keep training these sites further.

### Continued UJI fine-tuning (2026-07-14)

Since both UJI sites were still visibly climbing at epoch 10, continued
fine-tuning from the phase5c_v2 checkpoints for 15 more epochs each
(`scripts/phase5c_uji_continue.py`, same setup: xy_head frozen, LR=3e-5,
combined-3D-error model selection). Only `uji_b1`/`uji_b2` -- the other 4
sites showed no comparable still-climbing signal at epoch 10 (sod_hcxy/syl
trivial, sod_cetc331 flat, hdlc oscillating rather than trending).

| site | phase5c_v2 (10ep) | +15 more epochs | naive | literature |
|---|---|---|---|---|
| uji_b1 | 0.67m / 41.2% | 0.67m / 46.0% | 0.000 | 74.9% |
| uji_b2 | 0.50m / 42.9% | **0.35m / 57.9%** | 0.251 | 93.0% |

`uji_b2` improved substantially and its val-floor-accuracy curve climbed
smoothly and monotonically across all 15 additional epochs (50.9% ->
62.9%, before test-time evaluation on the best-selected checkpoint gave
57.9%) -- the gap to literature more than halved from where Phase 5c v1
started (68.4pp -> 35.1pp). `uji_b1` improved more modestly (41.2% ->
46.0%) and its val curve got visibly noisier this round (fluctuating in
the 53-59% band rather than climbing cleanly) -- plausibly because it only
has 12 total capture sessions to group-split on (the small-group
fragility flagged in Phase 5a), so further gains there may need a lower
learning rate or more sessions rather than just more epochs at the same
setup, unlike uji_b2 which still looks like it has clean room to keep
improving.

### Early-stopping fine-tune, both sites converged (2026-07-14)

Implemented proper patience-based early stopping
(`finetune_one_site_early_stop` in `scripts/phase5c_finetune.py`:
patience=6 epochs, min_delta=0.005m on the combined 3D validation metric,
max_epochs=40 safety cap) instead of hand-picking epoch counts. Continued
from the phase5c_uji_continue checkpoints: `uji_b2` at the same LR (3e-5,
its curve was climbing cleanly), `uji_b1` at a reduced LR (1e-5, since its
curve had gotten noisy rather than converging under the standard rate).

Both runs triggered genuine early stopping this time (did not hit the
max_epochs cap) -- the first results in this investigation that represent
real convergence rather than a time-budget-limited snapshot.

| site | converged at epoch | final xy/floor | naive floor | literature floor |
|---|---|---|---|---|
| uji_b1 | 22 (of 28 run) | 0.58m / **52.1%** | 0.000 | 74.9% |
| uji_b2 | 10 (of 16 run) | 0.27m / **66.9%** | 0.251 | 93.0% |

The lower LR for `uji_b1` worked as intended -- floor accuracy climbed
steadily to 72.2% on validation before plateauing/oscillating, well past
where the noisy standard-LR run had gotten stuck.

### Full uji_b1/uji_b2 trajectory across this session, for the record

| stage | uji_b1 floor acc | uji_b2 floor acc |
|---|---|---|
| v2 (joint, no fine-tune) | 30.5% | 33.7% |
| v1 fine-tune (5ep, buggy xy-only selection) | 37.0% | 37.2% |
| v2 fine-tune (10ep, fixed 3D-metric selection) | 41.2% | 42.9% |
| +15 more epochs (same LR) | 46.0% | 57.9% |
| early-stop, tuned LR, converged | **52.1%** | **66.9%** |

Net gain from the original joint backbone: uji_b1 +21.6pp, uji_b2 +33.2pp.
Remaining gap to literature: uji_b1 22.8pp (was 44.4pp), uji_b2 26.1pp
(was 59.3pp) -- both gaps cut roughly in half or better over the course of
this investigation, through legitimate training/tuning fixes (model-
selection bug, epoch budget, per-site learning rate), not by touching the
eval methodology.

## Phase 5d: error analysis (2026-07-14)

Implemented in `scripts/phase5d_error_analysis.py`: per-site spatial error
heatmaps, floor confusion matrices, and a pooled check of the "sparse
AP-position coverage drives error" hypothesis. Uses each site's best
available checkpoint (early-stopped for uji_b1/uji_b2, phase5c_v2 for the
rest). Figures in `outputs/phase5d_error_analysis/`.

**AP-sparsity hypothesis: NOT well supported, walking this back.**
Correlation between xy error and how many positioned APs were visible in
a given reading is weak everywhere (|r| < 0.2 for all 6 sites, some sites
even slightly negative as expected but barely so). This was one of the
two explanations offered earlier for UJI's weaker floor performance --
the per-row evidence doesn't support it as a row-level effect. Whatever
role AP sparsity plays, it isn't "fewer positioned APs in this specific
reading directly predicts worse error."

**Floor confusion is physically sensible where it exists.** HDLC confuses
ADJACENT floors far more than distant ones (floor 0<->1: 131/10, floor
1<->2: 76/174, vs. floor 0<->2: 52/11) -- consistent with signal
attenuating through one floor's thickness being a smaller effect than
through two. sod_cetc331's confusion matrix is nearly clean (only 18
off-diagonal out of 264), matching its 93%+ accuracy.

**Major finding, not merely a modeling weakness: both UJI sites are
missing an entire floor class from their TEST split**, a direct
consequence of the coarse session-level grouping (12/16 total groups --
the same fragility flagged back in Phase 5a). `uji_b1`'s test set has
ZERO true floor-2 samples (confirmed: confusion matrix row sums to 0 out
of ~915 rows), yet the model still predicts floor-2 for 185 rows (20.2%
of test) -- every one of those is automatically wrong, with no true
floor-2 sample it could ever match. `uji_b2` has the identical issue for
floor-4, but far smaller (50/2446 rows, ~2%). This directly explains why
`uji_b2` closed its literature gap so much faster than `uji_b1`
throughout this session -- `uji_b1` has been fighting a ~20-point
structural handicap in its own test set the whole time, not just a harder
learning problem. **This is a split-methodology artifact inflating the
apparent literature gap, not purely a representational limitation** --
worth a floor-stratified grouped split as a follow-up fix before drawing
final conclusions about UJI floor-classification quality, though not
implemented in this pass.

**Spatial error clusters at the AP-coverage periphery, consistently
across floors.** `uji_b2`'s heatmap shows a clear, repeated hotspot at the
building's geographic tail (the bottom-right extremity of its corridor
shape) across floors 1, 2, and 3 alike -- an edge/extrapolation effect
(the model interpolates well within the dense core of AP coverage,
struggles at the boundary), not randomly distributed noise. HDLC's
heatmap (narrow single corridor) didn't show as clear a spatial pattern,
plausibly because there's less room for a true "periphery" to exist in
that layout.

## Bundled retrain: v3 (negative result), v4 (fix + recovery), second
## zero-shot site (2026-07-14/15)

Per the user's request, three fixes motivated by the Phase 5d error
analysis were bundled into one retrain to save wall-clock time: (1)
`stratified_grouped_split_by_floor` instead of plain grouped splitting,
(2) ordinal floor regression instead of classification, (3) wider virtual-
space margin/extent (2.0m/10.0m -> 3.0m/15.0m) to help peripheral-point
error. Flagged explicitly before running that bundling forfeits the
ability to attribute which change caused which effect.

### v3 result: negative, not a clean win

| site | v2 joint | v3 joint (bundled) |
|---|---|---|
| hdlc | 0.91m / 59.3% | 1.06m / 61.1% |
| sod_cetc331 | 1.91m / 91.7% | 1.71m / 88.3% |
| sod_hcxy | 1.90m / 100% | 2.13m / 100% |
| sod_syl | 2.19m / 100% | 2.64m / 100% |
| uji_b1 | 1.05m / 30.5% | 2.23m / 32.0% |
| uji_b2 | 0.74m / 33.7% | 1.26m / 19.3% |
| zero-shot uji_b0 | 1.14m | 3.16m |

Almost every site's xy error got worse, and the headline zero-shot metric
nearly tripled (1.14m -> 3.16m). A second, distinct bug also surfaced:
`stratified_grouped_split_by_floor` v1 guaranteed test/val floor coverage
unconditionally, which for `uji_b2` (only 16 groups, 5 floors) meant
floor 4's only 2 groups both went to test+val, leaving TRAIN with zero
floor-4 examples -- a worse failure mode than the one it was built to fix
(a floor absent from train guarantees wrong predictions whenever it's the
true answer; a floor merely absent from test just isn't scored).

Diagnosis: the wider margin/extent was the most likely dominant cause,
since it's the one change applied to literally every image regardless of
site or floor-scarcity issues, matching the broad, near-universal xy
degradation pattern -- not isolated to UJI or to floor-related sites,
which is what a split- or loss-related cause would look like instead.

### Fix: `stratified_grouped_split_by_floor` rewritten (train-safe)

Rewrote the function to guarantee TRAIN coverage for every floor first
(non-negotiable), and only pull a group into test/val if doing so
provably leaves every floor it touches with >=1 other group still in
train. Added a hard assertion that no floor can end up missing from
train. Verified against the exact case that broke before (`uji_b2`, 16
groups, 5 floors): train coverage now holds for every floor.

### v4 result: recovers from v3, mostly beats v2

Reverted margin/extent to v2's (2.0m, 10.0m); kept the (now train-safe)
stratified split and the ordinal floor loss.

| site | v2 | v3 (bad bundle) | v4 (fixed) |
|---|---|---|---|
| hdlc | 0.91m / 59.3% | 1.06m / 61.1% | 0.68m / 62.5% |
| sod_cetc331 | 1.91m / 91.7% | 1.71m / 88.3% | 1.78m / 91.5% |
| sod_hcxy | 1.90m / 100% | 2.13m / 100% | 2.17m / 100% |
| sod_syl | 2.19m / 100% | 2.64m / 100% | 1.81m / 100% |
| uji_b1 | 1.05m / 30.5% | 2.23m / 32.0% | 1.19m / 34.3% |
| uji_b2 | 0.74m / 33.7% | 1.26m / 19.3% | 0.51m / 16.9%* |
| zero-shot uji_b0 | 1.14m | 3.16m | 1.12m |

*`uji_b2`'s floor number isn't a clean regression from v2: v2's test set
was still missing floor-4 entirely (the Phase 5d bug), so v2's 33.7% was
never actually scored against that floor. v4's test set genuinely
includes floor-4 now (confirmed: "floors missing from test: none"), so
16.9% is a harder, more complete, more honest number, not directly
comparable to v2's on that specific axis.

Zero-shot transfer fully recovered and slightly improved (1.14m -> 1.12m),
confirming the margin/extent diagnosis: reverting just that one change
(while keeping the split and loss fixes, which are net-positive) restored
and then exceeded v2's quality almost everywhere.

### Second independent zero-shot site: Tampere (2026-07-15)

Held out entirely per the earlier decision to test generalization beyond
the UJIIndoorLoc family specifically. Evaluated the v4 checkpoint, never
trained on Tampere at all (`scripts/phase5b_zeroshot_eval.py`):

**1.62m median xy error vs. 30.21m in-site naive baseline.**

This is the more important of the two zero-shot results for the "is this
actually foundational" question raised earlier in this project: `uji_b0`
alone only tested generalization WITHIN the UJIIndoorLoc family (two of
its three buildings were already in the training pool). Tampere is a
genuinely different dataset -- different country (Finland vs.
Spain/Malaysia/China), different collection methodology (crowdsourced
across 21 devices vs. professional/session-based collection), and zero
representation in the pretraining pool. Two independent zero-shot
successes across two unrelated dataset families is meaningfully stronger
evidence of real transfer than one success within a single family.

### Phase 5c v4: fine-tuning -- best result of the project so far (2026-07-15)

Same protocol as v2/5c (freeze `xy_head`, fine-tune trunk + per-site floor
head, 10 epochs, combined-3D-metric selection), applied to the v4 joint
checkpoint.

| site | joint (v4) | fine-tuned (v4) | literature |
|---|---|---|---|
| hdlc | 0.68m / 62.5% | **0.64m / 74.9%** | (Phase 2: 0.85m/78.4% -- gap now 3.5pp, down from 19.1pp originally) |
| sod_cetc331 | 1.78m / 91.5% | **1.78m / 95.1%** | 0.85m / 98.9% |
| sod_hcxy | 2.17m / 100% | 1.71m / 100% | 1.60m / -- |
| sod_syl | 1.81m / 100% | 1.72m / 100% | 2.20m / -- (beats literature) |
| uji_b1 | 1.19m / 34.3% | 1.24m / 37.9% | 7.85m / 74.9% |
| uji_b2 | 0.51m / 16.9% | **0.22m / 33.1%** | 8.15m / 93.0% |

Every site improved floor accuracy; most improved xy too. This is the
best checkpoint set produced in this project across all iterations
(v1 through v4, plus the continued/early-stop UJI experiments). Combined
with both zero-shot successes (uji_b0: 1.12m/41.80m naive; tampere:
1.62m/30.21m naive, both unaffected by this fine-tuning stage since they
evaluate the frozen v4 joint checkpoint directly), this is the headline
result set for the paper.

Outputs: `outputs/phase5c_v4/model_<site_id>.pt` (per-site specialized
checkpoints), `outputs/phase5c_v4/results.json`. The v4 joint checkpoint
(`outputs/phase5b_v4/model.pt`) remains the reusable "foundation model"
artifact for any future new/zero-shot site.

## Phase 6: oracle diagnostic retry with synthetic drift (2026-07-15)

Implemented in `scripts/phase6_oracle_diagnostic.py`. Per the earlier
pivot decision, retried the oracle diagnostic using SYNTHETIC,
spatially-generic drift (`random_drift_snapshot`, ported directly from
`wifi_tta/src/drift_simulator.py`: a buffer perturbed with one randomly
sampled random-walk-attenuation-severity + dropout-probability pair,
severity drawn independently per instance, decoupled from which physical
point it is) instead of HDLC's real, location-entangled Layout 1->2/3
obstruction drift. Same reference-relative buffer-stats design and
three-way comparison as Phase 4, run against the best available frozen
model (Phase 5c v4 fine-tuned HDLC checkpoint), on the same 116 points
held out from that model's training.

One implementation issue found and fixed: severe synthetic drift
(dropout up to 60%, walk severity up to 20dB) can occasionally zero out
every visible AP in a reading -- a genuine total-signal-loss scenario,
not a bug. Handled by scoring those rows with a fixed large penalty error
(30m) rather than silently skipping them, since skipping would bias the
diagnostic's target to look better than reality under severe drift.

### Result: a real, non-negative signal appears -- but still not a green light

| predictor | Phase 4 (HDLC real drift) | Phase 6 (synthetic drift) |
|---|---|---|
| Leave-points-out RandomForest | -0.070 | **0.096** |
| Random (non-grouped) RandomForest | 0.538 | 0.146 |
| Leave-points-out Ridge (linear) | -0.098 | 0.057 |
| Point-identity ALONE (oracle) | 0.538 | 0.333 |

**Confirms the Phase 4 root-cause diagnosis.** Switching to spatially-
generic synthetic drift produced a real, non-negative signal (both RF and
Ridge went from clearly negative to weakly positive R^2) where HDLC's
real, location-fixed obstruction drift produced essentially nothing. This
is direct evidence that the earlier negative result was about HDLC's
specific drift structure, not an inherent property of this backbone or
buffer-stats approach.

**Still not a green light, per the non-negotiable discipline.** Point
identity alone (R^2=0.333) still predicts drift-induced error better than
the actual buffer statistics do (0.096-0.146) -- the same confound
signature the diagnostic exists to catch, just less severe than Phase 4's
0.538-vs-0.538 near-total-overlap. This result now resembles the prior
project's CSI pipeline finding (real but location-dominated signal,
R^2~0.22 vs ~0.375) more than HDLC's original flat-zero result -- a
genuinely different, more informative negative result, not a repeat.

### Phase 6b: pooled across 6 sites -- confound gets WORSE, not better (2026-07-15)

Tried the natural next experiment: pooled buffer instances across all 6
pretraining sites (`scripts/phase6b_oracle_diagnostic_pooled.py`) to test
whether HDLC-alone's point-identity dominance was an artifact of only
having 116 points to memorize. Required switching from per-AP-dimension
buffer stats (3 x num_APs, incompatible across sites with 17 to 520 APs)
to fixed-size (9-dim) aggregate statistics (mean/std/max of detect-rate
and RSSI-diff, plus AP count) so every site produces a comparable feature
vector. Also added a leave-SITE-out test (not just leave-points-out) as
the strictest possible generalization check.

| predictor | HDLC alone (6) | Pooled, 6 sites (6b) |
|---|---|---|
| Leave-points-out RF | 0.096 | 0.152 |
| Leave-points-out Ridge | 0.057 | 0.163 |
| Point-identity alone | 0.333 | **0.685** |
| Leave-SITE-out RF | -- | **-0.052** |

Pooling made the confound WORSE: identity-alone R^2 roughly doubled, and
the leave-site-out test -- predicting an entirely unseen site's error
using only patterns learned from other sites -- is negative. Root cause:
the AP-count feature is a near-literal site fingerprint (each site has a
fixed, distinct AP count), and different sites have very different
baseline error scales, so pooling let the regressor learn "these stats
look like site X" rather than anything about drift.

### Phase 6c: final trial -- two principled fixes, still negative (2026-07-15)

Agreed with the user this would be the last trial regardless of outcome.
Two fixes, both conceptually motivated (not tuning-for-a-number):
(1) dropped the AP-count feature (`scripts/phase6c_oracle_diagnostic_normalized.py`);
(2) normalized the target per site (subtract each site's own median
per-buffer error) -- the conceptually correct framing for TTA, which
should predict drift-induced degradation RELATIVE TO a site's own normal
condition, not absolute difficulty.

| predictor | raw, AP-count dropped only | + per-site target normalization |
|---|---|---|
| Leave-points-out RF | 0.150 | 0.025 |
| Point-identity alone | 0.685 | 0.635 |
| Leave-SITE-out RF | -0.047 | **-0.268** |

Dropping AP-count alone barely moved anything (the other aggregate
features already encode enough site character on their own). Target
normalization made leave-site-out WORSE, not better: removing between-
site variance left only within-site variance, which point identity still
dominates (0.635) even there.

### Final verdict on Phase 6: negative, well-evidenced, not pursued further

Four independent tests -- HDLC-alone/real-drift (Phase 4), HDLC-alone/
synthetic-drift (Phase 6), pooled/synthetic-drift (Phase 6b), pooled/
synthetic-drift with two principled confound fixes (Phase 6c) -- all
converge on the same conclusion: reference-relative buffer statistics, in
this design, do not carry a cross-site-generalizable signal about
drift-induced error on this backbone and these datasets. Point/site
identity dominates every version tried. This is treated as a closed,
root-caused negative result, not an open question -- per the project's
own discipline against open-ended tuning once a finding replicates this
consistently.

**Standing result for the paper: Phase 6 (single-site HDLC, synthetic
drift)** -- the least-bad, most directly comparable version of this
finding (R^2=0.096 RF / 0.057 Ridge vs. identity=0.333), reported
alongside Phase 4's real-drift result as two independent negative
findings with a shared root cause, with Phase 6b/6c's pooled attempts
noted as confirming rather than overturning the conclusion.

Honest next steps for anyone continuing this line of work (not attempted
here): buffer statistics designed to NOT correlate with site/point
identity at all (a genuinely hard feature-design problem, not solved by
dropping one obviously-bad feature), or substantially more independent
deployment sessions per site (closer to the prior project's 12+ sessions
per building) so a regressor has enough real drift diversity to learn
from that isn't confounded with location in the first place.

Raw results: `outputs/phase6/oracle_diagnostic_synthetic_drift.json`,
`outputs/phase6b/oracle_diagnostic_pooled.json`,
`outputs/phase6c/oracle_diagnostic_normalized.json`.
