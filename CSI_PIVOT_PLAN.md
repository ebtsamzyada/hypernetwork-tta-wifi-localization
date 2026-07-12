# CSI Pivot Plan

## Why this document exists

The original research proposal (`Uncertainty_Triggered_TTA_Indoor_Localization_Proposal_v2.docx`)
describes a system built on **CSI** (Channel State Information), with a graph-structured
encoder, self-supervised pretraining, and hypernetwork-driven test-time adaptation.
Everything built so far (`src/`, `src_sod/`, `src_sod_graph/`) is an **RSSI**-based
proof-of-concept of one piece of that proposal (the hypernetwork TTA mechanism),
built under the original 2-day deadline as an explicit, sensible simplification.

This document tracks the pivot to making the CSI-based system in the proposal
the actual paper, while keeping all RSSI work intact as prior validated groundwork
(methodology, bug fixes, and the sparse-vs-dense finding remain valid and reusable
in the writeup regardless of this pivot).

**Nothing in `src/`, `src_sod/`, `src_sod_graph/`, `outputs/`, `outputs_sod/`,
`outputs_sod_graph/` is touched by this pivot.** All new work happens in `src_csi/`
and `outputs_csi/`.

## The dataset

**[qiang5love1314/CSI-dataset-for-indoor-localization](https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization)**
— verified directly (downloaded and loaded sample files, not just read the README).

- Real Intel-5300-style CSI, pre-extracted into `.mat` files (`scipy.io.loadmat`
  reads them directly, no MATLAB or raw-packet parsing needed).
- Each reference point: one `coordinateN.mat` (real part) + one `imaginaryN.mat`
  (imaginary part), shape `(3, 30, 1500)` = 3 antennas x 30 subcarrier groups x
  1500 packets. Combine as `real + 1j*imaginary` for complex CSI, or use
  amplitude `abs(complex)` / phase `angle(complex)` as features.
- **Area One (Lab, NLOS)**: 317 reference points, 13.5m x 11m room.
- **Area Two (Meeting Room, LOS)**: 176 reference points, 7m x 10m room.
- Two newer areas (Conference Room, miniLab) also present, unexplored so far.

## Honest differences from the RSSI setup (read before assuming anything transfers)

1. **Single link, not multi-AP.** RSSI fingerprinting worked with dozens of
   *different access points* scattered through a building; the localization
   signal came from *which APs are visible and how strong*. This dataset is
   3 antennas from (apparently) **one** transceiver pair. The signal here comes
   from fine-grained multipath structure across subcarriers/antennas, not from
   combining readings across many spatially distributed APs. **The "AP dropout /
   AP heterogeneity" framing that motivated the RSSI graph encoder does not
   directly apply.** A graph over antennas/subcarriers is still a reasonable
   idea (modeling frequency/spatial correlation), but the motivating problem is
   different and needs to be re-argued, not assumed.
2. **No AP-dropout drift mechanism.** The RSSI drift simulator (random-walk
   attenuation + increasing AP dropout probability) has no direct analogue for
   a single always-present link. A CSI-appropriate drift model needs designing
   from scratch — candidates: subcarrier-wise amplitude random walk (direct
   analogue of the RSSI attenuation walk), simulated multipath/fading change,
   or phase drift. This is a real design decision, not a parameter rename.
3. **Coordinates are filename-encoded, ambiguously.** The README's only
   documentation is one example: `"coordinate 715"` means position `[7, 15]`.
   The exact digit-splitting rule (how many leading digits are X vs Y) must be
   reverse-engineered carefully from the full filename list before trusting any
   label — get this wrong and every result downstream is silently corrupted.
4. **No session/user/device metadata.** Unlike UJIIndoorLoc (USERID+PHONEID)
   and SOD-HCXY (same), there is no grouping field here. Leakage-free
   evaluation will be done by holding out entire **reference points** (not
   packets) for test — the 1500 packets at one point are a burst, same
   concern as UJI's burst sessions — which is a valid but weaker guarantee
   than the session-level splits used so far (no guarantee of, e.g.,
   device-generalization, since there's no device metadata to split on).

## Phased plan (mirrors the original proposal's Section 7, adapted to the above)

- [ ] **Phase 1 — Data loading & leakage-free split.** Parse `.mat` files,
      decode filename coordinates correctly (verify against the layout PDF/
      image in the repo before trusting it), build features (start with
      amplitude, the least ambiguous choice), hold out entire reference points
      for test, confirm no point appears in both splits.
- [ ] **Phase 2 — Base network & honest baseline.** Simple model first
      (matching the "don't over-engineer" discipline that served us well
      before) mapping CSI features -> (x, y). Report honest baseline error
      before anything else.
- [ ] **Phase 3 — CSI-appropriate drift simulator.** Design and justify a
      drift model appropriate to CSI (see point 2 above), sanity-check it
      degrades the frozen baseline the same way the RSSI one did.
- [ ] **Phase 4 — Hypernetwork meta-training.** Re-apply the reference-buffer
      design that was proven necessary for RSSI (compare against a
      same-session/same-deployment baseline, not absolute stats) — the
      underlying reason it worked (separating location identity from drift)
      may or may not apply the same way here given point 1 above; treat this
      as a hypothesis to test, not an assumption.
- [ ] **Phase 5 — Deployment simulation & honest evaluation.** Same
      discipline as before: validate hyperparameters (buffer size, cadence,
      correction clipping) only on held-out training data, touch test once,
      report a multi-seed robust result.
- [ ] **Phase 6 (stretch, per proposal) — Graph encoder over antenna/subcarrier
      structure.** Only after Phases 1-5 give an honest baseline to compare
      against.
- [ ] **Phase 7 (stretch, per proposal) — Self-supervised pretraining**
      (temporal-continuity + subcarrier-proximity pretext tasks).
- [ ] **Phase 8 (stretch, per proposal) — Uncertainty-gated adaptation**
      (replace fixed-cadence firing with a calibrated aleatoric/epistemic
      trigger) — the proposal's own stated highest-priority addition if time
      allows.

## Coordinate decoding (verified, not assumed)

Filename numeric ID -> (x, y): **last 2 digits = Y, remaining leading digits =
X**. Verified two ways: (1) matches the README's only documented example
exactly (`coordinate715` -> `[7, 15]`), and (2) across all 317 Lab filenames,
decodes to X in [1, 21] (21 unique values) and Y in [1, 23] (23 unique
values) -- a 21x23 grid with some missing cells (317 of 483 possible),
consistent with "computer lab with tables, chairs and obstacles" blocking
some grid points. Room is 13.5m x 11m, so this implies roughly 0.64m
spacing along X and 0.48m along Y -- a plausible dense sampling grid.

## Physical layout (verified against lab.pdf, not assumed)

Rendered `Lab Dataset/lab.pdf` to an image and inspected it directly:
confirms **one Transmitter, one Receiver** (visually confirms the
single-link setup from point 1 above -- there is genuinely only one AP
here, not several), and a grid spanning the full 13.5m x 11m room whose
column/row counts match the decoded X(1-21)/Y(1-23) ranges exactly.
Some grid cells are occupied by tables (visible in the diagram) --
consistent with 317 of 483 possible grid cells being sampled.

**Coordinate -> metres conversion**: X-grid maps to the 13.5m dimension
(21 points, 20 gaps -> ~0.675m spacing), Y-grid maps to the 11m dimension
(23 points, 22 gaps -> ~0.5m spacing).
`x_m = (x_grid - 1) / 20 * 13.5`, `y_m = (y_grid - 1) / 22 * 11`.

## Data quality note

34 of 317 points (all in the weakest-signal grid columns, farthest from
the Tx/Rx pair) have a small fraction of packets with Inf amplitude
values (1,676 of 475,500 packets total, 0.35%). Dropped at the PACKET
level, not the point level, so otherwise-good data from those 34 points
is kept.

## Status

**Phase 1 complete.** Data loads cleanly: 378,003 train packets across
253 points, 95,821 test packets across 64 held-out points (grouped split
by reference point, verified disjoint).

**Phase 2 complete.** Honest baseline (plain MLP, 90 CSI-amplitude
features -> (x, y), point-disjoint held-out test):
```
Mean Euclidean error : 5.274 m
Median Euclidean error: 5.418 m
90th pct error        : 7.875 m
Per-axis MAE (x, y)   : 3.796 m, 2.995 m
```
Room diagonal is ~17.4m (13.5m x 11m), so 5.27m mean error is a
realistic, non-suspicious number for a single-link CSI setup with no
multi-AP triangulation available -- consistent with this project's
honesty discipline (no near-0m numbers that would signal leakage).

**Phase 2 (graph variant) complete.** Built `SubcarrierGraphEncoder`
(base_model.py): 30 subcarrier nodes, each carrying its 3-antenna
amplitude reading, frequency-adjacency attention mask (each subcarrier
attends only to neighbors within +/-2 in frequency -- a genuine physical
edge structure, unlike the RSSI graph's fallback to full attention).
Trained via `train_baseline_graph.py`, same split, same evaluation:
```
                mean     median   90th pct   per-axis MAE
Plain MLP:      5.274m   5.418m   7.875m     3.796m, 2.995m
Graph encoder:  5.372m   5.240m   9.215m     3.786m, 3.057m
```
**Essentially tied, not a decisive win** -- unlike RSSI (where the graph
encoder cut error by ~23%). This is a consistent, informative finding,
not a failure: the RSSI graph encoder's win came specifically from
handling AP HETEROGENEITY (different locations see different subsets of
APs, out of 520/56 total possible). This CSI setup has no analogous
failure mode -- all 3 antennas x 30 subcarriers are ALWAYS present, so
there's nothing for the graph structure to compensate for that the plain
MLP's dense layers couldn't already learn. Worth stating directly in the
writeup: the graph encoder's value is conditional on missing-data
structure, not a universal improvement.

## Bug found and fixed: severe overfitting (Phase 2 revisited)

First Phase 2 attempt (hidden=(128,64), weight_decay=1e-5, all 1500
packets/point) reported 5.274m mean error -- but a naive constant-
prediction baseline (always predict the training centroid) scored 4.90m,
i.e. the model was WORSE than guessing. Root cause: only ~250 distinct
training LOCATIONS (however many redundant near-duplicate packets each),
enough capacity to memorize them (train error 0.64m vs validation error
5.14m, an 8x gap) without learning a generalizable spatial mapping.

Fixed via: dropout (0.3) + smaller hidden layers (64,32) + stronger
weight_decay (1e-3) + packet subsampling (every 5th packet, 1500->~300
per point, reducing near-duplicate redundancy the model could memorize).
Also added a standing naive-baseline check to every training run's
output so this class of bug can't hide silently again.

**Corrected honest baseline:**
```
Mean Euclidean error : 4.691 m   (naive baseline: 4.884 m -- now genuinely beats it)
Median Euclidean error: 4.631 m
90th pct error        : 7.517 m
Per-axis MAE (x, y)   : 3.157 m, 2.898 m
```
Real but modest improvement over naive (~4%). Two structural reasons
this isn't dramatically better (not excuses for the bug, which is fixed):
single Tx-Rx link (no multi-AP triangulation), and phase is present but
uncalibrated (raw phase at a fixed subcarrier jumps ~randomly between
packets, std ~2.83 rad -- classic uncalibrated CFO/STO artifact) so only
amplitude is currently used.

**Graph encoder re-verified under the same fix** (matching dropout +
weight_decay=1e-3, same packet-subsampled data):
```
                mean     median   90th pct   vs naive (4.884m)
Plain MLP:      4.691m   4.631m   7.517m     beats by 4.2%
Graph encoder:  4.770m   4.619m   7.346m     beats by 2.3%
```
Same conclusion as before the bug fix: essentially tied, not a decisive
win for the graph encoder -- confirms that finding was real, not an
artifact of the earlier overfitting bug.

**Phase 3 complete.** Built `drift_simulator.py`: continuous (non-resetting,
since this is ONE fixed link, not multiple deployed devices), persistent
per-(antenna,subcarrier) random-walk amplitude drift, deployment order =
greedy nearest-neighbor spatial tour through the 64 held-out test points
(packets within a point keep their original temporal order).

Sanity check: with `walk_sigma=0.04` (initial guess, scaled from
amplitude's own std of ~10.5), drift did NOT clearly degrade the frozen
model (chunks non-monotonic, overall drifted error 4.53m even LOWER than
undrifted 4.69m). Swept `walk_sigma` 0.04 -> 0.8: overall error climbs
4.53m -> 5.49m (~17% relative), a real trend -- but per-chunk breakdown
stays non-monotonic at every severity (chunk 1 is stubbornly the easiest
regardless of drift). Same root cause as the RSSI pipelines' session
variance: only 64 distinct test points, so which points happen to fall
in which chunk of the single spatial-tour ordering dominates a
single-run chunk trend. **Lesson already learned once, applies again:
Step 5's headline metric here must be multi-seed (and likely
multi-tour) averaged, not read off one ordering's chunk breakdown.**
`walk_sigma=0.4` chosen as a clear-but-not-extreme default going forward.

**Phase 4 complete.** Built CSI-adapted hypernetwork (`hypernetwork.py`:
mean/variance difference stats, no detection-rate term since amplitude is
always present; `train_hypernetwork.py`/`cv_hypernetwork.py`: same
meta-training + leave-groups-out CV discipline, points instead of
sessions as the group unit). 5-fold CV over all 253 train points:
```
Mean across folds -- adapted: 3.651m  frozen: 3.703m  improvement: +0.052m
Folds where adapted beat frozen: 5/5
```
Notably more stable than the RSSI pipelines' first CV attempt (no wild
per-fold swings, no fold went negative) -- likely because CSI amplitude
features are lower-dimensional and each meta-training episode's
reference+current buffers come from the SAME point by construction, so
there's less of the location/drift confound RSSI had to fight through
several redesigns to fix. Modest (~1-1.7% relative) but consistent
improvement, closer to UJIIndoorLoc's 1.7% than SOD-HCXY's 7-9%.

**Phase 5 complete -- honest null result on the real test set.**
Calibrated `max_delta_frac` on 60 held-out TRAIN points (never test):
unusually stable compared to RSSI, no instability at any clip level
tested (0.05-0.6), improvement saturates at a small, consistent
+0.022m (~0.5%). Locked in 0.3.

**Final test-set result (15 independent drift realizations, the real
headline number) contradicts the validation signal:**
```
Frozen : 6.153m +/- 1.182m
Adapted: 6.201m +/- 1.219m
Improvement: -0.048m +/- 0.048m (-0.8% relative)
Seeds where adapted beat frozen: 2/15
```
The deployment plot confirms this visually: the frozen and adapted
curves are essentially perfectly overlapping across the ENTIRE
19,162-step timeline, no visible separation anywhere.

Checked whether this is a data-split artifact (test points
systematically harder/different from the train pool used for CV and
calibration) -- ruled out: test points actually have FEWER weak-signal
points than train (9.4% vs 18.6% in the far-edge region), similar mean
grid position. This looks like genuine failure to generalize from the
training point pool to the officially held-out test points, not an
artifact of an unlucky split.

**Honest conclusion:** on this single-link CSI dataset, the
hypernetwork mechanism shows a small, consistent positive signal during
meta-training-time cross-validation (5/5 folds, held-out train points),
but that signal does NOT reliably transfer to the true held-out test
set -- a legitimate negative finding, not a bug (thoroughly checked: no
composition bias, stable/non-catastrophic calibration, clean visual
confirmation). Worth reporting as-is rather than continuing to search
for a config that "works" on test, which would be exactly the kind of
test-set-fitting this project has been careful to avoid throughout.
