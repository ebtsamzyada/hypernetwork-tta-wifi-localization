# Hypernetwork-Driven Test-Time Adaptation for WiFi Indoor Localization

A continuous, label-free, hypernetwork-driven test-time adaptation (TTA) mechanism for indoor WiFi
localization under post-deployment drift. A frozen base network maps a fingerprint to (x, y); a
small hypernetwork reads a rolling buffer of unlabeled fingerprints and predicts a weight
correction for the base network's final layer only, to compensate for drift without ever seeing a
ground-truth label at deployment time.

Four pipelines, evaluated across two signal types (RSSI, CSI) and two base-network architectures
(plain MLP, graph/attention encoder).

## Results summary

| Pipeline | Dataset | Base network | Clean baseline | TTA relative improvement |
|---|---|---|---|---|
| `src/` | UJIIndoorLoc (520 APs, 3.8% visible) | MLP (+ floor classifier) | 8.05m | ~1.7% |
| `src_sod/` | SOD-HCXY (56 APs, 22.3% visible) | MLP | 8.47m | 7.6-8.9% (15/15 seeds positive) |
| `src_sod_graph/` | SOD-HCXY | Graph/set-attention over AP identities | 6.53m (23% better than MLP) | 5.3% |
| `src_csi/` | qiang5love1314 CSI dataset (single Tx-Rx link) | MLP / subcarrier-graph encoder | 4.69m / 4.77m | Positive in cross-validation, does not generalize to the held-out test set (-0.8%) |

**Headline finding:** AP/signal density is the key variable determining whether buffer-statistics-driven TTA
can separate "location" from "drift" -- sparse RSSI (UJIIndoorLoc) gives a weak effect, dense RSSI
(SOD-HCXY) gives a strong, robust effect, and single-link CSI (richer per-link information but no
multi-AP structure) gives a signal that's real in offline cross-validation but doesn't transfer to
genuinely new locations.

Full narrative and diagnostic history for the CSI pipeline: [`CSI_PIVOT_PLAN.md`](CSI_PIVOT_PLAN.md).

## Repository layout

```
src/                  UJIIndoorLoc pipeline (RSSI, sparse)
src_sod/               SOD-HCXY pipeline (RSSI, dense)
src_sod_graph/          SOD-HCXY with a graph/attention base network
src_csi/                CSI pipeline (single Tx-Rx link)
outputs*/               Trained checkpoints + result plots for each pipeline
CSI_PIVOT_PLAN.md       Design log and results for the CSI pipeline
TTA_Project_Summary.pptx  Slide summary
seed_data*.json         Raw per-seed multi-seed evaluation results (SOD-HCXY variants)
```

Each `src*/` directory is self-contained: `data_loader.py` (Step 1, leakage-free split) ->
`train_baseline.py` (Step 2, base network) -> `drift_simulator.py` (Step 3) ->
`train_hypernetwork.py` / `cv_hypernetwork.py` (Step 4, meta-training) -> `deploy_simulation.py`
(Step 5, the headline deployment simulation and multi-seed robustness evaluation).

## Data

Raw datasets are **not** included in this repo (excluded via `.gitignore`, ~1.5GB+ total).

- **UJIIndoorLoc** and **SOD-HCXY**: auto-downloaded by `data_loader.py` on first run (see
  `ensure_downloaded()` / `DATA_DIR` in each pipeline's `data_loader.py`).
- **CSI dataset**: manually download
  [qiang5love1314/CSI-dataset-for-indoor-localization](https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization)
  and place it at `data_csi/CSI-dataset-for-indoor-localization-main/` (matching the path in
  `src_csi/data_loader.py`).

## Running a pipeline

```bash
cd src_sod   # or src/, src_sod_graph/, src_csi/
python data_loader.py         # sanity check: leakage-free split, no leakage assertion
python train_baseline.py      # Step 2: honest baseline
python drift_simulator.py     # Step 3: sanity check drift accumulation
python cv_hypernetwork.py     # Step 4: cross-validated meta-training + production checkpoint
python deploy_simulation.py   # Step 5: headline result (plot + multi-seed evaluation)
```

## Methodology notes

- **Leakage-free evaluation throughout**: every split is grouped by physical location/session,
  never by row, with an explicit disjointness assertion.
- **Validation-only hyperparameter calibration**: correction-magnitude clipping and other
  deployment-time settings are always calibrated on held-out *training* data, never on the real
  test set, before a single final touch of test data.
- **Multi-seed robustness**: headline numbers for the SOD-HCXY and CSI pipelines are averaged
  across 15 independent random drift realizations, not a single run.
