"""Per-site config and loaders for the multi-site pretraining pool.

Sites (7 total, confirmed via audit -- see PIVOT_PLAN.md "Multi-site audit"):
  Pretraining pool (6):
    hdlc               -- Multimedia University, Malaysia (already Phase 1-4)
    sod_cetc331        -- SODIndoorLoc, 3 floors, ground-truth AP positions
    sod_hcxy           -- SODIndoorLoc, 1 floor,  ground-truth AP positions
    sod_syl            -- SODIndoorLoc, 1 floor,  ground-truth AP positions
    uji_b1             -- UJIIndoorLoc building 1 (4 floors, 12 sessions)
    uji_b2             -- UJIIndoorLoc building 2 (5 floors, 16 sessions, richest)
  Held out entirely, zero-shot test only (1):
    uji_b0             -- UJIIndoorLoc building 0 (4 floors, only 2 sessions --
                           too thin for pretraining diversity, which is exactly
                           why it's cheap to hold out instead)

Every site loader returns a DataFrame with a COMMON schema:
    site_id, floor_id (raw, site-local), x, y, _group, <wap_cols...>
plus (wap_positions, wap_cols). `_group` is the leakage-safe split key --
"point" (floor,x,y) for HDLC/SODIndoorLoc, "session" (USERID_PHONEID) for
UJIIndoorLoc, matching each dataset's own validated confound (see
PIVOT_PLAN.md and wifi_tta/src/data_loader.py's docstring).

Heterogeneous AP counts/positions across sites are NOT a problem for the
shared CNN: the virtual-space image is built per-ROW from just that row's
k-strongest APs, so the backbone never sees a fixed-width AP vector.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HDLC_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Hybrid-fingerprint Data with Layout Change (HDLC)"
SOD_ROOT = Path(r"B:\Uni materials\WRC Internship\17th JUly\wifi_tta\data_sod\SODIndoorLoc-main")
UJI_ROOT = Path(r"B:\Uni materials\WRC Internship\17th JUly\wifi_tta\data\UJIndoorLoc")

PRETRAIN_SITE_IDS = ["hdlc", "sod_cetc331", "sod_hcxy", "sod_syl", "uji_b1", "uji_b2"]
HELDOUT_SITE_ID = "uji_b0"

# Unified raw-dBm scale for the WHOLE multi-site pool -- matches SODIndoorLoc
# and UJIIndoorLoc's native (-104, 0) convention. HDLC's own single-site
# pipeline (data_loading.py, Phase 1-4) uses a DIFFERENT local scale
# (-100, -30) and is left untouched/reproducible; sites.py re-derives HDLC
# on this unified scale independently so all 7 sites share one consistent
# "detected vs not" threshold for the shared virtual-space image encoding.
# Mixing scales would be a real bug: HDLC's not-detected fill would sit
# ABOVE a -104 threshold and get misclassified as "detected".
RSSI_MIN = -104.0
RSSI_MAX = 0.0


def estimate_ap_positions_from_data(df, wap_cols, sentinel_min, sentinel_max,
                                     min_obs=5, strength_floor=0.15):
    """RSSI-weighted centroid of where each AP was strongly observed.
    Used for sites that don't publish ground-truth AP coordinates
    (UJIIndoorLoc). Ported from the original collaborator script's
    fallback -- an approximation, not ground truth; treat downstream
    xy-error on these sites as relative/comparative, not absolute."""
    xs = df["x"].to_numpy(dtype=np.float64)
    ys = df["y"].to_numpy(dtype=np.float64)
    positions = {}
    for col in wap_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        strength = np.clip((vals - sentinel_min) / (sentinel_max - sentinel_min), 0.0, 1.0)
        mask = strength >= strength_floor
        if mask.sum() < min_obs:
            continue
        w = strength[mask]
        positions[col] = (float(np.average(xs[mask], weights=w)), float(np.average(ys[mask], weights=w)))
    return positions


# ---------------------------------------------------------------- HDLC ----

HDLC_SENTINEL = -110.0


def _load_hdlc_raw_csv(path):
    """Independent of data_loading.load_raw_layout_csv -- reuses its
    low-level header/AP-position parsing helpers but applies THIS module's
    unified (-104, 0) scale instead of data_loading's local (-100, -30),
    so HDLC's rows are numerically consistent with every other site."""
    from data_loading import _clean_header, _is_wap_column, _parse_position_row

    raw = pd.read_csv(path, header=None, encoding="utf-8-sig")
    headers = [_clean_header(h) for h in raw.iloc[0].tolist()]
    coords_row = raw.iloc[1].tolist()
    all_positions = _parse_position_row(headers, coords_row)
    wap_cols = [h for h in headers if _is_wap_column(h)]
    wap_positions = {h: all_positions[h] for h in wap_cols if h in all_positions}

    body = raw.iloc[2:].reset_index(drop=True)
    body.columns = headers

    df = pd.DataFrame({
        "floor_id": pd.to_numeric(body["Floor"], errors="coerce").astype(int),
        "x": pd.to_numeric(body["x"], errors="coerce"),
        "y": pd.to_numeric(body["y"], errors="coerce"),
    })
    rssi = body[wap_cols].apply(pd.to_numeric, errors="coerce")
    rssi = rssi.mask(rssi == HDLC_SENTINEL, np.nan).fillna(RSSI_MIN).clip(RSSI_MIN, RSSI_MAX)
    df = pd.concat([df, rssi.astype(np.float32)], axis=1)

    return df, wap_positions, wap_cols


def _load_hdlc():
    train_raw, wap_pos, wap_cols = _load_hdlc_raw_csv(HDLC_ROOT / "Layout 1" / "Layout 1 - Raw - Training.csv")
    test_raw, _, _ = _load_hdlc_raw_csv(HDLC_ROOT / "Layout 1" / "Layout 1 - Raw - Testing.csv")
    df = pd.concat([train_raw, test_raw], ignore_index=True)
    df["site_id"] = "hdlc"
    df["_group"] = df["floor_id"].astype(str) + "_" + df["x"].astype(str) + "_" + df["y"].astype(str)
    return df[["site_id", "floor_id", "x", "y", "_group"] + wap_cols], wap_pos, wap_cols


# ---------------------------------------------------------- SODIndoorLoc ----

_SOD_FILES = {
    "sod_cetc331": ("CETC331/Training_CETC331.csv", "CETC331/Testing_CETC331.csv", "CETC331"),
    "sod_hcxy": ("HCXY/Training_HCXY_AP_30.csv", "HCXY/Testing_HCXY_AP.csv", "HCXY"),
    "sod_syl": ("SYL/Training_SYL_AP_30.csv", "SYL/Testing_SYL_AP.csv", "SYL"),
}
SOD_SENTINEL = 100.0


def _load_sod_ap_positions(building_sheet, mac_cols):
    import openpyxl
    wb = openpyxl.load_workbook(SOD_ROOT / "Information of pre-installed APs.xlsx", read_only=True)
    ws = wb[building_sheet]
    positions = {}
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    for r in rows:
        _id, ecoord, ncoord, _floor, mac24, _freq24, mac5, _freq5 = r
        if mac24 in mac_cols:
            positions[mac24] = (float(ecoord), float(ncoord))
        if mac5 in mac_cols:
            positions[mac5] = (float(ecoord), float(ncoord))
    return positions


def _load_sod_site(site_id):
    trainf, testf, sheet = _SOD_FILES[site_id]
    tr = pd.read_csv(SOD_ROOT / trainf)
    te = pd.read_csv(SOD_ROOT / testf)
    raw = pd.concat([tr, te], ignore_index=True)

    mac_cols = [c for c in raw.columns if c.startswith("MAC")]
    rssi = raw[mac_cols].apply(pd.to_numeric, errors="coerce")
    rssi = rssi.mask(rssi == SOD_SENTINEL, np.nan).fillna(RSSI_MIN).clip(RSSI_MIN, RSSI_MAX)

    df = pd.DataFrame({
        "site_id": site_id,
        "floor_id": raw["FloorID"].astype(int),
        "x": raw["ECoord"].astype(float),
        "y": raw["NCoord"].astype(float),
    })
    df = pd.concat([df, rssi.astype(np.float32)], axis=1)
    df["_group"] = df["floor_id"].astype(str) + "_" + df["x"].round(3).astype(str) + "_" + df["y"].round(3).astype(str)

    wap_positions = _load_sod_ap_positions(sheet, set(mac_cols))
    return df[["site_id", "floor_id", "x", "y", "_group"] + mac_cols], wap_positions, mac_cols


# ---------------------------------------------------------- UJIIndoorLoc ----

UJI_SENTINEL = 100.0
_UJI_BUILDING_IDS = {"uji_b0": 0, "uji_b1": 1, "uji_b2": 2}


def _load_uji_site(site_id):
    building_id = _UJI_BUILDING_IDS[site_id]
    raw = pd.read_csv(UJI_ROOT / "trainingData.csv")
    raw = raw[raw["BUILDINGID"] == building_id].reset_index(drop=True)

    wap_cols = [c for c in raw.columns if c.startswith("WAP")]
    rssi = raw[wap_cols].apply(pd.to_numeric, errors="coerce")
    rssi = rssi.mask(rssi == UJI_SENTINEL, np.nan).fillna(RSSI_MIN).clip(RSSI_MIN, RSSI_MAX)

    df = pd.DataFrame({
        "site_id": site_id,
        "floor_id": raw["FLOOR"].astype(int),
        "x": raw["LONGITUDE"].astype(float),
        "y": raw["LATITUDE"].astype(float),
    })
    df = pd.concat([df, rssi.astype(np.float32)], axis=1)
    # Leakage-safe group = capture session, NOT physical point -- UJIIndoorLoc
    # is burst-sampled per session (see wifi_tta/src/data_loader.py docstring);
    # a point-level group would still leak near-duplicate same-walk fingerprints.
    df["_group"] = raw["USERID"].astype(str) + "_" + raw["PHONEID"].astype(str)

    wap_positions = estimate_ap_positions_from_data(df, wap_cols, RSSI_MIN, RSSI_MAX)
    return df[["site_id", "floor_id", "x", "y", "_group"] + wap_cols], wap_positions, wap_cols


# ------------------------------------------------------------------ Tampere ----
# Lohan et al., "Crowdsourced WiFi database and benchmark software for
# indoor positioning", Zenodo 10.5281/zenodo.1001662 (v2), MIT/CC-BY.
# 4-floor university building, Tampere, Finland -- a genuinely new
# country/site relative to HDLC (Malaysia)/SODIndoorLoc (China)/
# UJIIndoorLoc (Spain). Audited: 992 AP columns, +100 sentinel (confirmed,
# UJI-style), no published AP positions (must estimate, like UJI), no
# floor column -- floor is derived from z (local coordinate, meters),
# which takes exactly 5 distinct values (0, 3.7, 7.4, 11.1, 14.8) spaced
# uniformly at 3.7m -- floor_id = round(z / 3.7). No user/session id
# column exists; device model + calendar day is used as the leakage-safe
# group instead (crowdsourced, bursty by device+day, same rationale as
# UJIIndoorLoc's session grouping -- see wifi_tta/src/data_loader.py).

TAMPERE_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw_tampere" / "FINGERPRINTING_DB"
TAMPERE_SENTINEL = 100.0
TAMPERE_FLOOR_HEIGHT_M = 3.7


def _load_tampere():
    tr_rss = pd.read_csv(TAMPERE_ROOT / "Training_rss_21Aug17.csv", header=None)
    te_rss = pd.read_csv(TAMPERE_ROOT / "Test_rss_21Aug17.csv", header=None)
    tr_crd = pd.read_csv(TAMPERE_ROOT / "Training_coordinates_21Aug17.csv", header=None, names=["x", "y", "z"])
    te_crd = pd.read_csv(TAMPERE_ROOT / "Test_coordinates_21Aug17.csv", header=None, names=["x", "y", "z"])
    tr_dev = pd.read_csv(TAMPERE_ROOT / "Training_device_21Aug17.csv", header=None, names=["device"])
    te_dev = pd.read_csv(TAMPERE_ROOT / "Test_device_21Aug17.csv", header=None, names=["device"])
    tr_date = pd.read_csv(TAMPERE_ROOT / "Training_date_21Aug17.csv", header=None, names=["date"])
    te_date = pd.read_csv(TAMPERE_ROOT / "Test_date_21Aug17.csv", header=None, names=["date"])

    wap_cols = [f"WAP{i:04d}" for i in range(tr_rss.shape[1])]
    rssi = pd.concat([tr_rss, te_rss], ignore_index=True)
    rssi.columns = wap_cols
    rssi = rssi.apply(pd.to_numeric, errors="coerce")
    rssi = rssi.mask(rssi == TAMPERE_SENTINEL, np.nan).fillna(RSSI_MIN).clip(RSSI_MIN, RSSI_MAX)

    crd = pd.concat([tr_crd, te_crd], ignore_index=True)
    dev = pd.concat([tr_dev, te_dev], ignore_index=True)
    date = pd.concat([tr_date, te_date], ignore_index=True)

    df = pd.DataFrame({
        "site_id": "tampere",
        "floor_id": (crd["z"] / TAMPERE_FLOOR_HEIGHT_M).round().astype(int),
        "x": crd["x"].astype(float),
        "y": crd["y"].astype(float),
    })
    df = pd.concat([df, rssi.astype(np.float32)], axis=1)
    day = date["date"].str.slice(0, 10)  # YYYY-MM-DD
    df["_group"] = dev["device"].astype(str) + "_" + day

    wap_positions = estimate_ap_positions_from_data(df, wap_cols, RSSI_MIN, RSSI_MAX)
    return df[["site_id", "floor_id", "x", "y", "_group"] + wap_cols], wap_positions, wap_cols


# --------------------------------------------------------------- registry ----

_LOADERS = {
    "hdlc": _load_hdlc,
    "sod_cetc331": lambda: _load_sod_site("sod_cetc331"),
    "sod_hcxy": lambda: _load_sod_site("sod_hcxy"),
    "sod_syl": lambda: _load_sod_site("sod_syl"),
    "uji_b0": lambda: _load_uji_site("uji_b0"),
    "uji_b1": lambda: _load_uji_site("uji_b1"),
    "uji_b2": lambda: _load_uji_site("uji_b2"),
    "tampere": _load_tampere,
}


def filter_locatable_rows(df, wap_positions, wap_cols, k, label):
    """Generic version of data_loading.filter_locatable_rows, parameterized
    on the unified RSSI_MIN above instead of HDLC's local constant.

    Checks "is ANY visible AP positioned", not "is any of the raw top-k
    positioned" -- the latter is fragile under train-time augmentation for
    partial-coverage sites (UJIIndoorLoc's estimated positions): noise/
    AP-dropping can shift the top-k away from a positioned AP even though
    one is still visible. build_virtual_space selects k-strongest AMONG
    positioned APs directly, so this check matches what it actually needs.
    No-op difference for 100%-coverage sites (HDLC, SODIndoorLoc)."""
    rssi_mat = df[wap_cols].to_numpy(dtype=np.float32)

    def _row_ok(row):
        visible = np.where(row > RSSI_MIN)[0]
        return any(wap_cols[i] in wap_positions for i in visible)

    mask = np.array([_row_ok(rssi_mat[i]) for i in range(len(df))])
    n_dropped = len(df) - int(mask.sum())
    if n_dropped:
        print(f"[{label}] dropped {n_dropped}/{len(df)} rows with no positioned visible AP "
              f"({n_dropped / len(df):.1%}).")
    return df[mask].reset_index(drop=True)


def build_floor_class_map(df):
    floor_ids_sorted = sorted(df["floor_id"].unique().tolist())
    return {fid: i for i, fid in enumerate(floor_ids_sorted)}


def attach_floor_class(df, floor_id_to_class):
    df = df.copy()
    df["floor_class"] = df["floor_id"].map(floor_id_to_class)
    if df["floor_class"].isna().any():
        raise ValueError("attach_floor_class found a floor id not in floor_id_to_class")
    df["floor_class"] = df["floor_class"].astype(int)
    return df


def load_site(site_id):
    """Returns (df, wap_positions, wap_cols) in the common schema."""
    if site_id not in _LOADERS:
        raise ValueError(f"Unknown site_id {site_id!r}. Known: {sorted(_LOADERS)}")
    df, wap_pos, wap_cols = _LOADERS[site_id]()
    n_positioned = sum(1 for c in wap_cols if c in wap_pos)
    print(f"[{site_id}] {len(df)} rows, {df['_group'].nunique()} groups, "
          f"{len(wap_cols)} AP columns ({n_positioned}/{len(wap_cols)} positioned), "
          f"floors={sorted(df['floor_id'].unique().tolist())}")
    return df, wap_pos, wap_cols
