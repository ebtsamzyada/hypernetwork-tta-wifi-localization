"""HDLC ("Hybrid-fingerprint Data with Layout Change") raw-CSV loader.

WiFi-only scope (17 WAP columns). BLE columns (42) are dropped by design --
see PIVOT_PLAN.md Item 2a / decisions log. Ground-truth AP positions come
from row 1 of each raw CSV, which HDLC always publishes (unlike UJIIndoorLoc
derivatives, which don't) -- see PIVOT_PLAN.md Item 1.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RSSI_MIN = -100.0
RSSI_MAX = -30.0

# HDLC's actual "not detected" convention, confirmed against the source
# paper (Nor Hisham et al., Data 2022, 7(11), 156). NOT the +100 sentinel
# the original collaborator script assumed (that was a UJIIndoorLoc
# convention that never matched this dataset -- see PIVOT_PLAN.md).
NO_SIGNAL_SENTINEL = -110.0

_WAP_RE = re.compile(r"^WAP\s*\d+$")


def _clean_header(h) -> str:
    return str(h).strip()


def _is_wap_column(header: str) -> bool:
    return bool(_WAP_RE.match(_clean_header(header)))


def _parse_position_row(headers, coords_row) -> dict:
    """Parse row 1 of a raw CSV ('(x,y)' per AP/beacon column) into a dict
    keyed by cleaned header name. HDLC publishes real ground-truth
    coordinates here for all 59 sources (42 BLE + 17 WAP)."""
    positions = {}
    for h, v in zip(headers, coords_row):
        if isinstance(v, str) and v.strip().startswith("(") and ")" in v:
            try:
                vx, vy = map(float, v.strip().strip("()").split(","))
                positions[_clean_header(h)] = (vx, vy)
            except ValueError:
                pass
    return positions


def load_raw_layout_csv(path):
    """Load one 'Layout N - Raw - {Training,Testing}.csv' file.

    Returns (df, wap_positions, wap_cols):
      df           -- columns: floor (int), x (float), y (float), plus one
                       float32 column per WAP (already sentinel-cleaned and
                       clipped to [RSSI_MIN, RSSI_MAX]).
      wap_positions-- {wap_col_name: (x, y)} ground-truth AP coordinates.
      wap_cols     -- list of WAP column names, in file order.
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig")
    headers = [_clean_header(h) for h in raw.iloc[0].tolist()]
    coords_row = raw.iloc[1].tolist()

    all_positions = _parse_position_row(headers, coords_row)
    wap_cols = [h for h in headers if _is_wap_column(h)]
    wap_positions = {h: all_positions[h] for h in wap_cols if h in all_positions}

    match_ratio = len(wap_positions) / len(wap_cols) if wap_cols else 0.0
    if match_ratio < 1.0:
        raise ValueError(
            f"[{path.name}] only {match_ratio:.0%} of {len(wap_cols)} WAP columns "
            f"had a parsed ground-truth position in row 1 -- expected 100% for "
            f"HDLC. Check for a header-format change in this file."
        )

    body = raw.iloc[2:].reset_index(drop=True)
    body.columns = headers

    out = pd.DataFrame({
        "floor": pd.to_numeric(body["Floor"], errors="coerce").astype(int),
        "x": pd.to_numeric(body["x"], errors="coerce"),
        "y": pd.to_numeric(body["y"], errors="coerce"),
    })

    rssi = body[wap_cols].apply(pd.to_numeric, errors="coerce")
    n_sentinel = int((rssi.to_numpy() == NO_SIGNAL_SENTINEL).sum())
    rssi = rssi.mask(rssi == NO_SIGNAL_SENTINEL, np.nan)
    rssi = rssi.fillna(RSSI_MIN).clip(RSSI_MIN, RSSI_MAX)
    for col in wap_cols:
        out[col] = rssi[col].astype(np.float32)

    total_cells = rssi.size
    print(
        f"[{path.name}] {len(out)} rows, {len(wap_cols)} WAP columns "
        f"(ground-truth positions: {len(wap_positions)}/{len(wap_cols)}, "
        f"{match_ratio:.0%}). {n_sentinel}/{total_cells} "
        f"({n_sentinel / max(1, total_cells):.1%}) cells were the "
        f"{NO_SIGNAL_SENTINEL} dBm 'not detected' sentinel."
    )

    return out, wap_positions, wap_cols


def build_floor_class_map(df: pd.DataFrame) -> dict:
    """Dense 0..num_floors-1 class-index mapping from raw floor ids."""
    floor_ids_sorted = sorted(df["floor"].unique().tolist())
    return {fid: i for i, fid in enumerate(floor_ids_sorted)}


def attach_floor_class(df: pd.DataFrame, floor_id_to_class: dict) -> pd.DataFrame:
    df = df.copy()
    df["floor_class"] = df["floor"].map(floor_id_to_class)
    if df["floor_class"].isna().any():
        raise ValueError(
            "attach_floor_class found floor id(s) not present in "
            "floor_id_to_class -- HDLC only has floors {0,1,2}; check input."
        )
    df["floor_class"] = df["floor_class"].astype(int)
    return df


def filter_locatable_rows(df: pd.DataFrame, wap_positions: dict, wap_cols: list, k: int, label: str) -> pd.DataFrame:
    """Drop rows where none of the k-strongest visible WAPs has a known
    position. With HDLC's 100% ground-truth WAP position coverage this
    should never drop anything -- kept as a safety net (see prior
    project's discipline of never silently leaking a trivial target)."""
    rssi_mat = df[wap_cols].to_numpy(dtype=np.float32)

    def _row_ok(row):
        visible = np.where(row > RSSI_MIN)[0]
        if len(visible) == 0:
            return False
        strongest = visible[np.argsort(-row[visible])][:k]
        return any(wap_cols[i] in wap_positions for i in strongest)

    mask = np.array([_row_ok(rssi_mat[i]) for i in range(len(df))])
    n_dropped = len(df) - int(mask.sum())
    if n_dropped:
        print(f"[{label}] dropped {n_dropped}/{len(df)} rows with no positioned "
              f"k-strongest WAP.")
    return df[mask].reset_index(drop=True)
