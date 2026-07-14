"""RSSI row -> "virtual space" image, GlobLoc-style: each of the
k-strongest visible WAPs is placed at its known real-world (x, y), pixel
intensity = normalized signal strength. Ported from the collaborator's
notebook, WiFi-only (no BLE), operating on the Phase 1 loader's clean
column layout.
"""
import numpy as np

from data_loading import RSSI_MIN as _HDLC_RSSI_MIN, RSSI_MAX as _HDLC_RSSI_MAX

PIXEL_METERS = 0.5


def select_k_strongest(rssi_row: np.ndarray, k: int, rssi_min: float = _HDLC_RSSI_MIN) -> np.ndarray:
    visible_idx = np.where(rssi_row > rssi_min)[0]
    return visible_idx[np.argsort(-rssi_row[visible_idx])][:k]


def build_virtual_space(x, y, rssi_row, wap_cols, wap_positions, k, margin_m, max_extent_m=None,
                         rssi_min: float = _HDLC_RSSI_MIN, rssi_max: float = _HDLC_RSSI_MAX):
    """Returns (ap_info, (vx, vy), (x_min, y_min), (height_px, width_px)).

    ap_info is a list of (row_px, col_px, intensity) to rasterize.
    (vx, vy) is the true (x, y) expressed in local pixel coordinates of the
    generated image -- the regression target.

    Every row reaching this function is assumed already filtered by
    `filter_locatable_rows` (i.e. guaranteed to have >=1 VISIBLE, positioned
    WAP -- not necessarily among the raw top-k). Selection below picks the
    k-strongest AMONG POSITIONED APs directly (not "raw top-k, then filter
    to positioned"), which matters for sites with partial AP-position
    coverage (e.g. UJIIndoorLoc's estimated positions, ~30% coverage):
    train-time augmentation (noise, random AP-dropping) can shift the raw
    top-k away from any positioned AP, and "raw top-k then filter" could
    empty out even though a positioned AP is still visible. This is a
    no-op for 100%-coverage sites (HDLC, SODIndoorLoc) since raw-top-k and
    top-k-among-positioned are then identical. Still raises rather than
    silently leaking a trivial target if truly no positioned AP is visible
    at all -- the historical bug this exact code family had, see
    PIVOT_PLAN.md Item 2b.
    """
    visible_idx = np.where(rssi_row > rssi_min)[0]
    positioned_visible = np.array([i for i in visible_idx if wap_cols[i] in wap_positions])

    if len(positioned_visible) == 0:
        raise ValueError(
            "build_virtual_space received a row with no positioned "
            "visible WAP -- filter_locatable_rows should have caught "
            "this upstream."
        )

    strongest_idx = positioned_visible[np.argsort(-rssi_row[positioned_visible])][:k]

    if max_extent_m is not None:
        anchor_i = strongest_idx[0]
        ax0, ay0 = wap_positions[wap_cols[anchor_i]]
        kept = [
            i for i in strongest_idx
            if abs(wap_positions[wap_cols[i]][0] - ax0) <= max_extent_m
            and abs(wap_positions[wap_cols[i]][1] - ay0) <= max_extent_m
        ]
        if kept:
            strongest_idx = kept

    ap_xy = np.array([wap_positions[wap_cols[i]] for i in strongest_idx])

    x_min = ap_xy[:, 0].min() - margin_m
    x_max = ap_xy[:, 0].max() + margin_m
    y_min = ap_xy[:, 1].min() - margin_m
    y_max = ap_xy[:, 1].max() + margin_m

    x_min = min(x_min, x - 0.5)
    x_max = max(x_max, x + 0.5)
    y_min = min(y_min, y - 0.5)
    y_max = max(y_max, y + 0.5)

    width_px = max(1, int(np.ceil(max(x_max - x_min, PIXEL_METERS) / PIXEL_METERS)))
    height_px = max(1, int(np.ceil(max(y_max - y_min, PIXEL_METERS) / PIXEL_METERS)))

    ap_info = []
    for i in strongest_idx:
        ax, ay = wap_positions[wap_cols[i]]
        col_px = int(np.clip((ax - x_min) / PIXEL_METERS, 0, width_px - 1))
        row_px = int(np.clip((ay - y_min) / PIXEL_METERS, 0, height_px - 1))
        rssi_val = float(np.clip(rssi_row[i], rssi_min, rssi_max))
        intensity = (rssi_val - rssi_min) / (rssi_max - rssi_min)
        ap_info.append((row_px, col_px, intensity))

    vx = (x - x_min) / PIXEL_METERS
    vy = (y - y_min) / PIXEL_METERS

    return ap_info, (vx, vy), (x_min, y_min), (height_px, width_px)


def generate_image(ap_info, img_hw):
    h, w = img_hw
    img = np.zeros((1, h, w), dtype=np.float32)
    for row_px, col_px, intensity in ap_info:
        img[0, row_px, col_px] = max(img[0, row_px, col_px], intensity)
    return img


def additive_noise(rssi_row, sigma, rng, rssi_min: float = _HDLC_RSSI_MIN, rssi_max: float = _HDLC_RSSI_MAX):
    noisy = rssi_row.copy()
    visible = noisy > rssi_min
    noise = rng.normal(0.0, sigma, size=noisy.shape)
    noisy[visible] = noisy[visible] + noise[visible]
    return np.clip(noisy, rssi_min, rssi_max)


def random_dropping(rssi_row, k, rng, rssi_min: float = _HDLC_RSSI_MIN, protect_idx=None):
    """protect_idx: indices that must not be dropped (e.g. the row's only
    remaining positioned-visible AP, for partial-coverage sites like
    UJIIndoorLoc -- dropping it would leave build_virtual_space with
    nothing to anchor an image on). No-op for 100%-coverage sites since
    protect_idx is only ever passed as non-None when coverage is partial."""
    dropped = rssi_row.copy()
    visible_idx = np.where(dropped > rssi_min)[0]
    if len(visible_idx) == 0:
        return dropped
    strongest_idx = visible_idx[np.argsort(-dropped[visible_idx])][:k]
    if protect_idx:
        strongest_idx = np.array([i for i in strongest_idx if i not in protect_idx])
    if len(strongest_idx) == 0:
        return dropped
    dropped[rng.choice(strongest_idx)] = rssi_min
    return dropped
