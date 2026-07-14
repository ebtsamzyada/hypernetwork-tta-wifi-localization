"""Multi-site Dataset: wraps several sites' dataframes WITHOUT concatenating
their (incompatible -- different AP columns/counts) raw columns. Keeps each
site's df/wap_positions/wap_cols separate and builds a flat (site_id, row)
index so a single DataLoader can shuffle across all sites while each
__getitem__ call still uses the right site's own AP set."""
import numpy as np
import torch
from torch.utils.data import Dataset

from sites import RSSI_MAX, RSSI_MIN
from virtual_space import additive_noise, build_virtual_space, generate_image, random_dropping


class MultiSiteDataset(Dataset):
    def __init__(self, site_dfs: dict, site_meta: dict, hp, augment=False, seed=0):
        """site_dfs: {site_id: df}. site_meta: {site_id: (wap_positions, wap_cols)}."""
        self.site_dfs = {sid: df.reset_index(drop=True) for sid, df in site_dfs.items()}
        self.site_meta = site_meta
        self.hp = hp
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.index = [(sid, i) for sid, df in self.site_dfs.items() for i in range(len(df))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        site_id, i = self.index[idx]
        df = self.site_dfs[site_id]
        wap_positions, wap_cols = self.site_meta[site_id]
        row = df.iloc[i]
        x, y = float(row["x"]), float(row["y"])
        floor_class = int(row["floor_class"])
        orig_rssi_row = row[wap_cols].to_numpy(dtype=np.float32)
        rssi_row = orig_rssi_row

        if self.augment:
            if self.rng.random() < 0.5:
                rssi_row = additive_noise(rssi_row, self.hp.noise_std, self.rng, RSSI_MIN, RSSI_MAX)
            if self.rng.random() < 0.5:
                # Protect the row's only remaining positioned-visible AP(s)
                # from being dropped -- matters for partial-coverage sites
                # (UJIIndoorLoc's estimated positions); see
                # PIVOT_PLAN.md "Phase 5b" bug note.
                visible_idx = np.where(rssi_row > RSSI_MIN)[0]
                positioned_visible = [i for i in visible_idx if wap_cols[i] in wap_positions]
                protect_idx = set(positioned_visible) if len(positioned_visible) <= 1 else None
                rssi_row = random_dropping(rssi_row, self.hp.k_strongest, self.rng, RSSI_MIN, protect_idx)

        try:
            ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
                x, y, rssi_row, wap_cols, wap_positions, self.hp.k_strongest, self.hp.margin_m, self.hp.max_extent_m,
                rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
            )
        except ValueError:
            # Residual edge case (e.g. additive noise pushing a borderline
            # reading below the visibility threshold): fall back to the
            # unaugmented row, which filter_locatable_rows already
            # guaranteed has >=1 positioned visible AP. Skips augmentation
            # for this one draw rather than crashing or fabricating a
            # label.
            ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
                x, y, orig_rssi_row, wap_cols, wap_positions, self.hp.k_strongest, self.hp.margin_m, self.hp.max_extent_m,
                rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
            )
        img = generate_image(ap_info, img_hw)

        return {
            "image": torch.from_numpy(img),
            "target_xy": torch.tensor([vx, vy], dtype=torch.float32),
            "target_floor_class": floor_class,
            "site_id": site_id,
            "origin": torch.tensor([x_min, y_min], dtype=torch.float32),
            "physical_xy": torch.tensor([x, y], dtype=torch.float32),
        }


def collate_single(batch):
    return batch
