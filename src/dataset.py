from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from virtual_space import (
    additive_noise,
    build_virtual_space,
    generate_image,
    random_dropping,
)


@dataclass
class HyperParams:
    k_strongest: int = 5
    noise_std: float = 2.0
    margin_m: float = 2.0
    max_extent_m: float = 10.0
    learning_rate: float = 1e-4
    pseudo_batch: int = 16
    lambda_floor: float = 1.0

    def as_dict(self):
        return dict(self.__dict__)


class GlobLocDataset(Dataset):
    """`df` must have x, y, floor, floor_class columns and already be
    filtered through filter_locatable_rows."""

    def __init__(self, df, wap_positions, wap_cols, hp: HyperParams, augment=False, seed=0):
        self.df = df.reset_index(drop=True)
        self.wap_positions = wap_positions
        self.wap_cols = wap_cols
        self.hp = hp
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y = float(row["x"]), float(row["y"])
        floor_id = int(row["floor"])
        floor_class = int(row["floor_class"])
        rssi_row = row[self.wap_cols].to_numpy(dtype=np.float32)

        if self.augment:
            if self.rng.random() < 0.5:
                rssi_row = additive_noise(rssi_row, self.hp.noise_std, self.rng)
            if self.rng.random() < 0.5:
                rssi_row = random_dropping(rssi_row, self.hp.k_strongest, self.rng)

        ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
            x, y, rssi_row, self.wap_cols, self.wap_positions,
            self.hp.k_strongest, self.hp.margin_m, self.hp.max_extent_m,
        )
        img = generate_image(ap_info, img_hw)

        return {
            "image": torch.from_numpy(img),
            "target_xy": torch.tensor([vx, vy], dtype=torch.float32),
            "target_floor_class": floor_class,
            "true_floor_id": floor_id,
            "origin": torch.tensor([x_min, y_min], dtype=torch.float32),
            "physical_xy": torch.tensor([x, y], dtype=torch.float32),
        }


def collate_single(batch):
    return batch
