"""Naive constant-prediction baseline. Report this on every run -- the prior
project's discipline: a loud warning if a real model doesn't beat it."""
import numpy as np
import pandas as pd


def naive_constant_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    pred_x, pred_y = float(train_df["x"].mean()), float(train_df["y"].mean())
    xy_err = np.sqrt((eval_df["x"] - pred_x) ** 2 + (eval_df["y"] - pred_y) ** 2)

    majority_floor = int(train_df["floor"].mode().iloc[0])
    floor_acc = float((eval_df["floor"] == majority_floor).mean())

    return {
        "pred_xy": (pred_x, pred_y),
        "majority_floor": majority_floor,
        "median_xy_err_m": float(xy_err.median()),
        "mean_xy_err_m": float(xy_err.mean()),
        "floor_accuracy": floor_acc,
        "n_eval": len(eval_df),
    }
