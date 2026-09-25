#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from wtf_code.config import P_TXT, P_FIG
from wtf_code.plotting import (
    plot_recession_slope_diagnostics,
    plot_recharge_bias_vs_slope,
    plot_validation_summary,
    plot_dh_correction_vs_slope_dt,
    assign_best_recharge,
)

bore_id = "61610918"
txt_dir = P_TXT
fig_dir = P_FIG
fig_dir.mkdir(parents=True, exist_ok=True)

# 1) Slope diagnostics directly from diagnostics CSV.
plot_recession_slope_diagnostics(
    txt_dir / f"{bore_id}_diagnostics.csv",
    outpath=fig_dir / f"{bore_id}_slope_diagnostics.png",
)

# 2) Recharge-bias vs slope requires merged summary + diagnostics.
summary = pd.read_csv(txt_dir / f"{bore_id}_recharge_summary.csv")
diag = pd.read_csv(txt_dir / f"{bore_id}_diagnostics.csv")

df = pd.merge(
    summary,
    diag[[
        "HydrologicalYear",
        "lin_slope_m_per_day",
        "fit_n",
        "r2_best",
        "wt_trough",
        "trough_to_peak_days",
    ]],
    on="HydrologicalYear",
    how="left",
)

if "R_P50_mm" not in df.columns:
    df["R_P50_mm"] = df.apply(assign_best_recharge, axis=1)

plot_recharge_bias_vs_slope(
    df,
    outpath=fig_dir / f"{bore_id}_bias_vs_slope.png",
)

plot_validation_summary(
    df,
    outpath=fig_dir / f"{bore_id}_validation_summary.png",
)

plot_dh_correction_vs_slope_dt(
    df,
    outpath=fig_dir / f"{bore_id}_dh_correction_vs_slope_dt.png",
)

plt.show()
