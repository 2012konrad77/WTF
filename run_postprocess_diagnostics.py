#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from wtf_code.config import RAW_RAIN_DIR
from wtf_code.config import P_TXT, P_FIG


# ============================================================
# SETTINGS
# ============================================================
bore_id = "61610918"

txt_dir = P_TXT
fig_dir = P_FIG
fig_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def find_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of these columns were found: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def add_flag_column(df):
    """
    Mark rows as flagged if qc_reason is non-empty or qc_status is not clearly 'ok'.
    """
    reason_col = find_column(df, ["qc_reason"], required=False)
    status_col = find_column(df, ["qc_status"], required=False)

    flagged = pd.Series(False, index=df.index)

    if reason_col is not None:
        flagged |= df[reason_col].fillna("").astype(str).str.strip().ne("")

    if status_col is not None:
        flagged |= ~df[status_col].fillna("").astype(str).str.lower().isin(["ok", "pass", "accepted"])

    df["is_flagged"] = flagged
    return df


def choose_recharge_column(row):
    """
    Build a single recharge value from best_model and model-specific p50 columns.
    """
    model = str(row["best_model"]).strip().lower()

    if model in ["lin", "linear"]:
        return row.get("R_lin_p50_mm", pd.NA)
    elif model in ["exp", "exponential"]:
        return row.get("R_exp_p50_mm", pd.NA)
    elif model in ["pow", "power", "power-law", "power_law"]:
        return row.get("R_pow_p50_mm", pd.NA)
    elif model in ["min", "minimum"]:
        if "R_min_p50_mm" in row.index:
            return row.get("R_min_p50_mm", pd.NA)
        return row.get("R_min_mm", pd.NA)

    # fallback: if unknown model, try the p50 columns in sensible order
    for c in ["R_lin_p50_mm", "R_exp_p50_mm", "R_pow_p50_mm", "R_min_p50_mm", "R_min_mm"]:
        if c in row.index and pd.notna(row[c]):
            return row[c]

    return pd.NA


def choose_delta_h_column(row):
    model = str(row["best_model"]).strip().lower()

    if model in ["lin", "linear"]:
        for c in ["delta_h_lin_diag", "delta_h_lin_sum"]:
            if c in row.index and pd.notna(row[c]):
                return row[c]

    elif model in ["exp", "exponential"]:
        for c in ["delta_h_exp_diag", "delta_h_exp_sum"]:
            if c in row.index and pd.notna(row[c]):
                return row[c]

    elif model in ["pow", "power", "power-law", "power_law"]:
        for c in ["delta_h_pow_diag", "delta_h_pow_sum"]:
            if c in row.index and pd.notna(row[c]):
                return row[c]

    elif model in ["min", "minimum"]:
        for c in ["delta_h_min_diag", "delta_h_min_sum"]:
            if c in row.index and pd.notna(row[c]):
                return row[c]

    for c in [
        "delta_h_lin_diag", "delta_h_lin_sum",
        "delta_h_exp_diag", "delta_h_exp_sum",
        "delta_h_pow_diag", "delta_h_pow_sum",
        "delta_h_min_diag", "delta_h_min_sum",
    ]:
        if c in row.index and pd.notna(row[c]):
            return row[c]

    return pd.NA


def assign_hydro_year(dates, start_month=4):
    """
    Assign hydrological year label.
    Example with start_month=4:
      2022-03-31 -> 2021
      2022-04-01 -> 2022
    """
    dates = pd.to_datetime(dates)
    return dates.dt.year.where(dates.dt.month >= start_month, dates.dt.year - 1)


def load_hydro_year_rainfall(rain_csv, start_month=4):
    """
    Read daily rainfall CSV and aggregate to hydrological-year totals.

    Expected columns:
      - one date column, e.g. Date / date / time
      - one rainfall column, e.g. Rain_mm / rainfall_mm / P_mm / rainfall

    Returns dataframe with:
      HydrologicalYear, Rainfall_mm
    """
    rain = pd.read_csv(rain_csv)

    date_col = find_column(
        rain,
        ["Date", "date", "time", "Time", "datetime", "Datetime"]
    )

    rain_col = find_column(
        rain,
        ["Rain_mm", "rain_mm", "rainfall_mm", "Rainfall_mm", "P_mm", "rainfall", "rain"]
    )

    rain[date_col] = pd.to_datetime(rain[date_col])
    rain[rain_col] = pd.to_numeric(rain[rain_col], errors="coerce").fillna(0.0)

    rain["HydrologicalYear"] = assign_hydro_year(rain[date_col], start_month=start_month)

    hy_rain = (
        rain.groupby("HydrologicalYear", as_index=False)[rain_col]
        .sum()
        .rename(columns={rain_col: "Rainfall_mm"})
    )

    return hy_rain


hy_rain = load_hydro_year_rainfall(rain_path, start_month=START_MONTH)

merged = pd.merge(
    merged,
    hy_rain,
    on="HydrologicalYear",
    how="left"
)

merged["Recharge_ratio"] = merged["Recharge_mm"] / merged["Rainfall_mm"]

##################
# Plotting
##################

def plot_scatter_with_flags(
    df,
    xcol,
    ycol,
    xlabel,
    ylabel,
    title,
    outpath,
    model_col="best_model",
    flagged_col="is_flagged"
):
    fig, ax = plt.subplots(figsize=(7, 5))

    if model_col in df.columns:
        for model_name, sub in df.groupby(model_col):
            ax.scatter(
                sub[xcol],
                sub[ycol],
                s=40,
                alpha=0.85,
                label=str(model_name)
            )
    else:
        ax.scatter(df[xcol], df[ycol], s=40, alpha=0.85, label="Years")

    if flagged_col in df.columns:
        flagged = df[df[flagged_col]]
        if not flagged.empty:
            ax.scatter(
                flagged[xcol],
                flagged[ycol],
                s=110,
                facecolors="none",
                edgecolors="red",
                linewidths=1.4,
                label="Flagged"
            )

    if "HydrologicalYear" in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row[xcol]) and pd.notna(row[ycol]):
                ax.annotate(
                    str(row["HydrologicalYear"]),
                    (row[xcol], row[ycol]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points"
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# 5. Recharge vs hydro-year rainfall
plot_scatter_with_flags(
    merged.dropna(subset=["Rainfall_mm", "Recharge_mm"]),
    xcol="Rainfall_mm",
    ycol="Recharge_mm",
    xlabel="Hydro-year rainfall (mm)",
    ylabel="Recharge (mm)",
    title=f"{bore_id}: Recharge vs hydro-year rainfall",
    outpath=fig_dir / f"{bore_id}_recharge_vs_rainfall.png",
)

# 6. Recharge ratio vs hydro-year rainfall
plot_scatter_with_flags(
    merged.dropna(subset=["Rainfall_mm", "Recharge_ratio"]),
    xcol="Rainfall_mm",
    ycol="Recharge_ratio",
    xlabel="Hydro-year rainfall (mm)",
    ylabel="Recharge ratio (-)",
    title=f"{bore_id}: Recharge ratio vs hydro-year rainfall",
    outpath=fig_dir / f"{bore_id}_rr_vs_rainfall.png",
)




# ============================================================
# READ DATA
# ============================================================
summary_path = txt_dir / f"{bore_id}_recharge_summary.csv"
diag_path = txt_dir / f"{bore_id}_diagnostics.csv"

summary = pd.read_csv(summary_path)
diag = pd.read_csv(diag_path)

print("Summary columns:")
print(list(summary.columns))
print("\nDiagnostics columns:")
print(list(diag.columns))


# ============================================================
# MERGE
# ============================================================
merged = pd.merge(
    summary,
    diag[
        [
            "HydrologicalYear",
            "qc_status",
            "qc_reason",
            "best_model",
            "r2_best",
            "n_peak_pts",
            "fit_max_gap_days",
            "trough_to_peak_days",
            "fit_n",
            "fit_duration_days",
            "lin_slope_m_per_day",
            "exp_k",
            "pow_k",
            "delta_h_min",
            "delta_h_lin",
            "delta_h_exp",
            "delta_h_pow",
        ]
    ],
    on="HydrologicalYear",
    how="left",
    suffixes=("_sum", "_diag")
)

# Prefer diagnostic best_model/qc_status if duplicated by merge
for col in ["best_model", "qc_status", "r2_best"]:
    c_diag = f"{col}_diag"
    c_sum = f"{col}_sum"
    if c_diag in merged.columns and c_sum in merged.columns:
        merged[col] = merged[c_diag].combine_first(merged[c_sum])
    elif c_diag in merged.columns:
        merged[col] = merged[c_diag]
    elif c_sum in merged.columns:
        merged[col] = merged[c_sum]

# Create single plotting columns
merged["Recharge_mm"] = merged.apply(choose_recharge_column, axis=1)
merged["delta_h_best"] = merged.apply(choose_delta_h_column, axis=1)

merged = add_flag_column(merged)

print("\nMerged columns:")
print(list(merged.columns))


# ============================================================
# PLOTS
# ============================================================

# 1. Recharge vs best delta_h
plot_scatter_with_flags(
    merged.dropna(subset=["delta_h_best", "Recharge_mm"]),
    xcol="delta_h_best",
    ycol="Recharge_mm",
    xlabel="Peak-trough rise, Δh (m)",
    ylabel="Recharge (mm)",
    title=f"{bore_id}: Recharge vs peak-trough rise",
    outpath=fig_dir / f"{bore_id}_recharge_vs_dh.png",
)

# 2. Recharge vs linear recession slope
plot_scatter_with_flags(
    merged.dropna(subset=["lin_slope_m_per_day", "Recharge_mm"]),
    xcol="lin_slope_m_per_day",
    ycol="Recharge_mm",
    xlabel="Linear recession slope (m/day)",
    ylabel="Recharge (mm)",
    title=f"{bore_id}: Recharge vs recession slope",
    outpath=fig_dir / f"{bore_id}_recharge_vs_slope.png",
)

# 3. Recharge vs number of fit points
plot_scatter_with_flags(
    merged.dropna(subset=["fit_n", "Recharge_mm"]),
    xcol="fit_n",
    ycol="Recharge_mm",
    xlabel="Number of fit points",
    ylabel="Recharge (mm)",
    title=f"{bore_id}: Recharge vs fit-point count",
    outpath=fig_dir / f"{bore_id}_recharge_vs_nfit.png",
)

# 4. Recharge vs max gap days
plot_scatter_with_flags(
    merged.dropna(subset=["fit_max_gap_days", "Recharge_mm"]),
    xcol="fit_max_gap_days",
    ycol="Recharge_mm",
    xlabel="Maximum gap in measurements (days)",
    ylabel="Recharge (mm)",
    title=f"{bore_id}: Recharge vs max gap days",
    outpath=fig_dir / f"{bore_id}_recharge_vs_gap.png",
)

print("\nDone. Figures saved to:")
print(fig_dir)
