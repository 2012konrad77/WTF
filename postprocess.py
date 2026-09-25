import numpy as np
import pandas as pd


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
    Add boolean column 'is_flagged' based on qc_reason and qc_status.
    """
    reason_col = find_column(df, ["qc_reason"], required=False)
    status_col = find_column(df, ["qc_status"], required=False)

    flagged = pd.Series(False, index=df.index)

    if reason_col is not None:
        flagged |= df[reason_col].fillna("").astype(str).str.strip().ne("")

    if status_col is not None:
        flagged |= ~df[status_col].fillna("").astype(str).str.lower().isin(
            ["ok", "pass", "accepted"]
        )

    df = df.copy()
    df["is_flagged"] = flagged
    return df


def choose_recharge_column(row):
    """
    Return a single recharge value based on best_model.
    """
    model = str(row["best_model"]).strip().lower()

    if model in ["lin", "linear"]:
        return row.get("R_lin_p50_mm", np.nan)
    elif model in ["exp", "exponential"]:
        return row.get("R_exp_p50_mm", np.nan)
    elif model in ["pow", "power", "power-law", "power_law"]:
        return row.get("R_pow_p50_mm", np.nan)
    elif model in ["min", "minimum"]:
        return row.get("R_min_p50_mm", row.get("R_min_mm", np.nan))

    for c in ["R_lin_p50_mm", "R_exp_p50_mm", "R_pow_p50_mm", "R_min_p50_mm", "R_min_mm"]:
        if c in row.index and pd.notna(row[c]):
            return row[c]

    return np.nan


def choose_delta_h_column(row):
    """
    Return a single delta-h value based on best_model, preferring
    diagnostic columns over summary columns.
    """
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

    return np.nan


def assign_best_recharge(row):
    """
    Return best-model recharge from unsuffixed columns, for use in
    validation plots working directly on summary-style tables.
    """
    model = str(row.get("best_model", "MIN")).strip().upper()

    if model == "LIN":
        return row.get("R_lin_p50_mm", np.nan)
    elif model == "EXP":
        return row.get("R_exp_p50_mm", np.nan)
    elif model == "POW":
        return row.get("R_pow_p50_mm", np.nan)
    else:
        return row.get("R_min_p50_mm", row.get("R_min_mm", np.nan))


def choose_best_dh(row):
    """
    Return best-model delta-h from unsuffixed columns, for validation plots.
    """
    model = str(row.get("best_model", "MIN")).strip().upper()

    if model == "LIN":
        return row.get("delta_h_lin", np.nan)
    elif model == "EXP":
        return row.get("delta_h_exp", np.nan)
    elif model == "POW":
        return row.get("delta_h_pow", np.nan)
    else:
        return row.get("delta_h_min", np.nan)


def assign_hydro_year(dates, start_month=4):
    """
    Assign hydrological year based on start_month.
    """
    dates = pd.to_datetime(dates)
    return dates.dt.year.where(dates.dt.month >= start_month, dates.dt.year - 1)


def load_hydro_year_rainfall(rain_csv, start_month=4):
    """
    Load daily rainfall CSV and aggregate to hydrological-year totals.
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


def merge_summary_and_diagnostics(summary, diag):
    """
    Merge summary and diagnostic dataframes, resolve duplicated fields,
    and create Recharge_mm, delta_h_best, and is_flagged.
    """
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

    for col in ["best_model", "qc_status", "r2_best"]:
        c_diag = f"{col}_diag"
        c_sum = f"{col}_sum"
        if c_diag in merged.columns and c_sum in merged.columns:
            merged[col] = merged[c_diag].combine_first(merged[c_sum])
        elif c_diag in merged.columns:
            merged[col] = merged[c_diag]
        elif c_sum in merged.columns:
            merged[col] = merged[c_sum]

    merged["Recharge_mm"] = merged.apply(choose_recharge_column, axis=1)
    merged["delta_h_best"] = merged.apply(choose_delta_h_column, axis=1)
    merged = add_flag_column(merged)
    return merged


def add_rainfall_to_merged(merged, rain_csv, start_month=4):
    """
    Add hydrological-year rainfall and recharge-ratio columns.
    """
    hy_rain = load_hydro_year_rainfall(rain_csv, start_month=start_month)
    merged = pd.merge(merged, hy_rain, on="HydrologicalYear", how="left")
    merged["Recharge_ratio"] = merged["Recharge_mm"] / merged["Rainfall_mm"].replace(0, np.nan)
    merged["Recharge_ratio_pct"] = 100.0 * merged["Recharge_ratio"]
    return merged