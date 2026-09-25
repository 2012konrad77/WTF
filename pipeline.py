from pathlib import Path
import pandas as pd
import numpy as np

from wtf_code.config import P_TXT, WT_TEST_DIR, WT_BY_BORE_DIR
from wtf_code.wtf_core import (
    build_summary,
    wtf_trough_peak,
    save_diagnostics_csv,
    derive_best_model_and_r2,
)
from wtf_code.utils import hydro_year_from_date


def build_diagnostics_rows(df):
    """
    Build per-hydrological-year diagnostic rows using the same logic as build_summary().
    """
    d = df.copy()
    d["hydro_year"] = hydro_year_from_date(d["Date"])

    diag_rows = []
    prev_peak_date = None

    for hy in sorted(d["hydro_year"].unique()):
        r = wtf_trough_peak(
            d,
            hy,
            fit_start_override=prev_peak_date,
            return_debug=False,
        )

        # derive best model for reporting
        best_model, r2_best = derive_best_model_and_r2(r)

        # preserve chaining only when current year passed QC
        if r.get("qc_status") == "OK":
            prev_peak_date = r.get("date_peak")
        else:
            prev_peak_date = None

        row = {
            "HydrologicalYear": int(hy),
            "qc_status": r.get("qc_status"),
            "qc_reason": r.get("qc_reason"),

            "best_model": best_model,
            "r2_best": r2_best,

            # NEW
            "best_r2": r.get("best_r2", np.nan),
            "best_model_by_r2": r.get("best_model_by_r2", np.nan),
            "flag_poor_fit": r.get("flag_poor_fit", np.nan),

            "n_peak_pts": r.get("n_peak_pts"),
            "fit_max_gap_days": r.get("fit_max_gap_days"),
            "date_trough": r.get("date_trough"),
            "wt_trough": r.get("wt_trough"),
            "date_peak": r.get("date_peak"),
            "wt_peak": r.get("wt_peak"),
            "trough_to_peak_days": r.get("trough_to_peak_days"),
            "fit_start": r.get("fit_start"),
            "fit_end": r.get("fit_end"),
            "fit_n": r.get("fit_n"),
            "fit_duration_days": r.get("fit_duration_days"),
            "delta_h_min": r.get("delta_h_min", np.nan),
            "delta_h_lin": r.get("delta_h_lin", np.nan),
            "delta_h_exp": r.get("delta_h_exp", np.nan),
            "delta_h_pow": r.get("delta_h_pow", np.nan),
            "r2_lin": r.get("r2_lin", np.nan),
            "r2_exp": r.get("r2_exp", np.nan),
            "r2_pow": r.get("r2_pow", np.nan),
            "lin_slope_m_per_day": r.get("lin_slope_m_per_day", np.nan),
            "exp_k": r.get("exp_k", np.nan),
            "pow_k": r.get("pow_k", np.nan),
            "hydro_year": int(hy),
        }

        diag_rows.append(row)

    return diag_rows


def run_one_bore(bore_id, ds_rain=None, use_test_dir=False):
    """
    Run WTF summary + diagnostics export for one bore.

    Parameters
    ----------
    bore_id : str
        Bore identifier, e.g. '61610918'
    ds_rain : xarray.Dataset, optional
        Included for compatibility with batch workflow; not used here unless
        you later extend the pipeline to add rainfall extraction.
    use_test_dir : bool, default False
        If True, read from WT_TEST_DIR. Otherwise read from WT_BY_BORE_DIR.
    """
    wt_dir = WT_TEST_DIR if use_test_dir else WT_BY_BORE_DIR
    wt_file = wt_dir / f"{bore_id}_WT.csv"

    if not wt_file.exists():
        raise FileNotFoundError(f"Missing WT file: {wt_file}")

    df = pd.read_csv(wt_file, parse_dates=["Date"])

    # summary
    summary = build_summary(df)
    summary_path = P_TXT / f"{bore_id}_recharge_summary.csv"

    if summary is None or summary.empty:
        print(f"No accepted summary rows for bore {bore_id}; not writing summary CSV.")
    else:
        summary.to_csv(summary_path, index=False)
        print(f"Wrote summary: {summary_path}")

    # diagnostics
    diag_rows = build_diagnostics_rows(df)
    save_diagnostics_csv(diag_rows, bore_id)

    return summary