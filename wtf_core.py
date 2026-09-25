import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from wtf_code.config import (
    RISE_TOL_M, RISE_CONFIRM_DAYS, MAX_DT_DAYS,
    MIN_POINTS, FALLBACK_BACKFIT_DAYS, MAX_FIT_DURATION_DAYS,
    T_SHIFT_POWER, MIN_FIT_POINTS_SOFT, MIN_PEAK_POINTS, MAX_GAP_DAYS_SOFT,
    TROUGH_MONTH_START, TROUGH_MONTH_END,
    MIN_R2_LIN, MIN_R2_EXP, MIN_R2_POW, MIN_R2_BEST,
    P_TXT,
)
from wtf_code.utils import hy_end, hydro_year_from_date, recharge_uncertainty, r2_score_safe, derive_best_model_and_r2


def fit_linear(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.unique(x).size < 2:
        return None
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    return float(m), float(b), yhat


def fit_exponential(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.unique(x).size < 2 or np.min(x) < 0:
        return None

    def f(t, c, A, k):
        return c + A * np.exp(-k * t)

    c0 = float(np.nanmin(y))
    A0 = max(float(np.nanmax(y) - c0), 1e-6)
    span = float(np.nanmax(x) - np.nanmin(x))
    k0 = 1.0 / max(span, 1.0)

    try:
        popt, _ = curve_fit(
            f, x, y, p0=(c0, A0, k0),
            bounds=((-np.inf, 0.0, 1e-6), (np.inf, np.inf, 10.0)),
            maxfev=20000,
        )
        c, A, k = popt
        yhat = f(x, c, A, k)
        return float(c), float(A), float(k), yhat
    except Exception:
        return None


def fit_powerlaw(x, y, t_shift=T_SHIFT_POWER):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.min(x) < 0 or t_shift <= 0:
        return None

    def f(t, c, A, k):
        return c + A * (t + t_shift) ** (-k)

    c0 = float(np.nanmin(y))
    A0 = max(float(np.nanmax(y) - c0), 1e-6)
    k0 = 1.0

    try:
        popt, _ = curve_fit(
            f, x, y, p0=(c0, A0, k0),
            bounds=((-np.inf, 0.0, 1e-6), (np.inf, np.inf, 10.0)),
            maxfev=20000,
        )
        c, A, k = popt
        yhat = f(x, c, A, k)
        return float(c), float(A), float(k), yhat
    except Exception:
        return None

def add_best_fit_metrics(out: dict, threshold: float = MIN_R2_BEST) -> dict:
    r2_series = pd.Series({
        "lin": out.get("r2_lin", np.nan),
        "exp": out.get("r2_exp", np.nan),
        "pow": out.get("r2_pow", np.nan),
    }, dtype=float)

    if r2_series.notna().any():
        out["best_r2"] = float(r2_series.max())
        out["best_model_by_r2"] = str(r2_series.idxmax())
        out["flag_poor_fit"] = bool(out["best_r2"] < threshold)
    else:
        out["best_r2"] = np.nan
        out["best_model_by_r2"] = np.nan
        out["flag_poor_fit"] = True

    return out

def wtf_trough_peak(df, hy, fit_start_override=None, return_debug=False):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    trough_start = pd.Timestamp(hy - 1, TROUGH_MONTH_START, 1)
    trough_end = pd.Timestamp(hy - 1, TROUGH_MONTH_END, 1) + pd.offsets.MonthEnd(0)
    cand = d[(d["Date"] >= trough_start) & (d["Date"] <= trough_end)].copy()

    if cand.empty:
        return {"hydro_year": int(hy), "qc_status": "SKIP", "qc_reason": "NO_TROUGH_IN_WINDOW"}

    best = None
    peak_start = pd.Timestamp(hy - 1, 8, 1)
    peak_end = pd.Timestamp(hy - 1, 12, 31)

    for _, row in cand.iterrows():
        date_tr = pd.to_datetime(row["Date"])
        wt_tr = float(row["WT_elev_m"])

        post = d[(d["Date"] > date_tr) & (d["Date"] <= date_tr + pd.Timedelta(days=RISE_CONFIRM_DAYS))]
        if post.empty:
            continue
        if float(post["WT_elev_m"].max() - wt_tr) < RISE_TOL_M:
            continue

        g_after = d[(d["Date"] > date_tr) & (d["Date"] <= hy_end(hy))].copy()
        g_after = g_after[(g_after["Date"] >= peak_start) & (g_after["Date"] <= peak_end)]
        if g_after.empty:
            continue

        i_pk = g_after["WT_elev_m"].idxmax()
        date_pk = pd.to_datetime(g_after.loc[i_pk, "Date"])
        wt_pk = float(g_after.loc[i_pk, "WT_elev_m"])
        n_peak_pts = int(len(g_after))

        dt_days = int((date_pk - date_tr).days)
        dh_obs = wt_pk - wt_tr
        if dt_days < 1 or dt_days > MAX_DT_DAYS or not np.isfinite(dh_obs) or dh_obs <= 0:
            continue

        score = float(dh_obs)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "date_tr": date_tr, "wt_tr": wt_tr,
                "date_pk": date_pk, "wt_pk": wt_pk,
                "n_peak_pts": n_peak_pts,
            }

    if best is None:
        return {"hydro_year": int(hy), "qc_status": "SKIP", "qc_reason": "NO_VALID_TROUGH_PEAK_PAIR"}

    date_tr = best["date_tr"]
    wt_tr = best["wt_tr"]
    date_pk = best["date_pk"]
    wt_pk = best["wt_pk"]
    n_peak_pts = best["n_peak_pts"]

    fit_start = None
    if fit_start_override is not None:
        fs = pd.to_datetime(fit_start_override)
        if (date_tr - fs).days <= MAX_FIT_DURATION_DAYS:
            fit_start = fs

    if fit_start is None:
        fit_start = max(d["Date"].min(), date_tr - pd.Timedelta(days=FALLBACK_BACKFIT_DAYS))
    if fit_start >= date_tr or (date_tr - fit_start).days > MAX_FIT_DURATION_DAYS:
        fit_start = max(d["Date"].min(), date_tr - pd.Timedelta(days=FALLBACK_BACKFIT_DAYS))

    win = d[(d["Date"] >= fit_start) & (d["Date"] <= date_tr)].copy()
    fit_max_gap_days = float(win["Date"].sort_values().diff().dt.days.max()) if len(win) >= 2 else np.nan

    qc_reasons = []
    if len(win) < MIN_POINTS:
        qc_reasons.append("FIT_N_LT_MIN_POINTS")
    if len(win) < MIN_FIT_POINTS_SOFT:
        qc_reasons.append("FIT_N_LT_MIN_FIT_POINTS_SOFT")
    if n_peak_pts < MIN_PEAK_POINTS:
        qc_reasons.append("PEAK_N_LT_MIN_PEAK_POINTS")
    if np.isfinite(fit_max_gap_days) and fit_max_gap_days > MAX_GAP_DAYS_SOFT:
        qc_reasons.append(f"FIT_MAX_GAP_GT_{MAX_GAP_DAYS_SOFT}D")

    if len(win) >= MIN_POINTS:
        t0 = pd.to_datetime(win["Date"].min())
        x = (win["Date"] - t0).dt.days.astype(float).to_numpy()
        y = win["WT_elev_m"].astype(float).to_numpy()
        x_pk = float((date_pk - t0).days)
    else:
        t0, x, y, x_pk = None, np.array([]), np.array([]), np.nan

    out = {
        "hydro_year": int(hy),
        "date_trough": date_tr,
        "wt_trough": float(wt_tr),
        "date_peak": date_pk,
        "wt_peak": float(wt_pk),
        "trough_to_peak_days": float((date_pk - date_tr).days),
        "fit_start": fit_start,
        "fit_end": date_tr,
        "fit_duration_days": float((date_tr - fit_start).days),
        "fit_n": int(len(win)),
        "fit_max_gap_days": float(fit_max_gap_days) if np.isfinite(fit_max_gap_days) else np.nan,
        "n_peak_pts": int(n_peak_pts),
    }

    if qc_reasons:
        out["qc_status"] = "SKIP"
        out["qc_reason"] = ";".join(qc_reasons)
        if return_debug:
            out["_debug"] = {
                "cand_window": cand.copy(),
                "series_full": d.copy(),
                "fit_window": win.copy(),
                "t0": t0,
                "x": x.copy(),
                "y": y.copy(),
                "x_pk": x_pk,
            }
        return out

    if np.unique(x).size < 2 or not np.isfinite(x_pk):
        out["qc_status"] = "SKIP"
        out["qc_reason"] = "INSUFFICIENT_X_VARIATION_OR_BAD_XPK"
        return out

    out["qc_status"] = "OK"
    out["qc_reason"] = ""
    out["delta_h_min"] = float(wt_pk - wt_tr)

    lin = fit_linear(x, y)
    if lin is None:
        out["lin_slope_m_per_day"] = np.nan
        out["r2_lin"] = np.nan
        out["delta_h_lin"] = np.nan
    else:
        m, b, yhat = lin
        out["lin_slope_m_per_day"] = float(m)
        out["r2_lin"] = float(r2_score_safe(y, yhat))
        out["delta_h_lin"] = (
            float(wt_pk - (m * x_pk + b))
            if np.isfinite(m) and m < 0 and out["r2_lin"] >= MIN_R2_LIN
            else np.nan
        )

    expf = fit_exponential(x, y)
    if expf is None:
        out["exp_k"] = np.nan
        out["r2_exp"] = np.nan
        out["delta_h_exp"] = np.nan
    else:
        c, A, k, yhat = expf
        out["exp_k"] = float(k)
        out["r2_exp"] = float(r2_score_safe(y, yhat))
        out["delta_h_exp"] = (
            float(wt_pk - (c + A * np.exp(-k * x_pk)))
            if out["r2_exp"] >= MIN_R2_EXP
            else np.nan
        )

    pw = fit_powerlaw(x, y, t_shift=T_SHIFT_POWER)
    if pw is None:
        out["pow_k"] = np.nan
        out["r2_pow"] = np.nan
        out["delta_h_pow"] = np.nan
    else:
        c, A, k, yhat = pw
        out["pow_k"] = float(k)
        out["r2_pow"] = float(r2_score_safe(y, yhat))
        base = x_pk + T_SHIFT_POWER
        out["delta_h_pow"] = (
            float(wt_pk - (c + A * base ** (-k)))
            if base > 0 and np.isfinite(k) and out["r2_pow"] >= MIN_R2_POW
            else np.nan
        )

    out = add_best_fit_metrics(out, threshold=MIN_R2_BEST)

    if return_debug:
        out["_debug"] = {
            "cand_window": cand.copy(),
            "series_full": d.copy(),
            "fit_window": win.copy(),
            "t0": t0,
            "x": x.copy(),
            "y": y.copy(),
            "x_pk": x_pk,
        }

    return out


def build_summary(df):
    df = df.copy()
    df["hydro_year"] = hydro_year_from_date(df["Date"])

    results = []
    prev_peak_date = None

    for hy in sorted(df["hydro_year"].unique()):
        r = wtf_trough_peak(df, hy, fit_start_override=prev_peak_date, return_debug=False)

        if r.get("qc_status") != "OK":
            prev_peak_date = None
            continue

        prev_peak_date = r["date_peak"]

        row = {
            "HydrologicalYear": int(hy),
            "qc_status": r.get("qc_status"),
            "date_trough": r.get("date_trough"),
            "wt_trough": r.get("wt_trough"),
            "date_peak": r.get("date_peak"),
            "wt_peak": r.get("wt_peak"),
            "delta_h_min": r.get("delta_h_min", np.nan),
            "delta_h_lin": r.get("delta_h_lin", np.nan),
            "delta_h_exp": r.get("delta_h_exp", np.nan),
            "delta_h_pow": r.get("delta_h_pow", np.nan),
            "r2_lin": r.get("r2_lin", np.nan),
            "r2_exp": r.get("r2_exp", np.nan),
            "r2_pow": r.get("r2_pow", np.nan),
            "best_r2": r.get("best_r2", np.nan),
            "best_model_by_r2": r.get("best_model_by_r2", np.nan),
            "flag_poor_fit": r.get("flag_poor_fit", np.nan),
        }

        for m in ["min", "lin", "exp", "pow"]:
            dh = r.get(f"delta_h_{m}", np.nan)
            p05, p50, p95 = recharge_uncertainty(dh)
            row[f"R_{m}_p05_mm"] = p05
            row[f"R_{m}_p50_mm"] = p50
            row[f"R_{m}_p95_mm"] = p95

        row["R_min_mm"] = row["R_min_p50_mm"]

        best_model, r2_best = derive_best_model_and_r2(r)
        row["best_model"] = best_model
        row["r2_best"] = r2_best

        results.append(row)

    return pd.DataFrame(results)

def build_diag_row(hy, r):
    best_model, r2_best = derive_best_model_and_r2(r)

    return {
        "HydrologicalYear": int(hy),
        "qc_status": r.get("qc_status"),
        "qc_reason": r.get("qc_reason"),
        "best_model": best_model,
        "r2_best": r2_best,
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
        "delta_h_min": r.get("delta_h_min"),
        "delta_h_lin": r.get("delta_h_lin"),
        "delta_h_exp": r.get("delta_h_exp"),
        "delta_h_pow": r.get("delta_h_pow"),
        "r2_lin": r.get("r2_lin"),
        "r2_exp": r.get("r2_exp"),
        "r2_pow": r.get("r2_pow"),
        "lin_slope_m_per_day": r.get("lin_slope_m_per_day"),
        "exp_k": r.get("exp_k"),
        "pow_k": r.get("pow_k"),
    }

def save_diagnostics_csv(diag_rows, bore_id):
    if not diag_rows:
        print("No diagnostics to save.")
        return

    diag = pd.DataFrame(diag_rows)

    preferred = [
        "HydrologicalYear",
        "qc_status", "qc_reason",
        "best_model", "r2_best",
        "best_r2", "best_model_by_r2", "flag_poor_fit",
        "n_peak_pts", "fit_max_gap_days",
        "date_trough", "wt_trough",
        "date_peak", "wt_peak",
        "trough_to_peak_days",
        "fit_start", "fit_end", "fit_n", "fit_duration_days",
        "delta_h_min", "delta_h_lin", "delta_h_exp", "delta_h_pow",
        "r2_lin", "r2_exp", "r2_pow",
        "lin_slope_m_per_day", "exp_k", "pow_k",
    ]

    cols = [c for c in preferred if c in diag.columns] + [c for c in diag.columns if c not in preferred]
    diag = diag[cols].sort_values("HydrologicalYear")
    num_cols = diag.select_dtypes(include="number").columns
    diag[num_cols] = diag[num_cols].round(4)

    outpath = P_TXT / f"{bore_id}_diagnostics.csv"
    diag.to_csv(outpath, index=False)
    print(f"Saved diagnostics CSV: {outpath}")