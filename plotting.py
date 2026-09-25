import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from wtf_code.config import P_FIG, T_SHIFT_POWER
from wtf_code.utils import rainfall_to_hydro_year
from wtf_code.wtf_core import fit_linear, fit_exponential, fit_powerlaw
from wtf_code.postprocess import assign_best_recharge, choose_best_dh


def plot_classic(rain_monthly, summary, bore_id):
    s = summary.copy()
    rain_hy = rainfall_to_hydro_year(rain_monthly)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(rain_hy["HydrologicalYear"], rain_hy["Rain_mm"], alpha=0.25, label="Rainfall")
    ax1.set_ylabel("Rainfall (mm)")

    ax2 = ax1.twinx()
    method_specs = {
        "MIN": ("R_min_p05_mm", "R_min_p50_mm", "R_min_p95_mm", "o"),
        "LIN": ("R_lin_p05_mm", "R_lin_p50_mm", "R_lin_p95_mm", "s"),
        "EXP": ("R_exp_p05_mm", "R_exp_p50_mm", "R_exp_p95_mm", "^"),
        "POW": ("R_pow_p05_mm", "R_pow_p50_mm", "R_pow_p95_mm", "D"),
    }

    for method, (c05, c50, c95, marker) in method_specs.items():
        if not all(c in s.columns for c in [c05, c50, c95]):
            continue

        sub = s[np.isfinite(s[c50])].sort_values("HydrologicalYear")
        if len(sub) == 0:
            continue

        ax2.plot(sub["HydrologicalYear"], sub[c50], marker=marker, linestyle="-", label=method)

        finite_band = np.isfinite(sub[c05]) & np.isfinite(sub[c95])
        if finite_band.any():
            ax2.fill_between(
                sub.loc[finite_band, "HydrologicalYear"],
                sub.loc[finite_band, c05],
                sub.loc[finite_band, c95],
                alpha=0.15,
            )

    ax2.set_ylabel("Recharge (mm)")
    ax2.set_xlabel("Hydrological year")
    ax2.set_title(f"{bore_id} recharge summary")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(P_FIG / f"{bore_id}_summary.png", dpi=250)
    plt.close(fig)


def plot_diagnostic_from_r(r, bore_id):
    if r is None or "_debug" not in r:
        return

    hy = int(r["hydro_year"])
    dbg = r["_debug"]
    d = dbg["series_full"]
    win = dbg["fit_window"]
    x = dbg["x"]
    y = dbg["y"]

    date_tr = pd.to_datetime(r["date_trough"])
    date_pk = pd.to_datetime(r["date_peak"])
    wt_tr = float(r["wt_trough"])
    wt_pk = float(r["wt_peak"])

    fit_start = pd.to_datetime(r["fit_start"])
    rec_start, rec_end = fit_start, date_tr
    recov_start, recov_end = date_tr, date_pk

    shade_recession = mcolors.to_rgba("tab:olive", alpha=0.12)
    shade_recovery = mcolors.to_rgba("tab:green", alpha=0.10)

    def add_spans(ax):
        ax.axvspan(rec_start, rec_end, color=shade_recession, zorder=0)
        ax.axvspan(recov_start, recov_end, color=shade_recovery, zorder=0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    axA, axB, axC, axD = axs.ravel()

    axA.plot(d["Date"], d["WT_elev_m"], lw=1)
    add_spans(axA)
    axA.scatter([date_pk], [wt_pk], marker="D", color="0.6", zorder=5, label="Peak", s=70)
    axA.scatter([date_tr], [wt_tr], marker="o", color="0.4", zorder=5, label="Trough")
    axA.set_title(f"Bore {bore_id} - HY {hy}")
    axA.legend()

    txt = (
        f"Bore: {bore_id}\n"
        f"HY: {hy}\n"
        f"Trough: {date_tr.date()}  WT={wt_tr:.3f} m\n"
        f"Peak:   {date_pk.date()}  WT={wt_pk:.3f} m\n"
        f"ΔT (days): {r['trough_to_peak_days']:.0f}\n"
        f"n_peak_pts: {r.get('n_peak_pts', np.nan)}\n"
        f"fit_n: {r.get('fit_n', np.nan)}  fit_max_gap_days: {r.get('fit_max_gap_days', np.nan)}\n"
        f"QC: {r.get('qc_status', '')} {r.get('qc_reason', '')}\n"
        f"Δh_min: {r.get('delta_h_min', np.nan):.3f} m\n"
        f"Δh_lin: {r.get('delta_h_lin', np.nan):.3f} m  R2={r.get('r2_lin', np.nan):.2f}\n"
        f"Δh_exp: {r.get('delta_h_exp', np.nan):.3f} m  R2={r.get('r2_exp', np.nan):.2f}\n"
        f"Δh_pow: {r.get('delta_h_pow', np.nan):.3f} m  R2={r.get('r2_pow', np.nan):.2f}\n"
    )
    axB.axis("off")
    axB.text(0.02, 0.98, txt, va="top", family="monospace")

    axC.plot(win["Date"], win["WT_elev_m"], marker="x", ms=4, ls="None", label="Measurements", zorder=10)
    add_spans(axC)
    axC.axhline(wt_pk, ls="--", color="0.6", linewidth=1.75, label="Peak level")
    axC.axvline(date_tr, ls=":", color="0.4", linewidth=0.75, label="Trough date")

    dates_fit = win["Date"]

    lin = fit_linear(x, y)
    if lin is not None:
        _, _, yhat = lin
        axC.plot(dates_fit, yhat, lw=2, label=f"LIN (R2={r.get('r2_lin', np.nan):.2f})")

    expf = fit_exponential(x, y)
    if expf is not None:
        _, _, _, yhat = expf
        axC.plot(dates_fit, yhat, lw=2, label=f"EXP (R2={r.get('r2_exp', np.nan):.2f})")

    pw = fit_powerlaw(x, y, t_shift=T_SHIFT_POWER)
    if pw is not None:
        _, _, _, yhat = pw
        axC.plot(dates_fit, yhat, lw=2, label=f"POW (R2={r.get('r2_pow', np.nan):.2f})")

    shade_handles = [
        Patch(facecolor=shade_recession, edgecolor="none", label="Recession window"),
        Patch(facecolor=shade_recovery, edgecolor="none", label="Recovery window"),
    ]
    h, _ = axC.get_legend_handles_labels()
    axC.legend(handles=shade_handles + h, fontsize=8)
    axC.set_title("Recession fit window")

    seg = d[(d["Date"] >= date_tr) & (d["Date"] <= date_pk)].copy().sort_values("Date")
    if not seg.empty:
        mid = seg[(seg["Date"] > date_tr) & (seg["Date"] < date_pk)]
        if not mid.empty:
            axD.plot(mid["Date"], mid["WT_elev_m"], marker="x", ms=4, ls="None", label="Measurements")

    axD.scatter([date_pk], [wt_pk], marker="D", color="0.6", zorder=7, label="Peak", s=70)
    axD.scatter([date_tr], [wt_tr], marker="o", color="0.4", zorder=7, label="Trough")
    add_spans(axD)
    axD.axvline(date_pk, ls="--", color="0.6", linewidth=1.75, label="Peak date/level")
    axD.axhline(wt_pk, ls="--", color="0.6", linewidth=1.75)
    axD.axvline(date_tr, ls=":", color="0.4", linewidth=0.75, label="Trough date")
    axD.set_title("Recovery segment")

    hD, _ = axD.get_legend_handles_labels()
    axD.legend(handles=shade_handles + hD, fontsize=8)

    fig.tight_layout()
    fig.savefig(P_FIG / f"{bore_id}_HY{hy}_diag.png", dpi=250)
    plt.close(fig)


def plot_recession_slope_diagnostics(diag_csv, outpath=None):
    df = pd.read_csv(
        diag_csv,
        parse_dates=["date_trough", "date_peak", "fit_start", "fit_end"],
    )

    d = df.copy()
    if "qc_status" in d.columns:
        d = d[d["qc_status"] == "OK"]

    d = d[np.isfinite(d["lin_slope_m_per_day"])]
    d = d[np.isfinite(d["wt_trough"])]

    if d.empty:
        raise ValueError("No valid rows with lin_slope_m_per_day and wt_trough found.")

    d["abs_lin_slope_m_per_day"] = np.abs(d["lin_slope_m_per_day"])

    fig = plt.figure(figsize=(12, 4.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.plot(
        d["HydrologicalYear"],
        d["lin_slope_m_per_day"],
        marker="o",
        linestyle="-",
    )
    ax1.axhline(0, linestyle="--", linewidth=1.0, color="black")
    ax1.set_xlabel("Hydrological year")
    ax1.set_ylabel("Linear slope (m/day)")
    ax1.set_title("(A) Recession slope through time")
    ax1.grid(True, alpha=0.25)

    sc = ax2.scatter(
        d["wt_trough"],
        d["abs_lin_slope_m_per_day"],
        c=d["HydrologicalYear"],
        s=42,
        alpha=0.9,
    )
    ax2.set_xlabel("Trough water level (m)")
    ax2.set_ylabel(r"$|$Linear slope$|$ (m/day)")
    ax2.set_title("(B) Slope vs trough level")
    ax2.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Hydrological year")

    if len(d) >= 2:
        x = d["wt_trough"].to_numpy()
        y = d["abs_lin_slope_m_per_day"].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax2.plot(xx, m * xx + b, linestyle="--", linewidth=1.2, color="black")

    ax3.hist(
        d["abs_lin_slope_m_per_day"],
        bins=15,
        edgecolor="black",
        alpha=0.8,
    )
    med = np.nanmedian(d["abs_lin_slope_m_per_day"])
    ax3.axvline(med, linestyle="-", linewidth=1.5, color="black", label=f"Median = {med:.4f}")
    ax3.set_xlabel(r"$|$Linear slope$|$ (m/day)")
    ax3.set_ylabel("Count")
    ax3.set_title("(C) Distribution of slope magnitude")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(frameon=False)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2, ax3)


def plot_recharge_bias_vs_slope(df_res, outpath=None):
    d = df_res.copy()

    if "R_P50_mm" not in d.columns:
        d["R_P50_mm"] = d.apply(assign_best_recharge, axis=1)

    required = ["R_min_mm", "R_P50_mm", "lin_slope_m_per_day"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = d[
        np.isfinite(d["R_min_mm"])
        & np.isfinite(d["R_P50_mm"])
        & np.isfinite(d["lin_slope_m_per_day"])
    ].copy()

    if d.empty:
        raise ValueError("No valid rows available for plotting.")

    d["abs_lin_slope"] = np.abs(d["lin_slope_m_per_day"])
    d["dR_mm"] = d["R_P50_mm"] - d["R_min_mm"]

    fig = plt.figure(figsize=(12, 4.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.scatter(
        d["abs_lin_slope"],
        d["dR_mm"],
        s=42,
        alpha=0.85,
    )

    if len(d) >= 2:
        x = d["abs_lin_slope"].to_numpy()
        y = d["dR_mm"].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax1.plot(xx, m * xx + b, linestyle="--", linewidth=1.2, color="black")

    ax1.set_xlabel(r"$|$Linear slope$|$ (m/day)")
    ax1.set_ylabel(r"$R_{P50} - R_{min}$ (mm)")
    ax1.set_title("(A) Recharge correction vs slope")
    ax1.grid(True, alpha=0.25)

    if "wt_trough" in d.columns and np.isfinite(d["wt_trough"]).any():
        sc2 = ax2.scatter(
            d["abs_lin_slope"],
            d["dR_mm"],
            c=d["wt_trough"],
            s=42,
            alpha=0.9,
        )
        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label("Trough water level (m)")
    else:
        ax2.scatter(
            d["abs_lin_slope"],
            d["dR_mm"],
            s=42,
            alpha=0.85,
        )

    ax2.set_xlabel(r"$|$Linear slope$|$ (m/day)")
    ax2.set_ylabel(r"$R_{P50} - R_{min}$ (mm)")
    ax2.set_title("(B) by trough level")
    ax2.grid(True, alpha=0.25)

    if "r2_best" in d.columns and np.isfinite(d["r2_best"]).any():
        sc3 = ax3.scatter(
            d["abs_lin_slope"],
            d["dR_mm"],
            c=d["r2_best"],
            s=42,
            alpha=0.9,
        )
        cbar3 = fig.colorbar(sc3, ax=ax3)
        cbar3.set_label(r"Best-model $R^2$")
        ax3.set_title("(C) Coloured by fit quality")
    elif "fit_n" in d.columns and np.isfinite(d["fit_n"]).any():
        sc3 = ax3.scatter(
            d["abs_lin_slope"],
            d["dR_mm"],
            c=d["fit_n"],
            s=42,
            alpha=0.9,
        )
        cbar3 = fig.colorbar(sc3, ax=ax3)
        cbar3.set_label("Number of fit points")
        ax3.set_title("(C) by data support")
    else:
        ax3.scatter(
            d["abs_lin_slope"],
            d["dR_mm"],
            s=42,
            alpha=0.85,
        )
        ax3.set_title("(C) Recharge correction vs slope")

    ax3.set_xlabel(r"$|$Linear slope$|$ (m/day)")
    ax3.set_ylabel(r"$R_{P50} - R_{min}$ (mm)")
    ax3.grid(True, alpha=0.25)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2, ax3)


def plot_validation_summary(df_res, outpath=None):
    d = df_res.copy()

    required = ["delta_h_min", "R_min_mm", "best_model"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "R_P50_mm" not in d.columns:
        d["R_P50_mm"] = d.apply(assign_best_recharge, axis=1)

    if "dh_min_m" not in d.columns:
        d["dh_min_m"] = d["delta_h_min"]

    d["dh_model_m"] = d.apply(choose_best_dh, axis=1)
    d["dR_mm"] = d["R_P50_mm"] - d["R_min_mm"]

    fig = plt.figure(figsize=(12, 4.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    sub1 = d[np.isfinite(d["dh_min_m"]) & np.isfinite(d["dh_model_m"])]
    if not sub1.empty:
        ax1.scatter(sub1["dh_min_m"], sub1["dh_model_m"], s=28, alpha=0.8)
        xymax1 = np.nanmax([sub1["dh_min_m"].max(), sub1["dh_model_m"].max()]) * 1.05
        ax1.plot([0, xymax1], [0, xymax1], ls="--", lw=1.2, color="black")
        ax1.set_xlim(0, xymax1)
        ax1.set_ylim(0, xymax1)

    ax1.set_xlabel(r"$\Delta h_{min}$ (m)")
    ax1.set_ylabel(r"$\Delta h_{model}$ (m)")
    ax1.set_title("(A) Rise comparison")
    ax1.grid(True, alpha=0.25)

    sub2 = d[np.isfinite(d["R_min_mm"]) & np.isfinite(d["R_P50_mm"])]
    if not sub2.empty:
        ax2.scatter(sub2["R_min_mm"], sub2["R_P50_mm"], s=28, alpha=0.8)
        xymax2 = np.nanmax([sub2["R_min_mm"].max(), sub2["R_P50_mm"].max()]) * 1.05
        ax2.plot([0, xymax2], [0, xymax2], ls="--", lw=1.2, color="black")
        ax2.set_xlim(0, xymax2)
        ax2.set_ylim(0, xymax2)

    ax2.set_xlabel(r"$R_{min}$ (mm)")
    ax2.set_ylabel(r"$R_{P50}$ (mm)")
    ax2.set_title("(B) Recharge comparison")
    ax2.grid(True, alpha=0.25)

    sub3 = d[np.isfinite(d["dR_mm"])]
    if not sub3.empty:
        ax3.hist(sub3["dR_mm"], bins=30, edgecolor="black", alpha=0.8)
        ax3.axvline(0, ls="--", lw=1.2, color="black")
        med = np.nanmedian(sub3["dR_mm"])
        ax3.axvline(med, lw=1.5, color="black")

    ax3.set_xlabel(r"$R_{P50} - R_{min}$ (mm)")
    ax3.set_ylabel("Count")
    ax3.set_title("(C) Added recharge")
    ax3.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2, ax3)


def plot_dh_correction_vs_slope_dt(df_res, outpath=None):
    d = df_res.copy()

    required = [
        "best_model",
        "delta_h_min",
        "lin_slope_m_per_day",
        "trough_to_peak_days",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d["dh_model_m"] = d.apply(choose_best_dh, axis=1)
    d["dh_corr_m"] = d["dh_model_m"] - d["delta_h_min"]
    d["dh_pred_m"] = np.abs(d["lin_slope_m_per_day"]) * d["trough_to_peak_days"]

    d = d[np.isfinite(d["dh_corr_m"]) & np.isfinite(d["dh_pred_m"])].copy()

    if d.empty:
        raise ValueError("No valid rows available for plotting.")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        d["dh_pred_m"],
        d["dh_corr_m"],
        s=40,
        alpha=0.85,
    )

    xymax = np.nanmax([d["dh_pred_m"].max(), d["dh_corr_m"].max()]) * 1.05
    ax.plot([0, xymax], [0, xymax], ls="--", lw=1.2, color="black")

    ax.set_xlim(0, xymax)
    ax.set_ylim(0, xymax)
    ax.set_xlabel(r"Predicted correction, $|dh/dt| \times \Delta t$ (m)")
    ax.set_ylabel(r"Observed correction, $\Delta h_{model} - \Delta h_{min}$ (m)")
    ax.set_title("Recession correction diagnostic")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_scatter_with_flags(
    df,
    xcol,
    ycol,
    xlabel,
    ylabel,
    title,
    outpath,
    model_col="best_model",
    flagged_col="is_flagged",
):
    fig, ax = plt.subplots(figsize=(7, 5))

    if model_col in df.columns:
        for model_name, sub in df.groupby(model_col):
            ax.scatter(
                sub[xcol],
                sub[ycol],
                s=40,
                alpha=0.85,
                label=str(model_name),
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
                label="Flagged",
            )

    if "HydrologicalYear" in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row[xcol]) and pd.notna(row[ycol]):
                ax.annotate(
                    str(row["HydrologicalYear"]),
                    (row[xcol], row[ycol]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
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


__all__ = [
    "plot_classic",
    "plot_diagnostic_from_r",
    "plot_recession_slope_diagnostics",
    "plot_recharge_bias_vs_slope",
    "plot_validation_summary",
    "plot_dh_correction_vs_slope_dt",
    "plot_scatter_with_flags",
]