from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from wtf_code.config import P_TXT
from wtf_code.postprocess import assign_best_recharge


# ============================================================
# IO HELPERS
# ============================================================

def load_bore_coordinates(path_csv):
    """
    Load bore coordinates and keep one row per ID.
    Expected columns include: ID, x, y
    """
    coords = pd.read_csv(path_csv, sep=None, engine="python")
    coords.columns = [c.strip() for c in coords.columns]

    if "ID" not in coords.columns:
        raise ValueError(f"'ID' column not found in {path_csv}")

    coords["ID"] = coords["ID"].astype(str).str.strip()
    return coords.drop_duplicates(subset="ID", keep="first")


def load_daily_rain_csv(path_csv):
    """
    Load daily rainfall CSV in SILO-style format.

    Supported cases:
    - columns already named Date and Rain
    - whitespace-delimited columns: Year Month Day ... Rain
    """
    try:
        rain_df = pd.read_csv(path_csv)
        rain_df.columns = [c.strip() for c in rain_df.columns]
    except Exception:
        rain_df = None

    if rain_df is not None and {"Date", "Rain"}.issubset(rain_df.columns):
        rain_df["Date"] = pd.to_datetime(rain_df["Date"])
        rain_df["Rain"] = pd.to_numeric(rain_df["Rain"], errors="coerce").fillna(0.0)
        return rain_df[["Date", "Rain"]].copy()

    rain_df = pd.read_csv(
        path_csv,
        sep=r"\s+",
        names=["Year", "Month", "Day", "A", "Rain"],
        engine="python",
    )
    rain_df["Date"] = pd.to_datetime(rain_df[["Year", "Month", "Day"]])
    rain_df["Rain"] = pd.to_numeric(rain_df["Rain"], errors="coerce").fillna(0.0)
    return rain_df[["Date", "Rain"]].copy()




def assign_hydro_year_label(dates, start_month=4):
    """
    Hydrological year label:
    Apr 2020 -> HY 2021 if using the 'year ending' convention,
    or HY 2020 if using 'year starting' convention.

    Here we use the same convention as your current WTF outputs:
    dates in Apr-Dec are assigned to year+1.
    """
    dates = pd.to_datetime(dates)
    return np.where(dates.dt.month >= start_month, dates.dt.year + 1, dates.dt.year)


def hydro_year_rainfall_from_daily(rain_df, start_month=4):
    """
    Aggregate daily rainfall to hydrological-year totals.
    Returns columns: HydrologicalYear, Rain_mm
    """
    d = rain_df.copy()
    d["HydrologicalYear"] = assign_hydro_year_label(d["Date"], start_month=start_month)

    return (
        d.groupby("HydrologicalYear", as_index=False)["Rain"]
        .sum()
        .rename(columns={"Rain": "Rain_mm"})
        .sort_values("HydrologicalYear")
    )


# ============================================================
# RECHARGE SUMMARY INGEST
# ============================================================

def infer_bore_id_from_filename(path):
    name = Path(path).name
    m = re.match(r"(.+?)_recharge_summary", name)
    if m:
        return m.group(1)
    return Path(path).stem

def load_one_summary(path_summary):
    path_summary = Path(path_summary)

    if not path_summary.exists():
        return None

    if path_summary.read_text(errors="ignore").strip() == "":
        print(f"Skipping blank summary file: {path_summary.name}")
        return None

    try:
        df = pd.read_csv(path_summary)
    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable/empty summary file: {path_summary.name}")
        return None

    if df.empty:
        print(f"Skipping dataframe with no rows: {path_summary.name}")
        return None

    bore_id = infer_bore_id_from_filename(path_summary)
    df["ID"] = str(bore_id)

    if "Recharge_mm" not in df.columns:
        df["Recharge_mm"] = df.apply(assign_best_recharge, axis=1)

    return df

def load_all_summaries(txt_dir=P_TXT, pattern="*_recharge_summary*.csv"):
    """
    Load all bore summary CSVs from Results/txt.
    Skips empty/broken files.
    """
    txt_dir = Path(txt_dir)
    files = sorted(txt_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No summary files found in {txt_dir} with pattern {pattern}")

    frames = []
    skipped = 0

    for f in files:
        df = load_one_summary(f)
        if df is None:
            skipped += 1
            continue
        frames.append(df)

    if not frames:
        raise ValueError(f"No valid summary files could be loaded from {txt_dir}")

    out = pd.concat(frames, ignore_index=True)
    out["ID"] = out["ID"].astype(str).str.strip()

    print(f"Loaded {len(frames)} summary files; skipped {skipped} empty/bad files.")
    return out

# ============================================================
# DIAGNOSTICS INGEST
# ============================================================

def infer_bore_id_from_diag_filename(path):
    name = Path(path).name
    m = re.match(r"(.+?)_diagnostics", name)
    if m:
        return m.group(1)
    return Path(path).stem


def load_one_diagnostics(path_diag):
    path_diag = Path(path_diag)

    if not path_diag.exists():
        return None

    if path_diag.read_text(errors="ignore").strip() == "":
        print(f"Skipping blank diagnostics file: {path_diag.name}")
        return None

    try:
        df = pd.read_csv(path_diag)
    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable/empty diagnostics file: {path_diag.name}")
        return None

    if df.empty:
        print(f"Skipping dataframe with no rows: {path_diag.name}")
        return None

    bore_id = infer_bore_id_from_diag_filename(path_diag)
    df["ID"] = str(bore_id)

    return df


def load_all_diagnostics(txt_dir=P_TXT, pattern="*_diagnostics*.csv"):
    """
    Load all bore diagnostics CSVs from Results/txt.
    Skips empty/broken files.
    """
    txt_dir = Path(txt_dir)
    files = sorted(txt_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No diagnostics files found in {txt_dir} with pattern {pattern}")

    frames = []
    skipped = 0

    for f in files:
        df = load_one_diagnostics(f)
        if df is None:
            skipped += 1
            continue
        frames.append(df)

    if not frames:
        raise ValueError(f"No valid diagnostics files could be loaded from {txt_dir}")

    out = pd.concat(frames, ignore_index=True)
    out["ID"] = out["ID"].astype(str).str.strip()

    print(f"Loaded {len(frames)} diagnostics files; skipped {skipped} empty/bad files.")
    return out

def summarise_poor_fit_by_bore(diag_df, hy_min=None, hy_max=None, min_frac=0.5):
    """
    Summarise poor-fit behaviour at bore level from diagnostics CSVs.

    Expected columns:
        ID, HydrologicalYear, best_r2, flag_poor_fit
    """
    d = diag_df.copy()
    d["ID"] = d["ID"].astype(str).str.strip()

    if "HydrologicalYear" not in d.columns:
        raise ValueError("HydrologicalYear column missing from diagnostics data")
    if "flag_poor_fit" not in d.columns:
        raise ValueError("flag_poor_fit column missing from diagnostics data")
    if "best_r2" not in d.columns:
        raise ValueError("best_r2 column missing from diagnostics data")

    if hy_min is not None:
        d = d[d["HydrologicalYear"] >= hy_min]
    if hy_max is not None:
        d = d[d["HydrologicalYear"] <= hy_max]

    d["flag_poor_fit"] = d["flag_poor_fit"].fillna(True).astype(bool)
    d["best_r2"] = pd.to_numeric(d["best_r2"], errors="coerce")

    out = (
        d.groupby("ID", as_index=False)
         .agg(
             n_diag_years=("HydrologicalYear", "count"),
             n_poor_fit=("flag_poor_fit", "sum"),
             frac_poor_fit=("flag_poor_fit", "mean"),
             median_best_r2=("best_r2", "median"),
         )
    )

    out["problem_well"] = out["frac_poor_fit"] >= 0.5

    return out


def merge_problem_wells_into_stats(stats_df, poor_fit_df):
    """
    Merge bore-level poor-fit diagnostics onto bore-level mapping stats.
    """
    out = stats_df.merge(poor_fit_df, on="ID", how="left")

    out["n_diag_years"] = out["n_diag_years"].fillna(0)
    out["n_poor_fit"] = out["n_poor_fit"].fillna(0)
    out["frac_poor_fit"] = out["frac_poor_fit"].fillna(0.0)
    out["median_best_r2"] = pd.to_numeric(out["median_best_r2"], errors="coerce")
    out["problem_well"] = out["problem_well"].fillna(False)

    return out

# ============================================================
# LONG-FORM R/P TABLE
# ============================================================

def build_rp_long_table(
    summaries_df,
    coords_df,
    hy_rain_df,
    keep_qc=("OK",),
    recharge_col="Recharge_mm",
    max_ratio=1.0,
    min_recharge=0.0,
):
    """
    Merge recharge summaries with coordinates and hydrological-year rainfall,
    then compute R/P ratio.

    keep_qc:
        tuple/list of qc_status values to retain, e.g.
        ("OK",) or ("OK", "SOFT_OK")
    """
    s = summaries_df.copy()
    c = coords_df.copy()
    r = hy_rain_df.copy()

    s["ID"] = s["ID"].astype(str).str.strip()
    c["ID"] = c["ID"].astype(str).str.strip()

    if "HydrologicalYear" not in s.columns:
        raise ValueError("HydrologicalYear column missing from summaries_df")
    if recharge_col not in s.columns:
        raise ValueError(f"{recharge_col} column missing from summaries_df")

    if keep_qc is not None and "qc_status" in s.columns:
        s = s[s["qc_status"].isin(list(keep_qc))].copy()

    merged = s.merge(c, on="ID", how="left")
    merged = merged.merge(r[["HydrologicalYear", "Rain_mm"]], on="HydrologicalYear", how="left")

    merged["Recharge_mm"] = pd.to_numeric(merged[recharge_col], errors="coerce")
    merged["Rain_mm"] = pd.to_numeric(merged["Rain_mm"], errors="coerce")

    merged["R_P_ratio"] = merged["Recharge_mm"] / merged["Rain_mm"]

    filt = (
        np.isfinite(merged["Recharge_mm"]) &
        np.isfinite(merged["Rain_mm"]) &
        np.isfinite(merged["R_P_ratio"]) &
        (merged["Recharge_mm"] > min_recharge) &
        (merged["R_P_ratio"] >= 0.0) &
        (merged["R_P_ratio"] < max_ratio)
    )

    return merged.loc[filt].copy()


def filter_period(df_long, hy_min=None, hy_max=None):
    d = df_long.copy()
    if hy_min is not None:
        d = d[d["HydrologicalYear"] >= hy_min]
    if hy_max is not None:
        d = d[d["HydrologicalYear"] <= hy_max]
    return d


# ============================================================
# BORE-LEVEL STATISTICS
# ============================================================

def summarise_rp_by_bore(df_long):
    """
    Compute bore-level statistics for mapping.
    """
    required = ["ID", "x", "y", "R_P_ratio"]
    missing = [c for c in required if c not in df_long.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stats_df = (
        df_long.groupby(["ID", "x", "y"], as_index=False)["R_P_ratio"]
        .agg(
            mean_RP="mean",
            median_RP="median",
            std_RP="std",
            n_years="count",
        )
    )

    stats_df["cv_RP"] = stats_df["std_RP"] / stats_df["mean_RP"]
    return stats_df


def stats_to_gdf(stats_df, source_crs="EPSG:28350", target_crs="EPSG:4326"):
    gdf = gpd.GeoDataFrame(
        stats_df,
        geometry=gpd.points_from_xy(stats_df["x"], stats_df["y"]),
        crs=source_crs,
    )
    return gdf.to_crs(target_crs)


# ============================================================
# DEPTH POLYGONS
# ============================================================

def load_depth_polygons(shp_path, value_col="Max_DTGW_R"):
    depth_gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")

    def extract_lower_bound(value):
        if value == "15 - Max":
            return 15.0
        if isinstance(value, str) and " - " in value:
            try:
                return float(value.split(" - ")[0])
            except Exception:
                return np.nan
        return np.nan

    def categorize_wt(value):
        if pd.isna(value):
            return "unknown"
        if value < 3:
            return "shallow (<3)"
        elif 3 <= value <= 14.9:
            return "medium (3-15)"
        else:
            return "deep (>15)"

    depth_gdf["WT_numeric"] = depth_gdf[value_col].apply(extract_lower_bound)
    depth_gdf["WT_category"] = depth_gdf["WT_numeric"].apply(categorize_wt)

    ordered_cats = ["shallow (<3)", "medium (3-15)", "deep (>15)"]
    category_colors = {
        "shallow (<3)": "#5E2514",
        "medium (3-15)": "#1f78b4",
        "deep (>15)": "#b2df8a",
    }

    depth_gdf["WT_index"] = depth_gdf["WT_category"].map(
        {cat: i for i, cat in enumerate(ordered_cats)}
    )

    depth_gdf_sorted = depth_gdf.sort_values("WT_numeric", ascending=False)

    return depth_gdf_sorted, ordered_cats, category_colors

# ============================================================
# DIAGNOSTICS INGEST
# ============================================================

def infer_bore_id_from_diag_filename(path):
    name = Path(path).name
    m = re.match(r"(.+?)_diagnostics", name)
    if m:
        return m.group(1)
    return Path(path).stem


def load_one_diagnostics(path_diag):
    path_diag = Path(path_diag)

    if not path_diag.exists():
        return None

    if path_diag.read_text(errors="ignore").strip() == "":
        print(f"Skipping blank diagnostics file: {path_diag.name}")
        return None

    try:
        df = pd.read_csv(path_diag)
    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable/empty diagnostics file: {path_diag.name}")
        return None

    if df.empty:
        print(f"Skipping dataframe with no rows: {path_diag.name}")
        return None

    bore_id = infer_bore_id_from_diag_filename(path_diag)
    df["ID"] = str(bore_id)

    return df


def load_all_diagnostics(txt_dir=P_TXT, pattern="*_diagnostics*.csv"):
    """
    Load all bore diagnostics CSVs from Results/txt.
    """
    txt_dir = Path(txt_dir)
    files = sorted(txt_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No diagnostics files found in {txt_dir} with pattern {pattern}")

    frames = []
    skipped = 0

    for f in files:
        df = load_one_diagnostics(f)
        if df is None:
            skipped += 1
            continue
        frames.append(df)

    if not frames:
        raise ValueError(f"No valid diagnostics files could be loaded from {txt_dir}")

    out = pd.concat(frames, ignore_index=True)
    out["ID"] = out["ID"].astype(str).str.strip()

    print(f"Loaded {len(frames)} diagnostics files; skipped {skipped} empty/bad files.")
    return out



# ============================================================
# MAPPING
# ============================================================

def plot_rp_maps_3panel(
    stats_gdf,
    depth_gdf=None,
    ordered_cats=None,
    category_colors=None,
    extent=None,
    outpath=None,
    hy_min=None,
    hy_max=None,
):
    if not HAS_CARTOPY:
        raise ImportError("cartopy is required for plot_rp_maps_3panel()")

    if hy_min is not None and hy_max is not None:
        period_label = f"{hy_min}–{hy_max}"
    elif hy_min is not None:
        period_label = f"{hy_min}+"
    elif hy_max is not None:
        period_label = f"≤{hy_max}"
    else:
        period_label = ""

    cmap_depth = None
    if depth_gdf is not None and ordered_cats and category_colors:
        cmap_depth = mcolors.ListedColormap([category_colors[c] for c in ordered_cats])

    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 11.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    plt.subplots_adjust(
        left=0.06,
        right=0.97,
        top=0.95,
        bottom=0.08,
        wspace=0.025,
    )

    def add_panel_label(ax, label):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            zorder=20,
        )

    def draw_panel(
        ax,
        column,
        title,
        cmap,
        vmin=None,
        vmax=None,
        add_depth_legend=False,
        panel_label="",
        show_left_labels=True,
        cbar_title=None,
    ):
        if cbar_title is None:
            cbar_title = title

        if extent is not None:
            ax.set_extent(extent)

        ax.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=5)

        if depth_gdf is not None and cmap_depth is not None:
            depth_gdf.plot(
                ax=ax,
                column="WT_index",
                cmap=cmap_depth,
                categorical=True,
                edgecolor="none",
                alpha=0.25,
                legend=False,
                zorder=1,
            )

        x = stats_gdf.geometry.x
        y = stats_gdf.geometry.y
        values = stats_gdf[column]

        # Layered scatter markers
        ax.scatter(
            x,
            y,
            s=38,
            facecolors="none",
            edgecolors="0.10",
            linewidths=0.7,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )

        ax.scatter(
            x,
            y,
            s=30,
            facecolors="white",
            edgecolors="none",
            alpha=0.95,
            transform=ccrs.PlateCarree(),
            zorder=11,
        )

        pts = ax.scatter(
            x,
            y,
            c=values,
            cmap=cmap,
            s=24,
            edgecolors="none",
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=12,
        )

        # Overlay problematic wells
        if "problem_well" in stats_gdf.columns:
            bad = stats_gdf[stats_gdf["problem_well"].fillna(False)]
            if len(bad) > 0:
                ax.scatter(
                    bad.geometry.x,
                    bad.geometry.y,
                    s=70,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                    transform=ccrs.PlateCarree(),
                    zorder=13,
                )

        # Taller framed colorbar
        ax_pos = ax.get_position()

        cbar_left = ax_pos.x0 + 0.025
        cbar_width_local = 0.011
        cbar_height_local = 0.78 * ax_pos.height
        cbar_bottom_local = ax_pos.y0 + 0.065 * ax_pos.height

        cbar_ax = fig.add_axes(
            [
                cbar_left,
                cbar_bottom_local,
                cbar_width_local,
                cbar_height_local,
            ],
            zorder=20,
        )

        cb = fig.colorbar(pts, cax=cbar_ax)
        cb.ax.tick_params(labelsize=10)

        cb.outline.set_visible(True)
        cb.outline.set_linewidth(0.8)
        cb.outline.set_edgecolor("0.2")

        cbar_ax.set_facecolor("white")
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.2")
            spine.set_linewidth(0.8)

        cbar_ax.set_title("", pad=2)

        # Depth legend
        if add_depth_legend and ordered_cats and category_colors:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=category_colors[c],
                    label=c,
                    markersize=10,
                )
                for c in ordered_cats
            ]

            if "problem_well" in stats_gdf.columns:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="red",
                        markerfacecolor="none",
                        label="Problem well (poor fit)",
                        markersize=9,
                        linewidth=0,
                        markeredgewidth=1.2,
                    )
                )

            leg = ax.legend(
                handles=legend_handles,
                title="Depth to WT",
                loc="lower left",
                frameon=True,
                fontsize=9,
                facecolor="white",
                edgecolor="0.2",
                framealpha=1.0,
                borderpad=0.4,
                labelspacing=0.35,
                handletextpad=0.6,
                fancybox=False,
            )
            leg.get_title().set_fontsize(10)

        # Gridlines / labels
        gl = ax.gridlines(
            draw_labels=True,
            linestyle="-.",
            linewidth=0.15,
            color="0.6",
            zorder=0,
        )

        gl.xlocator = mticker.FixedLocator([115.5, 115.75, 116.0])
        gl.ylocator = mticker.FixedLocator(
            [-31.4, -31.6, -31.8, -32.0, -32.2, -32.4, -32.6]
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = show_left_labels

        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}
        gl.xpadding = 6
        gl.ypadding = 6

        ax.set_title(title, fontsize=13)
        add_panel_label(ax, panel_label)

    draw_panel(
        axes[0],
        "median_RP",
        f"Median R/P ({period_label})" if period_label else "Median R/P",
        cmap="Blues",
        vmin=0.0,
        vmax=0.8,
        add_depth_legend=True,
        panel_label="A",
        show_left_labels=True,
        cbar_title=f"Median R/P\n({period_label})" if period_label else "Median R/P",
    )

    draw_panel(
        axes[1],
        "cv_RP",
        f"CoV R/P ({period_label})" if period_label else "CoV R/P",
        cmap="magma_r",
        vmin=0.0,
        vmax=0.30,
        add_depth_legend=False,
        panel_label="B",
        show_left_labels=False,
        cbar_title=f"CoV R/P\n({period_label})" if period_label else "CoV R/P",
    )

    draw_panel(
        axes[2],
        "n_years",
        f"Valid recharge years ({period_label})" if period_label else "Valid recharge years",
        cmap="cividis",
        vmin=3,
        vmax=max(5, stats_gdf["n_years"].max()),
        add_depth_legend=False,
        panel_label="C",
        show_left_labels=False,
        cbar_title="Valid\nyears",
    )

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, axes


# ============================================================
# HIGH-LEVEL CONVENIENCE
# ============================================================

def prepare_rp_stats_gdf(
    coords_csv,
    rain_csv,
    txt_dir=P_TXT,
    hy_min=None,
    hy_max=None,
    keep_qc=("OK",),
    source_crs="EPSG:28350",
    include_poor_fit=True,
    poor_fit_min_frac=0.5,
):
    """
    One-stop helper to build bore-level R/P mapping stats from summary CSVs.
    Optionally merges poor-fit diagnostics by bore.
    """


    
    coords_df = load_bore_coordinates(coords_csv)
    rain_df = load_daily_rain_csv(rain_csv)
    hy_rain_df = hydro_year_rainfall_from_daily(rain_df, start_month=4)

    summaries_df = load_all_summaries(txt_dir=txt_dir)
    df_long = build_rp_long_table(
        summaries_df,
        coords_df,
        hy_rain_df,
        keep_qc=keep_qc,
    )
    df_long = filter_period(df_long, hy_min=hy_min, hy_max=hy_max)

    stats_df = summarise_rp_by_bore(df_long)

    if include_poor_fit:
        diag_df = load_all_diagnostics(txt_dir=txt_dir)

        poor_fit_df = summarise_poor_fit_by_bore(
            diag_df,
            hy_min=hy_min,
            hy_max=hy_max,
        )

        stats_df = merge_problem_wells_into_stats(stats_df, poor_fit_df)

    stats_gdf = stats_to_gdf(stats_df, source_crs=source_crs, target_crs="EPSG:4326")

    return df_long, stats_df, stats_gdf


