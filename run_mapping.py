#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import matplotlib.pyplot as plt

from wtf_code.config import GEO_DIR, PROJECT_ROOT, P_FIG, P_TXT
from wtf_code.mapping import (
    prepare_rp_stats_gdf,
    load_depth_polygons,
    plot_rp_maps_3panel,
)


# ============================================================
# SETTINGS
# ============================================================
coords_csv = GEO_DIR / "BoreCoordinates_from_assets.csv"
rain_csv = GEO_DIR / "rain_perth_airport_silo.csv"
depth_shp = Path(os.environ.get(
    "WTF_DEPTH_SHP",
    PROJECT_ROOT / "data_external" / "SCP_WAVE_MaxDTGW.shp",
))

hy_min = 2011
hy_max = 2020

# Strict mapping by default
keep_qc = ("OK",)

# Optional final filter for map robustness
min_years_for_map = 3

extent = [115.35, 116.05, -32.70, -31.30]

outpath = P_FIG / f"rp_map_3panel_{hy_min}_{hy_max}.png"


# ============================================================
# MAIN
# ============================================================
def main():
    print()
    print("====================================")
    print("Preparing R/P mapping dataset")
    print("Summary folder:", P_TXT)
    print("Period:", f"{hy_min}–{hy_max}")
    print("QC filter:", keep_qc)
    print("====================================")
    print()

    df_long, stats_df, stats_gdf = prepare_rp_stats_gdf(
        coords_csv=coords_csv,
        rain_csv=rain_csv,
        txt_dir=P_TXT,
        hy_min=hy_min,
        hy_max=hy_max,
        keep_qc=keep_qc,
        source_crs="EPSG:28350",
        include_poor_fit=True,
        poor_fit_min_frac=0.5,
    )

    print(f"Long table rows: {len(df_long)}")
    print(f"Bores before n_years filter: {len(stats_df)}")

    if min_years_for_map is not None:
        stats_gdf = stats_gdf[stats_gdf["n_years"] >= min_years_for_map].copy()

    print(f"Bores after n_years >= {min_years_for_map}: {len(stats_gdf)}")
    print("HY range in df_long:", df_long["HydrologicalYear"].min(), df_long["HydrologicalYear"].max())
    print("Unique HYs:", sorted(df_long["HydrologicalYear"].unique()))

    depth_gdf, ordered_cats, category_colors = load_depth_polygons(depth_shp)

    plot_rp_maps_3panel(
        stats_gdf,
        depth_gdf=depth_gdf,
        ordered_cats=ordered_cats,
        category_colors=category_colors,
        extent=extent,
        outpath=outpath,
        hy_min=hy_min,
        hy_max=hy_max,
    )

    plt.show()

    # Also save tabular outputs for later use.
    long_csv = P_FIG / f"rp_long_{hy_min}_{hy_max}.csv"
    stats_csv = P_FIG / f"rp_stats_{hy_min}_{hy_max}.csv"

    df_long.to_csv(long_csv, index=False)
    stats_gdf.drop(columns="geometry").to_csv(stats_csv, index=False)

    print()
    print("====================================")
    print("Mapping finished")
    print("Figure:", outpath)
    print("Long table:", long_csv)
    print("Stats table:", stats_csv)
    print("====================================")


if __name__ == "__main__":
    main()
