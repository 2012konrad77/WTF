#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for the shareable WTF recharge workflow, version 4.

This release avoids hard-coded user-specific paths. Input data are expected
under WTF_DATA_ROOT, or under ./data if WTF_DATA_ROOT is not set.
"""

import datetime
from pathlib import Path
import os


# ------------------------------------------------------------
# CONFIGURATION: WTF recharge method, version 4 final
# ------------------------------------------------------------

# Hydrological year
START_MONTH = 4  # April-start hydrological year

# Trough selection and rise confirmation
RISE_TOL_M = 0.02
RISE_CONFIRM_DAYS = 120

# Recharge event timing
MAX_DT_DAYS = 280

# Recession fit window
MIN_POINTS = 3
FALLBACK_BACKFIT_DAYS = 240
MAX_FIT_DURATION_DAYS = 260
T_SHIFT_POWER = 1.0

# QC thresholds
MIN_FIT_POINTS_SOFT = 4
MIN_PEAK_POINTS = 2
MAX_GAP_DAYS_SOFT = 160

# Seasonal trough window for south-west Western Australia
TROUGH_MONTH_START = 4
TROUGH_MONTH_END = 7

# Model acceptance thresholds
MIN_R2_LIN = 0.30
MIN_R2_EXP = 0.30
MIN_R2_POW = 0.30
MIN_R2_BEST = 0.30

# Specific yield distribution
SY_MEAN = 0.1858
SY_STD = 0.0392
SY_LOWER = 0.01
SY_UPPER = 0.5

SAVE_DIAGNOSTICS = True
SAVE_DIAG_FIGS = True


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


_data_root = os.environ.get("WTF_DATA_ROOT")

if not _data_root:
    raise RuntimeError(
        "WTF_DATA_ROOT is not set. Set it to the WTF data directory, e.g.:\n"
        'export WTF_DATA_ROOT="$HOME/Documents/Recharge/wtf_input_data"'
    )

DATA_ROOT = Path(_data_root).expanduser()

RAW_RAIN_DIR = DATA_ROOT / "0_RainfallData"
GEO_DIR = DATA_ROOT / "1_Geo"
WT_DIR = DATA_ROOT / "2_WaterTableData"
WT_TEST_DIR = WT_DIR
WT_BY_BORE_DIR = WT_DIR / "by_bore"

RESULTS_DIR = PROJECT_ROOT / "Results"
P_TXT = RESULTS_DIR / "txt"
P_FIG = RESULTS_DIR / "figs"
P_TEX = RESULTS_DIR / "tex"

for p in [P_TXT, P_FIG, P_TEX]:
    p.mkdir(parents=True, exist_ok=True)


