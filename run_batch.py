#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time

from wtf_code.config import WT_TEST_DIR, WT_BY_BORE_DIR
from wtf_code.utils import load_rain_dataset
from wtf_code.pipeline import run_one_bore

# ============================================================
# SETTINGS
# ============================================================
USE_TEST_DATA = False   # True = 3 test files in 2_WaterTableData/
                       # False = full set in 2_WaterTableData/by_bore/


def main():
    t0 = time.time()

    wt_source = WT_TEST_DIR if USE_TEST_DATA else WT_BY_BORE_DIR
    wt_files = sorted(wt_source.glob("*_WT.csv"))

    print()
    print("====================================")
    print("WTF batch processing")
    print("Source:", wt_source)
    print("Number of bore files:", len(wt_files))
    print("====================================")
    print()

    ds_rain = load_rain_dataset()

    success = 0
    failed = 0

    for i, wt_file in enumerate(wt_files, start=1):
        bore_id = wt_file.stem.replace("_WT", "")
        pct = 100 * i / len(wt_files)

        print(f"[{i}/{len(wt_files)} | {pct:5.1f}%] Processing bore {bore_id}")

        try:
            run_one_bore(bore_id, ds_rain=ds_rain, use_test_dir=USE_TEST_DATA)
            success += 1

        except Exception as e:
            print(f"  ERROR: {bore_id}")
            print(f"  {type(e).__name__}: {e}")
            failed += 1

    elapsed = time.time() - t0

    print()
    print("====================================")
    print("Batch finished")
    print("Successful:", success)
    print("Failed:", failed)
    print(f"Elapsed time: {elapsed/60:.1f} minutes")
    print("====================================")


if __name__ == "__main__":
    main()