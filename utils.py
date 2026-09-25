import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.stats import truncnorm

from wtf_code.config import (
    START_MONTH,
    SY_MEAN, SY_STD, SY_LOWER, SY_UPPER,
    RAW_RAIN_DIR,
)


def hydro_year_from_date(dates, start_month=START_MONTH):
    dates = pd.to_datetime(dates)
    if hasattr(dates, "dt"):
        months = dates.dt.month
        years = dates.dt.year
    else:
        months = dates.month
        years = dates.year
    return np.where(months >= start_month, years + 1, years)


def hy_start(hy):
    return pd.Timestamp(hy - 1, START_MONTH, 1)


def hy_end(hy):
    return pd.Timestamp(hy, START_MONTH, 1) - pd.Timedelta(days=1)


def r2_score_safe(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2:
        return np.nan
    y = y[m]
    yhat = yhat[m]
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def recharge_uncertainty(delta_h, n=40000):
    if not np.isfinite(delta_h):
        return np.nan, np.nan, np.nan

    a = (SY_LOWER - SY_MEAN) / SY_STD
    b = (SY_UPPER - SY_MEAN) / SY_STD
    dist = truncnorm(a, b, loc=SY_MEAN, scale=SY_STD)

    sy = dist.rvs(n)
    R = 1000.0 * sy * delta_h
    return np.percentile(R, 5), np.percentile(R, 50), np.percentile(R, 95)


def derive_best_model_and_r2(r):
    raw_r2 = {
        "LIN": r.get("r2_lin", np.nan),
        "EXP": r.get("r2_exp", np.nan),
        "POW": r.get("r2_pow", np.nan),
    }
    raw_dh = {
        "LIN": r.get("delta_h_lin", np.nan),
        "EXP": r.get("delta_h_exp", np.nan),
        "POW": r.get("delta_h_pow", np.nan),
    }

    valid = {k: raw_r2[k] for k in raw_r2 if np.isfinite(raw_r2[k]) and np.isfinite(raw_dh[k])}

    if valid:
        best_model = max(valid, key=valid.get)
        r2_best = valid[best_model]
    else:
        any_fit_attempt = any(np.isfinite(v) for v in raw_r2.values())
        best_model = "MIN_REJECTED" if any_fit_attempt else "MIN_NOFIT"
        r2_best = np.nan

    return best_model, r2_best


def load_bore_lonlat_from_csv(bore_id, path_csv):
    dfc = pd.read_csv(path_csv, sep=None, engine="python")
    dfc.columns = [c.strip() for c in dfc.columns]
    dfc["ID"] = dfc["ID"].astype(str).str.strip()
    bore_id = str(bore_id).strip()

    row = dfc.loc[dfc["ID"] == bore_id]
    if row.empty:
        raise ValueError(f"Bore ID {bore_id} not found in {path_csv}")

    x = float(row.iloc[0]["x"])
    y = float(row.iloc[0]["y"])

    transformer = Transformer.from_crs("EPSG:28350", "EPSG:4283", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def load_rain_dataset(rain_dir=RAW_RAIN_DIR):
    nc_files = sorted(rain_dir.glob("*.daily_rain_ROI.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {rain_dir}")
    return xr.open_mfdataset(
        nc_files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )


def extract_monthly_rain_at_lonlat(ds, lon, lat, var="daily_rain"):
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")

    ts = ds[var].sel(lon=lon, lat=lat, method="nearest").to_series()
    ts.index = pd.to_datetime(ts.index)
    rain_monthly = ts.resample("MS").sum()
    rain_monthly.name = "Rain_mm"
    return rain_monthly


def rainfall_to_hydro_year(rain_monthly):
    if isinstance(rain_monthly, pd.Series):
        rm = rain_monthly.to_frame(name="Rain_mm").copy()
    else:
        rm = rain_monthly.copy()
        if "Rain_mm" not in rm.columns:
            numeric_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric rainfall column found")
            rm = rm.rename(columns={numeric_cols[0]: "Rain_mm"})

    if not isinstance(rm.index, pd.RangeIndex):
        rm = rm.reset_index()

    date_col = None
    for c in rm.columns:
        if pd.api.types.is_datetime64_any_dtype(rm[c]):
            date_col = c
            break

    if date_col is None:
        for c in ["time", "Time", "Date", "date", "index", "Month", "month"]:
            if c in rm.columns:
                rm[c] = pd.to_datetime(rm[c], errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(rm[c]):
                    date_col = c
                    break

    if date_col is None:
        raise ValueError(f"Could not identify date column in rainfall data. Columns are: {list(rm.columns)}")

    rm["HydrologicalYear"] = hydro_year_from_date(rm[date_col])

    return (
        rm.groupby("HydrologicalYear", as_index=False)["Rain_mm"]
        .sum()
        .sort_values("HydrologicalYear")
    )