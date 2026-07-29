from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import calc_zhd_era5  # noqa: E402
import calc_zhd_sounding  # noqa: E402
import diagnose_step1_bottom_boundary as step1  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(r"E:\ZHDplus\test\diagnostics")
ZHD_TOL_MM = 1e-6
TOP_IDENTITY_TOL_MM = 1e-10
WEIGHT_SUM_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step-2 top-boundary diagnostic for Profile-minus-Saas ZHD."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_DIR / "typhoon_events.json",
        help="Path to typhoon event JSON.",
    )
    parser.add_argument("--event", required=True, help="Single event id, e.g. hanna.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=r"Output root. Default: E:\ZHDplus\test\diagnostics",
    )
    parser.add_argument(
        "--include-outside",
        action="store_true",
        help="Keep outside records in plots. CSV always preserves all records.",
    )
    return parser.parse_args()


def duplicate_count(df: pd.DataFrame, cols: list[str]) -> int:
    existing = [c for c in cols if c in df.columns]
    return int(df.duplicated(existing).sum()) if existing and not df.empty else 0


def nan_inf_counts(df: pd.DataFrame, cols: list[str]) -> tuple[int, int]:
    existing = [c for c in cols if c in df.columns]
    if not existing or df.empty:
        return 0, 0
    numeric = df[existing].apply(pd.to_numeric, errors="coerce")
    return (
        int(numeric.isna().any(axis=1).sum()),
        int(np.isinf(numeric.to_numpy(dtype=float)).any(axis=1).sum()),
    )


def read_official(path: Path, key_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=key_cols + ["official_ZHD_mm"])
    df = pd.read_csv(path, parse_dates=["time_utc"])
    df["official_ZHD_mm"] = pd.to_numeric(df["ZHD_m"], errors="coerce") * 1000.0
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[key_cols + ["official_ZHD_mm"]]


def official_value(
    df: pd.DataFrame,
    time_utc: pd.Timestamp,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> float:
    if df.empty:
        return float("nan")
    mask = df["time_utc"] == time_utc
    if latitude is not None and "latitude" in df.columns:
        mask &= np.isclose(df["latitude"], latitude, rtol=0.0, atol=1e-9)
    if longitude is not None and "longitude" in df.columns:
        mask &= np.isclose(df["longitude"], longitude, rtol=0.0, atol=1e-9)
    m = df[mask]
    return float(m["official_ZHD_mm"].iloc[0]) if not m.empty else float("nan")


def finish_status(
    status: str,
    official_profile: float,
    official_saas: float,
    profile_mm: float,
    saas_mm: float,
) -> tuple[str, float, float]:
    if status != "ok":
        return status, float("nan"), float("nan")
    if pd.isna(official_profile) or pd.isna(official_saas):
        return "missing_official_output", float("nan"), float("nan")
    profile_err = profile_mm - official_profile
    saas_err = saas_mm - official_saas
    if abs(profile_err) > ZHD_TOL_MM or abs(saas_err) > ZHD_TOL_MM:
        return "official_output_mismatch", profile_err, saas_err
    return "ok", profile_err, saas_err


def top_metrics(
    bottom_pressure: float,
    bottom_height: float,
    top_pressure: float,
    top_height: float,
    layer_integral_mm: float,
    top_correction_mm: float,
    profile_mm: float,
    saas_mm: float,
) -> dict[str, float]:
    profile_minus_saas = profile_mm - saas_mm
    frac = top_correction_mm / profile_mm if profile_mm != 0 else np.nan
    return {
        "bottom_pressure_hPa": bottom_pressure,
        "bottom_geometric_height_m": bottom_height,
        "covered_pressure_range_hPa": bottom_pressure - top_pressure,
        "covered_geometric_height_range_m": top_height - bottom_height,
        "pressure_ratio_bottom_to_top": bottom_pressure / top_pressure
        if top_pressure > 0
        else np.nan,
        "log_pressure_range": np.log(bottom_pressure / top_pressure)
        if top_pressure > 0
        else np.nan,
        "profile_layer_integral_mm": layer_integral_mm,
        "profile_top_correction_mm": top_correction_mm,
        "profile_ZHD_mm": profile_mm,
        "surface_saas_ZHD_mm": saas_mm,
        "profile_minus_saas_mm": profile_minus_saas,
        "top_correction_fraction_of_profile": frac,
        "top_correction_fraction_percent": frac * 100.0,
    }


def sounding_effective(initial: pd.DataFrame) -> pd.DataFrame:
    return step1.sounding_effective_profile(initial).reset_index(drop=True)


def diagnose_sounding(event: dict[str, Any], paths: step1.EventPaths, stages: list[step1.StageDef]) -> pd.DataFrame:
    if not paths.sounding_input.exists():
        return pd.DataFrame(
            [
                {
                    "event_id": event["id"],
                    "display_name": event.get("display_name", event["id"]),
                    "station_id": str(event["station"]["id"]),
                    "stage": s.code,
                    "sounding_time_utc": pd.NaT,
                    "data_status": "missing_file",
                    "warning_message": str(paths.sounding_input),
                }
                for s in stages
            ]
        )
    raw = calc_zhd_sounding.read_sounding_csv(paths.sounding_input)
    raw["_original_index"] = np.arange(len(raw), dtype=int)
    off_saas = read_official(paths.sounding_saas, ["time_utc"])
    off_profile = read_official(paths.sounding_profile, ["time_utc"])
    rows: list[dict[str, Any]] = []
    for time_utc, group in raw.groupby(calc_zhd_sounding.TIME_COL, sort=True):
        time_utc = pd.Timestamp(time_utc)
        nan_count, inf_count = nan_inf_counts(
            group,
            [
                calc_zhd_sounding.P_COL,
                calc_zhd_sounding.H_COL,
                calc_zhd_sounding.TC_COL,
                calc_zhd_sounding.RH_PERCENT_COL,
            ],
        )
        row: dict[str, Any] = {
            "event_id": event["id"],
            "display_name": event.get("display_name", event["id"]),
            "station_id": str(event["station"]["id"]),
            "stage": step1.assign_stage(time_utc, stages),
            "sounding_time_utc": time_utc,
            "raw_level_count": int(len(group)),
            "duplicate_pressure_count": duplicate_count(group, [calc_zhd_sounding.P_COL]),
            "duplicate_height_count": duplicate_count(group, [calc_zhd_sounding.H_COL]),
            "nan_required_row_count": nan_count,
            "inf_required_row_count": inf_count,
            "warning_message": "",
        }
        status = "ok"
        if group.empty:
            status = "no_raw_rows"
        raw_top = group.loc[group[calc_zhd_sounding.P_COL].idxmin()] if status == "ok" else None
        initial = step1.sounding_initial_valid(group)
        row["initial_valid_level_count"] = int(len(initial))
        if status == "ok" and initial.empty:
            status = "no_initial_valid_layers"
        initial_top = initial.loc[initial[calc_zhd_sounding.P_COL].idxmin()] if status == "ok" else None
        effective = sounding_effective(initial) if status == "ok" else pd.DataFrame()
        row["profile_effective_level_count"] = int(len(effective))
        row["integrated_layer_count"] = max(int(len(effective)) - 1, 0)
        if status == "ok" and effective.empty:
            status = "no_effective_profile_layers"
        elif status == "ok" and len(effective) < 2:
            status = "insufficient_profile_layers"

        if status == "ok":
            bottom = initial.loc[initial[calc_zhd_sounding.P_COL].idxmax()]
            top = effective.loc[effective[calc_zhd_sounding.H_COL].idxmax()]
            lat0 = float(bottom[calc_zhd_sounding.LAT_COL])
            p0 = float(bottom[calc_zhd_sounding.P_COL])
            h0 = float(calc_zhd_sounding.geometric_height(float(bottom[calc_zhd_sounding.H_COL]), lat0))
            top_h = float(calc_zhd_sounding.geometric_height(float(top[calc_zhd_sounding.H_COL]), lat0))
            raw_top_h = float(calc_zhd_sounding.geometric_height(float(raw_top[calc_zhd_sounding.H_COL]), float(raw_top[calc_zhd_sounding.LAT_COL])))
            init_top_h = float(calc_zhd_sounding.geometric_height(float(initial_top[calc_zhd_sounding.H_COL]), float(initial_top[calc_zhd_sounding.LAT_COL])))
            layer_int, top_corr, profile_mm = step1.sounding_profile_components(effective, lat0)
            saas_mm_value = step1.saas_mm(p0, h0, lat0)
            top_removed = (
                int(raw_top["_original_index"]) != int(top["_original_index"])
                or abs(float(raw_top[calc_zhd_sounding.P_COL]) - float(top[calc_zhd_sounding.P_COL])) > 1e-12
            )
            reason = ""
            if top_removed:
                if int(raw_top["_original_index"]) != int(initial_top["_original_index"]):
                    reason = "missing_required_column"
                else:
                    reason = "Nh_or_effective_filter"
            official_profile = official_value(off_profile, time_utc)
            official_saas = official_value(off_saas, time_utc)
            status, profile_err, saas_err = finish_status(
                status, official_profile, official_saas, profile_mm, saas_mm_value
            )
            row.update(
                {
                    "raw_top_original_index": int(raw_top["_original_index"]),
                    "raw_top_pressure_hPa": float(raw_top[calc_zhd_sounding.P_COL]),
                    "raw_top_geopotential_height_m": float(raw_top[calc_zhd_sounding.H_COL]),
                    "raw_top_geometric_height_m": raw_top_h,
                    "raw_top_temperature_C": float(raw_top[calc_zhd_sounding.TC_COL]),
                    "raw_top_relative_humidity_percent": float(raw_top[calc_zhd_sounding.RH_PERCENT_COL]),
                    "initial_valid_top_original_index": int(initial_top["_original_index"]),
                    "initial_valid_top_pressure_hPa": float(initial_top[calc_zhd_sounding.P_COL]),
                    "initial_valid_top_geopotential_height_m": float(initial_top[calc_zhd_sounding.H_COL]),
                    "initial_valid_top_geometric_height_m": init_top_h,
                    "effective_top_original_index": int(top["_original_index"]),
                    "effective_top_pressure_hPa": float(top[calc_zhd_sounding.P_COL]),
                    "effective_top_geopotential_height_m": float(top[calc_zhd_sounding.H_COL]),
                    "effective_top_geometric_height_m": top_h,
                    "effective_top_latitude_deg": float(top[calc_zhd_sounding.LAT_COL]),
                    "effective_top_longitude_deg": float(top[calc_zhd_sounding.LON_COL]),
                    "effective_top_temperature_K": float(top["temperature_K"]),
                    "effective_top_vapor_pressure_hPa": float(top["e"]),
                    "effective_top_Nh": float(top["Nh"]),
                    "raw_to_effective_top_pressure_change_hPa": float(top[calc_zhd_sounding.P_COL]) - float(raw_top[calc_zhd_sounding.P_COL]),
                    "raw_to_effective_top_geometric_height_change_m": top_h - raw_top_h,
                    "top_level_removed_flag": bool(top_removed),
                    "top_level_removed_reason": reason,
                    "official_profile_ZHD_mm": official_profile,
                    "official_saas_ZHD_mm": official_saas,
                    "profile_recalculation_error_mm": profile_err,
                    "saas_recalculation_error_mm": saas_err,
                }
            )
            row.update(
                top_metrics(
                    p0,
                    h0,
                    float(top[calc_zhd_sounding.P_COL]),
                    top_h,
                    layer_int,
                    top_corr,
                    profile_mm,
                    saas_mm_value,
                )
            )
        row["data_status"] = status
        rows.append(row)
    return pd.DataFrame(rows)


def era5_effective(initial: pd.DataFrame, latitude: float) -> tuple[pd.DataFrame, float, float, float]:
    return step1.era5_profile_components(initial, latitude)


def diagnose_era5_grid(
    event: dict[str, Any],
    paths: step1.EventPaths,
    stages: list[step1.StageDef],
) -> tuple[pd.DataFrame, str]:
    station_lat = float(event["station"]["latitude"])
    station_lon = float(event["station"]["longitude"])
    calc_zhd_era5.STATION_LAT = station_lat
    calc_zhd_era5.STATION_LON = station_lon
    if not paths.era5_input.exists():
        return pd.DataFrame(), "missing"
    raw = calc_zhd_era5.read_era5_csv(paths.era5_input)
    raw["_original_index"] = np.arange(len(raw), dtype=int)
    lon_w, lon_e, lat_s, lat_n = calc_zhd_era5.find_surrounding_grid(raw)
    interp_mode, weights = step1.grid_weights(lon_w, lon_e, lat_s, lat_n, station_lon, station_lat)
    grid4 = calc_zhd_era5.select_grid4(raw, lon_w, lon_e, lat_s, lat_n)
    off_saas = read_official(paths.era5_saas_grid4, ["time_utc", "latitude", "longitude"])
    off_profile = read_official(paths.era5_profile_grid4, ["time_utc", "latitude", "longitude"])
    rows: list[dict[str, Any]] = []
    for (time_utc, lat, lon), group in grid4.groupby([calc_zhd_era5.TIME_COL, calc_zhd_era5.LAT_COL, calc_zhd_era5.LON_COL], sort=True):
        time_utc = pd.Timestamp(time_utc)
        lat = float(lat)
        lon = float(lon)
        pos, weight = weights.get((lat, lon), ("unused", 0.0))
        nan_count, inf_count = nan_inf_counts(
            group,
            [calc_zhd_era5.P_COL, calc_zhd_era5.H_COL, calc_zhd_era5.T_COL, calc_zhd_era5.RH_PERCENT_COL],
        )
        row: dict[str, Any] = {
            "event_id": event["id"],
            "display_name": event.get("display_name", event["id"]),
            "station_id": str(event["station"]["id"]),
            "stage": step1.assign_stage(time_utc, stages),
            "time_utc": time_utc,
            "grid_position": pos,
            "grid_latitude": lat,
            "grid_longitude": lon,
            "interpolation_weight": weight,
            "raw_level_count": int(len(group)),
            "duplicate_pressure_count": duplicate_count(group, [calc_zhd_era5.P_COL]),
            "duplicate_height_count": duplicate_count(group, [calc_zhd_era5.H_COL]),
            "nan_required_row_count": nan_count,
            "inf_required_row_count": inf_count,
            "warning_message": "",
        }
        status = "ok"
        if group.empty:
            status = "no_raw_rows"
        raw_top = group.loc[group[calc_zhd_era5.P_COL].idxmin()] if status == "ok" else None
        initial = step1.era5_initial_valid(group)
        row["initial_valid_level_count"] = int(len(initial))
        if status == "ok" and initial.empty:
            status = "no_initial_valid_layers"
        initial_top = initial.loc[initial[calc_zhd_era5.P_COL].idxmin()] if status == "ok" else None
        effective, layer_int, top_corr, profile_mm = era5_effective(initial, lat) if status == "ok" else (pd.DataFrame(), np.nan, np.nan, np.nan)
        row["profile_effective_level_count"] = int(len(effective))
        row["integrated_layer_count"] = max(int(len(effective)) - 1, 0)
        if status == "ok" and effective.empty:
            status = "no_effective_profile_layers"
        elif status == "ok" and len(effective) < 2:
            status = "insufficient_profile_layers"
        if status == "ok":
            bottom = initial.loc[initial[calc_zhd_era5.P_COL].idxmax()]
            top = effective.loc[effective[calc_zhd_era5.H_COL].idxmax()]
            p0 = float(bottom[calc_zhd_era5.P_COL])
            h0 = float(calc_zhd_era5.geometric_height(float(bottom[calc_zhd_era5.H_COL]), lat))
            saas_mm_value = step1.saas_mm(p0, h0, lat)
            raw_top_h = float(calc_zhd_era5.geometric_height(float(raw_top[calc_zhd_era5.H_COL]), lat))
            top_h = float(calc_zhd_era5.geometric_height(float(top[calc_zhd_era5.H_COL]), lat))
            top_removed = (
                int(raw_top["_original_index"]) != int(top["_original_index"])
                or abs(float(raw_top[calc_zhd_era5.P_COL]) - float(top[calc_zhd_era5.P_COL])) > 1e-12
            )
            reason = ""
            if top_removed:
                if int(raw_top["_original_index"]) != int(initial_top["_original_index"]):
                    reason = "missing_required_column"
                else:
                    reason = "Nh_or_effective_filter"
            official_profile = official_value(off_profile, time_utc, lat, lon)
            official_saas = official_value(off_saas, time_utc, lat, lon)
            status, profile_err, saas_err = finish_status(
                status, official_profile, official_saas, profile_mm, saas_mm_value
            )
            row.update(
                {
                    "raw_top_pressure_hPa": float(raw_top[calc_zhd_era5.P_COL]),
                    "raw_top_geopotential_height_m": float(raw_top[calc_zhd_era5.H_COL]),
                    "raw_top_geometric_height_m": raw_top_h,
                    "effective_top_pressure_hPa": float(top[calc_zhd_era5.P_COL]),
                    "effective_top_geopotential_height_m": float(top[calc_zhd_era5.H_COL]),
                    "effective_top_geometric_height_m": top_h,
                    "effective_top_temperature_K": float(top[calc_zhd_era5.T_COL]),
                    "effective_top_vapor_pressure_hPa": float(top["e"]),
                    "effective_top_Nh": float(top["Nh"]),
                    "raw_to_effective_top_pressure_change_hPa": float(top[calc_zhd_era5.P_COL]) - float(raw_top[calc_zhd_era5.P_COL]),
                    "raw_to_effective_top_geometric_height_change_m": top_h - raw_top_h,
                    "top_level_removed_flag": bool(top_removed),
                    "top_level_removed_reason": reason,
                    "official_profile_ZHD_mm": official_profile,
                    "official_saas_ZHD_mm": official_saas,
                    "profile_recalculation_error_mm": profile_err,
                    "saas_recalculation_error_mm": saas_err,
                }
            )
            row.update(
                top_metrics(
                    p0,
                    h0,
                    float(top[calc_zhd_era5.P_COL]),
                    top_h,
                    layer_int,
                    top_corr,
                    profile_mm,
                    saas_mm_value,
                )
            )
        row["data_status"] = status
        rows.append(row)
    return pd.DataFrame(rows), interp_mode


def diagnose_station(
    event: dict[str, Any],
    paths: step1.EventPaths,
    stages: list[step1.StageDef],
    grid: pd.DataFrame,
    interp_mode: str,
) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    off_saas = read_official(paths.era5_saas_station, ["time_utc"])
    off_profile = read_official(paths.era5_profile_station, ["time_utc"])
    rows: list[dict[str, Any]] = []
    ok_grid = grid[grid["data_status"] == "ok"].copy()
    for time_utc, group in ok_grid.groupby("time_utc", sort=True):
        time_utc = pd.Timestamp(time_utc)
        weight_sum = float(group["interpolation_weight"].sum())
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOL:
            rows.append({"event_id": event["id"], "time_utc": time_utc, "data_status": "interpolation_weight_error", "warning_message": f"weight_sum={weight_sum}"})
            continue

        def weighted(col: str) -> float:
            return float((group[col] * group["interpolation_weight"]).sum())

        layer_int = weighted("profile_layer_integral_mm")
        top_corr = weighted("profile_top_correction_mm")
        profile_mm = weighted("profile_ZHD_mm")
        if abs(profile_mm - (layer_int + top_corr)) > TOP_IDENTITY_TOL_MM:
            status = "invalid_top_correction"
        else:
            status = "ok"
        saas_mm_value = weighted("surface_saas_ZHD_mm")
        official_profile = official_value(off_profile, time_utc)
        official_saas = official_value(off_saas, time_utc)
        status, profile_err, saas_err = finish_status(
            status, official_profile, official_saas, profile_mm, saas_mm_value
        )
        row: dict[str, Any] = {
            "event_id": event["id"],
            "display_name": event.get("display_name", event["id"]),
            "station_id": str(event["station"]["id"]),
            "stage": step1.assign_stage(time_utc, stages),
            "time_utc": time_utc,
            "station_latitude": float(event["station"]["latitude"]),
            "station_longitude": float(event["station"]["longitude"]),
            "interpolation_mode": interp_mode,
            "weighted_effective_top_pressure_hPa": weighted("effective_top_pressure_hPa"),
            "weighted_effective_top_geopotential_height_m": weighted("effective_top_geopotential_height_m"),
            "weighted_effective_top_geometric_height_m": weighted("effective_top_geometric_height_m"),
            "weighted_covered_pressure_range_hPa": weighted("covered_pressure_range_hPa"),
            "weighted_covered_geometric_height_range_m": weighted("covered_geometric_height_range_m"),
            "weighted_integrated_layer_count": weighted("integrated_layer_count"),
            "profile_layer_integral_grid_then_interp_mm": layer_int,
            "profile_top_correction_grid_then_interp_mm": top_corr,
            "profile_ZHD_grid_then_interp_mm": profile_mm,
            "surface_saas_grid_then_interp_mm": saas_mm_value,
            "profile_minus_saas_mm": profile_mm - saas_mm_value,
            "top_correction_fraction_of_profile": top_corr / profile_mm if profile_mm else np.nan,
            "top_correction_fraction_percent": top_corr / profile_mm * 100.0 if profile_mm else np.nan,
            "official_profile_ZHD_mm": official_profile,
            "official_saas_ZHD_mm": official_saas,
            "profile_recalculation_error_mm": profile_err,
            "saas_recalculation_error_mm": saas_err,
            "data_status": status,
            "warning_message": "",
        }
        for _, g in group.iterrows():
            pos = str(g["grid_position"])
            row[f"{pos}_weight"] = g["interpolation_weight"]
            row[f"{pos}_top_pressure_hPa"] = g["effective_top_pressure_hPa"]
            row[f"{pos}_top_geometric_height_m"] = g["effective_top_geometric_height_m"]
            row[f"{pos}_top_correction_mm"] = g["profile_top_correction_mm"]
        rows.append(row)
    return pd.DataFrame(rows)


def add_fixed_top(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    ok_mask = df["data_status"].eq("ok") & df["stage"].isin(["pre", "during", "post"])
    if source == "era5_station_interp":
        top_col = "profile_top_correction_grid_then_interp_mm"
        layer_col = "profile_layer_integral_grid_then_interp_mm"
        saas_col = "surface_saas_grid_then_interp_mm"
    else:
        top_col = "profile_top_correction_mm"
        layer_col = "profile_layer_integral_mm"
        saas_col = "surface_saas_ZHD_mm"
    ref = float(df.loc[ok_mask, top_col].median()) if ok_mask.any() else np.nan
    df["reference_top_correction_mm"] = ref
    df["top_correction_anomaly_mm"] = df[top_col] - ref
    df["profile_ZHD_fixed_top_mm"] = df[layer_col] + ref
    df["profile_minus_saas_fixed_top_mm"] = df["profile_ZHD_fixed_top_mm"] - df[saas_col]
    df["top_variation_explained_component_mm"] = df["profile_minus_saas_mm"] - df["profile_minus_saas_fixed_top_mm"]
    df["top_component_identity_error_mm"] = df["top_variation_explained_component_mm"] - df["top_correction_anomaly_mm"]
    bad = df.loc[ok_mask, "top_component_identity_error_mm"].abs().max()
    if pd.notna(bad) and bad > TOP_IDENTITY_TOL_MM:
        raise ValueError(f"{source}: fixed-top identity error {bad} mm")
    return df


def rmse(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.sqrt(np.mean(s**2))) if not s.empty else np.nan


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(data) < 5 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method=method))


def summarize_one(source: str, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if source == "era5_station_interp":
        top_p = "weighted_effective_top_pressure_hPa"
        top_h = "weighted_effective_top_geometric_height_m"
        layers = "weighted_integrated_layer_count"
        top_corr = "profile_top_correction_grid_then_interp_mm"
    else:
        top_p = "effective_top_pressure_hPa"
        top_h = "effective_top_geometric_height_m"
        layers = "integrated_layer_count"
        top_corr = "profile_top_correction_mm"
    for stage in ["pre", "during", "post"]:
        sub = df[df["stage"].eq(stage)].copy()
        ok = sub[sub["data_status"].eq("ok")].copy()
        row = {
            "source": source,
            "stage": stage,
            "n_total": len(sub),
            "n_ok": len(ok),
            "n_invalid": len(sub) - len(ok),
            "mean_effective_top_pressure_hPa": ok[top_p].mean(),
            "median_effective_top_pressure_hPa": ok[top_p].median(),
            "std_effective_top_pressure_hPa": ok[top_p].std(),
            "min_effective_top_pressure_hPa": ok[top_p].min(),
            "max_effective_top_pressure_hPa": ok[top_p].max(),
            "range_effective_top_pressure_hPa": ok[top_p].max() - ok[top_p].min(),
            "mean_effective_top_geometric_height_m": ok[top_h].mean(),
            "median_effective_top_geometric_height_m": ok[top_h].median(),
            "std_effective_top_geometric_height_m": ok[top_h].std(),
            "min_effective_top_geometric_height_m": ok[top_h].min(),
            "max_effective_top_geometric_height_m": ok[top_h].max(),
            "range_effective_top_geometric_height_m": ok[top_h].max() - ok[top_h].min(),
            "mean_integrated_layer_count": ok[layers].mean(),
            "min_integrated_layer_count": ok[layers].min(),
            "max_integrated_layer_count": ok[layers].max(),
            "n_top_level_removed": int(ok.get("top_level_removed_flag", pd.Series(False, index=ok.index)).fillna(False).sum()),
            "top_level_removed_fraction": float(ok.get("top_level_removed_flag", pd.Series(False, index=ok.index)).fillna(False).mean()) if len(ok) else np.nan,
            "mean_profile_layer_integral_mm": ok["profile_layer_integral_mm"].mean() if "profile_layer_integral_mm" in ok else ok["profile_layer_integral_grid_then_interp_mm"].mean(),
            "std_profile_layer_integral_mm": ok["profile_layer_integral_mm"].std() if "profile_layer_integral_mm" in ok else ok["profile_layer_integral_grid_then_interp_mm"].std(),
            "mean_profile_top_correction_mm": ok[top_corr].mean(),
            "median_profile_top_correction_mm": ok[top_corr].median(),
            "std_profile_top_correction_mm": ok[top_corr].std(),
            "min_profile_top_correction_mm": ok[top_corr].min(),
            "max_profile_top_correction_mm": ok[top_corr].max(),
            "range_profile_top_correction_mm": ok[top_corr].max() - ok[top_corr].min(),
            "mean_top_correction_fraction_percent": ok["top_correction_fraction_percent"].mean(),
            "max_top_correction_fraction_percent": ok["top_correction_fraction_percent"].max(),
            "mean_profile_minus_saas_mm": ok["profile_minus_saas_mm"].mean(),
            "rmse_profile_minus_saas_mm": rmse(ok["profile_minus_saas_mm"]),
            "max_abs_profile_minus_saas_mm": ok["profile_minus_saas_mm"].abs().max(),
            "mean_top_correction_anomaly_mm": ok["top_correction_anomaly_mm"].mean(),
            "rmse_top_correction_anomaly_mm": rmse(ok["top_correction_anomaly_mm"]),
            "max_abs_top_correction_anomaly_mm": ok["top_correction_anomaly_mm"].abs().max(),
            "mean_profile_minus_saas_fixed_top_mm": ok["profile_minus_saas_fixed_top_mm"].mean(),
            "rmse_profile_minus_saas_fixed_top_mm": rmse(ok["profile_minus_saas_fixed_top_mm"]),
            "max_abs_profile_minus_saas_fixed_top_mm": ok["profile_minus_saas_fixed_top_mm"].abs().max(),
            "pearson_r_difference_vs_top_pressure": corr(ok["profile_minus_saas_mm"], ok[top_p], "pearson"),
            "pearson_r_difference_vs_top_height": corr(ok["profile_minus_saas_mm"], ok[top_h], "pearson"),
            "pearson_r_difference_vs_top_correction": corr(ok["profile_minus_saas_mm"], ok[top_corr], "pearson"),
            "spearman_r_difference_vs_top_pressure": corr(ok["profile_minus_saas_mm"], ok[top_p], "spearman"),
            "spearman_r_difference_vs_top_height": corr(ok["profile_minus_saas_mm"], ok[top_h], "spearman"),
            "spearman_r_difference_vs_top_correction": corr(ok["profile_minus_saas_mm"], ok[top_corr], "spearman"),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    means = out.set_index("stage")
    for prefix, a, b in [("during_minus_pre", "during", "pre"), ("during_minus_post", "during", "post")]:
        if a in means.index and b in means.index:
            out.loc[:, f"{prefix}_top_pressure_hPa"] = means.loc[a, "mean_effective_top_pressure_hPa"] - means.loc[b, "mean_effective_top_pressure_hPa"]
            out.loc[:, f"{prefix}_top_height_m"] = means.loc[a, "mean_effective_top_geometric_height_m"] - means.loc[b, "mean_effective_top_geometric_height_m"]
            num = means.loc[a, "mean_profile_top_correction_mm"] - means.loc[b, "mean_profile_top_correction_mm"]
            den = means.loc[a, "mean_profile_minus_saas_mm"] - means.loc[b, "mean_profile_minus_saas_mm"]
            out.loc[:, f"{prefix}_top_correction_mm"] = num
            out.loc[:, f"{prefix}_profile_minus_saas_mm"] = den
            out.loc[:, f"{prefix}_fixed_top_difference_mm"] = means.loc[a, "mean_profile_minus_saas_fixed_top_mm"] - means.loc[b, "mean_profile_minus_saas_fixed_top_mm"]
            ratio_name = "top_variation_explanation_ratio_pre_to_during" if b == "pre" else "top_variation_explanation_ratio_post_to_during"
            out.loc[:, ratio_name] = num / den if abs(den) >= 1e-6 else np.nan
    return out


def write_report(path: Path, event: dict[str, Any], sounding: pd.DataFrame, grid: pd.DataFrame, station: pd.DataFrame, summary: pd.DataFrame) -> None:
    ok_s = sounding[sounding["data_status"].eq("ok")]
    ok_g = grid[grid["data_status"].eq("ok")]
    ok_st = station[station["data_status"].eq("ok")]
    lines = [
        "Step-2 top-boundary diagnostic report",
        f"event_id: {event['id']}",
        f"display_name: {event.get('display_name', event['id'])}",
        f"station_id: {event['station']['id']}",
        "",
    ]
    lines.append(f"Sounding raw/effective top changed records: {int(ok_s['top_level_removed_flag'].sum())}")
    lines.append(f"ERA5 grid raw/effective top changed records: {int(ok_g['top_level_removed_flag'].sum())}")
    lines.append(
        f"ERA5 effective top pressure range: {ok_g['effective_top_pressure_hPa'].min():.12g} to {ok_g['effective_top_pressure_hPa'].max():.12g} hPa"
    )
    if ok_g["effective_top_pressure_hPa"].max() - ok_g["effective_top_pressure_hPa"].min() <= 1e-10:
        lines.append("在数值精度范围内，ERA5有效顶部气压恒定，顶部气压覆盖范围的时间变化不是当前ERA5差异变化来源。")
    top_range = ok_st["profile_top_correction_grid_then_interp_mm"].max() - ok_st["profile_top_correction_grid_then_interp_mm"].min()
    lines.append(f"ERA5 station top-correction range: {top_range:.12g} mm")
    if top_range <= 0.01:
        lines.append("ERA5顶部补偿的时间变化小于或等于0.01 mm，无法解释零点几毫米至毫米级的Profile−Saas变化。")
    rmse_orig = rmse(ok_st["profile_minus_saas_mm"])
    rmse_fixed = rmse(ok_st["profile_minus_saas_fixed_top_mm"])
    lines.append(f"ERA5 station original RMSE: {rmse_orig:.12g} mm")
    lines.append(f"ERA5 station fixed-top RMSE: {rmse_fixed:.12g} mm")
    lines.append(f"ERA5 station RMSE change: {rmse_fixed - rmse_orig:.12g} mm")
    if abs(rmse_orig - rmse_fixed) <= 0.01:
        lines.append("将顶部补偿固定后，RMSE变化不超过0.01 mm，顶部补偿时间变化不是当前主要差异来源。")
    lines.append("")
    lines.append("Stage summary:")
    lines.append(summary.to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diag(path_png: Path, path_pdf: Path, event: dict[str, Any], stages: list[step1.StageDef], sounding: pd.DataFrame, station: pd.DataFrame) -> None:
    s = sounding[sounding["data_status"].eq("ok") & sounding["stage"].isin(["pre", "during", "post"])]
    e = station[station["data_status"].eq("ok") & station["stage"].isin(["pre", "during", "post"])]
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    colors = {"pre": "#dbeafe", "during": "#fee2e2", "post": "#dcfce7"}
    for ax in axes:
        for st in stages:
            ax.axvspan(st.start, st.end, color=colors[st.code], alpha=0.45, lw=0)
        ax.axhline(0, color="#475569", lw=0.8)
        ax.grid(True, color="#e2e8f0", lw=0.6)
    axes[0].plot(s["sounding_time_utc"], s["profile_minus_saas_mm"], "o", color="#2563eb", label="sounding")
    axes[0].plot(e["time_utc"], e["profile_minus_saas_mm"], color="#dc2626", label="ERA5 station")
    axes[0].plot(e["time_utc"], e["profile_minus_saas_fixed_top_mm"], "--", color="#dc2626", label="ERA5 fixed top")
    axes[0].plot(s["sounding_time_utc"], s["profile_minus_saas_fixed_top_mm"], "o", mfc="white", mec="#2563eb", label="sounding fixed top")
    axes[1].plot(s["sounding_time_utc"], s["effective_top_pressure_hPa"], "o", color="#2563eb", label="sounding")
    axes[1].plot(e["time_utc"], e["weighted_effective_top_pressure_hPa"], color="#dc2626", label="ERA5 station")
    axes[1].invert_yaxis()
    axes[2].plot(s["sounding_time_utc"], s["effective_top_geometric_height_m"] / 1000, "o", color="#2563eb", label="sounding")
    axes[2].plot(e["time_utc"], e["weighted_effective_top_geometric_height_m"] / 1000, color="#dc2626", label="ERA5 station")
    axes[3].plot(s["sounding_time_utc"], s["profile_top_correction_mm"], "o", color="#2563eb", label="sounding")
    axes[3].plot(e["time_utc"], e["profile_top_correction_grid_then_interp_mm"], color="#dc2626", label="ERA5 station")
    axes[3].axhline(s["reference_top_correction_mm"].iloc[0], color="#2563eb", ls=":", lw=1)
    axes[3].axhline(e["reference_top_correction_mm"].iloc[0], color="#dc2626", ls=":", lw=1)
    axes[4].plot(s["sounding_time_utc"], s["top_correction_anomaly_mm"], "o", color="#2563eb", label="sounding")
    axes[4].plot(e["time_utc"], e["top_correction_anomaly_mm"], color="#dc2626", label="ERA5 station")
    labels = [
        "Profile - Saas ZHD (mm)",
        "Effective top pressure (hPa, inverted)",
        "Effective top height (km)",
        "Top correction (mm)",
        "Top-correction anomaly (mm)",
    ]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    axes[-1].set_xlabel("UTC time")
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle(f"Step-2 top-boundary diagnostic: {event.get('display_name', event['id'])} station {event['station']['id']}", fontweight="bold")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = step1.resolve_config_path(args.config)
    events = step1.load_events(config_path)
    event_id = args.event.lower().strip()
    if event_id not in events:
        print(f"Event not found: {event_id}")
        print(", ".join(sorted(events)))
        raise SystemExit(1)
    event = events[event_id]
    paths = step1.event_paths(event, config_path)
    stages = step1.stages_from_event(event)
    out_dir = (args.output_root / "step2_top_boundary" / event_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sounding = add_fixed_top(diagnose_sounding(event, paths, stages), "sounding")
    grid, interp_mode = diagnose_era5_grid(event, paths, stages)
    grid = add_fixed_top(grid, "era5_grid")
    station = add_fixed_top(diagnose_station(event, paths, stages, grid, interp_mode), "era5_station_interp")

    # Do not explain records whose official consistency failed.
    for name, frame in [("sounding", sounding), ("era5_grid", grid), ("era5_station", station)]:
        bad = frame[frame["data_status"].eq("official_output_mismatch")]
        if not bad.empty:
            print(f"{name}: official_output_mismatch found; affected records retained in CSV.")

    sounding_csv = out_dir / "sounding_top_boundary_detail.csv"
    grid_csv = out_dir / "era5_grid_top_boundary_detail.csv"
    station_csv = out_dir / "era5_station_top_boundary_detail.csv"
    summary_csv = out_dir / "step2_top_boundary_summary.csv"
    report_txt = out_dir / "step2_top_boundary_report.txt"
    png = out_dir / "step2_top_boundary_diagnostic.png"
    pdf = out_dir / "step2_top_boundary_diagnostic.pdf"
    sounding.to_csv(sounding_csv, index=False, encoding="utf-8-sig")
    grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    station.to_csv(station_csv, index=False, encoding="utf-8-sig")
    summary = pd.concat(
        [
            summarize_one("sounding", sounding),
            summarize_one("era5_grid", grid),
            summarize_one("era5_station_interp", station),
        ],
        ignore_index=True,
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_report(report_txt, event, sounding, grid, station, summary)
    plot_diag(png, pdf, event, stages, sounding, station)

    ok_s = sounding[sounding["data_status"].eq("ok")]
    ok_g = grid[grid["data_status"].eq("ok")]
    ok_st = station[station["data_status"].eq("ok")]
    print("Config file:", config_path.resolve())
    print("Event:", event_id, event.get("display_name", event_id), "station", event["station"]["id"])
    print("Output root:", args.output_root.resolve())
    print("Sounding input:", paths.sounding_input.resolve())
    print("ERA5 input:", paths.era5_input.resolve())
    print("Official outputs:", paths.sounding_saas.resolve(), paths.sounding_profile.resolve(), paths.era5_saas_station.resolve(), paths.era5_profile_station.resolve())
    print("Stage ok counts:", {s: {"sounding": int((ok_s.stage == s).sum()), "era5_grid": int((ok_g.stage == s).sum()), "era5_station": int((ok_st.stage == s).sum())} for s in ["pre", "during", "post"]})
    print("Sounding top removed count:", int(ok_s["top_level_removed_flag"].sum()))
    print("ERA5 grid top removed count:", int(ok_g["top_level_removed_flag"].sum()))
    print("Sounding top pressure range:", float(ok_s["effective_top_pressure_hPa"].min()), float(ok_s["effective_top_pressure_hPa"].max()))
    print("ERA5 top pressure range:", float(ok_g["effective_top_pressure_hPa"].min()), float(ok_g["effective_top_pressure_hPa"].max()))
    print("Sounding top correction range mm:", float(ok_s["profile_top_correction_mm"].max() - ok_s["profile_top_correction_mm"].min()))
    print("ERA5 station top correction range mm:", float(ok_st["profile_top_correction_grid_then_interp_mm"].max() - ok_st["profile_top_correction_grid_then_interp_mm"].min()))
    rmse_orig = rmse(ok_st["profile_minus_saas_mm"])
    rmse_fixed = rmse(ok_st["profile_minus_saas_fixed_top_mm"])
    print("ERA5 station original Profile-Saas RMSE mm:", rmse_orig)
    print("ERA5 station fixed-top RMSE mm:", rmse_fixed)
    print("ERA5 station RMSE change mm:", rmse_fixed - rmse_orig)
    ratio_cols = [c for c in summary.columns if "top_variation_explanation_ratio" in c]
    print("Top variation explanation ratios:")
    print(summary[["source", "stage"] + ratio_cols].drop_duplicates("source").to_string(index=False))
    print("Generated files:")
    for p in [sounding_csv, grid_csv, station_csv, summary_csv, report_txt, png, pdf]:
        print(p.resolve())


if __name__ == "__main__":
    main()
