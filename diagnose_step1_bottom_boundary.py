from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

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
import zhd_workflow  # noqa: E402


SAAS_COEFFICIENT = 0.0022793
ZHD_TOL_MM = 1e-6
WEIGHT_SUM_TOL = 1e-12


@dataclass(frozen=True)
class StageDef:
    code: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    include_start: bool
    include_end: bool


@dataclass(frozen=True)
class EventPaths:
    out_dir: Path
    diagnostic_dir: Path
    sounding_input: Path
    era5_input: Path
    sounding_saas: Path
    sounding_profile: Path
    era5_saas_grid4: Path
    era5_profile_grid4: Path
    era5_saas_station: Path
    era5_profile_station: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step-1 diagnostic for Profile-minus-Saastamoinen bottom-boundary "
            "and ERA5 interpolation-order effects."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_DIR / "typhoon_events.json",
        help="Path to typhoon event JSON.",
    )
    parser.add_argument(
        "--event",
        required=True,
        help="Single event id to diagnose, for example hanna.",
    )
    return parser.parse_args()


def load_events(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    events = raw.get("events", [])
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event["id"]).strip().lower()
        event["id"] = event_id
        by_id[event_id] = event
    return by_id


def resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    code_relative = CODE_DIR / path
    if code_relative.exists():
        return code_relative.resolve()
    return (ROOT_DIR / path).resolve()


def stages_from_event(event: dict[str, Any]) -> list[StageDef]:
    plot = event["plot"]
    time_start = pd.Timestamp(plot["time_start"])
    pre_end = pd.Timestamp(plot["pre_end"])
    impact_end = pd.Timestamp(plot["impact_end"])
    time_end = pd.Timestamp(plot["time_end"])
    # Keep the same boundary convention used by the current stage summaries:
    # pre: [time_start, pre_end)
    # during: [pre_end, impact_end]
    # post: (impact_end, time_end]
    return [
        StageDef("pre", "pre", time_start, pre_end, True, False),
        StageDef("during", "during", pre_end, impact_end, True, True),
        StageDef("post", "post", impact_end, time_end, False, True),
    ]


def assign_stage(time_utc: pd.Timestamp, stages: list[StageDef]) -> str:
    for stage in stages:
        left_ok = time_utc >= stage.start if stage.include_start else time_utc > stage.start
        right_ok = time_utc <= stage.end if stage.include_end else time_utc < stage.end
        if left_ok and right_ok:
            return stage.code
    return "outside"


def stage_mask(df: pd.DataFrame, stage: StageDef, time_col: str = "time_utc") -> pd.Series:
    if stage.include_start:
        left = df[time_col] >= stage.start
    else:
        left = df[time_col] > stage.start
    if stage.include_end:
        right = df[time_col] <= stage.end
    else:
        right = df[time_col] < stage.end
    return left & right


def event_paths(event: dict[str, Any], config_path: Path) -> EventPaths:
    if config_path.name == "typhoon_events.json":
        zhd_workflow.DEFAULT_CONFIG = config_path
    out_dir = zhd_workflow.event_out_dir(event)
    sounding_input = zhd_workflow.expected_sounding_qc_combined(event)
    era5_input = zhd_workflow.era5_data_dir(event) / "era5_pressure_raw_extract.csv"
    diag_dir = ROOT_DIR / "out" / "diagnostics" / "step1_bottom_boundary" / event["id"]
    return EventPaths(
        out_dir=out_dir,
        diagnostic_dir=diag_dir,
        sounding_input=sounding_input,
        era5_input=era5_input,
        sounding_saas=out_dir / zhd_workflow.SOUNDING_OUTPUTS["saas"],
        sounding_profile=out_dir / zhd_workflow.SOUNDING_OUTPUTS["profile"],
        era5_saas_grid4=out_dir / zhd_workflow.ERA5_OUTPUTS["saas_grid4"],
        era5_profile_grid4=out_dir / zhd_workflow.ERA5_OUTPUTS["profile_grid4"],
        era5_saas_station=out_dir / zhd_workflow.ERA5_OUTPUTS["saas_station"],
        era5_profile_station=out_dir / zhd_workflow.ERA5_OUTPUTS["profile_station"],
    )


def saas_denominator(height_m: float, latitude_deg: float) -> float:
    lat_rad = np.deg2rad(latitude_deg)
    return float(1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 2.8e-7 * height_m)


def saas_mm(pressure_hpa: float, height_m: float, latitude_deg: float) -> float:
    return float(calc_zhd_sounding.saastamoinen_zhd(pressure_hpa, height_m, latitude_deg) * 1000.0)


def pressure_equivalent_mm(delta_pressure_hpa: float, denominator: float) -> float:
    return float(SAAS_COEFFICIENT * delta_pressure_hpa / denominator * 1000.0)


def height_equivalent_mm(
    pressure_hpa: float,
    saas_height_m: float,
    profile_height_m: float,
    latitude_deg: float,
) -> float:
    profile_height_saas = calc_zhd_sounding.saastamoinen_zhd(
        pressure_hpa, profile_height_m, latitude_deg
    )
    saas_height_saas = calc_zhd_sounding.saastamoinen_zhd(
        pressure_hpa, saas_height_m, latitude_deg
    )
    return float((profile_height_saas - saas_height_saas) * 1000.0)


def safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def read_official_zhd(path: Path, key_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=key_cols + ["official_ZHD_mm"])
    df = pd.read_csv(path, parse_dates=["time_utc"])
    df["official_ZHD_mm"] = pd.to_numeric(df["ZHD_m"], errors="coerce") * 1000.0
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[key_cols + ["official_ZHD_mm"]].copy()


def find_official(
    official: pd.DataFrame,
    time_utc: pd.Timestamp,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> float:
    if official.empty:
        return float("nan")
    mask = official["time_utc"] == time_utc
    if latitude is not None and "latitude" in official.columns:
        mask &= np.isclose(official["latitude"], latitude, rtol=0.0, atol=1e-9)
    if longitude is not None and "longitude" in official.columns:
        mask &= np.isclose(official["longitude"], longitude, rtol=0.0, atol=1e-9)
    matched = official[mask]
    if matched.empty:
        return float("nan")
    return float(matched["official_ZHD_mm"].iloc[0])


def status_with_official_errors(
    current_status: str,
    official_saas: float,
    official_profile: float,
    saas_value: float,
    profile_value: float,
) -> tuple[str, float, float]:
    saas_error = float("nan")
    profile_error = float("nan")
    if current_status != "ok":
        return current_status, saas_error, profile_error
    if pd.isna(official_saas) or pd.isna(official_profile):
        return "missing_official_output", saas_error, profile_error
    saas_error = saas_value - official_saas
    profile_error = profile_value - official_profile
    if abs(saas_error) > ZHD_TOL_MM or abs(profile_error) > ZHD_TOL_MM:
        return "official_output_mismatch", saas_error, profile_error
    return "ok", saas_error, profile_error


def duplicate_count(df: pd.DataFrame, cols: list[str]) -> int:
    existing = [col for col in cols if col in df.columns]
    if not existing or df.empty:
        return 0
    return int(df.duplicated(existing).sum())


def required_nan_inf_counts(df: pd.DataFrame, cols: list[str]) -> tuple[int, int]:
    existing = [col for col in cols if col in df.columns]
    if not existing or df.empty:
        return 0, 0
    numeric = df[existing].apply(pd.to_numeric, errors="coerce")
    nan_count = int(numeric.isna().any(axis=1).sum())
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).any(axis=1).sum())
    return nan_count, inf_count


def sounding_profile_components(
    profile: pd.DataFrame,
    latitude_deg: float,
) -> tuple[float, float, float]:
    if len(profile) < 2:
        return float("nan"), float("nan"), float("nan")
    height_geometric = calc_zhd_sounding.geometric_height(
        profile[calc_zhd_sounding.H_COL].to_numpy(dtype=float),
        latitude_deg,
    )
    nh = profile["Nh"].to_numpy(dtype=float)
    dz_geometric = np.diff(height_geometric)
    zhd_int = 1e-6 * np.sum(
        dz_geometric * calc_zhd_sounding.log_mean(nh[:-1], nh[1:])
    )
    end_layer = profile.iloc[-1]
    p_top = float(end_layer[calc_zhd_sounding.P_COL])
    h_top_geopotential = float(end_layer[calc_zhd_sounding.H_COL])
    h_top_geometric = float(
        calc_zhd_sounding.geometric_height(h_top_geopotential, latitude_deg)
    )
    zhd_top = calc_zhd_sounding.saastamoinen_zhd(
        p_top,
        h_top_geometric,
        latitude_deg,
    )
    return float(zhd_int * 1000.0), float(zhd_top * 1000.0), float((zhd_int + zhd_top) * 1000.0)


def sounding_initial_valid(group: pd.DataFrame) -> pd.DataFrame:
    subset = [
        calc_zhd_sounding.LON_COL,
        calc_zhd_sounding.LAT_COL,
        calc_zhd_sounding.P_COL,
        calc_zhd_sounding.H_COL,
        calc_zhd_sounding.TC_COL,
        calc_zhd_sounding.RH_PERCENT_COL,
    ]
    return group.dropna(subset=subset).copy()


def sounding_effective_profile(initial: pd.DataFrame) -> pd.DataFrame:
    work = initial.sort_values(calc_zhd_sounding.H_COL).copy()
    work["temperature_K"] = work[calc_zhd_sounding.TC_COL] + 273.15
    work["RH"] = work[calc_zhd_sounding.RH_PERCENT_COL] / 100.0
    work["e"] = calc_zhd_sounding.saturation_water_vapor_pressure(
        work[calc_zhd_sounding.P_COL],
        work["temperature_K"],
        work["RH"],
    )
    work["Nh"] = (
        77.6890
        * (work[calc_zhd_sounding.P_COL] - 0.37803 * work["e"])
        / work["temperature_K"]
    )
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["Nh"])
    return work[work["Nh"] > 0.0].copy()


def diagnose_sounding(
    event: dict[str, Any],
    paths: EventPaths,
    stages: list[StageDef],
) -> pd.DataFrame:
    if not paths.sounding_input.exists():
        return pd.DataFrame(
            [
                {
                    "event_id": event["id"],
                    "display_name": event.get("display_name", event["id"]),
                    "station_id": str(event["station"]["id"]),
                    "stage": stage.code,
                    "sounding_time_utc": pd.NaT,
                    "data_status": "missing_file",
                    "warning_message": str(paths.sounding_input),
                }
                for stage in stages
            ]
        )

    df = calc_zhd_sounding.read_sounding_csv(paths.sounding_input)
    df["_original_index"] = np.arange(len(df), dtype=int)
    official_saas = read_official_zhd(paths.sounding_saas, ["time_utc"])
    official_profile = read_official_zhd(paths.sounding_profile, ["time_utc"])

    rows: list[dict[str, Any]] = []
    for time_utc, group in df.groupby(calc_zhd_sounding.TIME_COL, sort=True):
        time_utc = pd.Timestamp(time_utc)
        raw_count = int(len(group))
        nan_count, inf_count = required_nan_inf_counts(
            group,
            [
                calc_zhd_sounding.P_COL,
                calc_zhd_sounding.H_COL,
                calc_zhd_sounding.TC_COL,
                calc_zhd_sounding.RH_PERCENT_COL,
            ],
        )
        initial = sounding_initial_valid(group)
        effective = sounding_effective_profile(initial).reset_index(drop=True)

        row: dict[str, Any] = {
            "event_id": event["id"],
            "display_name": event.get("display_name", event["id"]),
            "station_id": str(event["station"]["id"]),
            "stage": assign_stage(time_utc, stages),
            "sounding_time_utc": time_utc,
            "raw_level_count": raw_count,
            "initial_valid_level_count": int(len(initial)),
            "profile_effective_level_count": int(len(effective)),
            "duplicate_time_count": duplicate_count(group, [calc_zhd_sounding.OBS_TIME_COL]),
            "duplicate_layer_count": duplicate_count(
                group,
                [calc_zhd_sounding.P_COL, calc_zhd_sounding.H_COL],
            ),
            "nan_required_row_count": nan_count,
            "inf_required_row_count": inf_count,
            "warning_message": "",
        }

        status = "ok"
        if raw_count == 0:
            status = "no_raw_rows"
        elif initial.empty:
            status = "no_initial_valid_layers"
        elif effective.empty:
            status = "no_effective_profile_layers"
        elif len(effective) < 2:
            status = "insufficient_profile_layers"

        if status == "ok":
            saas_bottom = initial.loc[initial[calc_zhd_sounding.P_COL].idxmax()]
            profile_bottom = effective.iloc[0]
            p0 = float(saas_bottom[calc_zhd_sounding.P_COL])
            lat0 = float(saas_bottom[calc_zhd_sounding.LAT_COL])
            lon0 = float(saas_bottom[calc_zhd_sounding.LON_COL])
            h0_gpot = float(saas_bottom[calc_zhd_sounding.H_COL])
            h0_geom = float(calc_zhd_sounding.geometric_height(h0_gpot, lat0))

            pp = float(profile_bottom[calc_zhd_sounding.P_COL])
            plat = float(profile_bottom[calc_zhd_sounding.LAT_COL])
            plon = float(profile_bottom[calc_zhd_sounding.LON_COL])
            ph_gpot = float(profile_bottom[calc_zhd_sounding.H_COL])
            ph_geom = float(calc_zhd_sounding.geometric_height(ph_gpot, lat0))

            denom = saas_denominator(h0_geom, lat0)
            saas_value = saas_mm(p0, h0_geom, lat0)
            layer_int, top_corr, profile_value = sounding_profile_components(effective, lat0)
            profile_minus_saas = profile_value - saas_value
            dp = pp - p0
            dh_gpot = ph_gpot - h0_gpot
            dh_geom = ph_geom - h0_geom
            p_equiv = pressure_equivalent_mm(dp, denom)
            h_equiv = height_equivalent_mm(p0, h0_geom, ph_geom, lat0)
            residual = profile_minus_saas - p_equiv - h_equiv

            off_saas = find_official(official_saas, time_utc)
            off_profile = find_official(official_profile, time_utc)
            status, saas_err, profile_err = status_with_official_errors(
                status, off_saas, off_profile, saas_value, profile_value
            )

            row.update(
                {
                    "saas_bottom_original_index": int(saas_bottom["_original_index"]),
                    "saas_bottom_pressure_hPa": p0,
                    "saas_bottom_geopotential_height_m": h0_gpot,
                    "saas_bottom_geometric_height_m": h0_geom,
                    "saas_bottom_latitude_deg": lat0,
                    "saas_bottom_longitude_deg": lon0,
                    "profile_bottom_original_index": int(profile_bottom["_original_index"]),
                    "profile_bottom_pressure_hPa": pp,
                    "profile_bottom_geopotential_height_m": ph_gpot,
                    "profile_bottom_geometric_height_m": ph_geom,
                    "profile_bottom_latitude_deg": plat,
                    "profile_bottom_longitude_deg": plon,
                    "bottom_pressure_difference_hPa": dp,
                    "bottom_geopotential_height_difference_m": dh_gpot,
                    "bottom_geometric_height_difference_m": dh_geom,
                    "bottom_latitude_difference_deg": plat - lat0,
                    "saas_denominator": denom,
                    "saas_ZHD_mm": saas_value,
                    "profile_layer_integral_mm": layer_int,
                    "profile_top_correction_mm": top_corr,
                    "profile_ZHD_mm": profile_value,
                    "profile_minus_saas_mm": profile_minus_saas,
                    "pressure_difference_equivalent_ZHD_mm": p_equiv,
                    "height_difference_equivalent_ZHD_mm": h_equiv,
                    "residual_after_bottom_boundary_mm": residual,
                    "official_saas_ZHD_mm": off_saas,
                    "official_profile_ZHD_mm": off_profile,
                    "saas_recalculation_error_mm": saas_err,
                    "profile_recalculation_error_mm": profile_err,
                }
            )
        row["data_status"] = status
        if int(row.get("initial_valid_level_count", 0)) > int(row.get("profile_effective_level_count", 0)):
            removed = int(row["initial_valid_level_count"]) - int(row["profile_effective_level_count"])
            row["warning_message"] = (row.get("warning_message") or "") + f"Nh_filter_removed={removed};"
        rows.append(row)

    return pd.DataFrame(rows)


def era5_initial_valid(group: pd.DataFrame) -> pd.DataFrame:
    return group.dropna(
        subset=[
            calc_zhd_era5.P_COL,
            calc_zhd_era5.H_COL,
            calc_zhd_era5.T_COL,
            calc_zhd_era5.RH_PERCENT_COL,
        ]
    ).copy()


def era5_profile_components(valid: pd.DataFrame, latitude: float) -> tuple[pd.DataFrame, float, float, float]:
    profile = valid.sort_values(calc_zhd_era5.H_COL).copy()
    profile["e"] = calc_zhd_era5.water_vapor_pressure(
        profile[calc_zhd_era5.P_COL],
        profile[calc_zhd_era5.T_COL],
        profile[calc_zhd_era5.RH_PERCENT_COL],
    )
    profile["Nh"] = 77.6890 * (
        profile[calc_zhd_era5.P_COL] - 0.37803 * profile["e"]
    ) / profile[calc_zhd_era5.T_COL]
    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(subset=["Nh"])
    profile = profile[profile["Nh"] > 0.0].copy()
    if len(profile) < 2:
        return profile, float("nan"), float("nan"), float("nan")
    height = profile[calc_zhd_era5.H_COL].to_numpy(dtype=float)
    height_geom = calc_zhd_era5.geometric_height(height, latitude)
    nh = profile["Nh"].to_numpy(dtype=float)
    d_height = np.diff(height)
    dz_geom = np.diff(height_geom)
    valid_interval = d_height >= 0.0
    zhd_int = 1e-6 * np.sum(
        dz_geom[valid_interval]
        * calc_zhd_era5.log_mean(nh[:-1][valid_interval], nh[1:][valid_interval])
    )
    top_layer = profile.loc[profile[calc_zhd_era5.H_COL].idxmax()]
    p_top = float(top_layer[calc_zhd_era5.P_COL])
    h_top = float(top_layer[calc_zhd_era5.H_COL])
    h_top_geom = float(calc_zhd_era5.geometric_height(h_top, latitude))
    zhd_top = calc_zhd_era5.saastamoinen_zhd(p_top, h_top_geom, latitude)
    return profile, float(zhd_int * 1000.0), float(zhd_top * 1000.0), float((zhd_int + zhd_top) * 1000.0)


def grid_weights(
    lon_w: float,
    lon_e: float,
    lat_s: float,
    lat_n: float,
    station_lon: float,
    station_lat: float,
) -> tuple[str, dict[tuple[float, float], tuple[str, float]]]:
    on_lon_grid = np.isclose(lon_w, lon_e, rtol=0.0, atol=1e-10)
    on_lat_grid = np.isclose(lat_s, lat_n, rtol=0.0, atol=1e-10)
    if on_lon_grid and on_lat_grid:
        return "nearest grid point", {(lat_s, lon_w): ("SW", 1.0)}
    if on_lon_grid:
        weight_n = (station_lat - lat_s) / (lat_n - lat_s)
        return "latitude linear", {
            (lat_s, lon_w): ("SW", float(1.0 - weight_n)),
            (lat_n, lon_w): ("NW", float(weight_n)),
        }
    if on_lat_grid:
        weight_e = (station_lon - lon_w) / (lon_e - lon_w)
        return "longitude linear", {
            (lat_s, lon_w): ("SW", float(1.0 - weight_e)),
            (lat_s, lon_e): ("SE", float(weight_e)),
        }
    denominator = (lon_e - lon_w) * (lat_n - lat_s)
    return "bilinear", {
        (lat_s, lon_w): ("SW", float(((lon_e - station_lon) * (lat_n - station_lat)) / denominator)),
        (lat_s, lon_e): ("SE", float(((station_lon - lon_w) * (lat_n - station_lat)) / denominator)),
        (lat_n, lon_w): ("NW", float(((lon_e - station_lon) * (station_lat - lat_s)) / denominator)),
        (lat_n, lon_e): ("NE", float(((station_lon - lon_w) * (station_lat - lat_s)) / denominator)),
    }


def diagnose_era5_grid(
    event: dict[str, Any],
    paths: EventPaths,
    stages: list[StageDef],
) -> tuple[pd.DataFrame, str, dict[tuple[float, float], tuple[str, float]]]:
    station_lat = float(event["station"]["latitude"])
    station_lon = float(event["station"]["longitude"])
    calc_zhd_era5.STATION_LAT = station_lat
    calc_zhd_era5.STATION_LON = station_lon
    if not paths.era5_input.exists():
        empty = pd.DataFrame(
            [
                {
                    "event_id": event["id"],
                    "stage": stage.code,
                    "time_utc": pd.NaT,
                    "data_status": "missing_file",
                    "warning_message": str(paths.era5_input),
                }
                for stage in stages
            ]
        )
        return empty, "missing", {}

    raw = calc_zhd_era5.read_era5_csv(paths.era5_input)
    raw["_original_index"] = np.arange(len(raw), dtype=int)
    lon_w, lon_e, lat_s, lat_n = calc_zhd_era5.find_surrounding_grid(raw)
    interp_mode, weights = grid_weights(lon_w, lon_e, lat_s, lat_n, station_lon, station_lat)
    weight_sum = sum(weight for _, weight in weights.values())
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"Interpolation weights sum to {weight_sum}, not 1.")

    grid4 = calc_zhd_era5.select_grid4(raw, lon_w, lon_e, lat_s, lat_n)
    official_saas = read_official_zhd(paths.era5_saas_grid4, ["time_utc", "latitude", "longitude"])
    official_profile = read_official_zhd(paths.era5_profile_grid4, ["time_utc", "latitude", "longitude"])
    rows: list[dict[str, Any]] = []
    group_cols = [calc_zhd_era5.TIME_COL, calc_zhd_era5.LAT_COL, calc_zhd_era5.LON_COL]
    for (time_utc, latitude, longitude), group in grid4.groupby(group_cols, sort=True):
        time_utc = pd.Timestamp(time_utc)
        latitude = float(latitude)
        longitude = float(longitude)
        pos, weight = weights.get((latitude, longitude), ("unused", 0.0))
        raw_count = int(len(group))
        nan_count, inf_count = required_nan_inf_counts(
            group,
            [
                calc_zhd_era5.P_COL,
                calc_zhd_era5.H_COL,
                calc_zhd_era5.T_COL,
                calc_zhd_era5.RH_PERCENT_COL,
            ],
        )
        initial = era5_initial_valid(group)
        row: dict[str, Any] = {
            "event_id": event["id"],
            "stage": assign_stage(time_utc, stages),
            "time_utc": time_utc,
            "grid_position": pos,
            "grid_latitude": latitude,
            "grid_longitude": longitude,
            "raw_level_count": raw_count,
            "initial_valid_level_count": int(len(initial)),
            "duplicate_layer_count": duplicate_count(
                group,
                [calc_zhd_era5.P_COL, calc_zhd_era5.H_COL],
            ),
            "nan_required_row_count": nan_count,
            "inf_required_row_count": inf_count,
            "interpolation_weight": weight,
            "warning_message": "",
        }
        status = "ok"
        if raw_count == 0:
            status = "no_raw_rows"
        elif initial.empty:
            status = "no_initial_valid_layers"
        elif len(initial) < 2:
            status = "insufficient_profile_layers"

        if status == "ok":
            saas_bottom = initial.loc[initial[calc_zhd_era5.P_COL].idxmax()]
            p0 = float(saas_bottom[calc_zhd_era5.P_COL])
            h0_gpot = float(saas_bottom[calc_zhd_era5.H_COL])
            h0_geom = float(calc_zhd_era5.geometric_height(h0_gpot, latitude))
            saas_value = saas_mm(p0, h0_geom, latitude)
            profile, layer_int, top_corr, profile_value = era5_profile_components(initial, latitude)
            row["profile_effective_level_count"] = int(len(profile))
            if profile.empty:
                status = "no_effective_profile_layers"
            elif len(profile) < 2:
                status = "insufficient_profile_layers"
            else:
                profile_bottom = profile.iloc[0]
                pp = float(profile_bottom[calc_zhd_era5.P_COL])
                ph_gpot = float(profile_bottom[calc_zhd_era5.H_COL])
                ph_geom = float(calc_zhd_era5.geometric_height(ph_gpot, latitude))
                denom = saas_denominator(h0_geom, latitude)
                profile_minus_saas = profile_value - saas_value
                dp = pp - p0
                dh_gpot = ph_gpot - h0_gpot
                dh_geom = ph_geom - h0_geom
                p_equiv = pressure_equivalent_mm(dp, denom)
                h_equiv = height_equivalent_mm(p0, h0_geom, ph_geom, latitude)
                residual = profile_minus_saas - p_equiv - h_equiv
                off_saas = find_official(official_saas, time_utc, latitude, longitude)
                off_profile = find_official(official_profile, time_utc, latitude, longitude)
                status, saas_err, profile_err = status_with_official_errors(
                    status, off_saas, off_profile, saas_value, profile_value
                )
                row.update(
                    {
                        "saas_bottom_pressure_hPa": p0,
                        "saas_bottom_geopotential_height_m": h0_gpot,
                        "saas_bottom_geometric_height_m": h0_geom,
                        "profile_bottom_pressure_hPa": pp,
                        "profile_bottom_geopotential_height_m": ph_gpot,
                        "profile_bottom_geometric_height_m": ph_geom,
                        "bottom_pressure_difference_hPa": dp,
                        "bottom_geopotential_height_difference_m": dh_gpot,
                        "bottom_geometric_height_difference_m": dh_geom,
                        "saas_denominator": denom,
                        "saas_ZHD_mm": saas_value,
                        "profile_layer_integral_mm": layer_int,
                        "profile_top_correction_mm": top_corr,
                        "profile_ZHD_mm": profile_value,
                        "profile_minus_saas_mm": profile_minus_saas,
                        "pressure_difference_equivalent_ZHD_mm": p_equiv,
                        "height_difference_equivalent_ZHD_mm": h_equiv,
                        "residual_after_bottom_boundary_mm": residual,
                        "official_saas_ZHD_mm": off_saas,
                        "official_profile_ZHD_mm": off_profile,
                        "saas_recalculation_error_mm": saas_err,
                        "profile_recalculation_error_mm": profile_err,
                    }
                )
        row["data_status"] = status
        if int(row.get("initial_valid_level_count", 0)) > int(row.get("profile_effective_level_count", 0)):
            removed = int(row["initial_valid_level_count"]) - int(row.get("profile_effective_level_count", 0))
            row["warning_message"] = (row.get("warning_message") or "") + f"Nh_filter_removed={removed};"
        rows.append(row)
    return pd.DataFrame(rows), interp_mode, weights


def station_from_grid(
    event: dict[str, Any],
    paths: EventPaths,
    stages: list[StageDef],
    grid_detail: pd.DataFrame,
    interp_mode: str,
    weights: dict[tuple[float, float], tuple[str, float]],
) -> pd.DataFrame:
    if grid_detail.empty or not weights:
        return pd.DataFrame()
    station_lat = float(event["station"]["latitude"])
    station_lon = float(event["station"]["longitude"])
    official_saas = read_official_zhd(paths.era5_saas_station, ["time_utc"])
    official_profile = read_official_zhd(paths.era5_profile_station, ["time_utc"])
    rows: list[dict[str, Any]] = []
    ok_grid = grid_detail[grid_detail["data_status"] == "ok"].copy()
    for time_utc, group in ok_grid.groupby("time_utc", sort=True):
        time_utc = pd.Timestamp(time_utc)
        weight_sum = float(group["interpolation_weight"].sum())
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError(f"{time_utc}: interpolation weights sum to {weight_sum}, not 1.")

        def weighted(col: str) -> float:
            return float((group[col] * group["interpolation_weight"]).sum())

        p0_interp = weighted("saas_bottom_pressure_hPa")
        h0_gpot_interp = weighted("saas_bottom_geopotential_height_m")
        h0_geom_interp = weighted("saas_bottom_geometric_height_m")
        profile_h_geom_interp = weighted("profile_bottom_geometric_height_m")
        profile_h_gpot_interp = weighted("profile_bottom_geopotential_height_m")
        saas_grid_then_interp = weighted("saas_ZHD_mm")
        profile_grid_then_interp = weighted("profile_ZHD_mm")
        saas_from_inputs = saas_mm(p0_interp, h0_geom_interp, station_lat)
        order_effect = saas_grid_then_interp - saas_from_inputs
        profile_minus_saas_current = profile_grid_then_interp - saas_grid_then_interp
        weighted_dp = weighted("bottom_pressure_difference_hPa")
        weighted_dh = weighted("bottom_geometric_height_difference_m")
        denom = saas_denominator(h0_geom_interp, station_lat)
        weighted_p_equiv = pressure_equivalent_mm(weighted_dp, denom)
        weighted_h_equiv = height_equivalent_mm(
            p0_interp, h0_geom_interp, profile_h_geom_interp, station_lat
        )
        residual_boundary = (
            profile_minus_saas_current - weighted_p_equiv - weighted_h_equiv
        )
        # Sign check for the interpolation-order residual:
        # current = profile_grid_then_interp - saas_grid_then_interp.
        # order_effect = saas_grid_then_interp - saas_from_interpolated_inputs.
        # Re-expressing the current difference against the diagnostic Saas value gives
        # profile_grid_then_interp - saas_from_interpolated_inputs
        # = current + order_effect. Therefore removing bottom-boundary terms while
        # accounting for interpolation order is current - bottom_terms + order_effect.
        residual_boundary_order = (
            profile_minus_saas_current
            - weighted_p_equiv
            - weighted_h_equiv
            + order_effect
        )
        off_saas = find_official(official_saas, time_utc)
        off_profile = find_official(official_profile, time_utc)
        saas_err = saas_grid_then_interp - off_saas if not pd.isna(off_saas) else float("nan")
        profile_err = (
            profile_grid_then_interp - off_profile if not pd.isna(off_profile) else float("nan")
        )
        status = "ok"
        if pd.isna(off_saas) or pd.isna(off_profile):
            status = "missing_official_output"
        elif abs(saas_err) > ZHD_TOL_MM or abs(profile_err) > ZHD_TOL_MM:
            status = "official_output_mismatch"

        row: dict[str, Any] = {
            "event_id": event["id"],
            "stage": assign_stage(time_utc, stages),
            "time_utc": time_utc,
            "station_latitude": station_lat,
            "station_longitude": station_lon,
            "interpolation_mode": interp_mode,
            "p0_interpolated_hPa": p0_interp,
            "h0_geopotential_interpolated_m": h0_gpot_interp,
            "h0_geometric_interpolated_m": h0_geom_interp,
            "profile_bottom_geopotential_interpolated_m": profile_h_gpot_interp,
            "profile_bottom_geometric_interpolated_m": profile_h_geom_interp,
            "saas_grid_then_interp_mm": saas_grid_then_interp,
            "saas_from_interpolated_inputs_mm": saas_from_inputs,
            "saas_interpolation_order_effect_mm": order_effect,
            "profile_grid_then_interp_mm": profile_grid_then_interp,
            "profile_minus_saas_current_mm": profile_minus_saas_current,
            "weighted_bottom_pressure_difference_hPa": weighted_dp,
            "weighted_bottom_height_difference_m": weighted_dh,
            "weighted_pressure_equivalent_ZHD_mm": weighted_p_equiv,
            "weighted_height_equivalent_ZHD_mm": weighted_h_equiv,
            "residual_after_bottom_boundary_mm": residual_boundary,
            "residual_after_bottom_and_interp_order_mm": residual_boundary_order,
            "official_saas_ZHD_mm": off_saas,
            "official_profile_ZHD_mm": off_profile,
            "saas_recalculation_error_mm": saas_err,
            "profile_recalculation_error_mm": profile_err,
            "data_status": status,
            "warning_message": "",
        }
        for _, g in group.iterrows():
            pos = str(g["grid_position"])
            row[f"{pos}_latitude"] = g["grid_latitude"]
            row[f"{pos}_longitude"] = g["grid_longitude"]
            row[f"{pos}_weight"] = g["interpolation_weight"]
            row[f"{pos}_saas_p0_hPa"] = g["saas_bottom_pressure_hPa"]
            row[f"{pos}_profile_bottom_p_hPa"] = g["profile_bottom_pressure_hPa"]
            row[f"{pos}_saas_h0_geopotential_m"] = g["saas_bottom_geopotential_height_m"]
            row[f"{pos}_profile_bottom_h_geopotential_m"] = g["profile_bottom_geopotential_height_m"]
            row[f"{pos}_saas_h0_geometric_m"] = g["saas_bottom_geometric_height_m"]
            row[f"{pos}_profile_bottom_h_geometric_m"] = g["profile_bottom_geometric_height_m"]
        rows.append(row)
    return pd.DataFrame(rows)


def rmse(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.sqrt(np.mean(clean**2)))


def max_abs(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.abs().max())


def summarize_source(source: str, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in ["pre", "during", "post"]:
        sub = df[df["stage"] == stage].copy() if not df.empty else pd.DataFrame()
        ok = sub[sub["data_status"] == "ok"].copy() if "data_status" in sub.columns else pd.DataFrame()
        row = {
            "source": source,
            "stage": stage,
            "n_total": int(len(sub)),
            "n_ok": int(len(ok)),
            "n_invalid": int(len(sub) - len(ok)),
            "mean_bottom_pressure_difference_hPa": ok.get("bottom_pressure_difference_hPa", pd.Series(dtype=float)).mean(),
            "median_bottom_pressure_difference_hPa": ok.get("bottom_pressure_difference_hPa", pd.Series(dtype=float)).median(),
            "max_abs_bottom_pressure_difference_hPa": max_abs(ok.get("bottom_pressure_difference_hPa", pd.Series(dtype=float))),
            "mean_bottom_geometric_height_difference_m": ok.get("bottom_geometric_height_difference_m", pd.Series(dtype=float)).mean(),
            "max_abs_bottom_geometric_height_difference_m": max_abs(ok.get("bottom_geometric_height_difference_m", pd.Series(dtype=float))),
            "mean_pressure_difference_equivalent_ZHD_mm": ok.get("pressure_difference_equivalent_ZHD_mm", pd.Series(dtype=float)).mean(),
            "max_abs_pressure_difference_equivalent_ZHD_mm": max_abs(ok.get("pressure_difference_equivalent_ZHD_mm", pd.Series(dtype=float))),
            "mean_height_difference_equivalent_ZHD_mm": ok.get("height_difference_equivalent_ZHD_mm", pd.Series(dtype=float)).mean(),
            "max_abs_height_difference_equivalent_ZHD_mm": max_abs(ok.get("height_difference_equivalent_ZHD_mm", pd.Series(dtype=float))),
            "mean_profile_minus_saas_mm": ok.get("profile_minus_saas_mm", pd.Series(dtype=float)).mean(),
            "rmse_profile_minus_saas_mm": rmse(ok.get("profile_minus_saas_mm", pd.Series(dtype=float))),
            "max_abs_profile_minus_saas_mm": max_abs(ok.get("profile_minus_saas_mm", pd.Series(dtype=float))),
            "mean_residual_after_bottom_boundary_mm": ok.get("residual_after_bottom_boundary_mm", pd.Series(dtype=float)).mean(),
            "rmse_residual_after_bottom_boundary_mm": rmse(ok.get("residual_after_bottom_boundary_mm", pd.Series(dtype=float))),
            "mean_saas_interpolation_order_effect_mm": np.nan,
            "max_abs_saas_interpolation_order_effect_mm": np.nan,
            "mean_residual_after_bottom_and_interp_order_mm": np.nan,
            "rmse_residual_after_bottom_and_interp_order_mm": np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_station(source: str, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in ["pre", "during", "post"]:
        sub = df[df["stage"] == stage].copy() if not df.empty else pd.DataFrame()
        ok = sub[sub["data_status"] == "ok"].copy() if "data_status" in sub.columns else pd.DataFrame()
        row = {
            "source": source,
            "stage": stage,
            "n_total": int(len(sub)),
            "n_ok": int(len(ok)),
            "n_invalid": int(len(sub) - len(ok)),
            "mean_bottom_pressure_difference_hPa": ok.get("weighted_bottom_pressure_difference_hPa", pd.Series(dtype=float)).mean(),
            "median_bottom_pressure_difference_hPa": ok.get("weighted_bottom_pressure_difference_hPa", pd.Series(dtype=float)).median(),
            "max_abs_bottom_pressure_difference_hPa": max_abs(ok.get("weighted_bottom_pressure_difference_hPa", pd.Series(dtype=float))),
            "mean_bottom_geometric_height_difference_m": ok.get("weighted_bottom_height_difference_m", pd.Series(dtype=float)).mean(),
            "max_abs_bottom_geometric_height_difference_m": max_abs(ok.get("weighted_bottom_height_difference_m", pd.Series(dtype=float))),
            "mean_pressure_difference_equivalent_ZHD_mm": ok.get("weighted_pressure_equivalent_ZHD_mm", pd.Series(dtype=float)).mean(),
            "max_abs_pressure_difference_equivalent_ZHD_mm": max_abs(ok.get("weighted_pressure_equivalent_ZHD_mm", pd.Series(dtype=float))),
            "mean_height_difference_equivalent_ZHD_mm": ok.get("weighted_height_equivalent_ZHD_mm", pd.Series(dtype=float)).mean(),
            "max_abs_height_difference_equivalent_ZHD_mm": max_abs(ok.get("weighted_height_equivalent_ZHD_mm", pd.Series(dtype=float))),
            "mean_profile_minus_saas_mm": ok.get("profile_minus_saas_current_mm", pd.Series(dtype=float)).mean(),
            "rmse_profile_minus_saas_mm": rmse(ok.get("profile_minus_saas_current_mm", pd.Series(dtype=float))),
            "max_abs_profile_minus_saas_mm": max_abs(ok.get("profile_minus_saas_current_mm", pd.Series(dtype=float))),
            "mean_residual_after_bottom_boundary_mm": ok.get("residual_after_bottom_boundary_mm", pd.Series(dtype=float)).mean(),
            "rmse_residual_after_bottom_boundary_mm": rmse(ok.get("residual_after_bottom_boundary_mm", pd.Series(dtype=float))),
            "mean_saas_interpolation_order_effect_mm": ok.get("saas_interpolation_order_effect_mm", pd.Series(dtype=float)).mean(),
            "max_abs_saas_interpolation_order_effect_mm": max_abs(ok.get("saas_interpolation_order_effect_mm", pd.Series(dtype=float))),
            "mean_residual_after_bottom_and_interp_order_mm": ok.get("residual_after_bottom_and_interp_order_mm", pd.Series(dtype=float)).mean(),
            "rmse_residual_after_bottom_and_interp_order_mm": rmse(ok.get("residual_after_bottom_and_interp_order_mm", pd.Series(dtype=float))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def check_official_consistency(*frames: pd.DataFrame) -> None:
    mismatches: list[str] = []
    for frame in frames:
        if frame.empty or "data_status" not in frame.columns:
            continue
        bad = frame[frame["data_status"] == "official_output_mismatch"]
        if not bad.empty:
            cols = [
                col
                for col in [
                    "event_id",
                    "stage",
                    "sounding_time_utc",
                    "time_utc",
                    "grid_position",
                    "grid_latitude",
                    "grid_longitude",
                    "saas_recalculation_error_mm",
                    "profile_recalculation_error_mm",
                ]
                if col in bad.columns
            ]
            mismatches.append(bad[cols].head(20).to_string(index=False))
    if mismatches:
        raise RuntimeError(
            "Recalculated ZHD is inconsistent with official output beyond "
            f"{ZHD_TOL_MM} mm. Stop analysis.\n" + "\n".join(mismatches)
        )


def write_report(
    path: Path,
    event: dict[str, Any],
    summary: pd.DataFrame,
    sounding: pd.DataFrame,
    grid: pd.DataFrame,
    station: pd.DataFrame,
) -> None:
    ok_sounding = sounding[sounding["data_status"] == "ok"]
    ok_grid = grid[grid["data_status"] == "ok"]
    ok_station = station[station["data_status"] == "ok"]

    def verdict_bottom(df: pd.DataFrame, source: str) -> list[str]:
        lines: list[str] = []
        if df.empty:
            lines.append(f"{source}: no ok records.")
            return lines
        max_dp = max_abs(df["bottom_pressure_difference_hPa"])
        max_dh = max_abs(df["bottom_geometric_height_difference_m"])
        if max_dp <= 1e-10:
            lines.append(
                f"{source}: 在数值精度范围内，底层气压完全一致，最大绝对差异 {max_dp:.12g} hPa；底层气压不一致不是当前差异来源。"
            )
        else:
            lines.append(
                f"{source}: 底层气压不完全一致，最大绝对差异 {max_dp:.12g} hPa。"
            )
        if max_dh <= 1e-8:
            lines.append(
                f"{source}: 在数值精度范围内，底层高度完全一致，最大绝对差异 {max_dh:.12g} m。"
            )
        else:
            lines.append(
                f"{source}: 底层高度不完全一致，最大绝对差异 {max_dh:.12g} m。"
            )
        return lines

    nh_removed_sounding = int(
        ((sounding["initial_valid_level_count"] - sounding["profile_effective_level_count"]) > 0).sum()
    ) if not sounding.empty and "initial_valid_level_count" in sounding.columns else 0
    nh_removed_grid = int(
        ((grid["initial_valid_level_count"] - grid["profile_effective_level_count"]) > 0).sum()
    ) if not grid.empty and "initial_valid_level_count" in grid.columns else 0

    lines = [
        f"Step-1 bottom-boundary diagnostic report",
        f"event_id: {event['id']}",
        f"display_name: {event.get('display_name', event['id'])}",
        f"station_id: {event['station']['id']}",
        "",
        "Numerical answers:",
    ]
    lines.extend(verdict_bottom(ok_sounding, "sounding"))
    lines.extend(verdict_bottom(ok_grid, "era5_grid"))
    lines.append(
        f"Nh filtering removed the lowest/effective-layer candidates in {nh_removed_sounding} sounding records and {nh_removed_grid} ERA5 grid records."
    )
    lines.append(
        "Maximum pressure-equivalent ZHD: sounding={:.12g} mm, era5_grid={:.12g} mm, era5_station={:.12g} mm.".format(
            max_abs(ok_sounding.get("pressure_difference_equivalent_ZHD_mm", pd.Series(dtype=float))),
            max_abs(ok_grid.get("pressure_difference_equivalent_ZHD_mm", pd.Series(dtype=float))),
            max_abs(ok_station.get("weighted_pressure_equivalent_ZHD_mm", pd.Series(dtype=float))),
        )
    )
    lines.append(
        "Maximum height-equivalent ZHD: sounding={:.12g} mm, era5_grid={:.12g} mm, era5_station={:.12g} mm.".format(
            max_abs(ok_sounding.get("height_difference_equivalent_ZHD_mm", pd.Series(dtype=float))),
            max_abs(ok_grid.get("height_difference_equivalent_ZHD_mm", pd.Series(dtype=float))),
            max_abs(ok_station.get("weighted_height_equivalent_ZHD_mm", pd.Series(dtype=float))),
        )
    )
    lines.append(
        "Maximum ERA5 Saas interpolation-order effect: {:.12g} mm.".format(
            max_abs(ok_station.get("saas_interpolation_order_effect_mm", pd.Series(dtype=float)))
        )
    )
    lines.append(
        "ERA5 station residual RMSE after bottom-boundary and interpolation-order terms: {:.12g} mm.".format(
            rmse(ok_station.get("residual_after_bottom_and_interp_order_mm", pd.Series(dtype=float)))
        )
    )
    lines.append("")
    lines.append("Stage summary:")
    lines.append(summary.to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diagnostic(
    path_png: Path,
    path_pdf: Path,
    event: dict[str, Any],
    stages: list[StageDef],
    sounding: pd.DataFrame,
    station: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for ax in axes:
        for stage in stages:
            color = {"pre": "#dbeafe", "during": "#fee2e2", "post": "#dcfce7"}[stage.code]
            ax.axvspan(stage.start, stage.end, color=color, alpha=0.45, lw=0)
        ax.axhline(0.0, color="#334155", lw=0.8, alpha=0.8)
        ax.grid(True, color="#e2e8f0", lw=0.6)

    sound_ok = sounding[sounding["data_status"] == "ok"].copy()
    stat_ok = station[station["data_status"] == "ok"].copy()
    axes[0].plot(
        sound_ok["sounding_time_utc"],
        sound_ok["profile_minus_saas_mm"],
        linestyle="none",
        marker="o",
        color="#2563eb",
        label="sounding",
    )
    axes[0].plot(
        stat_ok["time_utc"],
        stat_ok["profile_minus_saas_current_mm"],
        color="#dc2626",
        lw=1.6,
        label="ERA5 station",
    )
    axes[1].plot(
        sound_ok["sounding_time_utc"],
        sound_ok["pressure_difference_equivalent_ZHD_mm"],
        linestyle="none",
        marker="o",
        color="#2563eb",
        label="sounding",
    )
    axes[1].plot(
        stat_ok["time_utc"],
        stat_ok["weighted_pressure_equivalent_ZHD_mm"],
        color="#dc2626",
        lw=1.6,
        label="ERA5 station",
    )
    axes[2].plot(
        sound_ok["sounding_time_utc"],
        sound_ok["height_difference_equivalent_ZHD_mm"],
        linestyle="none",
        marker="o",
        color="#2563eb",
        label="sounding",
    )
    axes[2].plot(
        stat_ok["time_utc"],
        stat_ok["weighted_height_equivalent_ZHD_mm"],
        color="#dc2626",
        lw=1.6,
        label="ERA5 station",
    )
    axes[3].plot(
        stat_ok["time_utc"],
        stat_ok["saas_interpolation_order_effect_mm"],
        color="#7c3aed",
        lw=1.6,
        label="ERA5 Saas order effect",
    )

    labels = [
        "(a) Profile ZHD - Saas ZHD (mm)",
        "(b) Pressure-difference equivalent ZHD (mm)",
        "(c) Height-difference equivalent ZHD (mm)",
        "(d) ERA5 Saas interpolation-order effect (mm)",
    ]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    axes[-1].set_xlabel("UTC time")
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle(
        f"Step-1 bottom-boundary diagnostic: {event.get('display_name', event['id'])} station {event['station']['id']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_terminal_summary(
    config_path: Path,
    event: dict[str, Any],
    paths: EventPaths,
    summary: pd.DataFrame,
    sounding: pd.DataFrame,
    grid: pd.DataFrame,
    station: pd.DataFrame,
    generated: list[Path],
) -> None:
    ok_sounding = sounding[sounding["data_status"] == "ok"]
    ok_grid = grid[grid["data_status"] == "ok"]
    ok_station = station[station["data_status"] == "ok"]
    stage_counts = {
        stage: {
            "sounding_ok": int((ok_sounding["stage"] == stage).sum()),
            "era5_grid_ok": int((ok_grid["stage"] == stage).sum()),
            "era5_station_ok": int((ok_station["stage"] == stage).sum()),
        }
        for stage in ["pre", "during", "post"]
    }
    print("Config file:", config_path)
    print(
        "Event:",
        event["id"],
        event.get("display_name", event["id"]),
        "station",
        event["station"]["id"],
    )
    print("Sounding input:", paths.sounding_input)
    print("ERA5 input:", paths.era5_input)
    print("Official outputs:")
    for path in [
        paths.sounding_saas,
        paths.sounding_profile,
        paths.era5_saas_grid4,
        paths.era5_profile_grid4,
        paths.era5_saas_station,
        paths.era5_profile_station,
    ]:
        print(" ", path)
    print("Stage ok counts:", stage_counts)
    print(
        "Max bottom pressure difference hPa:",
        max(
            max_abs(ok_sounding.get("bottom_pressure_difference_hPa", pd.Series(dtype=float))),
            max_abs(ok_grid.get("bottom_pressure_difference_hPa", pd.Series(dtype=float))),
            max_abs(ok_station.get("weighted_bottom_pressure_difference_hPa", pd.Series(dtype=float))),
        ),
    )
    print(
        "Max bottom geometric height difference m:",
        max(
            max_abs(ok_sounding.get("bottom_geometric_height_difference_m", pd.Series(dtype=float))),
            max_abs(ok_grid.get("bottom_geometric_height_difference_m", pd.Series(dtype=float))),
            max_abs(ok_station.get("weighted_bottom_height_difference_m", pd.Series(dtype=float))),
        ),
    )
    print(
        "Max Saas interpolation-order effect mm:",
        max_abs(ok_station.get("saas_interpolation_order_effect_mm", pd.Series(dtype=float))),
    )
    print(
        "Original Profile-Saas RMSE mm:",
        rmse(
            pd.concat(
                [
                    ok_sounding.get("profile_minus_saas_mm", pd.Series(dtype=float)),
                    ok_station.get("profile_minus_saas_current_mm", pd.Series(dtype=float)),
                ],
                ignore_index=True,
            )
        ),
    )
    print(
        "RMSE after bottom-boundary terms mm:",
        rmse(
            pd.concat(
                [
                    ok_sounding.get("residual_after_bottom_boundary_mm", pd.Series(dtype=float)),
                    ok_station.get("residual_after_bottom_boundary_mm", pd.Series(dtype=float)),
                ],
                ignore_index=True,
            )
        ),
    )
    print(
        "ERA5 RMSE after bottom-boundary and interpolation-order terms mm:",
        rmse(ok_station.get("residual_after_bottom_and_interp_order_mm", pd.Series(dtype=float))),
    )
    print("Generated files:")
    for path in generated:
        print(" ", path)


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    events = load_events(config_path)
    event_id = args.event.strip().lower()
    if event_id not in events:
        print(f"Event not found: {event_id}")
        print("Available event IDs:")
        print(", ".join(sorted(events)))
        raise SystemExit(1)

    event = events[event_id]
    paths = event_paths(event, config_path)
    stages = stages_from_event(event)
    paths.diagnostic_dir.mkdir(parents=True, exist_ok=True)

    sounding = diagnose_sounding(event, paths, stages)
    grid, interp_mode, weights = diagnose_era5_grid(event, paths, stages)
    station = station_from_grid(event, paths, stages, grid, interp_mode, weights)
    check_official_consistency(sounding, grid, station)

    sounding_csv = paths.diagnostic_dir / "sounding_bottom_boundary_detail.csv"
    grid_csv = paths.diagnostic_dir / "era5_grid_bottom_boundary_detail.csv"
    station_csv = paths.diagnostic_dir / "era5_station_interpolation_diagnostic.csv"
    summary_csv = paths.diagnostic_dir / "step1_bottom_boundary_summary.csv"
    report_txt = paths.diagnostic_dir / "step1_bottom_boundary_report.txt"
    png = paths.diagnostic_dir / "step1_bottom_boundary_diagnostic.png"
    pdf = paths.diagnostic_dir / "step1_bottom_boundary_diagnostic.pdf"

    sounding.to_csv(sounding_csv, index=False, encoding="utf-8-sig")
    grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    station.to_csv(station_csv, index=False, encoding="utf-8-sig")

    summary = pd.concat(
        [
            summarize_source("sounding", sounding),
            summarize_source("era5_grid", grid),
            summarize_station("era5_station_interp", station),
        ],
        ignore_index=True,
    )
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_report(report_txt, event, summary, sounding, grid, station)
    plot_diagnostic(png, pdf, event, stages, sounding, station)

    generated = [sounding_csv, grid_csv, station_csv, summary_csv, report_txt, png, pdf]
    print_terminal_summary(config_path, event, paths, summary, sounding, grid, station, generated)


if __name__ == "__main__":
    main()
