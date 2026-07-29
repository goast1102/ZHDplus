from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
CODE_DIR = ROOT_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import calc_zhd_era5  # noqa: E402
import zhd_workflow  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "diagnostics" / "step3_batch"
LAYER_SUM_TOL_MM = 1e-8
EXPECTED_SUM_TOL_MM = 1e-10
CLOSURE_TOL_MM = 1e-8
WEIGHT_SUM_TOL = 1e-12

PRESSURE_SURFACES = [850.0, 700.0, 500.0, 300.0, 200.0, 100.0, 50.0, 10.0, 1.0]
PRESSURE_SURFACE_LABELS = ["surface", "850", "700", "500", "300", "200", "100", "50", "10", "1"]
STANDARD_BINS = [
    ("SFC_850", None, 850.0),
    ("850_700", 850.0, 700.0),
    ("700_500", 700.0, 500.0),
    ("500_300", 500.0, 300.0),
    ("300_200", 300.0, 200.0),
    ("200_100", 200.0, 100.0),
    ("100_50", 100.0, 50.0),
    ("50_10", 50.0, 10.0),
    ("10_1", 10.0, 1.0),
]
STANDARD_BIN_NAMES = [b[0] for b in STANDARD_BINS]
STAGE_ORDER = ["pre", "during", "post"]
EVENT_OUTPUT_NAMES = {
    "station": "native_layer_station.csv",
    "bins": "pressure_bin_diagnostic.csv",
    "envelope": "pressure_surface_envelope.csv",
    "summary": "step3_event_summary.csv",
    "pdf": "step3_diagnostics.pdf",
    "grid_debug": "native_layer_grid_debug.csv",
}


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
    sounding_input: Path
    era5_input: Path
    era5_saas_grid4: Path
    era5_profile_grid4: Path
    era5_saas_station: Path
    era5_profile_station: Path


def slug_event_id(value: str) -> str:
    out = str(value).strip().lower().replace(" ", "_")
    out = re.sub(r"[^a-z0-9_-]+", "", out)
    return out or "event"


def event_prefix(event: dict[str, Any]) -> str:
    return slug_event_id(event["id"])


def event_files(output_root: Path, event: dict[str, Any]) -> dict[str, Path]:
    prefix = event_prefix(event)
    return {key: output_root / f"{prefix}_{name}" for key, name in EVENT_OUTPUT_NAMES.items()}


def formal_outputs_exist(files: dict[str, Path]) -> bool:
    return all(files[key].exists() for key in ["station", "bins", "envelope", "summary", "pdf"])


def summary_status_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if "record_type" in df:
        event_rows = df[df["record_type"].eq("event")]
        if event_rows.empty:
            return False
        return str(event_rows["data_status"].iloc[0]) == "ok"
    if "data_status" in df and not df.empty:
        return str(df["data_status"].iloc[0]) == "ok"
    return False


def should_skip_event(files: dict[str, Path], force: bool) -> bool:
    return (not force) and formal_outputs_exist(files) and summary_status_ok(files["summary"])


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_text()}] {message}\n")


def empty_event_record(event: dict[str, Any], data_status: str, warning_message: str) -> dict[str, Any]:
    station = event.get("station", {})
    return {
        "event_id": event_prefix(event),
        "display_name": event.get("display_name", event["id"]),
        "region_name": event.get("region_name", ""),
        "station_id": str(station.get("id", "")),
        "station_latitude": station.get("latitude", np.nan),
        "station_longitude": station.get("longitude", np.nan),
        "data_status": data_status,
        "warning_message": warning_message,
    }


def write_status_summary(path: Path, event: dict[str, Any], data_status: str, warning_message: str) -> pd.DataFrame:
    row = empty_event_record(event, data_status, warning_message)
    row["record_type"] = "event"
    df = pd.DataFrame([row])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    code_relative = CODE_DIR / path
    if code_relative.exists():
        return code_relative.resolve()
    return (ROOT_DIR / path).resolve()


def resolve_step3_config_path(path: Path) -> Path:
    resolved = resolve_config_path(path)
    if resolved.exists():
        return resolved
    if path.name == "typhoon_events_expanded.json":
        fallback = CODE_DIR / "typhoon_events.json"
        if fallback.exists():
            return fallback.resolve()
    return resolved


def configure_workflow_roots(data_root: Path | None, out_root: Path | None) -> None:
    if data_root is not None:
        zhd_workflow.DATA_ROOT = data_root.expanduser().resolve()
    if out_root is not None:
        zhd_workflow.OUT_ROOT = out_root.expanduser().resolve()


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


def stages_from_event(event: dict[str, Any]) -> list[StageDef]:
    plot = event["plot"]
    time_start = pd.Timestamp(plot["time_start"])
    pre_end = pd.Timestamp(plot["pre_end"])
    impact_end = pd.Timestamp(plot["impact_end"])
    time_end = pd.Timestamp(plot["time_end"])
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


def event_paths(event: dict[str, Any], config_path: Path) -> EventPaths:
    if config_path.name == "typhoon_events.json":
        zhd_workflow.DEFAULT_CONFIG = config_path
    out_dir = zhd_workflow.event_out_dir(event)
    return EventPaths(
        out_dir=out_dir,
        sounding_input=zhd_workflow.expected_sounding_qc_combined(event),
        era5_input=zhd_workflow.era5_data_dir(event) / "era5_pressure_raw_extract.csv",
        era5_saas_grid4=out_dir / zhd_workflow.ERA5_OUTPUTS["saas_grid4"],
        era5_profile_grid4=out_dir / zhd_workflow.ERA5_OUTPUTS["profile_grid4"],
        era5_saas_station=out_dir / zhd_workflow.ERA5_OUTPUTS["saas_station"],
        era5_profile_station=out_dir / zhd_workflow.ERA5_OUTPUTS["profile_station"],
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step-3 ERA5 layer hydrostatic-consistency residual diagnostic."
    )
    parser.add_argument("--config", type=Path, default=CODE_DIR / "typhoon_events.json")
    parser.add_argument("--event")
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root directory containing ERA5/radiosonde data, for example /home/hongy/ZHDplus/data.",
    )
    parser.add_argument(
        "--workflow-out-root",
        type=Path,
        help="Root directory containing official workflow outputs, for example /home/hongy/ZHDplus/out.",
    )
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-grid-debug", action="store_true")
    args = parser.parse_args()
    if bool(args.event) == bool(args.all_events):
        parser.error("Specify exactly one of --event or --all-events.")
    return args


def rmse(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.sqrt(np.mean(x**2))) if not x.empty else np.nan


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(data) < 5 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method=method))


def official(path: Path, keys: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time_utc"])
    df["official_ZHD_mm"] = pd.to_numeric(df["ZHD_m"], errors="coerce") * 1000.0
    return df[keys + ["official_ZHD_mm"]]


def official_value(df: pd.DataFrame, time_utc: pd.Timestamp, lat: float | None = None, lon: float | None = None) -> float:
    m = df["time_utc"].eq(time_utc)
    if lat is not None and "latitude" in df:
        m &= np.isclose(df["latitude"], lat, rtol=0.0, atol=1e-9)
    if lon is not None and "longitude" in df:
        m &= np.isclose(df["longitude"], lon, rtol=0.0, atol=1e-9)
    got = df[m]
    return float(got["official_ZHD_mm"].iloc[0]) if not got.empty else np.nan


def grid_weights_for_event(event: dict[str, Any], raw: pd.DataFrame) -> tuple[str, dict[tuple[float, float], tuple[str, float]]]:
    calc_zhd_era5.STATION_LAT = float(event["station"]["latitude"])
    calc_zhd_era5.STATION_LON = float(event["station"]["longitude"])
    lon_w, lon_e, lat_s, lat_n = calc_zhd_era5.find_surrounding_grid(raw)
    mode, weights = grid_weights(
        lon_w,
        lon_e,
        lat_s,
        lat_n,
        float(event["station"]["longitude"]),
        float(event["station"]["latitude"]),
    )
    if abs(sum(w for _, w in weights.values()) - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError("interpolation weights do not sum to 1")
    return mode, weights


def prepare_effective_profile(valid: pd.DataFrame, lat: float) -> tuple[pd.DataFrame, float, float, float, float, float]:
    profile, layer_integral_mm, top_correction_mm, profile_mm = era5_profile_components(valid, lat)
    bottom = valid.loc[valid[calc_zhd_era5.P_COL].idxmax()]
    p0 = float(bottom[calc_zhd_era5.P_COL])
    h0_geopot = float(bottom[calc_zhd_era5.H_COL])
    h0_geom = float(calc_zhd_era5.geometric_height(h0_geopot, lat))
    saas_mm = float(calc_zhd_era5.saastamoinen_zhd(p0, h0_geom, lat) * 1000.0)
    return profile, layer_integral_mm, top_correction_mm, profile_mm, saas_mm, h0_geom


def native_layers_for_grid(
    event: dict[str, Any],
    stages: list[StageDef],
    paths: EventPaths,
) -> tuple[pd.DataFrame, str]:
    raw = calc_zhd_era5.read_era5_csv(paths.era5_input)
    mode, weights = grid_weights_for_event(event, raw)
    lon_w, lon_e, lat_s, lat_n = calc_zhd_era5.find_surrounding_grid(raw)
    grid4 = calc_zhd_era5.select_grid4(raw, lon_w, lon_e, lat_s, lat_n)
    off_profile = official(paths.era5_profile_grid4, ["time_utc", "latitude", "longitude"])
    off_saas = official(paths.era5_saas_grid4, ["time_utc", "latitude", "longitude"])
    rows: list[dict[str, Any]] = []
    for (time_utc, lat, lon), group in grid4.groupby([calc_zhd_era5.TIME_COL, calc_zhd_era5.LAT_COL, calc_zhd_era5.LON_COL], sort=True):
        time_utc = pd.Timestamp(time_utc)
        lat = float(lat)
        lon = float(lon)
        pos, weight = weights.get((lat, lon), ("unused", 0.0))
        stage = assign_stage(time_utc, stages)
        valid = era5_initial_valid(group)
        status = "ok"
        warning = ""
        if valid.empty:
            status = "no_raw_rows"
        elif len(valid) < 2:
            status = "insufficient_levels"
        if status != "ok":
            rows.append({"event_id": event["id"], "time_utc": time_utc, "grid_position": pos, "data_status": status, "warning_message": warning})
            continue
        profile, layer_int_sum, top_corr, profile_mm, saas_surface_mm, h0_geom = prepare_effective_profile(valid, lat)
        if len(profile) < 2:
            rows.append({"event_id": event["id"], "time_utc": time_utc, "grid_position": pos, "data_status": "insufficient_levels", "warning_message": warning})
            continue
        profile = profile.sort_values(calc_zhd_era5.H_COL).reset_index(drop=True)
        p = profile[calc_zhd_era5.P_COL].to_numpy(float)
        h_geopot = profile[calc_zhd_era5.H_COL].to_numpy(float)
        z_geom = calc_zhd_era5.geometric_height(h_geopot, lat)
        nh = profile["Nh"].to_numpy(float)
        e = profile["e"].to_numpy(float)
        temp = profile[calc_zhd_era5.T_COL].to_numpy(float)
        geop = profile.get("geopotential_m2_s2", h_geopot * 9.80665).to_numpy(float)
        top_p = float(p[-1])
        expected_top = calc_zhd_era5.saastamoinen_zhd(top_p, h0_geom, lat) * 1000.0
        top_closure = top_corr - expected_top
        actual_layers: list[float] = []
        expected_layers: list[float] = []
        residuals: list[float] = []
        cumulative = 0.0
        layer_status = "ok"
        for i in range(len(profile) - 1):
            p_lower = float(p[i])
            p_upper = float(p[i + 1])
            z_lower = float(z_geom[i])
            z_upper = float(z_geom[i + 1])
            if not (p_lower > p_upper and z_upper > z_lower):
                layer_status = "invalid_layer_order"
            nh_mean = float(calc_zhd_era5.log_mean(np.array([nh[i]]), np.array([nh[i + 1]]))[0])
            actual = float((z_upper - z_lower) * nh_mean * 1e-6 * 1000.0)
            # calc_zhd_era5.saastamoinen_zhd returns meters, so convert the pressure-difference
            # allocation to millimeters after subtracting the two diagnostic Saas values.
            expected = float(
                (
                    calc_zhd_era5.saastamoinen_zhd(p_lower, h0_geom, lat)
                    - calc_zhd_era5.saastamoinen_zhd(p_upper, h0_geom, lat)
                )
                * 1000.0
            )
            resid = actual - expected
            cumulative += resid
            actual_layers.append(actual)
            expected_layers.append(expected)
            residuals.append(resid)
            rows.append(
                {
                    "event_id": event["id"],
                    "display_name": event.get("display_name", event["id"]),
                    "station_id": str(event["station"]["id"]),
                    "stage": stage,
                    "time_utc": time_utc,
                    "grid_position": pos,
                    "grid_latitude": lat,
                    "grid_longitude": lon,
                    "interpolation_weight": weight,
                    "native_layer_index": i,
                    "pressure_lower_hPa": p_lower,
                    "pressure_upper_hPa": p_upper,
                    "pressure_mid_log_hPa": float(np.sqrt(p_lower * p_upper)),
                    "pressure_difference_hPa": p_lower - p_upper,
                    "geopotential_lower_m2_s2": float(geop[i]),
                    "geopotential_upper_m2_s2": float(geop[i + 1]),
                    "geopotential_height_lower_m": float(h_geopot[i]),
                    "geopotential_height_upper_m": float(h_geopot[i + 1]),
                    "geopotential_thickness_m": float(h_geopot[i + 1] - h_geopot[i]),
                    "geometric_height_lower_m": z_lower,
                    "geometric_height_upper_m": z_upper,
                    "geometric_thickness_m": z_upper - z_lower,
                    "temperature_lower_K": float(temp[i]),
                    "temperature_upper_K": float(temp[i + 1]),
                    "vapor_pressure_lower_hPa": float(e[i]),
                    "vapor_pressure_upper_hPa": float(e[i + 1]),
                    "Nh_lower": float(nh[i]),
                    "Nh_upper": float(nh[i + 1]),
                    "Nh_layer_mean": nh_mean,
                    "actual_profile_layer_ZHD_mm": actual,
                    "expected_hydrostatic_layer_ZHD_mm": expected,
                    "layer_hydrostatic_residual_mm": resid,
                    "cumulative_residual_to_upper_level_mm": cumulative,
                    "surface_saas_ZHD_mm": saas_surface_mm,
                    "actual_top_correction_mm": top_corr,
                    "expected_top_from_surface_reference_mm": expected_top,
                    "top_closure_residual_mm": top_closure,
                    "official_profile_ZHD_mm": official_value(off_profile, time_utc, lat, lon),
                    "official_saas_ZHD_mm": official_value(off_saas, time_utc, lat, lon),
                    "profile_minus_saas_mm": official_value(off_profile, time_utc, lat, lon) - official_value(off_saas, time_utc, lat, lon),
                    "data_status": layer_status,
                    "warning_message": warning,
                }
            )
        actual_sum_err = sum(actual_layers) + top_corr - profile_mm
        expected_sum_err = sum(expected_layers) - (saas_surface_mm - expected_top)
        closure_error = profile_mm - saas_surface_mm - sum(residuals) - top_closure
        if abs(actual_sum_err) > LAYER_SUM_TOL_MM:
            status = "layer_sum_mismatch"
        elif abs(expected_sum_err) > EXPECTED_SUM_TOL_MM:
            status = "expected_sum_mismatch"
        elif abs(closure_error) > CLOSURE_TOL_MM:
            status = "closure_mismatch"
        else:
            status = layer_status
        for r in rows[-(len(profile) - 1) :]:
            r["actual_layer_sum_error_mm"] = actual_sum_err
            r["expected_layer_sum_error_mm"] = expected_sum_err
            r["full_closure_error_mm"] = closure_error
            r["final_cumulative_with_top_mm"] = cumulative + top_closure
            r["data_status"] = status
    return pd.DataFrame(rows), mode


def station_layers(grid_layers: pd.DataFrame, event: dict[str, Any], paths: EventPaths) -> pd.DataFrame:
    off_profile = official(paths.era5_profile_station, ["time_utc"])
    off_saas = official(paths.era5_saas_station, ["time_utc"])
    rows: list[dict[str, Any]] = []
    ok = grid_layers[grid_layers["data_status"].eq("ok")].copy()
    for (time_utc, layer_idx), group in ok.groupby(["time_utc", "native_layer_index"], sort=True):
        wsum = float(group["interpolation_weight"].sum())
        if abs(wsum - 1.0) > WEIGHT_SUM_TOL:
            rows.append({"event_id": event["id"], "time_utc": time_utc, "native_layer_index": layer_idx, "data_status": "interpolation_weight_error", "warning_message": f"weight_sum={wsum}"})
            continue

        def wavg(col: str) -> float:
            return float((group[col] * group["interpolation_weight"]).sum())

        station_actual = wavg("actual_profile_layer_ZHD_mm")
        station_expected = wavg("expected_hydrostatic_layer_ZHD_mm")
        station_resid = station_actual - station_expected
        top_closure = wavg("top_closure_residual_mm")
        profile_mm = official_value(off_profile, pd.Timestamp(time_utc))
        saas_mm = official_value(off_saas, pd.Timestamp(time_utc))
        rows.append(
            {
                "event_id": event["id"],
                "display_name": event.get("display_name", event["id"]),
                "station_id": str(event["station"]["id"]),
                "stage": group["stage"].iloc[0],
                "time_utc": pd.Timestamp(time_utc),
                "station_latitude": float(event["station"]["latitude"]),
                "station_longitude": float(event["station"]["longitude"]),
                "native_layer_index": int(layer_idx),
                "pressure_lower_hPa": wavg("pressure_lower_hPa"),
                "pressure_upper_hPa": wavg("pressure_upper_hPa"),
                "pressure_mid_log_hPa": wavg("pressure_mid_log_hPa"),
                "station_geopotential_height_lower_m": wavg("geopotential_height_lower_m"),
                "station_geopotential_height_upper_m": wavg("geopotential_height_upper_m"),
                "station_geopotential_thickness_m": wavg("geopotential_thickness_m"),
                "station_geometric_height_lower_m": wavg("geometric_height_lower_m"),
                "station_geometric_height_upper_m": wavg("geometric_height_upper_m"),
                "station_geometric_thickness_m": wavg("geometric_thickness_m"),
                "station_actual_layer_ZHD_mm": station_actual,
                "station_expected_layer_ZHD_mm": station_expected,
                "station_layer_residual_mm": station_resid,
                "station_actual_top_correction_mm": wavg("actual_top_correction_mm"),
                "station_expected_top_reference_mm": wavg("expected_top_from_surface_reference_mm"),
                "station_top_closure_residual_mm": top_closure,
                "official_station_profile_ZHD_mm": profile_mm,
                "official_station_saas_ZHD_mm": saas_mm,
                "station_profile_minus_saas_mm": profile_mm - saas_mm,
                "data_status": "ok",
                "warning_message": "",
                **{f"{p}_weight": float(group.loc[group["grid_position"].eq(p), "interpolation_weight"].iloc[0]) if (group["grid_position"].eq(p)).any() else 0.0 for p in ["SW", "SE", "NW", "NE"]},
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["station_cumulative_residual_mm"] = df.groupby("time_utc")["station_layer_residual_mm"].cumsum()
    final = df.groupby("time_utc").tail(1).set_index("time_utc")
    closure_map = final["station_profile_minus_saas_mm"] - final["station_cumulative_residual_mm"] - final["station_top_closure_residual_mm"]
    df["station_full_closure_error_mm"] = df["time_utc"].map(closure_map)
    df.loc[df["station_full_closure_error_mm"].abs() > CLOSURE_TOL_MM, "data_status"] = "closure_mismatch"
    return df


def pressure_bin_for_layer(p_lower: float, p_upper: float, bottom_p: float) -> tuple[str, float, float, str]:
    for name, lo, up in STANDARD_BINS:
        lower = bottom_p if lo is None else lo
        if abs(p_lower - lower) < 1e-9 and abs(p_upper - up) < 1e-9:
            status = "partial_bottom_bin" if lo is None and bottom_p < 850.0 else "ok"
            return name, p_lower, p_upper, status
    return "unbinned", p_lower, p_upper, "missing_pressure_surface"


def standard_bins(station: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok = station[station["data_status"].eq("ok")].copy()
    for time_utc, group in ok.groupby("time_utc", sort=True):
        bottom_p = float(group["pressure_lower_hPa"].max())
        total_profile = float(group["official_station_profile_ZHD_mm"].iloc[0])
        total_resid = float(group["station_profile_minus_saas_mm"].iloc[0])
        cumulative = 0.0
        for name, lo, up in STANDARD_BINS:
            lower = bottom_p if lo is None else lo
            sub = group[
                (group["pressure_lower_hPa"] <= lower + 1e-9)
                & (group["pressure_upper_hPa"] >= up - 1e-9)
            ].copy()
            if sub.empty:
                rows.append({"event_id": group["event_id"].iloc[0], "stage": group["stage"].iloc[0], "time_utc": time_utc, "pressure_bin": name, "data_status": "missing_pressure_surface", "warning_message": ""})
                continue
            actual = float(sub["station_actual_layer_ZHD_mm"].sum())
            expected = float(sub["station_expected_layer_ZHD_mm"].sum())
            resid = actual - expected
            cumulative += resid
            rows.append(
                {
                    "event_id": sub["event_id"].iloc[0],
                    "stage": sub["stage"].iloc[0],
                    "time_utc": time_utc,
                    "pressure_bin": name,
                    "actual_lower_pressure_hPa": lower,
                    "actual_upper_pressure_hPa": up,
                    "lower_geopotential_height_m": float(sub["station_geopotential_height_lower_m"].iloc[0]),
                    "upper_geopotential_height_m": float(sub["station_geopotential_height_upper_m"].iloc[-1]),
                    "geopotential_thickness_m": float(sub["station_geopotential_height_upper_m"].iloc[-1] - sub["station_geopotential_height_lower_m"].iloc[0]),
                    "lower_geometric_height_m": float(sub["station_geometric_height_lower_m"].iloc[0]),
                    "upper_geometric_height_m": float(sub["station_geometric_height_upper_m"].iloc[-1]),
                    "geometric_thickness_m": float(sub["station_geometric_height_upper_m"].iloc[-1] - sub["station_geometric_height_lower_m"].iloc[0]),
                    "native_layer_count_in_bin": int(len(sub)),
                    "actual_profile_bin_ZHD_mm": actual,
                    "expected_hydrostatic_bin_ZHD_mm": expected,
                    "bin_hydrostatic_residual_mm": resid,
                    "cumulative_residual_at_bin_top_mm": cumulative,
                    "bin_fraction_of_total_profile_ZHD_percent": actual / total_profile * 100.0 if abs(total_profile) > 1e-6 else np.nan,
                    "bin_fraction_of_total_residual_percent": resid / total_resid * 100.0 if abs(total_resid) > 1e-6 else np.nan,
                    "data_status": "ok",
                    "warning_message": "",
                }
            )
    return pd.DataFrame(rows)


def envelope_and_thickness(station: pd.DataFrame, grid: pd.DataFrame, event: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    ok = station[station["data_status"].eq("ok")].copy()
    grid_ok = grid[grid["data_status"].eq("ok")].copy()
    for time_utc, group in ok.groupby("time_utc", sort=True):
        stage = group["stage"].iloc[0]
        bottom = group.iloc[0]
        surfaces: dict[float | str, tuple[float, float]] = {
            "surface": (float(bottom["station_geopotential_height_lower_m"]), float(bottom["station_geometric_height_lower_m"]))
        }
        for p in PRESSURE_SURFACES:
            hit = group[np.isclose(group["pressure_upper_hPa"], p)]
            if hit.empty:
                rows.append({"event_id": event["id"], "stage": stage, "time_utc": time_utc, "pressure_surface_hPa": p, "data_status": "missing_pressure_surface", "warning_message": ""})
                continue
            surfaces[p] = (float(hit["station_geopotential_height_upper_m"].iloc[0]), float(hit["station_geometric_height_upper_m"].iloc[0]))
        grid_time = grid_ok[grid_ok["time_utc"].eq(time_utc)]
        for label in ["surface"] + PRESSURE_SURFACES:
            if label not in surfaces:
                continue
            if label == "surface":
                gsurf = grid_time[grid_time["native_layer_index"].eq(0)]
                vals = {
                    f"{pos}_geopotential_height_m": float(gsurf.loc[gsurf["grid_position"].eq(pos), "geopotential_height_lower_m"].iloc[0]) if (gsurf["grid_position"].eq(pos)).any() else np.nan
                    for pos in ["SW", "SE", "NW", "NE"]
                }
                pval = np.nan
            else:
                gsurf = grid_time[np.isclose(grid_time["pressure_upper_hPa"], float(label))]
                vals = {
                    f"{pos}_geopotential_height_m": float(gsurf.loc[gsurf["grid_position"].eq(pos), "geopotential_height_upper_m"].iloc[0]) if (gsurf["grid_position"].eq(pos)).any() else np.nan
                    for pos in ["SW", "SE", "NW", "NE"]
                }
                pval = float(label)
            v = np.array([x for x in vals.values() if pd.notna(x)])
            rows.append(
                {
                    "event_id": event["id"],
                    "stage": stage,
                    "time_utc": time_utc,
                    "pressure_surface_hPa": pval,
                    "pressure_surface_label": "surface" if label == "surface" else str(int(float(label))),
                    "station_geopotential_height_m": surfaces[label][0],
                    "station_geometric_height_m": surfaces[label][1],
                    **vals,
                    "grid_min_geopotential_height_m": float(v.min()) if len(v) else np.nan,
                    "grid_max_geopotential_height_m": float(v.max()) if len(v) else np.nan,
                    "grid_spread_geopotential_height_m": float(v.max() - v.min()) if len(v) else np.nan,
                    "interpolation_weight_sum": 1.0,
                    "data_status": "ok",
                    "warning_message": "",
                }
            )
    env = pd.DataFrame(rows)
    thick = []
    for time_utc, eg in env[env["data_status"].eq("ok")].groupby("time_utc", sort=True):
        stage = eg["stage"].iloc[0]
        by_label = eg.set_index("pressure_surface_label")
        for name, lo, up in STANDARD_BINS:
            lower_label = "surface" if lo is None else str(int(lo))
            upper_label = str(int(up))
            if lower_label not in by_label.index or upper_label not in by_label.index:
                continue
            lower = by_label.loc[lower_label]
            upper = by_label.loc[upper_label]
            thick.append(
                {
                    "event_id": event["id"],
                    "stage": stage,
                    "time_utc": pd.Timestamp(time_utc),
                    "pressure_bin": name,
                    "pressure_lower_hPa": np.nan if lo is None else lo,
                    "pressure_upper_hPa": up,
                    "lower_geopotential_height_m": lower["station_geopotential_height_m"],
                    "upper_geopotential_height_m": upper["station_geopotential_height_m"],
                    "geopotential_thickness_m": upper["station_geopotential_height_m"] - lower["station_geopotential_height_m"],
                    "lower_geometric_height_m": lower["station_geometric_height_m"],
                    "upper_geometric_height_m": upper["station_geometric_height_m"],
                    "geometric_thickness_m": upper["station_geometric_height_m"] - lower["station_geometric_height_m"],
                }
            )
    th = pd.DataFrame(thick)
    if not th.empty:
        refs = th[th["stage"].eq("pre")].groupby("pressure_bin")["geopotential_thickness_m"].median()
        th["pre_reference_geopotential_thickness_m"] = th["pressure_bin"].map(refs)
        th["geopotential_thickness_anomaly_m"] = th["geopotential_thickness_m"] - th["pre_reference_geopotential_thickness_m"]
    return env, th


def crossing_summary(station: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok = station[station["data_status"].eq("ok")]
    bin_ok = bins[bins["data_status"].eq("ok")]
    for time_utc, group in ok.groupby("time_utc", sort=True):
        row: dict[str, Any] = {"time_utc": pd.Timestamp(time_utc), "stage": group["stage"].iloc[0]}
        for threshold, label in [(0.1, "0p1"), (0.25, "0p25"), (0.5, "0p5")]:
            hit = group[group["station_cumulative_residual_mm"].abs() >= threshold].head(1)
            row[f"first_crossing_{label}_pressure_hPa"] = float(hit["pressure_upper_hPa"].iloc[0]) if not hit.empty else np.nan
            row[f"first_crossing_{label}_geopotential_height_m"] = float(hit["station_geopotential_height_upper_m"].iloc[0]) if not hit.empty else np.nan
        bg = bin_ok[bin_ok["time_utc"].eq(time_utc)]
        if not bg.empty:
            idx_abs = bg["bin_hydrostatic_residual_mm"].abs().idxmax()
            idx_neg = bg["bin_hydrostatic_residual_mm"].idxmin()
            idx_pos = bg["bin_hydrostatic_residual_mm"].idxmax()
            row["largest_abs_layer_residual_bin"] = bg.loc[idx_abs, "pressure_bin"]
            row["largest_abs_layer_residual_mm"] = bg.loc[idx_abs, "bin_hydrostatic_residual_mm"]
            row["largest_incremental_negative_residual_bin"] = bg.loc[idx_neg, "pressure_bin"]
            row["largest_incremental_negative_residual_mm"] = bg.loc[idx_neg, "bin_hydrostatic_residual_mm"]
            row["largest_incremental_positive_residual_bin"] = bg.loc[idx_pos, "pressure_bin"]
            row["largest_incremental_positive_residual_mm"] = bg.loc[idx_pos, "bin_hydrostatic_residual_mm"]
        rows.append(row)
    return pd.DataFrame(rows)


def summary_bins(bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = bins[bins["data_status"].eq("ok")].copy()
    for pressure_bin in [b[0] for b in STANDARD_BINS]:
        for stage in ["pre", "during", "post"]:
            sub = bins[(bins["pressure_bin"].eq(pressure_bin)) & (bins["stage"].eq(stage))]
            gok = ok[(ok["pressure_bin"].eq(pressure_bin)) & (ok["stage"].eq(stage))]
            row = {
                "pressure_bin": pressure_bin,
                "stage": stage,
                "n_total": len(sub),
                "n_ok": len(gok),
                "n_invalid": len(sub) - len(gok),
                "mean_geopotential_thickness_m": gok["geopotential_thickness_m"].mean(),
                "std_geopotential_thickness_m": gok["geopotential_thickness_m"].std(),
                "min_geopotential_thickness_m": gok["geopotential_thickness_m"].min(),
                "max_geopotential_thickness_m": gok["geopotential_thickness_m"].max(),
                "mean_geopotential_thickness_anomaly_m": gok.get("geopotential_thickness_anomaly_m", pd.Series(dtype=float)).mean(),
                "rmse_geopotential_thickness_anomaly_m": rmse(gok.get("geopotential_thickness_anomaly_m", pd.Series(dtype=float))),
                "max_abs_geopotential_thickness_anomaly_m": gok.get("geopotential_thickness_anomaly_m", pd.Series(dtype=float)).abs().max(),
                "mean_actual_profile_bin_ZHD_mm": gok["actual_profile_bin_ZHD_mm"].mean(),
                "std_actual_profile_bin_ZHD_mm": gok["actual_profile_bin_ZHD_mm"].std(),
                "mean_expected_hydrostatic_bin_ZHD_mm": gok["expected_hydrostatic_bin_ZHD_mm"].mean(),
                "std_expected_hydrostatic_bin_ZHD_mm": gok["expected_hydrostatic_bin_ZHD_mm"].std(),
                "mean_bin_hydrostatic_residual_mm": gok["bin_hydrostatic_residual_mm"].mean(),
                "median_bin_hydrostatic_residual_mm": gok["bin_hydrostatic_residual_mm"].median(),
                "std_bin_hydrostatic_residual_mm": gok["bin_hydrostatic_residual_mm"].std(),
                "rmse_bin_hydrostatic_residual_mm": rmse(gok["bin_hydrostatic_residual_mm"]),
                "max_abs_bin_hydrostatic_residual_mm": gok["bin_hydrostatic_residual_mm"].abs().max(),
                "mean_cumulative_residual_at_bin_top_mm": gok["cumulative_residual_at_bin_top_mm"].mean(),
                "rmse_cumulative_residual_at_bin_top_mm": rmse(gok["cumulative_residual_at_bin_top_mm"]),
                "max_abs_cumulative_residual_at_bin_top_mm": gok["cumulative_residual_at_bin_top_mm"].abs().max(),
                "pearson_r_residual_vs_geopotential_thickness": corr(gok["bin_hydrostatic_residual_mm"], gok["geopotential_thickness_m"], "pearson"),
                "spearman_r_residual_vs_geopotential_thickness": corr(gok["bin_hydrostatic_residual_mm"], gok["geopotential_thickness_m"], "spearman"),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    means = out.pivot(index="pressure_bin", columns="stage")
    for pb in out["pressure_bin"].unique():
        for prefix, a, b in [("during_minus_pre", "during", "pre"), ("during_minus_post", "during", "post")]:
            mask = out["pressure_bin"].eq(pb)
            try:
                out.loc[mask, f"{prefix}_actual_ZHD_mm"] = means.loc[pb, ("mean_actual_profile_bin_ZHD_mm", a)] - means.loc[pb, ("mean_actual_profile_bin_ZHD_mm", b)]
                out.loc[mask, f"{prefix}_expected_ZHD_mm"] = means.loc[pb, ("mean_expected_hydrostatic_bin_ZHD_mm", a)] - means.loc[pb, ("mean_expected_hydrostatic_bin_ZHD_mm", b)]
                out.loc[mask, f"{prefix}_residual_mm"] = means.loc[pb, ("mean_bin_hydrostatic_residual_mm", a)] - means.loc[pb, ("mean_bin_hydrostatic_residual_mm", b)]
                out.loc[mask, f"{prefix}_geopotential_thickness_m"] = means.loc[pb, ("mean_geopotential_thickness_m", a)] - means.loc[pb, ("mean_geopotential_thickness_m", b)]
            except KeyError:
                pass
        mask = out["pressure_bin"].eq(pb)
        try:
            for metric, out_name in [
                ("mean_actual_profile_bin_ZHD_mm", "during_minus_reference_actual_ZHD_mm"),
                ("mean_expected_hydrostatic_bin_ZHD_mm", "during_minus_reference_expected_ZHD_mm"),
                ("mean_bin_hydrostatic_residual_mm", "during_minus_reference_residual_mm"),
                ("mean_geopotential_thickness_m", "during_minus_reference_geopotential_thickness_m"),
            ]:
                out.loc[mask, out_name] = (
                    means.loc[pb, (metric, "during")]
                    - 0.5 * (means.loc[pb, (metric, "pre")] + means.loc[pb, (metric, "post")])
                )
        except KeyError:
            pass
    return out


def add_event_metadata(df: pd.DataFrame, event: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["event_id"] = event_prefix(event)
    out["display_name"] = event.get("display_name", event["id"])
    out["station_id"] = str(event["station"]["id"])
    if "station_latitude" not in out:
        out["station_latitude"] = float(event["station"]["latitude"])
    if "station_longitude" not in out:
        out["station_longitude"] = float(event["station"]["longitude"])
    return out


def final_by_time(station: pd.DataFrame) -> pd.DataFrame:
    if station.empty:
        return station.copy()
    return station.sort_values(["time_utc", "native_layer_index"]).groupby("time_utc").tail(1).copy()


def stage_count(final: pd.DataFrame, stage: str) -> int:
    return int(final[final["stage"].eq(stage)]["time_utc"].nunique()) if not final.empty else 0


def stage_mean(final: pd.DataFrame, stage: str, col: str) -> float:
    sub = final[final["stage"].eq(stage)]
    return float(sub[col].mean()) if not sub.empty else np.nan


def stage_rmse(final: pd.DataFrame, stage: str, col: str) -> float:
    sub = final[final["stage"].eq(stage)]
    return rmse(sub[col]) if not sub.empty else np.nan


def get_summary_value(summary: pd.DataFrame, pressure_bin: str, stage: str, col: str) -> float:
    sub = summary[summary["pressure_bin"].eq(pressure_bin) & summary["stage"].eq(stage)]
    if sub.empty or col not in sub:
        return np.nan
    return float(sub[col].iloc[0])


def select_largest(summary: pd.DataFrame, stage: str, value_col: str, mode: str = "abs") -> tuple[str, float]:
    sub = summary[summary["stage"].eq(stage)].dropna(subset=[value_col]).copy()
    if sub.empty:
        return "", np.nan
    values = sub[value_col]
    if mode == "min":
        idx = values.idxmin()
    elif mode == "max":
        idx = values.idxmax()
    else:
        idx = values.abs().idxmax()
    return str(sub.loc[idx, "pressure_bin"]), float(sub.loc[idx, value_col])


def event_summary_record(event: dict[str, Any], station: pd.DataFrame, summary: pd.DataFrame, data_status: str, warning_message: str) -> dict[str, Any]:
    final = final_by_time(station)
    row = empty_event_record(event, data_status, warning_message)
    row["record_type"] = "event"
    for stage in STAGE_ORDER:
        row[f"n_{stage}"] = stage_count(final, stage)
        row[f"mean_profile_minus_saas_{stage}_mm"] = stage_mean(final, stage, "station_profile_minus_saas_mm")
        row[f"rmse_profile_minus_saas_{stage}_mm"] = stage_rmse(final, stage, "station_profile_minus_saas_mm")
    row["during_minus_pre_mean_difference_mm"] = row["mean_profile_minus_saas_during_mm"] - row["mean_profile_minus_saas_pre_mm"]
    row["during_minus_post_mean_difference_mm"] = row["mean_profile_minus_saas_during_mm"] - row["mean_profile_minus_saas_post_mm"]
    row["during_minus_reference_mean_difference_mm"] = row["mean_profile_minus_saas_during_mm"] - 0.5 * (
        row["mean_profile_minus_saas_pre_mm"] + row["mean_profile_minus_saas_post_mm"]
    )
    row["rmse_during_minus_pre_mm"] = row["rmse_profile_minus_saas_during_mm"] - row["rmse_profile_minus_saas_pre_mm"]
    row["rmse_during_minus_post_mm"] = row["rmse_profile_minus_saas_during_mm"] - row["rmse_profile_minus_saas_post_mm"]
    row["rmse_during_minus_reference_mm"] = row["rmse_profile_minus_saas_during_mm"] - 0.5 * (
        row["rmse_profile_minus_saas_pre_mm"] + row["rmse_profile_minus_saas_post_mm"]
    )
    row["largest_abs_during_residual_bin"], row["largest_abs_during_residual_mm"] = select_largest(summary, "during", "mean_bin_hydrostatic_residual_mm")
    row["largest_during_minus_pre_residual_bin"], row["largest_during_minus_pre_residual_mm"] = select_largest(summary, "during", "during_minus_pre_residual_mm")
    row["largest_during_minus_reference_residual_bin"], row["largest_during_minus_reference_residual_mm"] = select_largest(summary, "during", "during_minus_reference_residual_mm")
    row["largest_during_minus_reference_negative_bin"], row["largest_during_minus_reference_negative_mm"] = select_largest(summary, "during", "during_minus_reference_residual_mm", "min")
    row["largest_during_minus_reference_positive_bin"], row["largest_during_minus_reference_positive_mm"] = select_largest(summary, "during", "during_minus_reference_residual_mm", "max")
    row["largest_geopotential_thickness_change_bin"], row["largest_geopotential_thickness_change_m"] = select_largest(summary, "during", "during_minus_reference_geopotential_thickness_m")
    row["largest_thickness_change_bin"] = row["largest_geopotential_thickness_change_bin"]
    row["largest_thickness_change_m"] = row["largest_geopotential_thickness_change_m"]
    for pressure in [100.0, 50.0, 10.0, 1.0]:
        sub = station[np.isclose(station.get("pressure_upper_hPa", pd.Series(dtype=float)), pressure)] if not station.empty else pd.DataFrame()
        for stage in STAGE_ORDER:
            val = sub[sub["stage"].eq(stage)]["station_cumulative_residual_mm"].mean() if not sub.empty else np.nan
            row[f"mean_cumulative_residual_at_{int(pressure)}hPa_{stage}_mm"] = float(val) if pd.notna(val) else np.nan
    row["max_abs_layer_sum_error_mm"] = float(station.get("actual_layer_sum_error_mm", pd.Series(dtype=float)).abs().max()) if "actual_layer_sum_error_mm" in station else np.nan
    row["max_abs_expected_sum_error_mm"] = float(station.get("expected_layer_sum_error_mm", pd.Series(dtype=float)).abs().max()) if "expected_layer_sum_error_mm" in station else np.nan
    row["max_abs_full_closure_error_mm"] = float(station["station_full_closure_error_mm"].abs().max()) if "station_full_closure_error_mm" in station and not station.empty else np.nan
    diff = row["rmse_during_minus_reference_mm"]
    if pd.isna(diff):
        row["enhancement_class"] = "unknown"
    elif diff > 0.5:
        row["enhancement_class"] = "strong_enhancement"
    elif diff > 0.1:
        row["enhancement_class"] = "moderate_enhancement"
    elif diff < -0.1:
        row["enhancement_class"] = "reverse"
    else:
        row["enhancement_class"] = "stable"
    return row


def pressure_bin_stage_records(event: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, src in summary.iterrows():
        row = empty_event_record(event, "ok", "")
        row.update(
            {
                "record_type": "pressure_bin_stage",
                "stage": src["stage"],
                "pressure_bin": src["pressure_bin"],
                "n_valid": src.get("n_ok", np.nan),
                "mean_actual_profile_bin_ZHD_mm": src.get("mean_actual_profile_bin_ZHD_mm", np.nan),
                "mean_expected_hydrostatic_bin_ZHD_mm": src.get("mean_expected_hydrostatic_bin_ZHD_mm", np.nan),
                "mean_bin_residual_mm": src.get("mean_bin_hydrostatic_residual_mm", np.nan),
                "median_bin_residual_mm": src.get("median_bin_hydrostatic_residual_mm", np.nan),
                "rmse_bin_residual_mm": src.get("rmse_bin_hydrostatic_residual_mm", np.nan),
                "max_abs_bin_residual_mm": src.get("max_abs_bin_hydrostatic_residual_mm", np.nan),
                "mean_geopotential_thickness_m": src.get("mean_geopotential_thickness_m", np.nan),
                "mean_geopotential_thickness_anomaly_m": src.get("mean_geopotential_thickness_anomaly_m", np.nan),
                "mean_cumulative_residual_at_bin_top_mm": src.get("mean_cumulative_residual_at_bin_top_mm", np.nan),
                "during_minus_pre_residual_mm": src.get("during_minus_pre_residual_mm", np.nan),
                "during_minus_post_residual_mm": src.get("during_minus_post_residual_mm", np.nan),
                "during_minus_reference_residual_mm": src.get("during_minus_reference_residual_mm", np.nan),
                "during_minus_reference_actual_ZHD_mm": src.get("during_minus_reference_actual_ZHD_mm", np.nan),
                "during_minus_reference_expected_ZHD_mm": src.get("during_minus_reference_expected_ZHD_mm", np.nan),
                "during_minus_pre_geopotential_thickness_m": src.get("during_minus_pre_geopotential_thickness_m", np.nan),
                "during_minus_post_geopotential_thickness_m": src.get("during_minus_post_geopotential_thickness_m", np.nan),
                "during_minus_reference_geopotential_thickness_m": src.get("during_minus_reference_geopotential_thickness_m", np.nan),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_summary(event: dict[str, Any], station: pd.DataFrame, summary: pd.DataFrame, data_status: str, warning_message: str) -> pd.DataFrame:
    event_row = event_summary_record(event, station, summary, data_status, warning_message)
    bin_rows = pressure_bin_stage_records(event, summary)
    return pd.concat([pd.DataFrame([event_row]), bin_rows], ignore_index=True, sort=False)


def format_station_output(station: pd.DataFrame, event: dict[str, Any]) -> pd.DataFrame:
    out = add_event_metadata(station, event)
    cols = [
        "event_id", "display_name", "station_id", "stage", "time_utc",
        "station_latitude", "station_longitude", "native_layer_index",
        "pressure_lower_hPa", "pressure_upper_hPa", "pressure_mid_log_hPa",
        "station_geopotential_height_lower_m", "station_geopotential_height_upper_m", "station_geopotential_thickness_m",
        "station_geometric_height_lower_m", "station_geometric_height_upper_m", "station_geometric_thickness_m",
        "station_actual_layer_ZHD_mm", "station_expected_layer_ZHD_mm", "station_layer_residual_mm", "station_cumulative_residual_mm",
        "station_actual_top_correction_mm", "station_expected_top_reference_mm", "station_top_closure_residual_mm",
        "official_station_profile_ZHD_mm", "official_station_saas_ZHD_mm", "station_profile_minus_saas_mm",
        "station_full_closure_error_mm", "data_status", "warning_message",
    ]
    return out[[c for c in cols if c in out.columns]]


def format_pressure_bin_output(bins: pd.DataFrame, event: dict[str, Any], station: pd.DataFrame) -> pd.DataFrame:
    out = add_event_metadata(bins, event)
    if "actual_lower_pressure_hPa" in out:
        out["pressure_lower_hPa"] = out["actual_lower_pressure_hPa"]
    if "actual_upper_pressure_hPa" in out:
        out["pressure_upper_hPa"] = out["actual_upper_pressure_hPa"]
    final = final_by_time(station).set_index("time_utc") if not station.empty else pd.DataFrame()
    if not final.empty:
        out["profile_minus_saas_mm"] = out["time_utc"].map(final["station_profile_minus_saas_mm"])
        out["full_closure_error_mm"] = out["time_utc"].map(final["station_full_closure_error_mm"])
    cols = [
        "event_id", "display_name", "station_id", "stage", "time_utc", "pressure_bin",
        "pressure_lower_hPa", "pressure_upper_hPa", "lower_geopotential_height_m", "upper_geopotential_height_m",
        "geopotential_thickness_m", "pre_reference_geopotential_thickness_m", "geopotential_thickness_anomaly_m",
        "lower_geometric_height_m", "upper_geometric_height_m", "geometric_thickness_m",
        "native_layer_count_in_bin", "actual_profile_bin_ZHD_mm", "expected_hydrostatic_bin_ZHD_mm",
        "bin_hydrostatic_residual_mm", "cumulative_residual_at_bin_top_mm",
        "bin_fraction_of_total_profile_ZHD_percent", "bin_fraction_of_total_residual_percent",
        "profile_minus_saas_mm", "full_closure_error_mm", "data_status", "warning_message",
    ]
    return out[[c for c in cols if c in out.columns]]


def format_envelope_output(env: pd.DataFrame, event: dict[str, Any], save_grid_debug: bool) -> pd.DataFrame:
    out = add_event_metadata(env, event)
    cols = [
        "event_id", "display_name", "station_id", "stage", "time_utc", "pressure_surface_hPa",
        "pressure_surface_label", "station_geopotential_height_m", "station_geometric_height_m",
        "grid_min_geopotential_height_m", "grid_max_geopotential_height_m", "grid_spread_geopotential_height_m",
        "data_status", "warning_message",
    ]
    if save_grid_debug:
        cols.extend([f"{p}_geopotential_height_m" for p in ["SW", "SE", "NW", "NE"]])
    return out[[c for c in cols if c in out.columns]]


def add_stage_background(ax: plt.Axes, stages: list[StageDef]) -> None:
    colors = {"pre": "#dbeafe", "during": "#fee2e2", "post": "#dcfce7"}
    for st in stages:
        ax.axvspan(st.start, st.end, color=colors[st.code], alpha=0.35, lw=0)


def plot_envelope(env: pd.DataFrame, thick: pd.DataFrame, stages: list[StageDef], event: dict[str, Any], png: Path, pdf: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    add_stage_background(ax, stages)
    good = env[env["data_status"].eq("ok")].copy()
    labels = PRESSURE_SURFACE_LABELS
    pivot = good.pivot(index="time_utc", columns="pressure_surface_label", values="station_geopotential_height_m")
    fill_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(STANDARD_BINS)))
    for color, (name, lo, up) in zip(fill_colors, STANDARD_BINS):
        lower_label = "surface" if lo is None else str(int(lo))
        upper_label = str(int(up))
        if lower_label in pivot and upper_label in pivot:
            ax.fill_between(pivot.index, pivot[lower_label] / 1000, pivot[upper_label] / 1000, color=color, alpha=0.16)
    for label in labels:
        if label in pivot:
            ax.plot(pivot.index, pivot[label] / 1000, lw=1.2, label=label)
            ax.text(pivot.index.max(), pivot[label].iloc[-1] / 1000, label, fontsize=8, va="center")
    ax.set_ylabel("Geopotential height (km)")
    ax.set_title("(a) Fixed-pressure geopotential-height envelope", loc="left")
    ax.grid(True, color="#e2e8f0")
    ax2 = axes[1]
    add_stage_background(ax2, stages)
    for name in [b[0] for b in STANDARD_BINS]:
        sub = thick[thick["pressure_bin"].eq(name)]
        ax2.plot(sub["time_utc"], sub["geopotential_thickness_anomaly_m"], lw=1.1, label=name)
    ax2.axhline(0, color="#475569", lw=0.8)
    ax2.set_ylabel("Thickness anomaly (m)")
    ax2.set_title("(b) Fixed-pressure-layer geopotential thickness anomaly", loc="left")
    ax2.legend(ncol=3, fontsize=8, loc="upper left")
    ax2.grid(True, color="#e2e8f0")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle(f"Step 3 pressure envelope: {event.get('display_name', event['id'])} station {event['station']['id']}", fontweight="bold")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_contributions(summary: pd.DataFrame, event: dict[str, Any], png: Path, pdf: Path) -> None:
    stages = ["pre", "during", "post"]
    bins = [b[0] for b in STANDARD_BINS]
    y = np.arange(len(bins))
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True, constrained_layout=True)
    for stage in stages:
        sub = summary[summary["stage"].eq(stage)].set_index("pressure_bin").reindex(bins)
        axes[0].plot(sub["mean_actual_profile_bin_ZHD_mm"], y, marker="o", label=stage)
        axes[1].plot(sub["mean_expected_hydrostatic_bin_ZHD_mm"], y, marker="o", label=stage)
        axes[2].plot(sub["mean_bin_hydrostatic_residual_mm"], y, marker="o", label=stage)
    for ax, title in zip(axes, ["Actual profile contribution", "Hydrostatic expected contribution", "Actual - expected residual"]):
        ax.set_title(title)
        ax.grid(True, color="#e2e8f0")
        ax.legend()
    axes[2].axvline(0, color="#475569", lw=0.8)
    axes[0].set_yticks(y, bins)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ZHD contribution (mm)")
    axes[1].set_xlabel("ZHD contribution (mm)")
    axes[2].set_xlabel("Hydrostatic consistency residual (mm)")
    fig.suptitle(f"Step 3 layer contribution comparison: {event.get('display_name', event['id'])}", fontweight="bold")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative(station: pd.DataFrame, stages_def: list[StageDef], event: dict[str, Any], png: Path, pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for stage in ["pre", "during", "post"]:
        sub = station[station["stage"].eq(stage)]
        prof = sub.groupby("native_layer_index").agg(
            pressure=("pressure_upper_hPa", "mean"),
            height=("station_geopotential_height_upper_m", "mean"),
            cum=("station_cumulative_residual_mm", "mean"),
        )
        axes[0].plot(prof["cum"], prof["pressure"], marker="o", label=stage)
        axes[1].plot(prof["cum"], prof["height"] / 1000, marker="o", label=stage)
    axes[0].set_yscale("log")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#475569", lw=0.8)
    axes[0].set_xlabel("Cumulative residual (mm)")
    axes[0].set_ylabel("Pressure (hPa)")
    axes[1].axvline(0, color="#475569", lw=0.8)
    for x in [0.1, 0.25, 0.5, -0.1, -0.25, -0.5]:
        axes[0].axvline(x, color="#94a3b8", lw=0.5, ls=":")
        axes[1].axvline(x, color="#94a3b8", lw=0.5, ls=":")
    axes[1].set_xlabel("Cumulative residual (mm)")
    axes[1].set_ylabel("Geopotential height (km)")
    for ax in axes:
        ax.grid(True, color="#e2e8f0")
        ax.legend()
    fig.suptitle(f"Step 3 cumulative residual profiles: {event.get('display_name', event['id'])}", fontweight="bold")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_pressure(station: pd.DataFrame, stages: list[StageDef], event: dict[str, Any], png: Path, pdf: Path) -> None:
    piv = station.pivot(index="pressure_upper_hPa", columns="time_utc", values="station_cumulative_residual_mm").sort_index(ascending=False)
    vals = piv.to_numpy(float)
    vmax = np.nanquantile(np.abs(vals), 0.98)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    mesh = ax.pcolormesh(piv.columns, piv.index, vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_yscale("log")
    ax.invert_yaxis()
    for p in PRESSURE_SURFACES:
        ax.axhline(p, color="#334155", lw=0.5, alpha=0.5)
    for st in stages:
        ax.axvline(st.start, color="#111827", lw=0.8)
        ax.axvline(st.end, color="#111827", lw=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(f"Step 3 time-pressure cumulative residual: {event.get('display_name', event['id'])}", fontweight="bold")
    fig.colorbar(mesh, ax=ax, label="Cumulative hydrostatic consistency residual (mm)")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_closure(station: pd.DataFrame, event: dict[str, Any], png: Path, pdf: Path) -> None:
    final = station.groupby("time_utc").tail(1).copy()
    reconstructed = final["station_cumulative_residual_mm"] + final["station_top_closure_residual_mm"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(final["time_utc"], final["station_profile_minus_saas_mm"], label="official Profile-Saas", lw=2)
    axes[0].plot(final["time_utc"], reconstructed, "--", label="sum residuals + top closure")
    axes[1].plot(final["time_utc"], final["station_full_closure_error_mm"], label="closure error")
    axes[1].axhline(1e-8, color="#dc2626", ls=":", lw=0.8)
    axes[1].axhline(-1e-8, color="#dc2626", ls=":", lw=0.8)
    for ax in axes:
        ax.grid(True, color="#e2e8f0")
        ax.legend()
    axes[0].set_ylabel("mm")
    axes[1].set_ylabel("mm")
    axes[1].set_xlabel("UTC time")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle(f"Step 3 closure validation: {event.get('display_name', event['id'])}", fontweight="bold")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_envelope_figure(env: pd.DataFrame, thick: pd.DataFrame, stages: list[StageDef], event: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    add_stage_background(ax, stages)
    good = env[env["data_status"].eq("ok")].copy()
    if not good.empty:
        pivot = good.pivot(index="time_utc", columns="pressure_surface_label", values="station_geopotential_height_m")
        fill_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(STANDARD_BINS)))
        for color, (name, lo, up) in zip(fill_colors, STANDARD_BINS):
            lower_label = "surface" if lo is None else str(int(lo))
            upper_label = str(int(up))
            if lower_label in pivot and upper_label in pivot:
                ax.fill_between(pivot.index, pivot[lower_label] / 1000, pivot[upper_label] / 1000, color=color, alpha=0.16)
        for label in PRESSURE_SURFACE_LABELS:
            if label in pivot:
                ax.plot(pivot.index, pivot[label] / 1000, lw=1.2, label=label)
                ax.text(pivot.index.max(), pivot[label].iloc[-1] / 1000, label, fontsize=8, va="center")
    ax.set_ylabel("Geopotential height (km)")
    ax.set_title("(a) Fixed-pressure geopotential-height envelope", loc="left")
    ax.grid(True, color="#e2e8f0")
    ax2 = axes[1]
    add_stage_background(ax2, stages)
    for name in STANDARD_BIN_NAMES:
        sub = thick[thick["pressure_bin"].eq(name)]
        if not sub.empty:
            ax2.plot(sub["time_utc"], sub["geopotential_thickness_anomaly_m"], lw=1.1, label=name)
    ax2.axhline(0, color="#475569", lw=0.8)
    ax2.set_ylabel("Thickness anomaly (m)")
    ax2.set_title("(b) Fixed-pressure-layer geopotential thickness anomaly", loc="left")
    ax2.legend(ncol=3, fontsize=8, loc="upper left")
    ax2.grid(True, color="#e2e8f0")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.suptitle(f"Step 3 pressure envelope: {event.get('display_name', event['id'])} station {event['station']['id']}", fontweight="bold")
    return fig


def build_contribution_figure(summary: pd.DataFrame, event: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True, constrained_layout=True)
    y = np.arange(len(STANDARD_BIN_NAMES))
    for stage in STAGE_ORDER:
        sub = summary[summary["stage"].eq(stage)].set_index("pressure_bin").reindex(STANDARD_BIN_NAMES)
        axes[0].plot(sub["mean_actual_profile_bin_ZHD_mm"], y, marker="o", label=stage)
        axes[1].plot(sub["mean_expected_hydrostatic_bin_ZHD_mm"], y, marker="o", label=stage)
        axes[2].plot(sub["mean_bin_hydrostatic_residual_mm"], y, marker="o", label=stage)
    for ax, title in zip(axes, ["Actual profile contribution", "Hydrostatic expected contribution", "Actual - expected residual"]):
        ax.set_title(title)
        ax.grid(True, color="#e2e8f0")
        ax.legend()
    axes[2].axvline(0, color="#475569", lw=0.8)
    axes[0].set_yticks(y, STANDARD_BIN_NAMES)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ZHD contribution (mm)")
    axes[1].set_xlabel("ZHD contribution (mm)")
    axes[2].set_xlabel("Hydrostatic consistency residual (mm)")
    fig.suptitle(f"Step 3 layer contribution comparison: {event.get('display_name', event['id'])}", fontweight="bold")
    return fig


def build_cumulative_figure(station: pd.DataFrame, event: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for stage in STAGE_ORDER:
        sub = station[station["stage"].eq(stage)]
        if sub.empty:
            continue
        prof = sub.groupby("native_layer_index").agg(
            pressure=("pressure_upper_hPa", "mean"),
            height=("station_geopotential_height_upper_m", "mean"),
            cum=("station_cumulative_residual_mm", "mean"),
        )
        axes[0].plot(prof["cum"], prof["pressure"], marker="o", label=stage)
        axes[1].plot(prof["cum"], prof["height"] / 1000, marker="o", label=stage)
    axes[0].set_yscale("log")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Cumulative residual (mm)")
    axes[0].set_ylabel("Pressure (hPa)")
    axes[1].set_xlabel("Cumulative residual (mm)")
    axes[1].set_ylabel("Geopotential height (km)")
    for ax in axes:
        ax.axvline(0, color="#475569", lw=0.8)
        for x in [0.1, 0.25, 0.5, -0.1, -0.25, -0.5]:
            ax.axvline(x, color="#94a3b8", lw=0.5, ls=":")
        ax.grid(True, color="#e2e8f0")
        ax.legend()
    fig.suptitle(f"Step 3 cumulative residual profiles: {event.get('display_name', event['id'])}", fontweight="bold")
    return fig


def build_anomaly_figure(station: pd.DataFrame, bins: pd.DataFrame, stages: list[StageDef], event: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

    def datetime_edges(values: pd.Index) -> np.ndarray:
        nums = mdates.date2num(pd.to_datetime(values).to_pydatetime())
        if len(nums) == 1:
            return np.array([nums[0] - 0.5 / 24, nums[0] + 0.5 / 24])
        mids = (nums[:-1] + nums[1:]) / 2.0
        return np.r_[nums[0] - (mids[0] - nums[0]), mids, nums[-1] + (nums[-1] - mids[-1])]

    def pressure_edges(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        if len(vals) == 1:
            return np.array([vals[0] * 1.1, vals[0] * 0.9])
        mids = (vals[:-1] + vals[1:]) / 2.0
        first = vals[0] + (vals[0] - mids[0])
        last = max(vals[-1] - (mids[-1] - vals[-1]), 0.1)
        return np.r_[first, mids, last]

    b = bins[bins["data_status"].eq("ok")].copy()
    if not b.empty:
        pre_med = b[b["stage"].eq("pre")].groupby("pressure_bin")["bin_hydrostatic_residual_mm"].median()
        b["layer_residual_anomaly_mm"] = b["bin_hydrostatic_residual_mm"] - b["pressure_bin"].map(pre_med)
        piv = b.pivot(index="pressure_bin", columns="time_utc", values="layer_residual_anomaly_mm").reindex(STANDARD_BIN_NAMES)
        vals = piv.to_numpy(float)
        vmax = np.nanquantile(np.abs(vals), 0.98) if np.isfinite(vals).any() else 1.0
        mesh = axes[0].pcolormesh(datetime_edges(piv.columns), np.arange(len(piv.index) + 1), vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_yticks(np.arange(len(piv.index)) + 0.5, piv.index)
        axes[0].invert_yaxis()
        fig.colorbar(mesh, ax=axes[0], label="Layer residual anomaly (mm)")
    s = station[station["data_status"].eq("ok")].copy()
    if not s.empty:
        pre_med = s[s["stage"].eq("pre")].groupby("pressure_upper_hPa")["station_cumulative_residual_mm"].median()
        s["cumulative_residual_anomaly_mm"] = s["station_cumulative_residual_mm"] - s["pressure_upper_hPa"].map(pre_med)
        piv = s.pivot(index="pressure_upper_hPa", columns="time_utc", values="cumulative_residual_anomaly_mm").sort_index(ascending=False)
        vals = piv.to_numpy(float)
        vmax = np.nanquantile(np.abs(vals), 0.98) if np.isfinite(vals).any() else 1.0
        mesh = axes[1].pcolormesh(datetime_edges(piv.columns), pressure_edges(piv.index.to_numpy(float)), vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_yscale("log")
        axes[1].invert_yaxis()
        for p in PRESSURE_SURFACES:
            axes[1].axhline(p, color="#334155", lw=0.5, alpha=0.5)
        fig.colorbar(mesh, ax=axes[1], label="Cumulative residual anomaly (mm)")
    for ax, title in zip(axes, ["Layer residual anomaly relative to pre median", "Cumulative residual anomaly relative to pre median"]):
        for st in stages:
            ax.axvline(st.start, color="#111827", lw=0.8)
            ax.axvline(st.end, color="#111827", lw=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_title(title, loc="left")
        ax.set_ylabel("Pressure")
    fig.suptitle(f"Typhoon-related residual anomaly: {event.get('display_name', event['id'])}", fontweight="bold")
    return fig


def build_event_summary_figure(event_summary: pd.DataFrame, event: dict[str, Any]) -> plt.Figure:
    row = event_summary[event_summary["record_type"].eq("event")].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    ax.axis("off")
    lines = [
        f"Event: {row.get('display_name', event['id'])} ({row.get('event_id', event['id'])})",
        f"Station: {row.get('station_id', '')}    Region: {row.get('region_name', '')}",
        f"Data status: {row.get('data_status', '')}",
        "",
        "Profile - Saas stage means and RMSE (mm)",
        f"pre:    mean={row.get('mean_profile_minus_saas_pre_mm', np.nan):.6g}  rmse={row.get('rmse_profile_minus_saas_pre_mm', np.nan):.6g}  n={row.get('n_pre', 0)}",
        f"during: mean={row.get('mean_profile_minus_saas_during_mm', np.nan):.6g}  rmse={row.get('rmse_profile_minus_saas_during_mm', np.nan):.6g}  n={row.get('n_during', 0)}",
        f"post:   mean={row.get('mean_profile_minus_saas_post_mm', np.nan):.6g}  rmse={row.get('rmse_profile_minus_saas_post_mm', np.nan):.6g}  n={row.get('n_post', 0)}",
        "",
        f"during - pre mean:       {row.get('during_minus_pre_mean_difference_mm', np.nan):.6g} mm",
        f"during - post mean:      {row.get('during_minus_post_mean_difference_mm', np.nan):.6g} mm",
        f"during - reference mean: {row.get('during_minus_reference_mean_difference_mm', np.nan):.6g} mm",
        "",
        f"Largest during residual bin: {row.get('largest_abs_during_residual_bin', '')} = {row.get('largest_abs_during_residual_mm', np.nan):.6g} mm",
        f"Largest added negative residual bin: {row.get('largest_during_minus_reference_negative_bin', '')} = {row.get('largest_during_minus_reference_negative_mm', np.nan):.6g} mm",
        f"Largest positive compensation bin: {row.get('largest_during_minus_reference_positive_bin', '')} = {row.get('largest_during_minus_reference_positive_mm', np.nan):.6g} mm",
        f"Largest thickness change bin: {row.get('largest_geopotential_thickness_change_bin', '')} = {row.get('largest_geopotential_thickness_change_m', np.nan):.6g} m",
        "",
        "Mean cumulative residual at fixed pressure levels (mm)",
    ]
    for pressure in [100, 50, 10, 1]:
        lines.append(
            f"{pressure} hPa: pre={row.get(f'mean_cumulative_residual_at_{pressure}hPa_pre_mm', np.nan):.6g}, "
            f"during={row.get(f'mean_cumulative_residual_at_{pressure}hPa_during_mm', np.nan):.6g}, "
            f"post={row.get(f'mean_cumulative_residual_at_{pressure}hPa_post_mm', np.nan):.6g}"
        )
    lines.extend(
        [
            "",
            f"Max actual layer-sum error: {row.get('max_abs_layer_sum_error_mm', np.nan):.6g} mm",
            f"Max expected sum error:     {row.get('max_abs_expected_sum_error_mm', np.nan):.6g} mm",
            f"Max full closure error:     {row.get('max_abs_full_closure_error_mm', np.nan):.6g} mm",
            "This diagnostic verifies hydrostatic consistency and numerical closure; it is not a nonhydrostatic correction.",
        ]
    )
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10)
    fig.suptitle("Event summary", fontweight="bold")
    return fig


def write_event_pdf(path: Path, env: pd.DataFrame, thick: pd.DataFrame, summary: pd.DataFrame, station: pd.DataFrame, bins: pd.DataFrame, stages: list[StageDef], event: dict[str, Any], event_summary: pd.DataFrame) -> None:
    with PdfPages(path) as pdf:
        for fig in [
            build_envelope_figure(env, thick, stages, event),
            build_contribution_figure(summary, event),
            build_cumulative_figure(station, event),
            build_anomaly_figure(station, bins, stages, event),
            build_event_summary_figure(event_summary, event),
        ]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_report(path: Path, event: dict[str, Any], station: pd.DataFrame, bins: pd.DataFrame, thick: pd.DataFrame, summary: pd.DataFrame, crossing: pd.DataFrame) -> None:
    final = station.groupby("time_utc").tail(1)
    max_closure = station["station_full_closure_error_mm"].abs().max()
    during = summary[summary["stage"].eq("during")].copy()
    max_bin = during.loc[during["mean_bin_hydrostatic_residual_mm"].abs().idxmax()]
    max_change = summary.drop_duplicates("pressure_bin").loc[summary.drop_duplicates("pressure_bin")["during_minus_pre_residual_mm"].abs().idxmax()]
    thick_change = summary.drop_duplicates("pressure_bin").loc[summary.drop_duplicates("pressure_bin")["during_minus_pre_geopotential_thickness_m"].abs().idxmax()]
    lines = [
        "Step-3 ERA5 layer hydrostatic-consistency residual report",
        f"event_id: {event['id']}",
        f"display_name: {event.get('display_name', event['id'])}",
        f"station_id: {event['station']['id']}",
        "",
        "This is a hydrostatic consistency and numerical closure diagnostic. It is not a nonhydrostatic correction.",
        f"Effective ERA5 levels: {int(station['native_layer_index'].max()+2)} levels, {int(station['native_layer_index'].max()+1)} native adjacent layers.",
        f"Maximum full closure error: {max_closure:.12g} mm.",
        f"Largest during mean bin residual: {max_bin['pressure_bin']} = {max_bin['mean_bin_hydrostatic_residual_mm']:.6g} mm.",
        f"Largest during-pre residual change: {max_change['pressure_bin']} = {max_change['during_minus_pre_residual_mm']:.6g} mm.",
        f"Largest during-pre geopotential-thickness change: {thick_change['pressure_bin']} = {thick_change['during_minus_pre_geopotential_thickness_m']:.6g} m.",
        "",
        "Final cumulative residuals at selected pressure levels (stage means):",
    ]
    for p in [100.0, 50.0, 10.0, 1.0]:
        sub = station[np.isclose(station["pressure_upper_hPa"], p)]
        vals = sub.groupby("stage")["station_cumulative_residual_mm"].mean()
        lines.append(f"{p:g} hPa: " + ", ".join(f"{k}={v:.6g} mm" for k, v in vals.items()))
    lines.append("")
    lines.append("First crossings:")
    lines.append(crossing.to_string(index=False))
    lines.append("")
    lines.append("Summary:")
    lines.append(summary.to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_input_files(paths: EventPaths) -> tuple[str, str]:
    if not paths.era5_input.exists():
        return "missing_era5_file", str(paths.era5_input.resolve())
    official_files = [
        paths.era5_saas_grid4,
        paths.era5_profile_grid4,
        paths.era5_saas_station,
        paths.era5_profile_station,
    ]
    missing = [str(p.resolve()) for p in official_files if not p.exists()]
    if missing:
        return "missing_official_output", "; ".join(missing)
    return "ok", ""


def output_summary_rows(summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(summary_path)
    event_rows = df[df["record_type"].eq("event")].copy() if "record_type" in df else df.head(1).copy()
    bin_rows = df[df["record_type"].eq("pressure_bin_stage")].copy() if "record_type" in df else pd.DataFrame()
    return event_rows, bin_rows


def infer_event_status(event: dict[str, Any], station: pd.DataFrame, grid: pd.DataFrame) -> tuple[str, str]:
    final = final_by_time(station)
    for stage in STAGE_ORDER:
        if stage_count(final, stage) == 0:
            return f"insufficient_{stage}", f"No valid ERA5 station records in {stage} stage."
    if "station_profile_minus_saas_mm" in final and final["station_profile_minus_saas_mm"].isna().any():
        return "official_output_mismatch", "Official station Profile/Saas output could not be matched for one or more times."
    for status in ["invalid_layer_order", "interpolation_weight_error", "layer_sum_mismatch", "expected_sum_mismatch", "closure_mismatch"]:
        if (("data_status" in grid and grid["data_status"].eq(status).any()) or ("data_status" in station and station["data_status"].eq(status).any())):
            return status, f"{status} detected for {event_prefix(event)}."
    max_closure = station["station_full_closure_error_mm"].abs().max() if "station_full_closure_error_mm" in station and not station.empty else np.nan
    if pd.notna(max_closure) and max_closure > CLOSURE_TOL_MM:
        return "closure_mismatch", f"max_abs_full_closure_error_mm={max_closure:.12g}"
    return "ok", ""


def run_event(
    event: dict[str, Any],
    config_path: Path,
    output_root: Path,
    force: bool,
    save_grid_debug: bool,
    log_path: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    files = event_files(output_root, event)
    prefix = event_prefix(event)
    if should_skip_event(files, force):
        append_log(log_path, f"{prefix}: skipped existing ok outputs")
        event_rows, bin_rows = output_summary_rows(files["summary"])
        return {"event_id": prefix, "status": "skipped", "data_status": "ok", "event_rows": event_rows, "bin_rows": bin_rows, "files": files, "seconds": 0.0}

    try:
        paths = event_paths(event, config_path)
        input_status, input_warning = validate_input_files(paths)
        if input_status != "ok":
            summary_df = write_status_summary(files["summary"], event, input_status, input_warning)
            append_log(log_path, f"{prefix}: {input_status}: {input_warning}")
            return {"event_id": prefix, "status": "failed", "data_status": input_status, "event_rows": summary_df, "bin_rows": pd.DataFrame(), "files": files, "seconds": time.perf_counter() - start}

        stages = stages_from_event(event)
        grid, mode = native_layers_for_grid(event, stages, paths)
        station = station_layers(grid, event, paths)
        bins = standard_bins(station)
        env, thick = envelope_and_thickness(station, grid, event)
        if not thick.empty and not bins.empty:
            merge_cols = [
                "time_utc",
                "pressure_bin",
                "pre_reference_geopotential_thickness_m",
                "geopotential_thickness_anomaly_m",
            ]
            bins = bins.merge(thick[[c for c in merge_cols if c in thick.columns]], on=["time_utc", "pressure_bin"], how="left")
        summary = summary_bins(bins)
        data_status, warning_message = infer_event_status(event, station, grid)
        event_summary = build_event_summary(event, station, summary, data_status, warning_message)
        max_actual = grid["actual_layer_sum_error_mm"].abs().max() if "actual_layer_sum_error_mm" in grid else np.nan
        max_expected = grid["expected_layer_sum_error_mm"].abs().max() if "expected_layer_sum_error_mm" in grid else np.nan
        event_summary.loc[event_summary["record_type"].eq("event"), "max_abs_layer_sum_error_mm"] = max_actual
        event_summary.loc[event_summary["record_type"].eq("event"), "max_abs_expected_sum_error_mm"] = max_expected

        format_station_output(station, event).to_csv(files["station"], index=False, encoding="utf-8-sig")
        format_pressure_bin_output(bins, event, station).to_csv(files["bins"], index=False, encoding="utf-8-sig")
        format_envelope_output(env, event, save_grid_debug).to_csv(files["envelope"], index=False, encoding="utf-8-sig")
        event_summary.to_csv(files["summary"], index=False, encoding="utf-8-sig")
        if save_grid_debug:
            grid.to_csv(files["grid_debug"], index=False, encoding="utf-8-sig")
        write_event_pdf(files["pdf"], env, thick, summary, station, bins, stages, event, event_summary)

        elapsed = time.perf_counter() - start
        append_log(log_path, f"{prefix}: {data_status} in {elapsed:.2f}s; outputs={[str(p.resolve()) for k, p in files.items() if k != 'grid_debug' or save_grid_debug]}")
        event_rows, bin_rows = output_summary_rows(files["summary"])
        return {"event_id": prefix, "status": "success" if data_status == "ok" else "failed", "data_status": data_status, "event_rows": event_rows, "bin_rows": bin_rows, "files": files, "seconds": elapsed}
    except Exception as exc:
        warning = traceback.format_exc()
        summary_df = write_status_summary(files["summary"], event, "runtime_error", warning)
        append_log(log_path, f"{prefix}: runtime_error: {exc}\n{warning}")
        return {"event_id": prefix, "status": "failed", "data_status": "runtime_error", "event_rows": summary_df, "bin_rows": pd.DataFrame(), "files": files, "seconds": time.perf_counter() - start}


def all_events_event_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    out = event_rows.copy()
    rename = {
        "mean_profile_minus_saas_pre_mm": "pre_mean_difference_mm",
        "mean_profile_minus_saas_during_mm": "during_mean_difference_mm",
        "mean_profile_minus_saas_post_mm": "post_mean_difference_mm",
        "rmse_profile_minus_saas_pre_mm": "pre_rmse_mm",
        "rmse_profile_minus_saas_during_mm": "during_rmse_mm",
        "rmse_profile_minus_saas_post_mm": "post_rmse_mm",
        "during_minus_pre_mean_difference_mm": "during_minus_pre_mean_mm",
        "during_minus_post_mean_difference_mm": "during_minus_post_mean_mm",
        "during_minus_reference_mean_difference_mm": "during_minus_reference_mean_mm",
        "rmse_during_minus_pre_mm": "during_minus_pre_rmse_mm",
        "rmse_during_minus_post_mm": "during_minus_post_rmse_mm",
        "rmse_during_minus_reference_mm": "during_minus_reference_rmse_mm",
    }
    out = out.rename(columns=rename)
    cols = [
        "event_id", "display_name", "region_name", "station_id", "station_latitude", "station_longitude",
        "n_pre", "n_during", "n_post", "data_status", "warning_message",
        "pre_mean_difference_mm", "during_mean_difference_mm", "post_mean_difference_mm",
        "pre_rmse_mm", "during_rmse_mm", "post_rmse_mm",
        "during_minus_pre_mean_mm", "during_minus_post_mean_mm", "during_minus_reference_mean_mm",
        "during_minus_pre_rmse_mm", "during_minus_post_rmse_mm", "during_minus_reference_rmse_mm",
        "largest_during_minus_reference_negative_bin", "largest_during_minus_reference_negative_mm",
        "largest_during_minus_reference_positive_bin", "largest_during_minus_reference_positive_mm",
        "largest_thickness_change_bin", "largest_thickness_change_m",
        "max_abs_full_closure_error_mm", "enhancement_class",
    ]
    return out[[c for c in cols if c in out.columns]]


def all_events_pressure_bin_change(bin_rows: pd.DataFrame) -> pd.DataFrame:
    if bin_rows.empty:
        return pd.DataFrame()
    out = bin_rows[bin_rows["stage"].eq("during")].copy()
    cols = [
        "event_id", "display_name", "region_name", "station_id", "pressure_bin",
        "during_minus_pre_residual_mm", "during_minus_post_residual_mm", "during_minus_reference_residual_mm",
        "during_minus_pre_geopotential_thickness_m", "during_minus_post_geopotential_thickness_m", "during_minus_reference_geopotential_thickness_m",
    ]
    if "during_minus_reference_actual_ZHD_mm" not in out:
        out["during_minus_reference_actual_ZHD_mm"] = np.nan
    if "during_minus_reference_expected_ZHD_mm" not in out:
        out["during_minus_reference_expected_ZHD_mm"] = np.nan
    cols.extend(["during_minus_reference_actual_ZHD_mm", "during_minus_reference_expected_ZHD_mm"])
    return out[[c for c in cols if c in out.columns]]


def write_batch_report(path: Path, results: list[dict[str, Any]], event_summary: pd.DataFrame, bin_change: pd.DataFrame, config_count: int, enabled_count: int, total_seconds: float) -> None:
    ok_events = event_summary[event_summary.get("data_status", "").eq("ok")] if not event_summary.empty else pd.DataFrame()
    lines = [
        "Step-3 batch report",
        f"start/end generated: {now_text()}",
        f"configured events: {config_count}",
        f"enabled events: {enabled_count}",
        f"successful events: {len(ok_events)}",
        f"skipped events: {sum(1 for r in results if r['status'] == 'skipped')}",
        f"missing-file events: {sum(1 for r in results if str(r['data_status']).startswith('missing_'))}",
        f"failed events: {sum(1 for r in results if r['data_status'] not in ['ok', 'disabled'])}",
        "",
        "enhancement classes:",
    ]
    if "enhancement_class" in event_summary:
        for cls, n in event_summary["enhancement_class"].fillna("unknown").value_counts().items():
            lines.append(f"  {cls}: {n}")
    if not bin_change.empty:
        lines.append("")
        lines.append("Median during-reference residual change by pressure bin (mm):")
        med = bin_change.groupby("pressure_bin")["during_minus_reference_residual_mm"].median().reindex(STANDARD_BIN_NAMES)
        for pb, val in med.items():
            lines.append(f"  {pb}: {val:.6g}")
        lines.append("")
        lines.append("Sign consistency by pressure bin:")
        for pb, sub in bin_change.groupby("pressure_bin"):
            vals = pd.to_numeric(sub["during_minus_reference_residual_mm"], errors="coerce").dropna()
            if vals.empty:
                continue
            neg = float((vals < 0).mean())
            pos = float((vals > 0).mean())
            lines.append(f"  {pb}: negative={neg:.3f}, positive={pos:.3f}")
        lines.append("")
        neg_bins = bin_change.loc[bin_change.groupby("event_id")["during_minus_reference_residual_mm"].idxmin(), "pressure_bin"].value_counts()
        pos_bins = bin_change.loc[bin_change.groupby("event_id")["during_minus_reference_residual_mm"].idxmax(), "pressure_bin"].value_counts()
        lines.append(f"Most frequent largest negative bin: {neg_bins.idxmax() if not neg_bins.empty else ''}")
        lines.append(f"Most frequent largest positive bin: {pos_bins.idxmax() if not pos_bins.empty else ''}")
    lines.extend(
        [
            "",
            f"Max closure error across ok events: {event_summary.get('max_abs_full_closure_error_mm', pd.Series(dtype=float)).max():.12g} mm",
            "Closure tolerance: abs(full_closure_error_mm) <= 1e-8 mm",
            "This diagnostic verifies hydrostatic consistency and numerical closure; it must not be interpreted as a nonhydrostatic correction.",
            f"total runtime seconds: {total_seconds:.2f}",
            f"mean seconds per processed event: {np.mean([r['seconds'] for r in results]) if results else np.nan:.2f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overview_pdf(path: Path, event_summary: pd.DataFrame, bin_change: pd.DataFrame, output_root: Path) -> None:
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        ok = event_summary[event_summary["data_status"].eq("ok")].copy() if not event_summary.empty else pd.DataFrame()
        if not ok.empty and "during_minus_reference_mean_mm" in ok:
            ok = ok.sort_values("during_minus_reference_mean_mm")
            ax.barh(ok["event_id"], ok["during_minus_reference_mean_mm"], color="#2563eb")
            ax.axvline(0, color="#475569", lw=0.8)
            ax.set_xlabel("during - mean(pre, post) Profile-Saas (mm)")
        ax.set_title("Event ranking by typhoon-stage Profile-Saas change")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, max(6, 0.25 * max(len(ok), 1))), constrained_layout=True)
        if not bin_change.empty:
            piv = bin_change.pivot(index="event_id", columns="pressure_bin", values="during_minus_reference_residual_mm").reindex(columns=STANDARD_BIN_NAMES)
            vals = piv.to_numpy(float)
            vmax = np.nanquantile(np.abs(vals), 0.98) if np.isfinite(vals).any() else 1.0
            mesh = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(piv.columns)), piv.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(piv.index)), piv.index)
            fig.colorbar(mesh, ax=ax, label="Residual change (mm)")
        ax.set_title("Event x pressure-bin during-reference residual change")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        if not bin_change.empty:
            data = [pd.to_numeric(bin_change[bin_change["pressure_bin"].eq(pb)]["during_minus_reference_residual_mm"], errors="coerce").dropna() for pb in STANDARD_BIN_NAMES]
            ax.boxplot(data, labels=STANDARD_BIN_NAMES, showfliers=False)
            ax.axhline(0, color="#475569", lw=0.8)
            ax.set_ylabel("Residual change (mm)")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title("Residual-change distribution by pressure bin")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        if not bin_change.empty:
            ratios = []
            for pb in STANDARD_BIN_NAMES:
                vals = pd.to_numeric(bin_change[bin_change["pressure_bin"].eq(pb)]["during_minus_reference_residual_mm"], errors="coerce").dropna()
                ratios.append(((vals < 0).mean() if len(vals) else np.nan, (vals > 0).mean() if len(vals) else np.nan))
            x = np.arange(len(STANDARD_BIN_NAMES))
            ax.bar(x - 0.18, [r[0] for r in ratios], width=0.36, label="negative")
            ax.bar(x + 0.18, [r[1] for r in ratios], width=0.36, label="positive")
            ax.set_xticks(x, STANDARD_BIN_NAMES, rotation=45, ha="right")
            ax.set_ylabel("Event fraction")
            ax.legend()
        ax.set_title("Negative and positive residual-change fractions")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        if not ok.empty and "enhancement_class" in ok:
            for cls in ["strong_enhancement", "stable", "reverse"]:
                profiles = []
                for event_id in ok[ok["enhancement_class"].eq(cls)]["event_id"]:
                    p = output_root / f"{event_id}_native_layer_station.csv"
                    if not p.exists():
                        continue
                    s = pd.read_csv(p, parse_dates=["time_utc"])
                    pre = s[s["stage"].eq("pre")].groupby("pressure_upper_hPa")["station_cumulative_residual_mm"].median()
                    during = s[s["stage"].eq("during")].groupby("pressure_upper_hPa")["station_cumulative_residual_mm"].mean()
                    prof = (during - pre).dropna()
                    profiles.append(prof)
                if profiles:
                    joined = pd.concat(profiles, axis=1)
                    ax.plot(joined.mean(axis=1), joined.index, marker="o", label=cls)
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.axvline(0, color="#475569", lw=0.8)
            ax.set_xlabel("Mean cumulative residual anomaly (mm)")
            ax.set_ylabel("Pressure (hPa)")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        ax.set_title("Class-mean cumulative residual anomaly profiles")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        if not bin_change.empty:
            x = pd.to_numeric(bin_change["during_minus_reference_geopotential_thickness_m"], errors="coerce")
            y = pd.to_numeric(bin_change["during_minus_reference_residual_mm"], errors="coerce")
            data = pd.concat([x, y], axis=1).dropna()
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], s=18, alpha=0.7)
            r = data.iloc[:, 0].corr(data.iloc[:, 1]) if len(data) > 2 else np.nan
            ax.set_title(f"Thickness change vs layer residual change (r={r:.3f})")
            ax.set_xlabel("Thickness change (m)")
            ax.set_ylabel("Residual change (mm)")
            ax.axhline(0, color="#475569", lw=0.8)
            ax.axvline(0, color="#475569", lw=0.8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def write_all_events_outputs(output_root: Path, results: list[dict[str, Any]], config_count: int, enabled_count: int, total_seconds: float) -> dict[str, Path]:
    event_parts = [r["event_rows"] for r in results if not r["event_rows"].empty]
    bin_parts = [r["bin_rows"] for r in results if not r["bin_rows"].empty]
    event_rows = pd.concat(event_parts, ignore_index=True, sort=False) if event_parts else pd.DataFrame()
    bin_rows = pd.concat(bin_parts, ignore_index=True, sort=False) if bin_parts else pd.DataFrame()
    event_summary = all_events_event_summary(event_rows) if not event_rows.empty else pd.DataFrame()
    bin_summary = bin_rows.copy()
    bin_change = all_events_pressure_bin_change(bin_rows)
    files = {
        "event_summary": output_root / "all_events_step3_event_summary.csv",
        "bin_summary": output_root / "all_events_step3_pressure_bin_summary.csv",
        "bin_change": output_root / "all_events_step3_pressure_bin_change.csv",
        "report": output_root / "all_events_step3_batch_report.txt",
        "overview": output_root / "all_events_step3_overview.pdf",
    }
    event_summary.to_csv(files["event_summary"], index=False, encoding="utf-8-sig")
    bin_summary.to_csv(files["bin_summary"], index=False, encoding="utf-8-sig")
    bin_change.to_csv(files["bin_change"], index=False, encoding="utf-8-sig")
    write_batch_report(files["report"], results, event_summary, bin_change, config_count, enabled_count, total_seconds)
    write_overview_pdf(files["overview"], event_summary, bin_change, output_root)
    return files


def main() -> None:
    args = parse_args()
    configure_workflow_roots(args.data_root, args.workflow_out_root)
    config_path = resolve_step3_config_path(args.config)
    events = load_events(config_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "step3_batch_run.log"
    append_log(log_path, f"start config={config_path.resolve()} output_root={output_root}")

    if args.all_events:
        selected = [
            event
            for event in events.values()
            if args.include_disabled or event.get("enabled", False) is True
        ]
    else:
        event_id = args.event.lower().strip()
        if event_id not in events:
            print(f"Event not found: {event_id}")
            print(", ".join(sorted(events)))
            raise SystemExit(1)
        selected = [events[event_id]]
        if selected[0].get("enabled", False) is not True and not args.include_disabled:
            files = event_files(output_root, selected[0])
            summary_df = write_status_summary(files["summary"], selected[0], "disabled", "Event is disabled; use --include-disabled to process it.")
            append_log(log_path, f"{event_id}: disabled")
            print("Event is disabled. Use --include-disabled to process it.")
            print("Output directory:", output_root)
            print("Summary file:", files["summary"].resolve())
            return

    total_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    for event in selected:
        prefix = event_prefix(event)
        print(f"Processing {prefix} ({event.get('display_name', prefix)})")
        results.append(run_event(event, config_path, output_root, args.force, args.save_grid_debug, log_path))
    total_seconds = time.perf_counter() - total_start

    enabled_count = sum(1 for event in events.values() if event.get("enabled", False) is True)
    all_files = write_all_events_outputs(output_root, results, len(events), enabled_count, total_seconds)
    append_log(log_path, f"end total_seconds={total_seconds:.2f}")

    event_summary = pd.read_csv(all_files["event_summary"]) if all_files["event_summary"].exists() else pd.DataFrame()
    status_counts = event_summary["data_status"].value_counts().to_dict() if "data_status" in event_summary else {}
    class_counts = event_summary["enhancement_class"].fillna("unknown").value_counts().to_dict() if "enhancement_class" in event_summary else {}
    max_closure = event_summary.get("max_abs_full_closure_error_mm", pd.Series(dtype=float)).max() if not event_summary.empty else np.nan
    print("Configured events:", len(events))
    print("Enabled events:", enabled_count)
    print("Successful events:", int(status_counts.get("ok", 0)))
    print("Skipped events:", sum(1 for r in results if r["status"] == "skipped"))
    print("Missing-file events:", sum(n for k, n in status_counts.items() if str(k).startswith("missing_")))
    print("Failed events:", sum(n for k, n in status_counts.items() if k not in ["ok", "disabled"]))
    print("strong_enhancement events:", int(class_counts.get("strong_enhancement", 0)))
    print("moderate_enhancement events:", int(class_counts.get("moderate_enhancement", 0)))
    print("stable events:", int(class_counts.get("stable", 0)))
    print("reverse events:", int(class_counts.get("reverse", 0)))
    print("Max closure error:", max_closure)
    print("Total runtime seconds:", f"{total_seconds:.2f}")
    print("Mean seconds per event:", f"{np.mean([r['seconds'] for r in results]) if results else np.nan:.2f}")
    print("Output directory:", output_root)
    print("All-event outputs:")
    for p in all_files.values():
        print(p.resolve())


if __name__ == "__main__":
    main()
