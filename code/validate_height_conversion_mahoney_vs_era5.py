from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_JSON = Path(__file__).with_name("typhoon_events.json")
OUT_DIR = ROOT_DIR / "out1" / "height_conversion_validation"

OUT_SYNTHETIC_POINTS_CSV = OUT_DIR / "mahoney_vs_era5_synthetic_points.csv"
OUT_SYNTHETIC_SUMMARY_CSV = OUT_DIR / "mahoney_vs_era5_synthetic_summary.csv"
OUT_ACTUAL_POINTS_CSV = OUT_DIR / "mahoney_vs_era5_actual_profile_heights.csv"
OUT_ACTUAL_SUMMARY_CSV = OUT_DIR / "mahoney_vs_era5_actual_profile_summary.csv"
OUT_ACTUAL_TIMESERIES_CSV = OUT_DIR / "mahoney_vs_era5_actual_profile_timeseries.csv"
OUT_DELAY_PROFILES_CSV = OUT_DIR / "mahoney_vs_era5_delay_profiles.csv"
OUT_DELAY_SUMMARY_CSV = OUT_DIR / "mahoney_vs_era5_delay_summary.csv"
OUT_DELAY_TIMESERIES_CSV = OUT_DIR / "mahoney_vs_era5_delay_timeseries.csv"
OUT_CLOSURE_PROFILES_CSV = OUT_DIR / "mahoney_vs_era5_profile_saas_closure_profiles.csv"
OUT_CLOSURE_SUMMARY_CSV = OUT_DIR / "mahoney_vs_era5_profile_saas_closure_summary.csv"
OUT_CLOSURE_TIMESERIES_CSV = OUT_DIR / "mahoney_vs_era5_profile_saas_closure_timeseries.csv"
OUT_CLOSURE_ERA5_STATION_TIMESERIES_CSV = (
    OUT_DIR / "mahoney_vs_era5_profile_saas_closure_era5_station_interp_timeseries.csv"
)
OUT_CHECKS_TXT = OUT_DIR / "mahoney_vs_era5_validation_checks.txt"
OUT_SYNTHETIC_FIG_DIR = OUT_DIR / "synthetic_figures"
OUT_TIMESERIES_FIG_DIR = OUT_DIR / "timeseries_all_events"
OUT_DELAY_TIMESERIES_FIG_DIR = OUT_DIR / "delay_timeseries_all_events"
OUT_CLOSURE_TIMESERIES_FIG_DIR = OUT_DIR / "profile_saas_closure_timeseries_all_events"
OUT_CLOSURE_FIG_DPI = 450

REFERENCE_NOTE_ZH = (
    "这里的参考值是 Mahoney/WGS-84正常重力模型内部的数值积分参考值，"
    "不是实测真值或绝对真值。"
)
REFERENCE_NOTE_EN = (
    "The reference values here are numerical-integration reference values internal to the "
    "Mahoney/WGS-84 normal gravity model, not observed truth or absolute truth."
)

# All internal calculations use SI units: m and m/s^2.
GAMMA_45 = 9.80665
WGS84_A_M = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

GAMMA_E = 9.7803253359
SOMIGLIANA_K = 1.931853e-3
GRAVITY_RATIO_M = 0.003449787

ERA5_EARTH_RADIUS_M = 6_371_229.0

ERA5_H_COL = "geopotential_height_m"
SOUNDING_H_COL = "geopotential height_m"

ERA5_TIME_COL = "time_utc"
ERA5_LON_COL = "longitude"
ERA5_LAT_COL = "latitude"
ERA5_P_COL = "pressure_hPa"
ERA5_T_COL = "temperature_K"
ERA5_RH_COL = "relative_humidity_percent"

SOUNDING_TIME_COL = "sounding_time_utc"
SOUNDING_LON_COL = "longitude"
SOUNDING_LAT_COL = "latitude"
SOUNDING_P_COL = "pressure_hPa"
SOUNDING_TC_COL = "temperature_C"
SOUNDING_RH_COL = "relative humidity_%"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

SOURCE_COLORS = {
    "era5": "#2563eb",
    "sounding": "#16a34a",
}


def load_events() -> list[dict[str, Any]]:
    with CONFIG_JSON.open("r", encoding="utf-8") as f:
        return [
            event
            for event in json.load(f)["events"]
            if event.get("enabled", True)
        ]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def latitude_grid_deg() -> np.ndarray:
    default = np.arange(-90.0, 90.0 + 5.0, 5.0)
    required = np.array([0.0, 22.31, 30.0, 45.0, 60.0, 90.0, -22.31])
    return np.array(sorted(set(np.concatenate([default, required]))), dtype=float)


def height_grid_m() -> np.ndarray:
    default = np.arange(0.0, 30_000.0 + 100.0, 100.0)
    required = np.array(
        [0.0, 1000.0, 3000.0, 5000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0]
    )
    return np.array(sorted(set(np.concatenate([default, required]))), dtype=float)


def _sin2_lat(latitude_deg: float | np.ndarray) -> np.ndarray:
    phi = np.deg2rad(latitude_deg)
    return np.sin(phi) ** 2


def mahoney_surface_gravity(latitude_deg: float | np.ndarray) -> np.ndarray:
    """Mahoney Eq. (17), Somigliana normal gravity at the ellipsoid surface."""

    sin2 = _sin2_lat(latitude_deg)
    return GAMMA_E * (1.0 + SOMIGLIANA_K * sin2) / np.sqrt(1.0 - WGS84_E2 * sin2)


def mahoney_normal_gravity(z_m: float, latitude_deg: float) -> float:
    """Mahoney Eq. (18), normal gravity at geometric height z."""

    sin2 = float(_sin2_lat(latitude_deg))
    gamma_s = float(mahoney_surface_gravity(latitude_deg))
    vertical_factor = (
        1.0
        - 2.0
        * (1.0 + WGS84_F + GRAVITY_RATIO_M - 2.0 * WGS84_F * sin2)
        * z_m
        / WGS84_A_M
        + 3.0 * (z_m / WGS84_A_M) ** 2
    )
    return gamma_s * vertical_factor


def reference_geopotential_height(z_true_m: float, latitude_deg: float) -> float:
    """Numerical Mahoney/WGS-84 internal reference, not an observed or absolute truth.

    这里的参考值是 Mahoney/WGS-84正常重力模型内部的数值积分参考值，
    不是实测真值或绝对真值。
    """

    if np.isclose(z_true_m, 0.0):
        return 0.0

    integral, _ = quad(
        lambda z: mahoney_normal_gravity(z, latitude_deg),
        0.0,
        float(z_true_m),
        epsabs=1e-10,
        epsrel=1e-12,
        limit=200,
    )
    return float(integral / GAMMA_45)


def mahoney_integrated_geopotential_height(z_m: object, latitude_deg: object) -> np.ndarray:
    """Integral of Mahoney Eq. (18), divided by GAMMA_45.

    The integrand is quadratic in z, so the integral is evaluated exactly here.
    The synthetic reference grid still uses scipy.integrate.quad point by point.
    """

    z = np.asarray(z_m, dtype=float)
    sin2 = _sin2_lat(latitude_deg)
    gamma_s = mahoney_surface_gravity(latitude_deg)
    b = 1.0 + WGS84_F + GRAVITY_RATIO_M - 2.0 * WGS84_F * sin2
    return (gamma_s / GAMMA_45) * (
        z - b * z**2 / WGS84_A_M + z**3 / WGS84_A_M**2
    )


def mahoney_integral_inverse_height(geopotential_height_m: object, latitude_deg: object) -> np.ndarray:
    """Numerically invert the Mahoney/WGS-84 normal-gravity integral.

    This solves H = (1/GAMMA_45) * integral(gamma(z, phi), 0..Z) for Z.
    It is used for the actual ERA5/sounding event time-series comparison.
    """

    h = np.asarray(geopotential_height_m, dtype=float)
    sin2 = _sin2_lat(latitude_deg)
    gamma_s = mahoney_surface_gravity(latitude_deg)
    b = 1.0 + WGS84_F + GRAVITY_RATIO_M - 2.0 * WGS84_F * sin2
    c = gamma_s / GAMMA_45

    z = np.asarray(era5_geometric_height(h), dtype=float)
    for _ in range(8):
        integral_h = c * (z - b * z**2 / WGS84_A_M + z**3 / WGS84_A_M**2)
        derivative = c * (
            1.0 - 2.0 * b * z / WGS84_A_M + 3.0 * z**2 / WGS84_A_M**2
        )
        z -= (integral_h - h) / derivative
    return z


def mahoney_effective_radius(latitude_deg: float | np.ndarray) -> np.ndarray:
    """Mahoney Eq. (20) effective/fictitious radius for gravity variation, not ellipsoid radius."""

    sin2 = _sin2_lat(latitude_deg)
    return WGS84_A_M / (1.0 + WGS84_F + GRAVITY_RATIO_M - 2.0 * WGS84_F * sin2)


def mahoney_geometric_height(geopotential_height_m: object, latitude_deg: object) -> np.ndarray:
    h = np.asarray(geopotential_height_m, dtype=float)
    gamma_s = mahoney_surface_gravity(latitude_deg)
    radius = mahoney_effective_radius(latitude_deg)
    return radius * h / ((gamma_s / GAMMA_45) * radius - h)


def era5_geometric_height(geopotential_height_m: object) -> np.ndarray:
    h = np.asarray(geopotential_height_m, dtype=float)
    return ERA5_EARTH_RADIUS_M * h / (ERA5_EARTH_RADIUS_M - h)


def saastamoinen_zhd(pressure_hpa: float, height_m: float, latitude_deg: float) -> float:
    lat_rad = np.deg2rad(latitude_deg)
    denominator = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 2.8e-7 * height_m
    return 0.0022793 * pressure_hpa / denominator


def water_vapor_pressure_from_rh_percent(
    pressure_hpa: pd.Series,
    temperature_k: pd.Series,
    rh_percent: pd.Series,
) -> pd.Series:
    rh = rh_percent / 100.0
    temperature_c = temperature_k - 273.15
    exponent = ((18.729 - temperature_c / 227.3) * temperature_c) / (
        temperature_k - 15.28
    )
    return (
        6.1121
        * (1.0007 + 3.46e-6 * pressure_hpa)
        * rh
        * np.exp(exponent)
    )


def log_mean(nh_lower: np.ndarray, nh_upper: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.abs(nh_lower), np.abs(nh_upper))
    close = np.zeros_like(nh_lower, dtype=bool)
    nonzero = denominator > 0.0
    close[nonzero] = np.abs(nh_lower[nonzero] - nh_upper[nonzero]) / denominator[
        nonzero
    ] < 1e-8
    close[~nonzero] = True

    result = np.empty_like(nh_lower, dtype=float)
    result[close] = 0.5 * (nh_lower[close] + nh_upper[close])
    result[~close] = (nh_lower[~close] - nh_upper[~close]) / (
        np.log(nh_lower[~close]) - np.log(nh_upper[~close])
    )
    return result


def build_synthetic_points() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for latitude_deg in latitude_grid_deg():
        for z_true_m in height_grid_m():
            h_ref = reference_geopotential_height(float(z_true_m), float(latitude_deg))
            z_mahoney = float(mahoney_geometric_height(h_ref, latitude_deg))
            z_era5 = float(era5_geometric_height(h_ref))

            error_mahoney_m = z_mahoney - z_true_m
            error_era5_m = z_era5 - z_true_m
            method_difference_m = z_era5 - z_mahoney
            if np.isclose(z_true_m, 0.0):
                rel_mahoney_ppm = np.nan
                rel_era5_ppm = np.nan
            else:
                rel_mahoney_ppm = error_mahoney_m / z_true_m * 1e6
                rel_era5_ppm = error_era5_m / z_true_m * 1e6

            rows.append(
                {
                    "latitude_deg": float(latitude_deg),
                    "z_true_m": float(z_true_m),
                    "h_reference_m": h_ref,
                    "z_mahoney_numerical_reference_m": float(z_true_m),
                    "z_mahoney_m": z_mahoney,
                    "z_era5_m": z_era5,
                    "error_mahoney_m": error_mahoney_m,
                    "error_era5_m": error_era5_m,
                    "abs_error_mahoney_m": abs(error_mahoney_m),
                    "abs_error_era5_m": abs(error_era5_m),
                    "error_mahoney_mm": error_mahoney_m * 1000.0,
                    "error_era5_mm": error_era5_m * 1000.0,
                    "method_difference_m": method_difference_m,
                    "method_difference_mm": method_difference_m * 1000.0,
                    "relative_error_mahoney_ppm": rel_mahoney_ppm,
                    "relative_error_era5_ppm": rel_era5_ppm,
                }
            )
    return pd.DataFrame(rows)


def metric_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q05": np.nan,
            "q95": np.nan,
        }
    arr = clean.to_numpy(dtype=float)
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "MAE": float(clean.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(arr**2))),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q05": float(clean.quantile(0.05)),
        "q95": float(clean.quantile(0.95)),
    }


def build_synthetic_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "error_mahoney_mm",
        "error_era5_mm",
        "method_difference_mm",
        "relative_error_mahoney_ppm",
        "relative_error_era5_ppm",
    ]
    for metric in metrics:
        rows.append({"group": "ALL", "latitude_deg": np.nan, "metric": metric, **metric_stats(points[metric])})
    for latitude_deg, group in points.groupby("latitude_deg", sort=True):
        for metric in metrics:
            rows.append(
                {
                    "group": "latitude",
                    "latitude_deg": float(latitude_deg),
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def collect_actual_era5_heights(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for event in events:
        input_csv = resolve_path(event["era5"]["data_dir"]) / "era5_pressure_raw_extract.csv"
        if not input_csv.exists():
            continue
        usecols = ["time_utc", "latitude", "longitude", ERA5_H_COL]
        df = pd.read_csv(input_csv, usecols=usecols)
        df = df.rename(columns={ERA5_H_COL: "h_geopotential_m"})
        df["source"] = "era5"
        df["event_id"] = str(event["id"]).lower()
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def expected_sounding_qc_or_combined(event: dict[str, Any]) -> Path:
    station = str(event["station"]["id"])
    start = pd.Timestamp(event["sounding"]["start_time"])
    end = pd.Timestamp(event["sounding"]["end_time"])
    combined = resolve_path(event["sounding"]["data_dir"]) / (
        f"{station}_{start:%Y%m%d%H}_{end:%Y%m%d%H}_combined.csv"
    )
    qc = combined.with_name(f"{combined.stem}_qc.csv")
    return qc if qc.exists() else combined


def collect_actual_sounding_heights(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for event in events:
        input_csv = expected_sounding_qc_or_combined(event)
        if not input_csv.exists():
            continue
        usecols = ["sounding_time_utc", "latitude", "longitude", SOUNDING_H_COL]
        df = pd.read_csv(input_csv, usecols=usecols)
        df = df.rename(
            columns={
                "sounding_time_utc": "time_utc",
                SOUNDING_H_COL: "h_geopotential_m",
            }
        )
        df["source"] = "sounding"
        df["event_id"] = str(event["id"]).lower()
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_delay_profile(
    valid: pd.DataFrame,
    source: str,
    event_id: str,
    time_utc: object,
    latitude: float,
    longitude: float,
) -> dict[str, object] | None:
    if len(valid) < 2:
        return None

    if source == "era5":
        h_col = ERA5_H_COL
        p_col = ERA5_P_COL
        temperature_k = valid[ERA5_T_COL]
        rh_percent = valid[ERA5_RH_COL]
    else:
        h_col = SOUNDING_H_COL
        p_col = SOUNDING_P_COL
        temperature_k = valid[SOUNDING_TC_COL] + 273.15
        rh_percent = valid[SOUNDING_RH_COL]

    profile = valid.sort_values(h_col).copy()
    profile["temperature_K_for_delay"] = temperature_k.loc[profile.index]
    profile["e"] = water_vapor_pressure_from_rh_percent(
        profile[p_col],
        profile["temperature_K_for_delay"],
        rh_percent.loc[profile.index],
    )
    profile["Nh"] = 77.6890 * (
        profile[p_col] - 0.37803 * profile["e"]
    ) / profile["temperature_K_for_delay"]

    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[p_col, h_col, "Nh"]
    )
    profile = profile[profile["Nh"] > 0.0].copy()
    if len(profile) < 2:
        return None

    bottom_layer = profile.loc[profile[p_col].idxmax()]
    top_layer = profile.loc[profile[h_col].idxmax()]

    bottom_pressure_hpa = float(bottom_layer[p_col])
    bottom_geopotential_height_m = float(bottom_layer[h_col])
    bottom_era5_height_m = float(era5_geometric_height(bottom_geopotential_height_m))
    bottom_mahoney_height_m = float(
        mahoney_integral_inverse_height(bottom_geopotential_height_m, latitude)
    )

    top_pressure_hpa = float(top_layer[p_col])
    top_geopotential_height_m = float(top_layer[h_col])
    top_era5_height_m = float(era5_geometric_height(top_geopotential_height_m))
    top_mahoney_height_m = float(
        mahoney_integral_inverse_height(top_geopotential_height_m, latitude)
    )

    height_geopotential = profile[h_col].to_numpy(dtype=float)
    height_era5 = era5_geometric_height(height_geopotential)
    height_mahoney = mahoney_integral_inverse_height(height_geopotential, latitude)
    nh = profile["Nh"].to_numpy(dtype=float)

    d_height = np.diff(height_geopotential)
    valid_interval = d_height >= 0.0
    if not np.any(valid_interval):
        return None

    nh_lower = nh[:-1]
    nh_upper = nh[1:]
    nh_lm = log_mean(nh_lower[valid_interval], nh_upper[valid_interval])

    dz_era5 = np.diff(height_era5)
    dz_mahoney = np.diff(height_mahoney)
    zhd_int_era5_m = 1e-6 * np.sum(dz_era5[valid_interval] * nh_lm)
    zhd_int_mahoney_m = 1e-6 * np.sum(dz_mahoney[valid_interval] * nh_lm)

    zhd_saas_bottom_era5_m = saastamoinen_zhd(
        bottom_pressure_hpa,
        bottom_era5_height_m,
        latitude,
    )
    zhd_saas_bottom_mahoney_m = saastamoinen_zhd(
        bottom_pressure_hpa,
        bottom_mahoney_height_m,
        latitude,
    )
    zhd_top_era5_m = saastamoinen_zhd(top_pressure_hpa, top_era5_height_m, latitude)
    zhd_top_mahoney_m = saastamoinen_zhd(
        top_pressure_hpa,
        top_mahoney_height_m,
        latitude,
    )

    zhd_profile_era5_m = float(zhd_int_era5_m + zhd_top_era5_m)
    zhd_profile_mahoney_m = float(zhd_int_mahoney_m + zhd_top_mahoney_m)

    return {
        "source": source,
        "event_id": event_id,
        "time_utc": time_utc,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "n_layers": int(len(profile)),
        "bottom_pressure_hpa": bottom_pressure_hpa,
        "bottom_geopotential_height_m": bottom_geopotential_height_m,
        "bottom_era5_spherical_height_m": bottom_era5_height_m,
        "bottom_mahoney_numerical_height_m": bottom_mahoney_height_m,
        "top_pressure_hpa": top_pressure_hpa,
        "top_geopotential_height_m": top_geopotential_height_m,
        "top_era5_spherical_height_m": top_era5_height_m,
        "top_mahoney_numerical_height_m": top_mahoney_height_m,
        "zhd_saas_bottom_era5_spherical_m": float(zhd_saas_bottom_era5_m),
        "zhd_saas_bottom_mahoney_numerical_m": float(zhd_saas_bottom_mahoney_m),
        "zhd_top_era5_spherical_m": float(zhd_top_era5_m),
        "zhd_top_mahoney_numerical_m": float(zhd_top_mahoney_m),
        "zhd_integral_era5_spherical_m": float(zhd_int_era5_m),
        "zhd_integral_mahoney_numerical_m": float(zhd_int_mahoney_m),
        "zhd_profile_era5_spherical_m": zhd_profile_era5_m,
        "zhd_profile_mahoney_numerical_m": zhd_profile_mahoney_m,
        "zhd_profile_difference_m": zhd_profile_era5_m - zhd_profile_mahoney_m,
        "zhd_profile_difference_mm": (zhd_profile_era5_m - zhd_profile_mahoney_m)
        * 1000.0,
        "zhd_saas_bottom_difference_m": float(
            zhd_saas_bottom_era5_m - zhd_saas_bottom_mahoney_m
        ),
        "zhd_saas_bottom_difference_mm": float(
            (zhd_saas_bottom_era5_m - zhd_saas_bottom_mahoney_m) * 1000.0
        ),
        "zhd_top_difference_m": float(zhd_top_era5_m - zhd_top_mahoney_m),
        "zhd_top_difference_mm": float((zhd_top_era5_m - zhd_top_mahoney_m) * 1000.0),
        "dz_sum_era5_spherical_m": float(np.sum(dz_era5[valid_interval])),
        "dz_sum_mahoney_numerical_m": float(np.sum(dz_mahoney[valid_interval])),
        "dz_sum_difference_m": float(
            np.sum(dz_era5[valid_interval] - dz_mahoney[valid_interval])
        ),
    }


def collect_era5_delay_profiles(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usecols = [
        ERA5_TIME_COL,
        ERA5_LON_COL,
        ERA5_LAT_COL,
        ERA5_P_COL,
        ERA5_H_COL,
        ERA5_T_COL,
        ERA5_RH_COL,
    ]
    for event in events:
        event_id = str(event["id"]).lower()
        input_csv = resolve_path(event["era5"]["data_dir"]) / "era5_pressure_raw_extract.csv"
        if not input_csv.exists():
            continue
        df = pd.read_csv(input_csv, usecols=usecols)
        for col in [ERA5_LON_COL, ERA5_LAT_COL, ERA5_P_COL, ERA5_H_COL, ERA5_T_COL, ERA5_RH_COL]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=usecols).copy()
        for (time_utc, latitude, longitude), group in df.groupby(
            [ERA5_TIME_COL, ERA5_LAT_COL, ERA5_LON_COL],
            sort=True,
        ):
            row = prepare_delay_profile(
                group,
                "era5",
                event_id,
                time_utc,
                float(latitude),
                float(longitude),
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def collect_sounding_delay_profiles(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usecols = [
        SOUNDING_TIME_COL,
        SOUNDING_LON_COL,
        SOUNDING_LAT_COL,
        SOUNDING_P_COL,
        SOUNDING_H_COL,
        SOUNDING_TC_COL,
        SOUNDING_RH_COL,
    ]
    for event in events:
        event_id = str(event["id"]).lower()
        input_csv = expected_sounding_qc_or_combined(event)
        if not input_csv.exists():
            continue
        df = pd.read_csv(input_csv, usecols=usecols)
        for col in [
            SOUNDING_LON_COL,
            SOUNDING_LAT_COL,
            SOUNDING_P_COL,
            SOUNDING_H_COL,
            SOUNDING_TC_COL,
            SOUNDING_RH_COL,
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=usecols).copy()
        for time_utc, group in df.groupby(SOUNDING_TIME_COL, sort=True):
            if group.empty:
                continue
            latitude = float(group[SOUNDING_LAT_COL].iloc[0])
            longitude = float(group[SOUNDING_LON_COL].iloc[0])
            row = prepare_delay_profile(
                group,
                "sounding",
                event_id,
                time_utc,
                latitude,
                longitude,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def build_delay_profiles(events: list[dict[str, Any]]) -> pd.DataFrame:
    delay_profiles = pd.concat(
        [collect_era5_delay_profiles(events), collect_sounding_delay_profiles(events)],
        ignore_index=True,
    )
    if delay_profiles.empty:
        return delay_profiles
    delay_profiles["time_utc"] = pd.to_datetime(delay_profiles["time_utc"], errors="coerce")
    return delay_profiles.sort_values(["source", "event_id", "time_utc"]).reset_index(drop=True)


def build_actual_points(events: list[dict[str, Any]]) -> pd.DataFrame:
    actual = pd.concat(
        [collect_actual_era5_heights(events), collect_actual_sounding_heights(events)],
        ignore_index=True,
    )
    if actual.empty:
        return actual

    actual["time_utc"] = pd.to_datetime(actual["time_utc"], errors="coerce")
    for col in ["latitude", "longitude", "h_geopotential_m"]:
        actual[col] = pd.to_numeric(actual[col], errors="coerce")
    actual = actual.dropna(subset=["event_id", "source", "latitude", "h_geopotential_m"]).copy()

    actual["z_mahoney_numerical_m"] = mahoney_integral_inverse_height(
        actual["h_geopotential_m"],
        actual["latitude"],
    )
    actual["z_mahoney_formula_m"] = mahoney_geometric_height(
        actual["h_geopotential_m"],
        actual["latitude"],
    )
    actual["z_era5_m"] = era5_geometric_height(actual["h_geopotential_m"])
    actual["method_difference_m"] = actual["z_era5_m"] - actual["z_mahoney_numerical_m"]
    actual["method_difference_mm"] = actual["method_difference_m"] * 1000.0
    actual["abs_method_difference_mm"] = actual["method_difference_mm"].abs()
    actual["mahoney_formula_minus_numerical_m"] = (
        actual["z_mahoney_formula_m"] - actual["z_mahoney_numerical_m"]
    )
    actual["mahoney_formula_minus_numerical_mm"] = (
        actual["mahoney_formula_minus_numerical_m"] * 1000.0
    )
    return actual.sort_values(["source", "event_id", "time_utc"]).reset_index(drop=True)


def build_actual_summary(actual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if actual.empty:
        return pd.DataFrame(rows)
    metrics = [
        "method_difference_mm",
        "mahoney_formula_minus_numerical_mm",
    ]
    for keys, group in actual.groupby(["source", "event_id"], dropna=False, sort=True):
        source, event_id = keys
        for metric in metrics:
            rows.append(
                {
                    "source": source,
                    "event_id": event_id,
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for source, group in actual.groupby("source", dropna=False, sort=True):
        for metric in metrics:
            rows.append(
                {
                    "source": source,
                    "event_id": "ALL",
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for metric in metrics:
        rows.append(
            {
                "source": "ALL",
                "event_id": "ALL",
                "metric": metric,
                **metric_stats(actual[metric]),
            }
        )
    return pd.DataFrame(rows)


def build_actual_timeseries(actual: pd.DataFrame) -> pd.DataFrame:
    if actual.empty:
        return pd.DataFrame()

    rows = (
        actual.groupby(["source", "event_id", "time_utc"], dropna=False, sort=True)
        .agg(
            n_level_points=("method_difference_m", "size"),
            mean_difference_m=("method_difference_m", "mean"),
            median_difference_m=("method_difference_m", "median"),
            min_difference_m=("method_difference_m", "min"),
            max_difference_m=("method_difference_m", "max"),
            mean_abs_difference_m=("method_difference_m", lambda x: x.abs().mean()),
            mean_formula_minus_numerical_mm=(
                "mahoney_formula_minus_numerical_mm",
                "mean",
            ),
        )
        .reset_index()
    )
    mm_cols = {
        "mean_difference_m": "mean_difference_mm",
        "median_difference_m": "median_difference_mm",
        "min_difference_m": "min_difference_mm",
        "max_difference_m": "max_difference_mm",
        "mean_abs_difference_m": "mean_abs_difference_mm",
    }
    for col_m, col_mm in mm_cols.items():
        rows[col_mm] = rows[col_m] * 1000.0
    return rows


def build_delay_summary(delay_profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if delay_profiles.empty:
        return pd.DataFrame(rows)
    metrics = [
        "zhd_profile_difference_mm",
        "zhd_saas_bottom_difference_mm",
        "zhd_top_difference_mm",
        "dz_sum_difference_m",
    ]
    for keys, group in delay_profiles.groupby(["source", "event_id"], dropna=False, sort=True):
        source, event_id = keys
        for metric in metrics:
            rows.append(
                {
                    "source": source,
                    "event_id": event_id,
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for source, group in delay_profiles.groupby("source", dropna=False, sort=True):
        for metric in metrics:
            rows.append(
                {
                    "source": source,
                    "event_id": "ALL",
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for metric in metrics:
        rows.append(
            {
                "source": "ALL",
                "event_id": "ALL",
                "metric": metric,
                **metric_stats(delay_profiles[metric]),
            }
        )
    return pd.DataFrame(rows)


def build_delay_timeseries(delay_profiles: pd.DataFrame) -> pd.DataFrame:
    if delay_profiles.empty:
        return pd.DataFrame()
    return (
        delay_profiles.groupby(["source", "event_id", "time_utc"], dropna=False, sort=True)
        .agg(
            n_profiles=("zhd_profile_difference_m", "size"),
            mean_difference_m=("zhd_profile_difference_m", "mean"),
            median_difference_m=("zhd_profile_difference_m", "median"),
            min_difference_m=("zhd_profile_difference_m", "min"),
            max_difference_m=("zhd_profile_difference_m", "max"),
            mean_difference_mm=("zhd_profile_difference_mm", "mean"),
            median_difference_mm=("zhd_profile_difference_mm", "median"),
            min_difference_mm=("zhd_profile_difference_mm", "min"),
            max_difference_mm=("zhd_profile_difference_mm", "max"),
            mean_era5_spherical_zhd_m=("zhd_profile_era5_spherical_m", "mean"),
            mean_mahoney_numerical_zhd_m=("zhd_profile_mahoney_numerical_m", "mean"),
        )
        .reset_index()
    )


def build_profile_saas_closure_profiles(delay_profiles: pd.DataFrame) -> pd.DataFrame:
    if delay_profiles.empty:
        return pd.DataFrame()

    cols = [
        "source",
        "event_id",
        "time_utc",
        "latitude",
        "longitude",
        "n_layers",
        "zhd_saas_bottom_era5_spherical_m",
        "zhd_saas_bottom_mahoney_numerical_m",
        "zhd_profile_era5_spherical_m",
        "zhd_profile_mahoney_numerical_m",
    ]
    out = delay_profiles[cols].copy()
    out["profile_minus_saas_era5_spherical_m"] = (
        out["zhd_profile_era5_spherical_m"]
        - out["zhd_saas_bottom_era5_spherical_m"]
    )
    out["profile_minus_saas_mahoney_numerical_m"] = (
        out["zhd_profile_mahoney_numerical_m"]
        - out["zhd_saas_bottom_mahoney_numerical_m"]
    )
    out["profile_minus_saas_era5_spherical_mm"] = (
        out["profile_minus_saas_era5_spherical_m"] * 1000.0
    )
    out["profile_minus_saas_mahoney_numerical_mm"] = (
        out["profile_minus_saas_mahoney_numerical_m"] * 1000.0
    )
    out["closure_change_mahoney_minus_era5_mm"] = (
        out["profile_minus_saas_mahoney_numerical_mm"]
        - out["profile_minus_saas_era5_spherical_mm"]
    )
    return out


def build_profile_saas_closure_summary(closure_profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if closure_profiles.empty:
        return pd.DataFrame(rows)

    metrics = [
        "profile_minus_saas_era5_spherical_mm",
        "profile_minus_saas_mahoney_numerical_mm",
        "closure_change_mahoney_minus_era5_mm",
    ]
    for keys, group in closure_profiles.groupby(["source", "event_id"], dropna=False, sort=True):
        source, event_id = keys
        for metric in metrics:
            rows.append(
                {
                    "source": source,
                    "event_id": event_id,
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for source, group in closure_profiles.groupby("source", dropna=False, sort=True):
        for metric in metrics:
            rows.append(
                {
                    "source": source,
                    "event_id": "ALL",
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for metric in metrics:
        rows.append(
            {
                "source": "ALL",
                "event_id": "ALL",
                "metric": metric,
                **metric_stats(closure_profiles[metric]),
            }
        )
    return pd.DataFrame(rows)


def build_profile_saas_closure_timeseries(closure_profiles: pd.DataFrame) -> pd.DataFrame:
    if closure_profiles.empty:
        return pd.DataFrame()

    return (
        closure_profiles.groupby(["source", "event_id", "time_utc"], dropna=False, sort=True)
        .agg(
            n_profiles=("profile_minus_saas_era5_spherical_mm", "size"),
            era5_mean_mm=("profile_minus_saas_era5_spherical_mm", "mean"),
            era5_median_mm=("profile_minus_saas_era5_spherical_mm", "median"),
            era5_min_mm=("profile_minus_saas_era5_spherical_mm", "min"),
            era5_max_mm=("profile_minus_saas_era5_spherical_mm", "max"),
            mahoney_mean_mm=("profile_minus_saas_mahoney_numerical_mm", "mean"),
            mahoney_median_mm=("profile_minus_saas_mahoney_numerical_mm", "median"),
            mahoney_min_mm=("profile_minus_saas_mahoney_numerical_mm", "min"),
            mahoney_max_mm=("profile_minus_saas_mahoney_numerical_mm", "max"),
            closure_change_mean_mm=("closure_change_mahoney_minus_era5_mm", "mean"),
        )
        .reset_index()
    )


def bracket_values(values: np.ndarray, target: float) -> tuple[float, float] | None:
    values = np.array(sorted(np.unique(values.astype(float))), dtype=float)
    if values.size == 0 or target < values.min() or target > values.max():
        return None

    exact = values[np.isclose(values, target, rtol=0.0, atol=1e-10)]
    if exact.size:
        exact_value = float(exact[0])
        return exact_value, exact_value

    lower = values[values < target]
    upper = values[values > target]
    if lower.size == 0 or upper.size == 0:
        return None
    return float(lower.max()), float(upper.min())


def bilinear_value(
    values_by_point: dict[tuple[float, float], float],
    station_lat: float,
    station_lon: float,
    lat_s: float,
    lat_n: float,
    lon_w: float,
    lon_e: float,
) -> float | None:
    on_lon_grid = np.isclose(lon_w, lon_e, rtol=0.0, atol=1e-10)
    on_lat_grid = np.isclose(lat_s, lat_n, rtol=0.0, atol=1e-10)

    if on_lon_grid and on_lat_grid:
        required_points = [(lat_s, lon_w)]
    elif on_lon_grid:
        required_points = [(lat_s, lon_w), (lat_n, lon_w)]
    elif on_lat_grid:
        required_points = [(lat_s, lon_w), (lat_s, lon_e)]
    else:
        required_points = [
            (lat_s, lon_w),
            (lat_s, lon_e),
            (lat_n, lon_w),
            (lat_n, lon_e),
        ]

    if any(point not in values_by_point for point in required_points):
        return None

    if on_lon_grid and on_lat_grid:
        return float(values_by_point[(lat_s, lon_w)])
    if on_lon_grid:
        weight_n = (station_lat - lat_s) / (lat_n - lat_s)
        weight_s = 1.0 - weight_n
        return float(
            weight_s * values_by_point[(lat_s, lon_w)]
            + weight_n * values_by_point[(lat_n, lon_w)]
        )
    if on_lat_grid:
        weight_e = (station_lon - lon_w) / (lon_e - lon_w)
        weight_w = 1.0 - weight_e
        return float(
            weight_w * values_by_point[(lat_s, lon_w)]
            + weight_e * values_by_point[(lat_s, lon_e)]
        )

    denominator = (lon_e - lon_w) * (lat_n - lat_s)
    weights = {
        (lat_s, lon_w): ((lon_e - station_lon) * (lat_n - station_lat)) / denominator,
        (lat_s, lon_e): ((station_lon - lon_w) * (lat_n - station_lat)) / denominator,
        (lat_n, lon_w): ((lon_e - station_lon) * (station_lat - lat_s)) / denominator,
        (lat_n, lon_e): ((station_lon - lon_w) * (station_lat - lat_s)) / denominator,
    }
    return float(sum(weights[point] * values_by_point[point] for point in weights))


def build_era5_station_closure_timeseries(
    closure_profiles: pd.DataFrame,
    events: list[dict[str, Any]],
) -> pd.DataFrame:
    if closure_profiles.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    era5 = closure_profiles[closure_profiles["source"] == "era5"].copy()
    event_map = {str(event["id"]).lower(): event for event in events}
    metrics = [
        "profile_minus_saas_era5_spherical_mm",
        "profile_minus_saas_mahoney_numerical_mm",
    ]

    for event_id, event_group in era5.groupby("event_id", sort=True):
        event = event_map.get(str(event_id).lower())
        if event is None:
            continue
        station_lat = float(event["station"]["latitude"])
        station_lon = float(event["station"]["longitude"])
        lat_bracket = bracket_values(event_group["latitude"].to_numpy(dtype=float), station_lat)
        lon_bracket = bracket_values(event_group["longitude"].to_numpy(dtype=float), station_lon)
        if lat_bracket is None or lon_bracket is None:
            continue
        lat_s, lat_n = lat_bracket
        lon_w, lon_e = lon_bracket

        for time_utc, time_group in event_group.groupby("time_utc", sort=True):
            row: dict[str, object] = {
                "source": "era5_station_interp",
                "event_id": event_id,
                "time_utc": time_utc,
                "latitude": station_lat,
                "longitude": station_lon,
                "lat_s": lat_s,
                "lat_n": lat_n,
                "lon_w": lon_w,
                "lon_e": lon_e,
            }
            ok = True
            for metric in metrics:
                values = {
                    (float(item["latitude"]), float(item["longitude"])): float(item[metric])
                    for _, item in time_group.iterrows()
                }
                interpolated = bilinear_value(
                    values,
                    station_lat,
                    station_lon,
                    lat_s,
                    lat_n,
                    lon_w,
                    lon_e,
                )
                if interpolated is None:
                    ok = False
                    break
                row[metric] = interpolated
            if not ok:
                continue
            row["closure_change_mahoney_minus_era5_mm"] = (
                row["profile_minus_saas_mahoney_numerical_mm"]
                - row["profile_minus_saas_era5_spherical_mm"]
            )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["event_id", "time_utc"]).reset_index(drop=True)


def stage_times(event: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    plot_cfg = event["plot"]
    return (
        pd.Timestamp(plot_cfg["time_start"]),
        pd.Timestamp(plot_cfg["pre_end"]),
        pd.Timestamp(plot_cfg["impact_end"]),
        pd.Timestamp(plot_cfg["time_end"]),
    )


def add_stage_background(ax: plt.Axes, event: dict[str, Any], show_labels: bool) -> None:
    display_name = event.get("display_name", event["id"].title())
    time_start, pre_end, impact_end, time_end = stage_times(event)
    stages = [
        (time_start, pre_end, f"Pre-{display_name}", "#dbeafe"),
        (pre_end, impact_end, f"{display_name} impact", "#fee2e2"),
        (impact_end, time_end, f"Post-{display_name}", "#dcfce7"),
    ]
    for start, end, label, color in stages:
        ax.axvspan(start, end, color=color, alpha=0.55, lw=0, zorder=0)
        if show_labels:
            center = start + (end - start) / 2
            ax.text(
                center,
                1.06,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                color="#334155",
            )
    for boundary in [pre_end, impact_end]:
        ax.axvline(boundary, color="#475569", lw=1.0, ls="-", alpha=0.65, zorder=1)


def polish_timeseries_axis(ax: plt.Axes, event: dict[str, Any]) -> None:
    time_start, _, _, time_end = stage_times(event)
    ax.set_xlim(time_start, time_end)
    ax.axhline(0.0, color="#0f172a", lw=0.9, alpha=0.55, zorder=2)
    ax.grid(True, which="major", axis="y", color="#cbd5e1", lw=0.8, alpha=0.65)
    ax.grid(True, which="major", axis="x", color="#e2e8f0", lw=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0, colors="#334155")
    ax.set_ylabel("Height difference (m)")


def plot_source_timeseries(ax: plt.Axes, df: pd.DataFrame, source: str) -> None:
    color = SOURCE_COLORS[source]
    ax.fill_between(
        df["time_utc"],
        df["min_difference_m"],
        df["max_difference_m"],
        color=color,
        alpha=0.13,
        lw=0,
        label="level range",
        zorder=2,
    )
    ax.plot(
        df["time_utc"],
        df["mean_difference_m"],
        color=color,
        lw=2.0,
        label="mean",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["median_difference_m"],
        color="#0f172a",
        lw=1.2,
        ls="--",
        label="median",
        zorder=4,
    )


def save_actual_timeseries_plot(
    event_id: str,
    event: dict[str, Any],
    event_ts: pd.DataFrame,
) -> Path | None:
    panels = []
    for source in ["sounding", "era5"]:
        source_df = event_ts[event_ts["source"] == source].copy()
        if not source_df.empty:
            panels.append((source, source_df.sort_values("time_utc")))
    if not panels:
        return None

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13.5, max(3.2, 3.1 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for index, (source, source_df) in enumerate(panels):
        add_stage_background(axes[index], event, show_labels=index == 0)
        plot_source_timeseries(axes[index], source_df, source)
        axes[index].set_title(
            f"{chr(97 + index)}  {source} height conversion",
            loc="left",
            fontweight="bold",
        )
        axes[index].legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            facecolor="white",
            edgecolor="#cbd5e1",
            framealpha=0.92,
            ncol=1,
        )

    for ax in axes:
        polish_timeseries_axis(ax, event)

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    display_name = event.get("display_name", event_id.title())
    region_name = event.get("region_name", "")
    base_title = f"Mahoney Numerical Integral vs ERA5 Spherical Height During Typhoon {display_name}"
    title = f"{base_title} Around {region_name}" if region_name else base_title
    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.12)
    fig.text(
        0.5,
        1.02,
        "Plotted value: ERA5 spherical geometric height minus Mahoney/WGS-84 numerical-integral inverse height",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475569",
    )

    OUT_TIMESERIES_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_TIMESERIES_FIG_DIR / f"{event_id}_mahoney_numeric_vs_era5_sphere_timeseries.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_actual_timeseries_figures(
    events: list[dict[str, Any]],
    timeseries: pd.DataFrame,
) -> list[Path]:
    if timeseries.empty:
        return []

    event_map = {str(event["id"]).lower(): event for event in events}
    written: list[Path] = []
    for event_id in sorted(set(timeseries["event_id"]) & set(event_map)):
        event_ts = timeseries[timeseries["event_id"] == event_id].copy()
        out_png = save_actual_timeseries_plot(event_id, event_map[event_id], event_ts)
        if out_png is not None:
            written.append(out_png)
    return written


def plot_synthetic_figures(points: pd.DataFrame) -> list[Path]:
    OUT_SYNTHETIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    selected_lats = [0.0, 22.31, 45.0, 60.0, 90.0]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    for lat, color in zip(selected_lats, colors):
        subset = points[np.isclose(points["latitude_deg"], lat) & (points["z_true_m"] > 0.0)].copy()
        if subset.empty:
            continue
        x_km = subset["z_true_m"] / 1000.0
        axes[0].plot(
            x_km,
            subset["error_era5_mm"] / 1000.0,
            color=color,
            lw=1.8,
            label=f"{lat:g} deg",
        )
        axes[1].plot(
            x_km,
            subset["error_mahoney_mm"],
            color=color,
            lw=1.8,
            label=f"{lat:g} deg",
        )

    axes[0].set_title("a  ERA5 spherical approximation error", loc="left", fontweight="bold")
    axes[0].set_ylabel("Z_ERA5 - Z_ref (m)")
    axes[1].set_title("b  Mahoney closed-form approximation error", loc="left", fontweight="bold")
    axes[1].set_ylabel("Z_Mahoney formula - Z_ref (mm)")
    axes[1].set_xlabel("Reference geometric height (km)")

    for ax in axes:
        ax.axhline(0.0, color="#0f172a", lw=0.9, alpha=0.55)
        ax.grid(True, color="#e2e8f0", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="#cbd5e1",
            framealpha=0.92,
            ncol=2,
        )

    fig.suptitle(
        "Height Conversion Error Relative to Mahoney/WGS-84 Numerical-Integration Reference",
        fontsize=16,
        fontweight="bold",
    )
    out_png = OUT_SYNTHETIC_FIG_DIR / "synthetic_height_conversion_errors.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return [out_png]


def polish_delay_axis(ax: plt.Axes, event: dict[str, Any]) -> None:
    time_start, _, _, time_end = stage_times(event)
    ax.set_xlim(time_start, time_end)
    ax.axhline(0.0, color="#0f172a", lw=0.9, alpha=0.55, zorder=2)
    ax.grid(True, which="major", axis="y", color="#cbd5e1", lw=0.8, alpha=0.65)
    ax.grid(True, which="major", axis="x", color="#e2e8f0", lw=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0, colors="#334155")
    ax.set_ylabel("ZHD difference (mm)")


def plot_delay_source(ax: plt.Axes, df: pd.DataFrame, source: str) -> None:
    color = SOURCE_COLORS[source]
    ax.fill_between(
        df["time_utc"],
        df["min_difference_mm"],
        df["max_difference_mm"],
        color=color,
        alpha=0.13,
        lw=0,
        label="profile/grid range",
        zorder=2,
    )
    ax.plot(
        df["time_utc"],
        df["mean_difference_mm"],
        color=color,
        lw=2.0,
        label="mean",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["median_difference_mm"],
        color="#0f172a",
        lw=1.2,
        ls="--",
        label="median",
        zorder=4,
    )


def save_delay_timeseries_plot(
    event_id: str,
    event: dict[str, Any],
    event_ts: pd.DataFrame,
) -> Path | None:
    panels = []
    for source in ["sounding", "era5"]:
        source_df = event_ts[event_ts["source"] == source].copy()
        if not source_df.empty:
            panels.append((source, source_df.sort_values("time_utc")))
    if not panels:
        return None

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13.5, max(3.2, 3.1 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for index, (source, source_df) in enumerate(panels):
        add_stage_background(axes[index], event, show_labels=index == 0)
        plot_delay_source(axes[index], source_df, source)
        axes[index].set_title(
            f"{chr(97 + index)}  {source} ZHD comparison",
            loc="left",
            fontweight="bold",
        )
        axes[index].legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            facecolor="white",
            edgecolor="#cbd5e1",
            framealpha=0.92,
            ncol=1,
        )

    for ax in axes:
        polish_delay_axis(ax, event)

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    display_name = event.get("display_name", event_id.title())
    region_name = event.get("region_name", "")
    base_title = f"ZHD From Mahoney Numerical Height vs ERA5 Spherical Height During Typhoon {display_name}"
    title = f"{base_title} Around {region_name}" if region_name else base_title
    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.12)
    fig.text(
        0.5,
        1.02,
        "Plotted value: ZHD using ERA5 spherical heights minus ZHD using Mahoney/WGS-84 numerical-integral heights",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475569",
    )

    OUT_DELAY_TIMESERIES_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DELAY_TIMESERIES_FIG_DIR / f"{event_id}_mahoney_height_vs_era5_height_zhd_timeseries.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_delay_timeseries_figures(
    events: list[dict[str, Any]],
    timeseries: pd.DataFrame,
) -> list[Path]:
    if timeseries.empty:
        return []
    event_map = {str(event["id"]).lower(): event for event in events}
    written: list[Path] = []
    for event_id in sorted(set(timeseries["event_id"]) & set(event_map)):
        event_ts = timeseries[timeseries["event_id"] == event_id].copy()
        out_png = save_delay_timeseries_plot(event_id, event_map[event_id], event_ts)
        if out_png is not None:
            written.append(out_png)
    return written


def polish_closure_axis(ax: plt.Axes, event: dict[str, Any]) -> None:
    time_start, _, _, time_end = stage_times(event)
    ax.set_xlim(time_start, time_end)
    ax.axhline(0.0, color="#0f172a", lw=0.9, alpha=0.60, zorder=2)
    ax.grid(True, which="major", axis="y", color="#cbd5e1", lw=0.8, alpha=0.65)
    ax.grid(True, which="major", axis="x", color="#e2e8f0", lw=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0, colors="#334155")
    ax.set_ylabel("Profile - bottom Saas (mm)")


def plot_closure_source(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.fill_between(
        df["time_utc"],
        df["era5_min_mm"],
        df["era5_max_mm"],
        color="#2563eb",
        alpha=0.12,
        lw=0,
        label="ERA5 height range",
        zorder=2,
    )
    ax.fill_between(
        df["time_utc"],
        df["mahoney_min_mm"],
        df["mahoney_max_mm"],
        color="#16a34a",
        alpha=0.12,
        lw=0,
        label="Mahoney height range",
        zorder=2,
    )
    ax.plot(
        df["time_utc"],
        df["era5_mean_mm"],
        color="#2563eb",
        lw=2.0,
        label="ERA5 spherical height",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["mahoney_mean_mm"],
        color="#16a34a",
        lw=2.0,
        label="Mahoney numerical height",
        zorder=4,
    )


def plot_closure_station_interp(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.plot(
        df["time_utc"],
        df["profile_minus_saas_era5_spherical_mm"],
        color="#2563eb",
        lw=2.0,
        label="ERA5 spherical interp",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["profile_minus_saas_mahoney_numerical_mm"],
        color="#16a34a",
        lw=2.0,
        label="Mahoney numerical interp",
        zorder=4,
    )


def save_profile_saas_closure_plot(
    event_id: str,
    event: dict[str, Any],
    event_ts: pd.DataFrame,
    event_station_ts: pd.DataFrame | None = None,
) -> Path | None:
    panels = []
    for source in ["sounding", "era5"]:
        source_df = event_ts[event_ts["source"] == source].copy()
        if not source_df.empty:
            panel_name = "sounding" if source == "sounding" else "ERA5 grid"
            panels.append((panel_name, "range", source_df.sort_values("time_utc")))
    if event_station_ts is not None and not event_station_ts.empty:
        panels.append(
            (
                "era5 station interpolation",
                "station_interp",
                event_station_ts.sort_values("time_utc"),
            )
        )
    if not panels:
        return None

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13.5, max(3.2, 3.1 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for index, (source, panel_kind, source_df) in enumerate(panels):
        add_stage_background(axes[index], event, show_labels=index == 0)
        if panel_kind == "station_interp":
            plot_closure_station_interp(axes[index], source_df)
        else:
            plot_closure_source(axes[index], source_df)
        axes[index].set_title(
            f"{chr(97 + index)}  {source}",
            loc="left",
            fontweight="bold",
        )
        axes[index].legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            facecolor="white",
            edgecolor="#cbd5e1",
            framealpha=0.92,
            ncol=1,
        )

    for ax in axes:
        polish_closure_axis(ax, event)

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    display_name = event.get("display_name", event_id.title())
    region_name = event.get("region_name", "")
    if region_name:
        title = f"Profile-Saastamoinen Closure During Typhoon {display_name} Around {region_name}"
    else:
        title = f"Profile-Saastamoinen Closure During Typhoon {display_name}"
    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.10)
    fig.text(
        0.5,
        1.01,
        "Profile ZHD + top Saastamoinen compensation - bottom-layer Saastamoinen ZHD",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#475569",
    )

    OUT_CLOSURE_TIMESERIES_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_CLOSURE_TIMESERIES_FIG_DIR / f"{event_id}_profile_minus_bottom_saas_mahoney_vs_era5.png"
    fig.savefig(out_png, dpi=OUT_CLOSURE_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_profile_saas_closure_figures(
    events: list[dict[str, Any]],
    timeseries: pd.DataFrame,
    station_timeseries: pd.DataFrame | None = None,
) -> list[Path]:
    if timeseries.empty:
        return []
    event_map = {str(event["id"]).lower(): event for event in events}
    written: list[Path] = []
    for event_id in sorted(set(timeseries["event_id"]) & set(event_map)):
        event_ts = timeseries[timeseries["event_id"] == event_id].copy()
        if station_timeseries is None or station_timeseries.empty:
            event_station_ts = None
        else:
            event_station_ts = station_timeseries[
                station_timeseries["event_id"] == event_id
            ].copy()
        out_png = save_profile_saas_closure_plot(
            event_id,
            event_map[event_id],
            event_ts,
            event_station_ts,
        )
        if out_png is not None:
            written.append(out_png)
    return written


def check_north_south_symmetry(points: pd.DataFrame) -> dict[str, float]:
    pos = points[points["latitude_deg"] >= 0.0].copy()
    neg = points[points["latitude_deg"] <= 0.0].copy()
    neg["latitude_deg"] = -neg["latitude_deg"]
    merged = pos.merge(
        neg,
        on=["latitude_deg", "z_true_m"],
        suffixes=("_north", "_south"),
    )
    checks: dict[str, float] = {}
    for col in ["h_reference_m", "z_mahoney_m", "z_era5_m", "method_difference_m"]:
        checks[f"max_abs_north_south_delta_{col}"] = float(
            (merged[f"{col}_north"] - merged[f"{col}_south"]).abs().max()
        )
    return checks


def write_checks(points: pd.DataFrame, actual: pd.DataFrame) -> None:
    checks = check_north_south_symmetry(points)
    max_mahoney_error_mm = float(points["abs_error_mahoney_m"].max() * 1000.0)
    max_era5_error_mm = float(points["abs_error_era5_m"].max() * 1000.0)

    lines = [
        "Height conversion validation: Mahoney/WGS-84 vs ERA5 spherical approximation",
        "",
        "Reference statement:",
        REFERENCE_NOTE_ZH,
        REFERENCE_NOTE_EN,
        "",
        f"Synthetic point rows: {len(points)}",
        f"Actual profile-height rows: {len(actual)}",
        f"Max absolute Mahoney inversion error: {max_mahoney_error_mm:.6e} mm",
        f"Max absolute ERA5 spherical error: {max_era5_error_mm:.6e} mm",
        "Actual profile method_difference = ERA5 spherical height - Mahoney numerical-integral inverse height.",
        "Mahoney closed-form values are retained only as an auxiliary approximation check.",
        "Delay comparison uses the existing ZHD hydrostatic-delay formulas, not full ZTD with wet delay.",
    ]
    for key, value in checks.items():
        lines.append(f"{key}: {value:.6e}")
    OUT_CHECKS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()

    synthetic_points = build_synthetic_points()
    synthetic_summary = build_synthetic_summary(synthetic_points)
    actual_points = pd.DataFrame()
    actual_summary = pd.DataFrame()
    actual_timeseries = pd.DataFrame()
    delay_profiles = build_delay_profiles(events)
    delay_summary = build_delay_summary(delay_profiles)
    delay_timeseries = build_delay_timeseries(delay_profiles)
    closure_profiles = build_profile_saas_closure_profiles(delay_profiles)
    closure_summary = build_profile_saas_closure_summary(closure_profiles)
    closure_timeseries = build_profile_saas_closure_timeseries(closure_profiles)
    closure_station_timeseries = build_era5_station_closure_timeseries(
        closure_profiles,
        events,
    )

    synthetic_points.to_csv(OUT_SYNTHETIC_POINTS_CSV, index=False, encoding="utf-8-sig")
    synthetic_summary.to_csv(OUT_SYNTHETIC_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    actual_points.to_csv(OUT_ACTUAL_POINTS_CSV, index=False, encoding="utf-8-sig")
    actual_summary.to_csv(OUT_ACTUAL_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    actual_timeseries.to_csv(OUT_ACTUAL_TIMESERIES_CSV, index=False, encoding="utf-8-sig")
    delay_profiles.to_csv(OUT_DELAY_PROFILES_CSV, index=False, encoding="utf-8-sig")
    delay_summary.to_csv(OUT_DELAY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    delay_timeseries.to_csv(OUT_DELAY_TIMESERIES_CSV, index=False, encoding="utf-8-sig")
    closure_profiles.to_csv(OUT_CLOSURE_PROFILES_CSV, index=False, encoding="utf-8-sig")
    closure_summary.to_csv(OUT_CLOSURE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    closure_timeseries.to_csv(OUT_CLOSURE_TIMESERIES_CSV, index=False, encoding="utf-8-sig")
    closure_station_timeseries.to_csv(
        OUT_CLOSURE_ERA5_STATION_TIMESERIES_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    write_checks(synthetic_points, actual_points)
    synthetic_figures = plot_synthetic_figures(synthetic_points)
    timeseries_figures: list[Path] = []
    delay_figures: list[Path] = []
    closure_figures = plot_profile_saas_closure_figures(
        events,
        closure_timeseries,
        closure_station_timeseries,
    )

    print("Height conversion validation complete.")
    print(REFERENCE_NOTE_ZH)
    print(REFERENCE_NOTE_EN)
    print(OUT_SYNTHETIC_POINTS_CSV)
    print(OUT_SYNTHETIC_SUMMARY_CSV)
    print(OUT_ACTUAL_POINTS_CSV)
    print(OUT_ACTUAL_SUMMARY_CSV)
    print(OUT_ACTUAL_TIMESERIES_CSV)
    print(OUT_DELAY_PROFILES_CSV)
    print(OUT_DELAY_SUMMARY_CSV)
    print(OUT_DELAY_TIMESERIES_CSV)
    print(OUT_CLOSURE_PROFILES_CSV)
    print(OUT_CLOSURE_SUMMARY_CSV)
    print(OUT_CLOSURE_TIMESERIES_CSV)
    print(OUT_CLOSURE_ERA5_STATION_TIMESERIES_CSV)
    print(OUT_CHECKS_TXT)
    print(f"Synthetic figures written: {len(synthetic_figures)}")
    print(OUT_SYNTHETIC_FIG_DIR)
    print(f"Height-difference event time-series figures written: {len(timeseries_figures)}")
    print(OUT_TIMESERIES_FIG_DIR)
    print(f"Delay-comparison event time-series figures written: {len(delay_figures)}")
    print(OUT_DELAY_TIMESERIES_FIG_DIR)
    print(f"Profile-Saastamoinen closure event time-series figures written: {len(closure_figures)}")
    print(OUT_CLOSURE_TIMESERIES_FIG_DIR)


if __name__ == "__main__":
    main()
