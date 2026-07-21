from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import calc_zhd_sounding as base


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_JSON = Path(__file__).with_name("typhoon_events.json")
OUT_DIR = ROOT_DIR / "out1"

OUT_PROFILES_CSV = OUT_DIR / "zhd_dz_experiment_sounding_profiles.csv"
OUT_SUMMARY_CSV = OUT_DIR / "zhd_dz_experiment_sounding_summary.csv"
OUT_TIMESERIES_PNG = OUT_DIR / "zhd_dz_experiment_sounding_timeseries.png"
OUT_HISTOGRAM_PNG = OUT_DIR / "zhd_dz_experiment_sounding_histogram.png"

GEOPOTENTIAL_EARTH_RADIUS_M = 6_356_766.0
REPRO_TOL_M = 1e-12

METRIC_COLS = [
    "current_minus_saas_mm",
    "geometric_minus_saas_mm",
    "geometric_correction_mm",
    "current_interval_closure_mm",
    "geometric_interval_closure_mm",
    "dz_ratio",
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_events() -> list[dict[str, Any]]:
    with CONFIG_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)["events"]


def expected_sounding_combined(event: dict[str, Any]) -> Path:
    station = str(event["station"]["id"])
    start = datetime.fromisoformat(event["sounding"]["start_time"])
    end = datetime.fromisoformat(event["sounding"]["end_time"])
    return resolve_path(event["sounding"]["data_dir"]) / (
        f"{station}_{start:%Y%m%d%H}_{end:%Y%m%d%H}_combined.csv"
    )


def expected_sounding_qc_combined(event: dict[str, Any]) -> Path:
    combined = expected_sounding_combined(event)
    return combined.with_name(f"{combined.stem}_qc.csv")


def choose_sounding_input(event: dict[str, Any]) -> Path:
    qc = expected_sounding_qc_combined(event)
    if qc.exists():
        return qc
    return expected_sounding_combined(event)


def geometric_height(geopotential_height_m: np.ndarray) -> np.ndarray:
    return (
        GEOPOTENTIAL_EARTH_RADIUS_M
        * geopotential_height_m
        / (GEOPOTENTIAL_EARTH_RADIUS_M - geopotential_height_m)
    )


def quality_flags(
    duplicate_height_count: int,
    non_positive_interval_count: int,
    nonfinite_count: int,
    top_pressure_hpa: float | None,
    n_layers: int,
) -> str:
    flags: list[str] = []
    if duplicate_height_count:
        flags.append("duplicate_height")
    if non_positive_interval_count:
        flags.append("non_positive_height_interval")
    if nonfinite_count:
        flags.append("nan_or_inf")
    if top_pressure_hpa is not None and top_pressure_hpa > 100.0:
        flags.append("top_pressure_gt_100hPa")
    if n_layers < 2:
        flags.append("insufficient_layers")
    return ";".join(flags) if flags else "ok"


def calculate_profile_experiment(
    event_id: str,
    group: pd.DataFrame,
    time_utc: object,
) -> dict[str, object]:
    valid = group.dropna(
        subset=[base.P_COL, base.H_COL, base.TC_COL, base.RH_PERCENT_COL]
    ).copy()

    row: dict[str, object] = {
        "event_id": event_id,
        "time_utc": time_utc,
        "n_valid_input_layers": len(valid),
    }

    if valid.empty:
        row.update(
            {
                "n_layers": 0,
                "duplicate_height_count": 0,
                "non_positive_interval_count": 0,
                "nonfinite_count": 0,
                "quality_flag": "insufficient_layers",
            }
        )
        print(
            f"Sounding profile {event_id} {time_utc}: "
            "n_layers=0 duplicate_heights=0 non_positive_intervals=0 nonfinite=0"
        )
        return row

    bottom_layer = valid.loc[valid[base.P_COL].idxmax()]
    longitude = float(bottom_layer[base.LON_COL])
    latitude = float(bottom_layer[base.LAT_COL])
    bottom_pressure_hpa = float(bottom_layer[base.P_COL])
    bottom_height_m = float(bottom_layer[base.H_COL])
    zhd_saas_bottom_m = base.saastamoinen_zhd(
        bottom_pressure_hpa,
        bottom_height_m,
        latitude,
    )

    profile = valid.sort_values(base.H_COL).copy()
    profile["temperature_K"] = profile[base.TC_COL] + 273.15
    profile["RH"] = profile[base.RH_PERCENT_COL] / 100.0
    profile["e"] = base.saturation_water_vapor_pressure(
        profile[base.P_COL],
        profile["temperature_K"],
        profile["RH"],
    )
    profile["Nh"] = (
        77.6890
        * (profile[base.P_COL] - 0.37803 * profile["e"])
        / profile["temperature_K"]
    )

    nonfinite_count = int(
        (~np.isfinite(profile[[base.P_COL, base.H_COL, "temperature_K", "e", "Nh"]]))
        .to_numpy()
        .sum()
    )
    duplicate_height_count = int(profile[base.H_COL].duplicated().sum())

    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(subset=["Nh"])
    profile = profile[profile["Nh"] > 0.0].copy()
    n_layers = len(profile)

    if n_layers < 2:
        row.update(
            {
                "longitude": longitude,
                "latitude": latitude,
                "zhd_saas_bottom_m": float(zhd_saas_bottom_m),
                "n_layers": n_layers,
                "duplicate_height_count": duplicate_height_count,
                "non_positive_interval_count": 0,
                "nonfinite_count": nonfinite_count,
                "bottom_pressure_hPa": bottom_pressure_hpa,
                "bottom_height_m": bottom_height_m,
                "quality_flag": quality_flags(
                    duplicate_height_count,
                    0,
                    nonfinite_count,
                    None,
                    n_layers,
                ),
            }
        )
        print(
            f"Sounding profile {event_id} {time_utc}: "
            f"n_layers={n_layers} duplicate_heights={duplicate_height_count} "
            f"non_positive_intervals=0 nonfinite={nonfinite_count}"
        )
        return row

    height = profile[base.H_COL].to_numpy(dtype=float)
    nh = profile["Nh"].to_numpy(dtype=float)

    dz_current = np.diff(height)
    height_geometric = geometric_height(height)
    dz_geometric = np.diff(height_geometric)

    nh_lower = nh[:-1]
    nh_upper = nh[1:]
    log_nh = base.log_mean(nh_lower, nh_upper)

    valid_interval = dz_current >= 0.0
    zhd_int_current_m = 1e-6 * float(
        np.sum(dz_current[valid_interval] * log_nh[valid_interval])
    )
    zhd_int_geometric_m = 1e-6 * float(
        np.sum(dz_geometric[valid_interval] * log_nh[valid_interval])
    )

    top_layer = profile.loc[profile[base.H_COL].idxmax()]
    top_pressure_hpa = float(top_layer[base.P_COL])
    top_height_m = float(top_layer[base.H_COL])
    zhd_top_saas_m = base.saastamoinen_zhd(top_pressure_hpa, top_height_m, latitude)

    zhd_profile_current_m = zhd_int_current_m + zhd_top_saas_m
    zhd_profile_geometric_m = zhd_int_geometric_m + zhd_top_saas_m
    expected_interval_saas_m = zhd_saas_bottom_m - zhd_top_saas_m

    total_dH_m = float(np.sum(dz_current[valid_interval]))
    total_dz_geometric_m = float(np.sum(dz_geometric[valid_interval]))
    dz_ratio = (
        total_dz_geometric_m / total_dH_m
        if not np.isclose(total_dH_m, 0.0)
        else np.nan
    )
    non_positive_interval_count = int(np.sum(dz_current <= 0.0))

    row.update(
        {
            "longitude": longitude,
            "latitude": latitude,
            "zhd_saas_bottom_m": float(zhd_saas_bottom_m),
            "zhd_top_saas_m": float(zhd_top_saas_m),
            "zhd_int_current_m": zhd_int_current_m,
            "zhd_int_geometric_m": zhd_int_geometric_m,
            "zhd_profile_current_m": float(zhd_profile_current_m),
            "zhd_profile_geometric_m": float(zhd_profile_geometric_m),
            "expected_interval_saas_m": float(expected_interval_saas_m),
            "current_minus_saas_mm": 1000.0
            * (zhd_profile_current_m - zhd_saas_bottom_m),
            "geometric_minus_saas_mm": 1000.0
            * (zhd_profile_geometric_m - zhd_saas_bottom_m),
            "current_interval_closure_mm": 1000.0
            * (zhd_int_current_m - expected_interval_saas_m),
            "geometric_interval_closure_mm": 1000.0
            * (zhd_int_geometric_m - expected_interval_saas_m),
            "geometric_correction_mm": 1000.0
            * (zhd_int_geometric_m - zhd_int_current_m),
            "total_dH_m": total_dH_m,
            "total_dz_geometric_m": total_dz_geometric_m,
            "dz_ratio": dz_ratio,
            "n_layers": n_layers,
            "bottom_pressure_hPa": bottom_pressure_hpa,
            "top_pressure_hPa": top_pressure_hpa,
            "bottom_height_m": bottom_height_m,
            "top_height_m": top_height_m,
            "duplicate_height_count": duplicate_height_count,
            "non_positive_interval_count": non_positive_interval_count,
            "nonfinite_count": nonfinite_count,
            "quality_flag": quality_flags(
                duplicate_height_count,
                non_positive_interval_count,
                nonfinite_count,
                top_pressure_hpa,
                n_layers,
            ),
        }
    )

    print(
        f"Sounding profile {event_id} {time_utc}: "
        f"n_layers={n_layers} duplicate_heights={duplicate_height_count} "
        f"non_positive_intervals={non_positive_interval_count} "
        f"nonfinite={nonfinite_count}"
    )
    return row


def process_event(event: dict[str, Any]) -> pd.DataFrame:
    event_id = event["id"]
    input_csv = choose_sounding_input(event)
    if not input_csv.exists():
        print(f"Skip sounding event {event_id}: missing input {input_csv}")
        return pd.DataFrame()

    df = base.read_sounding_csv(input_csv)
    original_saas, original_profile = base.calculate_zhd(df)

    rows: list[dict[str, object]] = []
    for time_utc, group in df.groupby(base.TIME_COL, sort=True):
        rows.append(calculate_profile_experiment(event_id, group, time_utc))

    out = pd.DataFrame(rows)
    if not out.empty and not original_profile.empty:
        original_profile = original_profile.rename(
            columns={"ZHD_m": "original_profile_current_m"}
        )
        original_saas = original_saas.rename(columns={"ZHD_m": "original_saas_m"})
        out = out.merge(
            original_profile[["time_utc", "original_profile_current_m"]],
            on="time_utc",
            how="left",
        )
        out = out.merge(
            original_saas[["time_utc", "original_saas_m"]],
            on="time_utc",
            how="left",
        )
        out["current_reproduction_error_m"] = (
            out["zhd_profile_current_m"] - out["original_profile_current_m"]
        )
        out["saas_reproduction_error_m"] = (
            out["zhd_saas_bottom_m"] - out["original_saas_m"]
        )
    else:
        out["original_profile_current_m"] = np.nan
        out["original_saas_m"] = np.nan
        out["current_reproduction_error_m"] = np.nan
        out["saas_reproduction_error_m"] = np.nan

    out["input_csv"] = str(input_csv)
    return out


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
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "MAE": float(clean.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(clean.to_numpy(dtype=float) ** 2))),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q05": float(clean.quantile(0.05)),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "q95": float(clean.quantile(0.95)),
    }


def build_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event_id, group in profiles.groupby("event_id", dropna=False):
        for metric in METRIC_COLS:
            rows.append(
                {
                    "event_id": event_id,
                    "metric": metric,
                    **metric_stats(group[metric]),
                }
            )
    for metric in METRIC_COLS:
        rows.append({"event_id": "ALL", "metric": metric, **metric_stats(profiles[metric])})
    return pd.DataFrame(rows)


def plot_outputs(profiles: pd.DataFrame) -> None:
    plot_df = profiles.dropna(
        subset=["time_utc", "current_minus_saas_mm", "geometric_minus_saas_mm"]
    ).copy()
    if plot_df.empty:
        print("Skip sounding plots: no valid rows.")
        return

    plot_df["time_utc"] = pd.to_datetime(plot_df["time_utc"])

    plt.figure(figsize=(12, 6))
    for _, group in plot_df.groupby("event_id", sort=True):
        group = group.sort_values("time_utc")
        plt.plot(group["time_utc"], group["current_minus_saas_mm"], marker="o")
        plt.plot(group["time_utc"], group["geometric_minus_saas_mm"], marker="s")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Profile ZHD - Saastamoinen ZHD (mm)")
    plt.title("Sounding dz Experiment Time Series")
    plt.legend(["current dz", "geometric dz", "0 mm"], loc="best")
    plt.tight_layout()
    plt.savefig(OUT_TIMESERIES_PNG, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(
        [
            plot_df["current_minus_saas_mm"].dropna(),
            plot_df["geometric_minus_saas_mm"].dropna(),
        ],
        bins=40,
        label=["current dz", "geometric dz"],
    )
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Profile ZHD - Saastamoinen ZHD (mm)")
    plt.ylabel("Count")
    plt.title("Sounding dz Experiment Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HISTOGRAM_PNG, dpi=300)
    plt.close()


def print_terminal_conclusion(profiles: pd.DataFrame) -> None:
    current = profiles["current_minus_saas_mm"].dropna()
    geometric = profiles["geometric_minus_saas_mm"].dropna()
    correction = profiles["geometric_correction_mm"].dropna()

    current_mean = float(current.mean())
    geometric_mean = float(geometric.mean())
    correction_mean = float(correction.mean())
    current_rmse = float(np.sqrt(np.mean(current.to_numpy(dtype=float) ** 2)))
    geometric_rmse = float(np.sqrt(np.mean(geometric.to_numpy(dtype=float) ** 2)))
    rmse_reduction = current_rmse - geometric_rmse
    mean_reduction_ratio = (
        1.0 - abs(geometric_mean) / abs(current_mean)
        if not np.isclose(current_mean, 0.0)
        else np.nan
    )
    rmse_reduction_ratio = (
        1.0 - geometric_rmse / current_rmse
        if not np.isclose(current_rmse, 0.0)
        else np.nan
    )

    max_repro_error = profiles["current_reproduction_error_m"].abs().max()
    reproduces = bool(max_repro_error <= REPRO_TOL_M)

    print("\nSounding experiment conclusion")
    print(f"Current mean difference: {current_mean:.6f} mm")
    print(f"Geometric mean difference: {geometric_mean:.6f} mm")
    print(f"Mean geometric correction: {correction_mean:.6f} mm")
    print(f"RMSE reduction: {rmse_reduction:.6f} mm")
    print(f"Current method reproduces original program: {reproduces}")
    print(f"Max reproduction error: {max_repro_error:.3e} m")

    if mean_reduction_ratio >= 0.5 or rmse_reduction_ratio >= 0.5:
        print("实验支持 dz 定义是主要差异来源。")
    else:
        print("仅修正 dz 不能完全解释差异，需要继续检查层间静力一致性、顶部补偿和离散积分方法。")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_profiles = [process_event(event) for event in load_events()]
    profiles = pd.concat(
        [df for df in all_profiles if not df.empty],
        ignore_index=True,
        sort=False,
    )
    if profiles.empty:
        raise RuntimeError("No sounding profiles were processed.")

    top_gt_100_count = int((profiles["top_pressure_hPa"] > 100.0).sum())
    print(f"Sounding top pressure > 100 hPa profile count: {top_gt_100_count}")
    print(
        "Sounding current reproduction exact within 1e-12 m: "
        f"{bool(profiles['current_reproduction_error_m'].abs().max() <= REPRO_TOL_M)}"
    )

    profiles.to_csv(OUT_PROFILES_CSV, index=False, encoding="utf-8-sig")
    summary = build_summary(profiles)
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    plot_outputs(profiles)
    print_terminal_conclusion(profiles)

    print("\nSounding output files:")
    print(OUT_PROFILES_CSV)
    print(OUT_SUMMARY_CSV)
    print(OUT_TIMESERIES_PNG)
    print(OUT_HISTOGRAM_PNG)


if __name__ == "__main__":
    main()
