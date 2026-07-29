from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
EVENT_CONFIG = BASE / "code" / "typhoon_events.json"
OUT_ROOTS = [BASE / "out", BASE / "out" / "new"]
SKIP_DIRS = {
    "new",
    "method_c_sounding_era5_zhd",
    "method_c_sounding_era5_zhd_check",
    "verification",
}


def basin_from_lon_lat(lon: float, lat: float) -> str:
    if np.isnan(lon) or np.isnan(lat):
        return "unknown"
    if 100 <= lon <= 180 and -5 <= lat <= 45:
        return "WNP/East Asia"
    if -100 <= lon <= -50 and 0 <= lat <= 45:
        return "Atlantic/Gulf"
    if 40 <= lon < 100 and -5 <= lat <= 35:
        return "North Indian"
    return "other"


def load_event_config() -> dict:
    with EVENT_CONFIG.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {event["id"]: event for event in data["events"]}


def read_zhd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time_utc"])
    df["ZHD_m"] = pd.to_numeric(df["ZHD_m"], errors="coerce")
    return df.dropna(subset=["time_utc", "ZHD_m"]).copy()


def method_bias(event_dir: Path, method: str) -> Optional[pd.DataFrame]:
    rs = event_dir / f"sounding_zhd_{method}.csv"
    era = event_dir / f"era5_zhd_{method}_station_interp.csv"
    if not (rs.exists() and era.exists()):
        return None

    rs_df = read_zhd(rs)
    era_df = read_zhd(era)
    if rs_df.empty or era_df.empty:
        return None

    merged = (
        rs_df[["time_utc", "ZHD_m", "longitude", "latitude"]]
        .rename(
            columns={
                "ZHD_m": "sounding_ZHD_m",
                "longitude": "sounding_lon",
                "latitude": "sounding_lat",
            }
        )
        .merge(
            era_df[["time_utc", "ZHD_m"]].rename(columns={"ZHD_m": "era5_ZHD_m"}),
            on="time_utc",
            how="inner",
        )
    )
    if merged.empty:
        return None
    merged["method"] = method
    merged["bias_mm"] = (merged["era5_ZHD_m"] - merged["sounding_ZHD_m"]) * 1000.0
    merged["abs_bias_mm"] = merged["bias_mm"].abs()
    return merged


def stage_stats(df: pd.DataFrame, stage: str) -> dict:
    if df.empty:
        return {
            f"{stage}_n": 0,
            f"{stage}_mean_bias_mm": np.nan,
            f"{stage}_mae_mm": np.nan,
            f"{stage}_rmse_mm": np.nan,
            f"{stage}_max_abs_mm": np.nan,
            f"{stage}_peak_time_utc": pd.NaT,
            f"{stage}_peak_bias_mm": np.nan,
        }
    peak_idx = df["abs_bias_mm"].idxmax()
    return {
        f"{stage}_n": len(df),
        f"{stage}_mean_bias_mm": df["bias_mm"].mean(),
        f"{stage}_mae_mm": df["abs_bias_mm"].mean(),
        f"{stage}_rmse_mm": float(np.sqrt(np.mean(df["bias_mm"] ** 2))),
        f"{stage}_max_abs_mm": df["abs_bias_mm"].max(),
        f"{stage}_peak_time_utc": df.loc[peak_idx, "time_utc"],
        f"{stage}_peak_bias_mm": df.loc[peak_idx, "bias_mm"],
    }


def summarize_method(
    event_id: str,
    event_dir: Path,
    root_label: str,
    cfg: dict,
    method: str,
) -> Optional[dict]:
    bias = method_bias(event_dir, method)
    if bias is None:
        return None

    plot_cfg = cfg.get("plot", {})
    time_start = pd.Timestamp(plot_cfg["time_start"])
    pre_end = pd.Timestamp(plot_cfg["pre_end"])
    impact_end = pd.Timestamp(plot_cfg["impact_end"])
    time_end = pd.Timestamp(plot_cfg["time_end"])

    bias = bias[(bias["time_utc"] >= time_start) & (bias["time_utc"] <= time_end)].copy()
    if bias.empty:
        return None

    pre = bias[(bias["time_utc"] >= time_start) & (bias["time_utc"] < pre_end)]
    near = bias[(bias["time_utc"] >= pre_end) & (bias["time_utc"] <= impact_end)]
    post = bias[(bias["time_utc"] > impact_end) & (bias["time_utc"] <= time_end)]

    lon = float(bias["sounding_lon"].mean())
    lat = float(bias["sounding_lat"].mean())
    station = str(cfg.get("station", {}).get("id", ""))
    row = {
        "event": event_id,
        "root": root_label,
        "method": method,
        "display_name": cfg.get("display_name", event_id),
        "region_name": cfg.get("region_name", ""),
        "station": station,
        "lat": lat,
        "lon": lon,
        "basin": basin_from_lon_lat(lon, lat),
        "time_start": time_start,
        "pre_end": pre_end,
        "impact_end": impact_end,
        "time_end": time_end,
    }
    row.update(stage_stats(pre, "pre"))
    row.update(stage_stats(near, "near"))
    row.update(stage_stats(post, "post"))
    row["delta_near_pre_mae_mm"] = row["near_mae_mm"] - row["pre_mae_mm"]
    row["delta_near_pre_rmse_mm"] = row["near_rmse_mm"] - row["pre_rmse_mm"]
    row["delta_near_pre_max_abs_mm"] = row["near_max_abs_mm"] - row["pre_max_abs_mm"]
    row["near_pre_mae_ratio"] = (
        row["near_mae_mm"] / row["pre_mae_mm"] if row["pre_mae_mm"] > 0 else np.nan
    )
    return row


def classify_pair(group: pd.DataFrame) -> str:
    if len(group) != 2:
        return "incomplete"
    enough = (group["pre_n"] >= 2).all() and (group["near_n"] >= 2).all()
    if not enough:
        return "too_few_points"
    both_obvious = (
        (group["near_mae_mm"] >= 15.0)
        & (group["delta_near_pre_mae_mm"] >= 8.0)
        & (group["near_pre_mae_ratio"] >= 1.5)
    ).all()
    one_obvious = (
        (group["near_mae_mm"] >= 15.0)
        & (group["delta_near_pre_mae_mm"] >= 8.0)
        & (group["near_pre_mae_ratio"] >= 1.5)
    ).any()
    if both_obvious:
        return "both_methods_obvious_increase"
    if one_obvious:
        return "one_method_obvious_increase"
    return "no_obvious_increase"


def main() -> None:
    configs = load_event_config()
    rows = []
    missing_config = []
    for root in OUT_ROOTS:
        if not root.exists():
            continue
        root_label = "out/new" if root.name == "new" else "out"
        for event_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if event_dir.name in SKIP_DIRS:
                continue
            cfg = configs.get(event_dir.name)
            if cfg is None:
                missing_config.append(str(event_dir))
                continue
            for method in ["saas", "profile"]:
                row = summarize_method(event_dir.name, event_dir, root_label, cfg, method)
                if row is not None:
                    rows.append(row)

    method_summary = pd.DataFrame(rows)
    method_out = BASE / "out" / "era5_approach_bias_by_method.csv"
    method_summary.to_csv(method_out, index=False, encoding="utf-8-sig")

    classes = method_summary.groupby("event").apply(classify_pair).reset_index()
    classes = classes.rename(columns={0: "approach_bias_class"})

    event_summary = method_summary.pivot_table(
        index=[
            "event",
            "root",
            "display_name",
            "region_name",
            "station",
            "lat",
            "lon",
            "basin",
            "time_start",
            "pre_end",
            "impact_end",
            "time_end",
        ],
        columns="method",
        values=[
            "pre_n",
            "near_n",
            "pre_mae_mm",
            "near_mae_mm",
            "delta_near_pre_mae_mm",
            "near_pre_mae_ratio",
            "near_max_abs_mm",
            "near_peak_time_utc",
            "near_peak_bias_mm",
        ],
        aggfunc="first",
    )
    event_summary.columns = [f"{metric}_{method}" for metric, method in event_summary.columns]
    event_summary = event_summary.reset_index().merge(classes, on="event", how="left")
    event_summary = event_summary.sort_values(
        ["approach_bias_class", "delta_near_pre_mae_mm_saas"],
        ascending=[True, False],
    )
    event_out = BASE / "out" / "era5_approach_bias_event_summary.csv"
    event_summary.to_csv(event_out, index=False, encoding="utf-8-sig")

    obvious = event_summary[
        event_summary["approach_bias_class"] == "both_methods_obvious_increase"
    ].copy()
    watch = event_summary[
        event_summary["approach_bias_class"] == "one_method_obvious_increase"
    ].copy()

    print(f"method_rows={len(method_summary)}")
    print(f"events={len(event_summary)}")
    print(f"method_csv={method_out}")
    print(f"event_csv={event_out}")
    if missing_config:
        print("missing_config=" + ", ".join(missing_config))

    show_cols = [
        "event",
        "basin",
        "station",
        "pre_end",
        "impact_end",
        "pre_mae_mm_saas",
        "near_mae_mm_saas",
        "delta_near_pre_mae_mm_saas",
        "near_pre_mae_ratio_saas",
        "near_peak_time_utc_saas",
        "near_peak_bias_mm_saas",
        "pre_mae_mm_profile",
        "near_mae_mm_profile",
        "delta_near_pre_mae_mm_profile",
        "near_pre_mae_ratio_profile",
        "near_peak_time_utc_profile",
        "near_peak_bias_mm_profile",
    ]
    print("\nBoth methods obvious increase:")
    print(obvious[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nOne method obvious increase:")
    print(watch[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nClass counts by basin:")
    print(
        event_summary.groupby(["basin", "approach_bias_class"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()
