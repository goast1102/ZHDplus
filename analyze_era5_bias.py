from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ROOTS = [BASE / "out", BASE / "out" / "new"]
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
    if -180 <= lon < -100 and 0 <= lat <= 45:
        return "Central/East Pacific"
    if 40 <= lon < 100 and -5 <= lat <= 35:
        return "North Indian"
    return "other"


def event_summary(event_dir: Path, root_label: str) -> Optional[dict]:
    rs_path = event_dir / "sounding_zhd_profile.csv"
    era_path = event_dir / "era5_zhd_profile_station_interp.csv"
    q_path = event_dir / "sounding_quality_summary.csv"
    if not (rs_path.exists() and era_path.exists()):
        return None

    rs = pd.read_csv(rs_path, parse_dates=["time_utc"])
    era = pd.read_csv(era_path, parse_dates=["time_utc"])
    if rs.empty or era.empty:
        return None

    merged = (
        rs[["time_utc", "ZHD_m", "longitude", "latitude"]]
        .rename(
            columns={
                "ZHD_m": "sounding_ZHD_m",
                "longitude": "sounding_lon",
                "latitude": "sounding_lat",
            }
        )
        .merge(
            era[["time_utc", "ZHD_m"]].rename(columns={"ZHD_m": "era5_ZHD_m"}),
            on="time_utc",
            how="inner",
        )
    )
    if merged.empty:
        return None

    merged["bias_mm"] = (merged["era5_ZHD_m"] - merged["sounding_ZHD_m"]) * 1000.0
    peak_idx = merged["bias_mm"].abs().idxmax()

    station = ""
    qc_total = np.nan
    qc_pass = np.nan
    if q_path.exists():
        q = pd.read_csv(q_path)
        if "station" in q.columns and q["station"].notna().any():
            station = str(q["station"].dropna().iloc[0])
        if "passes_qc" in q.columns:
            qc_total = len(q)
            qc_pass = int(q["passes_qc"].astype(str).str.lower().eq("true").sum())

    lon = float(merged["sounding_lon"].mean())
    lat = float(merged["sounding_lat"].mean())
    abs_bias = merged["bias_mm"].abs()
    rmse = float(np.sqrt(np.mean(merged["bias_mm"] ** 2)))
    max_abs = abs_bias.max()
    if rmse >= 20.0:
        bias_class = "persistent_obvious"
    elif max_abs >= 35.0:
        bias_class = "peak_only_obvious"
    elif rmse >= 10.0:
        bias_class = "moderate"
    else:
        bias_class = "small"
    return {
        "event": event_dir.name,
        "root": root_label,
        "station": station,
        "start_utc": merged["time_utc"].min(),
        "end_utc": merged["time_utc"].max(),
        "year": int(merged["time_utc"].dt.year.mode().iloc[0]),
        "month": int(merged["time_utc"].dt.month.mode().iloc[0]),
        "lat": lat,
        "lon": lon,
        "basin": basin_from_lon_lat(lon, lat),
        "n_match": len(merged),
        "qc_pass": qc_pass,
        "qc_total": qc_total,
        "mean_bias_mm": merged["bias_mm"].mean(),
        "median_bias_mm": merged["bias_mm"].median(),
        "mae_mm": abs_bias.mean(),
        "rmse_mm": rmse,
        "p90_abs_bias_mm": abs_bias.quantile(0.90),
        "max_abs_bias_mm": max_abs,
        "peak_time_utc": merged.loc[peak_idx, "time_utc"],
        "peak_bias_mm": merged.loc[peak_idx, "bias_mm"],
        "frac_abs_gt10mm": float((abs_bias > 10.0).mean()),
        "frac_abs_gt20mm": float((abs_bias > 20.0).mean()),
        "bias_class": bias_class,
    }


def main() -> None:
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        root_label = "out/new" if root.name == "new" else "out"
        for event_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if event_dir.name in SKIP_DIRS:
                continue
            row = event_summary(event_dir, root_label)
            if row is not None:
                rows.append(row)

    summary = pd.DataFrame(rows).sort_values("rmse_mm", ascending=False)
    output = BASE / "out" / "era5_bias_event_summary.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", 40)
    print(f"events={len(summary)}")
    print(f"summary_csv={output}")
    print("\nTop events by RMSE:")
    print(
        summary[
            [
                "event",
                "basin",
                "start_utc",
                "end_utc",
                "lat",
                "lon",
                "n_match",
                "mean_bias_mm",
                "mae_mm",
                "rmse_mm",
                "max_abs_bias_mm",
                "peak_time_utc",
                "peak_bias_mm",
            ]
        ]
        .head(35)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )
    print("\nMetric quantiles:")
    print(
        summary[
            ["mae_mm", "rmse_mm", "max_abs_bias_mm", "frac_abs_gt10mm", "frac_abs_gt20mm"]
        ]
        .quantile([0.5, 0.75, 0.9, 0.95, 0.98])
        .to_string(float_format=lambda x: f"{x:.2f}")
    )
    print("\nBy basin:")
    print(
        summary.groupby("basin")
        .agg(
            events=("event", "count"),
            median_rmse_mm=("rmse_mm", "median"),
            p90_rmse_mm=("rmse_mm", lambda s: s.quantile(0.9)),
            median_max_abs_mm=("max_abs_bias_mm", "median"),
            persistent_obvious_events=("rmse_mm", lambda s: int((s >= 20.0).sum())),
        )
        .sort_values("median_rmse_mm", ascending=False)
        .to_string(float_format=lambda x: f"{x:.2f}")
    )
    print("\nBy year:")
    print(
        summary.groupby("year")
        .agg(
            events=("event", "count"),
            median_rmse_mm=("rmse_mm", "median"),
            p90_rmse_mm=("rmse_mm", lambda s: s.quantile(0.9)),
            persistent_obvious_events=("rmse_mm", lambda s: int((s >= 20.0).sum())),
        )
        .to_string(float_format=lambda x: f"{x:.2f}")
    )
    print("\nBy month:")
    print(
        summary.groupby("month")
        .agg(
            events=("event", "count"),
            median_rmse_mm=("rmse_mm", "median"),
            p90_rmse_mm=("rmse_mm", lambda s: s.quantile(0.9)),
            persistent_obvious_events=("rmse_mm", lambda s: int((s >= 20.0).sum())),
        )
        .to_string(float_format=lambda x: f"{x:.2f}")
    )


if __name__ == "__main__":
    main()
