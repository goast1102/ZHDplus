from pathlib import Path
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


SOURCES = {
    "sounding": {
        "saas": "sounding_zhd_saas.csv",
        "profile": "sounding_zhd_profile.csv",
    },
    "era5_station_interp": {
        "saas": "era5_zhd_saas_station_interp.csv",
        "profile": "era5_zhd_profile_station_interp.csv",
    },
}


def basin_from_lon_lat(lon, lat):
    if pd.isna(lon) or pd.isna(lat):
        return "Unknown"
    if -100 <= lon <= -10 and 0 <= lat <= 50:
        return "North Atlantic / Gulf"
    if 100 <= lon <= 180 and 0 <= lat <= 45:
        return "Western North Pacific / East Asia"
    if 40 <= lon < 100 and 0 <= lat <= 35:
        return "North Indian Ocean"
    if 100 <= lon <= 180 and -45 <= lat < 0:
        return "South Pacific / Australia"
    if -180 <= lon < -100 and 0 <= lat <= 45:
        return "Central/Eastern North Pacific"
    return "Other"


def load_configs():
    with EVENT_CONFIG.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {event["id"]: event for event in data["events"]}


def read_zhd_csv(path):
    df = pd.read_csv(path, parse_dates=["time_utc"])
    required = {"time_utc", "ZHD_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("{} missing columns: {}".format(path, sorted(missing)))
    df["ZHD_m"] = pd.to_numeric(df["ZHD_m"], errors="coerce")
    keep = ["time_utc", "ZHD_m"]
    for col in ["longitude", "latitude"]:
        if col in df.columns:
            keep.append(col)
    return df[keep].dropna(subset=["time_utc", "ZHD_m"]).copy()


def pair_profile_saas(event_dir, source_name):
    files = SOURCES[source_name]
    saas_path = event_dir / files["saas"]
    profile_path = event_dir / files["profile"]
    if not saas_path.exists() or not profile_path.exists():
        return None, "missing_files"

    saas = read_zhd_csv(saas_path).rename(columns={"ZHD_m": "saas_ZHD_m"})
    profile = read_zhd_csv(profile_path).rename(columns={"ZHD_m": "profile_ZHD_m"})
    paired = saas[["time_utc", "saas_ZHD_m"]].merge(
        profile[["time_utc", "profile_ZHD_m"]],
        on="time_utc",
        how="inner",
    )
    if paired.empty:
        return paired, "no_time_pairs"
    paired["profile_minus_saas_mm"] = (
        paired["profile_ZHD_m"] - paired["saas_ZHD_m"]
    ) * 1000.0
    paired["abs_profile_minus_saas_mm"] = paired["profile_minus_saas_mm"].abs()
    return paired.sort_values("time_utc").reset_index(drop=True), "ok"


def stage_defs(plot_cfg):
    time_start = pd.Timestamp(plot_cfg["time_start"])
    pre_end = pd.Timestamp(plot_cfg["pre_end"])
    impact_end = pd.Timestamp(plot_cfg["impact_end"])
    time_end = pd.Timestamp(plot_cfg["time_end"])
    return [
        ("pre", "typhoon_before", time_start, pre_end, True, False),
        ("during", "typhoon_during", pre_end, impact_end, True, True),
        ("post", "typhoon_after", impact_end, time_end, False, True),
    ]


def filter_stage(df, start, end, include_start, include_end):
    mask = pd.Series(True, index=df.index)
    if include_start:
        mask &= df["time_utc"] >= start
    else:
        mask &= df["time_utc"] > start
    if include_end:
        mask &= df["time_utc"] <= end
    else:
        mask &= df["time_utc"] < end
    return df[mask].copy()


def empty_metrics():
    return {
        "n_pairs": 0,
        "mean_saas_ZHD_mm": np.nan,
        "mean_profile_ZHD_mm": np.nan,
        "mean_profile_minus_saas_mm": np.nan,
        "median_profile_minus_saas_mm": np.nan,
        "mae_profile_minus_saas_mm": np.nan,
        "rmse_profile_minus_saas_mm": np.nan,
        "std_profile_minus_saas_mm": np.nan,
        "min_profile_minus_saas_mm": np.nan,
        "max_profile_minus_saas_mm": np.nan,
        "max_abs_profile_minus_saas_mm": np.nan,
        "peak_abs_time_utc": pd.NaT,
        "peak_profile_minus_saas_mm": np.nan,
        "mean_abs_relative_to_saas_percent": np.nan,
    }


def metrics(df):
    if df is None or df.empty:
        return empty_metrics()
    diff = df["profile_minus_saas_mm"]
    abs_diff = df["abs_profile_minus_saas_mm"]
    peak_idx = abs_diff.idxmax()
    out = empty_metrics()
    out.update(
        {
            "n_pairs": int(len(df)),
            "mean_saas_ZHD_mm": df["saas_ZHD_m"].mean() * 1000.0,
            "mean_profile_ZHD_mm": df["profile_ZHD_m"].mean() * 1000.0,
            "mean_profile_minus_saas_mm": diff.mean(),
            "median_profile_minus_saas_mm": diff.median(),
            "mae_profile_minus_saas_mm": abs_diff.mean(),
            "rmse_profile_minus_saas_mm": float(np.sqrt(np.mean(diff**2))),
            "std_profile_minus_saas_mm": diff.std(ddof=1) if len(df) > 1 else 0.0,
            "min_profile_minus_saas_mm": diff.min(),
            "max_profile_minus_saas_mm": diff.max(),
            "max_abs_profile_minus_saas_mm": abs_diff.max(),
            "peak_abs_time_utc": df.loc[peak_idx, "time_utc"],
            "peak_profile_minus_saas_mm": df.loc[peak_idx, "profile_minus_saas_mm"],
            "mean_abs_relative_to_saas_percent": (
                abs_diff.mean() / (df["saas_ZHD_m"].mean() * 1000.0) * 100.0
            ),
        }
    )
    return out


def main():
    configs = load_configs()
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

            station = cfg.get("station", {})
            lat = station.get("latitude", np.nan)
            lon = station.get("longitude", np.nan)
            base_row = {
                "event": event_dir.name,
                "display_name": cfg.get("display_name", event_dir.name),
                "region_name": cfg.get("region_name", ""),
                "station": str(station.get("id", "")),
                "station_lat": lat,
                "station_lon": lon,
                "basin": basin_from_lon_lat(lon, lat),
                "root": root_label,
                "event_dir": str(event_dir),
            }

            for source_name in ["sounding", "era5_station_interp"]:
                paired, status = pair_profile_saas(event_dir, source_name)
                for stage_code, stage_name, start, end, inc_start, inc_end in stage_defs(
                    cfg["plot"]
                ):
                    stage_df = (
                        filter_stage(paired, start, end, inc_start, inc_end)
                        if paired is not None and not paired.empty
                        else pd.DataFrame()
                    )
                    row = dict(base_row)
                    row.update(
                        {
                            "source": source_name,
                            "comparison": "profile_minus_saas",
                            "stage": stage_code,
                            "stage_name": stage_name,
                            "stage_start_utc": start,
                            "stage_end_utc": end,
                            "data_status": status,
                        }
                    )
                    row.update(metrics(stage_df))
                    rows.append(row)

    result = pd.DataFrame(rows)
    out_path = BASE / "out" / "profile_saas_stage_differences_all_events.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("rows={}".format(len(result)))
    print("events={}".format(result["event"].nunique() if not result.empty else 0))
    print("csv={}".format(out_path))
    if missing_config:
        print("missing_config={}".format(", ".join(missing_config)))
    print("\nSource/stage rows with data:")
    print(
        result.groupby(["source", "stage"])
        .agg(
            rows=("event", "count"),
            events_with_pairs=("n_pairs", lambda s: int((s > 0).sum())),
            median_n_pairs=("n_pairs", "median"),
            median_mae_mm=("mae_profile_minus_saas_mm", "median"),
            p75_mae_mm=("mae_profile_minus_saas_mm", lambda s: s.quantile(0.75)),
            median_mean_diff_mm=("mean_profile_minus_saas_mm", "median"),
        )
        .to_string(float_format=lambda x: "{:.3f}".format(x))
    )


if __name__ == "__main__":
    main()
