from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path(r"E:\ZHDplus\test\data\era5_tfsula\era5_pressure_raw_extract.csv")
OUT_LOWEST_CSV = Path(r"E:\ZHDplus\test\out\era5_lowest_layer_records.csv")
OUT_GRID_SUMMARY_CSV = Path(r"E:\ZHDplus\test\out\era5_lowest_layer_grid_summary.csv")

TIME_COL = "time_utc"
LAT_COL = "latitude"
LON_COL = "longitude"
P_COL = "pressure_hPa"
Z_COL = "geopotential_m2_s2"
H_COL = "geopotential_height_m"

REQUIRED_COLS = [TIME_COL, LAT_COL, LON_COL, P_COL, Z_COL, H_COL]


def read_era5_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    for col in [LAT_COL, LON_COL, P_COL, Z_COL, H_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=REQUIRED_COLS).copy()


def get_lowest_layers(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [TIME_COL, LAT_COL, LON_COL]
    idx = df.groupby(group_cols)[P_COL].idxmax()
    lowest = df.loc[idx, [TIME_COL, LON_COL, LAT_COL, P_COL, Z_COL, H_COL]].copy()
    return lowest.sort_values([LAT_COL, LON_COL, TIME_COL]).reset_index(drop=True)


def summarize_by_grid(lowest: pd.DataFrame) -> pd.DataFrame:
    summary = (
        lowest.groupby([LAT_COL, LON_COL], as_index=False)
        .agg(
            n_time=(TIME_COL, "nunique"),
            p_min=(P_COL, "min"),
            p_max=(P_COL, "max"),
            p_nunique=(P_COL, "nunique"),
            h_min_m=(H_COL, "min"),
            h_max_m=(H_COL, "max"),
            h_mean_m=(H_COL, "mean"),
            h_std_m=(H_COL, "std"),
            z_min_m2_s2=(Z_COL, "min"),
            z_max_m2_s2=(Z_COL, "max"),
            z_mean_m2_s2=(Z_COL, "mean"),
            z_std_m2_s2=(Z_COL, "std"),
        )
        .sort_values([LAT_COL, LON_COL])
        .reset_index(drop=True)
    )
    summary["h_range_m"] = summary["h_max_m"] - summary["h_min_m"]
    summary["z_range_m2_s2"] = summary["z_max_m2_s2"] - summary["z_min_m2_s2"]
    return summary


def print_global_checks(lowest: pd.DataFrame, summary: pd.DataFrame) -> None:
    n_profiles = len(lowest)
    n_grid = lowest[[LAT_COL, LON_COL]].drop_duplicates().shape[0]
    n_time = lowest[TIME_COL].nunique()

    print("ERA5 lowest-layer check")
    print(f"Input: {INPUT_CSV}")
    print(f"Profiles: {n_profiles}")
    print(f"Grid points: {n_grid}")
    print(f"Times: {n_time}, {lowest[TIME_COL].min()} to {lowest[TIME_COL].max()}")

    pressure_counts = lowest[P_COL].value_counts().sort_index()
    print("\nLowest-layer pressure counts:")
    print(pressure_counts.to_string())

    all_1000 = (lowest[P_COL] == 1000.0).all()
    print(f"\nIs lowest-layer pressure always 1000 hPa? {all_1000}")

    if not all_1000:
        bad = lowest[lowest[P_COL] != 1000.0]
        print("\nFirst non-1000 hPa lowest-layer records:")
        print(bad.head(20).to_string(index=False))

    print("\nLowest-layer geopotential height variation by grid:")
    print(
        summary[
            [
                LAT_COL,
                LON_COL,
                "n_time",
                "p_min",
                "p_max",
                "h_min_m",
                "h_max_m",
                "h_range_m",
                "z_min_m2_s2",
                "z_max_m2_s2",
                "z_range_m2_s2",
            ]
        ].to_string(index=False)
    )

    max_h_range = summary["h_range_m"].max()
    max_z_range = summary["z_range_m2_s2"].max()
    print(f"\nMax lowest-layer height range across grids: {max_h_range:.6f} m")
    print(f"Max lowest-layer geopotential range across grids: {max_z_range:.6f} m2/s2")
    print(f"Is lowest-layer geopotential height unchanged over time? {max_h_range == 0.0}")
    print(f"Is lowest-layer geopotential unchanged over time? {max_z_range == 0.0}")


def main() -> None:
    df = read_era5_csv(INPUT_CSV)
    lowest = get_lowest_layers(df)
    summary = summarize_by_grid(lowest)

    OUT_LOWEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    lowest.to_csv(OUT_LOWEST_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_GRID_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print_global_checks(lowest, summary)

    print("\nOutput files:")
    print(OUT_LOWEST_CSV)
    print(OUT_GRID_SUMMARY_CSV)


if __name__ == "__main__":
    main()
