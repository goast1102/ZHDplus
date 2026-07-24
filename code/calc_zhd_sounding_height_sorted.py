from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import calc_zhd_sounding as sounding
import zhd_workflow


# ============================================================
# Path settings
# ============================================================

INPUT_CSV = sounding.INPUT_CSV
OUT_SAAS_CSV = Path(r"E:\ZHDplus\test\out\mawar_sounding_zhd_saas_height_sorted.csv")
OUT_PROFILE_CSV = Path(
    r"E:\ZHDplus\test\out\mawar_sounding_zhd_profile_height_sorted.csv"
)


def calculate_height_sorted_profile_zhd(
    profile: pd.DataFrame,
    time_utc: object,
) -> float | None:
    """Original height-sorted profile integral plus top Saastamoinen compensation."""

    if len(profile) < 2:
        print(f"Warning: {time_utc} has fewer than 2 valid layers; skip profile ZHD.")
        return None

    profile = profile.sort_values(sounding.H_COL).copy()
    profile["temperature_K"] = profile[sounding.TC_COL] + 273.15
    profile["RH"] = profile[sounding.RH_PERCENT_COL] / 100.0

    profile["e"] = sounding.saturation_water_vapor_pressure(
        profile[sounding.P_COL],
        profile["temperature_K"],
        profile["RH"],
    )
    profile["Nh"] = (
        77.6890
        * (profile[sounding.P_COL] - 0.37803 * profile["e"])
        / profile["temperature_K"]
    )

    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(subset=["Nh"])
    profile = profile[profile["Nh"] > 0.0].copy()

    if len(profile) < 2:
        print(
            f"Warning: {time_utc} has fewer than 2 positive-Nh layers; "
            "skip profile ZHD."
        )
        return None

    bottom_layer = profile.loc[profile[sounding.P_COL].idxmax()]
    lat_deg = float(bottom_layer[sounding.LAT_COL])

    height = profile[sounding.H_COL].to_numpy(dtype=float)
    height_geometric = sounding.geometric_height(height, lat_deg)
    nh = profile["Nh"].to_numpy(dtype=float)

    d_height = np.diff(height)
    dz_geometric = np.diff(height_geometric)
    nh_lower = nh[:-1]
    nh_upper = nh[1:]

    valid_interval = d_height >= 0.0
    if not np.all(valid_interval):
        print(f"Warning: {time_utc} has negative height intervals after sorting.")

    zhd_int = 1e-6 * np.sum(
        dz_geometric[valid_interval]
        * sounding.log_mean(
            nh_lower[valid_interval],
            nh_upper[valid_interval],
        )
    )

    top_layer = profile.loc[profile[sounding.H_COL].idxmax()]
    p_top = float(top_layer[sounding.P_COL])
    h_top_geopotential = float(top_layer[sounding.H_COL])
    h_top_geometric = float(sounding.geometric_height(h_top_geopotential, lat_deg))

    if p_top > 100.0:
        print(
            f"Warning: {time_utc} top pressure is {p_top:.2f} hPa, "
            "greater than 100 hPa; still calculating profile ZHD."
        )

    zhd_top = sounding.saastamoinen_zhd(p_top, h_top_geometric, lat_deg)
    return float(zhd_int + zhd_top)


def calculate_zhd(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    saas_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    for time_utc, group in df.groupby(sounding.TIME_COL, sort=True):
        valid = group.dropna(
            subset=[
                sounding.LON_COL,
                sounding.LAT_COL,
                sounding.P_COL,
                sounding.H_COL,
                sounding.TC_COL,
                sounding.RH_PERCENT_COL,
            ]
        ).copy()

        if valid.empty:
            print(f"Warning: {time_utc} has no valid layers; skip this sounding.")
            continue

        bottom_layer = valid.loc[valid[sounding.P_COL].idxmax()]
        longitude = float(bottom_layer[sounding.LON_COL])
        latitude = float(bottom_layer[sounding.LAT_COL])
        p0 = float(bottom_layer[sounding.P_COL])
        h0_geopotential = float(bottom_layer[sounding.H_COL])
        h0_geometric = float(sounding.geometric_height(h0_geopotential, latitude))

        zhd_saas = sounding.saastamoinen_zhd(p0, h0_geometric, latitude)
        saas_rows.append(
            {
                "longitude": longitude,
                "latitude": latitude,
                "time_utc": time_utc,
                "ZHD_m": zhd_saas,
            }
        )

        zhd_profile = calculate_height_sorted_profile_zhd(valid, time_utc)
        if zhd_profile is not None:
            profile_rows.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "time_utc": time_utc,
                    "ZHD_m": zhd_profile,
                }
            )

    out_cols = ["longitude", "latitude", "time_utc", "ZHD_m"]
    saas_df = pd.DataFrame(saas_rows, columns=out_cols)
    profile_df = pd.DataFrame(profile_rows, columns=out_cols)
    return saas_df, profile_df


def run_default_paths() -> None:
    df = sounding.read_sounding_csv(INPUT_CSV)
    saas_df, profile_df = calculate_zhd(df)

    OUT_SAAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)

    saas_df.to_csv(OUT_SAAS_CSV, index=False, encoding="utf-8-sig")
    profile_df.to_csv(OUT_PROFILE_CSV, index=False, encoding="utf-8-sig")

    print("\nOutput files:")
    print(OUT_SAAS_CSV)
    print(OUT_PROFILE_CSV)

    print("\nHeight-sorted Saastamoinen ZHD head:")
    print(saas_df.head().to_string(index=False))

    print("\nHeight-sorted profile ZHD head:")
    print(profile_df.head().to_string(index=False))


def process_event(event: dict) -> None:
    input_csv = zhd_workflow.expected_sounding_qc_combined(event)
    if not input_csv.exists():
        print(f"Skip {event['id']}: input CSV not found: {input_csv}")
        return

    df = sounding.read_sounding_csv(input_csv)
    saas_df, profile_df = calculate_zhd(df)

    out_dir = zhd_workflow.event_out_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_saas = out_dir / "sounding_zhd_saas_height_sorted.csv"
    out_profile = out_dir / "sounding_zhd_profile_height_sorted.csv"

    saas_df.to_csv(out_saas, index=False, encoding="utf-8-sig")
    profile_df.to_csv(out_profile, index=False, encoding="utf-8-sig")

    print(f"{event['id']}:")
    print(out_saas)
    print(out_profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup radiosonde ZHD calculation using height-sorted profiles."
    )
    parser.add_argument("--config", type=Path, default=zhd_workflow.DEFAULT_CONFIG)
    parser.add_argument(
        "--event",
        action="append",
        help=(
            "Event id to process. Repeat for multiple events. "
            "Omit to run the script's default INPUT_CSV/OUT_* paths."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.event:
        run_default_paths()
        return

    events = zhd_workflow.load_events(args.config)
    selected_ids = [event_id.lower() for event_id in args.event]
    missing = [event_id for event_id in selected_ids if event_id not in events]
    if missing:
        raise ValueError(f"Events not found in config: {missing}")

    for event_id in selected_ids:
        process_event(events[event_id])


if __name__ == "__main__":
    main()
