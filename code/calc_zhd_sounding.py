from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Path settings
# ============================================================

INPUT_CSV = Path(
    r"E:\ZHDplus\test\data\radiosonde\91212_Mawar\91212_2023052100_2023052800_combined.csv"
)
OUT_SAAS_CSV = Path(r"E:\ZHDplus\test\out\mawar_sounding_zhd_saas.csv")
OUT_PROFILE_CSV = Path(r"E:\ZHDplus\test\out\mawar_sounding_zhd_profile.csv")


# ============================================================
# Column names
# ============================================================

TIME_COL = "sounding_time_utc"
LON_COL = "longitude"
LAT_COL = "latitude"
P_COL = "pressure_hPa"
H_COL = "geopotential height_m"
TC_COL = "temperature_C"
RH_PERCENT_COL = "relative humidity_%"

REQUIRED_COLS = [
    TIME_COL,
    LON_COL,
    LAT_COL,
    P_COL,
    H_COL,
    TC_COL,
    RH_PERCENT_COL,
]


def saastamoinen_zhd(pressure_hpa: float, height_m: float, latitude_deg: float) -> float:
    """Calculate Saastamoinen hydrostatic zenith delay in meters."""

    lat_rad = np.deg2rad(latitude_deg)
    denominator = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 2.8e-7 * height_m
    return 0.0022793 * pressure_hpa / denominator


def saturation_water_vapor_pressure(
    pressure_hpa: pd.Series,
    temperature_k: pd.Series,
    rh: pd.Series,
) -> pd.Series:
    """Calculate water vapor pressure from pressure, temperature, and RH."""

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
    """Calculate logarithmic mean, with arithmetic mean fallback for close values."""

    close = np.isclose(nh_lower, nh_upper, rtol=1e-10, atol=1e-12)
    result = np.empty_like(nh_lower, dtype=float)
    result[close] = 0.5 * (nh_lower[close] + nh_upper[close])
    result[~close] = (nh_lower[~close] - nh_upper[~close]) / (
        np.log(nh_lower[~close]) - np.log(nh_upper[~close])
    )
    return result


def calculate_profile_zhd(profile: pd.DataFrame, time_utc: object) -> float | None:
    """Calculate profile-integrated ZHD plus top Saastamoinen compensation."""

    if len(profile) < 2:
        print(f"Warning: {time_utc} has fewer than 2 valid layers; skip profile ZHD.")
        return None

    profile = profile.sort_values(H_COL).copy()
    profile["temperature_K"] = profile[TC_COL] + 273.15
    profile["RH"] = profile[RH_PERCENT_COL] / 100.0

    profile["e"] = saturation_water_vapor_pressure(
        profile[P_COL],
        profile["temperature_K"],
        profile["RH"],
    )
    profile["Nh"] = (
        77.6890
        * (profile[P_COL] - 0.37803 * profile["e"])
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

    height = profile[H_COL].to_numpy(dtype=float)
    nh = profile["Nh"].to_numpy(dtype=float)

    d_height = np.diff(height)
    nh_lower = nh[:-1]
    nh_upper = nh[1:]

    valid_interval = d_height >= 0.0
    if not np.all(valid_interval):
        print(f"Warning: {time_utc} has negative height intervals after sorting.")

    zhd_int = 1e-6 * np.sum(d_height[valid_interval] * log_mean(
        nh_lower[valid_interval],
        nh_upper[valid_interval],
    ))

    bottom_layer = profile.loc[profile[P_COL].idxmax()]
    lat_deg = float(bottom_layer[LAT_COL])

    top_layer = profile.loc[profile[H_COL].idxmax()]
    p_top = float(top_layer[P_COL])
    h_top = float(top_layer[H_COL])

    if p_top > 100.0:
        print(
            f"Warning: {time_utc} top pressure is {p_top:.2f} hPa, "
            "greater than 100 hPa; still calculating profile ZHD."
        )

    zhd_top = saastamoinen_zhd(p_top, h_top, lat_deg)
    return float(zhd_int + zhd_top)


def read_sounding_csv(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    for col in [LON_COL, LAT_COL, P_COL, H_COL, TC_COL, RH_PERCENT_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def calculate_zhd(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    saas_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    for time_utc, group in df.groupby(TIME_COL, sort=True):
        valid = group.dropna(subset=[P_COL, H_COL, TC_COL, RH_PERCENT_COL]).copy()

        if valid.empty:
            print(f"Warning: {time_utc} has no valid layers; skip this sounding.")
            continue

        bottom_layer = valid.loc[valid[P_COL].idxmax()]
        longitude = float(bottom_layer[LON_COL])
        latitude = float(bottom_layer[LAT_COL])
        p0 = float(bottom_layer[P_COL])
        h0 = float(bottom_layer[H_COL])

        zhd_saas = saastamoinen_zhd(p0, h0, latitude)
        saas_rows.append(
            {
                "longitude": longitude,
                "latitude": latitude,
                "time_utc": time_utc,
                "ZHD_m": zhd_saas,
            }
        )

        zhd_profile = calculate_profile_zhd(valid, time_utc)
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


def main() -> None:
    df = read_sounding_csv(INPUT_CSV)
    saas_df, profile_df = calculate_zhd(df)

    OUT_SAAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)

    saas_df.to_csv(OUT_SAAS_CSV, index=False, encoding="utf-8-sig")
    profile_df.to_csv(OUT_PROFILE_CSV, index=False, encoding="utf-8-sig")

    print("\nOutput files:")
    print(OUT_SAAS_CSV)
    print(OUT_PROFILE_CSV)

    print("\nSaastamoinen ZHD head:")
    print(saas_df.head().to_string(index=False))

    print("\nProfile ZHD head:")
    print(profile_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
