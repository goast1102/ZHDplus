from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Path and station settings
# ============================================================

INPUT_CSV = Path(r"E:\ZHDplus\test\data\era5_mawar\era5_pressure_raw_extract.csv")

OUT_SAAS_GRID4_CSV = Path(r"E:\ZHDplus\test\out\mawar_era5_zhd_saas_grid4.csv")
OUT_PROFILE_GRID4_CSV = Path(r"E:\ZHDplus\test\out\mawar_era5_zhd_profile_grid4.csv")
OUT_SAAS_STATION_CSV = Path(r"E:\ZHDplus\test\out\mawar_era5_zhd_saas_station_interp.csv")
OUT_PROFILE_STATION_CSV = Path(r"E:\ZHDplus\test\out\mawar_era5_zhd_profile_station_interp.csv")

STATION_LAT = 13.4776
STATION_LON = 144.7936


# ============================================================
# Column names
# ============================================================

TIME_COL = "time_utc"
LON_COL = "longitude"
LAT_COL = "latitude"
P_COL = "pressure_hPa"
H_COL = "geopotential_height_m"
T_COL = "temperature_K"
RH_PERCENT_COL = "relative_humidity_percent"

REQUIRED_COLS = [
    TIME_COL,
    LON_COL,
    LAT_COL,
    P_COL,
    H_COL,
    T_COL,
    RH_PERCENT_COL,
]

OUT_COLS = ["longitude", "latitude", "time_utc", "ZHD_m"]

ERA5_EARTH_RADIUS_M = 6_371_229.0


def geometric_height(geopotential_height_m: object) -> np.ndarray:
    H = np.asarray(geopotential_height_m, dtype=float)
    return ERA5_EARTH_RADIUS_M * H / (ERA5_EARTH_RADIUS_M - H)


def saastamoinen_zhd(pressure_hpa: float, height_m: float, latitude_deg: float) -> float:
    lat_rad = np.deg2rad(latitude_deg)
    denominator = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 2.8e-7 * height_m
    return 0.0022793 * pressure_hpa / denominator


def water_vapor_pressure(
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


def read_era5_csv(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    for col in [LON_COL, LAT_COL, P_COL, H_COL, T_COL, RH_PERCENT_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def find_surrounding_grid(df: pd.DataFrame) -> tuple[float, float, float, float]:
    lats = np.array(sorted(df[LAT_COL].dropna().unique()), dtype=float)
    lons = np.array(sorted(df[LON_COL].dropna().unique()), dtype=float)

    def bracket(values: np.ndarray, target: float, label: str) -> tuple[float, float]:
        if target < values.min() or target > values.max():
            raise ValueError(
                "Cannot find ERA5 grid points surrounding the station. "
                f"Station {label}: {target}. "
                f"Available {label} range: {values.min()} to {values.max()}."
            )

        exact = np.where(np.isclose(values, target, rtol=0.0, atol=1e-10))[0]
        if len(exact) > 0:
            print(
                f"Warning: station {label} lies exactly on ERA5 grid line "
                f"{target}; using on-line interpolation."
            )
            exact_value = float(values[int(exact[0])])
            return exact_value, exact_value

        lower_candidates = values[values < target]
        upper_candidates = values[values > target]

        if len(lower_candidates) == 0 or len(upper_candidates) == 0:
            raise ValueError(
                "Cannot find ERA5 grid points surrounding the station. "
                f"Station {label}: {target}. "
                f"Available {label} range: {values.min()} to {values.max()}."
            )

        return float(lower_candidates.max()), float(upper_candidates.min())

    lat_s, lat_n = bracket(lats, STATION_LAT, "latitude")
    lon_w, lon_e = bracket(lons, STATION_LON, "longitude")

    return lon_w, lon_e, lat_s, lat_n


def select_grid4(
    df: pd.DataFrame,
    lon_w: float,
    lon_e: float,
    lat_s: float,
    lat_n: float,
) -> pd.DataFrame:
    grid4 = df[
        df[LON_COL].isin([lon_w, lon_e]) & df[LAT_COL].isin([lat_s, lat_n])
    ].copy()

    if grid4.empty:
        raise ValueError("No rows found for the selected four ERA5 grid points.")

    return grid4


def calculate_grid_profile(
    group: pd.DataFrame,
    time_utc: object,
    longitude: float,
    latitude: float,
) -> tuple[float | None, float | None]:
    valid = group.dropna(subset=[P_COL, H_COL, T_COL, RH_PERCENT_COL]).copy()

    if valid.empty:
        print(
            f"Warning: {time_utc}, lon={longitude}, lat={latitude} "
            "has no valid layers; skip this grid profile."
        )
        return None, None

    bottom_layer = valid.loc[valid[P_COL].idxmax()]
    p0 = float(bottom_layer[P_COL])
    h0_geopotential = float(bottom_layer[H_COL])
    h0_geometric = float(geometric_height(h0_geopotential))

    if h0_geopotential < 0.0:
        print(
            f"Warning: {time_utc}, lon={longitude}, lat={latitude} "
            f"lowest-layer geopotential height is {h0_geopotential:.3f} m; keeping it."
        )

    zhd_saas = saastamoinen_zhd(p0, h0_geometric, latitude)

    if len(valid) < 2:
        print(
            f"Warning: {time_utc}, lon={longitude}, lat={latitude} "
            "has fewer than 2 valid layers; skip profile ZHD."
        )
        return float(zhd_saas), None

    profile = valid.sort_values(H_COL).copy()
    profile["e"] = water_vapor_pressure(
        profile[P_COL],
        profile[T_COL],
        profile[RH_PERCENT_COL],
    )
    profile["Nh"] = 77.6890 * (
        profile[P_COL] - 0.37803 * profile["e"]
    ) / profile[T_COL]

    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(subset=["Nh"])
    profile = profile[profile["Nh"] > 0.0].copy()

    if len(profile) < 2:
        print(
            f"Warning: {time_utc}, lon={longitude}, lat={latitude} "
            "has fewer than 2 positive-Nh layers; skip profile ZHD."
        )
        return float(zhd_saas), None

    height = profile[H_COL].to_numpy(dtype=float)
    height_geometric = geometric_height(height)
    nh = profile["Nh"].to_numpy(dtype=float)

    d_height = np.diff(height)
    dz_geometric = np.diff(height_geometric)
    nh_lower = nh[:-1]
    nh_upper = nh[1:]

    valid_interval = d_height >= 0.0
    zhd_int = 1e-6 * np.sum(
        dz_geometric[valid_interval]
        * log_mean(nh_lower[valid_interval], nh_upper[valid_interval])
    )

    top_layer = profile.loc[profile[H_COL].idxmax()]
    p_top = float(top_layer[P_COL])
    h_top_geopotential = float(top_layer[H_COL])
    h_top_geometric = float(geometric_height(h_top_geopotential))
    zhd_top = saastamoinen_zhd(p_top, h_top_geometric, latitude)

    return float(zhd_saas), float(zhd_int + zhd_top)


def calculate_grid_zhd(grid4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    saas_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    group_cols = [TIME_COL, LAT_COL, LON_COL]
    for (time_utc, latitude, longitude), group in grid4.groupby(group_cols, sort=True):
        longitude = float(longitude)
        latitude = float(latitude)
        zhd_saas, zhd_profile = calculate_grid_profile(
            group,
            time_utc,
            longitude,
            latitude,
        )

        if zhd_saas is not None:
            saas_rows.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "time_utc": time_utc,
                    "ZHD_m": zhd_saas,
                }
            )

        if zhd_profile is not None:
            profile_rows.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "time_utc": time_utc,
                    "ZHD_m": zhd_profile,
                }
            )

    saas_df = pd.DataFrame(saas_rows, columns=OUT_COLS)
    profile_df = pd.DataFrame(profile_rows, columns=OUT_COLS)
    return saas_df, profile_df


def bilinear_station_interp(
    grid_df: pd.DataFrame,
    lon_w: float,
    lon_e: float,
    lat_s: float,
    lat_n: float,
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    on_lon_grid = np.isclose(lon_w, lon_e, rtol=0.0, atol=1e-10)
    on_lat_grid = np.isclose(lat_s, lat_n, rtol=0.0, atol=1e-10)

    if on_lon_grid and on_lat_grid:
        interp_mode = "nearest grid point"
        required_points = [(lat_s, lon_w)]
    elif on_lon_grid:
        interp_mode = "latitude linear"
        required_points = [(lat_s, lon_w), (lat_n, lon_w)]
    elif on_lat_grid:
        interp_mode = "longitude linear"
        required_points = [(lat_s, lon_w), (lat_s, lon_e)]
    else:
        interp_mode = "bilinear"
        required_points = [
            (lat_s, lon_w),
            (lat_s, lon_e),
            (lat_n, lon_w),
            (lat_n, lon_e),
        ]

    print(f"Station interpolation mode for {label}: {interp_mode}")

    for time_utc, group in grid_df.groupby(TIME_COL, sort=True):
        values = {
            (float(row[LAT_COL]), float(row[LON_COL])): float(row["ZHD_m"])
            for _, row in group.iterrows()
        }

        missing = [point for point in required_points if point not in values]
        if missing:
            print(
                f"Warning: {time_utc} has incomplete {interp_mode} {label} ZHD; "
                f"missing {missing}. Skip station interpolation."
            )
            continue

        if on_lon_grid and on_lat_grid:
            z_station = values[(lat_s, lon_w)]
        elif on_lon_grid:
            weight_n = (STATION_LAT - lat_s) / (lat_n - lat_s)
            weight_s = 1.0 - weight_n
            z_station = (
                weight_s * values[(lat_s, lon_w)]
                + weight_n * values[(lat_n, lon_w)]
            )
        elif on_lat_grid:
            weight_e = (STATION_LON - lon_w) / (lon_e - lon_w)
            weight_w = 1.0 - weight_e
            z_station = (
                weight_w * values[(lat_s, lon_w)]
                + weight_e * values[(lat_s, lon_e)]
            )
        else:
            denominator = (lon_e - lon_w) * (lat_n - lat_s)
            weights = {
                (lat_s, lon_w): ((lon_e - STATION_LON) * (lat_n - STATION_LAT))
                / denominator,
                (lat_s, lon_e): ((STATION_LON - lon_w) * (lat_n - STATION_LAT))
                / denominator,
                (lat_n, lon_w): ((lon_e - STATION_LON) * (STATION_LAT - lat_s))
                / denominator,
                (lat_n, lon_e): ((STATION_LON - lon_w) * (STATION_LAT - lat_s))
                / denominator,
            }
            z_station = sum(weights[point] * values[point] for point in weights)

        rows.append(
            {
                "longitude": STATION_LON,
                "latitude": STATION_LAT,
                "time_utc": time_utc,
                "ZHD_m": float(z_station),
            }
        )

    return pd.DataFrame(rows, columns=OUT_COLS)


def write_csv(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def print_result(name: str, out_csv: Path, df: pd.DataFrame) -> None:
    print(f"\n{name}:")
    print(out_csv)
    print(f"Rows: {len(df)}")
    print(df.head().to_string(index=False))


def main() -> None:
    df = read_era5_csv(INPUT_CSV)
    lon_w, lon_e, lat_s, lat_n = find_surrounding_grid(df)

    print("Selected ERA5 grid points:")
    print(f"SW: lon={lon_w}, lat={lat_s}")
    print(f"SE: lon={lon_e}, lat={lat_s}")
    print(f"NW: lon={lon_w}, lat={lat_n}")
    print(f"NE: lon={lon_e}, lat={lat_n}")

    grid4 = select_grid4(df, lon_w, lon_e, lat_s, lat_n)
    saas_grid4, profile_grid4 = calculate_grid_zhd(grid4)

    saas_station = bilinear_station_interp(
        saas_grid4,
        lon_w,
        lon_e,
        lat_s,
        lat_n,
        "Saastamoinen",
    )
    profile_station = bilinear_station_interp(
        profile_grid4,
        lon_w,
        lon_e,
        lat_s,
        lat_n,
        "profile",
    )

    write_csv(saas_grid4, OUT_SAAS_GRID4_CSV)
    write_csv(profile_grid4, OUT_PROFILE_GRID4_CSV)
    write_csv(saas_station, OUT_SAAS_STATION_CSV)
    write_csv(profile_station, OUT_PROFILE_STATION_CSV)

    print_result("ERA5 Saastamoinen grid4", OUT_SAAS_GRID4_CSV, saas_grid4)
    print_result("ERA5 profile grid4", OUT_PROFILE_GRID4_CSV, profile_grid4)
    print_result("ERA5 Saastamoinen station interp", OUT_SAAS_STATION_CSV, saas_station)
    print_result("ERA5 profile station interp", OUT_PROFILE_STATION_CSV, profile_station)


if __name__ == "__main__":
    main()
