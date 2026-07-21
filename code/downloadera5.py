from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cdsapi
import pandas as pd
import xarray as xr


# ============================================================
# 1. ERA5 下载设置：Mawar 事件区域
# ============================================================

START_DATE = "2023-05-21"
END_DATE = "2023-05-28"

# CDS area 顺序：[北, 西, 南, 东]
# 下载最近四格点外扩一圈，约 4×4 个 ERA5 格点
AREA = [
    14.00,   # North
    144.50,  # West
    13.25,   # South
    145.25,  # East
]


# ============================================================
# 2. 路径设置
# ============================================================

BASE_DIR = Path(r"E:\ZHDplus\test\data\era5_mawar")

GRIB_DIR = BASE_DIR / "grib"
OUT_CSV = BASE_DIR / "era5_pressure_raw_extract.csv"

GRIB_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 运行控制
# ============================================================

# 是否下载 ERA5
DOWNLOAD_ERA5 = True

# 是否将 GRIB 提取为 CSV
EXTRACT_TO_CSV = True

# 已有文件是否重新下载
OVERWRITE = False

# Conservative daily download concurrency for CDS requests.
MAX_PARALLEL_DOWNLOADS = 3

# 是否只保留 00 UTC
# 建议先设 False，下载逐小时数据
ONLY_00UTC = False

G0 = 9.80665


# ============================================================
# 4. ERA5 参数
# ============================================================

DATASET = "reanalysis-era5-pressure-levels"

VARIABLES = [
    "geopotential",
    "relative_humidity",
    "temperature",
]

# ERA5 全部 37 个标准压力层
PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10",
    "20", "30", "50", "70",
    "100", "125", "150", "175",
    "200", "225", "250",
    "300", "350", "400", "450",
    "500", "550", "600", "650",
    "700", "750", "775",
    "800", "825", "850", "875",
    "900", "925", "950", "975", "1000",
]

TIMES = [f"{hour:02d}:00" for hour in range(24)]


# ============================================================
# 5. 下载单日 ERA5
# ============================================================

def download_one_day(
    client: cdsapi.Client,
    date: pd.Timestamp,
    output_file: Path,
) -> None:

    request: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [date.strftime("%Y")],
        "month": [date.strftime("%m")],
        "day": [date.strftime("%d")],
        "time": TIMES,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }

    print("=" * 70)
    print(f"下载日期：{date:%Y-%m-%d}")
    print(f"下载范围 [北, 西, 南, 东]：{AREA}")
    print(f"输出文件：{output_file}")

    client.retrieve(
        DATASET,
        request,
        str(output_file),
    )


def download_all_days() -> list[Path]:

    client = cdsapi.Client()

    dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    grib_files: list[Path] = []

    for date in dates:

        grib_file = GRIB_DIR / f"era5_pressure_{date:%Y%m%d}.grib"

        if grib_file.exists() and grib_file.stat().st_size > 0:

            if not OVERWRITE:
                print(f"已有文件，跳过下载：{grib_file}")
                grib_files.append(grib_file)
                continue

            grib_file.unlink()

        try:
            download_one_day(
                client=client,
                date=date,
                output_file=grib_file,
            )

            if not grib_file.exists():
                raise RuntimeError(
                    f"下载完成后没有找到文件：{grib_file}"
                )

            if grib_file.stat().st_size == 0:
                raise RuntimeError(
                    f"下载文件为空：{grib_file}"
                )

            grib_files.append(grib_file)

        except Exception as exc:
            print(f"下载失败：{date:%Y-%m-%d}")
            print(f"错误信息：{exc}")

    return grib_files


# ============================================================
# 6. 读取单个 GRIB
# ============================================================

def download_one_day_checked(date: pd.Timestamp) -> Path | None:

    grib_file = GRIB_DIR / f"era5_pressure_{date:%Y%m%d}.grib"

    try:
        client = cdsapi.Client()

        download_one_day(
            client=client,
            date=date,
            output_file=grib_file,
        )

        if not grib_file.exists():
            raise RuntimeError(f"Downloaded file not found: {grib_file}")

        if grib_file.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {grib_file}")

        return grib_file

    except Exception as exc:
        print(f"Download failed: {date:%Y-%m-%d}")
        print(f"Error: {exc}")
        return None


def download_all_days() -> list[Path]:

    dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    grib_files: list[Path] = []
    pending_dates: list[pd.Timestamp] = []

    for date in dates:

        grib_file = GRIB_DIR / f"era5_pressure_{date:%Y%m%d}.grib"

        if grib_file.exists() and grib_file.stat().st_size > 0:

            if not OVERWRITE:
                print(f"Existing file, skip download: {grib_file}")
                grib_files.append(grib_file)
                continue

            grib_file.unlink()

        pending_dates.append(date)

    if not pending_dates:
        return sorted(grib_files)

    worker_count = max(1, min(int(MAX_PARALLEL_DOWNLOADS), len(pending_dates)))

    if worker_count == 1:
        for date in pending_dates:
            grib_file = download_one_day_checked(date)
            if grib_file is not None:
                grib_files.append(grib_file)
    else:
        print(f"Parallel ERA5 daily downloads: {worker_count} workers")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(download_one_day_checked, date): date
                for date in pending_dates
            }

            for future in as_completed(futures):
                grib_file = future.result()
                if grib_file is not None:
                    grib_files.append(grib_file)

    return sorted(grib_files)


def read_grib(grib_file: Path) -> pd.DataFrame:

    print(f"\n读取：{grib_file.name}")

    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
        },
    )

    try:
        print("维度：", dict(ds.sizes))
        print("变量：", list(ds.data_vars))

        df = ds.to_dataframe().reset_index()

    finally:
        ds.close()

    rename_map: dict[str, str] = {}

    # 时间
    if "valid_time" in df.columns:
        rename_map["valid_time"] = "time_utc"
    elif "time" in df.columns:
        rename_map["time"] = "time_utc"

    # 压力层
    if "isobaricInhPa" in df.columns:
        rename_map["isobaricInhPa"] = "pressure_hPa"
    elif "pressure_level" in df.columns:
        rename_map["pressure_level"] = "pressure_hPa"
    elif "level" in df.columns:
        rename_map["level"] = "pressure_hPa"

    # 经纬度
    if "lat" in df.columns:
        rename_map["lat"] = "latitude"

    if "lon" in df.columns:
        rename_map["lon"] = "longitude"

    # ERA5 变量
    if "t" in df.columns:
        rename_map["t"] = "temperature_K"

    if "r" in df.columns:
        rename_map["r"] = "relative_humidity_percent"

    if "z" in df.columns:
        rename_map["z"] = "geopotential_m2_s2"

    df = df.rename(columns=rename_map)

    required_cols = [
        "time_utc",
        "longitude",
        "latitude",
        "pressure_hPa",
        "temperature_K",
        "relative_humidity_percent",
        "geopotential_m2_s2",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{grib_file.name} 缺少必要字段：{missing_cols}\n"
            f"实际字段：{df.columns.tolist()}"
        )

    # 时间
    df["time_utc"] = pd.to_datetime(
        df["time_utc"],
        errors="coerce",
    )

    # 数值列
    numeric_cols = [
        "longitude",
        "latitude",
        "pressure_hPa",
        "temperature_K",
        "relative_humidity_percent",
        "geopotential_m2_s2",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # 位势转换为位势高度
    df["geopotential_height_m"] = (
        df["geopotential_m2_s2"] / G0
    )

    # 经度统一到 [-180, 180)
    df["longitude"] = (
        (df["longitude"] + 180.0) % 360.0
    ) - 180.0

    # ERA5 相对湿度偶尔略超出 0–100
    df["relative_humidity_percent"] = (
        df["relative_humidity_percent"]
        .clip(lower=0.0, upper=100.0)
    )

    if ONLY_00UTC:
        df = df[
            df["time_utc"].dt.hour == 0
        ].copy()

    df["source_file"] = grib_file.name

    output_cols = [
        "time_utc",
        "longitude",
        "latitude",
        "pressure_hPa",
        "geopotential_height_m",
        "temperature_K",
        "relative_humidity_percent",
        "geopotential_m2_s2",
        "source_file",
    ]

    df = df[output_cols].copy()

    df = df.dropna(
        subset=[
            "time_utc",
            "longitude",
            "latitude",
            "pressure_hPa",
            "geopotential_height_m",
            "temperature_K",
            "relative_humidity_percent",
        ]
    )

    return df


# ============================================================
# 7. 合并全部 GRIB 并输出 CSV
# ============================================================

def extract_all_grib(
    grib_files: list[Path],
) -> pd.DataFrame:

    if not grib_files:
        grib_files = sorted(
            GRIB_DIR.glob("*.grib")
        )

    if not grib_files:
        raise FileNotFoundError(
            f"没有找到 GRIB 文件：{GRIB_DIR}"
        )

    all_df: list[pd.DataFrame] = []

    for index, grib_file in enumerate(
        grib_files,
        start=1,
    ):
        print(
            f"\n[{index}/{len(grib_files)}] "
            f"处理：{grib_file.name}"
        )

        df = read_grib(grib_file)

        print(f"有效记录数：{len(df)}")

        all_df.append(df)

    out = pd.concat(
        all_df,
        ignore_index=True,
    )

    # 去除重复记录
    out = out.drop_duplicates(
        subset=[
            "time_utc",
            "latitude",
            "longitude",
            "pressure_hPa",
        ],
        keep="last",
    )

    # 排序
    out = out.sort_values(
        [
            "time_utc",
            "latitude",
            "longitude",
            "pressure_hPa",
        ],
        ascending=[
            True,
            False,
            True,
            False,
        ],
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    return out


# ============================================================
# 8. 输出检查
# ============================================================

def print_check(out: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("ERA5 下载与提取完成")
    print("=" * 70)

    print(f"时间范围：{START_DATE} 至 {END_DATE}")
    print(f"下载区域 [北, 西, 南, 东]：{AREA}")
    print(f"输出 CSV：{OUT_CSV}")
    print(f"总记录数：{len(out)}")

    print("\n实际时间范围：")
    print(out["time_utc"].min())
    print(out["time_utc"].max())

    print("\nERA5 经度格点：")
    print(sorted(out["longitude"].unique()))

    print("\nERA5 纬度格点：")
    print(sorted(out["latitude"].unique()))

    print("\n压力层：")
    pressure_levels = sorted(
        out["pressure_hPa"].unique(),
        reverse=True,
    )
    print(pressure_levels)
    print(f"压力层数量：{len(pressure_levels)}")

    print("\n每个时刻、每个格点的层数统计：")

    layer_count = (
        out.groupby(
            ["time_utc", "latitude", "longitude"]
        )["pressure_hPa"]
        .nunique()
    )

    print(layer_count.describe())

    print("\n输出字段：")
    print(out.columns.tolist())

    print("\n前 20 行：")
    print(out.head(20).to_string(index=False))


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ERA5 压力层下载与提取")
    print("=" * 70)

    print(f"时间：{START_DATE} 至 {END_DATE}")
    print(f"区域 [北, 西, 南, 东]：{AREA}")

    grib_files: list[Path] = []

    if DOWNLOAD_ERA5:
        grib_files = download_all_days()

    if EXTRACT_TO_CSV:
        out = extract_all_grib(grib_files)
        print_check(out)


if __name__ == "__main__":
    main()
