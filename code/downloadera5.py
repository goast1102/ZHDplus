from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from calendar import monthrange
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

# 长时段会拆成多个较小请求；这里控制同时提交给 CDS 的请求数。
MAX_PARALLEL_DOWNLOADS = 3

# auto:
# - 短事件：在 CDS 单次请求限制内时，一次请求整个事件；
# - 长时段：按“月 × 变量”拆分，避免 request too large。
# event: 尽量按事件请求；monthly_variable: 始终按月和变量拆分。
DOWNLOAD_STRATEGY = "auto"

# CDS 页面提示的 Request cost 上限一般是 60000。这里用字段数近似估计：
# 变量数 × 天数 × 小时数 × 压力层数。
CDS_REQUEST_COST_LIMIT = 60000

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
# 5. ERA5 下载任务规划
# ============================================================

@dataclass(frozen=True)
class DownloadTask:
    label: str
    years: tuple[str, ...]
    months: tuple[str, ...]
    days: tuple[str, ...]
    variables: tuple[str, ...]
    output_file: Path


def request_cost(day_count: int, variable_count: int) -> int:
    return (
        int(day_count)
        * int(variable_count)
        * len(TIMES)
        * len(PRESSURE_LEVELS)
    )


def make_request(task: DownloadTask) -> dict[str, Any]:
    request: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "variable": list(task.variables),
        "year": list(task.years),
        "month": list(task.months),
        "day": list(task.days),
        "time": TIMES,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }
    return request


def month_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = pd.Timestamp(start.year, start.month, 1)

    while current <= end:
        last_day = monthrange(current.year, current.month)[1]
        month_start = current
        month_end = pd.Timestamp(current.year, current.month, last_day)

        chunks.append(
            (
                max(start, month_start),
                min(end, month_end),
            )
        )

        current = month_end + pd.Timedelta(days=1)

    return chunks


def days_for_chunk(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[str, ...]:
    return tuple(
        f"{date.day:02d}"
        for date in pd.date_range(start=start, end=end, freq="D")
    )


def event_task(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> DownloadTask:

    if start.year != end.year or start.month != end.month:
        raise ValueError(
            "CDS ERA5 pressure-level requests cannot safely express an exact "
            "cross-month event in one year/month/day selection. Split by month."
        )

    return DownloadTask(
        label=f"event {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        years=(f"{start.year:04d}",),
        months=(f"{start.month:02d}",),
        days=days_for_chunk(start, end),
        variables=tuple(VARIABLES),
        output_file=GRIB_DIR / f"era5_pressure_event_{start:%Y%m%d}_{end:%Y%m%d}.grib",
    )


def monthly_event_tasks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[DownloadTask]:

    tasks: list[DownloadTask] = []

    for chunk_start, chunk_end in month_chunks(start, end):
        tasks.append(
            DownloadTask(
                label=(
                    f"event chunk {chunk_start:%Y-%m-%d} "
                    f"to {chunk_end:%Y-%m-%d}"
                ),
                years=(f"{chunk_start.year:04d}",),
                months=(f"{chunk_start.month:02d}",),
                days=days_for_chunk(chunk_start, chunk_end),
                variables=tuple(VARIABLES),
                output_file=GRIB_DIR / (
                    f"era5_pressure_event_{chunk_start:%Y%m%d}_"
                    f"{chunk_end:%Y%m%d}.grib"
                ),
            )
        )

    return tasks


def monthly_variable_tasks(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[DownloadTask]:

    tasks: list[DownloadTask] = []

    for chunk_start, chunk_end in month_chunks(start, end):
        for variable in VARIABLES:
            tasks.append(
                DownloadTask(
                    label=(
                        f"{variable} {chunk_start:%Y-%m-%d} "
                        f"to {chunk_end:%Y-%m-%d}"
                    ),
                    years=(f"{chunk_start.year:04d}",),
                    months=(f"{chunk_start.month:02d}",),
                    days=days_for_chunk(chunk_start, chunk_end),
                    variables=(variable,),
                    output_file=GRIB_DIR / (
                        f"era5_pressure_{variable}_"
                        f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.grib"
                    ),
                )
            )

    return tasks


def plan_download_tasks() -> list[DownloadTask]:

    start = pd.Timestamp(START_DATE).normalize()
    end = pd.Timestamp(END_DATE).normalize()

    if end < start:
        raise ValueError(f"END_DATE is earlier than START_DATE: {START_DATE} > {END_DATE}")

    day_count = len(pd.date_range(start=start, end=end, freq="D"))
    full_variable_cost = request_cost(day_count, len(VARIABLES))
    same_month = start.year == end.year and start.month == end.month
    strategy = str(DOWNLOAD_STRATEGY).strip().lower()

    if strategy not in {"auto", "event", "monthly_variable"}:
        raise ValueError(
            "DOWNLOAD_STRATEGY must be one of: auto, event, monthly_variable"
        )

    if strategy == "monthly_variable":
        tasks = monthly_variable_tasks(start, end)
    elif same_month and (
        strategy == "event" or full_variable_cost <= CDS_REQUEST_COST_LIMIT
    ):
        tasks = [event_task(start, end)]
    elif strategy == "event" or full_variable_cost <= CDS_REQUEST_COST_LIMIT:
        tasks = monthly_event_tasks(start, end)
    else:
        tasks = monthly_variable_tasks(start, end)

    print("=" * 70)
    print("ERA5 download plan")
    print(f"Strategy: {DOWNLOAD_STRATEGY}")
    print(f"Date range: {start:%Y-%m-%d} to {end:%Y-%m-%d} ({day_count} days)")
    print(f"Estimated all-variable request cost: {full_variable_cost}")
    print(f"Tasks: {len(tasks)}")
    for index, task in enumerate(tasks, start=1):
        cost = request_cost(len(task.days), len(task.variables))
        print(
            f"[{index}/{len(tasks)}] {task.label}; "
            f"variables={list(task.variables)}; cost~{cost}; "
            f"file={task.output_file.name}"
        )

    if not same_month and tasks and len(tasks[0].variables) > 1:
        print(
            "Note: cross-month events are split by month because the CDS "
            "year/month/day request fields are not an exact date-range filter."
        )

    return tasks


def download_task_checked(task: DownloadTask) -> Path | None:

    try:
        client = cdsapi.Client()
        request = make_request(task)

        print("=" * 70)
        print(f"Downloading: {task.label}")
        print(f"Area [N, W, S, E]: {AREA}")
        print(f"Variables: {list(task.variables)}")
        print(f"Years: {list(task.years)} Months: {list(task.months)} Days: {list(task.days)}")
        print(f"Output file: {task.output_file}")

        client.retrieve(
            DATASET,
            request,
            str(task.output_file),
        )

        if not task.output_file.exists():
            raise RuntimeError(f"Downloaded file not found: {task.output_file}")

        if task.output_file.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {task.output_file}")

        return task.output_file

    except Exception as exc:
        print(f"Download failed: {task.label}")
        print(f"Error: {exc}")
        return None


def download_all_days() -> list[Path]:

    tasks = plan_download_tasks()

    grib_files: list[Path] = []
    pending_tasks: list[DownloadTask] = []

    for task in tasks:
        if task.output_file.exists() and task.output_file.stat().st_size > 0:

            if not OVERWRITE:
                print(f"Existing file, skip download: {task.output_file}")
                grib_files.append(task.output_file)
                continue

            task.output_file.unlink()

        pending_tasks.append(task)

    if not pending_tasks:
        return sorted(grib_files)

    worker_count = max(1, min(int(MAX_PARALLEL_DOWNLOADS), len(pending_tasks)))

    if worker_count == 1:
        for task in pending_tasks:
            grib_file = download_task_checked(task)
            if grib_file is not None:
                grib_files.append(grib_file)
    else:
        print(f"Parallel ERA5 downloads: {worker_count} workers")

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(download_task_checked, task): task
                for task in pending_tasks
            }

            for future in as_completed(futures):
                grib_file = future.result()
                if grib_file is not None:
                    grib_files.append(grib_file)

    return sorted(grib_files)


# ============================================================
# 6. 读取单个 GRIB
# ============================================================

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

    key_cols = [
        "time_utc",
        "longitude",
        "latitude",
        "pressure_hPa",
    ]
    value_cols = [
        "temperature_K",
        "relative_humidity_percent",
        "geopotential_m2_s2",
    ]
    required_cols = key_cols + value_cols

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    missing_key_cols = [
        col for col in key_cols
        if col not in df.columns
    ]
    available_value_cols = [
        col for col in value_cols
        if col in df.columns
    ]

    if missing_key_cols or not available_value_cols:
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
        *available_value_cols,
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if "geopotential_m2_s2" in df.columns:
        # 位势转换为位势高度
        df["geopotential_height_m"] = (
            df["geopotential_m2_s2"] / G0
        )

    # 经度统一到 [-180, 180)
    df["longitude"] = (
        (df["longitude"] + 180.0) % 360.0
    ) - 180.0

    if "relative_humidity_percent" in df.columns:
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
        *[
            col
            for col in [
                "geopotential_height_m",
                "temperature_K",
                "relative_humidity_percent",
                "geopotential_m2_s2",
            ]
            if col in df.columns
        ],
        "source_file",
    ]

    df = df[output_cols].copy()

    df = df.dropna(
        subset=[
            "time_utc",
            "longitude",
            "latitude",
            "pressure_hPa",
            *available_value_cols,
        ]
    )

    return df


def first_valid(series: pd.Series) -> Any:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return valid.iloc[0]


def join_sources(series: pd.Series) -> str:
    return ";".join(
        sorted(
            {
                str(value)
                for value in series.dropna()
                if str(value).strip()
            }
        )
    )


def combine_grib_frames(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:

    key_cols = [
        "time_utc",
        "latitude",
        "longitude",
        "pressure_hPa",
    ]
    value_cols = [
        "geopotential_height_m",
        "temperature_K",
        "relative_humidity_percent",
        "geopotential_m2_s2",
    ]

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    aggregation: dict[str, Any] = {
        col: first_valid
        for col in value_cols
        if col in combined.columns
    }
    aggregation["source_file"] = join_sources

    out = (
        combined.groupby(
            key_cols,
            as_index=False,
            dropna=False,
        )
        .agg(aggregation)
    )

    start = pd.Timestamp(START_DATE).normalize()
    end = pd.Timestamp(END_DATE).normalize() + pd.Timedelta(days=1)
    out = out[
        (out["time_utc"] >= start)
        & (out["time_utc"] < end)
    ].copy()

    required_cols = [
        "time_utc",
        "longitude",
        "latitude",
        "pressure_hPa",
        "geopotential_height_m",
        "temperature_K",
        "relative_humidity_percent",
        "geopotential_m2_s2",
    ]
    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]
    if missing_cols:
        raise ValueError(
            "合并后的 ERA5 数据缺少必要字段："
            f"{missing_cols}。如果按变量拆分下载，请确认所有变量文件都已下载。"
        )

    out = out.dropna(
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

    return out


# ============================================================
# 7. 合并全部 GRIB 并输出 CSV
# ============================================================

def extract_all_grib(
    grib_files: list[Path],
) -> pd.DataFrame:

    if not grib_files:
        planned_files = [
            task.output_file
            for task in plan_download_tasks()
            if task.output_file.exists()
            and task.output_file.stat().st_size > 0
        ]

        if planned_files:
            grib_files = sorted(planned_files)
        else:
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

    out = combine_grib_frames(all_df)

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
