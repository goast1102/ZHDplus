from __future__ import annotations

import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# 用户参数：Mawar—Guam
# ============================================================

STATION = "91212"

# Mawar影响关岛前后，UTC
START_TIME = datetime(2023, 5, 21, 0)
END_TIME = datetime(2023, 5, 28, 0)

# 常规探空时次
SOUNDING_HOURS = [0, 12]

OUT_DIR = Path(r"E:\ZHDplus\test\data\radiosonde\91212_Mawar")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Wyoming新接口
# ============================================================

BASE_URL = "https://weather.arcc.uwyo.edu/wsgi/sounding"

TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10
REQUEST_INTERVAL_SECONDS = 2


def generate_sounding_times(
    start_time: datetime,
    end_time: datetime,
    sounding_hours: list[int],
) -> list[datetime]:
    """生成日期范围内的00和12 UTC探空时次。"""

    times: list[datetime] = []

    current_date = start_time.date()
    end_date = end_time.date()

    while current_date <= end_date:
        for hour in sounding_hours:
            sounding_time = datetime(
                current_date.year,
                current_date.month,
                current_date.day,
                hour,
            )

            if start_time <= sounding_time <= end_time:
                times.append(sounding_time)

        current_date += timedelta(days=1)

    return sorted(times)


def download_one_sounding(
    session: requests.Session,
    sounding_time: datetime,
    station: str,
) -> pd.DataFrame | None:
    """下载并解析一个标准时次的探空数据。"""

    params = {
        "datetime": sounding_time.strftime("%Y-%m-%d %H:%M:%S"),
        "id": station,
        "type": "TEXT:CSV",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Downloading station={station}, "
                f"time={sounding_time:%Y-%m-%d %H:%M UTC}, "
                f"attempt={attempt}/{MAX_RETRIES}"
            )

            response = session.get(
                BASE_URL,
                params=params,
                timeout=TIMEOUT_SECONDS,
            )

            print(f"  HTTP status: {response.status_code}")
            print(f"  URL: {response.url}")

            response.raise_for_status()

            text = response.text.strip()

            if not text:
                print("  Empty response.")
                return None

            lower_text = text.lower()

            error_keywords = [
                "no data",
                "not found",
                "invalid station",
                "internal server error",
                "traceback",
                "<html",
                "<!doctype",
            ]

            if any(keyword in lower_text for keyword in error_keywords):
                print("  Server returned no valid sounding data.")
                print(text[:500])
                return None

            df = pd.read_csv(
                StringIO(text),
                skipinitialspace=True,
            )

            if df.empty:
                print("  Parsed DataFrame is empty.")
                return None

            required_columns = {
                "time",
                "pressure_hPa",
                "geopotential height_m",
                "temperature_C",
            }

            missing_columns = required_columns - set(df.columns)

            if missing_columns:
                raise ValueError(
                    f"返回数据缺少必要字段：{sorted(missing_columns)}；"
                    f"实际字段为：{df.columns.tolist()}"
                )

            # 转换时间
            df["time"] = pd.to_datetime(
                df["time"],
                errors="coerce",
            )

            # 给每层数据增加标准探空时次
            df.insert(
                0,
                "sounding_time_utc",
                sounding_time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            df.insert(
                1,
                "station_number",
                station,
            )

            # 将除时间外的探空变量转成数值
            numeric_columns = [
                "longitude",
                "latitude",
                "pressure_hPa",
                "geopotential height_m",
                "temperature_C",
                "dew point temperature_C",
                "ice point temperature_C",
                "relative humidity_%",
                "humidity wrt ice_%",
                "mixing ratio_g/kg",
                "wind direction_degree",
                "wind speed_m/s",
            ]

            for column in numeric_columns:
                if column in df.columns:
                    df[column] = pd.to_numeric(
                        df[column],
                        errors="coerce",
                    )

            # 删除没有气压或高度的无效层
            df = df.dropna(
                subset=[
                    "pressure_hPa",
                    "geopotential height_m",
                ]
            ).copy()

            # 按高度从低到高排序
            df = df.sort_values(
                "geopotential height_m"
            ).reset_index(drop=True)

            # 删除完全重复的层
            df = df.drop_duplicates(
                subset=[
                    "pressure_hPa",
                    "geopotential height_m",
                ],
                keep="first",
            ).reset_index(drop=True)

            if df.empty:
                print("  No valid vertical levels after QC.")
                return None

            return df

        except (
            requests.RequestException,
            pd.errors.ParserError,
            ValueError,
        ) as exc:
            print(f"  Failed: {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    return None


def main() -> None:
    sounding_times = generate_sounding_times(
        START_TIME,
        END_TIME,
        SOUNDING_HOURS,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/150.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
            "Referer": (
                "https://weather.arcc.uwyo.edu/"
                "upperair/sounding.shtml"
            ),
        }
    )

    profiles: list[pd.DataFrame] = []
    logs: list[dict] = []

    for sounding_time in sounding_times:
        output_file = OUT_DIR / (
            f"{STATION}_{sounding_time:%Y%m%d_%H}UTC.csv"
        )

        # 已存在时直接读取
        if output_file.exists():
            try:
                existing_df = pd.read_csv(
                    output_file,
                    parse_dates=["time"],
                )

                if not existing_df.empty:
                    print(f"Existing: {output_file.name}")

                    profiles.append(existing_df)

                    logs.append(
                        {
                            "station": STATION,
                            "sounding_time_utc": sounding_time,
                            "status": "existing",
                            "levels": len(existing_df),
                            "minimum_pressure_hPa": (
                                existing_df["pressure_hPa"].min()
                            ),
                            "maximum_height_m": (
                                existing_df[
                                    "geopotential height_m"
                                ].max()
                            ),
                            "file": str(output_file),
                        }
                    )
                    continue

            except Exception as exc:
                print(
                    f"  Existing file cannot be read, "
                    f"download again: {exc}"
                )

        df = download_one_sounding(
            session=session,
            sounding_time=sounding_time,
            station=STATION,
        )

        if df is None:
            logs.append(
                {
                    "station": STATION,
                    "sounding_time_utc": sounding_time,
                    "status": "missing_or_failed",
                    "levels": 0,
                    "minimum_pressure_hPa": None,
                    "maximum_height_m": None,
                    "file": "",
                }
            )

            time.sleep(REQUEST_INTERVAL_SECONDS)
            continue

        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S",
        )

        profiles.append(df)

        logs.append(
            {
                "station": STATION,
                "sounding_time_utc": sounding_time,
                "status": "downloaded",
                "levels": len(df),
                "minimum_pressure_hPa": df["pressure_hPa"].min(),
                "maximum_height_m": df[
                    "geopotential height_m"
                ].max(),
                "file": str(output_file),
            }
        )

        print(
            f"  Saved: {output_file.name}\n"
            f"  Levels: {len(df)}\n"
            f"  Pressure range: "
            f"{df['pressure_hPa'].max():.1f}–"
            f"{df['pressure_hPa'].min():.1f} hPa\n"
            f"  Maximum height: "
            f"{df['geopotential height_m'].max():.1f} m"
        )

        time.sleep(REQUEST_INTERVAL_SECONDS)

    # 合并全部探空数据
    if profiles:
        combined_df = pd.concat(
            profiles,
            ignore_index=True,
            sort=False,
        )

        combined_df["sounding_time_utc"] = pd.to_datetime(
            combined_df["sounding_time_utc"],
            errors="coerce",
        )

        combined_df = combined_df.sort_values(
            [
                "sounding_time_utc",
                "geopotential height_m",
            ]
        ).reset_index(drop=True)

        combined_file = OUT_DIR / (
            f"{STATION}_"
            f"{START_TIME:%Y%m%d%H}_"
            f"{END_TIME:%Y%m%d%H}_combined.csv"
        )

        combined_df.to_csv(
            combined_file,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S",
        )

        print("\nCombined file saved:")
        print(combined_file)
        print(f"Total profiles: {len(profiles)}")
        print(f"Total vertical records: {len(combined_df)}")

    # 下载日志
    log_df = pd.DataFrame(logs)

    if not log_df.empty:
        log_df = log_df.sort_values(
            "sounding_time_utc"
        ).reset_index(drop=True)

        log_file = OUT_DIR / "download_log.csv"

        log_df.to_csv(
            log_file,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S",
        )

        print("\nDownload log:")
        print(log_file)

        print("\nStatus summary:")
        print(log_df["status"].value_counts())

    print("\nFinished.")


if __name__ == "__main__":
    main()