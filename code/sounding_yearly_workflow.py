from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import calc_zhd_sounding
import downloadWyomingrs


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ZHDPLUS_DATA_ROOT", ROOT_DIR / "data")).expanduser()
OUT_ROOT = Path(os.environ.get("ZHDPLUS_OUT_ROOT", ROOT_DIR / "out")).expanduser()

WORKFLOW_ID = "sounding_yearly_2024_2026_20260728"

SOUNDING_HOURS = [0, 12]
MIN_SOUNDING_LEVELS = 80
MIN_SOUNDING_TOP_HEIGHT_M = 20000.0
MAX_SOUNDING_TOP_PRESSURE_HPA = 50.0

SOUNDING_OUTPUTS = {
    "saas": "sounding_zhd_saas.csv",
    "profile": "sounding_zhd_profile.csv",
}


@dataclass(frozen=True)
class Station:
    id: str
    slug: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Window:
    id: str
    label: str
    start_time: datetime
    end_time: datetime


STATIONS = [
    Station("55299", "naqu", "Naqu", 31.48, 92.07),
    Station("55591", "lhasa", "Lhasa", 29.67, 91.13),
    Station("56029", "yushu", "Yushu", 33.00, 96.97),
    Station("56137", "qamdo", "Qamdo", 31.15, 97.17),
    Station("52818", "golmud", "Golmud", 36.42, 94.90),
    Station("52836", "dulan", "Dulan", 36.30, 98.10),
    Station("52866", "xining", "Xining", 36.65, 101.72),
    Station("56046", "dari", "Dari", 33.75, 99.65),
    Station("11120", "innsbruck", "Innsbruck", 47.26, 11.34),
    Station("06610", "payerne", "Payerne", 46.81, 6.94),
    Station("10868", "munich", "Munich", 48.24, 11.55),
    Station("16113", "cuneo", "Cuneo", 44.54, 7.61),
    Station("16064", "novara", "Novara", 45.53, 8.67),
    Station("16045", "udine", "Udine", 45.98, 13.05),
    Station("14015", "ljubljana", "Ljubljana", 46.07, 14.51),
]

WINDOWS = [
    Window(
        "2024",
        "2024",
        datetime(2024, 1, 1, 0),
        datetime(2024, 12, 31, 12),
    ),
    Window(
        "2025",
        "2025",
        datetime(2025, 1, 1, 0),
        datetime(2025, 12, 31, 12),
    ),
    Window(
        "2026_to_20260728",
        "2026 to 2026-07-28",
        datetime(2026, 1, 1, 0),
        datetime(2026, 7, 28, 12),
    ),
]

ALL_STEPS = [
    "download",
    "qc",
    "zhd",
    "plot",
    "summarize",
]
RESULT_STEPS = [
    "qc",
    "zhd",
    "plot",
    "summarize",
]


def data_dir(station: Station, window: Window) -> Path:
    return DATA_ROOT / "sounding_yearly" / f"{station.id}_{station.slug}" / window.id


def out_dir(station: Station, window: Window) -> Path:
    return OUT_ROOT / WORKFLOW_ID / f"{station.id}_{station.slug}" / window.id


def combined_csv(station: Station, window: Window) -> Path:
    return data_dir(station, window) / (
        f"{station.id}_{window.start_time:%Y%m%d%H}_"
        f"{window.end_time:%Y%m%d%H}_combined.csv"
    )


def qc_csv(station: Station, window: Window) -> Path:
    combined = combined_csv(station, window)
    return combined.with_name(f"{combined.stem}_qc.csv")


def quality_summary_csv(station: Station, window: Window) -> Path:
    return out_dir(station, window) / "sounding_quality_summary.csv"


def zhd_saas_csv(station: Station, window: Window) -> Path:
    return out_dir(station, window) / SOUNDING_OUTPUTS["saas"]


def zhd_profile_csv(station: Station, window: Window) -> Path:
    return out_dir(station, window) / SOUNDING_OUTPUTS["profile"]


def zhd_plot_png(station: Station, window: Window) -> Path:
    return out_dir(station, window) / "sounding_zhd_yearly.png"


def generate_expected_times(window: Window) -> set[str]:
    rows: list[str] = []
    for day in pd.date_range(window.start_time.date(), window.end_time.date(), freq="D"):
        for hour in SOUNDING_HOURS:
            dt = datetime(day.year, day.month, day.day, hour)
            if window.start_time <= dt <= window.end_time:
                rows.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
    return set(rows)


def configure_downloader(station: Station, window: Window) -> None:
    downloadWyomingrs.STATION = station.id
    downloadWyomingrs.START_TIME = window.start_time
    downloadWyomingrs.END_TIME = window.end_time
    downloadWyomingrs.SOUNDING_HOURS = list(SOUNDING_HOURS)
    downloadWyomingrs.OUT_DIR = data_dir(station, window)
    downloadWyomingrs.OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_download(station: Station, window: Window) -> None:
    configure_downloader(station, window)
    downloadWyomingrs.main()


def quality_from_combined(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sounding_time_utc", "pressure_hPa", "geopotential height_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["sounding_time_utc"] = pd.to_datetime(df["sounding_time_utc"], errors="coerce")
    df["pressure_hPa"] = pd.to_numeric(df["pressure_hPa"], errors="coerce")
    df["geopotential height_m"] = pd.to_numeric(
        df["geopotential height_m"],
        errors="coerce",
    )
    valid = df.dropna(
        subset=["sounding_time_utc", "pressure_hPa", "geopotential height_m"]
    )
    return (
        valid.groupby("sounding_time_utc", as_index=False)
        .agg(
            levels=("pressure_hPa", "size"),
            minimum_pressure_hPa=("pressure_hPa", "min"),
            maximum_height_m=("geopotential height_m", "max"),
        )
        .sort_values("sounding_time_utc")
        .reset_index(drop=True)
    )


def load_quality(station: Station, window: Window) -> pd.DataFrame:
    log_csv = data_dir(station, window) / "download_log.csv"
    if log_csv.exists():
        quality = pd.read_csv(log_csv)
        quality["sounding_time_utc"] = pd.to_datetime(
            quality["sounding_time_utc"],
            errors="coerce",
        )
        for col in ["levels", "minimum_pressure_hPa", "maximum_height_m"]:
            quality[col] = pd.to_numeric(quality[col], errors="coerce")
        return quality

    return quality_from_combined(combined_csv(station, window))


def run_qc(station: Station, window: Window) -> bool:
    source_csv = combined_csv(station, window)
    target_qc_csv = qc_csv(station, window)
    target_summary_csv = quality_summary_csv(station, window)

    if not source_csv.exists():
        print(f"Skip QC, combined CSV not found: {source_csv}")
        if target_qc_csv.exists():
            target_qc_csv.unlink()
        return False

    quality = load_quality(station, window)
    quality["passes_qc"] = (
        (quality["levels"] >= MIN_SOUNDING_LEVELS)
        & (quality["maximum_height_m"] >= MIN_SOUNDING_TOP_HEIGHT_M)
        & (quality["minimum_pressure_hPa"] <= MAX_SOUNDING_TOP_PRESSURE_HPA)
    )

    expected_times = generate_expected_times(window)
    actual_times = set(
        quality["sounding_time_utc"].dropna().dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    quality["expected_profile"] = quality["sounding_time_utc"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    ).isin(expected_times)

    target_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(target_summary_csv, index=False, encoding="utf-8-sig")

    good_times = set(
        quality.loc[quality["passes_qc"], "sounding_time_utc"]
        .dropna()
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    print(
        f"QC {station.id} {window.id}: "
        f"{len(good_times)}/{len(expected_times)} expected profiles pass; "
        f"{len(actual_times)} profiles downloaded/found."
    )
    print(f"Quality summary: {target_summary_csv}")

    if not good_times:
        if target_qc_csv.exists():
            target_qc_csv.unlink()
        print("Skip ZHD for this station/year: no profiles passed QC.")
        return False

    combined = pd.read_csv(source_csv)
    combined["sounding_time_utc"] = pd.to_datetime(
        combined["sounding_time_utc"],
        errors="coerce",
    )
    filtered = combined[
        combined["sounding_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S").isin(good_times)
    ].copy()
    filtered.to_csv(
        target_qc_csv,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print(f"QC combined CSV: {target_qc_csv}")
    print(f"QC rows: {len(filtered)}")
    return True


def run_zhd(station: Station, window: Window) -> bool:
    source_csv = qc_csv(station, window)
    if not source_csv.exists():
        print(f"Skip ZHD, QC CSV not found: {source_csv}")
        return False

    calc_zhd_sounding.INPUT_CSV = source_csv
    calc_zhd_sounding.OUT_SAAS_CSV = zhd_saas_csv(station, window)
    calc_zhd_sounding.OUT_PROFILE_CSV = zhd_profile_csv(station, window)
    calc_zhd_sounding.OUT_SAAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    calc_zhd_sounding.main()
    return True


def read_zhd_csv(path: Path, window: Window) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skip missing plot input: {path}")
        return None
    df = pd.read_csv(path)
    required = {"time_utc", "ZHD_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df["ZHD_m"] = pd.to_numeric(df["ZHD_m"], errors="coerce")
    df = df.dropna(subset=["time_utc", "ZHD_m"]).copy()
    df = df[(df["time_utc"] >= window.start_time) & (df["time_utc"] <= window.end_time)]
    if df.empty:
        return None
    df["ZHD_mm"] = df["ZHD_m"] * 1000.0
    return df.sort_values("time_utc").reset_index(drop=True)


def read_quality_summary(station: Station, window: Window) -> pd.DataFrame | None:
    path = quality_summary_csv(station, window)
    if not path.exists():
        return None
    quality = pd.read_csv(path)
    if quality.empty or "sounding_time_utc" not in quality.columns:
        return None
    quality["sounding_time_utc"] = pd.to_datetime(
        quality["sounding_time_utc"],
        errors="coerce",
    )
    for col in ["levels", "minimum_pressure_hPa", "maximum_height_m"]:
        if col in quality.columns:
            quality[col] = pd.to_numeric(quality[col], errors="coerce")
    return quality.dropna(subset=["sounding_time_utc"]).copy()


def set_year_xaxis(ax: plt.Axes, window: Window) -> None:
    span_days = (window.end_time - window.start_time).total_seconds() / 86400.0
    if span_days > 240:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    elif span_days > 90:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=15))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))


def run_plot(station: Station, window: Window) -> bool:
    saas = read_zhd_csv(zhd_saas_csv(station, window), window)
    profile = read_zhd_csv(zhd_profile_csv(station, window), window)
    quality = read_quality_summary(station, window)

    if saas is None and profile is None:
        print(f"Skip plot, no ZHD CSV files for {station.id} {window.id}.")
        return False

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.0, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )

    if saas is not None:
        axes[0].plot(
            saas["time_utc"],
            saas["ZHD_mm"],
            color="#2563eb",
            lw=1.4,
            marker="o",
            ms=2.6,
            mfc="white",
            mec="#2563eb",
            mew=0.8,
            label="Saastamoinen",
        )
    if profile is not None:
        axes[0].plot(
            profile["time_utc"],
            profile["ZHD_mm"],
            color="#dc2626",
            lw=1.4,
            marker="s",
            ms=2.4,
            mfc="white",
            mec="#dc2626",
            mew=0.8,
            label="Profile integral",
        )

    axes[0].set_ylabel("ZHD (mm)")
    axes[0].set_title(
        f"{station.name} ({station.id}) sounding ZHD - {window.label}",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(loc="best", frameon=True)

    if quality is not None and "levels" in quality.columns:
        colors = quality.get("passes_qc", pd.Series(False, index=quality.index)).map(
            {True: "#16a34a", False: "#94a3b8"}
        )
        axes[1].scatter(
            quality["sounding_time_utc"],
            quality["levels"],
            s=12,
            c=colors,
            alpha=0.85,
            edgecolors="none",
        )
        axes[1].axhline(
            MIN_SOUNDING_LEVELS,
            color="#475569",
            lw=1.0,
            ls="--",
            label=f"QC threshold ({MIN_SOUNDING_LEVELS} levels)",
        )
        axes[1].legend(loc="best", frameon=True)
    axes[1].set_ylabel("Levels")
    axes[1].set_xlabel("Time (UTC)")

    for ax in axes:
        ax.set_xlim(window.start_time, window.end_time)
        ax.grid(True, which="major", color="#d1d5db", lw=0.7, alpha=0.65)
        ax.grid(True, which="minor", color="#e5e7eb", lw=0.5, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    set_year_xaxis(axes[-1], window)
    fig.autofmt_xdate(rotation=30, ha="right")

    target = zhd_plot_png(station, window)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {target}")
    return True


def summarize_job(station: Station, window: Window) -> dict[str, Any]:
    expected_count = len(generate_expected_times(window))
    quality = read_quality_summary(station, window)

    row: dict[str, Any] = {
        "station_id": station.id,
        "station_name": station.name,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "year_window": window.id,
        "start_time": window.start_time,
        "end_time": window.end_time,
        "expected_profiles": expected_count,
        "downloaded_profiles": 0,
        "qc_pass_profiles": 0,
        "qc_pass_rate": 0.0,
        "levels_mean": None,
        "levels_min": None,
        "levels_max": None,
        "top_height_mean_m": None,
        "top_height_min_m": None,
        "top_height_max_m": None,
        "top_pressure_mean_hPa": None,
        "top_pressure_min_hPa": None,
        "top_pressure_max_hPa": None,
        "combined_csv": combined_csv(station, window),
        "qc_csv": qc_csv(station, window),
        "quality_summary_csv": quality_summary_csv(station, window),
        "sounding_zhd_saas_csv": zhd_saas_csv(station, window),
        "sounding_zhd_profile_csv": zhd_profile_csv(station, window),
        "plot_png": zhd_plot_png(station, window),
    }

    if quality is None or quality.empty:
        return row

    downloaded = quality[quality["levels"].fillna(0) > 0].copy()
    passed = quality[
        quality.get("passes_qc", pd.Series(False, index=quality.index)).fillna(False)
    ].copy()
    row["downloaded_profiles"] = int(len(downloaded))
    row["qc_pass_profiles"] = int(len(passed))
    row["qc_pass_rate"] = (
        float(len(passed) / expected_count) if expected_count else 0.0
    )

    for prefix, column in [
        ("levels", "levels"),
        ("top_height", "maximum_height_m"),
        ("top_pressure", "minimum_pressure_hPa"),
    ]:
        if column not in downloaded.columns or downloaded.empty:
            continue
        values = downloaded[column].dropna()
        if values.empty:
            continue
        if prefix == "top_height":
            row["top_height_mean_m"] = float(values.mean())
            row["top_height_min_m"] = float(values.min())
            row["top_height_max_m"] = float(values.max())
        elif prefix == "top_pressure":
            row["top_pressure_mean_hPa"] = float(values.mean())
            row["top_pressure_min_hPa"] = float(values.min())
            row["top_pressure_max_hPa"] = float(values.max())
        else:
            row["levels_mean"] = float(values.mean())
            row["levels_min"] = int(values.min())
            row["levels_max"] = int(values.max())

    return row


def write_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    summary_dir = OUT_ROOT / WORKFLOW_ID
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(
        summary_dir / "sounding_yearly_station_summary.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    year_summary = (
        df.groupby("year_window", as_index=False)
        .agg(
            stations=("station_id", "count"),
            expected_profiles=("expected_profiles", "sum"),
            downloaded_profiles=("downloaded_profiles", "sum"),
            qc_pass_profiles=("qc_pass_profiles", "sum"),
            levels_mean=("levels_mean", "mean"),
            top_height_mean_m=("top_height_mean_m", "mean"),
            top_pressure_mean_hPa=("top_pressure_mean_hPa", "mean"),
        )
        .sort_values("year_window")
    )
    year_summary["download_rate"] = (
        year_summary["downloaded_profiles"] / year_summary["expected_profiles"]
    )
    year_summary["qc_pass_rate"] = (
        year_summary["qc_pass_profiles"] / year_summary["expected_profiles"]
    )
    year_summary.to_csv(
        summary_dir / "sounding_yearly_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Station summary: {summary_dir / 'sounding_yearly_station_summary.csv'}")
    print(f"Overall summary: {summary_dir / 'sounding_yearly_overall_summary.csv'}")


def expand_steps(raw_steps: list[str]) -> list[str]:
    if raw_steps == ["all"]:
        return ALL_STEPS
    if raw_steps == ["results"]:
        return RESULT_STEPS

    steps: list[str] = []
    for step in raw_steps:
        if step == "all":
            steps.extend(ALL_STEPS)
        elif step == "results":
            steps.extend(RESULT_STEPS)
        else:
            steps.append(step)

    invalid = [step for step in steps if step not in ALL_STEPS]
    if invalid:
        raise ValueError(f"Unknown steps: {invalid}. Valid steps: {ALL_STEPS}")

    deduped: list[str] = []
    for step in steps:
        if step not in deduped:
            deduped.append(step)
    return deduped


def selected_stations(raw_station_ids: list[str] | None) -> list[Station]:
    if not raw_station_ids:
        return list(STATIONS)
    station_ids = {station_id.strip() for station_id in raw_station_ids}
    stations_by_id = {station.id: station for station in STATIONS}
    missing = sorted(station_ids - set(stations_by_id))
    if missing:
        raise ValueError(f"Unknown station ids: {missing}")
    return [station for station in STATIONS if station.id in station_ids]


def selected_windows(raw_window_ids: list[str] | None) -> list[Window]:
    if not raw_window_ids:
        return list(WINDOWS)
    window_ids = {window_id.strip() for window_id in raw_window_ids}
    windows_by_id = {window.id: window for window in WINDOWS}
    missing = sorted(window_ids - set(windows_by_id))
    if missing:
        raise ValueError(f"Unknown year windows: {missing}")
    return [window for window in WINDOWS if window.id in window_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download radiosonde data, run sounding-only ZHD calculation, "
            "and create yearly station plots for 2024 through 2026-07-28."
        )
    )
    parser.add_argument(
        "--station",
        action="append",
        help="Station WMO id to run. Repeat for multiple stations. Omit for all.",
    )
    parser.add_argument(
        "--year",
        action="append",
        choices=[window.id for window in WINDOWS],
        help="Year window to run: 2024, 2025, or 2026_to_20260728. Omit for all.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help="Steps: all, results, download, qc, zhd, plot, summarize.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List selected station/year jobs and exit.",
    )
    return parser.parse_args()


def run_job(station: Station, window: Window, steps: list[str]) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"{station.id}_{station.slug} {window.id}")
    print("=" * 78)
    print(f"Data dir: {data_dir(station, window)}")
    print(f"Out dir: {out_dir(station, window)}")

    should_continue = True
    for step in steps:
        try:
            print("\n" + "-" * 64)
            print(f"Step: {step}")
            print("-" * 64)
            if step == "download":
                run_download(station, window)
            elif step == "qc":
                should_continue = run_qc(station, window)
            elif step == "zhd":
                should_continue = run_zhd(station, window)
            elif step == "plot":
                should_continue = run_plot(station, window)
            elif step == "summarize":
                pass
            else:
                raise ValueError(f"Unknown step: {step}")
        except Exception as exc:
            print(f"Step failed and workflow will continue: {station.id} {window.id} {step}")
            print(f"{type(exc).__name__}: {exc}")
            should_continue = False

        if not should_continue and step in {"qc", "zhd", "plot"}:
            break

    return summarize_job(station, window)


def main() -> None:
    args = parse_args()
    stations = selected_stations(args.station)
    windows = selected_windows(args.year)
    steps = expand_steps(args.steps)

    print(f"Workflow: {WORKFLOW_ID}")
    print("Selected stations:", ", ".join(station.id for station in stations))
    print("Selected years:", ", ".join(window.id for window in windows))
    print("Selected steps:", ", ".join(steps))

    if args.list_jobs:
        for station in stations:
            for window in windows:
                print(
                    f"{station.id}_{station.slug}_{window.id}: "
                    f"{station.name}, lat={station.latitude}, lon={station.longitude}, "
                    f"{window.start_time:%Y-%m-%d %H:%M} to "
                    f"{window.end_time:%Y-%m-%d %H:%M}"
                )
        return

    rows: list[dict[str, Any]] = []
    for station in stations:
        for window in windows:
            rows.append(run_job(station, window, steps))

    write_summary(rows)


if __name__ == "__main__":
    main()
