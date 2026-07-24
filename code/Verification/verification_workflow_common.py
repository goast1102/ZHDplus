from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT_DIR / "code"
CONFIG_JSON = Path(__file__).with_name("verification_stations_2025.json")

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import calc_zhd_era5  # noqa: E402
import calc_zhd_sounding  # noqa: E402
import downloadera5  # noqa: E402
import downloadWyomingrs  # noqa: E402
import plot_zhd_timeseries  # noqa: E402


RESULT_STEPS = [
    "sounding_qc",
    "era5_zhd",
    "sounding_zhd",
    "plot",
]

ALL_STEPS = [
    "sounding_download",
    "sounding_qc",
    "era5_download",
    "era5_extract",
    "era5_zhd",
    "sounding_zhd",
    "plot",
]

ERA5_OUTPUTS = {
    "saas_grid": "era5_zhd_saas_grid.csv",
    "profile_grid": "era5_zhd_profile_grid.csv",
    "saas_station": "era5_zhd_saas_station_interp.csv",
    "profile_station": "era5_zhd_profile_station_interp.csv",
}

SOUNDING_OUTPUTS = {
    "saas": "sounding_zhd_saas.csv",
    "profile": "sounding_zhd_profile.csv",
}

MIN_SOUNDING_LEVELS = 80
MIN_SOUNDING_TOP_HEIGHT_M = 20000.0
MAX_SOUNDING_TOP_PRESSURE_HPA = 50.0


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_config(config_path: Path = CONFIG_JSON) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def selected_workflow(config: dict[str, Any], workflow_id: str) -> dict[str, Any]:
    try:
        return config["workflow_sets"][workflow_id]
    except KeyError as exc:
        raise ValueError(f"Workflow set not found: {workflow_id}") from exc


def build_jobs(config: dict[str, Any], workflow_id: str) -> dict[str, dict[str, Any]]:
    workflow = selected_workflow(config, workflow_id)
    stations = {str(station["id"]): station for station in config["stations"]}

    jobs: dict[str, dict[str, Any]] = {}
    for station_id in workflow["station_ids"]:
        if station_id not in stations:
            raise ValueError(f"Station not found in catalog: {station_id}")
        station = stations[station_id]
        for window in workflow["windows"]:
            values = {
                "workflow_id": workflow_id,
                "station_id": station_id,
                "slug": station["slug"],
                "window_id": window["id"],
            }
            job_id = f"{station_id}_{station['slug']}_{window['id']}".lower()
            if job_id in jobs:
                raise ValueError(f"Duplicate job id: {job_id}")
            jobs[job_id] = {
                "id": job_id,
                "workflow_id": workflow_id,
                "display_name": station["name"],
                "window_label": window.get("label", window["id"]),
                "station": {
                    "id": station_id,
                    "name": station["name"],
                    "latitude": float(station["latitude"]),
                    "longitude": float(station["longitude"]),
                },
                "era5": {
                    "data_dir": workflow["era5_data_dir_template"].format(**values),
                    "area": list(station["era5_area"]),
                    "start_date": window["era5_start_date"],
                    "end_date": window["era5_end_date"],
                    "max_parallel_downloads": int(workflow.get("max_parallel_downloads", 3)),
                    "download_strategy": workflow.get("download_strategy", "auto"),
                },
                "sounding": {
                    "data_dir": workflow["sounding_data_dir_template"].format(**values),
                    "start_time": window["start_time"],
                    "end_time": window["end_time"],
                    "hours": list(config.get("sounding", {}).get("hours", [0, 12])),
                },
                "plot": {
                    "time_start": window["start_time"],
                    "time_end": window["end_time"],
                },
                "out_dir": workflow["output_dir_template"].format(**values),
            }
    return jobs


def job_out_dir(job: dict[str, Any]) -> Path:
    return resolve_path(job["out_dir"])


def era5_data_dir(job: dict[str, Any]) -> Path:
    return resolve_path(job["era5"]["data_dir"])


def sounding_data_dir(job: dict[str, Any]) -> Path:
    return resolve_path(job["sounding"]["data_dir"])


def expected_sounding_combined(job: dict[str, Any]) -> Path:
    station = str(job["station"]["id"])
    start = datetime.fromisoformat(job["sounding"]["start_time"])
    end = datetime.fromisoformat(job["sounding"]["end_time"])
    return sounding_data_dir(job) / f"{station}_{start:%Y%m%d%H}_{end:%Y%m%d%H}_combined.csv"


def expected_sounding_qc_combined(job: dict[str, Any]) -> Path:
    combined = expected_sounding_combined(job)
    return combined.with_name(f"{combined.stem}_qc.csv")


def configure_sounding_downloader(job: dict[str, Any]) -> None:
    downloadWyomingrs.STATION = str(job["station"]["id"])
    downloadWyomingrs.START_TIME = datetime.fromisoformat(job["sounding"]["start_time"])
    downloadWyomingrs.END_TIME = datetime.fromisoformat(job["sounding"]["end_time"])
    downloadWyomingrs.SOUNDING_HOURS = list(job["sounding"].get("hours", [0, 12]))
    downloadWyomingrs.OUT_DIR = sounding_data_dir(job)
    downloadWyomingrs.OUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_era5_downloader(job: dict[str, Any], download: bool, extract: bool) -> None:
    base_dir = era5_data_dir(job)
    downloadera5.START_DATE = job["era5"]["start_date"]
    downloadera5.END_DATE = job["era5"]["end_date"]
    downloadera5.AREA = list(job["era5"]["area"])
    downloadera5.BASE_DIR = base_dir
    downloadera5.GRIB_DIR = base_dir / "grib"
    downloadera5.OUT_CSV = base_dir / "era5_pressure_raw_extract.csv"
    downloadera5.DOWNLOAD_ERA5 = download
    downloadera5.EXTRACT_TO_CSV = extract
    downloadera5.MAX_PARALLEL_DOWNLOADS = int(job["era5"].get("max_parallel_downloads", 3))
    downloadera5.DOWNLOAD_STRATEGY = job["era5"].get("download_strategy", "auto")
    downloadera5.GRIB_DIR.mkdir(parents=True, exist_ok=True)
    downloadera5.OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def configure_era5_zhd(job: dict[str, Any]) -> None:
    out_dir = job_out_dir(job)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc_zhd_era5.INPUT_CSV = era5_data_dir(job) / "era5_pressure_raw_extract.csv"
    calc_zhd_era5.OUT_SAAS_GRID4_CSV = out_dir / ERA5_OUTPUTS["saas_grid"]
    calc_zhd_era5.OUT_PROFILE_GRID4_CSV = out_dir / ERA5_OUTPUTS["profile_grid"]
    calc_zhd_era5.OUT_SAAS_STATION_CSV = out_dir / ERA5_OUTPUTS["saas_station"]
    calc_zhd_era5.OUT_PROFILE_STATION_CSV = out_dir / ERA5_OUTPUTS["profile_station"]
    calc_zhd_era5.STATION_LAT = float(job["station"]["latitude"])
    calc_zhd_era5.STATION_LON = float(job["station"]["longitude"])


def configure_sounding_zhd(job: dict[str, Any]) -> None:
    out_dir = job_out_dir(job)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc_zhd_sounding.INPUT_CSV = expected_sounding_qc_combined(job)
    calc_zhd_sounding.OUT_SAAS_CSV = out_dir / SOUNDING_OUTPUTS["saas"]
    calc_zhd_sounding.OUT_PROFILE_CSV = out_dir / SOUNDING_OUTPUTS["profile"]


def sounding_quality_from_combined(combined_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(combined_csv)
    required = {"sounding_time_utc", "pressure_hPa", "geopotential height_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{combined_csv} is missing columns: {sorted(missing)}")

    df["sounding_time_utc"] = pd.to_datetime(df["sounding_time_utc"], errors="coerce")
    df["pressure_hPa"] = pd.to_numeric(df["pressure_hPa"], errors="coerce")
    df["geopotential height_m"] = pd.to_numeric(df["geopotential height_m"], errors="coerce")

    valid = df.dropna(subset=["sounding_time_utc", "pressure_hPa", "geopotential height_m"])
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


def run_sounding_quality_control(job: dict[str, Any]) -> bool:
    combined_csv = expected_sounding_combined(job)
    qc_csv = expected_sounding_qc_combined(job)
    summary_csv = job_out_dir(job) / "sounding_quality_summary.csv"
    log_csv = sounding_data_dir(job) / "download_log.csv"

    if not combined_csv.exists():
        print(f"Skip job: sounding combined CSV not found: {combined_csv}")
        if qc_csv.exists():
            qc_csv.unlink()
        return False

    if log_csv.exists():
        quality = pd.read_csv(log_csv)
        quality["sounding_time_utc"] = pd.to_datetime(quality["sounding_time_utc"], errors="coerce")
        for col in ["levels", "minimum_pressure_hPa", "maximum_height_m"]:
            quality[col] = pd.to_numeric(quality[col], errors="coerce")
    else:
        quality = sounding_quality_from_combined(combined_csv)

    quality["passes_qc"] = (
        (quality["levels"] >= MIN_SOUNDING_LEVELS)
        & (quality["maximum_height_m"] >= MIN_SOUNDING_TOP_HEIGHT_M)
        & (quality["minimum_pressure_hPa"] <= MAX_SOUNDING_TOP_PRESSURE_HPA)
    )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    good_times = quality.loc[quality["passes_qc"], "sounding_time_utc"].dropna()
    good_time_strings = set(good_times.dt.strftime("%Y-%m-%d %H:%M:%S"))
    print(
        "Sounding QC: "
        f"{len(good_time_strings)}/{len(quality)} profiles pass "
        f"(levels>={MIN_SOUNDING_LEVELS}, "
        f"top height>={MIN_SOUNDING_TOP_HEIGHT_M / 1000:.0f} km, "
        f"top pressure<={MAX_SOUNDING_TOP_PRESSURE_HPA:.0f} hPa)."
    )
    print(f"Sounding QC summary: {summary_csv}")

    if not good_time_strings:
        print("Skip job: no sounding profiles passed QC.")
        if qc_csv.exists():
            qc_csv.unlink()
        return False

    combined = pd.read_csv(combined_csv)
    combined["sounding_time_utc"] = pd.to_datetime(combined["sounding_time_utc"], errors="coerce")
    filtered = combined[
        combined["sounding_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S").isin(good_time_strings)
    ].copy()
    filtered.to_csv(
        qc_csv,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    print(f"Sounding QC combined file: {qc_csv}")
    print(f"Sounding QC rows: {len(filtered)}")
    return True


def stage_backgrounds(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    first = start + (end - start) / 3
    second = start + (end - start) * 2 / 3
    return [
        (start, first, "Early", "#dbeafe"),
        (first, second, "Middle", "#fee2e2"),
        (second, end, "Late", "#dcfce7"),
    ]


def configure_plot_globals(job: dict[str, Any]) -> None:
    out_dir = job_out_dir(job)
    start = pd.Timestamp(job["plot"]["time_start"])
    end = pd.Timestamp(job["plot"]["time_end"])

    plot_zhd_timeseries.OUT_DIR = out_dir
    plot_zhd_timeseries.RS_SAAS_CSV = out_dir / SOUNDING_OUTPUTS["saas"]
    plot_zhd_timeseries.RS_PROFILE_CSV = out_dir / SOUNDING_OUTPUTS["profile"]
    plot_zhd_timeseries.ERA5_SAAS_GRID4_CSV = out_dir / ERA5_OUTPUTS["saas_grid"]
    plot_zhd_timeseries.ERA5_PROFILE_GRID4_CSV = out_dir / ERA5_OUTPUTS["profile_grid"]
    plot_zhd_timeseries.ERA5_SAAS_STATION_CSV = out_dir / ERA5_OUTPUTS["saas_station"]
    plot_zhd_timeseries.ERA5_PROFILE_STATION_CSV = out_dir / ERA5_OUTPUTS["profile_station"]
    plot_zhd_timeseries.OUT_PNG = out_dir / "zhd_timeseries_rs_era5_era5inter.png"
    plot_zhd_timeseries.PLOT_TITLE = (
        f"ZHD Time Series for {job['display_name']} "
        f"({job['station']['id']}) - {job['window_label']}"
    )
    plot_zhd_timeseries.TIME_START = start
    plot_zhd_timeseries.PRE_END = start + (end - start) / 3
    plot_zhd_timeseries.DURING_END = start + (end - start) * 2 / 3
    plot_zhd_timeseries.TIME_END = end
    plot_zhd_timeseries.STAGES = stage_backgrounds(start, end)


def set_adaptive_xaxis(ax: Any, start: pd.Timestamp, end: pd.Timestamp) -> None:
    span_days = max((end - start).total_seconds() / 86400.0, 0.0)
    if span_days > 120:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    elif span_days > 35:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))


def run_verification_plot(job: dict[str, Any]) -> None:
    configure_plot_globals(job)

    rs_saas = plot_zhd_timeseries.read_optional_zhd_csv(plot_zhd_timeseries.RS_SAAS_CSV)
    rs_profile = plot_zhd_timeseries.read_optional_zhd_csv(plot_zhd_timeseries.RS_PROFILE_CSV)
    era5_saas_grid = plot_zhd_timeseries.read_optional_zhd_csv(
        plot_zhd_timeseries.ERA5_SAAS_GRID4_CSV
    )
    era5_profile_grid = plot_zhd_timeseries.read_optional_zhd_csv(
        plot_zhd_timeseries.ERA5_PROFILE_GRID4_CSV
    )
    era5_saas_station = plot_zhd_timeseries.read_optional_zhd_csv(
        plot_zhd_timeseries.ERA5_SAAS_STATION_CSV
    )
    era5_profile_station = plot_zhd_timeseries.read_optional_zhd_csv(
        plot_zhd_timeseries.ERA5_PROFILE_STATION_CSV
    )

    panels = []
    if rs_saas is not None or rs_profile is not None:
        panels.append(("rs", plot_zhd_timeseries.plot_rs, rs_saas, rs_profile))
    if era5_saas_grid is not None or era5_profile_grid is not None:
        panels.append(("era5", plot_zhd_timeseries.plot_era5_grid, era5_saas_grid, era5_profile_grid))
    if era5_saas_station is not None or era5_profile_station is not None:
        panels.append(
            ("era5_inter", plot_zhd_timeseries.plot_era5_station, era5_saas_station, era5_profile_station)
        )
    if not panels:
        raise FileNotFoundError("No available ZHD CSV files for plotting.")

    fig, axes = plot_zhd_timeseries.plt.subplots(
        len(panels),
        1,
        figsize=(13.5, max(3.2, 3.1 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for index, (panel_name, plot_func, first_df, second_df) in enumerate(panels):
        plot_zhd_timeseries.add_stage_background(axes[index], show_labels=index == 0)
        plot_func(axes[index], first_df, second_df)
        axes[index].set_title(f"{chr(97 + index)}  {panel_name}", loc="left", fontweight="bold")

    for ax in axes:
        plot_zhd_timeseries.polish_axis(ax)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                frameon=True,
                facecolor="white",
                edgecolor="#cbd5e1",
                framealpha=0.92,
                ncol=1,
            )

    axes[-1].set_xlabel("Time (UTC)")
    set_adaptive_xaxis(axes[-1], plot_zhd_timeseries.TIME_START, plot_zhd_timeseries.TIME_END)
    fig.suptitle(plot_zhd_timeseries.PLOT_TITLE, fontsize=17, fontweight="bold", y=1.02)

    plot_zhd_timeseries.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_zhd_timeseries.OUT_PNG, dpi=600, bbox_inches="tight")
    plot_zhd_timeseries.plt.close(fig)

    print("Figure file:")
    print(plot_zhd_timeseries.OUT_PNG)


def missing_plot_inputs(job: dict[str, Any]) -> list[Path]:
    out_dir = job_out_dir(job)
    return [
        path
        for path in [
            out_dir / SOUNDING_OUTPUTS["saas"],
            out_dir / SOUNDING_OUTPUTS["profile"],
            out_dir / ERA5_OUTPUTS["saas_grid"],
            out_dir / ERA5_OUTPUTS["profile_grid"],
            out_dir / ERA5_OUTPUTS["saas_station"],
            out_dir / ERA5_OUTPUTS["profile_station"],
        ]
        if not path.exists()
    ]


def run_step(job: dict[str, Any], step: str) -> bool:
    print("\n" + "=" * 72)
    print(f"{job['id']}: {step}")
    print("=" * 72)

    if step == "sounding_download":
        configure_sounding_downloader(job)
        downloadWyomingrs.main()
    elif step == "sounding_qc":
        return run_sounding_quality_control(job)
    elif step == "era5_download":
        configure_era5_downloader(job, download=True, extract=False)
        downloadera5.main()
    elif step == "era5_extract":
        configure_era5_downloader(job, download=False, extract=True)
        downloadera5.main()
    elif step == "era5_zhd":
        configure_era5_zhd(job)
        calc_zhd_era5.main()
    elif step == "sounding_zhd":
        sounding_csv = expected_sounding_qc_combined(job)
        if not sounding_csv.exists():
            print(f"Skip sounding_zhd: input CSV not found: {sounding_csv}")
            return True
        configure_sounding_zhd(job)
        calc_zhd_sounding.main()
    elif step == "plot":
        missing_inputs = missing_plot_inputs(job)
        if len(missing_inputs) == 6:
            print("Skip plot: no input files found:")
            for path in missing_inputs:
                print(path)
            return True
        if missing_inputs:
            print("Plot with available inputs; missing files:")
            for path in missing_inputs:
                print(path)
        run_verification_plot(job)
    else:
        raise ValueError(f"Unknown workflow step: {step}")

    return True


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


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=CONFIG_JSON)
    parser.add_argument(
        "--job",
        action="append",
        help="Job id to run. Repeat for multiple jobs. Omit to run all jobs in this workflow set.",
    )
    parser.add_argument(
        "--station",
        action="append",
        help="Station WMO id to run. Repeat for multiple stations.",
    )
    parser.add_argument(
        "--window",
        action="append",
        help="Window id to run, for example 2025_january or 2025_full_year.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help=(
            "Workflow steps: results, all, sounding_download, sounding_qc, "
            "era5_download, era5_extract, era5_zhd, sounding_zhd, plot."
        ),
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List selected jobs and exit without running workflow steps.",
    )
    return parser.parse_args()


def run_workflow(workflow_id: str, description: str) -> None:
    args = parse_args(description)
    config = load_config(args.config)
    jobs = build_jobs(config, workflow_id)
    steps = expand_steps(args.steps)

    selected_ids = list(jobs)
    if args.station:
        stations = {station.strip() for station in args.station}
        selected_ids = [
            job_id
            for job_id in selected_ids
            if str(jobs[job_id]["station"]["id"]) in stations
        ]
    if args.window:
        windows = {window.strip() for window in args.window}
        selected_ids = [
            job_id
            for job_id in selected_ids
            if job_id.endswith(tuple(f"_{window}" for window in windows))
        ]
    if args.job:
        requested = [job_id.lower() for job_id in args.job]
        missing = [job_id for job_id in requested if job_id not in jobs]
        if missing:
            raise ValueError(f"Jobs not found: {missing}")
        selected_ids = requested

    print("Workflow set:", workflow_id)
    print("Selected jobs:", ", ".join(selected_ids))
    print("Selected steps:", ", ".join(steps))

    if args.list_jobs:
        for job_id in selected_ids:
            job = jobs[job_id]
            print(
                f"{job_id}: station={job['station']['id']} "
                f"name={job['display_name']} "
                f"window={job['window_label']} "
                f"era5_area={job['era5']['area']}"
            )
        return

    for job_id in selected_ids:
        job = jobs[job_id]
        job_out_dir(job).mkdir(parents=True, exist_ok=True)
        for step in steps:
            try:
                should_continue_job = run_step(job, step)
                if not should_continue_job:
                    print(f"Stop current job after step: {job_id} {step}")
                    break
            except Exception as exc:
                print(f"Step failed and workflow will continue: {job_id} {step}")
                print(f"{type(exc).__name__}: {exc}")
