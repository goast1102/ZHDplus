from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import calc_zhd_era5
import calc_zhd_sounding
import downloadera5
import downloadWyomingrs
import plot_zhd_timeseries


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("typhoon_events.json")

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
    "saas_grid4": "era5_zhd_saas_grid4.csv",
    "profile_grid4": "era5_zhd_profile_grid4.csv",
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


def load_events(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    events = raw.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Config must contain an 'events' list.")

    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event["id"]).strip().lower()
        if not event_id:
            raise ValueError("Event id cannot be empty.")
        if event_id in by_id:
            raise ValueError(f"Duplicate event id: {event_id}")
        event["id"] = event_id
        by_id[event_id] = event

    return by_id


def event_out_dir(event: dict[str, Any]) -> Path:
    return ROOT_DIR / "out" / event["id"]


def era5_data_dir(event: dict[str, Any]) -> Path:
    return resolve_path(event["era5"]["data_dir"])


def sounding_data_dir(event: dict[str, Any]) -> Path:
    return resolve_path(event["sounding"]["data_dir"])


def expected_sounding_combined(event: dict[str, Any]) -> Path:
    station = str(event["station"]["id"])
    start = datetime.fromisoformat(event["sounding"]["start_time"])
    end = datetime.fromisoformat(event["sounding"]["end_time"])
    return sounding_data_dir(event) / (
        f"{station}_{start:%Y%m%d%H}_{end:%Y%m%d%H}_combined.csv"
    )


def expected_sounding_qc_combined(event: dict[str, Any]) -> Path:
    combined = expected_sounding_combined(event)
    return combined.with_name(f"{combined.stem}_qc.csv")


def migrate_legacy_mawar_outputs(event: dict[str, Any]) -> None:
    if event["id"] != "mawar":
        return

    out_dir = event_out_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_files = {
        ROOT_DIR / "out" / "mawar_era5_zhd_saas_grid4.csv": out_dir / ERA5_OUTPUTS["saas_grid4"],
        ROOT_DIR / "out" / "mawar_era5_zhd_profile_grid4.csv": out_dir / ERA5_OUTPUTS["profile_grid4"],
        ROOT_DIR / "out" / "mawar_era5_zhd_saas_station_interp.csv": out_dir / ERA5_OUTPUTS["saas_station"],
        ROOT_DIR / "out" / "mawar_era5_zhd_profile_station_interp.csv": out_dir / ERA5_OUTPUTS["profile_station"],
        ROOT_DIR / "out" / "mawar_sounding_zhd_saas.csv": out_dir / SOUNDING_OUTPUTS["saas"],
        ROOT_DIR / "out" / "mawar_sounding_zhd_profile.csv": out_dir / SOUNDING_OUTPUTS["profile"],
        out_dir / "mawar_era5_zhd_saas_grid4.csv": out_dir / ERA5_OUTPUTS["saas_grid4"],
        out_dir / "mawar_era5_zhd_profile_grid4.csv": out_dir / ERA5_OUTPUTS["profile_grid4"],
        out_dir / "mawar_era5_zhd_saas_station_interp.csv": out_dir / ERA5_OUTPUTS["saas_station"],
        out_dir / "mawar_era5_zhd_profile_station_interp.csv": out_dir / ERA5_OUTPUTS["profile_station"],
        out_dir / "mawar_sounding_zhd_saas.csv": out_dir / SOUNDING_OUTPUTS["saas"],
        out_dir / "mawar_sounding_zhd_profile.csv": out_dir / SOUNDING_OUTPUTS["profile"],
    }

    for src, dst in legacy_files.items():
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def configure_era5_downloader(event: dict[str, Any], download: bool, extract: bool) -> None:
    base_dir = era5_data_dir(event)
    downloadera5.START_DATE = event["era5"]["start_date"]
    downloadera5.END_DATE = event["era5"]["end_date"]
    downloadera5.AREA = list(event["era5"]["area"])
    downloadera5.BASE_DIR = base_dir
    downloadera5.GRIB_DIR = base_dir / "grib"
    downloadera5.OUT_CSV = base_dir / "era5_pressure_raw_extract.csv"
    downloadera5.DOWNLOAD_ERA5 = download
    downloadera5.EXTRACT_TO_CSV = extract
    downloadera5.MAX_PARALLEL_DOWNLOADS = int(
        event["era5"].get("max_parallel_downloads", 3)
    )
    downloadera5.DOWNLOAD_STRATEGY = event["era5"].get("download_strategy", "auto")
    downloadera5.GRIB_DIR.mkdir(parents=True, exist_ok=True)
    downloadera5.OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def configure_sounding_downloader(event: dict[str, Any]) -> None:
    downloadWyomingrs.STATION = str(event["station"]["id"])
    downloadWyomingrs.START_TIME = datetime.fromisoformat(event["sounding"]["start_time"])
    downloadWyomingrs.END_TIME = datetime.fromisoformat(event["sounding"]["end_time"])
    downloadWyomingrs.SOUNDING_HOURS = list(event["sounding"].get("hours", [0, 12]))
    downloadWyomingrs.OUT_DIR = sounding_data_dir(event)
    downloadWyomingrs.OUT_DIR.mkdir(parents=True, exist_ok=True)


def sounding_quality_from_combined(combined_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(combined_csv)
    required = {"sounding_time_utc", "pressure_hPa", "geopotential height_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{combined_csv} is missing columns: {sorted(missing)}")

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


def run_sounding_quality_control(event: dict[str, Any]) -> bool:
    combined_csv = expected_sounding_combined(event)
    qc_csv = expected_sounding_qc_combined(event)
    summary_csv = event_out_dir(event) / "sounding_quality_summary.csv"
    log_csv = sounding_data_dir(event) / "download_log.csv"

    if not combined_csv.exists():
        print(f"Skip event: sounding combined CSV not found: {combined_csv}")
        if qc_csv.exists():
            qc_csv.unlink()
        return False

    if log_csv.exists():
        quality = pd.read_csv(log_csv)
        quality["sounding_time_utc"] = pd.to_datetime(
            quality["sounding_time_utc"],
            errors="coerce",
        )
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
        print("Skip event: no sounding profiles passed QC.")
        if qc_csv.exists():
            qc_csv.unlink()
        return False

    combined = pd.read_csv(combined_csv)
    combined["sounding_time_utc"] = pd.to_datetime(
        combined["sounding_time_utc"],
        errors="coerce",
    )
    filtered = combined[
        combined["sounding_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S").isin(
            good_time_strings
        )
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


def configure_era5_zhd(event: dict[str, Any]) -> None:
    out_dir = event_out_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc_zhd_era5.INPUT_CSV = era5_data_dir(event) / "era5_pressure_raw_extract.csv"
    calc_zhd_era5.OUT_SAAS_GRID4_CSV = out_dir / ERA5_OUTPUTS["saas_grid4"]
    calc_zhd_era5.OUT_PROFILE_GRID4_CSV = out_dir / ERA5_OUTPUTS["profile_grid4"]
    calc_zhd_era5.OUT_SAAS_STATION_CSV = out_dir / ERA5_OUTPUTS["saas_station"]
    calc_zhd_era5.OUT_PROFILE_STATION_CSV = out_dir / ERA5_OUTPUTS["profile_station"]
    calc_zhd_era5.STATION_LAT = float(event["station"]["latitude"])
    calc_zhd_era5.STATION_LON = float(event["station"]["longitude"])


def configure_sounding_zhd(event: dict[str, Any]) -> None:
    out_dir = event_out_dir(event)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc_zhd_sounding.INPUT_CSV = expected_sounding_qc_combined(event)
    calc_zhd_sounding.OUT_SAAS_CSV = out_dir / SOUNDING_OUTPUTS["saas"]
    calc_zhd_sounding.OUT_PROFILE_CSV = out_dir / SOUNDING_OUTPUTS["profile"]


def configure_plot(event: dict[str, Any]) -> None:
    out_dir = event_out_dir(event)
    plot_cfg = event["plot"]
    display_name = event.get("display_name", event["id"].title())
    region_name = event.get("region_name", "")

    plot_zhd_timeseries.OUT_DIR = out_dir
    plot_zhd_timeseries.RS_SAAS_CSV = out_dir / SOUNDING_OUTPUTS["saas"]
    plot_zhd_timeseries.RS_PROFILE_CSV = out_dir / SOUNDING_OUTPUTS["profile"]
    plot_zhd_timeseries.ERA5_SAAS_GRID4_CSV = out_dir / ERA5_OUTPUTS["saas_grid4"]
    plot_zhd_timeseries.ERA5_PROFILE_GRID4_CSV = out_dir / ERA5_OUTPUTS["profile_grid4"]
    plot_zhd_timeseries.ERA5_SAAS_STATION_CSV = out_dir / ERA5_OUTPUTS["saas_station"]
    plot_zhd_timeseries.ERA5_PROFILE_STATION_CSV = out_dir / ERA5_OUTPUTS["profile_station"]
    plot_zhd_timeseries.OUT_PNG = out_dir / "zhd_timeseries_rs_era5_era5inter.png"

    plot_zhd_timeseries.TIME_START = pd.Timestamp(plot_cfg["time_start"])
    plot_zhd_timeseries.PRE_END = pd.Timestamp(plot_cfg["pre_end"])
    plot_zhd_timeseries.DURING_END = pd.Timestamp(plot_cfg["impact_end"])
    plot_zhd_timeseries.TIME_END = pd.Timestamp(plot_cfg["time_end"])
    plot_zhd_timeseries.STAGES = [
        (
            plot_zhd_timeseries.TIME_START,
            plot_zhd_timeseries.PRE_END,
            f"Pre-{display_name}",
            "#dbeafe",
        ),
        (
            plot_zhd_timeseries.PRE_END,
            plot_zhd_timeseries.DURING_END,
            f"{display_name} impact",
            "#fee2e2",
        ),
        (
            plot_zhd_timeseries.DURING_END,
            plot_zhd_timeseries.TIME_END,
            f"Post-{display_name}",
            "#dcfce7",
        ),
    ]

    if region_name:
        title = f"ZHD Time Series During Typhoon {display_name} Around {region_name}"
    else:
        title = f"ZHD Time Series During Typhoon {display_name}"

    plot_zhd_timeseries.PLOT_TITLE = title
    if hasattr(plot_zhd_timeseries, "_original_main"):
        plot_zhd_timeseries.main = plot_zhd_timeseries._original_main
    return

    original_main = plot_zhd_timeseries.main

    def main_with_title() -> None:
        rs_saas = plot_zhd_timeseries.read_zhd_csv(plot_zhd_timeseries.RS_SAAS_CSV)
        rs_profile = plot_zhd_timeseries.read_zhd_csv(plot_zhd_timeseries.RS_PROFILE_CSV)
        era5_saas_grid = plot_zhd_timeseries.read_zhd_csv(
            plot_zhd_timeseries.ERA5_SAAS_GRID4_CSV
        )
        era5_profile_grid = plot_zhd_timeseries.read_zhd_csv(
            plot_zhd_timeseries.ERA5_PROFILE_GRID4_CSV
        )
        era5_saas_station = plot_zhd_timeseries.read_zhd_csv(
            plot_zhd_timeseries.ERA5_SAAS_STATION_CSV
        )
        era5_profile_station = plot_zhd_timeseries.read_zhd_csv(
            plot_zhd_timeseries.ERA5_PROFILE_STATION_CSV
        )

        fig, axes = plot_zhd_timeseries.plt.subplots(
            3,
            1,
            figsize=(13.5, 9.2),
            sharex=True,
            constrained_layout=True,
        )

        plot_zhd_timeseries.add_stage_background(axes[0], show_labels=True)
        plot_zhd_timeseries.add_stage_background(axes[1])
        plot_zhd_timeseries.add_stage_background(axes[2])

        plot_zhd_timeseries.plot_rs(axes[0], rs_saas, rs_profile)
        plot_zhd_timeseries.plot_era5_grid(axes[1], era5_saas_grid, era5_profile_grid)
        plot_zhd_timeseries.plot_era5_station(
            axes[2],
            era5_saas_station,
            era5_profile_station,
        )

        for ax in axes:
            plot_zhd_timeseries.polish_axis(ax)
            ax.legend(
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="#cbd5e1",
                framealpha=0.92,
                ncol=2,
            )

        axes[-1].set_xlabel("Time (UTC)")
        axes[-1].xaxis.set_major_locator(
            plot_zhd_timeseries.mdates.HourLocator(byhour=[0, 12])
        )
        axes[-1].xaxis.set_major_formatter(
            plot_zhd_timeseries.mdates.DateFormatter("%m-%d\n%H:%M")
        )
        axes[-1].xaxis.set_minor_locator(
            plot_zhd_timeseries.mdates.HourLocator(interval=6)
        )

        fig.suptitle(title, fontsize=17, fontweight="bold", y=1.02)

        plot_zhd_timeseries.OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_zhd_timeseries.OUT_PNG, dpi=600, bbox_inches="tight")
        plot_zhd_timeseries.plt.close(fig)

        print("Figure file:")
        print(plot_zhd_timeseries.OUT_PNG)

    plot_zhd_timeseries.main = main_with_title
    plot_zhd_timeseries._original_main = original_main


def missing_plot_inputs(event: dict[str, Any]) -> list[Path]:
    out_dir = event_out_dir(event)
    return [
        path
        for path in [
            out_dir / SOUNDING_OUTPUTS["saas"],
            out_dir / SOUNDING_OUTPUTS["profile"],
            out_dir / ERA5_OUTPUTS["saas_grid4"],
            out_dir / ERA5_OUTPUTS["profile_grid4"],
            out_dir / ERA5_OUTPUTS["saas_station"],
            out_dir / ERA5_OUTPUTS["profile_station"],
        ]
        if not path.exists()
    ]


def run_step(event: dict[str, Any], step: str) -> bool:
    print("\n" + "=" * 72)
    print(f"{event['id']}: {step}")
    print("=" * 72)

    if step == "era5_download":
        configure_era5_downloader(event, download=True, extract=False)
        downloadera5.main()
    elif step == "era5_extract":
        configure_era5_downloader(event, download=False, extract=True)
        downloadera5.main()
    elif step == "sounding_download":
        configure_sounding_downloader(event)
        downloadWyomingrs.main()
    elif step == "sounding_qc":
        return run_sounding_quality_control(event)
    elif step == "era5_zhd":
        configure_era5_zhd(event)
        calc_zhd_era5.main()
    elif step == "sounding_zhd":
        sounding_csv = expected_sounding_qc_combined(event)
        if not sounding_csv.exists():
            print(f"Skip sounding_zhd: input CSV not found: {sounding_csv}")
            return True
        configure_sounding_zhd(event)
        calc_zhd_sounding.main()
    elif step == "plot":
        migrate_legacy_mawar_outputs(event)
        missing_inputs = missing_plot_inputs(event)
        if len(missing_inputs) == 6:
            print("Skip plot: no input files found:")
            for path in missing_inputs:
                print(path)
            return True
        if missing_inputs:
            print("Plot with available inputs; missing files:")
            for path in missing_inputs:
                print(path)
        configure_plot(event)
        plot_zhd_timeseries.main()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ERA5/radiosonde ZHD workflow for configured typhoon events."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to typhoon_events.json.",
    )
    parser.add_argument(
        "--event",
        action="append",
        help="Event id to run. Repeat for multiple events. Omit to run all events.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help=(
            "Workflow steps: results, all, era5_download, era5_extract, "
            "sounding_download, sounding_qc, era5_zhd, sounding_zhd, plot."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.config)
    steps = expand_steps(args.steps)

    selected_ids = (
        [event_id.lower() for event_id in args.event]
        if args.event
        else [
            event_id
            for event_id, event in events.items()
            if event.get("enabled", True)
        ]
    )
    missing = [event_id for event_id in selected_ids if event_id not in events]
    if missing:
        raise ValueError(f"Events not found in config: {missing}")

    print("Selected events:", ", ".join(selected_ids))
    print("Selected steps:", ", ".join(steps))

    for event_id in selected_ids:
        event = events[event_id]
        event_out_dir(event).mkdir(parents=True, exist_ok=True)
        for step in steps:
            try:
                should_continue_event = run_step(event, step)
                if not should_continue_event:
                    print(f"Stop current event after step: {event_id} {step}")
                    break
            except Exception as exc:
                print(f"Step failed and workflow will continue: {event_id} {step}")
                print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
