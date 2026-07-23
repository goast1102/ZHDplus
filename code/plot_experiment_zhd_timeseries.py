from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_JSON = Path(__file__).with_name("typhoon_events.json")
OUT_DIR = ROOT_DIR / "out1"
PLOT_DIR = OUT_DIR / "experiment_timeseries_all_events"

ERA5_PROFILES_CSV = OUT_DIR / "zhd_dz_experiment_era5_profiles.csv"
SOUNDING_PROFILES_CSV = OUT_DIR / "zhd_dz_experiment_sounding_profiles.csv"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
    }
)

COLORS = {
    "saas": "#2563eb",
    "current": "#dc2626",
    "geometric": "#16a34a",
}


def load_events() -> dict[str, dict[str, Any]]:
    with CONFIG_JSON.open("r", encoding="utf-8") as f:
        events = json.load(f)["events"]
    return {str(event["id"]).lower(): event for event in events}


def read_profiles(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Skip missing experiment CSV: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {
        "event_id",
        "time_utc",
        "zhd_saas_bottom_m",
        "zhd_profile_current_m",
        "zhd_profile_geometric_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["event_id"] = df["event_id"].astype(str).str.lower()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    for col in [
        "zhd_saas_bottom_m",
        "zhd_profile_current_m",
        "zhd_profile_geometric_m",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["event_id", "time_utc"]).copy()


def prepare_sounding(df: pd.DataFrame, event_id: str, event: dict[str, Any]) -> pd.DataFrame:
    out = filter_event_time(df, event_id, event)
    if out.empty:
        return out

    return (
        out.groupby("time_utc", as_index=False)
        .agg(
            saas_mm=("zhd_saas_bottom_m", lambda x: x.mean() * 1000.0),
            current_mm=("zhd_profile_current_m", lambda x: x.mean() * 1000.0),
            geometric_mm=("zhd_profile_geometric_m", lambda x: x.mean() * 1000.0),
        )
        .sort_values("time_utc")
        .reset_index(drop=True)
    )


def prepare_era5(df: pd.DataFrame, event_id: str, event: dict[str, Any]) -> pd.DataFrame:
    out = filter_event_time(df, event_id, event)
    if out.empty:
        return out

    return (
        out.groupby("time_utc", as_index=False)
        .agg(
            saas_mean_mm=("zhd_saas_bottom_m", lambda x: x.mean() * 1000.0),
            current_mean_mm=("zhd_profile_current_m", lambda x: x.mean() * 1000.0),
            current_min_mm=("zhd_profile_current_m", lambda x: x.min() * 1000.0),
            current_max_mm=("zhd_profile_current_m", lambda x: x.max() * 1000.0),
            geometric_mean_mm=("zhd_profile_geometric_m", lambda x: x.mean() * 1000.0),
            geometric_min_mm=("zhd_profile_geometric_m", lambda x: x.min() * 1000.0),
            geometric_max_mm=("zhd_profile_geometric_m", lambda x: x.max() * 1000.0),
            n_grid=("zhd_profile_geometric_m", "size"),
        )
        .sort_values("time_utc")
        .reset_index(drop=True)
    )


def filter_event_time(
    df: pd.DataFrame,
    event_id: str,
    event: dict[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    plot_cfg = event["plot"]
    time_start = pd.Timestamp(plot_cfg["time_start"])
    time_end = pd.Timestamp(plot_cfg["time_end"])
    out = df[df["event_id"] == event_id].copy()
    out = out[(out["time_utc"] >= time_start) & (out["time_utc"] <= time_end)]
    return out


def stage_times(event: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    plot_cfg = event["plot"]
    return (
        pd.Timestamp(plot_cfg["time_start"]),
        pd.Timestamp(plot_cfg["pre_end"]),
        pd.Timestamp(plot_cfg["impact_end"]),
        pd.Timestamp(plot_cfg["time_end"]),
    )


def add_stage_background(ax: plt.Axes, event: dict[str, Any], show_labels: bool) -> None:
    display_name = event.get("display_name", event["id"].title())
    time_start, pre_end, impact_end, time_end = stage_times(event)
    stages = [
        (time_start, pre_end, f"Pre-{display_name}", "#dbeafe"),
        (pre_end, impact_end, f"{display_name} impact", "#fee2e2"),
        (impact_end, time_end, f"Post-{display_name}", "#dcfce7"),
    ]

    for start, end, label, color in stages:
        ax.axvspan(start, end, color=color, alpha=0.55, lw=0, zorder=0)
        if show_labels:
            center = start + (end - start) / 2
            ax.text(
                center,
                1.06,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                color="#334155",
            )

    for boundary in [pre_end, impact_end]:
        ax.axvline(boundary, color="#475569", lw=1.0, ls="--", alpha=0.65, zorder=1)


def polish_axis(ax: plt.Axes, event: dict[str, Any]) -> None:
    time_start, _, _, time_end = stage_times(event)
    ax.set_xlim(time_start, time_end)
    ax.grid(True, which="major", axis="y", color="#cbd5e1", lw=0.8, alpha=0.65)
    ax.grid(True, which="major", axis="x", color="#e2e8f0", lw=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0, colors="#334155")
    ax.set_ylabel("ZHD (mm)")


def plot_sounding(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.plot(
        df["time_utc"],
        df["saas_mm"],
        color=COLORS["saas"],
        lw=2.0,
        marker="o",
        ms=4.8,
        mfc="white",
        mec=COLORS["saas"],
        mew=1.2,
        label="Saastamoinen",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["current_mm"],
        color=COLORS["current"],
        lw=2.0,
        marker="s",
        ms=4.5,
        mfc="white",
        mec=COLORS["current"],
        mew=1.2,
        label="Current dz",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["geometric_mm"],
        color=COLORS["geometric"],
        lw=2.0,
        marker="^",
        ms=4.5,
        mfc="white",
        mec=COLORS["geometric"],
        mew=1.2,
        label="Geometric dz",
        zorder=4,
    )


def plot_era5(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.fill_between(
        df["time_utc"],
        df["current_min_mm"],
        df["current_max_mm"],
        color=COLORS["current"],
        alpha=0.12,
        lw=0,
        label="Current dz grid range",
        zorder=2,
    )
    ax.fill_between(
        df["time_utc"],
        df["geometric_min_mm"],
        df["geometric_max_mm"],
        color=COLORS["geometric"],
        alpha=0.12,
        lw=0,
        label="Geometric dz grid range",
        zorder=2,
    )
    ax.plot(
        df["time_utc"],
        df["saas_mean_mm"],
        color=COLORS["saas"],
        lw=1.9,
        label="Saastamoinen grid mean",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["current_mean_mm"],
        color=COLORS["current"],
        lw=1.9,
        label="Current dz grid mean",
        zorder=4,
    )
    ax.plot(
        df["time_utc"],
        df["geometric_mean_mm"],
        color=COLORS["geometric"],
        lw=1.9,
        label="Geometric dz grid mean",
        zorder=4,
    )


def save_event_plot(
    event_id: str,
    event: dict[str, Any],
    sounding: pd.DataFrame,
    era5: pd.DataFrame,
) -> Path | None:
    panels = []
    if not sounding.empty:
        panels.append(("sounding experiment", plot_sounding, sounding))
    if not era5.empty:
        panels.append(("era5 experiment", plot_era5, era5))
    if not panels:
        return None

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13.5, max(3.2, 3.1 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]

    for index, (panel_name, plot_func, panel_df) in enumerate(panels):
        add_stage_background(axes[index], event, show_labels=index == 0)
        plot_func(axes[index], panel_df)
        axes[index].set_title(
            f"{chr(97 + index)}  {panel_name}",
            loc="left",
            fontweight="bold",
        )

    for ax in axes:
        polish_axis(ax, event)
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
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    display_name = event.get("display_name", event_id.title())
    region_name = event.get("region_name", "")
    if region_name:
        title = f"ZHD dz Experiment Time Series During Typhoon {display_name} Around {region_name}"
    else:
        title = f"ZHD dz Experiment Time Series During Typhoon {display_name}"
    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.12)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOT_DIR / f"{event_id}_zhd_dz_experiment_timeseries_rs_era5.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    events = load_events()
    era5_profiles = read_profiles(ERA5_PROFILES_CSV)
    sounding_profiles = read_profiles(SOUNDING_PROFILES_CSV)

    event_ids = sorted(
        set(events)
        & (
            set(era5_profiles["event_id"].unique())
            | set(sounding_profiles["event_id"].unique())
        )
    )
    if not event_ids:
        raise RuntimeError("No matching experiment profiles were found.")

    written: list[Path] = []
    for event_id in event_ids:
        event = events[event_id]
        sounding = prepare_sounding(sounding_profiles, event_id, event)
        era5 = prepare_era5(era5_profiles, event_id, event)
        out_png = save_event_plot(event_id, event, sounding, era5)
        if out_png is not None:
            written.append(out_png)
            print(out_png)

    print(f"\nExperiment time-series figures written: {len(written)}")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
