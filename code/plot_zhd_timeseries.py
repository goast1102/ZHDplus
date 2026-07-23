from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Path settings
# ============================================================

OUT_DIR = Path(r"E:\ZHDplus\test\out\mawar")

RS_SAAS_CSV = OUT_DIR / "mawar_sounding_zhd_saas.csv"
RS_PROFILE_CSV = OUT_DIR / "mawar_sounding_zhd_profile.csv"
ERA5_SAAS_GRID4_CSV = OUT_DIR / "mawar_era5_zhd_saas_grid4.csv"
ERA5_PROFILE_GRID4_CSV = OUT_DIR / "mawar_era5_zhd_profile_grid4.csv"
ERA5_SAAS_STATION_CSV = OUT_DIR / "mawar_era5_zhd_saas_station_interp.csv"
ERA5_PROFILE_STATION_CSV = OUT_DIR / "mawar_era5_zhd_profile_station_interp.csv"

OUT_PNG = OUT_DIR / "zhd_timeseries_rs_era5_era5inter.png"
PLOT_TITLE = "ZHD Time Series During Typhoon Mawar Around Guam"


# ============================================================
# Time settings
# ============================================================

TIME_START = pd.Timestamp("2023-05-21 00:00:00")
PRE_END = pd.Timestamp("2023-05-24 00:00:00")
DURING_END = pd.Timestamp("2023-05-25 00:00:00")
TIME_END = pd.Timestamp("2023-05-28 23:00:00")

STAGES = [
    (TIME_START, PRE_END, "Pre-Mawar", "#dbeafe"),
    (PRE_END, DURING_END, "Mawar impact", "#fee2e2"),
    (DURING_END, TIME_END, "Post-Mawar", "#dcfce7"),
]


# ============================================================
# Plot style
# ============================================================

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
    "profile": "#dc2626",
    "grid_band": "#64748b",
}


def read_zhd_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find input file: {path}")

    df = pd.read_csv(path)
    required = {"longitude", "latitude", "time_utc", "ZHD_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df["ZHD_m"] = pd.to_numeric(df["ZHD_m"], errors="coerce")
    df = df.dropna(subset=["time_utc", "ZHD_m"]).copy()
    df = df[(df["time_utc"] >= TIME_START) & (df["time_utc"] <= TIME_END)].copy()
    df["ZHD_mm"] = df["ZHD_m"] * 1000.0
    return df.sort_values("time_utc").reset_index(drop=True)


def read_optional_zhd_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skip missing plot input: {path}")
        return None

    df = read_zhd_csv(path)
    if df.empty:
        print(f"Skip empty plot input after time filter: {path}")
        return None

    return df


def summarize_grid4(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("time_utc", as_index=False)
        .agg(
            zhd_mean=("ZHD_mm", "mean"),
            zhd_min=("ZHD_mm", "min"),
            zhd_max=("ZHD_mm", "max"),
            n_grid=("ZHD_mm", "size"),
        )
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    expected_count = int(summary["n_grid"].mode().iloc[0])
    incomplete = summary[summary["n_grid"] != expected_count]
    if not incomplete.empty:
        times = ", ".join(incomplete["time_utc"].dt.strftime("%Y-%m-%d %H:%M").head(8))
        print(
            "Warning: inconsistent ERA5 grid point counts found: "
            f"expected {expected_count}; first times: {times}"
        )

    return summary


def add_stage_background(ax: plt.Axes, show_labels: bool = False) -> None:
    for start, end, label, color in STAGES:
        ax.axvspan(start, end, color=color, alpha=0.55, lw=0, zorder=0)
        if show_labels:
            center = start + (end - start) / 2
            ax.text(
                center,
                1.02,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                color="#334155",
            )

    for boundary in [PRE_END, DURING_END]:
        ax.axvline(boundary, color="#475569", lw=1.0, ls="--", alpha=0.65, zorder=1)


def polish_axis(ax: plt.Axes) -> None:
    ax.set_xlim(TIME_START, TIME_END)
    ax.grid(True, which="major", axis="y", color="#cbd5e1", lw=0.8, alpha=0.65)
    ax.grid(True, which="major", axis="x", color="#e2e8f0", lw=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=1.0, colors="#334155")
    ax.set_ylabel("ZHD (mm)")


def plot_rs(
    ax: plt.Axes,
    rs_saas: pd.DataFrame | None,
    rs_profile: pd.DataFrame | None,
) -> None:
    if rs_saas is not None:
        ax.plot(
            rs_saas["time_utc"],
            rs_saas["ZHD_mm"],
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
    if rs_profile is not None:
        ax.plot(
            rs_profile["time_utc"],
            rs_profile["ZHD_mm"],
            color=COLORS["profile"],
            lw=2.0,
            marker="s",
            ms=4.5,
            mfc="white",
            mec=COLORS["profile"],
            mew=1.2,
            label="Profile integral",
            zorder=4,
        )
    ax.set_title("a  rs", loc="left", fontweight="bold")


def plot_era5_grid(
    ax: plt.Axes,
    era5_saas_grid: pd.DataFrame | None,
    era5_profile_grid: pd.DataFrame | None,
) -> None:
    if era5_saas_grid is not None:
        saas = summarize_grid4(era5_saas_grid)
        ax.fill_between(
            saas["time_utc"],
            saas["zhd_min"],
            saas["zhd_max"],
            color=COLORS["saas"],
            alpha=0.12,
            lw=0,
            label="Saastamoinen grid range",
            zorder=2,
        )
        ax.plot(
            saas["time_utc"],
            saas["zhd_mean"],
            color=COLORS["saas"],
            lw=1.9,
            label="Saastamoinen grid mean",
            zorder=4,
        )

    if era5_profile_grid is not None:
        profile = summarize_grid4(era5_profile_grid)
        ax.fill_between(
            profile["time_utc"],
            profile["zhd_min"],
            profile["zhd_max"],
            color=COLORS["profile"],
            alpha=0.12,
            lw=0,
            label="Profile grid range",
            zorder=2,
        )
        ax.plot(
            profile["time_utc"],
            profile["zhd_mean"],
            color=COLORS["profile"],
            lw=1.9,
            label="Profile grid mean",
            zorder=4,
        )
    ax.set_title("b  era5", loc="left", fontweight="bold")


def plot_era5_station(
    ax: plt.Axes,
    era5_saas_station: pd.DataFrame | None,
    era5_profile_station: pd.DataFrame | None,
) -> None:
    if era5_saas_station is not None:
        ax.plot(
            era5_saas_station["time_utc"],
            era5_saas_station["ZHD_mm"],
            color=COLORS["saas"],
            lw=2.0,
            label="Saastamoinen interp",
            zorder=4,
        )
    if era5_profile_station is not None:
        ax.plot(
            era5_profile_station["time_utc"],
            era5_profile_station["ZHD_mm"],
            color=COLORS["profile"],
            lw=2.0,
            label="Profile interp",
            zorder=4,
        )
    ax.set_title("c  era5_inter", loc="left", fontweight="bold")


def main() -> None:
    rs_saas = read_optional_zhd_csv(RS_SAAS_CSV)
    rs_profile = read_optional_zhd_csv(RS_PROFILE_CSV)
    era5_saas_grid = read_optional_zhd_csv(ERA5_SAAS_GRID4_CSV)
    era5_profile_grid = read_optional_zhd_csv(ERA5_PROFILE_GRID4_CSV)
    era5_saas_station = read_optional_zhd_csv(ERA5_SAAS_STATION_CSV)
    era5_profile_station = read_optional_zhd_csv(ERA5_PROFILE_STATION_CSV)

    panels = []
    if rs_saas is not None or rs_profile is not None:
        panels.append(("rs", plot_rs, rs_saas, rs_profile))
    if era5_saas_grid is not None or era5_profile_grid is not None:
        panels.append(("era5", plot_era5_grid, era5_saas_grid, era5_profile_grid))
    if era5_saas_station is not None or era5_profile_station is not None:
        panels.append(
            ("era5_inter", plot_era5_station, era5_saas_station, era5_profile_station)
        )

    if not panels:
        raise FileNotFoundError("No available ZHD CSV files for plotting.")

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13.5, max(3.2, 3.1 * len(panels))),
        sharex=True,
        constrained_layout=True,
    )

    if len(panels) == 1:
        axes = [axes]

    for index, (panel_name, plot_func, first_df, second_df) in enumerate(panels):
        add_stage_background(axes[index], show_labels=index == 0)
        plot_func(axes[index], first_df, second_df)
        axes[index].set_title(
            f"{chr(97 + index)}  {panel_name}",
            loc="left",
            fontweight="bold",
        )

    for ax in axes:
        polish_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
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

    fig.suptitle(
        PLOT_TITLE,
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Figure file:")
    print(OUT_PNG)


if __name__ == "__main__":
    main()
