from __future__ import annotations

from verification_workflow_common import run_workflow


WORKFLOW_ID = "four_global_stations_full_year_2025"


def main() -> None:
    run_workflow(
        WORKFLOW_ID,
        "Run the 4-station full-year 2025 radiosonde and ERA5 ZHD verification workflow.",
    )


if __name__ == "__main__":
    main()
