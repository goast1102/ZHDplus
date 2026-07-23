from __future__ import annotations

from verification_workflow_common import run_workflow


WORKFLOW_ID = "twenty_stations_january_and_august_2025"


def main() -> None:
    run_workflow(
        WORKFLOW_ID,
        "Run the 20-station January/August 2025 radiosonde and ERA5 ZHD verification workflow.",
    )


if __name__ == "__main__":
    main()
