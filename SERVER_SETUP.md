# Server Setup

This project can run on Linux with the same event config in `code/typhoon_events.json`.
Use `code/zhd_workflow.py` as the server entry point instead of running the single-purpose scripts directly.

## 1. Copy the project

```bash
scp -r ZHDplus user@server:/path/to/ZHDplus
cd /path/to/ZHDplus
```

Large data can live outside the git/project directory. The workflow supports:

```bash
export ZHDPLUS_DATA_ROOT=/data/zhdplus/data
export ZHDPLUS_OUT_ROOT=/data/zhdplus/out
mkdir -p "$ZHDPLUS_DATA_ROOT" "$ZHDPLUS_OUT_ROOT"
```

Existing relative config paths such as `data/era5_milton` will be written under
`$ZHDPLUS_DATA_ROOT/era5_milton`. Results will be written under
`$ZHDPLUS_OUT_ROOT/<event_id>`.

Absolute paths in `typhoon_events.json` are also allowed if one event needs a special location.

## 2. Create the Python environment

Prefer conda or mamba on the server because ERA5 GRIB extraction needs ecCodes.

```bash
conda env create -f environment.yml
conda activate zhdplus
```

Quick import test:

```bash
python -c "import cdsapi, xarray, cfgrib, eccodes, pandas, matplotlib, scipy, requests; print('ok')"
python -m cfgrib selfcheck
```

## 3. Configure CDS API

Create `$HOME/.cdsapirc` on the server:

```text
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

Before the first ERA5 download, log in to the CDS website and accept the licence/terms for
`reanalysis-era5-pressure-levels`.

## 4. Run the workflow

Run one event end to end:

```bash
python code/zhd_workflow.py --event mawar --steps all
```

Run only ERA5 download and extraction:

```bash
python code/zhd_workflow.py --event mawar --steps era5_download era5_extract
```

Run calculations and plots after data already exists:

```bash
python code/zhd_workflow.py --event mawar --steps results
```

Run all enabled events:

```bash
python code/zhd_workflow.py --steps all
```

For long server jobs, use `tmux`, `screen`, or your scheduler:

```bash
tmux new -s zhdplus
python code/zhd_workflow.py --event mawar --steps all
```

## 5. ERA5 notes

The workflow writes ERA5 files as:

```text
$ZHDPLUS_DATA_ROOT/era5_<event>/grib/*.grib
$ZHDPLUS_DATA_ROOT/era5_<event>/era5_pressure_raw_extract.csv
```

If CDS rejects a large request, set the event's ERA5 config to split more aggressively:

```json
"download_strategy": "monthly_variable",
"max_parallel_downloads": 1
```

Keep `max_parallel_downloads` low on shared servers. `1` is slow but friendlier to CDS queues and easier to debug.

## 6. Data transfer option

If ERA5 files already exist locally, you can copy only `data/era5_*`, `data/radiosonde`, and `out`
to the server data roots. The workflow skips non-empty existing GRIB files unless `OVERWRITE` is changed in code.
