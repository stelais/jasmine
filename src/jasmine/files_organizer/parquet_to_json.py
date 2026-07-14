"""Convert RMDC event parquet files into the event JSON format.

The output matches the structure used by event JSON files:
  "id": 306455,
  "objname": "RMDC26_306455",
  "ra": 268.185535,
  "dec": -30.246211,
  "photometric_variability": null,
  "microlensing_event": {...metadata...},
  "light_curve": null,
  "light_curves": {
    "F146": {"time": [...], "flux": [...], "flux_err": [...]},
    "F087": {"time": [...], "flux": [...], "flux_err": [...]},
    "F213": {"time": [...], "flux": [...], "flux_err": [...]}
}
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any
import pandas as pd



OBS_FILE_RE = re.compile(r"RMDC26_ML_Data_obs_event_id_(\d+)\.parquet$")


def convert_directory(
    input_dir: Path,
    output_dir: Path | None,
    epoch_path: Path | None,
    indent: int | None,
) -> None:
    if output_dir is None:
        output_dir = input_dir / "json_events"
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_paths = sorted(input_dir.glob("RMDC26_ML_Data_obs_event_id_*.parquet"))
    if not obs_paths:
        raise SystemExit(f"No observation parquet files found in {input_dir}")

    for obs_path in obs_paths:
        event_id = event_id_from_path(obs_path)
        meta_path = input_dir / f"RMDC26_ML_Data_meta_event_id_{event_id}.parquet"
        if not meta_path.exists():
            print(f"Skipping {obs_path.name}: missing {meta_path.name}")
            continue
        output_path = output_dir / f"RMDC26_{event_id}.json"
        convert_event(obs_path, meta_path, output_path, epoch_path, indent)


def convert_event(
    obs_path: Path,
    meta_path: Path,
    output_path: Path,
    epoch_path: Path | None,
    indent: int | None = None,
) -> dict[str, Any]:
    obs = read_parquet(obs_path)
    meta = read_parquet(meta_path)

    if meta.empty:
        raise ValueError(f"Metadata file has no rows: {meta_path}")
    if len(meta) > 1:
        raise ValueError(f"Metadata file should contain one event row: {meta_path}")

    if epoch_path is not None:
        obs = add_bjd_time(obs, epoch_path)
        time_column = "bjd"
    else:
        time_column = "epoch_id"

    metadata = clean_mapping(meta.iloc[0].to_dict())
    event_id = metadata.get("event_id")
    objname = metadata.get("name") or f"event_{event_id}"

    event = {
        "id": event_id,
        "objname": objname,
        "ra": metadata.get("ra_deg"),
        "dec": metadata.get("dec_deg"),
        "photometric_variability": None,
        "metadata": metadata,
        "microlensing_event": None,
        "light_curve": None,
        "light_curves": make_light_curves(obs, time_column),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(event, indent=indent, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return event


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "Missing parquet engine. Install pyarrow with:\n"
            "  python -m pip install pyarrow"
        ) from exc


def add_bjd_time(obs: pd.DataFrame, epoch_path: Path) -> pd.DataFrame:
    epoch = read_parquet(epoch_path)
    required = {"epoch_id", "bjd"}
    missing = sorted(required - set(epoch.columns))
    if missing:
        raise ValueError(f"Epoch parquet is missing columns: {missing}")

    merged = obs.merge(epoch[["epoch_id", "bjd"]], on="epoch_id", how="left")
    if merged["bjd"].isna().any():
        missing_count = int(merged["bjd"].isna().sum())
        raise ValueError(
            f"{missing_count} observation rows did not match a bjd in {epoch_path}"
        )
    return merged


def make_light_curves(obs: pd.DataFrame, time_column: str) -> dict[str, Any]:
    required = {"filt", "flux_uJy", "flux_err_uJy", time_column}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"Observation parquet is missing columns: {missing}")

    light_curves: dict[str, Any] = {}
    for filt, group in obs.sort_values([time_column, "epoch_id"]).groupby(
        "filt", sort=True
    ):
        light_curves[str(filt)] = {
            "time": clean_list(group[time_column].tolist()),
            "flux": clean_list(group["flux_uJy"].tolist()),
            "flux_err": clean_list(group["flux_err_uJy"].tolist()),
        }

    return light_curves


def clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): clean_value(value) for key, value in values.items()}


def clean_list(values: list[Any]) -> list[Any]:
    return [clean_value(value) for value in values]


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def event_id_from_path(path: Path) -> str:
    match = OBS_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not find event id in filename: {path.name}")
    return match.group(1)


def infer_meta_path(obs_path: Path) -> Path:
    event_id = event_id_from_path(obs_path)
    return obs_path.with_name(f"RMDC26_ML_Data_meta_event_id_{event_id}.parquet")


def resolve_epoch_path(
    input_path: Path, explicit_epoch: Path | None, use_epoch_id_as_time: bool
) -> Path | None:
    if use_epoch_id_as_time:
        return None

    if explicit_epoch is not None:
        if not explicit_epoch.exists():
            raise SystemExit(f"epoch parquet file not found: {explicit_epoch}")
        return explicit_epoch

    inferred_epoch = infer_epoch_path(input_path)
    if inferred_epoch is None:
        raise SystemExit(
            "Could not find RMDC26_ML_Data_epoch.parquet. Pass --epoch or, "
            "for debugging only, pass --use-epoch-id-as-time."
        )
    return inferred_epoch


def infer_epoch_path(input_path: Path) -> Path | None:
    start = input_path if input_path.is_dir() else input_path.parent
    for directory in (start, *start.parents):
        candidate = directory / "RMDC26_ML_Data_epoch.parquet"
        if candidate.exists():
            return candidate
    return None

def output_filename_for_obs(obs_path: Path) -> str:
    event_id = event_id_from_path(obs_path)
    return f"RMDC26_{event_id}.json"

def one_parquet_to_json(
    obs_file: str | Path,
    metafile: str | Path | None = None,
    epoch_file: str | Path | None = None,
    *,
    output_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    indent: int | None = None,
) -> Path:
    """Convert one obs/meta parquet pair to one event JSON file.

    Args:
        obs_file: Event observation parquet file.
        metafile: Event metadata parquet file. If omitted, it is inferred from
            the obs filename.
        epoch_file: Shared epoch parquet file. Its ``bjd`` values become JSON
            light-curve ``time`` values. If omitted, it is inferred by walking
            upward from ``obs_file``.
        output_file: Exact JSON file to write.
        output_dir: Folder for the output JSON. Ignored when ``output_file`` is
            provided. Defaults to ``json_events`` beside the obs file.
        indent: Optional JSON indentation.

    Returns:
        Path to the written JSON file.
    """

    obs_path = Path(obs_file)
    meta_path = Path(metafile) if metafile is not None else infer_meta_path(obs_path)
    epoch_path = resolve_epoch_path(
        obs_path,
        Path(epoch_file) if epoch_file is not None else None,
        use_epoch_id_as_time=False,
    )

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata parquet file not found: {meta_path}")

    if output_file is not None:
        output_path = Path(output_file)
    else:
        if output_dir is None:
            output_dir = obs_path.parent / "json_events"
        output_path = Path(output_dir) / output_filename_for_obs(obs_path)

    convert_event(obs_path, meta_path, output_path, epoch_path, indent)
    return output_path


def convert_all_parquet_to_json(
    input_dir: str | Path,
    epoch_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    indent: int | None = None,
) -> list[Path]:
    """Convert every RMDC obs/meta parquet pair in a folder to JSON files."""

    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path / "json_events"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epoch_path = resolve_epoch_path(
        input_path,
        Path(epoch_file) if epoch_file is not None else None,
        use_epoch_id_as_time=False,
    )

    obs_paths = sorted(input_path.glob("RMDC26_ML_Data_obs_event_id_*.parquet"))
    if not obs_paths:
        raise FileNotFoundError(f"No observation parquet files found in {input_path}")

    written_paths: list[Path] = []
    print(f"Found {len(obs_paths)} observation parquet file(s).")
    for obs_path in obs_paths:
        meta_path = infer_meta_path(obs_path)
        if not meta_path.exists():
            print(f"Skipping {obs_path.name}: missing {meta_path.name}")
            continue
        written_paths.append(
            one_parquet_to_json(
                obs_path,
                meta_path,
                epoch_path,
                output_dir=output_path,
                indent=indent,
            )
        )

    return written_paths

def convert_all_categories_to_json(
    per_event_id_dir: str | Path,
    epoch_file: str | Path,
    output_dir: str | Path,
    *,
    indent: int | None = None,
) -> list[Path]:
    """Convert all category folders under per_event_id into matching JSON folders."""

    per_event_id_path = Path(per_event_id_dir)
    output_path = Path(output_dir)

    written_paths: list[Path] = []

    for category_dir in sorted(per_event_id_path.iterdir()):
        if not category_dir.is_dir():
            continue

        category_output_dir = output_path / category_dir.name

        category_outputs = convert_all_parquet_to_json(
            input_dir=category_dir,
            epoch_file=epoch_file,
            output_dir=category_output_dir,
            indent=indent,
        )

        written_paths.extend(category_outputs)

    return written_paths

def plot_test_json(event_path: Path):
    import matplotlib.pyplot as plt
    with open(event_path) as f:
        json_object = json.load(f)
    light_curves = json_object["light_curves"]

    for filter_name in sorted(light_curves):
        light_curve = light_curves[filter_name]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(
            light_curve["time"],
            light_curve["flux"],
            yerr=light_curve.get("flux_err"),
            fmt=".",
            markersize=3,
            elinewidth=0.5,
            capsize=1.5,
            linestyle="none",
            alpha=0.8,
        )

        event_name = json_object.get("objname", json_object["id"])
        ax.set_title(f"{event_name}: {filter_name}")
        ax.set_xlabel("Time (JD)")
        ax.set_ylabel("Flux")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_dir = Path("data/ml_datachallenge/")
    output_dir = data_dir / "json_events"
    epoch_file = data_dir / "RMDC26_ML_Data_epoch.parquet"

    # For one event conversion
    # input_dir = data_dir / "per_event_id/planetary2l1s"
    # obs_file = input_dir / "RMDC26_ML_Data_obs_event_id_306455.parquet"
    # metafile = input_dir / "RMDC26_ML_Data_meta_event_id_306455.parquet"
    # one_parquet_to_json(obs_file, metafile, epoch_file, output_dir=output_dir)

    # For all events conversion
    per_event_id_dir = data_dir / "per_event_id"
    convert_all_categories_to_json(
        per_event_id_dir=per_event_id_dir,
        epoch_file=epoch_file,
        output_dir=output_dir,
    )