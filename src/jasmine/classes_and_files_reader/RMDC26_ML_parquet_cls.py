"""
RMDC26_ML_parquet_cls: classes and functions for reading and splitting
RMDC26 ML data from parquet files.

Memory-efficient version:
- Does NOT read the full observations parquet into pandas.
- Streams metadata in batches.
- Streams observations in batches.
- Writes one parquet file per event category.

TODO:
- ephemeris file
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# 1 Jupiter mass in Solar masses
MJUP_TO_MSUN = 0.000954588
FFP_MASS_LIMIT_MSUN = 13.0 * MJUP_TO_MSUN


# Internal category name -> filename suffix
CATEGORY_SUFFIXES: Dict[str, str] = {
    "ffp": "ffp",
    "simple_1l1s": "simple_1l1s",
    "planetary_2l1s": "planetary2l1s",
    "stellar_bd_2l1s": "stellar_bd_2l1s",
    "planetary_2l2s": "planetary2l2s",
    "stellar_bd_2l2s": "stellar_bd_2l2s",
}

CATEGORIES: Tuple[str, ...] = tuple(CATEGORY_SUFFIXES.keys())

ParquetSource = Union[pd.DataFrame, str, Path]


@dataclass
class RMDC26Event:
    name: str
    event_id: int
    sim_label: str
    ra_deg: float
    dec_deg: float
    galactic_l: float
    galactic_b: float
    meta: pd.Series
    lightcurves: Dict[str, pd.DataFrame]

    @classmethod
    def from_event_id(
        cls,
        event_id: int,
        meta_source: ParquetSource,
        obs_source: ParquetSource,
    ) -> "RMDC26Event":
        """
        Load one event.

        meta_source can be:
        - a pandas DataFrame, or
        - a path to RMDC26_ML_Data_meta.parquet

        obs_source can be:
        - a pandas DataFrame, or
        - a path to RMDC26_ML_Data_obs.parquet

        For large files, pass file paths instead of full DataFrames.
        """
        event_id = int(event_id)

        meta_rows = _read_rows_by_column(
            source=meta_source,
            column="event_id",
            value=event_id,
        )

        if meta_rows.empty:
            raise ValueError(f"Event with event_id={event_id} not found in metadata.")

        meta = meta_rows.iloc[0]

        obs_event = _read_rows_by_column(
            source=obs_source,
            column="event_id",
            value=event_id,
        )

        if obs_event.empty:
            raise ValueError(f"Event with event_id={event_id} not found in observations.")

        lightcurves: Dict[str, pd.DataFrame] = {}

        for filt, filt_df in obs_event.groupby("filt"):
            filt_df = filt_df.sort_values("epoch_id")

            lightcurves[str(filt)] = filt_df[
                [
                    "epoch_id",
                    "flux_uJy",
                    "flux_err_uJy",
                    "true_flux_uJy",
                    "saturation_flag",
                ]
            ].copy()

        return cls(
            name=str(meta["name"]),
            event_id=int(meta["event_id"]),
            sim_label=str(meta["sim_label"]),
            ra_deg=float(meta["ra_deg"]),
            dec_deg=float(meta["dec_deg"]),
            galactic_l=float(meta["galactic_l"]),
            galactic_b=float(meta["galactic_b"]),
            meta=meta,
            lightcurves=lightcurves,
        )

    @classmethod
    def from_name(
        cls,
        name: str,
        meta_source: ParquetSource,
        obs_source: ParquetSource,
    ) -> "RMDC26Event":
        """
        Load one event by event name.
        """
        meta_rows = _read_rows_by_column(
            source=meta_source,
            column="name",
            value=name,
        )

        if meta_rows.empty:
            raise ValueError(f"Event with name={name} not found in metadata.")

        event_id = int(meta_rows.iloc[0]["event_id"])

        return cls.from_event_id(
            event_id=event_id,
            meta_source=meta_rows,
            obs_source=obs_source,
        )

    @property
    def simple_lightcurve(self) -> Dict[str, pd.DataFrame]:
        return {
            filt: df[["epoch_id", "flux_uJy", "flux_err_uJy"]].copy()
            for filt, df in self.lightcurves.items()
        }


def _read_rows_by_column(
    source: ParquetSource,
    column: str,
    value,
) -> pd.DataFrame:
    """
    Read rows matching one column value from either a DataFrame or parquet file.

    This is useful for loading a single event without loading the full obs table.
    """
    if isinstance(source, pd.DataFrame):
        return source[source[column] == value].copy()

    source_path = Path(source)

    return pd.read_parquet(
        source_path,
        filters=[(column, "==", value)],
    )


def _category_file(
    output_dir: Path,
    table_kind: str,
    category: str,
) -> Path:
    """
    Build output filename.

    Example:
    RMDC26_ML_Data_meta_ffp.parquet
    RMDC26_ML_Data_obs_planetary2l1s.parquet
    """
    suffix = CATEGORY_SUFFIXES[category]
    return output_dir / f"RMDC26_ML_Data_{table_kind}_{suffix}.parquet"


def _as_bool_binary_source(meta_df: pd.DataFrame) -> pd.Series:
    """
    Returns True for binary-source events.

    Uses Source_Is_Binary when available, and also checks sim_label for 2S.
    """
    binary_source_from_column = (
        pd.to_numeric(meta_df["Source_Is_Binary"], errors="coerce")
        .fillna(0)
        > 0
    )

    binary_source_from_label = meta_df["sim_label"].astype(str).str.contains(
        "2S",
        case=False,
        regex=False,
    )

    return binary_source_from_column | binary_source_from_label


def _as_bool_binary_lens(meta_df: pd.DataFrame) -> pd.Series:
    """
    Returns True for binary-lens events.

    Uses Planet_q when available, and also checks sim_label for 2L.
    """
    planet_q = pd.to_numeric(meta_df["Planet_q"], errors="coerce")

    binary_lens_from_planet_q = planet_q.notna()

    binary_lens_from_label = meta_df["sim_label"].astype(str).str.contains(
        "2L",
        case=False,
        regex=False,
    )

    return binary_lens_from_planet_q | binary_lens_from_label


def build_category_masks(meta_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build boolean masks for each requested event category.

    Categories:
    1. ffp:
       single lens, single source, Lens_Mass < 13 Jupiter masses

    2. simple_1l1s:
       single lens, single source, Lens_Mass >= 13 Jupiter masses

    3. planetary_2l1s:
       binary lens, single source, Planet_q < 0.04

    4. stellar_bd_2l1s:
       binary lens, single source, Planet_q >= 0.04

    5. planetary_2l2s:
       binary lens, binary source, Planet_q < 0.04

    6. stellar_bd_2l2s:
       binary lens, binary source, Planet_q >= 0.04
    """
    required_columns = [
        "sim_label",
        "Source_Is_Binary",
        "Planet_q",
        "Lens_Mass",
    ]

    missing_columns = [col for col in required_columns if col not in meta_df.columns]

    if missing_columns:
        raise KeyError(f"Missing required metadata columns: {missing_columns}")

    binary_source = _as_bool_binary_source(meta_df)
    single_source = ~binary_source

    binary_lens = _as_bool_binary_lens(meta_df)
    single_lens = ~binary_lens

    lens_mass = pd.to_numeric(meta_df["Lens_Mass"], errors="coerce")
    q = pd.to_numeric(meta_df["Planet_q"], errors="coerce")

    is_1l1s = single_lens & single_source
    is_2l1s = binary_lens & single_source
    is_2l2s = binary_lens & binary_source

    is_ffp = is_1l1s & lens_mass.notna() & (lens_mass < FFP_MASS_LIMIT_MSUN)
    is_simple_1l1s = is_1l1s & ~is_ffp

    is_planetary = q.notna() & (q < 0.04)
    is_stellar_bd = q.notna() & (q >= 0.04)

    masks = {
        "ffp": is_ffp,
        "simple_1l1s": is_simple_1l1s,
        "planetary_2l1s": is_2l1s & is_planetary,
        "stellar_bd_2l1s": is_2l1s & is_stellar_bd,
        "planetary_2l2s": is_2l2s & is_planetary,
        "stellar_bd_2l2s": is_2l2s & is_stellar_bd,
    }

    return masks


def split_meta_streaming(
    meta_file: Path,
    output_dir: Path,
    batch_size: int = 100_000,
    compression: str = "snappy",
    progress_every_batches: int = 25,
) -> Dict[str, List[int]]:
    """
    Stream the metadata parquet and write one metadata parquet per category.

    Returns
    -------
    category_event_ids:
        Dictionary mapping category name to event_id list.
        This is then used to split the large obs table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming metadata from: {meta_file}")

    meta_dataset = ds.dataset(str(meta_file), format="parquet")
    meta_schema = meta_dataset.schema

    meta_writers: Dict[str, pq.ParquetWriter] = {}
    category_event_ids: Dict[str, List[int]] = {category: [] for category in CATEGORIES}
    meta_counts: Dict[str, int] = {category: 0 for category in CATEGORIES}

    for category in CATEGORIES:
        output_file = _category_file(output_dir, "meta", category)
        meta_writers[category] = pq.ParquetWriter(
            str(output_file),
            schema=meta_schema,
            compression=compression,
        )

    unassigned_output = output_dir / "RMDC26_ML_Data_meta_unassigned.parquet"
    unassigned_writer = pq.ParquetWriter(
        str(unassigned_output),
        schema=meta_schema,
        compression=compression,
    )

    unassigned_count = 0
    rows_seen = 0

    try:
        scanner = meta_dataset.scanner(
            batch_size=batch_size,
            use_threads=True,
        )

        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            if batch.num_rows == 0:
                continue

            rows_seen += batch.num_rows

            # Convert only this metadata batch to pandas.
            # This is okay because the metadata batch is small.
            meta_batch_df = batch.to_pandas()

            masks = build_category_masks(meta_batch_df)

            assigned_mask = pd.Series(False, index=meta_batch_df.index)

            for category in CATEGORIES:
                mask = masks[category].fillna(False)
                assigned_mask |= mask

                meta_subset = meta_batch_df.loc[mask].copy()

                if meta_subset.empty:
                    continue

                event_ids = (
                    pd.to_numeric(meta_subset["event_id"], errors="raise")
                    .astype("int64")
                    .tolist()
                )

                category_event_ids[category].extend(event_ids)

                subset_table = pa.Table.from_pandas(
                    meta_subset,
                    schema=meta_schema,
                    preserve_index=False,
                )

                meta_writers[category].write_table(subset_table)
                meta_counts[category] += len(meta_subset)

            unassigned_subset = meta_batch_df.loc[~assigned_mask].copy()

            if not unassigned_subset.empty:
                unassigned_table = pa.Table.from_pandas(
                    unassigned_subset,
                    schema=meta_schema,
                    preserve_index=False,
                )

                unassigned_writer.write_table(unassigned_table)
                unassigned_count += len(unassigned_subset)

            if progress_every_batches and batch_number % progress_every_batches == 0:
                print(f"  metadata rows streamed: {rows_seen:,}")

    finally:
        for writer in meta_writers.values():
            writer.close()

        unassigned_writer.close()

    print()
    print("Metadata category summary")
    print("-------------------------")

    assigned_total = 0

    for category in CATEGORIES:
        count = meta_counts[category]
        assigned_total += count

        output_file = _category_file(output_dir, "meta", category)

        print(f"{category:20s}: {count:,}")
        print(f"  saved: {output_file}")

    print("-------------------------")
    print(f"assigned total       : {assigned_total:,}")
    print(f"metadata rows seen   : {rows_seen:,}")
    print(f"unassigned           : {unassigned_count:,}")
    print(f"unassigned saved     : {unassigned_output}")
    print()

    return category_event_ids


def split_obs_streaming(
    obs_file: Path,
    output_dir: Path,
    category_event_ids: Dict[str, List[int]],
    batch_size: int = 500_000,
    compression: str = "snappy",
    progress_every_batches: int = 25,
) -> Dict[str, int]:
    """
    Stream the large observations parquet once and write one obs parquet per category.

    This function does NOT convert the full obs table to pandas.
    It uses Arrow filtering batch-by-batch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming observations from: {obs_file}")

    obs_dataset = ds.dataset(str(obs_file), format="parquet")
    obs_schema = obs_dataset.schema

    if "event_id" not in obs_schema.names:
        raise KeyError("The observations parquet must contain an 'event_id' column.")

    event_id_type = obs_schema.field("event_id").type

    obs_writers: Dict[str, pq.ParquetWriter] = {}
    obs_counts: Dict[str, int] = {category: 0 for category in CATEGORIES}

    for category in CATEGORIES:
        output_file = _category_file(output_dir, "obs", category)
        obs_writers[category] = pq.ParquetWriter(
            str(output_file),
            schema=obs_schema,
            compression=compression,
        )

    # Convert event_id lists to Arrow arrays once.
    # These are used repeatedly while streaming the obs table.
    event_id_value_sets: Dict[str, pa.Array] = {}

    for category in CATEGORIES:
        event_id_value_sets[category] = pa.array(
            category_event_ids[category],
            type=event_id_type,
        )

    rows_seen = 0

    try:
        scanner = obs_dataset.scanner(
            batch_size=batch_size,
            use_threads=True,
        )

        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            if batch.num_rows == 0:
                continue

            rows_seen += batch.num_rows

            # Keep this as Arrow, not pandas.
            table = pa.Table.from_batches([batch])
            event_id_column = table["event_id"]

            for category in CATEGORIES:
                value_set = event_id_value_sets[category]

                if len(value_set) == 0:
                    continue

                mask = pc.is_in(
                    event_id_column,
                    value_set=value_set,
                )

                filtered_table = table.filter(mask)

                if filtered_table.num_rows == 0:
                    continue

                obs_writers[category].write_table(filtered_table)
                obs_counts[category] += filtered_table.num_rows

            if progress_every_batches and batch_number % progress_every_batches == 0:
                print(f"  observation rows streamed: {rows_seen:,}")

    finally:
        for writer in obs_writers.values():
            writer.close()

    print()
    print("Observation category summary")
    print("----------------------------")

    written_total = 0

    for category in CATEGORIES:
        count = obs_counts[category]
        written_total += count

        output_file = _category_file(output_dir, "obs", category)

        print(f"{category:20s}: {count:,}")
        print(f"  saved: {output_file}")

    print("----------------------------")
    print(f"observation rows seen    : {rows_seen:,}")
    print(f"observation rows written : {written_total:,}")
    print()

    return obs_counts


def main(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    meta_batch_size: int = 100_000,
    obs_batch_size: int = 500_000,
) -> None:
    """
    Split RMDC26 ML metadata and observations into event categories.

    Parameters
    ----------
    data_dir:
        Directory containing:

        RMDC26_ML_Data_meta.parquet
        RMDC26_ML_Data_obs.parquet

    output_dir:
        Directory where category files will be written.
        If None, files are written to:

        data_dir / "per_category"

    meta_batch_size:
        Number of metadata rows per streaming batch.

    obs_batch_size:
        Number of observation rows per streaming batch.
        Reduce this if the EC2 instance still runs out of memory.
    """
    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = data_dir / "per_category"
    else:
        output_dir = Path(output_dir)

    meta_file = data_dir / "RMDC26_ML_Data_meta.parquet"
    obs_file = data_dir / "RMDC26_ML_Data_obs.parquet"

    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file}")

    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    category_event_ids = split_meta_streaming(
        meta_file=meta_file,
        output_dir=output_dir,
        batch_size=meta_batch_size,
    )

    split_obs_streaming(
        obs_file=obs_file,
        output_dir=output_dir,
        category_event_ids=category_event_ids,
        batch_size=obs_batch_size,
    )

    print("Done.")


if __name__ == "__main__":
    DATA_DIR = Path("/home/ec2-user/msos_events_project/data/ml_datachallenge")

    main(
        data_dir=DATA_DIR,
        output_dir=DATA_DIR / "per_category",
        meta_batch_size=100_000,
        obs_batch_size=500_000,
    )