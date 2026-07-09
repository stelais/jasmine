"""
RMDC26_ML_parquet_cls: classes and functions for reading RMDC26ML data from parquet files.
TODO: ephemeris file
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# 1 Jupiter mass in Solar masses
MJUP_TO_MSUN = 0.000954588
FFP_MASS_LIMIT_MSUN = 13.0 * MJUP_TO_MSUN


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
            meta_df: pd.DataFrame,
            obs_df: pd.DataFrame,
    ) -> "RMDC26Event":
        meta_rows = meta_df[meta_df["event_id"] == event_id]

        if meta_rows.empty:
            raise ValueError(f"Event with event_id={event_id} not found in metadata.")

        meta = meta_rows.iloc[0]

        obs_event = obs_df[obs_df["event_id"] == event_id]

        if obs_event.empty:
            raise ValueError(f"Event with event_id={event_id} not found in observations.")

        lightcurves = {}
        for filt, filt_df in obs_event.groupby("filt"):
            lightcurves[filt] = filt_df[
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
            meta_df: pd.DataFrame,
            obs_df: pd.DataFrame,
    ) -> "RMDC26Event":
        meta_rows = meta_df[meta_df["name"] == name]

        if meta_rows.empty:
            raise ValueError(f"Event with name={name} not found in metadata.")

        event_id = int(meta_rows.iloc[0]["event_id"])
        return cls.from_event_id(event_id, meta_df, obs_df)

    @property
    def simple_lightcurve(self) -> Dict[str, pd.DataFrame]:
        return {
            filt: df[["epoch_id", "flux_uJy", "flux_err_uJy"]].copy()
            for filt, df in self.lightcurves.items()
        }


def _as_bool_binary_source(meta_df: pd.DataFrame) -> pd.Series:
    """
    Returns True for binary-source events.

    Uses Source_Is_Binary when available, and also checks sim_label for 2S.
    """
    binary_source_from_column = (
        meta_df["Source_Is_Binary"]
        .fillna(0)
        .astype(float)
        .astype(int)
        .astype(bool)
    )

    binary_source_from_label = meta_df["sim_label"].astype(str).str.contains(
        "2S", case=False, regex=False
    )

    return binary_source_from_column | binary_source_from_label


def _as_bool_binary_lens(meta_df: pd.DataFrame) -> pd.Series:
    """
    Returns True for binary-lens events.

    Uses Planet_q when available, and also checks sim_label for 2L.
    """
    binary_lens_from_planet_q = meta_df["Planet_q"].notna()

    binary_lens_from_label = meta_df["sim_label"].astype(str).str.contains(
        "2L", case=False, regex=False
    )

    return binary_lens_from_planet_q | binary_lens_from_label


def build_category_masks(meta_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build boolean masks for each requested event category.

    Categories:
    1. FFPs:
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
    print("Building category masks...")
    binary_source = _as_bool_binary_source(meta_df)
    single_source = ~binary_source

    binary_lens = _as_bool_binary_lens(meta_df)
    single_lens = ~binary_lens

    lens_mass = meta_df["Lens_Mass"]
    q = meta_df["Planet_q"]

    is_1l1s = single_lens & single_source
    is_2l1s = binary_lens & single_source
    is_2l2s = binary_lens & binary_source

    is_ffp = is_1l1s & lens_mass.notna() & (lens_mass < FFP_MASS_LIMIT_MSUN)
    is_simple_1l1s = is_1l1s & ~is_ffp

    is_planetary = q.notna() & (q < 0.04)
    is_stellar_bd = q.notna() & (q >= 0.04)

    print("Ready for the masks! ")
    masks = {
        "ffp": is_ffp,
        "simple_1l1s": is_simple_1l1s,
        "planetary_2l1s": is_2l1s & is_planetary,
        "stellar_bd_2l1s": is_2l1s & is_stellar_bd,
        "planetary_2l2s": is_2l2s & is_planetary,
        "stellar_bd_2l2s": is_2l2s & is_stellar_bd,
    }

    return masks


def save_meta_and_obs_subsets(
        meta_df: pd.DataFrame,
        obs_df: pd.DataFrame,
        output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = build_category_masks(meta_df)

    for category, mask in masks.items():
        meta_subset = meta_df.loc[mask].copy()

        event_ids = set(meta_subset["event_id"].astype(int))
        obs_subset = obs_df[obs_df["event_id"].isin(event_ids)].copy()

        meta_output = output_dir / f"RMDC26_ML_Data_meta_{category}.parquet"
        obs_output = output_dir / f"RMDC26_ML_Data_obs_{category}.parquet"

        meta_subset.to_parquet(meta_output, index=False)
        obs_subset.to_parquet(obs_output, index=False)

        print(f"{category}")
        print(f"  meta rows: {len(meta_subset):,}")
        print(f"  obs rows:  {len(obs_subset):,}")
        print(f"  saved: {meta_output}")
        print(f"  saved: {obs_output}")
        print()


def print_category_summary(meta_df: pd.DataFrame, data_dir) -> None:
    masks = build_category_masks(meta_df)

    print("Category summary")
    print("----------------")

    total_assigned = 0

    for category, mask in masks.items():
        n_events = int(mask.sum())
        total_assigned += n_events
        print(f"{category:20s}: {n_events:,}")

    print("----------------")
    print(f"assigned total       : {total_assigned:,}")
    print(f"metadata total       : {len(meta_df):,}")

    assigned_mask = pd.Series(False, index=meta_df.index)
    for mask in masks.values():
        assigned_mask |= mask

    n_unassigned = int((~assigned_mask).sum())
    print(f"unassigned           : {n_unassigned:,}")

    if n_unassigned > 0:
        unassigned = meta_df.loc[
            ~assigned_mask,
            ["name", "event_id", "sim_label", "Source_Is_Binary", "Planet_q", "Lens_Mass"],
        ].copy()

        unassigned_output = data_dir / "RMDC26_ML_Data_meta_unassigned.parquet"
        unassigned.to_parquet(unassigned_output, index=False)

        print()
        print(f"Saved unassigned events to: {unassigned_output}")


def main(data_dir) -> None:
    META_FILE = data_dir / "RMDC26_ML_Data_meta.parquet"
    OBS_FILE = data_dir / "RMDC26_ML_Data_obs.parquet"

    print(f"Reading metadata: {META_FILE}")
    meta_df = pd.read_parquet(META_FILE)

    print(f"Reading observations: {OBS_FILE}")
    obs_df = pd.read_parquet(OBS_FILE)

    print()
    print_category_summary(meta_df, data_dir)

    print()
    save_meta_and_obs_subsets(
        meta_df=meta_df,
        obs_df=obs_df,
        output_dir=data_dir / "per_category",
    )

    print("Example loading one event")
    example_event_id = int(meta_df.iloc[0]["event_id"])
    event = RMDC26Event.from_event_id(example_event_id, meta_df, obs_df)

    print("Event name: ", event.name)
    print("Event ID:   ", event.event_id)
    print("sim_label:  ", event.sim_label)
    print("RA, Dec:    ", event.ra_deg, event.dec_deg)
    print("l, b:       ", event.galactic_l, event.galactic_b)

    if "F146" in event.lightcurves:
        print()
        print("F146 light curve:")
        print(event.lightcurves["F146"].head())


if __name__ == "__main__":
    DATA_DIR = Path("/home/ec2-user/msos_events_project/data/ml_datachallenge")
    main(data_dir=DATA_DIR)
