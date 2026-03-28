from typing import Dict
from dataclasses import dataclass

import pandas as pd


@dataclass
class Event:
    name: str
    l_deg: float
    b_deg: float
    ra_deg: float
    dec_deg: float
    lightcurves: Dict[str, pd.DataFrame]

    @classmethod
    def from_name(cls, name: str, df):
        df_event = df[df["name"] == name]

        if df_event.empty:
            raise ValueError(f"Event {name} not found")

        meta = df_event.iloc[0]

        lightcurves = {}
        for filters, df_filters in df_event.groupby("filt"):
            # keep only relevant columns (and copy to avoid view issues)
            lightcurves[filters] = df_filters[
                ["bjd", "mag", "mag_err", "extinction",
                 "obs_x", "obs_y", "obs_z", "saturation_flag"]
            ].copy()

        return cls(
            name=name,
            l_deg=meta["l_deg"],
            b_deg=meta["b_deg"],
            ra_deg=meta["ra_deg"],
            dec_deg=meta["dec_deg"],
            lightcurves=lightcurves,
        )

    @property
    def simple_lightcurve(self) -> Dict[str, pd.DataFrame]:
        return {filters: df_event[["bjd", "mag", "mag_err"]].copy()
                for filters, df_event in self.lightcurves.items()}


if __name__ == "__main__":
    # Example usage
    # Read parquet file as a pandas dataframe
    df = pd.read_parquet("../../../../data_hugging_face/RMDC26_Beginner_Tier_test.parquet")
    df["bjd"] = df["bjd"] - 2_450_000 # adjust if you would like to

    # Load the event you want:
    event = Event.from_name("RMDC26_000001", df)
    print('Event name: ', event.name)
    print('Event l_deg: ', event.l_deg)
    print('Event b_deg: ', event.b_deg)
    print('Event ra_deg: ', event.ra_deg)
    print('Event dec_deg: ', event.dec_deg)

    complete_lightcurve = event.lightcurves["F146"]
    print("Part of complete lightcurve: \n", complete_lightcurve.head(3))
    simple_lightcurve = event.simple_lightcurve["F146"]
    print("Part of simple lightcurve: \n", simple_lightcurve.head(3))

    print()