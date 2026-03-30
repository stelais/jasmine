import os
import pandas as pd
from tqdm import tqdm

from astropy.coordinates import SkyCoord
from astropy import units

from jasmine.classes_and_files_reader import RMDC26_parquet_cls as RMDC26_cls


def creating_rtmodel_directories(
    parquet_path_="data_hugging_face/RMDC26_Beginner_Tier_test.parquet",
    output_dir_="data",
):
    # Load the full dataframe
    df = pd.read_parquet(parquet_path_)
    df["bjd"] = df["bjd"] - 2_450_000  # Adjust times

    # Check if output directory exists and raise error if it does
    if os.path.exists(output_dir_):
        raise FileExistsError(f"Output directory '{output_dir_}' already exists.")
    os.makedirs(output_dir_)

    # Iterate over unique events
    for event_name in tqdm(df["name"].unique()):
        # Create Event object
        event = RMDC26_cls.Event.from_name(event_name, df)

        event_dir = os.path.join(output_dir_, f"event_{event.name}")

        # Check if output directory exists and raise error if it does
        if os.path.exists(event_dir):
            raise FileExistsError(f"Event directory '{event_dir}' already exists.")
        os.makedirs(event_dir)

        # Iterate over available filters in simple_lightcurve
        for filt, lc_df in event.simple_lightcurve.items():
            save_path = os.path.join(event_dir, f"{filt}_.dat")

            # Write header manually and then the data
            with open(save_path, "w") as f:
                f.write("# Mag err HJD-2450000\n")
                # Ensure the columns are in correct order: mag, mag_err, bjd
                lc_df[["mag", "mag_err", "bjd"]].to_csv(
                    f, index=False, header=False, sep=" "
                )

        # Save events coordinates:
        event_coordinates = SkyCoord(
            event.ra_deg, event.dec_deg, unit=(units.deg, units.deg), obstime="J2000"
        )
        rtmodel_event_coordinates = (
            event_coordinates.to_string("hmsdms")
            .replace("h", ":")
            .replace("m", ":")
            .replace("d", ":")
            .replace("s", "")
        )
        if os.path.exists(f"{event_dir}/event.coordinates"):
            raise FileExistsError(
                f"File event.coordinates already exists in directory {event_dir}"
            )
        else:
            with open(f"{event_dir}/event.coordinates", "w") as f:
                f.write(f"{rtmodel_event_coordinates}")
        print(f"{event_dir} conclude!")


if __name__ == "__main__":
    # Example usage
    parquet_path = "data_hugging_face/RMDC26_Beginner_Tier_test.parquet"
    output_dir = "data"
    creating_rtmodel_directories(parquet_path, output_dir)
