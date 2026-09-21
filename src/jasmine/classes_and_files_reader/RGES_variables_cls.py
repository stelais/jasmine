from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


RAW_FOLDERS = {
    "cep": "CEP",
    "cv_dn": "DwarfNovae",
    "dsct": "DSCT",
    "ecl": "ECL",
    "ell": "ELL",
    "fl": "flare",
    "hb": "HB",
    "lpv": "LPV",
    "rrlyrae": "RRLYR",
    "t2cep": "T2CEP",
}


@lru_cache(maxsize=None)
def read_raw_coordinates(path: Path) -> tuple[str, float, float]:
    """Read and validate source coordinates in decimal degrees."""
    header = fits.getheader(path, ext=0)

    name = str(header["NAME"]).strip()
    ra = float(header["RA"])
    dec = float(header["DEC"])

    if not (np.isfinite(ra) and np.isfinite(dec)):
        raise ValueError(f"Non-finite coordinates in {path}")

    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError(f"Coordinates outside degree ranges in {path}")

    return name, ra, dec


@dataclass
class VariableStarEvent:
    fits_path: str | Path
    raw_dir: str | Path
    object_id: int | None = None
    zeropoint: float = 27.615
    header: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.fits_path = Path(self.fits_path)
        self.raw_dir = Path(self.raw_dir)

        if not self.fits_path.is_file():
            raise FileNotFoundError(self.fits_path)

        if not self.raw_dir.is_dir():
            raise NotADirectoryError(self.raw_dir)

        with fits.open(self.fits_path, memmap=True) as hdul:
            self.header = dict(hdul[0].header)

    @property
    def objname(self) -> str:
        return str(
            self.header.get("NAME") or self.fits_path.stem
        ).strip()

    @property
    def vartype(self) -> str | None:
        value = self.header.get("VARTYPE")
        return str(value).strip() if value is not None else None

    def find_coordinates(self) -> tuple[Path, float, float]:
        """Match the Roman event to its original raw FITS file."""
        vartype = (self.vartype or "").lower()

        if vartype not in RAW_FOLDERS:
            raise ValueError(f"Unknown Roman VARTYPE: {vartype!r}")

        folder = self.raw_dir / RAW_FOLDERS[vartype]
        names = [self.objname]

        # Match simulation names such as OGLE-BLG-DN-0001_ind0_3.
        source_name = re.sub(r"_ind\d+(?:_\d+)*$", "", self.objname)
        if source_name != self.objname:
            names.append(source_name)

        for candidate in names:
            path = folder / f"{candidate}_multiband_lc.fits"

            if not path.is_file():
                continue

            raw_name, ra, dec = read_raw_coordinates(path)

            if raw_name != candidate:
                raise ValueError(
                    f"NAME mismatch in {path}: "
                    f"expected {candidate!r}, found {raw_name!r}"
                )

            return path, ra, dec

        raise FileNotFoundError(
            f"No raw counterpart for {self.objname!r} "
            f"in {folder}; tried {names}"
        )

    @staticmethod
    def magnitude_to_flux(
        mag: np.ndarray,
        mag_err: np.ndarray,
        zeropoint: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert magnitudes and errors to linear flux units."""
        flux = 10.0 ** ((zeropoint - mag) / 2.5)
        flux_err = mag_err * flux * np.log(10.0) / 2.5
        return flux, flux_err

    def read_lightcurves(self) -> dict[str, pd.DataFrame]:
        """Read every Roman FITS filter as a flux light curve."""
        curves: dict[str, pd.DataFrame] = {}

        with fits.open(self.fits_path, memmap=True) as hdul:
            for hdu in hdul[1:]:
                table = hdu.data

                if table is None or not hasattr(table, "columns"):
                    continue

                columns = {
                    name.lower(): name
                    for name in table.columns.names
                }

                if not {"jd", "mag", "mag_error"}.issubset(columns):
                    continue

                time = np.asarray(table[columns["jd"]], dtype=float)
                mag = np.asarray(table[columns["mag"]], dtype=float)
                mag_err = np.asarray(
                    table[columns["mag_error"]], dtype=float
                )

                valid = (
                    np.isfinite(time)
                    & np.isfinite(mag)
                    & np.isfinite(mag_err)
                    & (mag_err >= 0)
                )

                flux, flux_err = self.magnitude_to_flux(
                    mag=mag[valid],
                    mag_err=mag_err[valid],
                    zeropoint=self.zeropoint,
                )

                curves[hdu.name] = (
                    pd.DataFrame({
                        "time": time[valid],
                        "flux": flux,
                        "flux_err": flux_err,
                    })
                    .sort_values("time")
                    .reset_index(drop=True)
                )

        if not curves:
            raise ValueError(
                f"No compatible light-curve tables in {self.fits_path}"
            )

        return curves

    def to_dataframe(self) -> pd.DataFrame:
        """Combine all Roman filters into one DataFrame."""
        chunks: list[pd.DataFrame] = []

        for filter_name, light_curve in self.read_lightcurves().items():
            frame = light_curve.copy()
            frame.insert(0, "filter", filter_name)
            chunks.append(frame)

        return pd.concat(chunks, ignore_index=True)

    def to_json_dict(self) -> dict:
        """Build an event dictionary with original source coordinates."""
        raw_path, ra, dec = self.find_coordinates()
        curves = self.read_lightcurves()

        light_curves = {
            filter_name: {
                column: frame[column].astype(float).tolist()
                for column in ("time", "flux", "flux_err")
            }
            for filter_name, frame in curves.items()
        }

        return {
            "id": self.object_id,
            "objname": self.objname,
            "ra": ra,
            "dec": dec,
            "photometric_variability": None,
            "metadata": {
                "name": self.objname,
                "vartype": self.vartype,
                "source_fits": self.fits_path.name,
                "coordinate_source_fits": str(raw_path),
                "coordinate_units": "deg",
                "input_photometry": "magnitude",
                "output_photometry": "flux",
                "magnitude_zeropoint": self.zeropoint,
            },
            "microlensing_event": None,
            "light_curve": None,
            "light_curves": light_curves,
        }

    def save_json(self, output_path: str | Path) -> Path:
        """Validate and serialize before opening the destination."""
        text = json.dumps(
            self.to_json_dict(),
            indent=2,
            allow_nan=False,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

        return output_path