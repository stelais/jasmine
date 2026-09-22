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
    """Read source coordinates in decimal degrees."""
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

        if not np.isfinite(self.zeropoint):
            raise ValueError("zeropoint must be finite")

        self.header = dict(fits.getheader(self.fits_path, ext=0))

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
        """Match a Roman event to its original raw FITS file."""
        vartype = (self.vartype or "").lower()

        if vartype not in RAW_FOLDERS:
            raise ValueError(f"Unknown Roman VARTYPE: {vartype!r}")

        folder = self.raw_dir / RAW_FOLDERS[vartype]
        names = [self.objname]

        # Try exact names before removing a simulation suffix.
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
        """Convert magnitudes and propagate their uncertainties."""
        with np.errstate(over="raise", invalid="raise"):
            flux = 10.0 ** ((zeropoint - mag) / 2.5)
            flux_err = mag_err * flux * np.log(10.0) / 2.5

        if not (
            np.all(np.isfinite(flux))
            and np.all(np.isfinite(flux_err))
        ):
            raise ValueError("Flux conversion produced non-finite values")

        # Apply the requested offset to the entire filter array.
        if flux.size > 0:
            minimum = np.min(flux)
            if minimum < 0:
                flux = flux - minimum + 1e-8

        return flux, flux_err

    def read_lightcurves(self) -> dict[str, pd.DataFrame]:
        """Read Roman filters as time, flux, and flux_err arrays."""
        curves: dict[str, pd.DataFrame] = {}

        with fits.open(self.fits_path, memmap=True) as hdul:
            for hdu in hdul[1:]:
                if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    continue

                table = hdu.data
                if table is None:
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

                if not np.any(valid):
                    raise ValueError(
                        f"No valid photometry in {self.fits_path}, "
                        f"filter {hdu.name}"
                    )

                flux, flux_err = self.magnitude_to_flux(
                    mag[valid], mag_err[valid], self.zeropoint
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
        chunks = []

        for filter_name, frame in self.read_lightcurves().items():
            frame = frame.copy()
            frame.insert(0, "filter", filter_name)
            chunks.append(frame)

        return pd.concat(chunks, ignore_index=True)

    def to_json_dict(self) -> dict:
        coordinate_error = None

        try:
            raw_path, ra, dec = self.find_coordinates()
            raw_source = str(raw_path)
            raw_match_status = "found"
        except FileNotFoundError as error:
            ra = None
            dec = None
            raw_source = None
            raw_match_status = "not_found"
            coordinate_error = str(error)

        curves = self.read_lightcurves()

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
                "raw_match_status": raw_match_status,
                "coordinate_source_fits": raw_source,
                "coordinate_error": coordinate_error,
                "coordinate_units": "deg",
                "input_photometry": "magnitude",
                "output_photometry": "flux",
                "magnitude_zeropoint": self.zeropoint,
            },
            "microlensing_event": None,
            "light_curve": None,
            "light_curves": {
                filter_name: {
                    column: frame[column].astype(float).tolist()
                    for column in ("time", "flux", "flux_err")
                }
                for filter_name, frame in curves.items()
            },
        }

    def save_json(self, output_path: str | Path) -> Path:
        return write_json(output_path, self.to_json_dict())


def write_json(path: str | Path, payload: dict) -> Path:
    """Serialize before opening the output file."""
    text = json.dumps(payload, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def all_events_to_json(base_path: str | Path) -> Path:
    base_path = Path(base_path)
    input_dir = base_path / "raw_roman"
    output_dir = base_path / "json_events"
    raw_dir = base_path / "raw"

    for directory in (input_dir, raw_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    fits_paths = sorted(input_dir.rglob("*.fits"))
    if not fits_paths:
        raise RuntimeError(f"No FITS files found in {input_dir}")

    saved = 0
    skipped_files = []
    duplicate_outputs = []
    missing_raw = []
    failures = []
    seen_destinations = {}

    print(f"Found {len(fits_paths)} FITS files", flush=True)

    for index, fits_path in enumerate(fits_paths, start=1):
        try:
            category = fits_path.parent.name.removeprefix(
                "RGES_filters_"
            ).removesuffix("_lightcurves_final")

            event = VariableStarEvent(
                fits_path=fits_path,
                raw_dir=raw_dir,
                zeropoint=27.615,
            )

            output_name = fits_path.stem.removeprefix(
                "RGES_filters_"
            ).removesuffix("_lightcurves_final")

            destination = output_dir / category / f"{output_name}.json"

            # Detect different inputs targeting the same output this run.
            if destination in seen_destinations:
                duplicate_outputs.append({
                    "fits_path": str(fits_path),
                    "first_fits_path": str(
                        seen_destinations[destination]
                    ),
                    "json_path": str(destination),
                    "objname": event.objname,
                    "reason": "Multiple FITS files target the same JSON",
                })
            else:
                seen_destinations[destination] = fits_path

                if destination.exists():
                    with destination.open("r", encoding="utf-8") as file:
                        payload = json.load(file)

                    skipped_files.append({
                        "fits_path": str(fits_path),
                        "json_path": str(destination),
                        "objname": event.objname,
                        "reason": "JSON already exists",
                        "stored_source_fits": (
                            payload.get("metadata") or {}
                        ).get("source_fits"),
                    })
                else:
                    payload = event.to_json_dict()
                    write_json(destination, payload)
                    saved += 1

                metadata = payload.get("metadata") or {}
                if metadata.get("raw_match_status") == "not_found":
                    missing_raw.append({
                        "fits_path": str(fits_path),
                        "json_path": str(destination),
                        "objname": payload.get("objname"),
                        "error": metadata.get("coordinate_error"),
                    })

        except Exception as error:
            failures.append({
                "fits_path": str(fits_path),
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"FAILED: {fits_path.name}: {error}", flush=True)

        if index % 100 == 0 or index == len(fits_paths):
            print(
                f"[{index}/{len(fits_paths)}] "
                f"Saved: {saved} | "
                f"Skipped: {len(skipped_files)} | "
                f"Duplicate outputs: {len(duplicate_outputs)} | "
                f"Missing raw: {len(missing_raw)} | "
                f"Failed: {len(failures)}",
                flush=True,
            )

    report_path = write_json(
        output_dir / "conversion_report.json",
        {
            "input_dir": str(input_dir),
            "total": len(fits_paths),
            "saved": saved,
            "skipped_existing": len(skipped_files),
            "skipped_files": skipped_files,
            "duplicate_output_count": len(duplicate_outputs),
            "duplicate_outputs": duplicate_outputs,
            "missing_raw_count": len(missing_raw),
            "missing_raw": missing_raw,
            "failed": len(failures),
            "failures": failures,
        },
    )

    print(f"\nOutput: {output_dir}")
    print(f"Report: {report_path}")
    return report_path


if __name__ == "__main__":
    # EXAMPLE
    # FOR ALL EVENTS
    base = Path("your_path")
    all_events_to_json(base)

    # FOR ONE EVENT
    event = VariableStarEvent(
        fits_path=(
                base / "raw_roman"
                / "RGES_filters_CV_lightcurves_final"
                / "RGES_filters_OGLE-BLG-DN-0001_ind0_3_lightcurves_final.fits"
        ),
        raw_dir=base / "raw",
    )

    raw_path, ra, dec = event.find_coordinates()
    print(f"Source: {raw_path}")
    print(f"RA: {ra}, Dec: {dec}")

    event.save_json(base / "json_events" / "CV" / f"{event.objname}.json")