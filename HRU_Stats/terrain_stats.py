"""Calculate PRMS terrain parameters (elevation, slope, aspect) per HRU.

Appends one row per HRU to the HRU stats parquet (through
``utils.StatsAppender``), keyed by ``catchment_id``:

    hru_elev    mean HRU elevation, in ``elev_units`` (1 = meters, 0 = feet)
    hru_slope   mean cell slope as a decimal fraction (rise / run)
    hru_aspect  mean downslope direction, degrees clockwise from north

Methods follow the National Hydrologic Model parameter derivation:

    * hru_elev is the mean DEM elevation inside the HRU outline.
    * hru_slope is the mean DEM cell slope inside the HRU.
    * hru_aspect averages the sine and cosine of each cell's aspect and
      takes the inverse tangent of the two means, so aspects on either side
      of north (for example 350 and 10 degrees) average to north instead of
      south.

Cell slope and aspect use the 3x3 Horn method.

DEM choice: use the NHDPlus HR ``elev_cm.tif`` (integer centimeters,
NAVD 88). Don't use hydro-dem!

QA columns (prefixed ``terrain_`` in the parquet): elev_min, elev_max,
n_cells, valid_cell_fraction, n_flat_cells, aspect_resultant, qa_flag.

Optional output: if ``output_dir`` is set, a ``hru_terrain_stats_summary.json``
QA summary is written there. Nothing else is written to disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import StatsAppender, check_hru_keys, load_hrus  # noqa: E402

GROUP_NAME = "terrain"

PARAMETER_COLUMNS = ("hru_elev", "hru_slope", "hru_aspect")

METERS_TO_FEET = 3.280839895

PRMS_ELEV_RANGE = (-1000.0, 30000.0)
PRMS_SLOPE_RANGE = (0.0, 10.0)


class TerrainStatsConfig:
    """Stores terrain-statistics settings.

    Args:
        hru_gpkg_path: Datasource containing the HRU polygon layer.
        parquet_path: HRU stats parquet created by ``HruParquetBuilder``.
        dem_path: Elevation raster. Use the NHDPlus HR ``elev_cm.tif``.
        hru_layer: HRU polygon layer name.
        elevation_scale_to_m: Multiplier that converts raster values to
            meters. 0.01 for ``elev_cm.tif``; 1.0 for a meter DEM.
        elev_units: Units written to ``hru_elev``: 1 meters, 0 feet. Set the
            PRMS ``elev_units`` parameter to match.
        min_cells_warn: HRUs with fewer valid cells are flagged.
        low_aspect_resultant: Mean-vector length below which an HRU's aspect
            is flagged as not well defined (cells face many directions).
        strict: Raise if any HRU has no valid terrain statistics.
        output_dir: Optional folder for the QA summary JSON.
    """

    def __init__(
        self,
        hru_gpkg_path: Path,
        parquet_path: Path,
        dem_path: Path,
        hru_layer: str = "model_hrus_orphan_adopted",
        elevation_scale_to_m: float = 0.01,
        elev_units: int = 1,
        min_cells_warn: int = 4,
        low_aspect_resultant: float = 0.05,
        strict: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        """Initializes and validates terrain settings."""
        if elev_units not in (0, 1):
            raise ValueError("elev_units must be 0 (feet) or 1 (meters).")

        if elevation_scale_to_m <= 0:
            raise ValueError("elevation_scale_to_m must be positive.")

        self.hru_gpkg_path = Path(hru_gpkg_path)
        self.parquet_path = Path(parquet_path)
        self.dem_path = Path(dem_path)
        self.hru_layer = str(hru_layer)
        self.elevation_scale_to_m = float(elevation_scale_to_m)
        self.elev_units = int(elev_units)
        self.min_cells_warn = int(min_cells_warn)
        self.low_aspect_resultant = float(low_aspect_resultant)
        self.strict = bool(strict)
        self.output_dir = (
            Path(output_dir) if output_dir is not None else None
        )


def horn_slope_aspect(
    elevation_m: np.ndarray,
    cell_dx: float,
    cell_dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates Horn slope and aspect for a 2-D elevation array.

    Args:
        elevation_m: Elevations in meters; NaN marks missing cells. Row index
            increases toward the south.
        cell_dx: Cell width in meters.
        cell_dy: Cell height in meters.

    Returns:
        Slope as a decimal fraction (rise / run), and aspect as the downslope
        direction in degrees clockwise from north. Border cells, cells
        touching missing data, and flat cells are NaN for aspect; border and
        missing-neighbor cells are NaN for slope.
    """
    z = np.asarray(elevation_m, dtype=np.float64)
    slope = np.full(z.shape, np.nan)
    aspect = np.full(z.shape, np.nan)

    if z.shape[0] < 3 or z.shape[1] < 3:
        return slope, aspect

    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_dx)
    dz_dy_north = ((a + 2.0 * b + c) - (g + 2.0 * h + i)) / (8.0 * cell_dy)

    inner_slope = np.hypot(dz_dx, dz_dy_north)

    inner_aspect = np.degrees(
        np.arctan2(-dz_dx, -dz_dy_north)
    ) % 360.0

    inner_aspect = np.where(inner_slope > 1e-12, inner_aspect, np.nan)

    slope[1:-1, 1:-1] = inner_slope
    aspect[1:-1, 1:-1] = inner_aspect

    return slope, aspect


def zonal_terrain(
    elevation_m: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Summarizes terrain for the cells selected by ``mask``.

    Returns:
        Dictionary with cell counts, mean/min/max elevation (meters), mean
        raw slope, circular-mean aspect (degrees), the mean-vector length
        (``aspect_resultant``), and the count of flat cells.
    """
    n_cells = int(mask.sum())

    elevation = elevation_m[mask]
    elevation = elevation[np.isfinite(elevation)]

    slopes = slope[mask]
    slopes = slopes[np.isfinite(slopes)]

    aspects = aspect[mask & np.isfinite(slope)]
    n_flat = int(np.isnan(aspects).sum())
    aspects = aspects[np.isfinite(aspects)]

    output: dict[str, Any] = {
        "n_cells": n_cells,
        "n_valid_elev_cells": int(elevation.size),
        "n_valid_slope_cells": int(slopes.size),
        "n_flat_cells": n_flat,
        "elev_mean_m": np.nan,
        "elev_min_m": np.nan,
        "elev_max_m": np.nan,
        "slope_mean_raw": np.nan,
        "aspect_deg": np.nan,
        "aspect_resultant": np.nan,
    }

    if elevation.size:
        output["elev_mean_m"] = float(elevation.mean())
        output["elev_min_m"] = float(elevation.min())
        output["elev_max_m"] = float(elevation.max())

    if slopes.size:
        output["slope_mean_raw"] = float(slopes.mean())

    if aspects.size:
        radians = np.radians(aspects)
        mean_sin = float(np.sin(radians).mean())
        mean_cos = float(np.cos(radians).mean())

        output["aspect_resultant"] = float(np.hypot(mean_sin, mean_cos))
        output["aspect_deg"] = float(
            np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0
        )

    return output


class HruTerrainStats:
    """Computes hru_elev, hru_slope, and hru_aspect for every HRU."""

    def __init__(self, config: TerrainStatsConfig) -> None:
        """Initializes the calculator."""
        self.config = config

    def run(self) -> pd.DataFrame:
        """Computes terrain statistics and appends them to the parquet.

        Returns:
            The terrain table that was appended (before column prefixing).

        Raises:
            FileNotFoundError: If the DEM does not exist.
            ValueError: If an HRU has no valid terrain (strict mode), an
                elevation is outside the PRMS range, or the HRU layer no
                longer matches the parquet.
        """
        if not self.config.dem_path.exists():
            raise FileNotFoundError(
                f"DEM does not exist: {self.config.dem_path}"
            )

        if "hydrodem" in self.config.dem_path.name.lower():
            tqdm.write(
                "WARNING: HydroDEM is hydrologically conditioned (stream "
                "burn and fill). The NHDPlus HR guide says to use "
                "elev_cm.tif for analyses other than flow direction."
            )

        hrus = load_hrus(
            self.config.hru_gpkg_path,
            self.config.hru_layer,
        )

        check_hru_keys(hrus)

        stats = self.calculate(hrus)

        StatsAppender(self.config.parquet_path, GROUP_NAME).append(
            table=stats,
            parameter_columns=PARAMETER_COLUMNS,
            inputs={
                "dem": self.config.dem_path.name,
                "elevation_scale_to_m": self.config.elevation_scale_to_m,
                "elev_units": self.config.elev_units,
            },
            hrus=hrus,
        )

        self._write_summary(stats)

        return stats

    def calculate(self, hrus: Any) -> pd.DataFrame:
        """Calculates terrain parameters for each HRU polygon."""
        records: list[dict[str, Any]] = []

        with rasterio.open(self.config.dem_path) as source:
            if source.crs is None or not source.crs.is_projected:
                raise ValueError(
                    "The DEM must be in a projected CRS with meter cells."
                )

            cell_dx = float(abs(source.transform.a))
            cell_dy = float(abs(source.transform.e))

            projected = hrus.to_crs(source.crs)

            for hru_id, catchment_id, geometry in tqdm(
                zip(
                    projected["hru_id"],
                    projected["catchment_id"],
                    projected.geometry,
                ),
                total=len(projected),
                desc="HRU terrain statistics",
                unit="HRU",
            ):
                record = self._one_hru(
                    source=source,
                    geometry=geometry,
                    cell_dx=cell_dx,
                    cell_dy=cell_dy,
                )
                record["hru_id"] = int(hru_id)
                record["catchment_id"] = str(catchment_id)
                records.append(record)

        return self._finalize(pd.DataFrame(records))

    def _one_hru(
        self,
        source: Any,
        geometry: Any,
        cell_dx: float,
        cell_dy: float,
    ) -> dict[str, Any]:
        """Reads a padded DEM window and summarizes one HRU."""
        min_x, min_y, max_x, max_y = geometry.bounds

        bounds_window = from_bounds(
            min_x,
            min_y,
            max_x,
            max_y,
            transform=source.transform,
        )

        row_start = max(int(np.floor(bounds_window.row_off)) - 1, 0)
        col_start = max(int(np.floor(bounds_window.col_off)) - 1, 0)
        row_stop = min(
            int(
                np.ceil(bounds_window.row_off + bounds_window.height)
            )
            + 1,
            source.height,
        )
        col_stop = min(
            int(np.ceil(bounds_window.col_off + bounds_window.width)) + 1,
            source.width,
        )

        if row_stop <= row_start or col_stop <= col_start:
            return self._empty_record("outside_dem")

        window = Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start,
        )

        raw = source.read(1, window=window, masked=True)

        elevation_m = (
            np.ma.filled(raw.astype(np.float64), np.nan)
            * self.config.elevation_scale_to_m
        )

        transform = window_transform(window, source.transform)

        mask = geometry_mask(
            [geometry],
            out_shape=elevation_m.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )

        flag = "ok"

        if not mask.any():
            mask = geometry_mask(
                [geometry],
                out_shape=elevation_m.shape,
                transform=transform,
                invert=True,
                all_touched=True,
            )
            flag = "small_hru_all_touched"

        if not mask.any():
            return self._empty_record("no_cells")

        slope, aspect = horn_slope_aspect(elevation_m, cell_dx, cell_dy)

        record = zonal_terrain(elevation_m, slope, aspect, mask)
        record["qa_flag"] = flag

        return record

    @staticmethod
    def _empty_record(flag: str) -> dict[str, Any]:
        """Returns an all-missing record with a QA flag."""
        return {
            "n_cells": 0,
            "n_valid_elev_cells": 0,
            "n_valid_slope_cells": 0,
            "n_flat_cells": 0,
            "elev_mean_m": np.nan,
            "elev_min_m": np.nan,
            "elev_max_m": np.nan,
            "slope_mean_raw": np.nan,
            "aspect_deg": np.nan,
            "aspect_resultant": np.nan,
            "qa_flag": flag,
        }

    def _finalize(self, table: pd.DataFrame) -> pd.DataFrame:
        """Converts units, applies PRMS limits, and assigns QA flags."""
        factor = METERS_TO_FEET if self.config.elev_units == 0 else 1.0

        output = table.copy()

        output["hru_elev"] = output["elev_mean_m"] * factor
        output["elev_min"] = output["elev_min_m"] * factor
        output["elev_max"] = output["elev_max_m"] * factor

        output["hru_slope"] = output["slope_mean_raw"].clip(
            *PRMS_SLOPE_RANGE
        )

        output["hru_aspect"] = output["aspect_deg"].fillna(0.0)

        output["valid_cell_fraction"] = np.where(
            output["n_cells"] > 0,
            output["n_valid_elev_cells"] / output["n_cells"].clip(lower=1),
            0.0,
        )

        flags: list[str] = []

        for row in output.itertuples():
            issues: list[str] = []

            if row.qa_flag != "ok":
                issues.append(str(row.qa_flag))

            if pd.isna(row.hru_elev) or pd.isna(row.hru_slope):
                issues.append("no_valid_terrain")
            else:
                if row.n_valid_elev_cells < self.config.min_cells_warn:
                    issues.append("few_cells")

                if row.slope_mean_raw > PRMS_SLOPE_RANGE[1]:
                    issues.append("slope_clipped")

                if pd.isna(row.aspect_deg):
                    issues.append("aspect_undefined_flat")
                elif (
                    row.aspect_resultant < self.config.low_aspect_resultant
                ):
                    issues.append("aspect_weak")

                if row.valid_cell_fraction < 0.9:
                    issues.append("dem_gaps")

            flags.append(";".join(dict.fromkeys(issues)) or "ok")

        output["qa_flag"] = flags

        lower, upper = PRMS_ELEV_RANGE

        out_of_range = output["hru_elev"].notna() & (
            (output["hru_elev"] < lower) | (output["hru_elev"] > upper)
        )

        if out_of_range.any():
            raise ValueError(
                f"{int(out_of_range.sum()):,} HRU(s) have hru_elev outside "
                f"the PRMS range {PRMS_ELEV_RANGE}. Check "
                "elevation_scale_to_m (0.01 for elev_cm.tif) and "
                "elev_units."
            )

        missing = output[
            output["hru_elev"].isna() | output["hru_slope"].isna()
        ]

        if not missing.empty and self.config.strict:
            raise ValueError(
                f"{len(missing):,} HRU(s) have no valid terrain statistics, "
                f"e.g. hru_id {missing['hru_id'].head(10).tolist()}. "
                "They may lie outside the DEM. Set strict=False to continue."
            )

        columns = [
            "hru_id",
            "catchment_id",
            "hru_elev",
            "hru_slope",
            "hru_aspect",
            "elev_min",
            "elev_max",
            "n_cells",
            "valid_cell_fraction",
            "n_flat_cells",
            "aspect_resultant",
            "qa_flag",
        ]

        return output[columns].sort_values("hru_id").reset_index(drop=True)

    def _write_summary(self, stats: pd.DataFrame) -> None:
        """Prints a QA summary and optionally writes it as JSON."""
        valid = stats.dropna(subset=["hru_elev"])

        summary = {
            "n_hrus": int(len(stats)),
            "n_with_terrain": int(len(valid)),
            "elev_units": (
                "meters" if self.config.elev_units == 1 else "feet"
            ),
            "hru_elev_min": float(valid["hru_elev"].min()),
            "hru_elev_max": float(valid["hru_elev"].max()),
            "hru_slope_mean": float(valid["hru_slope"].mean()),
            "hru_slope_max": float(valid["hru_slope"].max()),
            "qa_flag_counts": {
                str(flag): int(count)
                for flag, count in stats["qa_flag"]
                .value_counts()
                .items()
            },
        }

        tqdm.write(
            f"Terrain: {summary['n_with_terrain']:,} of "
            f"{summary['n_hrus']:,} HRUs have valid statistics. "
            f"QA flags: {summary['qa_flag_counts']}"
        )

        if self.config.output_dir is None:
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        (
            self.config.output_dir / "hru_terrain_stats_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )