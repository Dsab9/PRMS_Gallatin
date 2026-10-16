"""Calculate PRMS land-cover parameters for every HRU from Annual NLCD.

Appends one row per HRU to the HRU stats parquet (through
``utils.StatsAppender``), keyed by ``catchment_id``.

Inputs (all in one folder, one year):
    Annual NLCD Land Cover        (class codes 11-95, nodata 250)
    Annual NLCD Fractional Impervious Surface (0-100 percent, nodata 250)
    NLCD / USFS Tree Canopy Cover (0-100 percent; 254 and 255 are not data)

PRMS parameters appended:

    cov_type          dominant vegetation: 0 bare, 1 grasses, 2 shrubs,
                      3 deciduous trees, 4 coniferous trees
    covden_sum        summer vegetation cover density (0-1)
    covden_win        winter vegetation cover density (0-1)
    snow_intcp        winter snow interception storage capacity (inches)
    srain_intcp       summer rain interception storage capacity (inches)
    wrain_intcp       winter rain interception storage capacity (inches)
    hru_percent_imperv impervious fraction of the HRU (0-1)

How the values are made:

    * Each NLCD class maps to a PRMS cover type (``class_to_cov_type``).
      Open water is excluded. Developed classes count as grass for the
      pervious part; their imperviousness is handled by hru_percent_imperv.
    * cov_type is the PRMS type covering the most valid land cells.
    * For tree types (3, 4), covden_sum is the mean tree canopy cover over
      the whole HRU, which is the area fraction under canopy. For bare,
      grass, and shrub types there is no canopy raster, so covden_sum comes
      from ``vegetation_defaults``.
    * covden_win is covden_sum times a per-type winter factor (deciduous
      trees lose leaves).
    * Interception capacities come from ``vegetation_defaults`` by cov_type.
    * hru_percent_imperv is mean fractional impervious surface over the HRU.

IMPORTANT: the numbers in ``DEFAULT_VEGETATION_PARAMETERS`` (non-tree cover
density, winter factors, interception capacities) are starter placeholders,
not values taken from a PRMS or National Hydrologic Model table. Replace them
with the values you intend to use before parameterizing the model. The values
used, and the class map, are saved in the parquet manifest with this group so
they are never silent.

QA columns (prefixed ``land_cover_`` in the parquet): dominant_nlcd_class,
tcc_mean_pct, imperv_mean_pct, n_lc_cells, n_land_cells, n_canopy_cells,
n_imperv_cells, class-group fractions (frac_*), and qa_flag.

Optional output: if ``output_dir`` is set, a
``hru_land_cover_stats_summary.json`` QA summary is written there. Nothing
else is written to disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    StatsAppender,
    check_hru_keys,
    load_hrus,
    read_hru_values,
)

GROUP_NAME = "land_cover"

PARAMETER_COLUMNS = (
    "cov_type",
    "covden_sum",
    "covden_win",
    "snow_intcp",
    "srain_intcp",
    "wrain_intcp",
    "hru_percent_imperv",
)

COV_BARE = 0
COV_GRASS = 1
COV_SHRUB = 2
COV_DECIDUOUS = 3
COV_CONIFER = 4

NLCD_NODATA = 250

NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}

# NLCD class -> PRMS cov_type. None means excluded from vegetation (water).
DEFAULT_CLASS_TO_COV_TYPE: dict[int, int | None] = {
    11: None,
    12: COV_BARE,
    21: COV_GRASS,
    22: COV_GRASS,
    23: COV_GRASS,
    24: COV_GRASS,
    31: COV_BARE,
    41: COV_DECIDUOUS,
    42: COV_CONIFER,
    43: COV_CONIFER,
    52: COV_SHRUB,
    71: COV_GRASS,
    81: COV_GRASS,
    82: COV_GRASS,
    90: COV_DECIDUOUS,
    95: COV_GRASS,
}

# STARTER PLACEHOLDERS - review and replace. Keyed by PRMS cov_type.
DEFAULT_VEGETATION_PARAMETERS: dict[int, dict[str, float]] = {
    COV_BARE: {
        "covden_sum": 0.0,
        "covden_win_factor": 1.0,
        "snow_intcp": 0.0,
        "srain_intcp": 0.0,
        "wrain_intcp": 0.0,
    },
    COV_GRASS: {
        "covden_sum": 0.5,
        "covden_win_factor": 1.0,
        "snow_intcp": 0.1,
        "srain_intcp": 0.1,
        "wrain_intcp": 0.1,
    },
    COV_SHRUB: {
        "covden_sum": 0.5,
        "covden_win_factor": 1.0,
        "snow_intcp": 0.1,
        "srain_intcp": 0.1,
        "wrain_intcp": 0.1,
    },
    COV_DECIDUOUS: {
        "covden_sum": 0.5,
        "covden_win_factor": 0.5,
        "snow_intcp": 0.1,
        "srain_intcp": 0.1,
        "wrain_intcp": 0.1,
    },
    COV_CONIFER: {
        "covden_sum": 0.5,
        "covden_win_factor": 1.0,
        "snow_intcp": 0.1,
        "srain_intcp": 0.1,
        "wrain_intcp": 0.1,
    },
}

GROUP_NAMES = {
    "frac_water": (11,),
    "frac_ice": (12,),
    "frac_developed": (21, 22, 23, 24),
    "frac_barren": (31,),
    "frac_deciduous": (41,),
    "frac_evergreen": (42,),
    "frac_mixed": (43,),
    "frac_shrub": (52,),
    "frac_grass_crop": (71, 81, 82),
    "frac_wetland": (90, 95),
}


class LandCoverConfig:
    """Stores land-cover statistics settings.

    Args:
        hru_gpkg_path: Datasource containing the HRU polygon layer.
        parquet_path: HRU stats parquet created by ``HruParquetBuilder``.
        nlcd_dir: Folder holding the Annual NLCD rasters and tree canopy.
        year: Year in the raster file names (used to pick the files).
        hru_layer: HRU polygon layer name.
        land_cover_path: Optional explicit Land Cover raster.
        impervious_path: Optional explicit Fractional Impervious raster.
        canopy_path: Optional explicit Tree Canopy Cover raster.
        class_to_cov_type: NLCD class -> PRMS cov_type map (None excludes).
        vegetation_defaults: Per-cov_type parameter values (placeholders).
        min_cells_warn: HRUs with fewer valid land-cover cells are flagged.
        water_fraction_warn: Open-water fraction at or above which an HRU is
            flagged as mostly water.
        strict: Raise if any HRU has no valid land-cover cells.
        output_dir: Optional folder for the QA summary JSON.
    """

    def __init__(
        self,
        hru_gpkg_path: Path,
        parquet_path: Path,
        nlcd_dir: Path,
        year: int = 2025,
        hru_layer: str = "model_hrus_orphan_adopted",
        land_cover_path: Path | None = None,
        impervious_path: Path | None = None,
        canopy_path: Path | None = None,
        class_to_cov_type: dict[int, int | None] | None = None,
        vegetation_defaults: dict[int, dict[str, float]] | None = None,
        min_cells_warn: int = 4,
        water_fraction_warn: float = 0.5,
        strict: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        """Initializes and validates settings."""
        self.hru_gpkg_path = Path(hru_gpkg_path)
        self.parquet_path = Path(parquet_path)
        self.nlcd_dir = Path(nlcd_dir)
        self.year = int(year)
        self.hru_layer = str(hru_layer)
        self.land_cover_path = land_cover_path
        self.impervious_path = impervious_path
        self.canopy_path = canopy_path
        self.class_to_cov_type = dict(
            class_to_cov_type
            if class_to_cov_type is not None
            else DEFAULT_CLASS_TO_COV_TYPE
        )
        self.vegetation_defaults = {
            int(k): dict(v)
            for k, v in (
                vegetation_defaults
                if vegetation_defaults is not None
                else DEFAULT_VEGETATION_PARAMETERS
            ).items()
        }
        self.min_cells_warn = int(min_cells_warn)
        self.water_fraction_warn = float(water_fraction_warn)
        self.strict = bool(strict)
        self.output_dir = (
            Path(output_dir) if output_dir is not None else None
        )

        for cov_type in range(5):
            if cov_type not in self.vegetation_defaults:
                raise ValueError(
                    f"vegetation_defaults is missing cov_type {cov_type}."
                )

        bad = {
            c: v
            for c, v in self.class_to_cov_type.items()
            if v is not None and v not in range(5)
        }

        if bad:
            raise ValueError(f"Invalid cov_type in class map: {bad}")


def find_raster(
    folder: Path,
    year: int,
    include_any: tuple[str, ...],
    label: str,
    exclude_any: tuple[str, ...] = (),
) -> Path:
    """Finds exactly one GeoTIFF in ``folder`` by name tokens and year.

    Raises:
        FileNotFoundError: If the folder is missing or nothing matches.
        ValueError: If several files match (pass an explicit path).
    """
    if not folder.exists():
        raise FileNotFoundError(f"NLCD folder does not exist: {folder}")

    matches = []

    for path in sorted(folder.iterdir()):
        name = path.name.lower()

        if path.suffix.lower() not in (".tif", ".tiff"):
            continue

        if str(year) not in name:
            continue

        if not any(token in name for token in include_any):
            continue

        if any(token in name for token in exclude_any):
            continue

        matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"No {label} raster for {year} found in {folder} "
            f"(looked for names containing {include_any})."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Several {label} rasters match: "
            f"{[m.name for m in matches]}. Pass an explicit path."
        )

    return matches[0]


def summarize_land_cover(
    values: np.ndarray,
    class_to_cov_type: dict[int, int | None],
) -> dict[str, Any]:
    """Summarizes NLCD class values inside one HRU.

    Args:
        values: Land-cover class values (NaN or unknown codes are ignored).
        class_to_cov_type: NLCD class -> PRMS cov_type (None = water).

    Returns:
        Valid-cell count, class-group fractions, dominant NLCD class, and
        the dominant PRMS cov_type (None if there are no land cells).
    """
    valid_classes = np.array(sorted(class_to_cov_type), dtype=np.float64)

    valid = values[np.isin(values, valid_classes)].astype(int)
    n_valid = int(valid.size)

    output: dict[str, Any] = {
        "n_lc_cells": n_valid,
        "dominant_nlcd_class": None,
        "cov_type": None,
        "n_land_cells": 0,
    }

    for name in GROUP_NAMES:
        output[name] = np.nan

    if n_valid == 0:
        return output

    for name, classes in GROUP_NAMES.items():
        output[name] = float(np.isin(valid, classes).sum() / n_valid)

    counts = pd.Series(valid).value_counts()
    output["dominant_nlcd_class"] = int(counts.index[0])

    type_counts = np.zeros(5, dtype=int)

    for nlcd_class, count in counts.items():
        cov_type = class_to_cov_type.get(int(nlcd_class))

        if cov_type is not None:
            type_counts[cov_type] += int(count)

    output["n_land_cells"] = int(type_counts.sum())

    if type_counts.sum() > 0:
        output["cov_type"] = int(np.argmax(type_counts))

    return output


def mean_percent(
    values: np.ndarray,
    maximum: float = 100.0,
) -> tuple[float, int]:
    """Returns the mean of valid 0..maximum percent values as 0..1.

    Values above ``maximum`` (nodata and non-processing codes such as 250,
    254, 255) and NaN are excluded.

    Returns:
        Mean fraction (NaN if no valid cells) and the valid-cell count.
    """
    valid = values[np.isfinite(values) & (values >= 0) & (values <= maximum)]

    if valid.size == 0:
        return float("nan"), 0

    return float(valid.mean() / maximum), int(valid.size)


def derive_parameters(
    cov_type: int | None,
    canopy_fraction: float,
    impervious_fraction: float,
    vegetation_defaults: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """Builds the PRMS land-cover parameters for one HRU.

    Returns:
        Parameters plus a list of notes for the QA flag.
    """
    notes: list[str] = []

    if cov_type is None:
        cov_type = COV_BARE
        notes.append("no_land_cells")

    defaults = vegetation_defaults[cov_type]

    if cov_type in (COV_DECIDUOUS, COV_CONIFER):
        if np.isfinite(canopy_fraction):
            covden_sum = canopy_fraction
        else:
            covden_sum = defaults["covden_sum"]
            notes.append("canopy_missing_default_used")
    else:
        covden_sum = defaults["covden_sum"]

    covden_sum = float(np.clip(covden_sum, 0.0, 1.0))
    covden_win = float(
        np.clip(covden_sum * defaults["covden_win_factor"], 0.0, 1.0)
    )

    if np.isfinite(impervious_fraction):
        imperv = float(np.clip(impervious_fraction, 0.0, 1.0))
    else:
        imperv = 0.0
        notes.append("impervious_missing")

    return {
        "cov_type": int(cov_type),
        "covden_sum": covden_sum,
        "covden_win": covden_win,
        "snow_intcp": float(defaults["snow_intcp"]),
        "srain_intcp": float(defaults["srain_intcp"]),
        "wrain_intcp": float(defaults["wrain_intcp"]),
        "hru_percent_imperv": imperv,
        "notes": notes,
    }


class HruLandCoverStats:
    """Computes PRMS land-cover parameters for every HRU."""

    def __init__(self, config: LandCoverConfig) -> None:
        """Initializes the calculator."""
        self.config = config

    def resolve_rasters(self) -> dict[str, Path]:
        """Locates the three input rasters."""
        folder = self.config.nlcd_dir
        year = self.config.year

        paths = {
            "land_cover": self.config.land_cover_path
            or find_raster(
                folder,
                year,
                ("lndcov", "landcover", "land_cover"),
                "Land Cover",
                exclude_any=("lndchg", "lndcnf", "confidence", "change"),
            ),
            "impervious": self.config.impervious_path
            or find_raster(
                folder,
                year,
                ("fctimp", "fractional"),
                "Fractional Impervious",
            ),
            "canopy": self.config.canopy_path
            or find_raster(
                folder,
                year,
                ("tcc", "canopy"),
                "Tree Canopy Cover",
            ),
        }

        for label, path in paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} raster missing: {path}")

        return {label: Path(path) for label, path in paths.items()}

    def run(self) -> pd.DataFrame:
        """Computes land-cover parameters and appends them to the parquet.

        Returns:
            The land-cover table that was appended (before prefixing).

        Raises:
            FileNotFoundError: If a raster cannot be found.
            ValueError: If an HRU has no valid land cover (strict mode) or
                the HRU layer no longer matches the parquet.
        """
        rasters = self.resolve_rasters()

        for label, path in rasters.items():
            tqdm.write(f"{label}: {path.name}")

        hrus = load_hrus(
            self.config.hru_gpkg_path,
            self.config.hru_layer,
        )

        check_hru_keys(hrus)

        table = self.calculate(hrus, rasters)

        StatsAppender(self.config.parquet_path, GROUP_NAME).append(
            table=table,
            parameter_columns=PARAMETER_COLUMNS,
            inputs={
                "year": self.config.year,
                **{label: path.name for label, path in rasters.items()},
                "class_to_cov_type": json.dumps(
                    {str(k): v for k, v in self.config.class_to_cov_type.items()}
                ),
                "vegetation_defaults_REVIEW_REQUIRED": json.dumps(
                    {
                        str(k): v
                        for k, v in self.config.vegetation_defaults.items()
                    }
                ),
            },
            hrus=hrus,
        )

        self._write_summary(table, rasters)

        return table

    def calculate(
        self,
        hrus: Any,
        rasters: dict[str, Path],
    ) -> pd.DataFrame:
        """Calculates land-cover parameters for each HRU polygon."""
        records: list[dict[str, Any]] = []

        sources = {
            label: rasterio.open(path) for label, path in rasters.items()
        }

        try:
            for label, source in sources.items():
                if source.crs is None or not source.crs.is_projected:
                    raise ValueError(
                        f"{label} raster must be in a projected CRS."
                    )

            projected = {
                label: hrus.to_crs(source.crs)
                for label, source in sources.items()
            }

            catchment_ids = hrus["catchment_id"].astype(str).tolist()

            for position in tqdm(
                range(len(hrus)),
                desc="HRU land cover",
                unit="HRU",
            ):
                values: dict[str, np.ndarray] = {}
                flags: list[str] = []

                for label, source in sources.items():
                    data, flag = read_hru_values(
                        source,
                        projected[label].geometry.iloc[position],
                    )
                    values[label] = data

                    if flag != "ok":
                        flags.append(f"{label}:{flag}")

                land = summarize_land_cover(
                    values["land_cover"],
                    self.config.class_to_cov_type,
                )

                canopy_fraction, n_canopy = mean_percent(values["canopy"])
                imperv_fraction, n_imperv = mean_percent(
                    values["impervious"]
                )

                derived = derive_parameters(
                    land["cov_type"],
                    canopy_fraction,
                    imperv_fraction,
                    self.config.vegetation_defaults,
                )

                record = {
                    "hru_id": int(hrus["hru_id"].iloc[position]),
                    "catchment_id": catchment_ids[position],
                    **{key: derived[key] for key in PARAMETER_COLUMNS},
                    "dominant_nlcd_class": land["dominant_nlcd_class"],
                    "tcc_mean_pct": (
                        canopy_fraction * 100.0
                        if np.isfinite(canopy_fraction)
                        else np.nan
                    ),
                    "imperv_mean_pct": (
                        imperv_fraction * 100.0
                        if np.isfinite(imperv_fraction)
                        else np.nan
                    ),
                    "n_lc_cells": land["n_lc_cells"],
                    "n_land_cells": land["n_land_cells"],
                    "n_canopy_cells": n_canopy,
                    "n_imperv_cells": n_imperv,
                    **{name: land[name] for name in GROUP_NAMES},
                }

                record["qa_flag"] = self._flag(
                    record,
                    derived["notes"] + flags,
                )

                records.append(record)
        finally:
            for source in sources.values():
                source.close()

        table = pd.DataFrame(records)

        missing = table[table["n_lc_cells"] == 0]

        if not missing.empty and self.config.strict:
            raise ValueError(
                f"{len(missing):,} HRU(s) have no valid land-cover cells, "
                f"e.g. hru_id {missing['hru_id'].head(10).tolist()}. "
                "They may lie outside the NLCD raster. Set strict=False "
                "to continue."
            )

        return table.sort_values("hru_id").reset_index(drop=True)

    def _flag(
        self,
        record: dict[str, Any],
        notes: list[str],
    ) -> str:
        """Combines QA notes into one flag string."""
        issues = list(notes)

        if record["n_lc_cells"] == 0:
            issues.append("no_valid_land_cover")
        elif record["n_lc_cells"] < self.config.min_cells_warn:
            issues.append("few_cells")

        water = record["frac_water"]

        if np.isfinite(water) and water >= self.config.water_fraction_warn:
            issues.append("mostly_water")

        if (
            record["cov_type"] in (COV_DECIDUOUS, COV_CONIFER)
            and np.isfinite(record["frac_mixed"])
            and record["frac_mixed"] > 0.25
        ):
            issues.append("mixed_forest_assigned_by_config")

        return ";".join(dict.fromkeys(issues)) or "ok"

    def _write_summary(
        self,
        table: pd.DataFrame,
        rasters: dict[str, Path],
    ) -> None:
        """Prints a QA summary and optionally writes it as JSON."""
        summary = {
            "year": self.config.year,
            "rasters": {k: v.name for k, v in rasters.items()},
            "n_hrus": int(len(table)),
            "cov_type_counts": {
                str(k): int(v)
                for k, v in table["cov_type"].value_counts().items()
            },
            "qa_flag_counts": {
                str(k): int(v)
                for k, v in table["qa_flag"].value_counts().items()
            },
            "hru_percent_imperv_mean": float(
                table["hru_percent_imperv"].mean()
            ),
            "class_to_cov_type": {
                str(k): v for k, v in self.config.class_to_cov_type.items()
            },
            "vegetation_defaults_USED_REVIEW_REQUIRED": {
                str(k): v for k, v in self.config.vegetation_defaults.items()
            },
        }

        tqdm.write(
            f"Land cover: {summary['n_hrus']:,} HRUs. cov_type counts: "
            f"{summary['cov_type_counts']}. QA flags: "
            f"{summary['qa_flag_counts']}"
        )
        tqdm.write(
            "Review vegetation_defaults before using these values in PRMS."
        )

        if self.config.output_dir is None:
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        (
            self.config.output_dir / "hru_land_cover_stats_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
