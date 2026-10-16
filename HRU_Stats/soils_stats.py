"""Calculate PRMS soil-zone parameters for every HRU from gSSURGO.

Appends one row per HRU to the HRU stats parquet (through
``utils.StatsAppender``), keyed by ``catchment_id``.

PRMS parameters appended:

    soil_moist_max       maximum plant-available water in the soil zone (in)
    soil_rechr_max_frac  recharge-zone storage as a fraction of
                         soil_moist_max (0-1)
    soil_type            1 sand, 2 loam, 3 clay

Check the parameter names against the PRMS / pywatershed version you run.
PRMS-IV used ``soil_rechr_max`` (inches) where PRMS 5 uses
``soil_rechr_max_frac``.

Data: the gSSURGO file geodatabase (statewide). Layers used:

    MapunitRaster_10m   10 m raster whose cell values are map unit keys
                        (read through GDAL's OpenFileGDB raster driver,
                        GDAL 3.7+, or from an exported GeoTIFF)
    Valu1               pre-summarized map unit attributes, in millimeters:
                        rootznaws (available water storage in the root zone)
                        and aws025wta / aws050wta / aws0100wta / aws0150wta
                        (available water storage in the top 25 / 50 / 100 /
                        150 cm)
    component, chorizon horizon sand and clay percent, for soil_type

This gSSURGO download has no MUPOLYGON polygon layer, so map units come from
the raster.

Method:

    1. The map unit raster cells inside each HRU are counted by map unit
       key; the cell count is the weight of each map unit.
    2. soil_moist_max is the weighted mean of the chosen Valu1 storage
       field (default ``rootznaws``), converted from millimeters to inches.
       rootznaws uses the soil's own root-restricting depth, or 150 cm if
       none, so it does not depend on vegetation type.
    3. soil_rechr_max_frac is the storage in the top ``recharge_field``
       depth (default top 25 cm) divided by soil_moist_max, limited to 0-1.
    4. soil_type comes from weighted mean sand and clay percent over the top
       ``texture_depth_cm`` of soil: clay at or above
       ``clay_threshold_pct`` is 3 (clay); sand at or above
       ``sand_threshold_pct`` is 1 (sand); everything else is 2 (loam).

Not computed here: sat_threshold, ssr2gw_rate, carea_max, and the other
calibration-driven soil-zone and groundwater parameters.

IMPORTANT: the soil_type thresholds, the 0.01 inch floor on
soil_moist_max, and the domain-median fill for HRUs with no soil data are my
starting choices, not values from a PRMS or National Hydrologic Model table.
They are saved in the parquet manifest with this group.

QA columns (prefixed ``soils_`` in the parquet): covered_fraction (share of
HRU cells that have a map unit), valid_area_fraction (share whose map unit
has storage data), aws_mm, recharge_mm, sand_pct, clay_pct, n_mukeys,
qa_flag.

Optional output: if ``output_dir`` is set, a ``hru_soils_stats_summary.json``
QA summary is written there. Nothing else is written to disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (  # noqa: E402
    StatsAppender,
    check_hru_keys,
    list_layers,
    load_hrus,
    read_hru_values,
)

GROUP_NAME = "soils"

PARAMETER_COLUMNS = (
    "soil_moist_max",
    "soil_rechr_max_frac",
    "soil_type",
)

MM_PER_INCH = 25.4

SOIL_SAND = 1
SOIL_LOAM = 2
SOIL_CLAY = 3

STORAGE_FIELDS = (
    "rootznaws",
    "aws025wta",
    "aws050wta",
    "aws0100wta",
    "aws0150wta",
)


class SoilsConfig:
    """Stores soil-statistics settings.

    Args:
        hru_gpkg_path: Datasource containing the HRU polygon layer.
        parquet_path: HRU stats parquet created by ``HruParquetBuilder``.
        gssurgo_gdb_path: gSSURGO file geodatabase (for example
            ``gSSURGO_MT.gdb``).
        hru_layer: HRU polygon layer name.
        mukey_raster_layer: Map unit raster inside the geodatabase (cell
            values are map unit keys).
        mukey_raster_path: Optional GeoTIFF of the map unit raster. Use it
            if your GDAL cannot read geodatabase rasters (needs 3.7+).
        soil_moist_max_field: Valu1 field (millimeters) for the total soil
            water storage. ``rootznaws`` or one of the fixed-depth fields.
        recharge_field: Valu1 field (millimeters) for the recharge zone.
        texture_depth_cm: Depth over which sand and clay are averaged.
        clay_threshold_pct: Clay percent at or above which soil_type is 3.
        sand_threshold_pct: Sand percent at or above which soil_type is 1.
        min_soil_moist_max_in: Floor applied to soil_moist_max (inches).
        low_coverage_fraction: HRUs with less valid soil area than this
            fraction are flagged.
        fill_missing: Fill HRUs with no soil data using the domain median
            (mode for soil_type) and flag them. If False, such HRUs raise
            an error.
        output_dir: Optional folder for the QA summary JSON.
    """

    def __init__(
        self,
        hru_gpkg_path: Path,
        parquet_path: Path,
        gssurgo_gdb_path: Path,
        hru_layer: str = "model_hrus_orphan_adopted",
        mukey_raster_layer: str = "MapunitRaster_10m",
        mukey_raster_path: Path | None = None,
        soil_moist_max_field: str = "rootznaws",
        recharge_field: str = "aws025wta",
        texture_depth_cm: float = 50.0,
        clay_threshold_pct: float = 35.0,
        sand_threshold_pct: float = 70.0,
        min_soil_moist_max_in: float = 0.01,
        low_coverage_fraction: float = 0.5,
        fill_missing: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        """Initializes and validates settings."""
        for name, value in (
            ("soil_moist_max_field", soil_moist_max_field),
            ("recharge_field", recharge_field),
        ):
            if value not in STORAGE_FIELDS:
                raise ValueError(
                    f"{name} must be one of {STORAGE_FIELDS}, got "
                    f"'{value}'."
                )

        if texture_depth_cm <= 0:
            raise ValueError("texture_depth_cm must be positive.")

        self.hru_gpkg_path = Path(hru_gpkg_path)
        self.parquet_path = Path(parquet_path)
        self.gssurgo_gdb_path = Path(gssurgo_gdb_path)
        self.hru_layer = str(hru_layer)
        self.mukey_raster_layer = str(mukey_raster_layer)
        self.mukey_raster_path = (
            Path(mukey_raster_path) if mukey_raster_path is not None else None
        )
        self.soil_moist_max_field = soil_moist_max_field
        self.recharge_field = recharge_field
        self.texture_depth_cm = float(texture_depth_cm)
        self.clay_threshold_pct = float(clay_threshold_pct)
        self.sand_threshold_pct = float(sand_threshold_pct)
        self.min_soil_moist_max_in = float(min_soil_moist_max_in)
        self.low_coverage_fraction = float(low_coverage_fraction)
        self.fill_missing = bool(fill_missing)
        self.output_dir = (
            Path(output_dir) if output_dir is not None else None
        )


# =============================================================================
# gSSURGO reading
# =============================================================================


def resolve_layer(gdb_path: Path, wanted: str) -> str:
    """Finds a geodatabase table by case-insensitive name."""
    names = {name.lower(): name for name in list_layers(gdb_path)}

    if wanted.lower() not in names:
        raise ValueError(
            f"'{wanted}' not found in {gdb_path.name}. Available: "
            f"{sorted(names.values())}"
        )

    return names[wanted.lower()]


def open_mukey_raster(
    gdb_path: Path,
    raster_layer: str,
    raster_path: Path | None,
) -> Any:
    """Opens the map unit key raster.

    Uses ``raster_path`` (for example a GeoTIFF exported from the
    geodatabase) when given. Otherwise reads the raster layer straight from
    the file geodatabase, which needs GDAL 3.7 or newer.

    Raises:
        RuntimeError: If the raster cannot be opened, with the GDAL version
            and any raster layers that were found.
    """
    if raster_path is not None:
        return rasterio.open(raster_path)

    uri = f'OpenFileGDB:"{gdb_path}":{raster_layer}'

    try:
        return rasterio.open(uri)
    except Exception as error:
        found = "none found"

        try:
            with rasterio.open(str(gdb_path)) as probe:
                found = str(list(probe.subdatasets)) or "none found"
        except Exception:
            pass

        raise RuntimeError(
            f"Could not open raster '{raster_layer}' in {gdb_path.name}. "
            f"Rasterio is using GDAL {rasterio.__gdal_version__}; reading "
            "geodatabase rasters needs GDAL 3.7 or newer. Raster layers "
            f"found: {found}. If your GDAL is older, export the raster to a "
            "GeoTIFF (ArcGIS Copy Raster) and pass it as mukey_raster_path."
        ) from error


def read_table(
    gdb_path: Path,
    wanted: str,
    columns: tuple[str, ...],
    key_filter: tuple[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Reads a non-spatial geodatabase table with lowercase column names.

    Args:
        gdb_path: File geodatabase.
        wanted: Table name (case-insensitive).
        columns: Required columns, all returned.
        key_filter: Optional ``(column, allowed_values)``; rows whose
            string-cleaned value is not allowed are dropped.

    Raises:
        KeyError: If a required column is missing.
    """
    name = resolve_layer(gdb_path, wanted)

    frame = gpd.read_file(gdb_path, layer=name, engine="fiona")
    frame = pd.DataFrame(frame.drop(columns="geometry", errors="ignore"))
    frame.columns = [str(c).lower() for c in frame.columns]

    missing = set(columns) - set(frame.columns)

    if missing:
        raise KeyError(
            f"Table '{name}' is missing columns {sorted(missing)}. "
            f"Found: {sorted(frame.columns)}"
        )

    for column in ("mukey", "cokey"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()

    if key_filter is not None:
        column, allowed = key_filter
        frame = frame[frame[column].isin(allowed)]

    return frame[list(columns)].copy()


# =============================================================================
# Pure calculations
# =============================================================================


def texture_by_mukey(
    component: pd.DataFrame,
    chorizon: pd.DataFrame,
    depth_cm: float,
) -> pd.DataFrame:
    """Calculates map-unit mean sand and clay percent over the top soil.

    Horizon values are thickness-weighted over 0 to ``depth_cm`` within each
    component. Components are then weighted by ``comppct_r``. Components or
    horizons with no sand or clay value are ignored.

    Args:
        component: Columns mukey, cokey, comppct_r.
        chorizon: Columns cokey, hzdept_r, hzdepb_r, sandtotal_r,
            claytotal_r.
        depth_cm: Depth of soil to average.

    Returns:
        Frame indexed by mukey with sand_pct and clay_pct.
    """
    horizons = chorizon.dropna(
        subset=["sandtotal_r", "claytotal_r", "hzdept_r", "hzdepb_r"]
    ).copy()

    top = horizons["hzdept_r"].clip(lower=0.0)
    bottom = horizons["hzdepb_r"].clip(upper=depth_cm)
    horizons["thickness"] = (bottom - top).clip(lower=0.0)
    horizons = horizons[horizons["thickness"] > 0].copy()

    horizons["sand_w"] = horizons["sandtotal_r"] * horizons["thickness"]
    horizons["clay_w"] = horizons["claytotal_r"] * horizons["thickness"]

    by_component = horizons.groupby("cokey").agg(
        sand_w=("sand_w", "sum"),
        clay_w=("clay_w", "sum"),
        thickness=("thickness", "sum"),
    )

    by_component["sand_pct"] = (
        by_component["sand_w"] / by_component["thickness"]
    )
    by_component["clay_pct"] = (
        by_component["clay_w"] / by_component["thickness"]
    )

    joined = component.merge(
        by_component[["sand_pct", "clay_pct"]],
        left_on="cokey",
        right_index=True,
        how="inner",
    )

    joined = joined[joined["comppct_r"] > 0].copy()
    joined["sand_w"] = joined["sand_pct"] * joined["comppct_r"]
    joined["clay_w"] = joined["clay_pct"] * joined["comppct_r"]

    by_mukey = joined.groupby("mukey").agg(
        sand_w=("sand_w", "sum"),
        clay_w=("clay_w", "sum"),
        weight=("comppct_r", "sum"),
    )

    return pd.DataFrame(
        {
            "sand_pct": by_mukey["sand_w"] / by_mukey["weight"],
            "clay_pct": by_mukey["clay_w"] / by_mukey["weight"],
        }
    )


def classify_soil_type(
    sand_pct: float,
    clay_pct: float,
    sand_threshold_pct: float,
    clay_threshold_pct: float,
) -> float:
    """Maps mean sand and clay percent to PRMS soil_type 1, 2, or 3.

    Returns NaN if either value is missing.
    """
    if not (np.isfinite(sand_pct) and np.isfinite(clay_pct)):
        return float("nan")

    if clay_pct >= clay_threshold_pct:
        return float(SOIL_CLAY)

    if sand_pct >= sand_threshold_pct:
        return float(SOIL_SAND)

    return float(SOIL_LOAM)


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[float, float]:
    """Weighted mean over entries that have a value.

    Args:
        values: Values indexed by map unit key (NaN = no data).
        weights: Weights (cell counts) indexed by map unit key.

    Returns:
        Weighted mean (NaN if no entry has data) and the weight that had
        data.
    """
    aligned = values.reindex(weights.index)
    valid = aligned.notna()
    valid_weight = float(weights[valid].sum())

    if valid_weight <= 0.0:
        return float("nan"), 0.0

    mean = float((aligned[valid] * weights[valid]).sum() / valid_weight)

    return mean, valid_weight


# =============================================================================
# Calculator
# =============================================================================


class HruSoilStats:
    """Computes PRMS soil-zone parameters for every HRU."""

    def __init__(self, config: SoilsConfig) -> None:
        """Initializes the calculator."""
        self.config = config

    def run(self) -> pd.DataFrame:
        """Computes soil parameters and appends them to the parquet.

        Returns:
            The soils table that was appended (before column prefixing).

        Raises:
            FileNotFoundError: If the geodatabase does not exist.
            RuntimeError: If the map unit raster cannot be opened.
            ValueError: If no HRU has soil data, soil data is missing and
                ``fill_missing`` is False, or the HRU layer no longer
                matches the parquet.
        """
        config = self.config
        gdb = config.gssurgo_gdb_path

        if not gdb.exists():
            raise FileNotFoundError(f"gSSURGO geodatabase missing: {gdb}")

        hrus = load_hrus(config.hru_gpkg_path, config.hru_layer)

        check_hru_keys(hrus)

        tqdm.write("Counting gSSURGO map units in each HRU.")

        with open_mukey_raster(
            gdb,
            config.mukey_raster_layer,
            config.mukey_raster_path,
        ) as source:
            mukey_counts = self.count_mukeys(hrus, source)

        mukeys: set[str] = set()

        for counts, _ in mukey_counts:
            mukeys.update(counts.index)

        tqdm.write(f"  {len(mukeys):,} distinct map units in the HRUs.")

        if not mukeys:
            raise ValueError(
                "No map unit cells fall inside the HRUs. Check the "
                "raster covers your domain and uses the same area."
            )

        valu1 = read_table(
            gdb,
            "valu1",
            ("mukey", config.soil_moist_max_field, config.recharge_field),
            key_filter=("mukey", mukeys),
        )

        component = read_table(
            gdb,
            "component",
            ("mukey", "cokey", "comppct_r"),
            key_filter=("mukey", mukeys),
        )
        component["comppct_r"] = pd.to_numeric(
            component["comppct_r"], errors="coerce"
        )

        chorizon = read_table(
            gdb,
            "chorizon",
            (
                "cokey",
                "hzdept_r",
                "hzdepb_r",
                "sandtotal_r",
                "claytotal_r",
            ),
            key_filter=("cokey", set(component["cokey"])),
        )

        for column in ("hzdept_r", "hzdepb_r", "sandtotal_r", "claytotal_r"):
            chorizon[column] = pd.to_numeric(
                chorizon[column], errors="coerce"
            )

        texture = texture_by_mukey(
            component,
            chorizon,
            config.texture_depth_cm,
        )

        table = self.calculate(hrus, mukey_counts, valu1, texture)

        StatsAppender(config.parquet_path, GROUP_NAME).append(
            table=table,
            parameter_columns=PARAMETER_COLUMNS,
            inputs={
                "gssurgo": gdb.name,
                "mukey_raster": (
                    config.mukey_raster_path.name
                    if config.mukey_raster_path is not None
                    else config.mukey_raster_layer
                ),
                "soil_moist_max_field": config.soil_moist_max_field,
                "recharge_field": config.recharge_field,
                "texture_depth_cm": config.texture_depth_cm,
                "clay_threshold_pct": config.clay_threshold_pct,
                "sand_threshold_pct": config.sand_threshold_pct,
                "min_soil_moist_max_in": config.min_soil_moist_max_in,
                "fill_missing": config.fill_missing,
            },
            hrus=hrus,
        )

        self._write_summary(table)

        return table

    def count_mukeys(
        self,
        hrus: gpd.GeoDataFrame,
        source: Any,
    ) -> list[tuple[pd.Series, int]]:
        """Counts raster cells of each map unit key inside every HRU.

        Returns:
            One ``(counts, total_cells)`` pair per HRU, in HRU order.
            ``counts`` is indexed by mukey (string); ``total_cells`` includes
            cells with no map unit.
        """
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("The map unit raster must have a projected CRS.")

        projected = hrus.to_crs(source.crs)

        output: list[tuple[pd.Series, int]] = []

        for geometry in tqdm(
            projected.geometry,
            total=len(projected),
            desc="HRU map units",
            unit="HRU",
        ):
            values, _ = read_hru_values(source, geometry)

            valid = values[np.isfinite(values)].astype(np.int64)

            counts = pd.Series(valid, dtype="int64").value_counts()
            counts.index = counts.index.astype(str)

            output.append((counts, int(values.size)))

        return output

    def calculate(
        self,
        hrus: gpd.GeoDataFrame,
        mukey_counts: list[tuple[pd.Series, int]],
        valu1: pd.DataFrame,
        texture: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates soil parameters for each HRU.

        Args:
            hrus: HRU polygons.
            mukey_counts: Per-HRU ``(counts, total_cells)`` from
                ``count_mukeys``.
            valu1: Valu1 rows (mukey plus the two storage fields, mm).
            texture: Frame indexed by mukey with sand_pct and clay_pct.
        """
        config = self.config

        all_keys = sorted(
            {key for counts, _ in mukey_counts for key in counts.index}
        )

        lookup = pd.DataFrame(index=pd.Index(all_keys, dtype=str))

        values = valu1.drop_duplicates("mukey").set_index("mukey")

        lookup["aws_mm"] = pd.to_numeric(
            values[config.soil_moist_max_field], errors="coerce"
        ).reindex(lookup.index)
        lookup["recharge_mm"] = pd.to_numeric(
            values[config.recharge_field], errors="coerce"
        ).reindex(lookup.index)
        lookup["sand_pct"] = texture["sand_pct"].reindex(lookup.index)
        lookup["clay_pct"] = texture["clay_pct"].reindex(lookup.index)

        observed = lookup["aws_mm"].dropna()

        if len(observed) and float(observed.quantile(0.99)) < 30.0:
            tqdm.write(
                "WARNING: storage values look very small. Valu1 storage "
                "fields are millimeters; check you did not pass a "
                "centimeter field."
            )

        records: list[dict[str, Any]] = []

        for hru_id, catchment_id, (counts, total_cells) in zip(
            hrus["hru_id"],
            hrus["catchment_id"],
            mukey_counts,
        ):
            weights = counts.astype(float)

            record: dict[str, Any] = {
                "hru_id": int(hru_id),
                "catchment_id": str(catchment_id),
                "covered_fraction": 0.0,
                "valid_area_fraction": 0.0,
                "aws_mm": np.nan,
                "recharge_mm": np.nan,
                "sand_pct": np.nan,
                "clay_pct": np.nan,
                "n_mukeys": int(len(weights)),
            }

            if len(weights) and total_cells > 0:
                record["covered_fraction"] = float(
                    min(weights.sum() / total_cells, 1.0)
                )

                aws, aws_weight = weighted_mean(lookup["aws_mm"], weights)
                record["aws_mm"] = aws
                record["valid_area_fraction"] = float(
                    min(aws_weight / total_cells, 1.0)
                )

                for column in ("recharge_mm", "sand_pct", "clay_pct"):
                    record[column], _ = weighted_mean(
                        lookup[column], weights
                    )

            records.append(record)

        return self._finalize(pd.DataFrame(records))

    def _finalize(self, table: pd.DataFrame) -> pd.DataFrame:
        """Derives PRMS parameters, fills gaps, and assigns QA flags."""
        config = self.config
        output = table.copy()

        output["soil_moist_max"] = output["aws_mm"] / MM_PER_INCH

        recharge_fraction = np.where(
            output["aws_mm"] > 0,
            output["recharge_mm"] / output["aws_mm"],
            np.nan,
        )
        output["soil_rechr_max_frac"] = np.clip(recharge_fraction, 0.0, 1.0)

        output["soil_type"] = [
            classify_soil_type(
                sand,
                clay,
                config.sand_threshold_pct,
                config.clay_threshold_pct,
            )
            for sand, clay in zip(output["sand_pct"], output["clay_pct"])
        ]

        floored = output["soil_moist_max"].notna() & (
            output["soil_moist_max"] < config.min_soil_moist_max_in
        )
        output.loc[floored, "soil_moist_max"] = config.min_soil_moist_max_in

        filled_columns: dict[int, list[str]] = {}

        for column in PARAMETER_COLUMNS:
            missing = output[column].isna()

            if not missing.any():
                continue

            if output[column].notna().sum() == 0:
                raise ValueError(
                    f"No HRU has soil data for {column}. Check the "
                    "raster and Valu1 cover the HRU extent."
                )

            if not config.fill_missing:
                raise ValueError(
                    f"{int(missing.sum()):,} HRU(s) have no soil data for "
                    f"{column}, e.g. hru_id "
                    f"{output.loc[missing, 'hru_id'].head(10).tolist()}. "
                    "Set fill_missing=True to use the domain median."
                )

            fill = (
                float(output[column].mode().iloc[0])
                if column == "soil_type"
                else float(output[column].median())
            )

            for index in output.index[missing]:
                filled_columns.setdefault(int(index), []).append(column)

            output.loc[missing, column] = fill

        output["soil_type"] = output["soil_type"].astype(int)

        flags: list[str] = []

        for index, row in output.iterrows():
            issues: list[str] = []

            if row["covered_fraction"] <= 0:
                issues.append("no_map_units")
            elif row["valid_area_fraction"] < config.low_coverage_fraction:
                issues.append("low_soil_coverage")

            if int(index) in filled_columns:
                issues.append(
                    "filled_domain_median:"
                    + "+".join(filled_columns[int(index)])
                )

            if bool(floored.loc[index]):
                issues.append("soil_moist_max_floored")

            flags.append(";".join(issues) or "ok")

        output["qa_flag"] = flags

        columns = [
            "hru_id",
            "catchment_id",
            *PARAMETER_COLUMNS,
            "covered_fraction",
            "valid_area_fraction",
            "aws_mm",
            "recharge_mm",
            "sand_pct",
            "clay_pct",
            "n_mukeys",
            "qa_flag",
        ]

        return output[columns].sort_values("hru_id").reset_index(drop=True)

    def _write_summary(self, table: pd.DataFrame) -> None:
        """Prints a QA summary and optionally writes it as JSON."""
        summary = {
            "n_hrus": int(len(table)),
            "soil_type_counts": {
                str(k): int(v)
                for k, v in table["soil_type"].value_counts().items()
            },
            "soil_moist_max_in_min": float(table["soil_moist_max"].min()),
            "soil_moist_max_in_median": float(
                table["soil_moist_max"].median()
            ),
            "soil_moist_max_in_max": float(table["soil_moist_max"].max()),
            "qa_flag_counts": {
                str(k): int(v)
                for k, v in table["qa_flag"].value_counts().items()
            },
            "settings_REVIEW_REQUIRED": {
                "soil_moist_max_field": self.config.soil_moist_max_field,
                "recharge_field": self.config.recharge_field,
                "texture_depth_cm": self.config.texture_depth_cm,
                "clay_threshold_pct": self.config.clay_threshold_pct,
                "sand_threshold_pct": self.config.sand_threshold_pct,
                "min_soil_moist_max_in": self.config.min_soil_moist_max_in,
            },
        }

        tqdm.write(
            f"Soils: {summary['n_hrus']:,} HRUs. soil_type counts: "
            f"{summary['soil_type_counts']}. soil_moist_max (in) "
            f"median {summary['soil_moist_max_in_median']:.2f}. QA flags: "
            f"{summary['qa_flag_counts']}"
        )

        if self.config.output_dir is None:
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        (
            self.config.output_dir / "hru_soils_stats_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )