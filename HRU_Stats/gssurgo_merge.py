"""Merge several gSSURGO state geodatabases for the soils module.

Run this once before ``soils_stats`` when the HRUs cover more than one state
(for example Montana and Wyoming). It reads each state geodatabase and writes
a small merged set of inputs, clipped to the HRU extent:

    mukey_10m.tif          mosaic of the MapunitRaster_10m rasters
    valu1.parquet          Valu1 rows for the map units in the mosaic
    component.parquet      component rows for those map units
    chorizon.parquet       horizon rows for those components
    gssurgo_merge_manifest.json   sources, counts, overlaps, conflicts

Why merge: gSSURGO is distributed one state per geodatabase, but the soils
module reads one raster and one set of tables.

Rules:

    * Databases are listed in priority order. Where two rasters both have a
      map unit in the same cell, the first database wins. Cells that differ
      between databases are counted and reported.
    * Tables are concatenated and de-duplicated by key (mukey for Valu1,
      cokey for component, chkey for chorizon); the first database's row
      wins. If the same key has different values in two databases (for
      example two download dates), the conflicts are counted and, with
      ``strict_conflicts=True``, raise an error.
    * The mosaic grid is snapped to the first raster's cell grid and covers
      the HRU extent plus ``buffer_m``. Other rasters are matched to it with
      nearest-neighbor resampling, which is exact when the grids are aligned.
    * Map unit keys are assumed to be national (the same key means the same
      map unit in every state database). The conflict check tests that
      assumption; I have not verified it against NRCS documentation.

Needs GDAL 3.7+ for rasters inside a file geodatabase, ``fiona`` 1.9+ (for
``include_fields``), and ``pyarrow`` for parquet.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import fiona
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import list_layers, load_hrus  # noqa: E402

OUTPUT_RASTER = "mukey_10m.tif"
OUTPUT_MANIFEST = "gssurgo_merge_manifest.json"

VALU1_FIELDS = (
    "rootznaws",
    "aws025wta",
    "aws050wta",
    "aws0100wta",
    "aws0150wta",
)

COMPONENT_COLUMNS = ("mukey", "cokey", "comppct_r")

CHORIZON_COLUMNS = (
    "chkey",
    "cokey",
    "hzdept_r",
    "hzdepb_r",
    "sandtotal_r",
    "claytotal_r",
)

NODATA = 0


# =============================================================================
# Parquet helpers
# =============================================================================


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Writes a parquet file, with a clear message if pyarrow is missing."""
    try:
        frame.to_parquet(path, index=False)
    except ImportError as error:
        raise ImportError(
            "Writing parquet needs pyarrow. Install it with "
            "'conda install pyarrow'."
        ) from error


def read_parquet(path: Path) -> pd.DataFrame:
    """Reads a parquet file, with a clear message if pyarrow is missing."""
    try:
        return pd.read_parquet(path)
    except ImportError as error:
        raise ImportError(
            "Reading parquet needs pyarrow. Install it with "
            "'conda install pyarrow'."
        ) from error


# =============================================================================
# Pure calculations
# =============================================================================


def snap_grid(
    origin_x: float,
    origin_y: float,
    cell_x: float,
    cell_y: float,
    bounds: Sequence[float],
    buffer_m: float,
) -> tuple[float, float, int, int]:
    """Builds a window on a source raster's cell grid around some bounds.

    Args:
        origin_x: X of the source raster's upper-left corner.
        origin_y: Y of the source raster's upper-left corner.
        cell_x: Cell width (positive).
        cell_y: Cell height (positive).
        bounds: ``(minx, miny, maxx, maxy)`` in the raster CRS.
        buffer_m: Margin added on every side.

    Returns:
        ``(x_left, y_top, width, height)``: the upper-left corner of the
        window (an exact multiple of the cell size from the source origin)
        and its size in cells.
    """
    min_x, min_y, max_x, max_y = bounds

    col_start = int(np.floor((min_x - buffer_m - origin_x) / cell_x))
    col_stop = int(np.ceil((max_x + buffer_m - origin_x) / cell_x))
    row_start = int(np.floor((origin_y - (max_y + buffer_m)) / cell_y))
    row_stop = int(np.ceil((origin_y - (min_y - buffer_m)) / cell_y))

    return (
        origin_x + col_start * cell_x,
        origin_y - row_start * cell_y,
        col_stop - col_start,
        row_stop - row_start,
    )


def combine_first(
    blocks: Sequence[np.ndarray],
    nodata: int = NODATA,
) -> tuple[np.ndarray, list[int], int, int]:
    """Combines raster blocks so the earliest block with data wins.

    Args:
        blocks: Same-shape arrays in priority order.
        nodata: Value meaning "no map unit".

    Returns:
        The combined block, the number of cells each block supplied, the
        number of cells where a later block also had data, and how many of
        those overlapping cells held a different value.
    """
    combined = np.full(blocks[0].shape, nodata, dtype=blocks[0].dtype)
    supplied: list[int] = []
    overlap = 0
    different = 0

    for block in blocks:
        has_data = block != nodata
        fills = has_data & (combined == nodata)
        both = has_data & (combined != nodata)

        overlap += int(both.sum())
        different += int((both & (combined != block)).sum())

        combined[fills] = block[fills]
        supplied.append(int(fills.sum()))

    return combined, supplied, overlap, different


def merge_tables(
    frames: Sequence[pd.DataFrame],
    key: Sequence[str],
    labels: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Concatenates tables and de-duplicates by key, first table winning.

    Args:
        frames: One table per database, in priority order.
        key: Column(s) that identify a row.
        labels: One label per table, for the report.

    Returns:
        The merged table and a report with rows per source, duplicate keys,
        and keys whose values differ between sources.
    """
    key = list(key)

    combined = pd.concat(
        [f.assign(_source=label) for f, label in zip(frames, labels)],
        ignore_index=True,
    )

    value_columns = [
        c for c in combined.columns if c not in key and c != "_source"
    ]

    duplicate_rows = combined.duplicated(key, keep=False)
    duplicate_keys = 0
    conflicting_keys = 0

    if duplicate_rows.any():
        duplicates = combined[duplicate_rows]
        duplicate_keys = int(duplicates.drop_duplicates(key).shape[0])

        if value_columns:
            distinct = duplicates.groupby(key)[value_columns].nunique(
                dropna=False
            )
            conflicting_keys = int((distinct > 1).any(axis=1).sum())

    merged = (
        combined.drop_duplicates(key, keep="first")
        .drop(columns="_source")
        .reset_index(drop=True)
    )

    report = {
        "rows_per_source": {
            label: int(len(f)) for f, label in zip(frames, labels)
        },
        "rows_merged": int(len(merged)),
        "duplicate_keys": duplicate_keys,
        "conflicting_keys": conflicting_keys,
    }

    return merged, report


# =============================================================================
# Geodatabase table reading
# =============================================================================


def clean_id(value: Any) -> str:
    """Returns an ID as a stripped string."""
    return str(value).strip()


def read_gdb_table(
    gdb_path: Path,
    wanted: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
    filter_column: str | None = None,
    allowed: set[str] | None = None,
) -> pd.DataFrame:
    """Reads selected columns of a geodatabase table, filtering as it goes.

    Only the requested fields are loaded (``include_fields``) and rows are
    dropped while reading, so large tables such as chorizon stay small in
    memory.

    Args:
        gdb_path: File geodatabase.
        wanted: Table name (case-insensitive).
        required: Columns that must exist.
        optional: Columns included only when present.
        filter_column: Column to filter on (must be in ``required``).
        allowed: Keep only rows whose ``filter_column`` is in this set.

    Raises:
        ValueError: If the table does not exist.
        KeyError: If a required column is missing.
    """
    names = {name.lower(): name for name in list_layers(gdb_path)}

    if wanted.lower() not in names:
        raise ValueError(
            f"'{wanted}' not found in {gdb_path.name}. Available: "
            f"{sorted(names.values())}"
        )

    layer = names[wanted.lower()]

    with fiona.open(str(gdb_path), layer=layer) as probe:
        actual = {n.lower(): n for n in probe.schema["properties"]}

    missing = [c for c in required if c not in actual]

    if missing:
        raise KeyError(
            f"Table '{layer}' in {gdb_path.name} is missing {missing}. "
            f"Found: {sorted(actual)}"
        )

    columns = list(required) + [c for c in optional if c in actual]
    id_columns = {"mukey", "cokey", "chkey"}

    rows: list[list[Any]] = []

    with fiona.open(
        str(gdb_path),
        layer=layer,
        include_fields=[actual[c] for c in columns],
        ignore_geometry=True,
    ) as source:
        for feature in tqdm(
            source,
            total=len(source),
            desc=f"{gdb_path.stem}:{layer}",
            unit="row",
        ):
            props = feature["properties"]
            values = [props[actual[c]] for c in columns]

            if filter_column is not None and allowed is not None:
                value = clean_id(values[columns.index(filter_column)])

                if value not in allowed:
                    continue

            rows.append(values)

    frame = pd.DataFrame(rows, columns=columns)

    for column in columns:
        if column in id_columns:
            frame[column] = frame[column].map(clean_id)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


# =============================================================================
# Merge
# =============================================================================


class GssurgoMergeConfig:
    """Stores gSSURGO merge settings.

    Args:
        gdb_paths: gSSURGO file geodatabases in priority order.
        hru_gpkg_path: Datasource containing the HRU polygon layer.
        output_dir: Folder for the merged raster, tables, and manifest.
        hru_layer: HRU polygon layer name.
        mukey_raster_layer: Raster name inside each geodatabase.
        buffer_m: Margin added around the HRU extent.
        block_rows: Raster rows processed at a time.
        strict_conflicts: Raise if the same key has different values in two
            databases (otherwise warn and keep the first).
    """

    def __init__(
        self,
        gdb_paths: Sequence[Path],
        hru_gpkg_path: Path,
        output_dir: Path,
        hru_layer: str = "model_hrus_orphan_adopted",
        mukey_raster_layer: str = "MapunitRaster_10m",
        buffer_m: float = 500.0,
        block_rows: int = 512,
        strict_conflicts: bool = False,
    ) -> None:
        """Initializes and validates settings."""
        paths = [Path(p) for p in gdb_paths]

        if not paths:
            raise ValueError("Give at least one gSSURGO geodatabase.")

        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("The same geodatabase is listed twice.")

        if buffer_m < 0 or block_rows < 1:
            raise ValueError("buffer_m must be >= 0 and block_rows >= 1.")

        self.gdb_paths = paths
        self.hru_gpkg_path = Path(hru_gpkg_path)
        self.output_dir = Path(output_dir)
        self.hru_layer = str(hru_layer)
        self.mukey_raster_layer = str(mukey_raster_layer)
        self.buffer_m = float(buffer_m)
        self.block_rows = int(block_rows)
        self.strict_conflicts = bool(strict_conflicts)


class GssurgoMerger:
    """Merges state gSSURGO databases into one clipped input set."""

    def __init__(self, config: GssurgoMergeConfig) -> None:
        """Initializes the merger."""
        self.config = config

    def run(self) -> dict[str, Any]:
        """Builds the merged raster and tables and writes the manifest.

        Returns:
            The manifest dictionary.

        Raises:
            FileNotFoundError: If a geodatabase is missing.
            RuntimeError: If a raster cannot be opened.
            ValueError: If no map units fall in the HRU extent, or
                conflicts are found with ``strict_conflicts``.
        """
        config = self.config

        for path in config.gdb_paths:
            if not path.exists():
                raise FileNotFoundError(f"Geodatabase missing: {path}")

        config.output_dir.mkdir(parents=True, exist_ok=True)

        hrus = load_hrus(config.hru_gpkg_path, config.hru_layer)

        tqdm.write("Mosaicking map unit rasters.")
        mukeys, raster_report = self.build_mosaic(hrus)

        tqdm.write(f"  {len(mukeys):,} distinct map units in the mosaic.")

        labels = [p.name for p in config.gdb_paths]
        warnings: list[str] = []

        valu1_frames = [
            read_gdb_table(
                p,
                "valu1",
                required=("mukey",),
                optional=VALU1_FIELDS,
                filter_column="mukey",
                allowed=mukeys,
            )
            for p in config.gdb_paths
        ]
        valu1, valu1_report = merge_tables(valu1_frames, ["mukey"], labels)

        present = [c for c in VALU1_FIELDS if c in valu1.columns]

        if not present:
            raise KeyError(
                f"Valu1 has none of the storage fields {VALU1_FIELDS}."
            )

        component_frames = [
            read_gdb_table(
                p,
                "component",
                required=COMPONENT_COLUMNS,
                filter_column="mukey",
                allowed=mukeys,
            )
            for p in config.gdb_paths
        ]
        component, component_report = merge_tables(
            component_frames, ["cokey"], labels
        )

        cokeys = set(component["cokey"])

        chorizon_frames = [
            read_gdb_table(
                p,
                "chorizon",
                required=CHORIZON_COLUMNS,
                filter_column="cokey",
                allowed=cokeys,
            )
            for p in config.gdb_paths
        ]
        chorizon, chorizon_report = merge_tables(
            chorizon_frames, ["chkey"], labels
        )

        missing_valu1 = sorted(mukeys - set(valu1["mukey"]))

        if missing_valu1:
            warnings.append(
                f"{len(missing_valu1):,} map unit(s) in the raster have no "
                f"Valu1 row, e.g. {missing_valu1[:5]}."
            )

        reports = {
            "valu1": valu1_report,
            "component": component_report,
            "chorizon": chorizon_report,
        }

        conflicts = {
            name: r["conflicting_keys"]
            for name, r in reports.items()
            if r["conflicting_keys"]
        }

        if conflicts:
            message = (
                "The same key has different values in different "
                f"databases: {conflicts}. The first database's rows were "
                "kept. Check the databases are the same SSURGO release."
            )

            if config.strict_conflicts:
                raise ValueError(message)

            warnings.append(message)

        if raster_report["overlap_cells_different"]:
            warnings.append(
                f"{raster_report['overlap_cells_different']:,} overlapping "
                "raster cell(s) hold different map units in different "
                "databases; the first database was used."
            )

        write_parquet(valu1, config.output_dir / "valu1.parquet")
        write_parquet(component, config.output_dir / "component.parquet")
        write_parquet(chorizon, config.output_dir / "chorizon.parquet")

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "databases_in_priority_order": labels,
            "raster": raster_report,
            "n_mukeys": int(len(mukeys)),
            "tables": reports,
            "valu1_storage_fields": present,
            "warnings": warnings,
        }

        (config.output_dir / OUTPUT_MANIFEST).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        for warning in warnings:
            tqdm.write(f"WARNING: {warning}")

        tqdm.write(
            f"Merged {len(labels)} database(s): {len(mukeys):,} map units, "
            f"{len(component):,} components, {len(chorizon):,} horizons."
        )

        return manifest

    def build_mosaic(
        self,
        hrus: Any,
    ) -> tuple[set[str], dict[str, Any]]:
        """Writes the mosaic GeoTIFF and returns its map unit keys.

        Returns:
            Set of map unit keys (strings) present in the mosaic and a
            report dictionary.
        """
        config = self.config
        output_path = config.output_dir / OUTPUT_RASTER
        temp_path = config.output_dir / "mukey_10m.tmp.tif"

        uris = [
            f'OpenFileGDB:"{p}":{config.mukey_raster_layer}'
            for p in config.gdb_paths
        ]

        with ExitStack() as stack:
            sources = []

            for path, uri in zip(config.gdb_paths, uris):
                try:
                    sources.append(stack.enter_context(rasterio.open(uri)))
                except Exception as error:
                    raise RuntimeError(
                        f"Could not open raster "
                        f"'{config.mukey_raster_layer}' in {path.name} "
                        f"(GDAL {rasterio.__gdal_version__}): {error}"
                    ) from error

            first = sources[0]

            if first.crs is None or not first.crs.is_projected:
                raise ValueError("The first raster needs a projected CRS.")

            cell_x = float(abs(first.transform.a))
            cell_y = float(abs(first.transform.e))

            different_cells = [
                path.name
                for path, src in zip(config.gdb_paths, sources)
                if (abs(src.transform.a), abs(src.transform.e))
                != (cell_x, cell_y)
            ]

            if different_cells:
                tqdm.write(
                    f"Cell size differs in {different_cells}; those "
                    "rasters will be resampled (nearest) to the first."
                )

            bounds = tuple(hrus.to_crs(first.crs).total_bounds)

            x_left, y_top, width, height = snap_grid(
                first.transform.c,
                first.transform.f,
                cell_x,
                cell_y,
                bounds,
                config.buffer_m,
            )

            transform = Affine(cell_x, 0.0, x_left, 0.0, -cell_y, y_top)

            vrts = [
                stack.enter_context(
                    WarpedVRT(
                        src,
                        crs=first.crs,
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=Resampling.nearest,
                        src_nodata=(
                            src.nodata if src.nodata is not None else NODATA
                        ),
                        nodata=NODATA,
                        dtype="int32",
                    )
                )
                for src in sources
            ]

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "int32",
                "crs": first.crs,
                "transform": transform,
                "nodata": NODATA,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }

            mukeys: set[str] = set()
            supplied = [0] * len(vrts)
            overlap = 0
            different = 0

            with rasterio.open(temp_path, "w", **profile) as destination:
                for row in tqdm(
                    range(0, height, config.block_rows),
                    desc="Mosaic",
                    unit="block",
                ):
                    window = Window(
                        0,
                        row,
                        width,
                        min(config.block_rows, height - row),
                    )

                    blocks = [
                        vrt.read(1, window=window).astype("int32")
                        for vrt in vrts
                    ]

                    combined, counts, n_over, n_diff = combine_first(blocks)

                    destination.write(combined, 1, window=window)

                    supplied = [a + b for a, b in zip(supplied, counts)]
                    overlap += n_over
                    different += n_diff

                    mukeys.update(
                        str(int(v))
                        for v in np.unique(combined[combined != NODATA])
                    )

        total_cells = int(width) * int(height)
        filled = int(sum(supplied))

        if filled == 0:
            raise ValueError(
                "No map unit cells fall inside the HRU extent. Check the "
                "databases cover your domain."
            )

        os.replace(temp_path, output_path)

        report = {
            "path": output_path.name,
            "crs": str(first.crs),
            "cell_size": [cell_x, cell_y],
            "width": int(width),
            "height": int(height),
            "buffer_m": config.buffer_m,
            "cells_total": total_cells,
            "cells_with_map_unit": filled,
            "fraction_with_map_unit": round(filled / total_cells, 4),
            "cells_supplied_by_database": {
                path.name: int(count)
                for path, count in zip(config.gdb_paths, supplied)
            },
            "overlap_cells": int(overlap),
            "overlap_cells_different": int(different),
        }

        return mukeys, report


# =============================================================================
# Reading the merged set
# =============================================================================


class MergedGssurgo:
    """Reads the merged raster and tables written by ``GssurgoMerger``.

    Args:
        merged_dir: The merger's ``output_dir``.
    """

    def __init__(self, merged_dir: Path) -> None:
        """Checks that the merged files exist."""
        self.directory = Path(merged_dir)
        self.raster_path = self.directory / OUTPUT_RASTER

        needed = [
            self.raster_path,
            self.directory / "valu1.parquet",
            self.directory / "component.parquet",
            self.directory / "chorizon.parquet",
        ]

        missing = [p.name for p in needed if not p.exists()]

        if missing:
            raise FileNotFoundError(
                f"Merged gSSURGO files missing in {self.directory}: "
                f"{missing}. Run GssurgoMerger first."
            )

    def read_tables(
        self,
        mukeys: set[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns Valu1, component, and chorizon, optionally filtered.

        Args:
            mukeys: If given, keep only these map units (and their
                components and horizons).
        """
        valu1 = read_parquet(self.directory / "valu1.parquet")
        component = read_parquet(self.directory / "component.parquet")
        chorizon = read_parquet(self.directory / "chorizon.parquet")

        if mukeys is not None:
            valu1 = valu1[valu1["mukey"].isin(mukeys)]
            component = component[component["mukey"].isin(mukeys)]
            chorizon = chorizon[chorizon["cokey"].isin(set(component["cokey"]))]

        return (
            valu1.reset_index(drop=True),
            component.reset_index(drop=True),
            chorizon.reset_index(drop=True),
        )