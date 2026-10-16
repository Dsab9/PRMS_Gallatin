"""Shared helpers for the HRU_Stats workflow.

Workflow (driven from main.py):

    1. ``HruParquetBuilder`` copies the HRU feature class attributes (no
       geometry) into a parquet file, one row per catchment.
    2. Each stats module gets that parquet path in its config and calls
       ``StatsAppender`` to append its results.
    3. ``HruLayerJoiner`` joins the finished parquet back onto the HRU
       polygons and writes one layer.

Rows are keyed by ``catchment_id`` (with ``source_id`` as a cross-check)
because ``hru_id`` is renumbered whenever the HRU layer is rebuilt.

Column rules when a group appends:
    * PRMS parameter columns (for example ``hru_elev``) keep their names.
    * Every other column (QA flags, cell counts, class fractions) is
      prefixed with the group name, so groups never collide.
    * Rerunning a group replaces only that group's columns.
    * A group may not overwrite a column copied from the HRU layer.

A ``<parquet name>.manifest.json`` beside the parquet records the source
layer, the HRU-set hash, and, for each group, its columns, run time, and
inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Use the conda PROJ database when it exists (Windows conda layout).
PROJ_DIR = Path(sys.prefix) / "Library" / "share" / "proj"

if (PROJ_DIR / "proj.db").exists():
    os.environ["PROJ_LIB"] = str(PROJ_DIR)
    os.environ["PROJ_DATA"] = str(PROJ_DIR)

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from fiona import listlayers, remove
from tqdm import tqdm

if (PROJ_DIR / "proj.db").exists():
    pyproj.datadir.set_data_dir(str(PROJ_DIR))

KEY_COLUMN = "catchment_id"


# =============================================================================
# Layer helpers
# =============================================================================


def lower_columns(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Returns a copy with lowercase non-geometry attribute names."""
    output = gdf.copy()
    geometry_name = output.geometry.name

    output.columns = [
        column
        if column == geometry_name
        else str(column).lower()
        for column in output.columns
    ]

    return output


def list_layers(
    datasource_path: Path,
) -> list[str]:
    """Lists spatial layers in a GeoPackage or file geodatabase."""
    if not datasource_path.exists():
        raise FileNotFoundError(
            f"Datasource does not exist: {datasource_path}"
        )

    return sorted(listlayers(datasource_path))


def read_layer(
    datasource_path: Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Reads and normalizes one datasource layer."""
    available = list_layers(datasource_path)

    if layer_name not in available:
        raise ValueError(
            f"Layer '{layer_name}' is absent from "
            f"{datasource_path.name}. Available layers: {available}"
        )

    return lower_columns(
        gpd.read_file(
            datasource_path,
            layer=layer_name,
            engine="fiona",
        )
    )


def write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
    overwrite: bool = True,
) -> None:
    """Writes one spatial GeoDataFrame with the Fiona GPKG writer."""
    if gdf.empty:
        raise ValueError(
            f"Refusing to write empty layer '{layer_name}'."
        )

    if gdf.crs is None:
        raise ValueError(
            f"Layer '{layer_name}' has no CRS."
        )

    gpkg_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_layers = (
        listlayers(gpkg_path)
        if gpkg_path.exists()
        else []
    )

    if overwrite and layer_name in existing_layers:
        remove(
            gpkg_path,
            layer=layer_name,
            driver="GPKG",
        )

    gdf.to_file(
        gpkg_path,
        layer=layer_name,
        driver="GPKG",
        index=False,
        engine="fiona",
    )


def load_hrus(
    gpkg_path: Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Reads and validates the PRMS HRU polygon layer.

    Requires a unique integer ``hru_id`` and valid, non-empty polygons in a
    projected CRS.

    Raises:
        KeyError: If ``hru_id`` is missing.
        ValueError: If IDs repeat or geometries are null, empty, invalid,
            or in a geographic CRS.
    """
    hrus = read_layer(gpkg_path, layer_name)

    if "hru_id" not in hrus.columns:
        raise KeyError(f"HRU layer '{layer_name}' has no hru_id field.")

    if hrus["hru_id"].duplicated().any():
        raise ValueError("hru_id values must be unique.")

    bad_geometry = hrus.geometry.isna() | hrus.geometry.is_empty

    if bad_geometry.any():
        raise ValueError(
            f"{int(bad_geometry.sum()):,} HRU(s) have null or empty "
            "geometry."
        )

    if (~hrus.geometry.is_valid).any():
        raise ValueError(
            f"{int((~hrus.geometry.is_valid).sum()):,} HRU(s) have "
            "invalid geometry."
        )

    if hrus.crs is None or not hrus.crs.is_projected:
        raise ValueError("HRU layer must be in a projected CRS.")

    hrus["hru_id"] = pd.to_numeric(hrus["hru_id"]).astype(int)

    return hrus


def read_hru_values(
    source: Any,
    geometry: Any,
) -> tuple[np.ndarray, str]:
    """Reads raster values at the cell centers inside one HRU polygon.

    The polygon must already be in the raster CRS. Cells flagged as nodata by
    the raster are returned as NaN. HRUs smaller than one cell center fall
    back to every cell the polygon touches.

    Args:
        source: Open rasterio dataset.
        geometry: HRU polygon in the raster CRS.

    Returns:
        1-D float array of values (NaN where masked) and a flag: ``ok``,
        ``small_hru_all_touched``, ``no_cells``, or ``outside_raster``.
    """
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import transform as window_transform

    min_x, min_y, max_x, max_y = geometry.bounds

    bounds_window = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform=source.transform,
    )

    row_start = max(int(np.floor(bounds_window.row_off)), 0)
    col_start = max(int(np.floor(bounds_window.col_off)), 0)
    row_stop = min(
        int(np.ceil(bounds_window.row_off + bounds_window.height)),
        source.height,
    )
    col_stop = min(
        int(np.ceil(bounds_window.col_off + bounds_window.width)),
        source.width,
    )

    if row_stop <= row_start or col_stop <= col_start:
        return np.array([], dtype=np.float64), "outside_raster"

    window = Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )

    data = np.ma.filled(
        source.read(1, window=window, masked=True).astype(np.float64),
        np.nan,
    )

    transform = window_transform(window, source.transform)
    flag = "ok"

    mask = geometry_mask(
        [geometry],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    if not mask.any():
        mask = geometry_mask(
            [geometry],
            out_shape=data.shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )
        flag = "small_hru_all_touched"

    if not mask.any():
        return np.array([], dtype=np.float64), "no_cells"

    return data[mask], flag


# =============================================================================
# HRU keys and parquet storage
# =============================================================================


def hru_signature(hrus: gpd.GeoDataFrame) -> str:
    """Returns a short hash of the HRU set (catchment IDs and areas).

    The hash changes when HRUs are added, removed, or reshaped, which
    detects statistics built from an older HRU layer. Renumbering ``hru_id``
    alone does not change it.
    """
    parts = sorted(
        f"{cid}:{area:.2f}"
        for cid, area in zip(
            hrus[KEY_COLUMN].astype(str),
            hrus.geometry.area,
        )
    )

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def check_hru_keys(hrus: pd.DataFrame) -> None:
    """Raises unless every HRU has a unique catchment_id and a source_id."""
    for field in (KEY_COLUMN, "source_id"):
        if field not in hrus.columns:
            raise KeyError(
                f"HRU layer needs a '{field}' field to key the stats "
                "parquet."
            )

    catchment = hrus[KEY_COLUMN].astype(str).str.strip()

    if hrus[KEY_COLUMN].isna().any() or (catchment == "").any():
        raise ValueError("Every HRU needs a non-blank catchment_id.")

    if catchment.duplicated().any():
        raise ValueError("catchment_id values must be unique across HRUs.")

    if hrus["source_id"].isna().any():
        raise ValueError("Every HRU needs a source_id.")


class ParquetFile:
    """Reads and writes the stats parquet and its manifest.

    Writes are atomic (temp file, then replace) so an interrupted run never
    leaves a half-written parquet.
    """

    def __init__(self, path: Path) -> None:
        """Stores the parquet path and derives the manifest path."""
        self.path = Path(path)
        self.manifest_path = self.path.with_suffix(".manifest.json")

    def exists(self) -> bool:
        """Returns True if the parquet file exists."""
        return self.path.exists()

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        """Reads a parquet file."""
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise ImportError(
                "Reading parquet needs pyarrow. Install it with "
                "'conda install pyarrow'."
            ) from error

    @staticmethod
    def _write_frame(frame: pd.DataFrame, path: Path) -> None:
        """Writes a parquet file."""
        try:
            frame.to_parquet(path, index=False)
        except ImportError as error:
            raise ImportError(
                "Writing parquet needs pyarrow. Install it with "
                "'conda install pyarrow'."
            ) from error

    def read(self) -> pd.DataFrame:
        """Returns the table (catchment_id is a regular column)."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"Stats parquet does not exist: {self.path}. Run "
                "HruParquetBuilder first."
            )

        return self._read_frame(self.path)

    def read_manifest(self) -> dict[str, Any]:
        """Returns the manifest dictionary."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest missing beside parquet: {self.manifest_path}"
            )

        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write(self, frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
        """Atomically writes the parquet and then the manifest."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temp = self.path.with_name(self.path.name + ".tmp")
        self._write_frame(frame, temp)
        os.replace(temp, self.path)

        self.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


class HruParquetBuilder:
    """Copies the HRU feature class attributes into a parquet file.

    The parquet has one row per HRU (geometry dropped) and every attribute
    column of the layer, with lowercase names. ``catchment_id`` and
    ``source_id`` are stored as strings and ``hru_id`` as an integer.

    Args:
        hru_gpkg_path: Datasource holding the HRU polygon layer.
        hru_layer: HRU polygon layer name.
        parquet_path: Parquet file to create.
    """

    def __init__(
        self,
        hru_gpkg_path: Path,
        hru_layer: str,
        parquet_path: Path,
    ) -> None:
        """Stores the source layer and output path."""
        self.hru_gpkg_path = Path(hru_gpkg_path)
        self.hru_layer = str(hru_layer)
        self.store = ParquetFile(parquet_path)

    def create(self, overwrite: bool = False) -> Path:
        """Builds the parquet from the HRU layer.

        Args:
            overwrite: Replace an existing parquet. Replacing it discards
                every group already appended.

        Returns:
            Path to the parquet file.

        Raises:
            FileExistsError: If the parquet exists and overwrite is False.
            KeyError / ValueError: If the HRU layer fails validation.
        """
        if self.store.exists() and not overwrite:
            raise FileExistsError(
                f"{self.store.path} already exists. Pass overwrite=True "
                "to rebuild it (this drops all appended stats)."
            )

        if self.store.exists():
            try:
                existing = list(self.store.read_manifest()["groups"])
            except (FileNotFoundError, KeyError):
                existing = []

            tqdm.write(
                "Replacing existing stats parquet "
                f"(groups discarded: {existing or 'none'})."
            )

        hrus = load_hrus(self.hru_gpkg_path, self.hru_layer)
        check_hru_keys(hrus)

        table = pd.DataFrame(hrus.drop(columns=hrus.geometry.name))
        table[KEY_COLUMN] = table[KEY_COLUMN].astype(str)
        table["source_id"] = table["source_id"].astype(str)
        table["hru_id"] = table["hru_id"].astype(int)

        table = table.sort_values("hru_id").reset_index(drop=True)

        manifest = {
            "source": {
                "datasource": str(self.hru_gpkg_path),
                "layer": self.hru_layer,
                "crs": str(hrus.crs),
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_hrus": int(len(table)),
            "hru_signature": hru_signature(hrus),
            "base_columns": list(table.columns),
            "groups": {},
        }

        self.store.write(table, manifest)

        tqdm.write(
            f"Created {self.store.path.name}: {len(table):,} HRUs, "
            f"{len(table.columns)} attribute columns."
        )

        return self.store.path


class StatsAppender:
    """Appends one stats group's results to the HRU parquet.

    Stats modules create one of these from the ``parquet_path`` in their
    config and call ``append`` with their results table.

    Args:
        parquet_path: Parquet created by ``HruParquetBuilder``.
        group: Group name (``terrain``, ``land_cover``, ...). Non-parameter
            columns are prefixed with it.
    """

    def __init__(self, parquet_path: Path, group: str) -> None:
        """Stores the parquet path and group name."""
        if not group or not group.replace("_", "").isalnum():
            raise ValueError(
                "group must be a non-empty name of letters, digits, and "
                "underscores."
            )

        self.store = ParquetFile(parquet_path)
        self.group = group

    def append(
        self,
        table: pd.DataFrame,
        parameter_columns: tuple[str, ...],
        inputs: dict[str, str] | None = None,
        hrus: gpd.GeoDataFrame | None = None,
    ) -> list[str]:
        """Writes the group's columns into the parquet.

        Args:
            table: One row per HRU with a ``catchment_id`` column, the PRMS
                parameter columns, and optional QA columns.
            parameter_columns: Columns in ``table`` that are PRMS parameters
                (kept unprefixed).
            inputs: Description of the inputs, stored in the manifest.
            hrus: The HRU layer the group was computed from. If given, its
                hash must match the parquet's, which catches an HRU layer
                that was edited after the parquet was built.

        Returns:
            Final column names this group now owns.

        Raises:
            KeyError: If ``catchment_id`` or a parameter column is missing.
            ValueError: If rows do not match the parquet's catchments, a
                column clashes with another group or a copied HRU
                attribute, or the HRU layer changed.
        """
        frame = self.store.read()
        manifest = self.store.read_manifest()

        if hrus is not None and hru_signature(hrus) != manifest["hru_signature"]:
            raise ValueError(
                "The HRU layer changed after the parquet was built. "
                "Rebuild the parquet with HruParquetBuilder and rerun "
                "the groups."
            )

        if KEY_COLUMN not in table.columns:
            raise KeyError(f"Group table needs a {KEY_COLUMN} column.")

        data = table.copy()
        data[KEY_COLUMN] = data[KEY_COLUMN].astype(str)

        if data[KEY_COLUMN].duplicated().any():
            raise ValueError(f"Group '{self.group}' returned duplicate rows.")

        data = data.drop(columns=["hru_id"], errors="ignore").set_index(
            KEY_COLUMN
        )

        keys = pd.Index(frame[KEY_COLUMN])
        missing = keys.difference(data.index)
        extra = data.index.difference(keys)

        if len(missing) or len(extra):
            raise ValueError(
                f"Group '{self.group}' does not match the parquet's "
                f"catchments: {len(missing):,} missing, {len(extra):,} "
                "unknown."
            )

        absent = set(parameter_columns) - set(data.columns)

        if absent:
            raise KeyError(
                f"Group '{self.group}' did not return parameter columns "
                f"{sorted(absent)}."
            )

        data = data.rename(
            columns={
                column: (
                    column
                    if column in parameter_columns
                    or column.startswith(f"{self.group}_")
                    else f"{self.group}_{column}"
                )
                for column in data.columns
            }
        )

        protected = set(manifest["base_columns"])
        owned_elsewhere = {
            column
            for name, entry in manifest["groups"].items()
            if name != self.group
            for column in entry["columns"]
        }

        clashes = sorted(set(data.columns) & (protected | owned_elsewhere))

        if clashes:
            raise ValueError(
                f"Group '{self.group}' columns clash with copied HRU "
                f"attributes or other groups: {clashes}"
            )

        previous = manifest["groups"].get(self.group, {}).get("columns", [])
        frame = frame.drop(columns=previous, errors="ignore")

        values = data.reindex(keys).reset_index(drop=True)
        values.index = frame.index

        frame = pd.concat([frame, values], axis=1)

        manifest["groups"][self.group] = {
            "columns": list(data.columns),
            "parameter_columns": list(parameter_columns),
            "n_hrus": int(len(data)),
            "appended_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {k: str(v) for k, v in (inputs or {}).items()},
        }

        self.store.write(frame, manifest)

        tqdm.write(
            f"Appended {len(data.columns)} column(s) for group "
            f"'{self.group}' to {self.store.path.name}."
        )

        return list(data.columns)


class HruLayerJoiner:
    """Joins the finished parquet back onto the HRU polygons.

    Args:
        parquet_path: Parquet filled by the stats groups.
        hru_gpkg_path: Datasource holding the HRU polygon layer.
        hru_layer: HRU polygon layer name.
        output_gpkg_path: Datasource for the joined layer. Defaults to
            ``hru_gpkg_path``.
        output_layer: Name of the joined layer to write.
        include_qa: Also join QA and auxiliary columns, not just PRMS
            parameters.
        csv_path: Optional CSV (no geometry) of the joined table.
        strict: Raise on mismatched IDs, a changed HRU layer, or missing
            parameter values (otherwise warn).
    """

    def __init__(
        self,
        parquet_path: Path,
        hru_gpkg_path: Path,
        hru_layer: str,
        output_gpkg_path: Path | None = None,
        output_layer: str = "model_hrus_prms",
        include_qa: bool = False,
        csv_path: Path | None = None,
        strict: bool = True,
    ) -> None:
        """Stores join settings."""
        self.store = ParquetFile(parquet_path)
        self.hru_gpkg_path = Path(hru_gpkg_path)
        self.hru_layer = str(hru_layer)
        self.output_gpkg_path = (
            Path(output_gpkg_path)
            if output_gpkg_path is not None
            else self.hru_gpkg_path
        )
        self.output_layer = str(output_layer)
        self.include_qa = bool(include_qa)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.strict = bool(strict)

    def run(self) -> gpd.GeoDataFrame:
        """Joins and writes the HRU layer, returning the joined table."""
        hrus = load_hrus(self.hru_gpkg_path, self.hru_layer)
        joined = self.join(hrus)

        write_layer(joined, self.output_gpkg_path, self.output_layer)

        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            joined.drop(columns=joined.geometry.name).to_csv(
                self.csv_path,
                index=False,
            )

        tqdm.write(
            f"Wrote '{self.output_layer}': {len(joined):,} HRUs, "
            f"{len(joined.columns) - 1} attribute columns."
        )

        return joined

    def _problem(self, message: str) -> None:
        """Raises in strict mode, otherwise prints a warning."""
        if self.strict:
            raise ValueError(message)

        tqdm.write(f"WARNING: {message}")

    def join(self, hrus: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Joins group columns onto the HRU polygons by catchment_id."""
        check_hru_keys(hrus)

        frame = self.store.read().set_index(KEY_COLUMN)
        manifest = self.store.read_manifest()

        keys = hrus[KEY_COLUMN].astype(str)
        missing = pd.Index(keys).difference(frame.index)

        if len(missing):
            raise ValueError(
                f"{len(missing):,} HRU catchment(s) are not in the parquet, "
                f"e.g. {list(missing[:5])}. Rebuild and rerun the groups."
            )

        mismatch = (
            frame.loc[keys, "source_id"].values
            != hrus["source_id"].astype(str).values
        )

        if mismatch.any():
            raise ValueError(
                f"{int(mismatch.sum()):,} HRU(s) have a different "
                "source_id than the parquet. Rebuild and rerun the groups."
            )

        if hru_signature(hrus) != manifest["hru_signature"]:
            self._problem(
                "The HRU layer changed after the parquet was built. "
                "Rebuild the parquet and rerun the groups."
            )

        columns: list[str] = []
        parameter_columns: list[str] = []

        for entry in manifest["groups"].values():
            parameter_columns.extend(entry["parameter_columns"])
            columns.extend(
                entry["columns"]
                if self.include_qa
                else entry["parameter_columns"]
            )

        if not manifest["groups"]:
            self._problem("No stats groups have been appended yet.")

        empty = [
            column
            for column in parameter_columns
            if frame.loc[keys, column].isna().any()
        ]

        if empty:
            self._problem(
                f"Parameter columns with missing values: {empty}. Check "
                "each group's QA flags."
            )

        output = hrus.drop(columns=columns, errors="ignore").copy()

        values = frame.loc[keys, columns].reset_index(drop=True)
        values.index = output.index

        return gpd.GeoDataFrame(
            pd.concat([output, values], axis=1),
            geometry=hrus.geometry.name,
            crs=hrus.crs,
        )

