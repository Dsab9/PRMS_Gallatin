r"""Route orphan PRMS catchments using strict D8 and local sink recovery.

This module routes catchments that were not connected to an active stream
network during the source-agnostic routing stage.

Routing occurs in two passes:

    Pass 1: Strict D8 routing
        Select the highest accumulation candidate near the orphan catchment
        boundary. Trace using the supplied or calculated D8 direction raster.
        Assign the first active model segment or active routed catchment found.

    Pass 2: Local elevation-buffer recovery
        Run only if a strict D8 trace reaches a terminal cell before finding an
        active receiver. Search locally for a lower terrain exit that can be
        reached through a limited elevation barrier. From that exit, resume
        strict D8 tracing and attempt assignment again.

The elevation-buffer pass does not alter the source HydroDEM, D8 direction
raster, or flow accumulation raster. It produces an explicit proposed
connector for Arc Pro review.

Expected input datasource layers:
    model_catchments
        Required fields:
            catchment_id
            source_id
            routing_status
            manual_review
            is_active_hru
            geometry

    model_segments
        Required fields:
            segment_id
            source_id
            geometry

Output GeoPackage layers:
    orphan_routing_candidates
        Selected high-accumulation candidate outlet point for every processed
        orphan catchment.

    orphan_routing_paths
        Strict D8 path, or the strict path plus recovered connector plus
        post-recovery D8 path.

    orphan_routing_recovery_paths
        Only local elevation-buffer connector paths used during recovery.

    model_catchments_orphan_routed
        Full catchment review layer with proposed routing and recovery fields.

Additional files:
    orphan_routing_results.csv
    orphan_routing_summary.json
    hydrodem_d8_direction_internal.tif
    hydrodem_d8_accumulation_cells.tif

The input GeoPackage/file geodatabase is read-only to this module. Derived
layers are written to output_gpkg_path, avoiding repeated write/remove cycles
in the trusted routing-domain datasource.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

# =============================================================================
# PROJ configuration
# =============================================================================

PROJ_DIR = Path(sys.prefix) / "Library" / "share" / "proj"

if not (PROJ_DIR / "proj.db").exists():
    raise FileNotFoundError(
        "Could not find the conda PROJ database:\n"
        f"    {PROJ_DIR / 'proj.db'}"
    )

os.environ["PROJ_LIB"] = str(PROJ_DIR)
os.environ["PROJ_DATA"] = str(PROJ_DIR)

import pyproj

pyproj.datadir.set_data_dir(str(PROJ_DIR))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from fiona import listlayers, remove
from rasterio.features import geometry_mask
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# Configuration
# =============================================================================


class OrphanRoutingConfig:
    """Stores strict D8 and elevation-buffer orphan routing settings.

    Args:
        gpkg_path: Input GeoPackage or file geodatabase containing
            model_catchments and model_segments.
        output_gpkg_path: Output GeoPackage for all derived orphan-routing
            feature classes. If omitted, a new GeoPackage is created beside the
            input datasource.
        hydrodem_path: HydroDEM used for sink recovery and needed whenever D8
            direction or flow accumulation must be calculated.
        flow_direction_path: Optional existing D8 flow-direction raster.
        flow_accumulation_path: Optional existing flow-accumulation raster.
        flow_direction_encoding: ``internal`` for module 0-8 codes, or
            ``esri`` for standard ESRI D8 power-of-two codes.
        output_dir: Folder for CSV, JSON, and optional derived rasters.
        catchment_layer: Input catchment layer.
        segment_layer: Input active segment layer.
        orphan_statuses: Input routing_status values evaluated by this module.
        min_boundary_buffer_m: Inward boundary band used for outlet selection.
        minimum_accumulation_cells: Preferred candidate accumulation threshold.
        max_trace_steps: Maximum strict D8 cells followed per trace.
        segment_search_distance_m: Search radius for active segments.
        catchment_search_distance_m: Search radius for active HRUs.
        enable_sink_recovery: Enable local elevation-buffer recovery.
        flat_elevation_tolerance_m: Rise treated as approximately flat.
        max_breach_rise_m: Maximum permitted local elevation barrier above the
            terminal strict-D8 sink cell.
        max_breach_distance_m: Maximum planimetric search distance for a local
            recovery exit.
        minimum_exit_drop_m: Required drop below the terminal sink elevation
            for a recovery exit.
        write_flow_products: Write current D8/FAC products for QA.
    """

    DEFAULT_ORPHAN_STATUSES = (
        "missing_flowline",
        "sink",
        "downstream_missing",
        "outside_active_network",
    )

    VALID_DIRECTION_ENCODINGS = (
        "internal",
        "esri",
    )

    def __init__(
        self,
        gpkg_path: str | Path,
        output_gpkg_path: str | Path | None = None,
        hydrodem_path: str | Path | None = None,
        flow_direction_path: str | Path | None = None,
        flow_accumulation_path: str | Path | None = None,
        flow_direction_encoding: str = "internal",
        output_dir: str | Path | None = None,
        catchment_layer: str = "model_catchments",
        segment_layer: str = "model_segments",
        orphan_statuses: list[str] | tuple[str, ...] | None = None,
        min_boundary_buffer_m: float = 30.0,
        minimum_accumulation_cells: float = 1.0,
        max_trace_steps: int = 100_000,
        segment_search_distance_m: float = 75.0,
        catchment_search_distance_m: float = 25.0,
        enable_sink_recovery: bool = True,
        flat_elevation_tolerance_m: float = 0.25,
        max_breach_rise_m: float = 2.0,
        max_breach_distance_m: float = 50.0,
        minimum_exit_drop_m: float = 0.25,
        write_flow_products: bool = False,
    ) -> None:
        """Initializes configuration and validates all paths/settings."""
        self.gpkg_path = Path(gpkg_path)

        self.output_gpkg_path = (
            Path(output_gpkg_path)
            if output_gpkg_path is not None
            else self.gpkg_path.parent
            / f"{self.gpkg_path.stem}_orphan_routing.gpkg"
        )

        self.hydrodem_path = (
            Path(hydrodem_path)
            if hydrodem_path is not None
            else None
        )

        self.flow_direction_path = (
            Path(flow_direction_path)
            if flow_direction_path is not None
            else None
        )

        self.flow_accumulation_path = (
            Path(flow_accumulation_path)
            if flow_accumulation_path is not None
            else None
        )

        self.flow_direction_encoding = (
            str(flow_direction_encoding)
            .strip()
            .lower()
        )

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.gpkg_path.parent / "orphan_routing"
        )

        self.catchment_layer = str(catchment_layer)
        self.segment_layer = str(segment_layer)

        self.orphan_statuses = tuple(
            str(status)
            for status in (
                orphan_statuses
                if orphan_statuses is not None
                else self.DEFAULT_ORPHAN_STATUSES
            )
        )

        self.min_boundary_buffer_m = float(
            min_boundary_buffer_m
        )

        self.minimum_accumulation_cells = float(
            minimum_accumulation_cells
        )

        self.max_trace_steps = int(max_trace_steps)

        self.segment_search_distance_m = float(
            segment_search_distance_m
        )

        self.catchment_search_distance_m = float(
            catchment_search_distance_m
        )

        self.enable_sink_recovery = bool(
            enable_sink_recovery
        )

        self.flat_elevation_tolerance_m = float(
            flat_elevation_tolerance_m
        )

        self.max_breach_rise_m = float(
            max_breach_rise_m
        )

        self.max_breach_distance_m = float(
            max_breach_distance_m
        )

        self.minimum_exit_drop_m = float(
            minimum_exit_drop_m
        )

        self.write_flow_products = bool(
            write_flow_products
        )

        self._validate()

    @property
    def needs_direction_calculation(self) -> bool:
        """Returns whether flow direction must be calculated."""
        return self.flow_direction_path is None

    @property
    def needs_accumulation_calculation(self) -> bool:
        """Returns whether accumulation must be calculated."""
        return self.flow_accumulation_path is None

    @property
    def needs_hydrodem_for_d8(self) -> bool:
        """Returns whether HydroDEM is required for D8 calculation."""
        return (
            self.needs_direction_calculation
            or self.needs_accumulation_calculation
        )

    @property
    def needs_hydrodem(self) -> bool:
        """Returns whether HydroDEM is required for any configured operation."""
        return (
            self.needs_hydrodem_for_d8
            or self.enable_sink_recovery
        )

    def _validate(self) -> None:
        """Validates input paths and numerical routing settings."""
        if not self.gpkg_path.exists():
            raise FileNotFoundError(
                f"Input datasource does not exist: {self.gpkg_path}"
            )

        allowed_output_suffixes = (
            ".gpkg",
            ".gdb",
        )

        output_suffix = self.output_gpkg_path.suffix.lower()

        if output_suffix not in allowed_output_suffixes:
            raise ValueError(
                "output_gpkg_path must end with one of "
                f"{allowed_output_suffixes}. Received: "
                f"{self.output_gpkg_path}"
            )

        if (
            self.flow_direction_encoding
            not in self.VALID_DIRECTION_ENCODINGS
        ):
            raise ValueError(
                "flow_direction_encoding must be one of "
                f"{self.VALID_DIRECTION_ENCODINGS}; got "
                f"'{self.flow_direction_encoding}'."
            )

        if self.needs_hydrodem:
            if self.hydrodem_path is None:
                raise ValueError(
                    "hydrodem_path is required because D8 products and/or "
                    "sink recovery require terrain elevations."
                )

            if not self.hydrodem_path.exists():
                raise FileNotFoundError(
                    f"HydroDEM does not exist: {self.hydrodem_path}"
                )

        if (
            self.flow_direction_path is not None
            and not self.flow_direction_path.exists()
        ):
            raise FileNotFoundError(
                "Flow-direction raster does not exist: "
                f"{self.flow_direction_path}"
            )

        if (
            self.flow_accumulation_path is not None
            and not self.flow_accumulation_path.exists()
        ):
            raise FileNotFoundError(
                "Flow-accumulation raster does not exist: "
                f"{self.flow_accumulation_path}"
            )

        if not self.orphan_statuses:
            raise ValueError(
                "orphan_statuses cannot be empty."
            )

        if self.min_boundary_buffer_m < 0:
            raise ValueError(
                "min_boundary_buffer_m must be zero or greater."
            )

        if self.minimum_accumulation_cells < 0:
            raise ValueError(
                "minimum_accumulation_cells must be zero or greater."
            )

        if self.max_trace_steps <= 0:
            raise ValueError(
                "max_trace_steps must be greater than zero."
            )

        if self.segment_search_distance_m <= 0:
            raise ValueError(
                "segment_search_distance_m must be greater than zero."
            )

        if self.catchment_search_distance_m <= 0:
            raise ValueError(
                "catchment_search_distance_m must be greater than zero."
            )

        if self.flat_elevation_tolerance_m < 0:
            raise ValueError(
                "flat_elevation_tolerance_m must be zero or greater."
            )

        if self.max_breach_rise_m < 0:
            raise ValueError(
                "max_breach_rise_m must be zero or greater."
            )

        if self.max_breach_distance_m <= 0:
            raise ValueError(
                "max_breach_distance_m must be greater than zero."
            )

        if self.minimum_exit_drop_m < 0:
            raise ValueError(
                "minimum_exit_drop_m must be zero or greater."
            )


# =============================================================================
# Datasource and GeoPackage utility functions
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
    """Writes one spatial GeoDataFrame to the output GeoPackage.

    This uses the same Fiona/GPKG writer pattern as Routing.py. The only
    addition is that it can create a fresh output GeoPackage on the first write.
    """
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


def validate_output_layer(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    expected_geometry_types: (
        str
        | tuple[str, ...]
        | list[str]
        | set[str]
    ),
) -> gpd.GeoDataFrame:
    """Validates derived spatial output before GeoPackage write."""
    if gdf.empty:
        raise ValueError(
            f"Layer '{layer_name}' has no features to write."
        )

    if gdf.crs is None:
        raise ValueError(
            f"Layer '{layer_name}' has no CRS."
        )

    allowed_types = (
        {expected_geometry_types}
        if isinstance(expected_geometry_types, str)
        else set(expected_geometry_types)
    )

    null_count = int(gdf.geometry.isna().sum())

    if null_count > 0:
        raise ValueError(
            f"Layer '{layer_name}' contains "
            f"{null_count:,} null geometries."
        )

    empty_count = int(gdf.geometry.is_empty.sum())

    if empty_count > 0:
        raise ValueError(
            f"Layer '{layer_name}' contains "
            f"{empty_count:,} empty geometries."
        )

    geometry_types = set(
        gdf.geom_type.dropna().unique().tolist()
    )

    invalid_types = geometry_types - allowed_types

    if invalid_types:
        raise ValueError(
            f"Layer '{layer_name}' contains "
            f"geometry type(s) {sorted(geometry_types)}. "
            f"Allowed type(s): {sorted(allowed_types)}."
        )

    invalid_count = int(
        (~gdf.geometry.is_valid).sum()
    )

    if invalid_count > 0:
        raise ValueError(
            f"Layer '{layer_name}' contains "
            f"{invalid_count:,} invalid geometries."
        )

    return gdf


# =============================================================================
# Terrain processing
# =============================================================================


class D8Terrain:
    """Loads/calculates D8 terrain products and supports local recovery search.

    Internal D8 codes:

        0: terminal cell / sink / no downstream neighbor
        1: east
        2: southeast
        3: south
        4: southwest
        5: west
        6: northwest
        7: north
        8: northeast
    """

    OFFSETS = (
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    )

    def __init__(
        self,
        hydrodem_path: Path | None = None,
        flow_direction_path: Path | None = None,
        flow_accumulation_path: Path | None = None,
        flow_direction_encoding: str = "internal",
    ) -> None:
        """Initializes terrain products from source raster paths."""
        self.hydrodem_path = hydrodem_path
        self.flow_direction_path = flow_direction_path
        self.flow_accumulation_path = flow_accumulation_path

        self.flow_direction_encoding = (
            str(flow_direction_encoding)
            .strip()
            .lower()
        )

        self.dem: np.ndarray | None = None
        self.direction: np.ndarray | None = None
        self.accumulation: np.ndarray | None = None

        self.nodata: float | int | None = None
        self.profile: dict[str, Any] | None = None

        reference_path = (
            self.flow_direction_path
            or self.flow_accumulation_path
            or self.hydrodem_path
        )

        if reference_path is None:
            raise ValueError(
                "Provide hydrodem_path, flow_direction_path, "
                "or flow_accumulation_path."
            )

        with rasterio.open(reference_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.nrows = src.height
            self.ncols = src.width
            self.profile = src.profile.copy()

        if self.crs is None:
            raise ValueError(
                f"Raster has no CRS: {reference_path}"
            )

        self.cell_x = abs(self.transform.a)
        self.cell_y = abs(self.transform.e)

        if self.hydrodem_path is not None:
            self._load_hydrodem(
                self.hydrodem_path
            )

        if self.flow_direction_path is not None:
            self._load_direction(
                self.flow_direction_path
            )

        if self.flow_accumulation_path is not None:
            self._load_accumulation(
                self.flow_accumulation_path
            )

        self._validate_alignment()

    def build(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loads or calculates D8 direction and flow accumulation."""
        if self.direction is None:
            if self.dem is None:
                raise RuntimeError(
                    "HydroDEM is required to calculate D8 direction."
                )

            self.direction = self._calculate_d8_direction()

        if self.accumulation is None:
            self.accumulation = (
                self._calculate_flow_accumulation()
            )

        if not hasattr(self, "valid_mask"):
            self.valid_mask = np.isfinite(
                self.accumulation
            )

        return self.direction, self.accumulation

    def _load_hydrodem(
        self,
        hydrodem_path: Path,
    ) -> None:
        """Loads HydroDEM and creates valid terrain mask."""
        with rasterio.open(hydrodem_path) as src:
            self._assert_raster_matches_reference(
                src,
                hydrodem_path,
            )

            self.dem = src.read(1).astype(
                np.float64
            )

            self.nodata = src.nodata

        self.valid_mask = np.isfinite(self.dem)

        if self.nodata is not None:
            self.valid_mask &= (
                self.dem != self.nodata
            )

    def _load_direction(
        self,
        direction_path: Path,
    ) -> None:
        """Loads supplied direction and converts to internal 0-8 codes."""
        with rasterio.open(direction_path) as src:
            self._assert_raster_matches_reference(
                src,
                direction_path,
            )

            raw_direction = src.read(1)
            direction_nodata = src.nodata

        raw_direction = raw_direction.astype(
            np.int32,
            copy=True,
        )

        if direction_nodata is not None:
            raw_direction[
                raw_direction == direction_nodata
            ] = 0

        self.direction = (
            self._convert_direction_to_internal(
                raw_direction
            )
        )

        if not hasattr(self, "valid_mask"):
            self.valid_mask = (
                self.direction >= 0
            )

    def _load_accumulation(
        self,
        accumulation_path: Path,
    ) -> None:
        """Loads supplied flow accumulation."""
        with rasterio.open(accumulation_path) as src:
            self._assert_raster_matches_reference(
                src,
                accumulation_path,
            )

            accumulation = src.read(1).astype(
                np.float64
            )

            accumulation_nodata = src.nodata

        if accumulation_nodata is not None:
            accumulation[
                accumulation == accumulation_nodata
            ] = np.nan

        self.accumulation = accumulation

        if not hasattr(self, "valid_mask"):
            self.valid_mask = np.isfinite(
                self.accumulation
            )

    def _assert_raster_matches_reference(
        self,
        src: rasterio.io.DatasetReader,
        raster_path: Path,
    ) -> None:
        """Checks input raster alignment against reference grid."""
        if src.crs != self.crs:
            raise ValueError(
                f"CRS mismatch for {raster_path.name}. "
                f"Expected {self.crs}; found {src.crs}."
            )

        if (
            src.height != self.nrows
            or src.width != self.ncols
        ):
            raise ValueError(
                f"Grid-shape mismatch for {raster_path.name}. "
                f"Expected {(self.nrows, self.ncols)}; found "
                f"{(src.height, src.width)}."
            )

        if not src.transform.almost_equals(
            self.transform
        ):
            raise ValueError(
                f"Grid transform mismatch for {raster_path.name}. "
                "HydroDEM, FDR, and FAC must align exactly."
            )

    def _convert_direction_to_internal(
        self,
        direction: np.ndarray,
    ) -> np.ndarray:
        """Converts supplied direction codes to internal 0-8 codes."""
        direction = direction.astype(
            np.int32,
            copy=True,
        )

        if self.flow_direction_encoding == "internal":
            valid_codes = set(range(0, 9))
            found_codes = set(
                np.unique(direction).tolist()
            )

            invalid_codes = found_codes - valid_codes

            if invalid_codes:
                raise ValueError(
                    "Internal D8 raster has unsupported values: "
                    f"{sorted(invalid_codes)}"
                )

            return direction.astype(np.uint8)

        if self.flow_direction_encoding == "esri":
            esri_to_internal = {
                0: 0,
                1: 1,
                2: 2,
                4: 3,
                8: 4,
                16: 5,
                32: 6,
                64: 7,
                128: 8,
            }

            found_codes = set(
                np.unique(direction).tolist()
            )

            invalid_codes = (
                found_codes
                - set(esri_to_internal)
            )

            if invalid_codes:
                raise ValueError(
                    "ESRI D8 raster has unsupported values: "
                    f"{sorted(invalid_codes)}"
                )

            output = np.zeros(
                direction.shape,
                dtype=np.uint8,
            )

            for source_code, internal_code in (
                esri_to_internal.items()
            ):
                output[
                    direction == source_code
                ] = internal_code

            return output

        raise ValueError(
            "Unsupported flow-direction encoding: "
            f"{self.flow_direction_encoding}"
        )

    def _validate_alignment(
        self,
    ) -> None:
        """Validates loaded terrain array shapes."""
        expected_shape = (
            self.nrows,
            self.ncols,
        )

        for name, array in (
            ("HydroDEM", self.dem),
            ("flow direction", self.direction),
            ("flow accumulation", self.accumulation),
        ):
            if (
                array is not None
                and array.shape != expected_shape
            ):
                raise ValueError(
                    f"{name} has shape {array.shape}; expected "
                    f"{expected_shape}."
                )

    def _calculate_d8_direction(
        self,
    ) -> np.ndarray:
        """Calculates strict steepest-descent D8 directions from HydroDEM."""
        if self.dem is None:
            raise RuntimeError(
                "HydroDEM is required to calculate D8 direction."
            )

        direction = np.zeros(
            (self.nrows, self.ncols),
            dtype=np.uint8,
        )

        diagonal_distance = math.hypot(
            self.cell_x,
            self.cell_y,
        )

        distances = (
            self.cell_x,
            diagonal_distance,
            self.cell_y,
            diagonal_distance,
            self.cell_x,
            diagonal_distance,
            self.cell_y,
            diagonal_distance,
        )

        for row in tqdm(
            range(1, self.nrows - 1),
            desc="Calculating D8 direction",
            unit="row",
        ):
            for col in range(1, self.ncols - 1):
                if not self.valid_mask[row, col]:
                    continue

                elevation = self.dem[row, col]
                maximum_slope = 0.0
                selected_code = 0

                for code, (
                    (drow, dcol),
                    distance,
                ) in enumerate(
                    zip(self.OFFSETS, distances),
                    start=1,
                ):
                    neighbor_row = row + drow
                    neighbor_col = col + dcol

                    if not self.valid_mask[
                        neighbor_row,
                        neighbor_col,
                    ]:
                        continue

                    elevation_drop = (
                        elevation
                        - self.dem[
                            neighbor_row,
                            neighbor_col,
                        ]
                    )

                    slope = elevation_drop / distance

                    if slope > maximum_slope:
                        maximum_slope = slope
                        selected_code = code

                direction[row, col] = selected_code

        return direction

    def _calculate_flow_accumulation(
        self,
    ) -> np.ndarray:
        """Calculates true D8 contributing-upstream-cell accumulation."""
        if self.direction is None:
            raise RuntimeError(
                "D8 direction must exist before calculating "
                "flow accumulation."
            )

        indegree = np.zeros(
            (self.nrows, self.ncols),
            dtype=np.int32,
        )

        for row in tqdm(
            range(self.nrows),
            desc="Building D8 topology",
            unit="row",
        ):
            for col in range(self.ncols):
                if not self.valid_mask[row, col]:
                    continue

                direction_code = int(
                    self.direction[row, col]
                )

                if direction_code == 0:
                    continue

                drow, dcol = self.OFFSETS[
                    direction_code - 1
                ]

                downstream_row = row + drow
                downstream_col = col + dcol

                if (
                    0 <= downstream_row < self.nrows
                    and 0 <= downstream_col < self.ncols
                    and self.valid_mask[
                        downstream_row,
                        downstream_col,
                    ]
                ):
                    indegree[
                        downstream_row,
                        downstream_col,
                    ] += 1

        accumulation = np.zeros(
            (self.nrows, self.ncols),
            dtype=np.float64,
        )

        accumulation[self.valid_mask] = 1.0

        headwater_rows, headwater_cols = np.where(
            self.valid_mask
            & (indegree == 0)
        )

        queue: deque[tuple[int, int]] = deque(
            zip(
                headwater_rows.tolist(),
                headwater_cols.tolist(),
            )
        )

        processed_count = 0

        while queue:
            row, col = queue.popleft()
            processed_count += 1

            direction_code = int(
                self.direction[row, col]
            )

            if direction_code == 0:
                continue

            drow, dcol = self.OFFSETS[
                direction_code - 1
            ]

            downstream_row = row + drow
            downstream_col = col + dcol

            if not (
                0 <= downstream_row < self.nrows
                and 0 <= downstream_col < self.ncols
                and self.valid_mask[
                    downstream_row,
                    downstream_col,
                ]
            ):
                continue

            accumulation[
                downstream_row,
                downstream_col,
            ] += accumulation[row, col]

            indegree[
                downstream_row,
                downstream_col,
            ] -= 1

            if indegree[
                downstream_row,
                downstream_col,
            ] == 0:
                queue.append(
                    (
                        downstream_row,
                        downstream_col,
                    )
                )

        valid_count = int(
            self.valid_mask.sum()
        )

        if processed_count < valid_count:
            tqdm.write(
                "WARNING: Flow accumulation did not process "
                f"{valid_count - processed_count:,} valid cell(s)."
            )

        return accumulation

    def rowcol_to_point(
        self,
        row: int,
        col: int,
    ) -> Point:
        """Converts raster row/column to cell-center point."""
        x, y = rasterio.transform.xy(
            self.transform,
            row,
            col,
            offset="center",
        )

        return Point(x, y)

    def cell_elevation(
        self,
        row: int,
        col: int,
    ) -> float:
        """Returns HydroDEM elevation at one valid grid cell."""
        if self.dem is None:
            raise RuntimeError(
                "HydroDEM is required for elevation-buffer recovery."
            )

        return float(self.dem[row, col])

    def cell_is_valid(
        self,
        row: int,
        col: int,
    ) -> bool:
        """Returns whether row/column is inside valid terrain."""
        return bool(
            0 <= row < self.nrows
            and 0 <= col < self.ncols
            and self.valid_mask[row, col]
        )

    def terminal_cell(
        self,
        trace_cells: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Returns final strict-trace cell, if any."""
        if not trace_cells:
            return None

        return trace_cells[-1]

    def is_terminal_d8_cell(
        self,
        row: int,
        col: int,
    ) -> bool:
        """Returns whether a cell has no D8 downstream direction."""
        if self.direction is None:
            raise RuntimeError(
                "D8 direction must exist before terminal checks."
            )

        return bool(
            self.direction[row, col] == 0
        )

    def trace_downstream(
        self,
        start_row: int,
        start_col: int,
        max_steps: int,
    ) -> list[tuple[int, int]]:
        """Traces strict D8 cells downstream from a starting cell."""
        if self.direction is None:
            raise RuntimeError(
                "D8 direction must be built before tracing."
            )

        path: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()

        row = int(start_row)
        col = int(start_col)

        for _ in range(max_steps):
            if not self.cell_is_valid(row, col):
                break

            current_cell = (row, col)

            if current_cell in visited:
                break

            visited.add(current_cell)
            path.append(current_cell)

            direction_code = int(
                self.direction[row, col]
            )

            if direction_code == 0:
                break

            drow, dcol = self.OFFSETS[
                direction_code - 1
            ]

            row += drow
            col += dcol

        return path

    def find_local_recovery_path(
        self,
        start_row: int,
        start_col: int,
        flat_tolerance_m: float,
        max_rise_m: float,
        max_distance_m: float,
        minimum_exit_drop_m: float,
    ) -> dict[str, Any] | None:
        """Finds a constrained local terrain path out of a terminal D8 sink.

        The search is lexicographically optimized by:

            1. Lowest maximum elevation barrier above the sink elevation.
            2. Shortest horizontal path distance.
            3. Lowest exit elevation.

        A valid exit must:

            - be within max_distance_m of the sink,
            - require no more than max_rise_m above the sink elevation,
            - be lower than the sink by minimum_exit_drop_m,
            - have a non-terminal strict D8 direction.

        Args:
            start_row: Terminal strict-D8 sink row.
            start_col: Terminal strict-D8 sink column.
            flat_tolerance_m: Small tolerance used for reporting flat paths.
            max_rise_m: Maximum barrier above terminal sink elevation.
            max_distance_m: Maximum local search distance.
            minimum_exit_drop_m: Required elevation drop for exit cell.

        Returns:
            Dictionary with recovery cells, rise, distance, and exit elevation;
            otherwise None.
        """
        if self.dem is None:
            raise RuntimeError(
                "HydroDEM is required for local sink recovery."
            )

        if self.direction is None:
            raise RuntimeError(
                "D8 direction is required for local sink recovery."
            )

        if not self.cell_is_valid(
            start_row,
            start_col,
        ):
            return None

        start_elevation = self.cell_elevation(
            start_row,
            start_col,
        )

        maximum_exit_elevation = (
            start_elevation
            - minimum_exit_drop_m
        )

        max_row_offset = int(
            math.ceil(
                max_distance_m / self.cell_y
            )
        )

        max_col_offset = int(
            math.ceil(
                max_distance_m / self.cell_x
            )
        )

        minimum_row = max(
            0,
            start_row - max_row_offset,
        )

        maximum_row = min(
            self.nrows - 1,
            start_row + max_row_offset,
        )

        minimum_col = max(
            0,
            start_col - max_col_offset,
        )

        maximum_col = min(
            self.ncols - 1,
            start_col + max_col_offset,
        )

        start_cell = (
            int(start_row),
            int(start_col),
        )

        # Heap values:
        # (
        #     maximum_rise_above_sink,
        #     path_distance_m,
        #     row,
        #     col,
        # )
        queue: list[
            tuple[float, float, int, int]
        ] = [
            (
                0.0,
                0.0,
                start_row,
                start_col,
            )
        ]

        best_cost: dict[
            tuple[int, int],
            tuple[float, float],
        ] = {
            start_cell: (0.0, 0.0)
        }

        parent: dict[
            tuple[int, int],
            tuple[int, int] | None,
        ] = {
            start_cell: None
        }

        diagonal_distance = math.hypot(
            self.cell_x,
            self.cell_y,
        )

        neighbor_distances = (
            self.cell_x,
            diagonal_distance,
            self.cell_y,
            diagonal_distance,
            self.cell_x,
            diagonal_distance,
            self.cell_y,
            diagonal_distance,
        )

        while queue:
            (
                barrier_rise,
                path_distance,
                row,
                col,
            ) = heapq.heappop(queue)

            cell = (row, col)

            known_cost = best_cost.get(cell)

            if known_cost is None:
                continue

            if (
                barrier_rise,
                path_distance,
            ) != known_cost:
                continue

            elevation = self.cell_elevation(
                row,
                col,
            )

            is_start = cell == start_cell

            if (
                not is_start
                and elevation <= maximum_exit_elevation
                and self.direction[row, col] != 0
            ):
                cells = self._reconstruct_path(
                    parent,
                    cell,
                )

                return {
                    "cells": cells,
                    "max_rise_m": float(
                        barrier_rise
                    ),
                    "distance_m": float(
                        path_distance
                    ),
                    "start_elevation_m": float(
                        start_elevation
                    ),
                    "exit_elevation_m": float(
                        elevation
                    ),
                    "exit_row": int(row),
                    "exit_col": int(col),
                    "flat_only": bool(
                        barrier_rise
                        <= flat_tolerance_m
                    ),
                }

            for (
                (drow, dcol),
                step_distance,
            ) in zip(
                self.OFFSETS,
                neighbor_distances,
            ):
                neighbor_row = row + drow
                neighbor_col = col + dcol

                if (
                    neighbor_row < minimum_row
                    or neighbor_row > maximum_row
                    or neighbor_col < minimum_col
                    or neighbor_col > maximum_col
                ):
                    continue

                if not self.cell_is_valid(
                    neighbor_row,
                    neighbor_col,
                ):
                    continue

                candidate_distance = (
                    path_distance
                    + step_distance
                )

                if candidate_distance > max_distance_m:
                    continue

                neighbor_elevation = self.cell_elevation(
                    neighbor_row,
                    neighbor_col,
                )

                candidate_rise = max(
                    barrier_rise,
                    neighbor_elevation
                    - start_elevation,
                    0.0,
                )

                if candidate_rise > max_rise_m:
                    continue

                neighbor_cell = (
                    neighbor_row,
                    neighbor_col,
                )

                candidate_cost = (
                    candidate_rise,
                    candidate_distance,
                )

                existing_cost = best_cost.get(
                    neighbor_cell
                )

                if (
                    existing_cost is None
                    or candidate_cost < existing_cost
                ):
                    best_cost[neighbor_cell] = (
                        candidate_cost
                    )

                    parent[neighbor_cell] = cell

                    heapq.heappush(
                        queue,
                        (
                            candidate_rise,
                            candidate_distance,
                            neighbor_row,
                            neighbor_col,
                        ),
                    )

        return None

    @staticmethod
    def _reconstruct_path(
        parent: dict[
            tuple[int, int],
            tuple[int, int] | None,
        ],
        endpoint: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Reconstructs a local recovery path from parent links."""
        cells: list[tuple[int, int]] = []
        current: tuple[int, int] | None = endpoint

        while current is not None:
            cells.append(current)
            current = parent[current]

        cells.reverse()

        return cells

    def write_products(
        self,
        output_dir: Path,
    ) -> None:
        """Writes current D8 direction and accumulation raster products."""
        if (
            self.direction is None
            or self.accumulation is None
        ):
            raise RuntimeError(
                "Build terrain products before writing them."
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        direction_profile = self.profile.copy()

        direction_profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="lzw",
        )

        direction_path = (
            output_dir
            / "hydrodem_d8_direction_internal.tif"
        )

        with rasterio.open(
            direction_path,
            "w",
            **direction_profile,
        ) as destination:
            destination.write(
                self.direction.astype(np.uint8),
                1,
            )

        accumulation_profile = self.profile.copy()

        accumulation_profile.update(
            dtype="float64",
            count=1,
            nodata=-9999.0,
            compress="lzw",
        )

        output_accumulation = self.accumulation.copy()

        output_accumulation[
            ~self.valid_mask
        ] = -9999.0

        accumulation_path = (
            output_dir
            / "hydrodem_d8_accumulation_cells.tif"
        )

        with rasterio.open(
            accumulation_path,
            "w",
            **accumulation_profile,
        ) as destination:
            destination.write(
                output_accumulation,
                1,
            )


# =============================================================================
# Orphan router
# =============================================================================


class OrphanCatchmentRouter:
    """Routes orphan catchments using strict D8 and local recovery fallback."""

    REQUIRED_CATCHMENT_FIELDS = {
        "catchment_id",
        "source_id",
        "routing_status",
        "manual_review",
        "is_active_hru",
    }

    REQUIRED_SEGMENT_FIELDS = {
        "segment_id",
        "source_id",
    }

    def __init__(
        self,
        config: OrphanRoutingConfig,
        terrain: D8Terrain,
        catchments: gpd.GeoDataFrame,
        segments: gpd.GeoDataFrame,
    ) -> None:
        """Initializes router inputs."""
        self.config = config
        self.terrain = terrain

        self.catchments = (
            lower_columns(catchments)
            .to_crs(self.terrain.crs)
            .copy()
        )

        self.segments = (
            lower_columns(segments)
            .to_crs(self.terrain.crs)
            .copy()
        )

        self._validate_inputs()

        self.active_hrus = self.catchments[
            self.catchments["is_active_hru"] == 1
        ].copy()

        self.orphans = self.catchments[
            self.catchments["routing_status"].isin(
                self.config.orphan_statuses
            )
        ].copy()

        self.segment_sindex = self.segments.sindex
        self.active_hru_sindex = (
            self.active_hrus.sindex
        )
        self.catchment_sindex = self.catchments.sindex

    def route_orphans(
        self,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        pd.DataFrame,
        gpd.GeoDataFrame,
    ]:
        """Routes all configured orphan catchments.

        Returns:
            Candidate points, full routing paths, local recovery connector
            paths, tabular results, and updated catchment review layer.
        """
        point_records: list[dict[str, Any]] = []
        path_records: list[dict[str, Any]] = []
        recovery_records: list[dict[str, Any]] = []
        result_records: list[dict[str, Any]] = []

        for orphan in tqdm(
            self.orphans.itertuples(),
            total=len(self.orphans),
            desc="Routing orphan catchments",
            unit="catchment",
        ):
            (
                result,
                point_record,
                path_record,
                recovery_record,
            ) = self._route_one_orphan(orphan)

            result_records.append(result)

            if point_record is not None:
                point_records.append(point_record)

            if path_record is not None:
                path_records.append(path_record)

            if recovery_record is not None:
                recovery_records.append(recovery_record)

        results = pd.DataFrame(result_records)

        results = self._flag_circular_routing(results)

        outlet_points = self._records_to_gdf(
            point_records
        )

        routing_paths = self._records_to_gdf(
            path_records
        )

        recovery_paths = self._records_to_gdf(
            recovery_records
        )

        updated_catchments = self._apply_results(
            results
        )

        return (
            outlet_points,
            routing_paths,
            recovery_paths,
            results,
            updated_catchments,
        )

    def _records_to_gdf(
        self,
        records: list[dict[str, Any]],
    ) -> gpd.GeoDataFrame:
        """Creates an output GeoDataFrame from spatial records."""
        if not records:
            return gpd.GeoDataFrame(
                geometry=[],
                crs=self.terrain.crs,
            )

        return gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=self.terrain.crs,
        )

    def _route_one_orphan(
        self,
        orphan: Any,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        """Routes one orphan with strict D8 then optional sink recovery."""
        catchment_id = str(orphan.catchment_id)
        source_id = str(orphan.source_id)

        result = self._empty_result(
            catchment_id=catchment_id,
            source_id=source_id,
        )

        candidate = self._find_orphan_outlet_candidate(
            orphan.geometry
        )

        if candidate is None:
            result.update(
                routing_method="no_valid_dem_cell",
                routing_confidence="low",
                routing_notes=(
                    "No valid flow-accumulation cell was found "
                    "inside the orphan catchment."
                ),
                manual_review=1,
            )

            return result, None, None, None

        row, col, accumulation_value = candidate

        outlet_point = self.terrain.rowcol_to_point(
            row,
            col,
        )

        result.update(
            orphan_outlet_row=int(row),
            orphan_outlet_col=int(col),
            outlet_accumulation_cells=float(
                accumulation_value
            ),
        )

        strict_cells = self.terrain.trace_downstream(
            start_row=row,
            start_col=col,
            max_steps=self.config.max_trace_steps,
        )

        strict_assignment = self._find_trace_assignment(
            strict_cells,
            method_prefix="dem_trace",
            own_catchment_id=catchment_id,
        )

        all_cells = strict_cells.copy()
        recovery_record = None

        result["strict_trace_cell_count"] = int(
            len(strict_cells)
        )

        if strict_assignment is not None:
            self._apply_assignment(
                result,
                strict_assignment,
            )

        else:
            terminal = self.terrain.terminal_cell(
                strict_cells
            )

            can_recover = (
                self.config.enable_sink_recovery
                and terminal is not None
                and self.terrain.is_terminal_d8_cell(
                    terminal[0],
                    terminal[1],
                )
            )

            if can_recover:
                recovery = (
                    self.terrain.find_local_recovery_path(
                        start_row=terminal[0],
                        start_col=terminal[1],
                        flat_tolerance_m=(
                            self.config
                            .flat_elevation_tolerance_m
                        ),
                        max_rise_m=(
                            self.config
                            .max_breach_rise_m
                        ),
                        max_distance_m=(
                            self.config
                            .max_breach_distance_m
                        ),
                        minimum_exit_drop_m=(
                            self.config
                            .minimum_exit_drop_m
                        ),
                    )
                )

                result["sink_recovery_attempted"] = 1

                if recovery is not None:
                    (
                        recovery_assignment,
                        post_recovery_cells,
                    ) = self._route_from_recovery_exit(
                        recovery,
                        catchment_id,
                    )

                    self._record_recovery_metrics(
                        result,
                        recovery,
                    )

                    recovery_geometry = self._build_line_from_cells(
                        recovery["cells"]
                    )

                    if recovery_geometry is not None:
                        recovery_record = {
                            **result,
                            "geometry": recovery_geometry,
                        }

                    if recovery_assignment is not None:
                        self._apply_recovery_assignment(
                            result,
                            recovery_assignment,
                            recovery,
                        )

                        all_cells = (
                            strict_cells
                            + recovery["cells"][1:]
                            + post_recovery_cells[1:]
                        )

                    else:
                        result.update(
                            routing_method=(
                                "dem_elevation_buffer_unresolved"
                            ),
                            routing_confidence="low",
                            routing_notes=(
                                "A local lower terrain exit was found, "
                                "but strict D8 tracing from that exit did "
                                "not reach an active model segment or HRU."
                            ),
                            manual_review=1,
                        )

                        all_cells = (
                            strict_cells
                            + recovery["cells"][1:]
                        )

                else:
                    result.update(
                        routing_method="dem_internal_sink",
                        routing_confidence="high",
                        routing_notes=(
                            "Strict D8 trace ended at a terminal cell and "
                            "no local lower exit was found within the "
                            "configured elevation-buffer limits."
                        ),
                        manual_review=0,
                    )

            else:
                result.update(
                    routing_method="dem_unresolved",
                    routing_confidence="low",
                    routing_notes=(
                        "Strict D8 trace did not reach an active receiver "
                        "and did not end at a recoverable terminal D8 cell."
                    ),
                    manual_review=1,
                )

        result["trace_cell_count"] = int(
            len(all_cells)
        )

        full_path_geometry = self._build_line_from_cells(
            all_cells
        )

        if full_path_geometry is not None:
            result["trace_length_m"] = float(
                full_path_geometry.length
            )

        point_record = {
            **result,
            "geometry": outlet_point,
        }

        path_record = None

        if full_path_geometry is not None:
            path_record = {
                **result,
                "geometry": full_path_geometry,
            }

        if recovery_record is not None:
            recovery_record.update(result)

        return (
            result,
            point_record,
            path_record,
            recovery_record,
        )

    def _route_from_recovery_exit(
        self,
        recovery: dict[str, Any],
        own_catchment_id: str,
    ) -> tuple[
        dict[str, Any] | None,
        list[tuple[int, int]],
    ]:
        """Resumes strict D8 tracing from a local recovery exit."""
        exit_row = int(recovery["exit_row"])
        exit_col = int(recovery["exit_col"])

        post_recovery_cells = self.terrain.trace_downstream(
            start_row=exit_row,
            start_col=exit_col,
            max_steps=self.config.max_trace_steps,
        )

        assignment = self._find_trace_assignment(
            post_recovery_cells,
            method_prefix="dem_elevation_buffer",
            own_catchment_id=own_catchment_id,
        )

        return assignment, post_recovery_cells

    def _record_recovery_metrics(
        self,
        result: dict[str, Any],
        recovery: dict[str, Any],
    ) -> None:
        """Copies local recovery diagnostics into a result record."""
        result.update(
            sink_recovery_success=1,
            sink_recovery_max_rise_m=float(
                recovery["max_rise_m"]
            ),
            sink_recovery_distance_m=float(
                recovery["distance_m"]
            ),
            sink_recovery_exit_row=int(
                recovery["exit_row"]
            ),
            sink_recovery_exit_col=int(
                recovery["exit_col"]
            ),
            sink_recovery_start_elev_m=float(
                recovery["start_elevation_m"]
            ),
            sink_recovery_exit_elev_m=float(
                recovery["exit_elevation_m"]
            ),
            sink_recovery_flat_only=int(
                recovery["flat_only"]
            ),
        )

    def _apply_assignment(
        self,
        result: dict[str, Any],
        assignment: dict[str, Any],
    ) -> None:
        """Applies strict D8 assignment attributes."""
        result.update(
            assigned_source_id=str(
                assignment["source_id"]
            ),
            assigned_segment_id=(
                None
                if assignment["segment_id"] is None
                else int(assignment["segment_id"])
            ),
            assigned_catchment_id=assignment.get("catchment_id"),
            routing_method=str(
                assignment["routing_method"]
            ),
            routing_confidence=str(
                assignment["routing_confidence"]
            ),
            routing_notes=str(
                assignment["routing_notes"]
            ),
            manual_review=int(
                assignment["routing_confidence"]
                not in (
                    "high",
                    "medium_high",
                )
            ),
        )

    def _apply_recovery_assignment(
        self,
        result: dict[str, Any],
        assignment: dict[str, Any],
        recovery: dict[str, Any],
    ) -> None:
        """Applies local elevation-buffer recovery assignment attributes."""
        rise_m = float(recovery["max_rise_m"])
        distance_m = float(recovery["distance_m"])
        flat_only = bool(recovery["flat_only"])

        if (
            flat_only
            and distance_m <= (
                self.config.max_breach_distance_m
                * 0.60
            )
        ):
            confidence = "medium_high"
        else:
            confidence = "medium"

        receiver_type = assignment["receiver_type"]

        method = (
            "dem_flat_buffer_to_segment"
            if flat_only
            and receiver_type == "segment"
            else "dem_flat_buffer_to_catchment"
            if flat_only
            else "dem_elevation_buffer_to_segment"
            if receiver_type == "segment"
            else "dem_elevation_buffer_to_catchment"
        )

        result.update(
            assigned_source_id=str(
                assignment["source_id"]
            ),
            assigned_segment_id=(
                None
                if assignment["segment_id"] is None
                else int(assignment["segment_id"])
            ),
            assigned_catchment_id=assignment.get("catchment_id"),
            routing_method=method,
            routing_confidence=confidence,
            routing_notes=(
                "Strict D8 trace terminated at a sink. "
                f"Local elevation-buffer recovery crossed a maximum "
                f"rise of {rise_m:.3f} m over {distance_m:.1f} m, "
                f"then reached an active model {receiver_type}."
            ),
            manual_review=1,
        )

    @staticmethod
    def _empty_result(
        catchment_id: str,
        source_id: str,
    ) -> dict[str, Any]:
        """Creates an unresolved orphan-routing record."""
        return {
            "catchment_id": catchment_id,
            "source_id": source_id,
            "assigned_source_id": None,
            "assigned_segment_id": None,
            "assigned_catchment_id": None,
            "routing_method": "unresolved",
            "routing_confidence": "low",
            "routing_notes": None,
            "manual_review": 1,
            "orphan_outlet_row": None,
            "orphan_outlet_col": None,
            "outlet_accumulation_cells": None,
            "strict_trace_cell_count": 0,
            "trace_cell_count": 0,
            "trace_length_m": 0.0,
            "sink_recovery_attempted": 0,
            "sink_recovery_success": 0,
            "sink_recovery_max_rise_m": None,
            "sink_recovery_distance_m": None,
            "sink_recovery_exit_row": None,
            "sink_recovery_exit_col": None,
            "sink_recovery_start_elev_m": None,
            "sink_recovery_exit_elev_m": None,
            "sink_recovery_flat_only": 0,
        }

    def _find_orphan_outlet_candidate(
        self,
        polygon: BaseGeometry,
    ) -> tuple[int, int, float] | None:
        """Finds highest preferred accumulation cell near catchment boundary."""
        if self.terrain.accumulation is None:
            raise RuntimeError(
                "Flow accumulation must exist before routing."
            )

        inside_mask = geometry_mask(
            [polygon],
            out_shape=(
                self.terrain.nrows,
                self.terrain.ncols,
            ),
            transform=self.terrain.transform,
            invert=True,
        )

        valid_inside = (
            inside_mask
            & self.terrain.valid_mask
            & np.isfinite(self.terrain.accumulation)
        )

        if not valid_inside.any():
            return None

        inner_polygon = polygon.buffer(
            -self.config.min_boundary_buffer_m
        )

        if not inner_polygon.is_empty:
            inner_mask = geometry_mask(
                [inner_polygon],
                out_shape=(
                    self.terrain.nrows,
                    self.terrain.ncols,
                ),
                transform=self.terrain.transform,
                invert=True,
            )

            boundary_band = (
                valid_inside
                & ~inner_mask
            )
        else:
            boundary_band = valid_inside

        threshold_candidates = (
            boundary_band
            & (
                self.terrain.accumulation
                >= self.config.minimum_accumulation_cells
            )
        )

        candidate_mask = (
            threshold_candidates
            if threshold_candidates.any()
            else valid_inside
        )

        candidate_accumulation = np.where(
            candidate_mask,
            self.terrain.accumulation,
            -np.inf,
        )

        flat_index = int(
            np.argmax(candidate_accumulation)
        )

        row, col = np.unravel_index(
            flat_index,
            candidate_accumulation.shape,
        )

        accumulation_value = candidate_accumulation[
            row,
            col,
        ]

        if not np.isfinite(accumulation_value):
            return None

        return (
            int(row),
            int(col),
            float(accumulation_value),
        )

    def _find_trace_assignment(
        self,
        trace_cells: list[tuple[int, int]],
        method_prefix: str,
        own_catchment_id: str,
    ) -> dict[str, Any] | None:
        """Finds first segment, active HRU, or other catchment on a trace.

        Priority at each cell is active segment, then active HRU, then any
        other catchment (strict containment). The orphan's own polygon is
        never a valid receiver.
        """
        for trace_index, (
            row,
            col,
        ) in enumerate(trace_cells):
            point = self.terrain.rowcol_to_point(
                row,
                col,
            )

            segment_assignment = (
                self._match_active_segment(
                    point=point,
                    search_distance=(
                        self.config
                        .segment_search_distance_m
                    ),
                )
            )

            if segment_assignment is not None:
                return {
                    **segment_assignment,
                    "receiver_type": "segment",
                    "routing_method": (
                        f"{method_prefix}_to_segment"
                    ),
                    "routing_confidence": "high",
                    "routing_notes": (
                        "Strict D8 trace reached an active "
                        "model stream segment."
                    ),
                    "trace_index": int(trace_index),
                }

            catchment_assignment = (
                self._match_active_catchment(
                    point=point,
                    search_distance=(
                        self.config
                        .catchment_search_distance_m
                    ),
                )
            )

            if catchment_assignment is not None:
                return {
                    **catchment_assignment,
                    "receiver_type": "catchment",
                    "routing_method": (
                        f"{method_prefix}_to_catchment"
                    ),
                    "routing_confidence": "medium_high",
                    "routing_notes": (
                        "Strict D8 trace entered an active routed "
                        "catchment before reaching a segment."
                    ),
                    "trace_index": int(trace_index),
                }

            other_catchment = self._match_any_catchment(
                point=point,
                own_catchment_id=own_catchment_id,
            )

            if other_catchment is not None:
                return {
                    **other_catchment,
                    "receiver_type": "catchment",
                    "routing_method": (
                        f"{method_prefix}_to_catchment"
                    ),
                    "routing_confidence": "medium",
                    "routing_notes": (
                        "Strict D8 trace entered another catchment that "
                        "is not an active routed HRU."
                    ),
                    "trace_index": int(trace_index),
                }

        return None

    def _match_active_segment(
        self,
        point: Point,
        search_distance: float,
    ) -> dict[str, Any] | None:
        """Finds nearest active segment near one trace point."""
        candidate_indices = list(
            self.segment_sindex.query(
                point.buffer(search_distance),
                predicate="intersects",
            )
        )

        if not candidate_indices:
            return None

        candidates = self.segments.iloc[
            candidate_indices
        ].copy()

        candidates["distance_to_trace"] = (
            candidates.geometry.distance(point)
        )

        nearest = candidates.sort_values(
            "distance_to_trace"
        ).iloc[0]

        return {
            "source_id": str(nearest["source_id"]),
            "segment_id": int(
                nearest["segment_id"]
            ),
        }

    def _match_active_catchment(
        self,
        point: Point,
        search_distance: float,
    ) -> dict[str, Any] | None:
        """Finds active routed HRU containing or nearest one trace point."""
        candidate_indices = list(
            self.active_hru_sindex.query(
                point.buffer(search_distance),
                predicate="intersects",
            )
        )

        if not candidate_indices:
            return None

        candidates = self.active_hrus.iloc[
            candidate_indices
        ].copy()

        containing = candidates[
            candidates.geometry.contains(point)
            | candidates.geometry.touches(point)
        ]

        if containing.empty:
            candidates["distance_to_trace"] = (
                candidates.geometry.distance(point)
            )

            selected = candidates.sort_values(
                "distance_to_trace"
            ).iloc[0]
        else:
            selected = containing.iloc[0]

        return {
            "source_id": str(selected["source_id"]),
            "segment_id": int(
                selected["segment_id"]
            ),
        }

    def _match_any_catchment(
        self,
        point: Point,
        own_catchment_id: str,
    ) -> dict[str, Any] | None:
        """Finds another catchment that contains a trace point.

        Strict containment only (no search distance), so a trace is never
        sent to a neighbor it merely passes near. Used after active
        segment/HRU matching fails.
        """
        hits = self.catchment_sindex.query(
            point,
            predicate="intersects",
        )

        if len(hits) == 0:
            return None

        candidates = self.catchments.iloc[list(hits)]

        candidates = candidates[
            candidates["catchment_id"].astype(str)
            != own_catchment_id
        ]

        if candidates.empty:
            return None

        selected = candidates.sort_values(
            "catchment_id"
        ).iloc[0]

        return {
            "source_id": str(selected["source_id"]),
            "catchment_id": str(selected["catchment_id"]),
            "segment_id": None,
        }

    @staticmethod
    def _flag_circular_routing(
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clears proposals that route orphans in a circle (A -> B -> A)."""
        if results.empty:
            return results

        output = results.copy()

        next_id = {
            str(c): str(t)
            for c, t in zip(
                output["catchment_id"],
                output["assigned_catchment_id"],
            )
            if pd.notna(t)
        }

        in_cycle: set[str] = set()

        for start in next_id:
            path: list[str] = []
            node = start

            while node in next_id and node not in path:
                path.append(node)
                node = next_id[node]

            if node in path:
                in_cycle.update(path[path.index(node):])

        if not in_cycle:
            return output

        mask = output["catchment_id"].astype(str).isin(in_cycle)

        output.loc[
            mask,
            [
                "assigned_source_id",
                "assigned_segment_id",
                "assigned_catchment_id",
            ],
        ] = None

        output.loc[mask, "routing_method"] = "circular_routing"
        output.loc[mask, "routing_confidence"] = "low"
        output.loc[mask, "manual_review"] = 1
        output.loc[mask, "routing_notes"] = (
            "Proposed catchment routing forms a loop among orphan "
            "catchments, so the proposal was removed."
        )

        return output

    def _build_line_from_cells(
        self,
        cells: list[tuple[int, int]],
    ) -> LineString | None:
        """Converts ordered cells to valid non-degenerate line geometry."""
        coordinates: list[
            tuple[float, float]
        ] = []

        for row, col in cells:
            point = self.terrain.rowcol_to_point(
                row,
                col,
            )

            coordinate = (
                float(point.x),
                float(point.y),
            )

            if (
                not coordinates
                or coordinate != coordinates[-1]
            ):
                coordinates.append(coordinate)

        if len(coordinates) < 2:
            return None

        line = LineString(coordinates)

        if (
            line.is_empty
            or not line.is_valid
            or line.length <= 0
        ):
            return None

        return line

    def _apply_results(
        self,
        results: pd.DataFrame,
    ) -> gpd.GeoDataFrame:
        """Adds proposed strict/recovery routing attributes to catchments."""
        updated = self.catchments.copy()

        text_fields = (
            "proposed_source_id",
            "proposed_catchment_id",
            "orphan_routing_method",
            "orphan_routing_confidence",
            "orphan_routing_status",
            "orphan_routing_notes",
        )

        for field_name in text_fields:
            updated[field_name] = None

        integer_defaults = {
            "proposed_segment_id": 0,
            "orphan_outlet_row": -9999,
            "orphan_outlet_col": -9999,
            "orphan_trace_cells": 0,
            "strict_trace_cells": 0,
            "sink_recovery_attempted": 0,
            "sink_recovery_success": 0,
            "sink_recovery_exit_row": -9999,
            "sink_recovery_exit_col": -9999,
            "sink_recovery_flat_only": 0,
            "orphan_manual_review": 0,
        }

        for field_name, value in integer_defaults.items():
            updated[field_name] = value

        float_defaults = {
            "outlet_accum_cells": -9999.0,
            "orphan_trace_length_m": 0.0,
            "sink_recovery_max_rise_m": -9999.0,
            "sink_recovery_distance_m": -9999.0,
            "sink_recovery_start_elev_m": -9999.0,
            "sink_recovery_exit_elev_m": -9999.0,
        }

        for field_name, value in float_defaults.items():
            updated[field_name] = value

        updated["orphan_manual_review"] = (
            updated["manual_review"]
            .fillna(0)
            .astype(int)
        )

        updated["orphan_routing_status"] = (
            "not_orphan"
        )

        if results.empty:
            return updated

        result_lookup = results.set_index(
            "catchment_id",
            drop=False,
        )

        for index, catchment in updated.iterrows():
            catchment_id = str(
                catchment["catchment_id"]
            )

            if catchment_id not in result_lookup.index:
                continue

            result = result_lookup.loc[
                catchment_id
            ]

            updated.at[
                index,
                "orphan_routing_method",
            ] = result["routing_method"]

            updated.at[
                index,
                "orphan_routing_confidence",
            ] = result["routing_confidence"]

            updated.at[
                index,
                "orphan_routing_notes",
            ] = result["routing_notes"]

            updated.at[
                index,
                "orphan_routing_status",
            ] = (
                "proposed_assignment"
                if pd.notna(
                    result["assigned_source_id"]
                )
                else "sink_or_unresolved"
            )

            if result["routing_method"] == "circular_routing":
                updated.at[
                    index,
                    "orphan_routing_status",
                ] = "circular_routing"

            for result_field, output_field in (
                (
                    "orphan_outlet_row",
                    "orphan_outlet_row",
                ),
                (
                    "orphan_outlet_col",
                    "orphan_outlet_col",
                ),
                (
                    "trace_cell_count",
                    "orphan_trace_cells",
                ),
                (
                    "strict_trace_cell_count",
                    "strict_trace_cells",
                ),
                (
                    "sink_recovery_attempted",
                    "sink_recovery_attempted",
                ),
                (
                    "sink_recovery_success",
                    "sink_recovery_success",
                ),
                (
                    "sink_recovery_exit_row",
                    "sink_recovery_exit_row",
                ),
                (
                    "sink_recovery_exit_col",
                    "sink_recovery_exit_col",
                ),
                (
                    "sink_recovery_flat_only",
                    "sink_recovery_flat_only",
                ),
                (
                    "manual_review",
                    "orphan_manual_review",
                ),
            ):
                value = result[result_field]

                if pd.notna(value):
                    updated.at[
                        index,
                        output_field,
                    ] = int(value)

            for result_field, output_field in (
                (
                    "outlet_accumulation_cells",
                    "outlet_accum_cells",
                ),
                (
                    "trace_length_m",
                    "orphan_trace_length_m",
                ),
                (
                    "sink_recovery_max_rise_m",
                    "sink_recovery_max_rise_m",
                ),
                (
                    "sink_recovery_distance_m",
                    "sink_recovery_distance_m",
                ),
                (
                    "sink_recovery_start_elev_m",
                    "sink_recovery_start_elev_m",
                ),
                (
                    "sink_recovery_exit_elev_m",
                    "sink_recovery_exit_elev_m",
                ),
            ):
                value = result[result_field]

                if pd.notna(value):
                    updated.at[
                        index,
                        output_field,
                    ] = float(value)

            if pd.notna(
                result["assigned_source_id"]
            ):
                updated.at[
                    index,
                    "proposed_source_id",
                ] = str(
                    result["assigned_source_id"]
                )

                if pd.notna(result["assigned_segment_id"]):
                    updated.at[
                        index,
                        "proposed_segment_id",
                    ] = int(
                        result["assigned_segment_id"]
                    )

                if pd.notna(result["assigned_catchment_id"]):
                    updated.at[
                        index,
                        "proposed_catchment_id",
                    ] = str(
                        result["assigned_catchment_id"]
                    )

        return updated

    def _validate_inputs(
        self,
    ) -> None:
        """Validates required catchment and segment input schema."""
        missing_catchment = (
            self.REQUIRED_CATCHMENT_FIELDS
            - set(self.catchments.columns)
        )

        if missing_catchment:
            raise KeyError(
                "model_catchments is missing required field(s): "
                f"{sorted(missing_catchment)}"
            )

        missing_segment = (
            self.REQUIRED_SEGMENT_FIELDS
            - set(self.segments.columns)
        )

        if missing_segment:
            raise KeyError(
                "model_segments is missing required field(s): "
                f"{sorted(missing_segment)}"
            )

        if self.catchments.crs is None:
            raise ValueError(
                "model_catchments has no CRS."
            )

        if self.segments.crs is None:
            raise ValueError(
                "model_segments has no CRS."
            )

    def write_outputs(
        self,
        outlet_points: gpd.GeoDataFrame,
        routing_paths: gpd.GeoDataFrame,
        recovery_paths: gpd.GeoDataFrame,
        results: pd.DataFrame,
        updated_catchments: gpd.GeoDataFrame,
    ) -> None:
        """Writes spatial QA layers and tabular outputs."""
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_gpkg = self.config.output_gpkg_path

        if not outlet_points.empty:
            candidate_fields = [
                "catchment_id",
                "source_id",
                "assigned_source_id",
                "assigned_segment_id",
                "assigned_catchment_id",
                "routing_method",
                "routing_confidence",
                "orphan_outlet_row",
                "orphan_outlet_col",
                "outlet_accumulation_cells",
                "strict_trace_cell_count",
                "trace_cell_count",
                "trace_length_m",
                "sink_recovery_attempted",
                "sink_recovery_success",
                "sink_recovery_max_rise_m",
                "sink_recovery_distance_m",
                "manual_review",
                "routing_notes",
                "geometry",
            ]

            candidate_out = outlet_points[
                [
                    field
                    for field in candidate_fields
                    if field in outlet_points.columns
                ]
            ].copy()

            candidate_out = validate_output_layer(
                candidate_out,
                layer_name="orphan_routing_candidates",
                expected_geometry_types="Point",
            )

            write_layer(
                candidate_out,
                output_gpkg,
                "orphan_routing_candidates",
            )

        if not routing_paths.empty:
            path_fields = [
                "catchment_id",
                "source_id",
                "assigned_source_id",
                "assigned_segment_id",
                "assigned_catchment_id",
                "routing_method",
                "routing_confidence",
                "strict_trace_cell_count",
                "trace_cell_count",
                "trace_length_m",
                "sink_recovery_attempted",
                "sink_recovery_success",
                "sink_recovery_max_rise_m",
                "sink_recovery_distance_m",
                "manual_review",
                "routing_notes",
                "geometry",
            ]

            paths_out = routing_paths[
                [
                    field
                    for field in path_fields
                    if field in routing_paths.columns
                ]
            ].copy()

            paths_out = validate_output_layer(
                paths_out,
                layer_name="orphan_routing_paths",
                expected_geometry_types="LineString",
            )

            write_layer(
                paths_out,
                output_gpkg,
                "orphan_routing_paths",
            )

        if not recovery_paths.empty:
            recovery_fields = [
                "catchment_id",
                "source_id",
                "assigned_source_id",
                "assigned_segment_id",
                "assigned_catchment_id",
                "routing_method",
                "routing_confidence",
                "sink_recovery_attempted",
                "sink_recovery_success",
                "sink_recovery_max_rise_m",
                "sink_recovery_distance_m",
                "sink_recovery_exit_row",
                "sink_recovery_exit_col",
                "sink_recovery_start_elev_m",
                "sink_recovery_exit_elev_m",
                "sink_recovery_flat_only",
                "manual_review",
                "routing_notes",
                "geometry",
            ]

            recovery_out = recovery_paths[
                [
                    field
                    for field in recovery_fields
                    if field in recovery_paths.columns
                ]
            ].copy()

            recovery_out = validate_output_layer(
                recovery_out,
                layer_name="orphan_routing_recovery_paths",
                expected_geometry_types="LineString",
            )

            write_layer(
                recovery_out,
                output_gpkg,
                "orphan_routing_recovery_paths",
            )

        catchment_fields = [
            "catchment_id",
            "source_id",
            "to_source_id",
            "segment_id",
            "to_segment_id",
            "routing_target",
            "routing_status",
            "routing_detail",
            "is_active_hru",
            "is_outlet",
            "is_sink",
            "manual_review",
            "area_m2",
            "waterbody_frac",
            "source_type",
            "source_name",
            "drainage_area_km2",
            "stream_order",
            "divergence",
            "proposed_source_id",
            "proposed_segment_id",
            "proposed_catchment_id",
            "orphan_routing_method",
            "orphan_routing_confidence",
            "orphan_routing_status",
            "orphan_routing_notes",
            "orphan_outlet_row",
            "orphan_outlet_col",
            "outlet_accum_cells",
            "strict_trace_cells",
            "orphan_trace_cells",
            "orphan_trace_length_m",
            "sink_recovery_attempted",
            "sink_recovery_success",
            "sink_recovery_max_rise_m",
            "sink_recovery_distance_m",
            "sink_recovery_exit_row",
            "sink_recovery_exit_col",
            "sink_recovery_start_elev_m",
            "sink_recovery_exit_elev_m",
            "sink_recovery_flat_only",
            "orphan_manual_review",
            "geometry",
        ]

        catchment_out = updated_catchments[
            [
                field
                for field in catchment_fields
                if field in updated_catchments.columns
            ]
        ].copy()

        catchment_out = validate_output_layer(
            catchment_out,
            layer_name="model_catchments_orphan_routed",
            expected_geometry_types=(
                "Polygon",
                "MultiPolygon",
            ),
        )

        write_layer(
            catchment_out,
            output_gpkg,
            "model_catchments_orphan_routed",
        )

        results.to_csv(
            self.config.output_dir
            / "orphan_routing_results.csv",
            index=False,
        )

        summary = self._make_summary(results)

        (
            self.config.output_dir
            / "orphan_routing_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


    def plot_orphan_routing_map(
        self,
        results: pd.DataFrame,
        zoom_to_orphans: bool = False,
        zoom_buffer_m: float = 500.0,
        show: bool = True,
    ) -> None:
        """Maps orphan catchments by status with arrows to proposed receivers.

        Every catchment boundary is drawn solid. Orphan catchments are
        outlined and lightly filled by their original ``routing_status``.
        Each proposed routing fix is an arrow from the orphan's centroid to
        the centroid of the receiving HRU (the active HRU that owns the
        assigned segment). If that segment has no HRU, for example a
        tributary with no catchment of its own, the arrow ends at the
        segment's midpoint and is marked with a diamond.

        Arrow style shows how the fix was found: solid black for strict D8
        traces and dashed green for local elevation-buffer recovery.
        Orphans with no proposed receiver get a marker at their centroid.

        Args:
            results: Orphan routing result table from ``route_orphans``.
            zoom_to_orphans: Crop the view around the orphan catchments.
            zoom_buffer_m: Margin around orphans when zooming, in meters.
            show: Display the figure after saving it.
        """
        if results.empty:
            tqdm.write("No orphan results to plot.")
            return

        status_colors = {
            "sink": "#D55E00",
            "missing_flowline": "#CC79A7",
            "downstream_missing": "#E69F00",
            "outside_active_network": "#0072B2",
        }

        result_fields = [
            "catchment_id",
            "assigned_segment_id",
            "assigned_catchment_id",
            "routing_method",
            "routing_confidence",
        ]

        orphans = self.orphans[
            ["catchment_id", "routing_status", "geometry"]
        ].copy()
        orphans["catchment_id"] = orphans["catchment_id"].astype(str)

        result_table = results[result_fields].copy()
        result_table["catchment_id"] = result_table[
            "catchment_id"
        ].astype(str)

        orphans = orphans.merge(
            result_table,
            on="catchment_id",
            how="left",
        )

        hru_by_segment: dict[int, BaseGeometry] = {}

        for segment_id, geometry in zip(
            self.active_hrus["segment_id"],
            self.active_hrus.geometry,
        ):
            if pd.notna(segment_id):
                hru_by_segment.setdefault(int(segment_id), geometry)

        catchment_by_id = {
            str(c): g
            for c, g in zip(
                self.catchments["catchment_id"],
                self.catchments.geometry,
            )
        }

        segment_geometry = {
            int(segment_id): geometry
            for segment_id, geometry in zip(
                self.segments["segment_id"],
                self.segments.geometry,
            )
            if pd.notna(segment_id)
        }

        figure, axis = plt.subplots(figsize=(16, 16))

        self.active_hrus.plot(
            ax=axis,
            facecolor="#F0F0F0",
            edgecolor="none",
            zorder=0,
        )

        self.catchments.boundary.plot(
            ax=axis,
            color="#8C8C8C",
            linewidth=0.45,
            linestyle="-",
            zorder=1,
        )

        self.segments.plot(
            ax=axis,
            color="#7FB2D9",
            linewidth=0.7,
            linestyle="--",
            zorder=2,
        )

        legend_handles: list[Any] = [
            Line2D(
                [0], [0], color="#8C8C8C", linewidth=1.5,
                linestyle="-", label="Catchment boundary",
            ),
            Patch(
                facecolor="#F0F0F0", edgecolor="#8C8C8C",
                label="Active HRU",
            ),
            Line2D(
                [0], [0], color="#7FB2D9", linewidth=1.5,
                linestyle="--", label="Active segment",
            ),
        ]

        for status, color in status_colors.items():
            subset = orphans[orphans["routing_status"] == status]

            if subset.empty:
                continue

            subset.plot(
                ax=axis,
                facecolor=color,
                edgecolor="none",
                alpha=0.28,
                zorder=3,
            )

            subset.boundary.plot(
                ax=axis,
                color=color,
                linewidth=1.8,
                linestyle="-",
                zorder=4,
            )

            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                    label=f"{status} (n={len(subset):,})",
                )
            )

        unresolved_methods = {
            "dem_unresolved",
            "no_valid_dem_cell",
            "dem_elevation_buffer_unresolved",
            "unresolved",
            "circular_routing",
        }

        n_strict = 0
        n_buffer = 0
        n_sink = 0
        n_unresolved = 0
        n_segment_end = 0

        for row in orphans.itertuples():
            start = row.geometry.centroid
            method = (
                "" if pd.isna(row.routing_method)
                else str(row.routing_method)
            )
            segment_id = row.assigned_segment_id

            catchment_target = row.assigned_catchment_id

            if pd.isna(segment_id) and pd.isna(catchment_target):
                if method in unresolved_methods:
                    n_unresolved += 1
                    axis.scatter(
                        start.x, start.y, marker="x", s=60,
                        color="black", linewidths=1.6, zorder=8,
                    )
                else:
                    n_sink += 1
                    axis.scatter(
                        start.x, start.y, marker="o", s=55,
                        facecolor="white", edgecolor="black",
                        linewidths=1.4, zorder=8,
                    )
                continue

            if pd.notna(catchment_target):
                receiver_geometry = catchment_by_id[
                    str(catchment_target)
                ]
            else:
                segment_id = int(segment_id)
                receiver_geometry = hru_by_segment.get(segment_id)

            ends_at_segment = receiver_geometry is None

            if ends_at_segment:
                line = segment_geometry.get(segment_id)

                if line is None:
                    continue

                end = line.interpolate(0.5, normalized=True)
                n_segment_end += 1
            else:
                end = receiver_geometry.centroid

            is_buffer = "buffer" in method

            if is_buffer:
                n_buffer += 1
            else:
                n_strict += 1

            axis.annotate(
                "",
                xy=(end.x, end.y),
                xytext=(start.x, start.y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#00794F" if is_buffer else "#111111",
                    "linewidth": 1.5,
                    "linestyle": "--" if is_buffer else "-",
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 14,
                },
                zorder=9,
            )

            axis.scatter(
                start.x, start.y, s=14, color="black", zorder=8,
            )

            if ends_at_segment:
                axis.scatter(
                    end.x, end.y, marker="D", s=34,
                    facecolor="#7FB2D9", edgecolor="black",
                    linewidths=0.8, zorder=10,
                )

        legend_handles.extend(
            [
                Line2D(
                    [0], [0], color="#111111", linewidth=2.0,
                    linestyle="-",
                    label=f"Fix: strict D8 trace (n={n_strict:,})",
                ),
                Line2D(
                    [0], [0], color="#00794F", linewidth=2.0,
                    linestyle="--",
                    label=f"Fix: elevation-buffer recovery (n={n_buffer:,})",
                ),
                Line2D(
                    [0], [0], marker="D", color="white",
                    markerfacecolor="#7FB2D9", markeredgecolor="black",
                    markersize=8,
                    label=(
                        "Receiver has no HRU: arrow ends at segment "
                        f"midpoint (n={n_segment_end:,})"
                    ),
                ),
                Line2D(
                    [0], [0], marker="o", color="white",
                    markerfacecolor="white", markeredgecolor="black",
                    markersize=8,
                    label=f"No receiver, likely true sink (n={n_sink:,})",
                ),
                Line2D(
                    [0], [0], marker="x", color="black", linestyle="",
                    markersize=8,
                    label=f"Unresolved (n={n_unresolved:,})",
                ),
            ]
        )

        axis.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            fontsize=9,
        )

        if zoom_to_orphans:
            minx, miny, maxx, maxy = orphans.total_bounds

            axis.set_xlim(minx - zoom_buffer_m, maxx + zoom_buffer_m)
            axis.set_ylim(miny - zoom_buffer_m, maxy + zoom_buffer_m)

        axis.set_title(
            "Orphan Catchment Routing: proposed fixes\n"
            f"{len(orphans):,} orphans | "
            f"{n_strict + n_buffer:,} with a proposed receiver",
            fontsize=15,
            fontweight="bold",
        )

        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="box")
        figure.tight_layout()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        figure.savefig(
            self.config.output_dir / "orphan_routing_map.png",
            dpi=250,
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(figure)


    @staticmethod
    def _make_summary(
        results: pd.DataFrame,
    ) -> dict[str, Any]:
        """Creates JSON-serializable strict/recovery routing summary."""
        if results.empty:
            return {
                "n_orphans_considered": 0,
                "n_proposed_assignments": 0,
                "n_unresolved_or_sinks": 0,
                "n_sink_recovery_attempted": 0,
                "n_sink_recovery_success": 0,
                "routing_method_counts": {},
                "routing_confidence_counts": {},
            }

        return {
            "n_orphans_considered": int(
                len(results)
            ),
            "n_proposed_assignments": int(
                results["assigned_source_id"]
                .notna()
                .sum()
            ),
            "n_unresolved_or_sinks": int(
                results["assigned_source_id"]
                .isna()
                .sum()
            ),
            "n_sink_recovery_attempted": int(
                results["sink_recovery_attempted"]
                .sum()
            ),
            "n_sink_recovery_success": int(
                results["sink_recovery_success"]
                .sum()
            ),
            "routing_method_counts": {
                str(method): int(count)
                for method, count in results[
                    "routing_method"
                ]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            },
            "routing_confidence_counts": {
                str(confidence): int(count)
                for confidence, count in results[
                    "routing_confidence"
                ]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            },
        }


class OrphanAdoptionConfig:
    """Stores settings for applying reviewed ``proposed_source_id`` values.

    Args:
        gpkg_path: Datasource holding the prepared ``model_segments`` layer
            (read-only here).
        output_gpkg_path: Datasource holding the edited orphan layer and
            receiving the new layers.
        output_dir: Folder for CSVs and the JSON report.
        edited_layer: Orphan-routing layer that was manually edited.
        segment_layer: Active segment layer.
        output_catchment_layer: New full catchment layer to write.
        output_hru_layer: New trimmed PRMS HRU layer to write.
        results_csv_path: Original automatic results table, used only to
            label edits as auto-accepted or manual. Defaults to
            ``output_dir/orphan_routing_results.csv``.
        orphan_statuses: ``routing_status`` values treated as orphans.
        strict: Raise on invalid edits or routing cycles instead of skipping.
        extra_hru_fields: Additional catchment fields to keep in the HRU
            layer beyond the core PRMS fields.
        lake_waterbody_fraction: If set, HRUs with ``waterbody_frac`` at or
            above this value get ``hru_type`` 2 (lake). Default None keeps
            every HRU as land (1).
    """

    ACRE_M2 = 4046.8564224

    def __init__(
        self,
        gpkg_path: Path,
        output_gpkg_path: Path,
        output_dir: Path,
        edited_layer: str = "model_catchments_orphan_routed",
        segment_layer: str = "model_segments",
        output_catchment_layer: str = "model_catchments_orphan_adopted",
        output_hru_layer: str = "model_hrus_orphan_adopted",
        results_csv_path: Path | None = None,
        orphan_statuses: tuple[str, ...] = (
            OrphanRoutingConfig.DEFAULT_ORPHAN_STATUSES
        ),
        strict: bool = True,
        extra_hru_fields: tuple[str, ...] = (),
        lake_waterbody_fraction: float | None = None,
    ) -> None:
        """Initializes adoption settings."""
        self.gpkg_path = Path(gpkg_path)
        self.output_gpkg_path = Path(output_gpkg_path)
        self.output_dir = Path(output_dir)
        self.edited_layer = str(edited_layer)
        self.segment_layer = str(segment_layer)
        self.output_catchment_layer = str(output_catchment_layer)
        self.output_hru_layer = str(output_hru_layer)
        self.results_csv_path = (
            Path(results_csv_path)
            if results_csv_path is not None
            else self.output_dir / "orphan_routing_results.csv"
        )
        self.orphan_statuses = tuple(orphan_statuses)
        self.strict = bool(strict)
        self.extra_hru_fields = tuple(extra_hru_fields)
        self.lake_waterbody_fraction = lake_waterbody_fraction


class OrphanRoutingAdopter:
    """Applies reviewed routing and builds the PRMS HRU layer.

    Every catchment is retained and every catchment becomes an HRU, including
    sinks.

    Catchment layer (``output_catchment_layer``):
        - Each orphan with a ``proposed_source_id`` gets
          ``to_source_id = proposed_source_id``.
        - Every catchment whose ``to_source_id`` points to a known source
          gets ``routing_status = "routed"`` and a filled ``to_segment_id``.
        - ``is_active_hru`` is 1 for all catchments (the prior value is kept
          in ``was_active_hru``), and the old status in
          ``orphan_original_status``.

    HRU layer (``output_hru_layer``), trimmed to PRMS fields:
        hru_id, catchment_id, source_id, hru_segment, hru_area (acres),
        hru_lat, hru_lon, hru_type, geometry.

    ``hru_segment`` is the model segment the HRU contributes lateral flow to:
    its own segment if its flowline is in the active network, otherwise the
    first segment reached by following ``to_source_id``. HRUs whose chain
    never reaches a segment (true sinks) get 0.
    """

    REQUIRED_EDITED_FIELDS = (
        "catchment_id",
        "source_id",
        "routing_status",
        "proposed_source_id",
    )

    REQUIRED_SEGMENT_FIELDS = (
        "segment_id",
        "source_id",
    )

    def __init__(self, config: OrphanAdoptionConfig) -> None:
        """Initializes the adopter."""
        self.config = config

    def run(self) -> dict[str, Any]:
        """Loads, adopts, validates, and writes all outputs."""
        edited, segments, results = self.load()

        decisions, final, hrus, report = self.adopt(
            edited=edited,
            segments=segments,
            results=results,
        )

        self.write_outputs(
            decisions=decisions,
            final=final,
            hrus=hrus,
            report=report,
        )

        return report

    def load(
        self,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        pd.DataFrame | None,
    ]:
        """Reads the edited orphan layer, segments, and original results."""
        edited = read_layer(
            self.config.output_gpkg_path,
            self.config.edited_layer,
        )

        segments = read_layer(
            self.config.gpkg_path,
            self.config.segment_layer,
        )

        results = None

        if self.config.results_csv_path.exists():
            results = pd.read_csv(
                self.config.results_csv_path,
                dtype=str,
            )
            results.columns = [str(c).lower() for c in results.columns]
        else:
            tqdm.write(
                "Original results CSV not found; edits will not be "
                "labeled as auto-accepted versus manual."
            )

        return edited, segments, results

    @staticmethod
    def _clean_id(value: Any) -> str | None:
        """Normalizes an ID to a trimmed string, or None if blank."""
        if value is None or pd.isna(value):
            return None

        text = str(value).strip()

        if text == "" or text.lower() in {"none", "nan", "<na>"}:
            return None

        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]

        return text

    @staticmethod
    def _check_fields(
        frame: pd.DataFrame,
        required: tuple[str, ...],
        label: str,
    ) -> None:
        """Raises if required fields are absent."""
        missing = set(required) - set(frame.columns)

        if missing:
            raise KeyError(
                f"{label} layer is missing required fields: "
                f"{sorted(missing)}"
            )

    def _drop_blank_geometry(
        self,
        edited: gpd.GeoDataFrame,
    ) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
        """Drops blank rows with no geometry and no catchment ID.

        A feature with a catchment ID but no geometry is a damaged
        catchment, so it raises instead of being silently dropped.
        """
        no_geometry = edited.geometry.isna() | edited.geometry.is_empty

        if not no_geometry.any():
            return edited, []

        id_fields = [
            field
            for field in (
                "catchment_id",
                "source_id",
                "routing_status",
                "proposed_source_id",
            )
            if field in edited.columns
        ]

        blank = edited.loc[no_geometry, id_fields].copy()
        has_id = blank["catchment_id"].map(self._clean_id).notna()

        if has_id.any():
            raise ValueError(
                "These catchments have no geometry. Restore them from the "
                "original layer so no catchment is lost:\n"
                f"{blank[has_id].to_string(index=True)}"
            )

        tqdm.write(
            f"WARNING: dropping {len(blank):,} blank feature(s) with no "
            f"geometry and no catchment_id:\n{blank.to_string(index=True)}"
        )

        return (
            edited.loc[~no_geometry].copy(),
            blank.astype(str).to_dict("records"),
        )

    @staticmethod
    def _cycle_sources(next_map: dict[str, str]) -> set[str]:
        """Returns every source that lies on a routing cycle."""
        state: dict[str, int] = {}
        on_cycle: set[str] = set()

        for start in next_map:
            if state.get(start) == 2:
                continue

            path: list[str] = []
            node: str | None = start

            while node is not None and state.get(node, 0) == 0:
                state[node] = 1
                path.append(node)
                node = next_map.get(node)

            if node is not None and state.get(node) == 1:
                on_cycle.update(path[path.index(node):])

            for visited in path:
                state[visited] = 2

        return on_cycle

    def adopt(
        self,
        edited: gpd.GeoDataFrame,
        segments: gpd.GeoDataFrame,
        results: pd.DataFrame | None,
    ) -> tuple[
        pd.DataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        dict[str, Any],
    ]:
        """Applies proposed source IDs and builds the HRU layer.

        Returns:
            Per-orphan decision table, full final catchment layer, trimmed
            PRMS HRU layer, and a JSON-compatible QA report.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If a catchment is damaged, or strict validation
                finds invalid edits or routing cycles.
        """
        self._check_fields(edited, self.REQUIRED_EDITED_FIELDS, "edited")
        self._check_fields(segments, self.REQUIRED_SEGMENT_FIELDS, "segments")

        edited = edited.copy()
        n_input = int(len(edited))

        edited, dropped_rows = self._drop_blank_geometry(edited)

        segment_by_source = {
            self._clean_id(source): int(segment)
            for source, segment in zip(
                segments["source_id"],
                pd.to_numeric(segments["segment_id"]),
            )
            if self._clean_id(source) is not None
        }

        own_source = edited["source_id"].map(self._clean_id)
        known_sources = set(own_source.dropna()) | set(segment_by_source)

        auto_source: dict[str, str | None] = {}

        if results is not None and "assigned_source_id" in results.columns:
            auto_source = {
                str(c).strip(): self._clean_id(a)
                for c, a in zip(
                    results["catchment_id"],
                    results["assigned_source_id"],
                )
            }

        existing_next: dict[str, str] = {}

        if "to_source_id" in edited.columns:
            for source, target in zip(
                own_source, edited["to_source_id"].map(self._clean_id)
            ):
                if source is not None and target is not None:
                    existing_next[source] = target

        orphan_mask = edited["routing_status"].isin(
            self.config.orphan_statuses
        )

        records: list[dict[str, Any]] = []

        for index, row in edited[orphan_mask].iterrows():
            catchment_id = str(row["catchment_id"]).strip()
            own = self._clean_id(row["source_id"])
            target = self._clean_id(row["proposed_source_id"])

            issue = None

            if target is not None:
                if target == own:
                    issue = "self_route"
                elif target not in known_sources:
                    issue = "unknown_source_id"

            records.append(
                {
                    "_index": index,
                    "catchment_id": catchment_id,
                    "source_id": own,
                    "original_routing_status": str(row["routing_status"]),
                    "auto_source_id": auto_source.get(catchment_id),
                    "final_to_source_id": target,
                    "final_to_segment_id": (
                        segment_by_source.get(target)
                        if target is not None
                        else None
                    ),
                    "issue": issue,
                }
            )

        proposed_next = {
            r["source_id"]: r["final_to_source_id"]
            for r in records
            if r["issue"] is None
            and r["final_to_source_id"] is not None
            and r["source_id"] is not None
        }

        next_map = {**existing_next, **proposed_next}

        cycle_sources = self._cycle_sources(next_map)

        for record in records:
            if record["issue"] is not None:
                continue

            node = record["source_id"]
            seen: set[str] = set()

            while (
                record["final_to_source_id"] is not None
                and node is not None
                and node in next_map
                and node not in seen
            ):
                seen.add(node)
                node = next_map[node]

                if node in cycle_sources:
                    record["issue"] = "circular_routing"
                    break

        decisions = pd.DataFrame(
            records,
            columns=[
                "_index",
                "catchment_id",
                "source_id",
                "original_routing_status",
                "auto_source_id",
                "final_to_source_id",
                "final_to_segment_id",
                "issue",
            ],
        )

        decisions["adopted"] = (
            decisions["final_to_source_id"].notna()
            & decisions["issue"].isna()
        )

        decisions["adoption_origin"] = [
            self._origin(
                adopted=bool(adopted),
                issue=issue,
                final=final_source,
                auto=auto,
                have_results=results is not None,
            )
            for adopted, issue, final_source, auto in zip(
                decisions["adopted"],
                decisions["issue"],
                decisions["final_to_source_id"],
                decisions["auto_source_id"],
            )
        ]

        problems = decisions[decisions["issue"].notna()]

        if not problems.empty and self.config.strict:
            sample = problems[
                ["catchment_id", "source_id", "final_to_source_id", "issue"]
            ].head(10).to_string(index=False)

            raise ValueError(
                f"{len(problems):,} edited orphan record(s) are invalid. "
                "Fix them or set strict=False to skip them.\n"
                f"{sample}"
            )

        final = self._apply_routing(
            edited=edited,
            decisions=decisions,
            segment_by_source=segment_by_source,
            known_sources=known_sources,
        )

        final_next = self._next_map(final)
        final_cycles = self._cycle_sources(final_next)

        if final_cycles and self.config.strict:
            raise ValueError(
                f"{len(final_cycles):,} source(s) lie on a routing cycle "
                f"after adoption, e.g. {sorted(final_cycles)[:5]}"
            )

        hrus, hru_stats = self._build_hrus(
            final=final,
            segment_by_source=segment_by_source,
            next_map=final_next,
        )

        report = {
            "n_catchments_input": n_input,
            "n_catchments_output": int(len(final)),
            "n_hrus": int(len(hrus)),
            "dropped_blank_rows": dropped_rows,
            "n_orphans": int(len(decisions)),
            "n_adopted": int(decisions["adopted"].sum()),
            "n_orphans_left_unrouted": int((~decisions["adopted"]).sum()),
            "adoption_origin_counts": {
                str(k): int(v)
                for k, v in decisions["adoption_origin"]
                .value_counts()
                .items()
            },
            "issue_counts": {
                str(k): int(v)
                for k, v in decisions["issue"].value_counts().items()
            },
            "routing_status_counts": {
                str(k): int(v)
                for k, v in final["routing_status"].value_counts().items()
            },
            "n_sources_on_cycles": int(len(final_cycles)),
            "duplicate_catchment_source_ids": int(
                own_source.dropna().duplicated().sum()
            ),
            "n_segments": int(len(set(segment_by_source.values()))),
            **hru_stats,
        }

        return decisions.drop(columns="_index"), final, hrus, report

    @staticmethod
    def _origin(
        adopted: bool,
        issue: str | None,
        final: str | None,
        auto: str | None,
        have_results: bool,
    ) -> str:
        """Labels where an orphan's final routing came from."""
        if issue is not None:
            return "invalid_edit"

        if not have_results:
            return "adopted" if adopted else "unrouted"

        if adopted:
            if auto == final:
                return "auto_accepted"

            return "manual_changed" if auto is not None else "manual_added"

        return "manual_removed" if auto is not None else "unrouted"

    def _next_map(self, final: gpd.GeoDataFrame) -> dict[str, str]:
        """Builds source -> to_source from the final catchment layer."""
        sources = final["source_id"].map(self._clean_id)
        targets = final["to_source_id"].map(self._clean_id)

        return {
            s: t
            for s, t in zip(sources, targets)
            if s is not None and t is not None
        }

    def _apply_routing(
        self,
        edited: gpd.GeoDataFrame,
        decisions: pd.DataFrame,
        segment_by_source: dict[str, int],
        known_sources: set[str],
    ) -> gpd.GeoDataFrame:
        """Applies adopted routes and refreshes status for every catchment."""
        final = edited.copy()

        origin_by_index = dict(
            zip(decisions["_index"], decisions["adoption_origin"])
        )

        final["adoption_origin"] = [
            origin_by_index.get(index, "prepared") for index in final.index
        ]

        final["orphan_original_status"] = None

        orphan_mask = final["routing_status"].isin(
            self.config.orphan_statuses
        )

        final.loc[orphan_mask, "orphan_original_status"] = final.loc[
            orphan_mask, "routing_status"
        ]

        if "is_active_hru" in final.columns:
            final["was_active_hru"] = final["is_active_hru"]
        else:
            final["was_active_hru"] = 0

        for column in (
            "to_source_id",
            "routing_target",
            "routing_detail",
            "manual_review",
        ):
            if column not in final.columns:
                final[column] = None

        final["to_segment_id"] = (
            pd.to_numeric(final["to_segment_id"], errors="coerce")
            .astype("Int64")
            if "to_segment_id" in final.columns
            else pd.Series(pd.NA, index=final.index, dtype="Int64")
        )

        for row in decisions[decisions["adopted"]].itertuples():
            final.at[row._1, "to_source_id"] = row.final_to_source_id

        adopted_indexes = set(decisions.loc[decisions["adopted"], "_index"])

        for index in final.index:
            target = self._clean_id(final.at[index, "to_source_id"])

            if target is None or target not in known_sources:
                continue

            segment = segment_by_source.get(target)

            if segment is not None:
                final.at[index, "to_segment_id"] = int(segment)

            final.at[index, "routing_target"] = (
                f"segment {segment}"
                if segment is not None
                else f"source {target}"
            )

            previous = final.at[index, "orphan_original_status"]

            if index in adopted_indexes:
                final.at[index, "routing_detail"] = (
                    f"orphan routed to source {target} "
                    f"({final.at[index, 'adoption_origin']})"
                )
                final.at[index, "manual_review"] = 0
            elif previous is not None and not pd.isna(previous):
                final.at[index, "routing_detail"] = (
                    f"existing to_source_id {target} applied "
                    f"(was {previous})"
                )

            final.at[index, "routing_status"] = "routed"

        final["is_active_hru"] = 1

        return final

    def _resolve_segment(
        self,
        start: str | None,
        next_map: dict[str, str],
        segment_by_source: dict[str, int],
    ) -> int:
        """Follows to_source links to the first model segment, else 0."""
        node = next_map.get(start) if start is not None else None
        visited = {start}

        while node is not None and node not in visited:
            if node in segment_by_source:
                return int(segment_by_source[node])

            visited.add(node)
            node = next_map.get(node)

        return 0

    def _build_hrus(
        self,
        final: gpd.GeoDataFrame,
        segment_by_source: dict[str, int],
        next_map: dict[str, str],
    ) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
        """Builds the trimmed PRMS HRU layer from every catchment."""
        if final.crs is None or not final.crs.is_projected:
            raise ValueError(
                "Catchments must be in a projected CRS to calculate "
                "hru_area."
            )

        unit_name = str(final.crs.axis_info[0].unit_name).lower()

        if not unit_name.startswith("met"):
            raise ValueError(
                f"hru_area assumes a metric CRS; found unit '{unit_name}'."
            )

        own_source = final["source_id"].map(self._clean_id)

        hru_segment: list[int] = []
        own_count = 0
        chain_count = 0

        for source in own_source:
            if source is not None and source in segment_by_source:
                hru_segment.append(int(segment_by_source[source]))
                own_count += 1
                continue

            resolved = self._resolve_segment(
                source,
                next_map,
                segment_by_source,
            )
            chain_count += int(resolved > 0)
            hru_segment.append(resolved)

        centroids = gpd.GeoSeries(
            final.geometry.centroid,
            crs=final.crs,
        ).to_crs(4326)

        hru_type = np.ones(len(final), dtype=int)

        if (
            self.config.lake_waterbody_fraction is not None
            and "waterbody_frac" in final.columns
        ):
            fraction = pd.to_numeric(
                final["waterbody_frac"], errors="coerce"
            ).fillna(0.0)
            hru_type[
                (fraction >= self.config.lake_waterbody_fraction).to_numpy()
            ] = 2

        table = pd.DataFrame(
            {
                "catchment_id": final["catchment_id"].astype(str).values,
                "source_id": own_source.values,
                "hru_segment": hru_segment,
                "hru_area": (
                    final.geometry.area.values / self.config.ACRE_M2
                ),
                "hru_lat": np.round(centroids.y.values, 6),
                "hru_lon": np.round(centroids.x.values, 6),
                "hru_type": hru_type,
            }
        )

        for field in self.config.extra_hru_fields:
            if field in final.columns:
                table[field] = final[field].values

        hrus = gpd.GeoDataFrame(
            table,
            geometry=final.geometry.values,
            crs=final.crs,
        )

        hrus = hrus.sort_values(
            ["hru_segment", "source_id", "catchment_id"],
            na_position="last",
        ).reset_index(drop=True)

        hrus.insert(0, "hru_id", np.arange(1, len(hrus) + 1, dtype=int))

        used_segments = set(hrus["hru_segment"]) - {0}
        all_segments = set(segment_by_source.values())

        stats = {
            "hrus_with_own_segment": int(own_count),
            "hrus_routed_via_chain": int(chain_count),
            "hrus_with_no_segment_hru_segment_0": int(
                (hrus["hru_segment"] == 0).sum()
            ),
            "segments_without_hru": int(len(all_segments - used_segments)),
            "total_hru_area_acres": round(float(hrus["hru_area"].sum()), 2),
            "total_hru_area_km2": round(
                float(hrus["hru_area"].sum())
                * self.config.ACRE_M2
                / 1_000_000.0,
                3,
            ),
            "hrus_below_min_area_0_0001_acres": int(
                (hrus["hru_area"] < 0.0001).sum()
            ),
        }

        return hrus, stats

    def write_outputs(
        self,
        decisions: pd.DataFrame,
        final: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        report: dict[str, Any],
    ) -> None:
        """Writes the catchment layer, HRU layer, CSVs, and JSON report."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        catchment_out = validate_output_layer(
            final,
            layer_name=self.config.output_catchment_layer,
            expected_geometry_types=("Polygon", "MultiPolygon"),
        )

        write_layer(
            catchment_out,
            self.config.output_gpkg_path,
            self.config.output_catchment_layer,
        )

        hru_out = validate_output_layer(
            hrus,
            layer_name=self.config.output_hru_layer,
            expected_geometry_types=("Polygon", "MultiPolygon"),
        )

        write_layer(
            hru_out,
            self.config.output_gpkg_path,
            self.config.output_hru_layer,
        )

        hrus.drop(columns="geometry").to_csv(
            self.config.output_dir / "hrus_orphan_adopted.csv",
            index=False,
        )

        decisions.to_csv(
            self.config.output_dir / "orphan_adoption_decisions.csv",
            index=False,
        )

        (
            self.config.output_dir / "orphan_adoption_report.json"
        ).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        tqdm.write(
            f"Adopted {report['n_adopted']:,} of {report['n_orphans']:,} "
            f"orphans. HRUs written: {report['n_hrus']:,} "
            f"({report['hrus_with_no_segment_hru_segment_0']:,} with "
            f"hru_segment 0)."
        )


def main() -> None:
    """Runs strict D8 orphan routing plus local sink recovery."""
    config = OrphanRoutingConfig(
        # Input datasource: read-only in this workflow.
        gpkg_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\PlusHR_UG.gdb"
        ),

        output_gpkg_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\PlusHR_UG.gdb"
        ),

        hydrodem_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\HR_hydrodem.tif"
        ),

        # Reuse strict terrain products already generated by this module.
        flow_direction_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\orphan_routing"
            r"\hydrodem_d8_direction_internal.tif"
        ),

        flow_accumulation_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\orphan_routing"
            r"\hydrodem_d8_accumulation_cells.tif"
        ),

        flow_direction_encoding="internal",

        output_dir=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\orphan_routing"
        ),

        catchment_layer="model_catchments",
        segment_layer="model_segments",

        orphan_statuses=(
            "missing_flowline",
            "sink",
            "downstream_missing",
            "outside_active_network",
        ),

        min_boundary_buffer_m=75.0,
        minimum_accumulation_cells=1.0,
        max_trace_steps=100_000,
        segment_search_distance_m=100.0,
        catchment_search_distance_m=75.0,

        # Local recovery settings for the 10 m DEM.
        enable_sink_recovery=True,
        flat_elevation_tolerance_m=1.0,
        max_breach_rise_m=4.0,
        max_breach_distance_m=100.0,
        minimum_exit_drop_m=0.1,

        # Existing FDR/FAC are supplied above, so do not rewrite them.
        write_flow_products=False,
    )

    print("=" * 78)
    print("ORPHAN CATCHMENT ROUTING")
    print("=" * 78)
    print(f"Input datasource: {config.gpkg_path}")
    print(f"Output GeoPackage: {config.output_gpkg_path}")
    print(f"HydroDEM: {config.hydrodem_path}")
    print(f"Flow direction: {config.flow_direction_path}")
    print(
        f"Flow accumulation: "
        f"{config.flow_accumulation_path}"
    )
    print(
        f"Direction encoding: "
        f"{config.flow_direction_encoding}"
    )
    print(f"Output directory: {config.output_dir}")
    print(
        "Sink recovery: "
        f"{config.enable_sink_recovery}"
    )
    print(
        "Recovery limits: "
        f"flat={config.flat_elevation_tolerance_m:.2f} m, "
        f"rise={config.max_breach_rise_m:.2f} m, "
        f"distance={config.max_breach_distance_m:.1f} m, "
        f"exit_drop={config.minimum_exit_drop_m:.2f} m"
    )

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    catchments = read_layer(
        config.gpkg_path,
        config.catchment_layer,
    )

    segments = read_layer(
        config.gpkg_path,
        config.segment_layer,
    )

    terrain = D8Terrain(
        hydrodem_path=config.hydrodem_path,
        flow_direction_path=config.flow_direction_path,
        flow_accumulation_path=(
            config.flow_accumulation_path
        ),
        flow_direction_encoding=(
            config.flow_direction_encoding
        ),
    )

    print(
        "\nLoading or calculating basin-scale "
        "flow direction and accumulation."
    )

    terrain.build()

    if config.write_flow_products:
        print("\nWriting D8 terrain QA products.")

        terrain.write_products(
            config.output_dir
        )

    router = OrphanCatchmentRouter(
        config=config,
        terrain=terrain,
        catchments=catchments,
        segments=segments,
    )

    print(
        f"\nRouting {len(router.orphans):,} orphan "
        "catchment(s)."
    )

    (
        outlet_points,
        routing_paths,
        recovery_paths,
        results,
        updated_catchments,
    ) = router.route_orphans()

    print(
        "\nWriting orphan-routing outputs to the "
        "separate output GeoPackage."
    )

    router.write_outputs(
        outlet_points=outlet_points,
        routing_paths=routing_paths,
        recovery_paths=recovery_paths,
        results=results,
        updated_catchments=updated_catchments,
    )

    if results.empty:
        print(
            "\nNo catchments matched the configured "
            "orphan statuses."
        )
    else:
        print("\nRouting results by method/confidence:")

        print(
            results[
                [
                    "routing_method",
                    "routing_confidence",
                ]
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nSink-recovery results: "
            f"attempted="
            f"{int(results['sink_recovery_attempted'].sum()):,}, "
            f"successful="
            f"{int(results['sink_recovery_success'].sum()):,}"
        )

    router.plot_orphan_routing_map(results=results)

    print("\n" + "=" * 78)
    print("ORPHAN CATCHMENT ROUTING COMPLETE")
    print("=" * 78)


def main_adopt() -> None:
    """Adopts manually edited orphan routing into new final layers."""
    domain = Path(
        r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
        r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
    )

    config = OrphanAdoptionConfig(
        gpkg_path=domain / "PlusHR_UG.gdb",
        output_gpkg_path=domain / "PlusHR_UG.gdb",
        output_dir=domain / "orphan_routing",
        strict=True,
    )

    OrphanRoutingAdopter(config).run()

if __name__ == "__main__":
    main_adopt()