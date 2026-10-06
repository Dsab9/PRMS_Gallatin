r"""Repair missing downstream flowline links before PRMS routing.

This module identifies prepared flowlines that lack a valid downstream
``to_source_id`` and proposes downstream topology repairs using:

    - known valid prepared-network links,
    - HydroNet Junction point geometry,
    - downstream endpoint orientation,
    - forward linear extensions,
    - candidate receiving flowlines,
    - associated catchment polygons,
    - cycle prevention.

The intended workflow is:

    prepared_flowlines
    prepared_catchments
    hydro_net_junctions
        ↓
    Repair_Flowline_Topology.py
        ↓
    prepared_flowlines_topology_repaired
    prepared_catchments_topology_repaired
        ↓
    Routing.py
        ↓
    model_segments
    model_hrus
    model_catchments
        ↓
    Route_Orphan_Catchments.py
        ↓
    Strict D8 + elevation-buffer recovery only for residual orphan cases.

The source prepared layers are never modified. The output layers retain all
original features and add proposed topology-repair fields.

Expected flowline fields:
    source_id
    to_source_id
    geometry

Expected catchment fields:
    catchment_id
    source_id
    geometry

HydroNet Junction fields are optional. The junction layer must at minimum
contain point geometry. If it also contains fields related to flowline IDs,
Hydroseq, node IDs, or junction types, they are retained in the QA output but
are not required for the geometric repair workflow.
"""

from __future__ import annotations

import json
import math
import os
import sys
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
import networkx as nx
import numpy as np
import pandas as pd
from fiona import listlayers, remove
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



class TopologyRepairConfig:
    """Stores settings for junction-guided flowline topology repair.

    Args:
        input_path: GeoPackage or File Geodatabase containing prepared layers
            and HydroNet Junctions.
        output_gpkg_path: GeoPackage receiving topology-review outputs.
        output_dir: Folder receiving CSV and JSON QA products.
        flowline_layer: Prepared flowline layer name.
        catchment_layer: Prepared catchment layer name.
        junction_layer: HydroNet Junction point layer name.
        work_crs: Projected CRS used for distance and angle calculations.
        extension_distance_m: Maximum forward extension length.
        junction_snap_distance_m: Maximum endpoint-to-junction distance used
            to consider an endpoint associated with a HydroNet Junction.
        receiver_search_distance_m: Search radius around the extension line for
            candidate receiving flowlines.
        receiver_endpoint_snap_distance_m: Maximum distance from extension end
            to receiving flowline endpoint for a high-confidence endpoint snap.
        maximum_turn_angle_degrees: Maximum allowed directional turn from the
            broken flowline's downstream heading to receiving flowline heading.
        require_receiver_reaches_active_network: Require a candidate receiver
            to have a valid downstream path in the original prepared graph.
        apply_medium_confidence_repairs: If True, write medium-confidence
            proposed links into repaired ``to_source_id``. High-confidence
            links are always applied; low-confidence links never are.
    """

    def __init__(
        self,
        input_path: str | Path,
        output_gpkg_path: str | Path,
        output_dir: str | Path | None = None,
        flowline_layer: str = "prepared_flowlines",
        catchment_layer: str = "prepared_catchments",
        junction_layer: str = "hydro_net_junctions",
        work_crs: str = "EPSG:5070",
        extension_distance_m: float = 100.0,
        junction_snap_distance_m: float = 15.0,
        receiver_search_distance_m: float = 25.0,
        receiver_endpoint_snap_distance_m: float = 25.0,
        maximum_turn_angle_degrees: float = 60.0,
        require_receiver_reaches_active_network: bool = True,
        apply_medium_confidence_repairs: bool = True,
    ) -> None:
        """Initializes topology-repair configuration."""
        self.input_path = Path(input_path)
        self.output_gpkg_path = Path(output_gpkg_path)

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.input_path.parent
            / "flowline_topology_repair"
        )

        self.flowline_layer = str(flowline_layer)
        self.catchment_layer = str(catchment_layer)
        self.junction_layer = str(junction_layer)
        self.work_crs = str(work_crs)

        self.extension_distance_m = float(
            extension_distance_m
        )

        self.junction_snap_distance_m = float(
            junction_snap_distance_m
        )

        self.receiver_search_distance_m = float(
            receiver_search_distance_m
        )

        self.receiver_endpoint_snap_distance_m = float(
            receiver_endpoint_snap_distance_m
        )

        self.maximum_turn_angle_degrees = float(
            maximum_turn_angle_degrees
        )

        self.require_receiver_reaches_active_network = bool(
            require_receiver_reaches_active_network
        )

        self.apply_medium_confidence_repairs = bool(
            apply_medium_confidence_repairs
        )

        self._validate()

    def _validate(self) -> None:
        """Validates topology-repair paths and parameter ranges."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input datasource does not exist: "
                f"{self.input_path}"
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

        if self.extension_distance_m <= 0:
            raise ValueError(
                "extension_distance_m must be greater than zero."
            )

        if self.junction_snap_distance_m <= 0:
            raise ValueError(
                "junction_snap_distance_m must be greater than zero."
            )

        if self.receiver_search_distance_m <= 0:
            raise ValueError(
                "receiver_search_distance_m must be greater than zero."
            )

        if self.receiver_endpoint_snap_distance_m <= 0:
            raise ValueError(
                "receiver_endpoint_snap_distance_m must be greater than zero."
            )

        if not (
            0
            < self.maximum_turn_angle_degrees
            <= 180
        ):
            raise ValueError(
                "maximum_turn_angle_degrees must be greater than zero "
                "and no greater than 180."
            )


# =============================================================================
# Datasource utilities
# =============================================================================


def lower_columns(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Returns a copy with lowercase non-geometry field names."""
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
    """Lists spatial layers in a GeoPackage or File Geodatabase."""
    if not datasource_path.exists():
        raise FileNotFoundError(
            f"Datasource does not exist: {datasource_path}"
        )

    return sorted(listlayers(datasource_path))


def read_layer(
    datasource_path: Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Reads one spatial layer and normalizes field names."""
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
) -> None:
    """Writes one non-empty GeoDataFrame to a GeoPackage layer."""
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

    if layer_name in existing_layers:
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


def valid_geometry_subset(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Returns non-null, non-empty, valid-geometry records."""
    return gdf.loc[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.is_valid
    ].copy()


# =============================================================================
# Geometry helpers
# =============================================================================


def line_endpoints(
    geometry: BaseGeometry,
) -> tuple[Point, Point]:
    """Returns first and last coordinates of a LineString/MultiLineString.

    MultiLineStrings are reduced to their longest component because prepared
    network flowlines should ordinarily be simple single-part reaches.

    Args:
        geometry: Line geometry.

    Returns:
        Start and end points.

    Raises:
        ValueError: If geometry cannot provide at least two coordinates.
    """
    if geometry.geom_type == "MultiLineString":
        geometry = max(
            geometry.geoms,
            key=lambda part: part.length,
        )

    coordinates = list(geometry.coords)

    if len(coordinates) < 2:
        raise ValueError(
            "Flowline geometry has fewer than two coordinates."
        )

    return (
        Point(coordinates[0]),
        Point(coordinates[-1]),
    )


def endpoint_heading(
    geometry: BaseGeometry,
    endpoint: str,
) -> tuple[float, float]:
    """Returns an outward unit vector at a selected line endpoint.

    Args:
        geometry: Input LineString or MultiLineString.
        endpoint: Either ``start`` or ``end``.

    Returns:
        Unit vector directed outward from the selected endpoint.

    Raises:
        ValueError: If endpoint is invalid or a direction cannot be calculated.
    """
    if geometry.geom_type == "MultiLineString":
        geometry = max(
            geometry.geoms,
            key=lambda part: part.length,
        )

    coordinates = list(geometry.coords)

    if len(coordinates) < 2:
        raise ValueError(
            "Flowline geometry has fewer than two coordinates."
        )

    if endpoint == "start":
        x0, y0 = coordinates[1]
        x1, y1 = coordinates[0]

    elif endpoint == "end":
        x0, y0 = coordinates[-2]
        x1, y1 = coordinates[-1]

    else:
        raise ValueError(
            f"Unsupported endpoint: {endpoint}"
        )

    dx = float(x1 - x0)
    dy = float(y1 - y0)

    length = math.hypot(dx, dy)

    if length == 0:
        raise ValueError(
            "Cannot calculate heading from zero-length line segment."
        )

    return (
        dx / length,
        dy / length,
    )


def extend_endpoint(
    endpoint: Point,
    direction: tuple[float, float],
    distance_m: float,
) -> tuple[LineString, Point]:
    """Creates a forward extension line from an endpoint.

    Args:
        endpoint: Selected downstream endpoint.
        direction: Unit vector oriented downstream/outward.
        distance_m: Maximum extension distance.

    Returns:
        Extension LineString and extension endpoint.
    """
    dx, dy = direction

    extended_point = Point(
        endpoint.x + dx * distance_m,
        endpoint.y + dy * distance_m,
    )

    return (
        LineString([endpoint, extended_point]),
        extended_point,
    )


def angular_difference_degrees(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Returns smallest angle between two direction vectors in degrees."""
    dot_product = (
        first[0] * second[0]
        + first[1] * second[1]
    )

    dot_product = max(
        -1.0,
        min(1.0, dot_product),
    )

    return math.degrees(
        math.acos(dot_product)
    )


# =============================================================================
# Junction-guided topology repair
# =============================================================================


class JunctionGuidedTopologyRepair:
    """Proposes downstream repairs for dangling prepared flowlines.

    The repair process does not assume a flowline's digitized end coordinate is
    always downstream. It first uses known valid ``to_source_id`` links to
    infer whether the source data generally point downstream from line starts
    or line ends. For each broken flowline it then uses nearby HydroNet
    Junctions as evidence for endpoint selection.

    A proposed receiver must satisfy all of the following:

        - It is not the source flowline itself.
        - It is not already upstream of the source flowline.
        - Adding the proposed edge cannot create a directed cycle.
        - It lies near the source extension or its downstream endpoint.
        - It is directionally plausible relative to the source endpoint.
        - It has a valid source_id.

    All repair proposals are written for review. The repaired review layer
    applies only high-confidence proposals, plus medium-confidence proposals
    when configured.
    """

    REQUIRED_FLOWLINE_FIELDS = {
        "source_id",
        "to_source_id",
    }

    REQUIRED_CATCHMENT_FIELDS = {
        "catchment_id",
        "source_id",
    }

    def __init__(
        self,
        config: TopologyRepairConfig,
        flowlines: gpd.GeoDataFrame,
        catchments: gpd.GeoDataFrame,
        junctions: gpd.GeoDataFrame,
    ) -> None:
        """Initializes repair inputs in one projected working CRS."""
        self.config = config

        self.flowlines = (
            lower_columns(flowlines)
            .to_crs(config.work_crs)
            .copy()
        )

        self.catchments = (
            lower_columns(catchments)
            .to_crs(config.work_crs)
            .copy()
        )

        self.junctions = (
            lower_columns(junctions)
            .to_crs(config.work_crs)
            .copy()
        )

        self._validate_inputs()

        self.flowlines["source_id"] = (
            self.flowlines["source_id"]
            .astype(str)
        )

        self.flowlines["to_source_id"] = (
            self.flowlines["to_source_id"]
            .astype("string")
            .replace(
                {
                    "<NA>": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                    "": pd.NA,
                }
            )
        )

        self.catchments["source_id"] = (
            self.catchments["source_id"]
            .astype(str)
        )

        self.source_id_set = set(
            self.flowlines["source_id"]
        )

        self.catchment_sindex = self.catchments.sindex
        self.flowline_sindex = self.flowlines.sindex
        self.junction_sindex = self.junctions.sindex

        self.graph = self._build_original_graph()

        self.default_downstream_endpoint = (
            self._infer_default_downstream_endpoint()
        )

        self.flowline_endpoint_table = (
            self._build_endpoint_table()
        )

    def _validate_inputs(
            self,
    ) -> None:
        """Validates required topology-repair input schema and geometry.

        Raises:
            KeyError: If required source IDs are missing.
            ValueError: If input layers are empty, lack a CRS, or contain
                unsupported flowline/junction geometry.
        """
        missing_flowline_fields = (
                self.REQUIRED_FLOWLINE_FIELDS
                - set(self.flowlines.columns)
        )

        if missing_flowline_fields:
            raise KeyError(
                "prepared_flowlines is missing required field(s): "
                f"{sorted(missing_flowline_fields)}"
            )

        missing_catchment_fields = (
                self.REQUIRED_CATCHMENT_FIELDS
                - set(self.catchments.columns)
        )

        if missing_catchment_fields:
            raise KeyError(
                "prepared_catchments is missing required field(s): "
                f"{sorted(missing_catchment_fields)}"
            )

        if self.flowlines.empty:
            raise ValueError(
                "prepared_flowlines is empty."
            )

        if self.catchments.empty:
            raise ValueError(
                "prepared_catchments is empty."
            )

        if self.junctions.empty:
            raise ValueError(
                "hydronet_junctions is empty."
            )

        if self.flowlines.crs is None:
            raise ValueError(
                "prepared_flowlines has no CRS."
            )

        if self.catchments.crs is None:
            raise ValueError(
                "prepared_catchments has no CRS."
            )

        if self.junctions.crs is None:
            raise ValueError(
                "hydronet_junctions has no CRS."
            )

        invalid_flowline_count = int(
            (
                    self.flowlines.geometry.isna()
                    | self.flowlines.geometry.is_empty
                    | ~self.flowlines.geometry.is_valid
            ).sum()
        )

        if invalid_flowline_count > 0:
            raise ValueError(
                "prepared_flowlines contains "
                f"{invalid_flowline_count:,} null, empty, or invalid "
                "geometry record(s)."
            )

        allowed_flowline_types = {
            "LineString",
            "MultiLineString",
        }

        flowline_geometry_types = set(
            self.flowlines.geom_type
            .dropna()
            .unique()
            .tolist()
        )

        invalid_flowline_types = (
                flowline_geometry_types
                - allowed_flowline_types
        )

        if invalid_flowline_types:
            raise ValueError(
                "prepared_flowlines contains unsupported geometry type(s): "
                f"{sorted(flowline_geometry_types)}. Expected line geometry."
            )

        invalid_junction_count = int(
            (
                    self.junctions.geometry.isna()
                    | self.junctions.geometry.is_empty
                    | ~self.junctions.geometry.is_valid
            ).sum()
        )

        if invalid_junction_count > 0:
            raise ValueError(
                "hydronet_junctions contains "
                f"{invalid_junction_count:,} null, empty, or invalid "
                "geometry record(s)."
            )

        junction_geometry_types = set(
            self.junctions.geom_type
            .dropna()
            .unique()
            .tolist()
        )

        if junction_geometry_types != {"Point"}:
            raise ValueError(
                "hydronet_junctions must contain Point geometry. "
                f"Found: {sorted(junction_geometry_types)}"
            )

        duplicate_source_ids = self.flowlines.loc[
            self.flowlines["source_id"]
            .astype(str)
            .duplicated(keep=False),
            "source_id",
        ]

        if not duplicate_source_ids.empty:
            duplicate_values = (
                duplicate_source_ids
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "prepared_flowlines source_id values must be unique. "
                f"First duplicates: {duplicate_values[:10]}"
            )


    def run(
        self,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        pd.DataFrame,
    ]:
        """Builds topology repair proposals and review layers.

        Returns:
            Tuple containing:

                - proposed repair extension lines,
                - broken flowline endpoint points,
                - repaired-review flowline layer,
                - repaired-review catchment layer,
                - nonspatial repair table.
        """
        broken = self._find_broken_flowlines()

        extension_records: list[dict[str, Any]] = []
        endpoint_records: list[dict[str, Any]] = []
        repair_records: list[dict[str, Any]] = []

        for flowline in tqdm(
            broken.itertuples(),
            total=len(broken),
            desc="Proposing flowline topology repairs",
            unit="flowline",
        ):
            repair = self._propose_repair(flowline)

            repair_records.append(repair)

            endpoint_records.append(
                self._make_endpoint_record(
                    flowline,
                    repair,
                )
            )

            if repair["extension_geometry"] is not None:
                extension_records.append(
                    {
                        **repair,
                        "geometry": repair[
                            "extension_geometry"
                        ],
                    }
                )

        repairs = pd.DataFrame(repair_records)

        extension_lines = self._records_to_gdf(
            extension_records
        )

        endpoint_points = self._records_to_gdf(
            endpoint_records
        )

        repaired_flowlines = self._apply_repairs_to_flowlines(
            repairs
        )

        repaired_catchments = self._apply_repairs_to_catchments(
            repaired_flowlines
        )

        return (
            extension_lines,
            endpoint_points,
            repaired_flowlines,
            repaired_catchments,
            repairs,
        )

    def _find_broken_flowlines(
        self,
    ) -> gpd.GeoDataFrame:
        """Returns flowlines with missing or invalid downstream links."""
        to_source_id = self.flowlines["to_source_id"]

        missing_link = (
            to_source_id.isna()
            | ~to_source_id.astype(str).isin(
                self.source_id_set
            )
        )

        return self.flowlines.loc[
            missing_link
        ].copy()

    def _build_original_graph(
        self,
    ) -> nx.DiGraph:
        """Builds directed graph from valid original downstream links."""
        graph = nx.DiGraph()

        graph.add_nodes_from(
            self.flowlines["source_id"]
        )

        valid_links = self.flowlines[
            self.flowlines["to_source_id"].notna()
            & self.flowlines["to_source_id"]
            .astype(str)
            .isin(self.source_id_set)
        ].copy()

        graph.add_edges_from(
            zip(
                valid_links["source_id"],
                valid_links["to_source_id"].astype(
                    str
                ),
            )
        )

        return graph

    def _infer_default_downstream_endpoint(
        self,
    ) -> str:
        """Infers whether line starts or ends are normally downstream.

        Valid known topology links are used as evidence. For each source line,
        compare its start and end point distances to the geometry of its known
        downstream receiver. Whichever endpoint is closer most often becomes
        the default downstream endpoint.

        Returns:
            ``start`` or ``end``.
        """
        valid_links = self.flowlines[
            self.flowlines["to_source_id"].notna()
            & self.flowlines["to_source_id"]
            .astype(str)
            .isin(self.source_id_set)
        ].copy()

        if valid_links.empty:
            tqdm.write(
                "WARNING: No valid downstream links were available to "
                "infer flowline orientation. Defaulting to geometry end."
            )
            return "end"

        receiver_lookup = (
            self.flowlines
            .set_index("source_id")
            .geometry
            .to_dict()
        )

        start_votes = 0
        end_votes = 0

        sample = valid_links.head(10_000)

        for flowline in sample.itertuples():
            receiver_geometry = receiver_lookup.get(
                str(flowline.to_source_id)
            )

            if receiver_geometry is None:
                continue

            try:
                start_point, end_point = line_endpoints(
                    flowline.geometry
                )
            except ValueError:
                continue

            start_distance = start_point.distance(
                receiver_geometry
            )

            end_distance = end_point.distance(
                receiver_geometry
            )

            if start_distance < end_distance:
                start_votes += 1
            else:
                end_votes += 1

        selected = (
            "start"
            if start_votes > end_votes
            else "end"
        )

        total_votes = start_votes + end_votes

        if total_votes > 0:
            fraction = (
                max(start_votes, end_votes)
                / total_votes
            )

            tqdm.write(
                "Inferred default downstream geometry endpoint: "
                f"{selected} "
                f"({fraction:.1%} agreement from "
                f"{total_votes:,} valid links)."
            )

        return selected

    def _build_endpoint_table(
        self,
    ) -> gpd.GeoDataFrame:
        """Builds start/end flowline endpoint point features."""
        records: list[dict[str, Any]] = []

        for flowline in self.flowlines.itertuples():
            try:
                start_point, end_point = line_endpoints(
                    flowline.geometry
                )
            except ValueError:
                continue

            records.append(
                {
                    "source_id": str(flowline.source_id),
                    "endpoint_role": "start",
                    "geometry": start_point,
                }
            )

            records.append(
                {
                    "source_id": str(flowline.source_id),
                    "endpoint_role": "end",
                    "geometry": end_point,
                }
            )

        return gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=self.flowlines.crs,
        )

    def _propose_repair(self, flowline):
        """Select the first catchment before checking its network validity."""
        source_id = str(flowline.source_id)
        result = self._empty_repair_record(source_id,
            None if pd.isna(flowline.to_source_id) else str(flowline.to_source_id))
        result.update(proposed_catchment_id=None, first_intersection_distance_m=None,
                      first_intersection_x=None, first_intersection_y=None)
        choice = self._select_downstream_endpoint(flowline)
        if choice is None:
            result.update(repair_status="no_valid_endpoint", repair_notes="Cannot determine a usable line endpoint.")
            return result
        line, end = extend_endpoint(choice["endpoint_point"], choice["heading"], self.config.extension_distance_m)
        result.update(endpoint_role=choice["endpoint_name"],
            endpoint_x=choice["endpoint_point"].x, endpoint_y=choice["endpoint_point"].y,
            endpoint_junction_distance_m=choice["junction_distance_m"],
            endpoint_selection_method=choice["selection_method"], extension_geometry=line,
            extension_end_x=end.x, extension_end_y=end.y)
        candidates = self._find_receiver_candidates(source_id, line, end, choice["heading"])
        if candidates.empty:
            result.update(repair_status="no_receiver_candidate", repair_notes="Extension enters no other catchment interior.")
            return result
        first = candidates.iloc[0]
        distance = float(first["first_intersection_distance_m"])
        entry = line.interpolate(distance)
        receiver = first["candidate_source_id"]
        result.update(proposed_catchment_id=first["candidate_catchment_id"],
            proposed_to_source_id=receiver, first_intersection_distance_m=distance,
            first_intersection_x=entry.x, first_intersection_y=entry.y,
            repair_method="first_catchment_extension")
        tied = candidates[candidates["first_intersection_distance_m"] <= distance + 1e-6]
        if tied["candidate_source_id"].nunique(dropna=False) > 1:
            result.update(repair_status="ambiguous_first_catchment", repair_notes="Multiple receiving catchments enter at the same distance; review required.")
            return result
        if receiver not in self.source_id_set:
            result.update(repair_status="receiver_missing_flowline", repair_notes="First catchment has no matching prepared flowline. No farther catchment substituted.")
            return result
        if self._would_create_cycle(source_id, receiver):
            result.update(repair_status="cycle_rejected", repair_notes="First receiver would create a cycle; no farther catchment substituted.")
            return result
        receiver_row = self.flowlines[self.flowlines["source_id"] == receiver].iloc[0]
        result["receiver_distance_to_extension_m"] = receiver_row.geometry.distance(line)
        result["receiver_distance_to_extension_end_m"] = receiver_row.geometry.distance(entry)
        result["receiver_reaches_network_end"] = int(self._receiver_has_valid_downstream_path(receiver))
        result.update(repair_status="proposed", repair_confidence="medium",
            repair_notes="First entered catchment determines receiver. Junction proximity is endpoint QA, not directional proof. Review before accepting.")
        return result

    def _select_downstream_endpoint(self, flowline):
        """Use inferred digitizing orientation; junction proximity alone is not direction."""
        try:
            start, end = line_endpoints(flowline.geometry)
            role = self.default_downstream_endpoint
            point = start if role == "start" else end
            heading = endpoint_heading(flowline.geometry, role)
            distance = self._nearest_junction_distance(point)
        except (ValueError, AttributeError):
            return None
        return {"endpoint_name": role, "endpoint_point": point, "heading": heading,
                "junction_distance_m": distance if math.isfinite(distance) else None,
                "selection_method": "network_orientation_junction_qa"}

    def _nearest_junction_distance(
        self,
        point: Point,
    ) -> float:
        """Returns distance to nearest HydroNet Junction point."""
        if self.junctions.empty:
            return float("inf")

        candidate_indices = list(
            self.junction_sindex.query(
                point.buffer(
                    self.config.junction_snap_distance_m
                ),
                predicate="intersects",
            )
        )

        if not candidate_indices:
            return float("inf")

        candidates = self.junctions.iloc[
            candidate_indices
        ]

        return float(
            candidates.geometry.distance(point).min()
        )

    def _find_receiver_candidates(
        self, source_id, extension_line, extension_end, source_heading,
    ):
        """Record the first OTHER catchment interior entered; never skip it.

        Tangential boundary touches are excluded. A blocked first receiver
        remains blocked rather than selecting a farther polygon.
        """
        records = []
        for pos in self.catchment_sindex.query(extension_line, predicate="intersects"):
            row = self.catchments.iloc[int(pos)]
            if str(row["source_id"]) == source_id:
                continue
            intersection = extension_line.intersection(row.geometry)
            parts = [intersection]
            while parts:
                part = parts.pop()
                if part.geom_type in ("MultiLineString", "GeometryCollection"):
                    parts.extend(part.geoms)
                elif part.geom_type == "LineString" and part.length > 1e-8:
                    if not row.geometry.contains(part.interpolate(0.5, normalized=True)):
                        continue
                    distances = [extension_line.project(Point(c[:2])) for c in part.coords]
                    records.append({
                        "candidate_catchment_id": str(row["catchment_id"]),
                        "candidate_source_id": None if pd.isna(row["source_id"]) else str(row["source_id"]),
                        "first_intersection_distance_m": float(min(distances)),
                        "geometry": row.geometry,
                    })
        if not records:
            return gpd.GeoDataFrame(geometry=[], crs=self.catchments.crs)
        out = gpd.GeoDataFrame(records, geometry="geometry", crs=self.catchments.crs)
        return out.sort_values(["first_intersection_distance_m", "candidate_catchment_id"]).drop_duplicates("candidate_catchment_id").reset_index(drop=True)

    def _is_upstream_of_source(
        self,
        source_id: str,
        candidate_id: str,
    ) -> bool:
        """Returns whether candidate already lies upstream of source."""
        if (
            source_id not in self.graph
            or candidate_id not in self.graph
        ):
            return False

        return nx.has_path(
            self.graph,
            candidate_id,
            source_id,
        )

    def _would_create_cycle(
        self,
        source_id: str,
        receiver_id: str,
    ) -> bool:
        """Returns whether adding source -> receiver creates a cycle."""
        if source_id == receiver_id:
            return True

        if (
            source_id not in self.graph
            or receiver_id not in self.graph
        ):
            return False

        return nx.has_path(
            self.graph,
            receiver_id,
            source_id,
        )

    def _receiver_has_valid_downstream_path(
        self,
        receiver_id: str,
    ) -> bool:
        """Checks whether receiver eventually reaches a graph terminal.

        A terminal is any original graph node with out-degree zero. This does
        not claim the terminal is the selected PRMS outlet; Routing.py later
        determines whether the repaired link reaches the configured outlet.
        """
        if receiver_id not in self.graph:
            return False

        terminals = [
            node
            for node in self.graph.nodes
            if self.graph.out_degree(node) == 0
        ]

        return any(
            nx.has_path(
                self.graph,
                receiver_id,
                terminal,
            )
            for terminal in terminals
        )

    def _repair_confidence(
        self,
        candidate: pd.Series,
    ) -> str:
        """Assigns confidence based on extension geometry evidence."""
        endpoint_distance = float(
            candidate[
                "distance_to_extension_end_m"
            ]
        )

        turn_angle = float(
            candidate["turn_angle_degrees"]
        )

        if (
            endpoint_distance
            <= self.config
            .receiver_endpoint_snap_distance_m
            and turn_angle <= 30.0
        ):
            return "high"

        if (
            endpoint_distance
            <= self.config
            .receiver_search_distance_m
            and turn_angle
            <= self.config.maximum_turn_angle_degrees
        ):
            return "medium"

        return "low"

    @staticmethod
    def _empty_repair_record(
        source_id: str,
        original_to_source_id: str | None,
    ) -> dict[str, Any]:
        """Creates an unresolved topology repair record."""
        return {
            "source_id": source_id,
            "original_to_source_id": original_to_source_id,
            "proposed_to_source_id": None,
            "repair_status": "unresolved",
            "repair_method": None,
            "repair_confidence": "low",
            "repair_notes": None,
            "endpoint_role": None,
            "endpoint_x": None,
            "endpoint_y": None,
            "endpoint_junction_distance_m": None,
            "endpoint_selection_method": None,
            "extension_end_x": None,
            "extension_end_y": None,
            "receiver_endpoint_role": None,
            "receiver_distance_to_extension_m": None,
            "receiver_distance_to_extension_end_m": None,
            "receiver_turn_angle_degrees": None,
            "receiver_reaches_network_end": 0,
            "extension_geometry": None,
        }

    def _make_endpoint_record(
        self,
        flowline: Any,
        repair: dict[str, Any],
    ) -> dict[str, Any]:
        """Creates one QA point record for selected broken endpoint."""
        if repair["endpoint_x"] is None:
            point = flowline.geometry.representative_point()
        else:
            point = Point(
                repair["endpoint_x"],
                repair["endpoint_y"],
            )

        return {
            **repair,
            "geometry": point,
        }

    def _apply_repairs_to_flowlines(
            self,
            repairs: pd.DataFrame,
    ) -> gpd.GeoDataFrame:
        """Adds repair proposals and optionally applies downstream links.

        Original flowline attributes and geometry are preserved. New links
        are checked against a working graph that includes previously applied
        repairs, preventing cycles created by combinations of proposals.

        Args:
            repairs: One repair result record per evaluated source flowline.

        Returns:
            Flowline review layer with original, proposed, and applied links.
        """
        repaired = self.flowlines.copy().reset_index(drop=True)

        repaired["original_to_source_id"] = repaired[
            "to_source_id"
        ].copy()

        text_defaults = {
            "proposed_to_source_id": None,
            "proposed_catchment_id": None,
            "topology_repair_status": "not_evaluated",
            "topology_repair_method": None,
            "topology_repair_confidence": None,
            "topology_repair_notes": None,
        }

        for field, default in text_defaults.items():
            repaired[field] = pd.Series(
                [default] * len(repaired),
                index=repaired.index,
                dtype=object,
            )

        repaired["topology_repair_applied"] = 0

        numeric_defaults = {
            "first_intersection_distance_m": np.nan,
            "topology_repair_distance_m": np.nan,
            "topology_repair_turn_angle_deg": np.nan,
        }

        for field, default in numeric_defaults.items():
            repaired[field] = default

        if repairs.empty:
            return repaired

        if "source_id" not in repairs.columns:
            raise KeyError(
                "Repair results are missing 'source_id'."
            )

        repair_table = repairs.copy()
        repair_table["source_id"] = repair_table[
            "source_id"
        ].astype(str)

        if repair_table["source_id"].duplicated().any():
            duplicate_ids = repair_table.loc[
                repair_table["source_id"].duplicated(keep=False),
                "source_id",
            ].unique().tolist()

            raise ValueError(
                "Repair results must contain one record per source_id. "
                f"Duplicate IDs: {duplicate_ids[:10]}"
            )

        repair_lookup = repair_table.set_index(
            "source_id",
            drop=False,
        )

        application_graph = self.graph.copy()

        text_mapping = {
            "proposed_to_source_id": "proposed_to_source_id",
            "proposed_catchment_id": "proposed_catchment_id",
            "repair_status": "topology_repair_status",
            "repair_method": "topology_repair_method",
            "repair_confidence": "topology_repair_confidence",
            "repair_notes": "topology_repair_notes",
        }

        numeric_mapping = {
            "first_intersection_distance_m": (
                "first_intersection_distance_m"
            ),
            "receiver_turn_angle_degrees": (
                "topology_repair_turn_angle_deg"
            ),
        }

        for index, flowline in repaired.iterrows():
            source_id = str(flowline["source_id"])

            if source_id not in repair_lookup.index:
                continue

            repair = repair_lookup.loc[source_id]

            for result_field, output_field in text_mapping.items():
                value = repair.get(result_field)

                repaired.at[index, output_field] = (
                    None
                    if pd.isna(value)
                    else str(value)
                )

            for result_field, output_field in numeric_mapping.items():
                value = repair.get(result_field)

                if pd.notna(value):
                    repaired.at[index, output_field] = float(value)

            # Repair distance now means distance along the extension
            # to the first receiving catchment—not distance to its flowline.
            entry_distance = repair.get(
                "first_intersection_distance_m"
            )

            if pd.notna(entry_distance):
                repaired.at[
                    index,
                    "topology_repair_distance_m",
                ] = float(entry_distance)

            confidence = repair.get("repair_confidence")
            receiver = repair.get("proposed_to_source_id")

            apply_repair = (
                    confidence == "high"
                    or (
                            self.config.apply_medium_confidence_repairs
                            and confidence == "medium"
                    )
            )

            if (
                    repair.get("repair_status") != "proposed"
                    or not apply_repair
                    or pd.isna(receiver)
            ):
                continue

            receiver_id = str(receiver)

            if receiver_id not in self.source_id_set:
                repaired.at[
                    index,
                    "topology_repair_status",
                ] = "receiver_missing_flowline"

                repaired.at[
                    index,
                    "topology_repair_notes",
                ] = (
                    "The first receiving catchment has no matching "
                    "prepared flowline. Proposal was not applied."
                )
                continue

            creates_cycle = (
                    receiver_id == source_id
                    or nx.has_path(
                application_graph,
                receiver_id,
                source_id,
            )
            )

            if creates_cycle:
                repaired.at[
                    index,
                    "topology_repair_status",
                ] = "cycle_rejected_combined"

                repaired.at[
                    index,
                    "topology_repair_notes",
                ] = (
                    "Applying this proposal would create a cycle "
                    "when combined with previously applied repairs."
                )
                continue

            application_graph.add_edge(
                source_id,
                receiver_id,
            )

            repaired.at[
                index,
                "to_source_id",
            ] = receiver_id

            repaired.at[
                index,
                "topology_repair_applied",
            ] = 1

        return repaired

    def _apply_repairs_to_catchments(self, repaired_flowlines):
        """Preserve all catchment fields; inherit authoritative repaired links."""
        repaired = self.catchments.copy()
        lookup = repaired_flowlines.set_index("source_id")
        fields = [c for c in repaired_flowlines.columns if c.startswith("topology_repair_")]
        fields += ["to_source_id", "original_to_source_id", "proposed_to_source_id",
                        "proposed_catchment_id",
                        "first_intersection_distance_m",
                   "proposed_catchment_id", "first_intersection_distance_m"]
        for field in fields:
            if field in lookup.columns:
                repaired[field] = repaired["source_id"].map(lookup[field])
        repaired["topology_repair_status"] = repaired["topology_repair_status"].fillna("missing_flowline")
        repaired["topology_repair_applied"] = repaired["topology_repair_applied"].fillna(0).astype(int)
        return repaired

    def _records_to_gdf(
        self,
        records: list[dict[str, Any]],
    ) -> gpd.GeoDataFrame:
        """Creates a GeoDataFrame from output records."""
        if not records:
            return gpd.GeoDataFrame(
                geometry=[],
                crs=self.flowlines.crs,
            )

        return gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=self.flowlines.crs,
        )

    def write_outputs(
        self,
        extension_lines: gpd.GeoDataFrame,
        endpoint_points: gpd.GeoDataFrame,
        repaired_flowlines: gpd.GeoDataFrame,
        repaired_catchments: gpd.GeoDataFrame,
        repairs: pd.DataFrame,
    ) -> None:
        """Writes topology repair QA layers, review layers, CSV, and JSON."""
        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not extension_lines.empty:
            extension_out = extension_lines[
                [
                    field
                    for field in (
                        "source_id",
                        "original_to_source_id",
                        "proposed_to_source_id",
                        "proposed_catchment_id",
                        "first_intersection_distance_m",
                        "repair_status",
                        "repair_method",
                        "repair_confidence",
                        "endpoint_role",
                        "endpoint_junction_distance_m",
                        "receiver_distance_to_extension_m",
                        "receiver_distance_to_extension_end_m",
                        "receiver_turn_angle_degrees",
                        "receiver_reaches_network_end",
                        "repair_notes",
                        "geometry",
                    )
                    if field in extension_lines.columns
                ]
            ].copy()

            write_layer(
                extension_out,
                self.config.output_gpkg_path,
                "flowline_extension_candidates",
            )

        if not endpoint_points.empty:
            endpoint_out = endpoint_points[
                [
                    field
                    for field in (
                        "source_id",
                        "original_to_source_id",
                        "proposed_to_source_id",
                        "proposed_catchment_id",
                        "first_intersection_distance_m",
                        "repair_status",
                        "repair_method",
                        "repair_confidence",
                        "endpoint_role",
                        "endpoint_junction_distance_m",
                        "endpoint_selection_method",
                        "repair_notes",
                        "geometry",
                    )
                    if field in endpoint_points.columns
                ]
            ].copy()

            write_layer(
                endpoint_out,
                self.config.output_gpkg_path,
                "broken_flowline_endpoints",
            )

        flowline_out_fields = [
            "source_id",
            "to_source_id",
            "original_to_source_id",
            "proposed_to_source_id",
                        "proposed_catchment_id",
                        "first_intersection_distance_m",
            "topology_repair_status",
            "topology_repair_method",
            "topology_repair_confidence",
            "topology_repair_notes",
            "topology_repair_applied",
            "topology_repair_distance_m",
            "topology_repair_turn_angle_deg",
            "geometry",
        ]

        flowline_out_fields = list(repaired_flowlines.columns)
        flowline_out = repaired_flowlines[
            [
                field
                for field in flowline_out_fields
                if field in repaired_flowlines.columns
            ]
        ].copy()

        write_layer(
            flowline_out,
            self.config.output_gpkg_path,
            "prepared_flowlines_topology_repaired",
        )

        catchment_out_fields = [
            "catchment_id",
            "source_id",
            "to_source_id",
            "original_to_source_id",
            "proposed_to_source_id",
                        "proposed_catchment_id",
                        "first_intersection_distance_m",
            "topology_repair_status",
            "topology_repair_method",
            "topology_repair_confidence",
            "topology_repair_applied",
            "topology_repair_distance_m",
            "topology_repair_turn_angle_deg",
            "geometry",
        ]

        catchment_out_fields = list(repaired_catchments.columns)
        catchment_out = repaired_catchments[
            [
                field
                for field in catchment_out_fields
                if field in repaired_catchments.columns
            ]
        ].copy()

        write_layer(
            catchment_out,
            self.config.output_gpkg_path,
            "prepared_catchments_topology_repaired",
        )

        repairs_out = repairs.drop(
            columns=["extension_geometry"],
            errors="ignore",
        )

        repairs_out.to_csv(
            self.config.output_dir
            / "flowline_topology_repairs.csv",
            index=False,
        )

        if repairs.empty:
            summary = {
                "n_broken_flowlines": 0,
                "n_proposed": 0,
                "n_high_confidence": 0,
                "n_medium_confidence": 0,
                "n_applied_to_review_layer": 0,
                "repair_status_counts": {},
            }
        else:
            applied = repaired_flowlines[
                "topology_repair_applied"
            ].sum()

            summary = {
                "n_broken_flowlines": int(
                    len(repairs)
                ),
                "n_proposed": int(
                    (
                        repairs["repair_status"]
                        == "proposed"
                    ).sum()
                ),
                "n_high_confidence": int(
                    (
                        repairs["repair_confidence"]
                        == "high"
                    ).sum()
                ),
                "n_medium_confidence": int(
                    (
                        repairs["repair_confidence"]
                        == "medium"
                    ).sum()
                ),
                "n_applied_to_review_layer": int(
                    applied
                ),
                "repair_status_counts": {
                    str(status): int(count)
                    for status, count in repairs[
                        "repair_status"
                    ]
                    .value_counts(dropna=False)
                    .to_dict()
                    .items()
                },
            }

        (
            self.config.output_dir
            / "flowline_topology_repair_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def plot_repair_map(
            self,
            extension_lines: gpd.GeoDataFrame,
            endpoint_points: gpd.GeoDataFrame,
            repaired_flowlines: gpd.GeoDataFrame,
            repairs: pd.DataFrame,
            zoom_to_repairs: bool = False,
            zoom_buffer_m: float = 750.0,
    ) -> None:
        """Maps catchments (solid) and flowlines (dashed) with repair results.

        Catchment outlines are solid. All flowlines are dashed. Extensions
        are drawn as thick solid lines in status colors so they remain
        distinguishable from catchment boundaries. Catchments that receive a
        proposed repair are shaded green; catchments belonging to broken
        flowlines are shaded orange. Black x markers show the first
        catchment-entry point for each extension.

        Args:
            extension_lines: Forward extension candidates.
            endpoint_points: Selected downstream endpoints of broken lines.
            repaired_flowlines: Full repair-review flowline layer.
            repairs: Tabular repair results.
            zoom_to_repairs: If True, crop the map around repair extensions.
            zoom_buffer_m: Margin around repairs when zooming, in meters.
        """
        figure, axis = plt.subplots(figsize=(16, 16))

        catchments = self.catchments

        broken_ids = set(
            repaired_flowlines.loc[
                repaired_flowlines["topology_repair_status"]
                != "not_evaluated",
                "source_id",
            ].astype(str)
        )

        receiving_catchment_ids: set[str] = set()
        receiver_source_ids: set[str] = set()

        if not repairs.empty:
            proposed = repairs[repairs["repair_status"] == "proposed"]

            if "proposed_catchment_id" in proposed.columns:
                receiving_catchment_ids = set(
                    proposed["proposed_catchment_id"].dropna().astype(str)
                )

            receiver_source_ids = set(
                proposed["proposed_to_source_id"].dropna().astype(str)
            )

        catchment_ids = catchments["catchment_id"].astype(str)
        catchment_source_ids = catchments["source_id"].astype(str)

        broken_catchments = catchments[catchment_source_ids.isin(broken_ids)]
        receiving_catchments = catchments[
            catchment_ids.isin(receiving_catchment_ids)
        ]

        if not broken_catchments.empty:
            broken_catchments.plot(
                ax=axis,
                facecolor="#FDD9B5",
                edgecolor="none",
                alpha=0.55,
                zorder=0,
            )

        if not receiving_catchments.empty:
            receiving_catchments.plot(
                ax=axis,
                facecolor="#BDE5D0",
                edgecolor="none",
                alpha=0.65,
                zorder=0,
            )

        # Catchment outlines: solid.
        catchments.boundary.plot(
            ax=axis,
            color="#7A7A7A",
            linewidth=0.45,
            linestyle="-",
            zorder=1,
        )

        if not receiving_catchments.empty:
            receiving_catchments.boundary.plot(
                ax=axis,
                color="#1B7F4B",
                linewidth=1.3,
                linestyle="-",
                zorder=2,
            )

        if not broken_catchments.empty:
            broken_catchments.boundary.plot(
                ax=axis,
                color="#C75B12",
                linewidth=1.3,
                linestyle="-",
                zorder=2,
            )

        # Flowlines: all dashed.
        repaired_flowlines.plot(
            ax=axis,
            color="#9A9A9A",
            linewidth=0.7,
            linestyle="--",
            zorder=3,
        )

        broken_lines = repaired_flowlines[
            repaired_flowlines["source_id"].astype(str).isin(broken_ids)
        ]

        if not broken_lines.empty:
            broken_lines.plot(
                ax=axis,
                color="#111111",
                linewidth=1.6,
                linestyle="--",
                zorder=4,
            )

        receiver_lines = repaired_flowlines[
            repaired_flowlines["source_id"].astype(str).isin(
                receiver_source_ids
            )
        ]

        if not receiver_lines.empty:
            receiver_lines.plot(
                ax=axis,
                color="#0072B2",
                linewidth=1.8,
                linestyle="--",
                zorder=4,
            )

        status_colors = {
            "proposed": "#D55E00",
            "no_receiver_candidate": "#CC79A7",
            "cycle_rejected": "#E69F00",
            "receiver_missing_flowline": "#8C564B",
            "ambiguous_first_catchment": "#6A3D9A",
            "no_valid_endpoint": "#7F7F7F",
            "unresolved": "#7F7F7F",
        }

        if not extension_lines.empty:
            for status, color in status_colors.items():
                subset = extension_lines[
                    extension_lines["repair_status"] == status
                    ]

                if subset.empty:
                    continue

                subset.plot(
                    ax=axis,
                    color=color,
                    linewidth=2.4,
                    linestyle="-",
                    zorder=6,
                )

        if not endpoint_points.empty:
            for confidence, color in (
                    ("high", "#009E73"),
                    ("medium", "#56B4E9"),
                    ("low", "#999999"),
            ):
                subset = endpoint_points[
                    endpoint_points["repair_confidence"] == confidence
                    ]

                if subset.empty:
                    continue

                subset.plot(
                    ax=axis,
                    color=color,
                    marker="o",
                    markersize=34,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=7,
                )

        has_entry = (
                not repairs.empty
                and "first_intersection_x" in repairs.columns
        )

        if has_entry:
            entries = repairs.dropna(
                subset=[
                    "first_intersection_x",
                    "first_intersection_y",
                ]
            )

            if not entries.empty:
                axis.scatter(
                    entries["first_intersection_x"],
                    entries["first_intersection_y"],
                    marker="x",
                    s=55,
                    color="black",
                    linewidths=1.4,
                    zorder=8,
                )

        if zoom_to_repairs and not extension_lines.empty:
            minx, miny, maxx, maxy = extension_lines.total_bounds

            axis.set_xlim(minx - zoom_buffer_m, maxx + zoom_buffer_m)
            axis.set_ylim(miny - zoom_buffer_m, maxy + zoom_buffer_m)

        legend_handles = [
            Line2D([0], [0], color="#7A7A7A", linewidth=1.5,
                   linestyle="-", label="Catchment boundary"),
            Patch(facecolor="#FDD9B5", edgecolor="#C75B12",
                  label="Catchment of broken flowline"),
            Patch(facecolor="#BDE5D0", edgecolor="#1B7F4B",
                  label="Receiving catchment"),
            Line2D([0], [0], color="#9A9A9A", linewidth=1.5,
                   linestyle="--", label="Flowline"),
            Line2D([0], [0], color="#111111", linewidth=2.0,
                   linestyle="--", label="Broken flowline"),
            Line2D([0], [0], color="#0072B2", linewidth=2.0,
                   linestyle="--", label="Receiving flowline"),
            Line2D([0], [0], color="#D55E00", linewidth=2.6,
                   linestyle="-", label="Proposed extension"),
            Line2D([0], [0], color="#CC79A7", linewidth=2.6,
                   linestyle="-", label="Extension, no receiver"),
            Line2D([0], [0], marker="x", color="black", linestyle="",
                   markersize=8, label="First catchment entry"),
            Line2D([0], [0], marker="o", color="white",
                   markerfacecolor="#009E73", markeredgecolor="black",
                   markersize=8, label="High-confidence endpoint"),
            Line2D([0], [0], marker="o", color="white",
                   markerfacecolor="#56B4E9", markeredgecolor="black",
                   markersize=8, label="Medium-confidence endpoint"),
        ]

        axis.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.95,
        )

        proposed_count = (
            int((repairs["repair_status"] == "proposed").sum())
            if not repairs.empty
            else 0
        )

        axis.set_title(
            "Flowline Topology Repair: first-catchment extension\n"
            f"Broken flowlines: {len(broken_ids):,} | "
            f"Proposed receivers: {proposed_count:,}",
            fontsize=15,
            fontweight="bold",
        )

        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="box")
        figure.tight_layout()

        figure.savefig(
            self.config.output_dir / "flowline_topology_repair_map.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(figure)

# =============================================================================
# Standalone runner
# =============================================================================


def main() -> None:
    """Runs junction-guided prepared-flowline topology repair."""
    config = TopologyRepairConfig(
        input_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\PlusHR_UG.gdb"
        ),

        output_gpkg_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\PlusHR_UG.gdb"
        ),

        output_dir=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
            r"\flowline_topology_repair"
        ),

        flowline_layer="prepared_flowlines",
        catchment_layer="prepared_catchments",

        # Change this to the exact HydroNet Junction layer name in PlusHR_UG.gdb.
        junction_layer="hydronet_junctions",

        extension_distance_m=500.0,
        junction_snap_distance_m=15.0,
        receiver_search_distance_m=25.0,
        receiver_endpoint_snap_distance_m=25.0,
        maximum_turn_angle_degrees=60.0,

        require_receiver_reaches_active_network=True,

        # Start True so Routing.py can demonstrate the full upstream-network
        # effect. Set False if you want only high-confidence links applied.
        apply_medium_confidence_repairs=True,
    )

    print("=" * 78)
    print("JUNCTION-GUIDED FLOWLINE TOPOLOGY REPAIR")
    print("=" * 78)
    print(f"Input datasource: {config.input_path}")
    print(f"Output GeoPackage: {config.output_gpkg_path}")
    print(f"Flowline layer: {config.flowline_layer}")
    print(f"Catchment layer: {config.catchment_layer}")
    print(f"Junction layer: {config.junction_layer}")
    print(
        f"Extension distance: "
        f"{config.extension_distance_m:.1f} m"
    )
    print(
        f"Junction snap distance: "
        f"{config.junction_snap_distance_m:.1f} m"
    )
    print(
        f"Receiver search distance: "
        f"{config.receiver_search_distance_m:.1f} m"
    )

    config.output_dir.mkdir(            
        parents=True,
        exist_ok=True,
    )

    flowlines = read_layer(
        config.input_path,
        config.flowline_layer,
    )

    catchments = read_layer(
        config.input_path,
        config.catchment_layer,
    )

    junctions = read_layer(
        config.input_path,
        config.junction_layer,
    )

    repairer = JunctionGuidedTopologyRepair(
        config=config,
        flowlines=flowlines,
        catchments=catchments,
        junctions=junctions,
    )

    (
        extension_lines,
        endpoint_points,
        repaired_flowlines,
        repaired_catchments,
        repairs,
    ) = repairer.run()

    repairer.write_outputs(
        extension_lines=extension_lines,
        endpoint_points=endpoint_points,
        repaired_flowlines=repaired_flowlines,
        repaired_catchments=repaired_catchments,
        repairs=repairs,
    )

    print(
        "\nCreating flowline topology-repair QA map."
    )

    repairer.plot_repair_map(
        extension_lines=extension_lines,
        endpoint_points=endpoint_points,
        repaired_flowlines=repaired_flowlines,
        repairs=repairs,
    )

    print("\nRepair status counts:")

    if repairs.empty:
        print("No broken flowlines were found.")
    else:
        print(
            repairs["repair_status"]
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nRepair confidence counts:")

        print(
            repairs["repair_confidence"]
            .value_counts(dropna=False)
            .to_string()
        )

        applied_count = int(
            repaired_flowlines[
                "topology_repair_applied"
            ].sum()
        )

        print(
            f"\nApplied repair links in review layer: "
            f"{applied_count:,}"
        )

    print("\n" + "=" * 78)
    print("FLOWLINE TOPOLOGY REPAIR COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()