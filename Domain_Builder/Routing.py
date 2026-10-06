r"""Build a dendritic PRMS routing domain from standardized prepared layers.

This module does not know whether the source network came from:

    - NHDPlus HR,
    - NHDPlus V2,
    - 3D Hydrography Program data,
    - a custom Arc Pro hydrography layer,
    - or a future MODFLOW 6 DISV-derived network.

It only requires the standardized prepared-domain contract:

prepared_flowlines:
    source_id
    to_source_id
    geometry

prepared_catchments:
    catchment_id
    source_id
    geometry

prepared_waterbodies:
    waterbody_id
    geometry

prepared_huc12:
    huc12
    geometry

Outputs:
    model_segments
        Active routed stream segments with contiguous segment IDs.

    model_hrus
        Active routed parent HRUs used for PRMS/pywatershed parameterization.

    model_catchments
        All prepared catchments, including sinks, disconnected catchments, and
        catchments requiring manual Arc Pro routing corrections.

    unrouted_catchments
        Optional review subset of model_catchments that is not active in the
        selected model network.

    model_waterbodies
        Optional waterbody layer in the working projected CRS.

    segments.csv
    hru_to_segment.csv
    segment_routing.csv
    domain_qa/routing_qa.json

The central design distinction is:

    prepared layers
        Source-neutral routing fabric.

    model_catchments
        All catchments with model routing status, including manual-review
        candidates.

    model_hrus
        Only catchments in the active connected PRMS routing network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PROJ configuration
# ---------------------------------------------------------------------------
# Configure the conda PROJ data directory before importing pyproj, GeoPandas,
# Pyogrio, Fiona, or any project module that imports them.

_PROJ_DIR = Path(sys.prefix) / "Library" / "share" / "proj"

if not (_PROJ_DIR / "proj.db").exists():
    raise FileNotFoundError(
        "Conda PROJ database was not found:\n"
        f"    {_PROJ_DIR / 'proj.db'}"
    )

os.environ["PROJ_LIB"] = str(_PROJ_DIR)
os.environ["PROJ_DATA"] = str(_PROJ_DIR)

import pyproj

pyproj.datadir.set_data_dir(str(_PROJ_DIR))

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from fiona import listlayers, remove
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================


class RoutingConfig:
    """Stores source-agnostic routing and PRMS-domain settings.

    Args:
        gpkg_path: GeoPackage containing prepared domain layers and receiving
            model-domain output layers.
        target_huc12: HUC12 code used for automatic outlet selection.
        outlet_source_id: Optional explicit flowline source ID for the model
            outlet. If supplied, it takes precedence over target_huc12.
        work_crs: Projected CRS for model geometry, length, and area metrics.
        min_hru_area_m2: HRUs smaller than this area are flagged but retained.
        flowline_layer: Prepared flowline-layer name.
        catchment_layer: Prepared catchment-layer name.
        waterbody_layer: Optional prepared waterbody-layer name.
        huc12_layer: Optional prepared HUC12-layer name.
    """

    def __init__(
        self,
        gpkg_path: str | Path,
        target_huc12: str | None = None,
        outlet_source_id: str | None = None,
        work_crs: str = "EPSG:5070",
        min_hru_area_m2: float = 0.0,
        flowline_layer: str = "prepared_flowlines",
        catchment_layer: str = "prepared_catchments",
        waterbody_layer: str = "prepared_waterbodies",
        huc12_layer: str = "prepared_huc12",
    ) -> None:
        """Initializes source-agnostic routing configuration."""
        self.gpkg_path = Path(gpkg_path)

        self.target_huc12 = (
            str(target_huc12)
            if target_huc12 is not None
            else None
        )

        self.outlet_source_id = (
            str(outlet_source_id)
            if outlet_source_id is not None
            else None
        )

        self.work_crs = str(work_crs)
        self.min_hru_area_m2 = float(min_hru_area_m2)

        self.flowline_layer = str(flowline_layer)
        self.catchment_layer = str(catchment_layer)
        self.waterbody_layer = str(waterbody_layer)
        self.huc12_layer = str(huc12_layer)

    @property
    def output_dir(self) -> Path:
        """Returns the parent directory of the routing GeoPackage."""
        return self.gpkg_path.parent

    @property
    def qa_dir(self) -> Path:
        """Returns the routing QA-output directory."""
        return self.output_dir / "domain_qa"


# =============================================================================
# GIS utility functions
# =============================================================================


def lower_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Returns a copy with lowercase non-geometry attribute fields.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with lowercase non-geometry field names.
    """
    output = gdf.copy()
    geometry_name = output.geometry.name

    output.columns = [
        column if column == geometry_name else column.lower()
        for column in output.columns
    ]

    return output


def list_gpkg_layers(gpkg_path: Path) -> list[str]:
    """Lists available GeoPackage layers.

    Args:
        gpkg_path: GeoPackage path.

    Returns:
        Sorted list of layer names.

    Raises:
        FileNotFoundError: If the GeoPackage does not exist.
    """
    if not gpkg_path.exists():
        raise FileNotFoundError(
            f"GeoPackage does not exist: {gpkg_path}"
        )

    return sorted(listlayers(gpkg_path))


def read_layer(
    gpkg_path: Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Reads a required GeoPackage layer.

    Args:
        gpkg_path: GeoPackage path.
        layer_name: Layer name to read.

    Returns:
        GeoDataFrame with lowercase non-geometry field names.

    Raises:
        ValueError: If layer_name is not found in gpkg_path.
    """
    available = list_gpkg_layers(gpkg_path)

    if layer_name not in available:
        raise ValueError(
            f"Layer '{layer_name}' is absent from {gpkg_path.name}. "
            f"Available layers: {available}"
        )

    return lower_columns(
        gpd.read_file(gpkg_path, layer=layer_name)
    )


def write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
) -> None:
    """Writes a non-empty GeoDataFrame to a GeoPackage layer.

    An existing same-name layer is permanently removed before the new layer is
    written.

    Args:
        gdf: GeoDataFrame to write.
        gpkg_path: Output GeoPackage path.
        layer_name: Output layer name.

    Raises:
        ValueError: If gdf is empty.
    """
    if gdf.empty:
        raise ValueError(
            f"Refusing to write empty GeoPackage layer '{layer_name}'."
        )

    available = list_gpkg_layers(gpkg_path)

    if layer_name in available:
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


def union_geometries(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Dissolves all features from a GeoDataFrame into one geometry.

    Args:
        gdf: GeoDataFrame with an active geometry column.

    Returns:
        Unioned Shapely geometry.
    """
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()

    return gdf.geometry.unary_union


def empty_geodataframe(
    crs: Any,
) -> gpd.GeoDataFrame:
    """Creates an empty GeoDataFrame with a geometry column.

    Args:
        crs: Coordinate reference system for the empty output.

    Returns:
        Empty GeoDataFrame.
    """
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def resolve_downstream_catchments(
    catchments: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
    max_hops: int = 25,
) -> gpd.GeoDataFrame:
    """Finds each catchment's downstream catchment through the flowline graph.

    If the flowline named by ``to_source_id`` has no catchment (for example,
    a tributary that joins at a junction inside the downstream catchment),
    follow that flowline's own ``to_source_id`` until a flowline that owns a
    catchment is found.

    Adds:
        to_catchment_id: Downstream catchment ID, or None.
        to_catchment_source_id: source_id that owns that catchment.
        to_catchment_hops: Flowlines skipped (0 = direct, -1 = none).
        to_catchment_method: direct, via_downstream_flowline,
            no_downstream_link, or unresolved.

    Args:
        catchments: Catchments with source_id, catchment_id, to_source_id.
        flowlines: Flowlines with source_id and to_source_id.
        max_hops: Maximum flowlines to follow before giving up.

    Returns:
        Copy of catchments with the added fields.
    """
    to_map: dict[str, str | None] = {}

    for source_id, to_id in zip(
        flowlines["source_id"].astype(str),
        flowlines["to_source_id"],
    ):
        to_map[source_id] = None if pd.isna(to_id) else str(to_id)

    catchment_by_source: dict[str, str] = {}

    for catchment_id, source_id in zip(
        catchments["catchment_id"].astype(str),
        catchments["source_id"].astype(str),
    ):
        catchment_by_source.setdefault(source_id, catchment_id)

    ids, sources, hop_counts, methods = [], [], [], []

    for catchment_id, source_id, to_id in zip(
        catchments["catchment_id"].astype(str),
        catchments["source_id"].astype(str),
        catchments["to_source_id"],
    ):
        if pd.isna(to_id):
            ids.append(None)
            sources.append(None)
            hop_counts.append(-1)
            methods.append("no_downstream_link")
            continue

        node: str | None = str(to_id)
        visited = {source_id}
        hops = 0
        found = None

        while node is not None and node not in visited and hops <= max_hops:
            visited.add(node)
            candidate = catchment_by_source.get(node)

            if candidate is not None and candidate != catchment_id:
                found = (candidate, node)
                break

            node = to_map.get(node)
            hops += 1

        if found is None:
            ids.append(None)
            sources.append(None)
            hop_counts.append(-1)
            methods.append("unresolved")
        else:
            ids.append(found[0])
            sources.append(found[1])
            hop_counts.append(hops)
            methods.append(
                "direct" if hops == 0 else "via_downstream_flowline"
            )

    result = catchments.copy()
    result["to_catchment_id"] = ids
    result["to_catchment_source_id"] = sources
    result["to_catchment_hops"] = hop_counts
    result["to_catchment_method"] = methods

    return result


# =============================================================================
# Prepared-domain input loader
# =============================================================================


class PreparedDomainLoader:
    """Loads and validates source-agnostic prepared domain layers.

    Required prepared data contract:

        prepared_flowlines:
            source_id
            to_source_id
            geometry

        prepared_catchments:
            catchment_id
            source_id
            geometry
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
        config: RoutingConfig,
    ) -> None:
        """Initializes the prepared-domain loader.

        Args:
            config: Routing configuration.
        """
        self.config = config

    def load(
        self,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame | None,
    ]:
        """Loads prepared flowlines, catchments, and optional supporting layers.

        Returns:
            Tuple of:

                - prepared flowlines,
                - prepared catchments,
                - prepared waterbodies or empty GeoDataFrame,
                - prepared HUC12 polygons or None.

        Raises:
            KeyError: If required prepared-domain fields are missing.
            ValueError: If required prepared layers are unavailable.
        """
        flowlines = read_layer(
            self.config.gpkg_path,
            self.config.flowline_layer,
        )

        catchments = read_layer(
            self.config.gpkg_path,
            self.config.catchment_layer,
        )

        available = set(list_gpkg_layers(self.config.gpkg_path))

        if self.config.waterbody_layer in available:
            waterbodies = read_layer(
                self.config.gpkg_path,
                self.config.waterbody_layer,
            )
        else:
            waterbodies = empty_geodataframe(catchments.crs)

        if self.config.huc12_layer in available:
            huc12s = read_layer(
                self.config.gpkg_path,
                self.config.huc12_layer,
            )
        else:
            huc12s = None

        self._validate_flowlines(flowlines)
        self._validate_catchments(catchments)

        flowlines["source_id"] = flowlines["source_id"].astype(str)

        flowlines["to_source_id"] = (
            flowlines["to_source_id"]
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

        catchments["source_id"] = catchments["source_id"].astype(str)
        catchments["catchment_id"] = (
            catchments["catchment_id"].astype(str)
        )

        return flowlines, catchments, waterbodies, huc12s

    def _validate_flowlines(
        self,
        flowlines: gpd.GeoDataFrame,
    ) -> None:
        """Validates prepared flowline schema and unique source IDs.

        Args:
            flowlines: Prepared flowline GeoDataFrame.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If source_id values are duplicated.
        """
        missing = self.REQUIRED_FLOWLINE_FIELDS - set(
            flowlines.columns
        )

        if missing:
            raise KeyError(
                "Prepared flowline layer missing required fields: "
                f"{sorted(missing)}"
            )

        duplicate_mask = flowlines["source_id"].duplicated(
            keep=False
        )

        if duplicate_mask.any():
            duplicate_ids = (
                flowlines.loc[
                    duplicate_mask,
                    "source_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Prepared flowline source_id values must be unique. "
                f"First duplicate IDs: {duplicate_ids[:10]}"
            )

    def _validate_catchments(
        self,
        catchments: gpd.GeoDataFrame,
    ) -> None:
        """Validates prepared catchment schema.

        Args:
            catchments: Prepared catchment GeoDataFrame.

        Raises:
            KeyError: If required fields are missing.
        """
        missing = self.REQUIRED_CATCHMENT_FIELDS - set(
            catchments.columns
        )

        if missing:
            raise KeyError(
                "Prepared catchment layer missing required fields: "
                f"{sorted(missing)}"
            )


# =============================================================================
# Source-agnostic dendritic router
# =============================================================================


class DendriticRouter:
    """Builds PRMS segments, active HRUs, and catchment routing-review layers.

    The router is source-agnostic. It does not know whether the source network
    came from NHDPlus HR, NHDPlus V2, 3DHP, custom Arc Pro layers, or another
    future discretization.

    It uses only:

        source_id
        to_source_id

    to construct a directed primary-routing graph.

    ``model_catchments`` retains all prepared catchments. Active routed
    catchments are also written separately to ``model_hrus``.
    """

    def __init__(
        self,
        config: RoutingConfig,
        flowlines: gpd.GeoDataFrame,
        catchments: gpd.GeoDataFrame,
        waterbodies: gpd.GeoDataFrame | None = None,
        huc12s: gpd.GeoDataFrame | None = None,
    ) -> None:
        """Initializes the dendritic router.

        Args:
            config: Routing configuration.
            flowlines: Prepared flowline features.
            catchments: Prepared catchment polygons.
            waterbodies: Optional prepared waterbody polygons.
            huc12s: Optional prepared HUC12 polygons.
        """
        self.config = config
        self.flowlines = flowlines.copy()
        self.catchments = catchments.copy()

        if waterbodies is None:
            self.waterbodies = empty_geodataframe(
                crs=catchments.crs
            )
        else:
            self.waterbodies = waterbodies.copy()

        self.huc12s = (
            huc12s.copy()
            if huc12s is not None
            else None
        )

    def build(self) -> dict[str, Any]:
        """Builds routing outputs and writes them to the GeoPackage.

        Returns:
            JSON-compatible routing QA report.
        """
        outlet_source_id = self._select_outlet_source_id()

        graph, upstream_source_ids = self._trace_upstream(
            outlet_source_id
        )

        segments, source_to_segment = self._build_segments(
            upstream_source_ids=upstream_source_ids,
            outlet_source_id=outlet_source_id,
        )

        hrus, model_catchments, unrouted = self._build_catchment_layers(
            source_to_segment=source_to_segment,
            outlet_source_id=outlet_source_id,
        )

        (
            segments,
            hrus,
            model_catchments,
            unrouted,
            model_waterbodies,
        ) = self._add_waterbody_info(
            segments=segments,
            hrus=hrus,
            model_catchments=model_catchments,
            unrouted=unrouted,
        )

        unrouted = self._filter_unrouted_in_domain(
            unrouted=unrouted,
            hrus=hrus,
        )

        report = self._make_qa_report(
            segments=segments,
            hrus=hrus,
            model_catchments=model_catchments,
            unrouted=unrouted,
            graph=graph,
            outlet_source_id=outlet_source_id,
        )

        self._write_outputs(
            segments=segments,
            hrus=hrus,
            model_catchments=model_catchments,
            unrouted=unrouted,
            waterbodies=model_waterbodies,
            report=report,
        )

        return report

    def _select_outlet_source_id(self) -> str:
        """Selects an explicit or automatic routing outlet.

        Returns:
            Selected outlet source_id.

        Raises:
            ValueError: If no explicit or automatic outlet can be identified.
            KeyError: If prepared_huc12 lacks the huc12 field.
        """
        if self.config.outlet_source_id is not None:
            outlet = self.config.outlet_source_id

            if outlet not in set(self.flowlines["source_id"]):
                raise ValueError(
                    f"Configured outlet_source_id '{outlet}' does not occur "
                    "in prepared flowlines."
                )

            return outlet

        if self.huc12s is None or self.huc12s.empty:
            raise ValueError(
                "Set outlet_source_id or provide prepared_huc12 for automatic "
                "outlet selection."
            )

        if self.config.target_huc12 is None:
            raise ValueError(
                "Set target_huc12 or provide outlet_source_id."
            )

        if "huc12" not in self.huc12s.columns:
            raise KeyError(
                "prepared_huc12 must contain a 'huc12' field."
            )

        target_huc = self.huc12s[
            self.huc12s["huc12"].astype(str)
            == self.config.target_huc12
        ].copy()

        if target_huc.empty:
            raise ValueError(
                f"Target HUC12 '{self.config.target_huc12}' was not found "
                "in prepared_huc12."
            )

        target_geometry = union_geometries(target_huc)

        candidates = self.flowlines[
            self.flowlines.intersects(target_geometry)
        ].copy()

        if candidates.empty:
            raise ValueError(
                "No prepared flowlines intersect the target HUC12."
            )

        if "drainage_area_km2" in candidates.columns:
            outlet_row = candidates.sort_values(
                "drainage_area_km2",
                ascending=False,
            ).iloc[0]
        else:
            outlet_row = candidates.iloc[0]

        outlet = str(outlet_row["source_id"])

        tqdm.write(
            f"Auto-selected routing outlet source_id={outlet}"
        )

        return outlet

    def _trace_upstream(
        self,
        outlet_source_id: str,
    ) -> tuple[nx.DiGraph, set[str]]:
        """Builds a directed graph and traces all flowlines upstream of outlet.

        Args:
            outlet_source_id: Selected model outlet source ID.

        Returns:
            Tuple of:

                - directed downstream routing graph,
                - all source IDs upstream of outlet, including outlet.

        Raises:
            ValueError: If outlet_source_id is not found in the graph.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(self.flowlines["source_id"])

        valid_edges = self.flowlines[
            self.flowlines["to_source_id"].notna()
            & (self.flowlines["to_source_id"] != "")
            & (self.flowlines["to_source_id"] != "<NA>")
        ].copy()

        graph.add_edges_from(
            zip(
                valid_edges["source_id"],
                valid_edges["to_source_id"].astype(str),
            )
        )

        if outlet_source_id not in graph:
            raise ValueError(
                f"Outlet source_id '{outlet_source_id}' is absent from the "
                "prepared flowline graph."
            )

        upstream_source_ids = (
            nx.ancestors(graph, outlet_source_id)
            | {outlet_source_id}
        )

        return graph, upstream_source_ids

    def _build_segments(
        self,
        upstream_source_ids: set[str],
        outlet_source_id: str,
    ) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
        """Builds active model stream segments and downstream routing IDs.

        Args:
            upstream_source_ids: Prepared flowline IDs included in the active
                model network.
            outlet_source_id: Selected model outlet source ID.

        Returns:
            Tuple containing active model segments and a source-to-segment ID
            mapping.
        """
        segments = self.flowlines[
            self.flowlines["source_id"].isin(
                upstream_source_ids
            )
        ].copy()

        sort_fields = [
            field
            for field in (
                "drainage_area_km2",
                "stream_order",
                "source_id",
            )
            if field in segments.columns
        ]

        if sort_fields:
            segments = segments.sort_values(
                sort_fields,
                ascending=[False] * len(sort_fields),
            )

        segments = segments.reset_index(drop=True)

        segments["segment_id"] = np.arange(
            1,
            len(segments) + 1,
            dtype=int,
        )

        source_to_segment = dict(
            zip(
                segments["source_id"],
                segments["segment_id"],
            )
        )

        segments["to_segment_id"] = (
            segments["to_source_id"]
            .map(source_to_segment)
            .fillna(0)
            .astype(int)
        )

        segments["is_outlet"] = (
            segments["source_id"] == outlet_source_id
        ).astype(int)

        segments.loc[
            segments["is_outlet"] == 1,
            "to_segment_id",
        ] = 0

        segments = segments.to_crs(self.config.work_crs)
        segments["length_m"] = segments.geometry.length

        return segments, source_to_segment

    def _build_catchment_layers(
        self,
        source_to_segment: dict[str, int],
        outlet_source_id: str,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
    ]:
        """Builds active HRUs, all model catchments, and review subsets.

        Every prepared catchment is retained in the full
        ``model_catchments`` layer. Only connected catchments that trace to the
        selected outlet become active PRMS HRUs.

        Args:
            source_to_segment: Active source_id-to-segment_id mapping.
            outlet_source_id: Selected model outlet source ID.

        Returns:
            Tuple of:

                - active routed PRMS HRUs,
                - all catchments with routing-status fields,
                - non-active catchment review subset.
        """
        catchments = self.catchments.to_crs(
            self.config.work_crs
        ).copy()

        flowline_fields = [
            "source_id",
            "to_source_id",
            "source_name",
            "drainage_area_km2",
            "stream_order",
            "divergence",
        ]

        available_flowline_fields = [
            field
            for field in flowline_fields
            if field in self.flowlines.columns
        ]

        flowline_lookup = self.flowlines[
            available_flowline_fields
        ].copy()

        catchments = catchments.merge(
            flowline_lookup,
            on="source_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_flowline"),
        )

        catchments["area_m2"] = catchments.geometry.area

        catchments["segment_id"] = (
            catchments["source_id"]
            .map(source_to_segment)
            .astype("Int64")
        )

        catchments["to_segment_id"] = (
            catchments["to_source_id"]
            .map(source_to_segment)
            .astype("Int64")
        )

        catchments = resolve_downstream_catchments(
            catchments=catchments,
            flowlines=self.flowlines,
        )

        catchments["is_active_hru"] = (
            catchments["segment_id"].notna()
        ).astype(int)

        catchments["is_outlet"] = (
            catchments["source_id"] == outlet_source_id
        ).astype(int)

        source_flowline_ids = set(self.flowlines["source_id"])

        catchments["is_sink"] = (
            catchments["source_id"].isin(source_flowline_ids)
            & catchments["to_source_id"].isna()
            & (catchments["is_outlet"] == 0)
        ).astype(int)

        catchments["routing_status"] = "routed"
        catchments["routing_detail"] = "active_network"

        missing_flowline = ~catchments["source_id"].isin(
            source_flowline_ids
        )

        sink = (
            catchments["source_id"].isin(source_flowline_ids)
            & catchments["to_source_id"].isna()
            & (catchments["is_outlet"] == 0)
        )

        downstream_missing = (
            catchments["to_source_id"].notna()
            & ~catchments["to_source_id"].isin(
                source_flowline_ids
            )
        )

        outside_active_network = (
            catchments["to_source_id"].notna()
            & catchments["to_segment_id"].isna()
            & ~downstream_missing
            & (catchments["is_active_hru"] == 0)
        )

        basin_outlet = catchments["is_outlet"] == 1

        catchments.loc[
            missing_flowline,
            ["routing_status", "routing_detail"],
        ] = [
            "missing_flowline",
            "catchment source_id has no matching prepared flowline",
        ]

        catchments.loc[
            sink,
            ["routing_status", "routing_detail"],
        ] = [
            "sink",
            "source flowline has no primary downstream link",
        ]

        catchments.loc[
            downstream_missing,
            ["routing_status", "routing_detail"],
        ] = [
            "downstream_missing",
            "to_source_id is absent from prepared flowlines",
        ]

        catchments.loc[
            outside_active_network,
            ["routing_status", "routing_detail"],
        ] = [
            "outside_active_network",
            "catchment does not trace to selected model outlet",
        ]

        catchments.loc[
            basin_outlet,
            ["routing_status", "routing_detail"],
        ] = [
            "routed",
            "selected_model_outlet",
        ]

        catchments["manual_review"] = (
            catchments["routing_status"] != "routed"
        ).astype(int)

        catchments["routing_target"] = np.where(
            catchments["to_segment_id"].notna(),
            "segment_"
            + catchments["to_segment_id"].astype(str),
            np.where(
                catchments["to_source_id"].notna(),
                "source_"
                + catchments["to_source_id"].astype(str),
                "none",
            ),
        )

        model_catchments = catchments.copy()

        hrus = model_catchments[
            model_catchments["is_active_hru"] == 1
        ].copy()

        hrus = hrus.sort_values(
            "segment_id"
        ).reset_index(drop=True)

        hrus["hru_id"] = np.arange(
            1,
            len(hrus) + 1,
            dtype=int,
        )

        hrus["parent_source_id"] = hrus["source_id"]

        hrus["small_hru"] = (
            hrus["area_m2"] < self.config.min_hru_area_m2
        ).astype(int)

        unrouted = model_catchments[
            model_catchments["is_active_hru"] == 0
        ].copy()

        return hrus, model_catchments, unrouted

    def _add_waterbody_info(
        self,
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        model_catchments: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
    ]:
        """Adds waterbody flags and fractional coverage to model layers.

        Args:
            segments: Active model stream segments.
            hrus: Active routed model HRUs.
            model_catchments: All catchments retained for routing review.
            unrouted: Non-active catchment review subset.

        Returns:
            Updated segments, HRUs, all model catchments, unrouted subset, and
            projected waterbodies.
        """
        segments = segments.copy()
        hrus = hrus.copy()
        model_catchments = model_catchments.copy()
        unrouted = unrouted.copy()

        waterbodies = self.waterbodies.to_crs(
            self.config.work_crs
        ).copy()

        waterbodies = waterbodies[
            waterbodies.geometry.notna()
            & ~waterbodies.geometry.is_empty
        ].copy()

        if waterbodies.empty:
            segments["waterbody_id"] = None
            segments["in_waterbody"] = 0

            model_catchments["waterbody_frac"] = 0.0
            hrus["waterbody_frac"] = 0.0
            unrouted["waterbody_frac"] = 0.0

            unrouted["reason"] = unrouted["routing_status"]

            return (
                segments,
                hrus,
                model_catchments,
                unrouted,
                waterbodies,
            )

        if "waterbody_id" not in waterbodies.columns:
            waterbodies["waterbody_id"] = (
                waterbodies.index.astype(str)
            )

        waterbodies["geometry"] = waterbodies.geometry.buffer(0)

        segment_midpoints = segments[
            ["segment_id", "geometry"]
        ].copy()

        segment_midpoints["geometry"] = (
            segments.geometry.interpolate(
                0.5,
                normalized=True,
            )
        )

        segment_hits = gpd.sjoin(
            segment_midpoints,
            waterbodies[
                ["waterbody_id", "geometry"]
            ],
            how="left",
            predicate="within",
        )

        segment_to_waterbody = (
            segment_hits
            .drop_duplicates("segment_id")
            .set_index("segment_id")["waterbody_id"]
        )

        segments["waterbody_id"] = (
            segments["segment_id"].map(segment_to_waterbody)
        )

        segments["in_waterbody"] = (
            segments["waterbody_id"].notna()
        ).astype(int)

        model_catchments["waterbody_frac"] = (
            self._waterbody_fraction(
                polygons=model_catchments,
                key_field="catchment_id",
                waterbodies=waterbodies,
            )
        )

        water_fraction_lookup = model_catchments.set_index(
            "catchment_id"
        )["waterbody_frac"]

        hrus["waterbody_frac"] = (
            hrus["catchment_id"]
            .map(water_fraction_lookup)
            .fillna(0.0)
        )

        unrouted["waterbody_frac"] = (
            unrouted["catchment_id"]
            .map(water_fraction_lookup)
            .fillna(0.0)
        )

        unrouted["reason"] = np.where(
            unrouted["routing_status"] == "sink",
            "sink",
            np.where(
                unrouted["waterbody_frac"] >= 0.25,
                "waterbody",
                unrouted["routing_status"],
            ),
        )

        return (
            segments,
            hrus,
            model_catchments,
            unrouted,
            waterbodies,
        )

    @staticmethod
    def _waterbody_fraction(
        polygons: gpd.GeoDataFrame,
        key_field: str,
        waterbodies: gpd.GeoDataFrame,
    ) -> pd.Series:
        """Calculates waterbody fraction for each polygon.

        Args:
            polygons: Polygon GeoDataFrame.
            key_field: Unique polygon identifier.
            waterbodies: Waterbody polygons in the same projected CRS.

        Returns:
            Series indexed by key_field with fractions from 0 through 1.
        """
        if polygons.empty:
            return pd.Series(dtype=float)

        intersections = gpd.overlay(
            polygons[[key_field, "geometry"]],
            waterbodies[["geometry"]],
            how="intersection",
        )

        water_area = (
            intersections.geometry.area
            .groupby(intersections[key_field])
            .sum()
        )

        polygon_area = polygons.set_index(
            key_field
        ).geometry.area

        return (
            water_area.reindex(polygon_area.index).fillna(0.0)
            / polygon_area
        ).clip(0.0, 1.0)

    @staticmethod
    def _filter_unrouted_in_domain(
        unrouted: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Filters review catchments to the active-domain spatial envelope.

        The full unfiltered catchment layer remains available as
        ``model_catchments``. This filtered layer is only a convenience review
        subset for mapping likely in-domain sinks and disconnected catchments.

        Args:
            unrouted: All non-active catchments.
            hrus: Active routed HRUs.

        Returns:
            Unrouted catchments whose representative point is inside the convex
            hull of active HRUs.
        """
        if unrouted.empty or hrus.empty:
            return unrouted.copy()

        active_domain_hull = union_geometries(hrus).convex_hull

        in_domain = (
            unrouted.geometry
            .representative_point()
            .within(active_domain_hull)
        )

        return unrouted[in_domain].copy()

    @staticmethod
    def _make_qa_report(
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        model_catchments: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
        graph: nx.DiGraph,
        outlet_source_id: str,
    ) -> dict[str, Any]:
        """Creates routing, topology, and catchment-status QA metrics.

        Args:
            segments: Active routed stream segments.
            hrus: Active routed PRMS HRUs.
            model_catchments: Complete catchment review layer.
            unrouted: In-domain non-active catchment review subset.
            graph: Complete prepared-flowline routing graph.
            outlet_source_id: Active model outlet source ID.

        Returns:
            JSON-compatible QA report.
        """
        model_source_ids = set(segments["source_id"])

        model_graph = graph.subgraph(model_source_ids)

        status_counts = (
            model_catchments["routing_status"]
            .value_counts(dropna=False)
            .to_dict()
        )

        report: dict[str, Any] = {
            "outlet_source_id": outlet_source_id,
            "n_segments": int(len(segments)),
            "n_active_hrus": int(len(hrus)),
            "n_model_catchments": int(len(model_catchments)),
            "n_unrouted_catchments_in_domain": int(len(unrouted)),
            "is_dag": bool(
                nx.is_directed_acyclic_graph(model_graph)
            ),
            "all_segments_reach_outlet": bool(
                all(
                    node == outlet_source_id
                    or nx.has_path(
                        model_graph,
                        node,
                        outlet_source_id,
                    )
                    for node in model_graph.nodes
                )
            ),
            "null_downstream_nonoutlet": int(
                (
                    (segments["to_segment_id"] == 0)
                    & (segments["is_outlet"] == 0)
                ).sum()
            ),
            "duplicate_parent_source_id": int(
                hrus["parent_source_id"].duplicated().sum()
            ),
            "segments_without_hru": int(
                (~segments["source_id"].isin(hrus["source_id"])).sum()
            ),
            "segments_in_waterbody": int(
                segments["in_waterbody"].sum()
            ),
            "small_hrus": int(hrus["small_hru"].sum()),
            "routed_area_km2": round(
                hrus["area_m2"].sum() / 1_000_000,
                3,
            ),
            "all_catchment_area_km2": round(
                model_catchments["area_m2"].sum() / 1_000_000,
                3,
            ),
            "unrouted_area_km2_in_domain": round(
                unrouted["area_m2"].sum() / 1_000_000,
                3,
            ),
            "unrouted_waterbody_area_km2_in_domain": round(
                unrouted.loc[
                    unrouted["reason"] == "waterbody",
                    "area_m2",
                ].sum()
                / 1_000_000,
                3,
            ),
            "sink_count": int(
                model_catchments["is_sink"].sum()
            ),
            "manual_review_count": int(
                model_catchments["manual_review"].sum()
            ),
            "routing_status_counts": {
                str(key): int(value)
                for key, value in status_counts.items()
            },
            "n_to_catchment_via_downstream_flowline": int(
                (
                        model_catchments["to_catchment_method"]
                        == "via_downstream_flowline"
                ).sum()
            ),
            "n_to_catchment_unresolved": int(
                (
                        model_catchments["to_catchment_method"]
                        == "unresolved"
                ).sum()
            ),
        }

        if "drainage_area_km2" in segments.columns:
            outlet_area = segments.loc[
                segments["is_outlet"] == 1,
                "drainage_area_km2",
            ]

            report["outlet_drainage_area_km2"] = (
                float(outlet_area.iloc[0])
                if not outlet_area.empty
                else None
            )

            if report["outlet_drainage_area_km2"]:
                report["routed_area_difference_pct"] = round(
                    100.0
                    * (
                        report["routed_area_km2"]
                        - report["outlet_drainage_area_km2"]
                    )
                    / report["outlet_drainage_area_km2"],
                    3,
                )
            else:
                report["routed_area_difference_pct"] = None

        return report

    def _write_outputs(
        self,
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        model_catchments: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
        waterbodies: gpd.GeoDataFrame,
        report: dict[str, Any],
    ) -> None:
        """Writes model layers, routing tables, and QA report.

        Args:
            segments: Active model stream segments.
            hrus: Active routed PRMS HRUs.
            model_catchments: All catchments with routing review attributes.
            unrouted: Filtered in-domain manual-review subset.
            waterbodies: Projected model waterbody context.
            report: Routing QA metrics.
        """
        self.config.qa_dir.mkdir(parents=True, exist_ok=True)

        segment_fields = [
            "segment_id",
            "to_segment_id",
            "source_id",
            "to_source_id",
            "source_type",
            "source_name",
            "drainage_area_km2",
            "stream_order",
            "slope",
            "divergence",
            "is_outlet",
            "length_m",
            "waterbody_id",
            "in_waterbody",
            "geometry",
        ]

        hru_fields = [
            "hru_id",
            "catchment_id",
            "segment_id",
            "to_segment_id",
            "source_id",
            "to_source_id",
            "parent_source_id",
            "area_m2",
            "waterbody_frac",
            "small_hru",
            "is_outlet",
            "geometry",
            "to_catchment_id",
            "to_catchment_source_id",
            "to_catchment_hops",
            "to_catchment_method",
        ]

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
            "geometry",
            "to_catchment_id",
            "to_catchment_source_id",
            "to_catchment_hops",
            "to_catchment_method",
        ]

        unrouted_fields = [
            "catchment_id",
            "source_id",
            "to_source_id",
            "segment_id",
            "to_segment_id",
            "routing_target",
            "routing_status",
            "routing_detail",
            "is_sink",
            "manual_review",
            "reason",
            "area_m2",
            "waterbody_frac",
            "geometry",
            "to_catchment_id",
            "to_catchment_source_id",
            "to_catchment_hops",
            "to_catchment_method",
        ]

        segment_out = segments[
            [
                field
                for field in segment_fields
                if field in segments.columns
            ]
        ].copy()

        hru_out = hrus[
            [
                field
                for field in hru_fields
                if field in hrus.columns
            ]
        ].copy()

        catchment_out = model_catchments[
            [
                field
                for field in catchment_fields
                if field in model_catchments.columns
            ]
        ].copy()

        unrouted_out = unrouted[
            [
                field
                for field in unrouted_fields
                if field in unrouted.columns
            ]
        ].copy()

        write_layer(
            segment_out,
            self.config.gpkg_path,
            "model_segments",
        )

        write_layer(
            hru_out,
            self.config.gpkg_path,
            "model_hrus",
        )

        write_layer(
            catchment_out,
            self.config.gpkg_path,
            "model_catchments",
        )

        if not unrouted_out.empty:
            write_layer(
                unrouted_out,
                self.config.gpkg_path,
                "unrouted_catchments",
            )

        if not waterbodies.empty:
            write_layer(
                waterbodies,
                self.config.gpkg_path,
                "model_waterbodies",
            )

        segment_out.drop(
            columns="geometry"
        ).to_csv(
            self.config.output_dir / "segments.csv",
            index=False,
        )

        hru_out.drop(
            columns="geometry"
        ).to_csv(
            self.config.output_dir / "hru_to_segment.csv",
            index=False,
        )

        segment_out[
            [
                "segment_id",
                "to_segment_id",
                "source_id",
                "to_source_id",
            ]
        ].to_csv(
            self.config.output_dir / "segment_routing.csv",
            index=False,
        )

        catchment_out.drop(
            columns="geometry"
        ).to_csv(
            self.config.output_dir / "model_catchments.csv",
            index=False,
        )

        (self.config.qa_dir / "routing_qa.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )


# =============================================================================
# Standalone runner
# =============================================================================


def main() -> None:
    """Loads prepared layers and writes source-agnostic routing outputs."""
    gpkg_path = Path(
        r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
        r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
        r"\PlusHR_UG.gdb"
    )

    config = RoutingConfig(
        gpkg_path=gpkg_path,
        target_huc12="100200080507",
        flowline_layer="prepared_flowlines_topology_repaired",
        catchment_layer="prepared_catchments_topology_repaired",
        outlet_source_id="23002900012900",
        work_crs="EPSG:5070",
        min_hru_area_m2=0.0,
    )

    loader = PreparedDomainLoader(config)

    flowlines, catchments, waterbodies, huc12s = loader.load()

    router = DendriticRouter(
        config=config,
        flowlines=flowlines,
        catchments=catchments,
        waterbodies=waterbodies,
        huc12s=huc12s,
    )

    report = router.build()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()