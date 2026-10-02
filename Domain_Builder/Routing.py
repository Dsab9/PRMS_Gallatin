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
    model_hrus
    unrouted_catchments
    model_waterbodies
    segments.csv
    hru_to_segment.csv
    segment_routing.csv
    domain_qa/routing_qa.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Configure the conda PROJ directory before importing GIS libraries.
_PROJ_DIR = Path(sys.prefix) / "Library" / "share" / "proj"

if not (_PROJ_DIR / "proj.db").exists():
    raise FileNotFoundError(
        f"Conda PROJ database was not found: {_PROJ_DIR / 'proj.db'}"
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


class RoutingConfig:
    """Stores source-agnostic routing and PRMS-domain settings.

    Args:
        gpkg_path: GeoPackage holding prepared domain layers.
        target_huc12: HUC12 used for outlet auto-selection.
        outlet_source_id: Optional explicit source_id of the model outlet.
        work_crs: Projected CRS for length and area calculations.
        min_hru_area_m2: HRUs below this area are flagged but retained.
        flowline_layer: Standardized prepared-flowline layer name.
        catchment_layer: Standardized prepared-catchment layer name.
        waterbody_layer: Standardized prepared-waterbody layer name.
        huc12_layer: Standardized prepared-HUC12 layer name.
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
        self.target_huc12 = str(target_huc12) if target_huc12 else None
        self.outlet_source_id = (
            str(outlet_source_id)
            if outlet_source_id is not None
            else None
        )
        self.work_crs = str(work_crs)
        self.min_hru_area_m2 = float(min_hru_area_m2)

        self.flowline_layer = flowline_layer
        self.catchment_layer = catchment_layer
        self.waterbody_layer = waterbody_layer
        self.huc12_layer = huc12_layer

    @property
    def output_dir(self) -> Path:
        """Returns the parent directory of the routing GeoPackage."""
        return self.gpkg_path.parent

    @property
    def qa_dir(self) -> Path:
        """Returns the routing QA directory."""
        return self.output_dir / "domain_qa"


def lower_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalizes GeoDataFrame non-geometry column names to lowercase.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with lowercase field names.
    """
    output = gdf.copy()
    geometry_name = output.geometry.name
    output.columns = [
        column if column == geometry_name else column.lower()
        for column in output.columns
    ]
    return output


def list_gpkg_layers(gpkg_path: Path) -> list[str]:
    """Lists available layers in a GeoPackage.

    Args:
        gpkg_path: GeoPackage path.

    Returns:
        Sorted layer names.
    """
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage does not exist: {gpkg_path}")

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
        GeoDataFrame with lowercase field names.

    Raises:
        ValueError: If layer does not exist.
    """
    layers = list_gpkg_layers(gpkg_path)

    if layer_name not in layers:
        raise ValueError(
            f"Layer '{layer_name}' is absent from {gpkg_path.name}. "
            f"Available layers: {layers}"
        )

    return lower_columns(gpd.read_file(gpkg_path, layer=layer_name))


def write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
) -> None:
    """Writes a GeoDataFrame to a GeoPackage layer, replacing existing layer.

    Args:
        gdf: GeoDataFrame to write.
        gpkg_path: Output GeoPackage.
        layer_name: Output layer name.

    Raises:
        ValueError: If gdf is empty.
    """
    if gdf.empty:
        raise ValueError(f"Refusing to write empty layer: {layer_name}")

    existing = list_gpkg_layers(gpkg_path)
    if layer_name in existing:
        remove(gpkg_path, layer=layer_name, driver="GPKG")

    gdf.to_file(
        gpkg_path,
        layer=layer_name,
        driver="GPKG",
        index=False,
        engine="fiona",
    )


def union_geometries(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Dissolves all geometries from a GeoDataFrame.

    Args:
        gdf: GeoDataFrame.

    Returns:
        Unioned Shapely geometry.
    """
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()

    return gdf.geometry.unary_union


class PreparedDomainLoader:
    """Loads and validates source-agnostic prepared domain layers."""

    REQUIRED_FLOWLINE_FIELDS = {
        "source_id",
        "to_source_id",
    }

    REQUIRED_CATCHMENT_FIELDS = {
        "catchment_id",
        "source_id",
    }

    def __init__(self, config: RoutingConfig) -> None:
        """Initializes a prepared-domain loader.

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
        """Loads prepared source layers.

        Returns:
            Tuple of flowlines, catchments, waterbodies, and HUC12 polygons.
            Waterbodies and HUC12 polygons may be empty/None.

        Raises:
            KeyError: If required prepared schema fields are missing.
        """
        flowlines = read_layer(
            self.config.gpkg_path,
            self.config.flowline_layer,
        )
        catchments = read_layer(
            self.config.gpkg_path,
            self.config.catchment_layer,
        )

        layers = set(list_gpkg_layers(self.config.gpkg_path))

        waterbodies = (
            read_layer(
                self.config.gpkg_path,
                self.config.waterbody_layer,
            )
            if self.config.waterbody_layer in layers
            else gpd.GeoDataFrame(
                geometry=[],
                crs=catchments.crs,
            )
        )

        huc12s = (
            read_layer(
                self.config.gpkg_path,
                self.config.huc12_layer,
            )
            if self.config.huc12_layer in layers
            else None
        )

        self._validate_flowlines(flowlines)
        self._validate_catchments(catchments)

        flowlines["source_id"] = flowlines["source_id"].astype(str)
        flowlines["to_source_id"] = flowlines["to_source_id"].astype("string")
        catchments["source_id"] = catchments["source_id"].astype(str)
        catchments["catchment_id"] = catchments["catchment_id"].astype(str)

        return flowlines, catchments, waterbodies, huc12s

    def _validate_flowlines(
        self,
        flowlines: gpd.GeoDataFrame,
    ) -> None:
        """Validates generic prepared flowline schema.

        Args:
            flowlines: Prepared flowline layer.

        Raises:
            KeyError: If required fields are missing.
        """
        missing = self.REQUIRED_FLOWLINE_FIELDS - set(flowlines.columns)

        if missing:
            raise KeyError(
                f"Prepared flowline layer missing fields: {sorted(missing)}"
            )

        if flowlines["source_id"].duplicated().any():
            duplicate_ids = flowlines.loc[
                flowlines["source_id"].duplicated(keep=False),
                "source_id",
            ].unique()

            raise ValueError(
                "Prepared flowline source_id values must be unique. "
                f"First duplicates: {duplicate_ids[:10].tolist()}"
            )

    def _validate_catchments(
        self,
        catchments: gpd.GeoDataFrame,
    ) -> None:
        """Validates generic prepared catchment schema.

        Args:
            catchments: Prepared catchment layer.

        Raises:
            KeyError: If required fields are missing.
        """
        missing = self.REQUIRED_CATCHMENT_FIELDS - set(catchments.columns)

        if missing:
            raise KeyError(
                f"Prepared catchment layer missing fields: {sorted(missing)}"
            )


class DendriticRouter:
    """Builds PRMS model segments and parent HRUs from prepared source layers.

    The router is intentionally source-agnostic. It requires only a source ID
    and a precomputed primary downstream source ID for every flowline.
    """

    def __init__(
        self,
        config: RoutingConfig,
        flowlines: gpd.GeoDataFrame,
        catchments: gpd.GeoDataFrame,
        waterbodies: gpd.GeoDataFrame | None = None,
        huc12s: gpd.GeoDataFrame | None = None,
    ) -> None:
        """Initializes the source-agnostic dendritic router.

        Args:
            config: Routing configuration.
            flowlines: Prepared flowlines.
            catchments: Prepared catchments.
            waterbodies: Optional prepared waterbodies.
            huc12s: Optional prepared HUC12 polygons.
        """
        self.config = config
        self.flowlines = flowlines.copy()
        self.catchments = catchments.copy()
        self.waterbodies = (
            waterbodies.copy()
            if waterbodies is not None
            else gpd.GeoDataFrame(
                geometry=[],
                crs=catchments.crs,
            )
        )
        self.huc12s = huc12s.copy() if huc12s is not None else None

    def build(self) -> dict[str, Any]:
        """Builds routing outputs and writes them to the GeoPackage.

        Returns:
            Routing QA report.
        """
        outlet_source_id = self._select_outlet_source_id()
        graph, upstream_source_ids = self._trace_upstream(
            outlet_source_id
        )

        segments, source_to_segment = self._build_segments(
            upstream_source_ids,
            outlet_source_id,
        )

        hrus, unrouted = self._build_hrus(source_to_segment)

        (
            segments,
            hrus,
            unrouted,
            model_waterbodies,
        ) = self._add_waterbody_info(
            segments,
            hrus,
            unrouted,
        )

        unrouted = self._filter_unrouted_in_domain(
            unrouted,
            hrus,
        )

        report = self._make_qa_report(
            segments,
            hrus,
            unrouted,
            graph,
            outlet_source_id,
        )

        self._write_outputs(
            segments,
            hrus,
            unrouted,
            model_waterbodies,
            report,
        )

        return report

    def _select_outlet_source_id(self) -> str:
        """Selects an explicit or automatic outlet source ID.

        Returns:
            Outlet source_id.

        Raises:
            ValueError: If no explicit/automatic outlet can be determined.
        """
        if self.config.outlet_source_id is not None:
            if self.config.outlet_source_id not in set(
                self.flowlines["source_id"]
            ):
                raise ValueError(
                    f"Configured outlet source_id "
                    f"{self.config.outlet_source_id} is not in prepared "
                    "flowlines."
                )

            return self.config.outlet_source_id

        if self.huc12s is None or self.huc12s.empty:
            raise ValueError(
                "Set outlet_source_id or provide prepared_huc12 for "
                "automatic outlet selection."
            )

        if self.config.target_huc12 is None:
            raise ValueError(
                "Set target_huc12 or provide explicit outlet_source_id."
            )

        if "huc12" not in self.huc12s.columns:
            raise KeyError(
                "prepared_huc12 requires a 'huc12' field for automatic "
                "outlet selection."
            )

        target = self.huc12s[
            self.huc12s["huc12"].astype(str) == self.config.target_huc12
        ]

        if target.empty:
            raise ValueError(
                f"Target HUC12 {self.config.target_huc12} was not found "
                "in prepared_huc12."
            )

        target_geometry = union_geometries(target)

        candidates = self.flowlines[
            self.flowlines.intersects(target_geometry)
        ].copy()

        if candidates.empty:
            raise ValueError(
                "No prepared flowlines intersect the target HUC12."
            )

        area_field = (
            "drainage_area_km2"
            if "drainage_area_km2" in candidates.columns
            else None
        )

        if area_field is not None:
            outlet_row = candidates.sort_values(
                area_field,
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
        """Builds routing graph and identifies all source IDs upstream of outlet.

        Args:
            outlet_source_id: Selected outlet source ID.

        Returns:
            Directed graph and upstream source-ID set.

        Raises:
            ValueError: If outlet is not found.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(self.flowlines["source_id"])

        active_edges = self.flowlines[
            self.flowlines["to_source_id"].notna()
            & (self.flowlines["to_source_id"] != "")
            & (self.flowlines["to_source_id"] != "<NA>")
        ]

        graph.add_edges_from(
            zip(
                active_edges["source_id"],
                active_edges["to_source_id"].astype(str),
            )
        )

        if outlet_source_id not in graph:
            raise ValueError(
                f"Outlet source_id={outlet_source_id} is not in graph."
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
        """Creates contiguous model segment IDs and downstream links.

        Args:
            upstream_source_ids: Source flowline IDs included in the model.
            outlet_source_id: Selected outlet source ID.

        Returns:
            Model segment GeoDataFrame and source-ID-to-segment-ID lookup.
        """
        segments = self.flowlines[
            self.flowlines["source_id"].isin(upstream_source_ids)
        ].copy()

        sort_fields = [
            field
            for field in ("drainage_area_km2", "stream_order", "source_id")
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

    def _build_hrus(
        self,
        source_to_segment: dict[str, int],
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Creates parent HRUs and separate unrouted catchment features.

        Args:
            source_to_segment: Flowline source ID to model segment ID mapping.

        Returns:
            Routed HRU polygons and unrouted catchment polygons.
        """
        catchments = self.catchments.to_crs(
            self.config.work_crs
        ).copy()

        hrus = catchments[
            catchments["source_id"].isin(source_to_segment)
        ].copy()

        hrus["segment_id"] = (
            hrus["source_id"]
            .map(source_to_segment)
            .astype(int)
        )

        hrus["area_m2"] = hrus.geometry.area

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

        unrouted = catchments[
            ~catchments["source_id"].isin(source_to_segment)
        ].copy()

        unrouted["area_m2"] = unrouted.geometry.area

        return hrus, unrouted

    def _add_waterbody_info(
        self,
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
    ) -> tuple[
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
        gpd.GeoDataFrame,
    ]:
        """Calculates waterbody flags and fractional coverage.

        Args:
            segments: Model segments.
            hrus: Routed HRUs.
            unrouted: Unrouted catchments.

        Returns:
            Updated segments, HRUs, unrouted catchments, and waterbodies.
        """
        segments = segments.copy()
        hrus = hrus.copy()
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
            hrus["waterbody_frac"] = 0.0
            unrouted["waterbody_frac"] = 0.0
            unrouted["reason"] = "other"

            return segments, hrus, unrouted, waterbodies

        if "waterbody_id" not in waterbodies.columns:
            waterbodies["waterbody_id"] = waterbodies.index.astype(str)

        waterbodies["geometry"] = waterbodies.geometry.buffer(0)

        midpoint = segments[["segment_id", "geometry"]].copy()
        midpoint["geometry"] = segments.geometry.interpolate(
            0.5,
            normalized=True,
        )

        hits = gpd.sjoin(
            midpoint,
            waterbodies[["waterbody_id", "geometry"]],
            how="left",
            predicate="within",
        )

        segment_to_waterbody = (
            hits.drop_duplicates("segment_id")
            .set_index("segment_id")["waterbody_id"]
        )

        segments["waterbody_id"] = segments["segment_id"].map(
            segment_to_waterbody
        )
        segments["in_waterbody"] = (
            segments["waterbody_id"].notna()
        ).astype(int)

        hrus["waterbody_frac"] = self._waterbody_fraction(
            hrus,
            "hru_id",
            waterbodies,
        )

        unrouted["waterbody_frac"] = self._waterbody_fraction(
            unrouted,
            "catchment_id",
            waterbodies,
        )

        unrouted["reason"] = np.where(
            unrouted["waterbody_frac"] >= 0.25,
            "waterbody",
            "other",
        )

        return segments, hrus, unrouted, waterbodies

    @staticmethod
    def _waterbody_fraction(
        polygons: gpd.GeoDataFrame,
        key_field: str,
        waterbodies: gpd.GeoDataFrame,
    ) -> pd.Series:
        """Calculates waterbody fraction in polygons.

        Args:
            polygons: Polygons to evaluate.
            key_field: Unique polygon ID field.
            waterbodies: Waterbody polygons.

        Returns:
            Fraction series indexed by key_field.
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
            water_area.reindex(polygon_area.index).fillna(0)
            / polygon_area
        ).clip(0, 1)

    @staticmethod
    def _filter_unrouted_in_domain(
        unrouted: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Keeps unrouted catchments inside the routed-domain envelope.

        Args:
            unrouted: All unrouted catchments.
            hrus: Active routed HRUs.

        Returns:
            In-domain unrouted catchments.
        """
        if unrouted.empty or hrus.empty:
            return unrouted.copy()

        domain_hull = union_geometries(hrus).convex_hull
        in_domain = unrouted.geometry.representative_point().within(
            domain_hull
        )

        return unrouted[in_domain].copy()

    @staticmethod
    def _make_qa_report(
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
        graph: nx.DiGraph,
        outlet_source_id: str,
    ) -> dict[str, Any]:
        """Creates routing and area QA report.

        Args:
            segments: Model segments.
            hrus: Routed HRUs.
            unrouted: In-domain unrouted catchments.
            graph: Complete source routing graph.
            outlet_source_id: Selected source-network outlet ID.

        Returns:
            JSON-compatible QA report.
        """
        model_source_ids = set(segments["source_id"])
        model_graph = graph.subgraph(model_source_ids)

        report: dict[str, Any] = {
            "outlet_source_id": outlet_source_id,
            "n_segments": int(len(segments)),
            "n_hrus": int(len(hrus)),
            "is_dag": bool(nx.is_directed_acyclic_graph(model_graph)),
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
            "unrouted_area_km2": round(
                unrouted["area_m2"].sum() / 1_000_000,
                3,
            ),
            "unrouted_waterbody_area_km2": round(
                unrouted.loc[
                    unrouted["reason"] == "waterbody",
                    "area_m2",
                ].sum()
                / 1_000_000,
                3,
            ),
            "n_unrouted_catchments": int(len(unrouted)),
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
                    100
                    * (
                        report["routed_area_km2"]
                        - report["outlet_drainage_area_km2"]
                    )
                    / report["outlet_drainage_area_km2"],
                    3,
                )

        return report

    def _write_outputs(
        self,
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
        waterbodies: gpd.GeoDataFrame,
        report: dict[str, Any],
    ) -> None:
        """Writes model layers, routing tables, and QA report.

        Args:
            segments: Model segments.
            hrus: Parent HRUs.
            unrouted: In-domain unrouted catchments.
            waterbodies: Model waterbody context.
            report: Routing QA report.
        """
        self.config.qa_dir.mkdir(parents=True, exist_ok=True)

        write_layer(
            segments,
            self.config.gpkg_path,
            "model_segments",
        )
        write_layer(
            hrus,
            self.config.gpkg_path,
            "model_hrus",
        )

        if not unrouted.empty:
            write_layer(
                unrouted,
                self.config.gpkg_path,
                "unrouted_catchments",
            )

        if not waterbodies.empty:
            write_layer(
                waterbodies,
                self.config.gpkg_path,
                "model_waterbodies",
            )

        segments.drop(columns="geometry").to_csv(
            self.config.output_dir / "segments.csv",
            index=False,
        )

        hrus.drop(columns="geometry").to_csv(
            self.config.output_dir / "hru_to_segment.csv",
            index=False,
        )

        segments[
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

        (self.config.qa_dir / "routing_qa.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    """Loads prepared layers and writes PRMS routing domain outputs."""
    gpkg_path = Path(
        r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
        r"\PRMS_Gallatin\Domain_Builder\Domain_Files\nhdhr.gpkg"
    )

    config = RoutingConfig(
        gpkg_path=gpkg_path,
        target_huc12="100200080507",
        outlet_source_id=None,
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