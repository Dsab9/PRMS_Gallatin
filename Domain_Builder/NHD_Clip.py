r"""Clip a local NHDPlus HR geodatabase to a watershed shape for PRMS routing.

Workflow:

    1. Load a watershed polygon (shapefile, GeoPackage, or GDB layer).
    2. Read NHDPlus HR catchments, flowlines, VAA table, and waterbodies from
       a local HU4/HU8 geodatabase (no web service required).
    3. Keep catchments whose reference point (centroid by default) lies
       inside the watershed. Edge catchments that merely touch the shape are
       dropped; catchments without flowlines (sinks) are preserved.
    4. Keep flowlines that share an nhdplusid with the kept catchments, and
       join the value-added attributes (hydroseq, dnhydroseq, totdasqkm...).
    5. Keep waterbodies whose reference point lies inside the watershed.
    6. Write merged_catchments, merged_flowlines, merged_waterbodies, and
       watershed_boundary to the working GeoPackage.

The merged_* layers match the names produced by NHD_PlusHR.py, so
LocalNHDPlusHRLoader.load_and_prepare() can be run on the output unchanged.

Place this file in the same folder as NHD_PlusHR.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from tqdm import tqdm

# Importing NHD_PlusHR also configures the conda PROJ directory.
import NHD_PlusHR as nhd_source
from NHD_PlusHR import (
    LocalNHDPlusHRLoader,
    NHDPlusHRConfig,
    list_gpkg_layers,
    lower_columns,
)

BATCH_SIZE = 20000


class WatershedClipConfig:
    """Stores settings for clipping a local NHDPlus HR dataset.

    Args:
        source_gdb: Local NHDPlus HR geodatabase (HU4 or HU8) or GeoPackage.
        watershed_path: Watershed polygon datasource (shapefile, GPKG, GDB).
        output_dir: Folder containing the working GeoPackage.
        gpkg_name: Output GeoPackage filename.
        watershed_layer: Layer name inside watershed_path, if required.
        watershed_query: Optional pandas query to select watershed features,
            for example "huc8 == '10020008'".
        catchment_layer: Catchment layer name in the source datasource.
        flowline_layer: Flowline layer name in the source datasource.
        vaa_table: Flowline value-added attribute table name.
        waterbody_layer: Waterbody layer name in the source datasource.
        point_method: "centroid" or "representative_point".
        include_waterbodies: Whether to clip waterbodies.
    """

    VALID_POINT_METHODS = ("centroid", "representative_point")

    def __init__(
        self,
        source_gdb: str | Path,
        watershed_path: str | Path,
        output_dir: str | Path = "Domain_Files",
        gpkg_name: str = "nhdhr.gpkg",
        watershed_layer: str | None = None,
        watershed_query: str | None = None,
        catchment_layer: str = "NHDPlusCatchment",
        flowline_layer: str = "NHDFlowline",
        vaa_table: str = "NHDPlusFlowlineVAA",
        waterbody_layer: str = "NHDWaterbody",
        point_method: str = "centroid",
        include_waterbodies: bool = True,
    ) -> None:
        """Initializes watershed clip configuration."""
        self.source_gdb = Path(source_gdb)
        self.watershed_path = Path(watershed_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpkg_name = str(gpkg_name)
        self.watershed_layer = watershed_layer
        self.watershed_query = watershed_query
        self.catchment_layer = catchment_layer
        self.flowline_layer = flowline_layer
        self.vaa_table = vaa_table
        self.waterbody_layer = waterbody_layer
        self.include_waterbodies = bool(include_waterbodies)

        method = str(point_method).strip().lower()
        if method not in self.VALID_POINT_METHODS:
            raise ValueError(
                f"point_method must be one of {self.VALID_POINT_METHODS}."
            )
        self.point_method = method

    @property
    def gpkg_path(self) -> Path:
        """Returns the output GeoPackage path."""
        return self.output_dir / self.gpkg_name

    @property
    def summary_path(self) -> Path:
        """Returns the clip summary JSON path."""
        return self.output_dir / "watershed_clip_summary.json"


def write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
    overwrite: bool = True,
) -> None:
    """Writes a GeoDataFrame as a GeoPackage layer using pyogrio.

    The pyogrio engine stores 64-bit integers correctly. The fiona engine
    can overflow on Windows when a wide integer ID column exceeds 32 bits.

    Args:
        gdf: Non-empty GeoDataFrame to write.
        gpkg_path: Output GeoPackage path.
        layer_name: Output layer name.
        overwrite: Remove existing same-name layer before writing.

    Raises:
        ValueError: If gdf is empty.
    """
    if gdf.empty:
        raise ValueError(f"Refusing to write empty layer: {layer_name}")

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    existing = list_gpkg_layers(gpkg_path) if gpkg_path.exists() else []
    if overwrite and layer_name in existing:
        fiona.remove(gpkg_path, layer=layer_name, driver="GPKG")

    gdf.to_file(
        gpkg_path,
        layer=layer_name,
        driver="GPKG",
        index=False,
        engine="pyogrio",
    )


# Make the NHD_PlusHR loader use the same pyogrio writer.
nhd_source.write_layer = write_layer


def resolve_layer_name(path: Path, requested: str) -> str:
    """Finds a layer in a datasource, ignoring case.

    Args:
        path: Datasource path.
        requested: Requested layer or table name.

    Returns:
        Actual layer name in the datasource.

    Raises:
        ValueError: If the layer is not found.
    """
    layers = list_gpkg_layers(path)
    lookup = {name.lower(): name for name in layers}

    if requested.lower() not in lookup:
        raise ValueError(
            f"Layer '{requested}' not found in {path.name}. "
            f"Available layers: {layers}"
        )

    return lookup[requested.lower()]


def read_table(path: Path, layer_name: str) -> pd.DataFrame:
    """Reads a non-spatial table with lowercase column names.

    Args:
        path: Datasource path.
        layer_name: Table name.

    Returns:
        DataFrame with lowercase column names.
    """
    with fiona.open(path, layer=layer_name) as source:
        records = [
            dict(feature["properties"])
            for feature in tqdm(
                source,
                total=len(source),
                desc=f"Reading {layer_name}",
                unit="row",
            )
        ]

    table = pd.DataFrame(records)
    table.columns = [column.lower() for column in table.columns]
    return table


def force_2d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drops Z/M coordinates from all geometries.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with 2D geometries.
    """
    output = gdf.copy()
    output["geometry"] = gpd.GeoSeries(
        shapely.force_2d(output.geometry.to_numpy()),
        index=output.index,
        crs=output.crs,
    )
    return output


class WatershedClipper:
    """Clips local NHDPlus HR layers to a watershed using reference points.

    A feature is kept when its centroid (or representative point) lies inside
    the watershed polygon. Points are computed in a projected CRS to avoid
    distortion from geographic coordinates.
    """

    def __init__(self, config: WatershedClipConfig) -> None:
        """Initializes the clipper.

        Args:
            config: Watershed clip configuration.
        """
        self.config = config
        self._watershed: gpd.GeoDataFrame | None = None
        self._projected_crs: pyproj.CRS | None = None
        self._projected_boundary: Any = None

    def run(self) -> dict[str, Any]:
        """Runs the clip and writes merged_* layers.

        Returns:
            Summary dictionary of feature counts and edge-case diagnostics.
        """
        total_steps = 8 if self.config.include_waterbodies else 6
        summary: dict[str, Any] = {
            "source_gdb": str(self.config.source_gdb),
            "watershed_path": str(self.config.watershed_path),
            "point_method": self.config.point_method,
        }

        with tqdm(total=total_steps, desc="Watershed clip", unit="step") as bar:
            bar.set_postfix_str("watershed")
            watershed = self.load_watershed()
            bar.update(1)

            bar.set_postfix_str("catchments")
            catchments, stats = self.clip_catchments()
            summary.update(stats)
            bar.update(1)

            bar.set_postfix_str("flowlines")
            flowlines, stats = self.clip_flowlines(catchments)
            summary.update(stats)
            bar.update(1)

            waterbodies = None
            if self.config.include_waterbodies:
                bar.set_postfix_str("waterbodies")
                waterbodies, stats = self.clip_waterbodies()
                summary.update(stats)
                bar.update(1)

            bar.set_postfix_str("writing merged_catchments")
            write_layer(catchments, self.config.gpkg_path, "merged_catchments")
            bar.update(1)

            bar.set_postfix_str("writing merged_flowlines")
            write_layer(flowlines, self.config.gpkg_path, "merged_flowlines")
            bar.update(1)

            if waterbodies is not None:
                bar.set_postfix_str("writing merged_waterbodies")
                if not waterbodies.empty:
                    write_layer(
                        waterbodies,
                        self.config.gpkg_path,
                        "merged_waterbodies",
                    )
                bar.update(1)

            bar.set_postfix_str("writing watershed_boundary")
            write_layer(watershed, self.config.gpkg_path, "watershed_boundary")
            bar.update(1)

        self.config.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        return summary

    def load_watershed(self) -> gpd.GeoDataFrame:
        """Loads and dissolves the watershed polygon.

        Returns:
            Single-row watershed GeoDataFrame.

        Raises:
            ValueError: If no polygon features remain after filtering.
        """
        if self._watershed is not None:
            return self._watershed

        read_kwargs: dict[str, Any] = {}
        if self.config.watershed_layer is not None:
            read_kwargs["layer"] = self.config.watershed_layer

        watershed = lower_columns(
            gpd.read_file(self.config.watershed_path, **read_kwargs)
        )

        if self.config.watershed_query:
            watershed = watershed.query(self.config.watershed_query)

        watershed = watershed[
            watershed.geometry.notna() & ~watershed.geometry.is_empty
        ]

        if watershed.empty:
            raise ValueError("Watershed layer has no usable polygons.")

        if watershed.crs is None:
            raise ValueError("Watershed layer has no CRS defined.")

        geometry = watershed.geometry.make_valid().union_all()
        dissolved = gpd.GeoDataFrame(
            {"name": ["watershed"]},
            geometry=[geometry],
            crs=watershed.crs,
        )

        self._projected_crs = dissolved.estimate_utm_crs()
        self._projected_boundary = (
            dissolved.to_crs(self._projected_crs).geometry.iloc[0]
        )
        shapely.prepare(self._projected_boundary)
        self._watershed = dissolved

        return dissolved

    def read_source_layer(self, requested: str) -> gpd.GeoDataFrame:
        """Reads a source layer limited to the watershed bounding box.

        Args:
            requested: Layer name in the source datasource.

        Returns:
            GeoDataFrame with lowercase columns.
        """
        watershed = self.load_watershed()
        layer_name = resolve_layer_name(self.config.source_gdb, requested)

        with fiona.open(self.config.source_gdb, layer=layer_name) as source:
            layer_crs = pyproj.CRS.from_wkt(source.crs_wkt)

        bounds = tuple(watershed.to_crs(layer_crs).total_bounds)

        tqdm.write(f"Reading {layer_name} within watershed extent...")
        layer = gpd.read_file(
            self.config.source_gdb,
            layer=layer_name,
            bbox=bounds,
        )

        return lower_columns(layer)

    def classify(
        self,
        gdf: gpd.GeoDataFrame,
        label: str,
        compute_touches: bool = True,
    ) -> tuple[pd.Series, pd.Series]:
        """Tests reference points and intersection against the watershed.

        Args:
            gdf: Input features.
            label: Progress-bar label.
            compute_touches: Also compute geometry intersection flags.

        Returns:
            Tuple of boolean Series (inside, touches) aligned to gdf.index.
        """
        boundary = self._projected_boundary
        inside = np.zeros(len(gdf), dtype=bool)
        touches = np.zeros(len(gdf), dtype=bool)

        for start in tqdm(
            range(0, len(gdf), BATCH_SIZE),
            desc=f"Testing {label}",
            unit="batch",
        ):
            stop = start + BATCH_SIZE
            chunk = gdf.geometry.iloc[start:stop].to_crs(self._projected_crs)

            if self.config.point_method == "representative_point":
                points = chunk.representative_point()
            else:
                points = chunk.centroid

            inside[start:stop] = points.within(boundary).to_numpy()

            if compute_touches:
                touches[start:stop] = chunk.intersects(boundary).to_numpy()

        return (
            pd.Series(inside, index=gdf.index),
            pd.Series(touches, index=gdf.index),
        )

    def clip_catchments(
        self,
    ) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
        """Selects catchments with reference points inside the watershed.

        Returns:
            Tuple of kept catchments and diagnostic counts.

        Raises:
            KeyError: If nhdplusid is absent.
            ValueError: If no catchments are kept.
        """
        catchments = self.read_source_layer(self.config.catchment_layer)

        if "nhdplusid" not in catchments.columns:
            raise KeyError("Catchment layer has no nhdplusid field.")

        catchments["nhdplusid"] = catchments["nhdplusid"].astype("int64")
        catchments = catchments.drop_duplicates(subset="nhdplusid")

        inside, touches = self.classify(catchments, "catchments")
        kept = catchments[inside].copy()

        if kept.empty:
            raise ValueError(
                "No catchments have reference points inside the watershed. "
                "Check that the watershed overlaps the source geodatabase."
            )

        stats = {
            "catchments_read": int(len(catchments)),
            "catchments_kept": int(len(kept)),
            "catchments_touching_but_excluded": int((touches & ~inside).sum()),
        }

        return kept, stats

    def clip_flowlines(
            self,
            catchments: gpd.GeoDataFrame,
    ) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
        """Selects flowlines whose centroids lie inside the watershed.

        Flowline selection is independent of catchment selection, so
        connectors and artificial paths without catchments are retained.

        Args:
            catchments: Kept catchments (used only for diagnostics).

        Returns:
            Tuple of kept flowlines and diagnostic counts.

        Raises:
            KeyError: If required fields are absent.
            ValueError: If no flowlines are kept.
        """
        flowlines = self.read_source_layer(self.config.flowline_layer)

        if "nhdplusid" not in flowlines.columns:
            raise KeyError("Flowline layer has no nhdplusid field.")

        flowlines["nhdplusid"] = flowlines["nhdplusid"].astype("int64")
        flowlines = flowlines.drop_duplicates(subset="nhdplusid")

        inside, touches = self.classify(flowlines, "flowlines")
        kept = flowlines[touches].copy()

        if kept.empty:
            raise ValueError("No flowline centroids fall inside the watershed.")

        kept = force_2d(kept)
        kept_ids = set(kept["nhdplusid"])

        vaa_name = resolve_layer_name(
            self.config.source_gdb,
            self.config.vaa_table,
        )
        vaa = read_table(self.config.source_gdb, vaa_name)
        vaa["nhdplusid"] = vaa["nhdplusid"].astype("int64")
        vaa = vaa[vaa["nhdplusid"].isin(kept_ids)]
        vaa = vaa.drop_duplicates(subset="nhdplusid")

        overlap = (set(vaa.columns) & set(kept.columns)) - {"nhdplusid"}
        vaa = vaa.drop(columns=sorted(overlap))

        kept = kept.merge(vaa, on="nhdplusid", how="left")
        kept = gpd.GeoDataFrame(kept, geometry="geometry", crs=flowlines.crs)

        required = {"hydroseq", "dnhydroseq", "totdasqkm"}
        missing = required - set(kept.columns)
        if missing:
            raise KeyError(
                f"Flowlines missing VAA fields after join: {sorted(missing)}"
            )

        catchment_ids = set(catchments["nhdplusid"])
        stats = {
            "flowlines_kept": int(len(kept)),
            "catchments_without_flowline": int(len(catchment_ids - kept_ids)),
            "flowlines_without_catchment": int(len(kept_ids - catchment_ids)),
            "flowlines_without_vaa": int(kept["hydroseq"].isna().sum()),
        }

        return kept, stats

    def clip_waterbodies(
        self,
    ) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
        """Selects waterbodies with reference points inside the watershed.

        Returns:
            Tuple of kept waterbodies and diagnostic counts.
        """
        waterbodies = self.read_source_layer(self.config.waterbody_layer)

        if waterbodies.empty:
            return waterbodies, {"waterbodies_kept": 0}

        inside, _ = self.classify(
            waterbodies,
            "waterbodies",
            compute_touches=False,
        )
        kept = waterbodies[inside].copy()

        if kept.empty:
            return kept, {"waterbodies_kept": 0}

        kept = force_2d(kept)

        if "nhdplusid" in kept.columns:
            kept["wb_id"] = kept["nhdplusid"].astype("int64").astype(str)
        elif "permanent_identifier" in kept.columns:
            kept["wb_id"] = kept["permanent_identifier"].astype(str)
        else:
            kept["wb_id"] = kept.index.astype(str)

        return kept, {"waterbodies_kept": int(len(kept))}


def main() -> None:
    """Clips local NHDPlus HR data to a watershed and prepares layers."""
    print(f"Running {Path(__file__).resolve()} (pyogrio writer active)")

    clip_config = WatershedClipConfig(
        source_gdb=Path(
            r"C:\Users\CND367\Documents\PRMS\shapefiles\NHDPlus_HR"
            r"\NHDPLUS_H_1002_HU4_GDB.gdb"
        ),
        watershed_path=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\HUC8\nhdhr.gpkg"
        ),
        output_dir=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\HUC8"
        ),
        gpkg_name="nhdhr.gpkg",
        watershed_layer="gallatin_logan",
        # watershed_query="huc8 == '10020008'",
        point_method="centroid",  # or "representative_point"
        include_waterbodies=True,
    )

    summary = WatershedClipper(clip_config).run()

    print("\nClip summary:")
    print(json.dumps(summary, indent=2))

    # Reuse the existing loader to write prepared_* layers.
    prep_config = NHDPlusHRConfig(
        huc12="100200081401",
        output_dir=clip_config.output_dir,
        gpkg_name=clip_config.gpkg_name,
    )

    prepared = LocalNHDPlusHRLoader(prep_config).load_and_prepare()

    print("\nPrepared source layers:")
    for layer_name, gdf in prepared.items():
        print(f"  {layer_name}: {len(gdf):,} features")


if __name__ == "__main__":
    main()
