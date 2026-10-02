r"""Download and prepare NHDPlus HR data for source-agnostic PRMS routing.

This module performs source-specific work only:

    1. Identify upstream HUC12 polygons with NLDI.
    2. Download or locally load NHDPlus HR flowlines, catchments, and
       waterbodies.
    3. Merge and validate source data.
    4. Convert NHDPlus HR fields to a common prepared-domain schema.
    5. Write prepared source layers for Routing.py.

This module does not select a routing outlet, trace the model network, assign
model segment IDs, assign HRU IDs, or generate PRMS model routing tables.

Prepared outputs written to the GeoPackage:

    prepared_flowlines
    prepared_catchments
    prepared_waterbodies
    prepared_huc12

The prepared-domain schema is designed so that another source-specific script,
such as NHD_PlusV2.py, 3DHP.py, or DISV_Valley.py, can prepare equivalent
layers for Routing.py without changing the routing code.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
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
import numpy as np
import pandas as pd
from fiona import listlayers, remove
from pynhd import NHDPlusHR, NLDI, WaterData
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


class NHDPlusHRConfig:
    """Stores NHDPlus HR download and preparation settings.

    Args:
        huc12: Target HUC12 used for NLDI upstream navigation.
        output_dir: Folder containing the working GeoPackage.
        gpkg_name: GeoPackage filename.
        extra_huc12: Additional HUC12s included beyond NLDI results.
        download_layers: Logical layers to download.
        load_layers: Logical layers to load/prep locally.
        source_path: Optional alternate source GeoPackage/GDB.
        source_layer_map: Mapping of logical source names to GIS layer names.
        max_retries: Retry count after a failed web request.
        retry_seconds: Base wait time between retries.
    """

    VALID_LAYERS = ("flowline", "catchment", "waterbody")

    DEFAULT_SOURCE_LAYER_MAP = {
        "flowline": "merged_flowlines",
        "catchment": "merged_catchments",
        "waterbody": "merged_waterbodies",
    }

    def __init__(
        self,
        huc12: str,
        output_dir: str | Path = "Domain_Files",
        gpkg_name: str = "nhdhr.gpkg",
        extra_huc12: list[str] | tuple[str, ...] | None = None,
        download_layers: list[str] | tuple[str, ...] | None = None,
        load_layers: list[str] | tuple[str, ...] | None = None,
        source_path: str | Path | None = None,
        source_layer_map: dict[str, str] | None = None,
        max_retries: int = 1,
        retry_seconds: float = 10.0,
    ) -> None:
        """Initializes NHDPlus HR preparation configuration."""
        self.huc12 = str(huc12)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpkg_name = str(gpkg_name)
        self.extra_huc12 = [str(value) for value in (extra_huc12 or [])]
        self.download_layers = self.validate_layers(
            download_layers or self.VALID_LAYERS,
            "download_layers",
        )
        self.load_layers = self.validate_layers(
            load_layers or self.VALID_LAYERS,
            "load_layers",
        )
        self.source_path = (
            Path(source_path)
            if source_path is not None
            else self.gpkg_path
        )

        self.source_layer_map = self.DEFAULT_SOURCE_LAYER_MAP.copy()
        if source_layer_map is not None:
            self.source_layer_map.update(
                {
                    str(key).lower(): str(value)
                    for key, value in source_layer_map.items()
                }
            )

        self.max_retries = int(max_retries)
        self.retry_seconds = float(retry_seconds)

    @property
    def gpkg_path(self) -> Path:
        """Returns the working GeoPackage path."""
        return self.output_dir / self.gpkg_name

    @property
    def failure_log_path(self) -> Path:
        """Returns the failed-download CSV path."""
        return self.output_dir / "download_failures.csv"

    @classmethod
    def validate_layers(
        cls,
        layers: list[str] | tuple[str, ...],
        parameter_name: str,
    ) -> tuple[str, ...]:
        """Validates requested logical source layers.

        Args:
            layers: Candidate logical layer names.
            parameter_name: Error-message parameter name.

        Returns:
            Ordered tuple of unique valid layer names.

        Raises:
            ValueError: If the requested selection is empty or invalid.
        """
        requested = tuple(
            dict.fromkeys(str(layer).strip().lower() for layer in layers)
        )

        if not requested:
            raise ValueError(f"{parameter_name} cannot be empty.")

        invalid = set(requested) - set(cls.VALID_LAYERS)
        if invalid:
            raise ValueError(
                f"Invalid {parameter_name}: {sorted(invalid)}. "
                f"Valid choices: {cls.VALID_LAYERS}."
            )

        return requested


def lower_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Returns a copy with lowercase non-geometry fields.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with normalized lowercase attribute names.
    """
    output = gdf.copy()
    geometry_name = output.geometry.name
    output.columns = [
        column if column == geometry_name else column.lower()
        for column in output.columns
    ]
    return output


def safe_layer_name(name: str) -> str:
    """Converts text into a safe GeoPackage layer name.

    Args:
        name: Candidate layer name.

    Returns:
        Lowercase alphanumeric/underscore layer name.
    """
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def list_gpkg_layers(path: Path) -> list[str]:
    """Lists layers in a GeoPackage or supported spatial datasource.

    Args:
        path: Datasource path.

    Returns:
        Sorted layer names.

    Raises:
        FileNotFoundError: If the datasource does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Datasource does not exist: {path}")

    return sorted(listlayers(path))


def write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
    overwrite: bool = True,
) -> None:
    """Writes a GeoDataFrame as a GeoPackage layer.

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
        remove(gpkg_path, layer=layer_name, driver="GPKG")

    gdf.to_file(
        gpkg_path,
        layer=layer_name,
        driver="GPKG",
        index=False,
        engine="fiona",
    )


def read_layer(
    source_path: str | Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Reads a GIS layer with lowercase attribute names.

    Args:
        source_path: GeoPackage, GDB, or OGR-readable datasource.
        layer_name: Layer to load.

    Returns:
        GeoDataFrame with lowercase attribute fields.
    """
    source_path = Path(source_path)
    layers = list_gpkg_layers(source_path)

    if layer_name not in layers:
        raise ValueError(
            f"Layer '{layer_name}' is absent from {source_path.name}. "
            f"Available layers: {layers}"
        )

    return lower_columns(gpd.read_file(source_path, layer=layer_name))


def empty_geodataframe(crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
    """Creates an empty GeoDataFrame.

    Args:
        crs: CRS assigned to the output.

    Returns:
        Empty GeoDataFrame with a geometry column.
    """
    return gpd.GeoDataFrame(geometry=[], crs=crs)


class NHDPlusHRDownloader:
    """Downloads selected NHDPlus HR layers HUC12-by-HUC12.

    Successful requests are immediately written to raw GeoPackage layers.
    Raw chunks can be merged to standard merged source layers and then purged.

    The class does not calculate routing. Its only responsibilities are source
    acquisition, raw-chunk persistence, merging, and source validation.
    """

    RAW_PREFIX = {
        "flowline": "raw_flowline_",
        "catchment": "raw_catchment_",
        "waterbody": "raw_waterbody_",
    }

    MERGED_LAYER_MAP = {
        "flowline": "merged_flowlines",
        "catchment": "merged_catchments",
        "waterbody": "merged_waterbodies",
    }

    REQUIRED_FLOWLINE_COLUMNS = {
        "nhdplusid",
        "hydroseq",
        "dnhydroseq",
        "totdasqkm",
    }

    def __init__(self, config: NHDPlusHRConfig) -> None:
        """Initializes the NHDPlus HR downloader.

        Args:
            config: NHDPlus HR preparation configuration.
        """
        self.config = config
        self.gpkg_path = config.gpkg_path

    def download(
        self,
        layers: list[str] | tuple[str, ...] | None = None,
        overwrite_raw: bool = False,
        purge_raw_after_merge: bool = False,
    ) -> dict[str, int]:
        """Downloads selected source layers and writes merged GIS layers.

        Args:
            layers: Optional layer override. Defaults to config.download_layers.
            overwrite_raw: Replace already saved raw HUC12 chunks.
            purge_raw_after_merge: Delete raw chunks only after a requested
                source type completes without failures.

        Returns:
            Download and feature count summary.
        """
        requested = (
            NHDPlusHRConfig.validate_layers(layers, "layers")
            if layers is not None
            else self.config.download_layers
        )

        huc12s = self.get_upstream_huc12s()
        hucs = self.download_huc12_polygons(huc12s)
        huc_geometry_lookup = self._huc12_geometry_lookup(hucs, huc12s)

        result: dict[str, int] = {
            "huc12_count": len(huc12s),
            "failed_requests": 0,
        }

        for source_layer in requested:
            failures = self.download_layer_by_huc12(
                source_layer=source_layer,
                huc_geometries=huc_geometry_lookup,
                overwrite_raw=overwrite_raw,
            )

            result["failed_requests"] += failures

            merged = self.merge_saved_chunks(source_layer)
            result[f"{source_layer}_count"] = len(merged)

            if purge_raw_after_merge and failures == 0:
                deleted = self.purge_raw_layers((source_layer,))
                result[f"{source_layer}_raw_layers_deleted"] = deleted[
                    source_layer
                ]

        return result

    def get_upstream_huc12s(self) -> list[str]:
        """Gets target and upstream HUC12 IDs with NLDI.

        Returns:
            Sorted unique HUC12 code list.
        """
        navigation = NLDI().navigate_byid(
            fsource="huc12pp",
            fid=self.config.huc12,
            navigation="upstreamTributaries",
            source="huc12pp",
            distance=1000,
        )

        id_field = (
            "identifier"
            if "identifier" in navigation.columns
            else navigation.columns[0]
        )

        huc12s = set(
            navigation[id_field]
            .astype(str)
            .str.extract(r"(\d{12})")[0]
            .dropna()
        )

        huc12s.add(self.config.huc12)
        huc12s.update(self.config.extra_huc12)

        return sorted(huc12s)

    def download_huc12_polygons(
        self,
        huc12s: list[str],
    ) -> gpd.GeoDataFrame:
        """Downloads source HUC12 boundaries.

        Args:
            huc12s: HUC12 IDs to retrieve.

        Returns:
            HUC12 polygon GeoDataFrame.
        """
        layer_name = "download_huc12"

        if self._layer_exists(layer_name):
            return read_layer(self.gpkg_path, layer_name)

        try:
            hucs = NHDPlusHR("huc12").byids("huc12", huc12s)
        except Exception:
            hucs = WaterData("huc12").byid("huc12", huc12s)

        hucs = lower_columns(hucs)
        write_layer(hucs, self.gpkg_path, layer_name, overwrite=True)

        return hucs

    def download_layer_by_huc12(
        self,
        source_layer: str,
        huc_geometries: dict[str, BaseGeometry],
        overwrite_raw: bool = False,
    ) -> int:
        """Downloads one source layer per HUC12 and writes raw chunks.

        Args:
            source_layer: flowline, catchment, or waterbody.
            huc_geometries: Mapping from HUC12 ID to query geometry.
            overwrite_raw: Re-download existing raw source chunks.

        Returns:
            Number of failed HUC12 source requests.
        """
        service = NHDPlusHR(source_layer)
        failures = 0

        for huc12, geometry in tqdm(
            huc_geometries.items(),
            desc=f"Downloading {source_layer}",
            unit="HUC12",
        ):
            raw_layer = self._raw_layer_name(source_layer, huc12)

            if not overwrite_raw and self._layer_exists(raw_layer):
                continue

            try:
                chunk = self._request_by_geometry(
                    service=service,
                    geometry=geometry,
                    source_layer=source_layer,
                    huc12=huc12,
                )

            except RuntimeError as error:
                if source_layer == "catchment" and self._is_timeout_error(error):
                    try:
                        chunk = self._download_catchments_tiled(
                            huc12=huc12,
                            geometry=geometry,
                            ncols=3,
                            nrows=3,
                        )
                    except RuntimeError as tile_error:
                        failures += 1
                        self._log_failure(source_layer, huc12, tile_error)
                        tqdm.write(f"FAILED catchment {huc12}: {tile_error}")
                        continue
                else:
                    failures += 1
                    self._log_failure(source_layer, huc12, error)
                    tqdm.write(f"FAILED {source_layer} {huc12}: {error}")
                    continue

            if chunk is None or chunk.empty:
                continue

            chunk = lower_columns(chunk)
            chunk["source_huc12"] = str(huc12)

            write_layer(
                chunk,
                self.gpkg_path,
                raw_layer,
                overwrite=True,
            )

        return failures

    def merge_saved_chunks(
        self,
        source_layer: str,
    ) -> gpd.GeoDataFrame:
        """Merges and de-duplicates raw chunks for one source layer.

        Args:
            source_layer: flowline, catchment, or waterbody.

        Returns:
            Merged source GeoDataFrame.

        Raises:
            RuntimeError: If no raw chunks exist.
        """
        raw_layers = self._raw_layers(source_layer)

        if not raw_layers:
            raise RuntimeError(
                f"No raw chunks available for {source_layer}."
            )

        unique_id = (
            "nhdplusid"
            if source_layer in ("flowline", "catchment")
            else "wb_id"
        )

        merged: gpd.GeoDataFrame | None = None

        for raw_layer in tqdm(
            raw_layers,
            desc=f"Merging {source_layer}",
            unit="chunk",
        ):
            chunk = read_layer(self.gpkg_path, raw_layer)

            if merged is None:
                merged = chunk
            else:
                merged = gpd.GeoDataFrame(
                    pd.concat([merged, chunk], ignore_index=True),
                    crs=merged.crs,
                )

            if unique_id in merged.columns:
                merged = merged.drop_duplicates(
                    subset=unique_id,
                    keep="first",
                ).copy()

        if merged is None or merged.empty:
            raise RuntimeError(f"Merged {source_layer} result is empty.")

        if source_layer in ("flowline", "catchment"):
            merged["nhdplusid"] = merged["nhdplusid"].astype("int64")

        if source_layer == "waterbody":
            merged = self._ensure_waterbody_id(merged)

        if source_layer == "flowline":
            missing = self.REQUIRED_FLOWLINE_COLUMNS - set(merged.columns)
            if missing:
                raise KeyError(
                    f"Flowline data missing VAA fields: {sorted(missing)}"
                )

        write_layer(
            merged,
            self.gpkg_path,
            self.MERGED_LAYER_MAP[source_layer],
            overwrite=True,
        )

        return merged

    def purge_raw_layers(
        self,
        layers: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Purges raw HUC12 chunks for verified merged source layers.

        Args:
            layers: Source types to purge. Defaults to config.download_layers.

        Returns:
            Count of removed raw layers by source type.

        Raises:
            RuntimeError: If a merged layer does not exist.
        """
        requested = (
            NHDPlusHRConfig.validate_layers(layers, "layers")
            if layers is not None
            else self.config.download_layers
        )

        deleted: dict[str, int] = {}

        for source_layer in requested:
            merged_layer = self.MERGED_LAYER_MAP[source_layer]

            if not self._layer_exists(merged_layer):
                raise RuntimeError(
                    f"Cannot purge raw {source_layer}; "
                    f"{merged_layer} does not exist."
                )

            raw_layers = self._raw_layers(source_layer)

            for raw_layer in raw_layers:
                remove(self.gpkg_path, layer=raw_layer, driver="GPKG")

            deleted[source_layer] = len(raw_layers)

        return deleted

    def _request_by_geometry(
        self,
        service: NHDPlusHR,
        geometry: BaseGeometry,
        source_layer: str,
        huc12: str,
    ) -> gpd.GeoDataFrame | None:
        """Requests one NHDPlus HR geometry with retry handling.

        Args:
            service: Initialized NHDPlusHR service.
            geometry: Query polygon.
            source_layer: Service-layer label.
            huc12: Source HUC12 code.

        Returns:
            Query result or None if zero matching features occur.

        Raises:
            RuntimeError: If every request attempt fails.
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return service.bygeom(geometry)
            except Exception as error:  # noqa: BLE001
                last_error = error

                if self._is_zero_match_error(error):
                    return None

                if attempt < self.config.max_retries:
                    pause = self.config.retry_seconds * (attempt + 1)
                    tqdm.write(
                        f"Retrying {source_layer} for {huc12} "
                        f"in {pause:.0f} seconds."
                    )
                    time.sleep(pause)

        raise RuntimeError(
            f"{source_layer} request failed for HUC12 {huc12}. "
            f"Final cause: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def _download_catchments_tiled(
        self,
        huc12: str,
        geometry: BaseGeometry,
        ncols: int,
        nrows: int,
    ) -> gpd.GeoDataFrame:
        """Downloads catchments in smaller polygon tiles after a timeout.

        Args:
            huc12: HUC12 source identifier.
            geometry: HUC12 polygon geometry.
            ncols: Number of tile columns.
            nrows: Number of tile rows.

        Returns:
            De-duplicated catchment GeoDataFrame.
        """
        minx, miny, maxx, maxy = geometry.bounds
        dx = (maxx - minx) / ncols
        dy = (maxy - miny) / nrows

        tiles: list[BaseGeometry] = []

        for row in range(nrows):
            for col in range(ncols):
                candidate = box(
                    minx + col * dx,
                    miny + row * dy,
                    minx + (col + 1) * dx,
                    miny + (row + 1) * dy,
                )

                tile = candidate.intersection(geometry)
                if not tile.is_empty:
                    tiles.append(tile)

        service = NHDPlusHR("catchment")
        chunks: list[gpd.GeoDataFrame] = []

        for tile_number, tile in enumerate(
            tqdm(
                tiles,
                desc=f"Catchment tiles {huc12}",
                unit="tile",
                leave=False,
            ),
            start=1,
        ):
            chunk = self._request_by_geometry(
                service,
                tile,
                f"catchment tile {tile_number}",
                huc12,
            )

            if chunk is not None and not chunk.empty:
                chunks.append(lower_columns(chunk))

        if not chunks:
            return empty_geodataframe()

        catchments = gpd.GeoDataFrame(
            pd.concat(chunks, ignore_index=True),
            crs=chunks[0].crs,
        )

        if "nhdplusid" not in catchments.columns:
            raise KeyError(
                f"Tiled catchments for {huc12} have no nhdplusid."
            )

        return catchments.drop_duplicates(
            subset="nhdplusid",
            keep="first",
        ).copy()

    def _huc12_geometry_lookup(
        self,
        hucs: gpd.GeoDataFrame,
        requested_huc12s: list[str],
    ) -> dict[str, BaseGeometry]:
        """Builds HUC12 ID to polygon lookup.

        Args:
            hucs: HUC12 polygon source data.
            requested_huc12s: Expected HUC12 IDs.

        Returns:
            HUC12 geometry lookup.

        Raises:
            ValueError: If requested HUC12 polygons are missing.
        """
        id_field = next(
            (
                field
                for field in ("huc12", "huc_12", "identifier")
                if field in hucs.columns
            ),
            None,
        )

        if id_field is None:
            raise KeyError(
                f"No HUC12 ID field. Fields: {list(hucs.columns)}"
            )

        hucs = hucs.copy()
        hucs["huc12_code"] = (
            hucs[id_field].astype(str).str.extract(r"(\d{12})")[0]
        )

        missing = sorted(
            set(requested_huc12s) - set(hucs["huc12_code"].dropna())
        )

        if missing:
            raise ValueError(
                f"Missing HUC12 polygons: {missing[:10]}"
            )

        return dict(zip(hucs["huc12_code"], hucs.geometry))

    def _raw_layer_name(
        self,
        source_layer: str,
        huc12: str,
    ) -> str:
        """Returns raw GeoPackage layer name for a HUC12 source chunk."""
        return safe_layer_name(
            f"{self.RAW_PREFIX[source_layer]}{huc12}"
        )

    def _raw_layers(self, source_layer: str) -> list[str]:
        """Lists saved raw layers for a source type.

        Args:
            source_layer: flowline, catchment, or waterbody.

        Returns:
            Sorted raw layer names.
        """
        if not self.gpkg_path.exists():
            return []

        prefix = self.RAW_PREFIX[source_layer]

        return sorted(
            layer
            for layer in list_gpkg_layers(self.gpkg_path)
            if layer.startswith(prefix)
        )

    def _layer_exists(self, layer_name: str) -> bool:
        """Checks whether a GeoPackage layer exists."""
        return (
            self.gpkg_path.exists()
            and layer_name in list_gpkg_layers(self.gpkg_path)
        )

    def _log_failure(
        self,
        source_layer: str,
        huc12: str,
        error: Exception,
    ) -> None:
        """Writes failed source request metadata to CSV."""
        record = pd.DataFrame(
            [
                {
                    "timestamp_utc": pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat(),
                    "source_layer": source_layer,
                    "huc12": huc12,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            ]
        )

        record.to_csv(
            self.config.failure_log_path,
            mode="a",
            header=not self.config.failure_log_path.exists(),
            index=False,
        )

    @staticmethod
    def _ensure_waterbody_id(
        waterbodies: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Adds a stable waterbody identifier.

        Args:
            waterbodies: Input waterbody features.

        Returns:
            Waterbody GeoDataFrame with ``wb_id``.
        """
        waterbodies = waterbodies.copy()

        if "nhdplusid" in waterbodies.columns:
            waterbodies["wb_id"] = waterbodies["nhdplusid"].astype(str)
        elif "permanent_identifier" in waterbodies.columns:
            waterbodies["wb_id"] = waterbodies[
                "permanent_identifier"
            ].astype(str)
        else:
            waterbodies["wb_id"] = waterbodies.index.astype(str)

        return waterbodies

    @staticmethod
    def _is_zero_match_error(error: Exception) -> bool:
        """Returns True if a service error indicates zero matches."""
        text = f"{type(error).__name__}: {error}".lower()
        return (
            "zeromatched" in text
            or "zero matched" in text
            or "no features" in text
            or "no records" in text
        )

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        """Returns True if a service error is a timeout."""
        text = str(error).lower()
        return any(
            marker in text
            for marker in (
                "504",
                "gateway timeout",
                "cloudfront",
                "request could not be satisfied",
                "readtimeout",
                "timed out",
            )
        )


class LocalNHDPlusHRLoader:
    """Loads selected source layers and writes standardized prepared layers.

    This class supports either:

        - merged NHDPlus HR layers created by NHDPlusHRDownloader, or
        - layers manually created or corrected in Arc Pro.

    It translates NHDPlus HR fields to the generic prepared-domain contract
    used by Routing.py.
    """

    REQUIRED_FLOWLINE_COLUMNS = {
        "nhdplusid",
        "hydroseq",
        "dnhydroseq",
    }

    def __init__(self, config: NHDPlusHRConfig) -> None:
        """Initializes the local NHDPlus HR loader.

        Args:
            config: Source preparation configuration.
        """
        self.config = config

    def load_and_prepare(
        self,
        layers: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, gpd.GeoDataFrame]:
        """Loads selected NHDPlus HR datasets and writes prepared outputs.

        Args:
            layers: Optional source-layer override. Defaults to config.load_layers.

        Returns:
            Dictionary of standardized prepared source layers.

        Raises:
            KeyError: If required NHDPlus HR source fields are absent.
        """
        requested = (
            NHDPlusHRConfig.validate_layers(layers, "layers")
            if layers is not None
            else self.config.load_layers
        )

        loaded = self._load_source_layers(requested)
        prepared: dict[str, gpd.GeoDataFrame] = {}

        if "flowline" in loaded:
            prepared["flowline"] = self._prepare_flowlines(
                loaded["flowline"]
            )
            write_layer(
                prepared["flowline"],
                self.config.gpkg_path,
                "prepared_flowlines",
                overwrite=True,
            )

        if "catchment" in loaded:
            prepared["catchment"] = self._prepare_catchments(
                loaded["catchment"]
            )
            write_layer(
                prepared["catchment"],
                self.config.gpkg_path,
                "prepared_catchments",
                overwrite=True,
            )

        if "waterbody" in loaded:
            prepared["waterbody"] = self._prepare_waterbodies(
                loaded["waterbody"]
            )
            write_layer(
                prepared["waterbody"],
                self.config.gpkg_path,
                "prepared_waterbodies",
                overwrite=True,
            )

        self._prepare_huc12_if_available()
        self._write_preparation_metadata(prepared)

        return prepared

    def _load_source_layers(
        self,
        requested: tuple[str, ...],
    ) -> dict[str, gpd.GeoDataFrame]:
        """Loads selected NHDPlus HR source layers.

        Args:
            requested: Valid logical source layers.

        Returns:
            Dictionary keyed by logical source type.
        """
        available = set(list_gpkg_layers(self.config.source_path))
        loaded: dict[str, gpd.GeoDataFrame] = {}

        for source_layer in requested:
            physical_layer = self.config.source_layer_map[source_layer]

            if physical_layer not in available:
                raise ValueError(
                    f"Source layer '{physical_layer}' for '{source_layer}' "
                    f"is not available in {self.config.source_path.name}. "
                    f"Available layers: {sorted(available)}"
                )

            loaded[source_layer] = read_layer(
                self.config.source_path,
                physical_layer,
            )

        return loaded

    def _prepare_flowlines(
        self,
        flowlines: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Converts NHDPlus HR flowlines to prepared-domain schema.

        Args:
            flowlines: NHDPlus HR source flowlines.

        Returns:
            Prepared flowlines with source_id and to_source_id.

        Raises:
            KeyError: If required NHDPlus HR VAA fields are absent.
        """
        missing = self.REQUIRED_FLOWLINE_COLUMNS - set(flowlines.columns)
        if missing:
            raise KeyError(
                f"NHDPlus HR flowlines missing required fields: "
                f"{sorted(missing)}"
            )

        output = flowlines.copy()
        output["nhdplusid"] = output["nhdplusid"].astype("int64")

        hydroseq = output["hydroseq"].fillna(0).round().astype("int64")
        dnhydroseq = output["dnhydroseq"].fillna(0).round().astype("int64")

        hydroseq_to_id = dict(zip(hydroseq, output["nhdplusid"]))

        output["source_id"] = output["nhdplusid"].astype(str)
        output["to_source_id"] = dnhydroseq.map(hydroseq_to_id)

        output.loc[dnhydroseq == 0, "to_source_id"] = np.nan
        output["to_source_id"] = output["to_source_id"].astype("Int64").astype(str)
        output.loc[output["to_source_id"] == "<NA>", "to_source_id"] = None

        output["source_type"] = "nhdplus_hr"

        output["source_name"] = (
            output["gnis_name"]
            if "gnis_name" in output.columns
            else None
        )

        output["drainage_area_km2"] = (
            output["totdasqkm"]
            if "totdasqkm" in output.columns
            else np.nan
        )

        output["stream_order"] = (
            output["streamorde"]
            if "streamorde" in output.columns
            else np.nan
        )

        output["slope"] = (
            output["slope"]
            if "slope" in output.columns
            else np.nan
        )

        output["divergence"] = (
            output["divergence"]
            if "divergence" in output.columns
            else np.nan
        )

        keep = [
            "source_id",
            "to_source_id",
            "source_type",
            "source_name",
            "drainage_area_km2",
            "stream_order",
            "slope",
            "divergence",
            "geometry",
        ]

        return output[
            [column for column in keep if column in output.columns]
        ].copy()

    def _prepare_catchments(
        self,
        catchments: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Converts NHDPlus HR catchments to prepared-domain schema.

        Args:
            catchments: NHDPlus HR catchment polygons.

        Returns:
            Prepared catchment polygons.

        Raises:
            KeyError: If nhdplusid is absent.
        """
        if "nhdplusid" not in catchments.columns:
            raise KeyError(
                "NHDPlus HR catchments require an 'nhdplusid' field."
            )

        output = catchments.copy()
        output["nhdplusid"] = output["nhdplusid"].astype("int64")
        output["source_id"] = output["nhdplusid"].astype(str)
        output["catchment_id"] = output["source_id"]
        output["source_type"] = "nhdplus_hr"

        return output[
            [
                "catchment_id",
                "source_id",
                "source_type",
                "geometry",
            ]
        ].copy()

    @staticmethod
    def _prepare_waterbodies(
        waterbodies: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Converts NHDPlus HR waterbodies to prepared-domain schema.

        Args:
            waterbodies: NHDPlus HR waterbody polygons.

        Returns:
            Prepared waterbody polygons.
        """
        output = waterbodies.copy()

        if "nhdplusid" in output.columns:
            output["waterbody_id"] = output["nhdplusid"].astype(str)
        elif "permanent_identifier" in output.columns:
            output["waterbody_id"] = output[
                "permanent_identifier"
            ].astype(str)
        else:
            output["waterbody_id"] = output.index.astype(str)

        output["source_type"] = "nhdplus_hr"

        return output[
            [
                "waterbody_id",
                "source_type",
                "geometry",
            ]
        ].copy()

    def _prepare_huc12_if_available(self) -> None:
        """Writes prepared HUC12 source polygons if available."""
        if not self.config.gpkg_path.exists():
            return

        layers = list_gpkg_layers(self.config.gpkg_path)
        if "download_huc12" not in layers:
            return

        hucs = read_layer(self.config.gpkg_path, "download_huc12")

        id_field = next(
            (
                field
                for field in ("huc12", "huc_12", "identifier")
                if field in hucs.columns
            ),
            None,
        )

        if id_field is None:
            return

        output = hucs.copy()
        output["huc12"] = (
            output[id_field].astype(str).str.extract(r"(\d{12})")[0]
        )

        write_layer(
            output[["huc12", "geometry"]],
            self.config.gpkg_path,
            "prepared_huc12",
            overwrite=True,
        )

    def _write_preparation_metadata(
        self,
        prepared: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Writes basic source-preparation metadata to JSON.

        Args:
            prepared: Prepared source layers.
        """
        metadata = {
            "source_type": "nhdplus_hr",
            "target_huc12": self.config.huc12,
            "prepared_layers": {
                layer_name: int(len(gdf))
                for layer_name, gdf in prepared.items()
            },
        }

        metadata_path = self.config.output_dir / "prepared_domain_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    """Downloads/loads NHDPlus HR data and writes prepared routing layers."""
    config = NHDPlusHRConfig(
        huc12="100200080507",
        output_dir=Path(
            r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
            r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
        ),
        gpkg_name="nhdhr.gpkg",

        # Change these as needed:
        download_layers=("waterbody",),
        load_layers=("flowline", "catchment", "waterbody"),

        # If Arc Pro catchments are in another datasource, set source_path
        # and map the physical layer name here:
        #
        # source_path=Path(r"C:\...\gallatin_arcpro_review.gpkg"),
        # source_layer_map={
        #     "flowline": "merged_flowlines",
        #     "catchment": "catchments_arcpro",
        #     "waterbody": "merged_waterbodies",
        # },
    )

    downloader = NHDPlusHRDownloader(config)

    # Download only layers specified by config.download_layers.
    # Set purge_raw_after_merge=True only after verifying merged products.
    download_result = downloader.download(
        overwrite_raw=False,
        purge_raw_after_merge=False,
    )

    print("Download result:")
    print(json.dumps(download_result, indent=2))

    loader = LocalNHDPlusHRLoader(config)

    # Writes prepared_flowlines, prepared_catchments, and prepared_waterbodies.
    prepared = loader.load_and_prepare()

    print("\nPrepared source layers:")
    for layer_name, gdf in prepared.items():
        print(f"  {layer_name}: {len(gdf):,} features")


if __name__ == "__main__":
    main()