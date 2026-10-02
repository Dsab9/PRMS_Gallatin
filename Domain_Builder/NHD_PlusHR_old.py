"""Download, import, and route NHDPlus HR data for a PRMS domain.

This module separates the NHDPlus HR workflow into three main classes:

    DomainConfig:
        Stores project configuration, including target HUC12, requested
        downloads, requested local layers, data paths, and routing settings.

    NHDPlusHRDownloader:
        Uses NLDI to identify upstream HUC12s. Downloads selected NHDPlus HR
        layers one HUC12 at a time and writes each successful response
        immediately to a GeoPackage.

    LocalDomainImporter:
        Loads any selected subset of merged downloader data or Arc Pro–prepared
        data from a GeoPackage, file geodatabase, or other OGR-supported source.

    DendriticDomainRouter:
        Creates a dendritic stream-routing network from the NHDPlus HR VAA
        primary downstream relation:

            DnHydroseq -> Hydroseq

        It then creates model segments and parent HRUs.

Typical workflow:

    # Download all source data:
    config = DomainConfig(
        huc12="100200080507",
        download_layers=("flowline", "catchment", "waterbody"),
        load_layers=("flowline", "catchment", "waterbody"),
    )
    downloader = NHDPlusHRDownloader(config)
    downloader.download()

    # Recover from a catchment-only failure:
    config = DomainConfig(
        huc12="100200080507",
        download_layers=("catchment",),
        load_layers=("flowline", "catchment", "waterbody"),
    )
    downloader = NHDPlusHRDownloader(config)
    downloader.download()

    # Route from downloaded or Arc Pro–prepared data:
    importer = LocalDomainImporter(config)
    data = importer.load()
    router = DendriticDomainRouter(
        config,
        flowlines=data["flowline"],
        catchments=data["catchment"],
        waterbodies=data.get("waterbody"),
    )
    router.build(target_huc12=target_huc12)

The downloader saves raw source data as one layer per HUC12 and source type,
then writes de-duplicated merged layers. Reruns skip already completed raw
chunks unless overwrite_raw=True.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Configure PROJ before importing pyproj, geopandas, or pynhd.
_PROJ_DIR = Path(sys.prefix) / "Library" / "share" / "proj"

if not (_PROJ_DIR / "proj.db").exists():
    raise FileNotFoundError(
        "Could not find the conda PROJ database. Expected:\n"
        f"    {_PROJ_DIR / 'proj.db'}\n"
        "Confirm that pyproj and proj are installed in the active conda "
        "environment."
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
from pynhd import NHDPlusHR, NLDI, WaterData
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================


class DomainConfig:
    """Stores settings for NHDPlus HR acquisition, import, and routing.

    Args:
        huc12: Target outlet HUC12 used to define the upstream model domain.
        gpkg_name: GeoPackage filename written to ``output_dir``.
        output_dir: Directory for GeoPackage, CSV, failure log, and QA outputs.
        outlet_nhdplusid: Optional explicit NHDPlus HR outlet flowline ID. If
            None, the router chooses the intersecting flowline with the largest
            VAA ``totdasqkm`` inside the target HUC12.
        work_crs: Projected CRS used for length and area calculations.
        extra_huc12: Extra HUC12 IDs to include beyond NLDI upstream results.
        min_hru_area_m2: HRUs below this threshold are flagged, not removed.
        max_retries: Retry attempts after the first failed service request.
        retry_seconds: Initial wait between failed service request retries.
        download_layers: Logical source layers to download. Valid values are
            ``flowline``, ``catchment``, and ``waterbody``.
        load_layers: Logical layers to load from local source data.
        source_path: Optional local GeoPackage, file geodatabase, or other
            spatial datasource to load. Defaults to the output GeoPackage.
        source_layer_map: Optional mapping from logical layer names to physical
            layer names in ``source_path``. Defaults to merged downloader layers.
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
        gpkg_name: str = "nhdhr.gpkg",
        output_dir: str | Path = "Domain_Files",
        outlet_nhdplusid: int | None = None,
        work_crs: str = "EPSG:5070",
        extra_huc12: list[str] | tuple[str, ...] | None = None,
        min_hru_area_m2: float = 0.0,
        max_retries: int = 2,
        retry_seconds: float = 5.0,
        download_layers: list[str] | tuple[str, ...] | None = None,
        load_layers: list[str] | tuple[str, ...] | None = None,
        source_path: str | Path | None = None,
        source_layer_map: dict[str, str] | None = None,
    ) -> None:
        """Initializes project settings."""
        self.huc12 = str(huc12)
        self.gpkg_name = str(gpkg_name)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.outlet_nhdplusid = (
            int(outlet_nhdplusid)
            if outlet_nhdplusid is not None
            else None
        )
        self.work_crs = str(work_crs)
        self.extra_huc12 = [str(value) for value in (extra_huc12 or [])]
        self.min_hru_area_m2 = float(min_hru_area_m2)
        self.max_retries = int(max_retries)
        self.retry_seconds = float(retry_seconds)

        self.download_layers = self.validate_layers(
            layers=download_layers or self.VALID_LAYERS,
            parameter_name="download_layers",
        )
        self.load_layers = self.validate_layers(
            layers=load_layers or self.VALID_LAYERS,
            parameter_name="load_layers",
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
                    str(logical_name).lower(): str(physical_name)
                    for logical_name, physical_name in source_layer_map.items()
                }
            )

    @property
    def gpkg_path(self) -> Path:
        """Returns the output downloader GeoPackage path."""
        return self.output_dir / self.gpkg_name

    @property
    def qa_dir(self) -> Path:
        """Returns the routing QA output directory."""
        return self.output_dir / "domain_qa"

    @property
    def failure_log_path(self) -> Path:
        """Returns the download failure-log CSV path."""
        return self.output_dir / "download_failures.csv"

    @classmethod
    def validate_layers(
        cls,
        layers: list[str] | tuple[str, ...],
        parameter_name: str = "layers",
    ) -> tuple[str, ...]:
        """Validates and de-duplicates logical NHDPlus HR layer names.

        Args:
            layers: Requested logical layer names.
            parameter_name: Name used in an error message.

        Returns:
            Ordered tuple of unique valid layer names.

        Raises:
            ValueError: If the selection is empty or includes invalid names.
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
                f"Valid choices are: {cls.VALID_LAYERS}."
            )

        return requested


# =============================================================================
# General utilities
# =============================================================================


def lower_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Returns a GeoDataFrame with normalized lowercase attribute names.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        Copy of the input GeoDataFrame with lowercase non-geometry field names.
    """
    output = gdf.copy()
    geometry_name = output.geometry.name

    output.columns = [
        column if column == geometry_name else column.lower()
        for column in output.columns
    ]
    return output


def union_geometries(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Dissolves all geometries from a GeoDataFrame into one geometry.

    Args:
        gdf: GeoDataFrame with an active geometry column.

    Returns:
        Unioned Shapely geometry.
    """
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()

    return gdf.geometry.unary_union


def safe_layer_name(name: str) -> str:
    """Converts text into a conservative GeoPackage layer name.

    Args:
        name: Candidate layer name.

    Returns:
        Lowercase layer name containing only letters, numbers, and underscores.
    """
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def is_zero_match_error(error: Exception) -> bool:
    """Checks whether a service exception indicates no matching features.

    Args:
        error: Exception raised by a HyRiver request.

    Returns:
        True if the exception text indicates zero matching features.
    """
    message = f"{type(error).__name__}: {error}".lower()

    indicators = (
        "zeromatched",
        "zero matched",
        "no features",
        "no feature",
        "no records",
    )
    return any(indicator in message for indicator in indicators)


def available_layers(source_path: Path) -> list[str]:
    """Returns spatial layer names in a GeoPackage or file geodatabase.

    Args:
        source_path: Path to a spatial datasource.

    Returns:
        Sorted layer names.

    Raises:
        FileNotFoundError: If source_path does not exist.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Datasource does not exist: {source_path}")

    return sorted(listlayers(source_path))


def write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
    overwrite: bool = True,
) -> None:
    """Writes a non-empty GeoDataFrame into a GeoPackage layer.

    Args:
        gdf: Features to write.
        gpkg_path: Target GeoPackage path.
        layer_name: Target layer name.
        overwrite: Delete an existing same-name layer before writing.

    Raises:
        ValueError: If gdf is empty.
    """
    if gdf.empty:
        raise ValueError(f"Refusing to write empty layer '{layer_name}'.")

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    existing = available_layers(gpkg_path) if gpkg_path.exists() else []
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
    """Reads a vector layer and normalizes attribute names to lowercase.

    Args:
        source_path: GeoPackage, file geodatabase, or other OGR datasource.
        layer_name: Layer name to load.

    Returns:
        GeoDataFrame with lowercase non-geometry field names.

    Raises:
        FileNotFoundError: If the datasource does not exist.
        ValueError: If the specified layer is absent.
    """
    source_path = Path(source_path)

    layers = available_layers(source_path)
    if layer_name not in layers:
        raise ValueError(
            f"Layer '{layer_name}' not found in '{source_path.name}'. "
            f"Available layers: {layers}"
        )

    return lower_columns(gpd.read_file(source_path, layer=layer_name))


def empty_geodataframe(crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
    """Creates an empty GeoDataFrame with an active geometry column.

    Args:
        crs: CRS assigned to the empty output.

    Returns:
        Empty GeoDataFrame.
    """
    return gpd.GeoDataFrame(geometry=[], crs=crs)


class NHDPlusHRDownloader:
    """Downloads selected NHDPlus HR datasets HUC12-by-HUC12.

    Every successful NHDPlus HR response is immediately written to a raw
    GeoPackage layer. This makes interrupted downloads restartable: rerunning
    with ``overwrite_raw=False`` skips raw chunks already saved.

    Raw GeoPackage layers:
        raw_flowline_<huc12>
        raw_catchment_<huc12>
        raw_waterbody_<huc12>

    Merged GeoPackage layers:
        merged_flowlines
        merged_catchments
        merged_waterbodies

    Catchment requests can be much heavier than flowline requests. If a full
    HUC12 catchment request repeatedly returns an HTTP 504, the downloader
    falls back to tiled requests, then de-duplicates full returned polygons by
    ``nhdplusid`` before saving one raw catchment layer for that HUC12.
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

    def __init__(
        self,
        config: DomainConfig,
    ) -> None:
        """Initializes an NHDPlus HR downloader.

        Args:
            config: Project configuration containing source selection, paths,
                retry settings, and target HUC12.
        """
        self.config = config
        self.gpkg_path = config.gpkg_path
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
            self,
            layers: list[str] | tuple[str, ...] | None = None,
            overwrite_raw: bool = False,
            overwrite_merged: bool = True,
            continue_on_failure: bool = True,
            purge_raw_after_merge: bool = False,
    ) -> dict[str, int]:
        """Downloads and merges selected NHDPlus HR source layers.

        Args:
            layers: Optional logical layer override. If None, uses
                ``config.download_layers``. Valid layers are flowline,
                catchment, and waterbody.
            overwrite_raw: If True, downloads and replaces raw chunks already
                saved in the GeoPackage. If False, existing raw chunks are
                preserved and skipped.
            overwrite_merged: If True, rebuilds merged source layers after
                downloading each selected source type.
            continue_on_failure: If True, logs individual failed HUC12
                requests and continues downloading remaining HUC12s.

        Returns:
            Dictionary containing source HUC12 count, failure count, and
            feature count for each merged layer that was requested.

        Raises:
            RuntimeError: If no raw chunks exist for a requested layer when a
                merge is attempted.
        """
        requested_layers = (
            DomainConfig.validate_layers(layers, "layers")
            if layers is not None
            else self.config.download_layers
        )

        huc12s = self.get_upstream_huc12s()
        tqdm.write(f"Found {len(huc12s)} HUC12s in the upstream domain.")

        hucs = self.download_huc12_polygons(huc12s)
        huc_geometries = self._huc12_geometry_lookup(
            hucs=hucs,
            requested_huc12s=huc12s,
        )

        result: dict[str, int] = {
            "huc12_count": len(huc12s),
            "failed_requests": 0,
        }

        for source_layer in requested_layers:
            failures = self.download_layer_by_huc12(
                source_layer=source_layer,
                huc_geometries=huc_geometries,
                overwrite_raw=overwrite_raw,
                continue_on_failure=continue_on_failure,
            )

            result["failed_requests"] += failures

            if overwrite_merged:
                merged = self.merge_saved_chunks(
                    source_layer=source_layer,
                    overwrite=True,
                )
                result[f"{source_layer}_count"] = len(merged)

            if purge_raw_after_merge and failures == 0:
                deleted = self.purge_raw_layers(
                    layers=(source_layer,),
                )
                result[
                    f"{source_layer}_raw_layers_deleted"
                ] = deleted[source_layer]

            elif purge_raw_after_merge and failures > 0:
                tqdm.write(
                    f"Keeping raw {source_layer} chunks because "
                    f"{failures} HUC12 request(s) failed."
                )

        tqdm.write(f"Download summary: {result}")
        return result

    def get_upstream_huc12s(self) -> list[str]:
        """Finds target and upstream HUC12 IDs using NLDI navigation.

        NLDI supplies only the upstream HUC12 identifiers. NHDPlus HR layers
        are then queried independently using those HUC12 geometries.

        Returns:
            Sorted list of target, upstream, and configured extra HUC12 IDs.
        """
        navigation = NLDI().navigate_byid(
            fsource="huc12pp",
            fid=self.config.huc12,
            navigation="upstreamTributaries",
            source="huc12pp",
            distance=1000,
        )

        identifier_field = (
            "identifier"
            if "identifier" in navigation.columns
            else navigation.columns[0]
        )

        huc12s = set(
            navigation[identifier_field]
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
        overwrite: bool = False,
    ) -> gpd.GeoDataFrame:
        """Downloads and saves source HUC12 polygons.

        Args:
            huc12s: HUC12 IDs to retrieve.
            overwrite: If True, replace the saved ``download_huc12`` layer.

        Returns:
            HUC12 GeoDataFrame with lowercase attribute names.

        Raises:
            RuntimeError: If neither NHDPlus HR nor WaterData returns the
                requested HUC12 polygons.
        """
        layer_name = "download_huc12"

        if not overwrite and self._layer_exists(layer_name):
            return read_layer(self.gpkg_path, layer_name)

        try:
            hucs = NHDPlusHR("huc12").byids("huc12", huc12s)

        except Exception as first_error:  # noqa: BLE001
            tqdm.write(
                "NHDPlus HR HUC12 request failed. Trying WaterData.\n"
                f"Cause: {type(first_error).__name__}: {first_error}"
            )

            try:
                hucs = WaterData("huc12").byid("huc12", huc12s)

            except Exception as second_error:  # noqa: BLE001
                raise RuntimeError(
                    "Unable to download HUC12 polygons from NHDPlus HR or "
                    "WaterData."
                ) from second_error

        hucs = lower_columns(hucs)

        write_layer(
            gdf=hucs,
            gpkg_path=self.gpkg_path,
            layer_name=layer_name,
            overwrite=True,
        )

        return hucs

    def download_layer_by_huc12(
        self,
        source_layer: str,
        huc_geometries: dict[str, BaseGeometry],
        overwrite_raw: bool = False,
        continue_on_failure: bool = True,
    ) -> int:
        """Downloads a selected NHDPlus HR layer HUC12-by-HUC12.

        Every successful HUC12 query is immediately written as a raw layer.
        For catchments, a 504 gateway timeout invokes a tiled fallback.

        Args:
            source_layer: Logical layer name: flowline, catchment, or waterbody.
            huc_geometries: Mapping from HUC12 ID to query polygon geometry.
            overwrite_raw: Re-download and overwrite already saved raw chunks.
            continue_on_failure: Log failures and continue remaining HUC12s.

        Returns:
            Count of HUC12 requests that ultimately failed.

        Raises:
            ValueError: If source_layer is invalid.
            RuntimeError: If a request fails and continue_on_failure is False.
        """
        if source_layer not in DomainConfig.VALID_LAYERS:
            raise ValueError(
                f"Invalid source_layer '{source_layer}'. Valid values are: "
                f"{DomainConfig.VALID_LAYERS}."
            )

        service = NHDPlusHR(source_layer)
        failed_count = 0

        for huc12, geometry in tqdm(
            huc_geometries.items(),
            desc=f"Downloading {source_layer}",
            unit="HUC12",
        ):
            raw_layer_name = self._raw_layer_name(
                source_layer=source_layer,
                huc12=huc12,
            )

            if not overwrite_raw and self._layer_exists(raw_layer_name):
                continue

            try:
                chunk = self._request_by_geometry(
                    service=service,
                    geometry=geometry,
                    source_layer=source_layer,
                    huc12=huc12,
                )

            except RuntimeError as full_query_error:
                if (
                    source_layer == "catchment"
                    and self._is_timeout_error(full_query_error)
                ):
                    tqdm.write(
                        f"Full catchment request timed out for HUC12 {huc12}. "
                        "Trying 3 x 3 tiled requests."
                    )

                    try:
                        chunk = self._download_catchments_tiled(
                            huc12=huc12,
                            geometry=geometry,
                            ncols=3,
                            nrows=3,
                        )

                    except RuntimeError as tile_query_error:
                        if self._is_timeout_error(tile_query_error):
                            tqdm.write(
                                f"One or more 3 x 3 catchment tiles timed "
                                f"out for HUC12 {huc12}. Trying 5 x 5 tiles."
                            )

                            try:
                                chunk = self._download_catchments_tiled(
                                    huc12=huc12,
                                    geometry=geometry,
                                    ncols=5,
                                    nrows=5,
                                )

                            except RuntimeError as fine_tile_error:
                                failed_count += 1
                                self._log_failure(
                                    source_layer=source_layer,
                                    huc12=huc12,
                                    error=fine_tile_error,
                                )

                                tqdm.write(
                                    f"FAILED 5 x 5 tiled catchment request "
                                    f"for HUC12 {huc12}: {fine_tile_error}"
                                )

                                if not continue_on_failure:
                                    raise

                                continue

                        else:
                            failed_count += 1
                            self._log_failure(
                                source_layer=source_layer,
                                huc12=huc12,
                                error=tile_query_error,
                            )

                            tqdm.write(
                                f"FAILED tiled catchment request for HUC12 "
                                f"{huc12}: {tile_query_error}"
                            )

                            if not continue_on_failure:
                                raise

                            continue

                else:
                    failed_count += 1
                    self._log_failure(
                        source_layer=source_layer,
                        huc12=huc12,
                        error=full_query_error,
                    )

                    tqdm.write(
                        f"FAILED {source_layer} request for HUC12 {huc12}: "
                        f"{full_query_error}"
                    )

                    if not continue_on_failure:
                        raise

                    continue

            if chunk is None or chunk.empty:
                tqdm.write(
                    f"No {source_layer} features returned for HUC12 {huc12}."
                )
                continue

            chunk = lower_columns(chunk)
            chunk["source_huc12"] = str(huc12)

            write_layer(
                gdf=chunk,
                gpkg_path=self.gpkg_path,
                layer_name=raw_layer_name,
                overwrite=True,
            )

        return failed_count

    def merge_saved_chunks(
        self,
        source_layer: str,
        overwrite: bool = True,
    ) -> gpd.GeoDataFrame:
        """Merges saved raw layers into one de-duplicated source layer.

        Args:
            source_layer: Logical source layer: flowline, catchment, waterbody.
            overwrite: Replace an existing merged layer.

        Returns:
            Merged GeoDataFrame.

        Raises:
            RuntimeError: If no raw source chunks are available.
            KeyError: If merged flowlines lack required VAA fields.
        """
        raw_layers = self._raw_layers(source_layer)

        if not raw_layers:
            raise RuntimeError(
                f"No raw chunks are available for '{source_layer}'."
            )

        unique_field = self._unique_id_field(source_layer)
        merged: gpd.GeoDataFrame | None = None

        for raw_layer_name in tqdm(
            raw_layers,
            desc=f"Merging {source_layer}",
            unit="chunk",
        ):
            chunk = read_layer(self.gpkg_path, raw_layer_name)

            if merged is None:
                merged = chunk

            else:
                merged = gpd.GeoDataFrame(
                    pd.concat([merged, chunk], ignore_index=True),
                    crs=merged.crs,
                )

            if unique_field in merged.columns:
                merged = merged.drop_duplicates(
                    subset=unique_field,
                    keep="first",
                ).copy()

            else:
                merged = merged.drop_duplicates().copy()

        if merged is None or merged.empty:
            raise RuntimeError(
                f"Merged result for '{source_layer}' is empty."
            )

        if source_layer in ("flowline", "catchment"):
            if "nhdplusid" not in merged.columns:
                raise KeyError(
                    f"Merged '{source_layer}' layer has no 'nhdplusid' field."
                )

            merged["nhdplusid"] = merged["nhdplusid"].astype("int64")

        if source_layer == "flowline":
            missing = self.REQUIRED_FLOWLINE_COLUMNS - set(merged.columns)

            if missing:
                raise KeyError(
                    "Merged flowlines lack required VAA attributes: "
                    f"{sorted(missing)}"
                )

        if source_layer == "waterbody":
            merged = self._ensure_waterbody_id(merged)

        write_layer(
            gdf=merged,
            gpkg_path=self.gpkg_path,
            layer_name=self.MERGED_LAYER_MAP[source_layer],
            overwrite=overwrite,
        )

        return merged

    def _request_by_geometry(
        self,
        service: NHDPlusHR,
        geometry: BaseGeometry,
        source_layer: str,
        huc12: str,
    ) -> gpd.GeoDataFrame | None:
        """Requests one source layer for one geometry with retry handling.

        Args:
            service: Initialized NHDPlusHR object for a selected layer.
            geometry: Query geometry, normally in EPSG:4326.
            source_layer: Source-layer label used in status messages.
            huc12: HUC12 associated with this request.

        Returns:
            Downloaded GeoDataFrame, or None for a zero-feature response.

        Raises:
            RuntimeError: If all request attempts fail. The original exception
                is chained so the full PyNHD/HTTP cause remains available.
        """
        final_error: Exception | None = None
        total_attempts = self.config.max_retries + 1

        for attempt_index in range(total_attempts):
            attempt_number = attempt_index + 1

            try:
                return service.bygeom(geometry)

            except Exception as error:  # noqa: BLE001
                final_error = error

                if is_zero_match_error(error):
                    return None

                tqdm.write(
                    f"{source_layer} request failed for {huc12} "
                    f"({attempt_number}/{total_attempts}). "
                    f"{type(error).__name__}: {error}"
                )

                if attempt_number < total_attempts:
                    pause_seconds = (
                        self.config.retry_seconds * attempt_number
                    )

                    tqdm.write(
                        f"Retrying {source_layer} for {huc12} in "
                        f"{pause_seconds:.0f} seconds."
                    )

                    time.sleep(pause_seconds)

        raise RuntimeError(
            f"Failed downloading {source_layer} for HUC12 {huc12} after "
            f"{total_attempts} attempts. Final cause: "
            f"{type(final_error).__name__}: {final_error}"
        ) from final_error

    def _download_catchments_tiled(
        self,
        huc12: str,
        geometry: BaseGeometry,
        ncols: int,
        nrows: int,
    ) -> gpd.GeoDataFrame:
        """Downloads catchments by splitting a HUC12 into smaller query tiles.

        A returned catchment polygon may intersect more than one tile. The
        response is therefore de-duplicated using ``nhdplusid``. The service
        should return full original catchment polygons rather than clipped
        fragments, because each query is an intersection query.

        Args:
            huc12: Source HUC12 identifier.
            geometry: HUC12 polygon in geographic coordinates.
            ncols: Number of longitudinal tile columns.
            nrows: Number of latitudinal tile rows.

        Returns:
            De-duplicated catchment GeoDataFrame.

        Raises:
            RuntimeError: If a tile request fails after all retries.
            KeyError: If returned catchment data lack nhdplusid.
        """
        minx, miny, maxx, maxy = geometry.bounds
        delta_x = (maxx - minx) / ncols
        delta_y = (maxy - miny) / nrows

        tiles: list[BaseGeometry] = []

        for row_index in range(nrows):
            for col_index in range(ncols):
                candidate = box(
                    minx + col_index * delta_x,
                    miny + row_index * delta_y,
                    minx + (col_index + 1) * delta_x,
                    miny + (row_index + 1) * delta_y,
                )

                tile = candidate.intersection(geometry)

                if not tile.is_empty:
                    tiles.append(tile)

        service = NHDPlusHR("catchment")
        chunks: list[gpd.GeoDataFrame] = []

        for tile_number, tile in enumerate(
            tqdm(
                tiles,
                desc=f"Tiled catchments {huc12} ({ncols}x{nrows})",
                unit="tile",
                leave=False,
            ),
            start=1,
        ):
            result = self._request_by_geometry(
                service=service,
                geometry=tile,
                source_layer=f"catchment tile {tile_number}",
                huc12=huc12,
            )

            if result is not None and not result.empty:
                chunks.append(lower_columns(result))

        if not chunks:
            return empty_geodataframe(crs="EPSG:4326")

        catchments = gpd.GeoDataFrame(
            pd.concat(chunks, ignore_index=True),
            crs=chunks[0].crs,
        )

        if "nhdplusid" not in catchments.columns:
            raise KeyError(
                f"Tiled catchment data for HUC12 {huc12} lack 'nhdplusid'."
            )

        return catchments.drop_duplicates(
            subset="nhdplusid",
            keep="first",
        ).copy()

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        """Checks whether an exception represents a gateway/service timeout.

        Args:
            error: Exception produced by the source request or wrapper.

        Returns:
            True if the exception text indicates a 504 or CloudFront timeout.
        """
        message = str(error).lower()

        timeout_indicators = (
            "504",
            "gateway timeout",
            "cloudfront",
            "request could not be satisfied",
            "readtimeout",
            "read timed out",
            "connection timed out",
        )

        return any(indicator in message for indicator in timeout_indicators)

    def _huc12_geometry_lookup(
        self,
        hucs: gpd.GeoDataFrame,
        requested_huc12s: list[str],
    ) -> dict[str, BaseGeometry]:
        """Creates HUC12-to-geometry lookup from downloaded HUC polygons.

        Args:
            hucs: Source HUC12 polygon GeoDataFrame.
            requested_huc12s: HUC12 values expected from NLDI navigation.

        Returns:
            Mapping from HUC12 code to Shapely query geometry.

        Raises:
            KeyError: If no HUC12 identifier field is found.
            ValueError: If requested HUC12 polygons are absent.
        """
        candidate_fields = ("huc12", "huc_12", "identifier")

        id_field = next(
            (
                field
                for field in candidate_fields
                if field in hucs.columns
            ),
            None,
        )

        if id_field is None:
            raise KeyError(
                "Unable to identify a HUC12 attribute field. "
                f"Available fields: {list(hucs.columns)}"
            )

        output = hucs.copy()

        output["huc12_code"] = (
            output[id_field]
            .astype(str)
            .str.extract(r"(\d{12})")[0]
        )

        returned_hucs = set(output["huc12_code"].dropna())
        missing_hucs = sorted(set(requested_huc12s) - returned_hucs)

        if missing_hucs:
            raise ValueError(
                f"Missing {len(missing_hucs)} HUC12 polygons. "
                f"First missing values: {missing_hucs[:10]}"
            )

        return dict(zip(output["huc12_code"], output.geometry))

    def _raw_layer_name(
        self,
        source_layer: str,
        huc12: str,
    ) -> str:
        """Returns a GeoPackage-safe raw layer name.

        Args:
            source_layer: Logical source-layer name.
            huc12: Source HUC12 code.

        Returns:
            Raw GeoPackage layer name.
        """
        return safe_layer_name(
            f"{self.RAW_PREFIX[source_layer]}{huc12}"
        )

    def _raw_layers(
        self,
        source_layer: str,
    ) -> list[str]:
        """Returns saved raw chunk layer names for one source layer.

        Args:
            source_layer: Logical source-layer name.

        Returns:
            Sorted list of raw GeoPackage layer names.
        """
        if not self.gpkg_path.exists():
            return []

        prefix = self.RAW_PREFIX[source_layer]

        return sorted(
            layer_name
            for layer_name in available_layers(self.gpkg_path)
            if layer_name.startswith(prefix)
        )

    def purge_raw_layers(
            self,
            layers: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        """Deletes raw per-HUC12 layers after merged data are verified.

        This permanently removes the intermediate raw source layers:

            raw_flowline_<huc12>
            raw_catchment_<huc12>
            raw_waterbody_<huc12>

        It refuses to purge a source type unless its merged layer exists first.

        Args:
            layers: Logical source layers to purge. If None, uses
                ``self.config.download_layers``.

        Returns:
            Dictionary mapping each source layer to its count of deleted raw
            GeoPackage layers.

        Raises:
            FileNotFoundError: If the output GeoPackage does not exist.
            RuntimeError: If a requested source layer has no merged layer.
        """
        if not self.gpkg_path.exists():
            raise FileNotFoundError(
                f"Cannot purge raw layers because the GeoPackage does not exist: "
                f"{self.gpkg_path}"
            )

        requested_layers = (
            DomainConfig.validate_layers(layers, "layers")
            if layers is not None
            else self.config.download_layers
        )

        deleted_counts: dict[str, int] = {}

        for source_layer in requested_layers:
            merged_layer = self.MERGED_LAYER_MAP[source_layer]

            if not self._layer_exists(merged_layer):
                raise RuntimeError(
                    f"Refusing to purge raw '{source_layer}' layers because "
                    f"merged layer '{merged_layer}' does not exist. Build and "
                    "verify the merged layer first."
                )

            raw_layers = self._raw_layers(source_layer)

            for raw_layer_name in tqdm(
                    raw_layers,
                    desc=f"Purging raw {source_layer}",
                    unit="layer",
            ):
                remove(
                    self.gpkg_path,
                    layer=raw_layer_name,
                    driver="GPKG",
                )

            deleted_counts[source_layer] = len(raw_layers)

            tqdm.write(
                f"Deleted {len(raw_layers)} raw {source_layer} layer(s)."
            )

        return deleted_counts


    def _layer_exists(
        self,
        layer_name: str,
    ) -> bool:
        """Checks whether a named layer exists in the output GeoPackage.

        Args:
            layer_name: Candidate layer name.

        Returns:
            True if the named layer is present.
        """
        if not self.gpkg_path.exists():
            return False

        return layer_name in available_layers(self.gpkg_path)

    def _log_failure(
        self,
        source_layer: str,
        huc12: str,
        error: Exception,
    ) -> None:
        """Writes a failed source request to the persistent failure log.

        Args:
            source_layer: Failed logical source layer.
            huc12: HUC12 associated with the failed request.
            error: Final exception.
        """
        record = pd.DataFrame(
            [
                {
                    "timestamp_utc": pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat(),
                    "huc12": str(huc12),
                    "source_layer": source_layer,
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
    def _unique_id_field(
        source_layer: str,
    ) -> str:
        """Returns the unique identifier used for layer de-duplication.

        Args:
            source_layer: Logical source-layer name.

        Returns:
            Feature-ID field name.
        """
        if source_layer in ("flowline", "catchment"):
            return "nhdplusid"

        return "wb_id"

    @staticmethod
    def _ensure_waterbody_id(
        waterbodies: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Adds a stable project waterbody ID field.

        Args:
            waterbodies: Waterbody GeoDataFrame.

        Returns:
            Copy containing a string ``wb_id`` column.
        """
        output = waterbodies.copy()

        if "wb_id" in output.columns:
            output["wb_id"] = output["wb_id"].astype(str)
            return output

        if "nhdplusid" in output.columns:
            output["wb_id"] = output["nhdplusid"].astype(str)

        elif "permanent_identifier" in output.columns:
            output["wb_id"] = (
                output["permanent_identifier"].astype(str)
            )

        else:
            output["wb_id"] = output.index.astype(str)

        return output


class LocalDomainImporter:
    """Loads selected merged downloader layers or Arc Pro–prepared source data.

    The importer can load any subset of:

        - ``flowline``
        - ``catchment``
        - ``waterbody``

    By default, it loads the merged downloader layers in
    ``config.source_path``. To load Arc Pro–prepared layers, initialize
    ``DomainConfig`` with a different ``source_path`` and/or
    ``source_layer_map``.
    """

    REQUIRED_FLOWLINE_COLUMNS = {
        "nhdplusid",
        "hydroseq",
        "dnhydroseq",
        "totdasqkm",
    }

    def __init__(
        self,
        config: DomainConfig,
        source_path: str | Path | None = None,
        layer_map: dict[str, str] | None = None,
        load_layers: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Initializes a local layer importer.

        Args:
            config: Project configuration.
            source_path: Optional override for config.source_path.
            layer_map: Optional logical-to-physical layer-name override.
            load_layers: Optional requested logical-layer override.
        """
        self.config = config
        self.source_path = (
            Path(source_path)
            if source_path is not None
            else config.source_path
        )

        self.layer_map = config.source_layer_map.copy()

        if layer_map is not None:
            self.layer_map.update(
                {
                    str(logical_name).lower(): str(physical_name)
                    for logical_name, physical_name in layer_map.items()
                }
            )

        self.load_layers = (
            DomainConfig.validate_layers(load_layers, "load_layers")
            if load_layers is not None
            else config.load_layers
        )

    def load(
        self,
        layers: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, gpd.GeoDataFrame]:
        """Loads and validates selected spatial source layers.

        Args:
            layers: Optional load selection. If None, uses this importer's
                configured load_layers.

        Returns:
            Dictionary keyed by logical layer names. For example:
            ``data["flowline"]`` or ``data["catchment"]``.

        Raises:
            FileNotFoundError: If the datasource does not exist.
            ValueError: If a requested GIS layer is absent.
            KeyError: If a loaded layer lacks required fields.
        """
        requested = (
            DomainConfig.validate_layers(layers, "layers")
            if layers is not None
            else self.load_layers
        )

        existing = set(available_layers(self.source_path))
        result: dict[str, gpd.GeoDataFrame] = {}

        for logical_layer in requested:
            physical_layer = self.layer_map.get(logical_layer)

            if physical_layer is None:
                raise ValueError(
                    f"No physical layer mapping exists for '{logical_layer}'."
                )

            if physical_layer not in existing:
                raise ValueError(
                    f"Requested {logical_layer} layer '{physical_layer}' is "
                    f"not present in '{self.source_path.name}'. "
                    f"Available layers: {sorted(existing)}"
                )

            gdf = lower_columns(
                gpd.read_file(self.source_path, layer=physical_layer)
            )

            if logical_layer == "flowline":
                self._validate_flowlines(gdf)
                gdf["nhdplusid"] = gdf["nhdplusid"].astype("int64")

            elif logical_layer == "catchment":
                self._validate_catchments(gdf)
                gdf["nhdplusid"] = gdf["nhdplusid"].astype("int64")

            elif logical_layer == "waterbody":
                gdf = self._ensure_waterbody_id(gdf)

            result[logical_layer] = gdf

        return result

    def _validate_flowlines(self, flowlines: gpd.GeoDataFrame) -> None:
        """Checks fields required for dendritic VAA routing.

        Args:
            flowlines: Flowline GeoDataFrame.

        Raises:
            KeyError: If the flowline layer lacks required fields.
        """
        missing = self.REQUIRED_FLOWLINE_COLUMNS - set(flowlines.columns)

        if missing:
            raise KeyError(
                "Flowline layer lacks required NHDPlus HR VAA fields: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _validate_catchments(catchments: gpd.GeoDataFrame) -> None:
        """Checks the catchment link field.

        Args:
            catchments: Catchment GeoDataFrame.

        Raises:
            KeyError: If nhdplusid is absent.
        """
        if "nhdplusid" not in catchments.columns:
            raise KeyError(
                "Catchment layer must contain an 'nhdplusid' field."
            )

    @staticmethod
    def _ensure_waterbody_id(
        waterbodies: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Ensures waterbodies have a project-ready ``wb_id`` field.

        Args:
            waterbodies: Waterbody GeoDataFrame.

        Returns:
            Copy with a string ``wb_id`` field.
        """
        output = waterbodies.copy()

        if "wb_id" in output.columns:
            output["wb_id"] = output["wb_id"].astype(str)
            return output

        if "nhdplusid" in output.columns:
            output["wb_id"] = output["nhdplusid"].astype(str)

        elif "permanent_identifier" in output.columns:
            output["wb_id"] = output["permanent_identifier"].astype(str)

        else:
            output["wb_id"] = output.index.astype(str)

        return output


class DendriticDomainRouter:
    """Builds a dendritic PRMS parent-HRU and segment-routing domain.

    The active downstream connection is defined by the NHDPlus HR VAA primary
    downstream relation:

    ``flowline.dnhydroseq -> receiving_flowline.hydroseq``.

    This selects one primary downstream flowline per segment. Alternative
    divergence paths are not included in the initial active routing graph,
    although source VAA divergence fields are retained in exported segments
    for later QA and manual review.
    """

    REQUIRED_ROUTING_LAYERS = {"flowline", "catchment"}

    def __init__(
        self,
        config: DomainConfig,
        flowlines: gpd.GeoDataFrame,
        catchments: gpd.GeoDataFrame,
        waterbodies: gpd.GeoDataFrame | None = None,
    ) -> None:
        """Initializes the domain router.

        Args:
            config: Project configuration.
            flowlines: Merged or Arc Pro–prepared NHDPlus HR flowlines.
            catchments: Merged or Arc Pro–prepared catchments.
            waterbodies: Optional waterbody polygon layer.
        """
        self.config = config
        self.flowlines = lower_columns(flowlines)
        self.catchments = lower_columns(catchments)

        self.flowlines["nhdplusid"] = (
            self.flowlines["nhdplusid"].astype("int64")
        )
        self.catchments["nhdplusid"] = (
            self.catchments["nhdplusid"].astype("int64")
        )

        if waterbodies is None:
            self.waterbodies = empty_geodataframe(crs=catchments.crs)
        else:
            self.waterbodies = lower_columns(waterbodies)

    def build(
        self,
        target_huc12: gpd.GeoDataFrame | None = None,
    ) -> dict[str, Any]:
        """Builds and writes segments, parent HRUs, and routing QA outputs.

        Args:
            target_huc12: Target HUC12 polygon used to auto-select an outlet
                when config.outlet_nhdplusid is None.

        Returns:
            JSON-serializable routing and area QA report.

        Raises:
            ValueError: If the outlet cannot be identified or required source
                layers are invalid.
        """
        flowlines = self._add_dendritic_link(self.flowlines)
        outlet = self._select_outlet(flowlines, target_huc12)

        graph, upstream_ids = self._trace_upstream(
            flowlines=flowlines,
            outlet=outlet,
        )

        segments, id_to_segment = self._build_segments(
            flowlines=flowlines,
            upstream_ids=upstream_ids,
            outlet=outlet,
        )

        hrus, unrouted = self._build_hrus(id_to_segment)

        (
            segments,
            hrus,
            unrouted,
            waterbodies,
        ) = self._add_waterbody_info(
            segments=segments,
            hrus=hrus,
            unrouted=unrouted,
        )

        unrouted = self._filter_unrouted_in_domain(
            unrouted=unrouted,
            hrus=hrus,
        )

        report = self._make_qa_report(
            segments=segments,
            hrus=hrus,
            unrouted=unrouted,
            graph=graph,
            outlet=outlet,
        )

        self._write_model_outputs(
            segments=segments,
            hrus=hrus,
            unrouted=unrouted,
            waterbodies=waterbodies,
            report=report,
        )

        return report

    @staticmethod
    def _add_dendritic_link(
        flowlines: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Creates one primary downstream NHDPlus ID per flowline.

        Args:
            flowlines: Source flowlines containing nhdplusid, hydroseq, and
                dnhydroseq fields.

        Returns:
            Copy of flowlines with a ``to_nhdplusid`` field. Terminal flowlines
            have null downstream values.
        """
        output = flowlines.copy()

        hydroseq = output["hydroseq"].fillna(0).round().astype("int64")
        dnhydroseq = output["dnhydroseq"].fillna(0).round().astype("int64")

        hydroseq_to_id = dict(zip(hydroseq, output["nhdplusid"]))

        output["to_nhdplusid"] = dnhydroseq.map(hydroseq_to_id)
        output.loc[dnhydroseq == 0, "to_nhdplusid"] = np.nan

        return output

    def _select_outlet(
        self,
        flowlines: gpd.GeoDataFrame,
        target_huc12: gpd.GeoDataFrame | None,
    ) -> int:
        """Selects the configured or automatic outlet flowline.

        Args:
            flowlines: Flowlines including VAA drainage-area attributes.
            target_huc12: Target HUC12 geometry for automatic selection.

        Returns:
            Outlet NHDPlusID.

        Raises:
            ValueError: If no valid outlet can be identified.
        """
        if self.config.outlet_nhdplusid is not None:
            outlet = int(self.config.outlet_nhdplusid)

            if outlet not in set(flowlines["nhdplusid"]):
                raise ValueError(
                    f"Configured outlet NHDPlusID {outlet} does not occur in "
                    "the supplied flowline layer."
                )

            return outlet

        if target_huc12 is None or target_huc12.empty:
            raise ValueError(
                "An explicit outlet_nhdplusid is required when target_huc12 "
                "is not supplied."
            )

        target_geometry = union_geometries(target_huc12)

        candidates = flowlines[
            flowlines.intersects(target_geometry)
        ].copy()

        if candidates.empty:
            raise ValueError(
                "No candidate flowlines intersect the target HUC12."
            )

        outlet_row = candidates.sort_values(
            "totdasqkm",
            ascending=False,
        ).iloc[0]

        outlet = int(outlet_row["nhdplusid"])

        tqdm.write(
            "Auto-selected outlet: "
            f"NHDPlusID={outlet}, "
            f"name={outlet_row.get('gnis_name', '<unnamed>')}, "
            f"TotDASqKm={outlet_row['totdasqkm']}"
        )

        return outlet

    @staticmethod
    def _trace_upstream(
        flowlines: gpd.GeoDataFrame,
        outlet: int,
    ) -> tuple[nx.DiGraph, set[int]]:
        """Builds the directed graph and identifies flowlines upstream of outlet.

        Args:
            flowlines: Flowlines with to_nhdplusid.
            outlet: Selected outlet NHDPlusID.

        Returns:
            Directed downstream graph and set of upstream flowline IDs,
            including the outlet.

        Raises:
            ValueError: If outlet is absent from the graph.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(flowlines["nhdplusid"])

        edges = flowlines.dropna(subset=["to_nhdplusid"])

        graph.add_edges_from(
            zip(
                edges["nhdplusid"],
                edges["to_nhdplusid"].astype("int64"),
            )
        )

        if outlet not in graph:
            raise ValueError(
                f"Outlet NHDPlusID {outlet} is not present in the flowline "
                "routing graph."
            )

        upstream_ids = nx.ancestors(graph, outlet) | {outlet}

        return graph, upstream_ids

    def _build_segments(
        self,
        flowlines: gpd.GeoDataFrame,
        upstream_ids: set[int],
        outlet: int,
    ) -> tuple[gpd.GeoDataFrame, dict[int, int]]:
        """Creates model segment IDs and downstream segment relationships.

        Args:
            flowlines: Flowlines with to_nhdplusid.
            upstream_ids: NHDPlus IDs to retain in the model.
            outlet: Outlet NHDPlusID.

        Returns:
            Tuple containing model segment features and NHDPlusID-to-segment-ID
            mapping.
        """
        segments = (
            flowlines[
                flowlines["nhdplusid"].isin(upstream_ids)
            ]
            .sort_values("hydroseq", ascending=False)
            .reset_index(drop=True)
            .copy()
        )

        segments["segment_id"] = np.arange(
            1,
            len(segments) + 1,
            dtype=int,
        )

        id_to_segment = dict(
            zip(
                segments["nhdplusid"],
                segments["segment_id"],
            )
        )

        segments["to_segment_id"] = (
            segments["to_nhdplusid"]
            .map(id_to_segment)
            .fillna(0)
            .astype(int)
        )

        segments["is_outlet"] = (
            segments["nhdplusid"] == outlet
        ).astype(int)

        segments.loc[
            segments["is_outlet"] == 1,
            "to_segment_id",
        ] = 0

        segments = segments.to_crs(self.config.work_crs)
        segments["length_m"] = segments.geometry.length

        return segments, id_to_segment

    def _build_hrus(
        self,
        id_to_segment: dict[int, int],
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Creates routed parent HRUs and unrouted catchments.

        Args:
            id_to_segment: Mapping from routed NHDPlus IDs to model segment IDs.

        Returns:
            Tuple of routed HRU polygons and unrouted catchment polygons.
        """
        catchments = self.catchments.to_crs(
            self.config.work_crs
        ).copy()

        hrus = catchments[
            catchments["nhdplusid"].isin(id_to_segment)
        ].copy()

        hrus["segment_id"] = (
            hrus["nhdplusid"]
            .map(id_to_segment)
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

        hrus["parent_nhdplusid"] = hrus["nhdplusid"]

        hrus["small_hru"] = (
            hrus["area_m2"] < self.config.min_hru_area_m2
        ).astype(int)

        unrouted = catchments[
            ~catchments["nhdplusid"].isin(id_to_segment)
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
        """Adds waterbody flags and fractions to model features.

        Args:
            segments: Model segment GeoDataFrame.
            hrus: Routed HRU GeoDataFrame.
            unrouted: Unrouted catchment GeoDataFrame.

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

        if "wb_id" not in waterbodies.columns:
            waterbodies["wb_id"] = waterbodies.index.astype(str)

        waterbodies["geometry"] = waterbodies.geometry.buffer(0)

        midpoint = segments[["segment_id", "geometry"]].copy()
        midpoint["geometry"] = segments.geometry.interpolate(
            0.5,
            normalized=True,
        )

        segment_hits = gpd.sjoin(
            midpoint,
            waterbodies[["wb_id", "geometry"]],
            how="left",
            predicate="within",
        )

        segment_waterbody = (
            segment_hits
            .drop_duplicates("segment_id")
            .set_index("segment_id")["wb_id"]
        )

        segments["waterbody_id"] = segments["segment_id"].map(
            segment_waterbody
        )
        segments["in_waterbody"] = (
            segments["waterbody_id"].notna()
        ).astype(int)

        hrus["waterbody_frac"] = self._waterbody_fraction(
            polygons=hrus,
            key_field="hru_id",
            waterbodies=waterbodies,
        )

        unrouted["waterbody_frac"] = self._waterbody_fraction(
            polygons=unrouted,
            key_field="nhdplusid",
            waterbodies=waterbodies,
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
        """Calculates waterbody area fraction for each polygon.

        Args:
            polygons: Input polygons.
            key_field: Unique identifier in polygons.
            waterbodies: Waterbody polygons in the same projected CRS.

        Returns:
            Series indexed by the supplied key field, with fractions from 0 to 1.
        """
        if polygons.empty:
            return pd.Series(dtype=float)

        intersections = gpd.overlay(
            polygons[[key_field, "geometry"]],
            waterbodies[["geometry"]],
            how="intersection",
        )

        wet_area = (
            intersections.geometry.area
            .groupby(intersections[key_field])
            .sum()
        )

        total_area = polygons.set_index(
            key_field
        ).geometry.area

        return (
            wet_area.reindex(total_area.index).fillna(0)
            / total_area
        ).clip(0, 1)

    @staticmethod
    def _filter_unrouted_in_domain(
        unrouted: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Keeps unrouted catchments lying inside the routed-HRU envelope.

        Args:
            unrouted: Catchments not linked to active model segments.
            hrus: Routed parent HRUs.

        Returns:
            Subset of unrouted catchments whose representative points occur
            within the convex hull of active HRUs.
        """
        if unrouted.empty or hrus.empty:
            return unrouted.copy()

        domain_hull = union_geometries(hrus).convex_hull

        inside = (
            unrouted.geometry
            .representative_point()
            .within(domain_hull)
        )

        return unrouted[inside].copy()

    @staticmethod
    def _make_qa_report(
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
        graph: nx.DiGraph,
        outlet: int,
    ) -> dict[str, Any]:
        """Calculates routing and area QA metrics.

        Args:
            segments: Model stream segments.
            hrus: Routed parent HRUs.
            unrouted: In-domain unrouted catchments.
            graph: Complete flowline graph.
            outlet: Outlet NHDPlusID.

        Returns:
            JSON-serializable QA report.
        """
        model_ids = set(segments["nhdplusid"])
        model_graph = graph.subgraph(model_ids)

        report: dict[str, Any] = {
            "outlet_nhdplusid": int(outlet),
            "n_segments": int(len(segments)),
            "n_hrus": int(len(hrus)),
            "is_dag": bool(
                nx.is_directed_acyclic_graph(model_graph)
            ),
            "all_segments_reach_outlet": bool(
                all(
                    node == outlet
                    or nx.has_path(model_graph, node, outlet)
                    for node in model_graph.nodes
                )
            ),
            "null_downstream_nonoutlet": int(
                (
                    (segments["to_segment_id"] == 0)
                    & (segments["is_outlet"] == 0)
                ).sum()
            ),
            "duplicate_parent_nhdplusid": int(
                hrus["parent_nhdplusid"].duplicated().sum()
            ),
            "segments_without_hru": int(
                (~segments["nhdplusid"].isin(hrus["nhdplusid"])).sum()
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

        outlet_drainage_area = segments.loc[
            segments["is_outlet"] == 1,
            "totdasqkm",
        ]

        report["outlet_totdasqkm_vaa"] = (
            float(outlet_drainage_area.iloc[0])
            if not outlet_drainage_area.empty
            else None
        )

        if report["outlet_totdasqkm_vaa"]:
            report["routed_area_difference_pct"] = round(
                100.0
                * (
                    report["routed_area_km2"]
                    - report["outlet_totdasqkm_vaa"]
                )
                / report["outlet_totdasqkm_vaa"],
                3,
            )

        else:
            report["routed_area_difference_pct"] = None

        if "divergence" in segments.columns:
            report["segments_flagged_divergence"] = int(
                (segments["divergence"].fillna(0) > 0).sum()
            )

        return report

    def _write_model_outputs(
        self,
        segments: gpd.GeoDataFrame,
        hrus: gpd.GeoDataFrame,
        unrouted: gpd.GeoDataFrame,
        waterbodies: gpd.GeoDataFrame,
        report: dict[str, Any],
    ) -> None:
        """Writes model layers, mapping tables, and routing QA products.

        Args:
            segments: Model segments.
            hrus: Parent HRUs.
            unrouted: In-domain unrouted catchments.
            waterbodies: Processed waterbodies.
            report: Routing QA report.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.qa_dir.mkdir(parents=True, exist_ok=True)

        segment_fields = [
            "segment_id",
            "to_segment_id",
            "nhdplusid",
            "to_nhdplusid",
            "is_outlet",
            "length_m",
            "slope",
            "streamorde",
            "totdasqkm",
            "hydroseq",
            "levelpathi",
            "divergence",
            "gnis_name",
            "in_waterbody",
            "waterbody_id",
            "geometry",
        ]

        hru_fields = [
            "hru_id",
            "segment_id",
            "parent_nhdplusid",
            "area_m2",
            "waterbody_frac",
            "small_hru",
            "geometry",
        ]

        unrouted_fields = [
            "nhdplusid",
            "area_m2",
            "waterbody_frac",
            "reason",
            "geometry",
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
            layer_name="model_segments",
            overwrite=True,
        )

        write_layer(
            hru_out,
            self.config.gpkg_path,
            layer_name="model_hrus",
            overwrite=True,
        )

        if not unrouted_out.empty:
            write_layer(
                unrouted_out,
                self.config.gpkg_path,
                layer_name="unrouted_catchments",
                overwrite=True,
            )

        if not waterbodies.empty:
            write_layer(
                waterbodies,
                self.config.gpkg_path,
                layer_name="model_waterbodies",
                overwrite=True,
            )

        segment_out.drop(columns="geometry").to_csv(
            self.config.output_dir / "segments.csv",
            index=False,
        )

        hru_out.drop(columns="geometry").to_csv(
            self.config.output_dir / "hru_to_segment.csv",
            index=False,
        )

        segment_out[
            ["segment_id", "to_segment_id"]
        ].to_csv(
            self.config.output_dir / "segment_routing.csv",
            index=False,
        )

        (self.config.qa_dir / "routing_qa.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )


def get_target_huc12(
    config: DomainConfig,
) -> gpd.GeoDataFrame:
    """Loads the target HUC12 polygon from the downloader GeoPackage.

    Args:
        config: Project configuration.

    Returns:
        GeoDataFrame containing the requested target HUC12.

    Raises:
        ValueError: If the target HUC12 cannot be found.
    """
    huc12s = read_layer(config.gpkg_path, "download_huc12")

    candidate_fields = ("huc12", "huc_12", "identifier")
    id_field = next(
        (field for field in candidate_fields if field in huc12s.columns),
        None,
    )

    if id_field is None:
        raise ValueError(
            "Cannot identify the HUC12 ID field in download_huc12. "
            f"Fields: {list(huc12s.columns)}"
        )

    target = huc12s[
        huc12s[id_field].astype(str).str.contains(
            config.huc12,
            na=False,
        )
    ].copy()

    if target.empty:
        raise ValueError(
            f"Target HUC12 {config.huc12} was not found in "
            "download_huc12."
        )

    return target


def main() -> None:
    """Downloads catchments/waterbodies and then builds the PRMS domain."""
    config = DomainConfig(
        huc12="100200080507",
        download_layers=None,
        load_layers=("flowline", "catchment", "waterbody"),
        source_layer_map={
            "catchment": "NHDcatchment",
            "waterbody": "NHDwaterbody"
        },
        gpkg_name="upper_gallatin_nhdhr.gpkg"
    )

    # downloader = NHDPlusHRDownloader(config)
    #
    # cleanup_result = downloader.purge_raw_layers(
    #     layers=("flowline",),
    # )
    # download_result = downloader.download(
    #     overwrite_raw=False,
    #     overwrite_merged=True,
    #     continue_on_failure=True,
    # )
    #
    # print("\nDownload result:")
    # print(json.dumps(download_result, indent=2))

    importer = LocalDomainImporter(config)
    data = importer.load()

    required = {"flowline", "catchment"}
    missing = required - set(data)

    if missing:
        raise RuntimeError(
            "Cannot build the PRMS routing domain. Missing locally loaded "
            f"layer(s): {sorted(missing)}"
        )

    target_huc12 = get_target_huc12(config)

    router = DendriticDomainRouter(
        config=config,
        flowlines=data["flowline"],
        catchments=data["catchment"],
        waterbodies=data.get("waterbody"),
    )

    report = router.build(target_huc12=target_huc12)

    print("\nRouting QA report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

