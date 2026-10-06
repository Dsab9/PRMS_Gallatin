r"""Build, route, and plot the Upper Gallatin PRMS domain.

This is the top-level orchestration script for the NHDPlus HR workflow. It
calls source preparation, source-agnostic routing, and Matplotlib QA plotting
from separate modules:

    NHD_PlusHR.py
        Downloads or locally loads NHDPlus HR data and writes standardized
        prepared layers.

    Routing.py
        Reads prepared source layers and builds the dendritic model network,
        routed parent HRUs, and routing QA outputs.

    Plotting.py
        Reads model outputs and writes Matplotlib routing QA figures.

Keep this script intentionally small. It controls which stages run and what
source layers are requested, while the implementation remains in the relevant
module.

Workflow:
    1. Optionally download selected NHDPlus HR source layers.
    2. Load and prepare standardized source layers.
    3. Build the source-agnostic dendritic routing domain.
    4. Write routing map and routing-summary figures.

Typical recovery workflow:
    If flowlines were already downloaded but catchments are imported from Arc
    Pro, set:

        DOWNLOAD_LAYERS = ("waterbody",)
        LOAD_LAYERS = ("flowline", "catchment", "waterbody")

    and map the Arc Pro catchment layer with SOURCE_LAYER_MAP.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJ configuration
# ---------------------------------------------------------------------------
# This must occur before importing modules that import geopandas, pyproj,
# pyogrio, fiona, or pynhd. It overrides the system PostGIS PROJ path.

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

# ---------------------------------------------------------------------------
# Project-module imports
# ---------------------------------------------------------------------------
# These files must be in the same directory as this script.

from NHD_PlusHR import (
    LocalNHDPlusHRLoader,
    NHDPlusHRConfig,
    NHDPlusHRDownloader,
)
from Routing import (
    DendriticRouter,
    PreparedDomainLoader,
    RoutingConfig,
)
from Plotting import DomainPlotter

import geopandas as gpd
from fiona import listlayers
# =============================================================================
# User configuration
# =============================================================================

HUC12 = "100200080507"

OUTPUT_DIR = Path(
    r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
    r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
)

GPKG_NAME = "upper_gallatin_nhdhr.gpkg"

WORK_CRS = "EPSG:5070"

# Set this to a known prepared-flowline source ID if automatic outlet selection
# selects the wrong line. Keep None to select the largest drainage-area line
# intersecting the target HUC12.
OUTLET_SOURCE_ID: str | None = None

# ---------------------------------------------------------------------------
# Stage controls
# ---------------------------------------------------------------------------

# Set False once the selected source layers are already available locally.
RUN_DOWNLOAD = False

# Set True to create or refresh:
#   prepared_flowlines
#   prepared_catchments
#   prepared_waterbodies
#   prepared_huc12
RUN_PREPARE = True

# Set True to create or refresh:
#   model_segments
#   model_hrus
#   unrouted_catchments
#   model_waterbodies
RUN_ROUTING = True

# Set True to create:
#   domain_qa/routing_map.png
#   domain_qa/routing_summary.png
RUN_PLOTTING = True

# ---------------------------------------------------------------------------
# Layer selections
# ---------------------------------------------------------------------------

# Logical source layers that will be downloaded from NHDPlus HR.
#
# Valid names:
#   "flowline"
#   "catchment"
#   "waterbody"
#
# Example: Flowlines already downloaded; catchments imported manually:
# DOWNLOAD_LAYERS = ("waterbody",)
DOWNLOAD_LAYERS = (
    "flowline",
    "catchment",
    "waterbody",
)

# Logical source layers loaded locally and standardized into prepared layers.
#
# Routing requires both flowline and catchment.
# Waterbody is optional.
LOAD_LAYERS = (
    "flowline",
    "catchment",
    "waterbody",
)

# ---------------------------------------------------------------------------
# Source-layer mapping
# ---------------------------------------------------------------------------

# These are the physical layer names in the local source GeoPackage.
#
# Default downloader products:
#   merged_flowlines
#   merged_catchments
#   merged_waterbodies
#
# If you replace catchments in Arc Pro, export the layer into the GeoPackage,
# give it a name such as "catchments_arcpro", and change only the catchment
# mapping below.
SOURCE_LAYER_MAP = {
    "flowline": "merged_flowlines",
    "catchment": "NHDcatchment",
    "waterbody": "NHDwaterbody",
}

# Default source is the same GeoPackage used by the downloader. To load Arc
# Pro-edited layers stored in another GeoPackage or a file geodatabase, replace
# this with that source path.
SOURCE_PATH: Path | None = None

# ---------------------------------------------------------------------------
# Downloader controls
# ---------------------------------------------------------------------------

# Existing raw HUC12 chunks are skipped when False. Use False for recovery runs.
OVERWRITE_RAW = False

# Delete raw per-HUC12 layers after a merged layer succeeds with no failures.
# Keep False until you verify merged outputs in Arc Pro.
PURGE_RAW_AFTER_MERGE = False

# HTTP/service retries after initial request failure.
MAX_RETRIES = 1
RETRY_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Plot controls
# ---------------------------------------------------------------------------

PLOT_ROUTING_ARROWS = True
PLOT_SEGMENT_LABELS = False
PLOT_HRU_LABELS = False
PLOT_DPI = 300


# =============================================================================
# Stage functions
# =============================================================================


def build_source_config() -> NHDPlusHRConfig:
    """Creates NHDPlus HR source-preparation configuration.

    Returns:
        NHDPlusHRConfig used by NHDPlusHRDownloader and LocalNHDPlusHRLoader.
    """
    return NHDPlusHRConfig(
        huc12=HUC12,
        output_dir=OUTPUT_DIR,
        gpkg_name=GPKG_NAME,
        download_layers=DOWNLOAD_LAYERS,
        load_layers=LOAD_LAYERS,
        source_path=SOURCE_PATH,
        source_layer_map=SOURCE_LAYER_MAP,
        max_retries=MAX_RETRIES,
        retry_seconds=RETRY_SECONDS,
    )


def run_download_stage(
    source_config: NHDPlusHRConfig,
) -> dict[str, int]:
    """Downloads selected NHDPlus HR layers.

    Args:
        source_config: NHDPlus HR data-source configuration.

    Returns:
        Download result dictionary.
    """
    print("\n" + "=" * 78)
    print("STAGE 1: DOWNLOAD NHDPLUS HR SOURCE DATA")
    print("=" * 78)

    downloader = NHDPlusHRDownloader(source_config)

    result = downloader.download(
        overwrite_raw=OVERWRITE_RAW,
        purge_raw_after_merge=PURGE_RAW_AFTER_MERGE,
    )

    print("\nDownload result:")
    print(json.dumps(result, indent=2))

    if result.get("failed_requests", 0) > 0:
        print(
            "\nWARNING: One or more source requests failed. Review:\n"
            f"  {source_config.failure_log_path}\n"
            "The remaining successful chunks and merged data were retained."
        )

    return result


def run_prepare_stage(
    source_config: NHDPlusHRConfig,
) -> dict[str, int]:
    """Loads source layers and writes standardized prepared-domain layers.

    Args:
        source_config: NHDPlus HR data-source configuration.

    Returns:
        Dictionary containing feature counts by prepared layer.
    """
    print("\n" + "=" * 78)
    print("STAGE 2: PREPARE STANDARDIZED ROUTING INPUTS")
    print("=" * 78)

    loader = LocalNHDPlusHRLoader(source_config)

    prepared = loader.load_and_prepare()

    result = {
        layer_name: int(len(gdf))
        for layer_name, gdf in prepared.items()
    }

    print("\nPrepared-layer feature counts:")
    print(json.dumps(result, indent=2))

    return result


def run_routing_stage(
    source_config: NHDPlusHRConfig,
) -> dict[str, object]:
    """Builds the source-agnostic dendritic PRMS routing domain.

    Args:
        source_config: NHDPlus HR data-source configuration.

    Returns:
        Routing QA report.
    """
    print("\n" + "=" * 78)
    print("STAGE 3: BUILD DENDRITIC ROUTING DOMAIN")
    print("=" * 78)

    routing_config = RoutingConfig(
        gpkg_path=source_config.gpkg_path,
        target_huc12=HUC12,
        outlet_source_id=OUTLET_SOURCE_ID,
        work_crs=WORK_CRS,
        min_hru_area_m2=0.0,
        flowline_layer="prepared_flowlines",
        catchment_layer="prepared_catchments",
        waterbody_layer="prepared_waterbodies",
        huc12_layer="prepared_huc12",
    )

    prepared_loader = PreparedDomainLoader(routing_config)

    flowlines, catchments, waterbodies, huc12s = prepared_loader.load()

    router = DendriticRouter(
        config=routing_config,
        flowlines=flowlines,
        catchments=catchments,
        waterbodies=waterbodies,
        huc12s=huc12s,
    )

    report = router.build()

    print("\nRouting QA report:")
    print(json.dumps(report, indent=2))

    return report


def run_plotting_stage(
    source_config: NHDPlusHRConfig,
) -> dict[str, Path]:
    """Creates Matplotlib QA figures from model routing outputs.

    Args:
        source_config: NHDPlus HR data-source configuration.

    Returns:
        Dictionary of saved output image paths.
    """
    print("\n" + "=" * 78)
    print("STAGE 4: CREATE ROUTING QA MAPS")
    print("=" * 78)

    qa_dir = source_config.output_dir / "domain_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    plotter = DomainPlotter(source_config.gpkg_path)

    routing_map = plotter.save_routing_map(
        output_path=qa_dir / "routing_map.png",
        dpi=PLOT_DPI,
        show_arrows=PLOT_ROUTING_ARROWS,
        label_segments=PLOT_SEGMENT_LABELS,
        label_hrus=PLOT_HRU_LABELS,
    )

    routing_summary = plotter.save_routing_summary(
        output_path=qa_dir / "routing_summary.png",
        dpi=PLOT_DPI,
        show_arrows=PLOT_ROUTING_ARROWS,
    )

    outputs = {
        "routing_map": routing_map,
        "routing_summary": routing_summary,
    }

    print("\nSaved map outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    return outputs


def print_geopackage_inventory(
    gpkg_path: Path,
) -> None:
    """Prints GeoPackage layer names, feature counts, and geometry types.

    Args:
        gpkg_path: GeoPackage to inspect.
    """
    print("\n" + "=" * 78)
    print("GEOPACKAGE LAYER INVENTORY")
    print("=" * 78)
    print(f"GeoPackage: {gpkg_path}")

    if not gpkg_path.exists():
        print("GeoPackage does not exist; no layer inventory is available.")
        return

    layer_names = listlayers(gpkg_path)

    if not layer_names:
        print("No layers found in the GeoPackage.")
        return

    print(f"\n{'Layer name':<35} {'Features':>12}   Geometry type")
    print("-" * 78)

    for layer_name in sorted(layer_names):
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer_name)

            geometry_types = (
                ", ".join(
                    sorted(
                        str(value)
                        for value in gdf.geom_type.dropna().unique()
                    )
                )
                if not gdf.empty
                else "Empty"
            )

            print(
                f"{layer_name:<35} "
                f"{len(gdf):>12,}   "
                f"{geometry_types}"
            )

        except Exception as error:  # noqa: BLE001
            print(
                f"{layer_name:<35} "
                f"{'ERROR':>12}   "
                f"{type(error).__name__}: {error}"
            )

    print("-" * 78)
# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Runs enabled Upper Gallatin domain-building workflow stages."""
    print("=" * 78)
    print("UPPER GALLATIN PRMS DOMAIN BUILDER")
    print("=" * 78)

    print(f"Target HUC12: {HUC12}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GeoPackage: {OUTPUT_DIR / GPKG_NAME}")
    print(f"Download layers: {DOWNLOAD_LAYERS}")
    print(f"Load layers: {LOAD_LAYERS}")

    source_config = build_source_config()

    if RUN_DOWNLOAD:
        run_download_stage(source_config)

    if RUN_PREPARE:
        run_prepare_stage(source_config)

    if RUN_ROUTING:
        run_routing_stage(source_config)

    if RUN_PLOTTING:
        run_plotting_stage(source_config)

    print_geopackage_inventory(source_config.gpkg_path)

    print("\n" + "=" * 78)
    print("UPPER GALLATIN DOMAIN BUILD COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()