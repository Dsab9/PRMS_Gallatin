"""Run the HRU_Stats workflow.

Steps (switch any of them off in RUN):

    1. create_parquet  copy the HRU feature class attributes to a parquet
    2. terrain         hru_elev, hru_slope, hru_aspect
    3. land_cover      cov_type, covden_*, interception, imperviousness
    4. join_layer      join the parquet back onto the HRU polygons

Each stats module receives ``parquet_path`` in its config and appends its
results to that parquet through ``utils.StatsAppender``.

Keep ``create_parquet`` off when rerunning a single stats module: creating
the parquet again discards everything appended so far.
"""

from __future__ import annotations

from pathlib import Path

from utils import HruLayerJoiner, HruParquetBuilder

DOMAIN = Path(
    r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
    r"\PRMS_Gallatin\Domain_Builder\Domain_Files"
)

HRU_GPKG = DOMAIN / "PlusHR_UG.gdb"
HRU_LAYER = "model_hrus_orphan_adopted"

STATS_DIR = DOMAIN / "HRU_Stats"
PARQUET_PATH = STATS_DIR / "hru_stats.parquet"

OUTPUT_LAYER = "model_hrus_prms"

RUN = {
    "create_parquet": False,
    "terrain": False,
    "land_cover": False,
    "soils" : True,
    "join_layer": False,
}


def run_terrain() -> None:
    """Appends terrain statistics to the parquet."""
    from terrain_stats import HruTerrainStats, TerrainStatsConfig

    HruTerrainStats(
        TerrainStatsConfig(
            hru_gpkg_path=HRU_GPKG,
            hru_layer=HRU_LAYER,
            parquet_path=PARQUET_PATH,
            # NHDPlus HR elev_cm.tif (integer centimeters), not HydroDEM.
            dem_path=DOMAIN / "elev_cm.tif",
            elevation_scale_to_m=0.01,
            elev_units=1,
        )
    ).run()


def run_land_cover() -> None:
    """Appends land-cover statistics to the parquet."""
    from landcover_stats import HruLandCoverStats, LandCoverConfig

    HruLandCoverStats(
        LandCoverConfig(
            hru_gpkg_path=HRU_GPKG,
            hru_layer=HRU_LAYER,
            parquet_path=PARQUET_PATH,
            nlcd_dir=DOMAIN / "NLCD",
            year=2025,
        )
    ).run()


def run_soils() -> None:
    """Appends soil statistics to the parquet."""
    from soils_stats import HruSoilStats, SoilsConfig

    HruSoilStats(
        SoilsConfig(
            hru_gpkg_path=HRU_GPKG,
            hru_layer=HRU_LAYER,
            parquet_path=PARQUET_PATH,
            gssurgo_gdb_path=DOMAIN / "gSSURGO_MT" / "gSSURGO_MT.gdb",
        )
    ).run()


def main() -> None:
    """Runs the enabled steps in order."""
    if RUN["create_parquet"]:
        HruParquetBuilder(
            hru_gpkg_path=HRU_GPKG,
            hru_layer=HRU_LAYER,
            parquet_path=PARQUET_PATH,
        ).create(overwrite=True)

    if RUN["terrain"]:
        run_terrain()

    if RUN["land_cover"]:
        run_land_cover()

    if RUN["soils"]:
        run_soils()

    if RUN["join_layer"]:
        HruLayerJoiner(
            parquet_path=PARQUET_PATH,
            hru_gpkg_path=HRU_GPKG,
            hru_layer=HRU_LAYER,
            output_layer=OUTPUT_LAYER,
            csv_path=STATS_DIR / "hru_parameters.csv",
        ).run()


if __name__ == "__main__":
    main()