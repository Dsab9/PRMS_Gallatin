"""Plot PRMS domain catchments, stream segments, and routing relationships.

The DomainPlotter class reads model products written by NHD_PlusHR.py and
creates Matplotlib-based QA maps for reviewing:

    - Parent HRUs / catchments.
    - Routed flowline segments.
    - Segment-to-segment downstream routing.
    - Unrouted catchments.
    - Waterbody intersections.
    - Outlet location.
    - Optional segment and HRU ID labels.
"""

from __future__ import annotations

import os
import sys
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class DomainPlotter:
    """Plots a PRMS domain from saved GeoPackage layers.

    The class expects the following output layers created by the domain builder:

        model_hrus
        model_segments

    The following layers are optional:

        unrouted_catchments
        model_waterbodies
        download_huc12

    Args:
        gpkg_path: Path to the GeoPackage containing model domain products.
        hru_layer: GeoPackage layer containing routed HRU polygons.
        segment_layer: GeoPackage layer containing model stream segments.
        unrouted_layer: Optional layer containing unrouted catchments.
        waterbody_layer: Optional layer containing waterbodies.
        huc12_layer: Optional layer containing the source HUC12 domain.
    """

    DEFAULT_HRU_LAYER = "model_hrus"
    DEFAULT_SEGMENT_LAYER = "model_segments"
    DEFAULT_UNROUTED_LAYER = "unrouted_catchments"
    DEFAULT_WATERBODY_LAYER = "model_waterbodies"
    DEFAULT_HUC12_LAYER = "download_huc12"

    def __init__(
        self,
        gpkg_path: str | Path,
        hru_layer: str = DEFAULT_HRU_LAYER,
        segment_layer: str = DEFAULT_SEGMENT_LAYER,
        unrouted_layer: str = DEFAULT_UNROUTED_LAYER,
        waterbody_layer: str = DEFAULT_WATERBODY_LAYER,
        huc12_layer: str = DEFAULT_HUC12_LAYER,
    ) -> None:
        """Initializes the plotter and loads available GeoPackage layers.

        Args:
            gpkg_path: Path to the GeoPackage.
            hru_layer: Routed-HRU polygon layer name.
            segment_layer: Stream-segment line layer name.
            unrouted_layer: Optional unrouted-catchment polygon layer name.
            waterbody_layer: Optional waterbody polygon layer name.
            huc12_layer: Optional HUC12 polygon layer name.

        Raises:
            FileNotFoundError: If gpkg_path does not exist.
            ValueError: If required HRU or segment layers are not present.
        """
        self.gpkg_path = Path(gpkg_path)

        if not self.gpkg_path.exists():
            raise FileNotFoundError(
                f"GeoPackage does not exist: {self.gpkg_path}"
            )

        self.available_layers = set(gpd.list_layers(self.gpkg_path)["name"])

        self.hrus = self._read_required_layer(hru_layer)
        self.segments = self._read_required_layer(segment_layer)

        self.unrouted = self._read_optional_layer(unrouted_layer)
        self.waterbodies = self._read_optional_layer(waterbody_layer)
        self.huc12s = self._read_optional_layer(huc12_layer)

        self._validate_required_fields()
        self._harmonize_crs()

        self.segment_lookup = self.segments.set_index("segment_id").copy()

    def plot_routing(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (12, 12),
        show_hrus: bool = True,
        show_unrouted: bool = True,
        show_waterbodies: bool = True,
        show_huc12_boundary: bool = False,
        show_arrows: bool = True,
        label_segments: bool = False,
        label_hrus: bool = False,
        hru_alpha: float = 0.45,
        segment_linewidth: float = 0.8,
        title: str = "PRMS Domain Catchments and Dendritic Routing",
        zoom_to: str = "domain",
    ) -> tuple[Figure, Axes]:
        """Plots routed HRUs, stream segments, and optional routing arrows.

        Args:
            ax: Existing Matplotlib axes. Creates new figure/axes if None.
            figsize: Figure size when creating a new figure.
            show_hrus: Plot routed HRU polygons.
            show_unrouted: Plot unrouted catchments, if available.
            show_waterbodies: Plot waterbody polygons, if available.
            show_huc12_boundary: Plot HUC12 boundaries, if available.
            show_arrows: Draw downstream arrows at segment midpoints.
            label_segments: Label stream segments with ``segment_id``.
            label_hrus: Label HRU polygons with ``hru_id``.
            hru_alpha: Fill transparency for routed HRUs.
            segment_linewidth: Width of stream lines.
            title: Plot title.
            zoom_to: Extent source. One of ``domain``, ``hrus``, or
                ``segments``.

        Returns:
            Tuple of the Matplotlib figure and axes.

        Raises:
            ValueError: If zoom_to is invalid.
        """
        figure, ax = self._get_figure_and_axes(ax, figsize)

        if show_huc12_boundary and self.huc12s is not None:
            self.huc12s.boundary.plot(
                ax=ax,
                color="0.45",
                linewidth=0.55,
                linestyle="--",
                zorder=1,
            )

        if show_unrouted and self.unrouted is not None:
            self.unrouted.plot(
                ax=ax,
                color="#d95f02",
                edgecolor="#7f2704",
                linewidth=0.25,
                alpha=0.35,
                zorder=2,
            )

        if show_hrus:
            self.hrus.plot(
                ax=ax,
                color="#c6dbef",
                edgecolor="#6baed6",
                linewidth=0.20,
                alpha=hru_alpha,
                zorder=3,
            )

        if show_waterbodies and self.waterbodies is not None:
            self.waterbodies.plot(
                ax=ax,
                color="#3182bd",
                edgecolor="#08519c",
                linewidth=0.35,
                alpha=0.65,
                zorder=4,
            )

        self.segments.plot(
            ax=ax,
            color="#252525",
            linewidth=segment_linewidth,
            zorder=5,
        )

        if show_arrows:
            self._plot_routing_arrows(
                ax=ax,
                color="#252525",
                linewidth=max(segment_linewidth * 0.55, 0.35),
            )

        self._plot_outlet(
            ax=ax,
            marker_size=75,
        )

        if label_segments:
            self._label_segments(ax=ax)

        if label_hrus:
            self._label_hrus(ax=ax)

        self._set_extent(
            ax=ax,
            zoom_to=zoom_to,
        )

        self._add_legend(
            ax=ax,
            show_hrus=show_hrus,
            show_unrouted=show_unrouted,
            show_waterbodies=show_waterbodies,
            show_arrows=show_arrows,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_axis_off()
        ax.set_aspect("equal")

        figure.tight_layout()
        return figure, ax

    def plot_routing_summary(
        self,
        figsize: tuple[float, float] = (16, 10),
        show_arrows: bool = True,
    ) -> tuple[Figure, np.ndarray]:
        """Creates a multi-panel domain QA figure.

        Panels include:
            1. Catchments and routing.
            2. HRUs colored by waterbody fraction.
            3. Segments colored by stream order, where available.
            4. Unrouted catchments and waterbody context.

        Args:
            figsize: Overall Matplotlib figure size.
            show_arrows: Draw downstream routing arrows in the routing panel.

        Returns:
            Tuple of the figure and 2-by-2 axes array.
        """
        figure, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=figsize,
            constrained_layout=True,
        )

        self.plot_routing(
            ax=axes[0, 0],
            show_arrows=show_arrows,
            title="A. HRUs and Dendritic Routing",
        )

        self._plot_hru_water_fraction(axes[0, 1])
        self._plot_stream_order(axes[1, 0])
        self._plot_unrouted_context(axes[1, 1])

        return figure, axes

    def save_routing_map(
        self,
        output_path: str | Path,
        dpi: int = 300,
        **plot_kwargs: Any,
    ) -> Path:
        """Creates and saves a routing QA map.

        Args:
            output_path: Output image path, such as PNG, PDF, or SVG.
            dpi: Raster output resolution.
            **plot_kwargs: Keyword arguments passed to ``plot_routing``.

        Returns:
            The verified output path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, _ = self.plot_routing(**plot_kwargs)

        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(figure)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write plot: {output_path}")

        return output_path

    def save_routing_summary(
        self,
        output_path: str | Path,
        dpi: int = 300,
        **plot_kwargs: Any,
    ) -> Path:
        """Creates and saves the multi-panel routing summary figure.

        Args:
            output_path: Output image path.
            dpi: Raster output resolution.
            **plot_kwargs: Keyword arguments passed to ``plot_routing_summary``.

        Returns:
            The verified output path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, _ = self.plot_routing_summary(**plot_kwargs)

        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(figure)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write plot: {output_path}")

        return output_path

    def _read_required_layer(self, layer_name: str) -> gpd.GeoDataFrame:
        """Reads a required GeoPackage layer.

        Args:
            layer_name: Required layer name.

        Returns:
            Loaded GeoDataFrame.

        Raises:
            ValueError: If layer_name is not available.
        """
        if layer_name not in self.available_layers:
            raise ValueError(
                f"Required layer '{layer_name}' is not in "
                f"{self.gpkg_path.name}. Available layers: "
                f"{sorted(self.available_layers)}"
            )

        return self._lower_columns(
            gpd.read_file(self.gpkg_path, layer=layer_name)
        )

    def _read_optional_layer(
        self,
        layer_name: str,
    ) -> gpd.GeoDataFrame | None:
        """Reads an optional GeoPackage layer when present.

        Args:
            layer_name: Optional layer name.

        Returns:
            Loaded GeoDataFrame, or None if the layer is absent.
        """
        if layer_name not in self.available_layers:
            return None

        return self._lower_columns(
            gpd.read_file(self.gpkg_path, layer=layer_name)
        )

    def _validate_required_fields(self) -> None:
        """Checks fields required for plotting routing topology.

        Raises:
            KeyError: If required fields are absent.
        """
        required_segment_fields = {
            "segment_id",
            "to_segment_id",
            "is_outlet",
        }

        required_hru_fields = {
            "hru_id",
            "segment_id",
        }

        missing_segment = required_segment_fields - set(self.segments.columns)
        missing_hru = required_hru_fields - set(self.hrus.columns)

        if missing_segment:
            raise KeyError(
                "model_segments is missing required field(s): "
                f"{sorted(missing_segment)}"
            )

        if missing_hru:
            raise KeyError(
                "model_hrus is missing required field(s): "
                f"{sorted(missing_hru)}"
            )

        duplicate_segments = self.segments["segment_id"].duplicated(
            keep=False
        )

        if duplicate_segments.any():
            duplicate_ids = sorted(
                self.segments.loc[
                    duplicate_segments,
                    "segment_id",
                ].unique()
            )

            print(
                "WARNING: model_segments contains duplicate segment_id values. "
                f"First duplicate IDs: {duplicate_ids[:20]}"
            )

    def _harmonize_crs(self) -> None:
        """Reprojects all optional layers to the HRU CRS."""
        target_crs = self.hrus.crs

        if target_crs is None:
            raise ValueError("The HRU layer does not have a defined CRS.")

        if self.segments.crs != target_crs:
            self.segments = self.segments.to_crs(target_crs)

        for attribute_name in ("unrouted", "waterbodies", "huc12s"):
            layer = getattr(self, attribute_name)

            if layer is not None and layer.crs != target_crs:
                setattr(
                    self,
                    attribute_name,
                    layer.to_crs(target_crs),
                )

    @staticmethod
    def _lower_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Returns a copy with lowercase non-geometry field names.

        Args:
            gdf: Input GeoDataFrame.

        Returns:
            GeoDataFrame with lowercase attribute field names.
        """
        output = gdf.copy()
        geometry_name = output.geometry.name

        output.columns = [
            column if column == geometry_name else column.lower()
            for column in output.columns
        ]

        return output

    @staticmethod
    def _get_figure_and_axes(
        ax: Axes | None,
        figsize: tuple[float, float],
    ) -> tuple[Figure, Axes]:
        """Returns existing axes or creates new Matplotlib figure and axes.

        Args:
            ax: Existing axes, or None.
            figsize: New figure size if ax is None.

        Returns:
            Figure and axes.
        """
        if ax is not None:
            return ax.figure, ax

        return plt.subplots(figsize=figsize)

    def _plot_routing_arrows(
            self,
            ax: Axes,
            color: str,
            linewidth: float,
    ) -> None:
        """Draws downstream-routing arrows between model-segment midpoints.

        The arrow starts near the source segment midpoint and points toward the
        receiving downstream segment midpoint. Missing, duplicate, or zero-length
        routing connections are skipped.

        Args:
            ax: Matplotlib axes.
            color: Arrow color.
            linewidth: Arrow line width.
        """
        midpoint_lookup = self._segment_midpoint_lookup()

        for source_row in self.segments.itertuples():
            source_segment_id = int(source_row.segment_id)
            target_segment_id = int(source_row.to_segment_id)

            if target_segment_id <= 0:
                continue

            source_point = midpoint_lookup.get(source_segment_id)
            target_point = midpoint_lookup.get(target_segment_id)

            if source_point is None or target_point is None:
                continue

            dx = target_point.x - source_point.x
            dy = target_point.y - source_point.y
            distance = float(np.hypot(dx, dy))

            if distance <= 0:
                continue

            start_fraction = 0.60
            end_fraction = 0.12

            start_x = source_point.x + start_fraction * dx
            start_y = source_point.y + start_fraction * dy
            end_x = source_point.x + (1.0 - end_fraction) * dx
            end_y = source_point.y + (1.0 - end_fraction) * dy

            ax.annotate(
                text="",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": linewidth,
                    "mutation_scale": 8,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "alpha": 0.75,
                },
                zorder=6,
            )

    def _plot_outlet(
        self,
        ax: Axes,
        marker_size: float,
    ) -> None:
        """Plots the routing outlet as a distinctive marker.

        Args:
            ax: Matplotlib axes.
            marker_size: Matplotlib scatter marker size.
        """
        outlet = self.segments[
            self.segments["is_outlet"] == 1
        ]

        if outlet.empty:
            return

        outlet_point = outlet.geometry.iloc[0].interpolate(
            0.5,
            normalized=True,
        )

        ax.scatter(
            [outlet_point.x],
            [outlet_point.y],
            s=marker_size,
            marker="*",
            color="#e31a1c",
            edgecolor="black",
            linewidth=0.45,
            zorder=8,
        )

    def _segment_midpoint_lookup(self) -> dict[int, Any]:
        """Builds a segment-ID to Shapely midpoint lookup.

        Where duplicate segment IDs exist, the first geometry is retained. Segment
        IDs should be unique; this behavior prevents the plotting routine from
        failing while you inspect the underlying layer.

        Returns:
            Dictionary mapping each segment_id to its Shapely midpoint.
        """
        midpoint_lookup: dict[int, Any] = {}

        for row in self.segments[
            ["segment_id", "geometry"]
        ].itertuples(index=False):
            segment_id = int(row.segment_id)

            if segment_id not in midpoint_lookup:
                midpoint_lookup[segment_id] = row.geometry.interpolate(
                    0.5,
                    normalized=True,
                )

        return midpoint_lookup


    def _label_segments(
        self,
        ax: Axes,
        fontsize: float = 6.5,
    ) -> None:
        """Labels model stream segments using segment_id.

        Args:
            ax: Matplotlib axes.
            fontsize: Segment-label font size.
        """
        for row in self.segments.itertuples():
            point = row.geometry.interpolate(
                0.5,
                normalized=True,
            )

            ax.text(
                point.x,
                point.y,
                str(int(row.segment_id)),
                fontsize=fontsize,
                color="#252525",
                ha="center",
                va="center",
                zorder=9,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                },
            )

    def _label_hrus(
        self,
        ax: Axes,
        fontsize: float = 6.0,
    ) -> None:
        """Labels routed HRU polygons using hru_id.

        Args:
            ax: Matplotlib axes.
            fontsize: HRU-label font size.
        """
        for row in self.hrus.itertuples():
            point = row.geometry.representative_point()

            ax.text(
                point.x,
                point.y,
                str(int(row.hru_id)),
                fontsize=fontsize,
                color="#08519c",
                ha="center",
                va="center",
                zorder=7,
            )

    def _segment_midpoints(self) -> gpd.GeoDataFrame:
        """Returns stream-segment midpoint geometries indexed by segment_id.

        Returns:
            GeoDataFrame with point geometry and segment_id index.
        """
        points = self.segments[
            ["segment_id", "geometry"]
        ].copy()

        points["geometry"] = points.geometry.interpolate(
            0.5,
            normalized=True,
        )

        return points.set_index("segment_id")

    def _set_extent(
        self,
        ax: Axes,
        zoom_to: str,
        padding_fraction: float = 0.03,
    ) -> None:
        """Sets map extent with small padding around selected geometry.

        Args:
            ax: Matplotlib axes.
            zoom_to: ``domain``, ``hrus``, or ``segments``.
            padding_fraction: Fractional x/y bounds padding.

        Raises:
            ValueError: If zoom_to is invalid.
        """
        if zoom_to == "hrus":
            bounds = self.hrus.total_bounds

        elif zoom_to == "segments":
            bounds = self.segments.total_bounds

        elif zoom_to == "domain":
            layers = [self.hrus, self.segments]

            if self.unrouted is not None and not self.unrouted.empty:
                layers.append(self.unrouted)

            all_bounds = np.vstack(
                [layer.total_bounds for layer in layers]
            )

            bounds = np.array(
                [
                    all_bounds[:, 0].min(),
                    all_bounds[:, 1].min(),
                    all_bounds[:, 2].max(),
                    all_bounds[:, 3].max(),
                ]
            )

        else:
            raise ValueError(
                "zoom_to must be one of: 'domain', 'hrus', or 'segments'."
            )

        minx, miny, maxx, maxy = bounds
        pad_x = max((maxx - minx) * padding_fraction, 1.0)
        pad_y = max((maxy - miny) * padding_fraction, 1.0)

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    def _add_legend(
        self,
        ax: Axes,
        show_hrus: bool,
        show_unrouted: bool,
        show_waterbodies: bool,
        show_arrows: bool,
    ) -> None:
        """Adds a custom domain-map legend.

        Args:
            ax: Matplotlib axes.
            show_hrus: Whether HRUs were plotted.
            show_unrouted: Whether unrouted catchments were plotted.
            show_waterbodies: Whether waterbodies were plotted.
            show_arrows: Whether routing arrows were plotted.
        """
        handles: list[Any] = []

        if show_hrus:
            handles.append(
                Patch(
                    facecolor="#c6dbef",
                    edgecolor="#6baed6",
                    alpha=0.65,
                    label="Routed parent HRU / catchment",
                )
            )

        if show_unrouted and self.unrouted is not None:
            handles.append(
                Patch(
                    facecolor="#d95f02",
                    edgecolor="#7f2704",
                    alpha=0.50,
                    label="Unrouted catchment",
                )
            )

        if show_waterbodies and self.waterbodies is not None:
            handles.append(
                Patch(
                    facecolor="#3182bd",
                    edgecolor="#08519c",
                    alpha=0.75,
                    label="Waterbody",
                )
            )

        handles.append(
            Line2D(
                [0],
                [0],
                color="#252525",
                linewidth=1.2,
                label="Model stream segment",
            )
        )

        if show_arrows:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#252525",
                    linewidth=0.8,
                    marker=">",
                    markersize=6,
                    markevery=[1],
                    label="Primary downstream route",
                )
            )

        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#e31a1c",
                markeredgecolor="black",
                markersize=11,
                label="Model outlet",
            )
        )

        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.92,
        )

    def _plot_hru_water_fraction(
        self,
        ax: Axes,
    ) -> None:
        """Plots HRUs colored by waterbody fractional area.

        Args:
            ax: Matplotlib axes.
        """
        if "waterbody_frac" not in self.hrus.columns:
            self.hrus.plot(
                ax=ax,
                color="#c6dbef",
                edgecolor="#6baed6",
                linewidth=0.2,
            )

            ax.set_title("B. HRU Waterbody Fraction Not Available")
            ax.set_axis_off()
            ax.set_aspect("equal")
            return

        self.hrus.plot(
            ax=ax,
            column="waterbody_frac",
            cmap="Blues",
            edgecolor="#6baed6",
            linewidth=0.2,
            legend=True,
            legend_kwds={
                "label": "Waterbody fraction of HRU",
                "shrink": 0.70,
            },
        )

        self.segments.plot(
            ax=ax,
            color="#252525",
            linewidth=0.6,
        )

        self._plot_outlet(ax=ax, marker_size=60)
        self._set_extent(ax=ax, zoom_to="domain")

        ax.set_title("B. HRU Waterbody Fraction")
        ax.set_axis_off()
        ax.set_aspect("equal")

    def _plot_stream_order(
        self,
        ax: Axes,
    ) -> None:
        """Plots segments colored by stream order when available.

        Args:
            ax: Matplotlib axes.
        """
        self.hrus.plot(
            ax=ax,
            color="#f7f7f7",
            edgecolor="#d9d9d9",
            linewidth=0.15,
        )

        stream_order_field = next(
            (
                field
                for field in ("streamorde", "streamcalc", "streamorder")
                if field in self.segments.columns
            ),
            None,
        )

        if stream_order_field is None:
            self.segments.plot(
                ax=ax,
                color="#252525",
                linewidth=0.8,
            )

            ax.set_title("C. Model Segments")
            ax.set_axis_off()
            ax.set_aspect("equal")
            self._set_extent(ax=ax, zoom_to="domain")
            return

        self.segments.plot(
            ax=ax,
            column=stream_order_field,
            cmap="viridis",
            linewidth=1.1,
            legend=True,
            legend_kwds={
                "label": "Stream order",
                "shrink": 0.70,
            },
        )

        self._plot_outlet(ax=ax, marker_size=60)
        self._set_extent(ax=ax, zoom_to="domain")

        ax.set_title("C. Model Segment Stream Order")
        ax.set_axis_off()
        ax.set_aspect("equal")

    def _plot_unrouted_context(
        self,
        ax: Axes,
    ) -> None:
        """Plots unrouted catchments with stream and waterbody context.

        Args:
            ax: Matplotlib axes.
        """
        self.hrus.plot(
            ax=ax,
            color="#e5f5e0",
            edgecolor="#a1d99b",
            linewidth=0.15,
            alpha=0.55,
        )

        if self.unrouted is not None and not self.unrouted.empty:
            if "reason" in self.unrouted.columns:
                colors = {
                    "waterbody": "#3182bd",
                    "other": "#e6550d",
                }

                for reason, subset in self.unrouted.groupby("reason"):
                    subset.plot(
                        ax=ax,
                        color=colors.get(reason, "#756bb1"),
                        edgecolor="#252525",
                        linewidth=0.25,
                        alpha=0.60,
                    )

            else:
                self.unrouted.plot(
                    ax=ax,
                    color="#e6550d",
                    edgecolor="#252525",
                    linewidth=0.25,
                    alpha=0.60,
                )

        if self.waterbodies is not None:
            self.waterbodies.plot(
                ax=ax,
                color="#9ecae1",
                edgecolor="#3182bd",
                linewidth=0.25,
                alpha=0.65,
            )

        self.segments.plot(
            ax=ax,
            color="#252525",
            linewidth=0.65,
        )

        self._plot_outlet(ax=ax, marker_size=60)
        self._set_extent(ax=ax, zoom_to="domain")

        ax.set_title("D. Unrouted Catchment Context")
        ax.set_axis_off()
        ax.set_aspect("equal")


def main() -> None:
    """Creates standard routing QA figures from the current GeoPackage."""
    gpkg_path = Path(
        r"C:\Users\CND367\Documents\Python_Scripts\PRMS"
        r"\PRMS_Gallatin\Domain_Builder\Domain_Files\PlusHR_UG.gdb"
    )

    qa_dir = gpkg_path.parent / "domain_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    plotter = DomainPlotter(gpkg_path)

    map_path = plotter.save_routing_map(
        output_path=qa_dir / "routing_map.png",
        dpi=300,
        show_arrows=True,
        label_segments=False,
        label_hrus=False,
    )

    summary_path = plotter.save_routing_summary(
        output_path=qa_dir / "routing_summary.png",
        dpi=300,
        show_arrows=True,
    )

    print(f"Saved routing map: {map_path}")
    print(f"Saved routing summary: {summary_path}")


if __name__ == "__main__":
    main()