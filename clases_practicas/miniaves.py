import contextily as cx
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import KDEpy
from matplotlib import colorbar
from seaborn.axisgrid import FacetGrid, Grid
import numpy as np

# from aves.visualization.maps.utils import geographical_scale, north_arrow
def north_arrow(
    ax,
    x=0.98,
    y=0.06,
    arrow_length=0.04,
    text="N",
    font_name=None,
    font_size=None,
    color="#000000",
    arrow_color="#000000",
    arrow_width=3,
    arrow_headwidth=7,
):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(
            facecolor=arrow_color, width=arrow_width, headwidth=arrow_headwidth
        ),
        ha="center",
        va="center",
        fontsize=font_size,
        fontname=font_name,
        color=color,
        xycoords=ax.transAxes,
    )

def geographical_scale(ax, location="lower left"):
    ax.add_artist(ScaleBar(1, location="lower left"))


##### 

def bubble_map(
    ax,
    geodf: gpd.GeoDataFrame,
    size,
    scale=1,
    palette=None,
    color=None,
    add_legend=True,
    edgecolor="white",
    alpha=1.0,
    label=None,
    **kwargs
):
    marker = "o"

    if size is not None:
        if type(size) == str:
            marker_size = geodf[size]
        else:
            marker_size = float(size)
    else:
        marker_size = 1

    return geodf.plot(
        ax=ax,
        marker=marker,
        markersize=marker_size * scale,
        edgecolor=edgecolor,
        alpha=alpha,
        facecolor=color,
        legend=add_legend,
        label=label,
        **kwargs
    )


def dot_map(
    ax,
    geodf: gpd.GeoDataFrame,
    size=10,
    palette=None,
    add_legend=True,
    label=None,
    **kwargs
):
    return bubble_map(
        ax,
        geodf,
        size=float(size),
        palette=palette,
        add_legend=add_legend,
        edgecolor="none",
        label=label,
        **kwargs
    )


# from aves.visualization.colors import colormap_from_palette
def colormap_from_palette(palette_name, n_colors=10):
    return colors.ListedColormap(sns.color_palette(palette_name, n_colors=n_colors))


class GeoFacetGrid(FacetGrid):
    def __init__(self, geodataframe: gpd.GeoDataFrame, *args, **kwargs):
        geocontext = kwargs.pop("context", None)

        if geocontext is None:
            geocontext = geodataframe

        self.geocontext = geocontext

        self.bounds = geocontext.total_bounds
        self.aspect = (self.bounds[2] - self.bounds[0]) / (
            self.bounds[3] - self.bounds[1]
        )

        kwargs["aspect"] = self.aspect
        kwargs["xlim"] = (self.bounds[0], self.bounds[2])
        kwargs["ylim"] = (self.bounds[1], self.bounds[3])

        super().__init__(geodataframe, *args, **kwargs)

        for ax in self.axes.flatten():
            if kwargs.get("remove_axes", True):
                ax.set_axis_off()

            aspect = kwargs.get("aspect", "auto")
            # code from geopandas
            if aspect == "auto":
                if geodataframe.crs and geodataframe.crs.is_geographic:
                    bounds = geodataframe.total_bounds
                    y_coord = np.mean([bounds[1], bounds[3]])
                    ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
                    # formula ported from R package sp
                    # https://github.com/edzer/sp/blob/master/R/mapasp.R
                else:
                    ax.set_aspect("equal")
            elif aspect is not None:
                ax.set_aspect(aspect)

        self.zorder = 0

    def add_layer(self, func_or_data, *args, **kwargs):
        if isinstance(func_or_data, gpd.GeoDataFrame):
            # a direct geography
            for ax in self.axes.flatten():
                func_or_data.plot(*args, ax=ax, zorder=self.zorder, **kwargs)
        else:
            plot = lambda *a, **kw: func_or_data(
                plt.gca(), kw.pop("data"), *a, zorder=self.zorder, **kw
            )
            self.map_dataframe(plot, *args, **kwargs)

        self.zorder += 1

    def add_basemap(
        self, file_path, interpolation="hanning", reset_extent=False, **kwargs
    ):
        for ax in self.axes.flatten():
            cx.add_basemap(
                ax,
                crs=self.geocontext.crs.to_string(),
                source=file_path,
                interpolation=interpolation,
                zorder=self.zorder,
                reset_extent=reset_extent,
                **kwargs
            )

            if not reset_extent:
                ax.set_xlim((self.bounds[0], self.bounds[2]))
                ax.set_ylim((self.bounds[1], self.bounds[3]))

        self.zorder += 1

    def add_map_elements(
        self,
        all_axes=False,
        scale=True,
        arrow=True,
        scale_args={},
        arrow_args={},
    ):
        for ax in self.axes.flatten():
            if arrow:
                north_arrow(ax, **arrow_args)

            if scale:
                geographical_scale(ax, **scale_args)
            if not all_axes:
                break

    def add_global_colorbar(self, palette, k, title=None, title_args={}, **kwargs):

        orientation = kwargs.get("orientation", "horizontal")
        if orientation == "horizontal":
            cax = self.fig.add_axes([0.25, -0.012, 0.5, 0.01])
        elif orientation == "vertical":
            cax = self.fig.add_axes([1.01, 0.25, 0.01, 0.5])
        else:
            raise ValueError("unsupported colorbar orientation")

        if title:
            cax.set_title(title, **title_args)

        cb = colorbar.ColorbarBase(
            cax, cmap=colormap_from_palette(palette, n_colors=k), **kwargs
        )

        cax.set_axis_off()

        return cax, cb

    def set_title(self, title, **kwargs):
        self.fig.suptitle(title, **kwargs)


class GeoAttributeGrid(Grid):
    def __init__(
        self,
        geodataframe: gpd.GeoDataFrame,
        *,
        context: gpd.GeoDataFrame = None,
        vars=None,
        height=2.5,
        layout_pad=0.5,
        col_wrap=4,
        despine=True,
        remove_axes=True,
        set_limits=True,
        equal_aspect=True
    ):

        super().__init__()

        if vars is not None:
            vars = list(vars)
        else:
            vars = list(geodataframe.drop("geometry", axis=1).columns)

        if not vars:
            raise ValueError("No variables found for grid.")

        self.vars = vars

        if context is None:
            context = geodataframe

        self.geocontext = context

        self.bounds = self.geocontext.total_bounds
        self.aspect = (self.bounds[2] - self.bounds[0]) / (
            self.bounds[3] - self.bounds[1]
        )

        n_variables = len(vars)

        n_columns = min(col_wrap, len(vars))
        n_rows = n_variables // n_columns
        if n_rows * n_columns < n_variables:
            n_rows += 1

        with mpl.rc_context({"figure.autolayout": False}):
            fig, axes = plt.subplots(
                n_rows,
                n_columns,
                figsize=(n_columns * height * self.aspect, n_rows * height),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

        flattened = axes.flatten()

        if set_limits:
            for ax in flattened:
                ax.set_xlim((self.bounds[0], self.bounds[2]))
                ax.set_ylim((self.bounds[1], self.bounds[3]))

        if remove_axes:
            for ax in flattened:
                ax.set_axis_off()
        else:
            # deactivate only unneeded axes
            for i in range(n_variables, len(axes)):
                flattened[i].set_axis_off()

        if equal_aspect:
            for ax in flattened:
                ax.set_aspect("equal")

        self._figure = fig
        self.axes = axes
        self.data = geodataframe

        # Label the axes
        self._add_axis_labels()

        self._legend_data = {}

        # Make the plot look nice
        self._tight_layout_rect = [0.01, 0.01, 0.99, 0.99]
        self._tight_layout_pad = layout_pad
        self._despine = despine
        if despine:
            sns.despine(fig=fig)
        self.tight_layout(pad=layout_pad)

    def _add_axis_labels(self):
        for ax, label in zip(self.axes.flatten(), self.vars):
            ax.set_title(label)

    def add_layer(self, func_or_data, *args, **kwargs):
        if isinstance(func_or_data, gpd.GeoDataFrame):
            # a direct geography
            for ax, col in zip(self.axes.flatten(), self.vars):
                func_or_data.plot(*args, ax=ax, **kwargs)
        else:
            for ax, col in zip(self.axes.flatten(), self.vars):
                func_or_data(ax, self.data, col, *args, **kwargs)


def figure_from_geodataframe(
    geodf,
    height=5,
    bbox=None,
    remove_axes=True,
    set_limits=True,
    basemap=None,
    basemap_interpolation="hanning",
):
    if bbox is None:
        bbox = geodf.total_bounds

    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    fig, ax = plt.subplots(figsize=(height * aspect, height))

    if set_limits:
        ax.set_xlim([bbox[0], bbox[2]])
        ax.set_ylim([bbox[1], bbox[3]])

        # code from geopandas
        if geodf.crs and geodf.crs.is_geographic:
            bounds = geodf.total_bounds
            y_coord = np.mean([bounds[1], bounds[3]])
            ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            ax.set_aspect("equal")

    if remove_axes:
        ax.set_axis_off()

    if basemap is not None:
        cx.add_basemap(
            ax,
            crs=geodf.crs.to_string(),
            source=basemap,
            interpolation=basemap_interpolation,
            zorder=0,
        )

    return fig, ax


def small_multiples_from_geodataframe(
    geodf,
    n_variables,
    height=5,
    col_wrap=5,
    bbox=None,
    sharex=True,
    sharey=True,
    remove_axes=True,
    set_limits=True,
    flatten_axes=True,
    aspect="auto",
    basemap=None,
    basemap_interpolation="hanning",
):
    if n_variables <= 1:
        return figure_from_geodataframe(
            geodf,
            height=height,
            bbox=bbox,
            remove_axes=remove_axes,
            set_limits=set_limits,
            basemap=basemap,
            basemap_interpolation=basemap_interpolation,
        )

    if bbox is None:
        bbox = geodf.total_bounds

    # code from geopandas
    if aspect == "auto":
        if geodf.crs and geodf.crs.is_geographic:
            y_coord = np.mean([bbox[1], bbox[3]])
            aspect_ratio = 1 / np.cos(y_coord * np.pi / 180)
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            aspect_ratio = 1
    else:
        aspect_ratio = aspect

    n_columns = min(col_wrap, n_variables)
    n_rows = n_variables // n_columns
    if n_rows * n_columns < n_variables:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(n_columns * height / aspect_ratio, n_rows * height),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    flattened = axes.flatten()

    if set_limits:
        for ax in flattened:
            ax.set_xlim([bbox[0], bbox[2]])
            ax.set_ylim([bbox[1], bbox[3]])

    for ax in flattened:
        ax.set_aspect(aspect_ratio)

    if remove_axes:
        for ax in flattened:
            ax.set_axis_off()
    else:
        # deactivate only unneeded axes
        for i in range(n_variables, len(axes)):
            flattened[i].set_axis_off()

    if basemap is not None:
        for ax in flattened:
            cx.add_basemap(
                ax,
                crs=geodf.crs.to_string(),
                source=basemap,
                interpolation=basemap_interpolation,
                zorder=0,
            )

    if flatten_axes:
        return fig, flattened

    return fig, axes



# aves.visualization.maps.heatmap
def heat_map(
    ax,
    geodf,
    weight=None,
    low_threshold=0,
    max_threshold=1.0,
    n_levels=5,
    alpha=1.0,
    palette="magma",
    kernel="cosine",
    norm=2,
    bandwidth=1e-2,
    grid_points=2**6,
    return_heat=False,
    cbar_label=None,
    cbar_width=2.4,
    cbar_height=0.15,
    cbar_location="upper left",
    cbar_orientation="horizontal",
    cbar_pad=0.05,
    cbar_bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
    cbar_bbox_transform=None,
    legend_type="none",
    **kwargs
):
    heat = kde_from_points(
        geodf,
        kernel=kernel,
        norm=norm,
        bandwidth=bandwidth,
        grid_points=grid_points,
        weight_column=weight,
    )

    norm_heat = heat[2] / heat[2].max()

    if type(palette) == str:
        cmap = colormap_from_palette(palette, n_colors=n_levels)
    else:
        cmap = palette

    levels = np.linspace(low_threshold, max_threshold, n_levels)

    # TODO: this should be factorized into an utility function
    if legend_type == "colorbar":
        # add_ranged_color_legend(ax)
        if cbar_bbox_transform is None:
            cbar_bbox_transform = ax.transAxes

        if cbar_location != "out":
            cbar_ax = inset_axes(
                ax,
                width=cbar_width,
                height=cbar_height,
                loc=cbar_location,
                bbox_to_anchor=cbar_bbox_to_anchor,
                bbox_transform=cbar_bbox_transform,
                borderpad=0,
            )
        else:
            divider = make_axes_locatable(ax)
            cbar_main = divider.append_axes(
                "bottom" if cbar_orientation == "horizontal" else "right",
                size=cbar_height if cbar_orientation == "horizontal" else cbar_width,
                pad=cbar_pad,
            )
            cbar_main.set_axis_off()
            cbar_ax = inset_axes(
                cbar_main,
                width=cbar_width,
                height=cbar_height,
                loc="center",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=cbar_main.transAxes,
                borderpad=0,
            )

        color_legend(cbar_ax, cmap, levels, orientation=cbar_orientation)
    else:
        cbar_ax = None

    if not return_heat:
        return (
            ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap),
            cbar_ax,
        )
    else:
        return (
            ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap),
            cbar_ax,
            heat,
        )


# aves.features.geo
def kde_from_points(
    geodf,
    kernel="gaussian",
    norm=2,
    bandwidth=1e-2,
    grid_points=2 ** 9,
    weight_column=None,
):
    # La variable grid_points define la cantidad de puntos en el espacio en el que se estimará la densidad
    # hacemos una lista con las coordenadas de los viajes
    point_coords = np.vstack([geodf.geometry.x, geodf.geometry.y]).T
    # instanciamos la Fast-Fourier Transform Kernel Density Estimation
    kde = KDEpy.FFTKDE(bw=bandwidth, norm=norm, kernel=kernel)
    weights = None if weight_column is None else geodf[weight_column].values
    grid, points = kde.fit(point_coords, weights=weights).evaluate(grid_points)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T
    return x, y, z


# aves.visualizacion.colors

def color_legend(
    ax,
    color_list,
    bins=None,
    sizes=None,
    orientation="horizontal",
    remove_axes=False,
    bin_spacing="proportional",
    tick_labels=None,
):
    if bins is None:
        if type(color_list) == colors.ListedColormap:
            N = color_list.N
        else:
            N = len(color_list)
        bins = np.array(range(N))

    if sizes is not None:
        bar_width = bins[1:] - bins[0:-1]
        if orientation == "horizontal":
            ax.bar(
                bins[:-1],
                sizes,
                width=bar_width,
                align="edge",
                color=color_list,
                edgecolor=color_list,
            )
            ax.set_xticks(bins, labels=tick_labels)
        else:
            ax.barh(
                bins[:-1],
                sizes,
                height=bar_width,
                align="edge",
                color=color_list,
                edgecolor=color_list,
            )
            ax.set_yticks(bins, labels=tick_labels)
    else:
        cbar_norm = colors.BoundaryNorm(bins, len(bins) - 1)
        if type(color_list) == colors.ListedColormap:
            cmap = color_list
        else:
            cmap = colors.ListedColormap(color_list)
        cb = colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=cbar_norm,
            ticks=bins,
            spacing=bin_spacing,
            orientation=orientation,
        )
        if tick_labels:
            cb.set_ticklabels(tick_labels)

    sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)

    if remove_axes:
        ax.set_axis_off()

    return ax


#### NEW

def figure_grid_from_geodataframe(
    geodf,
    height=5,
    nrows=1,
    ncols=1,
    bbox=None,
    remove_axes=True,
    set_limits=True,
    basemap=None,
    basemap_interpolation="hanning",
):
    if bbox is None:
        bbox = geodf.total_bounds

    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    fig, axes = plt.subplots(ncols=ncols,nrows=nrows,figsize=(ncols * height * aspect, nrows*height))
    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]
    
    for ax in axes:
        if set_limits:
            ax.set_xlim([bbox[0], bbox[2]])
            ax.set_ylim([bbox[1], bbox[3]])
    
            # code from geopandas
            if geodf.crs and geodf.crs.is_geographic:
                bounds = geodf.total_bounds
                y_coord = np.mean([bounds[1], bounds[3]])
                ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
                # formula ported from R package sp
                # https://github.com/edzer/sp/blob/master/R/mapasp.R
            else:
                ax.set_aspect("equal")
    
        if remove_axes:
            ax.set_axis_off()
    
        if basemap is not None:
            cx.add_basemap(
                ax,
                crs=geodf.crs.to_string(),
                source=basemap,
                interpolation=basemap_interpolation,
                zorder=0,
            )

    return fig, axes

