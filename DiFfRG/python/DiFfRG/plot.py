import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as colors
import seaborn as sns
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from DiFfRG.utilities import globalize

# define a few cmaps
cmap1 = sns.color_palette("magma", as_cmap=True)
cmap2 = sns.color_palette("viridis", as_cmap=True)

# define a few palettes
palette1 = sns.color_palette("deep")
palette2 = sns.cubehelix_palette(start=0, rot=-0.5, dark=0.1, light=0.9)
palette3 = sns.color_palette("Set2")

sns.set_style("ticks")  # darkgrid, white grid, dark, white and ticks
# sns.despine()
sns.set_theme(style="ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})

plt.rcParams.update({"text.usetex": True, "axes.labelpad": 8.0})  # use latex rendering

# plt.rc("axes", titlesize=14)  # fontsize of the axes title
# plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
# plt.rc("xtick", labelsize=13)  # fontsize of the tick labels
# plt.rc("ytick", labelsize=13)  # fontsize of the tick labels
# plt.rc("legend", fontsize=13)  # legend fontsize
# plt.rc("font", size=13)  # controls default text sizes


def clear_fig():
    plt.cla()
    plt.clf()
    plt.close()

def plot_1D(
    datasets,
    xlabel=None,
    ylabel=None,
    ylim=None,
    xlim=None,
    log_x=False,
    log_y=False,
    file=None,
    modifiers=None,
    legend_loc="lower left",
    grid=False,
    force_legend=False,
    figsize=(6,4),
    axstyle="normal",
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
    for i, data in enumerate(datasets):
        # Ensure that the data has the right format
        if not "x" in data or not "y" in data:
            raise Exception("Data needs to have x and y keys!")
        if not "label" in data:
            data["label"] = None
        if not "color" in data:
            data["color"] = palette1[i]
        if not "scatter" in data:
            data["scatter"] = False
        if not "lw" in data:
            data["lw"] = 2.
        if not "linestyle" in data:
            data["linestyle"] = "-"
        if not "size" in data:
            data["size"] = 5.0
        if not "marker" in data:
            data["marker"] = "+"
        if not "yerr" in data:
            data["yerr"] = None
        if not "xerr" in data:
            data["xerr"] = None
        if not "alpha" in data:
            data["alpha"] = 1.0
        if not "zorder" in data:
            data["zorder"] = 2

        # Prepare the data
        sorted_idx = np.argsort(data["x"], axis=-1)
        data["x"] = data["x"][sorted_idx]
        data["y"] = data["y"][sorted_idx]

        # Plot the data
        if data["scatter"] == True:
            ax.errorbar(
                data["x"],
                data["y"],
                xerr=data["xerr"],
                yerr=data["yerr"],
                label=data["label"],
                color=data["color"],
                markersize=data["size"],
                marker=data["marker"],
                linestyle="None",
                alpha=data["alpha"],
                zorder=data["zorder"],
            )
        else:
            ax.plot(
                data["x"],
                data["y"],
                label=data["label"],
                color=data["color"],
                lw=data["lw"],
                linestyle=data["linestyle"],
                alpha=data["alpha"],
                zorder=data["zorder"],
            )

    # Set axis labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Set log scales, if desired
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Set axis limits, if desired
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set a grid, if desired
    if grid == True:
        ax.grid(color="grey", linestyle=":", linewidth=0.5, which="both")

    # Add a legend
    if force_legend or len(datasets) > 1:
        ax.legend(loc=legend_loc, borderpad=0.25, frameon=False)

    # Remove the margins
    ax.margins(0.0)

    ax.tick_params(
        axis="x",
        which="both",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=True,
        direction="in",
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        labelleft=True,
        right=True,
        labelright=False,
        direction="in",
    )

    if axstyle == "centered":
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    # Let the user add some things to the plot
    if modifiers != None and len(modifiers) > 0:
        for mod in modifiers:
            mod(ax)


    if file == None:
        plt.show()
    else:
        plt.savefig(file)
    clear_fig()

def scaled_triangulation(x, y, log_x=False, log_y=False):
    """Delaunay-triangulate (x, y) in the coordinates the plot actually shows.

    A Delaunay triangulation of the raw coordinates degenerates into sliver
    triangles whenever the two axes span very different ranges, which shows up
    as jagged, streaky structure in the plot. Rescaling both axes to unit range
    (in log space for logarithmic axes) first avoids this; the connectivity
    found there is then used with the original coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    u = np.log10(x) if log_x else x
    v = np.log10(y) if log_y else y
    scale_u = np.ptp(u)
    scale_v = np.ptp(v)
    if not np.isfinite(scale_u) or not np.isfinite(scale_v) or scale_u <= 0 or scale_v <= 0:
        return tri.Triangulation(x, y)

    triangles = tri.Triangulation(u / scale_u, v / scale_v).triangles
    return tri.Triangulation(x, y, triangles)

def plot_2D(
    data,
    cmap=cmap2,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    ylim=None,
    xlim=None,
    zlim=None,
    log_x=False,
    log_y=False,
    log_z=False,
    file=None,
    modifiers=None,
    grid=False,
    style="triplot",
    angle_3d=(30, 30),
    aspect=None,
    figsize=(6,4),
    axstyle="normal",
    levels=10,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)
    if not "x" in data or not "y" in data or not "z" in data:
        raise Exception("Data needs to have x, y and z keys!")
    
    if log_z and style != "3D" and style != "3d":
        norm = colors.LogNorm(vmin=np.min(data["z"]), vmax=np.max(data["z"]))
    else:
        norm = None
    
    if style == "triplot":
        # create the basic plot elements
        triangulation = scaled_triangulation(data["x"], data["y"], log_x, log_y)
        tripc = ax.tripcolor(triangulation, data["z"], shading="flat", cmap=cmap, norm = norm)
        # draw each triangle with its own color as edge, so that no background
        # shines through the seams between them in vector output
        tripc.set_edgecolor("face")
        tripc.set_linewidth(0.0)
        cbar = fig.colorbar(tripc)
        
        # Set a grid, if desired
        if grid == True:
            ax.triplot(triangulation, lw=0.05, color='white')
    elif style == "contour":
        x = np.array(data["x"])
        y = np.array(data["y"])
        z = np.array(data["z"])

        # Create grid values first.
        ngridx = 256
        ngridy = 256
        xi = np.linspace(x.min(), x.max(), ngridx)
        yi = np.linspace(y.min(), y.max(), ngridy)
        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = scaled_triangulation(x, y, log_x, log_y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        c = ax.contourf(xi, yi, zi, cmap=cmap, norm = norm, levels=levels)
        cbar = fig.colorbar(c, ax=ax)
        # Set a grid, if desired
        if grid == True:
            ax.triplot(triang, lw=0.05, color='white')
    elif style == "3D" or style == "3d":
        clear_fig()
        x = data["x"]
        y = data["y"]
        z = data["z"]
        # remove indices where x or y are beyond the limits
        if xlim:
            no_idx = np.less(x, xlim[0]) | np.greater(x, xlim[1])
            x = x[~no_idx]
            y = y[~no_idx]
            z = z[~no_idx]
        if ylim:
            no_idx = np.less(y, ylim[0]) | np.greater(y, ylim[1])
            x = x[~no_idx]
            y = y[~no_idx]
            z = z[~no_idx]
        if zlim:
            no_idx = np.less(z, zlim[0]) | np.greater(z, zlim[1])
            x = x[~no_idx]
            y = y[~no_idx]
            z = z[~no_idx]
        x = x if not log_x else np.log10(x)
        y = y if not log_y else np.log10(y)
        z = z if not log_z else np.log10(z)
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
        ax.plot_trisurf(x, y, z, cmap=cmap, norm = norm, edgecolor='none')
        ax.view_init(angle_3d[0], angle_3d[1])
        
        # turn off the grey background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # turn off the grid
        ax.grid(grid)
        
        if aspect: 
            ax.set_box_aspect(aspect)  # aspect ratio is 1:1:1 in data space
        
        # fix the axis ticks
        if log_x:
            # find the power of 10 closest to the minimum and maximum x values
            min_x = np.floor(np.log10(np.min(data["x"])))
            max_x = np.ceil(np.log10(np.max(data["x"])))
            ax.set_xticks([i for i in range(int(min_x)+1, int(max_x)+1)])
            ax.set_xticklabels([f"$10^{{{i}}}$" for i in range(int(min_x)+1, int(max_x)+1)])
        if log_y:
            # find the power of 10 closest to the minimum and maximum y values
            min_y = np.floor(np.log10(np.min(data["y"])))
            max_y = np.ceil(np.log10(np.max(data["y"])))
            ax.set_yticks([i for i in range(int(min_y)+1, int(max_y)+1)])
            ax.set_yticklabels([f"$10^{{{i}}}$" for i in range(int(min_y)+1, int(max_y)+1)])
        if log_z:
            # find the power of 10 closest to the minimum and maximum z values
            min_z = np.floor(np.log10(np.min(data["z"])))
            max_z = np.ceil(np.log10(np.max(data["z"])))
            ax.set_zticks([i for i in range(int(min_z), int(max_z)+1)])
            ax.set_zticklabels([f"$10^{{{i}}}$" for i in range(int(min_z), int(max_z)+1)])
        ax.xaxis.labelpad=8
        ax.yaxis.labelpad=8
        ax.zaxis.labelpad=4
        ax.zaxis.set_rotate_label(False) 
        ax.dist = 8
        if zlabel:
            ax.set_zlabel(zlabel, rotation=90)
    else:
        raise Exception("Unknown style!")

    # Set axis labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel and (style != "3D" and style != "3d"):
        cbar.ax.set_title(zlabel, loc="left", pad=12)

    # Set log scales, if desired
    if log_x and style != "3D" and style != "3d":
        ax.set_xscale("log")
    if log_y and style != "3D" and style != "3d": 
        ax.set_yscale("log")

    # Set axis limits, if desired
    if xlim and ((style != "3D" and style != "3d") or not log_x):
        ax.set_xlim(xlim)
    elif xlim:
        ax.set_xlim(np.log10(xlim))
    if ylim and ((style != "3D" and style != "3d") or not log_y):
        ax.set_ylim(ylim)
    elif ylim:
        ax.set_ylim(np.log10(ylim))
    if zlim and ((style != "3D" and style != "3d") or not log_z):
        ax.set_zlim(zlim)
    elif zlim:
        ax.set_zlim(np.log10(zlim))

    # Remove the margins
    ax.margins(0.0)

    if axstyle == "centered":
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.plot(1, 0.475, ">k", transform=ax.transAxes, clip_on=False)
        ax.plot(0.5, 1, "^k", transform=ax.transAxes, clip_on=False)
        ax.xaxis.set_label_coords(1.0, 0.46),
        ax.yaxis.set_label_coords(0.485, 1.0) 
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_axisbelow(False)
    else: 
        ax.tick_params(
            axis="x",
            which="both",
            top=True,
            labeltop=False,
            bottom=True,
            labelbottom=True,
            direction="in",
            zorder=10000
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            labelleft=True,
            right=True,
            labelright=False,
            direction="in",
            zorder=10000
        )

    # Let the user add some things to the plot
    if modifiers != None and len(modifiers) > 0:
        for mod in modifiers:
            mod(ax)

    if file == None:
        plt.show()
    else:
        plt.savefig(file, dpi=150)
    clear_fig()
