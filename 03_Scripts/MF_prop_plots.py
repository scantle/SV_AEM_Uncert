import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mapclassify
from flopy.plot import PlotMapView
import flopy
import seaborn as sns
from tqdm import tqdm

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

# MODFLOW settings
#model_dir = Path('C:/Projects/SVIHM/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW/')
model_dir = Path('/Volumes/Macintosh HD/Users/leland/Documents/ModelRuns/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW/')
model_name = 'SVIHM'

xoff = 499977
yoff = 4571330

start_date = pd.to_datetime("1990-10-01")  # Start of SP 1

# Plot Style
sns.set_theme(style="whitegrid")

# Plot settings
plt.rcParams.update({
    "font.family": "Bahnschrift",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 300,
    "axes.unicode_minus": False # for mac
})

# Output
plt_dir = Path('05_Plots/MF_props/')
plt_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------------------------------- #
# Functions/Classes
# -------------------------------------------------------------------------------------------------------------------- #

def is_active(k, i, j, ibound):
    """Return True if cell (k,i,j) is active (IBOUND != 0)."""
    return ibound[k, i, j] != 0

# -------------------------------------------------------------------------------------------------------------------- #

def _nice_scale_length(max_len):
    """
    Choose a 'nice' scale bar length given some max_len (in map units).
    """
    if max_len <= 0:
        return 0
    exp = np.floor(np.log10(max_len))
    frac = max_len / 10**exp
    for n in [1, 2, 5]:
        if frac <= n:
            return n * 10**exp
    return 10 * 10**exp

# -------------------------------------------------------------------------------------------------------------------- #

def add_north_arrow(ax, x=0.08, y=0.98, arrow_len=0.08):
    """
    Add a simple north arrow in axes-fraction coordinates.
    """
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - arrow_len),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="k"),
        annotation_clip=False,
        clip_on=False,
        zorder=10,
    )


# -------------------------------------------------------------------------------------------------------------------- #

def add_scale_bar(ax, units="km"):
    """
    Add a scale bar to a map axis.

    Assumes x-units are meters; text is in km by default.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = x_max - x_min

    # pick a reasonable bar length ~1/5 of axis width
    target = width / 5.0
    bar_len = _nice_scale_length(target)
    if bar_len == 0:
        return

    # position scale bar near bottom-left
    x0 = x_min + 0.05 * width
    y0 = y_min + 0.05 * (y_max - y_min)

    ax.plot([x0, x0 + bar_len], [y0, y0], lw=2, color='k')
    if units == "km":
        label = f"{bar_len / 1000:g} km"
    else:
        label = f"{bar_len:g} {units}"

    ax.text(
        x0 + bar_len / 2.0, y0,
        label,
        ha='center', va='bottom',
        fontsize=8,
    )

# -------------------------------------------------------------------------------------------------------------------- #

def plot_property_panel(
    ax,
    prop3d,
    ibound,
    mg,
    layer,
    title,
    n_classes=5,
    cmap_name="viridis",
    grid_alpha=0.05,
    use_jenks=True,
):
    """
    Plot a single property for one layer on a given Axes.

    Parameters
    ----------
    ax : matplotlib Axes
    prop3d : (nlay, nrow, ncol) ndarray
        Property values.
    ibound : (nlay, nrow, ncol) ndarray
        IBOUND array.
    mg : flopy modelgrid
    layer : int
        Zero-based layer index to plot.
    title : str
        Panel title.
    n_classes : int
        Number of Jenks classes (if use_jenks=True).
    cmap_name : str
        Base colormap name.
    grid_alpha : float
        Alpha for grid lines.
    use_jenks : bool
        If True → Jenks natural breaks (discrete classes).
        If False → continuous colormap from min to max.
    """
    arr = prop3d[layer, :, :].copy()
    ib = ibound[layer, :, :]

    # Mask inactive cells as NaN
    arr = np.where(ib != 0, arr, np.nan)
    valid = np.isfinite(arr)

    if not valid.any():
        ax.set_title(f"{title}\n(no data)", fontweight="bold")
        ax.set_axis_off()
        return

    vals = arr[valid]
    prop_name = title.lower()

    pmv = PlotMapView(modelgrid=mg, ax=ax, layer=layer)

    # -------------------------------------------------------------------------
    # Classification + colormap
    # -------------------------------------------------------------------------
    base_cmap = plt.colormaps.get_cmap(cmap_name)

    if use_jenks:
        # Jenks natural breaks → discrete classes
        nj = mapclassify.NaturalBreaks(y=vals, k=n_classes)
        bounds = np.concatenate(([vals.min()], nj.bins))

        # Human-friendly labels
        labels = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if "specific storage" in prop_name:
                labels.append(f"{lo:.1e} – {hi:.1e}")
            else:
                labels.append(f"{lo:.2f} – {hi:.2f}")

        # Discrete colormap
        colors = base_cmap(np.linspace(0, 1, n_classes))
        listed = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, listed.N)

        mesh = pmv.plot_array(
            arr,
            cmap=listed,
            norm=norm,
            zorder=1,
        )

        # Colorbar with midpoints and our custom labels
        mids = 0.5 * (bounds[:-1] + bounds[1:])
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=listed)
        cbar = plt.colorbar(
            mappable,
            ax=ax,
            fraction=0.046,
            pad=0.02,
        )
        cbar.set_ticks(mids)
        cbar.set_ticklabels(labels)

    else:
        # Continuous min–max scale
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmin == vmax:
            # Avoid zero range
            vmin -= 0.5 * abs(vmin) if vmin != 0 else -1.0
            vmax += 0.5 * abs(vmax) if vmax != 0 else 1.0

        mesh = pmv.plot_array(
            arr,
            cmap=base_cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=base_cmap)
        cbar = plt.colorbar(
            mappable,
            ax=ax,
            fraction=0.046,
            pad=0.02,
        )

    cbar.outline.set_visible(False)

    # -------------------------------------------------------------------------
    # Grid, axes style, north arrow, scale bar
    # -------------------------------------------------------------------------
    pmv.plot_grid(linewidth=0.1, color="k", alpha=grid_alpha, zorder=2)

    ax.set_title(title, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    add_north_arrow(ax)
    add_scale_bar(ax, units="km")

# -------------------------------------------------------------------------------------------------------------------- #
# Setup
# -------------------------------------------------------------------------------------------------------------------- #

# Read in MODFLOW model
gwf = flopy.modflow.Modflow.load(model_dir / (model_name + '.nam'), version='mfnwt',
                                 load_only=['dis','bas6','wel','rch','upw'],
                                 model_ws=model_dir)
gwf.modelgrid.set_coord_info(xoff=xoff, yoff=yoff)
mg = gwf.modelgrid
dis = gwf.dis
nlay = dis.nlay
nrow = dis.nrow
ncol = dis.ncol
nper = dis.nper

ibound = gwf.bas6.ibound.array  # shape (nlay, nrow, ncol)

# Time information
perlen = dis.perlen.array.astype(float)   # length of each stress period
total_time = perlen.sum()

# Boolean mask of active cells
active_mask_3d = ibound != 0  # shape (nlay, nrow, ncol)

# For each (i,j), find index of first active layer (or -1 if none)
top_active = np.full((nrow, ncol), -1, dtype=int)
has_active = active_mask_3d.any(axis=0)  # shape (nrow, ncol)
top_active[has_active] = np.argmax(active_mask_3d[:, has_active], axis=0)

# Read in SFR file - just the reach properties
sfr_file = model_dir / 'svihm.sfr'
with open(sfr_file, 'r') as f:
    line = f.readline()  # header
    line = f.readline()  # settings
    line = f.readline()  # nrch
    nrch = abs(int(line.split()[0]))
sfr_cols = ['layer', 'row', 'column', 'segment', 'reach', 'length', 'elevation', 'slope', 'thick', 'sbk']
sfr = pd.read_csv(sfr_file, sep='\\s+', skiprows=3, names=sfr_cols, nrows=nrch)

sfr_sbk = np.full((nlay, nrow, ncol), np.nan, dtype=float)
for _, r in sfr.iterrows():
    k = int(r['layer'] - 1)   # 1-indexed → 0-indexed
    i = int(r['row']   - 1)
    j = int(r['column']- 1)
    sfr_sbk[k, i, j] = r['sbk']


# -------------------------------------------------------------------------------------------------------------------- #
# Hydrogeologic Properties
# -------------------------------------------------------------------------------------------------------------------- #

props_dict = {
    "Kh":    gwf.upw.hk.array.astype(float).copy(),
    "Kv":    gwf.upw.vka.array.astype(float).copy(),
    "Ss":    gwf.upw.ss.array.astype(float).copy(),
    "Sy":    gwf.upw.sy.array.astype(float).copy(),
    "aniso": gwf.upw.hk.array.astype(float).copy() / gwf.upw.vka.array.astype(float).copy(),
    "sbk":   sfr_sbk,
}

titles = {
    "Kh":    "Horizontal Hydraulic Conductivity (m/d)",
    "Kv":    "Vertical Hydraulic Conductivity (m/d)",
    "Ss":    "Specific Storage (1/m)",
    "Sy":    "Specific Yield (–)",
    "aniso": "Anisotropy (Kh/Kv)",
    "sbk":   "Streambed Conductance (m/d)",
}

order = ["Kh", "Kv", "Ss", "Sy", "aniso", "sbk"]

for k in range(0,2):

    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    fig.patch.set_alpha(0)
    axes = axes.flatten()
    ax_track = 0

    for ax, key in zip(axes, order):
        if k>0 and key=='sbk': continue
        prop3d = props_dict[key]
        title = titles[key]
        cmap_name = "viridis"

        plot_property_panel(
            ax=ax,
            prop3d=prop3d,
            ibound=ibound,
            mg=mg,
            layer=k,
            title=title,
            n_classes=4,
            use_jenks=False,
            cmap_name=cmap_name,
        )
        ax_track += 1

    # Turn off any unused axes (if order shorter than 6)
    for ax in axes[ax_track:]:
        ax.set_visible(False)

    #fig.suptitle(f"Calibrated Hydrogeologic Properties (Layer {k+1})", fontsize=8, y=0.995)
    plt.tight_layout()
    plt.savefig(plt_dir / f"SVIHM_Calibrated_Properties_layer{k+1}.png", dpi=300)

plt.close('all')
