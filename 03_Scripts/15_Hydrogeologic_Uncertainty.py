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
model_dir = Path('C:/Projects/SVIHM/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW/')
model_name = 'SVIHM'

ens_dir = Path('06_Outputs/06_wtfx/UPW_SFR_ensemble/')

xoff = 499977
yoff = 4571330

start_date = pd.to_datetime("1990-10-01")  # Start of SP 1

# Plot Style
sns.set_theme(style="whitegrid")

# Plot settings
plt.rcParams.update({
    "font.family": "DM Serif Text",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "figure.dpi": 300,
})

# Output
plt_dir = Path('05_Plots/MF_props_uncert/')
plt_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------------------------------- #
# Functions
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

def add_north_arrow(ax, size=0.1):
    """
    Add a simple north arrow to an axis (axes fraction coordinates).
    """
    ax.annotate(
        'N',
        xy=(0.05, 0.99),
        xytext=(0.05, 0.9),
        xycoords='axes fraction',
        ha='center', va='center',
        arrowprops=dict(arrowstyle='-|>', lw=1.5),
        fontsize=8,
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

# Read in "base" MODFLOW model
gwf = flopy.modflow.Modflow.load(model_dir / (model_name + '.nam'), version='mfnwt',
                                 load_only=['dis','bas6'],
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

# Get list of files to read in
upw_files = sorted(ens_dir.glob("ftx_*.upw"))
sfr_files = sorted(ens_dir.glob("ftx_*.sfr"))

# -------------------------------------------------------------------------------------------------------------------- #
# Read in and process all the UPW/SFR files...
# -------------------------------------------------------------------------------------------------------------------- #

prop = {'kh': [],
       'kv': [],
       'ss': [],
       'sy': [],
       'sbk': [],
        }
nrch = 0

for fname in tqdm(upw_files, total=len(upw_files), desc="UPW Files"):
    # Read
    upw_obj = flopy.modflow.ModflowUpw.load(str(fname), gwf)
    # Store
    prop['kh'].append(upw_obj.hk.array)
    prop['kv'].append(upw_obj.vka.array)
    prop['ss'].append(upw_obj.ss.array)
    prop['sy'].append(upw_obj.sy.array)

for i, fname in tqdm(enumerate(sfr_files), total=len(sfr_files), desc="SFR Files"):
    if i == 0:
        with open(fname, 'r') as f:
            line = f.readline()  # header
            line = f.readline()  # settings
            line = f.readline()  # nrch
            nrch = abs(int(line.split()[0]))
    sfr_cols = ['layer', 'row', 'column', 'segment', 'reach', 'length', 'elevation', 'slope', 'thick', 'sbk']
    sfrfile = pd.read_csv(fname, sep='\\s+', skiprows=3, names=sfr_cols, nrows=nrch)

    sfr_sbk = np.full((nlay, nrow, ncol), np.nan, dtype=float)
    for _, r in sfrfile.iterrows():
        k = int(r['layer'] - 1)  # 1-indexed → 0-indexed
        i = int(r['row'] - 1)
        j = int(r['column'] - 1)
        sfr_sbk[k, i, j] = r['sbk']
    prop['sbk'].append(sfr_sbk)

stats = {}
for key in prop.keys():
    temp_stack = np.stack(prop[key], axis=0)
    stats[key] = np.std(temp_stack, axis=0)

# -------------------------------------------------------------------------------------------------------------------- #
# Plot
# -------------------------------------------------------------------------------------------------------------------- #

titles = {
    "kh":    "Horizontal Hydraulic Conductivity (m/d)",
    "kv":    "Vertical Hydraulic Conductivity (m/d)",
    "ss":    "Specific Storage (1/m)",
    "sy":    "Specific Yield (–)",
    "sbk":   "Streambed Conductance (m/d)",
}

for k in range(0,2):

    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    fig.patch.set_alpha(0)
    axes = axes.flatten()
    ax_track = 0

    for ax, key in zip(axes, prop.keys()):
        if k>0 and key=='sbk': continue
        prop3d = stats[key]
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
    plt.savefig(plt_dir / f"SVIHM_Prop_Uncertainty_layer{k+1}.png", dpi=300)

plt.close('all')

# -------------------------------------------------------------------------------------------------------------------- #
# Sing me a little song
# -------------------------------------------------------------------------------------------------------------------- #


def summarize_array(arr, label="", pct=(1, 5, 25, 50, 75, 95, 99)):
    arr = np.asarray(arr)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        print(f"{label:>4s}: (no finite values)")
        return

    p = np.percentile(v, pct)
    print(f"\n{label}: n={v.size:,}")
    print(f"  min / max : {v.min():.4g} / {v.max():.4g}")
    print(f"  mean      : {v.mean():.4g}")
    print(f"  median    : {np.median(v):.4g}")
    print(f"  p01,p05   : {p[0]:.4g}, {p[1]:.4g}")
    print(f"  p25,p50,p75: {p[2]:.4g}, {p[3]:.4g}, {p[4]:.4g}")
    print(f"  p95,p99   : {p[5]:.4g}, {p[6]:.4g}")

# Active mask for aquifer properties (same shape as 3D arrays)
active = (ibound != 0)

for key in prop.keys():
    std3d = stats[key]

    # Choose the right mask:
    if key == "sbk":
        mask = np.isfinite(std3d)   # only where sbk exists
    else:
        mask = active & np.isfinite(std3d)

    summarize_array(std3d[mask], label=key)

for key in prop.keys():
    std3d = stats[key]
    print("\n" + "="*60)
    print(f"{key} (std)")

    for k in range(nlay):
        if key == "sbk":
            mask = np.isfinite(std3d[k])
        else:
            mask = (ibound[k] != 0) & np.isfinite(std3d[k])

        summarize_array(std3d[k][mask], label=f"L{k+1}")