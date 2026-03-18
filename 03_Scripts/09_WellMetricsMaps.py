import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import pandas as pd
import numpy as np
import flopy
import geopandas as gpd
from pathlib import Path
from hydroeval import evaluator, nse, kge, rmse, pbias
from tqdm import tqdm

# import os
# os.chdir('../')

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

# Directories
data_dir = Path('01_Data/')
shp_dir = data_dir / 'GIS'
plt_dir = Path('05_Plots/WellMetricsMaps/')
plt_dir.mkdir(parents=True, exist_ok=True)
svihm_dir = Path('../SVIHM/')  # External to project, local SVIHM Git repo
svihm_ref_dir = svihm_dir / 'SVIHM_Input_Files/reference_data_for_plots/'
#model_dir = Path('C:/Projects/SVIHM/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW/')
model_dir = Path('/Volumes/Macintosh HD/Users/leland/Documents/ModelRuns/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW')
tex_file_dir = model_dir / '../preproc/'
hob_cache = data_dir / 'hobs_df_cached.pkl'
head_obs_file = data_dir / 'head_obs_master.csv'

# Texture files
tex_files = {
    'FINE': 't2p_FINE.csv',
    'MIXED_FINE': 't2p_MIXED_FINE.csv',
    'SAND': 't2p_SAND.csv',
    'MIXED_COARSE': 't2p_MIXED_COARSE.csv',
    'VERY_COARSE': 't2p_VERY_COARSE.csv'
}

# Shapefiles
sv_model_shp_file = shp_dir / 'grid_properties_rep.shp'

# Model Info
xoff = 499977
yoff = 4571330
origin_date = pd.to_datetime('1990-9-30')

# Plot Style
sns.set_theme(style="whitegrid")

# Plot settings
plt.rcParams.update({
    "font.family": "Bahnschrift",  # DM Serif Text
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "axes.unicode_minus": False # for mac
})

# -------------------------------------------------------------------------------------------------------------------- #
# Classes/Functions
# -------------------------------------------------------------------------------------------------------------------- #

def _nice_scale_length(max_len):
    if max_len <= 0:
        return 0
    exp = np.floor(np.log10(max_len))
    frac = max_len / 10**exp
    for n in [1, 2, 5]:
        if frac <= n:
            return n * 10**exp
    return 10 * 10**exp

# -------------------------------------------------------------------------------------------------------------------- #

def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.06, 0.98),
        xytext=(0.06, 0.90),
        xycoords="axes fraction",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-|>", lw=1.3),
        fontsize=8,
    )

# -------------------------------------------------------------------------------------------------------------------- #

def add_scale_bar(ax, units="km"):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = x_max - x_min
    if width <= 0:
        return

    target = width / 5.0
    bar_len = _nice_scale_length(target)
    if bar_len == 0:
        return

    x0 = x_min + 0.06 * width
    y0 = y_min + 0.06 * (y_max - y_min)

    ax.plot([x0, x0 + bar_len], [y0, y0], lw=2, color="k", zorder=10)
    label = f"{bar_len/1000:g} km" if units == "km" else f"{bar_len:g} {units}"
    ax.text(x0 + bar_len / 2.0, y0, label, ha="center", va="bottom", fontsize=8, zorder=10)

# -------------------------------------------------------------------------------------------------------------------- #

def upw_to_df(upw, ibound=None):
    """Convert UPW parameters (HK, VKA, SS, SY) and optionally IBOUND into a long-format DataFrame."""
    def to_series(arr, name):
        arr = np.asarray(arr)
        if arr.ndim == 2:  # single layer, expand to 3D
            arr = arr[np.newaxis, :, :]
        nlay, nrow, ncol = arr.shape
        lay, row, col = np.meshgrid(np.arange(nlay), np.arange(nrow), np.arange(ncol), indexing='ij')
        return pd.DataFrame({
            'Layer': lay.ravel() + 1,
            'Row': row.ravel() + 1,
            'Column': col.ravel() + 1,
            name: arr.ravel()
        })

    # Assemble individual DataFrames
    df = to_series(upw.hk.array, 'HK') \
        .merge(to_series(upw.vka.array, 'VK'), on=['Layer', 'Row', 'Column']) \
        .merge(to_series(upw.ss.array, 'SS'), on=['Layer', 'Row', 'Column']) \
        .merge(to_series(upw.sy.array, 'SY'), on=['Layer', 'Row', 'Column'])

    # Optionally add IBOUND
    if ibound is not None:
        df = df.merge(to_series(ibound, 'IBOUND'), on=['Layer', 'Row', 'Column'])

    return df

# -------------------------------------------------------------------------------------------------------------------- #

def calc_metrics(group, weight_col=None):
    obs = group["obsval"].to_numpy(dtype=float)
    sim = group["simval"].to_numpy(dtype=float)

    if len(obs) < 5 or np.any(~np.isfinite(obs)) or np.any(~np.isfinite(sim)):
        return pd.Series({"avg_res": np.nan})

    res = obs - sim

    if weight_col is None:
        avg_res = np.mean(res)
    else:
        w = group[weight_col].to_numpy(dtype=float)
        if np.nanmax(w) == 0:
            return pd.Series({"avg_res": np.nan})
        # weighted mean residual
        avg_res = np.average(res, weights=w)

    return pd.Series({"avg_res": avg_res})

# -------------------------------------------------------------------------------------------------------------------- #

def calc_PEST_res(df, weight_col='wt'):
    res_df = df.copy()
    res_df['res'] = res_df['obsval'] - res_df['simval']
    res_df['wtsqres'] = res_df['res']**2 * res_df[weight_col]**2
    return res_df

# -------------------------------------------------------------------------------------------------------------------- #

def repel_points(x, y, min_dist, iters=30, step=0.35, seed=13):
    """
    Simple deterministic repulsion in map units.
    - x,y: arrays
    - min_dist: minimum separation desired (same units as x,y)
    """
    rng = np.random.default_rng(seed)
    x = x.astype(float).copy()
    y = y.astype(float).copy()

    # Tiny initial jitter to break exact overlaps (deterministic)
    x += rng.normal(0, min_dist * 0.02, size=len(x))
    y += rng.normal(0, min_dist * 0.02, size=len(y))

    for _ in range(iters):
        moved = False
        for i in range(len(x)):
            dx = x[i] - x
            dy = y[i] - y
            dx[i] = 0.0
            dy[i] = 0.0
            dist = np.hypot(dx, dy)
            close = (dist > 0) & (dist < min_dist)
            if not np.any(close):
                continue

            # Push away from nearby points
            pushx = np.sum(dx[close] / (dist[close] + 1e-12))
            pushy = np.sum(dy[close] / (dist[close] + 1e-12))
            norm = np.hypot(pushx, pushy)
            if norm > 0:
                x[i] += step * (min_dist - np.min(dist[close])) * (pushx / norm)
                y[i] += step * (min_dist - np.min(dist[close])) * (pushy / norm)
                moved = True

        if not moved:
            break

    return x, y

# -------------------------------------------------------------------------------------------------------------------- #

def plot_well_residuals(well_gdf, grid_df, col='avg_res', prop=None, cmap='RdBu_r', ax=None):
    """
    Plot per-well residuals on the model grid with a diverging color map.

    Parameters
    ----------
    well_gdf : GeoDataFrame
        Must contain 'geometry' and the residual column (default 'resid_mean').
    grid_df : GeoDataFrame
        Model grid with geometry (and optional properties).
    col : str
        Column name in well_df to plot (e.g., 'resid_mean').
    prop : str or None
        Optional property in grid_df to use as a background (e.g., 'HK').
    cmap : str
        Matplotlib colormap name, typically diverging (e.g., 'RdBu_r').
    ax : matplotlib Axes or None
        If None, a new figure/axes is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Background grid
    if prop and prop in grid_df.columns:
        grid_df.plot(column=prop, ax=ax, cmap='Greys', edgecolor='none', legend=False)
    else:
        grid_df.plot(color='none', ax=ax, edgecolor='lightgrey')

    # Symmetric color limits around zero
    vals = well_gdf[col].values
    max_abs = np.nanmax(np.abs(vals))
    vmin, vmax = -max_abs, max_abs

    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    well_gdf.plot(
        ax=ax,
        column=col,
        cmap=cmap,
        norm=norm,
        edgecolor='black',
        linewidth=0.5,
        markersize=40,
        legend=True
    )

    ax.set_title(f'Per-well residuals ({col}, obs - sim)')
    ax.set_axis_off()
    plt.tight_layout()

# -------------------------------------------------------------------------------------------------------------------- #

def plot_residual_panel(ax, grid_layer_gdf, well_layer_gdf, norm, cmap="RdBu_r",
                        title="", grid_edgecolor="0.85", grid_lw=0.2,
                        marker_size=28, marker_edge_lw=0.5, jitter=False,
                        jitter_min_dist=None):
    """
    One layer panel.
    """
    # Background grid
    grid_layer_gdf.plot(ax=ax, color="none", edgecolor=grid_edgecolor, linewidth=grid_lw, zorder=1)

    if len(well_layer_gdf) > 0:
        g = well_layer_gdf.copy()

        # Optional repulsion/jitter for readability
        if jitter:
            if jitter_min_dist is None:
                # default: ~1/4 of a cell width (if cell size unknown, pick 150 m)
                jitter_min_dist = 150.0

            x = g.geometry.x.to_numpy()
            y = g.geometry.y.to_numpy()
            x2, y2 = repel_points(x, y, min_dist=jitter_min_dist, iters=35, step=0.35, seed=13)
            g["geometry"] = gpd.points_from_xy(x2, y2)

        g.plot(
            ax=ax,
            column="avg_res",
            cmap=cmap,
            norm=norm,
            markersize=marker_size,
            edgecolor="black",
            linewidth=marker_edge_lw,
            zorder=5,
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    add_north_arrow(ax)
    add_scale_bar(ax, units="km")

# -------------------------------------------------------------------------------------------------------------------- #

def plot_well_residuals_two_layers(
    well_gdf,
    grid_gdf,
    out_png=None,
    cmap="RdBu_r",
    jitter=True,
    jitter_min_dist=125.0,
    marker_size=28,
):
    """
    Side-by-side residual maps for Layer 1 and Layer 2.
    Expects:
      - well_gdf columns: ['avg_res', 'primary_layer'] where primary_layer is 0-indexed
      - grid_gdf columns: ['Layer'] where Layer is 1-indexed
    """
    # Ensure CRS consistency if you use CRS (optional)
    # if well_gdf.crs is not None and grid_gdf.crs is not None and well_gdf.crs != grid_gdf.crs:
    #     well_gdf = well_gdf.to_crs(grid_gdf.crs)

    # Global symmetric norm across BOTH layers (consistent color meaning)
    v = well_gdf["avg_res"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    max_abs = np.max(np.abs(v)) if v.size else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 6))
    fig.patch.set_alpha(0)

    for ax, layer_idx0 in zip(axes, [0, 1]):
        # Grid is 1-indexed
        grid_layer = grid_gdf[grid_gdf["Layer"] == (layer_idx0 + 1)]
        wells_layer = well_gdf[well_gdf["primary_layer"] == layer_idx0]

        plot_residual_panel(
            ax=ax,
            grid_layer_gdf=grid_layer,
            well_layer_gdf=wells_layer,
            norm=norm,
            cmap=cmap,
            title=f"Layer {layer_idx0+1} Residuals",
            marker_size=marker_size,
            jitter=jitter,
            jitter_min_dist=jitter_min_dist,
        )

    # Shared colorbar (clean + consistent)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.colormaps.get_cmap(cmap))
    sm.set_array([])

    # Manually position colorbar axis: [left, bottom, width, height]
    cax = fig.add_axes([0.46, 0.15, 0.02, 0.70])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Mean Head Residual (m), obs − sim")
    cbar.outline.set_visible(False)

    plt.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=300, transparent=True)
    return fig, axes

# -------------------------------------------------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------------------------------------------------- #

# Read in MODFLOW Model
print('Reading MODFLOW Model')
gwf = flopy.modflow.Modflow.load('SVIHM.nam', load_only=['dis','bas6','upw'], version='mfnwt', model_ws=model_dir)
gwf.modelgrid.set_coord_info(xoff=xoff, yoff=yoff)

# Read in hob key for XY locations...
hob_locs = pd.read_csv(svihm_ref_dir / '_hob_key.csv')

# Also reads in simulated values
sim_hobs = pd.read_csv(model_dir / 'HobData_SVIHM.dat', sep='\\s+', skiprows=1, names=['simval', 'obsval', 'obsnme'])

# Observed data sets with observations & std deviations
head_obs = pd.read_csv(head_obs_file)
head_obs["date"] = pd.to_datetime(head_obs["date"])

# Assemble hobs_df with sim, obs, weight, loc
hobs_df = pd.merge(sim_hobs, head_obs[['obsnme', 'weight', 'wellid']], on='obsnme', how='left')

# Calculate metrics for HOB wells
well_metrics = hobs_df.groupby('wellid').apply(calc_metrics, weight_col='weight', include_groups=False).reset_index()
#well_metrics = well_metrics[~well_metrics['RMSE'].isna()]
well_metrics = pd.merge(well_metrics,
                        hob_locs[['well_id','x_proj','y_proj','primary_layer']].drop_duplicates(),
                        left_on='wellid', right_on='well_id')
# Convert to GDF
well_gdf = gpd.GeoDataFrame(well_metrics, geometry=gpd.points_from_xy(well_metrics.x_proj, well_metrics.y_proj))

# Read MODFLOW grid shapefile
grid = gpd.read_file(sv_model_shp_file)
#grid['geometry'] = grid['geometry'].apply(convert_to_2d)

# Create UPW properties DataFrame
upw_df = upw_to_df(gwf.upw, ibound=gwf.bas6.ibound.array)

# Merge in properties
grid = grid.merge(upw_df, how='left', on=['Layer', 'Row', 'Column'])

# Drop ibound==0 cells
grid = grid[grid['IBOUND']==1]

# How PEST sees it
PEST_res = calc_PEST_res(hobs_df, weight_col='weight')
PEST_res.groupby('wellid')['wtsqres'].sum().sort_values().tail(10)

# -------------------------------------------------------------------------------------------------------------------- #

# Maps
fig, axes = plot_well_residuals_two_layers(
    well_gdf=well_gdf,
    grid_gdf=grid,
    out_png=plt_dir / "SVIHM_head_residuals_L1_L2.png",
    cmap="RdBu_r",
    jitter=True,
    jitter_min_dist=250.0,
    marker_size=14,
)

# Histogram
vals = well_gdf['avg_res'].to_numpy(dtype=float)
vals = vals[np.isfinite(vals)]

fig, ax = plt.subplots(figsize=(6, 4))

# Histogram
ax.hist(
    vals,
    bins=20,
    color="0.6",
    edgecolor="black",
    linewidth=0.6,
    density=False,
)

# Zero line
ax.axvline(0.0, color="k", lw=1.2, linestyle="--", label="Zero residual")

# Mean / median lines
mean_val = np.mean(vals)
med_val = np.median(vals)

ax.axvline(mean_val, color="#1f77b4", lw=1.2, linestyle="-", label=f"Mean = {mean_val:.2f} m")
ax.axvline(med_val, color="#d62728", lw=1.2, linestyle="--", label=f"Median = {med_val:.2f} m")

ax.set_xlabel("Obs. Well mean head residual (m)")
ax.set_ylabel("Count")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(plt_dir / '../avg_heads_residual_hist.png', dpi=300, transparent=True)