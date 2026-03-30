import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import flopy
from flopy.plot import PlotMapView

# ----------------------------------------------------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------------------------------------------------

data_dir = Path("01_Data")
f_dir = Path("06_Outputs/06_wtfx/")
out_dir = Path("06_Outputs/processed_pp_tables")
out_dir.mkdir(parents=True, exist_ok=True)
plt_dir = Path("05_Plots/pp_shifts")
plt_dir.mkdir(parents=True, exist_ok=True)

par_init_file = f_dir / "svihm_ies.0.par.csv"
par_final_file = f_dir / "svihm_ies.3.par.csv"

# PP location files
kv_pp_file = data_dir / "pp_init_csv" / "init_kv_mult_pp_kv_mult.csv"
scale_pp_file = data_dir / "pp_init_csv" / "init_scale_pp_all_textures.csv"

# Texture prior distribution file (for base scale constants)
tex_dist_file = data_dir / "lognorm_dist_clustered.par"

ordered_scale_map = [
    ("scale_1ff", "Fine"),
    ("scale_2mf", "Mixed_Fine"),
    ("scale_3sc", "Sand"),
    ("scale_3mc", "Mixed_Coarse"),
    ("scale_4vc", "Very_Coarse"),
]

# Optional weights for the average signed log-shift metric
# Larger weights toward coarser classes if you want the scalar to emphasize coarse-end movement.
texture_weights = {
    "Fine": 1.0,
    "Mixed_Fine": 1.0,
    "Sand": 1.0,
    "Mixed_Coarse": 1.0,
    "Very_Coarse": 1.0,
}

# MODFLOW settings
model_dir = Path('/Volumes/Macintosh HD/Users/leland/Documents/ModelRuns/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW/')
model_name = 'SVIHM'
xoff = 499977
yoff = 4571330

# Plot Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "Bahnschrift",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 12.5,
    "figure.dpi": 300,
    "axes.unicode_minus": False,  # for Mac
})

# Marker settings for combined L1/L2 map
layer_marker_map = {
    1: "o",   # Layer 1
    2: "^",   # Layer 2
}

marker_size = 55
marker_edgecolor = "k"
marker_linewidth = 0.35

# ----------------------------------------------------------------------------------------------------------------------
# Read files
# ----------------------------------------------------------------------------------------------------------------------

par_iter0 = pd.read_csv(par_init_file, dtype={"real_name": str})
par_iter3 = pd.read_csv(par_final_file, dtype={"real_name": str})

par_iter0 = par_iter0.set_index("real_name")
par_iter3 = par_iter3.set_index("real_name")

# normalize column case for easier merging/parsing
par_iter0.columns = [c.lower() for c in par_iter0.columns]
par_iter3.columns = [c.lower() for c in par_iter3.columns]

kv_pp = pd.read_csv(kv_pp_file)
scale_pp = pd.read_csv(scale_pp_file)

kv_pp.columns = [c.lower() for c in kv_pp.columns]
scale_pp.columns = [c.lower() for c in scale_pp.columns]

kv_pp["parnme"] = kv_pp["parnme"].str.lower()
scale_pp["parnme"] = scale_pp["parnme"].str.lower()

# read prior texture scales
tex_dists_df = pd.read_table(tex_dist_file, sep=r"\s+", skiprows=1)
prior_scale_map = (
    tex_dists_df
    .set_index("Texture")["Scale"]
    .to_dict()
)

# ----------------------------------------------------------------------------------------------------------------------
# Common realizations
# ----------------------------------------------------------------------------------------------------------------------

common_members = par_iter0.index.intersection(par_iter3.index)

par_iter0 = par_iter0.loc[common_members].copy()
par_iter3 = par_iter3.loc[common_members].copy()

# base is the deterministic reference; do not use it as an ensemble member for agreement
ensemble_members = [r for r in common_members if r != "base"]

# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def make_long(df: pd.DataFrame, par_cols: list[str], iteration_label: str) -> pd.DataFrame:
    out = (
        df[par_cols]
        .copy()
        .reset_index()
        .melt(id_vars="real_name", var_name="parnme", value_name="value")
    )
    out["iteration"] = iteration_label
    return out

def parse_kv_name(parnme: str) -> dict:
    parts = parnme.lower().split("_")
    out = {
        "par_family": "kv_mult",
        "layer_tag": None,
        "layer_num": None,
        "pp_id": None,
    }
    if len(parts) >= 4:
        out["layer_tag"] = parts[2]
        out["pp_id"] = parts[3]
        if parts[2].startswith("l"):
            out["layer_num"] = int(parts[2][1:])
    return out

def parse_scale_name(parnme: str) -> dict:
    parts = parnme.lower().split("_")
    out = {
        "par_family": "scale",
        "texture_group": None,
        "layer_tag": None,
        "layer_num": None,
        "pp_id": None,
    }
    if len(parts) >= 4:
        out["texture_group"] = f"{parts[0]}_{parts[1]}"
        out["layer_tag"] = parts[2]
        out["pp_id"] = parts[3]
        if parts[2].startswith("l"):
            out["layer_num"] = int(parts[2][1:])
    return out

def signed_agreement_from_delta(delta: pd.Series) -> float:
    return np.sign(delta).mean()

def frac_positive(delta: pd.Series) -> float:
    return np.mean(delta > 0)

def frac_negative(delta: pd.Series) -> float:
    return np.mean(delta < 0)

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return np.sum(w * x) / np.sum(w)

def slope_vs_rank(x: np.ndarray, ranks: np.ndarray) -> float:
    """
    Simple least-squares slope of x vs ranks.
    """
    xbar = np.mean(x)
    rbar = np.mean(ranks)
    denom = np.sum((ranks - rbar) ** 2)
    if denom == 0:
        return np.nan
    return np.sum((ranks - rbar) * (x - xbar)) / denom

# ----------------------------------------------------------------------------------------------------------------------
# Plotting Helpers
# ----------------------------------------------------------------------------------------------------------------------

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

def add_scale_bar(ax, units="km"):
    """
    Add a scale bar to a map axis.
    Assumes x-units are meters; text is in km by default.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    width = x_max - x_min

    target = width / 5.0
    bar_len = _nice_scale_length(target)
    if bar_len == 0:
        return

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
        fontsize=14,
    )

def get_symmetric_vlim(series, q=0.98):
    """
    Robust symmetric color limit around zero.
    Uses the q-quantile of absolute values.
    """
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 1.0
    vmax = float(s.abs().quantile(q))
    if vmax == 0:
        vmax = float(s.abs().max())
    if vmax == 0:
        vmax = 1.0
    return vmax

def plot_active_footprint(ax, mg, active_union_2d, grid_alpha=0.04):
    """
    Plot pale active model footprint for layers 1+2 combined.
    """
    pmv = PlotMapView(modelgrid=mg, ax=ax, layer=0)

    # pale gray active footprint
    footprint = np.where(active_union_2d, 1.0, np.nan)
    cmap = mcolors.ListedColormap(["#eeeeee"])
    pmv.plot_array(footprint, cmap=cmap, zorder=1)

    # faint grid over the active cells
    pmv.plot_grid(linewidth=0.1, color="k", alpha=grid_alpha, zorder=2)

    # approximate outer outline from active mask
    # draw boundaries by contouring the binary mask at 0.5
    xcenters = mg.xcellcenters
    ycenters = mg.ycellcenters
    arr = active_union_2d.astype(float)

    try:
        ax.contour(
            xcenters,
            ycenters,
            arr,
            levels=[0.5],
            colors="0.35",
            linewidths=0.8,
            zorder=3,
        )
    except Exception:
        # fallback if contouring fails for some reason
        pass

def plot_pp_metric_map(
    df,
    metric_col,
    out_file,
    title,
    cbar_label,
    mg,
    ibound,
    layer_col="layer_num",
    x_col="x",
    y_col="y",
    cmap_name="RdBu_r",
    qlim=0.98,
    marker="o",
    marker_size=55,
    marker_edgecolor="k",
    marker_linewidth=0.35,
):
    """
    Plot one metric as a two-panel map:
      - Layer 1 (model layer 0) on the left
      - Layer 2 (model layer 1) on the right

    Assumes `layer_col` may be zero-based (0, 1), which is what your parsed PP tables
    likely contain. Uses a shared divergent color scale centered at zero.
    """

    dat = df[[x_col, y_col, layer_col, metric_col]].copy()
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, layer_col, metric_col])

    if dat.empty:
        print(f"Skipping {metric_col}: no valid data.")
        return

    # Be explicit about integer layer indexing
    dat[layer_col] = dat[layer_col].astype(int)

    # Keep just first two model layers, assuming zero-based indexing
    dat = dat.loc[dat[layer_col].isin([0, 1])].copy()
    if dat.empty:
        print(f"Skipping {metric_col}: no layer 0/1 data.")
        return

    # Shared symmetric color scale around zero
    #if metric_col.startswith("signed_agreement"):
    #    vmax = 0.5
    #else:
    vmax = get_symmetric_vlim(dat[metric_col], q=qlim)

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.colormaps.get_cmap(cmap_name)

    # Figure with dedicated colorbar axis so it doesn't overlap Layer 2
    fig = plt.figure(figsize=(14 ,12))
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  #, wspace=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    axes = [ax1, ax2]

    # zero-based -> display labels
    layer_map = [
        (0, "Layer 1"),
        (1, "Layer 2"),
    ]

    for ax, (lay, lay_title) in zip(axes, layer_map):
        # Active footprint for this layer only, no outline
        active_2d = ibound[lay, :, :] != 0
        footprint = np.where(active_2d, 1.0, np.nan)

        pmv = PlotMapView(modelgrid=mg, ax=ax, layer=lay)
        fill_cmap = mcolors.ListedColormap(["#eeeeee"])
        pmv.plot_array(footprint, cmap=fill_cmap, zorder=1)
        pmv.plot_grid(linewidth=0.1, color="k", alpha=0.04, zorder=2)

        sub = dat.loc[dat[layer_col] == lay].copy()

        if not sub.empty:
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[metric_col],
                cmap=cmap,
                norm=norm,
                s=marker_size,
                marker=marker,
                edgecolor=marker_edgecolor,
                linewidth=marker_linewidth,
                zorder=5,
            )

        ax.set_title(lay_title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        add_north_arrow(ax)
        add_scale_bar(ax, units="km")

    # Shared colorbar in dedicated axis
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)

    fig.suptitle(title, fontweight="bold", y=0.98)

    plt.tight_layout() #rect=[0, 0, 1, 0.98])
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved {out_file}")

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

kv_cols = [c for c in par_iter0.columns if c.startswith("kv_mult")]
scale_cols = [c for c in par_iter0.columns if c.startswith("scale_")]

# Build long dataframes
kv_long = pd.concat(
    [
        make_long(par_iter0, kv_cols, "iter0"),
        make_long(par_iter3, kv_cols, "iter3"),
    ],
    ignore_index=True,
)

kv_meta = kv_long["parnme"].apply(parse_kv_name).apply(pd.Series)
kv_long = pd.concat([kv_long, kv_meta], axis=1)

scale_long = pd.concat(
    [
        make_long(par_iter0, scale_cols, "iter0"),
        make_long(par_iter3, scale_cols, "iter3"),
    ],
    ignore_index=True,
)

scale_meta = scale_long["parnme"].apply(parse_scale_name).apply(pd.Series)
scale_long = pd.concat([scale_long, scale_meta], axis=1)

# base reference from iteration 0
kv_base_ref = (
    kv_long.loc[
        (kv_long["real_name"] == "base") & (kv_long["iteration"] == "iter0"),
        ["parnme", "value"]
    ]
    .rename(columns={"value": "base_iter0_value"})
    .copy()
)

scale_base_ref = (
    scale_long.loc[
        (scale_long["real_name"] == "base") & (scale_long["iteration"] == "iter0"),
        ["parnme", "value"]
    ]
    .rename(columns={"value": "base_iter0_value"})
    .copy()
)

kv_long = kv_long.merge(kv_base_ref, on="parnme", how="left")
scale_long = scale_long.merge(scale_base_ref, on="parnme", how="left")

kv_long["delta_from_base"] = kv_long["value"] - kv_long["base_iter0_value"]
scale_long["delta_from_base"] = scale_long["value"] - scale_long["base_iter0_value"]

# Attach PP coordinates for kv_mult
kv_pp_use = kv_pp[["name", "parnme", "x", "y", "layer", "parval1"]].copy()
kv_pp_use = kv_pp_use.rename(columns={
    "layer": "pp_layer_from_file",
    "parval1": "pp_true_start_value",
})

kv_long = kv_long.merge(kv_pp_use, on="parnme", how="left")

# kv_mult agreement table
kv_post = kv_long.loc[
    (kv_long["iteration"] == "iter3") &
    (kv_long["real_name"].isin(ensemble_members))
].copy()

kv_agreement = (
    kv_post.groupby(["name", "parnme", "par_family", "layer_tag", "layer_num", "pp_id", "x", "y"], dropna=False)
    .agg(
        n=("delta_from_base", "size"),
        signed_agreement=("delta_from_base", signed_agreement_from_delta),
        frac_positive=("delta_from_base", frac_positive),
        frac_negative=("delta_from_base", frac_negative),
        mean_post=("value", "mean"),
        sd_post=("value", "std"),
        base_iter0_value=("base_iter0_value", "first"),
        pp_true_start_value=("pp_true_start_value", "first"),
    )
    .reset_index()
)

# Prepare scale PP coordinates
#
# scale_pp contains one row per parameter name, but coordinates are the same across texture groups for a given pp/layer.
# We'll parse and deduplicate to one location per pp/layer.
scale_pp_meta = scale_pp["parnme"].apply(parse_scale_name).apply(pd.Series)
scale_pp2 = pd.concat([scale_pp.copy(), scale_pp_meta], axis=1)

# One row per physical PP location (same across texture groups), keeping the original PP name
scale_pp_locs = (
    scale_pp2.sort_values(["layer_num", "pp_id", "name"])
    [["name", "layer_tag", "layer_num", "pp_id", "x", "y"]]
    .drop_duplicates(subset=["layer_tag", "layer_num", "pp_id"])
    .copy()
)

# Also preserve per-parameter starting values from the pp file in case you want them later
scale_pp_paramvals = scale_pp[["parnme", "parval1"]].copy().rename(columns={"parval1": "pp_true_start_value"})

scale_long = scale_long.merge(scale_pp_paramvals, on="parnme", how="left")

# Reconstruct actual ordered texture scales at PP locations
#
# For a given realization, pp, layer:
#   actual_scale(Fine)          = scale_1ff * prior_scale(Fine)
#   actual_scale(Mixed_Fine)    = scale_2mf * actual_scale(Fine)
#   actual_scale(Sand)          = scale_3sc * actual_scale(Mixed_Fine)
#   actual_scale(Mixed_Coarse)  = scale_3mc * actual_scale(Sand)
#   actual_scale(Very_Coarse)   = scale_4vc * actual_scale(Mixed_Coarse)
#
# We compute both iter0-base actual scales and iter3-realization actual scales,
# then log-shifts relative to the iter0-base scales.

# only the groups we want, in order
ordered_groups = [g for g, _ in ordered_scale_map]
ordered_textures = [t for _, t in ordered_scale_map]

scale_sub = scale_long.loc[scale_long["texture_group"].isin(ordered_groups)].copy()

# wide values by realization / iteration / pp
scale_wide = (
    scale_sub.pivot_table(
        index=["real_name", "iteration", "layer_tag", "layer_num", "pp_id"],
        columns="texture_group",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)

scale_wide.columns.name = None

# attach pp location once per pp/layer
scale_wide = scale_wide.merge(
    scale_pp_locs,
    on=["layer_tag", "layer_num", "pp_id"],
    how="left"
)

# add actual scales
for g, tex in ordered_scale_map:
    scale_wide[f"{tex}_actual_scale"] = np.nan

# sequential reconstruction
first_group, first_tex = ordered_scale_map[0]
scale_wide[f"{first_tex}_actual_scale"] = scale_wide[first_group] * prior_scale_map[first_tex]

for i in range(1, len(ordered_scale_map)):
    g, tex = ordered_scale_map[i]
    _, prev_tex = ordered_scale_map[i - 1]
    scale_wide[f"{tex}_actual_scale"] = scale_wide[g] * scale_wide[f"{prev_tex}_actual_scale"]

# Build base actual scales table (iteration 0 base)
base_actual_cols = [f"{tex}_actual_scale" for tex in ordered_textures]

scale_base_actual = (
    scale_wide.loc[
        (scale_wide["real_name"] == "base") &
        (scale_wide["iteration"] == "iter0"),
        ["layer_tag", "layer_num", "pp_id"] + base_actual_cols
    ]
    .copy()
)

rename_map = {c: f"base_{c}" for c in base_actual_cols}
scale_base_actual = scale_base_actual.rename(columns=rename_map)

# Posterior realization table with base actual scales attached
scale_post_wide = scale_wide.loc[
    (scale_wide["iteration"] == "iter3") &
    (scale_wide["real_name"].isin(ensemble_members))
].copy()

scale_post_wide = scale_post_wide.merge(
    scale_base_actual,
    on=["layer_tag", "layer_num", "pp_id"],
    how="left"
)

# log shifts for each texture actual scale relative to iter0 base
for tex in ordered_textures:
    scale_post_wide[f"{tex}_log_shift"] = (
        np.log(scale_post_wide[f"{tex}_actual_scale"]) -
        np.log(scale_post_wide[f"base_{tex}_actual_scale"])
    )

# Collapse the 5D texture-scale shift to scalar diagnostics
#
# 1) mean_log_shift:
#    average signed log-shift across ordered texture means
#
# 2) rank_slope:
#    optional diagnostic: whether coarse-end means moved more than fine-end means

w = np.array([texture_weights[tex] for tex in ordered_textures], dtype=float)
ranks = np.arange(1, len(ordered_textures) + 1, dtype=float)

log_shift_cols = [f"{tex}_log_shift" for tex in ordered_textures]

scale_post_wide["mean_log_shift"] = scale_post_wide[log_shift_cols].apply(
    lambda row: weighted_mean(row.to_numpy(dtype=float), w),
    axis=1
)

scale_post_wide["rank_slope"] = scale_post_wide[log_shift_cols].apply(
    lambda row: slope_vs_rank(row.to_numpy(dtype=float), ranks),
    axis=1
)

# ----------------------------------------------------------------------------------------------------------------------
# Agreement summaries for scale coarsening/fining diagnostics
# ----------------------------------------------------------------------------------------------------------------------

scale_agreement = (
    scale_post_wide.groupby(["name", "layer_tag", "layer_num", "pp_id", "x", "y"], dropna=False)
    .agg(
        n=("mean_log_shift", "size"),
        signed_agreement_meanlog=("mean_log_shift", signed_agreement_from_delta),
        frac_positive_meanlog=("mean_log_shift", frac_positive),
        frac_negative_meanlog=("mean_log_shift", frac_negative),
        mean_post_meanlog=("mean_log_shift", "mean"),
        sd_post_meanlog=("mean_log_shift", "std"),
        signed_agreement_rankslope=("rank_slope", signed_agreement_from_delta),
        frac_positive_rankslope=("rank_slope", frac_positive),
        frac_negative_rankslope=("rank_slope", frac_negative),
        mean_post_rankslope=("rank_slope", "mean"),
        sd_post_rankslope=("rank_slope", "std"),
    )
    .reset_index()
)

# ----------------------------------------------------------------------------------------------------------------------
# Also make a long table of actual texture scales and log shifts
# ----------------------------------------------------------------------------------------------------------------------

scale_actual_long_records = []

for tex in ordered_textures:
    tmp = scale_post_wide[
        ["real_name", "layer_tag", "layer_num", "pp_id", "x", "y",
         f"{tex}_actual_scale", f"base_{tex}_actual_scale", f"{tex}_log_shift"]
    ].copy()

    tmp = tmp.rename(columns={
        f"{tex}_actual_scale": "post_actual_scale",
        f"base_{tex}_actual_scale": "base_actual_scale",
        f"{tex}_log_shift": "log_shift",
    })
    tmp["texture"] = tex
    scale_actual_long_records.append(tmp)

scale_actual_long = pd.concat(scale_actual_long_records, ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------------------------------------------------

kv_long.to_csv(out_dir / "kv_long.csv", index=False)
kv_agreement.to_csv(out_dir / "kv_agreement_by_pp.csv", index=False)

scale_long.to_csv(out_dir / "scale_long_rawpars.csv", index=False)
scale_wide.to_csv(out_dir / "scale_wide_actual_scales_all.csv", index=False)
scale_post_wide.to_csv(out_dir / "scale_post_wide_with_indices.csv", index=False)
scale_actual_long.to_csv(out_dir / "scale_actual_long.csv", index=False)
scale_agreement.to_csv(out_dir / "scale_agreement_by_pp.csv", index=False)

print("Saved:")
for fn in [
    "kv_long.csv",
    "kv_agreement_by_pp.csv",
    "scale_long_rawpars.csv",
    "scale_wide_actual_scales_all.csv",
    "scale_post_wide_with_indices.csv",
    "scale_actual_long.csv",
    "scale_agreement_by_pp.csv",
]:
    print("  ", out_dir / fn)

# ----------------------------------------------------------------------------------------------------------------------
# Quick histograms of agreement metrics
# Run after the dataframe-building script
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Plot style to roughly match your manuscript figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "Bahnschrift",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 300,
    "axes.unicode_minus": False,  # for Mac
})

hist_dir = plt_dir / "pp_shifts"
hist_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------------------------------------------------

def summarize_series(name, s):
    s = s.dropna()
    print(f"\n{name}")
    print("-" * len(name))
    print(f"n      : {len(s)}")
    print(f"mean   : {s.mean(): .4f}")
    print(f"std    : {s.std(): .4f}")
    print(f"min    : {s.min(): .4f}")
    print(f"p05    : {s.quantile(0.05): .4f}")
    print(f"p25    : {s.quantile(0.25): .4f}")
    print(f"median : {s.median(): .4f}")
    print(f"p75    : {s.quantile(0.75): .4f}")
    print(f"p95    : {s.quantile(0.95): .4f}")
    print(f"max    : {s.max(): .4f}")


def make_hist(
        df,
        col,
        filename,
        xlabel,
        ylabel="Count",
        title=None,
        bins=30,
        kde=True,
        xlim=None,
        vline0=True,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=col,
        bins=bins,
        kde=kde,
        ax=ax,
    )

    if vline0:
        ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    fig.savefig(hist_dir / filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# 1. kv_mult agreement histograms
# ----------------------------------------------------------------------------------------------------------------------

summarize_series("kv_mult signed agreement", kv_agreement["signed_agreement"])
make_hist(
    kv_agreement,
    col="signed_agreement",
    filename="kv_signed_agreement_hist.png",
    xlabel="Signed Agreement (-1 = unanimous decrease, +1 = unanimous increase)",
    title="Distribution of kv_mult Signed Agreement",
    bins=25,
    kde=True,
    xlim=(-1.05, 1.05),
    vline0=True,
)

# mean posterior shift from base
if "base_iter0_value" in kv_agreement.columns and "mean_post" in kv_agreement.columns:
    kv_agreement["mean_shift_from_base"] = kv_agreement["mean_post"] - kv_agreement["base_iter0_value"]

    summarize_series("kv_mult mean posterior shift from base", kv_agreement["mean_shift_from_base"])
    make_hist(
        kv_agreement,
        col="mean_shift_from_base",
        filename="kv_mean_shift_from_base_hist.png",
        xlabel="Mean Posterior Shift from Iteration-0 Base",
        title="Distribution of kv_mult Mean Posterior Shift",
        bins=30,
        kde=True,
        vline0=True,
    )

# ----------------------------------------------------------------------------------------------------------------------
# 2. scale mean-log-shift histograms
# ----------------------------------------------------------------------------------------------------------------------

summarize_series("scale signed agreement (mean_log_shift)", scale_agreement["signed_agreement_meanlog"])
make_hist(
    scale_agreement,
    col="signed_agreement_meanlog",
    filename="scale_signed_agreement_meanlog_hist.png",
    xlabel="Signed Agreement of Mean Log-Shift",
    title="Distribution of Scale Signed Agreement (Mean Log-Shift)",
    bins=25,
    kde=True,
    xlim=(-1.05, 1.05),
    vline0=True,
)

summarize_series("scale mean posterior mean_log_shift", scale_agreement["mean_post_meanlog"])
make_hist(
    scale_agreement,
    col="mean_post_meanlog",
    filename="scale_mean_post_meanlog_hist.png",
    xlabel="Mean Posterior Mean Log-Shift",
    title="Distribution of Scale Mean Log-Shift",
    bins=30,
    kde=True,
    vline0=True,
)

# ----------------------------------------------------------------------------------------------------------------------
# 3. scale rank-slope histograms
# ----------------------------------------------------------------------------------------------------------------------

summarize_series("scale signed agreement (rank_slope)", scale_agreement["signed_agreement_rankslope"])
make_hist(
    scale_agreement,
    col="signed_agreement_rankslope",
    filename="scale_signed_agreement_rankslope_hist.png",
    xlabel="Signed Agreement of Rank-Slope",
    title="Distribution of Scale Signed Agreement (Rank-Slope)",
    bins=25,
    kde=True,
    xlim=(-1.05, 1.05),
    vline0=True,
)

summarize_series("scale mean posterior rank_slope", scale_agreement["mean_post_rankslope"])
make_hist(
    scale_agreement,
    col="mean_post_rankslope",
    filename="scale_mean_post_rankslope_hist.png",
    xlabel="Mean Posterior Rank-Slope",
    title="Distribution of Scale Rank-Slope",
    bins=30,
    kde=True,
    vline0=True,
)

print(f"\nSaved histogram figures to:\n{hist_dir}")

# -------------------------------------------------------------------------------------------------------------------- #
# Read model
# -------------------------------------------------------------------------------------------------------------------- #

gwf = flopy.modflow.Modflow.load(
    model_dir / f"{model_name}.nam",
    version="mfnwt",
    load_only=["dis", "bas6"],
    model_ws=model_dir,
)

gwf.modelgrid.set_coord_info(xoff=xoff, yoff=yoff)
mg = gwf.modelgrid
ibound = gwf.bas6.ibound.array  # shape (nlay, nrow, ncol)

# Never stop changing parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "Bahnschrift",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "figure.dpi": 300,
    "axes.unicode_minus": False,  # for Mac
})

# -------------------------------------------------------------------------------------------------------------------- #
# Plot configs
# -------------------------------------------------------------------------------------------------------------------- #

plot_jobs = [
    # kv_mult
    {
        "df": kv_agreement,
        "metric_col": "signed_agreement",
        "out_file": plt_dir / "kv_signed_agreement_map.png",
        "title": "Kriging Uncertainty Multiplier Directional Agreement",
        "cbar_label": "Directional Agreement",
        "markersize": 17,
    },
    {
        "df": kv_agreement,
        "metric_col": "mean_shift_from_base",
        "out_file": plt_dir / "kv_mean_shift_from_base_map.png",
        "title": "Kriging Uncertainty Multiplier Mean Posterior Shift",
        "cbar_label": "Mean Posterior Shift from Base",
        "markersize": 17,
    },

    # scale overall shift
    {
        "df": scale_agreement,
        "metric_col": "signed_agreement_meanlog",
        "out_file": plt_dir / "scale_signed_agreement_meanlog_map.png",
        "title": "Scale Directional Agreement (Mean Log-Shift)",
        "cbar_label": "Directional Agreement of Mean Log-Shift",
        "markersize": 55,
    },
    {
        "df": scale_agreement,
        "metric_col": "mean_post_meanlog",
        "out_file": plt_dir / "scale_mean_post_meanlog_map.png",
        "title": "Scale Mean Posterior Mean Log-Shift",
        "cbar_label": "Mean Posterior Mean Log-Shift",
        "markersize": 55,
    },

    # scale texture-rank tilt
    {
        "df": scale_agreement,
        "metric_col": "signed_agreement_rankslope",
        "out_file": plt_dir / "scale_signed_agreement_rankslope_map.png",
        "title": "ER-texture Transform Scale Distribution Directional Agreement",
        "cbar_label": "Directional Agreement of Rank-Slope",
        "markersize": 55,
    },
    {
        "df": scale_agreement,
        "metric_col": "mean_post_rankslope",
        "out_file": plt_dir / "scale_mean_post_rankslope_map.png",
        "title": "Scale Mean Posterior Rank-Slope",
        "cbar_label": "Mean Posterior Rank-Slope",
        "markersize": 55,
    },
]

# -------------------------------------------------------------------------------------------------------------------- #
# Make plots
# -------------------------------------------------------------------------------------------------------------------- #

for job in plot_jobs:
    plot_pp_metric_map(
        df=job["df"],
        metric_col=job["metric_col"],
        out_file=job["out_file"],
        title=job["title"],
        cbar_label=job["cbar_label"],
        mg=mg,
        ibound=ibound,
        layer_col="layer_num",
        x_col="x",
        y_col="y",
        cmap_name="PuOr",
        qlim=0.99,
        marker_size=job['markersize'],
    )

print(f"\nSaved maps to:\n{plt_dir}")