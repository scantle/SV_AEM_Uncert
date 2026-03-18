import matplotlib
matplotlib.use('TkAgg')
from matplotlib.font_manager import FontProperties
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

data_dir = Path('01_Data/')
f_dir = Path('06_Outputs/06_wtfx/')

par_init_file = f_dir / "svihm_ies.0.par.csv"
par_file   = f_dir / "svihm_ies.3.par.csv"
obs_file   = f_dir / "svihm_ies.3.obs.csv"
phi_actual_file = f_dir / 'svihm_ies.phi.actual.csv'
phi_comp_file   = f_dir / 'svihm_ies.phi.composite.csv'
phi_group_file  = f_dir / 'svihm_ies.phi.group.csv'
phi_meas_file   = f_dir / 'svihm_ies.phi.meas.csv'

# Extract iteration number...
iter = obs_file.name.split('.')[1]

plt_dir = Path('05_Plots/') / f'{f_dir.name}_iter{iter}' / 'par_analysis'
plt_dir.mkdir(parents=True, exist_ok=True)

# Fonts
title_font = FontProperties(family="Bahnschrift", size=14)
label_font = {"family": "Bahnschrift", "size": 11}
tick_font  = FontProperties(family="Bahnschrift", size=9)

# Plot Style
sns.set_theme(style="whitegrid")

#----------------------------------------------------------------------------------------------------------------------#
# Read In Files
#----------------------------------------------------------------------------------------------------------------------#

# PEST iteration results
par_iter0 = pd.read_csv(par_init_file, dtype={"real_name": str}, index_col=['real_name'])
par_results = pd.read_csv(par_file, dtype={"real_name": str}, index_col=['real_name'])
#obs_results = pd.read_csv(obs_file, dtype={"real_name": str}, index_col=['real_name'])
phi_actual = pd.read_csv(phi_actual_file)
phi_group  = pd.read_csv(phi_group_file)

# For phi_actual, columns 0..N and "base" contain per-member values; reshape long
member_cols = [c for c in phi_actual.columns if c not in
               ["iteration","total_runs","mean","standard_deviation","min","max"]]

phi_long = phi_actual.melt(
    id_vars=["iteration"],
    value_vars=member_cols,
    var_name="member",
    value_name="phi"
)

phi_long["member"] = phi_long["member"].replace({"base": -1}).astype(int)
phi_long["logphi"] = np.log10(phi_long["phi"])

# Stats of log10(phi) by iteration
phi_stats = (
    phi_long
    .groupby("iteration")["logphi"]
    .agg(["mean", "std"])
    .reset_index()
)

#phi_base = phi_long[phi_long['member']==-1]
#phi_base['logphi'] = np.log10(phi_base['phi'])

#----------------------------------------------------------------------------------------------------------------------#
# Plots
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Phi Reduction Across Iterations
fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)

ax.errorbar(
    phi_stats["iteration"],
    phi_stats["mean"],
    yerr=phi_stats["std"],
    fmt="o-",
    capsize=4,
    linewidth=1.3,
    markersize=4,
)

#ax.plot(phi_base["iteration"], phi_base["logphi"], label="base", color="black")

ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\log_{10}(\phi)$")
#ax.set_title("Ensemble objective function across iterations", fontproperties=title_font)

# Ticks font
# for label in ax.get_xticklabels():
#     label.set_fontproperties(tick_font)
# for label in ax.get_yticklabels():
#     label.set_fontproperties(tick_font)

ax.set_xticks(phi_stats["iteration"])

# Clean up frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(plt_dir / "phi_mean_by_iteration_log10phi.png", dpi=300, transparent=True)
plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Phi distribution by iteration
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)

sns.boxplot(
    data=phi_long,
    x="iteration",
    y="logphi",
    ax=ax,
    width=0.6,
    fliersize=1.5,
    linewidth=1.0,
)

ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\log_{10}(\phi)$")
ax.set_title("Distribution of ensemble objective function", fontproperties=title_font)

# Tick fonts
for label in ax.get_xticklabels():
    label.set_fontproperties(tick_font)
for label in ax.get_yticklabels():
    label.set_fontproperties(tick_font)

# Clean frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(plt_dir / "phi_distribution_by_iteration_log10phi.png", dpi=300)
plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Phi contribution by group per iteration

# Identify the group columns (exclude meta columns)
group_cols = [
    c for c in phi_group.columns
    if c not in ["iteration", "total_runs", "obs_realization", "par_realization"]
]

group_mean = phi_group.groupby("iteration")[group_cols].mean().reset_index()

group_long = group_mean.melt(
    id_vars="iteration",
    var_name="group",
    value_name="phi_mean",
)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(
    data=group_long,
    x="group",
    y="phi_mean",
    hue="iteration",
    ax=ax,
)

ax.set_xlabel("Observation Group")
ax.set_ylabel("Mean φ Contribution")
ax.set_title("Mean Group φ by Iteration")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Histogram of phi distribution (Iteration 0 vs Iteration 3 on same plot)

fig, ax = plt.subplots(figsize=(7.5,4.5), dpi=300)

iters_to_plot = [0, 3]
colors = sns.color_palette("deep", 2)

# Keep only finite, positive phi values before using log10 values
plot_df = phi_long[
    phi_long["iteration"].isin(iters_to_plot)
    & np.isfinite(phi_long["phi"])
    & (phi_long["phi"] > 0)
].copy()

plot_df["logphi"] = np.log10(plot_df["phi"])

# Shared bins based only on ensemble members
subset = plot_df[plot_df["member"] != -1]

bins = np.histogram_bin_edges(subset["logphi"].dropna(), bins=35)

for i, it in enumerate(iters_to_plot):

    df = plot_df[plot_df["iteration"] == it]
    df_ens = df[df["member"] != -1]
    df_base = df[df["member"] == -1]

    ax.hist(
        df_ens["logphi"],
        bins=bins,
        alpha=0.55,
        color=colors[i],
        label=f"Iteration {it}",
        edgecolor="none",
        density=True,
    )

    # Base model vertical line, only if it exists and is finite
    if not df_base.empty and np.isfinite(df_base["logphi"].iloc[0]):
        base_val = df_base["logphi"].iloc[0]
        ax.axvline(
            base_val,
            color=colors[i],
            linewidth=2.2,
            linestyle="--",
            label=f"Base (iter {it})"
        )

ax.set_xlabel(r"$\log_{10}(\phi)$")
ax.set_ylabel("Density")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()

fig.savefig(
    plt_dir / "phi_histogram_iter0_vs_iter3_overlay_log10phi.png",
    dpi=300,
)

plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Histogram of kV_mult

# Intersect members that exist in both iterations
#common_members = par_iter0.index.intersection(par_results.index)

kv_cols = [c for c in par_iter0.columns if c.startswith("kv_mult")]

kv0 = par_iter0.loc['base', kv_cols].to_numpy().ravel()

# note: this is from a base with parameters from a later iteration. Originally it was all zeros. However, we don't
# want to just replace all the data: to keep it honest, we'll leave the obvious manual calibration values (they're
# +/- 2 essentially)
kv0[(kv0>-1.5)&(kv0<1.5)] = 0.0

kv3 = par_results.loc['base', kv_cols].to_numpy().ravel()

kv_df = pd.DataFrame({
    "Value": np.concatenate([kv0, kv3]),
    "IES Iteration": (["Iteration 0"] * len(kv0)) + (["Iteration 3"] * len(kv3)),
})

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(
    data=kv_df,
    x="Value",
    hue="IES Iteration",
    bins=50,
    kde=True,
    element="step",
    ax=ax,
)

ax.set_xlabel("Kriging Uncertainty Multiplier Value")
ax.set_ylabel("Count")
#ax.set_title("Distribution of kv_mult Values")

plt.tight_layout()

fig.savefig(
    plt_dir / "kv_mult_hist.png",
    dpi=300,
)

plt.show()

# Let's get a little more wild with it:

# Histograms of kv_mult:
#   1) base realization
#   2) realization 127
#   3) full ensemble pooled (including base)

kv_cols = [c for c in par_iter0.columns if c.startswith("kv_mult")]

example_member = "127"

# ---- Base ----
kv0_base = par_iter0.loc["base", kv_cols].to_numpy(dtype=float).ravel()
kv3_base = par_results.loc["base", kv_cols].to_numpy(dtype=float).ravel()

kv0_base = np.zeros_like(kv0_base)

base_df = pd.DataFrame({
    "value": np.concatenate([kv0_base, kv3_base]),
    "iteration_label": (["Iteration 0"] * len(kv0_base)) + (["Iteration 3"] * len(kv3_base)),
    "panel": "Base Realization"
})

# ---- Example ensemble member ----
kv0_ex = par_iter0.loc[example_member, kv_cols].to_numpy(dtype=float).ravel()
kv3_ex = par_results.loc[example_member, kv_cols].to_numpy(dtype=float).ravel()

example_df = pd.DataFrame({
    "value": np.concatenate([kv0_ex, kv3_ex]),
    "iteration_label": (["Iteration 0"] * len(kv0_ex)) + (["Iteration 3"] * len(kv3_ex)),
    "panel": f"Realization {example_member}"
})

# ---- Full ensemble pooled (including base) ----
# Keep only realizations that exist in both files
common_members = par_iter0.index.intersection(par_results.index)

kv0_all = par_iter0.loc[common_members, kv_cols].to_numpy(dtype=float).ravel()
kv3_all = par_results.loc[common_members, kv_cols].to_numpy(dtype=float).ravel()

# Replace iteration 0 base values with zeros
# if "base" in common_members:
#     base_pos = np.where(common_members == "base")[0][0]
#     n_kv = len(kv_cols)
#     start = base_pos * n_kv
#     stop = start + n_kv
#     kv0_all[start:stop] = 0.0

all_df = pd.DataFrame({
    "value": np.concatenate([kv0_all, kv3_all]),
    "iteration_label": (["Iteration 0"] * len(kv0_all)) + (["Iteration 3"] * len(kv3_all)),
    "panel": "Full Ensemble"
})

# ---- Combine for plotting ----
plot_df = pd.concat([base_df, example_df, all_df], ignore_index=True)

# Shared bins across all panels
bins = np.histogram_bin_edges(plot_df["value"], bins=50)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=300, sharey=False)

panel_order = ["Base Realization", f"Realization {example_member}", "Full Ensemble"]

for ax, panel in zip(axes, panel_order):
    sub = plot_df[plot_df["panel"] == panel]

    sns.histplot(
        data=sub,
        x="value",
        hue="iteration_label",
        bins=bins,
        kde=True,
        element="step",
        common_norm=False,
        ax=ax,
    )

    # Fonts
    title_font = FontProperties(family="Bahnschrift", size=11)
    label_font = {"family": "Bahnschrift", "size": 11}
    tick_font = FontProperties(family="Bahnschrift", size=9)

    ax.set_title(panel, fontproperties=title_font)
    ax.set_xlabel("Kriging Uncertainty Multiplier Value")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep only one legend, on last panel
    if ax is not axes[2]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

fig.savefig(
    plt_dir / "kv_mult_hist_base_example_fullensemble.png",
    dpi=300,
)

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Scale parameters

common_members = par_iter0.index.intersection(par_results.index)

texture_groups = {
    "scale_1ff": [c for c in par_iter0.columns if c.startswith("scale_1ff")],
    "scale_2mf": [c for c in par_iter0.columns if c.startswith("scale_2mf")],
    "scale_3mc": [c for c in par_iter0.columns if c.startswith("scale_3mc")],
    "scale_3sc": [c for c in par_iter0.columns if c.startswith("scale_3sc")],
    "scale_4vc": [c for c in par_iter0.columns if c.startswith("scale_4vc")],
}

records = []

for tex_name, cols in texture_groups.items():
    if len(cols) == 0:
        continue  # skip any group that doesn't exist

    vals0 = par_iter0.loc['base', cols].to_numpy().ravel()
    vals3 = par_results.loc['base', cols].to_numpy().ravel()

    records.append(pd.DataFrame({
        "value": vals0,
        "iteration_label": "Iteration 0",
        "texture_group": tex_name,
    }))
    records.append(pd.DataFrame({
        "value": vals3,
        "iteration_label": "Iteration 3",
        "texture_group": tex_name,
    }))

scale_df = pd.concat(records, ignore_index=True)

g = sns.FacetGrid(
    scale_df,
    col="texture_group",
    hue="iteration_label",
    sharex=False,
    sharey=False,
    col_wrap=3,
    height=3.0,
)

g.map(
    sns.histplot,
    "value",
    bins=40,
    element="step",
    kde=True,
)

g.add_legend()
g.set_axis_labels("Scale Parameter Value", "Count")
g.fig.suptitle("Distributions of Texture Scale Parameters by Group (Iteration 0 vs 3)", y=1.02)

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Some investigation of members

# Use phi_long filtered to final iteration
final_iter = phi_long["iteration"].max()
phi_final = phi_long[phi_long["iteration"] == final_iter].copy()

# Identify best (minimum φ)
best_overall = phi_final.loc[phi_final["phi"].idxmin()]

best_overall_member = int(best_overall["member"])
best_overall_phi    = best_overall["phi"]

print("Best ensemble member overall:")
print(f"  member: {best_overall_member}")
print(f"  phi:    {best_overall_phi:,.3f}")

# Clean phi_group so member IDs are integers
phi_group2 = phi_group.copy()
phi_group2["member"] = phi_group2["obs_realization"].replace({"base": -1}).astype(int)

# Keep only the final iteration
phi_group_final = phi_group2[phi_group2["iteration"] == final_iter].copy()

# Identify group columns
group_cols = [
    c for c in phi_group_final.columns
    if c not in ["iteration", "total_runs", "obs_realization", "par_realization", "member"]
]

# Build a table of best member per group
records = []
for grp in group_cols:
    grp_vals = phi_group_final[["member", grp]].dropna()
    best_row = grp_vals.loc[grp_vals[grp].idxmin()]

    records.append({
        "group": grp,
        "best_member": int(best_row["member"]),
        "best_phi": float(best_row[grp])
    })

best_by_group = pd.DataFrame(records).sort_values("best_phi")
best_by_group

#----------------------------------------------------------------------------------------------------------------------#
# Calculate starting/ending values for t2p parameters
# Reconstruct derived hydraulic properties from par2par-style parameterization

sns.set_theme(style="whitegrid")

# Plot settings
plt.rcParams.update({
    "font.family": "Bahnschrift",
    "axes.unicode_minus": False, # for mac
    "font.size": 7,             # Overall base font size
    "axes.labelsize": 7,        # Size of X and Y labels
    "xtick.labelsize": 7,
    "axes.titlesize": 7        # For "Kh", "Kv", etc.
})

def build_hydraulic_props(par_df):
    """
    Reconstruct texture-specific hydraulic properties from PEST parameters.

    Parameters
    ----------
    par_df : pandas.DataFrame
        Rows are realizations, columns are parameter names (lowercase).

    Returns
    -------
    pandas.DataFrame
        Same index as par_df, with columns for Kh, Kv, Ss, and Sy by texture.
    """
    out = pd.DataFrame(index=par_df.index)

    # --- Kh ---
    out["kh_ff"] = par_df["kminff1"]
    out["kh_mf"] = out["kh_ff"] * par_df["kminmf1_m"]
    out["kh_sc"] = out["kh_mf"] * par_df["kminsc1_m"]
    out["kh_mc"] = out["kh_sc"] * par_df["kminmc1_m"]
    out["kh_vc"] = out["kh_mc"] * par_df["kminvc1_m"]

    # --- Anisotropy ---
    out["aniso_vc"] = par_df["anisovc1"]
    out["aniso_mc"] = out["aniso_vc"] * par_df["anisomc1_m"]
    out["aniso_sc"] = out["aniso_mc"] * par_df["anisosc1_m"]
    out["aniso_mf"] = out["aniso_sc"] * par_df["anisomf1_m"]
    out["aniso_ff"] = out["aniso_mf"] * par_df["anisoff1_m"]

    # --- Kv = Kh / aniso ---
    out["kv_ff"] = out["kh_ff"] / out["aniso_ff"]
    out["kv_mf"] = out["kh_mf"] / out["aniso_mf"]
    out["kv_sc"] = out["kh_sc"] / out["aniso_sc"]
    out["kv_mc"] = out["kh_mc"] / out["aniso_mc"]
    out["kv_vc"] = out["kh_vc"] / out["aniso_vc"]

    # --- Ss ---
    out["ss_ff"] = par_df["ssff1"]
    out["ss_mf"] = out["ss_ff"] * par_df["ssmf1_m"]
    out["ss_sc"] = out["ss_mf"] * par_df["sssc1_m"]
    out["ss_mc"] = out["ss_sc"] * par_df["ssmc1_m"]
    out["ss_vc"] = out["ss_mc"] * par_df["ssvc1_m"]

    # --- Sy ---
    out["sy_sc"] = par_df["sysc1"]
    out["sy_mf"] = out["sy_sc"] * par_df["symf1_m"]
    out["sy_ff"] = out["sy_mf"] * par_df["syff1_m"]
    out["sy_mc"] = out["sy_sc"] * par_df["symc1_m"]
    out["sy_vc"] = out["sy_mc"] * par_df["syvc1_m"]

    return out

# Build derived-property tables for iteration 0 and iteration 3

hydro0 = build_hydraulic_props(par_iter0.copy())
hydro3 = build_hydraulic_props(par_results.copy())

common_members = hydro0.index.intersection(hydro3.index)
hydro0 = hydro0.loc[common_members].copy()
hydro3 = hydro3.loc[common_members].copy()

hydro0["iteration_label"] = "Iteration 0"
hydro3["iteration_label"] = "Iteration 3"

hydro_df = pd.concat([hydro0, hydro3], axis=0)
hydro_df["real_name"] = hydro_df.index

# Reshape long for plotting

long_df = hydro_df.melt(
    id_vars=["real_name", "iteration_label"],
    var_name="param",
    value_name="value"
)

# Drop intermediate anisotropy columns
long_df = long_df[~long_df["param"].str.startswith("aniso_")].copy()

split_df = long_df["param"].str.split("_", n=1, expand=True)
long_df["property"] = split_df[0]
long_df["texture"] = split_df[1]

property_map = {
    "kh": "Kh",
    "kv": "Kv",
    "ss": "Ss",
    "sy": "Sy",
}

texture_map = {
    "ff": "Fine",
    "mf": "Mixed\nFine",
    "sc": "Sand",
    "mc": "Mixed\nCoarse",
    "vc": "Very\nCoarse",
}

long_df["property"] = long_df["property"].map(property_map)
long_df["texture"] = long_df["texture"].map(texture_map)

texture_order = ["Fine", "Mixed\nFine", "Sand", "Mixed\nCoarse", "Very\nCoarse"]
property_order = ["Kh", "Kv", "Ss", "Sy"]

long_df["texture"] = pd.Categorical(long_df["texture"], categories=texture_order, ordered=True)
long_df["property"] = pd.Categorical(long_df["property"], categories=property_order, ordered=True)

# Log-transform Kh, Kv, Ss

long_df["plot_value"] = long_df["value"]

log_props = ["Kh", "Kv", "Ss"]
mask = (
    long_df["property"].isin(log_props)
    & np.isfinite(long_df["value"])
    & (long_df["value"] > 0)
)
long_df.loc[mask, "plot_value"] = np.log10(long_df.loc[mask, "value"])

# Remove invalid values
long_df = long_df[np.isfinite(long_df["plot_value"])].copy()

# Determine common x-ranges and bins for each property column

prop_bins = {}
prop_xlim = {}

for prop in property_order:
    vals = long_df.loc[long_df["property"] == prop, "plot_value"].to_numpy()

    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        continue

    # Trim only very extreme tails for prettier axes, but still use almost all data
    xlo, xhi = np.nanpercentile(vals, [0.5, 99.5])

    # Add a little padding
    pad = 0.04 * (xhi - xlo) if xhi > xlo else 0.1
    xlo -= pad
    xhi += pad

    prop_xlim[prop] = (xlo, xhi)
    prop_bins[prop] = np.linspace(xlo, xhi, 30)

# Plot 5 x 4 grid

fig, axes = plt.subplots(
    nrows=len(texture_order),
    ncols=len(property_order),
    figsize=(16, 12),
    dpi=300,
    sharex="col",
    sharey=False
)

colors = sns.color_palette("deep", 2)
iter_order = ["Iteration 0", "Iteration 3"]

for i, texture in enumerate(texture_order):
    for j, prop in enumerate(property_order):
        ax = axes[i, j]

        sub = long_df[
            (long_df["texture"] == texture) &
            (long_df["property"] == prop)
        ].copy()

        bins = prop_bins[prop]

        for color, itlab in zip(colors, iter_order):
            sub_it = sub[sub["iteration_label"] == itlab]

            if sub_it.empty:
                continue

            ax.hist(
                sub_it["plot_value"],
                bins=bins,
                density=True,
                alpha=0.60,
                color=color,
                edgecolor="none",
                label=itlab if (i == 0 and j == 0) else None,
            )
            ax.set_yticklabels([])

            # # KDE overlay
            # sns.kdeplot(
            #     data=sub_it,
            #     x="plot_value",
            #     ax=ax,
            #     color=color,
            #     linewidth=1.4,
            #     fill=False,
            #     clip=prop_xlim[prop],
            #     warn_singular=False,
            #     legend=False,
            # )

        ax.set_xlim(prop_xlim[prop])

        # Column titles
        if i == 0:
            ax.set_title(prop, fontsize=10)

        # Row labels
        if j == 0:
            ax.set_ylabel(texture, fontsize=8)
        else:
            ax.set_ylabel("")

        # Bottom x-labels only
        if i == len(texture_order) - 1:
            if prop in ["Kh", "Kv", "Ss"]:
                ax.set_xlabel(f"log10({prop})")
            else:
                ax.set_xlabel(prop)
        else:
            ax.set_xlabel("")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# Single legend

handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        prop={'size': 8}
    )

plt.tight_layout(rect=[0.0, 0.0, 1, 0.94])

fig.savefig(
    plt_dir / "hydraulic_property_histogram_grid.png",
    dpi=300,
    transparent=False
)

plt.show()