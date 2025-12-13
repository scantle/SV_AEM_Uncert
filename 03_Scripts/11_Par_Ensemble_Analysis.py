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
title_font = FontProperties(family="DM Serif Text", size=14)
label_font = {"family": "Montserrat", "size": 11}
tick_font  = FontProperties(family="Montserrat", size=9)

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

#----------------------------------------------------------------------------------------------------------------------#
# Plots
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Phi Reduction Across Iterations
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)

ax.errorbar(
    phi_stats["iteration"],
    phi_stats["mean"],
    yerr=phi_stats["std"],
    fmt="o-",
    capsize=4,
    linewidth=1.3,
    markersize=4,
)

ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\log_{10}(\phi)$")
ax.set_title("Ensemble objective function across iterations", fontproperties=title_font)

# Ticks font
# for label in ax.get_xticklabels():
#     label.set_fontproperties(tick_font)
# for label in ax.get_yticklabels():
#     label.set_fontproperties(tick_font)

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
# Histogram of kV_mult

# Intersect members that exist in both iterations
#common_members = par_iter0.index.intersection(par_results.index)

kv_cols = [c for c in par_iter0.columns if c.startswith("kv_mult")]

kv0 = par_iter0.loc['base', kv_cols].to_numpy().ravel()

# note: this is from a base with parameters from a later iteration. Originally it was all zeros:
# so we're going to replace it with that
kv0.fill(0)

kv3 = par_results.loc['base', kv_cols].to_numpy().ravel()

kv_df = pd.DataFrame({
    "value": np.concatenate([kv0, kv3]),
    "iteration_label": (["Iteration 0"] * len(kv0)) + (["Iteration 3"] * len(kv3)),
})

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(
    data=kv_df,
    x="value",
    hue="iteration_label",
    bins=50,
    kde=True,
    element="step",
    ax=ax,
)

ax.set_xlabel("kv_mult Values")
ax.set_ylabel("Count")
ax.set_title("Distribution of kv_mult Values")

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
