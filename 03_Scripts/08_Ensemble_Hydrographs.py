import matplotlib
matplotlib.use('TkAgg')
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

data_dir = Path('01_Data/')
f_dir = Path('06_Outputs/06_wtfx/')

#par_file   = f_dir / "svihm_ies.2.par.csv"
obs_file   = f_dir / "svihm_ies.3.obs.csv"

# Extract iteration number...
iter = obs_file.name.split('.')[1]

plt_dir = Path('05_Plots/') / f'{f_dir.name}_iter{iter}'
hds_plot_dir = plt_dir / 'hds_plots'
hds_plot_dir.mkdir(parents=True, exist_ok=True)

head_obs_file = data_dir / 'head_obs_master.csv'
str_obs_file = data_dir / 'streamflow_obs_std.csv'

plt.ioff()  # faster

# Plot settings
plt.rcParams.update({
    "font.family": "Bahnschrift",
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "grid.alpha": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 300,
})

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def parse_daily_date_from_obsnme(obsnme: str):
    """
    Expect patterns like 'FJ_19901001', 'AS_20050715', etc.
    Takes the substring after the first '_' and parses YYYYMMDD.
    """
    try:
        date_part = obsnme.split("_", 1)[1]
        # daily flows are stored as YYYYMMDD
        return pd.to_datetime(date_part, format="%Y%m%d")
    except Exception:
        return pd.NaT

#----------------------------------------------------------------------------------------------------------------------#

def parse_monthly_date(obsnme):
    # e.g., "FJ_1990-10_VOL"
    try:
        return pd.to_datetime(obsnme.split("_")[1], format="%Y-%m")
    except:
        return pd.NaT

#----------------------------------------------------------------------------------------------------------------------#

def parse_yearly_date(obsnme):
    # e.g., "FJ_1991_VOL"
    try:
        return pd.to_datetime(obsnme.split("_")[1], format="%Y")
    except:
        return pd.NaT


#----------------------------------------------------------------------------------------------------------------------#
# Load Data
#----------------------------------------------------------------------------------------------------------------------#

# Observed data sets with observations & std deviations
head_obs = pd.read_csv(head_obs_file)
head_obs["date"] = pd.to_datetime(head_obs["date"])
str_obs = pd.read_csv(str_obs_file)

# PEST iteration results
run_results = pd.read_csv(obs_file, dtype={"real_name": str}, index_col=['real_name'])

# Make column names case-insensitive
run_results.columns = run_results.columns.str.lower()
str_obs["obsnme_lower"] = str_obs["obsnme"].str.lower()
head_obs["obsnme_lower"] = head_obs["obsnme"].str.lower()


# Little trick to get FJ streamflow into two groups for better plots

# Split FJ daily flows at Oct 1, 2010
mask_fj = str_obs["obsgnme"] == "str_FJ"

# Extract date from obsnme (YYYYMMDD)
# (Safe here because only daily FJ records are in str_FJ)
str_obs.loc[mask_fj, "date_tmp"] = pd.to_datetime(
    str_obs.loc[mask_fj, "obsnme"].str.split("_").str[1],
    format="%Y%m%d"
)

# Split into two pseudo-groups
# cutoff = pd.Timestamp("2010-10-01")
# str_obs.loc[mask_fj & (str_obs["date_tmp"] < cutoff),  "obsgnme"] = "str_FJ_pre2010"
# str_obs.loc[mask_fj & (str_obs["date_tmp"] >= cutoff), "obsgnme"] = "str_FJ_post2010"

# Cleanup temp column
str_obs.drop(columns=["date_tmp"], inplace=True)

# A little reporting because it's hard to read through these files
print(f"HDS R2: {run_results.loc['base','r2_heads']}")
print(f"FJ lNSE: {run_results.loc['base','fj_nse']} lkge: {run_results.loc['base','fj_kge']}")
print(f"AS lNSE: {run_results.loc['base','as_nse']} lkge: {run_results.loc['base','as_kge']}")
print(f"BY lNSE: {run_results.loc['base','by_nse']} lkge: {run_results.loc['base','by_kge']}")
print(f"SK lNSE: {run_results.loc['base','sck_nse']} lkge: {run_results.loc['base','sck_kge']}")

#----------------------------------------------------------------------------------------------------------------------#
# Plot streamflow groups
#----------------------------------------------------------------------------------------------------------------------#

# Only use the "str_*" groups for now (ignore vol_* for this first pass)
str_groups = [g for g in str_obs["obsgnme"].unique() if str(g).startswith("str_")]

for g in str_groups:
    # Subset obs for this stream group
    sub = str_obs.loc[str_obs["obsgnme"] == g].copy()

    # Parse dates (daily streamflow)
    sub["date"] = sub["obsnme"].apply(parse_daily_date_from_obsnme)

    # Drop any rows where date parsing failed (shouldn't be many if at all)
    sub = sub.dropna(subset=["date"])

    # Restrict to obs that we actually have in the run_results columns
    common_obs = sorted(set(sub["obsnme_lower"]) & set(run_results.columns))
    if len(common_obs) == 0:
        print(f"[WARN] No matching obs in run_results for group {g}")
        continue

    sub = sub[sub["obsnme_lower"].isin(common_obs)].copy()

    # Sort by date for nice plotting
    sub = sub.sort_values("date")

    # Re-establish the column ordering to match the sorted dates
    obs_cols = sub["obsnme_lower"].tolist()

    # Extract simulated values for these obs (all realizations)
    sim = run_results[obs_cols]

    # X-axis
    x = sub["date"].values

    # Observed values and std dev
    y_obs  = 10**sub["obsval"].values
    y_std  = sub["standard_deviation"].values
    y_hi   = y_obs + 1.0 * y_std
    y_lo   = y_obs - 1.0 * y_std

    # Separate base realization from the rest
    if "base" in sim.index:
        base_sim = 10**sim.loc["base"].values
        ens_sim  = sim.drop(index="base")
    else:
        base_sim = None
        ens_sim  = sim

    n_ens = ens_sim.shape[0]

    # Make the plot
    if g == 'str_FJ':
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig, ax = plt.subplots(figsize=(12, 5))

    # 1) Ensemble members (thin gray lines)
    for rname, row in ens_sim.iterrows():
        ax.plot(x, 10**row.values, color="0.7", linewidth=0.5, alpha=0.6)
    if n_ens > 0:
        ax.plot([], [], color="0.7", linewidth=0.5, alpha=0.6,
                label=f"ensemble members (n={n_ens})")

    # 2) Base realization (thick black line)
    if base_sim is not None:
        ax.plot(x,base_sim,color="k",linewidth=1.0,label="base realization", zorder=10)

    # 3) Observed values (thick blue line)
    ax.plot(x,y_obs,color="C0",linewidth=1.0,label="observed", zorder=8)

    # 4) ±1σ dashed blue lines
    #ax.plot(x, y_hi, color="C0", linestyle="--", linewidth=0.5, alpha=0.8, label="+1σ")
    #ax.plot(x,y_lo,color="C0",linestyle="--",linewidth=0.5,alpha=0.8,label="-1σ")

    # Axes labels & title
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.set_ylabel("Streamflow (m³/d) (log scale)")  # or cfs
    ax.set_title(f"Streamflow group: {g}")

    # Improve layout & legend
    ax.legend(loc="best")
    #ax.set_ylim(bottom=0)
    ax.set_xlim(x.min(), x.max())
    ax.margins(x=0)
    fig.autofmt_xdate()
    plt.tight_layout()

    # Save
    out_name = f"{g}_streamflow_ensemble.png"
    out_path = plt_dir / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved {out_path}")

#------------------------------------------------------------#
# Volume groups
#------------------------------------------------------------#

vol_groups = [g for g in str_obs["obsgnme"].unique() if str(g).startswith("vol_")]

for g in vol_groups:
    sub = str_obs.loc[str_obs["obsgnme"] == g].copy()

    if g == "vol_FJ_month":
        sub["date"] = sub["obsnme"].apply(parse_monthly_date)
    elif g == "vol_FJ_year":
        sub["date"] = sub["obsnme"].apply(parse_yearly_date)
    else:
        print(f"[WARN] no parser for group {g}")
        continue

    sub = sub.dropna(subset=["date"])
    sub = sub.sort_values("date")

    # Match to simulation columns
    obs_cols = sorted(set(sub["obsnme_lower"]) & set(run_results.columns))
    if len(obs_cols) == 0:
        print(f"[WARN] no matching obs in run_results for {g}")
        continue

    sub = sub[sub["obsnme_lower"].isin(obs_cols)]
    obs_cols = sub["obsnme_lower"].tolist()  # preserve date order

    sim = run_results[obs_cols]

    # Observations
    x = sub["date"].values
    y_obs = sub["obsval"].values
    y_std = sub["standard_deviation"].values
    y_hi = y_obs + 1*y_std
    y_lo = y_obs - 1*y_std

    # Base vs ensemble
    if "base" in sim.index:
        base_sim = sim.loc["base"].values
        ens_sim = sim.drop(index="base")
    else:
        base_sim = None
        ens_sim = sim

    n_ens = ens_sim.shape[0]

    # Plot
    fig, ax = plt.subplots(figsize=(12,8))

    # All ensemble members
    for _, row in ens_sim.iterrows():
        ax.plot(x, row.values, color="0.7", lw=0.5, alpha=0.5)
    if n_ens > 0:
        ax.plot([], [], color="0.7", lw=0.5, alpha=0.5, label=f"ensemble (n={n_ens})")

    # Base member
    if base_sim is not None:
        ax.plot(x, base_sim, "k-", lw=2, label="base realization")

    # Observed
    ax.plot(x, y_obs, "C0-", lw=2, label="observed")

    # ±1σ
    #ax.plot(x, y_hi, "C0--", lw=1, alpha=0.8, label="+1σ")
    #ax.plot(x, y_lo, "C0--", lw=1, alpha=0.8, label="-1σ")

    ax.set_xlim(x.min(), x.max())
    ax.margins(x=0)

    ax.set_title(f"Volume Group: {g}")
    ax.set_ylabel("Volume (same units as obsval)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    out = plt_dir / f"{g}_ensemble_volume.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print(f"Saved {out}")

#----------------------------------------------------------------------------------------------------------------------#
# Plot heads
#----------------------------------------------------------------------------------------------------------------------#

# use only difference targets for time series
diff_df = head_obs.loc[head_obs["group"] == "hds_diff"].copy()
wells = sorted(diff_df["wellid"].dropna().unique())

# accumulate observed vs base simulated elevations for final scatter
scat_obs = []
scat_sim = []
scat_obs_uw = []
scat_sim_uw = []

for wid in tqdm(wells, total=len(wells), desc=f"Well:"):
    sub = diff_df.loc[diff_df["wellid"] == wid].copy()
    sub = sub.dropna(subset=["date"])
    sub = sub.sort_values("date")

    # match obs names to simulation columns (lowercase)
    obs_cols = sorted(set(sub["obsnme_lower"]) & set(run_results.columns))
    if len(obs_cols) == 0:
        print(f"[WARN] no matching hds_diff obs in run_results for well {wid}")
        continue

    sub = sub[sub["obsnme_lower"].isin(obs_cols)]
    obs_cols = sub["obsnme_lower"].tolist()  # preserve time order

    sim_diff = run_results[obs_cols]

    x = sub["date"].values
    y_obs_diff = sub["obval"].values
    y_std_diff = sub["stdev"].values
    w_obs = sub["weight"].values

    # ±1σ bands for differences
    y_hi_diff = y_obs_diff + 1.0 * y_std_diff
    y_lo_diff = y_obs_diff - 1.0 * y_std_diff

    # base vs ensemble for differences
    if "base" in sim_diff.index:
        base_diff = sim_diff.loc["base"].values
        ens_diff = sim_diff.drop(index="base")
    else:
        base_diff = None
        ens_diff = sim_diff
    n_ens = ens_diff.shape[0]

    # --- get average obs + sim for this well ---

    avg_row = head_obs.loc[(head_obs["group"] == "hds_avg") & (head_obs["wellid"] == wid)]
    if len(avg_row) == 0:
        print(f"[WARN] no hds_avg row for well {wid}")
        continue
    avg_row = avg_row.iloc[0]

    avg_val = avg_row["obval"]
    avg_std = avg_row["stdev"]
    avg_name = avg_row["obsnme"].lower()

    if avg_name not in run_results.columns:
        print(f"[WARN] avg obs '{avg_name}' not found in run_results for well {wid}")
        continue

    sim_avg_series = run_results[avg_name]

    if "base" in sim_avg_series.index:
        base_avg = sim_avg_series.loc["base"]
        ens_avg_series = sim_avg_series.drop(index="base")
    else:
        base_avg = None
        ens_avg_series = sim_avg_series

    # --- reconstruct elevations: avg + diff ---

    # observed elevations: scalar avg + diff series
    y_obs_elev = avg_val + y_obs_diff

    # base elevation time series
    base_elev = None
    if base_diff is not None and base_avg is not None:
        base_elev = base_avg + base_diff
        mask_scatter = (w_obs > 0) & np.isfinite(y_obs_elev) & np.isfinite(base_elev)
        # weighted (used in R²)
        mask_w = (w_obs > 0) & np.isfinite(y_obs_elev) & np.isfinite(base_elev)
        if mask_w.any():
            scat_obs.append(y_obs_elev[mask_w])
            scat_sim.append(base_elev[mask_w])
        # unweighted (plotted only)
        mask_uw = (w_obs == 0) & np.isfinite(y_obs_elev) & np.isfinite(base_elev)
        if mask_uw.any():
            scat_obs_uw.append(y_obs_elev[mask_uw])
            scat_sim_uw.append(base_elev[mask_uw])

    # ensemble elevation time series
    # ens_diff and ens_avg_series share the same index
    ens_elev_dict = {}
    for rname, row in ens_diff.iterrows():
        if rname in ens_avg_series.index:
            ens_elev_dict[rname] = ens_avg_series.loc[rname] + row.values

    # --- figure with 3 axes: diff (top), elev + avg (bottom row) ---

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2], width_ratios=[3, 1])

    ax_diff = fig.add_subplot(gs[0, :])
    ax_elev = fig.add_subplot(gs[1, 0])
    ax_avg = fig.add_subplot(gs[1, 1], sharey=ax_elev)

    # TOP: differences

    # ensemble diff
    for _, row in ens_diff.iterrows():
        ax_diff.plot(x, row.values, color="0.7", lw=0.5, alpha=0.5)
    if n_ens > 0:
        ax_diff.plot([], [], color="0.7", lw=0.5, alpha=0.5)

    # base diff
    if base_diff is not None:
        ax_diff.plot(x, base_diff, "k-", lw=2, label="base diff")

    # obs diff: filled for weight>0, hollow for weight==0
    mask_wt = w_obs > 0
    if mask_wt.any():
        ax_diff.plot(x[mask_wt], y_obs_diff[mask_wt], "o", ms=4, color="C0")
    if (~mask_wt).any():
        ax_diff.plot(x[~mask_wt], y_obs_diff[~mask_wt], "o", ms=4, mfc="none", mec="C0")

    # ±1σ bands for differences
    #ax_diff.plot(x, y_hi_diff, "C0--", lw=1, alpha=0.8, label="+1σ (diff)")
    #ax_diff.plot(x, y_lo_diff, "C0--", lw=1, alpha=0.8)

    ax_diff.set_title(f"Head differences: {wid}")
    ax_diff.set_ylabel("Head difference (m)")
    ax_diff.set_xlabel("Date")
    ax_diff.grid(alpha=0.2)

    # BOTTOM-LEFT: head elevations

    # ensemble elevations
    for rname, vals in ens_elev_dict.items():
        ax_elev.plot(x, vals, color="0.7", lw=0.5, alpha=0.5)
    if len(ens_elev_dict) > 0:
        ax_elev.plot([], [], color="0.7", lw=0.5, alpha=0.5)

    # base elevation
    if base_elev is not None:
        ax_elev.plot(x, base_elev, "k-", lw=2)

    # obs elevations
    if mask_wt.any():
        ax_elev.plot(x[mask_wt], y_obs_elev[mask_wt], "o", ms=4, color="C0")
    if (~mask_wt).any():
        ax_elev.plot(x[~mask_wt], y_obs_elev[~mask_wt], "o", ms=4, mfc="none", mec="C0")

    ax_elev.set_title(f"Head elevations: {wid}")
    ax_elev.set_ylabel("Head elevation (m)")
    ax_elev.set_xlabel("Date")
    ax_elev.grid(alpha=0.2)

    # BOTTOM-RIGHT: averages

    # arbitrary x-range for horizontal lines
    x0, x1 = 0.0, 1.0

    # ensemble avg heads
    if len(ens_avg_series) > 0:
        for val in np.atleast_1d(ens_avg_series.values):
            ax_avg.hlines(val, x0, x1, colors="0.7", lw=0.5, alpha=0.5)
        ax_avg.hlines(np.nan, x0, x1, colors="0.7", lw=0.5, alpha=0.5)

    # base avg head
    if base_avg is not None:
        ax_avg.hlines(base_avg, x0, x1, colors="k", lw=2, label="base avg")

    # observed avg + std dev (±1σ)
    ax_avg.hlines(avg_val, x0, x1, colors="C0", lw=2, label="obs avg")
    # if avg_std is not None and not np.isnan(avg_std):
    #     ax_avg.hlines(avg_val + 1*avg_std, x0, x1, colors="C0", linestyles="--", lw=1, alpha=0.8)
    #     ax_avg.hlines(avg_val - 1*avg_std, x0, x1, colors="C0", linestyles="--", lw=1, alpha=0.8)

    ax_avg.set_xlim(x0, x1)
    ax_avg.set_xticks([])
    ax_avg.set_xlabel("Head Average Target")
    ax_avg.grid(alpha=0.2)

    # shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', lw=0, markersize=5, label='obs'),
        #Line2D([0], [0], color='C0', ls="--", lw=1, label=f'obs ±1σ'),
        Line2D([0], [0], color='k', lw=2, label='base'),
        Line2D([0], [0], color='0.7', lw=1, label=f'ensembles (n={n_ens})')
    ]

    ax_diff.legend(legend_elements, ['obs', 'base', f'ensembles (n={n_ens})'],  # f'obs ±1σ'
                   loc="upper left", bbox_to_anchor=(0, 1.02), ncol=1)

    fig.autofmt_xdate()
    plt.tight_layout()

    out = hds_plot_dir / f"heads_{wid}_ensemble.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Final scatter: head obs vs sim (base realization), using reconstructed elevations (avg + diff)
# ----------------------------------------------------------------------------------------------------------------------

obs_all = np.concatenate(scat_obs)
sim_all = np.concatenate(scat_sim)

# R^2
r2 = r2_score(obs_all, sim_all)

fig, ax = plt.subplots(figsize=(6, 6))

obs_uw = np.concatenate(scat_obs_uw)
sim_uw = np.concatenate(scat_sim_uw)
# ax.scatter(obs_uw, sim_uw, s=8, facecolors="none",
#            edgecolors="0.5", alpha=0.6, label="Zero-weight Observations")

ax.scatter(obs_all, sim_all, s=8, alpha=0.35, label='Head Observations')

# 1:1 line
lo = np.nanmin([obs_all.min(), sim_all.min()])
hi = np.nanmax([obs_all.max(), sim_all.max()])
ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, zorder=0, label='1:1 line')

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.margins(x=0, y=0)

ax.set_xlabel("Observed head elevation (m)")
ax.set_ylabel("Simulated head elevation (m)")
ax.set_title(f"Head Obs vs Sim (base)   $R^2$ = {r2:.3f}")

ax.legend(
   loc="upper left",
    frameon=False
)

plt.tight_layout()

#out = plt_dir / "heads_obs_vs_sim_scatter_base_wzeroweight.png"
out = plt_dir / "heads_obs_vs_sim_scatter_base.png"
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved {out}  (n={len(obs_all)} points)")

# ----------------------------------------------------------------------------------------------------------------------
# Residual statistics summary (for manuscript reporting)
# ----------------------------------------------------------------------------------------------------------------------

res = sim_all - obs_all
abs_res = np.abs(res)

summary = {
    "n": len(res),
    "mean_error": np.mean(res),
    "median_error": np.median(res),
    "rmse": np.sqrt(np.mean(res**2)),
    "mae": np.mean(abs_res),
    "std_error": np.std(res),
    "p05": np.percentile(res, 5),
    "p25": np.percentile(res, 25),
    "p50": np.percentile(res, 50),
    "p75": np.percentile(res, 75),
    "p95": np.percentile(res, 95),
    "max_abs_error": np.max(abs_res)
}

print("\nResidual Summary (Simulated − Observed Head Elevation)")
print("-----------------------------------------------------")
for k, v in summary.items():
    if k == "n":
        print(f"{k:>15}: {v}")
    else:
        print(f"{k:>15}: {v: .3f}")