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

str_obs_file = data_dir / 'streamflow_obs_std.csv'

#plt.ioff()  # faster

# Plot settings
plt.rcParams.update({
    "font.family": "DM Serif Text"
})

# Plot Style
sns.set_theme(style="whitegrid")

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
str_obs = pd.read_csv(str_obs_file)

# PEST iteration results
run_results = pd.read_csv(obs_file, dtype={"real_name": str}, index_col=['real_name'])

# Make column names case-insensitive
run_results.columns = run_results.columns.str.lower()
str_obs["obsnme_lower"] = str_obs["obsnme"].str.lower()

# Little trick to get FJ streamflow into two groups for better plots

# Split FJ daily flows at Oct 1, 2010
mask_fj = str_obs["obsgnme"] == "str_FJ"

# Extract date from obsnme (YYYYMMDD)
# (Safe here because only daily FJ records are in str_FJ)
str_obs.loc[mask_fj, "date_tmp"] = pd.to_datetime(
    str_obs.loc[mask_fj, "obsnme"].str.split("_").str[1],
    format="%Y%m%d"
)

# Cleanup temp column
str_obs.drop(columns=["date_tmp"], inplace=True)

# A little reporting because it's hard to read through these files
print(f"FJ lNSE: {run_results.loc['base','fj_nse']} lkge: {run_results.loc['base','fj_kge']}")
print(f"AS lNSE: {run_results.loc['base','as_nse']} lkge: {run_results.loc['base','as_kge']}")
print(f"BY lNSE: {run_results.loc['base','by_nse']} lkge: {run_results.loc['base','by_kge']}")
print(f"SK lNSE: {run_results.loc['base','sck_nse']} lkge: {run_results.loc['base','sck_kge']}")

#----------------------------------------------------------------------------------------------------------------------#
# Plot streamflow groups
#----------------------------------------------------------------------------------------------------------------------#

# Subset obs for this stream group
sub = str_obs.loc[str_obs["obsgnme"] == 'str_FJ'].copy()

# Parse dates (daily streamflow)
sub["date"] = sub["obsnme"].apply(parse_daily_date_from_obsnme)

# Sort by date for nice plotting
sub = sub.sort_values("date")

# Re-establish the column ordering to match the sorted dates
obs_cols = sub["obsnme_lower"].tolist()

# Extract simulated values for these obs (all realizations)
sim = run_results[obs_cols]

# X-axis
x = sub["date"].values

# Observed values and std dev
y_obs  = sub["obsval"].values
y_std  = sub["standard_deviation"].values
y_hi   = y_obs + 1.0 * y_std
y_lo   = y_obs - 1.0 * y_std

# Separate base realization from the rest
base_sim = sim.loc["base"].values
ens_sim  = sim.drop(index="base")
n_ens = ens_sim.shape[0]

# Make the plot
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_alpha(0)

# 1) Ensemble members (thin gray lines)
for rname, row in ens_sim.iterrows():
    ax.plot(x, row.values, color="0.7", linewidth=0.5, alpha=0.5)
if n_ens > 0:
    ax.plot([], [], color="0.7", linewidth=0.5, alpha=0.5,
            label=f"ensemble members (n={n_ens})")

# 2) Base realization (thick black line)
if base_sim is not None:
    ax.plot(x,base_sim,color="k",linewidth=0.9,label="base realization", zorder=10)

# 3) Observed values (thick blue line)
ax.plot(x,y_obs,color="C0",linewidth=1.0,label="observed", zorder=8)

# # 4) ±1σ dashed blue lines
# ax.plot(x, y_hi, color="C0", linestyle="--", linewidth=0.5, alpha=0.8, label="+1σ")
# ax.plot(x,y_lo,color="C0",linestyle="--",linewidth=0.5,alpha=0.8,label="-1σ")

# Axes labels & title
ax.set_xlabel("Date")
ax.set_ylabel("Streamflow (log10)")

# Improve layout & legend
ax.legend(loc="best")
#ax.set_ylim()
ax.set_axisbelow(True)
ax.grid(color="0.85")
ax.set_xlim(x.min(), x.max())
fig.autofmt_xdate()
plt.tight_layout()

# Save
out_name = "str_FJ_streamflow_ensemble_full.png"
out_path = plt_dir / out_name
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved {out_path}")
