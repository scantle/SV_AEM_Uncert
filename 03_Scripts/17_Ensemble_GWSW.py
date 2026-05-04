import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import geopandas as gpd
import flopy
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

origin_date = pd.to_datetime('1990-09-30')

# Parent directory containing ensemble LST outputs
ens_dir = Path('./06_Outputs/06_wtfx/BUD_SFRBUD_ensemble/')

# Reach shapefile
sfr_shpfile = Path('./01_Data/GIS/sfr_properties.shp')

out_dir = Path('./05_Plots/modflow_sfr_budget_uncertainty')
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------------------------------------------------- #

def make_seg_rch(seg, rch):
    return seg.astype(int).astype(str) + "_" + rch.astype(int).astype(str)

def signed_agreement_from_delta(delta: pd.Series) -> float:
    return np.sign(delta).mean()

# -------------------------------------------------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------------------------------------------------- #

print('Read SFR shapefile...')
sfr_shp = gpd.read_file(sfr_shpfile)

print("Loading ensemble realization net streamflow...")
out_files = sorted(ens_dir.glob("ftx_*.out"))

fsize = np.zeros(len(out_files))

builder = []
for i, f in tqdm(enumerate(out_files), desc="Processing SFR realizations", total=len(out_files)):
    if f.stat().st_size < 50679200:
        raise ValueError(f"File {i} is incomplete: {f}")
    sfr_out = pd.read_csv(f, delimiter='\\s+')
    if i==0:
        sfr_out[f'sptotal_{i}'] = sfr_out['AvgSPAqFlow'] * sfr_out['maxTS']
        #sfr_acc = sfr_out.copy()
        builder.append(sfr_out.copy())
    else:
        sfr_out[f'sptotal_{i}'] = sfr_out['AvgSPAqFlow'] * sfr_out['maxTS']
        #sfr_acc = pd.concat([sfr_acc, sfr_out[f'sptotal_{i}']], axis=1)
        builder.append(sfr_out[f'sptotal_{i}'])

sfr_acc = pd.concat(builder, axis=1).copy()

# Add some date-related columns
sfr_acc["Date"] = flopy.utils.totim_to_datetime(sfr_acc["Time"], start=origin_date)
sfr_acc['Month'] = sfr_acc['Date'].dt.month

# Get unique seg_rch col:
sfr_acc["seg_rch"] = make_seg_rch(sfr_acc["Seg"], sfr_acc["Reach"])
# and for shapefile
sfr_shp["seg_rch"] = make_seg_rch(sfr_shp["segment"], sfr_shp["reach"])

# -------------------------------------------------------------------------------------------------------------------- #
# Summarize across realizations
# -------------------------------------------------------------------------------------------------------------------- #

sp_cols = [c for c in sfr_acc.columns if c.startswith("sptotal_")]

# True reach identifiers/reference info: keep one copy only
reach_ref = (
    sfr_acc.groupby("seg_rch")
    .agg(
        Seg=("Seg", "first"),
        Reach=("Reach", "first"))
    .reset_index()
)

# Full-period totals by reach, summing only realization columns
total_sum = (
    sfr_acc.groupby("seg_rch")[sp_cols]
    .sum()
    .reset_index()
    .merge(reach_ref, on="seg_rch", how="left")
)

# Summer subset
sfr_summer = sfr_acc[sfr_acc["Month"].isin([7, 8, 9])].copy()

summer_sum = (
    sfr_summer.groupby("seg_rch")[sp_cols]
    .sum()
    .reset_index()
    .merge(reach_ref, on="seg_rch", how="left")
)

for df in [total_sum, summer_sum]:
    # Summarize ensemble uncertainty across realizations for each reach
    df["ens_mean"] = df[sp_cols].mean(axis=1, skipna=True)
    df["ens_std"]  = df[sp_cols].std(axis=1, skipna=True, ddof=1)
    df["ens_p05"]  = df[sp_cols].quantile(0.05, axis=1)
    df["ens_p95"]  = df[sp_cols].quantile(0.95, axis=1)
    df['signed_agreement'] = df[sp_cols].apply(signed_agreement_from_delta, axis=1)

# Merge to shapefile
sfr_total_map = sfr_shp.merge(total_sum, on="seg_rch", how="left")
sfr_summer_map = sfr_shp.merge(summer_sum, on="seg_rch", how="left")

# -------------------------------------------------------------------------------------------------------------------- #
# Plot 1: Map of mean exchange
# -------------------------------------------------------------------------------------------------------------------- #

absmax = np.nanmax(np.abs(sfr_total_map["ens_std"].values))

fig, ax = plt.subplots(figsize=(8, 10))
sfr_total_map.plot(
    column="ens_std",
    ax=ax,
    cmap="coolwarm",
    legend=True,
    vmin=-absmax, vmax=absmax,
    missing_kwds={"color": "lightgray", "label": "No data"},
)
ax.set_title("Ensemble Mean Net Stream–Aquifer Exchange by Reach\nCumulative Total Across the Full Simulation Period")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(out_dir / "sfr_mean_exchange_map.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Plot 2: Map of signed agreement (Total)
# -------------------------------------------------------------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(8, 10))
sfr_total_map.plot(
    column="signed_agreement",
    ax=ax,
    cmap="RdBu",
    legend=True,
    missing_kwds={"color": "lightgray", "label": "No data"},
)
ax.set_title("Sign Agreement in Net Stream–Aquifer Exchange by Reach\nTotal Exchange Over the Full Simulation Period")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(out_dir / "sfr_total_signed_agreement_map.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Plot 3: Map of signed agreement (Total)
# -------------------------------------------------------------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(8, 10))
sfr_summer_map.plot(
    column="signed_agreement",
    ax=ax,
    cmap="RdBu",
    legend=True,
    missing_kwds={"color": "lightgray", "label": "No data"},
)
ax.set_title("Sign Agreement in Net Stream–Aquifer Exchange by Reach\nJuly–September Total Exchange Over the Full Simulation Period")
ax.set_axis_off()
fig.tight_layout()
fig.savefig(out_dir / "sfr_summer_signed_agreement_map.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Segment-level total exchange distributions across realizations
# -------------------------------------------------------------------------------------------------------------------- #

# Convert to million m3
total_sum_mmm = total_sum.copy()
total_sum_mmm[sp_cols] = total_sum_mmm[sp_cols] / 1e6

# Drop artificial segments
total_sum_mmm = total_sum_mmm[total_sum_mmm['Seg']<=30]

segment_sum = (
    total_sum_mmm.groupby("Seg")[sp_cols]
    .sum()
    .reset_index()
)

segment_plot = segment_sum[["Seg"]].copy()
segment_plot["median"] = segment_sum[sp_cols].median(axis=1)
segment_plot["mean"] = segment_sum[sp_cols].mean(axis=1)
segment_plot["p05"] = segment_sum[sp_cols].quantile(0.05, axis=1)
segment_plot["p25"] = segment_sum[sp_cols].quantile(0.25, axis=1)
segment_plot["p75"] = segment_sum[sp_cols].quantile(0.75, axis=1)
segment_plot["p95"] = segment_sum[sp_cols].quantile(0.95, axis=1)
segment_plot["directional_agreement"] = segment_sum[sp_cols].apply(
    signed_agreement_from_delta, axis=1
)

# sort by segment number, or use median if you prefer
segment_plot = segment_plot.sort_values("Seg").reset_index(drop=True)
segment_plot["plot_y"] = np.arange(len(segment_plot))

# -------------------------------------------------------------------------------------------------------------------- #
# Dissolve reach shapefile to one geometry per segment
# -------------------------------------------------------------------------------------------------------------------- #

segment_map = (
    sfr_shp[["segment", "geometry"]]
    .rename(columns={"segment": "Seg"})
    .dissolve(by="Seg", as_index=False)
)

segment_map = segment_map.merge(
    segment_plot[["Seg", "directional_agreement"]],
    on="Seg",
    how="left"
)

# Label points
segment_labels = segment_map.copy()
segment_labels["label_point"] = segment_labels.geometry.representative_point()
segment_labels["label_x"] = segment_labels["label_point"].x
segment_labels["label_y"] = segment_labels["label_point"].y

# -------------------------------------------------------------------------------------------------------------------- #
# Combined figure: segment distribution + segment map
# -------------------------------------------------------------------------------------------------------------------- #

cmap = plt.get_cmap("PiYG")
norm = plt.Normalize(-1, 1)
colors = cmap(norm(segment_plot["directional_agreement"].values))

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(15, max(8, len(segment_plot) * 0.28)),
    gridspec_kw={"width_ratios": [1.15, 1.0]}
)

# ----------------------------------------
# Left panel: distribution strip plot
# ----------------------------------------

for y, p05, p25, p75, p95, med, c in zip(
    segment_plot["plot_y"],
    segment_plot["p05"],
    segment_plot["p25"],
    segment_plot["p75"],
    segment_plot["p95"],
    segment_plot["median"],
    colors,
):
    ax1.hlines(y=y, xmin=p05, xmax=p95, color=c, linewidth=2.0, zorder=1)
    ax1.hlines(y=y, xmin=p25, xmax=p75, color=c, linewidth=5.0, zorder=2)
    ax1.plot(med, y, marker="|", markersize=5, color="honeydew", linestyle="none", zorder=3)

ax1.axvline(0, color="red", linewidth=1.0, zorder=0)

ax1.set_title(
    "Distribution of Net Stream–Aquifer Exchange by Segment\n"
    "Total Exchange Over the Full Simulation Period"
)
ax1.set_xlabel("Net stream–aquifer exchange total ($Mm^3$)")
ax1.set_ylabel("Segment")
ax1.set_yticks(segment_plot["plot_y"])
ax1.set_yticklabels(segment_plot["Seg"].astype(int))
ax1.grid(axis="x", linestyle="--", alpha=0.4)

# ----------------------------------------
# Right panel: segment map
# ----------------------------------------

segment_map.plot(
    column="directional_agreement",
    ax=ax2,
    cmap="PiYG",
    vmin=-1,
    vmax=1,
    legend=False,
    edgecolor="black",
    linewidth=0.4,
    missing_kwds={"color": "lightgray", "label": "No data"},
)

LABEL_X_OFFSET = 500   # map units; probably meters in your CRS
LABEL_Y_OFFSET = 0     # optional vertical shift

for _, row in segment_labels.iterrows():
    ax2.text(
        row["label_x"] - LABEL_X_OFFSET,
        row["label_y"] + LABEL_Y_OFFSET,
        str(int(row["Seg"])),
        fontsize=7,
        ha="right",
        va="center",
        color="black",
        zorder=5,
    )

ax2.set_title("Segment Directional Agreement")
ax2.set_axis_off()

# Shared colorbar in a manual location
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# [left, bottom, width, height] in figure coordinates
cax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Directional agreement")

fig.tight_layout()
fig.savefig(out_dir / "sfr_total_segment_distribution_and_map.png", dpi=300)
plt.show()