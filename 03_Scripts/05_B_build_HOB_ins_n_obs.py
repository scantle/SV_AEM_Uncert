import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import flopy
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

# Model Info
model_name = 'SVIHM'
xoff = 499977
yoff = 4571330
origin_date = pd.to_datetime('1990-9-30')

# Directories
pest_dir = Path("04_PEST_setup")   # TPL, INS
data_dir  = Path('01_Data')
model_dir = Path('02_Models/SVIHM_MF_working/MODFLOW')
plot_dir  = Path('05_Plots/Weights')
plot_dir.mkdir(exist_ok=True)

# Specific files
hob_cache = data_dir / 'hobs_df_cached.pkl'

# Observation metadata
vertical_well_pairs = [
    ('ST201', 'ST201_2'),
    ('ST786', 'ST786_2')
]
wt_dict = {'N2': ['after_date', pd.to_datetime('01/01/2019')],   # outlier removal
           'Q32': ['after_date', pd.to_datetime('02/10/2020')],  # outlier removal
           'ST170': ['rolling_median', 0.5, 90, 0.0],  # Smooth to remove high-frequency oscillations
           'ST192': ['rolling_median', 1.5, 365, 0.0],  # Smooth to remove high-frequency oscillations
           'ST690': ['rolling_median', 1.0, 120, 0.0],  # Smooth to remove high-frequency oscillations
           'ST794': ['rolling_median', 0.7,  90, 0.0],  # Smooth to remove high-frequency oscillations
           'ST888': ['rolling_median', 1.0,  60, 0.0],  # Smooth to remove high-frequency oscillations
           'ST987': ['rolling_median', 1.0, 180, 0.0],  # Smooth to remove high-frequency oscillations
           'ST655': ['after_date', pd.to_datetime('10/01/1990')],  # Too close to model edge
           'ST202': ['after_date', pd.to_datetime('10/01/1990')],  # Too close to model edge
           #'G40'  : ['after_date', pd.to_datetime('10/01/1990')],  # Too close to model edge (but let's try it)
           #'N15'  : ['after_date', pd.to_datetime('10/01/1990')],  # Too close to model edge (but let's try it)
          }

# Don't show hydrographs
plt.ioff()

#----------------------------------------------------------------------------------------------------------------------#
# Classes/Functions
#----------------------------------------------------------------------------------------------------------------------#

def hob_to_df(hob, origin_date, out_file=None):
    obs_records = []
    for hob_entry in tqdm(hob.obs_data, desc='HOB Entry', total=len(hob.obs_data)):
        for ts_data in hob_entry.time_series_data:
            # Extract individual time series values
            totim = ts_data[0]  # Absolute model time
            sp = ts_data[1]  # Stress period (1-based)
            ts = ts_data[2]  # Time step (not 0-indexed)
            obsval = ts_data[3]
            obsname = ts_data[4].decode("utf-8") if isinstance(ts_data[4], bytes) else ts_data[4]  # Ensure string format

            # Convert totim to actual observation date
            obs_date = origin_date + pd.DateOffset(days=totim)

            # Append all relevant data as a row in the list
            obs_records.append({
                "obsnme": obsname,
                "wellid": obsname.split('.')[0],
                "obsval": obsval,
                "row": hob_entry.row,
                "col": hob_entry.column,
                "lay": hob_entry.layer,
                "multilay": hob_entry.multilayer,
                "roff": hob_entry.roff,
                "coff": hob_entry.coff,
                "sp": sp,
                "ts": ts,
                "date": obs_date
            })

    # Convert list of dictionaries into a DataFrame
    obs_df = pd.DataFrame(obs_records)

    if out_file is not None:
        hob_out = pd.read_csv(out_file, sep='\\s+', skiprows=1, header=None, names=['simval','obsval','obsnme'])
        assert hob_out.shape[0] == obs_df.shape[0]
        obs_df = obs_df.merge(hob_out[['obsnme','simval']], on=['obsnme'])

    return obs_df

#----------------------------------------------------------------------------------------------------------------------#

def calculate_hob_weights(hobs_df, wt_dict, bas, out_dir=None, by_well=False, default_weight=1.0, min_points=3):
    """
    Modify the weights in hobs_df based on predefined rules for selected wells.

    Parameters
    ----------
    hobs_df : pandas.DataFrame
        DataFrame containing HOB observations with at least columns:
            - 'wellid': Well identifier
            - 'obsnme': Observation name (unique ID)
            - 'obsval': Observed value
            - 'date': Timestamp
            - 'wt': Weight (to be modified)
    wt_dict : dict
        Dictionary specifying weighting rules for hand-picked wells.
        Format: {'wellid': ['rule_type', value, weight]} where:
            - 'after_date': Zero weight for observations after `value` (a date).
            - 'rolling_median': Zero weight for deviations from rolling median by more than `value`.
    bas : flopy.modflow.ModflowBas
        MODFLOW Basic Package object, used to check inactive cells.
    out_dir : Path or str, optional
        Directory where hydrograph plots should be saved. (no plots if None)
    by_well: True/False, optional (default = False)
        default_weight is assigned by well instead of by observation, i.e., observation weights are balanced
        (normalized) across wells
    default_weight: float, optional (default = 1.0)
        Weight assigned to observations (if by_well is False) or by well (if by_well is True)
    min_points: integer, optional (default = 3)
        Minimum number of observations for a well to have a non-zero weight. count < min_points => wt = 0

    Returns
    -------
    hobs_df : pandas.DataFrame
        Updated DataFrame with modified weight values.
    """

    # Create a copy of hobs_df to avoid modifying in place
    hobs_df = hobs_df.copy()
    if by_well:
        hobs_df['wt'] = default_weight / hobs_df.groupby('wellid')['obsnme'].transform('count')
    else:
        hobs_df['wt'] = default_weight

    # Track wells that were modified for reporting
    too_few_points = []
    inactive_wells = []

    # Process hand-picked wells with specific rules
    for well in wt_dict.keys():
        well_df = hobs_df[hobs_df['wellid'] == well].copy()
        if well_df.shape[0]==0:
            raise RuntimeError(f"No data for well {well}")
        well_df.sort_values("date", inplace=True)
        deviation_threshold = None
        window=None

        rule_type = wt_dict[well][0]
        if rule_type == 'after_date':
            cutoff_date = wt_dict[well][1]
            well_df.loc[(well_df['date'] > cutoff_date), "wt"] = 0.0

            # Merge changes back into hobs_df
            hobs_df.loc[hobs_df["obsnme"].isin(well_df["obsnme"]), "wt"] = well_df["wt"]

        elif rule_type == 'rolling_median':
            deviation_threshold = wt_dict[well][1]
            window = wt_dict[well][2]  # Window for rolling median
            new_weight = wt_dict[well][3]  # Weight to assign to outliers

            # Compute rolling median
            well_df["rolling_median"] = well_df["obsval"].rolling(window=window, center=True, min_periods=15).median()

            # Identify outliers that deviate too far from the rolling median
            outlier_mask = (well_df["rolling_median"] - well_df["obsval"]) > deviation_threshold
            well_df.loc[outlier_mask, "wt"] = new_weight

            # Merge changes back into hobs_df
            hobs_df.loc[hobs_df["obsnme"].isin(well_df["obsnme"]), "wt"] = well_df["wt"]

        # Plot the hydrograph with weights for visualization
        if out_dir is not None:
            plot_hydrograph_weights(well_df, deviation_threshold, window, out_dir)

    # Process all wells for general conditions
    for wellid, well_df in hobs_df.groupby('wellid'):
        # Zero weights for wells with less than 3 data points
        if well_df.shape[0] < min_points:
            hobs_df.loc[hobs_df['wellid'] == wellid, "wt"] = 0.0
            too_few_points.append(wellid)

        # Zero weights for wells in inactive cells using BAS
        row, col = int(well_df.iloc[0]['row']), int(well_df.iloc[0]['col'])  # Assume row & col are consistent per well
        if bas.ibound[0, row, col] == 0:  # Check if cell is inactive (ibound=0)
            hobs_df.loc[hobs_df['wellid'] == wellid, "wt"] = 0.0
            inactive_wells.append(wellid)

    # Print a report of changes
    if too_few_points:
        print(f"Wells with too few points (set to wt=0): {too_few_points}")
    if inactive_wells:
        print(f"Wells in inactive cells (set to wt=0): {inactive_wells}")

    return hobs_df  # Return the updated DataFrame

#----------------------------------------------------------------------------------------------------------------------#

def build_head_obs_sets_weighted(
    df: pd.DataFrame,
    vertical_pairs: List[Tuple[str, str]],
    base_sigma: float = 1.0,
    value_col: str = "obsval",
    time_key: str = "date",
    weight_col: str = "wt",
) -> pd.DataFrame:
    """
    Build AVG_HEAD, DIFF_HEAD, VDIFF using only rows with wt>0 when computing per-well means.
    For wells with NO wt>0 rows, we still include them:
      - mean fallback = unweighted mean over all rows for that well
      - sigma_mean fallback = base_sigma / sqrt(n_all)
      - AVG_HEAD weight = 0 (since sumw == 0)
    Output columns: ['obsnme','group','obval','stdev','weight','wellid','date']
    """
    req = {"obsnme", "wellid", value_col, weight_col}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"build_head_obs_sets_weighted: missing {missing}")

    d = df.copy()

    # ---------- positive-weight subset ----------
    pos = d[d[weight_col] > 0].copy()

    # Weighted mean via np.average where wt>0 exist
    if not pos.empty:
        wmean_pos = (
            pos.groupby("wellid")
               .apply(lambda g: np.average(g[value_col].to_numpy(),
                                           weights=g[weight_col].to_numpy()))
               .rename("well_wmean")
               .reset_index()
        )

        def _sigma_mean(g):
            w = g[weight_col].to_numpy()
            sw = w.sum()
            if sw <= 0:
                return np.nan
            return base_sigma * np.sqrt((w ** 2).sum()) / sw

        sigm_pos = (
            pos.groupby("wellid")
               .apply(_sigma_mean)
               .rename("sigma_mean")
               .reset_index()
        )
        meta_pos = wmean_pos.merge(sigm_pos, on="wellid", how="left")
    else:
        # no positive-weight obs anywhere
        meta_pos = pd.DataFrame(columns=["wellid", "well_wmean", "sigma_mean"])

    # ---------- fallback stats for wells with no wt>0 ----------
    # unweighted mean over all rows
    fallback_mean = (
        d.groupby("wellid", as_index=False)[value_col]
         .mean()
         .rename(columns={value_col: "fallback_mean"})
    )
    # n_all for fallback sigma of the mean
    n_all = (
        d.groupby("wellid", as_index=False)[value_col]
         .size()
         .rename(columns={"size": "n_all"})
    )

    # ensure every well appears, take weighted stats if available; else fallback
    all_wells = d[["wellid"]].drop_duplicates()
    meta_complete = (all_wells
        .merge(meta_pos, on="wellid", how="left")
        .merge(fallback_mean, on="wellid", how="left")
        .merge(n_all, on="wellid", how="left")
    )

    # Fill mean with fallback where missing
    meta_complete["well_wmean"] = meta_complete["well_wmean"].fillna(meta_complete["fallback_mean"])

    # Fallback sigma_mean = base_sigma / sqrt(n_all)
    sigma_fallback = base_sigma / np.sqrt(np.maximum(meta_complete["n_all"].fillna(0).to_numpy(), 1))
    meta_complete["sigma_mean"] = meta_complete["sigma_mean"].fillna(pd.Series(sigma_fallback))

    # Drop helpers we no longer need
    meta_complete = meta_complete.drop(columns=["fallback_mean"])

    # Merge means/sigmas back to the long table
    d = d.merge(meta_complete[["wellid", "well_wmean", "sigma_mean"]], on="wellid", how="left", validate="many_to_one")

    # ---------- AVG_HEAD ----------
    sumw = d.groupby("wellid", as_index=False)[weight_col].sum().rename(columns={weight_col: "sumw"})
    avg_rows = meta_complete.merge(sumw, on="wellid", how="left")
    avg_rows["obsnme"] = avg_rows["wellid"] + "_AVG"
    avg_rows["group"]  = "hds_avg"
    avg_rows["obval"]  = avg_rows["well_wmean"]
    avg_rows["stdev"]  = avg_rows["sigma_mean"]
    # AVG weight: cap at 1.0; wells with sumw==0 get 0 weight
    avg_rows["weight"] = avg_rows["sumw"].clip(upper=1.0)
    avg_rows.loc[avg_rows["sumw"].fillna(0) <= 0, "weight"] = 0.0
    avg_rows["date"]   = pd.NaT
    avg_rows = avg_rows[["obsnme", "group", "obval", "stdev", "weight", "wellid", "date"]]
    avg_rows = avg_rows.loc[:, ~avg_rows.columns.duplicated()]

    # ---------- DIFF_HEAD ----------
    d["diff_from_mean"] = d[value_col] - d["well_wmean"]
    d["obsnme_dm"]      = d["obsnme"].astype(str) + "_DM"
    d["group_dm"]       = "hds_diff"
    # DIFF stdev: sqrt(base_sigma^2 + sigma_mean^2)
    d["stdev_dm"] = np.sqrt(base_sigma**2 + np.where(np.isnan(d["sigma_mean"]), 0.0, d["sigma_mean"]**2))

    diff_rows = d.rename(columns={
        "obsnme_dm": "obsnme",
        "group_dm" : "group",
        "stdev_dm" : "stdev",
        weight_col : "weight",
        "diff_from_mean": "obval",
    })[["obsnme", "group", "obval", "stdev", "weight", "wellid", time_key]].copy()
    diff_rows = diff_rows.rename(columns={time_key: "date"})
    diff_rows = diff_rows.loc[:, ~diff_rows.columns.duplicated()]

    # ---------- VDIFF ----------
    vdiff_list = []
    for top_well, bot_well in vertical_pairs:
        top = d[d["wellid"] == top_well][[time_key, value_col, weight_col]].rename(
            columns={value_col: "top_val", weight_col: "wt_top"}
        )
        bot = d[d["wellid"] == bot_well][[time_key, value_col, weight_col]].rename(
            columns={value_col: "bot_val", weight_col: "wt_bot"}
        )
        m = pd.merge(top, bot, on=time_key, how="inner")
        if m.empty:
            continue
        m["obval"]  = m["top_val"] - m["bot_val"]
        m["weight"] = m[["wt_top", "wt_bot"]].min(axis=1)  # may be zero; fine
        m["stdev"]  = base_sigma * np.sqrt(2.0)
        m = m.reset_index(drop=True)
        m["obsnme"] = [f"{top_well}_VD.{i}" for i in m.index]
        m["group"]  = "hds_vdiff"
        m["wellid"] = top_well
        m = m.rename(columns={time_key: "date"})
        vdiff_list.append(m[["obsnme", "group", "obval", "stdev", "weight", "wellid", "date"]].copy())

    if vdiff_list:
        vdiff_rows = pd.concat(vdiff_list, ignore_index=True)
        vdiff_rows = vdiff_rows.loc[:, ~vdiff_rows.columns.duplicated()]
    else:
        vdiff_rows = pd.DataFrame(columns=["obsnme", "group", "obval", "stdev", "weight", "wellid", "date"])

    # ---------- combine ----------
    out = pd.concat([avg_rows, diff_rows, vdiff_rows], ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()]
    return out

#----------------------------------------------------------------------------------------------------------------------#

def plot_hydrograph_weights(df, deviation_threshold=None, window=None, out_dir=None):
    """
    Plot a hydrograph for a single well, coloring points by weight,
    with a rolling median and a shaded deviation threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns:
            - 'wellid': Well identifier
            - 'date': pandas datetime
            - 'obsval': Observed head (float)
            - 'weight': Weight (float) for each observation
    deviation_threshold : float, optional
        The threshold for outlier detection (default is None)
    window : float, optional
        Window used for rolling median (default is None).
    out_dir : Path or str
        Directory where plots will be saved.
    """

    if window is not None:

        # Define upper and lower bounds for the shaded region
        df["upper_bound"] = df["rolling_median"]  # + deviation_threshold
        df["lower_bound"] = df["rolling_median"] - deviation_threshold

        # Create figure
        plt.figure(figsize=(10, 5))

        # Plot rolling median as a solid line
        plt.plot(df["date"], df["rolling_median"], color="grey", linewidth=2, linestyle='--', label=f"Rolling Median, {window} days")

        # Fill between the deviation threshold
        plt.fill_between(df["date"], df["lower_bound"], df["upper_bound"], color="lightgray", alpha=0.5, label="Deviation Threshold")

    # Plot scatter where color is determined by weight (highlighting ignored observations)
    scatter = plt.scatter(df["date"], df["obsval"], c=df["wt"], cmap="coolwarm", edgecolor="k", alpha=0.8)

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Observed Head (m)")
    plt.title(f"Hydrograph for well {df['wellid'].iloc[0]}")

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add legend
    if window is not None: plt.legend()

    # Out
    if out_dir is not None:
        if window is not None:
            plot_filename = out_dir / f"{df['wellid'].iloc[0]}_{window}window_weight_hydrograph.png"
        else:
            plot_filename = out_dir / f"{df['wellid'].iloc[0]}_weight_hydrograph.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.clf()

#----------------------------------------------------------------------------------------------------------------------#
# Read HOBS
#----------------------------------------------------------------------------------------------------------------------#

# Setup GW model
gwf = flopy.modflow.Modflow.load((model_name + '.nam'), version='mfnwt', load_only=['dis','bas6'], model_ws=model_dir)

# Read in groundwater observations
if hob_cache.exists():
    # load the cached DataFrame instead of re‐reading the HOB
    print("Loading cached hobs_df from", hob_cache)
    hobs_df = pd.read_pickle(hob_cache)
else:
    hob_file = model_dir / "svihm.hob"
    print('Reading Hobs... (slow)')
    hob = flopy.modflow.ModflowHob.load(hob_file, model=gwf)
    print('Hobs read.')
    hobs_df = hob_to_df(hob, origin_date)
    hobs_df.to_pickle(hob_cache)
    print("Cached hobs_df to", hob_cache)

# Remove outlier data
hobs_w = calculate_hob_weights(
    hobs_df,
    wt_dict=wt_dict,
    bas=gwf.bas6,
    out_dir=plot_dir,
    by_well=False,
    default_weight=1.0,
    min_points=3,
)

# Get final obs/weights/std devs for PEST
obs_master = build_head_obs_sets_weighted(
    hobs_w,
    vertical_pairs=vertical_well_pairs,
    base_sigma=1.0,          # your assumed single-measurement std dev (m)
    value_col="obsval",
    time_key="date",
    weight_col="wt",
)

# Add in our r^2 dummy observation
r2_row = pd.DataFrame([{
    "obsnme": "R2_HEADS",
    "group": "STAT",
    "obval": 0.0,
    "stdev": 1.0,
    "weight": 0.0,
    "wellid": "NA",
    "date": pd.NaT
}])
obs_master = pd.concat([obs_master, r2_row], ignore_index=True)

out_csv = (data_dir / "head_obs_master.csv")
obs_master.sort_values(["group", "wellid", "obsnme"]).to_csv(out_csv, index=False)
print(f"Wrote observation definition CSV -> {out_csv}")

#----------------------------------------------------------------------------------------------------------------------#
# Instruction File (INS) Creation
#----------------------------------------------------------------------------------------------------------------------#

ins_path = pest_dir / "head_obs_reader.ins"
ordered = obs_master.sort_values(["group", "wellid", "obsnme"])["obsnme"].tolist()
with open(ins_path, "w") as f:
    f.write("pif #\n")
    for name in ordered:
        f.write(f"l1 w !{name}!\n")
print(f"Wrote INS file -> {ins_path}")