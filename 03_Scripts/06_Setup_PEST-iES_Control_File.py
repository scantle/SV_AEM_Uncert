import os
import numpy as np
import pandas as pd
import pyemu
from pyemu.utils import geostats
import flopy
from tqdm import tqdm
from pathlib import Path
import shutil
import glob

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

# Model Info
model_name = 'SVIHM'
xoff = 499977
yoff = 4571330
origin_date = pd.to_datetime('1990-9-30')

# Directories
orig_dir = os.getcwd()
pest_dir = Path("04_PEST_setup")   # TPL, INS
work_dir  = Path("C:/Projects/SVIHM/2025_R2P_PEST_Calib")  # build folder
data_dir  = Path('01_Data')
exe_dir   = Path('02_Models/Bin')
scpt_dir  = Path('03_Scripts/')
model_dir = Path('02_Models/SVIHM_MF_working')
mf_dir    = model_dir / 'MODFLOW'
pest_exe_dir = Path('C:/Users/lelan/Documents/Models/pest17')
pestpp_exe = Path('C:/Users/lelan/Documents/Models/pestpp-5.2.16-iwin/bin/pestpp-ies.exe')

# Out
pst_file = 'svihm_ies.pst'
ensemble_size = 50

# horizontal / vertical ranges (meters) and sills by PP family - Exponential Variogram
# must match 04_Write_Initial_PilotPoint_Files.py
# (but here we have a vertical to constrain variation between layers)
# (I've used the resistivity vertical range)
PP_GS = {
    "aem_var":   dict(ah=40*2,   av=78.4,   sill=1.0, nugget=0.0),
    "lth_var":   dict(ah=259*1,  av=78.4,   sill=1.0, nugget=0.0),
    "kv_mult":   dict(ah=93*2,   av=78.4,   sill=1.0, nugget=0.0),
    # 'scale_*' handled specially below per texture with same ranges
    "scale":     dict(ah=2317*2, av=78.4,   sill=1.0, nugget=0.0),
}

# set seed
np.random.seed(667)

#----------------------------------------------------------------------------------------------------------------------#
# Classes/Functions
#----------------------------------------------------------------------------------------------------------------------#

def get_model_zeds_at_locs(mg, loc_df, x_col='X', y_col='Y',layer_col='Layer'):
    top  = gwf.dis.top.array
    botm = gwf.dis.botm.array

    # map XY to row/col
    rows, cols = zip(*[mg.intersect(x, y) for x, y in loc_df[[x_col, y_col]].to_numpy()])
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    lays = loc_df[layer_col].to_numpy().astype(int)

    # midpoint elevation per layer
    zmid = np.where(
        lays == 0,
        0.5 * (top[rows, cols] + botm[0, rows, cols]),
        0.5 * (botm[lays - 1, rows, cols] + botm[lays, rows, cols])
    )
    return zmid

#----------------------------------------------------------------------------------------------------------------------#

def exp_cov3d(names, x, y, z, ah, av, sill=1.0, nugget=0.0):
    """
    3D anisotropic exponential covariance. Since pyemu can't handle 3d!
    Distance d = sqrt((dx/ah)^2 + (dy/ah)^2 + (dz/av)^2).
    Cov = sill * exp(-d), with optional hard taper at d > cutoff_mult.
    ah,av are the exponential SCALE parameters (not practical ranges).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    n = len(names)
    dx = (x[:,None] - x[None,:]) / ah
    dy = (y[:,None] - y[None,:]) / ah
    dz = (z[:,None] - z[None,:]) / av
    d  = np.sqrt(dx*dx + dy*dy + dz*dz)

    C = sill * np.exp(-d)

    # diagonal variance (include nugget on diag)
    np.fill_diagonal(C, sill + nugget)
    return pyemu.Cov(x=C, names=list(names))

#----------------------------------------------------------------------------------------------------------------------#

def build_full_cov_with_pp_blocks(pst, pp_pars, sigma_range=4.0):
    """
    pst: pyemu.Pst
    pp_pars: DataFrame indexed by parnme with columns ['pargp','X','Y','Z','Layer']
    returns: pyemu.Cov covering ALL adjustable params, with PP sub-blocks filled by 3D geostats
    """
    # 1) base diagonal from bounds for ALL params
    full_cov = pyemu.Cov.from_parameter_data(pst, sigma_range=sigma_range)
    F = full_cov.as_2d
    order = list(full_cov.row_names)

    # Fill in those diagonals from pilot point covariances
    families = ["aem_var","lth_var","kv_mult"] + sorted({g.split("_L")[0] for g in pp_pars["pargp"] if g.startswith("scale_")})
    for fam in families:
        sub = pp_pars[pp_pars["pargp"].str.startswith(fam)]
        if sub.empty:
            continue
        if fam.startswith("scale_"):
            cfg = PP_GS['scale']
        else:
            cfg = PP_GS[fam]
        C = exp_cov3d(names=sub.index.values,
                      x=sub["X"].values, y=sub["Y"].values, z=sub["Z"].values, ah=cfg["ah"], av=cfg["av"])
        # inject block
        idx = [order.index(nm) for nm in C.row_names]
        F[np.ix_(idx, idx)] = C.as_2d

    # 3) return as pyemu.Cov (same ordering as pst)
    return pyemu.Cov(x=F, names=order)

#----------------------------------------------------------------------------------------------------------------------#

def t2p_par2par_frompar(t2p_parameters):
    """
    Replicates the chain multiplications from the par2par file and
    returns a DataFrame of final parameter values for each texture.

    Parameters
    ----------
    t2p_parameters : pd.DataFrame
        A DataFrame with index = parameter names, and columns:
        ['parval1', 'scale', 'offset'] as written by PEST.

    Returns
    -------
    pd.DataFrame
        Final parameter values by texture class (FF, MF, SC, MC, VC),
        with columns: Kmin, Aniso, Ss, Sy.
    """
    # Ensure index is lowercase
    df = t2p_parameters.copy()
    df.index = df.index.str.lower()

    # Compute the true parameter values: parval1 * scale + offset
    pvals = df['parval1'] * df['scale'] + df['offset']

    # --- Kmin chain ---
    k_ff = pvals['kminff1']
    k_mf = k_ff * pvals['kminmf1_m']
    k_sc = k_mf * pvals['kminsc1_m']
    k_mc = k_sc * pvals['kminmc1_m']
    k_vc = k_mc * pvals['kminvc1_m']

    # --- Aniso chain ---
    an_vc = pvals['anisovc1']
    an_mc = an_vc * pvals['anisomc1_m']
    an_sc = an_mc * pvals['anisosc1_m']
    an_mf = an_sc * pvals['anisomf1_m']
    an_ff = an_mf * pvals['anisoff1_m']

    # --- Ss chain ---
    ss_ff = pvals['ssff1']
    ss_mf = ss_ff * pvals['ssmf1_m']
    ss_sc = ss_mf * pvals['sssc1_m']
    ss_mc = ss_sc * pvals['ssmc1_m']
    ss_vc = ss_mc * pvals['ssvc1_m']

    # --- Sy chain ---
    sy_sc = pvals['sysc1']
    sy_mf = sy_sc * pvals['symf1_m']
    sy_ff = sy_mf * pvals['syff1_m']
    sy_mc = sy_sc * pvals['symc1_m']
    sy_vc = sy_mc * pvals['syvc1_m']

    # Final values by texture
    final_vals = {
        "FF": {"Kmin": k_ff, "Aniso": an_ff, "Ss": ss_ff, "Sy": sy_ff},
        "MF": {"Kmin": k_mf, "Aniso": an_mf, "Ss": ss_mf, "Sy": sy_mf},
        "SC": {"Kmin": k_sc, "Aniso": an_sc, "Ss": ss_sc, "Sy": sy_sc},
        "MC": {"Kmin": k_mc, "Aniso": an_mc, "Ss": ss_mc, "Sy": sy_mc},
        "VC": {"Kmin": k_vc, "Aniso": an_vc, "Ss": ss_vc, "Sy": sy_vc},
    }

    df_out = pd.DataFrame(final_vals).T[["Kmin", "Aniso", "Ss", "Sy"]]
    df_out['Ss'] = df_out['Ss'].apply(lambda x: np.format_float_scientific(x, precision=2))
    df_out['Sy'] = df_out['Sy'].apply(lambda x: np.format_float_scientific(x, precision=2))

    return df_out

#----------------------------------------------------------------------------------------------------------------------#

def bound_pressure_report(pst, pe, top_k=15):
    pd = pst.parameter_data.loc[pst.adj_par_names, ["parlbnd","parubnd"]]
    vals = pe
    lo_hits = (vals.lt(pd["parlbnd"], axis=1)).sum(0)
    hi_hits = (vals.gt(pd["parubnd"], axis=1)).sum(0)
    hit = (lo_hits + hi_hits).sort_values(ascending=False)
    if (hit > 0).any():
        print(f"Params with at least one out-of-bounds draw (top {top_k}):")
        print(hit[hit>0].head(top_k))
    else:
        print("No out-of-bounds draws.")

#----------------------------------------------------------------------------------------------------------------------#

def group_weight_sums(obs_df: pd.DataFrame, groups=None, subset=None, group_col="obgnme", weight_col="weight"):
    """
    Return a Series of sum of weights per observation group.
    - groups: optional iterable of group names to include; if None, include all present
    - subset: optional boolean mask or index of obs to include in sums (others ignored)
    """
    if group_col not in obs_df or weight_col not in obs_df:
        raise KeyError(f"obs_df must contain columns '{group_col}' and '{weight_col}'")
    df = obs_df[[group_col, weight_col]].copy()

    if subset is not None:
        if isinstance(subset, (pd.Series, np.ndarray, list, tuple)):
            df = df.loc[obs_df.index[subset]] if (isinstance(subset, pd.Series) and subset.dtype==bool) else df.loc[subset]
        else:
            raise ValueError("subset must be a boolean Series/array aligned to obs_df or an index-like of obsnme")

    if groups is not None:
        df = df[df[group_col].isin(set(groups))]

    return df.groupby(group_col, dropna=False)[weight_col].sum().sort_index()

#----------------------------------------------------------------------------------------------------------------------#

def balance_group_weights(
    obs_df: pd.DataFrame,
    groups=None,
    subset=None,
    target_sum: float | None = None,
    group_col="obgnme",
    weight_col="weight",
    inplace: bool = True,
):
    """
    Rebalance weights across groups while preserving within-group relative weights.

    Args
    ----
    obs_df : DataFrame with columns [group_col, weight_col]
    groups : iterable of group names to balance; if None -> all groups present in obs_df
    subset : optional boolean mask or index of obs that are eligible to be scaled;
             other observations (even in the same group) remain unchanged.
    target_sum : if None -> each group's new sum becomes the *average* of the current sums
                 if provided:
                     - if >= total groups target: treated as the *total* to split equally among groups
                     - if <  total groups target: treated as the *per-group* target (applied to all)
    inplace : modify obs_df in place (default True). Returns the modified DF either way.

    Returns
    -------
    (df, info) where:
      - df is the DataFrame with adjusted weights
      - info is a dict with 'before', 'after', and 'scales' Series (per group)
    """
    if group_col not in obs_df or weight_col not in obs_df:
        raise KeyError(f"obs_df must contain columns '{group_col}' and '{weight_col}'")

    df = obs_df if inplace else obs_df.copy()

    # Build a working view + eligibility mask
    all_groups = df[group_col].dropna().unique().tolist()
    target_groups = set(all_groups) if groups is None else set(groups)
    if not target_groups:
        raise ValueError("No groups selected to balance.")

    if subset is None:
        eligible_mask = df[group_col].isin(target_groups)
    else:
        # subset can be boolean mask aligned to df.index, or list/Index of obsnme
        if isinstance(subset, pd.Series) and subset.dtype == bool:
            eligible_mask = subset & df[group_col].isin(target_groups)
        elif isinstance(subset, (np.ndarray, list, tuple, pd.Index)):
            eligible_mask = df.index.isin(subset) & df[group_col].isin(target_groups)
        else:
            raise ValueError("subset must be a boolean Series/array aligned to obs_df or an index-like of obsnme")

    # Current sums (only counting eligible obs within each group)
    before = df.loc[eligible_mask].groupby(group_col)[weight_col].sum().reindex(sorted(target_groups)).fillna(0.0)

    if len(before) == 0:
        raise ValueError("No eligible observations found in the selected groups.")

    # Decide the per-group target
    if target_sum is None:
        per_group_target = before.mean()
    else:
        # If target_sum looks like a total (>= sum of current per-group means), split equally.
        # Otherwise treat as per-group target directly.
        if target_sum >= before.sum():
            per_group_target = target_sum / len(before)
        else:
            per_group_target = float(target_sum)

    # Compute scale factors per group; handle zeros safely
    scales = pd.Series(index=before.index, dtype=float)
    for g, s in before.items():
        if s <= 0.0:
            # If a group has zero eligible weight, set scale=1.0 (no change)
            # and warn via scales value = np.nan to signal "unchanged"
            scales.loc[g] = np.nan
        else:
            scales.loc[g] = per_group_target / s

    # Apply scaling *only* to eligible obs, preserving within-group relative weights
    elig = df.loc[eligible_mask, [group_col, weight_col]].copy()
    # Map scale to each eligible row
    row_scales = elig[group_col].map(scales).to_numpy()
    # Keep rows with nan scale unchanged
    keep = np.isnan(row_scales)
    new_w = elig[weight_col].to_numpy()
    new_w[~keep] = new_w[~keep] * row_scales[~keep]
    df.loc[eligible_mask, weight_col] = new_w

    after = df.loc[eligible_mask].groupby(group_col)[weight_col].sum().reindex(before.index).fillna(0.0)

    info = {
        "before": before,
        "after": after,
        "per_group_target": per_group_target,
        "scales": scales,  # per-group multiplicative factors used (NaN = unchanged)
        "eligible_counts": df.loc[eligible_mask].groupby(group_col).size().reindex(before.index).fillna(0).astype(int),
    }
    return (df, info)

#----------------------------------------------------------------------------------------------------------------------#
# Read in necessary model files, setup spatial reference
#----------------------------------------------------------------------------------------------------------------------#

gwf = flopy.modflow.Modflow.load((model_name + '.nam'), version='mfnwt', load_only=['dis','bas6'], model_ws=mf_dir)
sr = pyemu.helpers.SpatialReference(delr=gwf.dis.delr.array, delc=gwf.dis.delc.array, xll=xoff, yll=yoff, epsg=26910)
end_date = origin_date + pd.DateOffset(months=gwf.nper)

#----------------------------------------------------------------------------------------------------------------------#
# Observations
#----------------------------------------------------------------------------------------------------------------------#

# Read in streamflow observations written by 05_A_Streamflow_Error_Models.py
sfr_obs = pd.read_csv(data_dir / 'streamflow_obs_std.csv', index_col='obsnme')
sfr_obs = sfr_obs.rename({'obsgnme': 'obgnme'}, axis=1)  # I hate these abbreviations

# Read in head observations written by 05_B_build_HOB_ins_n_obs.py
hob_obs = pd.read_csv(data_dir / 'head_obs_master.csv', index_col='obsnme')
hob_obs = hob_obs.rename({'obval': 'obsval', 'group': 'obgnme', 'stdev': 'standard_deviation'}, axis=1)

#----------------------------------------------------------------------------------------------------------------------#
# Parameters
#----------------------------------------------------------------------------------------------------------------------#

pp_init_files = [item for item in sorted((data_dir / 'pp_init_csv').glob("**/*.csv"))]
pp_init_dfs = []
for pp_init in pp_init_files:
    df = pd.read_csv(pp_init)
    pp_init_dfs.append(df)
pp_pars = pd.concat(pp_init_dfs)
pp_pars = pp_pars.set_index('parnme')

# Get z-coordinates for pilot points using MODFLOW layer midpoints
mg = gwf.modelgrid
mg.set_coord_info(xoff=xoff, yoff=yoff)
pp_pars['Z'] = get_model_zeds_at_locs(mg, pp_pars)

prev_par = pd.read_csv(data_dir / 'starting_values.csv', index_col='parnme')

#----------------------------------------------------------------------------------------------------------------------#
# File Management
#----------------------------------------------------------------------------------------------------------------------#

# Some handmade TPL files
other_tpls_files = sorted((data_dir / 'manualTPLs').glob("**/*.tpl"))
for f in other_tpls_files:
    shutil.copy2(f, pest_dir)

# Delete and remake PEST forward model folders
fmdir = pest_dir / 'SVIHM'
if fmdir.exists():
    shutil.rmtree(fmdir)

# Forward Model Files
shutil.copytree(model_dir, fmdir)

# Head obs file...
shutil.copy2(data_dir / 'head_obs_master.csv', fmdir / 'MODFLOW')

# Latest EXEs
shutil.copytree(exe_dir, fmdir / 'Bin')

# Latest python scripts
fm_scripts = ['GAGE2VOL.py', 'HOBPOSTPROC.py', 'LOGSTRSIM.py', 'RES2PAR_preproc.py', 'STREAM_ADJUSTER.py']
for f in fm_scripts:
    shutil.copy2(scpt_dir/ f, fmdir / 'Bin')

# Remove and remake target directory, copy in files
shutil.rmtree(work_dir)
shutil.copytree(pest_dir, work_dir)
shutil.copy2(pestpp_exe, work_dir)

# Set as working directory to appease pyemu
os.chdir(work_dir)

#----------------------------------------------------------------------------------------------------------------------#
# Setup PST
#----------------------------------------------------------------------------------------------------------------------#

# Template files with parameters => IN files
tpl_files = [item for item in glob.glob('*.tpl') if 'svihmt2p' not in item]
in_files = []
for f in tpl_files:  # Where they get written by PEST
    in_file = f.removesuffix('.tpl')
    if in_file in ['t2p_par2par.in', 'sfr2par.in']:
        in_files.append('SVIHM/' + in_file)
    elif in_file in ['catchment_mult.txt','landcover_table.txt','streamflow_multipliers.txt']:
        in_files.append('SVIHM/SWBM/' + in_file)
    else:  # most
        in_files.append('SVIHM/preproc/' + in_file)

# Instruction files with observations
ins_files = sorted(glob.glob("*.ins"))
out_files = []
for f in ins_files:  # overcomplicated this for myself
    out_file = f.removesuffix('.ins')
    if out_file=='Streamflow_FJ_SVIHM_VOL':
        out_files.append('SVIHM/MODFLOW/' + out_file + '.out')
    elif out_file=='head_obs_reader':
        out_files.append('SVIHM/MODFLOW/' + 'head_obs_for_pest.out')
    else:
        out_files.append('SVIHM/MODFLOW/' + out_file + '.dat')

pst = pyemu.Pst.from_io_files(tpl_files=tpl_files, in_files=in_files,
                              ins_files=ins_files, out_files=out_files,
                              pst_filename=pst_file)

#----------------------------------------------------------------------------------------------------------------------#
# Observation Weights & Std Dev
#----------------------------------------------------------------------------------------------------------------------#

# Get obs pointer
obs = pst.observation_data

# Add a standard dev column for observations
obs.loc[:, 'standard_deviation'] = np.nan

# Update observations
obs_updated = 0
for df in [sfr_obs, hob_obs]:
    df.index = df.index.str.lower()
    obs_updated += obs.loc[obs.index.intersection(df.index)].shape[0]
    for col in ["obsval","weight","obgnme", "standard_deviation"]:
        if col in df.columns:
            obs.loc[obs.index.intersection(df.index), col] = df.loc[obs.index.intersection(df.index), col]

# Stream NSE/KGE/RMSE not in observations
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'obsval'] = 1.0
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'weight'] = 0.0
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'standard_deviation'] = 0.0
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'obgnme'] = 'str_metrics'
obs_updated += obs.index.str.contains('nse|kge|rmse').sum()

print(f'Updated {obs_updated} / {obs.shape[0]} obs in PST, and {sfr_obs.shape[0]+hob_obs.shape[0]} in CSVs')

# Did we get everything??
if obs.loc[obs.standard_deviation.isna(), :].count().max() > 0:
    print('Not all observations set...')
    print(obs.loc[obs.standard_deviation.isna(), :])

# Onto weights...
print('Before balancing...')
group_weight_sums(obs)

# Assign metagroup labels
metagroup_mapping = {'AS': 'STREAMFLOW',
                     'AVG_HEAD': 'HEADS',
                     'BY': 'STREAMFLOW',
                     'DIFF_HEAD': 'HEADS',
                     'FJ': 'STREAMFLOW',
                     'FJ_month_vol': 'STREAMVOL',
                     'FJ_year_vol': 'STREAMVOL',
                     'SCK': 'STREAMFLOW',
                     'STAT': 'STAT',
                     'VDIFF': 'HEADS',
                     'str_metrics': 'STAT',
}
obs["metagroup"] = obs["obgnme"].map(metagroup_mapping)
_, info = balance_group_weights(obs, groups=['STREAMFLOW','HEADS'], group_col='metagroup')
print("Meta Before:\n", info["before"])
print("Meta After:\n", info["after"])

print('After balancing...')
group_weight_sums(obs)

#----------------------------------------------------------------------------------------------------------------------#
# Parameter Initial Values & Groups
#----------------------------------------------------------------------------------------------------------------------#

# Get param pointer
par = pst.parameter_data

# Pilot point init values
pp_pars.index = pp_pars.index.str.lower()
pp_updated = 0
for col in ['parval1', 'pargp']:
    par.loc[par.index.intersection(pp_pars.index), col] = pp_pars.loc[par.index.intersection(pp_pars.index), col]

# Checks
pp_updated += par.loc[par.index.intersection(pp_pars.index), col].count()
print(f'Updated {pp_updated} / {pp_pars.shape[0]} pars in PP CSVs')

# Pilot point transforms, bounds (by group)
par.loc[par.index.intersection(pp_pars.index), 'partrans'] = 'none'
par.loc[par.pargp=='aem_var', ['parlbnd','parubnd']] = (0.0, 25.0)
par.loc[par.pargp=='lth_var', ['parlbnd','parubnd']] = (0.0, 5.0)
par.loc[par.pargp=='kv_mult', ['parlbnd','parubnd']] = (-3.0, 3.0)
par.loc[par.pargp.str.contains('scale'), ['parlbnd','parubnd']] = (1.1, 3.3)
par.loc[par.pargp.str.contains('scale_Fine'), ['parlbnd','parubnd']] = (0.5, 1.5)

# Catchment, streamflow multipliers
wtr_updated = 0
par.loc[par.parnme.str.contains('catch_mult'),['parval1','parlbnd','parubnd','pargp']] = (0.5, 0.1, 1.0, 'catch_mult')
par.loc[par.parnme.str.contains('str_mult'),['parval1','parlbnd','parubnd','pargp']] = (1.0, 0.1, 1.0, 'str_mult')
wtr_updated += par.loc[par.parnme.str.contains('catch_mult')].shape[0] + par.loc[par.parnme.str.contains('str_mult')].shape[0]
print(f'Updated {wtr_updated} catchment/str mult using default values')

# Previous parameters (overwrite any previously set values)
par_updated = 0
for col in ['parval1', 'parlbnd', 'parubnd', 'pargp', 'scale', 'offset']:
    par.loc[par.index.intersection(prev_par.index), col] = prev_par.loc[par.index.intersection(prev_par.index), col]

# Check
par_updated += par.loc[par.index.intersection(prev_par.index), col].shape[0]
print(f'Updated {par_updated} / {prev_par.shape[0]} pars in prev CSV')
print(f'In total, {pp_updated + par_updated + wtr_updated} / {par.shape[0]} parameters have been updated.')

# Check (base) texture parameters
t2p_par2par_frompar(par)

#----------------------------------------------------------------------------------------------------------------------#
# Parameter Covariance
#----------------------------------------------------------------------------------------------------------------------#

# build full cov with PP blocks
full_cov = build_full_cov_with_pp_blocks(pst, pp_pars)
# full_cov.to_binary(work_dir / "parcov.jcb")  # Can't take names with 20+ characters!
# pst.pestpp_options["parcov"] = "parcov.jcb"

full_cov.to_ascii(work_dir / "parcov.cov")
pst.pestpp_options["parcov"] = "parcov.cov"

pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=full_cov, num_reals=ensemble_size)
pe.enforce(how="reset")
#bound_pressure_report(pst, pe)

pe.to_csv(work_dir / "prior_pe.csv")
pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.csv"
pst.pestpp_options["ies_num_reals"] = ensemble_size

#----------------------------------------------------------------------------------------------------------------------#
# LET'S GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#----------------------------------------------------------------------------------------------------------------------#

# --- ies options ---
pst.model_command = [str(Path("forward_run.bat"))]
pst.pestpp_options["ies_include_base"] = True
pst.pestpp_options['ies_reg_factor'] = 0.05  #
pst.pestpp_options["ies_bad_phi_sigma"] = 2.0  # middle ground value
pst.pestpp_options["ies_num_threads"] = 8
pst.control_data.noptmax = -2

# Localization
pst.pestpp_options["ies_localizer"] = "localizer.csv"
pst.pestpp_options["ies_autoadaloc"] = True
pst.pestpp_options["ies_autoadaloc_sigma_dist"] = 2

pst.write(pst_file, version=2)
os.chdir(orig_dir)