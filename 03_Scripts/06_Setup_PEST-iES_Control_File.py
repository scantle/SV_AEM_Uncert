import os
import numpy as np
import pandas as pd
import pyemu
import flopy
from tqdm import tqdm
from pathlib import Path
import project_paths as paths
import tex_ladder
import shutil
import glob
import json
import hashlib

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

# Model Info
model_name = 'SVIHM'
xoff = 499977
yoff = 4571330
origin_date = pd.to_datetime('1990-9-30')

# Include AEM/Lith KVME pilot points?
kvme_pp_flag = False

# Directories
orig_dir  = os.getcwd()
pest_dir  = paths.PEST_DIR   # TPL, INS
work_dir  = paths.RUNS_DIR / '2025_R2P_PEST_Calib'  # build folder
data_dir  = paths.DATA_DIR
exe_dir   = paths.BIN_DIR
scpt_dir  = paths.SCRIPT_DIR
model_dir = paths.MODEL_DIR
mf_dir    = model_dir / 'MODFLOW'
pestpp_exe = paths.pestpp_exe('pestpp-ies')

# Prior for non-pilot point parameters: center, transform, prior std dev (log10 units if log), hard limits
prior_file = data_dir / 'prior_parameters.csv'

# Out
pst_file = 'svihm_ies.pst'
ensemble_size = 500

# Bounds are placed at +/- n_sd prior std devs (transformed space) so they are rarely reached
n_sd = 3.0

# Pilot point priors by family (center from the pp_init CSVs written by 04)
#   sd in transformed units (log10 if log), hard_lo/hard_hi limit the bounds
#   scale_shift: local ln shift of all five ER-texture scales together (0.05 = all scales +5%)
PP_PRIOR = {
    "kv_mult":     dict(transform='none', sd=1.0,  hard_lo=-3.0, hard_hi=3.0),
    "scale_shift": dict(transform='none', sd=0.05, hard_lo=None, hard_hi=None),
    "aem_var":     dict(transform='none', sd=None, hard_lo=0.0,  hard_hi=5.0),   # KVME only
    "lth_var":   dict(transform='none', sd=None, hard_lo=0.0,  hard_hi=5.0),   # KVME only
}

# Prior filter on derived texture properties (reported, not silent)
filt_kvc = (10.0, 200.0)   # m/d, very coarse K
filt_anff_max = 100.0      # fine-fine anisotropy

# Global ER-texture scale ladder: prior mean and covariance from the bootstrap scale solutions (Scantlebury et al. 2026)
ladder_boot_file = data_dir / 'rho_dict.json'
tex_dist_file = data_dir / 'lognorm_dist_clustered.par'

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

# Observation noise correlation, fitted to the residual correlograms of the 2025 calibration (iteration 3)
#   rho(t) = (1 - sum w) exp(-t/L) + sum_k w_k cos(2 pi k t / 365.25), t in days
#   AS and BY records are too short to fit, use FJ (all mainstem gauges)
#   Heads: L only, no seasonal harmonics (would blur the seasonal amplitude the DIFF obs carry)
OBS_NOISE_KERNEL = {
    'FJ':    dict(L=36.0, w=[0.040, 0.123, 0.020]),
    'SCK':   dict(L=24.9, w=[0.169, 0.197, 0.027, 0.006]),
    'heads': dict(L=40.7, w=[]),
}
OBS_NOISE_KERNEL['AS'] = OBS_NOISE_KERNEL['FJ']
OBS_NOISE_KERNEL['BY'] = OBS_NOISE_KERNEL['FJ']
obs_noise_origin = pd.to_datetime('1990-10-01')

# Head reading error = per-well offset (reference point / screen) + short-term correlated error (m), must match 05_B
# Nested piezometers: bottom casing shares the top casing's error plus an independent nest term
head_offset_sd = 0.3
head_reading_sd = 0.3
head_nest_sd = 0.1

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

def prior_sd_and_bounds(center, transform, sd, hard_lo=None, hard_hi=None, n_sd=3.0):
    """
    Bounds at center +/- n_sd * sd in transformed space. If a hard limit falls inside that, the sd is
    reduced so the bound sits at the hard limit, keeping the bounds n_sd std devs from the center.
    Returns (sd, lbnd, ubnd, tightened)
    """
    if transform == 'log':
        f, finv = np.log10, lambda v: 10.0 ** v
    else:
        f, finv = (lambda v: v), (lambda v: v)
    c = f(center)
    sd_new = sd
    if hard_lo is not None and not pd.isna(hard_lo):
        sd_new = min(sd_new, (c - f(hard_lo)) / n_sd)
    if hard_hi is not None and not pd.isna(hard_hi):
        sd_new = min(sd_new, (f(hard_hi) - c) / n_sd)
    if sd_new <= 0:
        raise ValueError(f'Prior center {center} is on or outside its hard limits ({hard_lo}, {hard_hi})')
    return sd_new, finv(c - n_sd * sd_new), finv(c + n_sd * sd_new), sd_new < sd

#----------------------------------------------------------------------------------------------------------------------#

def build_full_cov_with_pp_blocks(pst, pp_pars, prior_sd, ladder_cov=None):
    """
    pst: pyemu.Pst
    pp_pars: DataFrame indexed by parnme with columns ['pargp','X','Y','Z','Layer']
    prior_sd: Series of prior std devs (transformed space) indexed by parnme, for ALL adjustable params
    ladder_cov: DataFrame (names x names) dense block for the scale ladder parameters
    returns: pyemu.Cov covering ALL adjustable params, with PP sub-blocks filled by 3D geostats
    """
    # 1) base diagonal from the prior std devs
    order = list(pst.adj_par_names)
    F = np.diag(prior_sd.loc[order].values ** 2)

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
        # correlation (sill 1) scaled by each pilot point's prior std dev
        C = exp_cov3d(names=sub.index.values,
                      x=sub["X"].values, y=sub["Y"].values, z=sub["Z"].values, ah=cfg["ah"], av=cfg["av"])
        sd = prior_sd.loc[C.row_names].values
        # inject block
        idx = [order.index(nm) for nm in C.row_names]
        F[np.ix_(idx, idx)] = C.as_2d * np.outer(sd, sd)

    # 2) dense block for the global scale ladder (bootstrap covariance)
    if ladder_cov is not None:
        idx = [order.index(nm) for nm in ladder_cov.index]
        F[np.ix_(idx, idx)] = ladder_cov.values

    # 3) return as pyemu.Cov (same ordering as pst)
    return pyemu.Cov(x=F, names=order)

#----------------------------------------------------------------------------------------------------------------------#

def t2p_par2par_chain(df):
    out = {}
    # --- K chain ---
    pvals = df.copy()

    out['k_ff'] = pvals['kminff1']
    out['k_mf'] = out['k_ff'] * pvals['kminmf1_m']
    out['k_sc'] = out['k_mf'] * pvals['kminsc1_m']
    out['k_mc'] = out['k_sc'] * pvals['kminmc1_m']
    out['k_vc'] = out['k_mc'] * pvals['kminvc1_m']

    # --- Aniso chain ---
    out['an_vc'] = pvals['anisovc1']
    out['an_mc'] = out['an_vc'] * pvals['anisomc1_m']
    out['an_sc'] = out['an_mc'] * pvals['anisosc1_m']
    out['an_mf'] = out['an_sc'] * pvals['anisomf1_m']
    out['an_ff'] = out['an_mf'] * pvals['anisoff1_m']

    # --- Ss chain ---
    out['ss_ff'] = pvals['ssff1']
    out['ss_mf'] = out['ss_ff'] * pvals['ssmf1_m']
    out['ss_sc'] = out['ss_mf'] * pvals['sssc1_m']
    out['ss_mc'] = out['ss_sc'] * pvals['ssmc1_m']
    out['ss_vc'] = out['ss_mc'] * pvals['ssvc1_m']

    # --- Sy chain ---
    out['sy_sc'] = pvals['sysc1']
    out['sy_mf'] = out['sy_sc'] * pvals['symf1_m']
    out['sy_ff'] = out['sy_mf'] * pvals['syff1_m']
    out['sy_mc'] = out['sy_sc'] * pvals['symc1_m']
    out['sy_vc'] = out['sy_mc'] * pvals['syvc1_m']

    return pd.DataFrame(out), list(out.keys())

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
    pvals = (df['parval1'] * df['scale'] + df['offset']).to_frame().T

    # Never break the chain
    chain_df, _ = t2p_par2par_chain(pvals)
    r = chain_df.iloc[0]
    df_out =  pd.DataFrame({
        "FF": {"Kmin": r.k_ff, "Aniso": r.an_ff, "Ss": r.ss_ff, "Sy": r.sy_ff},
        "MF": {"Kmin": r.k_mf, "Aniso": r.an_mf, "Ss": r.ss_mf, "Sy": r.sy_mf},
        "SC": {"Kmin": r.k_sc, "Aniso": r.an_sc, "Ss": r.ss_sc, "Sy": r.sy_sc},
        "MC": {"Kmin": r.k_mc, "Aniso": r.an_mc, "Ss": r.ss_mc, "Sy": r.sy_mc},
        "VC": {"Kmin": r.k_vc, "Aniso": r.an_vc, "Ss": r.ss_vc, "Sy": r.sy_vc},
    }).T[["Kmin", "Aniso", "Ss", "Sy"]]

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

def pp_family(pargp, families):
    for fam in families:
        if pargp.startswith(fam):
            return fam
    raise ValueError(f'No pilot point prior for group {pargp}')

#----------------------------------------------------------------------------------------------------------------------#

def assign_priors(par, pp_pars, prior_par, pp_prior, n_sd=3.0):
    """
    Sets parval1 (prior center), partrans, scale/offset and bounds (center +/- n_sd prior std devs) on the
    pst parameter data. Pilot point centers come from the pp_init CSVs, others from the prior table.
    Returns the prior std dev of every parameter (transformed space) and a summary table by group.
    """
    sd_out = {}
    rows = []

    # Pilot points
    pp_names = par.index.intersection(pp_pars.index)
    for nm in pp_names:
        grp = pp_pars.loc[nm, 'pargp']
        cfg = pp_prior[pp_family(grp, pp_prior.keys())]
        center = pp_pars.loc[nm, 'parval1']
        sd = cfg['sd'] if cfg['sd'] is not None else (cfg['hard_hi'] - cfg['hard_lo']) / (2 * n_sd)
        sd, lb, ub, tight = prior_sd_and_bounds(center, cfg['transform'], sd, cfg['hard_lo'], cfg['hard_hi'], n_sd)
        par.loc[nm, ['parval1', 'pargp', 'partrans', 'parlbnd', 'parubnd']] = [center, grp, cfg['transform'], lb, ub]
        sd_out[nm] = sd
        rows.append([nm, grp, cfg['transform'], center, sd, lb, ub, tight])

    # Everything else
    other = par.index.difference(pp_names)
    missing = other.difference(prior_par.index)
    if len(missing) > 0:
        raise ValueError(f'Parameters without a prior in {prior_file.name}: {list(missing)}')
    for nm in other:
        p = prior_par.loc[nm]
        sd, lb, ub, tight = prior_sd_and_bounds(p['center'], p['transform'], p['sd'], p['hard_lo'], p['hard_hi'], n_sd)
        par.loc[nm, ['parval1', 'pargp', 'partrans', 'parlbnd', 'parubnd', 'scale', 'offset']] = \
            [p['center'], p['pargp'], p['transform'], lb, ub, p['scale'], p['offset']]
        sd_out[nm] = sd
        rows.append([nm, p['pargp'], p['transform'], p['center'], sd, lb, ub, tight])

    df = pd.DataFrame(rows, columns=['parnme', 'pargp', 'transform', 'center', 'sd', 'lbnd', 'ubnd', 'sd_tightened'])
    summary = df.groupby('pargp').agg(n=('parnme', 'size'), transform=('transform', 'first'),
                                      center_min=('center', 'min'), center_max=('center', 'max'),
                                      sd_min=('sd', 'min'), sd_max=('sd', 'max'),
                                      lbnd_min=('lbnd', 'min'), ubnd_max=('ubnd', 'max'),
                                      n_sd_tightened=('sd_tightened', 'sum'))
    return pd.Series(sd_out), summary

#----------------------------------------------------------------------------------------------------------------------#

def draw_prior(pst, cov, n_reals, filt_kvc, filt_anff_max, shapes, oversample=5, seed=42):
    """
    Draws the prior ensemble from cov (full covariance, so cross-group/layer correlation is kept), reports
    draws outside bounds, then applies the texture filters and reports their acceptance rates.
    Scale ladder draws outside bounds are rejected rather than clipped (clipping breaks their correlation).
    """
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=n_reals*oversample, by_groups=False)
    bnds = pst.parameter_data.loc[pe.columns, ['parlbnd', 'parubnd']]
    oob = pe.lt(bnds['parlbnd'], axis=1) | pe.gt(bnds['parubnd'], axis=1)
    out = oob.mean(0)
    print(f'Draws outside bounds: max {out.max():.2%} of draws, {(out > 0.01).sum()} parameters above 1%')
    print(out.groupby(pst.parameter_data.loc[out.index, 'pargp']).max().sort_values(ascending=False).head(10))
    keep_ladder = ~oob[tex_ladder.ladder_names].any(axis=1)
    pe.enforce()

    # Filter on derived texture properties
    tex, _ = t2p_par2par_chain(pd.DataFrame(pe, columns=pe.columns))
    keep = (tex.k_vc > filt_kvc[0]) & (tex.k_vc < filt_kvc[1]) & (tex.an_ff < filt_anff_max)
    print(f'Texture filter acceptance: {keep.mean():.1%} ({keep.sum()} of {len(keep)})')

    # Filter on ER-texture probability order (max dip in E[texture] as ER increases)
    ln_scales = tex_ladder.ladder_lnscales(pe.loc[:, tex_ladder.ladder_names].to_numpy())
    dip = pd.Series([tex_ladder.order_dip(ls, shapes) for ls in ln_scales], index=pe.index)
    keep_order = keep_ladder & (dip <= tex_ladder.DIP_MAX)
    print(f'Scale ladder acceptance: {keep_ladder.mean():.1%} in bounds, {keep_order.mean():.1%} in bounds and '
          f'dip <= {tex_ladder.DIP_MAX}')
    keep = keep & keep_order
    print(f'Combined acceptance: {keep.mean():.1%} ({keep.sum()} of {len(keep)})')
    if keep.sum() < n_reals:
        raise ValueError('Not enough draws pass the texture filter; revisit the texture priors')
    return pe.loc[keep[keep].index].sample(frac=1.0, random_state=seed).iloc[:n_reals].reset_index(drop=True)

#----------------------------------------------------------------------------------------------------------------------#

def corr_noise(dates, sd, L, w, n_reals):
    """
    Draws N(0, C) at (irregular) dates with C_ij = sd_i sd_j rho(|t_i - t_j|),
    rho(t) = (1 - sum w) exp(-t/L) + sum_k w_k cos(2 pi k t / 365.25)
    Exponential part by exact AR(1) recursion; seasonal part as yearly harmonics with random amplitude and phase
    that repeat every year. Returns (n_reals, n) array, dates in the same order as given.
    """
    t = (pd.to_datetime(pd.Series(dates)) - obs_noise_origin).dt.days.to_numpy(dtype=float)
    order = np.argsort(t, kind='stable')
    ts = t[order]
    z = np.empty((n_reals, len(t)))
    z[:, 0] = np.random.standard_normal(n_reals)
    phi = np.exp(-np.diff(ts) / L)
    for k in range(1, len(ts)):
        z[:, k] = phi[k-1] * z[:, k-1] + np.sqrt(1 - phi[k-1]**2) * np.random.standard_normal(n_reals)
    z = np.sqrt(1 - np.sum(w)) * z
    for k, wk in enumerate(w, start=1):
        a, b = np.random.standard_normal((2, n_reals, 1)) * np.sqrt(wk)
        z += a * np.cos(2 * np.pi * k * ts / 365.25)[None, :] + b * np.sin(2 * np.pi * k * ts / 365.25)[None, :]
    out = np.empty_like(z)
    out[:, order] = z
    return out * np.broadcast_to(np.asarray(sd, dtype=float), (len(t),))[None, :]

#----------------------------------------------------------------------------------------------------------------------#

def streamflow_obs_noise(sfr_obs, n_reals):
    """Correlated noise on the 7-day log10 streamflow targets, by gauge"""
    blk = sfr_obs[sfr_obs['obgnme'].str.startswith('str7d_') & (sfr_obs['weight'] > 0)]
    noise = {}
    for g, b in blk.groupby(blk['obgnme'].str.removeprefix('str7d_').str.upper()):
        dates = pd.to_datetime(b.index.str.split('_w').str[1], format='%Y%m%d') + pd.Timedelta(days=3)
        e = corr_noise(dates, b['standard_deviation'], n_reals=n_reals, **OBS_NOISE_KERNEL[g])
        noise.update(dict(zip(b.index, e.T)))
    return pd.DataFrame(noise)

#----------------------------------------------------------------------------------------------------------------------#

def head_obs_noise(hob_obs, n_reals):
    """
    Noise drawn for each head reading (per-well offset + short-term correlated error), then carried through the
    same operations 05_B uses: well weighted mean (AVG), deviation from the mean (DIFF), and paired-well
    difference (VDIFF). Paired well bottom = top well + '_2'; it takes the top well's reading error (nearest date)
    plus an independent nest term.
    """
    diff = hob_obs[hob_obs['obgnme'] == 'hds_diff']
    avg = hob_obs[hob_obs['obgnme'] == 'hds_avg']
    vd = hob_obs[hob_obs['obgnme'] == 'hds_vdiff']
    nested = {f'{top}_2': top for top in vd['wellid'].unique()}
    noise, reading, head, sd_ratio = {}, {}, {}, []
    for well, g in diff.groupby('wellid'):
        dates = pd.to_datetime(g['date']).to_numpy()
        if well in nested:
            if nested[well] not in reading:
                raise ValueError(f'Top well {nested[well]} must be processed before {well}')
            top = reading[nested[well]]
            e = top.iloc[top.index.get_indexer(dates, method='nearest')].to_numpy().T + \
                corr_noise(g['date'], head_nest_sd, n_reals=n_reals, **OBS_NOISE_KERNEL['heads'])
        else:
            e = corr_noise(g['date'], head_reading_sd, n_reals=n_reals, **OBS_NOISE_KERNEL['heads']) + \
                np.random.standard_normal((n_reals, 1)) * head_offset_sd
        wt = g['weight'].to_numpy()
        if wt.sum() <= 0:
            wt = np.ones(len(g))
        m = e @ wt / wt.sum()
        noise.update(dict(zip(g.index, (e - m[:, None]).T)))
        avg_nme = f'{well}_avg'.lower()
        noise[avg_nme] = m
        sd_ratio.append([m.std() / avg.loc[avg_nme, 'standard_deviation'],
                         np.sqrt(((e - m[:, None]).var(0)).mean() / (g['standard_deviation']**2).mean())])
        reading[well] = pd.DataFrame(e.T, index=dates).groupby(level=0).first()
        head[well] = (g['obsval'] + avg.loc[avg_nme, 'obsval']).set_axis(dates).groupby(level=0).first()
    sd_ratio = np.array(sd_ratio)
    print(f'Head noise realized / 05_B sd: AVG {np.nanmin(sd_ratio[:, 0]):.2f}-{np.nanmax(sd_ratio[:, 0]):.2f}, '
          f'DIFF {np.nanmin(sd_ratio[:, 1]):.2f}-{np.nanmax(sd_ratio[:, 1]):.2f} (by well)')
    for top, v in vd.groupby('wellid'):
        bot = f'{top}_2'
        dates = pd.to_datetime(v['date']).to_numpy()
        if not np.allclose(head[top].loc[dates] - head[bot].loc[dates], v['obsval'], atol=1e-3):
            raise ValueError(f'VDIFF pairing check failed for {top}/{bot}')
        e = reading[top].loc[dates].to_numpy() - reading[bot].loc[dates].to_numpy()
        noise.update(dict(zip(v.index, e)))
    return pd.DataFrame(noise)

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
sfr_obs = pd.read_csv(paths.DERIVED_DIR / 'streamflow_obs_std.csv', index_col='obsnme')
sfr_obs = sfr_obs.rename({'obsgnme': 'obgnme'}, axis=1)  # I hate these abbreviations

# Read in head observations written by 05_B_build_HOB_ins_n_obs.py
hob_obs = pd.read_csv(paths.DERIVED_DIR / 'head_obs_master.csv', index_col='obsnme')
hob_obs = hob_obs.rename({'obval': 'obsval', 'group': 'obgnme', 'stdev': 'standard_deviation'}, axis=1)

#----------------------------------------------------------------------------------------------------------------------#
# Parameters
#----------------------------------------------------------------------------------------------------------------------#

pp_init_files = [item for item in sorted(paths.PP_INIT_DIR.glob("**/*.csv"))]
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

prior_par = pd.read_csv(prior_file, index_col='parnme')
prior_par.index = prior_par.index.str.lower()

# Scale ladder prior (ln S_FF + four ln gaps) from the bootstrap scale solutions, in texture order
with open(ladder_boot_file, 'r') as f:
    boot = json.load(f)
boot_scales = np.array([boot[k] for k in sorted(boot, key=int)]).T
ladder_mean, ladder_cov = tex_ladder.ladder_prior(boot_scales)
ladder_cov = pd.DataFrame(ladder_cov, index=tex_ladder.ladder_names, columns=tex_ladder.ladder_names)
for nm, c, sd in zip(tex_ladder.ladder_names, ladder_mean, np.sqrt(np.diag(ladder_cov))):
    prior_par.loc[nm, ['pargp', 'center', 'transform', 'sd', 'hard_lo', 'hard_hi', 'scale', 'offset', 'source']] = \
        ['ladder', c, 'none', sd, np.nan, np.nan, 1.0, 0.0, 'Bootstrap of Scantlebury et al. (2026) ER-texture scales']

# Texture shapes (fixed) for the order check
tex_shapes = pd.read_table(tex_dist_file, sep=r'\s+', skiprows=1).set_index('Texture').loc[tex_ladder.texs, 'Shape'].values

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

# Head obs file,
shutil.copy2(paths.DERIVED_DIR / 'head_obs_master.csv', fmdir / 'MODFLOW')

# Lithology and AEM logs written by 01_Write_Initial_Sim_Files.py
for f in ['lithologs.csv', 'aemlogs.csv']:
    shutil.copy2(paths.DERIVED_DIR / f, fmdir / 'preproc')

# Latest EXEs
shutil.copytree(exe_dir, fmdir / 'Bin')

# Latest python scripts
fm_scripts = ['GAGE2VOL.py', 'HOBPOSTPROC.py', 'LOGSTRSIM.py', 'RES2PAR_preproc.py', 'STREAM_ADJUSTER.py',
              'tex_ladder.py']
for f in fm_scripts:
    shutil.copy2(scpt_dir/ f, fmdir / 'Bin')

# Remove and remake target directory, copy in files
# (keeps .pixi so batches already launched from this folder keep their python)
if work_dir.exists():
    for f in work_dir.iterdir():
        if f.name == '.pixi':
            continue
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
shutil.copytree(pest_dir, work_dir, dirs_exist_ok=True)
shutil.copy2(pestpp_exe, work_dir)

# Worker python environment
shutil.copy2(paths.ROOT / 'pixi.toml', work_dir)
shutil.copy2(paths.ROOT / 'pixi.lock', work_dir)

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
for f in ins_files:
    out_file = f.removesuffix('.ins')
#    if out_file=='Streamflow_FJ_SVIHM_VOL':
#        out_files.append('SVIHM/MODFLOW/' + out_file + '.out')
    if out_file=='head_obs_reader':
        out_files.append('SVIHM/MODFLOW/' + 'head_obs_for_pest.out')
    elif out_file=='tex_order.out':
        out_files.append('SVIHM/preproc/' + out_file)
    else:
        out_files.append('SVIHM/MODFLOW/' + out_file + '.out')

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
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'obgnme'] = 'metrics'
obs_updated += obs.index.str.contains('nse|kge|rmse').sum()

# ER-texture probability order: inequality obs, realizations that violate it are dropped (no phi pressure while met)
# drop_violations respects weights, so it needs a non-zero weight
obs['drop_violations'] = False
obs.loc['tex_dip', ['obsval', 'weight', 'obgnme', 'standard_deviation', 'drop_violations']] = \
    [tex_ladder.DIP_MAX, 1.0, 'l_tex_order', 0.0, True]
obs.loc['tex_rev_aem', ['obsval', 'weight', 'obgnme', 'standard_deviation']] = [0.0, 0.0, 'tex_diag', 0.0]
obs_updated += 2

print(f'Updated {obs_updated} / {obs.shape[0]} obs in PST, and {sfr_obs.shape[0]+hob_obs.shape[0]} in CSVs')

# Did we get everything??
if obs.loc[obs.standard_deviation.isna(), :].count().max() > 0:
    print('Not all observations set...')
    print(obs.loc[obs.standard_deviation.isna(), :])

#----------------------------------------------------------------------------------------------------------------------#
# Weight Adjustments
#----------------------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------------------#
# Parameter Initial Values & Groups
#----------------------------------------------------------------------------------------------------------------------#

# Get param pointer
par = pst.parameter_data

# Prior centers, transforms, std devs and bounds for every parameter
pp_pars.index = pp_pars.index.str.lower()
prior_sd, prior_table = assign_priors(par, pp_pars, prior_par, PP_PRIOR, n_sd=n_sd)
prior_table.to_csv(paths.DERIVED_DIR / 'prior_summary.csv')
print(prior_table.to_string())

# Check (base) texture parameters
print(t2p_par2par_frompar(par))

#----------------------------------------------------------------------------------------------------------------------#
# Weight balancing
#----------------------------------------------------------------------------------------------------------------------#

# nz = obs.loc[obs['weight'] > 0, 'obgnme'].astype(str).values
# tags = ['hds','str_fj']
# factor_df = pd.DataFrame({'weight': 0.0}, index=tags)
# factor_df.loc['hds', 'weight'] = 0.3
# factor_df.loc['str_fj', 'weight'] = 0.35
# factor_df.loc['str_as', 'weight'] = .1
# factor_df.loc['str_by', 'weight'] = .1
# factor_df.loc['str_sck', 'weight'] = .15
# #factor_df.loc['vol', 'weight'] = 0.25
#
# # Make sure we got em all...
# if factor_df['weight'].min() <= 0:
#     print('Not all factor weights set...')
# if factor_df['weight'].sum() > 1:
#     print('Factor weights add up too high!')
#
# # write it out
# factor_df.to_csv('factor_weights.dat', header=False)
# pst.pestpp_options["ies_phi_factor_file"] = "factor_weights.dat"

#----------------------------------------------------------------------------------------------------------------------#
# Parameter Covariance
#----------------------------------------------------------------------------------------------------------------------#

# build full cov with PP blocks
full_cov = build_full_cov_with_pp_blocks(pst, pp_pars, prior_sd, ladder_cov=ladder_cov)
full_cov.to_coo("parcov.jcb")
pst.pestpp_options["parcov"] = "parcov.jcb"

# full_cov.to_ascii(work_dir / "parcov.cov")
# pst.pestpp_options["parcov"] = "parcov.cov"

prior_pe = draw_prior(pst, full_cov, ensemble_size, filt_kvc, filt_anff_max, tex_shapes)

prior_pe.to_csv("prior_pe.csv")
pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.csv"
pst.pestpp_options["ies_num_reals"] = ensemble_size

# If you get curious
good_tex_pars, _ = t2p_par2par_chain(prior_pe)

#----------------------------------------------------------------------------------------------------------------------#
# Observation Noise Ensemble
#----------------------------------------------------------------------------------------------------------------------#

# Correlated noise on non-zero weight streamflow targets and head readings; everything else (inequality, metrics,
# zero weight) stays at obsval
noise = pd.concat([streamflow_obs_noise(sfr_obs, ensemble_size),
                   head_obs_noise(hob_obs, ensemble_size)], axis=1)
noise.index = prior_pe.index
nz = obs.loc[(obs['weight'] > 0) & (obs['standard_deviation'] > 0)].index
print(f'Obs noise: {noise.shape[1]} obs, {len(nz.difference(noise.columns))} non-zero weight obs without noise')
print(nz.difference(noise.columns).tolist()[:20])
ratio = noise.std() / obs.loc[noise.columns, 'standard_deviation']
print(ratio.groupby(obs.loc[noise.columns, 'obgnme']).describe()[['mean', 'min', 'max']])

oe = pd.DataFrame(np.tile(obs['obsval'].to_numpy(dtype=float), (ensemble_size, 1)), index=prior_pe.index,
                  columns=obs.index)
oe.loc[:, noise.columns] += noise
pyemu.ObservationEnsemble(pst=pst, df=oe).to_dense('obs_en.bin')
pst.pestpp_options["ies_observation_ensemble"] = "obs_en.bin"

#----------------------------------------------------------------------------------------------------------------------#
# LET'S GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#----------------------------------------------------------------------------------------------------------------------#

# --- ies options ---
pst.model_command = [str(Path("forward_run.bat"))]
pst.pestpp_options["ies_include_base"] = True
pst.pestpp_options['ies_reg_factor'] = 0.25
pst.pestpp_options["ies_bad_phi_sigma"] = 2.0  # middle ground value
pst.pestpp_options["ies_num_threads"] = 8
pst.control_data.noptmax = -1
pst.pestpp_options["ies_enforce_bounds"] = True
pst.pestpp_options["overdue_giveup_fac"] = 3

# Localization
pst.pestpp_options["ies_localizer"] = "localizer.jcb"
pst.pestpp_options["ies_autoadaloc"] = True
pst.pestpp_options["ies_autoadaloc_sigma_dist"] = 2

pst.write(pst_file, version=2)

#----------------------------------------------------------------------------------------------------------------------#
# Build manifest (launch_ies.py copies only these files, and checks they haven't changed)
#----------------------------------------------------------------------------------------------------------------------#

manifest = {}
for f in sorted(work_dir.rglob('*')):
    rel = f.relative_to(work_dir)
    if f.is_dir() or rel.parts[0] == '.pixi' or '__pycache__' in rel.parts or rel.name == 'build_manifest.json':
        continue
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    manifest[rel.as_posix()] = h.hexdigest()
with open(work_dir / 'build_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

os.chdir(orig_dir)