import os
import numpy as np
import pandas as pd
import pyemu
from pyemu.utils import geostats
from pyemu.utils.helpers import read_pestpp_runstorage, parse_rmr_file
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

f_dir = Path('06_Outputs/01_manytimeouts')

par_file   = f_dir / "svihm_ies.0.par.csv"
rmr_file   = f_dir / "svihm_ies.rmr"

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def calc_chain(df):
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
# Analysis
#----------------------------------------------------------------------------------------------------------------------#

# Read files
rmr = parse_rmr_file(rmr_file)
par = pd.read_csv(par_file)

# Get success/overtime runs
success_runs = rmr.loc[rmr['action'].str.endswith('processed'), 'action'].str.strip('run_processed').astype(int).tolist()
overtime_runs = rmr.loc[rmr['reason']=='overdue.','run_id'].str.replace(',','').astype(int).tolist()

par = par.set_index("real_name", drop=True)

par['status'] = np.nan
par.loc[par.index.isin(success_runs), 'status'] = 'success'
par.loc[par.index.isin(overtime_runs), 'status'] = 'overtime'

derived, chain_cols = calc_chain(par)
df = par.join(derived, how="left")

#----------------------------------------------------------------------------------------------------------------------#
# Quick distribution diagnostics on the chains
q = df.groupby("status")[chain_cols].quantile([0.5,0.9,0.95,0.99]).unstack(level=0)
print("\nChain quantiles by status (level-0=stat, level-1=status):")

# suggested caps from overtime tail (useful to pre-filter or clamp)
caps = df.loc[df["status"]=="success", chain_cols].quantile(0.99)
print("\nSuggested initial caps (99th pct among success):")
print(caps.sort_values(ascending=False).head(15))

#----------------------------------------------------------------------------------------------------------------------#
# Okay let's look at the K multipliers so we can draw better ranges
kcols = ['kminff1', 'kminmf1_m', 'kminsc1_m', 'kminmc1_m', 'kminvc1_m']
kq = df.groupby("status")[kcols].quantile([0.1, 0.25,0.5,0.9,0.95,0.99]).unstack(level=0)

# Anisotropy
kcols = ['anisovc1', 'anisomc1_m', 'anisosc1_m', 'anisomf1_m', 'anisoff1_m']
anq = df.groupby("status")[kcols].quantile([0.1, 0.25,0.5,0.9,0.95,0.99]).unstack(level=0)

# sbc values (sort of not fair given the K values)
kcols = [pnme for pnme in df.columns if pnme.startswith('sbcm')]
sbcmq = df.groupby("status")[kcols].quantile([0.1, 0.25,0.5,0.9,0.95,0.99]).unstack(level=0)