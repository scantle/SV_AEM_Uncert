import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# import os
# os.chdir('../')

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

# Directories
data_dir = Path('01_Data/')
shp_dir = data_dir / 'GIS'
model_dir = Path('//BEHEMOTH/Users/lelan/Documents/ModelRuns/SVIHM/2025_r2p_ies/testing/pst04_iter01_goodpars_newweights/')
prev_cal_dir = Path('C:/Users/lelan/Documents/CodeProjects/PhD_SV_AEM_Uncert/06_Outputs/05_novolt_drnostreams')

out_dir = Path('06_Outputs')

# -------------------------------------------------------------------------------------------------------------------- #
# Classes/Functions
# -------------------------------------------------------------------------------------------------------------------- #

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

# -------------------------------------------------------------------------------------------------------------------- #
# Analysis
# -------------------------------------------------------------------------------------------------------------------- #

base_obs = pd.read_csv(model_dir / "svihm_ies.base.obs.csv").transpose()
obs_data = pd.read_csv(model_dir / "svihm_ies.obs_data.csv", index_col=0)
pdc_obs = pd.read_csv(prev_cal_dir / "svihm_iespdc.csv", index_col=0)

# Merge on observation name (assuming 'obsnme' is the column name in both)
df = obs_data.copy()
df['simval'] = base_obs.loc[obs_data.index]

# Compute residuals and contributions
df["res"] = df["simval"] - df["obsval"]
df["phi_contrib"] = (df["weight"] ** 2) * (df["res"] ** 2)

# Add some helpers
df["abs_res"] = df["res"].abs()

# Note if in PDC
df['pdc'] = False
df['pdc_distance'] = 0.0
df.loc[pdc_obs.index, 'pdc'] = True
df.loc[pdc_obs.index, 'pdc_distance'] = pdc_obs.loc[df[df['pdc']].index,'distance']

df.groupby('obgnme').phi_contrib.agg('sum')

df.sort_values(by=['phi_contrib'], ascending=False, inplace=True)
hds_df = df.loc[df.obgnme.str.startswith('hds'),:]

# Export just streams, by stream for analysis in Excel:
# cols = ['date', 'obsval','simval','res','weight','phi_contrib','abs_res','pdc','pdc_distance']
# for str in ['FJ','AS','BY','SCK']:
#     temp = df.loc[df['obgnme']==f'str_{str}',:].copy()
#     temp['obsnme'] = temp.index
#     temp["date"] = temp.obsnme.apply(parse_daily_date_from_obsnme)
#     temp[cols].to_csv(out_dir / f'str_{str}_residuals.csv')

# Setup a new df where we can keep str weight multipliers
w_df = df.loc[df.obgnme.str.startswith('str'), ['obgnme','obsval','simval','abs_res','pdc','weight']].copy()

# Fix ol' 1e-12 issue
w_df['obsval'] = w_df['obsval'].replace(-12,-1)
w_df['simval'] = w_df['simval'].replace(-12,-1)
w_df['res'] = w_df["simval"] - w_df["obsval"]

# adjust weight on FJ flows very difficult for the model to hit
w_df['wmult'] = 1.0
w_df.loc[(w_df.obgnme=='str_FJ') & (w_df.abs_res > 0.4), 'wmult'] = 0.2
w_df.loc[(w_df.obgnme=='str_FJ') & (w_df.abs_res > 0.8), 'wmult'] = 0.01

# remove conflicted values, mostly high flows not in PRMS model
w_df.loc[w_df.pdc, 'wmult'] = 0.0

# New estimated phi
w_df['new_weight'] = w_df['weight'] * w_df['wmult']
w_df['new_phi'] = (w_df["new_weight"] ** 2) * (w_df["res"] ** 2)

# phi by group
w_df.groupby('obgnme').new_phi.agg('sum')

# Write out stream weight multipliers
w_df['wmult'].to_csv(data_dir / "str_wmults.csv")