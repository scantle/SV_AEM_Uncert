"""
IES_Smart_Restart

Drops failed/timeout runs from prior iteration before writing files

Still need to run 06_Setup_PEST-iES_Control_File.py to correctly populat ref_pest_dir.
Localizer may require updates first if weights have changed.

"""

import os
import pyemu
import pandas as pd
from pathlib import Path
import shutil

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

pst_file = 'svihm_ies.pst'
f_dir = Path('06_Outputs/03_good_but_local/')

restart_dir = Path("C:/Projects/SVIHM/2025_R2P_PEST_Calib_restart")
restart_dir.mkdir(parents=True, exist_ok=True)

# Directories
orig_dir  = os.getcwd()
ref_pest_dir = Path("C:/Projects/SVIHM/2025_R2P_PEST_Calib")


#----------------------------------------------------------------------------------------------------------------------#
# File Management
#----------------------------------------------------------------------------------------------------------------------#

# Remove and remake target directory, copy in files
shutil.rmtree(restart_dir)
shutil.copytree(ref_pest_dir, restart_dir)

# Remove "old" files
os.remove(restart_dir / 'prior_pe.csv')

#----------------------------------------------------------------------------------------------------------------------#
# Subset using successful runs in iteration obs file
#----------------------------------------------------------------------------------------------------------------------#

# 1) Load the iter-0 OBS ensemble and define the keep set (drop BASE if present)
obs0 = pd.read_csv(orig_dir / f_dir / "svihm_ies.0.obs.csv", index_col=0)
keep = obs0.index.to_numpy()
keep[0:-1] = keep[0:-1].astype(int)
obs0_sub = obs0.loc[keep]
obs0_sub.to_csv(restart_dir / "obs_restart.csv")

# 2) Subset the iter-0 PAR ensemble to the same names
par0 = pd.read_csv(orig_dir / f_dir / "svihm_ies.0.par.csv", index_col=0)
dumb = par0.index.to_numpy()
dumb[0:-1] = dumb[0:-1].astype(int)
par0.index = dumb
par0_sub = par0.loc[keep]
par0_sub.to_csv(restart_dir / "par_restart.csv")

#) Subset noise...
noise = pd.read_csv(orig_dir / f_dir / "svihm_ies.obs+noise.csv", index_col=0)
dumb = noise.index.to_numpy()
dumb[0:-1] = dumb[0:-1].astype(int)
noise.index = dumb
noise_sub = noise.loc[keep]
noise_sub.to_csv(restart_dir / "noise_restart.csv")

print(f"{len(keep)} realizations kept")

#----------------------------------------------------------------------------------------------------------------------#
# Load & Update PST
#----------------------------------------------------------------------------------------------------------------------#

os.chdir(restart_dir)

pst = pyemu.Pst(pst_file)
pst.pestpp_options['ies_parameter_ensemble'] = 'par_restart.csv'
pst.pestpp_options['ies_restart_observation_ensemble'] = 'obs_restart.csv'
pst.pestpp_options['ies_observation_ensemble'] = 'noise_restart.csv'

pst.control_data.noptmax = -1
pst.write(restart_dir / pst_file, version=2)