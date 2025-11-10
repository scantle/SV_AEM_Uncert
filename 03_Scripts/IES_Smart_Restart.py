import pandas as pd
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

f_dir = Path('06_Outputs/02_still_timeouts_then_power_outage')


#----------------------------------------------------------------------------------------------------------------------#
# Subset using successful runs in iteration obs file
#----------------------------------------------------------------------------------------------------------------------#

# 1) Load the iter-0 OBS ensemble and define the keep set (drop BASE if present)
obs0 = pd.read_csv(f_dir / "svihm_ies.0.obs.csv", index_col=0)
keep = obs0.index.to_numpy()
keep[0:-1] = keep[0:-1].astype(int)
obs0_sub = obs0.loc[keep]
obs0_sub.to_csv(f_dir / "obs_restarts.csv")

# 2) Subset the iter-0 PAR ensemble to the same names
par0 = pd.read_csv(f_dir / "svihm_ies.0.par.csv", index_col=0)
dumb = par0.index.to_numpy()
dumb[0:-1] = dumb[0:-1].astype(int)
par0.index = dumb
par0_sub = par0.loc[keep]
par0_sub.to_csv(f_dir / "par_restart.csv")

#) Subset noise...
noise = pd.read_csv(f_dir / "svihm_ies.obs+noise.csv", index_col=0)
dumb = noise.index.to_numpy()
dumb[0:-1] = dumb[0:-1].astype(int)
noise.index = dumb
noise_sub = noise.loc[keep]
noise_sub.to_csv(f_dir / "noise_restart.csv")

# 4) Subset your prior parameter ensemble to the same names
prior = pd.read_csv(f_dir / "prior_pe.csv", index_col=0)
dumb = prior.index.to_numpy()
dumb[0:-1] = dumb[0:-1].astype(int)
prior.index = dumb
keep = keep[keep != "base"]
prior_sub = prior.loc[keep]
prior_sub.to_csv(f_dir / "prior_restart.csv")

print(f"{len(keep)} realizations kept")