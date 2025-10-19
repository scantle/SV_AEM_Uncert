import numpy as np
import pandas as pd
import pyemu
import flopy
from tqdm import tqdm
from pathlib import Path
import shutil

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
work_dir  = Path("C:/Projects/SVIHM/2025_R2P_PEST_Calib")  # build folder
data_dir  = Path('01_Data')
model_dir = Path('02_Models/SVIHM_MF_working')
mf_dir    = model_dir / 'MODFLOW'

# # for debugging console
# import os
# os.chdir("../")

#----------------------------------------------------------------------------------------------------------------------#
# Classes/Functions
#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
# Read in necessary model files, setup spatial reference
#----------------------------------------------------------------------------------------------------------------------#

gwf = flopy.modflow.Modflow.load((model_name + '.nam'), version='mfnwt', load_only=['dis','bas6'], model_ws=mf_dir)
sr = pyemu.helpers.SpatialReference(delr=gwf.dis.delr.array, delc=gwf.dis.delc.array, xll=xoff, yll=yoff, epsg=26910)
end_date = origin_date + pd.DateOffset(months=gwf.nper)

#----------------------------------------------------------------------------------------------------------------------#
# File Management
#----------------------------------------------------------------------------------------------------------------------#

# Some handmade TPL files
other_tpls_files = sorted((data_dir / 'manualTPLs').glob("**/*.tpl"))
for f in other_tpls_files:
    shutil.copy2(f, pest_dir)

# Latest EXEs


# Model folder

# Latest python scripts



#----------------------------------------------------------------------------------------------------------------------#
# Observations
#----------------------------------------------------------------------------------------------------------------------#

# Read in streamflow observations written by 05_A_Streamflow_Error_Models.py
sfr_obs = pd.read_csv(ref_dir / 'streamflow_obs_std.csv')

# Read in head observations written by 05_B_build_HOB_ins_n_obs.py
hob_obs = pd.read_csv(ref_dir / 'head_obs_master.csv')

#----------------------------------------------------------------------------------------------------------------------#
# Setup PST
#----------------------------------------------------------------------------------------------------------------------#

pf = pyemu.utils.PstFrom(
    original_d=model_dir, new_d=work_dir,
    spatial_reference=sr,
    longnames=True,
    start_datetime=origin_date,
    remove_existing=True
)

tpl_files = [item for item in tpl_files if 'par2par' not in item.name]
pf.add_parameters(
    filenames=[str(t) for t in tpl_files][0],
    par_type="direct",
    par_style="direct",
    pargp="base"
)