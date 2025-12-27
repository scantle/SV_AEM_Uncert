import os
import numpy as np
import pandas as pd
import pyemu
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

# Directories
orig_dir  = os.getcwd()
pest_dir  = Path("C:/Projects/SVIHM/2025_R2P_PEST_Calib")  # previous build folder
work_dir  = Path("C:/Projects/SVIHM/2025_Calib_iter3_natveg_low")  # build folder
data_dir  = Path('01_Data')
#exe_dir   = Path('02_Models/Bin')
scpt_dir  = Path('03_Scripts/')
#model_dir = Path('02_Models/SVIHM_MF_working')
#mf_dir    = model_dir / 'MODFLOW'
pest_exe_dir = Path('C:/Users/lelan/Documents/Models/pest17')
pestpp_exe = Path('C:/Users/lelan/Documents/Models/pestpp-5.2.16-iwin/bin/pestpp-swp.exe')

# Files from calibration
pst_file = 'svihm_ies.pst'
ensb_par = Path('06_Outputs/06_wtfx/svihm_ies.3.par.csv')

scen_name = 'natveg_high'


#----------------------------------------------------------------------------------------------------------------------#
# Open Calibration results
#----------------------------------------------------------------------------------------------------------------------#

par_df = pd.read_csv(ensb_par, index_col=0, dtype={"real_name": str})

#----------------------------------------------------------------------------------------------------------------------#
# Build Scenario
#----------------------------------------------------------------------------------------------------------------------#

# For natveg, no changes needed - everything is handled in SWBM input files and the landcover_table.tpl file

#----------------------------------------------------------------------------------------------------------------------#
# File Management
#----------------------------------------------------------------------------------------------------------------------#

# Assumes pest_dir is all set up from 06_Setup_PEST_iES_Control_File

exclude_files = ['prior_pe.csv', 'parcov.jcb', 'localizer.jcb','pestpp-ies.exe']

# Copy in files
for item in pest_dir.iterdir():
    dest = work_dir / item.name
    if item.is_dir():  # skip Model dir
        continue
    elif item.name in exclude_files:
        continue
    else:
        shutil.copy2(item, dest)
shutil.copy2(pestpp_exe, work_dir)

# Set as working directory to appease pyemu
os.chdir(work_dir)

#----------------------------------------------------------------------------------------------------------------------#
# Read/modify PST
#----------------------------------------------------------------------------------------------------------------------#

pst = pyemu.Pst(pst_file)
pst.control_data.noptmax = 0

# enforce only parameters that exist in pst (not sure where any others would come from...)
par_df = par_df.loc[:, par_df.columns.intersection(pst.par_names)]

pe = pyemu.ParameterEnsemble(pst=pst, df=par_df)
sweep_in = "calib_iter3_sweep.csv"
pe.to_csv(sweep_in)  # writes index as realization names (real_name)

# --- write a SWP-specific pst (copy of yours, with ++ lines added) ---
pst_swp = "svihm_swp.pst"
pst_lines = pst.write(pst_swp)  # writes the modified control file

# Append SWP directives:
# (pyemu doesn't have to "understand" these lines; PEST++ reads them.)
with open(pst_swp, "a", newline="\n") as f:
    f.write("\n")
    f.write(f"++sweep_parameter_csv_file({sweep_in})\n")
    f.write(f"++sweep_output_csv_file(svihm_{scen_name}.csv)\n")
    f.write(f"++sweep_chunk({par_df.shape[0]})\n")

with open('run_pest.bat', 'w', newline="\n") as f:
    f.write(f'call pestpp-swp.exe {pst_swp}')

with open('pest_host.bat', 'w', newline="\n") as f:
    f.write('title PEST Host\n')
    f.write(f'call ..\\bin\\pestpp-swp.exe {pst_swp} /h :5050\n')