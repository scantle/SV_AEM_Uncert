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
# Observations
#----------------------------------------------------------------------------------------------------------------------#

# Read in streamflow observations written by 05_A_Streamflow_Error_Models.py
sfr_obs = pd.read_csv(data_dir / 'streamflow_obs_std.csv')

# Read in head observations written by 05_B_build_HOB_ins_n_obs.py
hob_obs = pd.read_csv(data_dir / 'head_obs_master.csv')

# Munging
sfr_obs['weights'] = 1/ sfr_obs['obsstd']

#----------------------------------------------------------------------------------------------------------------------#
# Parameters
#----------------------------------------------------------------------------------------------------------------------#

pp_init_files = [item for item in sorted((data_dir / 'pp_init_csv').glob("**/*.csv"))]
pp_init_dfs = []
for pp_init in pp_init_files:
    df = pd.read_csv(pp_init)
    pp_init_dfs.append(df)
pp_pars = pd.concat(pp_init_dfs)

prev_par = pd.read_csv(data_dir / 'starting_values.csv')

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
    elif in_file=='landcover_table.txt':
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



#----------------------------------------------------------------------------------------------------------------------#
# Parameter Initial Values & Groups
#----------------------------------------------------------------------------------------------------------------------#

pst.write(pst_file, version=2)
os.chdir(orig_dir)