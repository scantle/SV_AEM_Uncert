"""
Project paths

Repo-relative folders are defined here so scripts work from any working directory. Machine-specific
locations (archive of previous runs, model run folders, PEST++ executables) are read from
local_paths.json in the repo root, which is not tracked. See local_paths_example.json.
"""
import json
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Repo folders
#----------------------------------------------------------------------------------------------------------------------#

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR    = ROOT / '01_Data'
GIS_DIR     = DATA_DIR / 'GIS'
DERIVED_DIR = DATA_DIR / 'derived'          # Written by workflow scripts, not tracked
DERIVED_GIS_DIR = DERIVED_DIR / 'GIS'
PP_LOC_DIR  = DERIVED_DIR / 'PPlocs'
PP_INIT_DIR = DERIVED_DIR / 'pp_init_csv'

MODELS_DIR  = ROOT / '02_Models'
BIN_DIR     = MODELS_DIR / 'Bin'
MODEL_DIR   = MODELS_DIR / 'SVIHM_MF_working'
MF_DIR      = MODEL_DIR / 'MODFLOW'
SWBM_DIR    = MODEL_DIR / 'SWBM'
PREPROC_DIR = MODEL_DIR / 'preproc'

SCRIPT_DIR  = ROOT / '03_Scripts'
PEST_DIR    = ROOT / '04_PEST_setup'
PLOT_DIR    = ROOT / '05_Plots'
OUTPUT_DIR  = ROOT / '06_Outputs'

for d in [DERIVED_GIS_DIR, PP_LOC_DIR, PP_INIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#----------------------------------------------------------------------------------------------------------------------#
# Machine-specific folders
#----------------------------------------------------------------------------------------------------------------------#

_local_file = ROOT / 'local_paths.json'
_local = {}
if _local_file.exists():
    with open(_local_file) as f:
        _local = json.load(f)

def _local_path(key, default=None):
    val = _local.get(key, default)
    return Path(val) if val is not None else None

ARCHIVE_DIR = _local_path('archive_dir')                      # Holds 06_Outputs/ of previous calibrations
RUNS_DIR    = _local_path('runs_dir', ROOT.parent / 'SVIHM_runs')  # PEST build folders & model run copies
PESTPP_BIN_DIR = _local_path('pestpp_bin_dir')
SVIHM_REPO  = _local_path('svihm_repo', ROOT.parent / 'SVIHM')

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def run_dir(name):
    """
    Folder of a PEST run's outputs (e.g. '06_wtfx'). Looks in 06_Outputs first, then the archive.
    """
    candidates = [OUTPUT_DIR / name]
    if ARCHIVE_DIR is not None:
        candidates.append(ARCHIVE_DIR / '06_Outputs' / name)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Run folder '{name}' not found in: " + ', '.join(str(c) for c in candidates))

def private_file(path):
    """
    Check for a file containing private well information, which is not distributed with the repo.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing private well data: {path}\n"
            f"This file is not included in the repository because it contains information on private wells.")
    return path

def pestpp_exe(name='pestpp-ies'):
    if PESTPP_BIN_DIR is None:
        raise ValueError('pestpp_bin_dir not set in local_paths.json')
    return PESTPP_BIN_DIR / f'{name}.exe'
