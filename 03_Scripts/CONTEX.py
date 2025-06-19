"""
CONTEX - CONtinuous-to-TEXture Geologic Simulator
"""
import numpy as np
import pandas as pd
import flopy as fp
import pyemu
from tqdm import tqdm
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Input Files
in_dir = Path('./04_InputFiles/CONTEX/')
scale_factor_file = in_dir / 'scale_factors.dat'

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330

# Log Distribution Prior
tex_dists = pd.DataFrame({
    "Texture":   ["Fine", "Mixed_Fine", "Sand", "Mixed_Coarse", "Very_Coarse"],
    "Shape":     [0.284487, 0.129485, 0.082697, 0.100896, 0.695543],
    "Location":  [0.0,      0.0,      0.0,      0.0,      190.029113],
    "Scale":     [31.790578, 76.562445, 120.040911, 147.877083, 113.941445],
}).set_index("Texture")

# Log Distribution pilot point variogram
scale_vario   = pyemu.geostats.SphVario(contribution=1.0, a=2317*3)
scale_gs  = pyemu.geostats.GeoStruct(variograms=[scale_vario])

#----------------------------------------------------------------------------------------------------------------------#
# Functions/Classes
#----------------------------------------------------------------------------------------------------------------------#

def model_to_grid_df(mf, xoff=0.0, yoff=0.0, remove_inactive=True):
    """
    Fast vectorized export of MODFLOW grid centers.

    Parameters
    ----------
    mf : flopy.modflow.Modflow
        Already‐loaded MODFLOW‐2005 / NWT model.
    xoff, yoff : float
        Optional extra offsets to add to the model grid (m).
    remove_inactive : bool, default True
        If True, rows with ibound==0 are dropped.

    Returns
    -------
    pd.DataFrame with columns:
        ['layer', 'row', 'col', 'X', 'Y', 'ibound']
    """
    grid = mf.modelgrid                    # StructuredGrid
    nlay, nrow, ncol = mf.dis.nlay, mf.dis.nrow, mf.dis.ncol

    # 2-D centres → 3-D by broadcasting
    x2d, y2d = grid.xcellcenters + xoff, grid.ycellcenters + yoff
    x3d = np.broadcast_to(x2d, (nlay, nrow, ncol))
    y3d = np.broadcast_to(y2d, (nlay, nrow, ncol))

    # Layer, row, col indices (also fully vectorised)
    lay = np.arange(nlay)[:, None, None]
    row, col = np.indices((nrow, ncol))
    lay3d = np.broadcast_to(lay, (nlay, nrow, ncol))
    row3d = np.broadcast_to(row, (nlay, nrow, ncol))
    col3d = np.broadcast_to(col, (nlay, nrow, ncol))

    # ibound from BAS6
    ibnd = mf.bas6.ibound.array

    # Flatten once and build the frame
    df = pd.DataFrame({
        "layer":  lay3d.ravel(),
        "row":    row3d.ravel(),
        "col":    col3d.ravel(),
        "X":      x3d.ravel(),
        "Y":      y3d.ravel(),
        "ibound": ibnd.ravel(),
    })

    if remove_inactive:
        df = df[df.ibound != 0].reset_index(drop=True)

    return df

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

# Read in MODFLOW model discretization
gwf = fp.modflow.Modflow.load((model_name + '.nam'), version='mfnwt', load_only=['dis','bas6'], model_ws=mf_dir)

# Read in log distribution pilot points
scale_pp = pd.read_csv(in_dir / 'pilot_point_scales.csv')
scale_pp['name'] = 'pp' + scale_pp.index.astype(str)
scale_pp.loc[scale_pp.Layer==1,'name'] = scale_pp.loc[scale_pp.Layer==0,'name'].values
scale_pp['zone'] = 0
scale_pp = scale_pp.rename({'X':'x', 'Y':'y'}, axis=1)
scale_pp_flat = scale_pp[scale_pp['Layer']==0]

# Get Kriging weights ("factors") for each point
grid_df = model_to_grid_df(gwf, xoff, yoff, remove_inactive=False)
grid_layer1_df = grid_df[grid_df['layer']==0]
scale_ok = pyemu.utils.geostats.OrdinaryKrige(scale_gs, scale_pp_flat)
if scale_factor_file.exists():
    print('Using existing factor file:', scale_factor_file)
else:
    scale_weights = scale_ok.calc_factors(grid_layer1_df['X'], grid_layer1_df['Y'], maxpts_interp=12)
    scale_ok.to_grid_factors_file(scale_factor_file, ncol=grid_layer1_df.shape[0])
    print('Cached scale factors to', scale_factor_file)

# Loop over layers, textures getting our final values
for k in range(0,gwf.nlay):
    for tex in tqdm(tex_dists.index):
        # write pp file
        this_pp = scale_pp[scale_pp['Layer']==k][['name','zone','x','y',tex]]
        this_pp = this_pp.rename({tex:'parval1'}, axis=1)
        pyemu.utils.pp_utils.write_pp_file(in_dir / f"scale_pp_{tex}.dat", this_pp)
        # Apply factors
        grid_df.loc[grid_df['layer']==k, tex] = (
            pyemu.utils.geostats.fac2real(pp_file=str(in_dir / f"scale_pp_{tex}.dat"),
                                      factors_file=str(scale_factor_file),
                                      out_file=None))[0]