"""
Texture2Par "RES2PAR" Preprocessor
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import flopy as fp
import pyemu
from tqdm import tqdm
from pathlib import Path
from scipy.stats import lognorm
import t2py

import geopandas as gpd

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Input Files
in_dir = Path('./04_InputFiles/RES2PAR/')
pp_factor_file = in_dir / 'pp_factors.dat'
tex_dist_file = in_dir / 'lognorm_dist_clustered.par'
sv_model_shp_file = Path('./01_Data/GIS/') / 'grid_properties_rep.shp'

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330

# Log Distribution pilot point variogram
scale_vario   = pyemu.geostats.SphVario(contribution=1.0, a=2317*3)
scale_gs  = pyemu.geostats.GeoStruct(variograms=[scale_vario])

seed = 667

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

    # Centers
    x2d, y2d = grid.xcellcenters + xoff, grid.ycellcenters + yoff
    x3d = np.broadcast_to(x2d, (nlay, nrow, ncol))
    y3d = np.broadcast_to(y2d, (nlay, nrow, ncol))

    # Assemble layer center z-elevations
    top2d  = mf.dis.top.array
    botm3d = mf.dis.botm.array
    z3d = np.empty((nlay, nrow, ncol), dtype=float)
    for k in range(nlay):
        if k == 0:
            z3d[k] = 0.5 * (top2d + botm3d[k])
        else:
            z3d[k] = 0.5 * (botm3d[k - 1] + botm3d[k])

    # Layer, row, col indices
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
        "Z":      z3d.ravel(),
        "ibound": ibnd.ravel(),
    })

    if remove_inactive:
        df = df[df.ibound != 0].reset_index(drop=True)

    return df

#----------------------------------------------------------------------------------------------------------------------#

def aem2texture(rho, parameters, scales=None):
    probabilities = {}
    psum = 0.0
    for tex in parameters.keys():
        if scales is None:
            probabilities[tex] = lognorm.pdf(rho, s=parameters[tex][0], loc=parameters[tex][1],
                                             scale=parameters[tex][2])
        else:
            probabilities[tex] = lognorm.pdf(rho, s=parameters[tex][0], loc=parameters[tex][1],
                                             scale=scales[tex].values[0])
        psum += probabilities[tex]
    # Normalize:
    for tex in list(parameters.keys()):
        probabilities[tex] = probabilities[tex]/ psum
    return probabilities

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

# Read in MODFLOW model discretization
gwf = fp.modflow.Modflow.load((model_name + '.nam'), version='mfnwt', load_only=['dis','bas6'], model_ws=mf_dir)

# Read in texture distribution priors
tex_dists_df = pd.read_table(tex_dist_file, sep='\\s+', skiprows=1)
tex_dists = tex_dists_df.set_index("Texture")[["Shape","Location","Scale"]].T.to_dict("list")

#----------------------------------------------------------------------------------------------------------------------#
# Pilot Point Kriging
#----------------------------------------------------------------------------------------------------------------------#

# Read in log distribution/nugget pilot points & values
pp = pd.read_csv(in_dir / 'pilot_point_values.csv')
pp['name'] = 'pp' + pp.index.astype(str)
pp.loc[pp.Layer==1,'name'] = pp.loc[pp.Layer==0,'name'].values
pp['zone'] = 0
pp = pp.rename({'X':'x', 'Y':'y'}, axis=1)
pp_flat = pp[pp['Layer']==0]

# Get Kriging weights ("factors") for each point
grid_df = model_to_grid_df(gwf, xoff, yoff, remove_inactive=False)
grid_layer1_df = grid_df[grid_df['layer']==0]
pp_ok = pyemu.utils.geostats.OrdinaryKrige(scale_gs, pp_flat)
if pp_factor_file.exists():
    print('Using existing factor file:', pp_factor_file)
else:
    pp_weight = pp_ok.calc_factors(grid_layer1_df['X'], grid_layer1_df['Y'], maxpts_interp=12)
    pp_ok.to_grid_factors_file(pp_factor_file, ncol=grid_layer1_df.shape[0])
    print('Cached pilot point factors to', pp_factor_file)

# Loop over layers, (textures, nuggets) getting our final values
for k in range(0,gwf.nlay):
    # Textures
    for tex in tqdm(tex_dists.keys(), f'Layer: {k} Texture', len(tex_dists.keys())):
        # write pp file
        this_pp = pp[pp['Layer']==k][['name','zone','x','y',tex]]
        this_pp = this_pp.rename({tex:'parval1'}, axis=1)
        pyemu.utils.pp_utils.write_pp_file(in_dir / f"scale_pp_{tex}.dat", this_pp)
        # Apply factors
        grid_df.loc[grid_df['layer']==k, tex] = (
            pyemu.utils.geostats.fac2real(pp_file=str(in_dir / f"scale_pp_{tex}.dat"),
                                      factors_file=str(pp_factor_file),
                                      out_file=None))[0]
    # Nuggets
    for nug in tqdm(['lth_nugget', 'aem_nugget'], f'Layer: {k} Nugget', 2):
        # write pp file
        this_pp = pp[pp['Layer']==k][['name','zone','x','y',nug]]
        this_pp = this_pp.rename({nug:'parval1'}, axis=1)
        pyemu.utils.pp_utils.write_pp_file(in_dir / f"pp_{nug}.dat", this_pp)
        # Apply factors
        grid_df.loc[grid_df['layer']==k, nug] = (
            pyemu.utils.geostats.fac2real(pp_file=str(in_dir / f"pp_{nug}.dat"),
                                      factors_file=str(pp_factor_file),
                                      out_file=None))[0]

#----------------------------------------------------------------------------------------------------------------------#
# Lithology Conversion
#----------------------------------------------------------------------------------------------------------------------#

# Read in lithology logs
litho = pd.read_csv(in_dir / 'lithologs.csv')
litho['data_type'] = 'litho'

# Get resistivity values
litho['RHO_I'] = np.nan
litho['RHO_I_STD'] = np.nan
for tex in tex_dists.keys():
    litho[f're_{tex}'] = np.nan

for idx, row in tqdm(litho.iterrows(), 'Interval', litho.shape[0]):
    # Get log dist mode for cell: loc + scale * np.exp(-shape**2)
    tex = row['tex']
    grid_cell = grid_df.loc[(grid_df.row == row.row) & (grid_df.col == row.col) & (grid_df.layer == row.layer),]
    scale = grid_cell[tex].values[0]
    litho.loc[idx,'RHO_I'] = np.log(scale)          # log(scale) is the log-space mean
    litho.loc[idx,'RHO_I_STD'] = tex_dists[tex][0]  # Shape parameter is log(std)
    # Test conversion back to texture
    retex = aem2texture(np.exp(litho.loc[idx,'RHO_I']), tex_dists, scales=grid_cell)
    for tex in tex_dists.keys():
        litho.loc[idx, f're_{tex}'] = retex[tex]

#----------------------------------------------------------------------------------------------------------------------#
# Resistivity Kriging
#----------------------------------------------------------------------------------------------------------------------#

# Read in AEM resistivity values
aem = pd.read_csv(in_dir / 'aemlogs.csv')
aem['data_type'] = 'aem'

# Need to convert RHO_I_STD to log using delta method

# Combine dataframes
use_cols = ['x','y','z','row','col','layer','RHO_I','RHO_I_STD','data_type']
resdf = pd.concat([litho[use_cols],aem[use_cols]])

# Let's work with the natural log of AEM
resdf['logrho'] = np.log(resdf['RHO_I'])
