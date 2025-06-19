import numpy as np
import pandas as pd
import geopandas as gpd
import flopy as fp
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

out_dir = Path('./04_InputFiles/CONTEX/')

out_dir.mkdir(parents=True, exist_ok=True)

layers = 2

#----------------------------------------------------------------------------------------------------------------------#
# Functions/Classes
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

#-- Setup initial pilot point file

# Read model outline
ppshp = gpd.read_file('./01_Data/GIS/t2p_pilot_points.shp')

# Read in initial shape parameters
lognorm_values = pd.read_table('./01_Data/lognorm_dist_clustered.par', sep='\\s+', skiprows=1)

# Assemble into pilot point dataframe

# Extract X, Y
ppshp['X'] = ppshp.geometry.x
ppshp['Y'] = ppshp.geometry.y

for i,row in lognorm_values.iterrows():
    ppshp[row.Texture] = row.Scale

# Add a layer column for each layer
npp = ppshp.shape[0]
ppshp['Layer'] = 0
pp_layers = pd.concat(
    [ppshp.assign(Layer=lyr) for lyr in range(layers)],
    ignore_index=True
)

# Pick out the columns you need and write CSV
out_cols = ['X', 'Y', 'Layer'] + list(lognorm_values.Texture)
pp_layers[out_cols].to_csv(out_dir / 'pilot_point_scales.csv', index=False)

print(f"Wrote {len(pp_layers)} pilot points × {lognorm_values.shape[0]} textures to pilot_point_scales.csv")

#----------------------------------------------------------------------------------------------------------------------#
# Pilot point distance average, to inform kriging

coords = ppshp[['X','Y']].to_numpy()
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
nn = nbrs.kneighbors(coords)[0][:,1]
mean_nn = nn.mean()

print(f'Mean PP neighbor distance:', round(mean_nn))