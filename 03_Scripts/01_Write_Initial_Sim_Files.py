import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import flopy as fp
import json
import pyemu
from pathlib import Path
from tqdm import tqdm
from scipy.stats import lognorm
from sklearn.neighbors import NearestNeighbors

# Local
import sys
sys.path.append('./03_Scripts/')
from aem_read import read_xyz, aem_wide2long

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Directories
out_dir = Path('./04_InputFiles/RES2PAR/')
data_dir = Path ('./01_Data/')
shp_dir = data_dir / 'GIS/'
plt_dir = Path('./05_Plots/')

# Files
litho_file = data_dir / 'AEM_WELL_LITHOLOGY_csv_WO2_20220710_CVHM.csv'
aem_sharp_file = data_dir / 'SCI_Sharp_10_West_I01_MOD_inv.xyz'

# Shapefiles
aem_hqwells_file = shp_dir / 'aem_sv_HQ_LithologyWells_UTM10N.shp'
aem_lqwells_file = shp_dir / 'aem_sv_LQ_LithologyWells_UTM10N.shp'
aem_sharp_sv_file = shp_dir / 'aem_sv_Sharp_I01_MOD_inv_UTM10N.shp'

out_dir.mkdir(parents=True, exist_ok=True)

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330

# T2P-like Settings
min_log_length = 5  # meters
use_model_gse = True

# Cluster colors
cc = ['#df263e', '#e37e26', '#e3c128', '#6da14d', '#5289db']

#----------------------------------------------------------------------------------------------------------------------#
# Functions/Classes
#----------------------------------------------------------------------------------------------------------------------#

def enforce_min_interval(litho_df: pd.DataFrame,
                         min_log_length: float,
                         top_col: str = "LITH_TOP_DEPTH_m",
                         bot_col: str = "LITH_BOT_DEPTH_m",
                         add_midpoint: bool = True
                         ) -> pd.DataFrame:
    """
    Split any lithologic interval longer than `min_log_length` into
    equally-sized sub-intervals <= min_log_length, and leave shorter
    intervals alone.

    Returns a new DataFrame with the same columns as `litho_df` (plus
    optionally 'midpoint_m').
    """
    out_rows = []
    for _, row in tqdm(litho_df.iterrows(), desc='Interval', total=litho_df.shape[0]):
        top = row[top_col]
        bot = row[bot_col]
        thickness = bot - top

        if thickness <= min_log_length:
            # keep as-is
            new_row = row.copy()
            new_row['split'] = False
            if add_midpoint:
                new_row["midpoint_m"] = 0.5 * (top + bot)
            out_rows.append(new_row)
        else:
            # split into N pieces
            n_segs = int(np.ceil(thickness / min_log_length))
            seg_len = thickness / n_segs
            for i in range(n_segs):
                seg_top = top + i * seg_len
                seg_bot = seg_top + seg_len
                new_row = row.copy()
                new_row[top_col] = seg_top
                new_row[bot_col] = seg_bot
                new_row['split'] = True
                if add_midpoint:
                    new_row["midpoint_m"] = 0.5 * (seg_top + seg_bot)
                out_rows.append(new_row)

    result = pd.DataFrame(out_rows)
    return result.reset_index(drop=True)

#----------------------------------------------------------------------------------------------------------------------#

def get_layer(df, gwf: fp.modflow.Modflow, row_col='row', col_col='col', midpoint_col='midpoint_m'):
    df['layer'] = np.nan
    for idx, row in tqdm(df.iterrows(), desc='Interval', total=df.shape[0]):
         df.loc[idx,'layer'] = gwf.dis.get_layer(row[row_col], row[col_col], row[midpoint_col])
    return df

#----------------------------------------------------------------------------------------------------------------------#

def reclassify_texture(intv):
    tex = intv['Texture']
    # Cluster 0
    if tex in ['shale','claystone']:
        tex = 'Fine'
    # Cluster 1
    elif tex in ['clay','silt','loam','top soil']:
        tex = 'Mixed_Fine'
    # Cluster 2
    elif tex in ['sand']:
        tex = 'Sand'
    # Cluster 3
    elif tex in ['gravel','cobbles']:
        tex = 'Mixed_Coarse'
    # Cluster 4
    elif tex in ['boulders','sandstone','lava','lime','rock']:
        tex = 'Very_Coarse'
    # elif tex == 'unknown':
    #     tex = -999
    else:
        print('UNKNOWN TEXTURE:', intv[['Texture','Primary_Texture_Modifier']])
    return tex

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

#-- Read in GW model
gwf = fp.modflow.Modflow.load((model_name + '.nam'), version='mfnwt', load_only=['dis','bas6'], model_ws=mf_dir)
gwf.modelgrid.set_coord_info(xoff=xoff, yoff=yoff)
layers = gwf.nlay

#----------------------------------------------------------------------------------------------------------------------#

#-- Redo lognormal distributions from Scantlebury & Harter (202X) with no loc parameter

# Read in texture-resistivity boostrap analysis from the previous project
with open(data_dir / 'rho_dict.json', 'r') as f:
    rho_dict = json.load(f)

# Convert keys to int and values to np.array
rho_dict = {int(k): np.array(v) for k, v in rho_dict.items()}

cluster_names = ['1 - Fine-grained', '2 - Mixed Fine', '3 - Sand', '4 - Mixed Coarse', '5 - Very Coarse']
tex_names = [name.split('-')[1].strip().replace(' ', '_') for name in cluster_names]

plt.style.use('seaborn-v0_8-colorblind')
hplt, hax = plt.subplots(figsize=(12, 8))
fit_dists = {}
hax.grid(which='both', linestyle='-', linewidth='0.5', color='lightgrey')
hist_patches = []

for tex in rho_dict.keys():
    bins = np.logspace(np.log10(np.min(rho_dict[tex])), np.log10(np.max(rho_dict[tex])), num=40)
    ptch = hax.hist(rho_dict[tex], bins=bins, alpha=0.5, density=True, zorder=2, label=cluster_names[tex], color=cc[tex])
    hist_patches.extend(ptch[2])
    shape, loc, scale = lognorm.fit(rho_dict[tex], floc=0)
    x = np.linspace(0, 500, 1000)
    y = lognorm.pdf(x, s=shape, loc=loc, scale=scale)
    plt.plot(x, y, zorder=2, color=cc[tex], lw=2)
    fit_dists[tex] = (shape, loc, scale)
    print(tex, shape, loc, scale)
hax.set_xscale('log')
medians = [round(np.median(rho_dict[tex])) for tex in rho_dict.keys()]
newticks = [10,20,50,100,150,200,300,400]
plt.xticks(newticks, [f'{t}' for t in newticks])
hax.set_xlim(10,500)

hax.set_xlabel(r'Resistivity (log scale), $\rho\, [\Omega\cdot\mathrm{m}]$', fontsize=15)
hax.set_ylabel('Density', fontsize=15)
leg = hax.legend(fontsize=13, title='Texture Clusters', title_fontsize=15)
hplt.tight_layout()
plt.savefig(plt_dir / '01_histogram_resistivity_clusters.png', dpi=300, bbox_inches='tight')

with open(out_dir / 'lognorm_dist_clustered.par', 'w') as f:
    f.write(f"{len(fit_dists.keys())}    # Number of texture classes\n")
    f.write(f"{'Texture':>15}{'Shape':>12}{'Location':>12}{'Scale':>12}\n")
    for tex in fit_dists.keys():
        f.write(f"{tex_names[tex]:15}{fit_dists[tex][0]:12.6f}{fit_dists[tex][1]:12.6f}{fit_dists[tex][2]:12.6f}\n")

#----------------------------------------------------------------------------------------------------------------------#

#-- Setup initial pilot point file

# Read model outline
ppshp = gpd.read_file(shp_dir / 't2p_pilot_points.shp')

# Re-read in file
lognorm_values = pd.read_table(out_dir / 'lognorm_dist_clustered.par', sep='\\s+', skiprows=1)

# Assemble into pilot point dataframe

# Extract X, Y
ppshp['X'] = ppshp.geometry.x
ppshp['Y'] = ppshp.geometry.y

# Add in conceptual pp nugget values
con_pp_cols = ['lth_nugget', 'aem_nugget']
for col in con_pp_cols:
    ppshp[col] = 0.0

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
out_cols = ['X', 'Y', 'Layer'] + list(lognorm_values.Texture) + con_pp_cols
pp_layers[out_cols].to_csv(out_dir / 'pilot_point_values.csv', index=False)

print(f"Wrote {len(pp_layers)} pilot points × {lognorm_values.shape[0]} textures to pilot_point_values.csv")

#----------------------------------------------------------------------------------------------------------------------#
# Pilot point distance average, to inform kriging

coords = ppshp[['X','Y']].to_numpy()
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
nn = nbrs.kneighbors(coords)[0][:,1]
mean_nn = nn.mean()

print(f'Mean PP neighbor distance:', round(mean_nn))

#----------------------------------------------------------------------------------------------------------------------#

#-- Setup lithology log with model lay, row, col

# Read in logs
litho = pd.read_csv(litho_file)

# Read in litho_df shapefiles
aem_hqwells_shp = gpd.read_file(aem_hqwells_file)
aem_hqwells_shp.set_index('WELLINFOID', inplace=True)
aem_lqwells_shp = gpd.read_file(aem_lqwells_file)
aem_lqwells_shp.set_index('WELLINFOID', inplace=True)

# Combine well shapefiles
litho_shp = pd.concat([aem_lqwells_shp, aem_hqwells_shp])

# Locate in GW model, limit to logs within GW domain (including inactive cells)
litho_shp['x'], litho_shp['y'] = (litho_shp.geometry.x, litho_shp.geometry.y)
litho_shp['row'], litho_shp['col'] = (np.nan, np.nan)
for row, values in tqdm(litho_shp.iterrows()):
    litho_shp.loc[row, ['row','col']] = gwf.modelgrid.intersect(values.x, values.y, forgive=True)
litho_shp['row'] = litho_shp['row'].astype(int).values
litho_shp['col'] = litho_shp['col'].astype(int).values

# Merge new spatial information with litho_df
litho_shp['WELL_INFO_ID'] = litho_shp.index
litho = litho.merge(litho_shp[['WELL_INFO_ID','x','y','row','col']], on='WELL_INFO_ID', how='inner')

# Apply texture reclassification, drop unknowns
litho['tex'] = litho.apply(reclassify_texture, axis=1)
litho = litho[litho['tex']!='unknown']

# # Enforce log minimum thickness
# litho = enforce_min_interval(litho, min_log_length)

# Determine layer each interval is within
litho['midpoint_m'] = 0.5 * (litho['LITH_TOP_DEPTH_m'] + litho['LITH_BOT_DEPTH_m'])
if use_model_gse:
    litho['z'] = gwf.dis.gettop()[litho['row'].values, litho['col'].values] - litho['midpoint_m']
else:
    litho['z'] = litho['GROUND_SURFACE_ELEVATION_m'] - litho['midpoint_m']
litho['layer'] = gwf.dis.get_layer(litho['row'], litho['col'], litho['z'])

# Rename some cols
litho = litho.rename({'LITH_TOP_DEPTH_m': 'TOP_DEPTH_m', 'LITH_BOT_DEPTH_m': 'BOT_DEPTH_m'}, axis=1)

# Write out litho_df file
out_cols = ['WELL_INFO_ID','row','col','layer','x','y','GROUND_SURFACE_ELEVATION_m','TOP_DEPTH_m','BOT_DEPTH_m','tex']
litho[out_cols].to_csv(out_dir / 'lithologs.csv', index=False)

#----------------------------------------------------------------------------------------------------------------------#

#-- Process AEM data into depth logs

# Read in sharp AEM data
aem_wide = read_xyz(aem_sharp_file, 26, x_col='UTMX', y_col='UTMY', delim_whitespace=True)

# Read in shapefiles (converted to correct datum)
aem_shp = gpd.read_file(aem_sharp_sv_file)
aem_shp['x'] = aem_shp.geometry.x
aem_shp['y'] = aem_shp.geometry.y
aem_shp['row'], aem_shp['col'] = (np.nan, np.nan)
for row, values in tqdm(aem_shp.iterrows()):
    aem_shp.loc[row, ['row','col']] = gwf.modelgrid.intersect(values.x, values.y, forgive=True)

# Subset to Scott Valley, add various shp columns to wide
aem_wide = aem_wide[aem_wide['LINE_NO'].isin(aem_shp['LINE_NO'])]
aem_wide = aem_wide.merge(aem_shp[['LINE_NO', 'FID', 'x', 'y','row','col']], on=['LINE_NO','FID'], how='inner')

# Made data long (one point per row)
aem_long = aem_wide2long(aem_wide,
                         id_col_prefixes=['RHO_I', 'RHO_I_STD', 'SIGMA_I', 'DEP_TOP', 'DEP_BOT', 'THK', 'THK_STD', 'DEP_BOT_STD'],
                         line_col='LINE_NO')

# There's no bottom for point 30 at each point (lowest pixel), so drop those values
aem_long = aem_long.dropna(subset='DEP_BOT')

# Drop entries below DOI (conservative)
aem_long = aem_long.loc[(aem_long['DEP_TOP'] < aem_long['DOI_CONSERVATIVE'])]

# Drop any NA Rho values
aem_long = aem_long.loc[~aem_long['RHO_I'].isna()]

# # Enforce log minimum thickness
# aem_long = enforce_min_interval(aem_long, min_log_length, top_col='DEP_TOP', bot_col='DEP_BOT')
#
# Determine layer each interval is within
aem_long['row'] = aem_long['row'].astype(int).values
aem_long['col'] = aem_long['col'].astype(int).values
aem_long['midpoint_m'] = 0.5 * (aem_long['DEP_TOP'] + aem_long['DEP_BOT'])
if use_model_gse:
    aem_long['z'] = gwf.dis.gettop()[aem_long['row'].values, aem_long['col'].values] - aem_long['midpoint_m']
else:
    aem_long['z'] = aem_long['ELEVATION'] - aem_long['midpoint_m']
aem_long['layer'] = gwf.dis.get_layer(aem_long['row'], aem_long['col'], aem_long['z'])

# Rename some cols
aem_long = aem_long.rename({'DEP_TOP': 'TOP_DEPTH_m', 'DEP_BOT': 'BOT_DEPTH_m', 'ELEVATION':'GROUND_SURFACE_ELEVATION_m'}, axis=1)

# Write out litho_df file
out_cols = ['LINE_NO','FID','row','col','layer','x','y','GROUND_SURFACE_ELEVATION_m','TOP_DEPTH_m','BOT_DEPTH_m','RHO_I','RHO_I_STD']
aem_long[out_cols].to_csv(out_dir / 'aemlogs.csv', index=False)

#----------------------------------------------------------------------------------------------------------------------#
