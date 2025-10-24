import numpy as np
import pandas as pd
import glob
import os
import geopandas as gpd
from shapely.geometry import Point
from collections import defaultdict
from math import sqrt
import pyemu
from sklearn.neighbors import KDTree  # fast radius queries
from pathlib import Path
from tqdm import tqdm


#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

data_dir = Path('01_Data')
pest_dir = Path("04_PEST_setup")
gis_dir  = data_dir / 'GIS'

# For various reasons, I'm not allowed to share this file with the well locations
# Please contact me if you have any questions!
hob_key_file = Path('C:/Users/lelan/Documents/CodeProjects/SVIHM/SVIHM_Input_Files/reference_data_for_plots/_hob_key.csv')

# horizontal / vertical ranges (meters) and sills by PP family - Exponential Variogram
# must match 04_Write_Initial_PilotPoint_Files.py
# (but here we have a vertical to constrain variation between layers)
# (I've used the resistivity vertical range)
PP_GS = {
    "aem_var":   dict(ah=40*2,   av=78.4,   sill=1.0, nugget=0.0),
    "lth_var":   dict(ah=259*1,  av=78.4,   sill=1.0, nugget=0.0),
    "kv_mult":   dict(ah=93*2,   av=78.4,   sill=1.0, nugget=0.0),
    # 'scale_*' handled specially below per texture with same ranges
    "scale":     dict(ah=2317*2, av=78.4,   sill=1.0, nugget=0.0),
}

# Parameters for localization kernel
# max multiple of horizontal range to search, correlation threshold to keep
R_MULT = 4.0          # search radius = R_MULT * ah
SCALE_R_MULT = 2.0    # search radius for scale params (have a long range already...)
RHO_MIN = 0.01        # prune tiny weights
USE_SQUARED = False   # optionally square rho for tighter localization

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def groupto_struct(pargp: str):
    if pargp in PP_GS:
        return PP_GS[pargp]
    # treat any 'scale_*' as 'scale'
    if pargp.startswith('scale_'):
        return PP_GS['scale']
    raise KeyError(f"pargp '{pargp}' not recognized in PP_GS (and not a 'scale_*').")

#----------------------------------------------------------------------------------------------------------------------#

def exp_kernel(dx, dy, ah):
    # anisotropic metric horizontally only
    m = np.sqrt((dx/ah)**2 + (dy/ah)**2)
    rho = np.exp(-m)
    return rho

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

# Read in private hob location file
hob_key = pd.read_csv(hob_key_file)
hob_key = hob_key[['well_id', 'x_proj', 'y_proj']].drop_duplicates()

# Read in head observations written by 05_B_build_HOB_ins_n_obs.py
hob_obs = pd.read_csv(data_dir / 'head_obs_master.csv')
hob_obs = hob_obs.rename({'obval': 'obsval', 'group': 'obgnme', 'stdev': 'standard_deviation'}, axis=1)

# Drop zero-weighted obs
hob_obs = hob_obs.loc[hob_obs['weight']>0.0]

# Read pilot point files
pp_init_files = [item for item in sorted((data_dir / 'pp_init_csv').glob("**/*.csv"))]
pp_init_dfs = []
for pp_init in pp_init_files:
    df = pd.read_csv(pp_init)
    pp_init_dfs.append(df)
pp_pars = pd.concat(pp_init_dfs)

# Merge to get HOB locations
hob_df = hob_obs.merge(hob_key, how='left', left_on='wellid', right_on='well_id')
if hob_df[hob_df[['x_proj', 'y_proj']].isna().any(axis=1)].shape[0] > 0:
    print('Error - still missing coordinates!')

# Read in TPL files to get remaining parameters
# Template files with parameters => IN files
tpl_files = [item for item in glob.glob( '04_PEST_setup/*.tpl') if 'svihmt2p' not in item]
in_files = []
for f in tpl_files:  # Where they get written by PEST
    in_file = f.removesuffix('.tpl')
    if in_file in ['t2p_par2par.in', 'sfr2par.in']:
        in_files.append('SVIHM/' + in_file)
    elif in_file in ['catchment_mult.txt','landcover_table.txt','streamflow_multipliers.txt']:
        in_files.append('SVIHM/SWBM/' + in_file)
    else:  # most
        in_files.append('SVIHM/preproc/' + in_file)

# Instruction files with observations
ins_files = sorted(glob.glob("04_PEST_setup/*.ins"))
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
                              pst_filename='temp.pst')
os.remove('temp.pst')  # immediate cleanup :)

# It would be much simpler to read in the PST file but that would create a circular dependency...
# so alas... update the pst object with the head weights
hob_obs = pd.read_csv(data_dir / 'head_obs_master.csv', index_col='obsnme')
hob_obs = hob_obs.rename({'obval': 'obsval', 'group': 'obgnme', 'stdev': 'standard_deviation'}, axis=1)
obs = pst.observation_data
for df in [hob_obs]:
    df.index = df.index.str.lower()
    for col in ["obsval","weight","obgnme", "standard_deviation"]:
        if col in df.columns:
            obs.loc[obs.index.intersection(df.index), col] = df.loc[obs.index.intersection(df.index), col]

# Stream NSE/KGE/RMSE not in observations
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'obsval'] = 1.0
obs.loc[obs.index.str.contains('nse|kge|rmse'), 'weight'] = 0.0

#----------------------------------------------------------------------------------------------------------------------#
# Separate out Quartz Valley using a shapefile
#----------------------------------------------------------------------------------------------------------------------#

qtz_gdf = gpd.read_file(gis_dir / 'quartz_valley_poly.shp')


#----------------------------------------------------------------------------------------------------------------------#
# Get those distances, build that matrix
#----------------------------------------------------------------------------------------------------------------------#

# Get unique locs in a specific format
obs_xy = (
    hob_df[['obsnme', 'x_proj', 'y_proj']]
    .drop_duplicates(subset=['obsnme'])
    .set_index('obsnme')[['x_proj', 'y_proj']]
    .rename(columns={'x_proj': 'x', 'y_proj': 'y'})
)
obs_xy.index = obs_xy.index.str.lower()  # to appease pyemu
obs_xy['quartz'] = [qtz_gdf.contains(Point(x, y))[0] for x, y in obs_xy[["x", "y"]].to_numpy()]

# Get parameters in a specific format
pars_xy = (
    pp_pars[['parnme', 'pargp', 'X', 'Y', 'Layer']]
    .rename(columns={'X': 'x', 'Y': 'y'})
    .set_index('parnme')
)
pars_xy.index = pars_xy.index.str.lower()  # to appease pyemu
pars_xy['quartz'] = [qtz_gdf.contains(Point(x, y))[0] for x, y in pars_xy[["x", "y"]].to_numpy()]

# Build KD-trees per param group for fast neighbor search
group_to_pars = defaultdict(list)
for p, row in pars_xy.iterrows():
    group_to_pars[row.pargp].append(p)

trees = {}
group_arrays = {}
for g, pnames in group_to_pars.items():
    pts = pars_xy.loc[pnames, ['x','y']].values
    group_arrays[g] = (np.array(pnames, dtype=object), pts)
    if pts.shape[0] > 0:
        trees[g] = KDTree(pts)
    else:
        trees[g] = None

# Build sparse triplets (row=obsnme, col=parnme, val=rho)
obs_names = obs_xy.index.values
obs_coords = obs_xy[['x','y']].values

trip_rows = []
trip_cols = []
trip_vals = []

for oi in tqdm(range(obs_coords.shape[0]), desc="Computing localizer"):
    ox, oy = obs_coords[oi, :]
    # loop over parameter groups so each group uses its own range
    for g, tree in trees.items():
        if tree is None:
            continue
        g_struct = groupto_struct(g)
        ah = float(g_struct['ah'])
        if g.startswith('scale'):
            # reign that crazy range in
            r_search = SCALE_R_MULT * ah
        else:
            r_search = R_MULT * ah

        # query neighbors
        idxs = tree.query_radius(np.array([[ox, oy]]), r=r_search, return_distance=False)[0]
        if idxs.size == 0:
            continue

        parnames_g, parpts_g = group_arrays[g]
        cand_xy = parpts_g[idxs, :]
        dx = cand_xy[:,0] - ox
        dy = cand_xy[:,1] - oy

        rho = exp_kernel(dx, dy, ah)

        if USE_SQUARED:
            rho = rho**2

        # prune tiny
        keep = rho >= RHO_MIN
        if not np.any(keep):
            continue

        kept_idxs = idxs[keep]
        kept_vals = rho[keep]
        kept_pars = parnames_g[kept_idxs]

        # store triplets
        trip_rows.extend([obs_names[oi]] * kept_pars.size)
        trip_cols.extend(kept_pars.tolist())
        trip_vals.extend(kept_vals.tolist())

# Make sure we have at least something
if len(trip_vals) == 0:
    raise RuntimeError("Localizer ended up empty. Check ranges (ah), R_MULT and RHO_MIN.")

# Assemble into a pyemu Matrix and write to PEST++ matrix format

# Build the full matrix with zeros everywhere else
all_obs = pst.nnz_obs_names  # obs_xy.index.tolist()
all_pars = pst.adj_par_names  # pars_xy.index.tolist()
M = pyemu.Matrix.from_names(row_names=all_obs, col_names=all_pars, isdiagonal=False)

# Map triplets to indices once (fast)
ri = pyemu.Matrix.find_rowcol_indices(trip_rows, M.row_names, M.col_names, axis=0)
ci = pyemu.Matrix.find_rowcol_indices(trip_cols, M.row_names, M.col_names, axis=1)

# Assign nonzeros; clip to [0,1] just in case
#vals = np.asarray(trip_vals, float)
#vals = np.clip(vals, 0.0, 1.0)

# forget all that complication, set all non-zero values to 1 so the auto-localizer isn't constrained
M.x[ri, ci] = 1.0  #vals

# Assign non-spatial ("global") parameters to influence all observations
glo_par = [par for par in all_pars if par not in pars_xy.index]
gi = pyemu.Matrix.find_rowcol_indices(glo_par, M.row_names, M.col_names, axis=1)
M.x[:, gi] = 1.0

# Make sure non-spatial (streamflow) observations can be influenced by parameters
glo_obs = [ob for ob in all_obs if ob not in obs_xy.index]
oi = pyemu.Matrix.find_rowcol_indices(glo_obs, M.row_names, M.col_names, axis=0)
M.x[oi, :] = 1.0

#-- It gets complicated.
ss_subs = ['str_','sbcm', 'catch_']
semi_spatial_pars = [item for item in glo_par if any(sub in item for sub in ss_subs)]

#-- Quartz Valley special rules
glo_par_quartz = ['str_mult_shackleford', 'str_mult_mill', 'sbcm27', 'sbcm28', 'sbcm29', 'catch_mult_08', 'catch_mult_01',
              'catch_mult_05', 'catch_mult_35', 'catch_mult_04', 'catch_mult_24']
glo_obs_quartz = [obs for obs in glo_obs if obs.startswith('sck_')]
glo_par_nonquartz = [par for par in semi_spatial_pars if par not in glo_par_quartz]
glo_obs_nonquartz = [obs for obs in glo_obs if not (obs.startswith('sck_') or obs.startswith('fj_'))]
q_obs_in_names  = obs_xy.index[obs_xy['quartz']].tolist() + glo_obs_quartz
q_obs_out_names = obs_xy.index[~obs_xy['quartz']].tolist() + glo_obs_nonquartz
q_par_in_names  = pars_xy.index[pars_xy['quartz']].tolist() + glo_par_quartz
q_par_out_names = pars_xy.index[~pars_xy['quartz']].tolist() + glo_par_nonquartz

# Convert to integer indices in M
qoi  = pyemu.Matrix.find_rowcol_indices(q_obs_in_names,  M.row_names, M.col_names, axis=0)
nqoi = pyemu.Matrix.find_rowcol_indices(q_obs_out_names, M.row_names, M.col_names, axis=0)
qpi  = pyemu.Matrix.find_rowcol_indices(q_par_in_names,  M.row_names, M.col_names, axis=1)
nqpi = pyemu.Matrix.find_rowcol_indices(q_par_out_names, M.row_names, M.col_names, axis=1)

# observations inside quartz valley should only influence pars in quartz valley
M.x[np.ix_(nqoi, qpi)] = 0.0  # obs outside, parameters inside
M.x[np.ix_(qoi, nqpi)] = 0.0  # obs inside, parameters outside

#-- Stream gauges should only influence streamflow, sfr k multipliers, catchment mult upstream

# BY is the most upstream gauge
by_catch_in = ['catch_mult_02', 'catch_mult_07', 'catch_mult_09', 'catch_mult_10', 'catch_mult_16', 'catch_mult_06',
               'catch_mult_11', 'catch_mult_37', 'catch_mult_40', 'catch_mult_38','catch_mult_36']
by_str_mult_in = ['str_mult_wildcat', 'str_mult_mcc_main', 'str_mult_mcc_branch', 'str_mult_miners', 'str_mult_french',
                  'str_mult_clark']
by_sbcm_in = ['sbcm01', 'sbcm02', 'sbcm03', 'sbcm04', 'sbcm05', 'sbcm06', 'sbcm07', 'sbcm08', 'sbcm09']

glo_par_by = by_catch_in + by_str_mult_in + by_sbcm_in
glo_par_nonby = [par for par in semi_spatial_pars if par not in glo_par_by]
glo_obs_by = [obs for obs in glo_obs if obs.startswith('by_')]

# Set BY obs to not influence downstream catchment/stream parameters
byoi  = pyemu.Matrix.find_rowcol_indices(glo_obs_by,  M.row_names, M.col_names, axis=0)
nbypi = pyemu.Matrix.find_rowcol_indices(glo_par_nonby, M.row_names, M.col_names, axis=1)
M.x[np.ix_(byoi, nbypi)] = 0.0  # by obs, parameters downstream

# AS is the next gauge downstream, pretty central, but on the main branch of the Scott
as_catch_out = ['catch_mult_08', 'catch_mult_01', 'catch_mult_05', 'catch_mult_35', 'catch_mult_04', 'catch_mult_24',
                'catch_mult_31', 'catch_mult_28', 'catch_mult_15', 'catch_mult_42', 'catch_mult_18', 'catch_mult_39',
                'catch_mult_12', 'catch_mult_20', 'catch_mult_29', 'catch_mult_19', 'catch_mult_21', 'catch_mult_14',
                'catch_mult_23', 'catch_mult_22']
as_catch_in = [item for item in semi_spatial_pars if (item.startswith('catch_') and item not in as_catch_out)]
as_str_mult_in =  by_str_mult_in + ['str_mult_etna', 'str_mult_hearts', 'str_mult_shell']
as_sbcm_in = by_sbcm_in + ['sbcm10', 'sbcm11', 'sbcm12', 'sbcm13']

glo_par_as = as_catch_in + as_str_mult_in + as_sbcm_in
glo_par_nonas = [par for par in semi_spatial_pars if par not in glo_par_as]
glo_obs_as = [obs for obs in glo_obs if obs.startswith('as_')]

# Set AS obs to not influence downstream catchment/stream parameters
asoi  = pyemu.Matrix.find_rowcol_indices(glo_obs_as,  M.row_names, M.col_names, axis=0)
naspi = pyemu.Matrix.find_rowcol_indices(glo_par_nonas, M.row_names, M.col_names, axis=1)
M.x[np.ix_(asoi, naspi)] = 0.0  # by obs, parameters downstream

# Write SPARSE binary localizer (rectangular)
out_path = pest_dir / "localizer.jcb"
M.to_coo(out_path)
print(f"Wrote sparse localizer to {out_path}")

#----------------------------------------------------------------------------------------------------------------------#
# QA/QC
#----------------------------------------------------------------------------------------------------------------------#
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def qa_plot_obs_influence(obs_name, M, obs_xy, pars_xy, wmin=1e-6, annotate_top=0, print_top=0, figsize=(8,7),
                          return_df = False):
    rn = [n.lower() for n in M.row_names]
    cn = [n.lower() for n in M.col_names]
    o = obs_name
    if o not in rn:
        raise KeyError(f"Obs '{obs_name}' not in M.row_names")
    i = rn.index(o)
    w = np.asarray(M.x[i, :]).ravel()
    w_series = pd.Series(w, index=cn, name='w')
    pars_xy_ = pars_xy.copy()
    df = pars_xy_.join(w_series, how='left').fillna({'w':0.0})
    n_pos = int((df['w']>0).sum()); n_strong = int((df['w']>=wmin).sum())
    print(f"[{obs_name}] nonzero pars: {n_pos:,}  (>= {wmin:g}: {n_strong:,})")
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(df['x'], df['y'], c=df['w'], s=14, cmap='viridis')
    cb = plt.colorbar(sc, ax=ax, shrink=0.9); cb.set_label('Localizer weight (param → obs)')
    ax.set_aspect('equal', adjustable='box')
    # mark the obs location
    obs_xy_ = obs_xy.copy()
    if o in obs_xy_.index:
        ox, oy = obs_xy_.loc[o, ['x','y']]
        ax.scatter([ox],[oy], s=60, marker='x', linewidths=2, color='red', label='observation')
        ax.legend(loc='best')
    ax.set_title(f"Influence map for observation: {obs_name}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    if annotate_top and n_pos>0:
        top = df.sort_values('w', ascending=False).head(annotate_top)
        for parnme, r in top.iterrows():
            ax.annotate(parnme, (r['x'], r['y']), fontsize=8, xytext=(3,3), textcoords='offset points')
    if print_top and n_pos>0:
        print(df.sort_values('w', ascending=False).head(print_top))
    plt.tight_layout()
    if return_df:
        return df

# HDS Tests
qa_plot_obs_influence("d31_avg", M, obs_xy, pars_xy, wmin=0.01)
qa_plot_obs_influence("qv04_avg", M, obs_xy, pars_xy, wmin=0.01)
qa_plot_obs_influence("st201_avg", M, obs_xy, pars_xy, wmin=0.01)
qa_plot_obs_influence("scv_11_avg", M, obs_xy, pars_xy, wmin=0.01)

# Stream tests
qa_plot_obs_influence("fj_2019-02_vol", M, obs_xy, pars_xy, wmin=0.01)  # should be everything

# Shackleford Creek
qa_plot_obs_influence("sck_20071203", M, obs_xy, pars_xy, wmin=0.01)