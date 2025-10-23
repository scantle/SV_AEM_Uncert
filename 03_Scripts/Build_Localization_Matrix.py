import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt
import pyemu
from sklearn.neighbors import KDTree  # fast radius queries
from pathlib import Path
from tqdm import tqdm

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

data_dir  = Path('01_Data')
pest_dir = Path("04_PEST_setup")

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
R_MULT = 3.0          # search radius = R_MULT * ah
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

# Get parameters in a specific format
pars_xy = (
    pp_pars[['parnme', 'pargp', 'X', 'Y', 'Layer']]
    .rename(columns={'X': 'x', 'Y': 'y'})
    .set_index('parnme')
)
pars_xy.index = pars_xy.index.str.lower()  # to appease pyemu

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

# Get the full row/col name sets (PEST++ requires full dimensions)
all_obs = obs_names.tolist()
all_pars = pars_xy.index.tolist()

M = pyemu.Matrix.from_names(row_names=all_obs, col_names=all_pars, isdiagonal=False)

# Map (row_name, col_name) -> integer indices efficiently
ri = pyemu.Matrix.find_rowcol_indices(trip_rows, M.row_names, M.col_names, axis=0)
ci = pyemu.Matrix.find_rowcol_indices(trip_cols, M.row_names, M.col_names, axis=1)
M.x[ri, ci] = trip_vals  # assign nonzeros

# Write the matrix for PEST++-IES
loc_df = pd.DataFrame(M.x, index=M.row_names, columns=M.col_names)
loc_path = pest_dir / "localizer.csv"
loc_df.to_csv(loc_path)

#----------------------------------------------------------------------------------------------------------------------#
# QA/QC
#----------------------------------------------------------------------------------------------------------------------#
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def qa_plot_obs_influence(obs_name, M, obs_xy, pars_xy, wmin=1e-6, annotate_top=0, print_top=0, figsize=(8,7)):
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


# Tests
qa_plot_obs_influence("d31_avg", M, obs_xy, pars_xy, wmin=0.01, print_top=100)
qa_plot_obs_influence("qv04_avg", M, obs_xy, pars_xy, wmin=0.01, print_top=20)
qa_plot_obs_influence("st201_avg", M, obs_xy, pars_xy, wmin=0.01, print_top=20)