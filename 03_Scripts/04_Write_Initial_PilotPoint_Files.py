import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import flopy as fp
import pyemu
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from sklearn.neighbors import NearestNeighbors

# for debugging console
# import os
# os.chdir("../")

# ----------------------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------------------

# Directories
out_dir   = Path('./04_InputFiles/RES2PAR/')
pest_dir  = Path('./04_InputFiles/PEST/')
data_dir  = Path('./01_Data/')
shp_dir   = data_dir / 'GIS/'
plt_dir   = Path('./05_Plots/')

out_dir.mkdir(parents=True, exist_ok=True)
pest_dir.mkdir(parents=True, exist_ok=True)

# MODFLOW
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330
layers = 2

# Filtering near Layer 2
boundary_threshold = 250.0  # meters

# Textures (for scale_pp parameters)
texs = ['Fine', 'Mixed_Fine', 'Sand', 'Mixed_Coarse', 'Very_Coarse']

# Variograms (one GeoStruct per PP-type)
scale_gs = pyemu.geostats.GeoStruct(variograms=[
    pyemu.geostats.SphVario(contribution=1.0, a=2317*3)
])
lth_nug_gs = pyemu.geostats.GeoStruct(variograms=[
    pyemu.geostats.SphVario(contribution=1.0, a=259*3)
])
aem_nug_gs = pyemu.geostats.GeoStruct(variograms=[
    pyemu.geostats.SphVario(contribution=1.0, a=40*3)
])
kv_mult_gs = pyemu.geostats.GeoStruct(variograms=[
    pyemu.geostats.SphVario(contribution=1.0, a=93*3)
])

# Per-PP-set config (locations are fixed here; values come from PEST later)
PPSETS = {
    "scale_pp": {
        "shp": shp_dir / 'scale_pp.shp',
        "targets": texs,                         # one param per texture (re-uses same factors)
        "gs": scale_gs,
        "tpl_pattern": 'scale_pp_{tex}_L{lay}.dat.tpl', # template per texture
        "dat_pattern": 'scale_pp_{tex}_L{lay}.dat',     # PEST will write this from tpl
        "maxpts": 16
    },
    "lth_var_pp": {
        "shp": shp_dir / 'litho_pp.shp',
        "targets": ['lth_var'],
        "gs": lth_nug_gs,
        "tpl_pattern": 'pp_lth_var_L{lay}.dat.tpl',
        "dat_pattern": 'pp_lth_var_L{lay}.dat',
        "maxpts": 16
    },
    "aem_var_pp": {
        "shp": shp_dir / 'aem_pp.shp',
        "targets": ['aem_var'],
        "gs": aem_nug_gs,
        "tpl_pattern": 'pp_aem_var_L{lay}.dat.tpl',
        "dat_pattern": 'pp_aem_var_L{lay}.dat',
        "maxpts": 16
    },
    "kv_mult_pp": {
        "shp": shp_dir / 'mult_pp.shp',
        "targets": ['kv_mult'],
        "gs": kv_mult_gs,
        "tpl_pattern": 'pp_kv_var_L{lay}.dat.tpl',
        "dat_pattern": 'pp_kv_var_L{lay}.dat',
        "maxpts": 16
    },
}

# ----------------------------------------------------------------------------------------------------------------------
# Functions/Classes
# ----------------------------------------------------------------------------------------------------------------------

def active_polygon_from_mask(mg, mask2d):
    xcc, ycc = mg.xcellcenters, mg.ycellcenters
    dx = np.median(np.diff(np.unique(xcc[0, :]))) if mg.ncol > 1 else 0.0
    dy = np.median(np.diff(np.unique(ycc[:, 0]))) if mg.nrow > 1 else 0.0
    half = min(dx, dy) / 2.0
    rr, cc = np.where(mask2d)
    if rr.size == 0:
        return Polygon()
    squares = [Point(xcc[r, c], ycc[r, c]).buffer(half, cap_style=3) for r, c in zip(rr, cc)]
    return unary_union(squares)

def model_to_grid_df(mf, xoff=0.0, yoff=0.0):
    grid = mf.modelgrid
    nlay, nrow, ncol = mf.dis.nlay, mf.dis.nrow, mf.dis.ncol
    x2d, y2d = grid.xcellcenters + xoff, grid.ycellcenters + yoff
    x3d = np.broadcast_to(x2d, (nlay, nrow, ncol))
    y3d = np.broadcast_to(y2d, (nlay, nrow, ncol))
    top2d  = mf.dis.top.array
    botm3d = mf.dis.botm.array
    z3d = np.empty((nlay, nrow, ncol), dtype=float)
    for k in range(nlay):
        z3d[k] = 0.5 * ((top2d if k == 0 else botm3d[k-1]) + botm3d[k])
    lay = np.arange(nlay)[:, None, None]
    row, col = np.indices((nrow, ncol))
    node = (lay * nrow * ncol + row * ncol + col)
    ibnd = mf.bas6.ibound.array
    df = pd.DataFrame({
        "node": node.ravel(order="C"),
        "layer": np.broadcast_to(np.arange(nlay)[:,None,None], (nlay,nrow,ncol)).ravel(),
        "row":   np.broadcast_to(row, (nlay,nrow,ncol)).ravel(),
        "col":   np.broadcast_to(col, (nlay,nrow,ncol)).ravel(),
        "X":     x3d.ravel(), "Y": y3d.ravel(), "Z": z3d.ravel(),
        "ibound": ibnd.ravel(),
    }).set_index("node")
    return df

def write_pp_tpl(df, tpl_path, parname_func):
    """
    Write a PEST/pyemu PP template file with columns:
    name zone x y parval1
    Values column is filled with $parameter_tokens$.
    """
    df = df.copy()
    df["zone"] = df.get("zone", 0)
    df["name"] = df["name"].astype(str)
    df["parval1"] = [f"$ {parname_func(r)} $" for _, r in df.iterrows()]
    with open(tpl_path, "w") as f:
        f.write("ptf $\n")
        for _, r in df.iterrows():
            # whitespace-delimited (no header)
            f.write(f"{r['name']} {int(r['zone'])} {r['X']:.3f} {r['Y']:.3f} {r['parval1']}\n")

def build_or_reuse_factors(pp_df_layer, gs, grid_X, grid_Y, out_fac_file, maxpts):
    """
    If factor file exists, do nothing. Else compute and save factors for this layer/grid.
    We use pyemu's to_grid_factors_file for native compatibility.
    """
    if Path(out_fac_file).exists():
        print(f"[factors] Using cached: {out_fac_file}")
        return
    print(f"[factors] Building: {out_fac_file}")
    ok = pyemu.utils.geostats.OrdinaryKrige(gs, pp_df_layer.rename(columns={"X":"x","Y":"y"}))
    fac = ok.calc_factors(grid_X, grid_Y, maxpts_interp=maxpts)
    # Write in pyemu format. ncol = number of grid targets on this layer
    ok.to_grid_factors_file(out_fac_file, ncol=len(grid_X))

def add_layer_column(df, nlay):
    """Duplicate rows across layers (Layer 0..nlay-1)."""
    out = pd.concat([df.assign(Layer=k) for k in range(nlay)], ignore_index=True)
    return out

def prep_pp_locations(shp_path, base_tag, allow_layer_dup=True):
    g = gpd.read_file(shp_path)
    g["X"] = g.geometry.x
    g["Y"] = g.geometry.y
    if "layer" in g.columns:
        # use layer from shapefile
        g = g.rename(columns={"layer":"Layer"})
    elif allow_layer_dup:
        g = add_layer_column(g[["X","Y"]], layers)
    else:
        g["Layer"] = 0

    # naming (stable)
    g = g.reset_index(drop=True)
    g["name"] = [f"pp_{base_tag}_L{int(L)}_{i+1}" for i, L in enumerate(g["Layer"].to_numpy())]
    g["zone"] = 0
    return g

# ----------------------------------------------------------------------------------------------------------------------
# Model/grid
# ----------------------------------------------------------------------------------------------------------------------

m = fp.modflow.Modflow.load(f"{model_name}.nam", model_ws=mf_dir, check=False, load_only=['DIS','BAS6'])
mg = m.modelgrid
mg.set_coord_info(xoff=xoff, yoff=yoff)

grid_df = model_to_grid_df(m)  # Already corrected x&y
ib = m.bas6.ibound.array
L2_mask = (ib[1, :, :] > 0)
L2_poly = active_polygon_from_mask(mg, L2_mask)

# ----------------------------------------------------------------------------------------------------------------------
# Read datasets for kriging AEM/borehole variance to
# ----------------------------------------------------------------------------------------------------------------------

borehole_data = pd.read_csv(out_dir / 'lithologs.csv')
aem_data = pd.read_csv(out_dir / 'aemlogs.csv')

# Constrain to unique locations, based on a name
aem_data['WELL_INFO_ID'] = aem_data['LINE_NO'].astype(int).astype(str) + "_" + aem_data['FID'].astype(int).astype(str)

# Duplicate targets across layers so we krige separately per layer
lth_targets = add_layer_column(
    borehole_data[['WELL_INFO_ID','x','y']].drop_duplicates().rename(columns={"x":"X","y":"Y"})[["WELL_INFO_ID","X","Y"]],
    layers
)
aem_targets = add_layer_column(
    aem_data[['WELL_INFO_ID','x','y']].drop_duplicates().rename(columns={"x":"X","y":"Y"})[["WELL_INFO_ID","X","Y"]],
    layers
)

# ----------------------------------------------------------------------------------------------------------------------
# Read texture initial scales (only to seed initial PP values for 'scale_pp')
# ----------------------------------------------------------------------------------------------------------------------
tex_file = data_dir / 'lognorm_dist_clustered.par'
tex_df = pd.read_table(tex_file, sep=r'\s+', skiprows=1)
tex_scale_init = dict(zip(tex_df.Texture, tex_df.Scale))

# ----------------------------------------------------------------------------------------------------------------------
# Build PP sets (locations, names, templates, and per-layer factor files)
# ----------------------------------------------------------------------------------------------------------------------

# Collect mean neighbor distances for reporting
nn_report = {}

for tag, cfg in tqdm(PPSETS.items(), 'PP Set', total=len(PPSETS.keys())):
    # PP locations
    base_tag = tag.replace("_pp","")
    pp_locs = prep_pp_locations(cfg["shp"], base_tag=base_tag, allow_layer_dup=True)

    # pyemu doesn't like uppercase letters in parameter names
    pp_locs.name = pp_locs.name.str.lower()

    # Special L2 filtering
    if "scale_pp" in tag:
        # mark near/far relative to L2
        def dist_to_L2(row):
            pt = Point(row["X"], row["Y"])
            if L2_poly.is_valid and pt.within(L2_poly):
                return 0.0
            return pt.distance(L2_poly)
        pp_locs["L2_dist_m"] = pp_locs.apply(dist_to_L2, axis=1)
        pp_locs["L2_status"] = pp_locs["L2_dist_m"].apply(
            lambda d: "inside" if d == 0.0 else ("near" if d <= boundary_threshold else "far")
        )
        before = len(pp_locs)
        mask_keep = ~((pp_locs["Layer"] == 1) & (pp_locs["L2_status"] == "far"))
        pp_locs = pp_locs[mask_keep].copy()
        print(f"[{tag}] Dropped {before - len(pp_locs)} far Layer-2 pilot points")

    # --- Write PP templates: one per layer ---
    for k in range(layers):
        ppl = pp_locs[pp_locs["Layer"] == k].copy()
        if ppl.empty:
            continue
        if tag == "scale_pp":
            # One template per texture per layer
            for tex in cfg["targets"]:
                tpl_path = pest_dir / cfg['tpl_pattern'].format(lay=k+1, tex=tex)
                def pname(row, t=tex, kk=k):
                    # unique, lowercase, layer-aware
                    return f"scale_{t}_l{kk}_{row.name + 1}"
                write_pp_tpl(ppl, tpl_path, parname_func=pname)
                print(f"[tpl] wrote {tpl_path}")
        else:
            par = cfg["targets"][0]
            tpl_path = pest_dir / cfg['tpl_pattern'].format(lay=k+1)
            def pname(row, p=par, kk=k):
                return f"{p}_l{kk}_{row.name + 1}"
            write_pp_tpl(ppl, tpl_path, parname_func=pname)
            print(f"[tpl] wrote {tpl_path}")

    # --- Build/reuse factor files per layer ---
    for k in range(layers):
        fac_file = out_dir / f"{tag}_L{k+1}.fac"  # reused across all textures if tag == scale_pp
        ppL = pp_locs[pp_locs["Layer"] == k][["name","zone","X","Y"]].copy()
        if ppL.empty:
            print(f"[factors] No PPs for {tag} layer {k+1} -> skip")
            continue

        # Choose kriging target set per PP type
        if tag == "lth_var_pp":
            tgtL = lth_targets[lth_targets["Layer"] == k]
            if tgtL.empty:
                print(f"[factors] No lithology targets for layer {k+1} -> skip")
                continue
            target_X = tgtL["X"].to_numpy()
            target_Y = tgtL["Y"].to_numpy()
            # save the target order used in the factor file rows:
            tgtL.assign(Layer=k)[["WELL_INFO_ID", "X", "Y", "Layer"]].to_csv(out_dir / f"{tag}_L{k + 1}_targets.csv", index=False)

        elif tag == "aem_var_pp":
            tgtL = aem_targets[aem_targets["Layer"] == k]
            if tgtL.empty:
                print(f"[factors] No AEM targets for layer {k+1} -> skip")
                continue
            target_X = tgtL["X"].to_numpy()
            target_Y = tgtL["Y"].to_numpy()
            tgtL.assign(Layer=k)[["WELL_INFO_ID", "X", "Y", "Layer"]].to_csv(out_dir / f"{tag}_L{k + 1}_targets.csv", index=False)

        else:
            # scale_pp and kv_mult_pp still target the model grid
            layer_nodes = grid_df[grid_df["layer"] == k]
            target_X = layer_nodes["X"].to_numpy()
            target_Y = layer_nodes["Y"].to_numpy()

        build_or_reuse_factors(
            pp_df_layer = ppL,
            gs          = cfg["gs"],
            grid_X      = target_X,
            grid_Y      = target_Y,
            out_fac_file= fac_file,
            maxpts      = cfg["maxpts"]
        )

    # --- Report mean NN distance (single layer geometry combined) ---
    coords = pp_locs[["X","Y"]].to_numpy()
    if coords.shape[0] >= 2:
        nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
        nn = nbrs.kneighbors(coords)[0][:,1]
        nn_report[tag] = float(np.mean(nn))

# ----------------------------------------------------------------------------------------------------------------------
# Write CSVs
# ----------------------------------------------------------------------------------------------------------------------

# Initial PP value tables: useful to seed starting values/out-of-loop inspection.
# For 'scale_pp', we can emit one CSV per texture with initial "parval1" from tex_scale_init.
init_dir = out_dir / "../pp_init_csv"
init_dir.mkdir(exist_ok=True, parents=True)

for tag, cfg in tqdm(PPSETS.items(), 'PP Set', total=len(PPSETS.keys())):
    base_tag = tag.replace("_pp","")
    if tag == "scale_pp":
        # create per-texture initial CSV
        locs = gpd.read_file(cfg["shp"])
        locs["X"], locs["Y"] = locs.geometry.x, locs.geometry.y
        locs = add_layer_column(locs[["X","Y"]], layers)
        locs = locs.reset_index(drop=True)
        locs["name"] = [f"pp_{base_tag}_L{int(L)}_{i+1}" for i, L in enumerate(locs["Layer"].to_numpy())]
        for tex in cfg["targets"]:
            df = locs[["name","X","Y","Layer"]].copy()
            df["parval1"] = tex_scale_init.get(tex, 0.0)
            df.to_csv(init_dir / f"init_{tag}_{tex}.csv", index=False)
    else:
        locs = gpd.read_file(cfg["shp"])
        locs["X"], locs["Y"] = locs.geometry.x, locs.geometry.y
        if "layer" in locs.columns:
            locs = locs.rename(columns={"layer":"Layer"})
        else:
            locs["Layer"] = 0
        locs = locs.reset_index(drop=True)
        locs["name"] = [f"pp_{base_tag}_L{int(L)}_{i+1}" for i, L in enumerate(locs["Layer"].to_numpy())]
        par = cfg["targets"][0]
        df = locs[["name","X","Y","Layer"]].copy()
        df["parval1"] = 0.0
        df.to_csv(init_dir / f"init_{tag}_{par}.csv", index=False)

# ----------------------------------------------------------------------------------------------------------------------
# Plot PP types by layer
# ----------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
colors = {"scale":"tab:blue","litho":"tab:green","aem":"tab:orange","mult":"tab:red"}

def plot_layer(ax, layer_idx, title):
    mask = (ib[layer_idx, :, :] > 0)
    poly = active_polygon_from_mask(mg, mask)
    if not poly.is_empty:
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="k", linewidth=0.5)

    # Reload per set (so we plot after any filtering/duplication)
    sc = gpd.read_file(PPSETS["scale_pp"]["shp"]); sc["X"], sc["Y"] = sc.geometry.x, sc.geometry.y
    sc = add_layer_column(sc[["X","Y"]], layers)
    sc = sc[sc["Layer"] == layer_idx]
    ax.scatter(sc["X"], sc["Y"], s=10, color=colors["scale"], label="Scale", alpha=0.7)

    li = gpd.read_file(PPSETS["lth_var_pp"]["shp"]); li["X"], li["Y"] = li.geometry.x, li.geometry.y
    if "layer" in li.columns: li = li.rename(columns={"layer":"Layer"})
    else: li["Layer"]=0
    li = li[li["Layer"] == layer_idx]
    ax.scatter(li["X"], li["Y"], s=10, color=colors["litho"], label="Lithology", alpha=0.7)

    ae = gpd.read_file(PPSETS["aem_var_pp"]["shp"]); ae["X"], ae["Y"] = ae.geometry.x, ae.geometry.y
    if "layer" in ae.columns: ae = ae.rename(columns={"layer":"Layer"})
    else: ae["Layer"]=0
    ae = ae[ae["Layer"] == layer_idx]
    ax.scatter(ae["X"], ae["Y"], s=10, color=colors["aem"], label="AEM", alpha=0.7)

    mu = gpd.read_file(PPSETS["kv_mult_pp"]["shp"]); mu["X"], mu["Y"] = mu.geometry.x, mu.geometry.y
    if "layer" in mu.columns: mu = mu.rename(columns={"layer":"Layer"})
    else: mu["Layer"]=0
    mu = mu[mu["Layer"] == layer_idx]
    ax.scatter(mu["X"], mu["Y"], s=10, color=colors["mult"], label="Multiplier", alpha=0.7)

    ax.set_title(title); ax.set_aspect("equal")

plot_layer(axes[0], 0, "Layer 1")
plot_layer(axes[1], 1, "Layer 2")
axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Neighbor-dist summaries
# ----------------------------------------------------------------------------------------------------------------------
for tag, mean_nn in nn_report.items():
    print(f"Mean NN distance [{tag}]: {round(mean_nn)} m")

print("\nDone. Templates written to", pest_dir)
print("Factor files written to", pest_dir)
print("Initial CSVs written to", init_dir)
