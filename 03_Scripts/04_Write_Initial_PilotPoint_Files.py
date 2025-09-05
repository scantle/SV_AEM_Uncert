import matplotlib
from IPython.core.profileapp import list_help

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import flopy as fp
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from sklearn.neighbors import NearestNeighbors

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Directories
out_dir = Path('./04_InputFiles/RES2PAR/')
pest_dir = Path('./04_InputFiles/PEST/')
data_dir = Path ('./01_Data/')
shp_dir = data_dir / 'GIS/'
plt_dir = Path('./05_Plots/')

out_dir.mkdir(parents=True, exist_ok=True)
pest_dir.mkdir(parents=True, exist_ok=True)

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330
layers = 2

# Remove outside L2 threshold
boundary_threshold = 250.0  # meters

#----------------------------------------------------------------------------------------------------------------------#
# Functions/Classes
#----------------------------------------------------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------------------------------------------------#
# Model object setup
#----------------------------------------------------------------------------------------------------------------------#

# Read in MODFLOW model
m = fp.modflow.Modflow.load(f"{model_name}.nam", model_ws=mf_dir, check=False, load_only=['DIS','BAS6'])
mg = m.modelgrid
mg.set_coord_info(xoff=xoff, yoff=yoff)
# Layer 2 active mask
ib = m.bas6.ibound.array
L2_mask = (ib[1, :, :] > 0)   # zero-based index 1 = layer 2
L2_poly = active_polygon_from_mask(mg, L2_mask)

#----------------------------------------------------------------------------------------------------------------------#
# Scale Pilot Points
#----------------------------------------------------------------------------------------------------------------------#

# Read pilot point file
scalepp = gpd.read_file(shp_dir / 'scale_pp.shp')

# Re-read in file
lognorm_values = pd.read_table(out_dir / 'lognorm_dist_clustered.par', sep='\\s+', skiprows=1)

# Assemble into pilot point dataframe

# Extract X, Y
scalepp['X'] = scalepp.geometry.x
scalepp['Y'] = scalepp.geometry.y

for i,row in lognorm_values.iterrows():
    scalepp[row.Texture] = row.Scale

# Add a layer column for each layer
scalepp['Layer'] = 0
scalepp_layers = pd.concat([scalepp.assign(Layer=lyr) for lyr in range(layers)], ignore_index=True)

# Find points outside of Layer 2
scalepp_layers["L2_dist_m"] = scalepp_layers.apply(
    lambda row: 0.0 if (L2_poly.is_valid and Point(row["X"], row["Y"]).within(L2_poly))
                else Point(row["X"], row["Y"]).distance(L2_poly),
    axis=1
)
scalepp_layers["L2_status"] = scalepp_layers["L2_dist_m"].apply(
    lambda d: "inside" if d == 0.0 else ("near" if d <= boundary_threshold else "far")
)

# Drop layer 2 points that are 'far'
before = len(scalepp_layers)
scalepp_layers = scalepp_layers[~((scalepp_layers["Layer"] == 1) & (scalepp_layers["L2_status"] == "far"))].copy()
after = len(scalepp_layers)
print(f"Dropped {before - after} far Layer-2 pilot points")

# Pick columns, write CSV
out_cols = ['X', 'Y', 'Layer'] + list(lognorm_values.Texture)
scalepp_layers[out_cols].to_csv(out_dir / 'ppscale_values.csv', index=False)  # naming conventions all over the place

print(f"Wrote {len(scalepp_layers)} pilot points × {lognorm_values.shape[0]} textures to ppscale_values.csv")

# Write PEST version
scalepp_pest = scalepp_layers.copy()
for tex in list(lognorm_values.Texture):
    scalepp_pest[tex] = scalepp_pest.apply(lambda row: f"$scale_{tex}_L{row['Layer']}_{row.name + 1}$", axis=1)

lithpp_tpl = pest_dir / 'ppscale_values.tpl'
with open(lithpp_tpl, "w") as f:
    f.write("ptf $\n")
scalepp_pest[out_cols].to_csv(lithpp_tpl, mode="a", index=False)
print(f"Wrote {lithpp_tpl}")

#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Lithology Pilot Points
#----------------------------------------------------------------------------------------------------------------------#

lithpp = gpd.read_file(shp_dir / 'litho_pp.shp')

# Extract X, Y
lithpp['X'] = lithpp.geometry.x
lithpp['Y'] = lithpp.geometry.y

# Add column for variance error with initial value
lithpp['lith_var'] = 0.0

# Pick columns, write CSV
out_cols = ['X', 'Y', 'layer', 'lith_var']
lithpp[out_cols].to_csv(out_dir / 'pplith_values.csv', index=False)
print(f"Wrote {len(lithpp)} pilot points to pplith_values.csv")

# Write PEST version
lithpp_pest = lithpp.copy()
lithpp_pest['lith_var'] = lithpp_pest.apply(lambda row: f"$lithvar_L{row['layer']}_{row.name + 1}$", axis=1)

lithpp_tpl = pest_dir / 'pplith_values.tpl'
with open(lithpp_tpl, "w") as f:
    f.write("ptf $\n")
lithpp_pest[out_cols].to_csv(lithpp_tpl, mode="a", index=False)
print(f"Wrote {lithpp_tpl}")

#----------------------------------------------------------------------------------------------------------------------#
# AEM Pilot Points
#----------------------------------------------------------------------------------------------------------------------#

aempp = gpd.read_file(shp_dir / 'aem_pp.shp')

# Extract X, Y
aempp['X'] = aempp.geometry.x
aempp['Y'] = aempp.geometry.y

# Add column for variance error with initial value
aempp['aem_var'] = 0.0

# Pick columns, write CSV
out_cols = ['X', 'Y', 'layer', 'aem_var']
aempp[out_cols].to_csv(out_dir / 'ppaem_values.csv', index=False)
print(f"Wrote {len(aempp)} pilot points to ppaem_values.csv")

# Write PEST version
aempp_pest = aempp.copy()
aempp_pest['aem_var'] = aempp_pest.apply(lambda row: f"$aemvar_L{row['layer']}_{row.name + 1}$", axis=1)

aempp_tpl = pest_dir / 'ppaem_values.tpl'
with open(aempp_tpl, "w") as f:
    f.write("ptf $\n")
aempp_pest[out_cols].to_csv(aempp_tpl, mode="a", index=False)
print(f"Wrote {aempp_tpl}")

#----------------------------------------------------------------------------------------------------------------------#
# Resistivity Variance Multiplier Pilot Points
#----------------------------------------------------------------------------------------------------------------------#

multpp = gpd.read_file(shp_dir / 'mult_pp.shp')

# Extract X, Y
multpp['X'] = multpp.geometry.x
multpp['Y'] = multpp.geometry.y

# Add column for variance error with initial value
multpp['mult'] = 0.0

# Pick columns, write CSV
out_cols = ['X', 'Y', 'layer', 'mult']
multpp[out_cols].to_csv(out_dir / 'ppmult_values.csv', index=False)
print(f"Wrote {len(multpp)} pilot points to ppmult_values.csv")

# Write PEST version
multpp_pest = multpp.copy()
multpp_pest['mult'] = multpp_pest.apply(lambda row: f"$multvar_L{row['layer']}_{row.name + 1}$", axis=1)

multpp_tpl = pest_dir / 'ppmult_values.tpl'
with open(multpp_tpl, "w") as f:
    f.write("ptf $\n")
multpp_pest[out_cols].to_csv(multpp_tpl, mode="a", index=False)
print(f"Wrote {multpp_tpl}")

#----------------------------------------------------------------------------------------------------------------------#
# Pilot point distance average, to inform kriging
#----------------------------------------------------------------------------------------------------------------------#

coords = scalepp[['X', 'Y']].to_numpy()
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
nn = nbrs.kneighbors(coords)[0][:,1]
mean_nn = nn.mean()

print(f'Mean Scale PP neighbor distance:', round(mean_nn))

coords = lithpp[['X', 'Y']].to_numpy()
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
nn = nbrs.kneighbors(coords)[0][:,1]
mean_nn = nn.mean()

print(f'Mean Lith PP neighbor distance:', round(mean_nn))

coords = aempp[['X', 'Y']].to_numpy()
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
nn = nbrs.kneighbors(coords)[0][:,1]
mean_nn = nn.mean()

print(f'Mean AEM PP neighbor distance:', round(mean_nn))

coords = multpp[['X', 'Y']].to_numpy()
nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
nn = nbrs.kneighbors(coords)[0][:,1]
mean_nn = nn.mean()

print(f'Mean Mult PP neighbor distance:', round(mean_nn))

#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Plot pilot points by layer
#----------------------------------------------------------------------------------------------------------------------#

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

colors = {
    "scale": "tab:blue",
    "litho": "tab:green",
    "aem": "tab:orange",
    "mult": "tab:red"
}

def plot_layer(ax, layer_idx, title):
    # Active domain outline for this layer
    mask = (ib[layer_idx, :, :] > 0)
    poly = active_polygon_from_mask(mg, mask)
    if not poly.is_empty:
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="k", linewidth=0.5)

    # Pilot points
    # Scale (from scalepp_layers DataFrame)
    sc = scalepp_layers[scalepp_layers["Layer"] == layer_idx]
    ax.scatter(sc["X"], sc["Y"], s=10, color=colors["scale"], label="Scale", alpha=0.7)

    # Lithology
    li = lithpp[lithpp["layer"] == layer_idx]
    ax.scatter(li["X"], li["Y"], s=10, color=colors["litho"], label="Lithology", alpha=0.7)

    # AEM
    ae = aempp[aempp["layer"] == layer_idx]
    ax.scatter(ae["X"], ae["Y"], s=10, color=colors["aem"], label="AEM", alpha=0.7)

    # Multiplier
    mu = multpp[multpp["layer"] == layer_idx]
    ax.scatter(mu["X"], mu["Y"], s=10, color=colors["mult"], label="Multiplier", alpha=0.7)

    ax.set_title(title)
    ax.set_aspect("equal")

plot_layer(axes[0], 0, "Layer 1")
plot_layer(axes[1], 1, "Layer 2")

axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

