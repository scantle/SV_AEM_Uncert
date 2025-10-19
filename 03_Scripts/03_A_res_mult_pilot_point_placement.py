import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pathlib import Path
import flopy

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Directories
data_dir = Path('./01_Data/')
shp_dir = data_dir / 'GIS/'

out_dir = data_dir / 'PPlocs'
out_dir.mkdir(exist_ok=True)

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330

out_prefix="mult_pp"

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def get_active_mask_from_flopy(m):
    """
    Returns:
      - active_mask: (nrow, ncol) boolean
      - xcc, ycc: (nrow, ncol) arrays of cell centers in projected coords
    """
    mg = m.modelgrid  # works for DIS and DISV; below assumes structured
    xcc, ycc = mg.xcellcenters, mg.ycellcenters  # (nrow, ncol)

    # Try BAS6 (MODFLOW-2005/NWT) then IDOMAIN if available
    active_mask = None
    if hasattr(m, 'bas6') and hasattr(m.bas6, 'ibound'):
        ib = m.bas6.ibound.array  # (nlay, nrow, ncol)
        active_mask = np.any(ib > 0, axis=0)
    elif hasattr(mg, 'idomain') and mg.idomain is not None:
        # mg.idomain can be 2D or 3D
        idm = np.array(mg.idomain)
        if idm.ndim == 3:
            active_mask = np.any(idm > 0, axis=0)
        else:
            active_mask = idm > 0
    else:
        raise RuntimeError("No IBOUND/IDOMAIN found; provide a polygon mask path instead.")

    return active_mask, xcc, ycc

#----------------------------------------------------------------------------------------------------------------------#

def get_active_mask_from_polygon_shp(shp_path, xcc, ycc):
    """
    Burn a polygon mask shapefile onto the model centers.
    Returns boolean (nrow, ncol) mask.
    """
    gdf = gpd.read_file(shp_path).to_crs(epsg=None)  # keep CRS as-is
    poly = unary_union(gdf.geometry)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xcc.ravel(), ycc.ravel()), crs=gdf.crs)
    inside = pts.within(poly)
    return inside.values.reshape(xcc.shape)

#----------------------------------------------------------------------------------------------------------------------#

def group_blocks(nrow, ncol, block=6):
    """Yield (r0, r1, c0, c1) index slices for 6x6 blocks over the grid."""
    for r0 in range(0, nrow, block):
        for c0 in range(0, ncol, block):
            r1 = min(r0 + block, nrow)
            c1 = min(c0 + block, ncol)
            yield r0, r1, c0, c1

#----------------------------------------------------------------------------------------------------------------------#

def block_center_xy(xcc, ycc, r0, r1, c0, c1):
    """Center of the block in map coords from cell centers."""
    x_block = xcc[r0:r1, c0:c1]
    y_block = ycc[r0:r1, c0:c1]
    return float(np.mean(x_block)), float(np.mean(y_block))

#----------------------------------------------------------------------------------------------------------------------#

def block_polygon_from_edges(mg, r0, r1, c0, c1):
    """
    Build a polygon for a block using modelgrid edges (structured only).
    If you only have centers, a bbox from min/max centers is fine as well.
    """
    # With centers, a robust approximation:
    x_min = mg.xcellcenters[r0:r1, c0:c1].min()
    x_max = mg.xcellcenters[r0:r1, c0:c1].max()
    y_min = mg.ycellcenters[r0:r1, c0:c1].min()
    y_max = mg.ycellcenters[r0:r1, c0:c1].max()
    # Expand by half-cell approx:
    dx = np.median(np.diff(np.unique(mg.xcellcenters[0, :]))) if mg.ncol > 1 else 0.0
    dy = np.median(np.diff(np.unique(mg.ycellcenters[:, 0]))) if mg.nrow > 1 else 0.0
    poly = Polygon([(x_min - dx/2, y_min - dy/2),
                    (x_max + dx/2, y_min - dy/2),
                    (x_max + dx/2, y_max + dy/2),
                    (x_min - dx/2, y_max + dy/2)])
    return poly

#----------------------------------------------------------------------------------------------------------------------#

def make_refine_buffer(data_points_gdf=None, streams_gdf=None,
                       point_buffer=500.0, stream_buffer=300.0):
    """
    Build a unioned buffer polygon for refinement.
      - point_buffer: radius for wells/AEM support (meters)
      - stream_buffer: half-width for river corridors (meters)
    """
    polys = []
    if data_points_gdf is not None and len(data_points_gdf) > 0:
        polys.append(data_points_gdf.buffer(point_buffer))
    if streams_gdf is not None and len(streams_gdf) > 0:
        polys.append(streams_gdf.buffer(stream_buffer))
    if not polys:
        return None
    return unary_union(gpd.GeoSeries(pd.concat(polys, ignore_index=True)))

#----------------------------------------------------------------------------------------------------------------------#

def enforce_minimum_spacing(points_gdf, dmin):
    """
    Greedy thinning to enforce minimum spacing (meters).
    Keeps refined blocks preferentially by sorting block size ascending (3 first, then 6).
    """
    gdf = points_gdf.copy()
    gdf = gdf.sort_values(by=["block_cells"])  # 3x3 first, then 6x6
    kept = []
    tree_coords = []

    for idx, row in gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        if not tree_coords:
            kept.append(idx)
            tree_coords.append((x, y))
            continue
        # quick check
        too_close = False
        for (xx, yy) in tree_coords:
            if (x - xx) ** 2 + (y - yy) ** 2 < dmin ** 2:
                too_close = True
                break
        if not too_close:
            kept.append(idx)
            tree_coords.append((x, y))

    return gdf.loc[kept].copy()

#----------------------------------------------------------------------------------------------------------------------#

def active_polygon_from_mask(mg, layer_mask):
    """
    Build a polygon for the active domain of a single layer mask (2D bool array).
    Approximates each active cell as a square of size dx×dy centered at the cell center,
    then dissolves (unary_union) all squares.
    """
    xcc, ycc = mg.xcellcenters, mg.ycellcenters
    # cell size estimates
    dx = np.median(np.diff(np.unique(xcc[0, :]))) if mg.ncol > 1 else 0.0
    dy = np.median(np.diff(np.unique(ycc[:, 0]))) if mg.nrow > 1 else 0.0
    halfx, halfy = dx / 2.0, dy / 2.0

    # fast square-by-buffer (cap_style=3 => square)
    squares = []
    rr, cc = np.where(layer_mask)
    for r, c in zip(rr, cc):
        squares.append(Point(xcc[r, c], ycc[r, c]).buffer(min(halfx, halfy), cap_style=3))
    if not squares:
        # empty mask -> empty polygon (use an empty geometry)
        return Polygon()
    return unary_union(squares)

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

m = flopy.modflow.Modflow.load(f"{model_name}.nam", model_ws=mf_dir, check=False, forgive=True, load_only=['DIS','BAS6'])

wells_shpfile = shp_dir / "hobwells.shp"

# Analysis Settings
streams_path=None       # e.g., river polyline shapefile
block_size=6            # 6x6 coarse blocks
refine_subdivisions=2   # split 6x6 into 2×2 => each sub-block is 3x3
point_buffer=100.0      # meters around wells/AEM
stream_buffer=100.0     # meters along streams
min_separation=200.0    # enforce minimum spacing (meters)
l2_near_thresh = 200.0  # Outside layer 2 cutoff

# Read in model info
mg = m.modelgrid
mg.set_coord_info(xoff=xoff, yoff=yoff)
xcc, ycc = mg.xcellcenters, mg.ycellcenters
nrow, ncol = xcc.shape

# Get active cells
active_mask, xcc, ycc = get_active_mask_from_flopy(m)

# Optional refinement buffers
refine_poly = None

dp_gdf = gpd.read_file(wells_shpfile) if wells_shpfile else None
#st_gdf = gpd.read_file(streams_path) if streams_path else None
refine_poly = make_refine_buffer(dp_gdf, point_buffer=point_buffer)

pilots = []

for r0, r1, c0, c1 in group_blocks(nrow, ncol, block=block_size):
    block_mask = active_mask[r0:r1, c0:c1]
    if not np.any(block_mask):
        continue  # skip blocks with no active cells

    # Coarse 6x6 center
    bx, by = block_center_xy(xcc, ycc, r0, r1, c0, c1)
    block_poly = block_polygon_from_edges(mg, r0, r1, c0, c1)

    # Decide whether to refine this block
    refine_this = False
    if refine_poly is not None:
        # Intersects any refinement corridor?
        refine_this = block_poly.intersects(refine_poly)

    if not refine_this:
        pilots.append((bx, by, r0, r1, c0, c1, 6))
        continue

    # Refined: split 6x6 into 2x2 sub-blocks (3x3 each)
    sub = refine_subdivisions
    r_splits = np.linspace(r0, r1, sub + 1, dtype=int)
    c_splits = np.linspace(c0, c1, sub + 1, dtype=int)
    for i in range(sub):
        for j in range(sub):
            rr0, rr1 = r_splits[i], r_splits[i + 1]
            cc0, cc1 = c_splits[j], c_splits[j + 1]
            sub_mask = active_mask[rr0:rr1, cc0:cc1]
            if not np.any(sub_mask):
                continue
            sx, sy = block_center_xy(xcc, ycc, rr0, rr1, cc0, cc1)
            pilots.append((sx, sy, rr0, rr1, cc0, cc1, 3))

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    {
        "x": [p[0] for p in pilots],
        "y": [p[1] for p in pilots],
        "r0": [p[2] for p in pilots],
        "r1": [p[3] for p in pilots],
        "c0": [p[4] for p in pilots],
        "c1": [p[5] for p in pilots],
        "block_cells": [p[6] for p in pilots],  # 6 or 3 (per side)
    },
    geometry=[Point(p[0], p[1]) for p in pilots],
    crs=mg.proj4 if getattr(mg, "proj4", None) else None
)

# Enforce minimum spacing
if min_separation and len(gdf) > 1:
    gdf = enforce_minimum_spacing(gdf, min_separation)

# Layer-2 active mask (uses BAS6 IBOUND already loaded)
ib = m.bas6.ibound.array  # shape (nlay, nrow, ncol)
L2_mask = (ib[1, :, :] > 0)   # zero-based index: layer 2 => [1]

# Build L2 active polygon for distance tests
L2_poly = active_polygon_from_mask(mg, L2_mask)

# Distance to L2 active (meters in your projected CRS)
near_thresh = 200.0  # n (m): tweak 150–300 m
gdf["L2_dist_m"] = gdf.geometry.apply(lambda g: 0.0 if (L2_poly.is_valid and g.within(L2_poly)) else g.distance(L2_poly))

def _l2_status(d):
    if d == 0.0:
        return "inside"
    return "near" if d <= l2_near_thresh else "far"

gdf["L2_status"]   = gdf["L2_dist_m"].apply(_l2_status)
gdf["L2_within_n"] = gdf["L2_status"].isin(["inside", "near"])

# Create stable IDs/names, then duplicate per layer
gdf = gdf.reset_index(drop=True)
gdf["pp_id"] = (np.arange(len(gdf)) + 1).astype(int)
gdf["name"]  = gdf["pp_id"].map(lambda i: f"pp_{i:05d}")

gdf_L1 = gdf.copy()
gdf_L1["layer"] = 0
gdf_L1["name"]  = gdf_L1["name"] + "_L1"

gdf_L2 = gdf.copy()
gdf_L2["layer"] = 1
gdf_L2["name"]  = gdf_L2["name"] + "_L2"

# Option A (recommended): drop "far" points from L2 entirely
gdf_L2 = gdf_L2.query("L2_status != 'far'").copy()

# Merge and save layer-specific outputs
pp_by_layer = pd.concat([gdf_L1, gdf_L2], ignore_index=True)
pp_by_layer.to_file(shp_dir / f"{out_prefix}.shp")
pp_by_layer.drop(columns="geometry").to_csv(out_dir / f"{out_prefix}.csv", index=False)

print(f"Saved: {(shp_dir / f'{out_prefix}.shp').name}")
print(f"Saved: {(shp_dir / f'{out_prefix}.csv').name}")