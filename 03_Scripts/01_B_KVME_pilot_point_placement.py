import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.affinity import translate
import random
from pathlib import Path
import flopy

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Directories
data_dir = Path('./04_InputFiles/RES2PAR')
shp_dir = Path('01_Data/GIS/')

out_dir=Path("./06_Outputs/")  # for csvs, shapefiles go in shp_dir

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330

borehole_prefix = "litho_pp"
aem_prefix = "aem_pp"

# spacing (meters)
aem_buffer_radius = 150
aem_target_spacing = 300
aem_min_separation = 300
lith_min_separation = 300
boundary_threshold = 300

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def gdf_from_points_df(df, xcol="x", ycol="y", crs=None):
    g = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[xcol], df[ycol]), crs=crs)
    return g

#----------------------------------------------------------------------------------------------------------------------#

def build_influence_mask(pts_gdf, radius_m):
    """Dissolve buffers around points into a single polygon mask."""
    if len(pts_gdf) == 0:
        return None
    buf = pts_gdf.buffer(radius_m)
    return unary_union(buf)

#----------------------------------------------------------------------------------------------------------------------#

def active_polygon_from_mask(mg, mask2d):
    """Union of square cells approximating the active area."""
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

def lattice_points_in_polygon(poly, spacing, jitter_frac=0.3):
    """
    Fast 'blue-noise-ish' seeding: start with a square lattice at 'spacing',
    jitter each node by up to ±jitter_frac*spacing in x/y, keep if inside poly.
    """
    if poly is None or poly.is_empty:
        return []
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    j = spacing * jitter_frac
    y = miny
    row = 0
    while y <= maxy:
        x = minx
        # simple row offset to reduce alignment artefacts
        x_offset = (row % 2) * (0.5 * spacing)
        while x <= maxx:
            rx = x + x_offset + random.uniform(-j, j)
            ry = y + random.uniform(-j, j)
            p = Point(rx, ry)
            if poly.contains(p):
                pts.append(p)
            x += spacing
        y += spacing
        row += 1
    return pts

#----------------------------------------------------------------------------------------------------------------------#

def greedy_thin(points, dmin):
    """Greedy minimum-spacing filter (keep earlier points first)."""
    kept = []
    coords = []
    d2 = dmin * dmin
    for p in points:
        x, y = p.x, p.y
        too_close = False
        for (xx, yy) in coords:
            if (x-xx)*(x-xx) + (y-yy)*(y-yy) < d2:
                too_close = True
                break
        if not too_close:
            kept.append(p)
            coords.append((x, y))
    return kept

#----------------------------------------------------------------------------------------------------------------------#

def duplicate_per_layer(pp_gdf, m, near_thresh=200.0):
    """Duplicate points to layers, compute L2 distance flags, drop far in L2."""
    ib = m.bas6.ibound.array
    L2_mask = (ib[1, :, :] > 0)  # layer index 1 == second layer
    L2_poly = active_polygon_from_mask(m.modelgrid, L2_mask)

    pp = pp_gdf.copy().reset_index(drop=True)
    pp["pp_id"] = (np.arange(len(pp)) + 1).astype(int)
    pp["name"] = pp["pp_id"].map(lambda i: f"pp_{i:05d}")

    # distance to L2 active
    pp["L2_dist_m"] = pp.geometry.apply(lambda g: 0.0 if (L2_poly.is_valid and g.within(L2_poly)) else g.distance(L2_poly))
    pp["L2_status"] = pp["L2_dist_m"].apply(lambda d: "inside" if d == 0.0 else ("near" if d <= near_thresh else "far"))
    pp["L2_within_n"] = pp["L2_status"].isin(["inside", "near"])

    L1 = pp.copy()
    L1["layer"] = 1
    L1["name"] = L1["name"] + "_L1"

    L2 = pp.copy()
    L2["layer"] = 2
    L2["name"] = L2["name"] + "_L2"
    L2 = L2.query("L2_status != 'far'").copy()

    return pd.concat([L1, L2], ignore_index=True)

#----------------------------------------------------------------------------------------------------------------------#

def dedupe_sites(df, id_col=None, xcol="x", ycol="y", layer_col="layer"):
    """
    Collapse raw log rows to unique site-layer points.
    Priority: use an ID column if present; otherwise use rounded coords.
    Returns a DataFrame with columns [site_id, layer, x, y].
    """
    if id_col and id_col in df.columns:
        # mean x/y in case of tiny jitter; keeps 1 row per (site, layer)
        out = (df.groupby([id_col, layer_col], as_index=False)
                 .agg({xcol: "mean", ycol: "mean"}))
        out = out.rename(columns={id_col: "site_id", xcol: "x", ycol: "y", layer_col: "layer"})
    else:
        # fallback: spatial de-dupe (round to 1 m)
        df = df.copy()
        df["_xr"] = df[xcol].round(0)
        df["_yr"] = df[ycol].round(0)
        out = (df.groupby(["_xr", "_yr", layer_col], as_index=False)
                 .agg({xcol: "mean", ycol: "mean"}))
        out["site_id"] = np.arange(1, len(out) + 1)
        out = out[["site_id", layer_col, "x", "y"]].rename(columns={layer_col: "layer"})
    return out

#----------------------------------------------------------------------------------------------------------------------#

def keep_inside_active(pp_gdf, active_poly):
    """Drop points entirely outside the active model polygon."""
    if active_poly is None or active_poly.is_empty:
        return pp_gdf
    return pp_gdf[pp_gdf.geometry.within(active_poly)].copy()

#----------------------------------------------------------------------------------------------------------------------#

def count_logs_per_site_layer(df, id_col="WELL_INFO_ID", layer_col="layer"):
    """Return a DF with columns [site_id, layer, n_logs]."""
    c = (df.groupby([id_col, layer_col])
           .size()
           .reset_index(name="n_logs")
           .rename(columns={id_col: "site_id"}))
    return c

#----------------------------------------------------------------------------------------------------------------------#

def thin_per_layer_priority(pp_gdf, dmin, priority_col="n_logs"):
    """
    Greedy min-distance thinning per layer.
    Keeps higher 'priority_col' first (e.g., wells with more logs),
    then enforces dmin spacing within each layer.
    """
    kept_layers = []
    d2 = dmin * dmin
    for lyr, sub in pp_gdf.groupby("layer"):
        sub = sub.sort_values(priority_col, ascending=False).copy()
        kept_idx, coords = [], []
        for idx, row in sub.iterrows():
            x, y = row.geometry.x, row.geometry.y
            if all((x-xx)*(x-xx) + (y-yy)*(y-yy) >= d2 for (xx, yy) in coords):
                kept_idx.append(idx)
                coords.append((x, y))
        kept_layers.append(sub.loc[kept_idx])
    return pd.concat(kept_layers).sort_values(["layer", priority_col], ascending=[True, False])

#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

# Read in model
m = flopy.modflow.Modflow.load(f"{model_name}.nam", model_ws=mf_dir, check=False, forgive=True, load_only=['DIS','BAS6'])

# Read in data
borehole_data = pd.read_csv(data_dir / 'lithologs.csv')
aem_data = pd.read_csv(data_dir / 'aemlogs.csv')

# Store model info
mg = m.modelgrid
mg.set_coord_info(xoff=xoff, yoff=yoff)

# Model, grid, and active mask
mg = m.modelgrid
mg.set_coord_info(xoff=xoff, yoff=yoff)
active_mask, _, _ = (lambda mm: (
    (np.any(mm.bas6.ibound.array > 0, axis=0), mg.xcellcenters, mg.ycellcenters)
))(m)
active_poly = active_polygon_from_mask(mg, active_mask)

# AEM pilot points
aem_data = (aem_data.sort_values(["LINE_NO","FID"]).
            drop_duplicates(["LINE_NO","FID"])[["x","y","layer"]].
            rename(columns={"layer":"layer"}) )

aem_gdf = gdf_from_points_df(aem_data, xcol="x", ycol="y", crs=mg.proj4 if getattr(mg, "proj4", None) else None)
aem_mask = build_influence_mask(aem_gdf, radius_m=aem_buffer_radius)
aem_mask = aem_mask.intersection(active_poly) if aem_mask is not None else None

aem_candidates = lattice_points_in_polygon(aem_mask, spacing=aem_target_spacing, jitter_frac=0.3)
aem_kept = greedy_thin(aem_candidates, dmin=aem_min_separation)

aem_pp = gpd.GeoDataFrame(
    {"x": [p.x for p in aem_kept], "y": [p.y for p in aem_kept], "source": "AEM"},
    geometry=aem_kept,
    crs=aem_gdf.crs
)

aem_by_layer = duplicate_per_layer(aem_pp, m, near_thresh=boundary_threshold)

# Save AEM
(shp_dir / f"{aem_prefix}.shp").parent.mkdir(parents=True, exist_ok=True)
# aem_pp.to_file(shp_dir / f"{aem_prefix}.shp")
# aem_pp.drop(columns="geometry").to_csv(out_dir / f"{aem_prefix}.csv", index=False)
aem_by_layer.to_file(shp_dir / f"{aem_prefix}.shp")
aem_by_layer.drop(columns="geometry").to_csv(out_dir / f"{aem_prefix}.csv", index=False)

print(f"AEM pilots: {len(aem_pp)} base / {len(aem_by_layer)} with layers")

# ----- Borehole pilots: 1 PP per well per layer -----

# Collapse raw logs to unique (site, layer) points
bh_sites = dedupe_sites(borehole_data, id_col="WELL_INFO_ID", xcol="x", ycol="y", layer_col="layer")

# Count how many raw rows each site/layer has (priority)
bh_counts = count_logs_per_site_layer(borehole_data, id_col="WELL_INFO_ID", layer_col="layer")
bh_sites = bh_sites.merge(bh_counts, on=["site_id", "layer"], how="left").fillna({"n_logs": 1})

# GeoDF & clip to active polygon
bh_pp = gpd.GeoDataFrame(
    {"site_id": bh_sites["site_id"], "layer": bh_sites["layer"], "n_logs": bh_sites["n_logs"], "source": "BOREHOLE"},
    geometry=gpd.points_from_xy(bh_sites["x"], bh_sites["y"]),
    crs=mg.proj4 if getattr(mg, "proj4", None) else None
)
active_mask = np.any(m.bas6.ibound.array > 0, axis=0)
active_poly  = active_polygon_from_mask(mg, active_mask)
#bh_pp = keep_inside_active(bh_pp, active_poly)

# Enforce minimum spacing per layer
bh_pp = thin_per_layer_priority(bh_pp, dmin=lith_min_separation, priority_col="n_logs")

# Stable names/ids
bh_pp = bh_pp.reset_index(drop=True)
bh_pp["pp_id"] = (np.arange(len(bh_pp)) + 1).astype(int)
bh_pp["name"]  = bh_pp["pp_id"].map(lambda i: f"{borehole_prefix}_{i:05d}")

# L2 distance flags & drop 'far' only for layer 2
ib = m.bas6.ibound.array
L2_poly = active_polygon_from_mask(mg, (ib[1, :, :] > 0))
bh_pp["L2_dist_m"] = bh_pp.geometry.apply(lambda g: 0.0 if (L2_poly.is_valid and g.within(L2_poly)) else g.distance(L2_poly))
bh_pp["L2_status"] = bh_pp["L2_dist_m"].apply(lambda d: "inside" if d == 0.0 else ("near" if d <= boundary_threshold else "far"))
bh_pp = bh_pp[~((bh_pp["layer"] == 1) & (bh_pp["L2_status"] == "far"))].copy()

# Save
bh_pp.to_file(shp_dir / f"{borehole_prefix}.shp")
bh_pp[["name","site_id","layer","n_logs","L2_dist_m","L2_status"]].to_csv(out_dir / f"{borehole_prefix}.csv", index=False)

print(f"Borehole pilots (after {lith_min_separation:.0f} m min-sep): {len(bh_pp)}")
