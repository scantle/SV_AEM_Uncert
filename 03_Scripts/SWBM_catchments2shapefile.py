import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import geopandas as gpd
import warnings
from pathlib import Path
from shapely import wkb

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

# Directories
swbm_dir = Path('02_Models/SVIHM_MF_working/SWBM/')
data_dir = Path('01_Data')
shp_dir = data_dir / 'GIS'

# Shapefiles
sv_model_shp_file = shp_dir / 'grid_properties_rep.shp'

out_file = shp_dir / 'SWBM_catchments.shp'

# -------------------------------------------------------------------------------------------------------------------- #
# Classes/Functions
# -------------------------------------------------------------------------------------------------------------------- #

def force_2d(geom):
    """Strip Z from any geometry (Polygon/MultiPolygon/LineString/etc.)."""
    try:
        return wkb.loads(wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom

# -------------------------------------------------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------------------------------------------------- #

grid = gpd.read_file(sv_model_shp_file)
# ensure 2D
grid["geometry"] = grid["geometry"].apply(force_2d)

# restrict to layer 1 (keep only what you need to save memory)
grid_lay1 = grid.loc[grid["Layer"] == 1, ["Row", "Column", "geometry"]].copy()

# SWBM maps
catch_cell = pd.read_csv(
    swbm_dir / "modflow_cell_to_catchment.txt",
    sep=r"\s+",
    skiprows=1,
    names=["Row", "Column", "Catchment"],
    dtype={"Row": int, "Column": int, "Catchment": int},
)

catch_mult = pd.read_table(
    data_dir / "manualTPLs/catchment_mult.txt.tpl",
    sep=r"\s+",
    skiprows=2,
    names=["Catchment", "delim1", "mult", "delim2"],
    dtype={"Catchment": int, "mult": str},
)

# ---------- join grid cells to catchments ----------
g = grid_lay1.merge(catch_cell, on=["Row", "Column"], how="inner")

# sanity checks
missing = catch_cell.merge(grid_lay1[["Row","Column"]], on=["Row","Column"], how="left", indicator=True)
n_missing = (missing["_merge"] == "left_only").sum()
if n_missing > 0:
    warnings.warn(f"{n_missing} catch_cell rows had no matching grid cell (Row/Column mismatch).")

# ---------- bring in multiplier names ----------
g = g.merge(catch_mult[["Catchment", "mult"]], on="Catchment", how="left")
n_unmatched_mult = g["mult"].isna().sum()
if n_unmatched_mult > 0:
    warnings.warn(f"{n_unmatched_mult} cells had no multiplier match in catch_mult.")

# ---------- dissolve to one feature per catchment (MultiPolygon if disjoint) ----------
# (Dissolve also handles non-contiguous pieces by returning a MultiPolygon.)
# We’ll also add a simple attribute: number of cells per dissolved feature.
g["_cell"] = 1
counts = g.groupby(["Catchment", "mult"])["_cell"].sum().rename("n_cells").reset_index()

diss = g.dissolve(by=["Catchment", "mult"], as_index=False)

# attach counts (optional but handy)
diss = diss.merge(counts, on=["Catchment", "mult"], how="left")

# keep CRS
diss = gpd.GeoDataFrame(diss, geometry="geometry", crs=grid.crs)

# shapefile field name hygiene (<=10 chars); yours are short already
# but just in case, we can rename to conservative names
diss = diss.rename(columns={"Catchment": "catchment"})

# ensure 2D again after dissolve (some stacks re-inject Zs)
diss["geometry"] = diss["geometry"].apply(force_2d)

# ---------- write output ----------
out_file = shp_dir / "SWBM_catchments.shp"   # write to outputs folder
diss.to_file(out_file, driver="ESRI Shapefile")

print(f"Wrote {len(diss)} dissolved catchment feature(s) to {out_file}")