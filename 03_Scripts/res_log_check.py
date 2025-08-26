import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Directories
out_dir = Path('./06_Outputs/')
data_dir = Path ('./01_Data/')
shp_dir = data_dir / 'GIS/'
plt_dir = Path('./05_Plots/')

# Shapefiles
sv_model_domain_file = shp_dir / 'Model_Domain_20180222.shp'

#----------------------------------------------------------------------------------------------------------------------#
# Read/Setup
#----------------------------------------------------------------------------------------------------------------------#

all_df = pd.read_csv(out_dir / 'res_log_all.csv')
lim_df = pd.read_csv(out_dir / 'res_log.csv')

domain = gpd.read_file(sv_model_domain_file)
# If the shapefile is multipart, dissolve to a single polygon for clean overlay/join
domain_poly = domain.dissolve().geometry.iloc[0]
domain_gdf  = gpd.GeoDataFrame(geometry=[domain_poly], crs=domain.crs)

# Build GeoDataFrames; assume same CRS as domain
all_gdf = gpd.GeoDataFrame(all_df.copy(),
                           geometry=gpd.points_from_xy(all_df['X'], all_df['Y']),
                           crs=domain_gdf.crs)
lim_gdf = gpd.GeoDataFrame(lim_df.copy(),
                           geometry=gpd.points_from_xy(lim_df['X'], lim_df['Y']),
                           crs=domain_gdf.crs)

# ----------------------------------------------------------------------------------------------------------------------
# Compare coordinate pairs (exact equality on x,y)
# ----------------------------------------------------------------------------------------------------------------------
all_pairs = set(map(tuple, all_df[['X','Y']].to_numpy()))
lim_pairs = set(map(tuple, lim_df[['X','Y']].to_numpy()))

only_in_all = all_pairs - lim_pairs
only_in_lim = lim_pairs - all_pairs
shared      = all_pairs & lim_pairs

print(f"Total ALL points: {len(all_df)}")
print(f"Total LIM points: {len(lim_df)}")
print(f"Shared coordinate pairs: {len(shared)}")
print(f"Only-in-ALL pairs: {len(only_in_all)}")
print(f"Only-in-LIM pairs: {len(only_in_lim)}")

# Create GDFs for the “only in …” sets (useful to visualize)
if only_in_all:
    only_all_df = pd.DataFrame(list(only_in_all), columns=['X','Y'])
    only_all_gdf = gpd.GeoDataFrame(
        only_all_df, geometry=gpd.points_from_xy(only_all_df.X, only_all_df.Y), crs=domain_gdf.crs
    )
else:
    only_all_gdf = gpd.GeoDataFrame(geometry=[], crs=domain_gdf.crs)

if only_in_lim:
    only_lim_df = pd.DataFrame(list(only_in_lim), columns=['X','Y'])
    only_lim_gdf = gpd.GeoDataFrame(
        only_lim_df, geometry=gpd.points_from_xy(only_lim_df.X, only_lim_df.Y), crs=domain_gdf.crs
    )
else:
    only_lim_gdf = gpd.GeoDataFrame(geometry=[], crs=domain_gdf.crs)

# ----------------------------------------------------------------------------------------------------------------------
# Flag points outside the model domain
# ----------------------------------------------------------------------------------------------------------------------
all_inside_mask = all_gdf.within(domain_poly)
lim_inside_mask = lim_gdf.within(domain_poly)

all_outside = all_gdf.loc[~all_inside_mask]
lim_outside = lim_gdf.loc[~lim_inside_mask]

print(f"ALL points outside domain: {len(all_outside)}")
print(f"LIM points outside domain: {len(lim_outside)}")

# ----------------------------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Domain background (light gray fill, darker edge)
domain_gdf.plot(ax=ax, facecolor='#DDDDDD', edgecolor='#888888', linewidth=0.8)

# Shared points (appear in both): plot beneath to avoid clutter if large
if len(shared) > 0:
    shared_df = pd.DataFrame(list(shared), columns=['x','y'])
    shared_gdf = gpd.GeoDataFrame(shared_df, geometry=gpd.points_from_xy(shared_df.x, shared_df.y), crs=domain_gdf.crs)
    shared_gdf.plot(ax=ax, markersize=10, marker='o', alpha=0.5, label='Shared (ALL ∩ LIM)')

# Exclusives
if len(only_all_gdf) > 0:
    only_all_gdf.plot(ax=ax, markersize=15, marker='x', linewidth=1.2, alpha=0.9, label='Only in ALL')
if len(only_lim_gdf) > 0:
    only_lim_gdf.plot(ax=ax, markersize=15, marker='+', linewidth=1.2, alpha=0.9, label='Only in LIM')

ax.set_title('ALL vs LIM with Model Domain')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.legend(loc='best', frameon=True)

out_png = plt_dir / 'all_vs_lim_with_domain.png'
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print(f"Saved plot to: {out_png}")