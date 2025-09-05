import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import gstools as gs
from gstools.covmodel import SumModel
from pathlib import Path
from math import radians, pi

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Input Files
in_dir = Path('./04_InputFiles/RES2PAR/')
plt_dir = Path('./05_Plots')

# MODFLOW Model
mf_dir = Path('./02_Models/SVIHM_MF/')
model_name = 'svihm'
xoff = 499977
yoff = 4571330

#----------------------------------------------------------------------------------------------------------------------#
# Functions/Classes
#----------------------------------------------------------------------------------------------------------------------#

def azimuth_to_vector(az_deg):
    """Return a unit vector pointing at az_deg clockwise from North."""
    theta = np.deg2rad(az_deg)
    return np.array([np.sin(theta),  # x  (East)
                     np.cos(theta),  # y  (North)
                     0.0])           # z  (horizontal)

#----------------------------------------------------------------------------------------------------------------------#

def rose_of_pair_directions(xs, ys, n_pairs=200000, bin_deg=10, seed=42):
    """
    Quick rose plot of separation-vector azimuths for a random subset of point pairs.

    Parameters
    ----------
    xs, ys : 1-D arrays
        Horizontal coordinates of your data points (same length).
    n_pairs : int
        How many random point pairs to sample.
    bin_deg : int
        Width of histogram bins in degrees.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # 1) choose random indices for the two ends of each pair
    i = rng.integers(0, len(xs), n_pairs)
    j = rng.integers(0, len(xs), n_pairs)

    # 2) separation-vector components (horizontal only)
    dx = xs[j] - xs[i]
    dy = ys[j] - ys[i]

    # 3) azimuth clockwise from North
    az_rad = np.arctan2(dx, dy)
    az_deg = np.mod(np.degrees(az_rad), 360.0)

    # 4) histogram
    bins = np.arange(0, 360 + bin_deg, bin_deg)
    counts, _ = np.histogram(az_deg, bins=bins)

    # 5) polar bar plot
    theta = np.deg2rad(bins[:-1] + bin_deg / 2)
    width = np.deg2rad(bin_deg)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(theta, counts, width=width, bottom=0.0, edgecolor="k", alpha=0.7)

    # cosmetic: 0° at North, clockwise positive
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Pair-direction rose ({n_pairs:,} pairs, {bin_deg}° bins)")
    plt.show()

#----------------------------------------------------------------------------------------------------------------------#

def fit_dir_variogram(bc, gamma, kind="sph", nugget=True):
    kind = kind.lower()
    if kind.startswith("sph"):
        m = gs.Spherical(dim=1)
    elif kind.startswith("exp"):
        m = gs.Exponential(dim=1)
    elif kind.startswith("gau"):
        m = gs.Gaussian(dim=1)
    else:
        raise ValueError("kind must be sph/exp/gau")
    m.fit_variogram(bc, gamma, nugget=nugget)
    return m

#----------------------------------------------------------------------------------------------------------------------#

def dir_vario(xs, ys, zs, vals, az_deg, bin_edges, tol_deg=15, samp=30000, seed=0):
    bc, g, n = gs.vario_estimate(
        (xs, ys, zs), vals,
        bin_edges=bin_edges,
        direction=[azimuth_to_vector(az_deg)],
        angles_tol=np.deg2rad(tol_deg),
        sampling_size=samp,
        sampling_seed=seed,
        return_counts=True,
    )
    g = g[0] if g.ndim > 1 else g
    n = n[0] if n.ndim > 1 else n
    return bc, g, n

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

# Read in AEM resistivity values
aem = pd.read_csv(in_dir / 'aemlogs.csv')
aem['data_type'] = 'aem'

# Need to convert RHO_I_STD to log using delta method

# Combine dataframes
use_cols = ['x','y','z','row','col','layer','RHO_I','RHO_I_STD','data_type']
resdf = aem.copy()

# Let's work with the natural log of AEM
resdf['logrho'] = np.log(resdf['RHO_I'])

#----------------------------------------------------------------------------------------------------------------------#
# Setup data in a way that GSTOOLS likes
xs     = resdf['x'].to_numpy()
ys     = resdf['y'].to_numpy()
zs     = resdf['z'].to_numpy()
vals   = resdf['logrho'].to_numpy()
xy_max = 4000
z_max  = 300
bin_h = np.linspace(0, xy_max / 2, 20)
bin_v = np.linspace(0, z_max  / 2, 20)

#----------------------------------------------------------------------------------------------------------------------#
# Try out different principal directions
#----------------------------------------------------------------------------------------------------------------------#

rose_of_pair_directions(resdf['x'].to_numpy(),
                        resdf['y'].to_numpy(),
                        n_pairs = 100000,
                        bin_deg = 5)

azimuths = np.arange(-45, 46, 5)
n = len(azimuths)
ncols = 5
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

# 4) Loop over azimuths: estimate, fit, plot
for idx, az in tqdm(enumerate(azimuths), 'Azimuth', len(azimuths)):

    # Estimate directional variogram
    bc, gamma = gs.vario_estimate(
        (xs, ys, zs), vals,
        bin_edges     = bin_h,
        direction     = azimuth_to_vector(az),
        angles_tol    = np.deg2rad(15),
        sampling_size = 10000,
        sampling_seed = 0,
        return_counts = False
    )

    exp_gamma = gamma[0] if gamma.ndim > 1 else gamma

    # Fit an Exponential model (1D fit)
    model = gs.Spherical(dim=1)
    model.fit_variogram(bc, exp_gamma, nugget=True)

    # Plot empirical and fitted variogram
    ax = axes[idx // ncols, idx % ncols]
    ax.scatter(bc, exp_gamma, s=10, label="data")
    model.plot("variogram", ax=ax, x_max=bc.max(), label="exp fit")
    ax.set_title(f"{az:+.0f}° from N | range: {model.len_scale}")
    ax.set_xlabel("Lag (m)")
    ax.set_ylabel("Semivariance")
    ax.legend(fontsize="small")

# Remove any unused axes
for j in range(n, nrows * ncols):
    fig.delaxes(axes[j // ncols, j % ncols])

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Empirical variograms for E–W, N–S, Vertical
#----------------------------------------------------------------------------------------------------------------------#
# Directional variograms
major_angle = 0
minor_angle = major_angle + 90

bc_EW, gamma_EW = gs.vario_estimate(
    (xs, ys, zs), vals,
    bin_edges     = bin_h,
    direction     = [azimuth_to_vector(minor_angle)],
    angles_tol=np.deg2rad(15),
    sampling_size = 25000,
    sampling_seed = 0,
)

bc_NS,  gamma_NS = gs.vario_estimate(
    (xs, ys, zs), vals,
    bin_edges     = bin_h,
    direction     = [azimuth_to_vector(major_angle)],  # N–S
    angles_tol=np.deg2rad(15),
    sampling_size = 25000,
    sampling_seed = 0,
)

bc_z, gamma_V = gs.vario_estimate(
    (xs, ys, zs), vals,
    bin_edges     = bin_v,
    direction     = [np.array([0,0,1])],  # vertical
    sampling_size = 25000,
    angles_tol=np.deg2rad(15),
    sampling_seed = 0,
)

# Fit variograms
m_EW = gs.Spherical(dim=1); m_EW.fit_variogram(bc_EW, gamma_EW, nugget=True)
m_NS = gs.Spherical(dim=1); m_NS.fit_variogram(bc_NS, gamma_NS, nugget=True)
m_V  = gs.Spherical(dim=1); m_V.fit_variogram(bc_z, gamma_V, nugget=True)

print("E–W:", m_EW)
print("N–S:", m_NS)
print("Vert:", m_V)

maj_mod =  m_NS  # For SV, major scale along NS
rot_z = np.pi/2 - np.deg2rad(major_angle)

anis_model = gs.Spherical(
    dim       = 3,
    var       = maj_mod.var,              # sill from major axis
    len_scale = [m_NS.len_scale,          # x-range
                 m_EW.len_scale,          # y-range
                 m_V.len_scale],          # z-range
    nugget    = maj_mod.nugget,           # nugget from major axis
    angles    = [rot_z, 0.0, 0.0],           # Tait–Bryan angles
)

print("Anisotropic model:", anis_model)

axes_dirs = {
    "E–W":      azimuth_to_vector(minor_angle),
    "N–S":      azimuth_to_vector(major_angle),
    "Vertical": np.array([0, 0, 1])
}
fig, (ax_xy, ax_z) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

# 1) Horizontal variograms (E–W & N–S) on ax_xy
ax_xy.scatter(bc_EW, gamma_EW, label=f"{minor_angle}° Experimental", marker="o")
ax_xy.scatter(bc_NS, gamma_NS, label=f"{major_angle}° Experimental", marker="^")

# fitted directional curves for axes 0 and 1
m_EW.plot("vario_axis", axis=0, ax=ax_xy, x_max=bc_EW.max(), label=f"{minor_angle}° only fit")
anis_model.plot("vario_axis", axis=1, ax=ax_xy, x_max=bc_NS.max(), label=f"{major_angle}° fit")
anis_model.plot("vario_axis", axis=0, ax=ax_xy, x_max=bc_EW.max(), label=f"{minor_angle}° fit")
ax_xy.set(title="Horizontal Variograms", ylabel="Semivariance")
ax_xy.legend(loc="best")

# 2) Vertical variogram on ax_z
ax_z.scatter(bc_z, gamma_V, label="Vertical", marker="s")

# fitted directional curve for axis 2
m_V.plot("vario_axis", axis=0, ax=ax_z, x_max=bc_z.max(), label="Vertical only fit")
anis_model.plot("vario_axis", axis=2, ax=ax_z, x_max=bc_z.max(), label="Vertical fit")

ax_z.set(title="Vertical Variogram", xlabel="Lag Distance", ylabel="Semivariance")
ax_z.legend(loc="best")

plt.tight_layout()
plt.show()
plt.savefig(plt_dir / f'{major_angle}_degree_major_variogram.png', dpi=300)

#----------------------------------------------------------------------------------------------------------------------#
s