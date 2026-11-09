"""
ER-texture scale ladder
Global log-scale ladder (ln S_FF + four ln gaps), soft order floor, and the texture probability-order check.
Used by RES2PAR_preproc.py (forward run) and 06_Setup_PEST-iES_Control_File.py (prior draws).
"""
import numpy as np

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

texs = ['Fine', 'Mixed_Fine', 'Sand', 'Mixed_Coarse', 'Very_Coarse']
ladder_names = ['ladder_lnff', 'ladder_g1', 'ladder_g2', 'ladder_g3', 'ladder_g4']

# Soft floor on the ln gaps: ratios never drop below ~1.02, so texture order holds without a hard bound
GMIN = np.log(1.02)
BETA = 0.02

# Order check: max allowed dip in the expected texture index, evaluated over the AEM 5-95% ER range
DIP_MAX = 0.25
DIP_GRID = np.exp(np.linspace(np.log(18.0), np.log(780.0), 1500))

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def soft_gap(g, gmin=GMIN, beta=BETA):
    """gmin + beta*ln(1 + exp((g - gmin)/beta)), written to be stable for large |g - gmin|"""
    return gmin + beta * np.logaddexp(0.0, (np.asarray(g, dtype=float) - gmin) / beta)

#----------------------------------------------------------------------------------------------------------------------#

def ladder_lnscales(theta, gmin=GMIN, beta=BETA):
    """
    theta = [ln S_FF, g1, g2, g3, g4] (one set, or n x 5)
    Returns ln S for Fine..Very_Coarse (5, or n x 5)
    """
    theta = np.asarray(theta, dtype=float)
    gaps = soft_gap(theta[..., 1:], gmin, beta)
    return np.concatenate([theta[..., :1], theta[..., :1] + np.cumsum(gaps, axis=-1)], axis=-1)

#----------------------------------------------------------------------------------------------------------------------#

def ladder_theta_from_scales(scales):
    """Inverse (ignoring the floor): scales (5,) or (n,5) in ohm-m -> theta"""
    ln_s = np.log(np.asarray(scales, dtype=float))
    return np.concatenate([ln_s[..., :1], np.diff(ln_s, axis=-1)], axis=-1)

#----------------------------------------------------------------------------------------------------------------------#

def tex_probs(rho, ln_s, shapes):
    """
    Texture probabilities at ER values rho (ohm-m), same as PreProc.f90 calc_normalized_likelihoods
    (without its +/- 1 ohm-m integration), including the tail rule: below the lowest scale all goes to that
    texture, above the highest all goes to that one.
    Returns 5 x len(rho)
    """
    x = np.log(np.atleast_1d(np.asarray(rho, dtype=float)))
    ln_s = np.asarray(ln_s, dtype=float)
    shapes = np.asarray(shapes, dtype=float)
    z = (x[None, :] - ln_s[:, None]) / shapes[:, None]
    p = np.exp(-0.5 * z**2) / shapes[:, None]
    p = p / p.sum(axis=0)
    lo = x < ln_s.min()
    hi = x > ln_s.max()
    p[:, lo] = 0.0
    p[ln_s.argmin(), lo] = 1.0
    p[:, hi] = 0.0
    p[ln_s.argmax(), hi] = 1.0
    return p

#----------------------------------------------------------------------------------------------------------------------#

def order_deficit(rho, ln_s, shapes):
    """How far E[texture index] sits below the largest value it reached at a lower ER (0 where monotone)"""
    et = (np.arange(len(ln_s))[:, None] * tex_probs(rho, ln_s, shapes)).sum(axis=0)
    return np.maximum.accumulate(et) - et

#----------------------------------------------------------------------------------------------------------------------#

def order_dip(ln_s, shapes, rho_grid=DIP_GRID):
    """Largest dip in E[texture index] as ER increases. Shift (delta) only slides this along ER, so it
    depends on the global ladder only"""
    return order_deficit(rho_grid, ln_s, shapes).max()

#----------------------------------------------------------------------------------------------------------------------#

def ladder_prior(boot_scales):
    """
    Prior for theta from the bootstrap scale solutions (n x 5, ohm-m, Fine..Very_Coarse)
    Returns mean (5,) and covariance (5 x 5) of theta
    """
    th = ladder_theta_from_scales(boot_scales)
    return th.mean(axis=0), np.cov(th.T)
