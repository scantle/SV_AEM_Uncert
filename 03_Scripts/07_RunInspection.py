import matplotlib
matplotlib.use('TkAgg')
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#----------------------------------------------------------------------------------------------------------------------#
# Setup
#----------------------------------------------------------------------------------------------------------------------#

f_dir = Path('06_Outputs/02_still_timeouts_then_power_outage')

par_file   = f_dir / "svihm_ies.2.par.csv"
obs_file   = f_dir / "svihm_ies.2.obs.csv"
rmr_file   = f_dir / "svihm_ies.rmr"

# Metric columns
FJ_METRIC     = "fj_kge"
HEADS_METRIC  = "r2_heads"

# Define "good" by quantile cutoffs (can also use absolute thresholds below)
FJ_GOOD_Q       = 0.75
HEADS_GOOD_Q    = 0.75

# Optional absolute thresholds (set to None to ignore)
FJ_MIN          = None   # e.g., 0.6
HEADS_MIN       = 0.0

# Realizations to export as .par files:
REALIZATIONS_TO_EXPORT = [284, 323]

# Exclusions
EXCLUDE_SUBSTRINGS = ["catch_mult_",'kv_mult']

# Columns that are clearly not parameters and should never end up in .par files
NON_PARAM_COLS = {"real_name"}

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def calc_chain(df):
    out = {}
    # --- K chain ---
    pvals = df.copy()

    out['k_ff'] = pvals['kminff1']
    out['k_mf'] = out['k_ff'] * pvals['kminmf1_m']
    out['k_sc'] = out['k_mf'] * pvals['kminsc1_m']
    out['k_mc'] = out['k_sc'] * pvals['kminmc1_m']
    out['k_vc'] = out['k_mc'] * pvals['kminvc1_m']

    # --- Aniso chain ---
    out['an_vc'] = pvals['anisovc1']
    out['an_mc'] = out['an_vc'] * pvals['anisomc1_m']
    out['an_sc'] = out['an_mc'] * pvals['anisosc1_m']
    out['an_mf'] = out['an_sc'] * pvals['anisomf1_m']
    out['an_ff'] = out['an_mf'] * pvals['anisoff1_m']

    # --- Ss chain ---
    out['ss_ff'] = pvals['ssff1']
    out['ss_mf'] = out['ss_ff'] * pvals['ssmf1_m']
    out['ss_sc'] = out['ss_mf'] * pvals['sssc1_m']
    out['ss_mc'] = out['ss_sc'] * pvals['ssmc1_m']
    out['ss_vc'] = out['ss_mc'] * pvals['ssvc1_m']

    # --- Sy chain ---
    out['sy_sc'] = pvals['sysc1']
    out['sy_mf'] = out['sy_sc'] * pvals['symf1_m']
    out['sy_ff'] = out['sy_mf'] * pvals['syff1_m']
    out['sy_mc'] = out['sy_sc'] * pvals['symc1_m']
    out['sy_vc'] = out['sy_mc'] * pvals['syvc1_m']

    return pd.DataFrame(out), list(out.keys())

#----------------------------------------------------------------------------------------------------------------------#

def pick_good_sets(obs_df: pd.DataFrame, fj_metric: str, heads_metric: str,
                   fj_q=0.75, heads_q=0.75, fj_min=None, heads_min=None):
    """Return index (real_name) for good FJ, good Heads, and intersection."""
    # Compute quantile thresholds
    fj_thr     = obs_df[fj_metric].quantile(fj_q)
    heads_thr  = obs_df[heads_metric].quantile(heads_q)

    if fj_min is not None:
        fj_thr = max(fj_thr, fj_min)
    if heads_min is not None:
        heads_thr = max(heads_thr, heads_min)

    good_fj = obs_df.index[obs_df[fj_metric] >= fj_thr].astype(str)
    good_h  = obs_df.index[obs_df[heads_metric] >= heads_thr].astype(str)
    both    = good_fj.intersection(good_h)

    return good_fj, good_h, both, fj_thr, heads_thr

#----------------------------------------------------------------------------------------------------------------------#

def filter_parameter_columns(df: pd.DataFrame,
                             exclude_substrings=(),
                             extra_drop=()):
    """Return list of columns that look like parameters (exclude obvious non-pars and excluded substrings)."""
    cols = []
    for c in df.columns:
        if c in NON_PARAM_COLS:  # never parameters
            continue
        if c in extra_drop:
            continue
        if any(sub in c for sub in exclude_substrings):
            continue
        # skip clearly non-parameter derived/metric columns if they exist
        cols.append(c)
    return cols

#----------------------------------------------------------------------------------------------------------------------#

def group_summary(df_params: pd.DataFrame, idx_good, idx_all, label="good"):
    """Compute simple effect-size style summaries (mean difference / pooled std) for quick screening."""
    g = df_params.loc[idx_good]
    a = df_params.loc[idx_all]

    mu_g = g.mean()
    mu_a = a.mean()
    sd_p = np.sqrt(0.5 * (g.std(ddof=1)**2 + a.std(ddof=1)**2))
    smd  = (mu_g - mu_a) / sd_p.replace(0.0, np.nan)

    out = pd.concat({
        f"{label}_mean": mu_g,
        "all_mean": mu_a,
        f"{label}_minus_all": mu_g - mu_a,
        f"{label}_SMD": smd
    }, axis=1)

    # Rank by absolute SMD (largest absolute differences first)
    out["rank_abs_SMD"] = out[f"{label}_SMD"].abs().rank(ascending=False, method="dense")
    return out.sort_values("rank_abs_SMD")

#----------------------------------------------------------------------------------------------------------------------#

def sparse_logit_screen(df_params: pd.DataFrame, idx_good, idx_other, label="good", max_features=25):
    """L1-logistic regression to screen for predictive parameters (sign and magnitude are informative)."""

    # coerce to strings and intersect with df_params.index
    idx_good  = pd.Index(idx_good).astype(str).intersection(df_params.index)
    idx_other = pd.Index(idx_other).astype(str).intersection(df_params.index)

    y = pd.Series(0, index=df_params.index, dtype=int)
    y.loc[idx_good] = 1
    y = y.loc[df_params.index]  # align

    use_idx = idx_good.union(idx_other)
    X_use = df_params.loc[use_idx]
    y_use = y.loc[use_idx]

    # Fit L1-logistic with standardization; increase C if everything zeros out
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                         LogisticRegression(penalty="l1", solver="saga", max_iter=5000, C=0.5))
    pipe.fit(X_use.values, y_use.values)

    coefs = pd.Series(pipe.named_steps['logisticregression'].coef_.ravel(),
                      index=df_params.columns, name=f"logit_coef_{label}")
    coefs = coefs[coefs != 0].sort_values(key=lambda s: s.abs(), ascending=False)
    return coefs.head(max_features)

#----------------------------------------------------------------------------------------------------------------------#

def write_par_files_for_realizations(par_df: pd.DataFrame, real_ids, out_dir: Path):
    """
    Write two files per realization:
      1) Classic PEST .par (lines: 'parname  value')
      2) Two-column CSV 'parnme,parval1' (handy for pyemu and many utilities)
    Excludes non-parameter columns automatically.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify parameter columns = all columns except NON_PARAM_COLS
    param_cols = [c for c in par_df.columns if c not in NON_PARAM_COLS]

    for rid in real_ids:
        # Realizations are usually integer strings in 'real_name'; normalize
        rid_str = str(rid)
        if rid_str not in par_df.index:
            # Try zero-padded if needed (PEST++ often uses bare ints; adjust here if your naming differs)
            raise ValueError(f"Realization {rid} not found in par_df.index")

        row = par_df.loc[rid_str, param_cols]

        csv_path = out_dir / f"real_{rid}_parnme_parval1.csv"
        pd.DataFrame({"parnme": row.index, "parval1": row.values}).to_csv(csv_path, index=False)

    return

#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Analysis
#----------------------------------------------------------------------------------------------------------------------#

# Read
par = pd.read_csv(par_file, dtype={"real_name": str}, index_col=['real_name'])
obs = pd.read_csv(obs_file, dtype={"real_name": str}, index_col=['real_name'])

# Derived chain values
derived, chain_cols = calc_chain(par)
df_all = par.join(derived, how="left")

# collect metric columns for your own inspection
metric_cols = [c for c in obs.columns if any(sub in c for sub in ["nse", "r2_", "kge", "rmse", "mae"])]

# A little manual exploration
obs_metrics = obs.loc[:, metric_cols]
obs_metrics.sort_values(by=['fj_kge','r2_heads'], ascending=[False, False], inplace=True)

# Identify good sets
good_fj, good_h, good_both, fj_thr, heads_thr = pick_good_sets(
    obs, FJ_METRIC, HEADS_METRIC, FJ_GOOD_Q, HEADS_GOOD_Q, FJ_MIN, HEADS_MIN
)

print(f"[Cutoffs] {FJ_METRIC} >= {fj_thr:.4g} ; {HEADS_METRIC} >= {heads_thr:.4g}")
print(f"Counts: good FJ={len(good_fj)}, good heads={len(good_h)}, both={len(good_both)} "
      f"out of N={len(obs)}")

# Build parameter matrices to analyze (exclude the untrustworthy and any obvious non-par columns)
# We’ll analyze BOTH the original parameters and the derived chain values.
exclude = tuple(EXCLUDE_SUBSTRINGS)
base_param_cols = filter_parameter_columns(par, exclude_substrings=exclude, extra_drop=())
derived_cols = chain_cols  # always keep
analyze_cols = sorted(set(base_param_cols).union(derived_cols))

X = df_all.loc[:, analyze_cols].copy()

# ---- Summaries: mean differences / SMD
smry_fj = group_summary(X, good_fj, X.index, label="good_FJ")
smry_heads = group_summary(X, good_h, X.index, label="good_heads")
smry_both = group_summary(X, good_both, X.index, label="good_both")

# Save quick tables
out_dir = f_dir / "ies_iter_analysis"
out_dir.mkdir(parents=True, exist_ok=True)
smry_fj.to_csv(out_dir / "summary_good_FJ.csv")
smry_heads.to_csv(out_dir / "summary_good_heads.csv")
smry_both.to_csv(out_dir / "summary_good_both.csv")

# ---- Sparse logistic screens (which pars help predict 'good'?)
# For a fair contrast, define "other" as those NOT in the target set
other_fj = X.index.difference(good_fj)
other_heads = X.index.difference(good_h)
other_both = X.index.difference(good_both)

logit_fj = sparse_logit_screen(X, good_fj, other_fj, label="good_FJ", max_features=30)
logit_heads = sparse_logit_screen(X, good_h, other_heads, label="good_heads", max_features=30)
logit_both = sparse_logit_screen(X, good_both, other_both, label="good_both", max_features=30)

logit_fj.to_csv(out_dir / "logit_screen_good_FJ.csv", header=True)
logit_heads.to_csv(out_dir / "logit_screen_good_heads.csv", header=True)
logit_both.to_csv(out_dir / "logit_screen_good_both.csv", header=True)

# ---- Quick, human-readable top lists in console
def show_top(s: pd.Series, n=12, title=""):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(s.head(n))

show_top(smry_fj[f"good_FJ_SMD"].abs().sort_values(ascending=False), title="Top abs SMD: good FJ")
show_top(smry_heads[f"good_heads_SMD"].abs().sort_values(ascending=False), title="Top abs SMD: good heads")
show_top(smry_both[f"good_both_SMD"].abs().sort_values(ascending=False), title="Top abs SMD: good both")

show_top(logit_fj, title="Sparse logit screen: good FJ (coef magnitude)")
show_top(logit_heads, title="Sparse logit screen: good heads (coef magnitude)")
show_top(logit_both, title="Sparse logit screen: good both (coef magnitude)")

# pick top parameters (here by logistic coefficients)
top_pars = logit_both.abs().sort_values(ascending=False).head(30).index

plot_dir = out_dir / "param_histograms"
plot_dir.mkdir(parents=True, exist_ok=True)

for p in top_pars:
    plt.figure(figsize=(6, 4))
    sns.histplot(X[p], color="0.7", label="All", bins=25, kde=False)
    sns.histplot(X.loc[good_fj, p], color="tab:blue", label="Good FJ", bins=25, kde=False, alpha=0.5)
    sns.histplot(X.loc[good_h, p], color="tab:green", label="Good Heads", bins=25, kde=False, alpha=0.5)
    sns.histplot(X.loc[good_both, p], color="tab:orange", label="Good Both", bins=25, kde=False, alpha=0.5)

    plt.legend()
    plt.title(f"Distribution of {p}")
    plt.xlabel(p)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{p}_hist.png", dpi=200)
    plt.close()

# ---- Export .par files for specific realizations
write_par_files_for_realizations(par, REALIZATIONS_TO_EXPORT, out_dir / "selected_pars")

print("\nDone. Outputs written to:", out_dir.resolve())

#----------------------------------------------------------------------------------------------------------------------#
