import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import copy
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import t2py

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

# Files
mod_dir = Path('./02_Models/r2p_cv/MODFLOW/')
t2p_dir = Path('./02_Models/r2p_cv/preproc/')
t2p_inf = t2p_dir / 'svihm.t2p'
t2p_log = t2p_dir / 'res_log.csv'
t2p_path = Path('./02_Models/Bin/Texture2Par.exe')
run_dir = Path('./02_Models/CV_runs/')
run_dir.mkdir(parents=True, exist_ok=True)

t2p_files = ['interp_tex_dists.csv', 'kv_mult.csv','SVIHM_TEMPLATE.upw']

classes = ['logrho']
textures = ['Fine', 'Mixed_Fine','Sand', 'Mixed_Coarse', 'Very_Coarse']
FOLDS = 10
NN_LIST = [16]  #[32, 48, 64, 96, 128, 200, 300]
INTERVAL_LIST = [10] #[30, 15, 10, 5, 3]

np.random.seed(667)

# -------------------------------------------------------------------------------------------------------------------- #
# Functions/Classes
# -------------------------------------------------------------------------------------------------------------------- #

def update_nnear(t2p_list, nnear_lines, new_nnear):
    new_t2p_list = copy.copy(t2p_list)
    for i in nnear_lines:
        target = new_t2p_list[i].split()[-1]
        new_t2p_list[i] = new_t2p_list[i].replace(target, str(new_nnear))
    return new_t2p_list

# -------------------------------------------------------------------------------------------------------------------- #

def update_max_log_length(t2p_list, max_log_line, max_log_length):
    new_t2p_list = copy.copy(t2p_list)
    target = new_t2p_list[max_log_line].split()[-1]
    new_t2p_list[max_log_line] = new_t2p_list[max_log_line].replace(target, str(max_log_length))
    return new_t2p_list

# -------------------------------------------------------------------------------------------------------------------- #

def run_t2p(dir, t2p_path, t2p_infile):
    """Run the external program with given argument list."""
    try:
        result = subprocess.run(
            [t2p_path, t2p_infile],
            cwd=dir,
            capture_output=True,
            text=True,
            check=True      # raise exception on failure
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e}\n{e.stdout}\n{e.stderr}"

# -------------------------------------------------------------------------------------------------------------------- #

def job(run_config):
    run_dir, t2p_infile = run_config
    print(f" - STARTED:  {run_dir.name}")
    result = run_t2p(run_dir, t2p_path.absolute(), t2p_infile)
    print(f" - FINISHED: {run_dir.name}")
    return result

# -------------------------------------------------------------------------------------------------------------------- #

def get_texture_results(dir, textures):
    texture_data = {tex: pd.read_csv(dir / f't2p_{tex.upper()}.csv', na_values=-999) for tex in textures}
    combined = None
    for tex_name, df in texture_data.items():
        layers = sum(df.columns.str.startswith('Layer'))
        df = df.rename(columns={df.columns[0]: 'Row', df.columns[1]: 'Col'})
        for k in range(0,layers):
            if k==0:  laycombined = df['Layer1']
            else:
                laycombined = pd.concat([laycombined, df[f'Layer{k+1}']])
        if combined is None:
            combined = pd.DataFrame({tex_name : laycombined.values})
        else:
            combined[tex_name] = laycombined.values

    # Normalize & return
    return combined.div(combined.sum(axis=1), axis=0)

# -------------------------------------------------------------------------------------------------------------------- #

def calculate_cv_metrics(full_df, fold_df, textures):
    # Combine prediction and truth
    combined = pd.concat([full_df[textures], fold_df[textures]], axis=1, keys=['true', 'pred'])
    combined = combined.dropna()
    if combined.empty:
        return None

    true_vals = combined['true'].values
    pred_vals = combined['pred'].values

    brier_score = ((true_vals - pred_vals) ** 2).sum(axis=1).mean()
    mae_score = np.abs(true_vals - pred_vals).mean(axis=1).mean()

    true_class = true_vals.argmax(axis=1)
    pred_class = pred_vals.argmax(axis=1)
    misclass_rate = (true_class != pred_class).mean()

    out_of_bounds = ((pred_vals < 0) | (pred_vals > 1)).sum()
    out_of_bounds_prop = out_of_bounds / pred_vals.size

    return {
        'brier_score': brier_score,
        'mae_score': mae_score,
        'misclass_rate': misclass_rate,
        'out_of_bounds_prop': out_of_bounds_prop,
        'n_samples': len(pred_vals)
    }

# -------------------------------------------------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------------------------------------------------- #

# Setup results DF
all_logs = pd.DataFrame()

# Read in well log file, setup folds
basecase = t2py.Dataset(classes=classes,
                        variance_col=True,
                        filename=t2p_log, file_sep=',')
fold_tag = np.random.randint(0, FOLDS, size=basecase.max_id)

# Read in main input file as list
t2p_in = []
log_length_line = 0
nnear_lines = []
in_vario_block = False
in_opt_block = False
with open(t2p_inf, 'r') as f:
    for i, line in enumerate(f):
        t2p_in.append(line)
        # OPTIONS
        if line.startswith('BEGIN OPTIONS'):
            in_opt_block = True
        elif line.startswith('END OPTIONS'):
            in_opt_block = False
        # NNEAR
        elif line.startswith('BEGIN VARIOGRAMS'):
            in_vario_block = True
        elif line.startswith('END VARIOGRAMS'):
            in_vario_block = False
        if in_opt_block and line.strip().startswith('MAX_LOG_LENGTH'):
            log_length_line = i
        if in_vario_block and line.strip().startswith('CLASS'):
            nnear_lines.append(i+1)
print(f'Found max_log_length on line {log_length_line}')
print(f'Found {len(nnear_lines)} variogram nnear specifications in {t2p_inf.name}')

# Setup tests
for n in NN_LIST:
    for k in INTERVAL_LIST:
        print('\n*----------------------------------------------------')
        print(f'* Starting nnear = {n}, INTERVAL = {k}')
        print('*----------------------------------------------------\n')

        # Setup T2P file
        t2p_scen = update_nnear(t2p_in, nnear_lines, n)
        t2p_scen = update_max_log_length(t2p_scen, log_length_line, k)

        # Folder
        scen_dir = f'{n}_nnear_{k}_max_int'

        nruns = []

        # Setup full run
        full_dir = run_dir / scen_dir / f'full_run'
        full_dir.mkdir(parents=True, exist_ok=True)

        # Copy in essential files
        for f in t2p_files:
            shutil.copy(t2p_dir / f, full_dir / f)

        # Write files
        with open(full_dir / t2p_inf.name, "w") as f:
            for line in t2p_scen:
                f.write(line)
        nruns.append((full_dir.absolute(), t2p_inf.name))
        print(f'Wrote {full_dir / t2p_inf.name}')

        basecase.write_file(filename=full_dir / t2p_log.name)
        print(f'Wrote {full_dir / t2p_log.name}', sep=',')

        # Setup Folds
        for f in range(0, FOLDS):

            # Create a new folder
            fold_dir = run_dir / scen_dir / f'fold_{f}'
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Copy in essential files
            for item in t2p_files:
                shutil.copy(t2p_dir / item, fold_dir / item)

            # Setup dataset
            folddf = basecase.df.copy()
            loc_ids = [idx for idx, tag in enumerate(fold_tag) if tag != f]
            folddf = folddf[folddf.ID.isin(loc_ids)]
            foldcase = t2py.Dataset(classes, variance_col=True)
            foldcase.add_wells_by_df(folddf, name_col='Location', fill_missing=False)

            # Write files
            with open(fold_dir / t2p_inf.name, "w") as f:
                for line in t2p_scen:
                    f.write(line)
            nruns.append((fold_dir.absolute(), t2p_inf.name))
            print(f'Wrote {fold_dir / t2p_inf.name}')

            foldcase.write_file(filename=fold_dir / t2p_log.name, sep=',')
            print(f'Wrote {fold_dir / t2p_log.name}')

        # Copy in model folder
        try:
            shutil.copytree(mod_dir, run_dir / scen_dir / mod_dir.name)
        except FileExistsError:
            pass

        # Run for all nnear folders
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(job, nruns))

        # Read in full results
        full = get_texture_results(full_dir, textures)

        # Loop over folds getting results
        cv_results = []

        for f in range(FOLDS):
            fold_dir = run_dir / scen_dir / f'fold_{f}'
            fold_pred = get_texture_results(fold_dir, textures)

            # Remove NAs and calculate metrics
            result = calculate_cv_metrics(full, fold_pred, textures)

            if result is not None:
                result.update({'nnear': n, 'interval_len': k, 'fold': f})
                cv_results.append(result)
            else:
                print(f"Fold {f}: No valid data after dropping NAs.")
        cv_log_df = pd.DataFrame(cv_results)
        all_logs = pd.concat([all_logs, cv_log_df], ignore_index=True)

# Write Out Log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = run_dir / f'CV_Results_{timestamp}.csv'
all_logs.to_csv(log_path, index=False)

# --- Analysis ---
print("\nCross-validation complete.")
print(f"Saved raw CV logs to: {log_path}\n")

required_cols = {"nnear","interval_len","brier_score","mae_score","misclass_rate","out_of_bounds_prop","n_samples"}
missing = required_cols - set(all_logs.columns)
if missing:
    raise ValueError(f"Missing required columns in all_logs: {missing}. Did you add 'interval_len' to result.update(...)?")

# 1) Aggregate across folds for each (interval_len, nnear)
grouped = (
    all_logs
    .groupby(["interval_len","nnear"], as_index=False)
    .agg({
        "brier_score":"mean",
        "mae_score":"mean",
        "misclass_rate":"mean",
        "out_of_bounds_prop":"mean",
        "n_samples":"sum"
    })
    .sort_values(["interval_len","nnear"])
)

# Save the grouped summary
summary_csv = run_dir / f"CV_Summary_by_interval_nnear_{timestamp}.csv"
grouped.to_csv(summary_csv, index=False)
print(f"Saved summary table: {summary_csv}")

# 2) Best by each metric (global)
def _best_row(df, metric):
    idx = df[metric].idxmin()
    return df.loc[idx, ["interval_len","nnear",metric]]

best_brier    = _best_row(grouped, "brier_score")
best_mae      = _best_row(grouped, "mae_score")
best_misclass = _best_row(grouped, "misclass_rate")

print("\n*----------------- Best by metric (global) -----------------*")
print(f"- Best Brier Score    : interval={best_brier.interval_len}, nnear={best_brier.nnear}, score={best_brier.brier_score:.6f}")
print(f"- Best MAE            : interval={best_mae.interval_len}, nnear={best_mae.nnear}, score={best_mae.mae_score:.6f}")
print(f"- Best Misclass. Rate : interval={best_misclass.interval_len}, nnear={best_misclass.nnear}, rate={best_misclass.misclass_rate:.6f}")
print("*-----------------------------------------------------------*")

# 3) Rank-sum across the three error metrics
rank_df = grouped.copy()
for m in ["brier_score","mae_score","misclass_rate"]:
    rank_df[f"rank_{m}"] = rank_df[m].rank(method="min", ascending=True)

rank_df["rank_sum"] = rank_df[["rank_brier_score","rank_mae_score","rank_misclass_rate"]].sum(axis=1)
best_overall_idx = rank_df["rank_sum"].idxmin()
best_overall = rank_df.loc[best_overall_idx, ["interval_len","nnear","rank_sum"]]

# Best nnear within each interval_len by rank-sum
best_per_interval = (
    rank_df.sort_values(["interval_len","rank_sum"])
           .groupby("interval_len", as_index=False)
           .first()[["interval_len","nnear","rank_sum"]]
)

# Save rank tables
rank_csv = run_dir / f"CV_RankSum_{timestamp}.csv"
rank_df.to_csv(rank_csv, index=False)
best_per_interval_csv = run_dir / f"CV_BestPerInterval_{timestamp}.csv"
best_per_interval.to_csv(best_per_interval_csv, index=False)

print("\n*----------------- Rank-sum selection -----------------*")
print(f"- Overall best (rank-sum): interval={best_overall.interval_len}, nnear={best_overall.nnear}, rank_sum={int(best_overall.rank_sum)}")
print("\nPer-interval best (rank-sum):")
print(best_per_interval.to_string(index=False))
print("*------------------------------------------------------*")

# 4) Pretty tables (pivot) and heatmaps for quick inspection
def _pivot(metric):
    pv = grouped.pivot(index="interval_len", columns="nnear", values=metric).sort_index(ascending=True)
    return pv

metrics = ["brier_score","mae_score","misclass_rate"]
pivots = {m: _pivot(m) for m in metrics}

# Save pivots
for m, pv in pivots.items():
    pv_path = run_dir / f"Pivot_{m}_{timestamp}.csv"
    pv.to_csv(pv_path)
    print(f"Saved pivot for {m}: {pv_path}")

# Heatmap helper (matplotlib-only)
def plot_heatmap(pv, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pv.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("nnear")
    ax.set_ylabel("max_log_length")
    # tick labels
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    # annotate values
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            val = pv.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3g}", ha="center", va="center")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("score")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

# Plot heatmaps for each metric
for m in metrics:
    pv = pivots[m]
    if pv.size > 0:
        outp = run_dir / f"Heatmap_{m}_{timestamp}.png"
        plot_heatmap(pv, f"{m} vs (interval_len, nnear)", outp)
        print(f"Saved heatmap: {outp}")

# 5) Quick textual table
print("\n*----------------- Summary table -----------------*")
disp = grouped.copy()
for col in ["brier_score","mae_score","misclass_rate","out_of_bounds_prop"]:
    disp[col] = disp[col].round(6)
print(disp.to_string(index=False))
print("*------------------------------------------------*")
