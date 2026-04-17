import numpy as np
import pandas as pd
from pathlib import Path
from hydroeval import evaluator, nse, kgeprime, rmse, pbias
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------------------

data_dir = Path("01_Data/")
f_dir    = Path("06_Outputs/06_wtfx/")
obs_file = f_dir / "svihm_ies.3.obs.csv"
str_obs_file = data_dir / "streamflow_obs_std.csv"
head_obs_file = data_dir / 'head_obs_master.csv'

out_dir  = Path("05_Outputs/")
out_dir.mkdir(parents=True, exist_ok=True)

# Low-flow cutoffs in linear space
LOW_CUTOFF = {
    "FJ":  2.44e5,
    "SCK": 18294.0,
    "AS":  56370.0,
    "BY":  43079.0,
}

STREAM_GROUPS = ["str_FJ", "str_SCK", "str_AS", "str_BY"]

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

BACKTRANSFORM = lambda x: np.power(10.0, x)


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def parse_daily_date_from_obsnme(obsnme: str):
    try:
        date_part = obsnme.split("_", 1)[1]
        return pd.to_datetime(date_part, format="%Y%m%d")
    except Exception:
        return pd.NaT


def _safe_metric_table(obs: np.ndarray, sim: np.ndarray) -> dict:
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return {"RMSE": np.nan, "NSE": np.nan, "KGE": np.nan, "PBIAS": np.nan}
    o = obs[m]
    s = sim[m]
    try:
        return {
            "RMSE": float(evaluator(rmse,     s, o, axis=0)[0]),
            "NSE":  float(evaluator(nse,      s, o, axis=0)[0]),
            "KGE":  float(evaluator(kgeprime, s, o, axis=0)[0][0]),
            "PBIAS":float(evaluator(pbias,    s, o, axis=0)[0]),
        }
    except Exception:
        return {"RMSE": np.nan, "NSE": np.nan, "KGE": np.nan, "PBIAS": np.nan}


def _summarize_over_ensemble(obs_vec: np.ndarray, sim_df: pd.DataFrame) -> dict:
    out = {}

    # Base
    if "base" in sim_df.index:
        base_sim = sim_df.loc["base"].to_numpy(dtype=float)
        bm = _safe_metric_table(obs_vec, base_sim)
        out.update({f"base_{k}": v for k, v in bm.items()})
    else:
        out.update({f"base_{k}": np.nan for k in ["RMSE","NSE","KGE","PBIAS"]})

    # Ensemble
    ens_df = sim_df.drop(index="base", errors="ignore")
    ens_arr = ens_df.to_numpy(dtype=float)
    out["n_ens"] = int(ens_arr.shape[0])

    if ens_arr.shape[0] == 0:
        out.update({f"ens_{stat}_{k}": np.nan
                    for stat in ["min","mean","max"]
                    for k in ["RMSE","NSE","KGE","PBIAS"]})
        return out

    met = [_safe_metric_table(obs_vec, ens_arr[r, :]) for r in range(ens_arr.shape[0])]
    met_df = pd.DataFrame(met)

    for k in ["RMSE","NSE","KGE","PBIAS"]:
        out[f"ens_min_{k}"]  = float(met_df[k].min())
        out[f"ens_mean_{k}"] = float(met_df[k].mean())
        out[f"ens_max_{k}"]  = float(met_df[k].max())

    return out


def _stream_code_from_group(obsgnme: str) -> str:
    return obsgnme.split("_", 1)[1]


# ----------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------

str_obs = pd.read_csv(str_obs_file)
str_obs["obsnme"] = str_obs["obsnme"].astype(str)
str_obs["obsnme_lower"] = str_obs["obsnme"].str.lower()

head_obs = pd.read_csv(head_obs_file)
head_obs["date"] = pd.to_datetime(head_obs["date"])

run_results = pd.read_csv(obs_file, dtype={"real_name": str}, index_col=["real_name"])
run_results.columns = run_results.columns.astype(str).str.lower()

if "obsval" not in str_obs.columns:
    raise KeyError(f"{str_obs_file.name} must contain 'obsval'. Found: {list(str_obs.columns)}")

str_obs["date"] = str_obs["obsnme"].apply(parse_daily_date_from_obsnme)
str_obs["month"] = str_obs["date"].dt.month

# ----------------------------------------------------------------------------------------------------------------------
# Streamflow Metrics
# ----------------------------------------------------------------------------------------------------------------------

rows = []

for g in STREAM_GROUPS:
    sub = str_obs.loc[str_obs["obsgnme"] == g].copy()
    if sub.empty:
        print(f"[WARN] No rows found in {str_obs_file.name} for group {g}")
        continue

    stream = _stream_code_from_group(g)
    if stream not in LOW_CUTOFF:
        print(f"[WARN] No low-flow cutoff provided for {stream}; skipping")
        continue

    # match obs names to simulation columns
    common_obs = [o for o in sub["obsnme_lower"].tolist() if o in run_results.columns]
    if len(common_obs) == 0:
        print(f"[WARN] No matching obs in run_results for {g}")
        continue

    # align to common_obs order
    sub = sub.set_index("obsnme_lower").loc[common_obs].reset_index()

    # observed values in log space (use as-is for metrics)
    obs_log = sub["obsval"].to_numpy(dtype=float)

    # flow-class assignment based on linear back-transform of observed log values
    obs_lin = BACKTRANSFORM(obs_log)
    cutoff = LOW_CUTOFF[stream]
    sub["flow_class"] = np.where(obs_lin < cutoff, "low", "inbank")

    # sims in log space
    sim_df_all = run_results[common_obs]

    # ----------------------------------------------------------------------------------
    # A) Full-period metrics by flow class: All / Low / Inbank
    # ----------------------------------------------------------------------------------
    for cls, mask in [
        ("all",    np.ones(len(sub), dtype=bool)),
        ("low",    (sub["flow_class"].to_numpy() == "low")),
        ("inbank", (sub["flow_class"].to_numpy() == "inbank")),
    ]:
        if mask.sum() < 2:
            continue

        obs_vec = obs_log[mask]
        sim_df  = sim_df_all.loc[:, mask]   # boolean mask on columns
        stats = _summarize_over_ensemble(obs_vec, sim_df)

        rows.append({
            "stream": stream,
            "group": g,
            "period": "All",
            "month": "",
            "flow_class": cls,
            "n_obs": int(mask.sum()),
            **stats
        })

    # ----------------------------------------------------------------------------------
    # B) Monthly metrics (Jan..Dec) WITHOUT flow-class split
    # ----------------------------------------------------------------------------------
    for m in range(1, 13):
        month_mask = (sub["month"].to_numpy() == m)
        if month_mask.sum() < 2:
            continue

        obs_vec = obs_log[month_mask]
        sim_df  = sim_df_all.loc[:, month_mask]
        stats = _summarize_over_ensemble(obs_vec, sim_df)

        rows.append({
            "stream": stream,
            "group": g,
            "period": "Month",
            "month": MONTH_NAMES[m],
            "flow_class": "all",          # single monthly row per month
            "n_obs": int(month_mask.sum()),
            **stats
        })

summary_df = pd.DataFrame(rows)

# nice ordering
month_order = [""] + [MONTH_NAMES[m] for m in range(1, 13)]
summary_df["month"] = pd.Categorical(summary_df["month"], categories=month_order, ordered=True)

flow_order = ["all", "low", "inbank"]
summary_df["flow_class"] = pd.Categorical(summary_df["flow_class"], categories=flow_order, ordered=True)

summary_df = summary_df.sort_values(["stream", "period", "month", "flow_class"])

out_csv = out_dir / f"stream_metrics_flowclass_and_month_{obs_file.stem}.csv"
summary_df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

# ----------------------------------------------------------------------------------------------------------------------
# Head Metrics
# ----------------------------------------------------------------------------------------------------------------------

def _metrics_one(obs: np.ndarray, sim: np.ndarray) -> dict:
    """R2 (sklearn), RMSE/PBIAS (hydroeval)."""
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return {"R2": np.nan, "RMSE": np.nan, "PBIAS": np.nan}
    o = obs[m]
    s = sim[m]
    return {
        "R2":    float(r2_score(o, s)),
        "RMSE":  float(evaluator(rmse,  s, o, axis=0)[0]),
        "PBIAS": float(evaluator(pbias, s, o, axis=0)[0]),
    }

def head_metrics_avg_plus_diff(
    head_obs: pd.DataFrame,
    run_results: pd.DataFrame,
    use_weighted_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (overall_df, per_well_df)

    overall_df: 1 row with base + ens min/mean/max metrics, pooled across all wells/times
    per_well_df: one row per well with base + ens min/mean/max metrics
    """

    # --- split obs tables ---
    diff_df = head_obs.loc[head_obs["group"] == "hds_diff"].copy()
    avg_df  = head_obs.loc[head_obs["group"] == "hds_avg"].copy()

    # normalize names
    if "obsnme_lower" not in diff_df.columns:
        diff_df["obsnme_lower"] = diff_df["obsnme"].astype(str).str.lower()
    if "obsnme_lower" not in avg_df.columns:
        avg_df["obsnme_lower"] = avg_df["obsnme"].astype(str).str.lower()

    run_cols = set(run_results.columns.astype(str).str.lower())
    run_results.columns = run_results.columns.astype(str).str.lower()

    real_names = run_results.index.astype(str).tolist()
    ens_names = [r for r in real_names if r != "base"]

    # collectors for pooled metrics
    pooled_obs_parts = []
    pooled_base_parts = []
    pooled_ens_parts = {r: [] for r in ens_names}

    per_well_rows = []

    wells = sorted(diff_df["wellid"].dropna().unique())
    for wid in wells:
        sub = diff_df.loc[diff_df["wellid"] == wid].copy()
        sub = sub.dropna(subset=["date"])
        if sub.empty:
            continue

        avg_row = avg_df.loc[avg_df["wellid"] == wid]
        if avg_row.empty:
            continue
        avg_row = avg_row.iloc[0]

        avg_name = str(avg_row["obsnme_lower"])
        if avg_name not in run_cols:
            continue

        # diff obs columns that exist in run_results
        obs_cols = [o for o in sub["obsnme_lower"].astype(str).tolist() if o in run_cols]
        if len(obs_cols) < 2:
            continue

        # align to obs_cols order
        sub = sub.set_index("obsnme_lower").loc[obs_cols].reset_index()

        # observed components (match your plotting script naming: obval, weight)
        diff_obs = sub["obval"].to_numpy(dtype=float)
        w_obs = sub["weight"].to_numpy(dtype=float) if "weight" in sub.columns else np.ones_like(diff_obs)
        avg_obs = float(avg_row["obval"])

        if use_weighted_only:
            mask = (w_obs > 0) & np.isfinite(diff_obs)
        else:
            mask = np.isfinite(diff_obs)

        if mask.sum() < 2:
            continue

        obs_elev = (avg_obs + diff_obs)[mask]

        # sims (all realizations) for diff + avg
        sim_diff = run_results[obs_cols]     # DataFrame: rows=realizations, cols=time
        sim_avg  = run_results[avg_name]     # Series: index=realizations

        # ---- base (per well + pooled) ----
        base_stats = {"R2": np.nan, "RMSE": np.nan, "PBIAS": np.nan}
        if "base" in run_results.index:
            base_elev = (float(sim_avg.loc["base"]) + sim_diff.loc["base"].to_numpy(dtype=float))[mask]
            base_stats = _metrics_one(obs_elev, base_elev)
            pooled_obs_parts.append(obs_elev)
            pooled_base_parts.append(base_elev)

        # ---- ensemble metrics per realization (per well + pooled) ----
        ens_stats_list = []
        for r in ens_names:
            if r not in sim_diff.index or r not in sim_avg.index:
                continue
            r_elev = (float(sim_avg.loc[r]) + sim_diff.loc[r].to_numpy(dtype=float))[mask]
            pooled_ens_parts[r].append(r_elev)
            ens_stats_list.append(_metrics_one(obs_elev, r_elev))

        if len(ens_stats_list) > 0:
            ens_df = pd.DataFrame(ens_stats_list)
            well_row = {
                "wellid": wid,
                "n_obs": int(mask.sum()),
                "base_R2": base_stats["R2"],
                "base_RMSE": base_stats["RMSE"],
                "base_PBIAS": base_stats["PBIAS"],
                "ens_min_R2":  float(ens_df["R2"].min()),
                "ens_mean_R2": float(ens_df["R2"].mean()),
                "ens_max_R2":  float(ens_df["R2"].max()),
                "ens_min_RMSE":  float(ens_df["RMSE"].min()),
                "ens_mean_RMSE": float(ens_df["RMSE"].mean()),
                "ens_max_RMSE":  float(ens_df["RMSE"].max()),
                "ens_min_PBIAS":  float(ens_df["PBIAS"].min()),
                "ens_mean_PBIAS": float(ens_df["PBIAS"].mean()),
                "ens_max_PBIAS":  float(ens_df["PBIAS"].max()),
                "n_ens": int(len(ens_stats_list)),
            }
        else:
            well_row = {
                "wellid": wid,
                "n_obs": int(mask.sum()),
                "base_R2": base_stats["R2"],
                "base_RMSE": base_stats["RMSE"],
                "base_PBIAS": base_stats["PBIAS"],
                "ens_min_R2": np.nan, "ens_mean_R2": np.nan, "ens_max_R2": np.nan,
                "ens_min_RMSE": np.nan, "ens_mean_RMSE": np.nan, "ens_max_RMSE": np.nan,
                "ens_min_PBIAS": np.nan, "ens_mean_PBIAS": np.nan, "ens_max_PBIAS": np.nan,
                "n_ens": 0,
            }

        per_well_rows.append(well_row)

    # ----------------------------
    # Overall pooled metrics
    # ----------------------------
    if len(pooled_obs_parts) == 0 or len(pooled_base_parts) == 0:
        overall = {
            "n_obs": 0,
            "n_ens": int(len(ens_names)),
            "base_R2": np.nan, "base_RMSE": np.nan, "base_PBIAS": np.nan,
            "ens_min_R2": np.nan, "ens_mean_R2": np.nan, "ens_max_R2": np.nan,
            "ens_min_RMSE": np.nan, "ens_mean_RMSE": np.nan, "ens_max_RMSE": np.nan,
            "ens_min_PBIAS": np.nan, "ens_mean_PBIAS": np.nan, "ens_max_PBIAS": np.nan,
        }
        return pd.DataFrame([overall]), pd.DataFrame(per_well_rows)

    obs_all  = np.concatenate(pooled_obs_parts)
    base_all = np.concatenate(pooled_base_parts)

    base_m = _metrics_one(obs_all, base_all)

    # ensemble pooled
    ens_metrics = []
    ens_used = 0
    for r, parts in pooled_ens_parts.items():
        if len(parts) == 0:
            continue
        sim_all = np.concatenate(parts)
        ens_metrics.append(_metrics_one(obs_all, sim_all))
        ens_used += 1

    if ens_used > 0:
        em = pd.DataFrame(ens_metrics)
        overall = {
            "n_obs": int(obs_all.size),
            "n_ens": int(ens_used),
            "base_R2": base_m["R2"],
            "base_RMSE": base_m["RMSE"],
            "base_PBIAS": base_m["PBIAS"],
            "ens_min_R2":  float(em["R2"].min()),
            "ens_mean_R2": float(em["R2"].mean()),
            "ens_max_R2":  float(em["R2"].max()),
            "ens_min_RMSE":  float(em["RMSE"].min()),
            "ens_mean_RMSE": float(em["RMSE"].mean()),
            "ens_max_RMSE":  float(em["RMSE"].max()),
            "ens_min_PBIAS":  float(em["PBIAS"].min()),
            "ens_mean_PBIAS": float(em["PBIAS"].mean()),
            "ens_max_PBIAS":  float(em["PBIAS"].max()),
        }
    else:
        overall = {
            "n_obs": int(obs_all.size),
            "n_ens": 0,
            "base_R2": base_m["R2"],
            "base_RMSE": base_m["RMSE"],
            "base_PBIAS": base_m["PBIAS"],
            "ens_min_R2": np.nan, "ens_mean_R2": np.nan, "ens_max_R2": np.nan,
            "ens_min_RMSE": np.nan, "ens_mean_RMSE": np.nan, "ens_max_RMSE": np.nan,
            "ens_min_PBIAS": np.nan, "ens_mean_PBIAS": np.nan, "ens_max_PBIAS": np.nan,
        }

    overall_df = pd.DataFrame([overall])
    per_well_df = pd.DataFrame(per_well_rows).sort_values("wellid")

    return overall_df, per_well_df

overall_heads_df, perwell_heads_df = head_metrics_avg_plus_diff(
    head_obs=head_obs,
    run_results=run_results,
    use_weighted_only=True,   # recommended
)

print(overall_heads_df)
print(perwell_heads_df.head())

overall_heads_df.to_csv(out_dir / f"heads_metrics_overall_{obs_file.stem}.csv", index=False)
perwell_heads_df.to_csv(out_dir / f"heads_metrics_perwell_{obs_file.stem}.csv", index=False)
