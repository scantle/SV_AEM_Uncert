import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import flopy
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------------------------------------------------------- #
# Settings
# -------------------------------------------------------------------------------------------------------------------- #

# Base realization
run_base_dir = Path('/Users/leland/Documents/ModelRuns/2025_R2P_PEST_Calib_iter3/SVIHM/MODFLOW')

# Parent directory containing ensemble LST outputs
ens_dir = Path('./06_Outputs/06_wtfx/BUD_SFRBUD_ensemble/')

out_dir = Path('./05_Plots/modflow_budget_uncertainty')
out_dir.mkdir(parents=True, exist_ok=True)

mf_nam = 'SVIHM'
origin_date = pd.to_datetime('1990-09-30')
TAF_FACTOR = 1.0 / (1_000 * 1233.48183754752)
convert_to_TAF = True

# Choose uncertainty style:
#   "std"        -> symmetric +/- 1 std
#   "percentile" -> asymmetric interval from LOWER_PCTL to UPPER_PCTL
UNCERTAINTY_STYLE = "percentile"
LOWER_PCTL = 5.0
UPPER_PCTL = 95.0
YMIN = None
YMAX = None

# Same range as stacked plots:
# YMIN = -120
# YMAX = 120

# Show base realization as a short horizontal marker on each bar
SHOW_BASE_MARKER = True
BASE_MARKER_COLOR = 'k'
BASE_MARKER_LINEWIDTH = 1.2
BASE_MARKER_FRACTION = 0.55   # fraction of bar width occupied by marker

# Save individual CSV files with summary stats
WRITE_SUMMARY_CSV = True

plt.rcParams.update({
    "font.sans-serif": ["Merriweather", "Arial"],
})

# -------------------------------------------------------------------------------------------------------------------- #
# Colors / plotting order
# -------------------------------------------------------------------------------------------------------------------- #

C_RECHARGE = 'blue'
C_PUMPING  = 'green'
C_ET       = 'goldenrod'
C_STORAGE  = 'orangered'

color_map_mf = {
    "MFR/Ditch":      'cornflowerblue',
    "Recharge":       C_RECHARGE,
    "GW Pumping":     C_PUMPING,
    "Drains":         'darkorchid',
    "GW ET":          C_ET,
    "MF Storage":     C_STORAGE,
    "Stream Leakage": '#E6D81E',
}

mf_components = [
    "MF Storage",
    "Stream Leakage",
    "Recharge",
    "MFR/Ditch",
    "GW Pumping",
    "Drains",
    "GW ET",
]

# -------------------------------------------------------------------------------------------------------------------- #
# Functions
# -------------------------------------------------------------------------------------------------------------------- #

def load_modflow_annual_budget(file_path: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load and process annual MODFLOW budget from a run directory.
    Expects MODFLOW/<mf_nam>.lst under run_dir.
    Returns annual sums indexed by water year end (YE-SEP), in native volume units.
    """

    lst = flopy.utils.MfListBudget(file_path)
    mfdf, _ = lst.get_dataframes(start_datetime=start_date)
    if mfdf.shape[0] < 12754:  # undercooked
        raise EOFError('Incomplete MODFLOW budget file.')
    mfdf = mfdf.resample("YE-SEP").sum()

    # Derived columns
    mfdf['MF Storage'] = mfdf['STORAGE_IN'] - mfdf['STORAGE_OUT']
    mfdf['Stream Leakage'] = -1.0 * (mfdf['STREAM_LEAKAGE_OUT'] - mfdf['STREAM_LEAKAGE_IN'])

    # Rename columns
    mfdf = mfdf.rename(columns={
        'WELLS_IN': 'MFR/Ditch',
        'RECHARGE_IN': 'Recharge',
        'WELLS_OUT': 'GW Pumping',
        'DRNO_DRAINS_OUT': 'Drains',
        'ET_SEGMENTS_OUT': 'GW ET',
    })

    # Keep only desired columns
    keep_cols = [
        'MFR/Ditch', 'Recharge', 'GW Pumping', 'Drains',
        'GW ET', 'MF Storage', 'Stream Leakage'
    ]
    missing = [c for c in keep_cols if c not in mfdf.columns]
    if missing:
        raise ValueError(f"Missing expected MODFLOW budget columns in {file_path.name}: {missing}")

    mfdf = mfdf[keep_cols].copy()

    # Enforce sign convention for outflows
    for c in ['GW Pumping', 'Drains', 'GW ET']:
        mfdf[c] = -np.abs(mfdf[c].values)

    # Convert to TAF
    if convert_to_TAF:
        mfdf = mfdf * TAF_FACTOR

    return mfdf


def collect_ensemble_budgets(lst_files: list, start_date: pd.Timestamp):
    """
    Load annual MODFLOW budgets for all realization folders.
    Returns dict: {run_name: annual_budget_df}
    """
    budgets = []
    failed = []

    for i, f in tqdm(enumerate(lst_files), desc="Loading ensemble MODFLOW budgets", total=len(lst_files)):
        if f.stat().st_size < 152079900:  # first line of defense
            continue
        try:
            budgets.append(load_modflow_annual_budget(f, start_date))
        except Exception as e:
            failed.append((i, str(e)))

    if not budgets:
        raise RuntimeError("No ensemble budgets were loaded successfully.")

    if failed:
        print(f"\nWARNING: {len(failed)} runs failed to load.")
        for name, msg in failed[:15]:
            print(f"  {name}: {msg}")
        if len(failed) > 15:
            print("  ...")

    print(f"\nLoaded {len(budgets)}/{len(lst_files)} ensemble runs successfully.")
    return budgets


def summarize_component(ensemble_budgets: list[pd.DataFrame], component: str):
    """
    Build a summary dataframe for one component across an ensemble stored
    as a list of annual budget DataFrames.

    Returns
    -------
    summary : pd.DataFrame
        Index is annual dates, columns include:
          - mean
          - std
          - p05
          - p95
          - p25
          - p75
          - n
    comp_df : pd.DataFrame
        One column per realization (integer column labels).
    """
    if not ensemble_budgets:
        raise ValueError("ensemble_budgets is empty")

    # Union of all dates across realizations, then align
    all_dates = sorted(set().union(*[df.index for df in ensemble_budgets]))
    all_dates = pd.DatetimeIndex(all_dates)

    data = {}
    for i, df in enumerate(ensemble_budgets):
        if component not in df.columns:
            raise ValueError(f"Component '{component}' not found in realization {i}")
        data[i] = df[component].reindex(all_dates)

    comp_df = pd.DataFrame(data, index=all_dates)

    summary = pd.DataFrame(index=all_dates)
    summary['mean'] = comp_df.mean(axis=1, skipna=True)
    summary['std'] = comp_df.std(axis=1, skipna=True, ddof=1)
    summary['p05'] = comp_df.quantile(0.05, axis=1, interpolation='linear')
    summary['p95'] = comp_df.quantile(0.95, axis=1, interpolation='linear')
    summary['p25'] = comp_df.quantile(0.25, axis=1, interpolation='linear')
    summary['p75'] = comp_df.quantile(0.75, axis=1, interpolation='linear')
    summary['n'] = comp_df.notna().sum(axis=1)

    return summary, comp_df


def get_yerr_and_center(summary: pd.DataFrame):
    """
    Returns:
      center_vals: values to plot as bars
      yerr: either 1D symmetric or 2 x N asymmetric error array
    """
    center_vals = summary['mean'].values

    if UNCERTAINTY_STYLE == "std":
        yerr = summary['std'].fillna(0.0).values
    elif UNCERTAINTY_STYLE == "percentile":
        lo = summary['p05'].values
        hi = summary['p95'].values
        lower = center_vals - lo
        upper = hi - center_vals
        lower = np.where(np.isfinite(lower), np.maximum(lower, 0.0), 0.0)
        upper = np.where(np.isfinite(upper), np.maximum(upper, 0.0), 0.0)
        yerr = np.vstack([lower, upper])
    else:
        raise ValueError(f"Unknown UNCERTAINTY_STYLE: {UNCERTAINTY_STYLE}")

    return center_vals, yerr


def plot_component_bars(
    component: str,
    base_series: pd.Series,
    summary: pd.DataFrame,
    out_dir: Path,
    color_map: dict,
):
    """
    Create a single annual bar chart with uncertainty for one MODFLOW component.
    Bars = ensemble mean.
    Error bars = uncertainty interval.
    Optional horizontal marker = base realization value.
    """
    dates = summary.index.to_pydatetime()
    width = (dates[1] - dates[0]).days * 0.8 if len(dates) > 1 else 20

    center_vals, yerr = get_yerr_and_center(summary)
    base_vals = base_series.reindex(summary.index).values

    fig, ax = plt.subplots(figsize=(14, 4.8))

    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.6)

    # Main bars: ensemble mean
    ax.bar(
        dates,
        center_vals,
        width=width,
        color=color_map.get(component, "0.5"),
        edgecolor='k',
        linewidth=0.3,
        zorder=2,
    )

    # Error bars
    ax.errorbar(
        dates,
        center_vals,
        yerr=yerr,
        fmt='none',
        ecolor='k',
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=3,
    )

    # Base realization marker: short horizontal line segment across each bar
    if SHOW_BASE_MARKER:
        half_marker_width_days = 0.5 * width * BASE_MARKER_FRACTION
        for d, y in zip(dates, base_vals):
            if np.isfinite(y):
                x0 = mdates.date2num(d) - half_marker_width_days
                x1 = mdates.date2num(d) + half_marker_width_days
                ax.hlines(
                    y=y,
                    xmin=x0,
                    xmax=x1,
                    colors=BASE_MARKER_COLOR,
                    linewidth=BASE_MARKER_LINEWIDTH,
                    zorder=4,
                )

    ax.axhline(0, color='k', linewidth=1.0)
    if convert_to_TAF:
        ylabel = "Annual Volume (TAF)"
    else:
        ylabel = 'Annual Volume ($m^3$)'
    if UNCERTAINTY_STYLE == "std":
        subtitle = "bars = ensemble mean, error bars = ±1 SD"
    else:
        subtitle = f"bars = ensemble mean, error bars = {int(LOWER_PCTL)}–{int(UPPER_PCTL)}th percentile"

    if SHOW_BASE_MARKER:
        subtitle += ", black tick = base realization"

    ax.set_title(f"{component} — MODFLOW Annual Budget\n{subtitle}")
    ax.set_ylabel(ylabel)

    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    tdel = pd.to_timedelta(180, unit='d')
    ax.set_xlim(summary.index.min() - tdel, summary.index.max() + tdel)

    # Optional fixed y limits
    if YMIN is not None or YMAX is not None:
        ymin_cur, ymax_cur = ax.get_ylim()
        ax.set_ylim(
            YMIN if YMIN is not None else ymin_cur,
            YMAX if YMAX is not None else ymax_cur,
        )

    fig.tight_layout()

    safe_name = component.lower().replace("/", "_").replace(" ", "_")
    out_png = out_dir / f"modflow_budget_{safe_name}_annual_uncertainty.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    if WRITE_SUMMARY_CSV:
        out_csv = out_dir / f"modflow_budget_{safe_name}_annual_summary.csv"
        summary.to_csv(out_csv)

    return out_png


# -------------------------------------------------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------------------------------------------------- #

# Base realization
print("Loading base realization MODFLOW budget...")
mf_base = load_modflow_annual_budget(run_base_dir / 'SVIHM.lst', start_date=origin_date)

print("Loading ensemble realization MODFLOW budgets...")
lst_files = sorted(ens_dir.glob("ftx_*.lst"))

# Ensemble
ensemble_budgets = collect_ensemble_budgets(lst_files=lst_files, start_date=origin_date)

# Make plots
made_plots = []
for component in mf_components:
    print(f"Plotting: {component}")
    summary, comp_df = summarize_component(ensemble_budgets, component)
    out_png = plot_component_bars(
        component=component,
        base_series=mf_base[component],
        summary=summary,
        out_dir=out_dir,
        color_map=color_map_mf,
    )
    made_plots.append(out_png)

print("\nDone.")
print(f"Saved {len(made_plots)} plots to:\n  {out_dir}")