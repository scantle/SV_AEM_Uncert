import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

data_dir = Path('01_Data/')
out_dir  = data_dir
pest_dir = Path("04_PEST_setup")   # TPL, INS

streams = ['FJ','SCK','AS','BY']
stream_files = ['FJ (USGS 11519500) Daily Flow, 1990-10-01_2025-08-31.csv',
                'SCK_F25484_Stream_Flow_Rate_Daily_Means_09292025.csv',
                'Scott River Above Serpa Lane.txt',
                'Scott River Below Youngs Dam.txt']

stream_files = [data_dir / f for f in stream_files]

fj_field_meas_file = data_dir / 'FJ_field_measurements.txt'

# FJ Flow Classifications (following Tolley et al. 2019)
# m3/d, numbers represent highest flow in category
fj_q_classes = {
    'low': 2.44E5,
    'medium': 2.44E6,
}
# To be determined in code for other streams

# Uncertainty values by flow regime (McMillan et al., 2012) as used by Martin & White (2023)
q_uncertainty = {
    'low': 1.0,
    'inbank': 0.2,
    'out-of-bank': 0.4,
}

USGS_field_meas_cols = ['agency_cd', 'site_no', 'measurement_nu', 'measurement_dt', 'tz_cd', 'q_meas_used_fg',
                        'party_nm', 'site_visit_coll_agency_cd', 'gage_height_va', 'discharge_va',
                        'measured_rating_diff', 'gage_va_change', 'gage_va_time', 'control_type_cd', 'discharge_cd',
                        'chan_nu', 'chan_name', 'meas_type', 'streamflow_method', 'velocity_method', 'chan_discharge',
                        'chan_width', 'chan_area', 'chan_velocity', 'chan_stability', 'chan_material', 'chan_evenness',
                        'long_vel_desc', 'horz_vel_desc', 'vert_vel_desc', 'chan_loc_cd', 'chan_loc_dist']

cfs_to_m3d = (0.3048)**3 * 86400

np.random.seed(667)

origin_date = pd.to_datetime('1990-9-30')

#----------------------------------------------------------------------------------------------------------------------#
# Classes/Functions
#----------------------------------------------------------------------------------------------------------------------#

class StreamflowUncertainty:
    """
    Implements Martin & White (2023) 'expected uncertainty envelope' for discharge:
      - Determine flow regime per day (either via FDC thresholds per M&W, or user thresholds)
      - For each day, perturb observed discharge d_i using regime-specific Pf and a random variate:
            x_{i,r} = d_i + z_{i,r} * Pf * d*_i
        where z_{i,r} ~ N(0,1) for 'Good' quality; z_{i,r} ~ N(1, sqrt(2)) for 'Fair'/'Poor' quality,
        and d*_i = max(d_i, low_threshold) for "low" regime; otherwise d*_i = d_i.
      - Compute RMSE_i across R realizations; use as SD for PEST-IES (linear space).
      - (Optional) Aggregate to monthly and/or convert to log-space SD via delta method.
    """

    def __init__(self,
                 df_daily,                         # DataFrame with data (date_col, flow_col, quality_col)
                 q_uncertainty,                    # dict: {'low':1.0,'inbank':0.2,'out-of-bank':0.4}
                 n_realizations=1000,
                 regime_method="fdc",              # "fdc" (M&W) or "thresholds"
                 thresholds=None,                  # dict with 'low' and 'medium' in m3/d if regime_method="thresholds"
                 quality_col='measured_rating_diff',
                 date_col='Date',
                 flow_col='Flow_m3d',
                 oob_quantile=0.98,              # for FDC; set to None to disable out-of-bank
                 low_scale_floor=None            # None (no substitution), 'mdf', or numeric (m3/d)
                 ):
        self.df = df_daily[[date_col, flow_col, quality_col]].copy()
        self.df.rename(columns={date_col:'date', flow_col:'q', quality_col:'qual'}, inplace=True)
        self.q_unc = q_uncertainty
        self.nR = int(n_realizations)
        self.regime_method = regime_method
        self.thresholds = thresholds or {}
        self.oob_quantile = oob_quantile
        self.low_scale_floor = low_scale_floor

        # Prepare
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Assign regimes
        if self.regime_method == "fdc":
            self._assign_regimes_fdc()
        elif self.regime_method == "thresholds":
            self._assign_regimes_thresholds()
        else:
            raise ValueError("regime_method must be 'fdc' or 'thresholds'.")

        # Map regime -> Pf (relative error fraction)
        self.df['Pf'] = self.df['regime'].map(self.q_unc).astype(float)

        # Map quality -> variate type (unbiased vs biased)
        # Expect values like 'Good','Fair','Poor' (case-insensitive). Unknown/NaN -> treat as Good (conservative)
        self.df['qual_norm'] = self.df['qual'].astype(str).str.strip().str.lower()
        self.df['is_biased'] = self.df['qual_norm'].isin(['fair','poor'])

        # Decide the floor used ONLY for scaling noise on 'low' days
        if self.low_scale_floor is None:
            # no substitution
            floor_value = None
        elif isinstance(self.low_scale_floor, str) and self.low_scale_floor.lower() == 'mdf':
            floor_value = self.low_thresh_mdf if self.low_thresh_mdf is not None else float(np.nanmean(self.df['q']))
        else:
            floor_value = float(self.low_scale_floor)

        if floor_value is None:
            self.df['q_scale'] = self.df['q'].to_numpy()
        else:
            q_arr = self.df['q'].to_numpy()
            self.df['q_scale'] = np.where(self.df['regime'] == 'low', np.maximum(q_arr, floor_value), q_arr)

    def _assign_regimes_fdc(self):
        q = self.df['q'].to_numpy()
        self.low_thresh_mdf = float(np.nanmean(q))

        if self.oob_quantile is None:
            # no out-of-bank class
            regime = np.where(q < self.low_thresh_mdf, 'low', 'inbank')
            self.oob_thresh_p02 = None
        else:
            self.oob_thresh_p02 = float(np.nanquantile(q, self.oob_quantile))
            regime = np.where(q > self.oob_thresh_p02, 'out-of-bank',
                              np.where(q < self.low_thresh_mdf, 'low', 'inbank'))
        self.df['regime'] = regime

    def _assign_regimes_thresholds(self):
        low = float(self.thresholds.get('low', np.nan))
        med = float(self.thresholds.get('medium', np.nan))
        hi = self.thresholds.get('high', None)  # optional

        q = self.df['q'].to_numpy()
        regime = np.full(len(q), 'inbank', dtype=object)
        regime[q < low] = 'low'

        if hi is not None:
            hi = float(hi)
            regime[q >= hi] = 'out-of-bank'  # only if 'high' provided

        self.df['regime'] = regime

        # for M&W-style low-floor default when asked for 'mdf'
        self.low_thresh_mdf = low if not np.isnan(low) else float(np.nanmean(q))
        self.oob_thresh_p02 = hi

    def simulate_rmse(self):
        """
        Run the Monte Carlo perturbations and compute RMSE_i for each day:
            x_{i,r} = d_i + z_{i,r} * Pf * d*_i
            RMSE_i = sqrt( mean_r (x_{i,r} - d_i)^2 )
        Returns a DataFrame with daily RMSE_i and metadata.
        """
        d = self.df['q'].to_numpy()                # observed discharge
        Pf = self.df['Pf'].to_numpy()              # regime percent (fraction)
        q_scale = self.df['q_scale'].to_numpy()    # d*_i scaling per M&W low-flow rule
        is_biased = self.df['is_biased'].to_numpy()

        n = len(d)
        RMSE = np.empty(n, dtype=float)

        # Vectorized generation by splitting biased/unbiased rows for efficiency
        idx_unbiased = np.where(~is_biased)[0]
        idx_biased   = np.where(is_biased)[0]

        # Prepare z-matrices (draws per subset to save memory)
        RMSE[:] = np.nan

        # Unbiased: z ~ N(0,1)
        if idx_unbiased.size:
            zU = np.random.normal(loc=0.0, scale=1.0, size=(idx_unbiased.size, self.nR))
            eU = zU * (Pf[idx_unbiased, None] * q_scale[idx_unbiased, None])
            # RMSE across realizations around di
            RMSE[idx_unbiased] = np.sqrt(np.mean(eU**2, axis=1))

        # Biased: z ~ N(1, sqrt(2))
        if idx_biased.size:
            zB = np.random.normal(loc=1.0, scale=np.sqrt(2.0), size=(idx_biased.size, self.nR))
            eB = zB * (Pf[idx_biased, None] * q_scale[idx_biased, None])
            RMSE[idx_biased] = np.sqrt(np.mean(eB**2, axis=1))

        out = self.df.copy()
        out['sd_lin'] = RMSE  # linear-space SD (this is what M&W pass to PEST-IES)
        return out

    @staticmethod
    def delta_method_log_sd(sd_lin, q, base=10):
        """
        Delta-method approximation for SD in log space when obs are log-transformed:
            Var[log Q] ≈ Var(Q) / E[Q]^2  ⇒  sd_log ≈ sd_lin / Q
        For base-10 logs: divide further by ln(10).
        """
        # avoid division by zero
        eps = np.finfo(float).eps
        if base == 10:
            return (sd_lin / np.maximum(q, eps)) / np.log(10.0)
        elif base is None:  # natural log
            return sd_lin / np.maximum(q, eps)
        else:
            raise ValueError("base must be 10 or None (for natural log).")

    def aggregate_rmse(self, freq='M', agg='sum', min_coverage=0.8, year_end='SEP'):
        """
        freq: 'M' (monthly) or 'A' (annual). If freq='A', year_end is the month name
              that ends the year, e.g. 'DEC' for calendar years, 'SEP' for water years.
        """
        import calendar
        df = self.df.copy().sort_values('date')

        # daily error scale s_i = Pf * q_scale
        s = (df['Pf'].to_numpy() * df['q_scale'].to_numpy())
        is_biased = df['is_biased'].to_numpy()

        # per-day error mean/variance (analytic)
        mu = np.zeros_like(s, dtype=float)
        var = np.empty_like(s, dtype=float)
        mu[~is_biased] = 0.0
        var[~is_biased] = s[~is_biased] ** 2
        mu[is_biased] = s[is_biased]
        var[is_biased] = 2.0 * s[is_biased] ** 2

        # ---- period keys (PeriodIndex) ----
        if freq.upper() == 'M':
            per = df['date'].dt.to_period('M')
        elif freq.upper() == 'Y':
            # e.g., 'Y-DEC' = calendar year, 'Y-SEP' = water year (Oct–Sep)
            per = df['date'].dt.to_period(f'Y-{year_end.upper()}')
        else:
            raise ValueError("freq must be 'M' or 'Y'.")

        # exact start/end timestamps for each period row
        per_start = per.dt.to_timestamp(how='start')
        per_end = per.dt.to_timestamp(how='end')
        # vectorized number of calendar days in each period row
        ndays_row = (per_end - per_start).dt.days + 1

        gdf = pd.DataFrame({
            'period': per,
            'q': df['q'].to_numpy(),
            'mu': mu,
            'var': var,
            'ndays_row': ndays_row.to_numpy()
        })

        out = (gdf
               .groupby('period')
               .agg(n_obs=('q', lambda x: int(np.isfinite(x).sum())),
                    obsval_sum=('q', 'sum'),
                    obsval_mean=('q', 'mean'),
                    mu_sum=('mu', 'sum'),
                    var_sum=('var', 'sum'),
                    # since all rows in a period share the same ndays_row value, 'max' is fine
                    ndays_calendar=('ndays_row', 'max'))
               .reset_index())

        # coverage fraction
        out['coverage'] = out['n_obs'] / out['ndays_calendar'].clip(lower=1)

        # choose obs value and RMSE for the aggregate
        if agg == 'sum':
            out['obsval'] = out['obsval_sum']
            out['obsstd'] = np.sqrt(out['mu_sum'] ** 2 + out['var_sum'])
        elif agg == 'mean':
            n = out['n_obs'].clip(lower=1)
            out['obsval'] = out['obsval_mean']
            out['obsstd'] = np.sqrt((out['mu_sum'] / n) ** 2 + (out['var_sum'] / (n ** 2)))
        else:
            raise ValueError("agg must be 'sum' or 'mean'.")

        # apply coverage filter
        out = out[out['coverage'] >= float(min_coverage)].copy()

        # tidy
        out = out[['period', 'n_obs', 'coverage', 'obsval', 'obsstd']].sort_values('period').reset_index(drop=True)
        return out

    def make_obs_table_aggregate(self, gauge_id: str,
                                 freq='M', agg='sum', min_coverage=0.8,
                                 name_suffix=None, label_which='end'):  # 'start' or 'end'
        ag = self.aggregate_rmse(freq=freq, agg=agg, min_coverage=min_coverage)

        # pick start/end of period for labeling
        how = 'end' if str(label_which).lower().startswith('e') else 'start'
        ts = ag['period'].dt.to_timestamp(how=how)

        if freq.upper() == 'M':
            labels = ts.dt.strftime('%Y-%m')
        else:  # 'A'
            labels = ts.dt.strftime('%Y')

        tag = []
        if name_suffix:
            tag.append(name_suffix)
        tag.append('VOL' if agg == 'sum' else 'MEAN')
        tag = '_'.join(tag)

        obsnme = [f"{gauge_id}_{lab}_{tag}" for lab in labels]

        out = pd.DataFrame({
            'obsnme': obsnme,
            'obsval': ag['obsval'].to_numpy(),
            'obsstd': ag['obsstd'].to_numpy(),
            'period': ag['period'].to_numpy(),  # still a Period
            'period_ts': ts.to_numpy(),  # Timestamp at start/end (handy)
            'coverage': ag['coverage'].to_numpy(),
            'n_days': ag['n_obs'].to_numpy()
        })
        return out

#----------------------------------------------------------------------------------------------------------------------#

def stat_plots(daily_sd):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # (A) Regime counts
    daily_sd['regime'].value_counts().reindex(['low', 'inbank', 'out-of-bank']).plot(kind='bar', ax=axes[0])
    axes[0].set_title('Regime counts')
    axes[0].set_ylabel('days')

    # (B) Biased vs unbiased split
    daily_sd['is_biased'].map({True: 'biased', False: 'unbiased'}).value_counts().plot(kind='pie', autopct='%1.0f%%',
                                                                                       ax=axes[1])
    axes[1].set_ylabel('')
    axes[1].set_title('Quality-driven perturbation type')

#----------------------------------------------------------------------------------------------------------------------#

def uncert_hydrograph(daily_sd, start=None, end=None, name="", freq="D", logy=False):
    """
    Plot hydrograph with ± sd_lin band, without visually interpolating across missing days.
    We reindex onto a complete daily range so gaps remain gaps (NaNs).
    """
    df = daily_sd.copy()
    df = df.sort_values('date').set_index('date')

    # Determine plotting window
    if start is None:
        start = df.index.min()
    else:
        start = pd.to_datetime(start)
    if end is None:
        end = df.index.max()
    else:
        end = pd.to_datetime(end)

    # Full daily index and reindex to insert NaNs on missing days
    full = pd.date_range(start, end, freq=freq)
    df = df.reindex(full)

    # Build mask where both q and sd are present
    q = df['q']
    sd = df['sd_lin']
    valid = q.notna() & sd.notna()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # Line breaks at NaNs automatically
    ax.plot(df.index, q, label='Q (m³/d)', linewidth=1)

    # Uncertainty band only where valid data exist (no interpolation)
    ax.fill_between(df.index, (q - sd), (q + sd),
                    where=valid, alpha=0.3, label='± sd_lin', interpolate=False)

    if logy:
        ax.set_yscale('log')

    ax.set_title(f'{name} Streamflow ± Std. Dev.')
    ax.set_xlim(start, end)
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('m³/d')
    fig.tight_layout()

#----------------------------------------------------------------------------------------------------------------------#

def plot_low_flow_diagnostics(df, date_col="Date", flow_col="Flow", low_thresh=None, logy=True):
    """
    Plot two panels:
      1. Violin plot of flow distribution with a red horizontal line at low_thresh
      2. Hydrograph with flow below low_thresh plotted in red, others in blue
    """
    if low_thresh is None:
        raise ValueError("You must provide low_thresh (in same units as flow_col).")

    # Make figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Panel 1: Violin plot
    sns.violinplot(y=df[flow_col], ax=axes[0], inner="quartile", color="lightgray")
    axes[0].axhline(low_thresh, color="red", linestyle="--", label=f"low_thresh={low_thresh:.2f}")
    axes[0].set_title("Flow distribution")
    axes[0].set_ylabel(flow_col)
    axes[0].legend()

    if logy:
        axes[0].set_yscale("log")

    # Panel 2: Hydrograph
    below = df[df[flow_col] < low_thresh]
    above = df[df[flow_col] >= low_thresh]

    axes[1].plot(df[date_col], df[flow_col], color="lightgray", linewidth=1)
    axes[1].scatter(above[date_col], above[flow_col], s=10, c="blue", label=">= low_thresh")
    axes[1].scatter(below[date_col], below[flow_col], s=10, c="red", label="< low_thresh")

    axes[1].axhline(low_thresh, color="red", linestyle="--")
    axes[1].set_title("Hydrograph with low flows highlighted")
    axes[1].set_ylabel(flow_col)
    axes[1].set_xlabel(date_col)
    if logy:
        axes[1].set_yscale("log")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------------------------------------------------#

def make_obs_table(
    sd_df: pd.DataFrame,
    gauge_id: str,
    transform: str = "log10",   # "log10" or "linear"
    freq: str = "D",            # "D" (daily) or "M" (monthly)
    date_col: str = "date",
    q_col: str = "q",
    sd_lin_col: str = "sd_lin",
    sd_log10_col: str = "sd_log10",
    q_floor: float = 1e-12,     # protects logs
) -> pd.DataFrame:
    """
    Build an observation table ready to export:
      columns: obsnme, obsval, obsstd
    - transform = 'linear' → obsval = q,   obsstd = sd_lin
    - transform = 'log10'  → obsval = log10(q), obsstd = sd_log10
    - freq = 'D' uses daily rows already in sd_df
    - freq = 'M' aggregates to calendar months (mean obsval; RMS obsstd)
    """
    df = sd_df.copy()
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in dataframe.")
    if q_col not in df.columns or sd_lin_col not in df.columns:
        raise ValueError(f"Expected '{q_col}' and '{sd_lin_col}' in dataframe.")
    if transform == "log10" and sd_log10_col not in df.columns:
        raise ValueError(f"transform='log10' requires '{sd_log10_col}' column. Compute it first.")

    # choose value + sd columns
    if transform == "linear":
        val = df[q_col].astype(float)
        sd  = df[sd_lin_col].astype(float)
    elif transform == "log10":
        # protect logs (should already be OK, but guard anyway)
        val = np.log10(np.maximum(df[q_col].astype(float).to_numpy(), q_floor))
        sd  = df[sd_log10_col].astype(float).to_numpy()
    else:
        raise ValueError("transform must be 'linear' or 'log10'.")

    base = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "obsval": val,
        "obsstd": sd
    })

    # drop NA rows (missing flow or sd)
    base = base.dropna(subset=["obsval", "obsstd"])

    if freq.upper() == "M":
        # monthly aggregation: mean of obsval; RMS of obsstd over the month
        base["month"] = base["date"].values
        grp = base.groupby("month").agg(
            obsval=("obsval", "mean"),
            obsstd=("obsstd", lambda x: float(np.sqrt(np.mean(np.square(x)))))
        ).reset_index().rename(columns={"month": "date"})
        out = grp
        # name like FJ_2000-10
        out["obsnme"] = [f"{gauge_id}_{d.strftime('%Y-%m')}" for d in out["date"]]
    elif freq.upper() == "D":
        out = base.copy()
        # name like FJ_20001031
        out["obsnme"] = [f"{gauge_id}_{d.strftime('%Y%m%d')}" for d in out["date"]]
    else:
        raise ValueError("freq must be 'D' or 'M'.")

    # final order + types
    out = out[["obsnme", "obsval", "obsstd", "date"]].sort_values("date").reset_index(drop=True)
    return out

#----------------------------------------------------------------------------------------------------------------------#

def write_ts_ins_file(obs_df, origin_date, skip_rows, ins_filename, column_str=None, markers=None, date_col="date"):
    """
    Writes a PEST instruction (ins) file for streamflow observations with optimized skipping.

    Parameters:
    obs_df (pd.DataFrame): DataFrame containing columns ['Date', 'obsnme']
    origin_date (pd.Timestamp): Model start date
    ins_filename (str or Path): Name of the output instruction file
    """
    obs_df = obs_df.sort_values(by=date_col).reset_index(drop=True)

    # Open file
    with open(ins_filename, 'w') as f:
        f.write("pif @\n")  # PEST instruction file header

        current_date = origin_date + pd.DateOffset(days=1)
        for i, row in obs_df.iterrows():
            days_skipped = (row[date_col] - current_date).days
            if i == 0:
                days_skipped += skip_rows
            if column_str is not None:
                f.write(f"l{days_skipped + 1} [{row['obsnme']}]{column_str}\n")
            elif markers is not None:
                f.write(f"l{days_skipped + 1} {markers} !{row['obsnme']}!\n")
            else:
                raise ValueError("Must pass either markers or column_str")
            current_date = row[date_col] + pd.Timedelta(days=1)

    print(f"Instruction file written: {ins_filename}")

#----------------------------------------------------------------------------------------------------------------------#

def write_static_ins_file(obs_df, ins_filename, markers):
    # Open file
    with open(ins_filename, 'w') as f:
        f.write("pif @\n")  # PEST instruction file header
        for i, row in obs_df.iterrows():
            f.write(f"l1 {markers} !{row['obsnme']}!\n")
    print(f"Instruction file written: {ins_filename}")

#----------------------------------------------------------------------------------------------------------------------#
# Fort Jones
#----------------------------------------------------------------------------------------------------------------------#

# Read data
q_df = pd.read_csv(stream_files[0], parse_dates=['Date'])
field_meas_df = pd.read_table(fj_field_meas_file, skiprows=16, names=USGS_field_meas_cols, parse_dates=['measurement_dt'])

# Drop field measurements with no rating
field_meas_df = field_meas_df.dropna(axis=0, subset=['measured_rating_diff'])

# Convert to m3/d
q_df['Flow_m3d'] = q_df['Flow'] * cfs_to_m3d

# Get closest streamflow measurement
q_df = q_df.sort_values('Date')
fmd = field_meas_df[['measurement_dt','measured_rating_diff']].sort_values('measurement_dt')
q_df = pd.merge_asof(q_df, fmd, left_on='Date', right_on='measurement_dt', direction='nearest')

# Build model
sim = StreamflowUncertainty(
    df_daily=q_df,
    q_uncertainty=q_uncertainty,
    n_realizations=1000,
    regime_method="thresholds",
    thresholds=fj_q_classes,
    oob_quantile=None,
    low_scale_floor=None
)

# Run sim
fj_daily_sd = sim.simulate_rmse()

# Get log-space values for use in PEST-IES
fj_daily_sd['sd_log10'] = StreamflowUncertainty.delta_method_log_sd(fj_daily_sd['sd_lin'].to_numpy(),
                                                                 fj_daily_sd['q'].to_numpy(),
                                                                 base=10)

# Plots
stat_plots(fj_daily_sd)
uncert_hydrograph(fj_daily_sd, start='10/01/2020', end='10/01/2025', name='Fort Jones')

fj_obs = make_obs_table(sd_df=fj_daily_sd, gauge_id=streams[0], transform="log10")

# Monthly volumes (sum of daily m³/d → m³/month) with ≥80% day coverage
fj_month_vol = sim.make_obs_table_aggregate(gauge_id="FJ", freq='M', agg='sum')

# Annual volumes
fj_year_vol = sim.make_obs_table_aggregate(gauge_id="FJ", freq='Y', agg='sum')

#----------------------------------------------------------------------------------------------------------------------#
# Shackleford (SCK)
#----------------------------------------------------------------------------------------------------------------------#

# Read data
sck_df = pd.read_csv(stream_files[1], names=['Date','Flow','quality'], parse_dates=[0], skiprows=9)

# Reclassify quality to Good/Fair/Poor (1,2 = Good, 151,252 = no data, 55 = extrapolated, 70 = estimated)
sck_df['code'] = sck_df['quality'].str.extract(r'^(\d+)').astype(int)[0]
sck_df = sck_df[~sck_df.code.isin([151,255])]
sck_df['quality_rev'] = 'Good'
sck_df.loc[sck_df.code==55,'quality_rev'] = 'Fair'  # Extrapolated as Fair (biased distribution)
sck_df.loc[sck_df.code==70,'quality_rev'] = 'Poor'  # Estimated as Poor (biased distribution)

# Convert to m3/d
sck_df['Flow_m3d'] = sck_df['Flow'].astype(float) * cfs_to_m3d

# Determine thresholds
sck_low_cut = sck_df['Flow_m3d'].quantile(0.33)
plot_low_flow_diagnostics(sck_df, date_col="Date", flow_col="Flow_m3d", low_thresh=sck_low_cut, logy=False)
sck_q_classes = {'low': sck_low_cut, 'medium': sck_df['Flow_m3d'].quantile(0.66)}

# Build model
sck_sim = StreamflowUncertainty(
    df_daily=sck_df,
    q_uncertainty=q_uncertainty,
    n_realizations=1000,
    regime_method="thresholds",
    quality_col='quality_rev',
    thresholds=sck_q_classes,
    oob_quantile=None,
    low_scale_floor=None
)

# Run sim
sck_daily_sd = sck_sim.simulate_rmse()

# Get log-space values for use in PEST-IES
sck_daily_sd['sd_log10'] = StreamflowUncertainty.delta_method_log_sd(sck_daily_sd['sd_lin'].to_numpy(),
                                                                 sck_daily_sd['q'].to_numpy(),
                                                                 base=10)

# Plots
stat_plots(sck_daily_sd)
uncert_hydrograph(sck_daily_sd, start='10/01/2016', end='10/01/2018', name='Shackleford')

sck_obs = make_obs_table(sd_df=sck_daily_sd, gauge_id=streams[1], transform="log10")

#----------------------------------------------------------------------------------------------------------------------#
# Above Serpa Lane (AS)
#----------------------------------------------------------------------------------------------------------------------#

# Read data
as_df = pd.read_table(stream_files[2], sep='\s+', names=['Date','model_time','Flow','Flow_m3d'], parse_dates=[0], skiprows=1)

as_df['quality'] = 'Good'  # No data for quality, assume all Good

# Determine thresholds
as_low_cut = as_df['Flow_m3d'].quantile(0.4)
plot_low_flow_diagnostics(as_df, date_col="Date", flow_col="Flow_m3d", low_thresh=as_low_cut, logy=False)
as_q_classes = {'low': as_low_cut, 'medium': as_df['Flow_m3d'].quantile(0.66)}

# Build model
as_sim = StreamflowUncertainty(
    df_daily=as_df,
    q_uncertainty=q_uncertainty,
    n_realizations=1000,
    regime_method="thresholds",
    quality_col='quality',
    thresholds=as_q_classes,
    oob_quantile=None,
    low_scale_floor=None
)

# Run sim
as_daily_sd = as_sim.simulate_rmse()

# Get log-space values for use in PEST-IES
as_daily_sd['sd_log10'] = StreamflowUncertainty.delta_method_log_sd(as_daily_sd['sd_lin'].to_numpy(),
                                                                 as_daily_sd['q'].to_numpy(),
                                                                 base=10)

# Plots
stat_plots(as_daily_sd)
uncert_hydrograph(as_daily_sd, name='Above Serpa Lane')

as_obs = make_obs_table(sd_df=as_daily_sd, gauge_id=streams[2], transform="log10")

#----------------------------------------------------------------------------------------------------------------------#
# Below Young's Dam (BY)
#----------------------------------------------------------------------------------------------------------------------#

# Read data
by_df = pd.read_table(stream_files[3], sep='\s+', names=['Date','model_time','Flow','Flow_m3d'], parse_dates=[0], skiprows=1)

by_df['quality'] = 'Good'  # No data for quality, assume all Good

# Determine thresholds
by_low_cut = by_df['Flow_m3d'].quantile(0.4)
plot_low_flow_diagnostics(by_df, date_col="Date", flow_col="Flow_m3d", low_thresh=by_low_cut, logy=False)
by_q_classes = {'low': by_low_cut, 'medium': by_df['Flow_m3d'].quantile(0.66)}

# Build model
by_sim = StreamflowUncertainty(
    df_daily=by_df,
    q_uncertainty=q_uncertainty,
    n_realizations=1000,
    regime_method="thresholds",
    quality_col='quality',
    thresholds=by_q_classes,
    oob_quantile=None,
    low_scale_floor=None
)

# Run sim
by_daily_sd = by_sim.simulate_rmse()

# Get log-space values for use in PEST-IES
by_daily_sd['sd_log10'] = StreamflowUncertainty.delta_method_log_sd(by_daily_sd['sd_lin'].to_numpy(),
                                                                 by_daily_sd['q'].to_numpy(),
                                                                 base=10)

# Plots
stat_plots(by_daily_sd)
uncert_hydrograph(by_daily_sd, name="Below Young's Dam")

by_obs = make_obs_table(sd_df=by_daily_sd, gauge_id=streams[3], transform="log10")

#----------------------------------------------------------------------------------------------------------------------#
# Combine obs tables, write output file
#----------------------------------------------------------------------------------------------------------------------#
cols = ['obsnme', 'obsval', 'obsstd']
all_obs = pd.concat([fj_obs[cols], fj_month_vol[cols], fj_year_vol[cols], sck_obs[cols], as_obs[cols], by_obs[cols]], axis=0)

all_obs.to_csv(out_dir / 'streamflow_obs_std.csv', index=False)

# Write INS files
write_ts_ins_file(fj_obs, origin_date,0, pest_dir /'Streamflow_FJ_SVIHM_MidptFlow_LOG.ins', markers='w')
write_ts_ins_file(sck_obs,origin_date,0, pest_dir / 'Streamflow_SCK_SVIHM_MidptFlow_LOG.ins', markers='w')
write_ts_ins_file(as_obs, origin_date,0, pest_dir /'Streamflow_AS_SVIHM_MidptFlow_LOG.ins', markers='w')
write_ts_ins_file(by_obs, origin_date,0, pest_dir /'Streamflow_BY_SVIHM_MidptFlow_LOG.ins', markers='w')

# Crude INS metrics addition
for stream in streams:
    with open(pest_dir / f'Streamflow_{stream}_SVIHM_MidptFlow_LOG.ins', 'a') as f:
        f.write(f'l1 w !{stream}_NSE!\n')
        f.write(f'l1 w !{stream}_KGE!\n')
        f.write(f'l1 w !{stream}_RMSE!\n')

# Write obs files so we can have per-stream stats as observation targets
# fj_obs.to_csv(out_dir / 'FJ_log.csv', index=False)
# sck_obs.to_csv(out_dir / 'SCK_log.csv', index=False)
# as_obs.to_csv(out_dir / 'AS_log.csv', index=False)
# by_obs.to_csv(out_dir / 'BY_log.csv', index=False)

# Write VOLUME INS files
fj_vol = pd.concat([fj_year_vol[['obsnme', 'obsval',]],
                    fj_month_vol[['obsnme', 'obsval',]]]).reset_index(drop=True)
write_static_ins_file(fj_vol, pest_dir / 'Streamflow_FJ_SVIHM_VOL.ins', markers='w')