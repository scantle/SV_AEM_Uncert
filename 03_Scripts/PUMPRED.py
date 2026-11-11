import re
import argparse
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

# Pumping reduced by MF-NWT (WEL SPECIFY/PHIRAMP) as a fraction of requested extraction:
#   pmpr_total            whole run
#   pmpr_wyYYYY           basin, by water year
#   pmpr_lK_rRRR_cCCC     whole run, by ag well cell (fixed list from ag_well_summary.txt)

parser = argparse.ArgumentParser(description="Pumping reduction fractions from MF-NWT Reduced_Wells output.")
parser.add_argument("wel_file", type=str, help="MODFLOW WEL file (requested rates).")
parser.add_argument("reduced_file", type=str, help="Reduced wells file written by the WEL package.")
parser.add_argument("dis_file", type=str, help="MODFLOW DIS file (stress period lengths).")
parser.add_argument("well_summary", type=str, help="SWBM ag_well_summary.txt (layer, row_MF, col_MF).")
parser.add_argument("--origin", type=str, default="1990-10-01", help="Start date of SP 1.")
parser.add_argument("--out", type=str, default="Pumping_Reduction.out", help="Output file.")
parser.add_argument("--ins", type=str, default=None, help="Also write a PEST instruction file.")

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def read_periods(dis_file):
    lines = [l for l in open(dis_file) if not l.lstrip().startswith('#')]
    nper = int(lines[0].split()[3])
    sp = [l.split() for l in lines if re.match(r'^\s*[\d.Ee+-]+\s+\d+\s+[\d.Ee+-]+\s+(SS|TR)\s*$', l, re.I)]
    if len(sp) != nper:
        raise ValueError(f'Found {len(sp)} stress period lines in {dis_file}, expected {nper}')
    per = pd.DataFrame({'perlen': [float(s[0]) for s in sp], 'nstp': [int(s[1]) for s in sp],
                        'tsmult': [float(s[2]) for s in sp]}, index=np.arange(1, nper + 1))
    if (per['tsmult'] != 1.0).any():
        raise ValueError('Time step multipliers other than 1 are not supported')
    per['dt'] = per['perlen'] / per['nstp']
    return per

def water_years(per, origin):
    sp_start = pd.Timestamp(origin) + pd.to_timedelta(per['perlen'].cumsum().shift(fill_value=0), unit='D')
    return pd.Series(np.where(sp_start.dt.month >= 10, sp_start.dt.year + 1, sp_start.dt.year), index=per.index)

def well_cells(well_summary):
    wells = pd.read_csv(well_summary, sep=r'\s+')
    cells = wells[['layer', 'row_MF', 'col_MF']].drop_duplicates().sort_values(['layer', 'row_MF', 'col_MF'])
    return list(cells.itertuples(index=False, name=None))

def obs_names(dis_file, well_summary, origin='1990-10-01'):
    per = read_periods(dis_file)
    names = ['pmpr_total'] + [f'pmpr_wy{wy}' for wy in sorted(water_years(per, origin).unique())]
    names += [f'pmpr_l{k}_r{r:03d}_c{c:03d}' for k, r, c in well_cells(well_summary)]
    return names

def write_ins(names, ins_file):
    with open(ins_file, 'w') as f:
        f.write('pif ~\n')
        for nm in names:
            f.write(f'l1 w !{nm}!\n')

def read_wel(wel_file, nper):
    lines = open(wel_file).read().splitlines()
    i = 0
    while lines[i].lstrip().startswith('#'):
        i += 1
    i += 1                                   # MXACTW line
    if lines[i].strip().upper().startswith('SPECIFY'):
        i += 1
    rows, last = [], None
    for sp in range(1, nper + 1):
        n = int(lines[i].split()[0])
        if n >= 0:
            t = [l.split() for l in lines[i + 1:i + 1 + n]]
            last = pd.DataFrame({'k': [int(v[0]) for v in t], 'r': [int(v[1]) for v in t],
                                 'c': [int(v[2]) for v in t], 'q': [float(v[3]) for v in t]})
        if last is None:
            raise ValueError(f'WEL reuse flag in SP {sp} with no previous data')
        rows.append(last.assign(sp=sp))
        i += 1 + max(n, 0)
    return pd.concat(rows, ignore_index=True)

def read_reduced(reduced_file):
    rows, sp, ts = [], None, None
    with open(reduced_file) as f:
        for line in f:
            if 'STRESS PERIOD' in line:
                m = re.search(r'STRESS PERIOD\s+(\d+)\s+TIME STEP\s+(\d+)', line)
                sp, ts = int(m.group(1)), int(m.group(2))
                continue
            t = line.split()
            if len(t) >= 5 and t[0].isdigit() and sp is not None:
                rows.append((sp, ts, int(t[0]), int(t[1]), int(t[2]), float(t[3]), float(t[4])))
    return pd.DataFrame(rows, columns=['sp', 'ts', 'k', 'r', 'c', 'appl_q', 'act_q'])

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Communicate
    print("PUMPRED.py")

    # parse args
    args = parser.parse_args()

    # Stress periods and water years
    per = read_periods(args.dis_file)
    per['wy'] = water_years(per, args.origin)

    # Ag well cells (fixed obs list)
    cells = well_cells(args.well_summary)

    # Requested extraction volume (rates constant within each SP)
    wel = read_wel(args.wel_file, len(per))
    wel = wel[wel['q'] < 0].copy()
    wel['vol'] = -wel['q'] * per.loc[wel['sp'], 'perlen'].to_numpy()
    wel['wy'] = per.loc[wel['sp'], 'wy'].to_numpy()

    # Reduced volume, extraction only (a header-only file means no reductions)
    red = read_reduced(args.reduced_file)
    red = red[red['appl_q'] < 0].copy()
    red['vol'] = (red['act_q'] - red['appl_q']) * per.loc[red['sp'], 'dt'].to_numpy()
    red['wy'] = per.loc[red['sp'], 'wy'].to_numpy()

    # Fractions
    def frac(r, q):
        return float(r / q) if q > 0 else 0.0

    out = [('pmpr_total', frac(red['vol'].sum(), wel['vol'].sum()))]
    req_wy = wel.groupby('wy')['vol'].sum()
    red_wy = red.groupby('wy')['vol'].sum()
    for wy in sorted(per['wy'].unique()):
        out.append((f'pmpr_wy{wy}', frac(red_wy.get(wy, 0.0), req_wy.get(wy, 0.0))))
    req_c = wel.groupby(['k', 'r', 'c'])['vol'].sum()
    red_c = red.groupby(['k', 'r', 'c'])['vol'].sum()
    for k, r, c in cells:
        out.append((f'pmpr_l{k}_r{r:03d}_c{c:03d}', frac(red_c.get((k, r, c), 0.0), req_c.get((k, r, c), 0.0))))

    # Reductions in cells outside the ag well list are still in the totals; report them
    other = red_c.drop(index=[c for c in cells if c in red_c.index]).sum()
    if other > 0:
        print(f'Note: {other:.4g} m3 reduced in cells not in {args.well_summary}')

    # Write
    with open(args.out, 'w') as f:
        for nm, v in out:
            f.write(f'{nm:<24s}{v:20.10e}\n')
    print(f'Total reduction {out[0][1]:.2%} of requested extraction; wrote {len(out)} values to {args.out}')

    if args.ins:
        write_ins([nm for nm, v in out], args.ins)
