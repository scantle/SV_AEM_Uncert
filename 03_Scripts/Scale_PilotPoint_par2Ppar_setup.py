"""
No longer being used
"""

import pandas as pd
import numpy as np
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

tpl_dir = Path('04_InputFiles/PEST')
data_dir  = Path('./01_Data/')
out_dir = Path('./06_Outputs')
texs = ['Fine', 'Mixed_Fine', 'Sand', 'Mixed_Coarse', 'Very_Coarse']
tex_dist_file = data_dir / 'lognorm_dist_clustered.par'
pp_file_formula = 'scale_pp_{tex}_L{lay}.dat.tpl'
pp_out_formula = 'scale_pp_{tex}_L{lay}.dat'
layers = 2
pp_colnames = ['locname', 'n', 'x', 'y', 'sym1', 'parname', 'sym2']
outfile = tpl_dir / 'scale_par2par.tpl'
out_par_mult_file = out_dir / 'scale_pp_par2par_par.csv'

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

# Read in initial values
tex_df = pd.read_table(tex_dist_file, sep=r'\s+', skiprows=1, index_col=0)

# Read in ppfiles
pars = {}
file_connections = []
for tex in texs:
    for k in range(0,layers):
        pp_in = pp_file_formula.format(tex=tex, lay=k+1)
        pp_out = pp_out_formula.format(tex=tex, lay=k+1)
        ppdf = pd.read_table(tpl_dir/pp_in, skiprows=1, sep='\s+', names=pp_colnames)
        if k==0:
            pars[tex] = ppdf['parname'].to_list()
        else:
            pars[tex] = pars[tex] + ppdf['parname'].to_list()
        file_connections.append((pp_in, pp_out))

# Start writing par2par file
tex_mult_parnames = []
with open(outfile, 'w') as f:
    f.write('*parameter data\n')
    # Write each texture
    for i, tex in enumerate(texs):
        for j in range(0, len(pars[tex])):
            splits = pars[tex][j].replace(tex,'').replace('__','_').split('_')
            layer = int(splits[1][1]) + 1  # to 1-based indexing, for ease, and probably some later confusion
            parname = f'{splits[0]}_{tex}_{layer}_mult_{j+1}'
            default_value = 1.0
            if i==0:  # base
                f.write(f"{pars[tex][j]} = {tex_df.loc['Fine','Scale']} * ${parname}$\n")
            else:
                prevname = pars[tex][j].replace(tex,texs[i-1])
                f.write(f"{pars[tex][j]} = {prevname} * ${parname}$\n")
                default_value = tex_df.loc[tex,'Scale'] / tex_df.loc[texs[i-1],'Scale']
            tex_mult_parnames.append((parname, default_value))
    f.write('*template and model input files\n')
    for item in file_connections:
        f.write(f'{item[0]} {item[1]}\n')

# Write par2par multiplier names to output file
mults = pd.DataFrame(tex_mult_parnames, columns=['parname','value'])
mults.to_csv(out_par_mult_file, index=False)