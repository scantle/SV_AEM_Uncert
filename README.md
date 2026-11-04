# SV_AEM_Uncert

Calibration and uncertainty analysis of the Scott Valley Integrated Hydrologic Model (SVIHM) using a
parameterized texture model built from airborne electromagnetic (AEM) resistivity and lithologic logs.
Resistivity is kriged and perturbed with pilot point multipliers, converted to texture probabilities
with a spatially varying resistivity-texture transform, and power-law averaged to hydraulic properties
(Texture2Par). Everything is calibrated together with PEST-IES.

The model itself (SWBM + MODFLOW-NWT) is maintained in the [SVIHM repository](https://github.com/UCDavisHydro/SVIHM).
This repository holds the calibration setup, workflow scripts, and analysis.

## Layout

| Folder | Contents |
|---|---|
| `01_Data/` | Input data (AEM, lithology, streamflow, GIS) and hand-made PEST templates (`manualTPLs/`) |
| `01_Data/derived/` | Files written by the workflow scripts (not tracked) |
| `02_Models/` | Executables (`Bin/`), forward model template (`SVIHM_MF_working/`), kriging cross-validation setup (`r2p_cv/`) |
| `03_Scripts/` | Numbered workflow scripts, forward run utilities, analysis scripts |
| `04_PEST_setup/` | PEST build staging. Templates/instruction files are generated; only the batch files are tracked |
| `05_Plots/`, `06_Outputs/` | Plots and PEST outputs (not tracked) |

## Setup

1. Create the conda environment:
   ```
   conda env create -f conda_env.yml
   conda activate SV_AEM_T2P
   ```
2. Copy `local_paths_example.json` to `local_paths.json` and edit it for your machine:
   - `archive_dir`: folder holding `06_Outputs/` from previous calibrations (optional)
   - `runs_dir`: where PEST build folders and model run copies go
   - `pestpp_bin_dir`: folder with the PEST++ executables
   - `svihm_repo`: local clone of the SVIHM repository (defaults to `../SVIHM`)
3. Some well data is not included in this repository (see [Private well data](#private-well-data)).

## Paths

All scripts get their folders from `03_Scripts/project_paths.py`:

```python
import project_paths as paths

obs = pd.read_csv(paths.DERIVED_DIR / 'head_obs_master.csv')
f_dir = paths.run_dir('06_wtfx')   # looks in 06_Outputs/, then archive_dir/06_Outputs/
```

Paths are absolute (built from the repo root), so scripts can be run from any working directory.
Machine-specific locations come from `local_paths.json`, which is not tracked. Add new
locations to `project_paths.py` rather than hard-coding them in scripts.

The forward run utilities (`RES2PAR_preproc.py`, `STREAM_ADJUSTER.py`, `LOGSTRSIM.py`, `GAGE2VOL.py`,
`HOBPOSTPROC.py`) are copied into the model's `Bin/` folder and run inside each PEST worker, so they
use paths relative to the model folder instead.

## Workflow

Building the PEST setup:

| Script | Does |
|---|---|
| `01_Write_Initial_Sim_Files.py` | Lithology and AEM data to resistivity logs; fits resistivity-texture distributions |
| `02_Resistivity_Variography.py` | Resistivity variograms |
| `03_A_res_mult_pilot_point_placement.py` | Kriging uncertainty multiplier pilot points (denser near observation wells) |
| `04_Write_Initial_PilotPoint_Files.py` | Pilot point templates, factor files, initial values |
| `05_A_Streamflow_Error_Models.py` | Streamflow observations, error models, instruction files |
| `05_B_build_HOB_ins_n_obs.py` | Head observations (well means, deviations, vertical differences) |
| `Build_Localization_Matrix.py` | Localization matrix for PEST-IES |
| `06_Setup_PEST-iES_Control_File.py` | Assembles the PEST build folder, prior covariance, prior ensemble and control file |

`03_B_KVME_pilot_point_placement.py` places pilot points for the KVME pilot point method
(`kvme_pp_flag` in script 06). This method is not used, has not been tested, and is not recommended.

Forward run (`02_Models/SVIHM_MF_working/svihm.bat`): par2par → `RES2PAR_preproc.py` → Texture2Par →
`STREAM_ADJUSTER.py` → SWBM → SFR2PAR → MODFLOW-NWT → post-processors.

Analysis of a finished run: scripts `07`–`17`, plus `big_fj_plot.py`, `par_shift_analysis.py`, and
`10_Residual_Analysis.py`. Script 10 writes optional observation weight multipliers
(`01_Data/derived/str_wmults.csv`) that `05_A` applies if present.

## Watershed model inputs

Tributary inflows and mountain front recharge come from a watershed model (PRMS or LSPC) and enter
the model only through:

- `SWBM/subwatershed_irrigation_inflows_raw.txt`, `SWBM/subwatershed_nonirrigation_inflows_raw.txt`
  (daily, m³/day, one column per tributary)
- `SWBM/monthly_MFR_by_catchment.txt`
- the multiplier templates `streamflow_multipliers.txt.tpl` and `catchment_mult.txt.tpl`

## Private well data

The groundwater level observations include data from private wells, so some well information is not
included in this repository. Scripts that need it will stop with an error saying which data is missing.
