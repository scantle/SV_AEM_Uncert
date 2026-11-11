:: Python environment (set by run_ies.cmd, or the active pixi/conda env)
if not defined IES_PYTHON if defined CONDA_PREFIX set "IES_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined IES_PYTHON (
    echo ERROR: IES_PYTHON is not set. Start runs with start_ies.bat, or use pixi run -e worker.
    exit /b 2
)

:: Each step stops the run on failure, so files left from a previous run are never used

:: Run par2par
call Bin\par2par.exe t2p_par2par.in
if errorlevel 1 exit /b 1

:: Run preproc, T2P
cd preproc
"%IES_PYTHON%" ..\Bin\RES2PAR_preproc.py
if errorlevel 1 exit /b 1
call ..\Bin\Texture2Par.exe svihm.t2p
if errorlevel 1 exit /b 1

:: Adjust streamflow, Run SWBM
cd ..\SWBM
del subwatershed_nonirrigation_inflows.txt
del subwatershed_irrigation_inflows.txt
"%IES_PYTHON%" ..\Bin\STREAM_ADJUSTER.py
if errorlevel 1 exit /b 1
call ..\Bin\SWBM.exe svihm.swbm
if errorlevel 1 exit /b 1

:: Copy over new SWBM-generated MODFLOW files
cd ..\
xcopy SWBM\SVIHM.* MODFLOW /Y /I
xcopy SWBM\SVIHM_tabfile_seg*.tab MODFLOW /Y /I

:: Run SFR2Par
copy MODFLOW\SVIHM.sfr MODFLOW\SVIHM_intermediate.sfr /Y
del MODFLOW\SVIHM.sfr
call Bin\SFR2PAR.exe sfr2par.in
if errorlevel 1 exit /b 1

:: Run MODFLOW
cd MODFLOW
call ..\Bin\MODFLOW-NWT.exe SVIHM.nam
if errorlevel 1 exit /b 1

:: Run python post processors
"%IES_PYTHON%" ..\Bin\LOGSTRSIM.py Streamflow_FJ_SVIHM.dat MidptFlow --obs-csv FJ_log.csv --blocks FJ_log7d.csv
if errorlevel 1 exit /b 1
"%IES_PYTHON%" ..\Bin\LOGSTRSIM.py Streamflow_AS_SVIHM.dat MidptFlow --obs-csv AS_log.csv --blocks AS_log7d.csv
if errorlevel 1 exit /b 1
"%IES_PYTHON%" ..\Bin\LOGSTRSIM.py Streamflow_BY_SVIHM.dat MidptFlow --obs-csv BY_log.csv --blocks BY_log7d.csv
if errorlevel 1 exit /b 1
"%IES_PYTHON%" ..\Bin\LOGSTRSIM.py Streamflow_SCK_SVIHM.dat MidptFlow --obs-csv SCK_log.csv --blocks SCK_log7d.csv
if errorlevel 1 exit /b 1
"%IES_PYTHON%" ..\Bin\GAGE2VOL.py Streamflow_FJ_SVIHM.dat 1990-09-30 5
if errorlevel 1 exit /b 1
"%IES_PYTHON%" ..\Bin\HOBPOSTPROC.py HobData_SVIHM.dat head_obs_master.csv
if errorlevel 1 exit /b 1
"%IES_PYTHON%" ..\Bin\PUMPRED.py SVIHM.wel Reduced_Wells.dat SVIHM.dis ..\SWBM\ag_well_summary.txt --out Pumping_Reduction.out
if errorlevel 1 exit /b 1

cd ..\
