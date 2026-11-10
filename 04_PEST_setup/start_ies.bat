@echo off
setlocal
:: Starts PEST-IES from a PEST build folder: checks pixi, installs the worker environment, prepares a batch
set "IES_PIXI="
for %%I in (pixi.exe) do set "IES_PIXI=%%~$PATH:I"
if not defined IES_PIXI if exist "%LOCALAPPDATA%\pixi\bin\pixi.exe" set "IES_PIXI=%LOCALAPPDATA%\pixi\bin\pixi.exe"
if not defined IES_PIXI if exist "%USERPROFILE%\.pixi\bin\pixi.exe" set "IES_PIXI=%USERPROFILE%\.pixi\bin\pixi.exe"
if not defined IES_PIXI (
    echo ERROR: Cannot find pixi. Install it in PowerShell with:
    echo    powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
    goto failed
)
for %%F in (pixi.toml pixi.lock launch_ies.py build_manifest.json svihm_ies.pst pestpp-ies.exe) do (
    if not exist "%~dp0%%F" (
        echo ERROR: Missing %%F. Copy the complete PEST build folder.
        goto failed
    )
)
echo Checking or creating the worker environment...
"%IES_PIXI%" install --locked -e worker --manifest-path "%~dp0pixi.toml"
if errorlevel 1 goto failed
"%IES_PIXI%" run --locked -e worker --manifest-path "%~dp0pixi.toml" python -c "import numpy, pandas, scipy, flopy, pyemu, tqdm, hydroeval, sklearn; print('Worker environment ready: numpy', numpy.__version__, 'pandas', pandas.__version__, 'pyemu', pyemu.__version__)"
if errorlevel 1 goto failed
echo.
echo Start the host computer first. Workers connect on TCP port 4004.
echo This computer: %COMPUTERNAME%
echo.
set "IES_BATCH_FILE=%TEMP%\ies_batch_%RANDOM%_%RANDOM%.txt"
"%IES_PIXI%" run --locked -e worker --manifest-path "%~dp0pixi.toml" python "%~dp0launch_ies.py" --template "%~dp0." --batch-path-file "%IES_BATCH_FILE%"
if errorlevel 1 goto failed
set /p IES_BATCH_DIR=<"%IES_BATCH_FILE%"
del "%IES_BATCH_FILE%"
if not exist "%IES_BATCH_DIR%\start_runs.bat" goto failed
call "%IES_BATCH_DIR%\start_runs.bat"
if errorlevel 1 goto failed
echo.
echo Each manager/worker has its own command window.
echo To stop this batch, run "%IES_BATCH_DIR%\stop_runs.bat".
pause
exit /b 0

:failed
echo.
echo ERROR: Setup or launch failed.
pause
exit /b 1
