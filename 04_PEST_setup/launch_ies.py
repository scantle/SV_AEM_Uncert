"""
PEST-IES manager/worker launcher (standard library only)

Run by start_ies.bat from a PEST build folder (written by 06_Setup_PEST-iES_Control_File.py). Copies the
files listed in build_manifest.json into a new batch folder with a manager and/or worker subfolders, then writes
start_runs.bat and stop_runs.bat. Existing folders are never reused or deleted, and the build folder must be local.
"""
import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path

#----------------------------------------------------------------------------------------------------------------------#
# Settings
#----------------------------------------------------------------------------------------------------------------------#

pst_file = 'svihm_ies.pst'
pestpp_exe = 'pestpp-ies.exe'
manifest_file = 'build_manifest.json'
default_port = 4004

# PEST++ options naming files only the manager reads (not copied to workers)
manager_only_options = ['parcov', 'ies_parameter_ensemble', 'ies_observation_ensemble', 'ies_localizer']

# Imports needed by the forward run scripts
worker_imports = 'import numpy, pandas, scipy, flopy, pyemu, tqdm, hydroeval, sklearn'

#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#

def local_path(value):
    """
    Absolute path, refusing network drives (P:, UNC) and links/junctions.
    """
    path = Path(value).absolute()
    for part in [path, *path.parents]:
        if part.drive.casefold() == 'p:' or str(part).startswith('\\\\'):
            raise ValueError(f'Only local paths are allowed: {path}')
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError(f'Links and junctions are not allowed: {part}')
    return path

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def checked_files(template):
    """
    Files listed in the build manifest, after checking none have changed since the build.
    """
    manifest = json.loads((template / manifest_file).read_text())
    files = []
    for relative, expected in manifest.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'Unsafe path in manifest: {relative}')
        path = local_path(template / relative)
        if sha256(path) != expected:
            raise ValueError(f'Build file changed since 06 was run: {relative}. Rebuild the PEST folder.')
        files.append(relative)
    files.append(manifest_file)
    return files

def launch(template, destination, python, count, host, hostname, port):
    template, destination, python = map(local_path, [template, destination, python])
    if count < 0 or (not host and count == 0) or not 1 <= port <= 65535:
        raise ValueError('Invalid worker count or port')
    if destination.exists():
        raise ValueError(f'Batch folder already exists: {destination}')
    python = python.resolve(strict=True)
    subprocess.run([str(python), '-c', worker_imports], check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for value in [str(python), hostname]:
        if any(c in value for c in '\r\n"%&|<>^'):
            raise ValueError(f'Unsupported character in batch command argument: {value}')

    print('Checking build files...')
    files = checked_files(template)
    names = (['manager'] if host else []) + [f'worker_{i:03d}' for i in range(count)]
    manager_only = set()
    for line in (template / pst_file).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].lower() in manager_only_options:
            manager_only.add(parts[1])

    # Complete every copy before starting any process
    for name in names:
        print(f'Copying {name}')
        folder = destination / name
        for relative in files:
            if name != 'manager' and relative in manager_only:
                continue
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template / relative, target)

    title = 'IES_' + uuid.uuid4().hex[:12]
    processes = []
    start_lines = ['@echo off', 'setlocal DisableDelayedExpansion', 'cd /d "%~dp0"']
    for name in names:
        address = f':{port}' if name == 'manager' else f'{hostname}:{port}'
        script = [
            '@echo off',
            'setlocal DisableDelayedExpansion',
            f'title {title}_{name}',
            'cd /d "%~dp0"',
            f'set "IES_PYTHON={python}"',
            f'"{pestpp_exe}" {pst_file} /h "{address}"',
            'set "IES_EXIT=%ERRORLEVEL%"',
            'echo.',
            f'echo {name} finished with exit code %IES_EXIT%.',
            'exit /b %IES_EXIT%',
        ]
        (destination / name / 'run_ies.cmd').write_text('\n'.join(script) + '\n')
        start_lines.append(f'start "{title}_{name}" /D "%~dp0{name}" cmd.exe /k call run_ies.cmd')
        processes.append({'role': name, 'address': address, 'window_title': f'{title}_{name}'})
    start_lines.extend(['echo IES windows started.', 'exit /b 0'])
    (destination / 'start_runs.bat').write_text('\n'.join(start_lines) + '\n')
    (destination / 'stop_runs.bat').write_text(
        '@echo off\n'
        f'taskkill /FI "WINDOWTITLE eq {title}_*" /T /F\n'
        'pause\n')
    info = {'template': str(template), 'python': str(python), 'created_utc': datetime.now(timezone.utc).isoformat(),
            'processes': processes}
    (destination / 'launch.json').write_text(json.dumps(info, indent=2) + '\n')
    return processes

#----------------------------------------------------------------------------------------------------------------------#
# Main
#----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare a PEST-IES manager/worker batch')
    parser.add_argument('--template', type=Path, required=True, help='PEST build folder')
    parser.add_argument('--destination', type=Path, help='New batch folder (default: next to the build folder)')
    parser.add_argument('--python', type=Path, help='Worker python (default: the python running this script)')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--host', type=str.lower, choices=['y', 'yes', 'n', 'no'])
    parser.add_argument('--hostname', help='Manager computer name (workers only)')
    parser.add_argument('--port', type=int, default=default_port)
    parser.add_argument('--batch-path-file', type=Path, help='Write the batch folder location here')
    args = parser.parse_args()

    count = args.workers if args.workers is not None else int(input('Number of workers: '))
    answer = args.host
    while answer not in {'y', 'yes', 'n', 'no'}:
        answer = input('Is this the host computer? [y/n]: ').strip().lower()
    host = answer in {'y', 'yes'}
    print(f"Preparing {'MANAGER + workers' if host else 'WORKERS ONLY'}.")
    hostname = args.hostname or (socket.gethostname() if host else input('Manager computer name: ').strip())
    python = args.python or Path(sys.executable)
    template = args.template.absolute()
    destination = args.destination or template.parent / (
        f"{template.name}_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    processes = launch(template, destination.absolute(), python, count, host, hostname, args.port)
    print(json.dumps(processes, indent=2))
    print(f'Prepared batch: {destination.absolute()}')
    if args.batch_path_file:
        args.batch_path_file.write_text(str(destination.absolute()) + '\n')
