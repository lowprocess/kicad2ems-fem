"""Running Elmer: ElmerGrid (gmsh mesh to Elmer's), ElmerSolver (a .sif), SaveScalars output.

    found()                                  # (True, where) or (False, why)
    elmergrid(msh, folder)                   # Elmer's mesh folder from a gmsh .msh (version 2)
    solve(case_folder, sif_text)             # ElmerSolver's log; raises with the log's tail on failure
    scalars(path)                            # {column name: value} of a SaveScalars file

Elmer is found as K2E_ELMER (the folder with ElmerSolver and ElmerGrid) or on PATH.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _tool(name):
    where = os.environ.get('K2E_ELMER')
    if where:
        path = Path(where) / name
        return str(path) if path.is_file() else None
    return shutil.which(name)


def found():
    missing = [n for n in ('ElmerSolver', 'ElmerGrid') if _tool(n) is None]
    if missing:
        return False, (f"Elmer not found ({', '.join(missing)}): install Elmer (elmerfem), or set K2E_ELMER to the "
                       "folder with ElmerSolver and ElmerGrid")
    return True, f"Elmer ({Path(_tool('ElmerSolver')).parent})"


def elmergrid(msh, folder):
    """Elmer's mesh folder from a gmsh .msh. No -autoclean: it renumbers the physical tags the .sif refers to."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    p = subprocess.run([_tool('ElmerGrid'), '14', '2', str(msh), '-out', str(folder)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError('ElmerGrid failed: ' + (p.stdout + p.stderr)[-1500:])
    return folder


def solve(case, sif_text, timeout=7200):
    case = Path(case)
    (case / 'case.sif').write_text(sif_text)
    (case / 'ELMERSOLVER_STARTINFO').write_text('case.sif\n1\n')
    p = subprocess.run([_tool('ElmerSolver'), 'case.sif'], cwd=case, capture_output=True, text=True, timeout=timeout)
    (case / 'elmer.log').write_text(p.stdout + p.stderr)
    if p.returncode or 'ALL DONE' not in p.stdout:
        raise RuntimeError('ElmerSolver failed; see ' + str(case / 'elmer.log') + '\n' + (p.stdout + p.stderr)[-1500:])
    return p.stdout


def scalars(path):
    """{name: value} of a SaveScalars file (the last row) and its .names."""
    names = Path(str(path) + '.names').read_text().splitlines()
    cols = [re.sub(r'^\s*\d+:\s*', '', l).strip() for l in names if re.match(r'^\s*\d+:', l)]
    return dict(zip(cols, np.loadtxt(path, ndmin=2)[-1]))
