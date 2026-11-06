"""
Running Palace and reading its S-parameters.

    launcher()                            # how Palace is started here: ('native', path) or ('podman', rootfs), or None
    run(workdir, 'palace.json', np=8)     # runs it there, returns its log
    f, S = read_s(workdir + '/postpro', nports)

Palace is found as
  K2E_PALACE          a `palace` executable (its MPI wrapper), or `palace` on PATH: run directly
  K2E_PALACE_ROOTFS   an unpacked Palace image (a root filesystem with Palace's Spack view in /opt/views/view):
                      run in rootless podman with that as its root, no network, as the user; default
                      ~/opt/palace/rootfs
"""
import glob
import os
import shutil
import subprocess

import numpy as np

ROOTFS = os.path.expanduser('~/opt/palace/rootfs')
VIEW_ENV = '/.singularity.d/env/91-environment.sh'       # the image's PATH to its Spack view


def launcher():
    exe = os.environ.get('K2E_PALACE') or shutil.which('palace')
    if exe:
        return 'native', exe
    root = os.environ.get('K2E_PALACE_ROOTFS', ROOTFS)
    if os.path.isdir(os.path.join(root, 'opt/views/view/bin')) and shutil.which('podman'):
        return 'podman', root
    return None


def command(workdir, cfg, np=8, threads=1, how=None):
    how = how or launcher()
    if how is None:
        raise RuntimeError("Palace not found: set K2E_PALACE to its executable, or K2E_PALACE_ROOTFS to an unpacked "
                           "Palace image (README)")
    kind, where = how
    args = f"palace -np {int(np)} -nt {int(threads)} {cfg}"
    if kind == 'native':
        return [where, '-np', str(int(np)), '-nt', str(int(threads)), cfg], workdir
    env = f". {VIEW_ENV} 2>/dev/null || export PATH=/opt/views/view/bin:$PATH; cd /work && {args}"
    # --rootfs takes the root as the first positional argument: every option goes before it
    return ['podman', 'run', '--rm', '--network', 'none', '--userns=keep-id', '--shm-size', '2g',
            '-v', f'{os.path.abspath(workdir)}:/work:Z', '--rootfs', f'{where}:O', 'bash', '-c', env], None


def run(workdir, cfg, np=8, threads=1, log=print, how=None, timeout=None):
    cmd, cwd = command(workdir, cfg, np, threads, how)
    with open(os.path.join(workdir, 'palace.log'), 'w') as fh:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in p.stdout:
            fh.write(line)
            tail = (tail + [line.rstrip()])[-40:]
            if line.lstrip().startswith(('It ', 'Computing', 'Adaptive', 'Elapsed', 'Total')):
                log('palace: ' + line.strip())
        rc = p.wait(timeout)
    if rc:
        raise RuntimeError(f"palace exited {rc}:\n" + '\n'.join(tail[-20:]))
    return '\n'.join(tail)


def read_s(postpro, n):
    """(f Hz, S (nf, n, n) complex) from Palace's port-S.csv: columns f (GHz), |S[i][j]| (dB), arg(S[i][j]) (deg.)
    for each excited port j. NaN where a term is missing."""
    path = os.path.join(postpro, 'port-S.csv')
    head = open(path).readline().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    cols = [h.strip() for h in head]
    f = data[:, 0] * 1e9
    S = np.full((len(f), n, n), np.nan + 0j)
    import re
    for c, h in enumerate(cols):
        m = re.match(r'\|S\[(\d+)\]\[(\d+)\]\| \(dB\)', h)
        if not m:
            continue
        i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
        a = cols.index(f'arg(S[{i + 1}][{j + 1}]) (deg.)')
        S[:, i, j] = 10 ** (data[:, c] / 20) * np.exp(1j * np.radians(data[:, a]))
    return f, S
