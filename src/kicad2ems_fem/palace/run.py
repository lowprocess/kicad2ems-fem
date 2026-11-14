"""
Running Palace and reading its S-parameters.

    launcher()                            # how Palace is started here: ('native', path) or ('podman', rootfs), or None
    run(workdir, 'palace.json', np=8)     # runs it there, returns its log
    f, S = read_s(workdir + '/postpro', nports)
    f, P = read_incident(workdir + '/postpro', 1)         # the driven port's incident power, each solve
    f, th, ph, E = read_farfield(workdir + '/postpro')    # the far field r E at the config's angles

Palace is found as
  K2E_PALACE          a `palace` executable (its MPI wrapper), or `palace` on PATH: run directly, with its own
                      folder first on PATH (a Spack view's: its mpirun beside it)
  K2E_PALACE_ROOTFS   an unpacked Palace image (a root filesystem with Palace's Spack view in /opt/views/view):
                      run in rootless podman with that as its root, no network, as the user; default
                      ~/opt/palace/rootfs
  each also as kicad2ems's installer recorded it (kicad2ems-site.toml: palace, palace_rootfs), after the
  environment's and before PATH's
"""
import os
import shutil
import subprocess
import time

import numpy as np

ROOTFS = os.path.expanduser('~/opt/palace/rootfs')
VIEW_ENV = '/.singularity.d/env/91-environment.sh'       # the image's PATH to its Spack view


def _site(key):
    from ..elmer.run import site
    return site(key)


def launcher():
    exe = os.environ.get('K2E_PALACE') or _site('palace') or shutil.which('palace')
    if exe:
        return 'native', exe
    root = os.environ.get('K2E_PALACE_ROOTFS') or _site('palace_rootfs') or ROOTFS
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


def environment(cmd):
    """The environment a Palace command runs in: a native palace's own folder first on PATH (a Spack view's,
    where its mpirun is too: the launcher finds MPI on PATH)."""
    env = dict(os.environ)
    if cmd and cmd[0] != 'podman':
        env['PATH'] = os.pathsep.join(filter(None, (os.path.dirname(os.path.abspath(cmd[0])), env.get('PATH'))))
    return env


def run(workdir, cfg, np=8, threads=1, log=print, how=None, timeout=None, logfile='palace.log'):
    cmd, cwd = command(workdir, cfg, np, threads, how)
    with open(os.path.join(workdir, logfile), 'w') as fh:
        from .. import children                             # its own process group: stopped with the tool
        p = children.popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env=environment(cmd))
        tail, shown = [], 0.0
        for line in p.stdout:
            fh.write(line)
            fh.flush()                                      # the log can be followed while Palace runs
            tail = (tail + [line.rstrip()])[-40:]
            head = line.lstrip()
            if head.startswith('It '):                      # one a second: the swept model answers 100s at once
                if time.time() - shown < 1.0:
                    continue
                shown = time.time()
            if head.startswith(('It ', 'Computing', 'Adaptive', 'Greedy', 'Elapsed', 'Total')):
                log('palace: ' + line.strip())
        rc = p.wait(timeout)
    if children.stopped(p):
        children.raise_stopped()
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
        if i >= n or j >= n:                                # a lumped part's boundary: not a port of the board
            continue
        a = cols.index(f'arg(S[{i + 1}][{j + 1}]) (deg.)')
        S[:, i, j] = 10 ** (data[:, c] / 20) * np.exp(1j * np.radians(data[:, a]))
    return f, S


def table(path):
    """(column names, rows) of one of Palace's CSV files."""
    with open(path) as fh:
        cols = [h.strip() for h in fh.readline().split(',')]
    return cols, np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)


def read_incident(postpro, index):
    """(f Hz, P W (nf,)) of the driven lumped port `index` (1-based): the incident wave's power, 0.5 V_inc I_inc
    of port-V.csv and port-I.csv (peak phasors). The port's own V and I there are those of its resistor, so the
    power into the model is this times 1 - |S11|^2."""
    out = []
    for name, q in (('port-V.csv', 'V'), ('port-I.csv', 'I')):
        cols, data = table(os.path.join(postpro, name))
        c = [k for k, h in enumerate(cols) if h.startswith(f'{q}_inc[{index}]')]
        if len(c) != 1:
            raise ValueError(f"{name}: no single column of {q}_inc[{index}] in {cols}")
        out.append(data[:, c[0]])
    return data[:, 0] * 1e9, 0.5 * out[0] * out[1]


def read_farfield(postpro):
    """(f Hz (nf,), theta, phi deg (n,), E (nf, n, 3) complex) from Palace's farfield-rE.csv: the far field times
    the distance, r E (V), in Cartesian components, at the config's angles (one excitation)."""
    cols, data = table(os.path.join(postpro, 'farfield-rE.csv'))
    f = data[:, 0]
    freqs = np.unique(f)
    first = data[f == freqs[0]]
    n = len(first)
    if len(data) != n * len(freqs):
        raise ValueError(f"farfield-rE.csv: {len(data)} rows, not {len(freqs)} frequencies x {n} angles")
    E = np.empty((len(freqs), n, 3), complex)
    comp = [[c for c, h in enumerate(cols) if h.startswith(f'r*{part}{{E_{x}}}')] for x in 'xyz' for part in ('Re', 'Im')]
    ang = [[c for c, h in enumerate(cols) if h.startswith(a)] for a in ('theta', 'phi')]
    if any(len(c) != 1 for c in comp + ang):
        raise ValueError(f"farfield-rE.csv: unexpected columns {cols}")
    th, ph = ang[0][0], ang[1][0]
    for i, fi in enumerate(freqs):
        d = data[f == fi]
        if not (np.allclose(d[:, th], first[:, th]) and np.allclose(d[:, ph], first[:, ph])):
            raise ValueError("farfield-rE.csv: the angles differ between frequencies")
        for x in range(3):
            E[i, :, x] = d[:, comp[2 * x][0]] + 1j * d[:, comp[2 * x + 1][0]]
    return freqs * 1e9, first[:, th], first[:, ph], E
