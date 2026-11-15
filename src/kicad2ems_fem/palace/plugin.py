"""
Palace (AWS's finite-element solver, awslabs/palace) as a kicad2ems solver plugin: the board meshed in
tetrahedra by gmsh (model.py), a driven Palace run over the band with its adaptive frequency sweep
(config.py), every port excited in turn (run.py).

Experimental. What it models: the dielectric slabs inside the board outline, lossy by their tand; copper as
PEC sheets of zero thickness (or lossy ones, --palace-loss); vias as PEC holes; lumped ports; lumped R / L / C
parts as lumped boundaries across their gaps; edge walls as PEC sheets; an air box with an absorbing boundary
around the board, or a closed metal can (--palace-box pec, --palace-margin its distance from the board). Antenna
mode: a second run at a few frequencies for the far field of the driven port (antenna.py). Eigenmodes
(--palace-modes N, R-E19): after the driven run, the model's N resonances above fmin, each one's frequency and
Q loaded by the ports' resistors, in the summary and the report. Not yet: roughness, surface finish, phased
drives.
"""
import os
import time

import numpy as np

from kicad2ems.api.solver import Option, Solver, Result, Deviation

G = 'palace'
ANTENNA_MARGIN = 0.5            # antenna mode's air around the board, wavelengths at the band centre
DAMPED_Q = 5.0                  # an eigenmode below this Q: a feed line damped by its port (rf_shielded: Q 1.1, 2.1
                                # beside the resonator's 158 and the can's 559), not a resonance


class _quiet:
    """stdout and stderr at the descriptors to /dev/null, C's buffers flushed on both sides: TetGen prints past
    gmsh's General.Terminal, and what sat in C's buffer had reached the next redirect of them."""

    def __enter__(self):
        import ctypes
        import sys
        self.libc = ctypes.CDLL(None) if os.name != 'nt' else None
        for f in (sys.stdout, sys.stderr):
            f.flush()
        if self.libc is not None:
            self.libc.fflush(None)
        self.saved = [os.dup(1), os.dup(2)]
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        os.close(null)
        return self

    def __exit__(self, *exc):
        if self.libc is not None:
            self.libc.fflush(None)
        for fd, keep in zip((1, 2), self.saved):
            os.dup2(keep, fd)
            os.close(keep)
        return False


def palace_status():
    from .run import launcher
    how = launcher()
    if how is None:
        return False, ("Palace not found: set K2E_PALACE to its `palace` executable, or K2E_PALACE_ROOTFS to an "
                       "unpacked Palace image run in podman (kicad2ems-fem README)")
    try:
        import gmsh                                         # noqa: F401
    except ImportError:
        return False, "gmsh's Python module is missing: python -m pip install gmsh"
    kind, where = how
    return True, f"Palace via {kind} ({where})"


class Palace(Solver):
    api = (2, 1)
    name, aliases, label = 'palace', ('fem',), 'Palace (FEM)'
    in_auto = False                 # experimental: only when named
    rank, order = 2, 40
    capabilities = dict(ports='yes: lumped', gap_ports='yes: lumped', diff_pairs='yes',
                        parts='yes: lumped sheets across the gap', vias='yes',
                        outline='yes', planes='yes', edge_walls='yes: PEC sheets',
                        thickness='approximated: zero-thickness sheets',
                        conductor_loss='optional (--palace-loss): finite-thickness surface impedance on the sheets; '
                                       'good to ~10 MHz, high above (edge singularity)',
                        roughness='no', finish='no', dispersion='no: er, tand at one value',
                        radiation='yes: absorbing boundary', cavity='yes: --palace-box pec, a metal can',
                        antenna='yes: the finite board in air, one driven port (no --drive)')
    options = [
        Option('palace_order', 2, "Palace's element order (default {default})", int, name='order',
               gui='Palace element order', group=G),
        Option('palace_tol', 1e-3, "the adaptive frequency sweep's tolerance (default {default}; 0: every point solved)",
               float, name='tol', gui='Palace sweep tolerance', group=G),
        Option('palace_feature', 4.0, "mesh cells across the smallest feature at the copper edges (default {default})",
               float, name='feature', gui='Palace cells per feature', group=G),
        Option('palace_wave', 8.0, "mesh cells per wavelength at fmax away from the copper (default {default})",
               float, name='wave', gui='Palace cells per wavelength', group=G),
        Option('palace_margin', None, "air around the board, mm (default: a quarter wavelength at fmax, at least 10x "
               "the stack; antenna mode: half a wavelength at the band centre)", float, name='margin',
               gui='Palace air margin (mm)', group=G),
        Option('palace_np', 0, "Palace's MPI processes (default {default}: one per fast core, emsdev's count)", int, name='np',
               gui='Palace processes', group=G),
        Option('palace_loss', False, "copper loss: the sheets as Palace's finite-thickness conductivity boundary "
               "instead of PEC (good to ~10 MHz; high above)", bool, name='loss', gui='Palace copper loss', group=G),
        Option('palace_amr', 0, "passes of Palace's adaptive mesh refinement: the cells where its error estimate is "
               "largest split, until it is below --palace-amr-tol (default {default}: none)", int, name='amr',
               gui='Palace mesh refinement passes', group=G, cli=dict(metavar='N')),
        Option('palace_amr_tol', 1e-2, "the adaptive mesh refinement's error estimate to stop at (default {default})",
               float, name='amr_tol', group=G),
        Option('palace_box', 'absorbing', "the air box's faces: absorbing (the board in free space) or pec (a closed "
               "metal can, --palace-margin its distance from the board on every side)", str, name='box',
               choices=('absorbing', 'pec'), gui='Palace box', group=G),
        Option('palace_modes', 0, "after the driven run, the model's N resonances above fmin (an eigenmode run): each "
               "one's frequency and Q, the ports loading it (default {default}: none)", int, name='modes',
               gui='Palace resonances', group=G, cli=dict(metavar='N')),
        Option('palace_fine_ground', False, "the feature size at the ground's edges too: the ground planes' own edges, "
               "the vias on ground nets (then round) and the edge walls. Several times the mesh", bool,
               name='fine_ground', gui='Palace fine mesh at the ground', group=G),
    ]

    def check(self, s=None):
        return palace_status()

    def validate(self, s):
        if getattr(s, 'palace_modes', 0) and s.palace_modes < 0:
            return "--palace-modes: a number of resonances, 0 or more"
        return None

    def assess(self, run):
        out = []
        if getattr(run, 'antenna', False) and getattr(run.s, 'drive', None):
            out.append("antenna mode: --drive (a phased drive) is not modelled, one port is driven")
        if getattr(run, 'antenna', False) and getattr(run.s, 'palace_box', 'absorbing') == 'pec':
            out.append("antenna mode: the far field needs the absorbing box (--palace-box absorbing), not a can")
        return out

    # its cost, from the runs of 2026-11 (the examples on 8 ranks of an i7-13700K, order 2, 391-401 points of the
    # adaptive sweep): unknowns about 1.53 a tetrahedron; Palace's peak memory 1.5 GB + 55 MB a thousand unknowns
    # (within ~30 %); the sweep's time ~1.8e-5 s x tets^1.5 on 8 ranks (within x2-3: its samples follow the board)
    UNKNOWNS_PER_TET, GB0, GB_PER_KUNKNOWN, T0 = 1.53, 1.5, 0.055, 1.8e-5

    def estimate(self, run):
        """Its rough cost (R-B21): the mesh built as the run would (gmsh, seconds), its tetrahedra scaled by the
        runs above. Only for a run that has it (a preview estimates every solver: the mesh is seconds to minutes),
        gmsh's own output kept out of the run's."""
        import tempfile
        from . import model
        from .run import ranks
        if self.name not in [*(getattr(run, 'plan', None) or ()), *(getattr(run, 'previewed', None) or ())]:
            return None
        s = run.s
        margin = s.palace_margin
        if margin is None and getattr(run, 'antenna', False):
            margin = ANTENNA_MARGIN * 299792458.0 / (0.5 * (run.fmin + run.fmax)) * 1e3
        with tempfile.TemporaryDirectory() as d, _quiet():
            mdl = model.build(run.m, float(np.max(run.freq)), os.path.join(d, 'model.msh'), margin=margin,
                              per_feature=s.palace_feature, per_wave=s.palace_wave, planes=not s.palace_fine_ground,
                              log=lambda *a: None)
        order = int(getattr(s, 'palace_order', 2) or 2)
        unknowns = self.UNKNOWNS_PER_TET * mdl.tets * (order / 2) ** 3
        return dict(seconds=self.T0 * (unknowns / self.UNKNOWNS_PER_TET) ** 1.5 * 8 / max(ranks(s.palace_np), 1),
                    gb=self.GB0 + self.GB_PER_KUNKNOWN * unknowns / 1e3,
                    what=f"{mdl.tets:,} tetrahedra, ~{unknowns / 1e3:.0f} k unknowns at order {order}")

    def deviations(self, run, prep):
        from .model import part_elements
        out = [Deviation(f'copper:{lay}', 'thickness', 'a PEC sheet of zero thickness') for lay in run.m.copper]
        for pt in run.m.parts:
            n = len(part_elements(pt))
            if n > 1:
                out.append(Deviation(f'part:{pt.ref}', 'parts', f'{n} sheets in series along its gap, one element each'))
        return out

    def prepare(self, run):
        from . import model
        s = run.s
        work = run.path('_palace')
        os.makedirs(work, exist_ok=True)
        t = time.time()
        margin = s.palace_margin
        if margin is None and getattr(run, 'antenna', False):  # the absorbing faces well away from the antenna
            margin = ANTENNA_MARGIN * 299792458.0 / (0.5 * (run.fmin + run.fmax)) * 1e3
        mdl = model.build(run.m, float(np.max(run.freq)), os.path.join(work, 'model.msh'), margin=margin,
                          per_feature=s.palace_feature, per_wave=s.palace_wave, planes=not s.palace_fine_ground,
                          log=run.log)
        return dict(work=work, model=mdl, mesh_s=time.time() - t)

    def solve(self, run, prep):
        import json
        from kicad2ems.api.solver import mixed_mode
        from . import config, run as prun
        s, work, mdl = run.s, prep['work'], prep['model']
        box = getattr(s, 'palace_box', None) or 'absorbing'
        cfg = config.config(mdl, run.freq, order=s.palace_order, tol=s.palace_tol, loss=bool(s.palace_loss),
                            amr=getattr(s, 'palace_amr', 0) or 0, amr_tol=getattr(s, 'palace_amr_tol', 1e-2), box=box)
        with open(os.path.join(work, 'palace.json'), 'w') as fh:
            json.dump(cfg, fh, indent=1)
        t = time.time()
        prun.run(work, 'palace.json', np=s.palace_np, log=run.log)
        seconds = time.time() - t
        f, S = prun.read_s(os.path.join(work, 'postpro'), len(run.m.ports))
        S = np.stack([np.interp(run.freq, f, S[:, i, j].real) + 1j * np.interp(run.freq, f, S[:, i, j].imag)
                      for i in range(S.shape[1]) for j in range(S.shape[2])], -1).reshape(len(run.freq), *S.shape[1:])
        if getattr(run, 'antenna', False):
            t = time.time()
            self._antenna(run, prep, S)
            seconds += time.time() - t
        modes = None
        if getattr(s, 'palace_modes', 0):
            t = time.time()
            modes = self._modes(run, prep, box)
            seconds += time.time() - t
        S, names, zref = mixed_mode(S, run.m)
        info = dict(tets=mdl.tets, nodes=mdl.nodes, hmin=mdl.hmin, hmax=mdl.hmax, margin=mdl.margin, box=box,
                    mesh_s=round(prep['mesh_s'], 1), order=s.palace_order, seconds=round(seconds, 1),
                    parts=len(mdl.parts), edge_walls=mdl.edge_wall is not None, engine=prun.engine(prep['work']))
        if modes is not None:
            info['modes'] = modes
        return Result(S, names, zref, seconds + prep['mesh_s'], info=info,
                      comment=f"Palace FEM, order {s.palace_order}, {mdl.tets} tets ({run.name})")

    def _modes(self, run, prep, box):
        """The eigenmode run (R-E19): the model's --palace-modes resonances above fmin, on the driven run's mesh and
        settings, the ports as their resistors. Logged as a table, written as <name>_palace_modes.json; returns
        dict(about, rows=[dict(mode, f_ghz, Q, q_ext={port: Q}, error)]) for the summary and the report."""
        import json
        from . import config, run as prun
        s, work, mdl = run.s, prep['work'], prep['model']
        n = int(s.palace_modes)
        cfg = config.eigen(mdl, n, run.fmin, order=s.palace_order, loss=bool(s.palace_loss), box=box)
        with open(os.path.join(work, 'palace_eig.json'), 'w') as fh:
            json.dump(cfg, fh, indent=1)
        run.log(f"palace: eigenmodes, {n} above {run.fmin / 1e9:g} GHz ({'a metal can' if box == 'pec' else 'open box'})")
        prun.run(work, 'palace_eig.json', np=s.palace_np, log=run.log, logfile='palace_eig.log')
        found = prun.read_modes(os.path.join(work, 'postpro_eig'))
        names = {p['index']: p['name'] for p in mdl.ports}
        rows = [dict(mode=i + 1, f_ghz=m['f_Hz'] / 1e9, Q=m['Q'], error=m['error'],
                     q_ext={names.get(k, str(k)): v for k, v in m['q_ext'].items() if k in names},
                     **({'kind': 'a feed line damped by its port'} if m['Q'] < DAMPED_Q else {}))
                for i, m in enumerate(found)]
        about = (f"Palace's eigenmodes above {run.fmin / 1e9:g} GHz, "
                 + ("in the metal can" if box == 'pec' else "the open board (radiation in Q)")
                 + "; Q loaded by the ports' resistors" + (", the copper's loss" if s.palace_loss else "")
                 + f" and the dielectric's tan d; Q_ext: each port's share. A Q under {DAMPED_Q:g} is a feed line "
                 "damped by its port's resistor, not a resonance")
        for r in rows:
            ext = ', '.join(f"{p} {q:.4g}" for p, q in r['q_ext'].items())
            run.log(f"  palace mode {r['mode']}: {r['f_ghz']:.5g} GHz, Q {r['Q']:.4g}" + (f" (Q_ext {ext})" if ext else '')
                    + (f": {r['kind']}" if r.get('kind') else ''))
        out = dict(about=about, rows=rows)
        js = run.path('_palace_modes.json')
        with open(js, 'w') as fh:
            json.dump(out, fh, indent=1)
        run.wrote('eigenmodes, Palace (JSON)', js)
        return out

    def _antenna(self, run, prep, S):
        """Antenna mode: the far field of the driven port (the first of --excite, else the first port) in a second
        Palace run at frequencies over the band and the pattern frequency, written as the other solvers' antenna
        results (the JSON, the sphere, the plots and the report section)."""
        import json
        from kicad2ems.api.solver import match_summary, describe_antenna, save_antenna, plot_antenna, plot_pattern
        from . import antenna, config, run as prun
        s, m, work, mdl = run.s, run.m, prep['work'], prep['model']
        names = [p.name for p in m.ports]
        want = getattr(s, 'excite', None) or []
        want = [x.strip() for x in (want.split(',') if isinstance(want, str) else want)]
        k = next((j for j, nm in enumerate(names) if nm in want), 0)
        s11 = S[:, k, k]
        match = match_summary(run.freq, s11, m.ports[k].z0)
        ff = getattr(s, 'ff_freq', None)
        f_pat = float(ff[0]) * 1e9 if ff else match['f_res']
        step = float(getattr(s, 'ff_step', None) or 2.0)
        ffr = antenna.far_freqs(run.fmin, run.fmax, f_pat)
        run.log(f"palace: antenna, {names[k]} driven: far field at {len(ffr)} frequencies")
        cfg = config.config(mdl, ffr, order=s.palace_order, tol=0, loss=bool(s.palace_loss), excite=[k],
                            farfield=antenna.angles(step), out='postpro_ff')
        with open(os.path.join(work, 'palace_ff.json'), 'w') as fh:
            json.dump(cfg, fh, indent=1)
        prun.run(work, 'palace_ff.json', np=s.palace_np, log=run.log, logfile='palace_ff.log')
        post = os.path.join(work, 'postpro_ff')
        f, pinc = prun.read_incident(post, k + 1)
        _, Sf = prun.read_s(post, len(names))
        _, th, ph, E = prun.read_farfield(post)
        ant, sph = antenna.analyse(f, Sf[:, k, k], pinc, th, ph, E, step, f_pat)
        sph_file = run.path('_palace_sphere.npz')
        np.savez_compressed(sph_file, **sph)
        ant.update(match=match, port=names[k], solver='palace', sphere_file=os.path.basename(sph_file),
                   model=f"the finite board in air, absorbing faces {mdl.margin:.0f} mm away; "
                         + ('lossy copper sheets' if s.palace_loss else 'lossless copper'))
        for line in describe_antenna(match, ant['rows'], ant['f_pattern']):
            run.log(f"  palace {line}")
        tag = '' if 'antenna' not in run.summary else '_palace'           # another solver's results come first
        run.figure(plot_antenna(run.freq, s11, m.ports[k].z0, ant, f"{run.name} - antenna (Palace)"),
                   "_palace_antenna.png", f"antenna{tag}", 'antenna (Palace): match, gain, efficiency')
        run.figure(plot_pattern(ant, f"{run.name} - gain pattern (Palace)"), "_palace_pattern.png", f"pattern{tag}",
                   'antenna (Palace): patterns')
        js = run.path('_palace_antenna.json')
        save_antenna(ant, js)
        run.wrote('antenna results, Palace (JSON)', js)
        run.wrote('antenna sphere, Palace', sph_file)
        run.summary['antenna' + tag] = {k_: v for k_, v in ant.items() if k_ != 'sphere'}
