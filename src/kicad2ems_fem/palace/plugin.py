"""
Palace (AWS's finite-element solver, awslabs/palace) as a kicad2ems solver plugin: the board meshed in
tetrahedra by gmsh (model.py), a driven Palace run over the band with its adaptive frequency sweep
(config.py), every port excited in turn (run.py).

Experimental. What it models: the dielectric slabs inside the board outline, lossy by their tand; copper as
PEC sheets of zero thickness (or lossy ones, --palace-loss); vias as PEC holes; lumped ports; lumped R / L / C
parts as lumped boundaries across their gaps; edge walls as PEC sheets; an air box with an absorbing boundary
around the board. Antenna mode: a second run at a few frequencies for the far field of the driven port
(antenna.py). Not yet: roughness, surface finish, phased drives.
"""
import os
import time

import numpy as np

from kicad2ems.api.solver import Option, Solver, Result, Deviation

G = 'palace'
ANTENNA_MARGIN = 0.5            # antenna mode's air around the board, wavelengths at the band centre


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
                        radiation='yes: absorbing boundary', cavity='yes',
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
        Option('palace_fine_ground', False, "the feature size at the ground's edges too: the ground planes' own edges, "
               "the vias on ground nets (then round) and the edge walls. Several times the mesh", bool,
               name='fine_ground', gui='Palace fine mesh at the ground', group=G),
    ]

    def check(self, s=None):
        return palace_status()

    def assess(self, run):
        out = []
        if getattr(run, 'antenna', False) and getattr(run.s, 'drive', None):
            out.append("antenna mode: --drive (a phased drive) is not modelled, one port is driven")
        return out

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
        cfg = config.config(mdl, run.freq, order=s.palace_order, tol=s.palace_tol, loss=bool(s.palace_loss))
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
        S, names, zref = mixed_mode(S, run.m)
        info = dict(tets=mdl.tets, nodes=mdl.nodes, hmin=mdl.hmin, hmax=mdl.hmax, margin=mdl.margin,
                    mesh_s=round(prep['mesh_s'], 1), order=s.palace_order, seconds=round(seconds, 1),
                    parts=len(mdl.parts), edge_walls=mdl.edge_wall is not None)
        return Result(S, names, zref, seconds + prep['mesh_s'], info=info,
                      comment=f"Palace FEM, order {s.palace_order}, {mdl.tets} tets ({run.name})")

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
