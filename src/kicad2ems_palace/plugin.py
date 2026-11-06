"""
Palace (AWS's finite-element solver, awslabs/palace) as a kicad2ems solver plugin: the board meshed in
tetrahedra by gmsh (model.py), a driven Palace run over the band with its adaptive frequency sweep
(config.py), every port excited in turn (run.py).

Experimental. What it models: the dielectric slabs inside the board outline, lossy by their tand; copper as
PEC sheets of zero thickness; vias as PEC holes; lumped ports; an air box with an absorbing boundary around
the board. Not yet: lumped parts, conductor loss, roughness, the antenna mode's far field.
"""
import os
import time

import numpy as np

from kicad2ems.api.solver import Option, Solver, Result, Deviation

G = 'palace'


def palace_status():
    from .run import launcher, command
    how = launcher()
    if how is None:
        return False, ("Palace not found: set K2E_PALACE to its `palace` executable, or K2E_PALACE_ROOTFS to an "
                       "unpacked Palace image run in podman (kicad2ems-palace README)")
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
    capabilities = dict(ports='yes: lumped', gap_ports='yes: lumped', diff_pairs='yes', parts='no', vias='yes',
                        outline='yes', planes='yes', edge_walls='no', thickness='approximated: zero-thickness sheets',
                        conductor_loss='no: PEC', roughness='no', finish='no', dispersion='no: er, tand at one value',
                        radiation='yes: absorbing boundary', cavity='yes', antenna='no')
    options = [
        Option('palace_order', 2, "Palace's element order (default {default})", int, name='order',
               gui='Palace element order', group=G),
        Option('palace_tol', 1e-3, "the adaptive frequency sweep's tolerance (default {default}; 0: every point solved)",
               float, name='tol', gui='Palace sweep tolerance', group=G),
        Option('palace_feature', 1.0, "mesh cells across the smallest feature at the copper edges (default {default})",
               float, name='feature', gui='Palace cells per feature', group=G),
        Option('palace_wave', 8.0, "mesh cells per wavelength at fmax away from the copper (default {default})",
               float, name='wave', gui='Palace cells per wavelength', group=G),
        Option('palace_margin', None, "air around the board, mm (default: a quarter wavelength at fmax, at least 10x "
               "the stack)", float, name='margin', gui='Palace air margin (mm)', group=G),
        Option('palace_np', 8, "Palace's MPI processes (default {default}: the P-cores)", int, name='np',
               gui='Palace processes', group=G),
    ]

    def check(self, s=None):
        return palace_status()

    def assess(self, run):
        m, out = run.m, []
        if m.parts:
            out.append(f"{len(m.parts)} lumped part(s): not modelled yet")
        if getattr(run, 'antenna', False):
            out.append("antenna mode: no far field yet")
        return out

    def deviations(self, run, prep):
        out = [Deviation(f'copper:{lay}', 'thickness', 'a PEC sheet of zero thickness') for lay in run.m.copper]
        out += [Deviation(f'wall:{i}', 'edge_walls', 'left out', dropped=True) for i, _ in enumerate(run.m.walls)]
        return out

    def prepare(self, run):
        from . import model
        s = run.s
        work = run.path('_palace')
        os.makedirs(work, exist_ok=True)
        t = time.time()
        mdl = model.build(run.m, float(np.max(run.freq)), os.path.join(work, 'model.msh'), margin=s.palace_margin,
                          per_feature=s.palace_feature, per_wave=s.palace_wave, log=run.log)
        return dict(work=work, model=mdl, mesh_s=time.time() - t)

    def solve(self, run, prep):
        import json
        from kicad2ems.api.solver import mixed_mode
        from . import config, run as prun
        s, work, mdl = run.s, prep['work'], prep['model']
        cfg = config.config(mdl, run.freq, order=s.palace_order, tol=s.palace_tol)
        with open(os.path.join(work, 'palace.json'), 'w') as fh:
            json.dump(cfg, fh, indent=1)
        t = time.time()
        prun.run(work, 'palace.json', np=s.palace_np, log=run.log)
        seconds = time.time() - t
        f, S = prun.read_s(os.path.join(work, 'postpro'), len(run.m.ports))
        S = np.stack([np.interp(run.freq, f, S[:, i, j].real) + 1j * np.interp(run.freq, f, S[:, i, j].imag)
                      for i in range(S.shape[1]) for j in range(S.shape[2])], -1).reshape(len(run.freq), *S.shape[1:])
        S, names, zref = mixed_mode(S, run.m)
        info = dict(tets=mdl.tets, nodes=mdl.nodes, hmin=mdl.hmin, hmax=mdl.hmax, margin=mdl.margin,
                    mesh_s=round(prep['mesh_s'], 1), order=s.palace_order, seconds=round(seconds, 1))
        return Result(S, names, zref, seconds + prep['mesh_s'], info=info,
                      comment=f"Palace FEM, order {s.palace_order}, {mdl.tets} tets ({run.name})")
