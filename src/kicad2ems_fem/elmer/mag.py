"""Coils with ferrite, with Elmer: the inductance of wound coils and the coupling between them, beside ferrite plates.

    python -m kicad2ems_fem.elmer.mag --coil tx:turns=10,inner=10,outer=22,z=0,t=1 \\
        --coil rx:turns=14,inner=8,outer=16,z=5,t=0.5 --plate z=-2.5,t=2.5,size=50x50,mur=2000 --out mag

A coil is a flat ring of winding (mm: inner and outer radius, z of its lower face, thickness t; x, y of its centre, 0
when not given) that carries its turns' current spread evenly over its section: litz or fine wire well below its
resonance, not one solid turn. A plate is a ferrite slab (size=AxB or r=RADIUS, z, t, mur; x, y). Every coil's axis
is z. Elmer solves the static field of each coil alone and of each pair together in an air ball; the inductances come
from the field energy. Writes <out>/mag.json: L (H, the matrix), k, and `air`, the same coils without ferrite from
the ring formulas (air_core below), which no ferrite run needs but every result can be read against.

    from kicad2ems_fem.elmer.mag import solve, air_core
    r = solve([dict(name='tx', turns=10, inner=10, outer=22, z=0, t=1)], [dict(z=-2.5, t=2.5, size=(50, 50), mur=2000)], 'mag')
    r['L'][0][0], r['air'][0][0]

What is left out: the coil's own resistance, skin and proximity loss, the ferrite's loss and saturation, metal
nearby (a shield or a battery behind the ferrite carries eddy currents: not in a static field), and coils drawn as
board copper (pcbqs has those in air).
"""
import argparse
import re
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import run as elmer

MU0 = 4e-7 * np.pi
AIR, OUTER = 1, 1                                                # the air body; the ball's face
COIL0 = 2                                                        # coil bodies from here, then the plates


# -- the rings in air ---------------------------------------------------------------------------------------------

def _maxwell(r1, r2, d):
    """Mutual inductance of two coaxial circles (m), H: Maxwell's elliptic-integral formula."""
    from scipy.special import ellipe, ellipk
    m = 4 * r1 * r2 / ((r1 + r2) ** 2 + d ** 2)
    k = np.sqrt(m)
    return MU0 * np.sqrt(r1 * r2) * ((2 / k - k) * ellipk(m) - 2 / k * ellipe(m))


def _neumann(r1, r2, off, d, n=360):
    """Mutual inductance of two parallel circles whose axes are `off` apart: Neumann's integral."""
    a = (np.arange(n) + 0.5) * 2 * np.pi / n
    c1, s1, c2, s2 = np.cos(a)[:, None], np.sin(a)[:, None], np.cos(a)[None, :], np.sin(a)[None, :]
    dist = np.sqrt((r1 * c1 - off - r2 * c2) ** 2 + (r1 * s1 - r2 * s2) ** 2 + d ** 2)
    return MU0 / (4 * np.pi) * r1 * r2 * ((s1 * s2 + c1 * c2) / dist).sum() * (2 * np.pi / n) ** 2


def _filaments(c, nr, nz):
    """A coil's section as nr x nz circles: (radius, z) of each, m, and the cell's sides."""
    w, t = (c['outer'] - c['inner']) * 1e-3, c['t'] * 1e-3
    r = c['inner'] * 1e-3 + (np.arange(nr) + 0.5) * w / nr
    z = c['z'] * 1e-3 + (np.arange(nz) + 0.5) * t / nz
    R, Z = np.meshgrid(r, z, indexing='ij')
    return R.ravel(), Z.ravel(), w / nr, t / nz


def air_core(coils, cells=0.25):
    """The inductance matrix (H) of the coils in air: each coil's section cut into circles about `cells` mm apart,
    their mutual inductances summed (Maxwell's formula; Neumann's integral between coils off one axis), a circle's
    own from its cell's geometric mean distance."""
    n = len(coils)
    L = np.zeros((n, n))
    fil = []
    for c in coils:
        w, t = c['outer'] - c['inner'], c['t']
        fil.append(_filaments(c, max(2, min(60, int(np.ceil(w / cells)))), max(1, min(12, int(np.ceil(t / cells))))))
    for i, ci in enumerate(coils):
        r, z, a, b = fil[i]
        M = _maxwell(r[:, None], r[None, :], z[:, None] - z[None, :] + np.eye(len(r)))   # the diagonal is replaced
        np.fill_diagonal(M, MU0 * r * (np.log(8 * r / (0.2235 * (a + b))) - 2))
        L[i, i] = ci['turns'] ** 2 * M.mean()
        for j in range(i):
            cj = coils[j]
            off = float(np.hypot(ci.get('x', 0.0) - cj.get('x', 0.0), ci.get('y', 0.0) - cj.get('y', 0.0))) * 1e-3
            if off < 1e-9:
                r2, z2, _, _ = fil[j]
                M = _maxwell(r[:, None], r2[None, :], z[:, None] - z2[None, :]).mean()
            else:                                                # fewer circles: the integral is the cost
                ra, za, _, _ = _filaments(ci, 12, 1)
                rb, zb, _, _ = _filaments(cj, 12, 1)
                M = np.mean([_neumann(p, q, off, u - v) for p, u in zip(ra, za) for q, v in zip(rb, zb)])
            L[i, j] = L[j, i] = ci['turns'] * cj['turns'] * M
    return L


# -- the model ----------------------------------------------------------------------------------------------------

def _check(coils, plates):
    if not coils:
        raise ValueError('one coil or more')
    names = []
    for i, c in enumerate(coils):
        c.setdefault('name', f'coil{i + 1}')
        c.setdefault('x', 0.0), c.setdefault('y', 0.0), c.setdefault('z', 0.0)
        for key in ('turns', 'inner', 'outer', 't'):
            if key not in c or c[key] <= 0:
                raise ValueError(f"coil {c['name']}: {key} is needed, above 0")
        if c['outer'] <= c['inner']:
            raise ValueError(f"coil {c['name']}: outer radius above inner")
        names.append(c['name'])
    if len(set(names)) != len(names):
        raise ValueError('coil names must differ')
    for p in plates:
        p.setdefault('x', 0.0), p.setdefault('y', 0.0)
        if ('size' in p) == ('r' in p):
            raise ValueError('plate: size=AxB or r=RADIUS, one of them')
        for key in ('z', 't', 'mur'):
            if key not in p:
                raise ValueError(f'plate: {key} is needed')
        if p['t'] <= 0 or p['mur'] < 1:
            raise ValueError('plate: t above 0, mur 1 or more')


def _span(o):
    """(x0, y0, z0, x1, y1, z1) of a coil or plate."""
    a, b = ((o['outer'],) * 2 if 'outer' in o else (o['r'],) * 2 if 'r' in o else (o['size'][0] / 2, o['size'][1] / 2))
    return o['x'] - a, o['y'] - b, o['z'], o['x'] + a, o['y'] + b, o['z'] + o['t']


def mag_model(coils, plates, path, h=None, margin=6.0, log=print, threads=8, grow=0.35):
    """The mesh (.msh version 2, mm, ten-node tetrahedra): the coils' rings, the plates and an air ball `margin`
    times the parts' reach around them. h: mesh size at the coils and plates (None: from their sections); it grows
    by `grow` of the distance from them. Returns dict(bodies {body: (kind, index)}, tets, radius, sizes)."""
    import gmsh
    _check(coils, plates)
    spans = np.array([_span(o) for o in [*coils, *plates]])
    lo, hi = spans[:, :3].min(0), spans[:, 3:].max(0)
    centre = 0.5 * (lo + hi)
    reach = 0.5 * float(np.linalg.norm(hi - lo))
    radius = margin * max(reach, max(c['outer'] for c in coils))
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', threads)
        occ = gmsh.model.occ
        parts, sizes = [], []
        for c in coils:
            ring, _ = occ.cut([(3, occ.addCylinder(c['x'], c['y'], c['z'], 0, 0, c['t'], c['outer']))],
                              [(3, occ.addCylinder(c['x'], c['y'], c['z'], 0, 0, c['t'], c['inner']))])
            parts.append(ring[0])
            w = c['outer'] - c['inner']
            sizes.append(h or float(np.clip(min(w, c['t']), c['outer'] / 30, c['outer'] / 8)))
        for p in plates:
            if 'r' in p:
                parts.append((3, occ.addCylinder(p['x'], p['y'], p['z'], 0, 0, p['t'], p['r'])))
                reach_p = p['r']
            else:
                a, b = p['size']
                parts.append((3, occ.addBox(p['x'] - a / 2, p['y'] - b / 2, p['z'], a, b, p['t'])))
                reach_p = max(a, b) / 2
            sizes.append(h or float(np.clip(p['t'], reach_p / 25, reach_p / 6)))
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                both, _ = occ.intersect([a], [b], removeObject=False, removeTool=False)
                if both and occ.getMass(3, both[0][1]) > 1e-9:
                    raise ValueError('a coil and a plate, or two of them, overlap')
                occ.remove(both, recursive=True)
        air = (3, occ.addSphere(*centre, radius))
        _, maps = occ.fragment([air], parts)
        occ.synchronize()
        owned = {tag for m in maps[1:] for _, tag in m}
        bodies = {AIR: ('air', 0)}
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in maps[0] if tag not in owned], AIR)
        for i, m in enumerate(maps[1:]):
            body = COIL0 + i
            gmsh.model.addPhysicalGroup(3, [tag for _, tag in m], body)
            bodies[body] = ('coil', i) if i < len(coils) else ('plate', i - len(coils))
        ball = [tag for _, tag in gmsh.model.getBoundary(gmsh.model.getEntities(3), combined=True, oriented=False)]
        gmsh.model.addPhysicalGroup(2, ball, OUTER)
        fields = []
        for m, size in zip(maps[1:], sizes):                     # the size at each part, growing with the distance
            faces = [tag for _, tag in gmsh.model.getBoundary(m, combined=False, oriented=False)]
            dist = gmsh.model.mesh.field.add('Distance')
            gmsh.model.mesh.field.setNumbers(dist, 'SurfacesList', faces)
            gmsh.model.mesh.field.setNumber(dist, 'Sampling', 60)
            far = gmsh.model.mesh.field.add('MathEval')
            gmsh.model.mesh.field.setString(far, 'F', f'{size} + {grow} * F{dist}')
            fields.append(far)
        low = gmsh.model.mesh.field.add('Min')
        gmsh.model.mesh.field.setNumbers(low, 'FieldsList', fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(low)
        for name in ('MeshSizeExtendFromBoundary', 'MeshSizeFromPoints', 'MeshSizeFromCurvature'):
            gmsh.option.setNumber('Mesh.' + name, 0)
        gmsh.option.setNumber('Mesh.MeshSizeMax', radius / 3)
        gmsh.option.setNumber('Mesh.Algorithm3D', 10)
        gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.option.setNumber('Mesh.Binary', 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)                              # see sif(): second-order elements
        tets = int(sum(len(t) for t in gmsh.model.mesh.getElements(3)[1]))
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    log(f'mesh: {tets:,} tetrahedra; {min(sizes):.3g} mm at the parts, air to {radius:.4g} mm')
    return dict(bodies=bodies, tets=tets, radius=radius, sizes=sizes)


def sif(m, coils, plates, amps):
    """The .sif with coil i carrying amps[i] in each turn: CoilSolver lays the winding's current around each ring,
    the same density everywhere in it (a wound coil: `Normalize Coil Current`; without it the current crowds to the
    inner radius as in one solid turn), WhitneyAVSolver solves the static field, the energy goes to scalars.dat.

    Second-order edge elements on second-order tetrahedra. First-order ones are 2 to 4% low at any mesh this tool
    would make (a thin winding has one element through it), and second-order ones on a first-order mesh do not
    converge: the current is made divergence-free on the mesh's nodes, which must then be second-order too."""
    bodies = materials = components = ''
    for k, (body, (kind, i)) in enumerate(m['bodies'].items()):
        coil = kind == 'coil'
        bodies += (f"Body {k + 1}\n  Name = \"{kind}{'' if kind == 'air' else i + 1}\"\n  Target Bodies(1) = {body}\n"
                   f"  Equation = {2 if coil else 1}\n  Material = {k + 1}\nEnd\n")
        materials += f"Material {k + 1}\n  Relative Permeability = {plates[i]['mur'] if kind == 'plate' else 1:g}\nEnd\n"
        if coil:
            c = coils[i]
            density = c['turns'] * amps[i] / ((c['outer'] - c['inner']) * c['t'] * 1e-6)
            components += (f"Component {i + 1}\n  Name = \"{c['name']}\"\n  Coil Type = \"test\"\n"
                           f"  Master Bodies(1) = {k + 1}\n  Desired Current Density = Real {density:.9g}\n"
                           f"  Coil Normal(3) = 0 0 1\nEnd\n")
    return f"""Header
  Mesh DB "." "mesh"
End
Simulation
  Coordinate System = Cartesian 3D
  Coordinate Scaling = 1.0e-3
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Max Output Level = 5
End
Constants
  Permeability Of Vacuum = {MU0:.12g}
End
{bodies}Equation 1
  Active Solvers(2) = 2 3
End
Equation 2
  Active Solvers(3) = 1 2 3
End
{materials}{components}Solver 1
  Equation = "CoilSolver"
  Procedure = "CoilSolver" "CoilSolver"
  Coil Closed = Logical True
  Narrow Interface = Logical True
  Normalize Coil Current = Logical True
  Calculate Elemental Fields = Logical True
  Fix Input Current Density = Logical True
  Linear System Solver = "Iterative"
  Linear System Iterative Method = "BiCGStab"
  Linear System Preconditioning = "ILU1"
  Linear System Max Iterations = 2000
  Linear System Convergence Tolerance = 1.0e-9
End
Solver 2
  Equation = "MGDynamics"
  Procedure = "MagnetoDynamics" "WhitneyAVSolver"
  Use Elemental CoilCurrent = Logical True
  Fix Input Current Density = Logical True
  Use Piola Transform = Logical True
  Quadratic Approximation = Logical True
  Nonlinear System Max Iterations = 1
  Linear System Solver = "Iterative"
  Linear System Iterative Method = "BiCGStabl"
  BiCGStabl Polynomial Degree = 4
  Linear System Preconditioning = "none"
  Linear System Max Iterations = 20000
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Abort Not Converged = Logical True
End
Solver 3
  Equation = "MGDynamicsCalc"
  Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"
  Linear System Solver = "Iterative"
  Linear System Symmetric = Logical True
  Linear System Iterative Method = "CG"
  Linear System Preconditioning = "ILU0"
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-8
  Discontinuous Bodies = Logical True
End
Solver 4
  Exec Solver = After All
  Equation = "SaveScalars"
  Procedure = "SaveData" "SaveScalars"
  Filename = "scalars.dat"
End
Boundary Condition 1
  Target Boundaries(1) = {OUTER}
  AV {{e}} = 0.0
  AV = 0.0
  jfix = Real 0.0
End
"""


def solve(coils, plates=(), out='mag', h=None, margin=6.0, log=print):
    """The coils' inductance matrix beside the plates. coils: dicts(name, turns, inner, outer, z, t, x, y), mm;
    plates: dicts(z, t, mur, size=(a, b) or r, x, y). Returns the dict written to <out>/mag.json: names, L (H), k,
    air (the coils without the plates, air_core), tets, seconds."""
    ok, why = elmer.found()
    if not ok:
        raise RuntimeError(why)
    coils, plates = [dict(c) for c in coils], [dict(p) for p in plates]
    _check(coils, plates)
    out = Path(out)
    case = out / 'elmer'
    case.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    m = mag_model(coils, plates, case / 'model.msh', h=h, margin=margin, log=log)
    mesh_s = time.perf_counter() - t
    elmer.elmergrid(case / 'model.msh', case / 'mesh')
    t = time.perf_counter()
    n = len(coils)

    def energy(amps):
        elmer.solve(case, sif(m, coils, plates, amps))
        s = elmer.scalars(case / 'scalars.dat')
        return float(next(v for c, v in s.items() if 'electromagnetic field energy' in c.lower()))
    L = np.zeros((n, n))
    for i in range(n):                                           # each coil alone, 1 A: W = L / 2
        L[i, i] = 2 * energy([float(j == i) for j in range(n)])
        log(f"{coils[i]['name']}: {L[i, i] * 1e6:.4g} uH")
    for i in range(n):                                           # each pair: W = (L1 + L2) / 2 + M
        for j in range(i):
            L[i, j] = L[j, i] = energy([float(q in (i, j)) for q in range(n)]) - 0.5 * (L[i, i] + L[j, j])
    d = np.sqrt(np.diag(L))
    result = dict(format=1, tool='kicad2ems_fem.elmer.mag', names=[c['name'] for c in coils], L=L.tolist(),
                  k=(L / np.outer(d, d)).tolist(),
                  air=air_core(coils).tolist(), coils=coils, plates=plates, tets=m['tets'], margin=margin,
                  radius=m['radius'], sizes=m['sizes'],
                  seconds=dict(mesh=round(mesh_s, 1), solve=round(time.perf_counter() - t, 1)))
    (out / 'mag.json').write_text(json.dumps(result, indent=1))
    return result


def _item(ap, text, flag):
    """'name:key=value,...' as a dict; size=AxB as a pair."""
    name, sep, rest = text.partition(':')
    out = dict(name=name.strip()) if sep else {}
    for part in re.split(r'[,\s]+', (rest if sep else text).strip()):     # commas or spaces between them
        key, _, v = part.partition('=')
        try:
            out[key.strip()] = tuple(float(q) for q in v.split('x')) if key.strip() == 'size' else float(v)
        except ValueError:
            ap.error(f'{flag} {text!r}: {part!r} is not key=number')
        if key.strip() == 'size' and len(out['size']) != 2:
            ap.error(f'{flag} {text!r}: size=AxB')
    return out


def parser():
    ap = argparse.ArgumentParser(prog='python -m kicad2ems_fem.elmer.mag', description=__doc__.split('\n\n')[0])
    ap.add_argument('--coil', action='append', required=True,
                    help='[NAME:] turns=N inner=R outer=R z=Z t=T [x=X y=Y] (mm, spaces or commas between; '
                         'repeat: a coil each)')
    ap.add_argument('--plate', action='append', help='z=Z t=T mur=MU size=AxB or r=R [x=X y=Y] (mm; repeat: a '
                                                     'ferrite plate each)')
    ap.add_argument('--out', default='mag', help='output folder')
    ap.add_argument('--h', type=float, help='mesh size at the coils and plates, mm (default: from their sections)')
    ap.add_argument('--margin', type=float, default=6.0, help="the air ball's radius over the parts' reach (default 6)")
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    say = lambda s: print('fem mag:', s, flush=True)
    try:
        r = solve([_item(ap, c, '--coil') for c in a.coil], [_item(ap, p, '--plate') for p in a.plate or []], a.out,
                  h=a.h, margin=a.margin, log=say)
    except (ValueError, RuntimeError) as exc:
        print(f'fem mag: error: {exc}', file=sys.stderr)
        return 2
    L, air = np.array(r['L']), np.array(r['air'])
    for i, name in enumerate(r['names']):
        say(f"{name}: {L[i, i] * 1e6:.4g} uH ({air[i, i] * 1e6:.4g} uH in air by the ring formulas"
            + (f", x{L[i, i] / air[i, i]:.3g} with the ferrite)" if r['plates'] else ')'))
    for i in range(len(L)):
        for j in range(i):
            say(f"{r['names'][j]} to {r['names'][i]}: M {L[i, j] * 1e6:.4g} uH, k {r['k'][i][j]:.4f} "
                f"(in air {air[i, j] * 1e6:.4g} uH, k {air[i, j] / np.sqrt(air[i, i] * air[j, j]):.4f})")
    say(f"{r['tets']:,} tetrahedra, {r['seconds']['solve']:.0f} s; wrote {Path(a.out) / 'mag.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
