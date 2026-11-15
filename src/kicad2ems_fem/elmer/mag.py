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

With --freq F (R-E21), each coil alone again in the time-harmonic field at F: its inductance there and its series
resistance from the ferrite's loss (a plate's mur_im, mu''), from the eddy currents of metal plates (sigma=: a
shield, a battery's can, as a surface impedance) and its winding's own (r=, given), and its Q: `ac` in mag.json.

What is left out: the winding's skin and proximity loss (the turns' current is spread over the section), the
ferrite's saturation, a metal plate thinner than a few skin depths, and coils drawn as board copper (pcbqs has
those in air).
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
METAL0 = 101                                                     # a metal plate's faces (a hole in the air), j-th
FERRITE = 'ferrite'                                              # a lossy ferrite plate's body mask, j-th from 1


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
        if p.get('sigma'):                                       # a metal plate: a shield, a battery's can
            p.setdefault('mur', 1.0)
            if p['sigma'] <= 0 or p.get('mur_im'):
                raise ValueError('plate: sigma above 0, and no mur_im on a metal plate')
        for key in ('z', 't', 'mur'):
            if key not in p:
                raise ValueError(f'plate: {key} is needed')
        if p['t'] <= 0 or p['mur'] < 1 or p.get('mur_im', 0) < 0:
            raise ValueError('plate: t above 0, mur 1 or more, mur_im 0 or more')


def _span(o):
    """(x0, y0, z0, x1, y1, z1) of a coil or plate."""
    a, b = ((o['outer'],) * 2 if 'outer' in o else (o['r'],) * 2 if 'r' in o else (o['size'][0] / 2, o['size'][1] / 2))
    return o['x'] - a, o['y'] - b, o['z'], o['x'] + a, o['y'] + b, o['z'] + o['t']


def mag_model(coils, plates, path, h=None, margin=6.0, log=print, threads=8, grow=0.35, metal=False):
    """The mesh (.msh version 2, mm, ten-node tetrahedra): the coils' rings, the plates and an air ball `margin`
    times the parts' reach around them. h: mesh size at the coils and plates (None: from their sections); it grows
    by `grow` of the distance from them. A metal plate (sigma) is a hole in the air whose faces are boundary
    METAL0 + j with metal=True (the harmonic run's surface impedance), and is left out with metal=False (at DC a
    non-magnetic metal is transparent). Returns dict(bodies {body: (kind, index)}, metals {boundary: index}, tets,
    radius, sizes)."""
    import gmsh
    _check(coils, plates)
    holes = [j for j, p in enumerate(plates) if p.get('sigma')] if metal else []
    keep = [j for j, p in enumerate(plates) if not p.get('sigma')]
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
        def slab(p):
            if 'r' in p:
                return (3, occ.addCylinder(p['x'], p['y'], p['z'], 0, 0, p['t'], p['r'])), p['r']
            a, b = p['size']
            return (3, occ.addBox(p['x'] - a / 2, p['y'] - b / 2, p['z'], a, b, p['t'])), max(a, b) / 2
        kinds = [('coil', i) for i in range(len(coils))]
        for j in keep:
            dt, reach_p = slab(plates[j])
            parts.append(dt)
            kinds.append(('plate', j))
            sizes.append(h or float(np.clip(plates[j]['t'], reach_p / 25, reach_p / 6)))
        cutters = []
        for j in holes:
            dt, reach_p = slab(plates[j])
            cutters.append(dt)
            sizes.append(h or float(np.clip(plates[j]['t'], reach_p / 25, reach_p / 6)))
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                both, _ = occ.intersect([a], [b], removeObject=False, removeTool=False)
                if both and occ.getMass(3, both[0][1]) > 1e-9:
                    raise ValueError('a coil and a plate, or two of them, overlap')
                occ.remove(both, recursive=True)
        for i, a in enumerate(cutters):
            for b in parts + cutters[i + 1:]:
                both, _ = occ.intersect([a], [b], removeObject=False, removeTool=False)
                if both and occ.getMass(3, both[0][1]) > 1e-9:
                    raise ValueError('a metal plate overlaps a coil or another plate')
                occ.remove(both, recursive=True)
        air = (3, occ.addSphere(*centre, radius))
        if cutters:
            (air,), _ = occ.cut([air], cutters)
        _, maps = occ.fragment([air], parts)
        occ.synchronize()
        owned = {tag for m in maps[1:] for _, tag in m}
        bodies = {AIR: ('air', 0)}
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in maps[0] if tag not in owned], AIR)
        for i, m in enumerate(maps[1:]):
            body = COIL0 + i
            gmsh.model.addPhysicalGroup(3, [tag for _, tag in m], body)
            bodies[body] = kinds[i]
        outside = [tag for _, tag in gmsh.model.getBoundary(gmsh.model.getEntities(3), combined=True, oriented=False)]
        metals, hole_faces = {}, []
        for k, j in enumerate(holes):                            # a hole's faces: those inside its box
            x0, y0, z0, x1, y1, z1 = _span(plates[j])
            e = 1e-6 * radius
            faces = [tag for _, tag in gmsh.model.getEntitiesInBoundingBox(x0 - e, y0 - e, z0 - e, x1 + e, y1 + e,
                                                                           z1 + e, 2)]
            gmsh.model.addPhysicalGroup(2, faces, METAL0 + k)
            metals[METAL0 + k] = j
            hole_faces.append(faces)
        ball = [t for t in outside if not any(t in f for f in hole_faces)]
        gmsh.model.addPhysicalGroup(2, ball, OUTER)
        fields = []
        for m, size in zip(maps[1:] + [[(2, t) for t in f] for f in hole_faces], sizes):   # the size at each part,
            faces = ([tag for d, tag in m if d == 2] if m and m[0][0] == 2 else           # growing with distance
                     [tag for _, tag in gmsh.model.getBoundary(m, combined=False, oriented=False)])
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
    log(f'mesh: {tets:,} tetrahedra; {min(sizes):.3g} mm at the parts, air to {radius:.4g} mm'
        + (f'; {len(holes)} metal plate(s) as surface impedances' if holes else ''))
    return dict(bodies=bodies, metals=metals, tets=tets, radius=radius, sizes=sizes)


def sif(m, coils, plates, amps, freq=None):
    """The .sif with coil i carrying amps[i] in each turn: CoilSolver lays the winding's current around each ring,
    the same density everywhere in it (a wound coil: `Normalize Coil Current`; without it the current crowds to the
    inner radius as in one solid turn), WhitneyAVSolver solves the static field, the energy goes to scalars.dat.

    Second-order edge elements on second-order tetrahedra. First-order ones are 2 to 4% low at any mesh this tool
    would make (a thin winding has one element through it), and second-order ones on a first-order mesh do not
    converge: the current is made divergence-free on the mesh's nodes, which must then be second-order too.

    freq (Hz): the time-harmonic field at that frequency instead (WhitneyAVHarmonicSolver; R-E21), the current
    amps[i] its peak, a ferrite plate at its real mu' (its loss: _ac). A metal plate (sigma) is a hole whose faces
    are a surface impedance (Layer Electric Conductivity: a skin depth well under its thickness), its loss
    Elmer's surface current power."""
    bodies = materials = components = saves = ''
    n_save = 0
    for k, (body, (kind, i)) in enumerate(m['bodies'].items()):
        coil = kind == 'coil'
        lossy = bool(freq) and kind == 'plate' and bool(plates[i].get('mur_im'))
        bodies += (f"Body {k + 1}\n  Name = \"{kind}{'' if kind == 'air' else i + 1}\"\n  Target Bodies(1) = {body}\n"
                   f"  Equation = {2 if coil else 1}\n  Material = {k + 1}\n"
                   + (f"  {FERRITE}{i + 1} = Logical True\n" if lossy else '') + "End\n")
        if lossy:                                            # its |B|^2 integral (m^3 T^2), for its loss (_ac)
            n_save += 1
            saves += (f'  Variable {n_save} = "Magnetic Flux Density E"\n  Operator {n_save} = "body int square"\n'
                      f'  Mask Name {n_save} = "{FERRITE}{i + 1}"\n')
        mur = plates[i]['mur'] if kind == 'plate' else 1.0
        materials += f"Material {k + 1}\n  Relative Permeability = {mur:.12g}\nEnd\n"
        if coil:
            c = coils[i]
            density = c['turns'] * amps[i] / ((c['outer'] - c['inner']) * c['t'] * 1e-6)
            components += (f"Component {i + 1}\n  Name = \"{c['name']}\"\n  Coil Type = \"test\"\n"
                           f"  Master Bodies(1) = {k + 1}\n  Desired Current Density = Real {density:.9g}\n"
                           f"  Coil Normal(3) = 0 0 1\nEnd\n")
    harm = bool(freq)
    w = 2 * np.pi * float(freq or 0)
    av = 'AV[AV re:1 AV im:1]' if harm else 'AV'
    zero = ('  AV re {e} = 0.0\n  AV im {e} = 0.0\n  AV re = 0.0\n  AV im = 0.0\n' if harm else
            '  AV {e} = 0.0\n  AV = 0.0\n')
    metal_bcs = ''
    for n_, (b, j) in enumerate((m.get('metals') or {}).items() if harm else ()):
        metal_bcs += (f"Boundary Condition {n_ + 2}\n  Target Boundaries(1) = {b}\n"
                      f"  Layer Electric Conductivity = Real {plates[j]['sigma']:.9g}\n"
                      f"  Layer Relative Permeability = Real {plates[j].get('mur', 1.0):g}\nEnd\n")
    solver2 = (f'  Procedure = "MagnetoDynamics" "WhitneyAVHarmonicSolver"\n  Variable = "{av}"\n'
               f'  Angular Frequency = Real {w:.12g}\n' if harm else
               '  Procedure = "MagnetoDynamics" "WhitneyAVSolver"\n')
    calc = (f'  Angular Frequency = Real {w:.12g}\n' + ('  Calculate Magnetic Flux Density = Logical True\n  Calculate Elemental Fields = Logical True\n'
                                                   if saves else '')
            if harm else '')
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
{solver2}  Use Elemental CoilCurrent = Logical True
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
  Potential Variable = String "AV"
{calc}  Linear System Solver = "Iterative"
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
{saves}End
Boundary Condition 1
  Target Boundaries(1) = {OUTER}
{zero}  jfix = Real 0.0
End
{metal_bcs}"""


def solve(coils, plates=(), out='mag', h=None, margin=6.0, log=print, freq=None):
    """The coils' inductance matrix beside the plates. coils: dicts(name, turns, inner, outer, z, t, x, y, r: the
    winding's own resistance, ohm, at freq), mm; plates: dicts(z, t, mur, mur_im, sigma, size=(a, b) or r, x, y).
    Returns the dict written to <out>/mag.json: names, L (H), k, air (the coils without the plates, air_core),
    tets, seconds; with freq (Hz), `ac`: each coil alone at that frequency (R-E21), dict(name, L, R_ferrite,
    R_metal, R_winding, R, Q), from the harmonic field: L from its energy, R twice the time-averaged loss of a 1 A
    peak current."""
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
    said = []                                                       # the last log: which Elmer it was

    def energy(amps):
        said[:] = [elmer.solve(case, sif(m, coils, plates, amps))]
        s = elmer.scalars(case / 'scalars.dat')
        return float(next(v for c, v in s.items() if 'electromagnetic field energy' in c.lower()))
    L = np.zeros((n, n))
    for i in range(n):                                           # each coil alone, 1 A: W = L / 2
        L[i, i] = 2 * energy([float(j == i) for j in range(n)])
        log(f"{coils[i]['name']}: {L[i, i] * 1e6:.4g} uH")
    for i in range(n):                                           # each pair: W = (L1 + L2) / 2 + M
        for j in range(i):
            L[i, j] = L[j, i] = energy([float(q in (i, j)) for q in range(n)]) - 0.5 * (L[i, i] + L[j, j])
    ac = None
    if freq:
        ac, ac_tets = _ac(coils, plates, out, m, case, float(freq), h, margin, log, said)
    d = np.sqrt(np.diag(L))
    result = dict(format=1, tool='kicad2ems_fem.elmer.mag', engine=elmer.engine(said[0] if said else ''), names=[c['name'] for c in coils], L=L.tolist(),
                  k=(L / np.outer(d, d)).tolist(),
                  air=air_core(coils).tolist(), coils=coils, plates=plates, tets=m['tets'], margin=margin,
                  radius=m['radius'], sizes=m['sizes'],
                  seconds=dict(mesh=round(mesh_s, 1), solve=round(time.perf_counter() - t, 1)))
    if ac is not None:
        result.update(freq=float(freq), ac=ac, ac_tets=ac_tets)
    (out / 'mag.json').write_text(json.dumps(result, indent=1))
    return result


def _ac(coils, plates, out, m, case, freq, h, margin, log, said):
    """Each coil alone, 1 A peak, in the harmonic field at freq: ([dict(name, L, R_ferrite, R_metal, R_winding, R,
    Q)], tetrahedra). With metal plates on its own mesh (them as holes, see mag_model), else on the static one.

    A ferrite's loss is (w / 2) mu0 mu'' |H|^2 = pi f mu'' |B|^2 / (mu0 mu'^2) a unit volume, the time average,
    from the integral of |B|^2 over its body that the run saves (sif: first order in mu'' / mu', its field that of
    mu'). Elmer's own harmonic loss reads 0 in its 26.2, its test case's too; and differencing two runs' energies
    (mu' moved by 5 %) was the solver's noise, a mu' of 2000 holding ~0.15 % of the energy."""
    if any(p.get('sigma') for p in plates):
        case = out / 'elmer_ac'
        case.mkdir(parents=True, exist_ok=True)
        m = mag_model(coils, plates, case / 'model.msh', h=h, margin=margin, log=log, metal=True)
        elmer.elmergrid(case / 'model.msh', case / 'mesh')
    w = 2 * np.pi * freq
    rows = []
    for i, c in enumerate(coils):
        said[:] = [elmer.solve(case, sif(m, coils, plates, [float(j == i) for j in range(len(coils))], freq=freq))]
        sc = elmer.scalars(case / 'scalars.dat')

        def get(name):
            return float(next((v for k, v in sc.items() if name in k.lower()), 0.0))
        W = get('electromagnetic field energy')
        L = 2 * W
        r_metal = 2 * (get('surface current power') + get('eddy current power'))
        r_fer = 0.0
        for j, pl in enumerate(plates):
            if pl.get('mur_im'):
                b2 = get(f'mask {FERRITE}{j + 1}')                  # SaveScalars' "... mask ferrite1" column
                r_fer += 2 * np.pi * freq * pl['mur_im'] / (MU0 * pl['mur'] ** 2) * b2     # (m^3: Coordinate Scaling)
        r_wind = float(c.get('r') or 0.0)
        R = r_fer + r_metal + r_wind
        rows.append(dict(name=c['name'], L=L, R_ferrite=r_fer, R_metal=r_metal, R_winding=r_wind, R=R,
                         Q=w * L / R if R > 0 else None))
        log(f"{c['name']} at {freq / 1e3:g} kHz: L {L * 1e6:.4g} uH, R {R * 1e3:.4g} mohm (ferrite {r_fer * 1e3:.3g}, "
            f"metal {r_metal * 1e3:.3g}, winding {r_wind * 1e3:.3g})" + (f", Q {w * L / R:.4g}" if R > 0 else ''))
    return rows, m['tets']


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
                    help='[NAME:] turns=N inner=R outer=R z=Z t=T [x=X y=Y] [r=OHM: its winding\'s resistance at '
                         '--freq] (mm, spaces or commas between; repeat: a coil each)')
    ap.add_argument('--plate', action='append', help='z=Z t=T mur=MU [mur_im=MU2: its loss, mu\'\'] size=AxB or r=R '
                                                     '[x=X y=Y] (mm; a ferrite plate), or sigma=S/m instead of mur (a '
                                                     'metal plate: a shield, a battery); repeat: a plate each')
    ap.add_argument('--out', default='mag', help='output folder')
    ap.add_argument('--h', type=float, help='mesh size at the coils and plates, mm (default: from their sections)')
    ap.add_argument('--margin', type=float, default=6.0, help="the air ball's radius over the parts' reach (default 6)")
    ap.add_argument('--freq', type=float, help="also each coil at this frequency, Hz (R-E21): its L, R and Q with the "
                                               "ferrite's loss (a plate's mur_im), the eddy currents of metal plates "
                                               "(sigma=S/m) and the winding's own r=ohm")
    return ap


def main(argv=None):
    from .. import children
    children.guard()                                        # its solvers stopped with it
    ap = parser()
    a = ap.parse_args(argv)
    say = lambda s: print('fem mag:', s, flush=True)
    try:
        r = solve([_item(ap, c, '--coil') for c in a.coil], [_item(ap, p, '--plate') for p in a.plate or []], a.out,
                  h=a.h, margin=a.margin, log=say, freq=a.freq)
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
