"""
Capacitance between nets, by Palace's electrostatic solver on the real stack: the dielectric slabs inside the
board outline (finite, as built), air around them, the copper of each net a conductor.

    python -m kicad2ems_fem.palace.cap board.kicad_pcb --net SW --net VIN --net GND --out cap_sw
    r = solve(board, ['SW', 'VIN', 'GND'], 'cap_sw')      # r['C']: the Maxwell matrix (F), r['names']

- Each `--net` is one conductor (a terminal); `A+B` joins nets into one. Copper of other nets is left out, not
  floating: name what matters.
- Copper is a zero-thickness sheet on its layer, or with `--thick` a solid of its thickness (the outer layers
  outward): the side walls' capacitance, which is not small between close traces. Vias are solid cylinders.
- The outer faces of the air box carry no charge (`--far open`, the default: the conductors' charges sum to zero)
  or are grounded (`--far ground`: a shield `--margin` mm away, default half the board's longer side). The two
  bracket the capacitance between two conductors far from anything else, open from above; they differ by a few %
  on a small board in air and less on a dielectric. The capacitance to the surroundings needs `--far ground`
  and falls as 1 / margin.
- Mesh: `--h` at the copper edges (default: the smallest of the slab thicknesses and the gaps between conductors
  on a layer), growing to `--hmax` away from them. The charge peaks at the edges, so the answer falls towards
  the true one in proportion to `--h`: about 5% high at the default on a trace 1.5 slabs wide, 1% at a tenth of
  it. Refine `--h` until the answer stops moving.

The Maxwell matrix C: Q_i = sum_j C_ij V_j, so C_ii is conductor i's capacitance with every other grounded and
-C_ij the capacitance between i and j. Palace also writes the mutual (SPICE) form, r['Cm'].
"""
import json
import os
import time

import numpy as np
from shapely.geometry import box, shape

from ..geometry import copper_of, layers as layer_names, load_board, polygons, slab
from .model import _polys, _surface

DIEL0, AIR, OUTER, TERM0 = 1, 99, 101, 201


def _groups(nets):
    """[(name, {nets})] from ['A', 'B+C', ...]."""
    out = []
    for n in nets:
        if isinstance(n, str):
            out.append((n, {x.strip() for x in n.split('+') if x.strip()}))
        else:
            out.append(('+'.join(sorted(n)), set(n)))
    return out


def feature_size(board, groups, crop=None, tol=0.01):
    sizes = [l['z1'] - l['z0'] for l in board['stack']['layers'] if l['z1'] > l['z0']]
    for lay in layer_names(board):
        gs = [copper_of(board, nets, lay) for _, nets in groups]
        gs = [g if crop is None or g is None else g.intersection(crop) for g in gs]
        gs = [g for g in gs if g is not None and not g.is_empty]
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                d = gs[i].distance(gs[j])
                if d > 1e-4:
                    sizes.append(d)
    return float(min(sizes)) if sizes else 1.0


def cap_model(board, nets, path, thick=False, h=None, hmax=None, margin=None, crop=None, grade=3.0, tol=0.01,
              log=print, threads=8, binary=True):
    """Mesh the dielectric and the air around the conductors of `nets` and write it to `path` (gmsh 2.2). Returns
    dict(diel=[(attribute, er)], air, outer, terminals=[(name, attribute)], h, hmax, margin, tets, nodes)."""
    import gmsh
    groups = _groups(nets)
    outline = shape(board['outline']) if isinstance(board['outline'], dict) else board['outline']
    if crop is not None:
        outline = outline.intersection(crop)
    slabs = [l for l in board['stack']['layers'] if l['z1'] > l['z0']]
    names = layer_names(board)
    zs = {c['name']: c['z'] for c in board['stack']['copper']}
    x0, y0, x1, y1 = outline.bounds
    zb = min([l['z0'] for l in slabs] + [slab(board, l)[0] if thick else zs[l] for l in names])
    zt = max([l['z1'] for l in slabs] + [slab(board, l)[1] if thick else zs[l] for l in names])
    h = h or max(feature_size(board, groups, crop, tol), 0.02)
    hmax = hmax or max(max(x1 - x0, y1 - y0) / 6, 4 * h)
    margin = margin if margin is not None else max(10 * (zt - zb), 0.5 * max(x1 - x0, y1 - y0), 2.0)

    def end(z, side):                       # thick copper: a via ending on a layer goes through its slab
        if not thick:
            return z
        hit = [slab(board, l) for l in names if abs(zs[l] - z) < 1e-6]
        return (hit[0][0] if side < 0 else hit[0][1]) if hit else z

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', threads)
        gmsh.model.add('cap')
        occ = gmsh.model.occ
        ents, tags = [], {}

        def add(role, dimtags):
            tags.setdefault(role, []).extend(dimtags)
            ents.extend(dimtags)

        for i, l in enumerate(slabs):
            for poly in _polys(outline):
                ext = occ.extrude([(2, _surface(occ, poly, l['z0']))], 0, 0, l['z1'] - l['z0'])
                add(('diel', i), [d for d in ext if d[0] == 3])
        add('air', [(3, occ.addBox(x0 - margin, y0 - margin, zb - margin, x1 - x0 + 2 * margin,
                                   y1 - y0 + 2 * margin, zt - zb + 2 * margin))])
        for gi, (name, gnets) in enumerate(groups):
            n = 0
            for lay in names:
                g = copper_of(board, gnets, lay)
                if g is not None and crop is not None:
                    g = g.intersection(crop)
                for poly in polygons(g, tol):
                    n += 1
                    if thick:
                        za, zc = slab(board, lay)
                        ext = occ.extrude([(2, _surface(occ, poly, za))], 0, 0, zc - za)
                        add(('cu', gi), [d for d in ext if d[0] == 3])
                    else:
                        add(('cu', gi), [(2, _surface(occ, poly, zs[lay]))])
            for v in board.get('vias', []):
                if v.get('net') in gnets and v['z1'] > v['z0'] and \
                        (crop is None or crop.contains(box(v['x'] - v['r'], v['y'] - v['r'], v['x'] + v['r'], v['y'] + v['r']))):
                    za, zc = end(v['z0'], -1), end(v['z1'], 1)
                    add(('cu', gi), [(3, occ.addCylinder(v['x'], v['y'], za, 0, 0, zc - za, v['r']))])
            if not n:
                raise ValueError(f"no copper of {name}" + (' in the region' if crop is not None else ''))
        _, omap = occ.fragment(ents, [])
        occ.synchronize()
        role_of = {}
        for dt, kids in zip(ents, omap):
            for role, dts in tags.items():
                if dt in dts:
                    for kid in kids:
                        role_of.setdefault(kid, set()).add(role)

        def group_of(dt):
            return sorted({r[1] for r in role_of.get(dt, ()) if isinstance(r, tuple) and r[0] == 'cu'})

        vols = gmsh.model.getEntities(3)
        metal = {dt[1]: group_of(dt) for dt in vols if group_of(dt)}        # conductor volumes: left out of the mesh
        kept = {dt[1] for dt in vols} - set(metal)
        diel_v = {}
        for dt in vols:
            if dt[1] in kept:
                d = [r[1] for r in role_of.get(dt, ()) if isinstance(r, tuple) and r[0] == 'diel']
                if d:
                    diel_v.setdefault(d[0], []).append(dt[1])
        air_v = sorted(kept - {t for ts in diel_v.values() for t in ts})
        term = {gi: set() for gi in range(len(groups))}                     # each conductor's faces on the mesh
        for dt in gmsh.model.getEntities(2):
            for gi in group_of(dt):
                term[gi].add(dt[1])
        for v, gis in metal.items():
            for f in gmsh.model.getBoundary([(3, v)], oriented=False):
                for gi in gis:
                    term[gi].add(abs(f[1]))
        for gi in term:                                                     # only faces a meshed volume touches
            term[gi] = {s for s in term[gi] if kept.intersection(gmsh.model.getAdjacencies(2, s)[0])}
            if not term[gi]:
                raise RuntimeError(f"{groups[gi][0]}: its copper is lost in the mesh")
        rim = {gi: {abs(c[1]) for f in fs for c in gmsh.model.getBoundary([(2, f)], oriented=False)}
               for gi, fs in term.items()}
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if term[a] & term[b] or rim[a] & rim[b]:   # a shared face, or sheets edge to edge
                    raise ValueError(f"{groups[a][0]} and {groups[b][0]} touch: join them (A+B) or drop one")
        bb = (x0 - margin, y0 - margin, zb - margin, x1 + margin, y1 + margin, zt + margin)
        outer = []
        for _, s in gmsh.model.getEntities(2):
            b = gmsh.model.getBoundingBox(2, s)
            if any(abs(b[ax] - bb[ax + k]) < 1e-6 and abs(b[ax + 3] - bb[ax + k]) < 1e-6 for ax in range(3) for k in (0, 3)):
                outer.append(s)

        diel = []
        for i, l in enumerate(slabs):
            if i in diel_v:
                gmsh.model.addPhysicalGroup(3, diel_v[i], DIEL0 + i)
                diel.append((DIEL0 + i, float(l['er'])))
        gmsh.model.addPhysicalGroup(3, air_v, AIR)
        gmsh.model.addPhysicalGroup(2, outer, OUTER)
        terminals = []
        for gi, (name, _) in enumerate(groups):
            gmsh.model.addPhysicalGroup(2, sorted(term[gi]), TERM0 + gi)
            terminals.append((name, TERM0 + gi))

        edges = set().union(*rim.values())
        fd = gmsh.model.mesh.field
        dist = fd.add('Distance')
        fd.setNumbers(dist, 'CurvesList', sorted(edges))
        fd.setNumber(dist, 'Sampling', 40)
        thr = fd.add('Threshold')
        for k, v in dict(InField=dist, SizeMin=h, SizeMax=hmax, DistMin=h, DistMax=grade * hmax).items():
            fd.setNumber(thr, k, v)
        fd.setAsBackgroundMesh(thr)
        for k, v in {'Mesh.MeshSizeExtendFromBoundary': 0, 'Mesh.MeshSizeFromPoints': 0, 'Mesh.MeshSizeFromCurvature': 0,
                     'Mesh.MeshSizeMax': hmax * 2, 'Mesh.Algorithm': 6, 'Mesh.Algorithm3D': 10}.items():
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.option.setNumber('Mesh.Binary', int(binary))
        gmsh.option.setNumber('Mesh.SaveAll', 0)                # the conductors' insides are left out
        gmsh.write(str(path))
        nodes = len(gmsh.model.mesh.getNodes()[0])
        tets = sum(len(e) for t in kept for e in gmsh.model.mesh.getElements(3, t)[1])
    finally:
        gmsh.finalize()
    log(f"mesh: {tets:,} tetrahedra; h {h:.3g}..{hmax:.3g} mm, air margin {margin:.3g} mm, "
        f"{'solid' if thick else 'sheet'} copper")
    return dict(diel=diel, air=AIR, outer=OUTER, terminals=terminals, h=h, hmax=hmax, margin=margin, tets=tets,
                nodes=nodes)


def config(m, mesh='model.msh', order=2, far='open', out='postpro'):
    if far not in ('open', 'ground'):
        raise ValueError("far: 'open' or 'ground'")
    materials = [dict(Attributes=[a], Permittivity=er) for a, er in m['diel']] + \
                [dict(Attributes=[m['air']], Permittivity=1.0)]
    boundaries = dict(Terminal=[dict(Index=i + 1, Attributes=[a]) for i, (_, a) in enumerate(m['terminals'])])
    boundaries['ZeroCharge' if far == 'open' else 'Ground'] = dict(Attributes=[m['outer']])
    return {
        'Problem': dict(Type='Electrostatic', Verbose=2, Output=out),
        'Model': dict(Mesh=mesh, L0=1e-3),
        'Domains': dict(Materials=materials),
        'Boundaries': boundaries,
        'Solver': dict(Order=order, Device='CPU', Electrostatic=dict(Save=0),
                       Linear=dict(Type='BoomerAMG', KSPType='CG', Tol=1e-9, MaxIts=300)),
    }


def read_c(postpro, n, name='terminal-C.csv'):
    """An (n, n) matrix (F) from Palace's terminal-C.csv (Maxwell) or terminal-Cm.csv (mutual)."""
    from .run import table
    _, data = table(os.path.join(postpro, name))
    if data.shape != (n, n + 1):
        raise ValueError(f"{name}: {data.shape} values, expected {n} rows of 1 + {n}")
    return data[:, 1:]


def solve(board, nets, out, thick=False, h=None, hmax=None, margin=None, crop=None, order=2, far='open', np_=8,
          log=print):
    """The capacitance matrix between `nets` (each a conductor; 'A+B' joins nets). Returns dict(names, C, Cm (F),
    tets, h, hmax, margin, seconds) and writes it to <out>/cap.json; Palace's case is in <out>/palace."""
    from . import run as prun
    if isinstance(board, (str, os.PathLike)):
        board = load_board(board, {n for _, ns in _groups(nets) for n in ns})
    work = os.path.join(str(out), 'palace')
    os.makedirs(work, exist_ok=True)
    t = time.time()
    m = cap_model(board, nets, os.path.join(work, 'model.msh'), thick=thick, h=h, hmax=hmax, margin=margin,
                  crop=crop, log=log)
    mesh_s = time.time() - t
    with open(os.path.join(work, 'palace.json'), 'w') as fh:
        json.dump(config(m, order=order, far=far), fh, indent=1)
    t = time.time()
    prun.run(work, 'palace.json', np=np_, log=log)
    n = len(m['terminals'])
    post = os.path.join(work, 'postpro')
    C, Cm = read_c(post, n), read_c(post, n, 'terminal-Cm.csv')
    r = dict(tool='kicad2ems_fem.palace.cap', names=[name for name, _ in m['terminals']], C=C.tolist(), Cm=Cm.tolist(),
             thick=bool(thick), far=far,
             order=order, tets=m['tets'], h=m['h'], hmax=m['hmax'], margin=m['margin'], mesh_s=round(mesh_s, 1),
             seconds=round(time.time() - t, 1))
    with open(os.path.join(str(out), 'cap.json'), 'w') as fh:
        json.dump(r, fh, indent=1)
    return r


def report(r):
    """The matrix as text, pF."""
    names, C = r['names'], np.asarray(r['C']) * 1e12
    w = max(8, *(len(n) for n in names)) + 2
    lines = [' ' * w + ''.join(f"{n:>{w}}" for n in names)]
    lines += [f"{n:<{w}}" + ''.join(f"{v:>{w}.4g}" for v in row) for n, row in zip(names, C)]
    return '\n'.join(lines)


def parser():
    import argparse
    ap = argparse.ArgumentParser(prog='kicad2ems-fem-cap', description=__doc__.split('\n\n')[0])
    ap.add_argument('board', help='a .kicad_pcb, or a kicad2ems <name>_board.json')
    ap.add_argument('--net', action='append', required=True, help='a conductor: a net, or nets joined as A+B; repeat')
    ap.add_argument('--out', required=True, help='the folder of the results')
    ap.add_argument('--thick', action='store_true', help='copper as solids of its thickness, not sheets')
    ap.add_argument('--h', type=float, help='mesh size at the copper edges, mm')
    ap.add_argument('--hmax', type=float, help='mesh size away from the copper, mm')
    ap.add_argument('--margin', type=float, help='air around the board, mm')
    ap.add_argument('--region', help='x0,y0,x1,y1 (mm, model coordinates): only the board inside it')
    ap.add_argument('--far', choices=('open', 'ground'), default='open', help="the air box's faces (default open)")
    ap.add_argument('--order', type=int, default=2, help="Palace's element order (default 2)")
    ap.add_argument('--np', type=int, default=8, help="Palace's MPI processes (default 8)")
    return ap


def main(argv=None):
    from .. import children
    children.guard()                                        # its solvers stopped with it
    a = parser().parse_args(argv)
    crop = box(*[float(v) for v in a.region.split(',')]) if a.region else None
    r = solve(a.board, a.net, a.out, thick=a.thick, h=a.h, hmax=a.hmax, margin=a.margin, crop=crop, order=a.order,
              far=a.far, np_=a.np)
    print(f"capacitance (Maxwell), pF: {r['tets']:,} tetrahedra, {r['seconds']} s")
    print(report(r))


if __name__ == '__main__':
    main()
