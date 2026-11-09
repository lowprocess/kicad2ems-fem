"""
The board as a Palace model: a tetrahedral gmsh mesh of the dielectric slabs inside an air box, and the
attributes the config refers to (config.py).

    mdl = build(board, fmax=10e9, path='model.msh')       # Model: attributes, sizes, counts

Copper is a zero-thickness PEC sheet on each copper layer at its height (cu_z); a via is a hole through the
dielectric whose wall is PEC (its barrel, closed by the pads it ends on). Lumped ports are rectangles: a 'z'
port a vertical sheet across the trace from the reference copper to the trace, a 'gap' port a sheet in the
copper plane across its gap (board.Port.box). The air box is the board's extent grown by `margin` on every
side, with an absorbing (first-order) boundary on its faces. Units: mm (the config's L0 = 1e-3).

Mesh size: `hmin` at the copper edges (a fraction of the smallest feature: the gap between copper on a layer, a
port's width, a slab's thickness), growing with distance to `hmax` (a fraction of the wavelength at fmax in
the densest dielectric); Palace's second-order elements carry the field between.
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

C0 = 299792458.0
# physical-group attributes (Palace's mesh attributes)
DIEL0, AIR = 1, 99                  # dielectric slab i: DIEL0 + i
PEC, ABSORB, PORT0 = 100, 101, 201  # PEC: the via walls; port i (0-based, board order): PORT0 + i
CU0 = 110                           # copper layer i's sheets (board.cu_z order): CU0 + i (its own thickness when lossy)


@dataclass
class Model:
    path: str
    diel: list                      # [(attribute, er, tand)] one per slab
    air: int
    pec: object                     # the metal's attributes (a list; an int in older callers): PEC when lossless
    absorb: int
    ports: list                     # [dict(attr, index, R, direction, name)]
    hmin: float
    hmax: float
    margin: float
    nodes: int = 0
    tets: int = 0
    sheets: list = field(default_factory=list)     # [(attribute, thickness mm)]: each copper layer's sheets
    via_wall: int = None                            # the via walls' attribute (PEC), or None without vias
    notes: list = field(default_factory=list)


def _polys(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, 'geoms', []) if isinstance(g, Polygon)]


def _ring(occ, coords, z):
    """A closed curve loop through a ring's points (the last repeats the first) at height z."""
    pts = np.asarray(coords, float)[:-1]
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-6]          # drop repeated points
    pts = pts[keep]
    if len(pts) > 2 and np.hypot(*(pts[-1] - pts[0])) < 1e-6:
        pts = pts[:-1]
    p = [occ.addPoint(float(x), float(y), z) for x, y in pts]
    lines = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    return occ.addCurveLoop(lines)


def _surface(occ, poly, z):
    """A plane surface of a shapely polygon (with its holes) at height z."""
    loops = [_ring(occ, poly.exterior.coords, z)] + [_ring(occ, h.coords, z) for h in poly.interiors]
    return occ.addPlaneSurface(loops)


def feature_size(board):
    """The smallest feature the mesh must resolve, mm: the gap between separate copper on a layer, a port's
    width, a slab's thickness."""
    sizes = [l['z1'] - l['z0'] for l in board.layers if l['z1'] > l['z0']]
    sizes += [p.w for p in board.ports if p.w > 0]
    sizes += [abs(p.gap[1] - p.gap[0]) for p in board.ports if p.gap is not None]
    for polys in board.copper.values():
        ps = [q for g in polys for q in _polys(g)]
        if len(ps) > 60:                            # many pieces: sample the gaps of the first ones
            ps = ps[:60]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                d = ps[i].distance(ps[j])
                if d > 1e-4:
                    sizes.append(d)
    return float(min(sizes)) if sizes else 1.0


def build(board, fmax, path, margin=None, hmin=None, hmax=None, per_feature=1.0, per_wave=8.0, grade=3.0,
          log=print, threads=8):
    """Mesh the board for Palace and write it to `path` (gmsh 2.2, binary). fmax in Hz."""
    import gmsh
    feat = feature_size(board)
    er_max = max([l['er'] for l in board.layers] + [1.0])
    lam0 = C0 / fmax * 1e3                                      # mm
    hmin = hmin or max(feat / per_feature, 0.02)
    hmax = hmax or max(lam0 / per_wave / np.sqrt(er_max), 4 * hmin)
    x0, y0, x1, y1 = board.outline.bounds
    zb = min([l['z0'] for l in board.layers] + list(board.cu_z.values()))
    zt = max([l['z1'] for l in board.layers] + list(board.cu_z.values()))
    margin = margin if margin is not None else max(lam0 / 4, 10 * (zt - zb))

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', threads)
        gmsh.model.add('board')
        occ = gmsh.model.occ
        ents, tags = [], {}                                     # tags: role -> [(dim, tag)] before the fragment

        def add(role, dimtags):
            tags.setdefault(role, []).extend(dimtags)
            ents.extend(dimtags)

        for i, l in enumerate(board.layers):                   # the dielectric slabs, the outline extruded
            if l['z1'] <= l['z0']:
                continue
            for poly in _polys(board.outline):
                s = _surface(occ, poly, l['z0'])
                ext = occ.extrude([(2, s)], 0, 0, l['z1'] - l['z0'])
                add(('diel', i), [d for d in ext if d[0] == 3])
        add('air', [(3, occ.addBox(x0 - margin, y0 - margin, zb - margin, x1 - x0 + 2 * margin,
                                   y1 - y0 + 2 * margin, zt - zb + 2 * margin))])
        for lay, polys in board.copper.items():                 # the copper sheets
            z = board.cu_z[lay]
            for g in polys:
                for poly in _polys(g):
                    add(('cu', lay), [(2, _surface(occ, poly.buffer(0), z))])
        for v in board.vias:                                    # the via barrels (holes)
            x, y, r, za, zc = v[:5]
            if zc > za:
                add('via', [(3, occ.addCylinder(x, y, za, 0, 0, zc - za, r))])
        for k, p in enumerate(board.ports):                     # the lumped-port sheets
            s, e = p.box()
            if p.kind == 'z':                                   # vertical: across the trace at c[axis]
                r = _vertical(occ, p.axis, s, e)
            else:                                               # in the copper plane, across the gap
                r = occ.addRectangle(s[0], s[1], s[2], e[0] - s[0], e[1] - s[1])
            add(('port', k), [(2, r)])
        out, omap = occ.fragment(ents, [])
        occ.synchronize()
        role_of = {}                                            # fragment piece -> roles
        for dt, kids in zip(ents, omap):
            for role, dts in tags.items():
                if dt in dts:
                    for kid in kids:
                        role_of.setdefault(kid, set()).add(role)
        vols = [dt for dt in gmsh.model.getEntities(3)]
        via_v = {dt for dt in vols if 'via' in role_of.get(dt, ())}
        diel_v = {}
        for dt in vols:
            if dt in via_v:
                continue
            rs = role_of.get(dt, set())
            d = [r for r in rs if isinstance(r, tuple) and r[0] == 'diel']
            if d:
                diel_v.setdefault(d[0][1], []).append(dt[1])
        air_v = [dt[1] for dt in vols if dt not in via_v and not any(isinstance(r, tuple) and r[0] == 'diel'
                                                                         for r in role_of.get(dt, ()))]
        # the copper: each layer's sheet pieces; the via walls (the faces of the via holes not shared with air)
        cu_l = {lay: {dt[1] for dt in gmsh.model.getEntities(2) if ('cu', lay) in role_of.get(dt, ())}
                for lay in board.copper}
        walls = set()
        for dt in via_v:
            for f in gmsh.model.getBoundary([dt], oriented=False):
                walls.add(abs(f[1]))
        for s in cu_l.values():
            s.difference_update(walls)
        bb = (x0 - margin, y0 - margin, zb - margin, x1 + margin, y1 + margin, zt + margin)
        outer = []
        for _, s in gmsh.model.getEntities(2):
            b = gmsh.model.getBoundingBox(2, s)
            for ax in range(3):
                if (abs(b[ax] - bb[ax]) < 1e-6 and abs(b[ax + 3] - bb[ax]) < 1e-6) or \
                        (abs(b[ax] - bb[ax + 3]) < 1e-6 and abs(b[ax + 3] - bb[ax + 3]) < 1e-6):
                    outer.append(s)
                    break
        port_s = {k: [dt[1] for dt in gmsh.model.getEntities(2) if ('port', k) in role_of.get(dt, ())]
                  for k in range(len(board.ports))}
        notes = []
        for k, ss in port_s.items():
            if not ss:
                raise RuntimeError(f"port {board.ports[k].name}: its sheet is lost in the mesh")
            walls.difference_update(ss)
            for s in cu_l.values():
                s.difference_update(ss)

        diel = []
        for i, l in enumerate(board.layers):
            if i in diel_v:
                gmsh.model.addPhysicalGroup(3, diel_v[i], DIEL0 + i)
                diel.append((DIEL0 + i, l['er'], l['tand']))
        gmsh.model.addPhysicalGroup(3, air_v, AIR)
        sheets = []
        order = list(board.cu_z)
        for lay, s in cu_l.items():
            if s:
                attr = CU0 + order.index(lay)
                gmsh.model.addPhysicalGroup(2, sorted(s), attr)
                sheets.append((attr, float(getattr(board, 'cu_t', {}).get(lay, 0.035))))
        if walls:
            gmsh.model.addPhysicalGroup(2, sorted(walls), PEC)
        gmsh.model.addPhysicalGroup(2, outer, ABSORB)
        ports = []
        for k, p in enumerate(board.ports):
            gmsh.model.addPhysicalGroup(2, port_s[k], PORT0 + k)
            if p.kind == 'z':
                d = '+Z' if p.z_sig >= p.z_ref else '-Z'
            else:
                d = '+' + 'XY'[p.axis]
            ports.append(dict(attr=PORT0 + k, index=k + 1, R=float(p.z0), direction=d, name=p.name))

        # sizes: fine at the copper edges and the ports, graded out to hmax
        edges = set()
        cu = set(walls).union(*cu_l.values())
        for s in list(cu) + [t for ss in port_s.values() for t in ss]:
            for c in gmsh.model.getBoundary([(2, s)], oriented=False):
                edges.add(abs(c[1]))
        fd = gmsh.model.mesh.field
        dist = fd.add('Distance')
        fd.setNumbers(dist, 'CurvesList', sorted(edges))
        fd.setNumber(dist, 'Sampling', 40)
        thr = fd.add('Threshold')
        fd.setNumber(thr, 'InField', dist)
        fd.setNumber(thr, 'SizeMin', hmin)
        fd.setNumber(thr, 'SizeMax', hmax)
        fd.setNumber(thr, 'DistMin', hmin)
        fd.setNumber(thr, 'DistMax', grade * hmax)          # hmin to hmax over this distance
        fd.setAsBackgroundMesh(thr)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeMax', hmax * 2)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.Algorithm3D', 10)          # HXT: parallel
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.option.setNumber('Mesh.Binary', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 0)                # only the physical groups: the via holes are left out
        gmsh.write(path)
        nodes = len(gmsh.model.mesh.getNodes()[0])
        tets = _count_tets(gmsh)
    finally:
        gmsh.finalize()
    log(f"palace mesh: {tets:,} tetrahedra, {nodes:,} nodes; h {hmin:.3g}..{hmax:.3g} mm, air margin {margin:.3g} mm")
    metal = [a for a, _ in sheets] + ([PEC] if walls else [])
    return Model(path, diel, AIR, metal, ABSORB, ports, hmin, hmax, margin, nodes=nodes, tets=tets, notes=notes,
                 sheets=sheets, via_wall=PEC if walls else None)


def _vertical(occ, axis, s, e):
    """A vertical rectangle at x = s[0] (axis 0: spanning y and z) or y = s[1] (axis 1: spanning x and z)."""
    if axis == 0:
        x = s[0]
        pts = [(x, s[1], s[2]), (x, e[1], s[2]), (x, e[1], e[2]), (x, s[1], e[2])]
    else:
        y = s[1]
        pts = [(s[0], y, s[2]), (e[0], y, s[2]), (e[0], y, e[2]), (s[0], y, e[2])]
    p = [occ.addPoint(*q) for q in pts]
    loop = occ.addCurveLoop([occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)])
    return occ.addPlaneSurface([loop])


def _count_tets(gmsh):
    n = 0
    for _, v in gmsh.model.getEntities(3):
        types, elems, _ = gmsh.model.mesh.getElements(3, v)
        n += sum(len(e) for e in elems)
    return n
