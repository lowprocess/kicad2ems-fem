"""
The board as a Palace model: a tetrahedral gmsh mesh of the dielectric slabs inside an air box, and the
attributes the config refers to (config.py).

    mdl = build(board, fmax=10e9, path='model.msh')       # Model: attributes, sizes, counts

Copper is a zero-thickness PEC sheet on each copper layer at its height (cu_z); a via is a hole through the
dielectric whose wall is PEC (its barrel, closed by the pads it ends on). Lumped ports are rectangles: a 'z'
port a vertical sheet across the trace from the reference copper to the trace, a 'gap' port a sheet in the
copper plane across its gap (board.Port.box). The air box is the board's extent grown by `margin` on every
side, with an absorbing (first-order) boundary on its faces. Units: mm (the config's L0 = 1e-3).

A lumped part (board.parts) is a sheet across the gap between its pads, in the copper plane: a 0 ohm part is
copper, any other one or more lumped boundaries the config gives its R, L or C (part_elements: a part with
series parasitics is as many sheets one after another along the gap, one element each, as Palace's lumped
boundary is a parallel R-L-C). Its terminals are copper as wide as the part, reaching a little way onto each
pad (part_lands), so the sheet is fed over its whole width. An edge wall (board.walls) is a vertical PEC sheet.

Mesh size: `hmin` at the copper edges (a fraction of the smallest feature: the gap between copper on a layer, a
port's width, a slab's thickness), growing with distance to a fraction of the wavelength at fmax: `hmax` in the
slabs (the densest dielectric's), the free-space one in the air; Palace's second-order elements carry the field
between. The ground does not ask for `hmin` (`planes`): a ground plane's own edge, a via on a ground net (a
hexagonal hole of the barrel's cross-section, not a cylinder) and an edge wall take the size the mesh has reached
at them, growing from the signal copper, the ports and the parts. That needs the board's nets (copper_nets,
via_nets, ground_nets); a board without them is meshed at `hmin` along every copper edge.
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box as shp_box
from shapely.ops import unary_union

C0 = 299792458.0
# physical-group attributes (Palace's mesh attributes)
DIEL0, AIR = 1, 99                  # dielectric slab i: DIEL0 + i
PEC, ABSORB, PORT0 = 100, 101, 201  # PEC: the via walls; port i (0-based, board order): PORT0 + i
WALL, PART0 = 102, 301              # the edge walls (always PEC); lumped-part sheet j (Model.parts order): PART0 + j
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
    edge_wall: int = None                           # the edge walls' attribute (WALL), or None without them
    parts: list = field(default_factory=list)       # [dict(attr, index, direction, rlc={'R'|'L'|'C': value}, ref)]
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


def part_elements(pt):
    """A part's lumped sheets along its gap: [({'R'|'L'|'C': value}, start, stop)], empty for a 0 ohm part (metal).
    One element: one sheet over the whole gap. Series parasitics (ESR, ESL): the gap in equal lengths, one sheet
    per element, in series along it (zero-valued ones left out)."""
    vals = [(k, float(v)) for k, v in (('R', pt.R), ('L', pt.L), ('C', pt.C)) if v is not None and float(v) > 0]
    if pt.short or not vals:
        return []
    s, e = (np.asarray(v, float) for v in pt.box())
    a = int(pt.axis)
    cuts = np.linspace(s[a], e[a], len(vals) + 1)
    out = []
    for i, (k, v) in enumerate(vals):
        ps, pe = s.copy(), e.copy()
        ps[a], pe[a] = cuts[i], cuts[i + 1]
        out.append(({k: v}, ps, pe))
    return out


def part_lands(board):
    """{layer: [box]}: each part's two terminals, metal as wide as the part reaching a little way onto its pads.
    Without them the sheet meets a rounded pad only along the flat of its edge and the current crowds there: the
    demo low-pass's 0402 parts read 30% high in R, L and 1/C."""
    out = {}
    for pt in getattr(board, 'parts', None) or []:
        s, e = (np.asarray(v, float) for v in pt.box())
        a = int(pt.axis)
        b = 1 - a
        ext = min(0.5 * (e[a] - s[a]), 0.25 * (e[b] - s[b]))
        if ext <= 1e-6 or pt.layer not in board.copper:
            continue
        for lo, hi in ((s[a] - ext, s[a]), (e[a], e[a] + ext)):
            r = shp_box(lo, s[b], hi, e[b]) if a == 0 else shp_box(s[b], lo, e[b], hi)
            out.setdefault(pt.layer, []).append(r)
    return out


def feature_size(board):
    """The smallest feature the mesh must resolve, mm: the gap between separate copper on a layer, a port's
    width, a slab's thickness. (A part's sheets are refined on their own: build.)"""
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


def wall_ends(board, axis, c, t0, t1, z0, z1, snap=5e-3):
    """An edge wall's ends, each moved to the wall it meets at a corner, or to a corner of the copper it joins (the
    layers at z0 and z1), when within `snap` mm of one: a wall drawn a hair past either would leave elements that
    small."""
    a = 0 if axis in ('x', 0) else 1
    ts = [w[1] for w in getattr(board, 'walls', None) or []
          if (0 if w[0] in ('x', 0) else 1) != a and w[2] - snap < c < w[3] + snap]
    for lay, z in board.cu_z.items():
        if min(abs(z - z0), abs(z - z1)) > 1e-6:
            continue
        for g in board.copper.get(lay, ()):
            for poly in _polys(g):
                for ring in [poly.exterior, *poly.interiors]:
                    xy = np.asarray(ring.coords)
                    ts.extend(xy[np.abs(xy[:, a] - c) < 1e-6, 1 - a])
    if not ts:
        return t0, t1
    ts = np.asarray(ts)
    near = lambda t: (lambda v: v if abs(v - t) < snap else t)(float(ts[np.argmin(np.abs(ts - t))]))
    return near(t0), near(t1)


def ground_planes(board, lay, reach):
    """Which of a layer's copper polygons are ground planes: on a ground net (board.ground_nets) and mostly further
    than `reach` from their own edge. [bool per polygon]; all False for a board that does not name its nets."""
    nets = (getattr(board, 'copper_nets', None) or {}).get(lay) or []
    gnd = getattr(board, 'ground_nets', None) or ()
    polys = board.copper[lay]
    return [i < len(nets) and nets[i] in gnd and g.buffer(-reach).area > 0.5 * g.area for i, g in enumerate(polys)]


def build(board, fmax, path, margin=None, hmin=None, hmax=None, per_feature=1.0, per_wave=8.0, grade=3.0,
          log=print, threads=8, planes=True, flat=None):
    """Mesh the board for Palace and write it to `path` (gmsh 2.2, binary). fmax in Hz.

    planes: the ground planes (ground_planes), the vias on ground nets and the edge walls take the size the mesh
    has reached at them, growing from the other copper and the ports, instead of `hmin` at their own edges.
    flat: a slab's elements no wider than this many times its thickness (None: as wide as the sizes above allow)."""
    import gmsh
    feat = feature_size(board)
    er_max = max([l['er'] for l in board.layers] + [1.0])
    lam0 = C0 / fmax * 1e3                                      # mm
    hmin = hmin or max(feat / per_feature, 0.02)
    hmax = hmax or max(lam0 / per_wave / np.sqrt(er_max), 4 * hmin)
    hair = max(lam0 / per_wave, hmax)                           # the air's own wavelength, outside the slabs
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
        lands = part_lands(board)
        reach = max(4 * hmin, 1.0)
        gnd = getattr(board, 'ground_nets', None) or ()
        for lay, polys in board.copper.items():                 # the copper sheets
            z = board.cu_z[lay]
            plane = ground_planes(board, lay, reach) if planes else [False] * len(polys)
            if lands.get(lay):                                  # a part's terminals join the pads they sit on
                polys, plane = [unary_union([g.buffer(0) for g in polys] + lands[lay])], [False]
            for g, pl in zip(polys, plane):
                for poly in _polys(g):
                    dt = [(2, _surface(occ, poly.buffer(0), z))]
                    add(('cu', lay), dt)
                    if pl:
                        tags.setdefault('plane', []).extend(dt)
        via_nets = getattr(board, 'via_nets', None) or []
        for i, v in enumerate(board.vias):                      # the via barrels (holes)
            x, y, r, za, zc = v[:5]
            if zc > za:
                if planes and i < len(via_nets) and via_nets[i] in gnd:    # a ground via: a hexagonal hole, whose
                    dt = [(3, _prism(occ, x, y, r, za, zc))]                # flat walls take wide triangles
                    tags.setdefault('plane', []).extend(dt)
                else:
                    dt = [(3, occ.addCylinder(x, y, za, 0, 0, zc - za, r))]
                add('via', dt)
        for k, p in enumerate(board.ports):                     # the lumped-port sheets
            s, e = p.box()
            if p.kind == 'z':                                   # vertical: across the trace at c[axis]
                r = _vertical(occ, p.axis, s, e)
            else:                                               # in the copper plane, across the gap
                r = occ.addRectangle(s[0], s[1], s[2], e[0] - s[0], e[1] - s[1])
            add(('port', k), [(2, r)])
        part_list = []                                          # [(ref, axis, rlc)]: one per lumped sheet
        for pt in getattr(board, 'parts', None) or []:          # the lumped parts: sheets across their gaps
            els = part_elements(pt)
            if not els:                                         # 0 ohm: copper between the pads
                s, e = pt.box()
                add(('cu', pt.layer), [(2, occ.addRectangle(s[0], s[1], s[2], e[0] - s[0], e[1] - s[1]))])
                continue
            for rlc, s, e in els:
                add(('part', len(part_list)), [(2, occ.addRectangle(s[0], s[1], s[2], e[0] - s[0], e[1] - s[1]))])
                part_list.append((pt.ref, int(pt.axis), rlc, float(min(v for v in np.abs(e - s)[:2] if v > 1e-6))))
        for axis, c, t0, t1, z0, z1 in getattr(board, 'walls', None) or []:    # the edge walls: vertical sheets
            t0, t1 = wall_ends(board, axis, c, t0, t1, z0, z1)
            if z1 - z0 < 1e-6 or t1 - t0 < 1e-6:
                continue
            s, e = ([c, t0, z0], [c, t1, z1]) if axis in ('x', 0) else ([t0, c, z0], [t1, c, z1])
            add('wall', [(2, _vertical(occ, 0 if axis in ('x', 0) else 1, s, e))])
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
        meshed = {dt[1] for dt in vols if dt not in via_v}

        def live(s):                        # a face of a meshed volume: not one inside a via hole
            up = gmsh.model.getAdjacencies(2, s)[0]
            if len(up):
                return bool(meshed.intersection(up))
            c = occ.getCenterOfMass(2, s)   # a sheet inside a volume (an edge wall in from the edge) bounds none
            return not any(gmsh.model.isInside(3, v, c) for _, v in via_v)
        walls = set()
        for dt in via_v:
            for f in gmsh.model.getBoundary([dt], oriented=False):
                walls.add(abs(f[1]))
        for lay, s in cu_l.items():
            cu_l[lay] = {t for t in s - walls if live(t)}
        walls = {t for t in walls if live(t)}       # a hole through several slabs is several pieces: not the discs between
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
        part_s = {j: [dt[1] for dt in gmsh.model.getEntities(2) if ('part', j) in role_of.get(dt, ())]
                  for j in range(len(part_list))}
        notes = []
        for k, ss in port_s.items():
            if not ss:
                raise RuntimeError(f"port {board.ports[k].name}: its sheet is lost in the mesh")
        for j, ss in part_s.items():
            if not ss:
                raise RuntimeError(f"part {part_list[j][0]}: its sheet is lost in the mesh")
        lumped = {t for ss in list(port_s.values()) + list(part_s.values()) for t in ss}
        edge_s = {dt[1] for dt in gmsh.model.getEntities(2) if 'wall' in role_of.get(dt, ()) and live(dt[1])} - lumped - walls
        walls.difference_update(lumped)
        for s in cu_l.values():
            s.difference_update(lumped | edge_s)

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
        if edge_s:
            gmsh.model.addPhysicalGroup(2, sorted(edge_s), WALL)
        gmsh.model.addPhysicalGroup(2, outer, ABSORB)
        ports = []
        for k, p in enumerate(board.ports):
            gmsh.model.addPhysicalGroup(2, port_s[k], PORT0 + k)
            if p.kind == 'z':
                d = '+Z' if p.z_sig >= p.z_ref else '-Z'
            else:
                d = '+' + 'XY'[p.axis]
            ports.append(dict(attr=PORT0 + k, index=k + 1, R=float(p.z0), direction=d, name=p.name))
        parts = []
        for j, (ref, axis, rlc, _) in enumerate(part_list):     # Palace's port indices go on after the ports'
            gmsh.model.addPhysicalGroup(2, part_s[j], PART0 + j)
            parts.append(dict(attr=PART0 + j, index=len(ports) + j + 1, direction='+' + 'XY'[axis], rlc=rlc, ref=ref))

        # sizes: fine at the copper edges and the ports, graded out to hmax; not at the ground planes' own edges,
        # the ground vias or the edge walls (planes), which the mesh reaches at the size it has grown to
        edges = set()
        cu = set(walls).union(edge_s, *cu_l.values())
        coarse = set()
        if planes:
            coarse = {dt[1] for dt in gmsh.model.getEntities(2) if 'plane' in role_of.get(dt, ())} | set(edge_s)
            for dt in via_v:
                if 'plane' in role_of.get(dt, ()):
                    coarse.update(abs(f[1]) for f in gmsh.model.getBoundary([dt], oriented=False))
            if not (cu - coarse) and not lumped:                # nothing else to lead the mesh
                coarse = set()
        for s in sorted(cu - coarse) + sorted(lumped):
            for c in gmsh.model.getBoundary([(2, s)], oriented=False):
                edges.add(abs(c[1]))
        fd = gmsh.model.mesh.field
        dist = fd.add('Distance')
        fd.setNumbers(dist, 'CurvesList', sorted(edges))
        fd.setNumber(dist, 'Sampling', 40)
        thr = fd.add('Threshold')
        fd.setNumber(thr, 'InField', dist)
        fd.setNumber(thr, 'SizeMin', hmin)
        fd.setNumber(thr, 'SizeMax', hair)
        fd.setNumber(thr, 'DistMin', hmin)
        fd.setNumber(thr, 'DistMax', grade * hair)          # hmin to the air's size over this distance
        fields = [thr]
        for i, ts in diel_v.items():                        # the slabs: their shorter wavelength, their thickness
            hs = min(hmax, flat * (board.layers[i]['z1'] - board.layers[i]['z0'])) if flat else hmax
            if hs < hair:
                slab = fd.add('Constant')
                fd.setNumbers(slab, 'VolumesList', sorted(ts))
                fd.setNumber(slab, 'VIn', hs)
                fd.setNumber(slab, 'VOut', hair)
                fields.append(slab)
        hpart = max(min([w for *_, w in part_list], default=hmin) / per_feature, 0.02)
        if hpart < hmin:                                    # the part sheets narrower than that: finer at them only
            near = fd.add('Distance')
            fd.setNumbers(near, 'CurvesList', sorted({abs(c[1]) for ss in part_s.values() for s in ss
                                                      for c in gmsh.model.getBoundary([(2, s)], oriented=False)}))
            fd.setNumber(near, 'Sampling', 40)
            fine = fd.add('Threshold')
            for k, v in dict(InField=near, SizeMin=hpart, SizeMax=hair, DistMin=hpart, DistMax=grade * hair).items():
                fd.setNumber(fine, k, v)
            fields.append(fine)
        if len(fields) > 1:
            thr = fd.add('Min')
            fd.setNumbers(thr, 'FieldsList', fields)
        fd.setAsBackgroundMesh(thr)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeMax', hair * 2)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        for alg in (10, 1):                                     # HXT (parallel); Delaunay where it cannot recover
            gmsh.option.setNumber('Mesh.Algorithm3D', alg)      # the faces of a thin slab from wide triangles
            try:
                gmsh.model.mesh.generate(3)
                break
            except Exception:
                if alg == 1:
                    raise
                gmsh.model.mesh.clear()
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.option.setNumber('Mesh.Binary', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 0)                # only the physical groups: the via holes are left out
        gmsh.write(path)
        nodes = len(gmsh.model.mesh.getNodes()[0])
        tets = _count_tets(gmsh)
        if not tets:
            raise RuntimeError('gmsh made no volume mesh of the board')
    finally:
        gmsh.finalize()
    log(f"palace mesh: {tets:,} tetrahedra, {nodes:,} nodes; h {hmin:.3g}..{hmax:.3g} mm in the board, to "
        f"{hair:.3g} mm in the air; air margin {margin:.3g} mm")
    metal = [a for a, _ in sheets] + ([PEC] if walls else []) + ([WALL] if edge_s else [])
    return Model(path, diel, AIR, metal, ABSORB, ports, hmin, hmax, margin, nodes=nodes, tets=tets, notes=notes,
                 sheets=sheets, via_wall=PEC if walls else None, edge_wall=WALL if edge_s else None, parts=parts)


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


def _prism(occ, x, y, r, za, zc, n=6):
    """A vertical n-sided prism of a circle's cross-section area: its volume tag."""
    R = r * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    a = 2 * np.pi * np.arange(n) / n
    p = [occ.addPoint(x + R * np.cos(v), y + R * np.sin(v), za) for v in a]
    s = occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(p[i], p[(i + 1) % n]) for i in range(n)])])
    return [d for d in occ.extrude([(2, s)], 0, 0, zc - za) if d[0] == 3][0][1]


def _count_tets(gmsh):
    n = 0
    for _, v in gmsh.model.getEntities(3):
        types, elems, _ = gmsh.model.mesh.getElements(3, v)
        n += sum(len(e) for e in elems)
    return n
