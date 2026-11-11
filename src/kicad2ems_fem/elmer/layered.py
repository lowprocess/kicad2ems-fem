"""The whole board as a layered mesh for Elmer: one triangle mesh of the board's outline that every layer's copper
edges, the vias and the terminals' pads are lines of, swept through the stack as wedges. Each wedge is copper, a via
barrel, board material or (beside the copper of an outer layer) absent, by where its triangle lies in that layer.

    m = layered_mesh(board, 'case/mesh', electric={'VHB'}, terminals={'Q1': [(layer, polygon)]}, h=0.3)
    m['boundaries']        # {tag: (names, area mm2)}: 1 the exposed faces, 10.. the terminals' pads (also exposed)
    m['bodies']            # the bodies present: COPPER_E, BARREL_E (of `electric` nets), COPPER, BARREL, BOARD

No 3D booleans: the copper of real boards (planes with hundreds of holes, on several layers) breaks them, and all the
board's copper matters to where the heat goes. Shapely does the layout's arithmetic in 2D, on a grid of `grid` mm.
A via is a prism of its hole's area through the layers it joins.
"""
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ..geometry import copper_of, layers as layer_names, slab

COPPER_E, BARREL_E, COPPER, BARREL, BOARD = 1, 2, 3, 4, 5
EXPOSED, TERM0 = 1, 10


def _cross(u, v):
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def _ngon(x, y, r, n):
    """A regular polygon of a circle's area."""
    R = r * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    a = (np.arange(n) + 0.5) * 2 * np.pi / n
    return Polygon(np.c_[x + R * np.cos(a), y + R * np.sin(a)])


def z_cells(board, dz=0.5):
    """The stack as cells bottom to top: [(z0, z1, copper layer or None)]. A copper layer is one cell of its
    thickness, the board between them cells of at most `dz` mm."""
    names = layer_names(board)
    slabs = {l: slab(board, l) for l in names}
    zs = sorted({round(z, 6) for s in slabs.values() for z in s} |
                {round(l[k], 6) for l in board['stack'].get('layers', []) for k in ('z0', 'z1')})
    out = []
    for a, b in zip(zs[:-1], zs[1:]):
        mid = 0.5 * (a + b)
        lay = next((l for l, (s0, s1) in slabs.items() if s0 - 1e-9 < mid < s1 + 1e-9), None)
        n = 1 if lay else max(1, int(np.ceil((b - a) / dz - 1e-9)))
        out += [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n, lay) for i in range(n)]
    return out


def _rings(g):
    for p in (g.geoms if hasattr(g, 'geoms') else [g]):
        if isinstance(p, Polygon) and not p.is_empty:
            yield from [p.exterior, *p.interiors]


def _clean(g, region, tol, grid):
    """A layout geometry inside the region, simplified to tol and on the grid; None when nothing is left."""
    if g is None or g.is_empty:
        return None
    g = g.intersection(region)
    if tol:
        g = g.simplify(tol)
    g = shapely.set_precision(g.buffer(0), grid)
    return None if g.is_empty or g.area < 1e-6 else g


def triangulate(region, lines, size, grid, log=print):
    """Triangles of `region` (a polygon) with the segments of `lines` (noded, on the grid) as edges. size: (x0, y0,
    step, array[ny, nx]) of the element size wanted, mm. Returns (xy [n, 2], triangles [m, 3], counter-clockwise)."""
    import gmsh
    x0, y0, step, raster = size
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('board')
        geo = gmsh.model.geo
        pts, segs = {}, {}

        def point(c):
            k = (round(c[0] / grid), round(c[1] / grid))
            if k not in pts:
                pts[k] = geo.addPoint(k[0] * grid, k[1] * grid, 0.0)
            return pts[k]

        def seg(a, b):
            a, b = point(a), point(b)
            if a == b:
                return None
            k = (min(a, b), max(a, b))
            new = k not in segs
            if new:
                segs[k] = geo.addLine(*k)
            return (segs[k] if a < b else -segs[k]), new

        # the outline's rings, through the points where lines end on them
        ends = shapely.points([c for l in lines for c in (l.coords[0], l.coords[-1])]) if lines else np.array([])
        loops, edge = [], set()
        for ring in _rings(orient(region, 1.0)):
            on = [(ring.project(p), (p.x, p.y)) for p in ends[shapely.dwithin(ends, ring, 0.25 * grid)]] if len(ends) else []
            on += [(ring.project(shapely.Point(c)), c) for c in ring.coords[:-1]]
            chain = [c for _, c in sorted(on)]
            tags = [s[0] for s in (seg(chain[i], chain[(i + 1) % len(chain)]) for i in range(len(chain))) if s]
            edge |= {abs(t) for t in tags}
            loops.append(geo.addCurveLoop(tags))
        surface = geo.addPlaneSurface(loops)
        inner = []
        for l in lines:
            c = list(l.coords)
            for a, b in zip(c[:-1], c[1:]):
                s = seg(a, b)
                if s and s[1] and abs(s[0]) not in edge:
                    inner.append(abs(s[0]))
        geo.synchronize()
        if inner:
            gmsh.model.mesh.embed(1, inner, 2, surface)
        ny, nx = raster.shape

        def wanted(dim, tag, x, y, z, lc):
            i = min(max(int((y - y0) / step), 0), ny - 1)
            j = min(max(int((x - x0) / step), 0), nx - 1)
            return float(raster[i, j])
        gmsh.model.mesh.setSizeCallback(wanted)
        for k, v in {'Mesh.MeshSizeExtendFromBoundary': 0, 'Mesh.MeshSizeFromPoints': 0, 'Mesh.MeshSizeFromCurvature': 0,
                     'Mesh.MeshSizeMax': float(raster.max()), 'Mesh.Algorithm': 5}.items():
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        tri = gmsh.model.mesh.getElementsByType(2)[1].reshape(-1, 3)
    finally:
        gmsh.finalize()
    if not len(tri):
        raise RuntimeError('gmsh made no triangles of the board')
    used = np.unique(tri)
    index = np.full(int(tags.max()) + 1, -1)
    index[used] = np.arange(len(used))
    where = np.zeros(int(tags.max()) + 1, int)
    where[tags.astype(int)] = np.arange(len(tags))
    xy = xyz.reshape(-1, 3)[where[used], :2]
    tri = index[tri.astype(int)]
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    flip = _cross(b - a, c - a) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return xy, tri


def layered_mesh(board, folder, electric=(), copper=None, terminals=None, crop=None, h=0.3, hmax=2.0, dz=0.5,
                 tol=0.02, grid=0.005, grade=0.35, log=print):
    """Write Elmer's mesh of the board to `folder`. electric: the nets that carry current (bodies COPPER_E and
    BARREL_E, meshed at `h`); copper: the other nets to keep (None: all); terminals: {name: [(layer, polygon)]}, pads
    on the outer layers, each a boundary (pads shared by several names are one boundary of those names). Returns the
    mesh's description (see the module's text) with the arrays the results are read with."""
    electric = set(electric)
    terminals = terminals or {}
    names = layer_names(board)
    cells = z_cells(board, dz)
    region = shape(board['outline']) if board.get('outline') else None
    if region is None or region.is_empty:
        raise ValueError('the board has no outline')
    if crop is not None:
        region = region.intersection(crop)
    region = max(region.geoms, key=lambda p: p.area) if hasattr(region, 'geoms') else region
    region = shapely.set_precision(region.simplify(tol), grid)
    every = {c.get('net') for l in names for c in board['copper'].get(l, [])}
    passive = (every if copper is None else set(copper)) - electric
    cu = {l: (_clean(copper_of(board, electric, l), region, tol, grid) if electric else None,
              _clean(copper_of(board, passive, l), region, 2 * tol, grid) if passive else None) for l in names}
    if electric and not any(e is not None for e, _ in cu.values()):
        raise ValueError(f"no copper of {', '.join(sorted(electric))}")
    slabs = {l: slab(board, l) for l in names}
    zs = {c['name']: c['z'] for c in board['stack']['copper']}
    bottom, top = min(names, key=lambda l: zs[l]), max(names, key=lambda l: zs[l])

    def span(v):                             # a via ending at a layer's sheet goes through that layer's slab
        ends = []
        for z, side in ((v['z0'], 0), (v['z1'], 1)):
            hit = [slabs[l][side] for l in names if abs(zs[l] - z) < 1e-6]
            ends.append(hit[0] if hit else z)
        return ends
    vias = []
    for v in board.get('vias', []):
        cls = BARREL_E if v.get('net') in electric else BARREL if v.get('net') in passive else 0
        g = _ngon(v['x'], v['y'], v['r'], 8 if cls == BARREL_E else 6)
        if cls and region.contains(g):
            vias.append((shapely.set_precision(g, grid), cls, *span(v), v['r']))
    pads = {}
    for name, ps in terminals.items():
        gs = {l: _clean(unary_union([p for lay, p in ps if lay == l]), region, 0.0, grid) for l in (bottom, top)}
        if all(g is None for g in gs.values()):
            raise ValueError(f'terminal {name}: no pad on an outer layer inside the board')
        pads[name] = gs

    # every edge of the layout, noded on the grid
    work = [r for e, p in cu.values() for g in (e, p) if g is not None for r in _rings(g)]
    work += [v[0].exterior for v in vias] + [r for gs in pads.values() for g in gs.values() if g is not None
                                             for r in _rings(g)]
    lines = []
    if work:
        noded = shapely.unary_union([LineString(r.coords) for r in work] + [LineString(r.coords) for r in _rings(region)],
                                    grid_size=grid)
        lines = [l for l in (noded.geoms if hasattr(noded, 'geoms') else [noded]) if l.geom_type == 'LineString']
        shapely.prepare(region)
        lines = [l for l in lines if region.covers(l)]

    # the element size wanted: h in the copper that carries current and at the terminals, growing away from them
    x0, y0, x1, y1 = region.bounds
    step = max(0.5 * h, max(x1 - x0, y1 - y0) / 2000)
    gx, gy = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
    fine = [e for e, _ in cu.values() if e is not None] + [g for gs in pads.values() for g in gs.values() if g is not None]
    raster = np.full((len(gy), len(gx)), float(hmax))
    if fine:
        from scipy import ndimage
        X, Y = np.meshgrid(gx, gy)
        inside_fine = shapely.contains_xy(unary_union(fine).buffer(step), X, Y)
        raster = np.clip(h + grade * step * ndimage.distance_transform_edt(~inside_fine), h, hmax)
    xy, tri = triangulate(region, lines, (x0, y0, step, raster), grid, log)

    # what each triangle is in each cell
    cen = xy[tri].mean(1)
    nt, nk = len(tri), len(cells)
    barrel = np.zeros((nk, nt), np.int8)
    mid = np.array([0.5 * (a + b) for a, b, _ in cells])
    if vias:
        tree = shapely.STRtree([v[0] for v in vias])
        ti, vi = tree.query(shapely.points(cen), predicate='within')
        for t, v in zip(ti, vi):
            _, cls, za, zb, _ = vias[v]
            k = (mid > za) & (mid < zb)
            barrel[k, t] = np.where(barrel[k, t] == BARREL_E, BARREL_E, cls)
    body = np.zeros((nk, nt), np.int8)
    for k, (_, _, lay) in enumerate(cells):
        if lay is None:
            body[k] = np.where(barrel[k] > 0, barrel[k], BOARD)
            continue
        e, p = cu[lay]
        c = np.zeros(nt, np.int8)
        if p is not None:
            c[shapely.contains_xy(p, cen[:, 0], cen[:, 1])] = COPPER
        if e is not None:
            c[shapely.contains_xy(e, cen[:, 0], cen[:, 1])] = COPPER_E
        rest = BOARD if lay not in (bottom, top) else 0
        body[k] = np.where(c > 0, c, np.where(barrel[k] > 0, barrel[k], rest))
    exists = body > 0
    if not exists.any(axis=0).all():
        tri, cen, body, exists = tri[exists.any(axis=0)], cen[exists.any(axis=0)], body[:, exists.any(axis=0)], \
            exists[:, exists.any(axis=0)]
        nt = len(tri)

    # nodes: (2D node, level), those in use
    n2 = len(xy)
    levels = np.array([cells[0][0]] + [c[1] for c in cells])
    used = np.zeros((nk + 1, n2), bool)
    for k in range(nk):
        used[k:k + 2, np.unique(tri[exists[k]])] = True
    node = np.full((nk + 1, n2), -1)
    node[used] = np.arange(used.sum())
    lev, i2 = np.nonzero(used)
    nodes = np.c_[xy[i2], levels[lev]]
    wk, wt = np.nonzero(exists)
    wedges = np.c_[node[wk[:, None], tri[wt]], node[wk[:, None] + 1, tri[wt]]]
    wbody = body[wk, wt]
    wid = np.full((nk, nt), -1)
    wid[wk, wt] = np.arange(len(wk))

    # the exposed faces: above and below, and at the sides
    up = exists & ~np.vstack([exists[1:], np.zeros((1, nt), bool)])
    down = exists & ~np.vstack([np.zeros((1, nt), bool), exists[:-1]])
    tris, tparent, tkey = [], [], []
    key = {'top': np.zeros(nt, int), 'bottom': np.zeros(nt, int)}               # bit i: terminal i's pad
    for i, (name, gs) in enumerate(pads.items()):
        for side, lay in (('top', top), ('bottom', bottom)):
            if gs[lay] is not None:
                key[side][shapely.contains_xy(gs[lay].buffer(0.5 * grid), cen[:, 0], cen[:, 1])] |= 1 << i
    ktop, kbot = nk - 1, 0
    for k in range(nk):
        for mask, off, side, kk in ((up[k], 1, 'top', ktop), (down[k], 0, 'bottom', kbot)):
            t = np.flatnonzero(mask)
            if len(t):
                tris.append(node[k + off][tri[t]])
                tparent.append(wid[k, t])
                on_pad = (k == kk) & np.isin(body[k, t], (COPPER_E, COPPER))
                tkey.append(np.where(on_pad, key[side][t], 0))
    tris, tparent, tkey = np.vstack(tris), np.concatenate(tparent), np.concatenate(tkey)
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    et = np.tile(np.arange(nt), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, et = e[order], et[order]
    same = np.r_[(e[1:] == e[:-1]).all(1), False]
    pair = np.flatnonzero(same)
    single = np.flatnonzero(~same & ~np.r_[False, same[:-1]])
    quads, qparent = [], []
    for k in range(nk):
        a, b = et[pair], et[pair + 1]
        d = exists[k, a] != exists[k, b]
        own = np.where(exists[k, a[d]], a[d], b[d])
        s = single[exists[k, et[single]]]
        ed = np.vstack([e[pair[d]], e[s]])
        quads.append(np.c_[node[k][ed[:, 0]], node[k][ed[:, 1]], node[k + 1][ed[:, 1]], node[k + 1][ed[:, 0]]])
        qparent.append(np.r_[wid[k, own], wid[k, et[s]]])
    quads, qparent = np.vstack(quads), np.concatenate(qparent)

    keys = sorted(set(tkey.tolist()) - {0})
    tag_of = {0: EXPOSED, **{k: TERM0 + i for i, k in enumerate(keys)}}
    ttag = np.array([tag_of[k] for k in tkey.tolist()])
    p = nodes[tris]
    tarea = 0.5 * np.abs(_cross(p[:, 1, :2] - p[:, 0, :2], p[:, 2, :2] - p[:, 0, :2]))
    q = nodes[quads]
    qarea = np.hypot(*(q[:, 1, :2] - q[:, 0, :2]).T) * (q[:, 2, 2] - q[:, 1, 2])
    tnames = list(pads)
    boundaries = {EXPOSED: ((), float(tarea[ttag == EXPOSED].sum() + qarea.sum()))}
    for k in keys:
        boundaries[tag_of[k]] = (tuple(n for i, n in enumerate(tnames) if k >> i & 1), float(tarea[ttag == tag_of[k]].sum()))
    for i, name in enumerate(tnames):
        if not any(k >> i & 1 for k in keys):
            raise ValueError(f'terminal {name}: no copper under its pads')

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nw, nb = len(wedges), len(tris) + len(quads)
    (folder / 'mesh.header').write_text(f'{len(nodes)} {nw} {nb}\n3\n706 {nw}\n303 {len(tris)}\n404 {len(quads)}\n')
    with open(folder / 'mesh.nodes', 'w') as fh:
        np.savetxt(fh, np.c_[np.arange(1, len(nodes) + 1), np.full(len(nodes), -1), nodes], fmt='%d %d %.9g %.9g %.9g')
    with open(folder / 'mesh.elements', 'w') as fh:
        np.savetxt(fh, np.c_[np.arange(1, nw + 1), wbody, np.full(nw, 706), wedges + 1], fmt='%d')
    with open(folder / 'mesh.boundary', 'w') as fh:
        np.savetxt(fh, np.c_[np.arange(1, len(tris) + 1), ttag, tparent + 1, np.zeros(len(tris), int),
                             np.full(len(tris), 303), tris + 1], fmt='%d')
        np.savetxt(fh, np.c_[np.arange(len(tris) + 1, nb + 1), np.full(len(quads), EXPOSED), qparent + 1,
                             np.zeros(len(quads), int), np.full(len(quads), 404), quads + 1], fmt='%d')
    r = np.array([v[4] for v in vias]) if vias else np.array([0.15])
    vol = lambda b: float((0.5 * np.abs(_cross(xy[tri[wt, 1]] - xy[tri[wt, 0]], xy[tri[wt, 2]] - xy[tri[wt, 0]])) *
                           (levels[wk + 1] - levels[wk]))[wbody == b].sum())
    log(f'mesh: {nw:,} wedges on {nt:,} triangles in {nk} layers, {len(nodes):,} nodes, {len(vias)} vias')
    return dict(boundaries=boundaries, bodies=sorted(set(wbody.tolist())), wedges=nw, triangles=nt, cells=cells,
                vias=len(vias), via_r=r, nodes=nodes, xy=xy, tri=tri, node=node, body=body, faces=tris, face_tag=ttag,
                face_area=tarea, sides=quads, side_area=qarea, volume={b: vol(b) for b in set(wbody.tolist())},
                top=top, bottom=bottom)
