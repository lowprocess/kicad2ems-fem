"""A net's copper as a 3D gmsh model for Elmer: each layer's copper extruded through its thickness (geometry.slab),
each via a solid cylinder through the board whose conductivity gives its plated annulus's conductance, and the
terminals (pads) tagged on the copper's surface.

    m = copper_model(board, nets, terminals, 'copper.msh', h=0.25, h_fine=0.05)
    m['boundaries']            # {terminal: physical tag}; bodies 1 (copper) and 2 (barrels)

Real copper needs care (each point found on wpthb and Jetson boards, pcbqs/validation/dc_elmer.py):
  - polygons simplified to `tol` (near-duplicate vertices make overlapping facets);
  - holes cut by a boolean (as curve loops of one face OCC silently dropped some on real planes);
  - copper fragmented against the via cylinders only (fragmenting everything against everything left planes unfused);
  - pads tagged on the mesh, not split in the geometry (split pads' slivers fail to mesh);
  - the size-field points cleared after meshing (orphan nodes make Elmer's system singular).
"""
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from ..geometry import copper_of, layers as layer_names, polygons, slab

SIGMA_CU = 5.8e7


def copper_model(board, nets, terminals, path, crop=None, h=0.25, h_fine=0.05, plating=0.025, log=print, tol=0.01):
    """Mesh the copper of `nets` (and their vias) in 3D, written to `path` (gmsh .msh version 2). terminals:
    {name: [(layer, shapely polygon)]}. Returns dict(boundaries={terminal: tag}, barrel_sigma, vias, tets,
    barrels=bool): body 1 the copper, body 2 the barrels."""
    import gmsh
    nets = set(nets)
    layers = layer_names(board)
    zs = {c['name']: c['z'] for c in board['stack']['copper']}
    zmin = min(slab(board, l)[0] for l in layers)
    zmax = max(slab(board, l)[1] for l in layers)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('copper')
        occ = gmsh.model.occ

        def disc(ring, z):
            tags = [occ.addPoint(x, y, z) for x, y in list(ring.coords)[:-1]]
            return occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(tags[i], tags[(i + 1) % len(tags)])
                                                          for i in range(len(tags))])])

        def surfaces(poly, z):
            outer = disc(poly.exterior, z)
            if not poly.interiors:
                return [outer]
            res, _ = occ.cut([(2, outer)], [(2, disc(r, z)) for r in poly.interiors])
            return [t for d, t in res if d == 2]

        volumes = []
        for lay in layers:
            g = copper_of(board, nets, lay)
            if g is None:
                continue
            if crop is not None:
                g = g.intersection(crop)
            z0, z1 = slab(board, lay)
            ext = occ.extrude([(2, s) for p in polygons(g, tol) for s in surfaces(p, z0)], 0, 0, z1 - z0)
            volumes += [t for d, t in ext if d == 3]
        if not volumes:
            raise ValueError(f"no copper of {', '.join(sorted(nets))}")

        def end(z, side):                    # a via ending at a layer's sheet goes through that layer's slab
            hit = [slab(board, l) for l in layers if abs(zs[l] - z) < 1e-6]
            return (hit[0][0] if side < 0 else hit[0][1]) if hit else z
        vias = [v for v in board['vias'] if v.get('net') in nets and
                (crop is None or crop.contains(box(v['x'] - v['r'], v['y'] - v['r'], v['x'] + v['r'], v['y'] + v['r'])))]
        cyl = [occ.addCylinder(v['x'], v['y'], end(v['z0'], -1), 0, 0, end(v['z1'], 1) - end(v['z0'], -1), v['r'])
               for v in vias]
        if cyl:
            occ.fragment([(3, t) for t in volumes], [(3, t) for t in cyl])
        occ.synchronize()
        slabs = [slab(board, l) for l in layers]
        copper, barrel = [], []
        for _, t in gmsh.model.getEntities(3):
            z = occ.getCenterOfMass(3, t)[2]
            (copper if any(a - 1e-6 <= z <= b + 1e-6 for a, b in slabs) else barrel).append(t)
        gmsh.model.addPhysicalGroup(3, copper, 1, 'copper')
        if barrel:
            gmsh.model.addPhysicalGroup(3, barrel, 2, 'barrel')
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], 3, 'surface')
        # mesh size: h, h_fine at the vias and along the pad edges
        marks = [(v['x'], v['y'], 0.5 * (zmin + zmax)) for v in vias]
        for lay, poly in [x for ts in terminals.values() for x in ts]:
            zf = _face_z(board, lay, zmin)
            marks += [(x, y, zf) for x, y in poly.exterior.segmentize(h_fine).coords]
        pts = []
        if marks:
            pts = [occ.addPoint(*m) for m in marks]
            occ.synchronize()
            f = gmsh.model.mesh.field.add('Distance')
            gmsh.model.mesh.field.setNumbers(f, 'PointsList', pts)
            th = gmsh.model.mesh.field.add('Threshold')
            dist = max([2 * v['r'] for v in vias], default=h_fine)
            for k, val in dict(InField=f, SizeMin=h_fine, SizeMax=h, DistMin=dist, DistMax=dist + 3 * h).items():
                gmsh.model.mesh.field.setNumber(th, k, val)
            gmsh.model.mesh.field.setAsBackgroundMesh(th)
        for k, val in {'Mesh.MeshSizeMax': h, 'Mesh.MeshSizeExtendFromBoundary': 0, 'Mesh.MeshSizeFromPoints': 0,
                       'Mesh.MeshSizeFromCurvature': 0, 'Mesh.MshFileVersion': 2.2}.items():
            gmsh.option.setNumber(k, val)
        for algorithm in (10, 1):                       # HXT, then Delaunay
            gmsh.option.setNumber('Mesh.Algorithm3D', algorithm)
            try:
                gmsh.model.mesh.generate(3)
                break
            except Exception as exc:
                if algorithm == 1:
                    raise
                log(f'gmsh: HXT failed ({str(exc).splitlines()[0]}); trying Delaunay')
                gmsh.model.mesh.clear()
        if pts:
            gmsh.model.mesh.clear([(0, t) for t in pts])
        n_tets = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
        if not n_tets:
            raise RuntimeError('gmsh produced no tetrahedra')
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    tags = tag_terminals(path, board, terminals, zmin)
    r = np.array([v['r'] for v in vias]) if vias else np.array([0.15])
    factor = float(np.mean((r ** 2 - np.maximum(r - plating, 0) ** 2) / r ** 2))
    log(f'gmsh: {n_tets:,} tetrahedra, {len(vias)} vias, {len(tags)} terminals')
    return dict(boundaries=tags, barrel_sigma=SIGMA_CU * factor, barrels=bool(barrel), vias=len(vias), tets=n_tets)


def _face_z(board, layer, zmin):
    """The z of a layer's outer copper face: the bottom of the lowest layer, else the top."""
    z0, z1 = slab(board, layer)
    return z0 if z0 < zmin + 1e-9 else z1


def tag_terminals(path, board, terminals, zmin):
    """Re-tag the .msh's horizontal surface triangles that lie on a terminal's pads (the outer copper face of the
    pad's layer) with the terminal's physical tag (10, 11, ...). Returns {terminal: tag}."""
    text = Path(path).read_text().split('\n')
    i = text.index('$Nodes')
    n = int(text[i + 1])
    node = np.array([list(map(float, l.split()[:4])) for l in text[i + 2:i + 2 + n]])
    index = {int(k): j for j, k in enumerate(node[:, 0])}
    xyz = node[:, 1:]
    j = text.index('$Elements')
    m = int(text[j + 1])
    lines = list(range(j + 2, j + 2 + m))
    tri = [k for k in lines if text[k].split()[1] == '2']
    fields = [text[k].split() for k in tri]
    verts = np.array([[index[int(q)] for q in f[3 + int(f[2]):]] for f in fields])
    pts = xyz[verts]                                       # (ntri, 3, 3)
    flat = np.ptp(pts[:, :, 2], axis=1) < 1e-9
    cen = pts.mean(1)
    done = np.zeros(len(tri), bool)
    tags = {}
    for t, (name, pads) in enumerate(terminals.items()):
        hit = np.zeros(len(tri), bool)
        for zf in {_face_z(board, lay, zmin) for lay, _ in pads}:
            area = unary_union([poly.buffer(1e-3) for lay, poly in pads if abs(_face_z(board, lay, zmin) - zf) < 1e-9])
            sel = flat & ~done & (abs(cen[:, 2] - zf) < 1e-6)
            hit[sel] = shapely.contains_xy(area, cen[sel, 0], cen[sel, 1])
        if not hit.any():
            raise ValueError(f'terminal {name}: no copper surface under its pads')
        for k in np.flatnonzero(hit):
            fields[k][3] = str(10 + t)
            text[tri[k]] = ' '.join(fields[k])
        done |= hit
        tags[name] = 10 + t
    Path(path).write_text('\n'.join(text))
    return tags
