"""The board's copper, pads and vias as shapely pieces (mm), from kicad2ems's board.json dict, shared by the solvers.

    load_board(path, nets)              # board.json's dict: a .kicad_pcb through kicad2ems, or a <name>_board.json
    copper_of(board, nets, layer)       # one shapely geometry, or None
    polygons(geom, tol)                 # its polygons, simplified to tol mm, slivers and tiny holes dropped
    pads_of(board, ref, nets)           # [(layer, polygon)]: a part's pads on those nets (through-hole: every layer)
    slab(board, layer)                  # (bottom, top) z of a copper layer's thickness, mm
"""
import json
from pathlib import Path

from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def load_board(path, nets=None):
    """board.json's dict from a .kicad_pcb (through kicad2ems; only `nets` when given) or a kicad2ems board.json."""
    path = str(path)
    if path.endswith('.json'):
        return json.loads(Path(path).read_text())
    from kicad2ems.importers import open_board
    from kicad2ems.board import Board
    only = dict(nets=sorted(nets), parts='none') if nets else {}
    return Board(open_board(path), need_ports=False, log=lambda *a: None, **only).to_dict()


def copper_of(board, nets, layer):
    nets = {nets} if isinstance(nets, str) else set(nets)
    gs = [shape(c['polygon']) for c in board['copper'].get(layer, []) if c.get('net') in nets]
    g = unary_union(gs).buffer(0) if gs else None
    return None if g is None or g.is_empty else g


def polygons(g, tol=0.0):
    """The polygons of a shapely geometry; tol > 0 simplifies them (mm) and drops tiny holes and slivers (real
    copper's near-duplicate vertices otherwise break 3D meshing)."""
    if g is None or g.is_empty:
        return []
    if tol:
        g = g.simplify(tol).buffer(0)
    out = []
    for p in (g.geoms if hasattr(g, 'geoms') else [g]):
        if isinstance(p, Polygon) and p.area > 1e-4:
            out.append(Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area > 4 * tol * tol + 1e-4]) if tol else p)
    return out


def layers(board):
    return [c['name'] for c in board['stack']['copper']]


def slab(board, layer):
    """(bottom, top) z of a copper layer's slab, mm: board.json's copper is a sheet at z with thickness t; the outer
    layers grow outward, the inner ones upward."""
    zs = {c['name']: (c['z'], c['t']) for c in board['stack']['copper']}
    z, t = zs[layer]
    if z <= min(v[0] for v in zs.values()):
        return z - t, z
    return z, z + t


def pads_of(board, ref, nets):
    """[(layer, polygon)] of part `ref`'s pads on `nets`: an SMD pad on its layer, a through-hole one (*.Cu or
    several layers) on every copper layer."""
    nets = {nets} if isinstance(nets, str) else set(nets)
    names = layers(board)
    fp = next((f for f in board['prims']['footprints'] if f['ref'] == ref), None)
    if fp is None:
        raise ValueError(f'no part {ref} on the board')
    out = []
    for p in fp.get('pads', []):
        if p.get('net') not in nets or not p.get('geom'):
            continue
        ls = names if any(l in ('*.Cu', '*.Cu*') for l in p.get('layers', [])) else [l for l in p.get('layers', []) if l in names]
        out += [(l, shape(p['geom'])) for l in ls]
    if not out:
        raise ValueError(f'part {ref} has no pad on {", ".join(sorted(nets))}')
    return out
