"""A converter's input loop in Palace (R-E22): the loop pcbqs's near-field run found (its converter, input net,
ground and input capacitors, and the copper it cropped to), solved again in 3D, for its inductance and resistance
beside pcbqs's.

    python -m kicad2ems_fem.palace.loop --mqs <pcbqs near-field run> --out loop

The board is kicad2ems's of the KiCad file the run read: the input net's and the ground's copper inside the crop,
its vias, as Palace's PEC sheets and holes (model.py). The input capacitors are metal straps across their pads
(0 ohm): what pcbqs's copper_L_H and copper_R_ohm are, the copper without the parts' own impedance (pcbqs takes
their ESR and ESL apart). The other parts are left out, as pcbqs leaves them. The source is a lumped port across
the gap between the converter's closest input and ground pads, in their copper plane (pcbqs: a source between the
pads' nodes at part height 0). Palace solves the run's harmonics; Z = R (1 + S11) / (1 - S11), L = Im Z / w,
the port's R near the loop's own |Z|.

PEC copper is the high-frequency limit of pcbqs's: its current is at the copper's surfaces facing the loop, as
pcbqs's at the harmonics where the copper is a few skin depths thick (35 um copper: ~10 MHz and up), so compare at
the higher harmonics; at the switching frequency pcbqs's L is higher (the current still in the bulk). Writes
<out>/loop.json: the frequencies, L_H, R_ohm (Palace's, ~0 with PEC), pcbqs's beside them, the model's counts.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import nearest_points



def read_mqs(folder):
    """A pcbqs near-field run's mqs.json, and the KiCad file it read."""
    path = Path(folder) / 'mqs.json'
    if not path.is_file():
        raise ValueError(f"{folder}: no mqs.json (a pcbqs near-field run's folder)")
    r = json.loads(path.read_text())
    sel = r.get('selection') or {}
    if sel.get('low') or sel.get('switch'):
        raise ValueError("a discrete switching cell's loop (two switches) is not modelled here yet: a converter IC's")
    board = r.get('source_path') or ''
    if not board.lower().endswith('.kicad_pcb') or not os.path.isfile(board):
        raise ValueError(f"the run's board {board!r} is not a KiCad file here")
    return r, board


def port_spec(raw, converter, vin, ground):
    """'P1@x,y,layer,rot' (KiCad mm) across the gap between the converter's closest input and ground pads on one
    layer, whose centres are apart along x or y (a lumped port feeds along an axis); and the gap's length."""
    fp = next((f for f in raw.footprints if f.get('ref') == converter), None)
    if fp is None:
        raise ValueError(f"converter {converter}: not on the board")
    pads = [p for p in fp['pads'] if p.get('net') in (vin, ground)]
    best = None
    for a in (p for p in pads if p['net'] == vin):
        for b in (p for p in pads if p['net'] == ground):
            for lay in set(a['layers']) & set(b['layers']):
                if not lay.endswith('.Cu') and lay != '*.Cu':
                    continue
                pa, pb = nearest_points(a['geom'], b['geom'])
                d = np.array([pb.x - pa.x, pb.y - pa.y])
                gap = float(np.hypot(*d))
                if gap < 1e-6:
                    continue
                axis = int(np.argmax(np.abs(d)))
                if abs(d[1 - axis]) > 0.3 * abs(d[axis]):          # not along an axis: a lumped port cannot feed it
                    continue
                if best is None or gap < best[0]:
                    lay_ = 'F.Cu' if lay == '*.Cu' else lay
                    mid = [(pa.x + pb.x) / 2, (pa.y + pb.y) / 2]
                    ba, bb = a['geom'].bounds, b['geom'].bounds          # across the feed: the middle of where the
                    o = 1 - axis                                          # pads face each other (nearest_points
                    lo, hi = max(ba[o], bb[o]), min(ba[o + 2], bb[o + 2])  # gives a corner)
                    if hi > lo:
                        mid[o] = (lo + hi) / 2
                    best = (gap, mid[0], mid[1], lay_, 0 if axis == 0 else 90)
    if best is None:
        raise ValueError(f"converter {converter}: no input pad ({vin}) and ground pad ({ground}) facing each other "
                         f"along x or y on one layer")
    gap, x, y, lay, rot = best
    return f"P1@{x:.6g},{y:.6g},{lay},{rot}", gap


def solve(mqs, out='loop', order=2, feature=1.0, hmin=0.25, hmax=2.0, margin=10.0, np_=0, log=print):
    from kicad2ems.board import Board
    from kicad2ems.importers import open_board
    from . import config, model, run as prun
    r, board_file = read_mqs(mqs)
    sel = r['selection']
    vin, ground, caps = sel['vin'], sel.get('ground') or 'GND', list(sel.get('capacitors') or [])
    f = np.asarray(r['frequencies_Hz'], float)
    out = Path(out)
    work = out / '_palace'
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    raw = open_board(board_file)
    spec, gap = port_spec(raw, sel['converter'], vin, ground)
    crop = shape(r['crop'])                                       # pcbqs's, model coordinates (y up)
    region = [(x, -y) for x, y in list(crop.exterior.coords)[:-1]]
    b = Board(raw, nets=[vin, ground], region=region, ports=[spec])
    found = {pt.ref for pt in b.parts}
    b.parts = [pt for pt in b.parts if pt.ref in caps]
    for pt in b.parts:                                            # the input capacitors: metal straps
        pt.short = True
    left = [c for c in caps if c not in found]
    log(f"palace loop: {sel['converter']} {vin} / {ground}, port {spec} (gap {gap:.3f} mm), capacitors "
        f"{', '.join(pt.ref for pt in b.parts) or 'none'} as straps" + (f"; not on the cropped copper: {', '.join(left)}"
                                                                        if left else ''))
    h0 = max(model.feature_size(b) / feature, hmin)              # (an RF mesh's 0.02 mm at the Jetson's pads: 2.2 M
    mdl = model.build(b, float(f.max()), str(work / 'model.msh'), margin=margin, hmin=h0, hmax=hmax,   # tets)
                      per_feature=feature, log=log)
    mesh_s = time.perf_counter() - t0
    # the port's reference resistance near the loop's own |Z| (pcbqs's wL at the middle harmonic): against 50 ohm a
    # loop of milliohms has S11 within 1e-4 of -1, and Palace's S there read 1.43 ohm of loss out of PEC copper
    pl = [v for v in (r.get('copper_L_H') or []) if isinstance(v, (int, float)) and v > 0]
    zref = float(max(2 * np.pi * np.sqrt(f.min() * f.max()) * (np.median(pl) if pl else 1e-9), 1e-4))
    for q in mdl.ports:
        q['R'] = zref
    cfg = config.config(mdl, f, order=order, tol=0)
    (work / 'palace.json').write_text(json.dumps(cfg, indent=1))
    t1 = time.perf_counter()
    prun.run(str(work), 'palace.json', np=np_, log=log)
    fs, S = prun.read_s(str(work / 'postpro'), 1)
    s11 = np.interp(f, fs, S[:, 0, 0].real) + 1j * np.interp(f, fs, S[:, 0, 0].imag)
    Z = zref * (1 + s11) / (1 - s11)
    L, R = Z.imag / (2 * np.pi * f), Z.real
    result = dict(format=1, tool='kicad2ems_fem.palace.loop', engine=prun.engine(str(work)), mqs=str(Path(mqs).resolve()),
                  board=board_file, selection=sel, port=spec, port_gap_mm=gap, port_ohm=zref,
                  straps=[pt.ref for pt in b.parts],
                  missing_capacitors=left, frequencies_Hz=f.tolist(), L_H=L.tolist(), R_ohm=R.tolist(),
                  pcbqs=dict(L_H=r.get('copper_L_H'), R_ohm=r.get('copper_R_ohm')),
                  tets=mdl.tets, order=order, feature=feature, hmin=mdl.hmin, hmax=mdl.hmax, margin=mdl.margin,
                  seconds=dict(mesh=round(mesh_s, 1), solve=round(time.perf_counter() - t1, 1)))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'loop.json').write_text(json.dumps(result, indent=1))
    for fi, li, pi in zip(f, L, r.get('copper_L_H') or [None] * len(f)):
        log(f"palace loop: {fi / 1e6:g} MHz: L {li * 1e9:.4g} nH" + (f" (pcbqs {pi * 1e9:.4g} nH)" if pi else ''))
    return result


def parser():
    ap = argparse.ArgumentParser(prog='python -m kicad2ems_fem.palace.loop', description=__doc__.split('\n\n')[0])
    ap.add_argument('--mqs', required=True, help="a pcbqs near-field run's folder (its mqs.json: the loop)")
    ap.add_argument('--out', default='loop', help='output folder')
    ap.add_argument('--order', type=int, default=2, help="Palace's element order (default 2)")
    ap.add_argument('--feature', type=float, default=1.0, help='mesh cells across the smallest feature (default 1)')
    ap.add_argument('--hmin', type=float, default=0.25, help='the mesh at the copper edges no finer than this, mm '
                                                         '(default 0.25: the Jetson U69 loop, 123,000 tets, ~12 GB)')
    ap.add_argument('--hmax', type=float, default=2.0, help='the mesh in the board no coarser than this, mm (default 2)')
    ap.add_argument('--margin', type=float, default=10.0, help='air around the cropped copper, mm (default 10)')
    ap.add_argument('--np', type=int, default=0, help="Palace's MPI processes (default: one per fast core)")
    return ap


def main(argv=None):
    from .. import children
    children.guard()
    a = parser().parse_args(argv)
    say = lambda s: print(s, flush=True)
    try:
        solve(a.mqs, a.out, order=a.order, feature=a.feature, hmin=a.hmin, hmax=a.hmax, margin=a.margin, np_=a.np,
              log=say)
    except (ValueError, RuntimeError) as exc:
        print(f'palace loop: error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
