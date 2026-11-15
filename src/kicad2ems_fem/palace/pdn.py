"""A rail's PDN in Palace (R-E22): the rail pcbqs's PDN run found (its nets, ground, decoupling capacitors, regulator
and loads), solved again in 3D, its impedance at each load beside pcbqs's.

    python -m pcbqs.pdn board.kicad_pcb --rail +3V3 ... --ports-touchstone --out pdn_3v3
    python -m kicad2ems_fem.palace.pdn --pdn pdn_3v3 --out pdn_3v3_palace

Every part on the rail is a lumped port across the gap between its closest rail pad and ground pad (a series part's:
between its two pads), and Palace solves the bare copper between them: the rail's and the ground's copper and vias
as PEC sheets and barrels, the dielectric between. Its port matrix (Z = R (I + S)(I - S)^-1, the ports' R near the
rail's impedances) is then closed by the parts, as kicad2ems closes part ports (embed.close): each series part, the
decaps (C, ESR, ESL, or their vendor file) and the regulator (R + L, or its file) by its own impedance, the loads left
open, which is what each load sees. Palace's ports stand at the parts' pads, so the copper from the pads to the
planes (pcbqs's mount inductance) is Palace's own here.

Writes <out>/pdn_palace.json: the frequencies, the bare port matrix's diagonal, each load's |Z| and phase, pcbqs's
beside them (its z and, where the run wrote --ports-touchstone, its bare port matrix), the differences, the model's
counts; <out>/pdn_palace_ports.zNp the bare port matrix. Palace's driven solver is full-wave: below a few MHz its
system is poorly conditioned, so the band starts at --fmin (default 1 MHz); pcbqs's PDN is the low-frequency answer.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from shapely.ops import nearest_points


def read_pdn(folder, board=None):
    """A pcbqs PDN run's pdn.json, and the KiCad file it read (`board` where the run did not record one)."""
    path = Path(folder) / 'pdn.json'
    if not path.is_file():
        raise ValueError(f"{folder}: no pdn.json (a pcbqs PDN run's folder)")
    r = json.loads(path.read_text())
    if 'pdn' not in r or not r['pdn'].get('parts'):
        raise ValueError(f"{folder}: the run has no PDN impedance (--no-pdn, or no plane pair)")
    board = board or r.get('board') or ''
    if not board.lower().endswith('.kicad_pcb') or not os.path.isfile(board):
        raise ValueError(f"the run's board {board!r} is not a KiCad file here (--board)")
    return r, board


def _box(g, shrink=0.0):
    from shapely.geometry import box
    a, o = g['axis'], 1 - g['axis']
    lo, hi = [0.0, 0.0], [0.0, 0.0]
    lo[a], hi[a], lo[o], hi[o] = g['lo'], g['hi'], g['c0'], g['c1']
    return box(*lo, *hi).buffer(-shrink) if shrink else box(*lo, *hi)


def pad_gap(raw, ref, nets_a, nets_b, ground_copper=None):
    """The copper gap a port of part `ref` spans (KiCad mm): on the line between its closest pads of nets_a and nets_b
    (on one copper layer, apart along x or y), from the edge of the copper its nets_a pad is in (a pour, a trace) to
    the edge of the copper of its nets_b pad, as wide as those pads face each other: where a lumped element joining
    the two would be. A part with no pad of nets_b: its nets_a pad against the nearest copper of ground_copper
    {layer: geometry} (a regulator's output). None where no such gap is clear of other copper.
    dict(layer, axis, lo, hi, c0, c1)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    fp = next((f for f in raw.footprints if f.get('ref') == ref), None)
    if fp is None:
        return None
    a_pads = [p for p in fp['pads'] if p.get('net') in nets_a]
    b_pads = [p for p in fp['pads'] if p.get('net') in nets_b]
    cands = [(a, b['geom'], b.get('net'), lay) for a in a_pads for b in b_pads for lay in set(a['layers']) & set(b['layers'])]
    if not b_pads and ground_copper:
        cands = [(a, ground_copper[lay], None, lay) for a in a_pads for lay in a['layers'] if lay in ground_copper]
    items = getattr(raw, 'items', [])
    best = None
    for a, gb, bnet, lay in cands:
        if not lay.endswith('.Cu'):
            continue
        pa, pb = nearest_points(a['geom'], gb)
        d = np.array([pb.x - pa.x, pb.y - pa.y])
        if float(np.hypot(*d)) < 1e-6:
            continue
        axis = int(np.argmax(np.abs(d)))
        if abs(d[1 - axis]) > 0.3 * abs(d[axis]):
            continue
        o = 1 - axis
        ba, bb = a['geom'].bounds, gb.bounds
        c0, c1 = max(ba[o], bb[o]), min(ba[o + 2], bb[o + 2])
        if c1 <= c0:
            c0, c1 = ba[o], ba[o + 2]
        # the copper each end is in: its pad and the copper of its net touching it
        mine = lambda pad_geom, net: unary_union([pad_geom] + [g_ for l_, n_, g_ in items
                                                              if l_ == lay and n_ == net and g_.intersects(pad_geom)])
        ca = mine(a['geom'], a.get('net'))
        cb = mine(gb, bnet) if bnet else gb
        mid = 0.5 * (c0 + c1)
        p0, p1 = [0.0, 0.0], [0.0, 0.0]
        p0[o] = p1[o] = mid
        p0[axis], p1[axis] = getattr(pa, 'xy'[axis]), getattr(pb, 'xy'[axis])
        seg = LineString([p0, p1])
        free = seg.difference(ca).difference(cb)                     # the line outside both ends' copper
        parts_ = [g_ for g_ in getattr(free, 'geoms', [free]) if not g_.is_empty]
        if not parts_:
            continue
        gap_line = min(parts_, key=lambda g_: g_.distance(a['geom']))
        xs = [c[axis] for c in gap_line.coords]
        g = dict(layer=lay, axis=axis, lo=min(xs), hi=max(xs), c0=c0, c1=c1)
        sheet = _box(g, 1e-3)
        others = [g_ for l_, n_, g_ in items if l_ == lay] + [q['geom'] for f in raw.footprints for q in f['pads']
                                                            if lay in q['layers']]
        if any(sheet.intersection(x).area > 1e-6 for x in others):  # other copper in the way: it would short it
            continue
        if best is None or (g['hi'] - g['lo']) < best[0]:
            best = (g['hi'] - g['lo'], g)
    return best[1] if best else None


def gap_port(b, name, g, z0):
    """The kicad2ems gap Port of pad_gap's gap g on board b (model coordinates: y up)."""
    from kicad2ems.board import Port
    a, o = g['axis'], 1 - g['axis']
    flip = (lambda v: -v) if a == 1 else (lambda v: v)
    lo, hi = sorted((flip(g['lo']), flip(g['hi'])))
    c = np.zeros(2)
    c[a] = 0.5 * (lo + hi)
    c[o] = 0.5 * (g['c0'] + g['c1']) * (-1 if o == 1 else 1)
    z = b.cu_z[g['layer']]
    return Port(name, g['layer'], z, z, c, a, 1, float(g['c1'] - g['c0']), float(z0), ref_layer=g['layer'], kind='gap',
                gap=(float(lo), float(hi)), source=f"part {name}")


def element_z(el, f, files=None):
    """A part's own impedance at f (its element in pdn.json: C, ESR, ESL; R, L; or its file, read again from the
    run's settings), without its mount (Palace's copper has that)."""
    w = 2 * np.pi * np.asarray(f, float)
    if el.get('file'):
        from pcbqs.pdn.models import read_impedance, table_z
        path = (files or {}).get(el['file'].split(' (')[0])
        if path is None:
            raise ValueError(f"its model file {el['file']} is not in the run's settings")
        ft, Zt, _ = read_impedance(path)
        return table_z(ft, Zt, f)
    if 'C' in el:
        return el.get('ESR', 0.0) + 1j * w * el.get('ESL', 0.0) + 1 / (1j * w * el['C'])
    return el.get('R', 0.0) + 1j * w * el.get('L', 0.0)


def close(Z, keep, loads):
    """Z (nf, n, n) with the ports not in `keep` closed by loads {index: z (nf,)}: Z_kk - Z_kp (Z_pp + z)^-1 Z_pk."""
    p = sorted(loads)
    if not p:
        return Z[:, keep][:, :, keep]
    out = np.zeros((Z.shape[0], len(keep), len(keep)), complex)
    for i in range(Z.shape[0]):
        Zpp = Z[i][np.ix_(p, p)] + np.diag([loads[k][i] for k in p])
        out[i] = Z[i][np.ix_(keep, keep)] - Z[i][np.ix_(keep, p)] @ np.linalg.solve(Zpp, Z[i][np.ix_(p, keep)])
    return out


def _settings_files(r):
    s = r.get('settings') or {}
    files = {}
    for item in s.get('cap_file') or []:
        path = item.partition('=')[2]
        path = path.rpartition(':')[0] if path.endswith((':shunt', ':series')) else path
        files[os.path.basename(path)] = path
    if s.get('vrm_file'):
        files[os.path.basename(s['vrm_file'])] = s['vrm_file']
    return files


def solve(pdn, out='pdn_palace', board=None, fmin=1e6, fmax=None, points=None, order=2, feature=1.0, hmin=0.25,
          hmax=2.0, margin=10.0, tol=1e-3, np_=0, log=print):
    from kicad2ems.board import Board
    from kicad2ems.importers import open_board
    from . import config, model, run as prun
    r, board_file = read_pdn(pdn, board)
    nets, ground = list(r['nets']), list(r['ground'])
    parts = r['pdn']['parts']
    f0 = np.asarray(r['pdn']['f'], float)
    f = f0[(f0 >= fmin) & (f0 <= (fmax or f0.max()))]
    if points:
        f = np.logspace(np.log10(f.min()), np.log10(f.max()), int(points))
    if len(f) < 2:
        raise ValueError(f"no frequencies of the run's in {fmin:g} Hz .. {fmax or f0.max():g} Hz")
    out, t0 = Path(out), time.perf_counter()
    work = out / '_palace'
    work.mkdir(parents=True, exist_ok=True)
    raw = open_board(board_file)
    from shapely.ops import unary_union
    gcu = {}
    for lay, net, geom in getattr(raw, 'items', []):              # the ground's copper by layer (KiCad mm)
        if net in ground:
            gcu.setdefault(lay, []).append(geom)
    gcu = {k: unary_union(v) for k, v in gcu.items()}
    series = r.get('series') or {}
    refs, gaps, skipped = [], [], []
    for ref in list(parts) + [s for s in series if s not in parts]:      # the parts on the rail, then the series ones
        if ref in series:
            pads = [p.get('net') for f_ in raw.footprints if f_.get('ref') == ref for p in f_['pads']]
            a = [n for n in pads if n in nets]
            g = pad_gap(raw, ref, {a[0]} if a else set(), set(nets) - ({a[0]} if a else set()))
        else:
            g = pad_gap(raw, ref, set(nets), set(ground), gcu)
        if g is None or any(_box(g).intersection(_box(h)).area > 1e-9 for h in gaps):    # (one gap, one port)
            skipped.append(ref)
            continue
        refs.append(ref)
        gaps.append(g)
    if skipped:
        log(f"palace pdn: no port for {', '.join(skipped)} (no copper gap of its own between its rail and ground, along "
            f"x or y and clear of other copper and ports): left out, as an open port in pcbqs's bare matrix")
    loads = [x for x in refs if parts.get(x, {}).get('role') == 'load']
    if not loads:
        raise ValueError("no load of the run has a port here")
    b = Board(raw, nets=sorted(set(nets) | set(ground)), need_ports=False)
    b.parts = []                                                    # each part is a port (or not on the rail)
    b.ports = [gap_port(b, x, g, 1.0) for x, g in zip(refs, gaps)]
    specs = [f"{g['layer']} {'xy'[g['axis']]} {g['lo']:.3f}..{g['hi']:.3f} mm" for g in gaps]
    log(f"palace pdn: {r['rail']} / {', '.join(ground)}, {len(refs)} ports: " +
        ', '.join(f"{x} ({parts.get(x, {}).get('role', 'series')})" for x in refs))
    h0 = max(model.feature_size(b) / feature, hmin)
    mdl = model.build(b, float(f.max()), str(work / 'model.msh'), margin=margin, hmin=h0, hmax=hmax,
                      per_feature=feature, log=log)
    mesh_s = time.perf_counter() - t0
    zl = [np.abs(np.asarray(r['pdn']['z'][x]['mag'])) for x in loads if x in r['pdn'].get('z', {})]
    zref = float(np.clip(np.median(np.concatenate(zl)) if zl else 0.1, 1e-3, 10.0))   # (the rail's own level)
    for q in mdl.ports:
        q['R'] = zref
    cfg = config.config(mdl, f, order=order, tol=tol)
    (work / 'palace.json').write_text(json.dumps(cfg, indent=1))
    t1 = time.perf_counter()
    prun.run(str(work), 'palace.json', np=np_, log=log)
    fs, S = prun.read_s(str(work / 'postpro'), len(refs))
    S = np.array([[np.interp(f, fs, S[:, i, j].real) + 1j * np.interp(f, fs, S[:, i, j].imag)
                   for j in range(len(refs))] for i in range(len(refs))]).transpose(2, 0, 1)
    I = np.eye(len(refs))
    Z = np.array([zref * (I + s) @ np.linalg.inv(I - s) for s in S])
    files = _settings_files(r)
    zs = {}
    for x in refs:
        if x in series:
            s_ = series[x]
            zs[refs.index(x)] = (s_.get('R', 0.0) or 0.0) + 1j * 2 * np.pi * f * (s_.get('L', 0.0) or 0.0)
    keep = [k for k, x in enumerate(refs) if x not in series]
    bare = close(Z, keep, zs)                                       # the series parts closed: pcbqs's bare network
    names = [refs[k] for k in keep]
    from pcbqs.pdn.cli import write_z
    write_z(str(out / f'pdn_palace_ports.z{len(names)}p'), f, bare, names,
            what="Palace's bare PDN (the copper, its series parts) between its parts' ports")
    shut = {k: element_z(parts[x]['element'], f, files) for k, x in enumerate(names) if parts[x]['role'] != 'load'}
    lk = [k for k, x in enumerate(names) if parts[x]['role'] == 'load']
    loaded = close(bare, lk, shut)
    result = dict(format=1, tool='kicad2ems_fem.palace.pdn', engine=prun.engine(str(work)), pdn=str(Path(pdn).resolve()),
                  board=board_file, rail=r['rail'], ground=ground, ports={x: s for x, s in zip(refs, specs)},
                  left_out=skipped, port_ohm=zref, frequencies_Hz=f.tolist(), z={}, bare_diag={}, tets=mdl.tets,
                  order=order, hmin=mdl.hmin, hmax=mdl.hmax, margin=mdl.margin,
                  seconds=dict(mesh=round(mesh_s, 1), solve=round(time.perf_counter() - t1, 1)))
    for i, x in enumerate(names):
        result['bare_diag'][x] = dict(mag=np.abs(bare[:, i, i]).tolist(), deg=np.degrees(np.angle(bare[:, i, i])).tolist())
    pz = r['pdn'].get('z', {})
    for j, k in enumerate(lk):
        x = names[k]
        zx = loaded[:, j, j]
        entry = dict(mag=np.abs(zx).tolist(), deg=np.degrees(np.angle(zx)).tolist())
        if x in pz:
            pm = np.interp(np.log(f), np.log(f0), np.asarray(pz[x]['mag']))
            entry.update(pcbqs_mag=pm.tolist(), ratio=(np.abs(zx) / pm).tolist())
            log(f"palace pdn: {x}: |Z| Palace / pcbqs " + ', '.join(f"{v:.3g}" for v in (np.abs(zx) / pm)[::max(1, len(f) // 6)]))
        result['z'][x] = entry
    pt = r['pdn'].get('ports') or {}
    pfile = Path(pdn) / pt['touchstone'] if pt.get('touchstone') else None
    if pfile is not None and pfile.is_file():                      # pcbqs's own bare matrix, term by term
        from pcbqs.pdn.models import read_touchstone
        fq, Mq, _, _ = read_touchstone(str(pfile))
        common = [x for x in names if x in pt.get('names', [])]
        ia, ib = [names.index(x) for x in common], [pt['names'].index(x) for x in common]
        Mi = np.array([[np.interp(np.log(f), np.log(fq), Mq[:, a, c].real) + 1j * np.interp(np.log(f), np.log(fq), Mq[:, a, c].imag)
                        for c in ib] for a in ib]).transpose(2, 0, 1)
        mine = bare[:, ia][:, :, ia]
        result['bare_vs_pcbqs'] = {x: (np.abs(mine[:, i, i]) / np.abs(Mi[:, i, i])).tolist() for i, x in enumerate(common)}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pdn_palace.json').write_text(json.dumps(result, indent=1))
    return result


def parser():
    ap = argparse.ArgumentParser(prog='python -m kicad2ems_fem.palace.pdn', description=__doc__.split('\n\n')[0])
    ap.add_argument('--pdn', required=True, help="a pcbqs PDN run's folder (its pdn.json: the rail and its parts)")
    ap.add_argument('--board', default='', help="the KiCad board, where the run did not record it")
    ap.add_argument('--out', default='pdn_palace', help='output folder')
    ap.add_argument('--fmin', type=float, default=1e6, help="lowest frequency, Hz (default 1e6: Palace's full-wave "
                                                            "solve is poorly conditioned below a few MHz)")
    ap.add_argument('--fmax', type=float, default=0.0, help="highest frequency, Hz (default the run's)")
    ap.add_argument('--points', type=int, default=0, help="frequencies, log-spaced (default the run's in the band)")
    ap.add_argument('--order', type=int, default=2, help="Palace's element order (default 2)")
    ap.add_argument('--feature', type=float, default=1.0, help='mesh cells across the smallest feature (default 1)')
    ap.add_argument('--hmin', type=float, default=0.25, help='the mesh at the copper edges no finer than this, mm')
    ap.add_argument('--hmax', type=float, default=2.0, help='the mesh in the board no coarser than this, mm')
    ap.add_argument('--margin', type=float, default=10.0, help='air around the board, mm (default 10)')
    ap.add_argument('--tol', type=float, default=1e-3, help="Palace's adaptive sweep tolerance (0: every frequency)")
    ap.add_argument('--np', type=int, default=0, help="Palace's MPI processes (default: one per fast core)")
    return ap


def main(argv=None):
    from .. import children
    children.guard()
    a = parser().parse_args(argv)
    say = lambda s: print(s, flush=True)
    try:
        solve(a.pdn, a.out, board=a.board or None, fmin=a.fmin, fmax=a.fmax or None, points=a.points or None,
              order=a.order, feature=a.feature, hmin=a.hmin, hmax=a.hmax, margin=a.margin, tol=a.tol, np_=a.np, log=say)
    except (ValueError, RuntimeError) as exc:
        print(f'palace pdn: error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
