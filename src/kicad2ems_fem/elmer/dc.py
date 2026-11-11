"""DC IR drop of a net's copper with Elmer: steady current conduction in the 3D copper and via barrels.

    python -m kicad2ems_fem.elmer.dc board.kicad_pcb --net VHB --source L3 --load Q1=0.67 --load Q2=0.67 --out dc_vhb

The source part's pads on the net are held at 0 V; each load draws its current through its pads on the net as one
equipotential terminal. Several --net are one conductor (a rail through 0 ohm links joined copper); series parts
(sense resistors, beads) are not modelled: pcbqs.pdn does rail trees and returns, this is the 3D check of their
copper. Writes <out>/dc.json (each load's voltage, the copper's loss, the mesh) and Elmer's case in <out>/elmer.

    from kicad2ems_fem.elmer.dc import solve
    r = solve(board, {'VHB'}, 'L3', {'Q1': 0.67}, out='dc_vhb')   # r['V']['Q1'] (V, negative: the drop), r['loss_W']
"""
import argparse
import json
import sys
import time
from pathlib import Path

from ..geometry import load_board, pads_of
from . import run as elmer
from .model import SIGMA_CU, copper_model

PLATING = 0.025


def sif(m, loads):
    """The .sif: the source pads at 0 V, each load an equipotential terminal drawing its current (Potential Constant
    with a flux); each load's mean potential saved."""
    bodies = "Body 1\n  Target Bodies(1) = 1\n  Equation = 1\n  Material = 1\nEnd\n"
    if m['barrels']:
        bodies += "Body 2\n  Target Bodies(1) = 2\n  Equation = 1\n  Material = 2\nEnd\n"
    bcs = [f"Boundary Condition 1\n  Target Boundaries(1) = {m['boundaries']['source']}\n  Potential = 0.0\nEnd\n"]
    masks = []
    for k, (name, amps) in enumerate(loads.items()):
        bcs.append(f"Boundary Condition {k + 2}\n  Name = \"{name}\"\n  Target Boundaries(1) = {m['boundaries'][name]}\n"
                   f"  Potential Constant = Logical True\n  Potential Constant Flux = Real {-amps:.12g}\n"
                   f"  load{k} = Logical True\nEnd\n")
        masks.append(f"  Variable {k + 1} = Potential\n  Operator {k + 1} = boundary mean\n  Mask Name {k + 1} = load{k}\n")
    return f"""Header
  Mesh DB "." "mesh"
End
Simulation
  Coordinate System = Cartesian 3D
  Coordinate Scaling = 1.0e-3
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Max Output Level = 4
End
{bodies}Equation 1
  Active Solvers(2) = 1 2
End
Material 1
  Electric Conductivity = {SIGMA_CU:.6g}
End
Material 2
  Electric Conductivity = {m['barrel_sigma']:.6g}
End
Solver 1
  Equation = "StatCurrent"
  Variable = "Potential"
  Procedure = "StatCurrentSolveVec" "StatCurrentSolver"
  ! direct: the thin copper slabs' flat elements made the iterative solvers diverge
  Linear System Solver = "Direct"
  Linear System Direct Method = "Umfpack"
End
Solver 2
  Exec Solver = After All
  Equation = "SaveScalars"
  Procedure = "SaveData" "SaveScalars"
  Filename = "scalars.dat"
{''.join(masks)}End
{''.join(bcs)}"""


def solve(board, nets, source, loads, out='dc', h=0.25, h_fine=0.05, plating=PLATING, log=print):
    """Each load's voltage (V, relative to the source: negative, the drop) and the copper's loss (W)."""
    ok, why = elmer.found()
    if not ok:
        raise RuntimeError(why)
    if not loads or any(a <= 0 for a in loads.values()):
        raise ValueError('loads: one or more, each a positive current')
    out = Path(out)
    case = out / 'elmer'
    case.mkdir(parents=True, exist_ok=True)
    nets = {nets} if isinstance(nets, str) else set(nets)
    terms = {'source': pads_of(board, source, nets), **{ref: pads_of(board, ref, nets) for ref in loads}}
    t = time.perf_counter()
    m = copper_model(board, nets, terms, case / 'copper.msh', h=h, h_fine=h_fine, plating=plating, log=log)
    mesh_s = time.perf_counter() - t
    elmer.elmergrid(case / 'copper.msh', case / 'mesh')
    t = time.perf_counter()
    elmer.solve(case, sif(m, loads))
    s = elmer.scalars(case / 'scalars.dat')
    V = {ref: float(next(v for c, v in s.items() if 'potential' in c.lower() and f'load{k}' in c.lower()))
         for k, ref in enumerate(loads)}
    # the power the loads draw through the copper. Not Elmer's Joule heating field: the loop that fills it is not
    # safe with several threads (StatCurrentSolver_post), and now and then gives another figure
    loss = float(sum(-V[ref] * amps for ref, amps in loads.items()))
    result = dict(format=1, nets=sorted(nets), source=source, loads=dict(loads), V=V,
                  drop_mV={r: -v * 1e3 for r, v in V.items()}, loss_W=loss, tets=m['tets'], vias=m['vias'],
                  h=h, h_fine=h_fine, seconds=dict(mesh=round(mesh_s, 1), solve=round(time.perf_counter() - t, 1)))
    (out / 'dc.json').write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m kicad2ems_fem.elmer.dc', description=__doc__.split('\n\n')[0])
    ap.add_argument('board', help='.kicad_pcb, or a kicad2ems <name>_board.json')
    ap.add_argument('--net', action='append', required=True, help='the net (repeat: nets joined as one conductor)')
    ap.add_argument('--source', required=True, help="the part whose pads on the net are held at 0 V")
    ap.add_argument('--load', action='append', required=True, help='REF=AMPS (repeat)')
    ap.add_argument('--out', default='dc', help='output folder')
    ap.add_argument('--h', type=float, default=0.25, help='mesh size in the copper, mm (default 0.25)')
    ap.add_argument('--h-fine', type=float, default=0.05, help='mesh size at vias and pad edges, mm (default 0.05)')
    ap.add_argument('--plating', type=float, default=PLATING, help='via plating, mm (default 0.025)')
    a = ap.parse_args(argv)
    loads = {}
    for item in a.load:
        ref, _, amps = item.partition('=')
        try:
            loads[ref.strip()] = float(amps)
        except ValueError:
            ap.error(f'--load {item!r}: expected REF=AMPS')
    try:
        board = load_board(a.board, set(a.net))
        r = solve(board, set(a.net), a.source, loads, out=a.out, h=a.h, h_fine=a.h_fine, plating=a.plating,
                  log=lambda s: print('fem dc:', s, flush=True))
    except (ValueError, RuntimeError) as exc:
        print(f'fem dc: error: {exc}', file=sys.stderr)
        return 2
    for ref, mv in r['drop_mV'].items():
        print(f'fem dc: {ref}: {mv:.4f} mV')
    print(f"fem dc: {r['loss_W'] * 1e3:.4f} mW in the copper; {r['tets']:,} tetrahedra; wrote {Path(a.out) / 'dc.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
