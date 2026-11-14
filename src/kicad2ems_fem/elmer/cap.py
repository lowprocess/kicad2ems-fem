"""
Capacitance between nets with Elmer's electrostatic solver: the same model and mesh as Palace's tool
(palace/cap.py: the finite dielectric, air around it, each net's copper a conductor), solved by a second program.

    python -m kicad2ems_fem.elmer.cap board.kicad_pcb --net SW --net VIN --net GND --out cap_sw
    r = solve(board, ['SW', 'VIN', 'GND'], 'cap_sw')      # r['C']: the Maxwell matrix (F), r['names']

The options are Palace's tool's (--thick, --h, --hmax, --margin, --region, --far, --order). The two share the mesh
and so its error; what this checks is the solver and the reading of its output. The mesh's own error shows in
refining --h, with either.
"""
import json
import os
import time

import numpy as np
from shapely.geometry import box

from ..geometry import load_board
from ..palace.cap import _groups, cap_model, report
from . import run as elmer

EPS0 = 8.8541878128e-12


def sif(m, order=2, far='open'):
    """The .sif: each conductor a constraint mode (1 V on it, 0 V on the others), the charges they draw the
    capacitance matrix. The air box's faces carry no charge (nothing set: the natural condition) or are at 0 V."""
    if far not in ('open', 'ground'):
        raise ValueError("far: 'open' or 'ground'")
    mats = [(a, er) for a, er in m['diel']] + [(m['air'], 1.0)]
    bodies = ''.join(f"Body {i + 1}\n  Target Bodies(1) = {a}\n  Equation = 1\n  Material = {i + 1}\nEnd\n"
                     for i, (a, _) in enumerate(mats))
    materials = ''.join(f"Material {i + 1}\n  Relative Permittivity = {er:.9g}\nEnd\n" for i, (_, er) in enumerate(mats))
    bcs = ''.join(f"Boundary Condition {i + 1}\n  Name = \"{name}\"\n  Target Boundaries(1) = {a}\n"
                  f"  Capacitance Body = {i + 1}\nEnd\n" for i, (name, a) in enumerate(m['terminals']))
    if far == 'ground':
        bcs += (f"Boundary Condition {len(m['terminals']) + 1}\n  Target Boundaries(1) = {m['outer']}\n"
                "  Potential = 0.0\nEnd\n")
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
Constants
  Permittivity Of Vacuum = {EPS0:.10g}
End
{bodies}Equation 1
  Active Solvers(1) = 1
End
{materials}Solver 1
  Equation = "Stat Elec Solver"
  Variable = "Potential"
  Procedure = "StatElecSolveVec" "StatElecSolver"
  Element = "p:{int(order)}"
  Calculate Capacitance Matrix = Logical True
  Capacitance Matrix Filename = "capacitance.dat"
  Fix Constraint Modes p = Logical True
  Nonlinear System Max Iterations = 1
  Linear System Solver = "Iterative"
  Linear System Iterative Method = "CG"
  Linear System Preconditioning = "ILU0"
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0e-10
  Linear System Abort Not Converged = Logical True
End
{bcs}"""


def maxwell(E):
    """The Maxwell matrix from Elmer's: Elmer writes each conductor's capacitance to everything else at 0 V other
    than the conductors (the row's sum) on the diagonal, and the mutual capacitances, positive, off it."""
    E = np.asarray(E, float)
    C = -E
    np.fill_diagonal(C, E.sum(axis=1))
    return C


def solve(board, nets, out, thick=False, h=None, hmax=None, margin=None, crop=None, order=2, far='open', log=print):
    """The capacitance matrix between `nets` (each a conductor; 'A+B' joins nets). Returns dict(names, C (F), tets,
    h, hmax, margin, seconds) and writes it to <out>/cap.json; Elmer's case is in <out>/elmer."""
    ok, why = elmer.found()
    if not ok:
        raise RuntimeError(why)
    if isinstance(board, (str, os.PathLike)):
        board = load_board(board, {n for _, ns in _groups(nets) for n in ns})
    case = os.path.join(str(out), 'elmer')
    os.makedirs(case, exist_ok=True)
    t = time.time()
    m = cap_model(board, nets, os.path.join(case, 'model.msh'), thick=thick, h=h, hmax=hmax, margin=margin,
                  crop=crop, log=log, binary=False)
    mesh_s = time.time() - t
    elmer.elmergrid(os.path.join(case, 'model.msh'), os.path.join(case, 'mesh'))
    t = time.time()
    elmer.solve(case, sif(m, order=order, far=far))
    n = len(m['terminals'])
    C = maxwell(np.loadtxt(os.path.join(case, 'capacitance.dat'), ndmin=2))
    if C.shape != (n, n):
        raise RuntimeError(f"Elmer's capacitance matrix is {C.shape}, expected {n} x {n}")
    r = dict(tool='kicad2ems_fem.elmer.cap', names=[name for name, _ in m['terminals']], C=C.tolist(), thick=bool(thick),
             far=far, order=order, solver='elmer', tets=m['tets'], h=m['h'], hmax=m['hmax'], margin=m['margin'], mesh_s=round(mesh_s, 1),
             seconds=round(time.time() - t, 1))
    with open(os.path.join(str(out), 'cap.json'), 'w') as fh:
        json.dump(r, fh, indent=1)
    return r


def parser():
    import argparse
    ap = argparse.ArgumentParser(prog='python -m kicad2ems_fem.elmer.cap', description=__doc__.split('\n\n')[0])
    ap.add_argument('board', help='a .kicad_pcb, or a kicad2ems <name>_board.json')
    ap.add_argument('--net', action='append', required=True, help='a conductor: a net, or nets joined as A+B; repeat')
    ap.add_argument('--out', required=True, help='the folder of the results')
    ap.add_argument('--thick', action='store_true', help='copper as solids of its thickness, not sheets')
    ap.add_argument('--h', type=float, help='mesh size at the copper edges, mm')
    ap.add_argument('--hmax', type=float, help='mesh size away from the copper, mm')
    ap.add_argument('--margin', type=float, help='air around the board, mm')
    ap.add_argument('--region', help='x0,y0,x1,y1 (mm, model coordinates): only the board inside it')
    ap.add_argument('--far', choices=('open', 'ground'), default='open', help="the air box's faces (default open)")
    ap.add_argument('--order', type=int, default=2, help="the element order (default 2)")
    return ap


def main(argv=None):
    from .. import children
    children.guard()                                        # its solvers stopped with it
    a = parser().parse_args(argv)
    crop = box(*[float(v) for v in a.region.split(',')]) if a.region else None
    r = solve(a.board, a.net, a.out, thick=a.thick, h=a.h, hmax=a.hmax, margin=a.margin, crop=crop, order=a.order,
              far=a.far)
    print(f"capacitance (Maxwell), pF: {r['tets']:,} tetrahedra, {r['seconds']} s")
    print(report(r))


if __name__ == '__main__':
    main()
