"""The board's steady temperature with Elmer: heat conducted in all the copper, the vias and the board material, given
off by convection at the board's faces; heated by parts and by the DC current of a net (as elmer/dc.py solves it),
whose copper's resistance follows its temperature.

    python -m kicad2ems_fem.elmer.thermal board.kicad_pcb --heat Q1=0.3 --heat Q2=0.3 \\
        --net VHB --source L3 --load Q1=3 --load Q2=3 --out th_vhb

--heat REF=WATTS puts a part's dissipation into the copper under its pads, spread evenly (REF.PAD: under that pad
only); REF=WATTS@DUTY is a part that dissipates WATTS for that fraction of the time, heated by the mean. --heat-from
takes every part's dissipation from a circuit run: pcbqs's case.json (pcbqs.circuit's parts: their power_W, a
coupled winding's own loss_W, a switch's gate drive with it), or any JSON {REF: watts}; FILE@DUTY for a converter that runs part of the
time. A --heat for the same part replaces the file's figure. The temperature is the steady one for the mean power:
right when the on and off times are short beside the seconds a part's copper takes to warm. --net, --source and
--load are dc.py's: the current's own heat is added, and the drops are those of the warm copper. Writes
<out>/thermal.json (the hottest point, each part's pad temperature, the drops, the heat balance), <out>/thermal.png
(the board's faces) and Elmer's case with a .vtu in <out>/elmer.

    from kicad2ems_fem.elmer.thermal import solve
    r = solve(board, 'th', heat={'Q1': 0.3}, nets={'VHB'}, source='L3', loads={'Q1': 3.0})
    r['T_max'], r['parts']['Q1'], r['drop_mV']['Q1'], r['balance']

What is left out: the parts' own bodies (a part's heat all enters the board, its case gives none to the air: the pad
temperatures are upper bounds), solder mask, heat taken by connectors and wires, and radiation and air flow other
than as the one coefficient --htc (W/m2/K; 10 is still air with radiation near 60 C).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape

from ..geometry import layers as layer_names, load_board, pads_of
from . import run as elmer
from .layered import BARREL, BARREL_E, BOARD, COPPER, COPPER_E, layered_mesh
from .model import SIGMA_CU

K_CU = 385.0                    # W/m/K
RHO_CU, C_CU = 8960.0, 385.0    # kg/m3, J/kg/K: 3.45e6 J/m3/K, pcbqs's RHO_C_CU
RHO_FR4, C_FR4 = 1850.0, 1081.0 # 2.0e6 J/m3/K, pcbqs's RHO_C_FR4
STEPS, DECADES = 40, 4          # a transient's steps a decade, from its end / 10^(DECADES - 1): pcbqs's
K_BOARD = (0.8, 0.3)            # FR-4 along the glass weave and through it
ALPHA_CU = 0.00393              # per K, about 20 C
PLATING = 0.025
NAMES = {COPPER_E: 'copper, current', BARREL_E: 'vias, current', COPPER: 'copper', BARREL: 'vias', BOARD: 'board'}


def part_pads(board, spec):
    """[(layer, polygon)] of a part's pads that are copper ('REF'), or of one pad ('REF.PAD')."""
    ref, _, pad = spec.partition('.')
    names = layer_names(board)
    fp = next((f for f in board['prims']['footprints'] if f['ref'] == ref), None)
    if fp is None:
        raise ValueError(f'no part {ref} on the board')
    out = []
    for p in fp.get('pads', []):
        if not p.get('geom') or (pad and str(p.get('name')) != pad):
            continue
        ls = names if any(l in ('*.Cu', '*.Cu*') for l in p.get('layers', [])) else [l for l in p.get('layers', []) if l in names]
        out += [(l, shape(p['geom'])) for l in ls]
    if not out:
        raise ValueError(f"part {ref} has no copper pad{' ' + pad if pad else ''}")
    return out


def steps(seconds):
    """A transient's time steps (s): DECADES decades to `seconds`, STEPS equal steps in each, the first from 0 (as
    pcbqs's grid transient takes them, so the two compare step for step)."""
    first, out, t = seconds / 10 ** (DECADES - 1), [], 0.0
    for k in range(DECADES):
        end = first * 10 ** k
        out += [(end - t) / STEPS] * STEPS
        t = end
    return out


def sif(m, heat=None, loads=None, htc=10.0, ambient=25.0, k_board=K_BOARD, plating=PLATING, max_iterations=20,
        transient=None):
    """The .sif. m: layered_mesh's description, its boundaries named 'P:<part>' (heated), 'I:source' and 'I:<load>'.
    heat: {part: W}; loads: {ref: A} (None: no current). transient (s): the board heating from the ambient over
    that long instead (R-E21; heat only), BDF2 on steps(), the hottest point and each part's pads saved each step."""
    heat, electric = heat or {}, loads is not None
    if transient and electric:
        raise ValueError("a transient takes the parts' heat only, not a net's current")
    r = m['via_r']
    fill = float(np.mean((r ** 2 - np.maximum(r - plating, 0) ** 2) / r ** 2))
    temps = np.arange(-60.0, 401.0, 20.0)
    sigma = lambda s0: ("Variable Temperature\n    Real\n" +
                        ''.join(f'      {t:g} {s0 / (1 + ALPHA_CU * (t - 20.0)):.6g}\n' for t in temps) + "    End")
    mats = {COPPER_E: (K_CU, SIGMA_CU), BARREL_E: (K_CU * fill, SIGMA_CU * fill), COPPER: (K_CU, None),
            BARREL: (K_CU * fill, None), BOARD: (None, None)}
    rho_c = {COPPER_E: (RHO_CU, C_CU), BARREL_E: (RHO_CU * fill, C_CU), COPPER: (RHO_CU, C_CU),
             BARREL: (RHO_CU * fill, C_CU), BOARD: (RHO_FR4, C_FR4)}          # (a barrel's: its plating's share)
    bodies = materials = ''
    for i, b in enumerate(m['bodies']):
        on = electric and b in (COPPER_E, BARREL_E)
        bodies += (f"Body {i + 1}\n  Name = \"{NAMES[b]}\"\n  Target Bodies(1) = {b}\n  Equation = {1 if on or not electric else 2}\n"
                   f"  Material = {i + 1}\n  Initial Condition = 1\n{'  Body Force = 1' + chr(10) if on else ''}End\n")
        k, s = mats[b]
        materials += f"Material {i + 1}\n  Density = {rho_c[b][0]:.6g}\n  Heat Capacity = {rho_c[b][1]:.6g}\n"
        materials += (f"  Heat Conductivity = {k:.6g}\n" if k else
                      f"  Heat Conductivity(3) = {k_board[0]:g} {k_board[0]:g} {k_board[1]:g}\n")
        if on:
            materials += f"  Electric Conductivity = {sigma(s)}\n"
        materials += "End\n"
    area = {}                                                    # of each name's pads, m2
    for tag, (names, a) in m['boundaries'].items():
        for n in names:
            area[n] = area.get(n, 0.0) + a * 1e-6
    bcs, masks, k = '', '', 0
    for i, (tag, (names, _)) in enumerate(m['boundaries'].items()):
        bcs += (f"Boundary Condition {i + 1}\n  Target Boundaries(1) = {tag}\n  Heat Transfer Coefficient = {htc:g}\n"
                f"  External Temperature = {ambient:g}\n")
        q = sum(heat[n[2:]] / area[n] for n in names if n.startswith('P:'))
        if q:
            bcs += f"  Heat Flux = {q:.9g}\n"
        if transient:
            for j, spec in enumerate(heat):
                if f'P:{spec}' in names:
                    bcs += f"  part{j} = Logical True\n"
        for n in names:
            if not electric or not n.startswith('I:'):
                continue
            if n == 'I:source':
                bcs += "  Potential = 0.0\n"
            else:
                if sum(n in ns for ns, _ in m['boundaries'].values()) > 1:
                    raise ValueError(f"load {n[2:]}: its pads are heated in part; heat all of them or none")
                bcs += (f"  Potential Constant = Logical True\n  Potential Constant Flux = Real {-loads[n[2:]]:.12g}\n"
                        f"  load{k} = Logical True\n")
                masks += (f"  Variable {k + 2} = Potential\n  Operator {k + 2} = boundary mean\n"
                          f"  Mask Name {k + 2} = load{k}\n")
                k += 1
        bcs += "End\n"
    current = f"""Solver 1
  Equation = "StatCurrent"
  Variable = "Potential"
  Procedure = "StatCurrentSolveVec" "StatCurrentSolver"
  Calculate Joule Heating = Logical True
  Calculate Nodal Heating = Logical True
  Constant Weights = Logical True                 ! without it Elmer scales the nodal heat by each node's volume
  Linear System Solver = "Direct"
  Linear System Direct Method = "Umfpack"
  Steady State Convergence Tolerance = 1.0e-6
End
Solver 4
  Equation = "SaveScalars"
  Procedure = "SaveData" "SaveScalars"
  Filename = "scalars.dat"
  Variable 1 = Nodal Joule Heating
  Operator 1 = sum
{masks}End
Body Force 1
  Temperature Load = Equals "Nodal Joule Heating"
End
""" if electric else ''
    first = 1 if electric else 0                                 # the solvers run in their numbers' order
    record = ''
    if transient:                                                # each step: the hottest point, each part's pads
        record = (f'Solver {first + 2}\n  Exec Solver = After Timestep\n  Equation = "SaveScalars"\n'
                  '  Procedure = "SaveData" "SaveScalars"\n  Filename = "transient.dat"\n'
                  '  Variable 1 = Temperature\n  Operator 1 = max\n')
        for j, spec in enumerate(heat):
            record += (f'  Variable {j + 2} = Temperature\n  Operator {j + 2} = boundary int mean\n'
                       f'  Mask Name {j + 2} = part{j}\n')
        record += 'End\n'
        dts = steps(transient)
        simulation = ('  Simulation Type = Transient\n  Timestepping Method = BDF\n  BDF Order = 2\n'
                      f'  Timestep Intervals({len(dts)}) = {" ".join("1" for _ in dts)}\n'
                      f'  Timestep Sizes({len(dts)}) = {" ".join(f"{d:.9g}" for d in dts)}\n'
                      '  Steady State Max Iterations = 1\n')
        output = ''
    else:
        simulation = (f'  Simulation Type = Steady State\n'
                      f'  Steady State Max Iterations = {max_iterations if electric else 1}\n')
        output = (f'Solver {first + 2}\n  Exec Solver = After All\n  Equation = "ResultOutput"\n'
                  '  Procedure = "ResultOutputSolve" "ResultOutputSolver"\n  Output Directory = "."\n'
                  '  Output File Name = "case"\n  Vtu Format = Logical True\n  Ascii Output = Logical True\n'
                  '  Save Geometry Ids = Logical True\nEnd\n')
    equations = ("Equation 1\n  Active Solvers(2) = 1 2\nEnd\nEquation 2\n  Active Solvers(1) = 2\nEnd\n" if electric
                 else "Equation 1\n  Active Solvers(1) = 1\nEnd\n")
    return f"""Header
  Mesh DB "." "mesh"
End
Simulation
  Coordinate System = Cartesian 3D
  Coordinate Scaling = 1.0e-3
{simulation}  Max Output Level = 4
End
{bodies}{equations}Initial Condition 1
  Temperature = {ambient:g}
End
{materials}{current}Solver {first + 1}
  Equation = "Heat Equation"
  Variable = "Temperature"
  Procedure = "HeatSolveVec" "HeatSolver"
  Vector Assembly = Logical False                 ! the vectorized assembly takes no conductivity tensor
  Nonlinear System Max Iterations = 1
  Linear System Solver = "Direct"
  Linear System Direct Method = "Umfpack"
  Steady State Convergence Tolerance = 1.0e-6
End
{output}{record}{bcs}"""


def read_vtu(path, name):
    """A nodal field of an ASCII .vtu, and the points."""
    text = Path(path).read_text()

    def array(pattern):
        hit = re.search(r'<DataArray[^>]*' + pattern + r'[^>]*>(.*?)</DataArray>', text, re.S | re.I)
        if hit is None:
            raise RuntimeError(f'{path}: no {pattern}')
        return np.array(hit.group(1).split(), float)
    pts = re.search(r'<Points>\s*<DataArray[^>]*>(.*?)</DataArray>', text, re.S)
    return array(f'Name="{name}"'), np.array(pts.group(1).split(), float).reshape(-1, 3)


def circuit_heat(board, path, duty=1.0):
    """The parts' dissipation from a circuit run, for solve(heat=): ({REF: W}, {name: W left out}). path: a
    case.json of pcbqs's (pcbqs.circuit's parts: each one's power_W; a coupled winding's own loss_W, its V·I being
    the power it passes on as well; a switch's gate_power_W with it) or a JSON {REF: W}. Left out: what is not a part
    with copper pads on the board (the copper model's ports, the source, the coil and what lies beyond it), what
    dissipates nothing, and a coupled winding whose own loss the run did not measure."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and isinstance(data.get('parts'), dict):
        # pcbqs.circuit's case.json: {REF: {type, power_W, ...}} (a source's is negative: what it delivers)
        own = lambda p: p['loss_W'] if 'loss_W' in p else (None if p.get('coupled_with') else p['power_W'])
        data = {ref: own(p) + p.get('gate_power_W', 0.0) for ref, p in data['parts'].items()
                if isinstance(p, dict) and isinstance(p.get('power_W'), (int, float)) and own(p) is not None}
    if not isinstance(data, dict) or not all(isinstance(v, (int, float)) for v in data.values()):
        raise ValueError(f"{path}: expected a circuit run's case.json (its parts), or {{REF: watts}}")
    if not 0 < duty <= 1:
        raise ValueError('duty: a fraction of the time, above 0 and at most 1')
    refs = {f['ref'] for f in board['prims']['footprints'] if any(p.get('geom') for p in f.get('pads', []))}
    heat = {ref: w * duty for ref, w in data.items() if ref in refs and w > 0}
    for ref in list(heat):
        try:
            part_pads(board, ref)
        except ValueError:
            del heat[ref]
    return heat, {ref: w * duty for ref, w in data.items() if ref not in heat}


def solve(board, out='thermal', heat=None, nets=None, source=None, loads=None, copper=None, htc=10.0, ambient=25.0,
          h=0.3, hmax=2.0, crop=None, k_board=K_BOARD, plating=PLATING, log=print, transient=0.0):
    """The board's temperatures. heat: {REF or REF.PAD: W}; nets, source, loads: the net that carries current (as
    dc.solve), or none; copper: the other nets kept (None: all the board's). transient (s): also the heating from
    the ambient over that long (heat only; R-E21), <out>/thermal-transient.json, and its summary in the result's
    `transient`. Returns the dict written to <out>/thermal.json."""
    if transient and (nets or loads or source):
        raise ValueError("--transient takes the parts' heat only, not a net's current")
    if transient and transient < 0:
        raise ValueError('--transient: seconds, positive')
    ok, why = elmer.found()
    if not ok:
        raise RuntimeError(why)
    heat, loads = dict(heat or {}), dict(loads or {})
    nets = set() if nets is None else {nets} if isinstance(nets, str) else set(nets)
    if nets and (not source or not loads or any(a <= 0 for a in loads.values())):
        raise ValueError('a net that carries current needs its source and loads, each a positive current')
    if (loads or source) and not nets:
        raise ValueError('source and loads need their net')
    if not heat and not nets:
        raise ValueError('nothing heats the board: give parts (heat) or a net with its source and loads')
    if any(w < 0 for w in heat.values()):
        raise ValueError('heat: each part a dissipation in watts, not negative')
    if isinstance(board, (str, Path)):
        board = load_board(board)
    out = Path(out)
    case = out / 'elmer'
    case.mkdir(parents=True, exist_ok=True)
    terms = {f'P:{spec}': part_pads(board, spec) for spec in heat}
    if nets:
        terms['I:source'] = pads_of(board, source, nets)
        terms.update({f'I:{ref}': pads_of(board, ref, nets) for ref in loads})
    t = time.perf_counter()
    m = layered_mesh(board, case / 'mesh', electric=nets, copper=copper, terminals=terms, crop=crop, h=h, hmax=hmax,
                     log=log)
    mesh_s = time.perf_counter() - t
    t = time.perf_counter()
    # one thread with a current: the loop that fills Elmer's nodal Joule heating (StatCurrentSolver_post) is not
    # safe with several, and now and then gives the heat equation another source
    said = elmer.solve(case, sif(m, heat, loads if nets else None, htc=htc, ambient=ambient, k_board=k_board, plating=plating),
                threads=1 if nets else None)
    solve_s = time.perf_counter() - t
    T, pts = read_vtu(case / 'case_t0001.vtu', 'temperature')
    if len(T) != len(m['nodes']) or np.abs(pts - m['nodes'] * 1e-3).max() > 1e-6 * np.abs(pts).max():  # 8 digits
        raise RuntimeError("Elmer's .vtu does not have the mesh's nodes in their order")
    tf, ts = T[m['faces']].mean(1), T[m['sides']].mean(1)
    given = htc * 1e-6 * float(((tf - ambient) * m['face_area']).sum() + ((ts - ambient) * m['side_area']).sum())
    hot = int(np.argmax(T))
    z = m['nodes'][hot, 2]
    result = dict(format=1, tool='kicad2ems_fem.elmer.thermal', engine=elmer.engine(said), T_max=float(T.max()), T_min=float(T.min()),
                  ambient=ambient, htc=htc,
                  hottest=dict(x=float(m['nodes'][hot, 0]), y=float(m['nodes'][hot, 1]),
                               layer=min(m['cells'], key=lambda c: abs(0.5 * (c[0] + c[1]) - z))[2]),
                  parts={}, heat=heat, wedges=m['wedges'], triangles=m['triangles'], vias=m['vias'], h=h, hmax=hmax,
                  seconds=dict(mesh=round(mesh_s, 1), solve=round(solve_s, 1)))
    for spec in heat:
        sel = np.isin(m['face_tag'], [tag for tag, (names, _) in m['boundaries'].items() if f'P:{spec}' in names])
        result['parts'][spec] = float((tf[sel] * m['face_area'][sel]).sum() / m['face_area'][sel].sum())
    joule = 0.0
    if nets:
        rows = np.loadtxt(case / 'scalars.dat', ndmin=2)
        cols = [re.sub(r'^\s*\d+:\s*', '', l).strip().lower() for l in
                Path(str(case / 'scalars.dat') + '.names').read_text().splitlines() if re.match(r'^\s*\d+:', l)]
        col = lambda *words: next(i for i, c in enumerate(cols) if all(w in c for w in words))
        joule = float(rows[-1, col('joule')])
        drop = lambda row: {ref: float(-row[col('potential', f'load{k}')] * 1e3) for k, ref in enumerate(loads)}
        result.update(nets=sorted(nets), source=source, loads=loads, drop_mV=drop(rows[-1]),
                      drop_ambient_mV=drop(rows[0]), loss_W=joule, loss_ambient_W=float(rows[0, col('joule')]),
                      iterations=len(rows))
    put = joule + sum(heat.values())
    result['balance'] = dict(in_W=put, out_W=given, error=(given - put) / put if put else 0.0)
    if transient:
        t = time.perf_counter()
        elmer.solve(case, sif(m, heat, None, htc=htc, ambient=ambient, k_board=k_board, plating=plating,
                              transient=transient))
        tr = _transient(case / 'transient.dat', transient, heat, result, ambient)
        tr['seconds'] = round(time.perf_counter() - t, 1)
        (out / 'thermal-transient.json').write_text(json.dumps(tr, indent=1))
        try:
            transient_picture(out / 'thermal-transient.png', tr, ambient)
        except Exception as exc:                                 # the numbers stand without the picture
            log(f'no transient picture: {exc}')
        result['transient'] = dict(end_s=transient, to_63_s=tr['to_63_s'], T_max_end=tr['T_max'][-1],
                                   seconds=tr['seconds'])
        result['seconds']['transient'] = tr['seconds']
        log(f"transient: {len(tr['times_s']) - 1} steps to {transient:g} s, {tr['seconds']:.0f} s; the hottest point "
            f"{tr['T_max'][-1]:.1f} C at the end (steady {result['T_max']:.1f} C)")
    (out / 'thermal.json').write_text(json.dumps(result, indent=1))
    np.savez_compressed(out / 'thermal.npz', T=T, nodes=m['nodes'], xy=m['xy'], tri=m['tri'], node=m['node'])
    try:
        picture(out / 'thermal.png', m, T, result, board)
    except Exception as exc:                                     # the numbers stand without the picture
        log(f'no picture: {exc}')
    return result


def _transient(path, end, heat, steady, ambient):
    """The transient run's record (transient.dat: a row a step, the hottest point then each part's pads) as
    dict(times_s, T_max, parts={spec: [C]}, to_63_s={spec: s}): 0 s is the ambient, before the heat; to_63_s when
    a part's pads are 63 % of the way to their steady temperature (pcbqs's grid transient reports the same)."""
    rows = np.loadtxt(path, ndmin=2)
    dts = steps(end)
    if len(rows) != len(dts):
        raise RuntimeError(f"Elmer's transient record has {len(rows)} steps, not {len(dts)}")
    times = np.r_[0.0, np.cumsum(dts)]
    tmax = np.r_[ambient, rows[:, 0]]
    parts, to63 = {}, {}
    for j, spec in enumerate(heat):
        c = np.r_[ambient, rows[:, j + 1]]
        parts[spec] = c.tolist()
        target = ambient + 0.632 * (steady['parts'][spec] - ambient)
        hit = int(np.argmax(c >= target)) if (c >= target).any() else 0
        if hit:
            a, b = c[hit - 1], c[hit]
            to63[spec] = float(times[hit - 1] + (times[hit] - times[hit - 1]) * (target - a) / max(b - a, 1e-12))
    return dict(times_s=times.tolist(), T_max=tmax.tolist(), parts=parts, to_63_s=to63, end_s=end,
                steady=dict(T_max=steady['T_max'], parts=steady['parts']))


def transient_picture(path, tr, ambient):
    """Each part's pads and the hottest point against time (log), the steady values dashed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array(tr['times_s'])
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    ax.semilogx(t[1:], tr['T_max'][1:], 'k', lw=1.4, label='hottest point')
    for spec, c in tr['parts'].items():
        line, = ax.semilogx(t[1:], c[1:], lw=1.2, label=spec)
        ax.axhline(tr['steady']['parts'][spec], color=line.get_color(), ls='--', lw=0.7)
    ax.axhline(ambient, color='0.6', lw=0.7)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('temperature (°C)')
    ax.set_title('Heating from the ambient (Elmer); steady dashed')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def picture(path, m, T, result, board=None):
    """The temperature of the board's top and bottom faces, one scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    node, xy, tri = m['node'], m['xy'], m['tri']
    nk = node.shape[0]
    have = node >= 0
    first = np.argmax(have, axis=0)
    last = nk - 1 - np.argmax(have[::-1], axis=0)
    cols = np.arange(node.shape[1])
    faces = {'top': T[node[last, cols]], 'bottom': T[node[first, cols]]}
    lo, hi = float(T.min()), float(T.max())
    x0, y0, x1, y1 = (*xy.min(0), *xy.max(0))
    w = 4.6 * (x1 - x0) / max(y1 - y0, 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(2 * min(max(w, 3.0), 7.0) + 1.2, 5.2), constrained_layout=True)
    tr = Triangulation(xy[:, 0], xy[:, 1], tri)
    for ax, (side, v) in zip(axes, faces.items()):
        im = ax.tripcolor(tr, v, cmap='YlOrRd', vmin=lo, vmax=hi, shading='gouraud', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'{side}: {v.min():.1f} to {v.max():.1f} °C', fontsize=10, loc='left')
        ax.tick_params(labelsize=8, length=2, color='0.5')
        for s in ax.spines.values():
            s.set_color('0.7')
        ax.set_xlabel('mm', fontsize=8)
        if side == 'bottom':
            ax.invert_xaxis()                                    # seen from below
    h = result['hottest']
    axes[0].plot(h['x'], h['y'], marker='+', color='0.1', ms=9, mew=1.2)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label('°C', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    b = result['balance']
    fig.suptitle(f"hottest {result['T_max']:.1f} °C (+) in {result['ambient']:g} °C air, {b['in_W']:.3g} W, "
                 f"{result['htc']:g} W/m²/K", fontsize=11, x=0.02, ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pairs(ap, items, flag, what):
    """{REF: value} of REF=VALUE items; VALUE@DUTY is the value for that fraction of the time: its mean."""
    out = {}
    for item in items or []:
        ref, _, v = item.partition('=')
        v, _, duty = v.partition('@')
        try:
            out[ref.strip()] = float(v) * (float(duty) if duty else 1.0)
            if duty and not 0 < float(duty) <= 1:
                raise ValueError
        except ValueError:
            ap.error(f'{flag} {item!r}: expected REF={what}' + ('[@DUTY], the duty above 0 and at most 1' if flag == '--heat' else ''))
    return out


def parser():
    ap = argparse.ArgumentParser(prog='python -m kicad2ems_fem.elmer.thermal', description=__doc__.split('\n\n')[0])
    ap.add_argument('board', help='.kicad_pcb, or a kicad2ems <name>_board.json')
    ap.add_argument('--heat', action='append', help='REF=WATTS or REF.PAD=WATTS; WATTS@DUTY for part of the time (repeat)')
    ap.add_argument('--heat-from', help="FILE[@DUTY]: a pcbqs circuit run's case.json, or a JSON {REF: watts}")
    ap.add_argument('--net', action='append', help='a net that carries current (repeat: joined as one conductor)')
    ap.add_argument('--source', help="the part whose pads on the net are held at 0 V")
    ap.add_argument('--load', action='append', help='REF=AMPS (repeat)')
    ap.add_argument('--copper', action='append', help='keep only these other nets (default: all the copper)')
    ap.add_argument('--htc', type=float, default=10.0, help='heat given to the air, W/m2/K (default 10)')
    ap.add_argument('--ambient', type=float, default=25.0, help='the air, C (default 25)')
    ap.add_argument('--k-board', default='0.8,0.3', help="the board's conductivity in plane,through, W/m/K (default 0.8,0.3)")
    ap.add_argument('--region', help='x0,y0,x1,y1 (mm, model coordinates): only the board inside it')
    ap.add_argument('--out', default='thermal', help='output folder')
    ap.add_argument('--h', type=float, default=0.3, help='mesh size in the copper with current and at pads, mm (default 0.3)')
    ap.add_argument('--hmax', type=float, default=2.0, help='mesh size elsewhere, mm (default 2)')
    ap.add_argument('--plating', type=float, default=PLATING, help='via plating, mm (default 0.025)')
    ap.add_argument('--transient', type=float, default=0.0,
                    help="s: also the heating from the ambient over this long, the parts' heat on at 0 (no net's "
                         "current): thermal-transient.json")
    return ap


def main(argv=None):
    from .. import children
    children.guard()                                        # its solvers stopped with it
    ap = parser()
    a = ap.parse_args(argv)
    heat, loads = _pairs(ap, a.heat, '--heat', 'WATTS'), _pairs(ap, a.load, '--load', 'AMPS')
    say = lambda s: print('fem thermal:', s, flush=True)
    try:
        board = load_board(a.board)
        if a.heat_from:
            path, _, duty = a.heat_from.rpartition('@') if re.search(r'@[0-9.eE+-]+$', a.heat_from) else (a.heat_from, '', '')
            found, left = circuit_heat(board, path, float(duty) if duty else 1.0)
            say(f"{path}: {len(found)} parts, {sum(found.values()):.4g} W" + (f" at {float(duty):g} of the time" if duty else '')
                + (f"; {len(left)} entries are not parts on the board or dissipate nothing "
                   f"({', '.join(list(left)[:4])}{', ...' if len(left) > 4 else ''})" if left else ''))
            specified = {spec.partition('.')[0] for spec in heat}
            heat = {**{ref: w for ref, w in found.items() if ref not in specified}, **heat}
        r = solve(board, a.out, heat=heat, nets=set(a.net or []), source=a.source, loads=loads, copper=a.copper,
                  htc=a.htc, ambient=a.ambient, h=a.h, hmax=a.hmax, plating=a.plating,
                  crop=box(*[float(v) for v in a.region.split(',')]) if a.region else None,
                  k_board=tuple(float(v) for v in a.k_board.split(',')), log=say, transient=a.transient)
    except (ValueError, RuntimeError) as exc:
        print(f'fem thermal: error: {exc}', file=sys.stderr)
        return 2
    say(f"hottest {r['T_max']:.1f} C at ({r['hottest']['x']:.1f}, {r['hottest']['y']:.1f}) on {r['hottest']['layer'] or 'the board'}; "
        f"coolest {r['T_min']:.1f} C; air {r['ambient']:g} C")
    for ref, t in r['parts'].items():
        say(f'{ref}: {t:.1f} C at its pads ({r["heat"][ref]:g} W)')
    for ref, mv in r.get('drop_mV', {}).items():
        say(f"{ref}: {mv:.4f} mV ({r['drop_ambient_mV'][ref]:.4f} mV with the copper at the air's temperature)")
    if 'loss_W' in r:
        say(f"{r['loss_W'] * 1e3:.4f} mW in the copper that carries the current")
    for ref, t63 in (r.get('transient') or {}).get('to_63_s', {}).items():
        say(f"{ref}: 63 % of its rise at {t63:.4g} s")
    b = r['balance']
    say(f"{b['in_W']:.4g} W in, {b['out_W']:.4g} W to the air ({b['error'] * 100:+.2f}%); {r['wedges']:,} wedges; "
        f"wrote {Path(a.out) / 'thermal.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
