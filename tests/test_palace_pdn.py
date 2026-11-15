"""A rail's PDN in Palace (R-E22, palace/pdn.py): each part's port at its copper gap, the port matrix closed by the
parts, and a short run on the buck example's output rail against pcbqs's PDN."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from kicad2ems_fem.palace import pdn as P

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUCK = os.path.join(ROOT, 'pcbqs', 'src', 'pcbqs', 'circuit', 'examples', '04_buck.kicad_pcb')
need_buck = pytest.mark.skipif(not os.path.isfile(BUCK), reason='no pcbqs checkout beside this one')


def test_closing_ports_is_the_networks_own_reduction():
    """close(): ports shut by impedances, the others kept: as attaching them to the network."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Z = (A @ A.conj().T + 3 * np.eye(3))[None]                     # a passive 3-port's Z at one frequency
    z2 = np.array([0.7 + 0.2j])
    got = P.close(Z, [0, 1], {2: z2})[0]
    Y = np.linalg.inv(Z[0]); Y[2, 2] += 1 / z2[0]                  # (the load's admittance at port 3)
    want = np.linalg.inv(Y)[:2, :2]
    assert np.allclose(got, want)
    assert np.allclose(P.close(Z, [0], {})[0], Z[0][:1, :1])
    f = np.array([1e6, 1e8])
    zc = P.element_z(dict(C=1e-6, ESR=5e-3, ESL=1e-9), f)
    assert zc[0] == pytest.approx(5e-3 + 1j * (2 * np.pi * 1e6 * 1e-9 - 1 / (2 * np.pi * 1e6 * 1e-6)))


@need_buck
def test_each_part_s_port_is_its_own_copper_gap():
    """On the buck's output rail (its pads in pours, the out and GND pours 0.4 mm apart on F.Cu): C2's and J2's
    ports span that gap at their own pads; L1's would lie on C2's: left out."""
    from kicad2ems.importers import open_board
    from shapely.ops import unary_union
    raw = open_board(BUCK)
    gcu = {}
    for lay, net, geom in raw.items:
        if net == 'GND':
            gcu.setdefault(lay, []).append(geom)
    gcu = {k: unary_union(v) for k, v in gcu.items()}
    c2, j2 = (P.pad_gap(raw, r, {'out'}, {'GND'}, gcu) for r in ('C2', 'J2'))
    assert c2['layer'] == 'F.Cu' and c2['axis'] == 1 and (c2['lo'], c2['hi']) == pytest.approx((12.0, 12.4))
    assert c2['c1'] - c2['c0'] == pytest.approx(2.7) and j2['c0'] > c2['c1']
    l1 = P.pad_gap(raw, 'L1', {'out'}, {'GND'}, gcu)
    assert l1 is not None and P._box(l1).intersection(P._box(c2)).area > 0


@need_buck
@pytest.mark.skipif(__import__('kicad2ems_fem.palace.run', fromlist=['launcher']).launcher() is None,
                    reason='Palace not found')
def test_the_buck_rail_in_palace(tmp_path):
    """The buck's output rail: pcbqs's PDN run, then Palace's; each load's |Z| and the bare matrix against pcbqs's
    (the 3D model has the ground pour beside the rail on F.Cu, which pcbqs's plane pair does not: it reads more
    capacitance and less inductance, steadily over the band)."""
    run = tmp_path / 'pdn'
    subprocess.check_call([sys.executable, '-m', 'pcbqs.pdn', BUCK, '--rail', 'out', '--volts', '5', '--load', 'J2=1',
                           '--points', '20', '--ports-touchstone', '--out', str(run)], stdout=subprocess.DEVNULL)
    r = P.solve(str(run), str(tmp_path / 'palace'), fmin=1e7, fmax=3e8, points=4, log=lambda *a: None)
    assert set(r['ports']) == {'C2', 'J2'} and r['left_out'] == ['L1']
    ratio = np.asarray(r['z']['J2']['ratio'])
    assert np.all((ratio > 0.3) & (ratio < 1.0)) and np.ptp(ratio) < 0.15
    bare = np.asarray(r['bare_vs_pcbqs']['J2'])
    assert np.all((bare > 0.5) & (bare < 0.9))
    assert json.loads((tmp_path / 'palace' / 'pdn_palace.json').read_text())['tool'] == 'kicad2ems_fem.palace.pdn'
