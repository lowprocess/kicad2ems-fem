"""Palace itself on small models, against circuit and closed-form answers. Skipped where Palace is not found
(palace/run.py: K2E_PALACE, or an unpacked image under podman)."""
import json
import os

import numpy as np
import pytest

from kicad2ems_fem.palace import cap, config, model, run
from test_palace import with_part
from test_palace_cap import board as cap_board

pytestmark = pytest.mark.skipif(run.launcher() is None, reason='Palace not found')
quiet = lambda *a: None


def series_arm(b, tmp_path, freq):
    """The series impedance between the two ports (the pi network's arm), ohm, at each of `freq`."""
    b.layers[0]['tand'] = 0.0
    mdl = model.build(b, 2e9, str(tmp_path / 'model.msh'), log=quiet)
    with open(tmp_path / 'palace.json', 'w') as fh:
        json.dump(config.config(mdl, freq, tol=0), fh)
    run.run(str(tmp_path), 'palace.json', log=quiet)
    f, S = run.read_s(str(tmp_path / 'postpro'), 2)
    out = []
    for s in S:
        Y = np.linalg.inv(50 * (np.eye(2) + s) @ np.linalg.inv(np.eye(2) - s))
        out.append(-1 / Y[1, 0])
    return np.array(out)


def test_a_series_part_is_its_circuit_value(tmp_path):
    """0.4 ohm + 8.2 nH across a cut in a 50 ohm microstrip: two sheets in series. The arm also holds the
    10 mm line's own inductance (2.6 nH: the line alone)."""
    w = 2 * np.pi * 0.1e9
    z = series_arm(with_part(R=20.0, L=8.2e-9), tmp_path, [0.1e9, 0.2e9])[0]
    assert z.real == pytest.approx(20.0, rel=0.01)
    assert z.imag / w * 1e9 == pytest.approx(8.2 + 2.6, abs=0.3)


def test_a_series_capacitor(tmp_path):
    w = 2 * np.pi * 0.1e9
    z = series_arm(with_part(C=10e-12), tmp_path, [0.1e9, 0.2e9])[0]
    assert abs(z.real) < 0.5
    assert -1 / (w * (z.imag - w * 2.6e-9)) * 1e12 == pytest.approx(10.0, rel=0.02)


def test_microstrip_capacitance_is_hammerstad_jensen(tmp_path):
    """A 0.3 mm trace 0.2 mm over a plane: C per length 6% over the closed form, in air and on FR-4: the 6 mm
    trace's open ends add 2.5%, the mesh at the edges about as much at this h, the closed box a little."""
    for er, c_m in ((1.0, 32.05e-12), (4.4, 104.6e-12)):          # Hammerstad-Jensen, F/m
        r = cap.solve(cap_board(er=er), ['T', 'G'], str(tmp_path / f'er{er}'), h=0.05, log=quiet)
        C = np.array(r['C'])
        assert r['names'] == ['T', 'G'] and C[0, 0] > 0
        assert np.abs(C.sum(0)).max() < 1e-4 * C[0, 0]            # an open box: the charges balance
        assert -C[0, 1] / 6e-3 / c_m == pytest.approx(1.06, abs=0.03)


def test_a_grounded_box_takes_some_of_the_charge(tmp_path):
    """Open and grounded boxes bracket the capacitance between two conductors far from anything else."""
    C = {far: np.array(cap.solve(cap_board(er=1.0), ['T', 'G'], str(tmp_path / far), h=0.1, far=far, log=quiet)['C'])
         for far in ('open', 'ground')}
    g = C['ground']
    assert g[0, 1] == pytest.approx(g[1, 0], rel=1e-3) and g[0, 0] > -g[0, 1] > 0 and g[1, 1] > -g[0, 1]
    assert 0.85 < g[0, 1] / C['open'][0, 1] < 0.99
