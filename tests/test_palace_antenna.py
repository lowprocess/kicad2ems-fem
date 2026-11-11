"""The antenna mode's far-field reading and analysis, without Palace."""
import numpy as np
import pytest

from kicad2ems_fem.palace import antenna, run

HEAD = ('        f (GHz),     exc,               theta (deg.),                 phi (deg.),'
        '              r*Re{E_x} (V),              r*Im{E_x} (V),              r*Re{E_y} (V),'
        '              r*Im{E_y} (V),              r*Re{E_z} (V),              r*Im{E_z} (V)\n')


def dipole_z(theta, phi, amp=1.0):
    """r E of a short dipole along z: E_theta = amp sin(theta), Cartesian (n, 3)."""
    t, p = np.radians(theta), np.radians(phi)
    th = np.stack([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)], -1)
    return amp * np.sin(t)[:, None] * th


def palace_angles(step=2.0):
    """The directions Palace writes for antenna.angles: each pole once, whatever phi was asked."""
    a = antenna.angles(step)
    return a[:, 0], a[:, 1]


def test_each_direction_is_asked_once_and_the_poles_once():
    a = antenna.angles(2.0)
    assert len({tuple(r) for r in a}) == len(a)
    assert (a[:, 0] == 0).sum() == 1 and (a[:, 0] == 180).sum() == 1
    assert a[:, 0].min() == 0 and a[:, 0].max() == 180 and a[:, 1].min() == 0 and a[:, 1].max() < 360


def test_read_farfield_of_palace_csv(tmp_path):
    rows = [(2.4, 1, 0, 0, 1, 2, 3, 4, 5, 6), (2.4, 1, 90, 45, 0, 0, 0, 0, -1, 0.5),
            (2.5, 1, 0, 0, 2, 0, 0, 0, 0, 0), (2.5, 1, 90, 45, 0, 0, 0, 0, 0, 1)]
    (tmp_path / 'farfield-rE.csv').write_text(HEAD + ''.join(','.join(f'{v:+.6e}' for v in r) + '\n' for r in rows))
    f, th, ph, E = run.read_farfield(str(tmp_path))
    assert f.tolist() == [2.4e9, 2.5e9] and th.tolist() == [0, 90] and ph.tolist() == [0, 45]
    assert E.shape == (2, 2, 3) and E[0, 0].tolist() == [1 + 2j, 3 + 4j, 5 + 6j] and E[1, 1, 2] == 1j


def test_a_short_dipole_has_1p76_dbi_and_the_efficiency_of_its_power():
    th, ph = palace_angles()
    f = np.array([2.0e9, 2.4e9])
    E = np.stack([dipole_z(th, ph), dipole_z(th, ph, 2.0)])
    prad = 4 * np.pi / (3 * antenna.ETA0) * np.array([1.0, 4.0])       # of |E_theta| = amp sin(theta)
    s11 = np.array([0.5, 0.1j])
    pinc = prad / 0.8 / (1 - np.abs(s11) ** 2)                         # 80% of the accepted power radiated
    ant, sph = antenna.analyse(f, s11, pinc, th, ph, E, f_pattern=2.4e9)
    for r in ant['rows']:
        assert r['D_dbi'] == pytest.approx(10 * np.log10(1.5), abs=0.02)
        assert r['eff'] == pytest.approx(0.8, rel=0.005)
        assert r['G_dbi'] == pytest.approx(10 * np.log10(1.5 * 0.8), abs=0.03)
    assert ant['rows'][0]['Grealized_dbi'] == pytest.approx(10 * np.log10(1.5 * 0.8 * 0.75), abs=0.03)
    assert sph['prad'] == pytest.approx(prad, rel=0.005) and ant['f_pattern'] == 2.4e9
    xz = ant['cuts']['xz (phi = 0)']
    g = dict(zip(xz['angle'], xz['gain_dbi']))
    assert g[90.0] == pytest.approx(10 * np.log10(1.2), abs=0.03) and g[-90.0] == pytest.approx(g[90.0], abs=1e-6)
    assert g[0.0] < -50 and g[30.0] == pytest.approx(10 * np.log10(1.2 * 0.25), abs=0.03)
    xy = ant['cuts']['xy (theta = 90)']['gain_dbi']
    assert max(xy) - min(xy) < 1e-6
