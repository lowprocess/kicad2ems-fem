"""The Palace capacitance model, config and output reading, without Palace."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from kicad2ems_fem.palace import cap
from test_palace import groups


def board(er=4.4, h=0.2, extra=()):
    """A 0.3 mm trace, 6 mm long, over a 4 x 8 mm plane; a second trace 0.2 mm beside it with `extra`."""
    top = [dict(net='T', polygon=mapping(box(-0.15, -3, 0.15, 3)))] + list(extra)
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=h, t=0.035)],
                           layers=[dict(name='core', z0=0.0, z1=h, er=er, tand=0.0)], thickness=h),
                copper={'F.Cu': top, 'B.Cu': [dict(net='G', polygon=mapping(box(-2, -4, 2, 4)))]},
                vias=[], outline=mapping(box(-2, -4, 2, 4)), prims=dict(footprints=[]))


def test_nets_join_with_a_plus():
    assert cap._groups(['A', 'B+C']) == [('A', {'A'}), ('B+C', {'B', 'C'})]


@pytest.mark.parametrize('thick', [False, True])
def test_mesh_has_a_terminal_per_conductor(tmp_path, thick):
    p = str(tmp_path / 'm.msh')
    m = cap.cap_model(board(), ['T', 'G'], p, thick=thick, h=0.2, log=lambda *a: None)
    g = groups(p)
    assert set(g) == {(3, cap.DIEL0), (3, cap.AIR), (2, cap.OUTER), (2, cap.TERM0), (2, cap.TERM0 + 1)}
    assert all(n > 0 for n in g.values())
    assert m['terminals'] == [('T', cap.TERM0), ('G', cap.TERM0 + 1)] and m['diel'] == [(cap.DIEL0, 4.4)]


def test_default_mesh_size_is_the_smallest_gap():
    b = board(extra=[dict(net='U', polygon=mapping(box(0.25, -3, 0.55, 3)))])
    assert cap.feature_size(b, cap._groups(['T', 'U', 'G'])) == pytest.approx(0.1)
    assert cap.feature_size(b, cap._groups(['T', 'G'])) == pytest.approx(0.2)        # the slab


def test_touching_conductors_are_refused(tmp_path):
    b = board(extra=[dict(net='U', polygon=mapping(box(0.15, -3, 0.45, 3)))])
    with pytest.raises(ValueError, match='touch'):
        cap.cap_model(b, ['T', 'U', 'G'], str(tmp_path / 'm.msh'), h=0.2, log=lambda *a: None)
    cap.cap_model(b, ['T+U', 'G'], str(tmp_path / 'm.msh'), h=0.2, log=lambda *a: None)


def test_a_net_without_copper_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='no copper of X'):
        cap.cap_model(board(), ['T', 'X'], str(tmp_path / 'm.msh'), h=0.2, log=lambda *a: None)


def test_config_is_electrostatic_with_an_open_or_grounded_box():
    m = dict(diel=[(1, 4.4)], air=99, outer=101, terminals=[('T', 201), ('G', 202)])
    c = cap.config(m)
    assert c['Problem']['Type'] == 'Electrostatic'
    assert c['Boundaries'] == dict(Terminal=[dict(Index=1, Attributes=[201]), dict(Index=2, Attributes=[202])],
                                   ZeroCharge=dict(Attributes=[101]))
    assert c['Domains']['Materials'] == [dict(Attributes=[1], Permittivity=4.4), dict(Attributes=[99], Permittivity=1.0)]
    assert cap.config(m, far='ground')['Boundaries']['Ground'] == dict(Attributes=[101])
    with pytest.raises(ValueError):
        cap.config(m, far='pec')


def test_read_c_and_report(tmp_path):
    (tmp_path / 'terminal-C.csv').write_text(
        '            i,             C[i][1] (F),             C[i][2] (F)\n'
        ' 1.0e+00, 2.0e-12, -1.5e-12\n 2.0e+00, -1.5e-12, 3.0e-12\n')
    C = cap.read_c(str(tmp_path), 2)
    assert np.allclose(C, [[2e-12, -1.5e-12], [-1.5e-12, 3e-12]])
    text = cap.report(dict(names=['T', 'G'], C=C.tolist()))
    assert text.splitlines()[1].split() == ['T', '2', '-1.5']
    with pytest.raises(ValueError):
        cap.read_c(str(tmp_path), 3)
