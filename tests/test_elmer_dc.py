"""Elmer DC IR drop: a strip against its closed form; pads and terminals from board.json."""
import pytest
from shapely.geometry import box, mapping

from kicad2ems_fem.elmer import run as elmer
from kicad2ems_fem.geometry import pads_of


def strip_board(t=0.035):
    """A 50 x 5 mm strip on F.Cu, 5 x 5 mm pads of parts S and L at its ends."""
    pad = lambda ref, x: dict(ref=ref, pads=[dict(name='1', net='N', layers=['F.Cu'], geom=mapping(box(x, 0, x + 5, 5)))])
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=t), dict(name='F.Cu', z=1.6, t=t)],
                           layers=[dict(name='core', z0=0.0, z1=1.6, er=4.5, tand=0.02)]),
                copper={'F.Cu': [dict(net='N', polygon=mapping(box(0, 0, 50, 5)))]}, vias=[],
                prims=dict(footprints=[pad('S', 0), pad('L', 45)]))


def test_pads_of_takes_a_parts_pads_on_the_net():
    b = strip_board()
    (lay, poly), = pads_of(b, 'L', {'N'})
    assert lay == 'F.Cu' and poly.bounds == (45, 0, 50, 5)
    with pytest.raises(ValueError):
        pads_of(b, 'L', {'other'})
    with pytest.raises(ValueError):
        pads_of(b, 'X', {'N'})


@pytest.mark.skipif(not elmer.found()[0], reason='Elmer not installed')
def test_strip_drop_matches_closed_form(tmp_path):
    """R = 40 mm / (sigma t w) between the pads' inner edges; the pads are tagged on the mesh, so their edges are
    resolved to h_fine (0.15% at h_fine 0.06)."""
    from kicad2ems_fem.elmer.dc import solve
    r = solve(strip_board(), {'N'}, 'S', {'L': 1.0}, out=tmp_path, h=0.25, h_fine=0.06, log=lambda *a: None)
    R0 = 0.040 / (5.8e7 * 35e-6 * 0.005)
    assert -r['V']['L'] == pytest.approx(R0, rel=3e-3)
    assert r['loss_W'] == pytest.approx(-r['V']['L'] * 1.0, rel=1e-6)
    assert (tmp_path / 'dc.json').is_file()
