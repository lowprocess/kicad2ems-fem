"""The coil tool: the ring formulas, the mesh and the .sif without Elmer; coils in air and on ferrite against the
formulas with."""
import json

import numpy as np
import pytest

from kicad2ems_fem.elmer import mag, run as elmer

needs_elmer = pytest.mark.skipif(not elmer.found()[0], reason=elmer.found()[1] or 'no Elmer')
quiet = lambda *_: None
TX = dict(name='tx', turns=10, inner=10.25, outer=22, z=0, t=1)
RX = dict(name='rx', turns=14, inner=8, outer=16, z=5, t=0.5)


def test_a_thin_ring_is_the_wire_loop_formula():
    ring = [dict(turns=1, inner=19.9, outer=20.1, z=0, t=0.2)]
    loop = mag.MU0 * 0.02 * (np.log(8 * 0.02 / (0.44705 * 0.2e-3)) - 2)      # a square wire's mean distance
    assert mag.air_core(ring, cells=0.02)[0, 0] == pytest.approx(loop, rel=1e-3)
    assert mag.air_core([dict(ring[0], turns=7)], cells=0.02)[0, 0] == pytest.approx(49 * loop, rel=1e-3)


def test_far_rings_couple_as_dipoles():
    a = dict(turns=3, inner=4.9, outer=5.1, z=0, t=0.2)
    L = mag.air_core([a, dict(a, turns=2, z=200)])
    assert L[0, 1] == L[1, 0] == pytest.approx(6 * mag.MU0 * np.pi * 0.005 ** 4 / (2 * 0.2 ** 3), rel=5e-3)


def test_rings_off_one_axis_meet_the_coaxial_formula():
    on = mag.air_core([TX, RX])
    assert mag.air_core([TX, dict(RX, x=1e-4)])[0, 1] == pytest.approx(on[0, 1], rel=2e-3)
    off = [mag.air_core([TX, dict(RX, x=x)])[0, 1] for x in (5.0, 15.0, 40.0)]
    assert on[0, 1] > off[0] > off[1] > 0 > off[2]               # past the edge the flux returns: negative
    assert mag.air_core([TX, dict(RX, y=-5.0)])[0, 1] == pytest.approx(off[0], rel=1e-9)


def test_what_a_coil_and_a_plate_must_have():
    for bad in (dict(TX, outer=5), dict(TX, turns=0), {k: v for k, v in TX.items() if k != 't'}):
        with pytest.raises(ValueError):
            mag._check([dict(bad)], [])
    with pytest.raises(ValueError):
        mag._check([dict(TX), dict(TX)], [])                     # one name twice
    for bad in (dict(z=-3, t=2, mur=2000), dict(z=-3, t=2, mur=2000, r=30, size=(50, 50)), dict(z=-3, t=2, r=30),
                dict(z=-3, t=2, mur=0.5, r=30)):
        with pytest.raises(ValueError):
            mag._check([dict(TX)], [dict(bad)])


def test_mesh_has_a_body_for_each_part(tmp_path):
    plate = dict(z=-3, t=2.5, size=(50, 50), mur=2000, x=0.0, y=0.0)
    m = mag.mag_model([dict(TX, x=0.0, y=0.0), dict(RX, x=4.0, y=0.0)], [plate], tmp_path / 'm.msh', h=3.0, log=quiet)
    assert m['bodies'] == {1: ('air', 0), 2: ('coil', 0), 3: ('coil', 1), 4: ('plate', 0)}
    assert m['radius'] > 5 * 25 and m['tets'] > 1000
    text = (tmp_path / 'm.msh').read_text()
    assert text.startswith('$MeshFormat\n2.2 0')
    kinds = {int(line.split()[1]) for line in text.split('$Elements\n')[1].split('\n$End')[0].splitlines()[1:]}
    assert kinds == {9, 11}                                      # six-node triangles, ten-node tetrahedra
    with pytest.raises(ValueError):                              # the coil inside the plate
        mag.mag_model([dict(TX, x=0.0, y=0.0)], [dict(plate, z=-1)], tmp_path / 'm.msh', log=quiet)


def test_sif_drives_each_coil_with_its_turns():
    m = dict(bodies={1: ('air', 0), 2: ('coil', 0), 3: ('coil', 1), 4: ('plate', 0)})
    s = mag.sif(m, [TX, RX], [dict(mur=2000)], [1.0, 0.0])
    assert 'Desired Current Density = Real 851063.83' in s       # 10 A over 11.75 mm x 1 mm
    assert s.count('Desired Current Density = Real 0\n') == 1 and s.count('Component ') == 2
    assert 'Master Bodies(1) = 2' in s and 'Master Bodies(1) = 3' in s
    assert s.count('Relative Permeability = 1\n') == 3 and 'Relative Permeability = 2000' in s
    assert 'Normalize Coil Current = Logical True' in s and 'Quadratic Approximation = Logical True' in s
    assert 'Coordinate Scaling = 1.0e-3' in s


def test_items_on_the_command_line():
    import argparse
    ap = argparse.ArgumentParser()
    assert mag._item(ap, 'tx:turns=10,inner=10.25,outer=22,z=0,t=1', '--coil') == dict(TX, turns=10.0)
    assert mag._item(ap, 'z=-3,t=2.5,size=50x40,mur=2000', '--plate') == dict(z=-3.0, t=2.5, size=(50.0, 40.0), mur=2000.0)
    for bad in ('turns=ten', 'size=50', 'size=1x2x3'):
        with pytest.raises(SystemExit):
            mag._item(ap, bad, '--coil')


@needs_elmer
def test_two_coils_in_air_against_the_ring_formulas(tmp_path):
    r = mag.solve([TX, RX], [], tmp_path, h=2.0, log=quiet)
    L, air = np.array(r['L']), np.array(r['air'])
    assert L == pytest.approx(air, rel=0.01)
    assert r['k'][0][1] == pytest.approx(air[0, 1] / np.sqrt(air[0, 0] * air[1, 1]), rel=0.005)
    assert json.loads((tmp_path / 'mag.json').read_text())['names'] == ['tx', 'rx']


@needs_elmer
def test_a_coil_on_a_wide_ferrite_plate_against_its_image(tmp_path):
    gap, t, mur = 0.5, 2.5, 2000
    r = mag.solve([TX], [dict(z=-gap - t, t=t, r=60.0, mur=mur)], tmp_path, h=3.0, log=quiet)
    two = mag.air_core([TX, dict(TX, name='image', z=-2 * gap - TX['t'])])
    assert r['L'][0][0] == pytest.approx(two[0, 0] + (mur - 1) / (mur + 1) * two[0, 1], rel=0.015)
    assert r['air'][0][0] == pytest.approx(two[0, 0])
