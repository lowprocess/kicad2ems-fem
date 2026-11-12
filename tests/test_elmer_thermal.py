"""The thermal tool: the layered mesh and the .sif without Elmer; a fin and a warm strip against their equations
with."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from kicad2ems_fem.elmer import layered, run as elmer, thermal

needs_elmer = pytest.mark.skipif(not elmer.found()[0], reason=elmer.found()[1] or 'no Elmer')


def strip(length=20.0, width=5.0, tb=1.6, tc=0.035, parts=(('S', 0.0, 1.0),), via=None):
    """A strip of top copper the size of its board; parts: (ref, x, pad length) pads across the strip."""
    pad = lambda ref, x, a: dict(ref=ref, pads=[dict(name='1', net='N', layers=['F.Cu'],
                                                     geom=mapping(box(x, 0, x + a, width)))])
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=tc), dict(name='F.Cu', z=tb, t=tc)],
                           layers=[dict(name='core', z0=0.0, z1=tb, er=4.5, tand=0.02)]),
                copper={'F.Cu': [dict(net='N', polygon=mapping(box(0, 0, length, width)))]},
                vias=[via] if via else [], outline=mapping(box(0, 0, length, width)),
                prims=dict(footprints=[pad(*p) for p in parts]))


def test_z_cells_are_the_copper_layers_and_slices_of_the_board():
    cells = layered.z_cells(strip(), dz=0.5)
    assert [c[2] for c in cells] == ['B.Cu', None, None, None, None, 'F.Cu']
    assert cells[0][:2] == pytest.approx((-0.035, 0.0)) and cells[-1][:2] == pytest.approx((1.6, 1.635))
    assert all(b - a <= 0.5 + 1e-9 for a, b, _ in cells)
    assert [a for a, _, _ in cells[1:]] == pytest.approx([b for _, b, _ in cells[:-1]])


def test_mesh_volumes_and_faces(tmp_path):
    b = strip(parts=(('S', 0.0, 1.0), ('L', 18.0, 2.0)))
    m = layered.layered_mesh(b, tmp_path / 'mesh', electric={'N'}, h=1.0, hmax=2.0, log=lambda *_: None,
                             terminals={'P:S': thermal.part_pads(b, 'S'), 'I:L': thermal.part_pads(b, 'L')})
    assert m['bodies'] == [layered.COPPER_E, layered.BOARD]      # nothing where an outer layer has no copper
    assert m['volume'][layered.COPPER_E] == pytest.approx(20 * 5 * 0.035, rel=1e-9)
    assert m['volume'][layered.BOARD] == pytest.approx(20 * 5 * 1.6, rel=1e-9)
    area = {names: a for names, a in m['boundaries'].values()}
    assert area[('P:S',)] == pytest.approx(5.0, rel=1e-9) and area[('I:L',)] == pytest.approx(10.0, rel=1e-9)
    assert sum(area.values()) == pytest.approx(2 * 100 + 50 * 1.635, rel=1e-9)   # every outer face is exposed
    header = (tmp_path / 'mesh' / 'mesh.header').read_text().split()
    assert int(header[1]) == m['wedges'] and len(m['nodes']) == int(header[0])


def test_a_via_is_a_barrel_through_the_board(tmp_path):
    b = strip(via=dict(x=10.0, y=2.5, r=0.3, net='N', z0=0.0, z1=1.6))
    m = layered.layered_mesh(b, tmp_path / 'mesh', h=1.0, hmax=2.0, log=lambda *_: None)
    assert m['vias'] == 1 and layered.BARREL in m['bodies']
    assert m['volume'][layered.BARREL] == pytest.approx(np.pi * 0.3 ** 2 * 1.6, rel=0.02)


def mesh_of(names):
    return dict(bodies=[layered.COPPER_E, layered.COPPER, layered.BOARD], via_r=np.array([0.15]),
                boundaries={1: ((), 100.0), **{10 + i: (n, 4.0) for i, n in enumerate(names)}})


def test_sif_with_heat_alone_has_no_current_solver():
    s = thermal.sif(mesh_of([('P:Q1',)]), heat={'Q1': 0.5}, htc=12, ambient=40)
    assert 'StatCurrent' not in s and 'Solver 1\n  Equation = "Heat Equation"' in s and 'Solver 3' not in s
    assert s.count('Heat Transfer Coefficient = 12') == 2 and s.count('External Temperature = 40') == 2
    assert 'Heat Flux = 125000' in s                             # 0.5 W over 4 mm2
    assert 'Heat Conductivity(3) = 0.8 0.8 0.3' in s and 'Body Force' not in s


def test_sif_with_current_heats_the_copper_and_follows_its_temperature():
    s = thermal.sif(mesh_of([('I:source',), ('I:Q1', 'P:Q1')]), heat={'Q1': 0.5}, loads={'Q1': 3.0})
    assert 'Solver 1\n  Equation = "StatCurrent"' in s and 'Solver 2\n  Equation = "Heat Equation"' in s
    assert 'Temperature Load = Equals "Nodal Joule Heating"' in s and 'Constant Weights = Logical True' in s
    assert 'Electric Conductivity = Variable Temperature' in s and '20 5.8e+07' in s
    assert 'Potential = 0.0' in s and 'Potential Constant Flux = Real -3' in s
    assert s.count('Body Force = 1') == 1                        # the copper with current only
    with pytest.raises(ValueError):                              # a load on pads of which only some are heated
        thermal.sif(mesh_of([('I:source',), ('I:Q1', 'P:Q1.1'), ('I:Q1',)]), heat={'Q1.1': 0.5}, loads={'Q1': 3.0})


def fin(length, width, tb, tc, watts, pad, htc, n=8001):
    """The fin equation -kA T'' + hP T = q in 1D, both ends convecting: the rise along the strip, K."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    x = np.linspace(0, length * 1e-3, n)
    dx = x[1] - x[0]
    tall = (tb + tc) * 1e-3
    kA = (thermal.K_CU * tc + thermal.K_BOARD[0] * tb) * width * 1e-6
    hP = htc * 2 * (width * 1e-3 + tall)
    A = sp.diags([-kA / dx ** 2 * np.ones(n - 1), (2 * kA / dx ** 2 + hP) * np.ones(n), -kA / dx ** 2 * np.ones(n - 1)],
                 [-1, 0, 1]).tolil()
    q = np.where(x <= pad * 1e-3, watts / (pad * 1e-3), 0.0)
    for i in (0, n - 1):                                         # half a cell, its end face convecting
        A[i, i] = kA / dx ** 2 + hP / 2 + htc * width * 1e-3 * tall / dx
        q[i] /= 2
    return x * 1e3, spl.spsolve(A.tocsr(), q)


@needs_elmer
def test_a_heated_strip_against_the_fin_equation(tmp_path):
    L, w, tb, tc, Q, htc = 100.0, 5.0, 0.2, 0.035, 0.2, 10.0     # a thin board: one temperature through it
    r = thermal.solve(strip(L, w, tb, tc), tmp_path, heat={'S': Q}, htc=htc, h=0.5, hmax=1.0, log=lambda *_: None)
    assert abs(r['balance']['error']) < 1e-6 and r['balance']['in_W'] == pytest.approx(Q)
    d = np.load(tmp_path / 'thermal.npz')
    x1, rise = fin(L, w, tb, tc, Q, 1.0, htc)
    top = np.isclose(d['nodes'][:, 2], d['nodes'][:, 2].max())
    for x in (10.0, 40.0, 80.0):
        near = top & (abs(d['nodes'][:, 0] - x) < 0.6)
        assert d['T'][near].mean() - 25.0 == pytest.approx(np.interp(x, x1, rise), rel=0.02)
    assert r['parts']['S'] - 25.0 == pytest.approx(np.interp(0.5, x1, rise), rel=0.02)
    assert (tmp_path / 'thermal.png').stat().st_size > 10_000


@needs_elmer
def test_a_strip_warmed_by_its_own_current(tmp_path):
    b = strip(50.0, 5.0, parts=(('S', 0.0, 5.0), ('L', 45.0, 5.0)))
    r = thermal.solve(b, tmp_path, nets={'N'}, source='S', loads={'L': 10.0}, log=lambda *_: None)
    assert abs(r['balance']['error']) < 1e-6
    assert r['loss_W'] == pytest.approx(10.0 * r['drop_mV']['L'] * 1e-3, rel=1e-6)
    cold = 0.040 / (thermal.SIGMA_CU * 5e-3 * 35e-6) * 10.0 * 1e3 * (1 + thermal.ALPHA_CU * 5.0)   # 40 mm between the pads
    assert r['drop_ambient_mV']['L'] == pytest.approx(cold, rel=0.03)
    warm = lambda t: (1 + thermal.ALPHA_CU * (t - 20.0)) / (1 + thermal.ALPHA_CU * 5.0)
    assert warm(r['T_min']) < r['drop_mV']['L'] / r['drop_ambient_mV']['L'] < warm(r['T_max'])
    rise = r['loss_W'] / (r['htc'] * (2 * 250 + 110 * 1.635) * 1e-6)                # the mean over the faces
    assert r['T_min'] - 25.0 < rise < r['T_max'] - 25.0


def test_heat_from_a_circuit_run_keeps_the_parts_on_the_board(tmp_path):
    import json
    b = strip(parts=(('Q1', 0.0, 1.0), ('C2', 10.0, 1.0)))
    b['prims']['footprints'].append(dict(ref='H1', pads=[]))     # a mounting hole: no copper to heat
    (tmp_path / 'case.json').write_text(json.dumps(dict(load_power_W=10.0, component_power_W={
        'Q1': 0.09, 'C2': -1e-4, 'H1': 0.5, 'PCB0': 0.002, 'PRIMARY': 10.4})))
    heat, left = thermal.circuit_heat(b, tmp_path / 'case.json', duty=0.5)
    assert heat == {'Q1': pytest.approx(0.045)} and set(left) == {'C2', 'H1', 'PCB0', 'PRIMARY'}
    (tmp_path / 'flat.json').write_text(json.dumps({'Q1': 0.2, 'C2': 0.1}))
    assert thermal.circuit_heat(b, tmp_path / 'flat.json')[0] == {'Q1': 0.2, 'C2': 0.1}
    # pcbqs.circuit's case.json: each part a table with its power (the source's negative, a load off the board)
    (tmp_path / 'general.json').write_text(json.dumps(dict(name='run', period_s=1e-5, parts={
        'V1': dict(type='source', power_W=-10.7, avg_V=12.0), 'Q1': dict(type='switch', power_W=0.058, rms_A=1.3),
        'C2': dict(type='capacitor', power_W=0.002), 'R1': dict(type='resistor', power_W=10.0), 'K1': dict(type='coupling')})))
    heat, left = thermal.circuit_heat(b, tmp_path / 'general.json')
    assert heat == {'Q1': 0.058, 'C2': 0.002} and left == {'V1': -10.7, 'R1': 10.0}
    (tmp_path / 'other.json').write_text(json.dumps(dict(parts=['Q1'])))
    with pytest.raises(ValueError, match="a circuit run's case.json"):
        thermal.circuit_heat(b, tmp_path / 'other.json')
    with pytest.raises(ValueError):
        thermal.circuit_heat(b, tmp_path / 'flat.json', duty=0.0)


def test_duty_on_the_command_line_is_the_mean(capsys):
    import argparse
    ap = argparse.ArgumentParser()
    assert thermal._pairs(ap, ['Q1=2@0.25', 'Q2.5=0.3'], '--heat', 'WATTS') == {'Q1': 0.5, 'Q2.5': 0.3}
    with pytest.raises(SystemExit):
        thermal._pairs(ap, ['Q1=2@1.5'], '--heat', 'WATTS')
