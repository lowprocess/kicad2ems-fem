"""The Palace model, config and output reading, without Palace (tests/test_palace_run.py runs it)."""
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from kicad2ems.board import Port
from kicad2ems_fem.palace import config, model, run


def microstrip(vias=()):
    """A 10 x 6 mm board, 0.5 mm of er 3.5, a 1.1 mm strip on top over a full ground, a port at each end."""
    return SimpleNamespace(
        layers=[dict(name='d', z0=0.0, z1=0.5, er=3.5, tand=0.002, kind='diel')],
        cu_z={'B.Cu': 0.0, 'F.Cu': 0.5}, copper={'B.Cu': [box(0, -3, 10, 3)], 'F.Cu': [box(0, -0.55, 10, 0.55)]},
        vias=list(vias), outline=box(0, -3, 10, 3), parts=[], walls=[],
        ports=[Port('P1', 'F.Cu', 0.5, 0.0, (0.5, 0.0), 0, 1, 1.1, 50.0),
               Port('P2', 'F.Cu', 0.5, 0.0, (9.5, 0.0), 0, -1, 1.1, 50.0)])


def groups(path):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.open(path)
        out = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            n = 0
            for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                n += sum(len(x) for x in gmsh.model.mesh.getElements(dim, e)[1])
            out[(dim, tag)] = n
        return out
    finally:
        gmsh.finalize()


def test_microstrip_mesh_has_every_attribute(tmp_path):
    p = str(tmp_path / 'm.msh')
    mdl = model.build(microstrip(), 10e9, p, log=lambda *a: None)
    g = groups(p)
    assert set(g) == {(3, model.DIEL0), (3, model.AIR), (2, model.CU0), (2, model.CU0 + 1), (2, model.ABSORB),
                      (2, model.PORT0), (2, model.PORT0 + 1)}             # each copper layer its own sheets
    assert mdl.sheets == [(model.CU0, 0.035), (model.CU0 + 1, 0.035)] and mdl.via_wall is None
    assert all(n > 0 for n in g.values())
    assert mdl.hmin == pytest.approx(0.5)                   # the slab's thickness is the smallest feature
    assert [q['direction'] for q in mdl.ports] == ['+Z', '+Z']


def test_vias_are_holes_with_pec_walls(tmp_path):
    a, b = str(tmp_path / 'a.msh'), str(tmp_path / 'b.msh')
    model.build(microstrip(), 10e9, a, log=lambda *x: None)
    model.build(microstrip([(5.0, 2.0, 0.2, 0.0, 0.5)]), 10e9, b, log=lambda *x: None)
    ga, gb = groups(a), groups(b)
    assert (2, model.PEC) in gb and (2, model.PEC) not in ga   # the barrel's wall is PEC
    # the hole is not meshed: no tetrahedron of the slab has its centre inside the barrel
    import gmsh
    gmsh.initialize()
    try:
        gmsh.open(b)
        for e in gmsh.model.getEntitiesForPhysicalGroup(3, model.DIEL0):
            _, tags, nodes = gmsh.model.mesh.getElements(3, e)
            xyz = {}
            nt, coords, _ = gmsh.model.mesh.getNodes()
            xyz = dict(zip(nt, coords.reshape(-1, 3)))
            c = np.array([np.mean([xyz[n] for n in nodes[0][4 * k:4 * k + 4]], 0) for k in range(len(tags[0]))])
            assert (np.hypot(c[:, 0] - 5.0, c[:, 1] - 2.0) > 0.2 - 0.05).all()
    finally:
        gmsh.finalize()


def test_config_excites_every_port_on_the_band():
    mdl = model.Model('w/model.msh', [(1, 3.5, 0.002)], model.AIR, model.PEC, model.ABSORB,
                      [dict(attr=201, index=1, R=50.0, direction='+Z', name='P1'),
                       dict(attr=202, index=2, R=50.0, direction='+Z', name='P2')], 0.1, 1.0, 5.0)
    c = config.config(mdl, np.linspace(1e9, 10e9, 91))
    assert c['Model'] == dict(Mesh='model.msh', L0=1e-3)
    assert [p['Excitation'] for p in c['Boundaries']['LumpedPort']] == [1, 2]
    assert c['Solver']['Driven']['Samples'] == [dict(Type='Linear', MinFreq=1.0, MaxFreq=10.0, NSample=91)]
    assert c['Solver']['Driven']['AdaptiveTol'] == 1e-3
    assert c['Domains']['Materials'][0] == dict(Attributes=[1], Permittivity=3.5, LossTan=0.002)


def test_read_s_of_palace_csv(tmp_path):
    head = ('f (GHz),|S[1][1]| (dB),arg(S[1][1]) (deg.),|S[2][1]| (dB),arg(S[2][1]) (deg.),'
            '|S[1][2]| (dB),arg(S[1][2]) (deg.),|S[2][2]| (dB),arg(S[2][2]) (deg.)')
    rows = ['1.0,-20,90,-0.1,-45,-0.1,-45,-21,10', '2.0,-18,80,-0.2,-90,-0.2,-90,-19,0']
    (tmp_path / 'port-S.csv').write_text(head + '\n' + '\n'.join(rows) + '\n')
    f, S = run.read_s(str(tmp_path), 2)
    assert f.tolist() == [1e9, 2e9]
    assert abs(S[0, 1, 0]) == pytest.approx(10 ** (-0.1 / 20))
    assert np.angle(S[1, 0, 1], deg=True) == pytest.approx(-90)


def test_podman_command_puts_options_before_the_root(tmp_path):
    cmd, cwd = run.command(str(tmp_path), 'palace.json', 4, how=('podman', '/r'))
    assert cmd.index('--rootfs') == len(cmd) - 5 and cmd[-4] == '/r:O' and cmd[-3:-1] == ['bash', '-c']
    assert 'palace -np 4' in cmd[-1]


def test_config_loss_puts_conductivity_on_the_sheets_and_via_walls():
    mdl = model.Model('w/model.msh', [(1, 3.5, 0.002)], model.AIR, [model.CU0, model.PEC], model.ABSORB,
                      [dict(attr=201, index=1, R=50.0, direction='+Z', name='P1')], 0.1, 1.0, 5.0,
                      sheets=[(model.CU0, 0.035)], via_wall=model.PEC)
    lossless = config.config(mdl, [1e9])['Boundaries']
    assert lossless['PEC'] == dict(Attributes=[model.CU0, model.PEC]) and 'Conductivity' not in lossless
    lossy = config.config(mdl, [1e9], loss=True)['Boundaries']
    assert 'PEC' not in lossy
    assert lossy['Conductivity'] == [
        dict(Attributes=[model.CU0], Conductivity=5.8e7, Thickness=0.035, External=False),
        dict(Attributes=[model.PEC], Conductivity=5.8e7, Thickness=0.025, External=True)]
