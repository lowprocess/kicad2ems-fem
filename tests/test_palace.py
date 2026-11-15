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


def orphans(path):
    """Boundary triangles that are no tetrahedron's face (MFEM aborts on one)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.open(path)
        faces = set()
        for _, e in gmsh.model.getEntities(3):
            for t, _, nodes in zip(*gmsh.model.mesh.getElements(3, e)):
                n = np.asarray(nodes, np.int64).reshape(-1, 4)
                for c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                    faces.update(map(tuple, np.sort(n[:, c], 1)))
        bad = 0
        for _, e in gmsh.model.getEntities(2):
            for t, _, nodes in zip(*gmsh.model.mesh.getElements(2, e)):
                bad += sum(tuple(r) not in faces for r in np.sort(np.asarray(nodes, np.int64).reshape(-1, 3), 1))
        return bad
    finally:
        gmsh.finalize()


def test_a_via_through_two_slabs_leaves_no_face_inside_its_hole(tmp_path):
    b = microstrip([(5.0, 2.0, 0.2, 0.0, 0.5)])
    b.layers = [dict(name='d1', z0=0.0, z1=0.2, er=3.5, tand=0.002, kind='diel'),
                dict(name='d2', z0=0.2, z1=0.5, er=4.4, tand=0.002, kind='diel')]
    b.cu_z['In1.Cu'] = 0.2
    b.copper['In1.Cu'] = [box(4, 1, 6, 3)]                  # a pad the barrel passes through
    p = str(tmp_path / 'm.msh')
    model.build(b, 10e9, p, log=lambda *x: None)
    assert (2, model.PEC) in groups(p) and orphans(p) == 0


def fenced():
    """A 20 x 16 mm board of named nets: a strip between two ground pours over a ground plane, a ground via in
    each pour's corners, an edge wall along each long side."""
    b = microstrip()
    b.layers = [dict(name='d', z0=0.0, z1=0.2, er=3.5, tand=0.002, kind='diel')]
    b.cu_z = {'B.Cu': 0.0, 'F.Cu': 0.2}
    b.outline = box(0, -8, 20, 8)
    b.copper = {'B.Cu': [box(0, -8, 20, 8)], 'F.Cu': [box(0, -0.2, 20, 0.2), box(0, 1.5, 20, 8), box(0, -8, 20, -1.5)]}
    b.copper_nets = {'B.Cu': ['GND'], 'F.Cu': ['SIG', 'GND', 'GND']}
    b.ground_nets = {'GND'}
    b.vias = [(x, y, 0.2, 0.0, 0.2) for x in (5.0, 15.0) for y in (-5.0, 5.0)]
    b.via_nets = ['GND'] * 4
    b.walls = [('y', 8.0, 0.0, 20.0, 0.0, 0.2), ('y', -8.0, 0.0, 20.0, 0.0, 0.2)]
    b.ports = [Port('P1', 'F.Cu', 0.2, 0.0, (0.5, 0.0), 0, 1, 0.4, 50.0),
               Port('P2', 'F.Cu', 0.2, 0.0, (19.5, 0.0), 0, -1, 0.4, 50.0)]
    return b


def test_ground_planes_are_wide_copper_on_a_ground_net():
    b = fenced()
    b.copper['F.Cu'].append(box(2, 0.8, 18, 1.0))           # a ground trace, not a plane
    b.copper_nets['F.Cu'].append('GND')
    assert model.ground_planes(b, 'F.Cu', 1.0) == [False, True, True, False]
    assert model.ground_planes(b, 'B.Cu', 1.0) == [True]
    assert model.ground_planes(microstrip(), 'B.Cu', 1.0) == [False]     # no nets named: nothing is taken for a plane


def test_a_wall_ends_on_the_wall_it_crosses():
    b = fenced()
    b.walls = [('x', 0.5005, -8.0, 8.0, 0.0, 0.2), ('y', -7.9995, 0.5, 19.5, 0.0, 0.2), ('y', 7.9, 0.5, 19.5, 0.0, 0.2)]
    assert model.wall_ends(b, *b.walls[0]) == (-7.9995, 8.0)            # half a micron past one, 0.1 mm past the other
    assert model.wall_ends(b, *b.walls[1]) == (0.5005, 19.5)
    b.walls[0] = ('x', 0.5005, -8.0, 7.0, 0.0, 0.2)                      # no longer reaching the third wall,
    b.copper['F.Cu'][1] = box(0.502, 1.5, 20, 7.9)                       # which ends on the corner of the copper it joins
    assert model.wall_ends(b, *b.walls[2]) == (0.502, 19.5)


def test_planes_take_the_mesh_size_reached_at_them(tmp_path):
    a, c = str(tmp_path / 'a.msh'), str(tmp_path / 'c.msh')
    fine = model.build(fenced(), 5e9, a, log=lambda *x: None, planes=False)
    mdl = model.build(fenced(), 5e9, c, log=lambda *x: None)
    assert mdl.tets < 0.5 * fine.tets
    assert set(groups(c)) == set(groups(a)) and (2, model.WALL) in groups(c) and (2, model.PEC) in groups(c)
    assert all(n > 0 for n in groups(c).values()) and orphans(c) == 0
    assert (mdl.hmin, mdl.hmax) == (fine.hmin, fine.hmax)


def test_the_coarse_solve_is_complex():
    mdl = model.Model('w/model.msh', [(1, 3.5, 0.002)], model.AIR, model.PEC, model.ABSORB,
                      [dict(attr=201, index=1, R=50.0, direction='+Z', name='P1')], 0.1, 1.0, 5.0)
    assert config.config(mdl, np.linspace(1e9, 2e9, 3))['Solver']['Linear']['ComplexCoarseSolve'] is True


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


# ---- lumped parts and edge walls ----

def with_part(R=None, L=None, C=None, gap=0.4):
    """The microstrip cut at its middle, a part across the cut."""
    from kicad2ems.parts import Part
    b = microstrip()
    g0, g1 = 5 - gap / 2, 5 + gap / 2
    b.copper['F.Cu'] = [box(0, -0.55, g0, 0.55), box(g1, -0.55, 10, 0.55)]
    b.parts = [Part('X1', R, L, C, 'F.Cu', 0.5, 0, (g0, -0.55, 0.5), (g1, 0.55, 0.5))]
    return b


def test_a_part_is_one_sheet_per_series_element():
    from kicad2ems.parts import Part
    one = model.part_elements(Part('R1', 50.0, None, None, 'F.Cu', 0.5, 0, (1, 0, 0.5), (1.4, 1, 0.5)))
    assert [(rlc, s[0], e[0]) for rlc, s, e in one] == [({'R': 50.0}, 1.0, 1.4)]
    esr = model.part_elements(Part('C1', 0.1, 5e-10, 6.8e-12, 'F.Cu', 0.5, 1, (0, 1, 0.5), (1, 1.6, 0.5)))
    assert [next(iter(rlc)) for rlc, _, _ in esr] == ['R', 'L', 'C']
    assert [round(float(s[1]), 6) for _, s, _ in esr] == [1.0, 1.2, 1.4] and esr[-1][2][1] == pytest.approx(1.6)
    assert all(s[0] == 0 and e[0] == 1 for _, s, e in esr)                      # the whole width, each
    ideal = model.part_elements(Part('C2', 0.0, None, 1e-12, 'F.Cu', 0.5, 0, (1, 0, 0.5), (1.4, 1, 0.5)))
    assert [rlc for rlc, _, _ in ideal] == [{'C': 1e-12}]                       # a zero ESR is no sheet
    assert model.part_elements(Part('J1', 0.0, None, None, 'F.Cu', 0.5, 0, (1, 0, 0.5), (1.4, 1, 0.5))) == []


def test_part_terminals_reach_onto_the_pads():
    lands = model.part_lands(with_part(R=50.0))['F.Cu']
    assert [tuple(round(v, 3) for v in g.bounds) for g in lands] == [(4.6, -0.55, 4.8, 0.55), (5.2, -0.55, 5.4, 0.55)]


def test_a_wall_in_from_the_edge_is_kept(tmp_path):
    """Edge walls stand a keepout in from the outline, so inside the slab and the boundary of no volume."""
    b = with_part(R=50.0)
    b.walls = [('y', 2.5, 0.0, 10.0, 0.0, 0.5), ('y', -2.5, 0.0, 10.0, 0.0, 0.5)]
    p = str(tmp_path / 'm.msh')
    mdl = model.build(b, 10e9, p, log=lambda *a: None)
    assert mdl.edge_wall == model.WALL and groups(p)[(2, model.WALL)] > 0


def test_part_sheets_and_walls_have_their_own_attributes(tmp_path):
    b = with_part(R=0.4, L=8.2e-9)
    b.walls = [('x', 0.0, -3.0, 3.0, 0.0, 0.5)]
    p = str(tmp_path / 'm.msh')
    mdl = model.build(b, 10e9, p, log=lambda *a: None)
    g = groups(p)
    assert {(2, model.PART0), (2, model.PART0 + 1), (2, model.WALL)} <= set(g) and all(n > 0 for n in g.values())
    assert mdl.edge_wall == model.WALL and model.WALL in mdl.pec
    assert [(q['attr'], q['index'], q['direction'], q['rlc'], q['ref']) for q in mdl.parts] == [
        (model.PART0, 3, '+X', {'R': 0.4}, 'X1'), (model.PART0 + 1, 4, '+X', {'L': 8.2e-9}, 'X1')]
    assert mdl.hmin == pytest.approx(0.4)                   # the cut; the sheets are refined around themselves


def test_a_zero_ohm_part_is_copper(tmp_path):
    p = str(tmp_path / 'm.msh')
    mdl = model.build(with_part(R=0.0), 10e9, p, log=lambda *a: None)
    assert mdl.parts == [] and (2, model.PART0) not in groups(p)


def part_model():
    return model.Model('w/model.msh', [(1, 3.5, 0.002)], model.AIR, [model.CU0, model.WALL], model.ABSORB,
                       [dict(attr=201, index=1, R=50.0, direction='+Z', name='P1'),
                        dict(attr=202, index=2, R=50.0, direction='+Z', name='P2')], 0.1, 1.0, 5.0,
                       sheets=[(model.CU0, 0.035)], edge_wall=model.WALL,
                       parts=[dict(attr=301, index=3, direction='+X', rlc={'R': 0.4}, ref='L1'),
                              dict(attr=302, index=4, direction='+X', rlc={'L': 8.2e-9}, ref='L1')])


def test_config_parts_are_passive_boundaries():
    c = config.config(part_model(), np.linspace(1e9, 2e9, 11))
    lp = c['Boundaries']['LumpedPort']
    assert lp[2] == dict(Index=3, Attributes=[301], Direction='+X', R=0.4)
    assert lp[3] == dict(Index=4, Attributes=[302], Direction='+X', L=8.2e-9)
    assert [p.get('Excitation') for p in lp] == [1, 2, None, None]
    lossy = config.config(part_model(), [1e9], loss=True)['Boundaries']
    assert lossy['PEC'] == dict(Attributes=[model.WALL])    # the edge walls stay PEC beside lossy copper


def test_config_far_field_of_one_port_at_chosen_frequencies():
    ang = np.array([[0.0, 0.0], [90.0, 45.0]])
    c = config.config(part_model(), [1e9, 1.5e9, 3e9], tol=0, excite=[1], farfield=ang, out='postpro_ff')
    assert [p.get('Excitation') for p in c['Boundaries']['LumpedPort']] == [None, 2, None, None]
    assert c['Boundaries']['Postprocessing'] == dict(FarField=dict(Attributes=[model.ABSORB],
                                                                   ThetaPhis=[[0.0, 0.0], [90.0, 45.0]]))
    assert c['Solver']['Driven']['Samples'] == [dict(Type='Point', Freq=[1.0, 1.5, 3.0])]
    assert 'AdaptiveTol' not in c['Solver']['Driven'] and c['Problem']['Output'] == 'postpro_ff'


def test_read_s_leaves_out_the_parts_boundaries(tmp_path):
    head = ','.join(f'|S[{i}][1]| (dB),arg(S[{i}][1]) (deg.)' for i in (1, 2, 3))
    (tmp_path / 'port-S.csv').write_text(f'f (GHz),{head}\n1.0,-20,90,-0.1,-45,-30,0\n')
    f, S = run.read_s(str(tmp_path), 2)
    assert S.shape == (1, 2, 2) and abs(S[0, 1, 0]) == pytest.approx(10 ** (-0.1 / 20)) and np.isnan(S[0, 0, 1])


def test_read_incident_power(tmp_path):
    for q, unit, v in (('V', 'V', '8.0,0.5,-0.25'), ('I', 'A', '0.25,0.01,0.005')):
        (tmp_path / f'port-{q}.csv').write_text(
            f'        f (GHz),               {q}_inc[1] ({unit}),               Re{{{q}[1]}} ({unit}),'
            f'               Im{{{q}[1]}} ({unit})\n2.4,{v}\n')
    f, p = run.read_incident(str(tmp_path), 1)
    assert f.tolist() == [2.4e9] and p[0] == pytest.approx(1.0)


def test_config_asks_for_mesh_refinement():
    """--palace-amr N (R-E3): Palace's Model.Refinement, N passes to the error estimate's tolerance; none by
    default."""
    mdl = model.Model('w/model.msh', [(1, 3.5, 0.002)], model.AIR, model.PEC, model.ABSORB,
                      [dict(attr=201, index=1, R=50.0, direction='+Z', name='P1')], 0.1, 1.0, 5.0)
    assert 'Refinement' not in config.config(mdl, np.linspace(1e9, 2e9, 3))['Model']
    r = config.config(mdl, np.linspace(1e9, 2e9, 3), amr=2, amr_tol=5e-3)['Model']['Refinement']
    assert r == dict(MaxIts=2, Tol=5e-3, SaveAdaptIterations=False)


def test_the_feature_size_is_every_pieces_nearest_gap_and_the_copper_is_straightened():
    """R-B21: a layer of 100 pieces whose narrowest gap is between the last two (the first 60 had been sampled), and
    an arc drawn in 256 segments straightened to the tolerance, its area kept."""
    from shapely.geometry import Point
    b = microstrip()
    pads = [box(k * 2.0, 2.0, k * 2.0 + 1.0, 2.5) for k in range(99)] + [box(197.05, 2.0, 198.0, 2.5)]
    b.copper = dict(b.copper, **{'F.Cu': b.copper['F.Cu'] + pads})
    assert model.feature_size(b) == pytest.approx(0.05)
    disc = Point(0, 0).buffer(1.0, quad_segs=64)
    (flat,) = model.simplified([disc], 0.005)
    assert len(flat.exterior.coords) < len(disc.exterior.coords) / 2 and flat.area == pytest.approx(disc.area, rel=1e-2)


def test_the_estimate_is_its_mesh_scaled_by_the_runs():
    """R-B21: Palace's estimate builds the mesh as the run would and scales its tetrahedra by the examples' runs."""
    from kicad2ems_fem.palace.plugin import Palace
    pl = Palace()
    s = SimpleNamespace(palace_margin=None, palace_feature=4.0, palace_wave=8.0, palace_fine_ground=False, palace_order=2,
                        palace_np=8)
    run_ = SimpleNamespace(m=microstrip(), s=s, freq=np.array([1e9, 10e9]), fmin=1e9, fmax=10e9, antenna=False,
                           plan=['openems'])
    assert pl.estimate(run_) is None                                   # not in the run: no mesh built for it
    run_.plan = ['palace']
    e = pl.estimate(run_)
    tets = int(e['what'].split()[0].replace(',', ''))
    assert tets > 100 and e['gb'] == pytest.approx(pl.GB0 + pl.GB_PER_KUNKNOWN * pl.UNKNOWNS_PER_TET * tets / 1e3)
    assert e['seconds'] == pytest.approx(pl.T0 * tets ** 1.5) and 'order 2' in e['what']


def _two_ports():
    return model.Model('w/model.msh', [(1, 3.5, 0.002)], model.AIR, [model.PEC], model.ABSORB,
                       [dict(attr=201, index=1, R=50.0, direction='+Z', name='P1'),
                        dict(attr=202, index=2, R=50.0, direction='+Z', name='P2')], 0.1, 1.0, 5.0)


def test_a_pec_box_is_a_can():
    """--palace-box pec: the air box's faces join the metal; no absorbing boundary is left, and a far field
    (which needs one) is refused."""
    mdl = _two_ports()
    c = config.config(mdl, np.linspace(1e9, 2e9, 3), box='pec')
    assert 'Absorbing' not in c['Boundaries']
    assert c['Boundaries']['PEC']['Attributes'] == [model.PEC, model.ABSORB]
    assert config.config(mdl, np.linspace(1e9, 2e9, 3))['Boundaries']['Absorbing'] == dict(
        Attributes=[model.ABSORB], Order=1)
    with pytest.raises(ValueError):
        config.config(mdl, [1e9], box='pec', farfield=[[0, 0]])
    with pytest.raises(ValueError):
        config.config(mdl, [1e9], box='open')


def test_eigen_config_loads_every_port_and_excites_none():
    """R-E19: the eigenmode run's ports are their resistors (Q loaded, the ports' share in port-Q.csv)."""
    mdl = _two_ports()
    c = config.eigen(mdl, 3, 4.3e9)
    assert c['Problem']['Type'] == 'Eigenmode' and c['Problem']['Output'] == 'postpro_eig'
    assert c['Solver']['Eigenmode'] == dict(N=3, Tol=1e-6, Target=4.3, Save=0)
    assert [p['R'] for p in c['Boundaries']['LumpedPort']] == [50.0, 50.0]
    assert not any('Excitation' in p for p in c['Boundaries']['LumpedPort'])
    assert model.ABSORB in c['Boundaries']['PEC']['Attributes']
    assert config.eigen(mdl, 1, 1e9, box='absorbing')['Boundaries']['Absorbing']['Attributes'] == [model.ABSORB]


def test_read_modes_of_palace_csv(tmp_path):
    (tmp_path / 'eig.csv').write_text(
        '            m,          Re{f} (GHz),          Im{f} (GHz),                    Q,'
        '        Error (Bkwd.),         Error (Abs.)\n'
        ' 2.000000e+00, +5.100000000000e+00, +1.000000000000e-02, +2.550000000000e+02, +1.0e-10, +1.0e-09\n'
        ' 1.000000e+00, +4.620000000000e+00, +1.500000000000e-02, +1.540000000000e+02, +2.0e-10, +3.0e-09\n')
    (tmp_path / 'port-Q.csv').write_text(
        '            m,            Q_ext[1],    κ_ext[1] (GHz),            Q_ext[2],    κ_ext[2] (GHz)\n'
        ' 1.000000e+00, +3.1e+02, +1.5e-02, +3.2e+02, +1.4e-02\n'
        ' 2.000000e+00, +5.0e+02, +1.0e-02, +6.0e+02, +8.0e-03\n', encoding='utf-8')     # matched by m, not row
    m = run.read_modes(str(tmp_path))
    assert [round(x['f_Hz'] / 1e9, 3) for x in m] == [4.62, 5.1]
    assert m[0]['Q'] == pytest.approx(154) and m[0]['q_ext'] == {1: 310.0, 2: 320.0}
    assert m[1]['f_im_Hz'] == pytest.approx(1e7)



def test_refinement_has_a_cap_and_reaches_the_eigenmode_run():
    """R-E20: the refinement stops at amr_max unknowns, in the driven run and in the eigenmode run alike."""
    mdl = _two_ports()
    r = config.config(mdl, np.linspace(1e9, 2e9, 3), amr=3, amr_max=250000)['Model']['Refinement']
    assert r == dict(MaxIts=3, Tol=1e-2, SaveAdaptIterations=False, MaxSize=250000)
    assert config.eigen(mdl, 2, 4e9, amr=2, amr_tol=1e-3)['Model']['Refinement'] == dict(
        MaxIts=2, Tol=1e-3, SaveAdaptIterations=False)
    assert 'Refinement' not in config.eigen(mdl, 2, 4e9)['Model']


def test_the_absorbing_boundary_can_be_second_order():
    mdl = _two_ports()
    assert config.config(mdl, [1e9], abc=2)['Boundaries']['Absorbing']['Order'] == 2
    with pytest.raises(ValueError):
        config.config(mdl, [1e9], abc=3)


def test_the_refinement_cap_follows_the_memory_available(monkeypatch):
    from kicad2ems_fem.palace.plugin import Palace
    import emsdev.memory
    monkeypatch.setattr(emsdev.memory, 'available', lambda: 31e9)
    logs = []
    r = SimpleNamespace(s=SimpleNamespace(palace_amr=2, palace_amr_tol=1e-2, palace_amr_max=0), log=logs.append)
    got = Palace()._amr(r)
    assert got['amr'] == 2 and got['amr_max'] == int((0.7 * 31 - Palace.GB0) / Palace.GB_PER_KUNKNOWN * 1e3)
    assert 'unknowns' in logs[0]
    r.s.palace_amr_max = 123456
    assert Palace()._amr(r)['amr_max'] == 123456
    r.s.palace_amr = 0
    assert Palace()._amr(r) == dict(amr=0)
