"""The DC, thermal and capacitance tools as kicad2ems analysis plugins (analyses.py): engines of a study's questions,
their settings read from each tool's parser, their commands, their reports, whose result a folder is; one of them
run for real through its command where Elmer is installed."""
import json
import os
import subprocess
import sys

import pytest
from shapely.geometry import box, mapping

from kicad2ems import analyses as registry_module, study
from kicad2ems_fem import analyses as A
from kicad2ems_fem.elmer import run as elmer

ALL = (A.ElmerDC, A.ElmerThermal, A.ElmerCap, A.PalaceCap)


class _EP:
    def __init__(self, cls):
        self.name, self.value, self.cls = cls.name, f"kicad2ems_fem.analyses:{cls.__name__}", cls

    def load(self):
        return self.cls


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(registry_module, '_entry_points', lambda: [_EP(c) for c in ALL])
    monkeypatch.setattr(registry_module, '_registry', None)
    monkeypatch.setattr(registry_module, 'broken', [])
    yield registry_module.registry()
    monkeypatch.setattr(registry_module, '_registry', None)


def test_they_register_as_engines_of_questions_others_answer_too(registered):
    assert list(registered) == ['elmer_dc', 'elmer_thermal', 'elmer_cap', 'palace_cap'] and not registry_module.broken
    es = study.engines()
    assert [es[n].provides for n in registered] == [('dc',), ('thermal',), ('capacitance',), ('capacitance',)]
    assert [es[n].label for n in registered] == ['Elmer', 'Elmer', 'Elmer', 'Palace']
    rows = {r['question'].name: r for g in study.study_tree() for r in g['questions']}
    assert [e.name for e in rows['capacitance']['engines']] == ['elmer_cap', 'palace_cap']
    import tomllib
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pyproject.toml'), 'rb') as fh:
        declared = tomllib.load(fh)['project']['entry-points']['kicad2ems.analyses']
    assert declared == {c.name: f"kicad2ems_fem.analyses:{c.__name__}" for c in ALL}


def test_the_settings_are_the_tools_own_options():
    dc = {o['key']: o for o in A.ElmerDC().options()}
    assert list(dc) == ['net', 'source', 'load', 'h', 'h_fine', 'plating'] and 'out' not in dc
    assert dc['net']['kind'] == 'list' and dc['net']['required'] and dc['h']['default'] == 0.25 and dc['h']['kind'] == 'float'
    assert dc['source']['group'] == 'Nets and loads' and dc['plating']['group'] == 'Mesh'
    th = {o['key']: o for o in A.ElmerThermal().options()}
    assert {'heat', 'heat_from', 'htc', 'ambient', 'k_board', 'region'} <= set(th) and th['htc']['default'] == 10.0
    assert [o['group'] for o in A.ElmerThermal().options()] == sorted((o['group'] for o in A.ElmerThermal().options()),
                                                                    key=A.ORDER.index)
    cap = {o['key']: o for o in A.PalaceCap().options()}
    assert cap['thick']['kind'] == 'bool' and cap['far']['choices'] == ['open', 'ground'] and cap['np']['kind'] == 'int'


def test_the_commands(tmp_path):
    board = str(tmp_path / 'b.kicad_pcb')
    open(board, 'w').close()
    dc = A.ElmerDC()
    line = dc.command(dict(net='VHB', source='L3', load='Q1=0.67, Q2=0.67', h=0.25, h_fine='0.1'), board, 'out')
    assert line == [sys.executable, '-m', 'kicad2ems_fem.elmer.dc', board, '--out', 'out', '--net', 'VHB', '--source',
                    'L3', '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--h-fine', '0.1']
    assert dc.tag(dict(net=['VHB', 'VIN'])) == 'VHB' and dc.status('fem dc: meshing') == 'meshing' and dc.status('x') is None
    with pytest.raises(ValueError, match='set --source, --load'):
        dc.command(dict(net='VHB'), board, 'out')
    with pytest.raises(ValueError, match='choose a board first'):
        dc.command(dict(net='VHB', source='L3', load='Q1=1'), '', 'out')
    cap = A.ElmerCap()
    assert cap.command(dict(net='A\nB+C', thick=True, order=2), board, 'o')[6:] == ['--net', 'A', '--net', 'B+C', '--thick']
    assert cap.tag(dict(net='A, B+C')) == 'A-B+C'


def test_the_temperature_takes_a_circuit_run_s_losses(tmp_path):
    board = str(tmp_path / 'b.kicad_pcb')
    open(board, 'w').close()
    th = A.ElmerThermal()
    with pytest.raises(ValueError, match='give the heat'):
        th.command({}, board, 'out')
    run = tmp_path / 'circuit'
    for name, loss in (('light', 0.1), ('heavy', 0.9)):
        (run / name).mkdir(parents=True)
        (run / name / 'case.json').write_text(json.dumps(dict(parts=dict(Q1=dict(power_W=loss), V1=dict(power_W=-5.0)))))
    line = th.command({}, board, 'out', results=dict(circuit=str(run)))
    assert line[6:] == ['--heat-from', str(run / 'heavy' / 'case.json')]
    assert th.command(dict(heat='Q1=0.3'), board, 'out', results=dict(circuit=str(run)))[6:] == ['--heat', 'Q1=0.3']


def test_reports_and_whose_result_a_folder_is(tmp_path, registered):
    d = tmp_path / 'dc'
    d.mkdir()
    (d / 'dc.json').write_text(json.dumps(dict(
        format=1, tool='kicad2ems_fem.elmer.dc', nets=['VHB'], source='L3', loads=dict(Q1=0.67, Q2=0.67),
        V=dict(Q1=-0.0021, Q2=-0.0034), drop_mV=dict(Q1=2.1, Q2=3.4), loss_W=0.0037, tets=12345, vias=4,
        seconds=dict(mesh=1.5, solve=2.0))))
    assert registry_module.analysis_of(str(d)) == ('elmer_dc', False)
    rep = A.ElmerDC().report(str(d))
    assert rep['title'] == 'VHB: DC drop from L3' and 'Worst drop 3.400 mV at Q2' in rep['sections'][0]['text']
    assert rep['sections'][1]['table']['rows'] == [['Q1', 0.67, 2.1], ['Q2', 0.67, 3.4]] and '12,345' in rep['footer']
    # the figures another engine of the rail's copper names alike (kicad2ems.study.compare)
    assert [(f['name'], f['value'], f['unit']) for f in A.ElmerDC().figures(str(d))] == \
        [('drop Q1', 2.1, 'mV'), ('drop Q2', 3.4, 'mV'), ('loss', 3.7, 'mW')]
    assert A.ElmerDC().figures(str(tmp_path)) == [] and A.ElmerThermal().figures(str(tmp_path)) == []
    t = tmp_path / 'th'
    t.mkdir()
    ours = dict(format=1, tool='kicad2ems_fem.elmer.thermal', T_max=71.2, T_min=30.0, ambient=25.0, htc=10.0,
                hottest=dict(x=1.0, y=2.0, layer='F.Cu'), parts=dict(Q1=70.5, L1=48.0), heat=dict(Q1=0.3, L1=0.1),
                wedges=999, seconds=dict(mesh=1.0, solve=3.0), balance=dict(in_W=0.4, out_W=0.4, error=0.0))
    (t / 'thermal.json').write_text(json.dumps(ours))
    assert registry_module.analysis_of(str(t)) == ('elmer_thermal', False)
    rep = A.ElmerThermal().report(str(t))
    assert rep['title'] == 'Temperature: hottest 71.2 °C' and rep['sections'][1]['table']['rows'][0] == ['Q1', 0.3, 70.5, 45.5]
    assert [(f['name'], f['value']) for f in A.ElmerThermal().figures(str(t))] == \
        [('hottest', 71.2), ('Q1', 70.5), ('L1', 48.0), ('coolest', 30.0)]
    (t / 'thermal.json').write_text(json.dumps(dict(method='grid', hottest_C=70.0)))       # pcbqs's file: not Elmer's
    assert not A.ElmerThermal().owns(str(t)) and registry_module.analysis_of(str(t)) is None
    c = tmp_path / 'cap'
    c.mkdir()
    (c / 'cap.json').write_text(json.dumps(dict(tool='kicad2ems_fem.palace.cap', names=['A', 'B'], thick=False, far='open',
                                                C=[[2e-12, -1.5e-12], [-1.5e-12, 3e-12]], tets=500, seconds=4.0, mesh_s=1.0)))
    assert registry_module.analysis_of(str(c)) == ('palace_cap', False)
    rep = A.PalaceCap().report(str(c))
    assert rep['sections'][0]['table']['columns'] == ['', 'A', 'B'] and rep['sections'][0]['table']['rows'][0] == ['A', 2.0, -1.5]
    assert rep['footer'].startswith('Palace, 500')
    assert [(f['name'], round(f['value'], 6)) for f in A.PalaceCap().figures(str(c))] == [('C A', 2.0), ('C B', 3.0), ('C A,B', 1.5)]
    assert A.ElmerCap().report(str(tmp_path)) is None


@pytest.mark.skipif(not elmer.found()[0], reason='Elmer not installed')
def test_a_drop_run_through_its_command(tmp_path, registered):
    """A 20 x 5 mm strip of 35 um copper carrying 1 A between two pads, run as a front end runs it: the command of
    the plugin, its mark, its report; the drop is the strip's own resistance."""
    pad = lambda ref, x, a: dict(ref=ref, pads=[dict(name='1', net='N', layers=['F.Cu'], geom=mapping(box(x, 0, x + a, 5.0)))])
    board = dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=1.6, t=0.035)],
                            layers=[dict(name='core', z0=0.0, z1=1.6, er=4.5, tand=0.02)]),
                 copper={'F.Cu': [dict(net='N', polygon=mapping(box(0, 0, 20.0, 5.0)))]}, vias=[],
                 outline=mapping(box(0, 0, 20.0, 5.0)), prims=dict(footprints=[pad('S', 0.0, 1.0), pad('L', 19.0, 1.0)]))
    path = tmp_path / 'strip_board.json'
    path.write_text(json.dumps(board))
    out = tmp_path / 'run'
    a = registered['elmer_dc']
    line = a.command(a.for_questions(dict(net='N', source='S', load='L=1', h=0.5, h_fine=0.25), ('dc',)), str(path), str(out))
    study.mark(str(out), 'elmer_dc', ('dc',))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    done = subprocess.run(line, capture_output=True, text=True, env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-800:]
    assert any(a.status(l) for l in done.stdout.splitlines())
    assert registry_module.analysis_of(str(out)) == ('elmer_dc', False)
    rep = a.report(str(out))
    drop = rep['sections'][1]['table']['rows'][0][2]
    assert drop == pytest.approx(1.0 * 18.0e-3 / (5.8e7 * 35e-6 * 5e-3) * 1e3, rel=0.03)       # 18 mm between the pads
