"""Which Elmer or Palace solved a case is in each tool's result (R-B14): Elmer's from its log's Version line,
Palace's changeset from what its run wrote beside its log."""
import json


def test_elmer_s_version_from_its_log():
    from kicad2ems_fem.elmer.run import engine
    log = ("ELMER SOLVER (v 26.2) STARTED AT: 2026/11/09 19:05:44\n"
           "MAIN: Version: 26.2-devel (Rev: 033b105e2, Compiled: 2026-06-22)\n")
    e = engine(log)
    assert (e['program'], e['version'], e['revision']) == ('Elmer', '26.2-devel', '033b105e2') and e['kicad2ems_fem']
    assert engine('')['version'] is None


def test_palace_s_changeset_beside_its_log(tmp_path):
    from kicad2ems_fem.palace.run import engine
    (tmp_path / 'palace.log.json').write_text(json.dumps(dict(changeset='0dc74cd', ranks=8, unconverged=[])))
    assert engine(str(tmp_path))['revision'] == '0dc74cd' and engine(str(tmp_path / 'none'))['revision'] is None
