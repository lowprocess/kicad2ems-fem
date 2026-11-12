"""Where Elmer and Palace are found: the environment first, then what kicad2ems's installer recorded
(kicad2ems.sitecfg: elmer, palace, palace_rootfs), then PATH."""
import os
import sys

import pytest

from kicad2ems_fem.elmer import run as elmer
from kicad2ems_fem.palace import run as palace


@pytest.fixture
def site(monkeypatch):
    """The installer's settings as a dict the test fills."""
    from kicad2ems import sitecfg
    values = {}
    monkeypatch.setattr(sitecfg, 'get', lambda key: values.get(key))
    for key in ('K2E_ELMER', 'K2E_PALACE', 'K2E_PALACE_ROOTFS'):
        monkeypatch.delenv(key, raising=False)
    return values


def _exe(folder, name):
    path = folder / (name + ('.exe' if os.name == 'nt' else ''))
    path.write_text('')
    path.chmod(0o755)
    return path


def test_elmer_from_the_environment_then_the_installer(site, tmp_path, monkeypatch):
    rec, env = tmp_path / 'recorded', tmp_path / 'env'
    rec.mkdir(), env.mkdir()
    for name in ('ElmerSolver', 'ElmerGrid'):
        _exe(rec, name)
    site['elmer'] = str(rec)
    assert elmer._tool('ElmerSolver') == str(rec / ('ElmerSolver' + ('.exe' if os.name == 'nt' else '')))
    assert elmer.found()[0] and str(rec) in elmer.found()[1]
    monkeypatch.setenv('K2E_ELMER', str(env))                       # the environment first: there is none there
    assert elmer._tool('ElmerSolver') is None and not elmer.found()[0]


def test_palace_from_the_environment_then_the_installer_then_path(site, tmp_path, monkeypatch):
    monkeypatch.setattr(palace.shutil, 'which', lambda name: '/on/the/path/palace' if name == 'palace' else None)
    assert palace.launcher() == ('native', '/on/the/path/palace')
    site['palace'] = str(tmp_path / 'view' / 'bin' / 'palace')
    assert palace.launcher() == ('native', site['palace'])
    monkeypatch.setenv('K2E_PALACE', '/from/the/environment/palace')
    assert palace.launcher() == ('native', '/from/the/environment/palace')
    monkeypatch.delenv('K2E_PALACE')
    site.clear()
    monkeypatch.setattr(palace.shutil, 'which', lambda name: '/usr/bin/podman' if name == 'podman' else None)
    image = tmp_path / 'image'
    (image / 'opt' / 'views' / 'view' / 'bin').mkdir(parents=True)
    site['palace_rootfs'] = str(image)
    assert palace.launcher() == ('podman', str(image))


@pytest.mark.skipif(sys.platform == 'win32', reason='Palace runs on Linux and macOS')
def test_a_native_palace_runs_with_its_own_folder_first_on_path(tmp_path, monkeypatch):
    """A Spack view's palace launcher looks for mpirun on PATH: its own folder (the view's bin) is put first."""
    monkeypatch.setenv('PATH', '/usr/bin')
    cmd, _ = palace.command(str(tmp_path), 'palace.json', np=2, how=('native', str(tmp_path / 'view' / 'bin' / 'palace')))
    assert cmd[:3] == [str(tmp_path / 'view' / 'bin' / 'palace'), '-np', '2']
    assert palace.environment(cmd)['PATH'] == os.pathsep.join([str(tmp_path / 'view' / 'bin'), '/usr/bin'])
    pod, _ = palace.command(str(tmp_path), 'palace.json', how=('podman', str(tmp_path)))
    assert palace.environment(pod)['PATH'] == '/usr/bin'           # the image has its own
