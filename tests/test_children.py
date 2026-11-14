"""The solvers a tool starts: in a process group of their own, stopped with what started them (children.py)."""
import os
import subprocess
import sys
import time

import pytest

from kicad2ems_fem import children

posix = pytest.mark.skipif(os.name != 'posix', reason='process groups')
TREE = [sys.executable, '-c', "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"]


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


@posix
def test_a_solver_and_what_it_started_stop_together():
    """mpirun's ranks, podman's container: the solver's whole group goes, not only the process started."""
    p = children.popen(TREE)
    time.sleep(0.5)
    assert os.getpgid(p.pid) == p.pid != os.getpgid(0)           # a session of its own
    children.stop(p, grace=1.0)
    time.sleep(0.2)
    assert p.poll() is not None and not _group_alive(p.pid)


@posix
def test_a_stop_request_stops_it(tmp_path, monkeypatch):
    """A run's STOP file (the window's Stop of a kicad2ems run that runs Palace): the solver is stopped within
    POLL_S, and run() raises kicad2ems's Stopped."""
    pytest.importorskip('kicad2ems.stop')
    from kicad2ems import stop
    monkeypatch.setenv(stop.ENV, str(tmp_path / 'STOP'))
    t0 = time.time()
    (tmp_path / 'STOP').write_text('')
    with pytest.raises(stop.Stopped):
        children.run(TREE, timeout=30)
    assert time.time() - t0 < 10


def test_a_timeout_says_so():
    with pytest.raises(RuntimeError, match='did not finish in 0.5 s'):
        children.run([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=0.5)


@posix
def test_a_tool_s_sigterm_stops_its_solvers(tmp_path):
    """The window's Stop sends a tool SIGTERM: guard() stops the tool's solvers before it exits (they were left
    running)."""
    code = ("import time, sys; from kicad2ems_fem import children; children.guard(); "
            f"p = children.popen({TREE!r}); open({str(tmp_path / 'pid')!r}, 'w').write(str(p.pid)); time.sleep(60)")
    tool = subprocess.Popen([sys.executable, '-c', code])
    for _ in range(100):
        if (tmp_path / 'pid').exists() and (tmp_path / 'pid').read_text():
            break
        time.sleep(0.05)
    pid = int((tmp_path / 'pid').read_text())
    tool.terminate()
    assert tool.wait(10) == 143
    time.sleep(0.2)
    assert not _group_alive(pid)


def test_palace_s_unconverged_solves_are_said(tmp_path, monkeypatch):
    """Palace's own words for a solve that did not converge are a WARNING and kept with its changeset beside the log
    (only its exit code had been read); its ranks default to the fast cores (8 had been fixed)."""
    import json, sys
    from kicad2ems_fem.palace import run as prun
    script = tmp_path / 'fake_palace.py'
    script.write_text("print('Git changeset ID: abc123')\nprint('GMRES solver did NOT converge in 400 iterations')\n"
                      "print('Linear solver did not converge, norm(Ax-b)/norm(b) = 1.3e-05')\n")
    monkeypatch.setattr(prun, 'command', lambda workdir, cfg, np, threads, how: ([sys.executable, str(script)], workdir))
    said = []
    prun.run(str(tmp_path), 'palace.json', log=said.append)
    info = json.loads((tmp_path / 'palace.log.json').read_text())
    assert info['changeset'] == 'abc123' and len(info['unconverged']) == 2
    assert any(s.startswith('WARNING palace: its solver did not converge 2 time(s)') for s in said)
    assert prun.ranks(None) >= 1 and prun.ranks(3) == 3
