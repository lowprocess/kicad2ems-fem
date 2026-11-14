"""
The solvers a tool starts (ElmerGrid, ElmerSolver, Palace with its MPI ranks, podman), each in a process group of
its own and stopped with the tool: on the tool's SIGTERM (the window's Stop), at its exit, and when the run's stop
is requested (kicad2ems.stop: a Palace driven run inside kicad2ems, whose STOP file the window writes). Started
plainly, they outlived a Stop: 8 Palace ranks went on for their half hour.

    p = popen(cmd, **kw)          # subprocess.Popen in a session of its own, watched as above
    r = run(cmd, timeout=None)    # subprocess.run's result (text captured); a timeout stops it: RuntimeError
    stopped(p)                    # whether it was stopped for a stop request (raise_stopped then)
    guard()                       # in a tool's main: SIGTERM stops its solvers, then the tool
"""
import atexit
import os
import signal
import subprocess
import threading
import time

_live = set()
_lock = threading.Lock()
POLL_S = 0.5                    # how often a running solver's run is asked whether it is to stop
GRACE_S = 3.0                   # how long a solver has after SIGTERM before SIGKILL


def _requested():
    try:
        from kicad2ems.stop import requested
    except ImportError:                                         # (kicad2ems older than its stop module)
        return False
    return requested()


def _signal(p, sig):
    try:
        if os.name == 'posix':
            os.killpg(p.pid, sig)                               # the solver and what it started (mpirun's ranks)
        else:
            p.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop(p, grace=GRACE_S):
    """Stop p and its process group: SIGTERM, then SIGKILL after `grace` s (podman stops its container on TERM)."""
    if p.poll() is not None:
        return
    _signal(p, signal.SIGTERM)
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        _signal(p, getattr(signal, 'SIGKILL', signal.SIGTERM))
        try:
            p.wait(grace)
        except subprocess.TimeoutExpired:
            pass


def stop_all():
    with _lock:
        live = list(_live)
    for p in live:
        stop(p)


def _watch(p):
    while p.poll() is None:
        if _requested():
            p.stop_requested = True
            stop(p)
            break
        time.sleep(POLL_S)
    with _lock:
        _live.discard(p)


def popen(cmd, **kw):
    """subprocess.Popen(cmd, **kw) in a session of its own (POSIX), stopped with the tool and on a stop request."""
    if os.name == 'posix':
        kw.setdefault('start_new_session', True)
    p = subprocess.Popen(cmd, **kw)
    p.stop_requested = False
    with _lock:
        _live.add(p)
    threading.Thread(target=_watch, args=(p,), daemon=True).start()
    return p


def stopped(p):
    """Whether p ended because the run was asked to stop."""
    return bool(getattr(p, 'stop_requested', False))


def raise_stopped():
    """kicad2ems.stop.Stopped (the run gives no result), or RuntimeError without kicad2ems's stop module."""
    try:
        from kicad2ems.stop import Stopped
    except ImportError:
        Stopped = RuntimeError
    raise Stopped('stop requested')


def run(cmd, timeout=None, cwd=None, env=None):
    """subprocess.run(cmd, capture_output=True, text=True)'s result, the solver started as popen's. A timeout stops
    it and raises RuntimeError saying so (subprocess.TimeoutExpired had escaped the tools' main); a stop request,
    raise_stopped."""
    p = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, env=env)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop(p)
        out, err = p.communicate()
        raise RuntimeError(f"{os.path.basename(str(cmd[0]))} did not finish in {timeout:g} s: stopped\n"
                           + ((out or '') + (err or ''))[-1500:]) from None
    if stopped(p):
        raise_stopped()
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def guard():
    """In a tool's main: its solvers stopped with it, on SIGTERM (the window's Stop: the tool then exits 143) and
    at its exit."""
    atexit.register(stop_all)
    if threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGTERM'):
        def on_term(signum, frame):
            stop_all()
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, on_term)
