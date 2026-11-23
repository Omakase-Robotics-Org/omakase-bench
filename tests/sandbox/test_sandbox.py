import pytest

from omakase_bench.sandbox import run_program


def dispatch(fn, args, kwargs):
    if fn == "add":
        return args[0] + args[1]
    raise ValueError(f"no function {fn}")


def run(code, timeout=5.0):
    return run_program(code, dispatch, timeout=timeout)


def test_robot_calls_work():
    r = run("print(robot.add(2, 3)); import numpy as np; print(int(np.arange(4).sum()))")
    assert r.ok and r.stdout.split() == ["5", "6"] and r.calls == 1


ESCAPES = {
    "open builtin": "open('/etc/hostname').read()",
    "io module": "import io; io.open('/etc/hostname')",
    "os import": "import os; os.listdir('/')",
    "sys import": "import sys",
    "subprocess": "import subprocess; subprocess.run(['true'])",
    "socket": "import socket; socket.create_connection(('1.1.1.1', 80), timeout=1)",
    "ctypes": "import ctypes",
    "real builtins via module": "json.__builtins__['open']('/tmp/omakase-bench-escape', 'w').write('x')",
    "posix via importer": ("imp = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'BuiltinImporter'][0]\n"
                           "imp.load_module('posix').system('touch /tmp/omakase-bench-escape')"),
    "raw socket via importer": ("imp = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'BuiltinImporter'][0]\n"
                                "s = imp.load_module('_socket').socket(); s.connect(('1.1.1.1', 80))"),
    "fork via importer": ("imp = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'BuiltinImporter'][0]\n"
                          "imp.load_module('posix').fork()"),
}


@pytest.mark.parametrize("name", list(ESCAPES))
def test_escape_attempts_are_refused(name, tmp_path):
    import os
    marker = "/tmp/omakase-bench-escape"
    if os.path.exists(marker):
        os.remove(marker)
    r = run(ESCAPES[name])
    assert r.exit_code != 0, f"{name} ran: {r.stdout} {r.stderr}"
    assert any(word in r.stderr for word in ("PermissionError", "ImportError", "NameError", "KeyError")), r.stderr
    assert not os.path.exists(marker)


def test_wall_timeout_kills_the_program():
    r = run("while True:\n    pass", timeout=2.0)
    assert r.timed_out and not r.ok and r.wall_seconds < 6


def test_refusals_reach_the_program_as_robot_error():
    r = run("try:\n    robot.fly()\nexcept RobotError as e:\n    print('caught', e)")
    assert r.ok and "caught" in r.stdout and "no function fly" in r.stdout


def test_stop_episode_ends_the_program():
    from omakase_bench.harness.session import StopEpisode

    def stopping(fn, args, kwargs):
        raise StopEpisode("budget_exceeded:api_calls")
    r = run_program("for i in range(1000):\n    robot.ping()", stopping, timeout=5)
    assert r.stop is not None and r.stop.reason == "budget_exceeded:api_calls" and r.calls == 1
