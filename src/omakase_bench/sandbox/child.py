"""The sandboxed interpreter. Started by ``runner.run_program`` as ``python -I child.py REQ RESP``.

It reads the program from stdin, then locks itself down before running it:
an audit hook refuses file opens, process creation, sockets and other OS
access; imports are limited to a short allow-list of modules loaded before the
lock; and the parent has set resource limits (no new processes, no file
writes, bounded memory and CPU). The only way out is the robot proxy, which
sends JSON requests over an inherited pipe.
"""
import builtins
import json
import linecache
import math
import os
import random
import sys
import traceback

import numpy

ALLOWED_IMPORTS = {"math", "numpy", "json", "random", "itertools", "collections", "functools",
                   "statistics", "time", "dataclasses", "typing"}
BLOCKED_EVENT_PREFIXES = ("open", "os.", "subprocess.", "socket.", "shutil.", "ctypes.", "_ctypes",
                          "mmap.", "fcntl.", "pty.", "glob.", "pathlib.", "tempfile.", "urllib.",
                          "http.", "ftplib.", "smtplib.", "telnetlib.", "webbrowser.", "winreg.",
                          "sqlite3.", "socketserver", "_posixsubprocess", "signal.", "resource.",
                          "sys.remote_exec", "sys.monitoring", "cpython.run_", "sys.setprofile",
                          "sys.settrace", "code.__new__", "marshal.", "pickle.")
for _m in ("itertools", "collections", "functools", "statistics", "time", "dataclasses", "typing"):
    __import__(_m)


class RobotError(Exception):
    """The robot refused a call (nothing moved) or the call was invalid."""


class EpisodeOver(BaseException):
    """The harness ended the episode; the program must stop."""


def main() -> None:
    req_fd, resp_fd = int(sys.argv[1]), int(sys.argv[2])
    code = sys.stdin.read()
    sys.stdin.close()
    buf = b""

    def rpc(fn, args, kwargs):
        nonlocal buf
        line = json.dumps({"fn": fn, "args": args, "kwargs": kwargs}).encode() + b"\n"
        os.write(req_fd, line)
        while b"\n" not in buf:
            chunk = os.read(resp_fd, 65536)
            if not chunk:
                raise EpisodeOver("harness closed the connection")
            buf += chunk
        line, _, buf = buf.partition(b"\n")
        reply = json.loads(line)
        if reply.get("stop"):
            raise EpisodeOver(reply["stop"])
        if not reply["ok"]:
            raise RobotError(reply["error"])
        return reply["result"]

    class Robot:
        __slots__ = ()

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return lambda *a, **k: rpc(name, list(a), k)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox; "
                              f"allowed: {', '.join(sorted(ALLOWED_IMPORTS))}")
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = guarded_import
    for name in ("open", "input", "breakpoint", "help", "exit", "quit"):
        safe_builtins.pop(name, None)

    linecache.cache["<program>"] = (len(code), None, code.splitlines(True), "<program>")

    def audit(event, args):
        if event.startswith(BLOCKED_EVENT_PREFIXES):
            raise PermissionError(f"sandbox: {event} is not allowed")
        if event == "import" and str(args[0]).split(".")[0] not in ALLOWED_IMPORTS and args[0] not in sys.modules:
            raise PermissionError(f"sandbox: import of {args[0]} is not allowed")

    sys.addaudithook(audit)
    env = {"__name__": "__program__", "__builtins__": safe_builtins, "robot": Robot(), "RobotError": RobotError,
           "math": math, "np": numpy, "numpy": numpy, "json": json, "random": random}
    try:
        exec(compile(code, "<program>", "exec"), env)
    except EpisodeOver as stop:
        print(f"[sandbox] episode ended: {stop}", file=sys.stderr)
        sys.exit(3)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
