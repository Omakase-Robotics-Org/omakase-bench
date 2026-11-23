"""The robot daemon API the bench speaks: the vendored document, command lines, the simulated daemon, the
HTTP client (only against a local mock server, or printing)."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from omakase_bench.core.suite import find_task

from omakase_bench.core.task import robot_for
from omakase_bench.backends.real import http
from omakase_bench.backends.real.d1 import api, mirror
from omakase_bench.harness import command_api
from omakase_bench.robots.spec import load_robot

ROBOT = load_robot("d1")
CFG = api.robot_api(ROBOT)
SPEC = CFG.spec
SYN = CFG.syntax({"left": "L", "right": "R"}, 7)
LINES = ["joints: L 10 -20 0 -90 0 0 0 vel=0.3", "joints: both 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0", "joint: R 3 -45",
         "gripper: L 0.4 force=30", "gripper: R 1", "lift: 0.15 speed=0.5", "head: -0.3 0.2",
         "base: vel 0.1 0 0.6", "base: drive -0.06", "base: turn 0.2"]


def test_the_vendored_document_has_every_operation_the_bench_sends():
    ops = {(m.lower(), p) for p, v in SPEC["paths"].items() for m in v}
    assert SPEC["x-vendored"]["commit"] and len(ops) == len(SPEC["x-vendored"]["operations"])
    sent = set()
    for line in LINES:
        kind, rest = line.split(":", 1)
        for r in api.translate(kind, rest, SYN):
            op, _ = command_api.operation(SPEC, r.method, r.path)
            sent.add((r.method.lower(), r.path))
    assert len(sent) >= 9


@pytest.mark.parametrize("line", LINES)
def test_every_command_line_is_a_valid_request(line):
    kind, rest = line.split(":", 1)
    reqs = api.translate(kind, rest, SYN)
    for r in reqs:
        assert command_api.validate(SPEC, r.method, r.path, r.body) == [], (line, r)
    assert reqs[-1].then_wait_s == 0


def test_requests_follow_the_documented_fields_and_units():
    r = api.translate("joints", " L 10 -20 0 -90 0 0 0 vel=0.3", SYN)
    assert [(x.path, x.body) for x in r] == [
        ("/v1/arm/a/mode", {"mode": "position", "vel_ratio": 0.3, "acc_ratio": 0.3}),
        ("/v1/arm/a/move_joints", {"joints_deg": [10, -20, 0, -90, 0, 0, 0], "wait": True})]
    assert api.translate("gripper", "R 0.4 force=30", SYN)[0].body == {"openness": 0.4, "force_percent": 30, "wait": True}
    base = api.translate("base", "drive -0.06", SYN)
    assert [x.body for x in base] == [{"vx": -0.1, "omega": 0.0}, {"vx": 0.0, "omega": 0.0}]
    assert base[0].then_wait_s == pytest.approx(0.6)
    # the document refuses what the bench must not send
    assert command_api.validate(SPEC, "POST", "/v1/arm/c/move_joints", {"joints_deg": [0] * 7})
    assert command_api.validate(SPEC, "POST", "/v1/neck/cmd", {"pitch": "up"})
    assert command_api.validate(SPEC, "POST", "/v1/chassis/cmd_vel", {"vx": 0.1})
    assert command_api.validate(SPEC, "POST", "/v1/arm/move_joints", {}) != []


def test_simulated_daemon_serves_every_vendored_operation_and_answers_in_its_envelope():
    from omakase_bench.backends.mujoco.backend import MujocoBackend
    from omakase_bench.harness.session import RunBudget, Session
    from omakase_bench.harness.ee_control import EEController, WholeBody
    task = find_task("manip/chess_move")
    ep = task.resolve(0)
    be = MujocoBackend()
    be.load(robot_for(task, ep.scene), ep.scene)
    be.reset(ep.scene, seed=0)
    s = Session(ep, be, robot_for(task, ep.scene), run_budget=RunBudget(None, None))
    ctl = EEController(s)
    ctl.body = WholeBody(base=True, lift=True, head=True, joints=True)
    sim = mirror.SimDaemon(ctl, {"a": "left", "b": "right"})
    for path, ops in SPEC["paths"].items():
        for method in ops:
            concrete = path.replace("{side}", "a")
            body = {"/v1/arm/{side}/mode": {"mode": "position", "vel_ratio": 0.2, "acc_ratio": 0.2},
                    "/v1/arm/{side}/move_joint": {"joint": 0, "position": 0.0},
                    "/v1/arm/{side}/move_joints": {"joints_deg": [0.0] * 7},
                    "/v1/arm/move_joints_both": None,
                    "/v1/end_effectors/{side}/set": {"openness": 0.5},
                    "/v1/slider/set_height": {"height_m": 0.12},
                    "/v1/neck/cmd": {"pitch": -0.2, "yaw": 0.0}, "/v1/chassis/cmd_vel": {"vx": 0.0, "omega": 0.0},
                    "/v1/arm/lease": {"holder": "test", "class": "policy"}}.get(path)
            if path == "/v1/arm/move_joints_both":
                q = be.joint_positions()
                import math
                body = {side: [math.degrees(q[j]) for j in ROBOT.arms[arm].joints] for side, arm in sim.sides.items()}
            env = sim.request(method.upper(), concrete, body)
            assert env.get("http_status") != 404, (path, env)
            assert set(env) >= {"status", "data", "message"}
    # its state operations answer in the document's format, with the simulated joints
    for path in mirror.STATE:
        envelope = SPEC["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert command_api.check(SPEC, envelope, sim.request("GET", path), path) == [], path
    feedback = sim.request("GET", "/v1/arm/feedback")["data"]
    q = be.joint_positions()
    import math
    assert feedback["sides"][0]["pos"] == [math.degrees(q[j]) for j in ROBOT.arms["left"].joints]
    assert sim.request("GET", "/v1/slider/state")["data"]["height_m"] == q["lift_joint"]
    # the lease: first come within a class, a higher class takes it, only the holder hands it back
    assert sim.request("GET", "/v1/arm/lease")["data"] is None
    grant = sim.request("POST", "/v1/arm/lease", {"holder": "bench", "class": "policy", "ttl_s": 30})["data"]
    assert grant["epoch"] == sim.request("POST", "/v1/arm/lease", {"holder": "bench", "class": "policy"})["data"]["epoch"]
    assert sim.request("POST", "/v1/arm/lease", {"holder": "other", "class": "policy"})["http_status"] == 409
    assert sim.request("DELETE", "/v1/arm/lease", {"holder": "other"})["kind"] == "lease_held"
    assert sim.request("POST", "/v1/arm/lease", {"holder": "console", "class": "operator"})["data"]["preempted_from"] == "bench"
    assert sim.request("DELETE", "/v1/arm/lease", {"holder": "console"})["status"] == "ok"
    q = be.joint_positions()
    pose = sim.request("GET", "/v1/chassis/amr/localization")["data"]["mission_frame"]["odometry"]
    assert pose == {"x": q["base_x_joint"], "y": q["base_y_joint"], "theta": q["base_yaw_joint"]}
    limits = sim.request("GET", "/v1/limits")
    response = SPEC["paths"]["/v1/limits"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert command_api.check(SPEC, response, limits) == []
    assert sim.request("POST", "/v1/end_effectors/a/set", {"openness": 1.5})["kind"] == "unprocessable"
    assert sim.request("POST", "/v1/end_effectors/a/set", {"openness": 0.5, "force_percent": 120})["http_status"] == 422
    assert sim.request("POST", "/v1/arm/a/mode", {"mode": "position", "vel_ratio": 2, "acc_ratio": 0.1})["kind"] == "invalid"
    assert sim.request("POST", "/v1/slider/set_height", {"height_m": 0.4})["http_status"] == 400
    assert sim.request("POST", "/v1/arm/a/move_joint", {"joint": 6, "position": 120})["kind"] == "refused"
    assert "clamped" in sim.request("POST", "/v1/neck/cmd", {"pitch": 3.0})["note"]


def test_the_http_client_never_starts_unless_enabled_twice(monkeypatch):
    monkeypatch.delenv(http.ENABLE_ENV, raising=False)
    monkeypatch.delenv(http.URL_ENV, raising=False)
    with pytest.raises(PermissionError):
        http.HttpDaemonClient(SPEC, "http://127.0.0.1:1")
    with pytest.raises(PermissionError):
        http.HttpDaemonClient(SPEC, "http://127.0.0.1:1", enable=True)
    monkeypatch.setenv(http.ENABLE_ENV, "1")
    with pytest.raises(ValueError):
        http.HttpDaemonClient(SPEC, enable=True)                       # no URL: there is no default host


def test_dry_run_prints_requests_and_opens_no_socket(monkeypatch):
    def no_socket(*a, **k):
        raise AssertionError("a dry run opened a socket")
    monkeypatch.setattr(socket, "socket", no_socket)
    printed = []
    client = http.HttpDaemonClient(SPEC, dry_run=True, echo=printed.append)
    out = api.run_lines(client, SPEC, SYN, "joints: L 10 -20 0 -90 0 0 0 vel=0.3\nbase: drive 0.06")
    assert out == ["POST /v1/arm/a/mode -> ok", "POST /v1/arm/a/move_joints -> ok", "POST /v1/chassis/cmd_vel -> ok",
                   "POST /v1/chassis/cmd_vel -> ok"]
    assert printed[1] == 'POST <robot>/v1/arm/a/move_joints {"joints_deg": [10.0, -20.0, 0.0, -90.0, 0.0, 0.0, 0.0], "wait": true}'
    assert printed[3] == "# wait 0.6 s"
    assert client.request("POST", "/v1/neck/cmd", {"pitch": "up"})["kind"] == "invalid"     # checked, not sent


def test_http_client_against_a_local_mock_server(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            if body.get("pitch") == 0.9:
                import time
                time.sleep(0.6)
            if self.path.endswith("/move_joints") and body["joints_deg"][0] > 90:
                payload, code = {"status": "error", "data": None, "message": "outside travel", "kind": "refused"}, 409
            else:
                payload, code = {"status": "ok", "data": None, "message": None}, 200
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.handle_error = lambda *args: None        # the impatient client hangs up on purpose
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv(http.ENABLE_ENV, "1")
        client = http.HttpDaemonClient(SPEC, f"http://127.0.0.1:{server.server_port}", enable=True)
        assert client.request("POST", "/v1/arm/b/move_joints", {"joints_deg": [0.0] * 7, "wait": True})["status"] == "ok"
        env = client.request("POST", "/v1/arm/b/move_joints", {"joints_deg": [100.0] + [0.0] * 6, "wait": True})
        assert env["kind"] == "refused" and env["http_status"] == 409
        assert seen[0] == ("/v1/arm/b/move_joints", {"joints_deg": [0.0] * 7, "wait": True})
        # a robot that does not answer in time is an error envelope, and the line stops there
        impatient = http.HttpDaemonClient(SPEC, f"http://127.0.0.1:{server.server_port}", enable=True, timeout_s=0.2)
        monkeypatch.setattr(http, "TIMEOUT_MARGIN_S", 0.0)
        out = api.run_lines(impatient, SPEC, SYN, "head: 0.9 0")
        assert len(out) == 1 and "error 0 unreachable: no answer from the robot within 0.2 s" in out[0]
    finally:
        server.shutdown()


def test_blocking_operations_get_the_time_the_document_gives_them(monkeypatch):
    import urllib.request
    waits = []

    def fake_urlopen(req, timeout):
        waits.append((req.full_url.rsplit("/v1/", 1)[1], timeout))
        raise TimeoutError

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv(http.ENABLE_ENV, "1")
    client = http.HttpDaemonClient(SPEC, "http://127.0.0.1:1", enable=True)
    client.request("POST", "/v1/slider/set_height", {"height_m": 0.1, "wait": True})
    client.request("POST", "/v1/neck/cmd", {"pitch": 0.0, "yaw": 0.0})
    assert waits == [("slider/set_height", 305.0), ("neck/cmd", 35.0)]


def test_the_simulated_daemons_error_statuses_are_ones_the_document_lists():
    answered = {("POST", "/v1/end_effectors/{side}/set"): {422, 409}, ("POST", "/v1/arm/{side}/mode"): {400, 409},
                ("POST", "/v1/arm/{side}/move_joints"): {400, 409}, ("POST", "/v1/arm/{side}/move_joint"): {400, 409},
                ("POST", "/v1/arm/move_joints_both"): {409}, ("POST", "/v1/slider/set_height"): {400, 409}}
    for (method, path), codes in answered.items():
        listed = {int(c) for c in SPEC["paths"][path][method.lower()]["responses"]}
        assert codes <= listed and 404 in listed, (path, listed)


def test_an_interrupted_velocity_hold_still_sends_the_stop():
    sent = []

    class Interrupted:
        def request(self, method, path, body=None):
            sent.append(body)
            return command_api.ok()

        def wait(self, seconds):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        api.run_lines(Interrupted(), SPEC, SYN, "base: drive 0.06")
    assert sent == [{"vx": 0.1, "omega": 0.0}, {"vx": 0.0, "omega": 0.0}]
