"""A stand-in robot on the loopback interface: the D1 daemon's state and motion operations over HTTP and
WebSocket in the formats of the vendored API document, and JPEG camera snapshots. No test in this
directory opens a connection to anything else."""
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from omakase_bench.backends.real import http
from omakase_bench.harness import command_api
from omakase_bench.robots.spec import load_robot

ROBOT = load_robot("d1")
SPEC = command_api.load_spec(str(ROBOT.preset_dir / ROBOT.api["spec"]))
#: what the stand-in adds to arm joint 1 of each side, so that a reading is told from an echoed target
OFFSET_DEG = 0.5


def example(schema):
    return command_api.skeleton(SPEC, schema)


def response_schema(path):
    return {"properties": {"data": command_api.response_data(SPEC, "GET", path)}}


class Robot:
    """The stand-in's state: joint targets it accepted become its measured positions."""

    def __init__(self):
        self.arms = {"a": [0.0] * 7, "b": [0.0] * 7}          # degrees
        self.openness = {"a": 1.0, "b": 1.0}
        self.height = 0.1
        self.pitch, self.yaw = -0.45, 0.0
        self.posts, self.frames, self.gets = [], [], []
        self.refuse_above_deg = 170.0
        self.lease, self.lease_calls, self.epoch = None, [], 0
        self.amr, self.odometry, self.velocities = True, {"x": 0.0, "y": 0.0, "theta": 0.0}, []

    RANK = {"ambient": 10, "policy": 50, "operator": 90}

    def lease_call(self, method, body):
        """The daemon's arm lease: first come within a class, a higher class preempts."""
        self.lease_calls.append((method, body))
        held = self.lease
        if method == "GET":
            return {"status": "ok", "data": None if held is None else {k: v for k, v in held.items() if k != "token"}, "message": None}, 200
        if method == "DELETE":
            if held is not None and held["holder"] != body["holder"]:
                return {"status": "error", "data": None, "message": f"held by {held['holder']}", "kind": "lease_held"}, 409
            self.lease = None
            return {"status": "ok", "data": None, "message": None}, 200
        rank = self.RANK[body.get("class", "ambient")]
        if held is not None and held["holder"] != body["holder"] and rank <= self.RANK[held["class"]]:
            return {"status": "error", "data": None, "message": f"the arms are held by {held['holder']} ({held['class']})",
                    "kind": "lease_held"}, 409
        if held is None or held["holder"] != body["holder"]:
            self.epoch += 1
        ttl = body.get("ttl_s") or 30
        self.lease = {"holder": body["holder"], "class": body.get("class", "ambient"), "epoch": self.epoch,
                      "expires_in_s": ttl, "ttl_s": ttl, "preempted_from": None, "token": "0123456789abcdef"}
        return {"status": "ok", "data": dict(self.lease), "message": None}, 200

    def read(self, path):
        if path == "/v1/chassis/amr/localization":
            if not self.amr:
                return {"status": "error", "data": None, "message": "the chassis backend is not d1amr", "kind": "unavailable"}
            return {"status": "ok", "data": {"pose": dict(self.odometry), "mission_frame": {"frame": "odom", "odometry": dict(self.odometry)},
                                             "status": {"state": "confident"}}, "message": None}
        data = example(response_schema(path)["properties"]["data"])
        if path == "/v1/arm/feedback":
            side = data["sides"][0] if data["sides"] else example(command_api.member(SPEC, command_api.response_data(SPEC, "GET", path), "properties", "sides", "items"))
            data["sides"] = [{**side, "pos": [v + (OFFSET_DEG if i == 0 else 0.0) for i, v in enumerate(self.arms[s])]}
                             for s in ("a", "b")]
            data["header"]["mono_ns"] = 1_000_000 * (len(self.gets) + 1)
        elif path == "/v1/end_effectors":
            schema = command_api.member(SPEC, command_api.response_data(SPEC, "GET", path), "items")
            item, state = example(schema), example(command_api.member(SPEC, schema, "properties", "state"))
            data = [{**item, "side": s, "fitted": True, "available": True, "state": {**state, "openness": self.openness[s]}}
                    for s in ("a", "b")]
        elif path == "/v1/slider/state":
            data["height_m"] = self.height
        elif path == "/v1/neck/state":
            data.update(pitch=self.pitch, yaw=self.yaw)
        return {"status": "ok", "data": data, "message": None}

    def command(self, path, body):
        """Apply a motion request; returns (envelope, http status)."""
        self.posts.append((path, body))
        arm_motion = path.startswith("/v1/arm/")
        if arm_motion and self.lease is not None and body.get("holder") != self.lease["holder"]:
            return {"status": "error", "data": None, "message": f"the arms are held by {self.lease['holder']}", "kind": "lease_held"}, 409
        if path == "/v1/chassis/cmd_vel":
            self.velocities.append((body["vx"], body["omega"]))
        if path == "/v1/arm/move_joints_both":
            targets = {"a": body["a"], "b": body["b"]}
        elif path.endswith("/move_joints"):
            targets = {path.split("/")[3]: body["joints_deg"]}
        else:
            targets = {}
        if any(abs(v) > self.refuse_above_deg for t in targets.values() for v in t):
            return {"status": "error", "data": None, "message": "outside travel", "kind": "refused"}, 409
        self.arms.update(targets)
        if "/end_effectors/" in path:
            self.openness[path.split("/")[3]] = body["openness"]
        if path == "/v1/slider/set_height":
            self.height = body["height_m"]
        if path == "/v1/neck/cmd":
            self.pitch, self.yaw = body["pitch"], body["yaw"]
        return {"status": "ok", "data": None, "message": None}, 200


def jpeg(camera, size=(640, 480)):
    """A synthetic frame: a gradient whose colour says which camera it is."""
    tint = {"head": (255, 0, 0), "left_wrist": (0, 255, 0), "right_wrist": (0, 0, 255)}[camera]
    ramp = np.linspace(0.2, 1.0, size[0])[None, :, None] * np.ones((size[1], 1, 1))
    out = io.BytesIO()
    Image.fromarray((ramp * np.array(tint)).astype(np.uint8)).save(out, format="JPEG", quality=95)
    return out.getvalue()


@pytest.fixture
def mock_robot(monkeypatch):
    from websockets.sync.server import serve
    robot = Robot()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, raw, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else None

        def do_GET(self):
            if self.path.startswith("/cam/"):
                name = self.path[len("/cam/"):-len("/frame.jpg")]
                robot.frames.append(name)
                return self._send(200, jpeg(name), "image/jpeg")
            if self.path == "/v1/arm/lease":
                payload, code = robot.lease_call("GET", None)
                return self._send(code, json.dumps(payload).encode())
            robot.gets.append(self.path)
            payload = robot.read(self.path)
            self._send(200 if payload["status"] == "ok" else 502, json.dumps(payload).encode())

        def do_POST(self):
            payload, code = robot.lease_call("POST", self._body()) if self.path == "/v1/arm/lease" else robot.command(self.path, self._body())
            self._send(code, json.dumps(payload).encode())

        def do_DELETE(self):
            payload, code = robot.lease_call("DELETE", self._body())
            self._send(code, json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    def socket(connection):
        connection.send(json.dumps({"event": "soft_kill", "latched": False}))     # an unsolicited frame
        for message in connection:
            frame = json.loads(message)
            path = {"arm.move_joints_both": "/v1/arm/move_joints_both"}.get(frame["method"])
            if path is None and frame["method"].endswith(".move_joints"):
                path = f"/v1/arm/{frame['method'].split('.')[1]}/move_joints"
            payload, _ = robot.command(path or frame["method"], frame.get("params") or {})
            connection.send(json.dumps({"id": frame["id"] + 1000, "status": "ok", "data": None, "message": None}))  # another call's answer
            connection.send(json.dumps({"id": frame["id"], **payload}))

    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=web.serve_forever, daemon=True).start()
    ws = serve(socket, "127.0.0.1", 0)
    threading.Thread(target=ws.serve_forever, daemon=True).start()
    monkeypatch.setenv(http.ENABLE_ENV, "1")
    robot.url = f"http://127.0.0.1:{web.server_port}"
    robot.ws_url = f"ws://127.0.0.1:{ws.socket.getsockname()[1]}/v1/ws"
    robot.cameras = {f"camera.{name}": f"{robot.url}/cam/{name}/frame.jpg" for name in ROBOT.cameras}
    robot.degrees = math.degrees
    yield robot
    ws.shutdown()
    web.shutdown()
    web.server_close()


# --- stand-ins for what is on the robot only -----------------------------------------------------------

class FileCapture:
    """A video device whose frames are an image file (BGR, as OpenCV delivers them)."""

    def __init__(self, path):
        self.path, self.released = path, False

    def read(self):
        rgb = np.asarray(Image.open(self.path).convert("RGB"))
        return True, rgb[:, :, ::-1].copy()

    def release(self):
        self.released = True


class FakeBus:
    """The shared-memory layer: a queue of buffers per topic."""

    def __init__(self):
        self.queues, self.closed = {}, False

    def publish(self, topic, buffer):
        self.queues.setdefault(topic, []).append(buffer)

    def subscribe(self, topic):
        queue = self.queues.setdefault(topic, [])
        return lambda: queue.pop(0) if queue else None

    def close(self):
        self.closed = True


class LiveBus(FakeBus):
    """A topic that keeps publishing: every other read delivers the newest buffer again."""

    def subscribe(self, topic):
        queue, turn = self.queues.setdefault(topic, []), [False]

        def receive():
            turn[0] = not turn[0]
            return queue[-1] if turn[0] and queue else None
        return receive


def srvd_colour(rgb, frame_id="cam"):
    """One colour frame as the camera daemon publishes it (rgba8)."""
    from omakase_bench.backends.real import srvd
    h, w = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
    return srvd.HEADER.pack(srvd.MAGIC, srvd.VERSION, 0, 0, 123, 0, frame_id.encode(), w, h, 1) + rgba.tobytes()


def srvd_depth(metres, intrinsics, scale=0.001, frame_id="depth"):
    """One depth frame as the camera daemon publishes it (gray16 with a depth block)."""
    from omakase_bench.backends.real import srvd
    h, w = metres.shape
    block = srvd.DEPTH_BLOCK.pack(1, 16, 1, 0, scale, 0.0, 10.0, scale, *intrinsics, *([0.0] * 8))
    values = np.round(metres / scale).astype("<u2")
    return srvd.HEADER.pack(srvd.MAGIC, srvd.VERSION, 0, srvd.FLAG_DEPTH, 456, 0, frame_id.encode(), w, h, srvd.GRAY16) + block + values.tobytes()


def calibration_file(path, cameras):
    """A calibration file of schema omakase.camera_calibration/2 with the given lenses and mounts."""
    gate = {"verdict": "PASS", "checks": []}
    prov = {"tool": "test", "date": "2026-01-01"}
    out = {"schema": "omakase.camera_calibration/2", "robot": "d1-test", "updated": "2026-01-01T00:00:00Z", "cameras": {}}
    for name, cam in cameras.items():
        entry = {"stream": {"width": cam["width"], "height": cam["height"]},
                 "intrinsics": {"fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"],
                                "distortion": {"model": cam.get("model", "none"), "coefficients": cam.get("coefficients", [])},
                                "gate": cam.get("gate", gate), "provenance": prov}}
        if "mount" in cam:
            parent, xyz, quat = cam["mount"]
            pose = {"xyz_m": list(xyz), "quat_xyzw": list(quat)}
            entry["mount"] = {"parent_link": parent, "T_parent_camera": pose, "nominal": {"T_parent_camera": pose},
                              "gate": gate, "provenance": prov}
        out["cameras"][name] = entry
    path.write_text(json.dumps(out))
    return path


@pytest.fixture
def kit():
    """The stand-ins for what exists on the robot only."""
    from types import SimpleNamespace
    return SimpleNamespace(FakeBus=FakeBus, LiveBus=LiveBus, FileCapture=FileCapture, srvd_colour=srvd_colour, srvd_depth=srvd_depth,
                           calibration_file=calibration_file)
