"""The real backend: disabled unless enabled twice, no default host, measured state, cameras and streamed
motion, tested only against the stand-in robot on the loopback interface or with dry_run (no socket)."""
import json
import math

import numpy as np
import pytest

from omakase_bench.backends.real import http, ws
from omakase_bench.backends.real.backend import RealBackend
from omakase_bench.core.backend import (CAPABILITIES, SIMULATOR, Backend, BackendUnsupported, GroundTruth, SceneReset,
                                        check_supported, make_backend, required_capabilities)
from omakase_bench.core.runner import RunConfig, run
from omakase_bench.core.suite import find_task
from omakase_bench.core.task import SceneSpec, load_task
from omakase_bench.harness import command_api
from omakase_bench.harness.session import RunBudget
from omakase_bench.models.base import Capabilities, Image, Reply, Text, Usage
from omakase_bench.robots.spec import load_robot
from omakase_bench.subjects import build_subject
from omakase_bench.subjects.ee_agent import EEAgent

ROBOT = load_robot("d1")
SCENE = SceneSpec(robot="d1")
STATE_PATHS = ["/v1/arm/feedback", "/v1/end_effectors", "/v1/slider/state", "/v1/neck/state"]


def test_it_is_a_backend_with_none_of_a_simulators_capabilities():
    real, sim = make_backend("real", dry_run=True), make_backend("mujoco")
    assert isinstance(real, Backend) and isinstance(sim, Backend)
    assert not isinstance(real, SceneReset) and not isinstance(real, GroundTruth) and not hasattr(real, "render_depth")
    assert isinstance(sim, SceneReset) and isinstance(sim, GroundTruth)
    assert real.info().capabilities == frozenset() and SIMULATOR <= sim.info().capabilities
    assert all(isinstance(sim, CAPABILITIES[c]) for c in sim.info().capabilities if c in CAPABILITIES)


def test_it_stays_disabled_without_the_option_and_the_environment_and_has_no_default_host(monkeypatch):
    for name in (http.ENABLE_ENV, http.URL_ENV, ws.WS_URL_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PermissionError):
        RealBackend("http://127.0.0.1:1").load(ROBOT, SCENE)
    with pytest.raises(PermissionError):
        RealBackend("http://127.0.0.1:1", enable=True).load(ROBOT, SCENE)          # the environment says no
    monkeypatch.setenv(http.ENABLE_ENV, "1")
    with pytest.raises(PermissionError):
        RealBackend("http://127.0.0.1:1").load(ROBOT, SCENE)                       # the option says no
    with pytest.raises(ValueError, match="no robot URL"):
        RealBackend(enable="1").load(ROBOT, SCENE)
    with pytest.raises(ValueError, match="expected an http\\(s\\) URL"):
        RealBackend("http://127.0.0.1:1", enable="1", **{"camera.head": "head"}).load(ROBOT, SCENE)
    with pytest.raises(ValueError, match="has no camera"):
        RealBackend(dry_run=True, **{"camera.chest": "http://127.0.0.1:1/x"}).load(ROBOT, SCENE)
    with pytest.raises(ValueError, match="unknown option"):
        RealBackend(host="127.0.0.1")
    with pytest.raises(BackendUnsupported, match="names no driver"):
        RealBackend(dry_run=True).load(load_robot("arm6"), SceneSpec(robot="arm6"))


def test_a_dry_run_prints_the_requests_and_opens_no_socket():
    printed = []
    backend = RealBackend(dry_run=True, echo=printed.append)
    backend.load(ROBOT, SCENE)
    q = backend.joint_positions()
    backend.command_joints({"lift_joint": 0.2})
    assert printed == ['POST <robot>/v1/slider/set_height {"height_m": 0.2, "wait": false}']
    assert backend.joint_positions() == {**q, "lift_joint": 0.2} and backend.info().detail["joint_state"] == "commanded"
    assert backend.cameras() == ()
    with pytest.raises(BackendUnsupported, match="a dry run has no cameras"):
        backend.render("head", 64, 48)


def test_the_stand_in_robot_answers_in_the_format_of_the_api_document(mock_robot):
    for path in STATE_PATHS:
        op, _ = command_api.operation(command_api.load_spec(str(ROBOT.preset_dir / ROBOT.api["spec"])), "GET", path)
        spec = command_api.load_spec(str(ROBOT.preset_dir / ROBOT.api["spec"]))
        schema = command_api._resolve(spec, op["responses"]["200"])["content"]["application/json"]["schema"]
        assert command_api.check(spec, schema, mock_robot.read(path), path) == [], path


def test_joint_state_is_measured_from_the_robots_state_operations(mock_robot):
    backend = make_backend("real", url=mock_robot.url, enable="1")
    backend.load(ROBOT, SCENE)
    mock_robot.arms["a"] = [10.0, -20.0, 0.0, -90.0, 0.0, 30.0, 0.0]
    mock_robot.openness["b"], mock_robot.height, mock_robot.pitch, mock_robot.yaw = 0.25, 0.17, 0.3, -0.2
    mock_robot.odometry = {"x": 0.4, "y": -0.1, "theta": 0.3}
    q = backend.joint_positions()
    assert mock_robot.gets == STATE_PATHS[:2] + ["/v1/neck/state", "/v1/slider/state", "/v1/chassis/amr/localization"]
    left = ROBOT.arms["left"].joints
    assert [round(math.degrees(q[j]), 3) for j in left] == [10.5, -20.0, 0.0, -90.0, 0.0, 30.0, 0.0]   # the reading, offset included
    assert q["lift_joint"] == 0.17 and q["neck_tilt_joint"] == -0.3 and q["neck_pan_joint"] == -0.2
    finger = ROBOT.hand_of("right").fingers[0]
    assert q[finger.joint] == pytest.approx(finger.open + 0.75 * (finger.closed - finger.open))
    assert (q["base_x_joint"], q["base_y_joint"], q["base_yaw_joint"]) == (0.4, -0.1, 0.3)     # the base's odometry pose
    assert backend.info().detail["joint_state"] == "measured"
    # a reading is reused for a moment; the lift, a slow serial read, for longer
    backend.joint_positions()
    assert len(mock_robot.gets) == 5
    backend.command_joints({"neck_pan_joint": 0.1})
    backend.joint_positions()
    assert mock_robot.gets[5:] == STATE_PATHS[:2] + ["/v1/neck/state", "/v1/chassis/amr/localization"]
    backend.close()


def test_joint_targets_become_requests_over_http_and_a_refusal_is_an_error(mock_robot):
    backend = make_backend("real", url=mock_robot.url, enable="1")
    backend.load(ROBOT, SCENE)
    backend.command_joints({j: backend._target[j] for j in ROBOT.arms["left"].joints})          # nothing changed: nothing sent
    assert mock_robot.posts == []
    left = ROBOT.arms["left"].joints
    backend.command_joints({left[1]: 0.2, "lift_joint": 0.2, "neck_tilt_joint": 0.3})
    finger = ROBOT.hand_of("right").fingers[0]
    backend.command_joints({finger.joint: finger.closed})
    assert [p for p, _ in mock_robot.posts] == ["/v1/arm/a/move_joints", "/v1/slider/set_height", "/v1/neck/cmd",
                                                "/v1/end_effectors/b/set"]
    assert len(mock_robot.posts[0][1]["joints_deg"]) == 7 and mock_robot.posts[0][1]["wait"] is False
    assert mock_robot.posts[1][1] == {"height_m": 0.2, "wait": False} and mock_robot.posts[2][1]["pitch"] == -0.3
    assert mock_robot.posts[3][1] == {"openness": 0.0, "wait": False}
    assert backend.info().detail["motion"] == "http"
    before = backend.joint_positions()[left[0]]
    with pytest.raises(BackendUnsupported, match="refused a command: POST /v1/arm/a/move_joints -> error 409 refused"):
        backend.command_joints({left[0]: math.radians(175.0)})
    assert backend.joint_positions()[left[0]] == before and backend._target[left[0]] != math.radians(175.0)
    backend.step(0.01)
    assert backend.sim_time() == pytest.approx(0.01)
    backend.close()


def test_motion_is_streamed_over_websocket_when_a_websocket_url_is_given(mock_robot):
    backend = make_backend("real", url=mock_robot.url, ws_url=mock_robot.ws_url, enable="1")
    backend.load(ROBOT, SCENE)
    assert backend.info().detail["motion"] == "websocket stream"
    left, right = ROBOT.arms["left"].joints, ROBOT.arms["right"].joints
    for k in range(1, 4):
        backend.command_joints({left[1]: 0.1 * k, right[1]: 0.05 * k})
    assert [p for p, _ in mock_robot.posts] == ["/v1/arm/move_joints_both"] * 3      # both arms in one streamed request
    last = mock_robot.posts[-1][1]
    assert last["wait"] is False and last["a"][1] == pytest.approx(math.degrees(0.3), abs=1e-3)
    assert last["b"][1] == pytest.approx(math.degrees(0.15), abs=1e-3)
    assert backend.stream.method("POST", "/v1/arm/b/move_joints") == "arm.b.move_joints"
    q = backend.joint_positions()                                # read back over HTTP: what the robot measured
    assert math.degrees(q[left[1]]) == pytest.approx(math.degrees(0.3), abs=1e-3)
    assert math.degrees(q[left[0]]) == pytest.approx(math.degrees(backend._target[left[0]]) + 0.5, abs=1e-3)
    backend.close()


def test_cameras_deliver_the_robots_frames_at_the_size_asked_for(mock_robot):
    backend = make_backend("real", url=mock_robot.url, enable="1", **mock_robot.cameras)
    backend.load(ROBOT, SCENE)
    assert sorted(backend.cameras()) == ["head", "left_wrist", "right_wrist"]
    head = backend.render("head", 320, 240)
    assert head.shape == (240, 320, 3) and head.dtype == np.uint8
    assert head[..., 0].mean() > 100 and head[..., 1].mean() < 10            # the head camera's frame is red
    assert backend.render("left_wrist", 640, 480)[..., 1].mean() > 100
    assert mock_robot.frames == ["head", "left_wrist"]
    partial = make_backend("real", url=mock_robot.url, enable="1", **{"camera.head": mock_robot.cameras["camera.head"]})
    partial.load(ROBOT, SCENE)
    with pytest.raises(BackendUnsupported, match="camera 'left_wrist' has no source; configured: head"):
        partial.render("left_wrist", 64, 48)
    broken = make_backend("real", url=mock_robot.url, enable="1", **{"camera.head": mock_robot.url + "/v1/neck/state"})
    broken.load(ROBOT, SCENE)
    with pytest.raises(BackendUnsupported, match="expected a JPEG image, got application/json"):
        broken.render("head", 64, 48)


def test_a_task_that_needs_ground_truth_is_refused_on_the_real_backend(tmp_path):
    task = find_task("core/cube_to_tray")
    assert required_capabilities(task) == {"reset", "ground_truth"}
    assert required_capabilities(find_task("games/chess")) == {"reset", "ground_truth", "multi_robot", "contacts", "object_pose"}
    assert required_capabilities(find_task("team/block_relay")) == {"reset", "ground_truth", "multi_robot"}
    cfg = RunConfig(out_dir=tmp_path, backend="real", backend_options={"dry_run": "1", "echo": lambda s: None},
                    video=False, run_budget=RunBudget(None, None))
    with pytest.raises(BackendUnsupported, match="task core/cube_to_tray needs ground_truth, reset, which backend real "
                                                 "does not provide"):
        run([build_subject("scripted/oracle")], [task], [0], cfg, progress=lambda s: None)
    check_supported(task, make_backend("mujoco").info())


class Rows:
    """A stand-in model for the ee interface: keeps what it was shown and replies with the given rows."""
    capabilities = Capabilities(images=True)
    name = "test/rows"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def respond(self, request) -> Reply:
        self.seen.append(request)
        text = self.replies.pop(0) if self.replies else "a: L hold | R hold"
        return Reply(text=text, stop="end", usage=Usage(input_tokens=10, output_tokens=5, usd=0.0, seconds=0.0), model=self.name)


def _hold_task(tmp_path, mock_robot, requires=()):
    """An ee task without a scene to build, and the stand-in's arms at the pose it starts from."""
    raw = dict(find_task("manip/chess_move").raw)
    raw.update(id="demo/hold", requires=list(requires), interfaces=["ee"], budget={"sim_seconds": 60})
    raw["ee"] = {**raw["ee"], "horizon": 2, "start": {}}
    for key in ("generator", "params", "gripper"):
        raw.pop(key, None)
    raw["instruction"], raw["success"], raw["subgoals"] = "Hold still.", {"all": []}, []
    path = tmp_path / f"hold_{len(requires)}.yaml"
    path.write_text(json.dumps(raw))                                              # JSON is YAML
    task = load_task(path)
    home = {j: math.degrees(v) for a in ROBOT.arms.values() for j, v in zip(a.joints, ROBOT.with_home(task.scene.home_pose).arms[a.name].home)}
    mock_robot.arms = {"a": [home[j] for j in ROBOT.arms["left"].joints], "b": [home[j] for j in ROBOT.arms["right"].joints]}
    return task, raw


def test_the_ee_interface_gets_the_same_observations_from_the_robot_as_from_the_simulator(tmp_path, mock_robot):
    """Images from the robot's cameras, the observation row from measured joints, no depth lookup."""
    task, raw = _hold_task(tmp_path, mock_robot)
    model = Rows([])
    cfg = RunConfig(out_dir=tmp_path / "run", backend="real", video=False, run_budget=RunBudget(None, None),
                    backend_options={"url": mock_robot.url, "ws_url": mock_robot.ws_url, "enable": "1", **mock_robot.cameras})
    episode = run([EEAgent(model, "test/rows")], [task], [0], cfg, progress=lambda s: None)["episodes"][0]
    assert episode["exit_reason"] not in ("error:subject",), episode["exit_detail"]
    assert episode["evidence"] == "not judged: the backend has no ground truth" and len(model.seen) == 2
    first = model.seen[0].messages[0].parts
    images = [p for p in first if isinstance(p, Image)]
    text = " ".join(p.text for p in first if isinstance(p, Text))
    assert len(images) == 1 and sorted(mock_robot.frames[:3]) == ["head", "left_wrist", "right_wrist"]
    assert "locate:" not in text and "t00" in text                # the depth lookup is not offered; the row is
    trace = (tmp_path / "run" / episode["artifacts"]["trace"]).read_text()
    assert "ee_lookups_off" in trace
    # the same task in the simulator offers the lookup: the difference is the declared capability, not the code
    sim_model = Rows([])
    sim_cfg = RunConfig(out_dir=tmp_path / "sim", video=False, run_budget=RunBudget(None, None))
    sim_raw = {**raw, "requires": ["reset"]}
    sim_path = tmp_path / "hold_sim.yaml"
    sim_path.write_text(json.dumps(sim_raw))
    run([EEAgent(sim_model, "test/rows")], [load_task(sim_path)], [0], sim_cfg, progress=lambda s: None)
    sim_text = " ".join(p.text for p in sim_model.seen[0].messages[0].parts if isinstance(p, Text))
    assert "locate:" in sim_text


def _prompt(model):
    return " ".join(p.text for p in model.seen[0].messages[0].parts if isinstance(p, Text))


def test_the_depth_lookup_is_offered_on_the_robot_only_for_cameras_with_a_depth_topic(tmp_path, mock_robot, kit):
    """Depth is the head camera's: the prompt offers the lookup for that view alone, a lookup on it is answered
    through the calibration, and without a depth topic the lookup is left out of the prompt."""
    head = ROBOT.cameras["head"]
    focal = 0.5 * head.height / math.tan(math.radians(head.fovy) / 2)
    lens = {"width": head.width, "height": head.height, "fx": focal, "fy": focal, "cx": head.width / 2, "cy": head.height / 2}
    calibration = kit.calibration_file(tmp_path / "camera_calibration.json", {"head": lens})
    needs_depth, plain = _hold_task(tmp_path, mock_robot, requires=["depth_camera"])[0], _hold_task(tmp_path, mock_robot)[0]

    def episode(out, task, replies, **options):
        bus = kit.LiveBus()
        for name in ROBOT.cameras:
            bus.publish(f"dxr/camera/{name}/raw", kit.srvd_colour(np.full((480, 640, 3), 90, np.uint8), name))
        bus.publish("dxr/camera/head/depth", kit.srvd_depth(np.full((480, 640), 0.6, np.float32), (focal, focal, 320.0, 240.0)))
        model = Rows(replies)
        cfg = RunConfig(out_dir=tmp_path / out, backend="real", video=False, run_budget=RunBudget(None, None),
                        backend_options={"url": mock_robot.url, "enable": "1", "camera_source": "iceoryx", "topic_bus": bus,
                                         "calibration": str(calibration), **options})
        results = run([EEAgent(model, "test/rows")], [task], [0], cfg, progress=lambda s: None)
        trace = [json.loads(line) for line in (tmp_path / out / results["episodes"][0]["artifacts"]["trace"]).read_text().splitlines()]
        return model, results, trace

    model, results, trace = episode("with", needs_depth, ["locate: head 160 120\nlocate: left_wrist 10 10\na: L hold | R hold"])
    assert results["episodes"][0]["exit_reason"] != "error:subject", results["episodes"][0]["exit_detail"]
    text = _prompt(model)
    assert "locate: <view> <u> <v>' (view = head;" in text and "left_wrist:" not in text.split("Depth lookup:")[1].split("Lookups take")[0]
    lookups = [e for e in trace if e["kind"] == "lookup"]
    assert [e["view"] for e in lookups] == ["head"] and lookups[0]["point_mm"] is not None     # the wrist has no depth
    assert results["provenance"]["backend"]["detail"]["depth"] == ["head"] and not any(e["kind"] == "ee_lookups_off" for e in trace)

    model, results, trace = episode("without", plain, [], depth="0")
    assert "locate:" not in _prompt(model) and any(e["kind"] == "ee_lookups_off" for e in trace)
    assert results["provenance"]["backend"]["detail"]["depth_note"] == "switched off (depth=0)"
    # a task that needs depth is refused where there is none, before anything is sent
    with pytest.raises(BackendUnsupported, match="task demo/hold needs depth_camera, which backend real does not provide"):
        episode("refused", needs_depth, [], depth="0")


def test_a_dropped_connection_or_a_foreign_answer_is_an_error_envelope_not_an_exception(mock_robot):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    backend = make_backend("real", url=mock_robot.url, ws_url=mock_robot.ws_url, enable="1")
    backend.load(ROBOT, SCENE)
    backend.stream._socket.close()
    with pytest.raises(BackendUnsupported, match="refused a command: POST /v1/arm/move_joints_both -> error 0 unreachable"):
        backend.command_joints({ROBOT.arms["left"].joints[1]: 0.2})
    backend.close()

    class Page(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>not the robot</html>")

        def log_message(self, *args):
            pass

    other = HTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        client = http.HttpDaemonClient(command_api.load_spec(str(ROBOT.preset_dir / ROBOT.api["spec"])),
                                       f"http://127.0.0.1:{other.server_port}", enable=True)
        answer = client.request("GET", "/v1/neck/state")
        assert answer["status"] == "error" and answer["kind"] == "device" and "not the API's JSON envelope" in answer["message"]
    finally:
        other.shutdown()
        other.server_close()


def test_the_robot_api_works_without_ground_truth(tmp_path, mock_robot):
    """list_objects is refused and the base drives: neither asks the backend for an oracle it does not have."""
    from omakase_bench.harness.api import Refused
    from omakase_bench.harness.session import Session
    task, _ = _hold_task(tmp_path, mock_robot)
    mock_robot.odometry = {"x": 1.0, "y": 2.0, "theta": 0.5}
    backend = make_backend("real", url=mock_robot.url, enable="1")
    episode = task.resolve(0)
    backend.load(ROBOT, episode.scene)
    try:
        session = Session(episode, backend, ROBOT, run_budget=RunBudget(None, None))
        with pytest.raises(Refused, match="this backend has no perception"):
            session.api.list_objects()
        session.api.move_base(forward_m=0.05)
        moving = [v for v in mock_robot.velocities if v != (0.0, 0.0)]
        assert moving and all(v == pytest.approx((0.25, 0.0), abs=2e-3) for v in moving)       # 50 mm in 0.2 s
        assert mock_robot.velocities[0] == (0.0, 0.0) and mock_robot.velocities[-1] == (0.0, 0.0)
    finally:
        backend.close()
