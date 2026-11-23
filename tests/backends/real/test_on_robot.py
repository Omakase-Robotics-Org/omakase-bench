"""The real backend as it runs on the robot: the arm lease, the base, cameras from video devices and
shared-memory topics, depth with the calibration file. Everything on the robot is a stand-in: a loopback
daemon, image files as video devices, a queue as the shared-memory layer, a calibration file in tmp_path."""
import json
import math
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from PIL import Image

from omakase_bench.backends.real import srvd
from omakase_bench.backends.real.backend import RealBackend
from omakase_bench.backends.real.calibration import CalibrationError, Lens, load_calibration
from omakase_bench.core.backend import BackendUnsupported, make_backend
from omakase_bench.core.task import SceneSpec
from omakase_bench.harness import command_api, depth
from omakase_bench.harness.kinematics.model import Kinematics
from omakase_bench.robots.spec import load_robot

ROBOT = load_robot("d1")
SCENE = SceneSpec(robot="d1")
HEAD = ROBOT.cameras["head"]
#: the ideal lens of the description's head camera: what the simulator renders with
FOCAL = 0.5 * HEAD.height / math.tan(math.radians(HEAD.fovy) / 2)
PINHOLE = {"width": HEAD.width, "height": HEAD.height, "fx": FOCAL, "fy": FOCAL, "cx": HEAD.width / 2, "cy": HEAD.height / 2}


# --- lease ------------------------------------------------------------------------------------------------

def test_the_lease_is_taken_for_the_run_carried_by_arm_commands_renewed_and_handed_back(mock_robot, monkeypatch):
    backend = make_backend("real", url=mock_robot.url, enable="1", lease_ttl_s="3")
    backend.load(ROBOT, SCENE)
    assert mock_robot.lease["holder"] == "omakase-bench" and mock_robot.lease["class"] == "policy"
    assert backend.info().detail["lease"] == {"holder": "omakase-bench", "class": "policy", "epoch": 1}
    left = ROBOT.arms["left"].joints
    backend.command_joints({left[1]: 0.2})
    assert mock_robot.posts[-1][1]["holder"] == "omakase-bench"                 # without it the daemon refuses: see below
    backend.command_joints({"lift_joint": 0.2})
    assert "holder" not in mock_robot.posts[-1][1]                               # the lease arbitrates the arms only
    calls = len(mock_robot.lease_calls)
    backend.step(0.0)
    assert len(mock_robot.lease_calls) == calls                                  # well inside the TTL: no renewal yet
    backend._lease_at -= 1.5                                                     # a third of the TTL has passed
    backend.step(0.0)
    assert mock_robot.lease_calls[-1][0] == "POST" and mock_robot.epoch == 1     # a renewal keeps the grant
    backend.close()
    assert mock_robot.lease is None and mock_robot.lease_calls[-1] == ("DELETE", {"holder": "omakase-bench"})


def test_a_lease_another_client_holds_refuses_the_run_before_anything_moves(mock_robot):
    mock_robot.lease_call("POST", {"holder": "vr-teleop", "class": "operator", "ttl_s": 60})
    backend = make_backend("real", url=mock_robot.url, enable="1")
    with pytest.raises(BackendUnsupported, match=r"the arms are leased to 'vr-teleop' \(class operator, 60 s left\)"):
        backend.load(ROBOT, SCENE)
    assert mock_robot.posts == [] and mock_robot.lease["holder"] == "vr-teleop"  # nothing sent, their lease untouched
    # without a lease of its own (lease=0) the daemon itself refuses the arm command
    free = make_backend("real", url=mock_robot.url, enable="1", lease="0")
    free.load(ROBOT, SCENE)
    with pytest.raises(BackendUnsupported, match="409 lease_held"):
        free.command_joints({ROBOT.arms["left"].joints[1]: 0.2})
    free.close()
    assert mock_robot.lease["holder"] == "vr-teleop"


def test_a_lease_lost_between_renewals_stops_the_run(mock_robot):
    backend = make_backend("real", url=mock_robot.url, enable="1", lease_ttl_s="3")
    backend.load(ROBOT, SCENE)
    mock_robot.lease = None                    # it expired and somebody took and returned the arms meanwhile
    mock_robot.epoch += 1
    backend._lease_at -= 2.0
    with pytest.raises(BackendUnsupported, match="the arm lease was lost"):
        backend.step(0.0)
    backend.close()


def test_the_lease_is_kept_while_no_step_is_taken_and_its_loss_stops_the_run(mock_robot):
    """A model's thinking time passes without a step and lasts longer than the lease."""
    backend = make_backend("real", url=mock_robot.url, enable="1", lease_ttl_s="1")
    backend.load(ROBOT, SCENE)
    time.sleep(0.9)                                                               # no step, no command
    renewals = [body for method, body in mock_robot.lease_calls if method == "POST"][1:]
    assert len(renewals) >= 2 and mock_robot.epoch == 1 and mock_robot.lease["holder"] == "omakase-bench"
    backend.step(0.0)                                                             # still its own grant
    mock_robot.lease_call("POST", {"holder": "vr-teleop", "class": "operator", "ttl_s": 60})   # an operator takes the arms
    time.sleep(0.5)
    with pytest.raises(BackendUnsupported, match="the arm lease: POST /v1/arm/lease -> error 409 lease_held"):
        backend.step(0.0)
    backend.close()
    assert mock_robot.lease["holder"] == "vr-teleop"
    calls = len(mock_robot.lease_calls)
    time.sleep(0.5)
    assert len(mock_robot.lease_calls) == calls                                   # closed: no renewal follows


def test_loading_again_closes_what_the_load_before_it_opened(mock_robot, tmp_path, kit):
    """The runner loads once per episode."""
    opened = []

    def open_capture(device):
        opened.append(kit.FileCapture(_picture(tmp_path / "head.png", (200, 10, 10))))
        return opened[-1]

    backend = RealBackend(mock_robot.url, ws_url=mock_robot.ws_url, enable="1", camera_source="v4l2", depth="0",
                          open_capture=open_capture)
    backend.load(ROBOT, SCENE)
    first_stream = backend.stream
    backend.load(ROBOT, SCENE)
    assert len(opened) == 2 and opened[0].released and not opened[1].released
    assert backend.stream is not first_stream and first_stream._socket.close_code is not None
    assert [m for m, _ in mock_robot.lease_calls] == ["GET", "POST", "DELETE", "GET", "POST"]
    time.sleep(0.05)
    assert sum(t.name == "arm lease" for t in threading.enumerate()) == 1
    backend.close()


def test_the_lease_is_handed_back_when_setting_up_fails(mock_robot):
    def no_device(device):
        raise BackendUnsupported(f"video device {device} could not be opened")

    backend = RealBackend(mock_robot.url, enable="1", camera_source="v4l2", depth="0", open_capture=no_device)
    with pytest.raises(BackendUnsupported, match="video device /dev/video100 could not be opened"):
        backend.load(ROBOT, SCENE)
    assert mock_robot.lease is None and [m for m, _ in mock_robot.lease_calls] == ["GET", "POST", "DELETE"]


# --- base -------------------------------------------------------------------------------------------------

def test_base_targets_become_velocity_commands_and_the_base_is_stopped(mock_robot):
    backend = make_backend("real", url=mock_robot.url, enable="1")
    backend.load(ROBOT, SCENE)
    x = 0.0
    for _ in range(3):                                    # the harness's base path: a target per control period
        x += 0.02
        backend.command_joints({"base_x_joint": x})
        backend.step(0.0)
    assert mock_robot.velocities == [(0.2, 0.0)] * 3      # 20 mm in a 0.1 s period
    backend.step(0.0)                                     # a period without a new target: the velocity is not held
    assert mock_robot.velocities[-1] == (0.0, 0.0) and len(mock_robot.velocities) == 4
    backend.step(0.0)
    assert len(mock_robot.velocities) == 4
    backend.command_joints({"base_yaw_joint": 0.05})
    assert mock_robot.velocities[-1] == (0.0, 0.5)
    with pytest.raises(BackendUnsupported, match="to the side: the base is differential drive"):
        backend.command_joints({"base_y_joint": 0.05})
    backend.close()                                       # closing while a velocity is commanded stops the base
    assert mock_robot.velocities[-1] == (0.0, 0.0)
    assert all(path.startswith("/v1/") for path, _ in mock_robot.posts)      # the daemon's routes only


def test_a_base_motion_starts_from_where_the_base_is_and_stays_within_the_ceilings(mock_robot):
    """The odometry frame is not the scene's: the first target of a motion is far from the last one commanded."""
    mock_robot.odometry = {"x": 3.2, "y": 1.5, "theta": 0.7}
    backend = make_backend("real", url=mock_robot.url, enable="1")
    backend.load(ROBOT, SCENE)
    q = backend.joint_positions()
    names = ("base_x_joint", "base_y_joint", "base_yaw_joint")
    backend.command_joints({j: q[j] for j in names})              # the harness's first target: the measured pose
    assert mock_robot.velocities == [(0.0, 0.0)]
    backend.command_joints({"base_x_joint": 3.2 + 0.02 * math.cos(0.7), "base_y_joint": 1.5 + 0.02 * math.sin(0.7)})
    assert mock_robot.velocities[-1] == pytest.approx((0.2, 0.0), abs=1e-3)
    backend.command_joints({"base_x_joint": 3.2 + 1.0 * math.cos(0.7), "base_y_joint": 1.5 + 1.0 * math.sin(0.7)})
    assert mock_robot.velocities[-1] == (0.5, 0.0)                # a metre in one period is not asked for
    backend.close()


def test_a_velocity_command_without_an_answer_is_followed_by_a_stop(mock_robot):
    backend = make_backend("real", url=mock_robot.url, enable="1")
    backend.load(ROBOT, SCENE)
    answer = backend.client.request
    backend.client.request = lambda method, path, body=None: (
        command_api.error("unreachable", "no answer", 0) if path == "/v1/chassis/cmd_vel" and body["vx"] else answer(method, path, body))
    with pytest.raises(BackendUnsupported, match="unreachable"):
        backend.command_joints({"base_x_joint": 0.02})
    assert mock_robot.velocities == []
    backend.close()                                               # whether the base took the command is unknown
    assert mock_robot.velocities == [(0.0, 0.0)]


def test_a_step_ends_one_period_after_the_step_before_it(mock_robot):
    """A base velocity is the step per period: the time a period's requests take is not added to it."""
    backend = make_backend("real", url=mock_robot.url, enable="1", lease="0")
    backend.load(ROBOT, SCENE)
    start = time.monotonic()
    for _ in range(3):
        time.sleep(0.06)                                          # the period's commands and readings
        backend.step(0.1)
    assert 0.3 <= time.monotonic() - start < 0.4                  # three periods and the first one's work, not 0.48 s
    backend.close()


def test_without_the_in_house_base_the_pose_is_declared_commanded(mock_robot):
    mock_robot.amr = False
    backend = make_backend("real", url=mock_robot.url, enable="1")
    backend.load(ROBOT, SCENE)
    q = backend.joint_positions()
    assert q["base_x_joint"] == 0.0 and backend.driver.base_pose.startswith("commanded: the robot's state could not be read")
    backend.command_joints({"base_x_joint": 0.02})
    assert backend.joint_positions()["base_x_joint"] == 0.02                    # unmeasured: the base is where it was sent
    backend._read_at = -1e9
    backend.joint_positions()
    assert mock_robot.gets.count("/v1/chassis/amr/localization") == 1          # asked once, not every reading
    assert backend.info().detail["base_pose"].startswith("commanded: ")         # and the results say so
    backend.close()
    mock_robot.amr = True
    measured = make_backend("real", url=mock_robot.url, enable="1")
    measured.load(ROBOT, SCENE)
    measured.joint_positions()
    assert measured.info().detail["base_pose"] == "measured"
    measured.close()


def test_the_lease_class_is_policy_unless_an_option_names_another(mock_robot):
    assert ROBOT.real["lease"]["class"] == "policy"
    backend = make_backend("real", url=mock_robot.url, enable="1", lease_class="operator", lease_holder="bench-operator")
    backend.load(ROBOT, SCENE)
    assert (mock_robot.lease["class"], mock_robot.lease["holder"]) == ("operator", "bench-operator")
    backend.close()


# --- cameras on the robot -----------------------------------------------------------------------------------

def _picture(path, colour, size=(640, 480)):
    Image.fromarray(np.full((size[1], size[0], 3), colour, np.uint8)).save(path)
    return str(path)


def test_colour_from_video_devices_named_by_the_robot_description_or_an_option(mock_robot, tmp_path, kit):
    opened = {}

    def open_capture(device):
        opened[device] = kit.FileCapture(_picture(tmp_path / f"{device.strip('/').replace('/', '_')}.png",
                                              (200, 10, 10) if device == "/dev/video100" else (10, 200, 10)))
        return opened[device]

    backend = RealBackend(mock_robot.url, enable="1", camera_source="v4l2", depth="0", open_capture=open_capture)
    backend.load(ROBOT, SCENE)
    assert backend.cameras() == ("head",) and list(opened) == ["/dev/video100"]          # the description's on-robot default
    head = backend.render("head", 320, 240)
    assert head.shape == (240, 320, 3) and tuple(head[0, 0]) == (200, 10, 10)             # RGB, not the device's BGR
    with pytest.raises(BackendUnsupported, match="camera 'left_wrist' has no device; configured: head"):
        backend.render("left_wrist", 64, 48)
    backend.close()
    assert opened["/dev/video100"].released
    mapped = RealBackend(mock_robot.url, enable="1", camera_source="v4l2", depth="0", open_capture=open_capture,
                         **{"camera.left_wrist": "/dev/video104"})
    mapped.load(ROBOT, SCENE)
    assert sorted(mapped.cameras()) == ["head", "left_wrist"] and tuple(mapped.render("left_wrist", 64, 48)[0, 0]) == (10, 200, 10)
    assert mapped.info().detail["camera_source"] == "v4l2" and mapped.info().capabilities == frozenset()
    mapped.close()


def test_colour_from_shared_memory_topics(mock_robot, kit):
    bus = kit.FakeBus()
    for k, name in enumerate(("head", "left_wrist", "right_wrist")):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[..., k] = 220
        bus.publish(f"dxr/camera/{name}/raw", kit.srvd_colour(frame, name))
    backend = RealBackend(mock_robot.url, enable="1", camera_source="iceoryx", depth="0", topic_bus=bus)
    backend.load(ROBOT, SCENE)
    assert sorted(backend.cameras()) == ["head", "left_wrist", "right_wrist"]
    assert tuple(backend.render("left_wrist", 640, 480)[5, 5]) == (0, 220, 0)
    assert backend.render("head", 320, 240).shape == (240, 320, 3)
    bus.publish("dxr/camera/head/raw", b"not a frame")
    time.sleep(0.05)
    with pytest.raises(BackendUnsupported, match="camera 'head': topic dxr/camera/head/raw: .*header alone is 72"):
        backend.render("head", 64, 48)
    backend.close()
    assert bus.closed


def test_frames_decode_as_the_camera_daemon_publishes_them(kit):
    yuyv = np.tile(np.array([128, 128, 128, 128], np.uint8), (4, 4))            # mid grey, 8 x 4
    buffer = srvd.HEADER.pack(srvd.MAGIC, srvd.VERSION, 0, 0, 7, 0, b"cam", 8, 4, 4) + yuyv.tobytes()
    frame = srvd.decode_colour(buffer)
    assert frame.rgb.shape == (4, 8, 3) and frame.timestamp_ns == 7 and abs(int(frame.rgb[0, 0, 0]) - 130) <= 2
    metres = np.full((4, 6), 0.5, np.float32)
    decoded = srvd.decode_depth(kit.srvd_depth(metres, (300.0, 300.0, 3.0, 2.0)))
    assert decoded.metres == pytest.approx(metres) and decoded.intrinsics == (300.0, 300.0, 3.0, 2.0)
    with pytest.raises(srvd.FrameError, match="magic"):
        srvd.decode_colour(b"\0" * 100)
    with pytest.raises(srvd.FrameError, match="a depth frame is gray16"):
        srvd.decode_depth(buffer + b"\0" * 200)


# --- calibration and depth ------------------------------------------------------------------------------------

def test_the_calibration_file_gives_lenses_and_mounts_and_a_failed_gate_is_refused(tmp_path, kit):
    fish = {"width": 640, "height": 480, "fx": 300.0, "fy": 300.0, "cx": 320.0, "cy": 240.0, "model": "kannala_brandt",
            "coefficients": [0.02, -0.005, 0.001, 0.0]}
    brown = {**PINHOLE, "model": "brown_conrady", "coefficients": [0.08, -0.02, 0.001, -0.0005, 0.003],
             "mount": ("head_link", (0.05, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0))}
    cal = load_calibration(kit.calibration_file(tmp_path / "cal.json", {"head": brown, "left_wrist": fish}))
    assert cal.robot == "d1-test" and cal.mounts["head"].parent_link == "head_link" and "left_wrist" not in cal.mounts
    for lens in cal.lenses.values():
        u, v = np.meshgrid(np.linspace(60, 580, 5), np.linspace(60, 420, 4))
        back = lens.pixels(lens.rays(u, v) * 0.8)
        assert np.abs(back[0] - u).max() < 0.01 and np.abs(back[1] - v).max() < 0.01, lens.model
    assert Lens(**PINHOLE).rays(320.0, 240.0) == pytest.approx([0.0, 0.0, 1.0])
    failed = {**PINHOLE, "gate": {"verdict": "FAIL", "checks": [], "reasons": ["coverage too low"]}}
    with pytest.raises(CalibrationError, match=r"camera head intrinsics: its gate failed \(coverage too low\)"):
        load_calibration(kit.calibration_file(tmp_path / "bad.json", {"head": failed}))
    overridden = {**PINHOLE, "gate": {"verdict": "FAIL", "checks": [], "override": {"reason": "accepted"}}}
    assert load_calibration(kit.calibration_file(tmp_path / "ok.json", {"head": overridden})).lenses["head"].fx == FOCAL
    with pytest.raises(CalibrationError, match="no camera calibration file"):
        load_calibration(tmp_path / "none.json")
    (tmp_path / "v1.json").write_text(json.dumps({"schema": "omakase.camera_calibration/1"}))
    with pytest.raises(CalibrationError, match="expected 'omakase.camera_calibration/2'"):
        load_calibration(tmp_path / "v1.json")


def _on_robot(kit, mock_robot, tmp_path, cameras, bus, **options):
    path = kit.calibration_file(tmp_path / "camera_calibration.json", cameras)
    backend = RealBackend(mock_robot.url, enable="1", camera_source="iceoryx", topic_bus=bus, calibration=str(path), **options)
    backend.load(ROBOT, SCENE)
    return backend


def test_with_depth_topics_and_the_calibration_file_the_depth_lookup_works_through_the_harness(mock_robot, tmp_path, kit):
    bus = kit.FakeBus()
    bus.publish("dxr/camera/head/depth", kit.srvd_depth(np.full((480, 640), 0.6, np.float32), (FOCAL, FOCAL, 320.0, 240.0)))
    backend = _on_robot(kit, mock_robot, tmp_path, {"head": PINHOLE}, bus)
    assert backend.info().capabilities == {"depth_camera"} and backend.info().detail["depth"] == ["head"]
    image = backend.render_depth("head", HEAD.width, HEAD.height)
    assert image.shape == (480, 640) and float(np.median(image)) == pytest.approx(0.6, abs=1e-3)
    # the harness's own lookup: the same function a simulator's depth goes through
    kin = Kinematics(ROBOT)
    views = {"head": "head"}
    found = depth.lookup(backend, kin, ROBOT, views, "head", 200, 90, 320, 240)
    assert found.point_mm is not None, found.reason
    # what the simulator's path gives for the same depth: an ideal lens and the description's camera pose.
    # The calibration here states exactly those, so the two agree.
    kin.set_q(backend.joint_positions())
    pos, rot = depth.camera_pose(kin, ROBOT, "head")
    uf, vf = (200 + 0.5) * 2, (90 + 0.5) * 2
    expected = pos + rot @ (np.array([(uf - 320) / FOCAL, (vf - 240) / FOCAL, 1.0]) * 0.6)
    assert np.abs(np.array(found.point_mm) - expected * 1000).max() <= 2
    backend.close()


def test_a_calibrated_lens_and_mount_change_where_a_pixel_lands(mock_robot, tmp_path, kit):
    bus = kit.FakeBus()
    bus.publish("dxr/camera/head/depth", kit.srvd_depth(np.full((480, 640), 0.6, np.float32), (FOCAL, FOCAL, 320.0, 240.0)))
    kin = Kinematics(ROBOT)
    kin.set_q(ROBOT.default_joint_positions())
    link = kin.data.body("head_link")
    cam = kin.data.body(HEAD.link)
    # the nominal mount, moved 20 mm along the head link's x axis
    rot_parent = link.xmat.reshape(3, 3)
    offset = rot_parent.T @ (cam.xpos - link.xpos) + np.array([0.02, 0.0, 0.0])
    turn = rot_parent.T @ cam.xmat.reshape(3, 3)
    w = math.sqrt(max(0.0, 1 + np.trace(turn))) / 2
    quat = ((turn[2, 1] - turn[1, 2]) / (4 * w), (turn[0, 2] - turn[2, 0]) / (4 * w), (turn[1, 0] - turn[0, 1]) / (4 * w), w)
    lens = {**PINHOLE, "model": "inverse_brown_conrady", "coefficients": [0.1, 0.0, 0.0, 0.0, 0.0],
            "mount": ("head_link", tuple(offset), quat)}
    backend = _on_robot(kit, mock_robot, tmp_path, {"head": lens}, bus)
    mock_robot.arms = {s: [math.degrees(v) for v in ROBOT.arms[a].home] for s, a in (("a", "left"), ("b", "right"))}
    mock_robot.height, mock_robot.pitch = ROBOT.defaults["lift"], -ROBOT.defaults["neck"]["tilt"]
    centre = depth.lookup(backend, kin, ROBOT, {"head": "head"}, "head", 160, 120, 320, 240)
    corner = depth.lookup(backend, kin, ROBOT, {"head": "head"}, "head", 20, 20, 320, 240)
    kin.set_q(backend.joint_positions())
    pos, rot = depth.camera_pose(kin, ROBOT, "head")
    nominal_centre = pos + rot @ np.array([0.0, 0.0, 0.6])
    shift = (np.array(centre.point_mm) / 1000 - nominal_centre)
    assert np.linalg.norm(shift) == pytest.approx(0.02, abs=0.004)                      # the mount's 20 mm
    # off the axis the distortion bends the ray away from the ideal lens's
    ideal = np.array([(41 - 320) / FOCAL, (41 - 240) / FOCAL, 1.0])
    assert np.abs(backend.pixel_ray("head", 41, 41, 640, 480) - ideal).max() > 0.01 and corner.point_mm is not None
    backend.close()


def test_the_depth_imager_sits_at_the_colour_camera_unless_its_offset_is_configured(mock_robot, tmp_path, kit):
    assert ROBOT.real["depth_to_colour"]["head"]["xyz_m"] == [0.0, 0.0, 0.0]            # the D1 has no measured value
    metres = np.zeros((480, 640), np.float32)
    metres[:, 300:340] = 0.6                                                             # a strip 0.6 m away

    def columns(**options):
        bus = kit.FakeBus()
        bus.publish("dxr/camera/head/depth", kit.srvd_depth(metres, (FOCAL, FOCAL, 320.0, 240.0)))
        backend = _on_robot(kit, mock_robot, tmp_path, {"head": PINHOLE}, bus, **options)
        seen = np.nonzero(backend.render_depth("head", HEAD.width, HEAD.height)[240] > 0)[0]
        backend.close()
        return int(seen.min()), int(seen.max())

    assert columns() == (300, 339)                                                       # zero: one frame
    shift = FOCAL * 0.05 / 0.6                                                           # 50 mm to the right, seen at 0.6 m
    lo, hi = columns(**{"depth_to_colour.head": "0.05,0,0"})
    assert lo == pytest.approx(300 + shift, abs=1) and hi == pytest.approx(339 + shift, abs=1)
    with pytest.raises(BackendUnsupported, match="depth_to_colour"):
        columns(**{"depth_to_colour.head": "0.05,0"})
    with pytest.raises(ValueError, match="has no camera"):
        columns(**{"depth_to_colour.chest": "0,0,0"})


def test_depth_already_on_the_colour_grid_is_used_as_it_is(mock_robot, tmp_path, kit):
    """The camera daemon can render depth into the colour camera's view: the frame then carries the colour
    camera's pinhole, and neither a calibrated lens nor an offset applies."""
    assert ROBOT.real["depth_frame"] == {"head": "depth"}                               # the D1's default topic is not aligned
    metres = np.zeros((240, 320), np.float32)
    metres[:, 150:170] = 0.6                                                             # a strip, on a 320 x 240 colour grid
    printed = []
    robot = load_robot("d1")
    object.__setattr__(robot, "real", {**robot.real, "calibration": str(tmp_path / "absent.json")})

    def aligned(**options):
        bus = kit.FakeBus()
        bus.publish("dxr/camera/head/depth_color", kit.srvd_depth(metres, (300.0, 300.0, 160.0, 120.0)))
        backend = RealBackend(mock_robot.url, enable="1", camera_source="iceoryx", topic_bus=bus, echo=printed.append,
                              **{"depth.head": "dxr/camera/head/depth_color", "depth_frame.head": "color", **options})
        backend.load(robot, SCENE)
        return backend

    backend = aligned()                                                                  # no calibration file at all
    assert backend.info().capabilities == {"depth_camera"} and backend.info().detail["depth_frame"] == {"head": "color"}
    image = backend.render_depth("head", HEAD.width, HEAD.height)
    seen = np.nonzero(image[240] > 0)[0]
    assert image.shape == (480, 640) and (int(seen.min()), int(seen.max())) == (300, 339)       # the same pixels, scaled
    ray = backend.pixel_ray("head", 384.0, 240.0, 640, 480)                              # the frame's own intrinsics
    assert ray == pytest.approx([(192.0 - 160.0) / 300.0, 0.0, 1.0])
    assert backend.camera_mount("head") is None                                          # the description's nominal pose
    found = depth.lookup(backend, Kinematics(ROBOT), ROBOT, {"head": "head"}, "head", 160, 120, 320, 240)
    assert found.point_mm is not None and printed == []
    backend.close()

    shifted = aligned(**{"depth_to_colour.head": "0.05,0,0"})                            # an offset has nothing to act on
    seen = np.nonzero(shifted.render_depth("head", HEAD.width, HEAD.height)[240] > 0)[0]
    assert (int(seen.min()), int(seen.max())) == (300, 339)
    assert any("depth_to_colour.head is ignored" in line for line in printed)
    shifted.close()
    with pytest.raises(ValueError, match="depth_frame head=colour; a depth topic is on the grid of: depth, color"):
        aligned(**{"depth_frame.head": "colour"})
    # the default mode still needs the camera's lens from the calibration file
    with pytest.raises(BackendUnsupported, match="depth needs the camera calibration"):
        aligned(**{"depth_frame.head": "depth"})


def test_depth_is_declared_missing_where_its_inputs_are(mock_robot, tmp_path, kit):
    bus = kit.FakeBus()
    # no calibration file at the description's default path: no depth, said so, not an error
    plain = RealBackend(mock_robot.url, enable="1", camera_source="iceoryx", topic_bus=bus, calibration=None)
    ROBOT.real["calibration"]                                                          # the default exists in the description
    robot = load_robot("d1")
    object.__setattr__(robot, "real", {**robot.real, "calibration": str(tmp_path / "absent.json")})
    plain.load(robot, SCENE)
    assert plain.info().capabilities == frozenset() and not hasattr(plain, "render_depth")
    assert plain.info().detail["depth_note"].startswith("off: no camera calibration file at")
    plain.close()
    # asked for by option without a usable calibration: an error that names it
    with pytest.raises(BackendUnsupported, match="depth needs the camera calibration: no camera calibration file"):
        RealBackend(mock_robot.url, enable="1", camera_source="iceoryx", topic_bus=kit.FakeBus(),
                    calibration=str(tmp_path / "absent.json")).load(ROBOT, SCENE)
    # a depth topic without frames: the lookup names the topic and the usual causes
    silent = _on_robot(kit, mock_robot, tmp_path, {"head": PINHOLE}, kit.FakeBus())
    found = None
    with pytest.raises(BackendUnsupported, match="no frame on topic dxr/camera/head/depth"):
        found = silent.depths._readers["head"].newest(wait_s=0.05)
    assert found is None
    silent.close()
    # the http source carries no depth at all
    remote = make_backend("real", url=mock_robot.url, enable="1")
    remote.load(ROBOT, SCENE)
    assert remote.info().detail["depth_note"] == "no depth over the http camera source" and not hasattr(remote, "render_depth")
    remote.close()
    with pytest.raises(ValueError, match="camera_source 'usb'; sources: v4l2, iceoryx, http"):
        RealBackend(camera_source="usb")


def test_the_shared_memory_binding_is_asked_for_only_when_topics_are_read(mock_robot):
    backend = RealBackend(mock_robot.url, enable="1", camera_source="iceoryx", depth="0")
    with pytest.raises(BackendUnsupported, match="need the iceoryx2 Python binding at the camera daemon's version"):
        backend.load(ROBOT, SCENE)
    assert mock_robot.lease is None                                                    # and the lease went back


# --- no renderer on the robot ---------------------------------------------------------------------------------

def test_kinematics_and_ik_need_no_renderer():
    """On the robot the bench computes FK and IK with MuJoCo but never renders: with no GL library to load,
    kinematics, the solver and the real backend's dry run still work."""
    code = (
        "import os, sys\n"
        "os.environ['MUJOCO_GL'] = 'egl'\n"
        "os.environ['PYOPENGL_PLATFORM'] = 'nonexistent'\n"
        "import mujoco\n"
        "def no_renderer(*a, **k): raise RuntimeError('a renderer was created')\n"
        "mujoco.Renderer = no_renderer\n"
        "from omakase_bench.robots.spec import load_robot\n"
        "from omakase_bench.harness.kinematics.model import Kinematics\n"
        "from omakase_bench.backends.real.backend import RealBackend\n"
        "from omakase_bench.core.task import SceneSpec\n"
        "robot = load_robot('d1'); kin = Kinematics(robot); q = robot.default_joint_positions(); kin.set_q(q)\n"
        "p, _ = kin.tcp_pose('left')\n"
        "assert kin.solve_ik('left', p + (0.03, 0, 0.02), kin.named_rotation('left', 'down', 0.0), q).ok\n"
        "b = RealBackend(dry_run=True, echo=lambda s: None); b.load(robot, SceneSpec(robot='d1')); b.command_joints({'lift_joint': 0.2})\n"
        "assert 'OpenGL' not in sys.modules and 'glfw' not in sys.modules\n"
        "print('ok')\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-1500:]
