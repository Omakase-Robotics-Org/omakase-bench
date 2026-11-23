"""Parts of the Isaac backend that run without Isaac Sim: the wire form and generated assets."""
import xml.etree.ElementTree as ET

import pytest

from omakase_bench.backends.isaac import server
from omakase_bench.core.suite import find_task
from omakase_bench.core.task import rgba, scene_from_dict, scene_to_dict


@pytest.mark.parametrize("task_id", ["core/cube_to_tray", "core/press_button", "drafts/elevator_call"])
def test_scene_round_trip(task_id):
    scene = find_task(task_id).resolve(5).scene
    assert scene_from_dict(scene_to_dict(scene)) == scene


@pytest.mark.parametrize("task_id", ["core/press_button", "drafts/elevator_call"])
def test_articulated_urdfs_parse(task_id):
    for a in find_task(task_id).scene.articulated:
        urdf, actuators, moving = server._articulated_urdf(a, rgba)
        root = ET.fromstring(urdf)
        links = {l.get("name") for l in root.iter("link")}
        assert set(moving) <= links and actuators


def test_mimic_map_from_preset_urdf():
    from omakase_bench.robots.spec import load_robot
    mimics = server._mimics(load_robot("d1").urdf_path.read_text())
    assert mimics["left_ee_jaw_r_finger_joint"][0] == "left_ee_jaw_r_joint"
    assert len(mimics) == 12


def test_site_settings_come_from_options_then_the_environment_and_name_no_machine(monkeypatch):
    from omakase_bench.backends.isaac.client import IsaacBackend
    for name in ("HOST", "REMOTE_DIR", "SETUP", "PYTHON"):
        monkeypatch.delenv(f"OMAKASE_BENCH_ISAAC_{name}", raising=False)
    b = IsaacBackend()
    assert (b.host, b.remote_dir, b.setup, b.python) == ("local", "~/omakase-bench", "true", "python")
    monkeypatch.setenv("OMAKASE_BENCH_ISAAC_HOST", "gpu-box")
    monkeypatch.setenv("OMAKASE_BENCH_ISAAC_SETUP", ". ~/env.sh")
    b = IsaacBackend(python="/opt/isaac/python")
    assert (b.host, b.setup, b.python) == ("gpu-box", ". ~/env.sh", "/opt/isaac/python")
    assert IsaacBackend(host="local").host == "local"
