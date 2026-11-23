"""robot.yaml: fields that used to be constants in code, and their defaults."""
import shutil

import pytest

from omakase_bench.robots.spec import PRESETS_DIR, Actuation, load_robot


def _variant(tmp_path, edit):
    d = tmp_path / "robot"
    shutil.copytree(PRESETS_DIR / "arm6", d)
    (d / "robot.yaml").write_text(edit((d / "robot.yaml").read_text()))
    return d


def test_the_d1_states_what_the_code_used_to_assume():
    d1 = load_robot("d1")
    assert d1.actuation == Actuation() and d1.exempt_geoms == "*_exempt" and d1.ik_solver == "dls"
    assert {n: (a.side, a.self_collision_from) for n, a in d1.arms.items()} == {
        "left": ("left", "left_arm_joint3"), "right": ("right", "right_arm_joint3")}
    assert d1.hand_of("left").tip_reach == 0.011 and (d1.arm_for(0.1), d1.arm_for(-0.1)) == ("left", "right")
    assert d1.api["adapter"] == "omakase_bench.backends.real.d1.api:adapter"


def test_defaults_of_a_description_that_states_none(tmp_path):
    d = _variant(tmp_path, lambda t: t.split("actuation:")[0].replace("    self_collision_from: joint3\n", "")
                 .replace("    tip_reach: 0.010\n", ""))
    robot = load_robot(d)
    assert robot.actuation == Actuation() and robot.arms["arm"].self_collision_from == "joint3"
    assert robot.hand_of("arm").tip_reach == 0.010          # fingertip_offset.open
    assert robot.api == {}


@pytest.mark.parametrize("old, new, message", [
    ("    self_collision_from: joint3", "    self_collision_from: joint9", "is not a joint of the arm"),
    ("    max_joint_speed: 1.0 ", "    side: front\n    max_joint_speed: 1.0 ", "side must be one of"),
    ("  arm_proximal_joints: 3", "  arm_proximal_joints: 3\n  wrist: [1, 2]", "actuation has unknown field"),
])
def test_a_wrong_field_is_named(tmp_path, old, new, message):
    d = _variant(tmp_path, lambda t: t.replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_robot(d)
