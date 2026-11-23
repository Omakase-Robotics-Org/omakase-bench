import math

import pytest

from omakase_bench.core.backend import ArticulatedState, ObjectState, OracleSnapshot
from omakase_bench.core.predicates import PredicateError, evaluate, validate
from omakase_bench.core.task import Region


def snap(cube=(0.42, -0.02, 0.775), held=None, contacts_ever=(), contacts=(), pressed=False, base=(0, 0, 0),
         vel=(0, 0, 0), tray_yaw=0.0):
    q = (math.cos(tray_yaw / 2), 0, 0, math.sin(tray_yaw / 2))
    objects = {
        "cube": ObjectState("cube", "object", "cube", cube, (1, 0, 0, 0), (0.04, 0.04, 0.04), vel),
        "tray": ObjectState("tray", "fixture", "tray", (0.42, -0.03, 0.775), q, (0.18, 0.18, 0.05)),
        "table": ObjectState("table", "fixture", "table", (0.63, 0, 0.375), (1, 0, 0, 0), (0.6, 1.0, 0.75)),
    }
    arts = {"button": ArticulatedState("button", "button", 0.006 if pressed else 0.0, pressed, pressed),
            "door": ArticulatedState("door", "sliding_door", 0.5, open_fraction=0.5)}
    return OracleSnapshot(time=1.0, objects=objects, articulated=arts, base=base, base_footprint=(0, 0),
                          tcp={"left": (0.3, 0.2, 1.0), "right": (0.3, -0.2, 1.0)},
                          gripper_opening={"left": 0.1, "right": 0.1}, contacts=frozenset(contacts),
                          contacts_ever=frozenset(contacts_ever), held_by=held or {},
                          initial={"cube": (0.42, 0.2, 0.77)},
                          regions={"car": Region("car", (2.0, -0.5, -0.1), (3.0, 0.5, 0.5))})


def test_inside_and_not_held():
    assert evaluate({"inside": ["cube", "tray"]}, snap()).ok
    assert not evaluate({"inside": ["cube", "tray"]}, snap(cube=(0.42, 0.2, 0.77))).ok
    assert not evaluate({"inside": ["cube", "tray"]}, snap(vel=(0.3, 0, 0))).ok, "a moving cube is not placed"
    assert not evaluate({"inside": ["cube", "tray"]}, snap(cube=(0.42, -0.02, 0.95))).ok, "above the rim"
    assert evaluate({"not_held": "cube"}, snap()).ok
    assert not evaluate({"not_held": "cube"}, snap(held={"cube": "left"})).ok
    assert evaluate({"held": "cube"}, snap(held={"cube": "left"})).ok


def test_inside_respects_container_yaw():
    # 0.07 m off-centre along x: inside an axis-aligned tray, outside one turned 45 degrees
    s0 = snap(cube=(0.42 + 0.065, -0.03 + 0.065, 0.775))
    s45 = snap(cube=(0.42 + 0.065, -0.03 + 0.065, 0.775), tray_yaw=math.radians(45))
    assert evaluate({"inside": ["cube", "tray"]}, s0).ok
    assert not evaluate({"inside": ["cube", "tray"]}, s45).ok


def test_touch_and_combinators():
    s = snap(contacts_ever=[("cube", "left_hand")])
    assert evaluate({"touched": ["cube", "any_hand"]}, s).ok
    assert evaluate({"touched": ["cube", "left_hand"]}, s).ok
    assert not evaluate({"touched": ["cube", "right_hand"]}, s).ok
    assert evaluate({"all": [{"touched": ["cube", "any_hand"]}, {"not": {"held": "cube"}}]}, s).ok
    assert evaluate({"any": [{"held": "cube"}, {"touched": ["cube", "left_hand"]}]}, s).ok


def test_geometry_predicates():
    s = snap(cube=(0.42, 0.2, 0.84))
    assert evaluate({"height_above": ["cube", "table", 0.03]}, s).ok
    assert not evaluate({"height_above": ["cube", "table", 0.10]}, s).ok
    assert evaluate({"displaced_lt": ["cube", 0.1]}, s).ok
    assert evaluate({"hands_clear_of": ["cube", 0.1]}, s).ok
    assert evaluate({"distance_lt": ["left_tcp", "cube", 0.3]}, s).ok


def test_articulated_and_regions():
    assert evaluate({"pressed": "button"}, snap(pressed=True)).ok
    assert not evaluate({"pressed": "button"}, snap()).ok
    assert evaluate({"door_open": ["door", 0.4]}, snap()).ok
    assert not evaluate({"door_open": ["door", 0.8]}, snap()).ok
    fp = ((0.3, 0.25), (-0.3, 0.25), (-0.3, -0.25), (0.3, -0.25))
    assert evaluate({"base_in_region": "car"}, snap(base=(2.5, 0, 0)), {"footprint": fp}).ok
    assert not evaluate({"base_in_region": "car"}, snap(base=(2.2, 0, 0)), {"footprint": fp}).ok, "rear corners outside"


def test_unknown_predicate_is_rejected():
    with pytest.raises(PredicateError):
        validate({"all": [{"teleported": "cube"}]})
    with pytest.raises(PredicateError):
        evaluate({"inside": ["cube", "tray"], "held": "cube"}, snap())
