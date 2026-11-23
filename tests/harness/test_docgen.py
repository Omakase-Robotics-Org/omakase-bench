from omakase_bench.harness.docgen import coerce_arguments, python_stubs, tool_specs


def test_tool_specs_come_from_signatures():
    specs = {s.name: s for s in tool_specs()}
    assert {"move_ee", "list_objects", "close_gripper", "move_base", "camera_image"} <= set(specs)
    p = specs["move_ee"].parameters
    assert p["properties"]["arm"]["enum"] == ["left", "right"]
    assert p["properties"]["x"]["type"] == "number"
    assert set(p["required"]) == {"arm", "x", "y", "z"}
    assert "straight line" in specs["move_ee"].description


def test_python_stubs():
    text = python_stubs()
    assert "def move_ee(arm: Literal['left', 'right'], x: float" in text
    assert "def get_state() -> dict" in text


def test_argument_checking():
    assert coerce_arguments("move_ee", {"arm": "left", "x": 1, "y": 0, "z": 1}) == {"arm": "left", "x": 1.0, "y": 0.0, "z": 1.0}
    for bad in ({"arm": "middle", "x": 0, "y": 0, "z": 0}, {"arm": "left", "x": "far", "y": 0, "z": 0},
                {"arm": "left", "x": 0, "y": 0}, {"arm": "left", "x": 0, "y": 0, "z": 0, "speed": 3}):
        try:
            coerce_arguments("move_ee", bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
