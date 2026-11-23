# Adding a task

A task is a package `src/omakase_bench/tasks/<family>/<task>/`. This walkthrough follows
`tasks/manipulation/cube_to_tray`, the simplest complete task: one robot picks a cube and puts it into a
tray.

Families: `manipulation` (a fixed base and its arms), `mobile_manipulation` (the base moves as part of the
task), `multi_robot` (several robots, one agent each), `games` (multi-robot tasks ruled move by move).

## 1. Make the package

```
tasks/manipulation/pick_place/
  __init__.py     the reference solution and the registration
  task.yaml       the definition
```

## 2. Define the task (`task.yaml`)

```yaml
id: manipulation/pick_place
version: 0.1.0
status: draft
title: Cube into the tray
interfaces: [tools, code, scripted]
instruction: Put the {cube_color} cube into the white tray.
scene:
  robot: {preset: d1, base: [0, 0, 0], lift: 0.10, neck: {pan: 0.0, tilt: 0.55}}
  fixtures:
    - {name: table, type: box, size: [0.60, 1.00, 0.75], pos: [0.63, 0.0, 0.375], color: wood, label: table}
    - {name: tray, type: tray, size: [0.18, 0.18, 0.05], pos: [0.42, -0.03, 0.775], color: white, label: tray}
  objects:
    - {name: cube, type: box, size: [0.04, 0.04, 0.04], pos: [0.41, 0.21, 0.77], mass: 0.05, color: gray, label: cube}
  cameras:
    showcase: {pos: [1.55, -1.25, 1.55], lookat: [0.45, 0.0, 0.85], fovy: 45}
variation:
  cube.xy: {uniform_box: [[0.38, 0.18], [0.44, 0.24]]}
  cube.yaw_deg: {uniform: [-40, 40]}
  cube.color: {choice: [gray, red, blue, green]}
min_separation: {"cube|tray": 0.19}
success:
  all:
    - {inside: [cube, tray]}
    - {not_held: cube}
subgoals:
  - {name: grasped, pred: {held: cube}, weight: 0.3}
  - {name: placed, pred: {all: [{inside: [cube, tray]}, {not_held: cube}]}, weight: 0.7}
budget: {sim_seconds: 90, wall_seconds: 900, api_calls: 40, model_calls: 40, usd: 1.5}
```

| Block | What it is |
| --- | --- |
| `id` | how the task is named on the command line and in results |
| `interfaces` | which subjects may attempt it: `tools` (one robot API call per model turn), `code` (a written program), `ee` (fingertip rows from camera images), `chunks` (joint action chunks), `scripted` |
| `instruction` | what the subject is told; `{cube_color}` is filled from the variation |
| `scene` | the robot and its start, static `fixtures`, free `objects`, `articulated` elements (buttons, doors, drawers), named `regions`, `cameras` |
| `variation` | what changes per seed: `uniform`, `uniform_box`, `choice` on an element's `xy`, `yaw_deg`, `color`, or on `robot.base` |
| `success` | a predicate over simulator ground truth; a model's claim never scores. The vocabulary is in `core/predicates.py`. |
| `subgoals` | weighted partial credit, reported beside success |
| `budget` | what one episode may spend |
| `requires` | optional: the backend capabilities the task needs. Left out, a task needs `reset` and `ground_truth`. |

When independent distributions cannot describe the scene, or success depends on what the episode asked
for, name functions of the package:

```yaml
generator: omakase_bench.tasks.manipulation.pick_place:build     # (scene, rng, params) -> (scene, values)
success: {goal: "omakase_bench.tasks.manipulation.pick_place:placed"}   # (snapshot, values, params) -> Verdict
params: {…}
```

`tasks/manipulation/tabletop/generators.py` has examples of both. A task with several robots adds a
`robots:` block, a `schedule:` and `referee: module:Class` (`core/team.py` documents the interface;
`tasks/multi_robot/block_relay` is the small example, `tasks/games/chess` the large one).

## 3. Write the reference solution and register the package (`__init__.py`)

The reference solution is a program that reads ground truth and calls the robot API a model calls. It is
the task's upper bound and its test: if it fails, the task is broken.

```python
"""Pick the cube from the table and put it into the tray."""
from __future__ import annotations

from pathlib import Path

from ....harness.api import Refused, RobotAPI
from ....harness.session import Session
from ....subjects.scripted import grasp_wrist, object_pose
from ...base import TaskPackage


def solve(api: RobotAPI, session: Session) -> None:
    p, yaw, size = object_pose(session, "cube")          # ground truth, in the base frame
    t, tyaw, tsize = object_pose(session, "tray")
    arm = api.spec.arm_for(p[1])                         # the arm on the cube's side
    wrist = grasp_wrist(yaw)
    api.open_gripper(arm)
    try:
        api.move_ee(arm, p[0], p[1], p[2] + 0.12, "down", wrist)
    except Refused:
        wrist += 90.0                                    # a cube can be gripped across either pair of faces
        api.move_ee(arm, p[0], p[1], p[2] + 0.12, "down", wrist)
    api.move_ee(arm, p[0], p[1], p[2] + 0.006, "down", wrist)
    api.close_gripper(arm)
    api.move_ee_by(arm, dz=0.12)
    rim = t[2] + tsize[2] / 2
    api.move_ee(arm, t[0], t[1], rim + 0.10, "down", grasp_wrist(tyaw))
    api.move_ee(arm, t[0], t[1], rim + 0.035, "down", grasp_wrist(tyaw))
    api.open_gripper(arm)
    api.move_ee_by(arm, dz=0.10)


PACKAGE = TaskPackage(
    name="manipulation/pick_place",                    # the package's path under tasks/
    summary="pick a cube and place it in a tray",
    scripted={"manipulation/pick_place": solve},       # task id -> reference solution
    directory=Path(__file__).parent,
)
```

That is all the registration there is: the registry imports every `tasks/<family>/<task>` package, and
the scripted subject, the CLI and task lookup ask the registry.

A package may also add commands to the CLI (`commands=`; `tasks/games/chess/cli.py` adds
`omakase-bench chess …`).

## 4. Run it

```bash
uv run omakase-bench list tasks                                      # the task loads and its predicates are known
uv run omakase-bench run --task manipulation/pick_place --model scripted/oracle --seeds 0-4
uv run omakase-bench run --task manipulation/pick_place --model mock/place --interface tools     # a stand-in model, no key
uv run omakase-bench run --task manipulation/pick_place --model anthropic/<model> --interface code --max-usd 2
```

Each run writes `results.json`, `report.html` and per episode a trace and a video.

## 5. Test it

`tests/tasks/manipulation/cube_to_tray/test_cube_to_tray.py` is the pattern: the scripted player succeeds
on several seeds; a stand-in model runs the task end to end through the tools and the code interface; the
same seed gives the same start state.

## 6. Publish it in a suite

A task under development is defined by its package's `task.yaml` and can change. To compare models on it,
publish a suite:

```bash
uv run omakase-bench suite freeze starter@1 --tasks manipulation/pick_place,core/press_button --seeds 0-19
```

This copies each task's YAML into `suites/starter/v1/`, writes `suite.yaml` and records the suite's hash
in `suites/frozen.yaml`. From then on the copy is what `--suite starter@1` runs, and the hash covers the
YAML, the predicate vocabulary and the generator and goal modules the tasks name. A published suite never
changes: the next change to a task is `starter@2`. A task id then resolves to the newest published copy;
to run the package's working `task.yaml` pass its path to `--task`. That is why `cube_to_tray`'s own YAML
is `suites/core/v1/cube_to_tray.yaml` and its package holds only code.
