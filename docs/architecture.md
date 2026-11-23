# Architecture

omakase-bench runs a subject (a language model, a policy or a scripted program) on robot tasks in
simulation and scores the result from simulator ground truth. This document describes the layout of the
repository and the abstractions the layout rests on.

## Top-level folders

The word "harness" is used throughout the documentation and the prompts for everything that stands between
a model and the robot, so that is the name of the folder holding it.

| Folder | What belongs there |
| --- | --- |
| `core/` | Definitions and protocols with no robot, task or simulator in them: `Backend`, `Task`, `Suite`, the success predicates, the referee protocol, the runner and results. |
| `harness/` | What mediates between a subject and a robot: the session and turn loop, parsing of command lines, the single robot API every command passes through with its safety checks, IK invocation, observation building, the move between idle and working poses. |
| `harness/kinematics/` | Robot-agnostic kinematics: the robot-only model, forward kinematics, and IK solvers behind one interface. |
| `robots/` | Robot descriptions and nothing else: one folder per robot (URDF, meshes, `robot.yaml`, poses, limits, and the robot's API document where it has one), plus the loader of `robot.yaml`. |
| `backends/` | What executes joint commands: the simulators (`mujoco/`, `isaac/`) and the real robot (`real/`), each behind the `Backend` protocol. |
| `tasks/` | One package per task, grouped by family; a task package owns everything that exists only for that task. |
| `suites/` | Published benchmarks: frozen task lists with seeds and the task YAML they were published with. |
| `subjects/` | What is measured: one module per interface (tools, code, ee, chunks, scripted). |
| `models/` | Language-model provider adapters behind one request/reply seam. |
| `sandbox/`, `report/`, `objects/` | Execution of model-written programs, reports and videos, procedural scene objects. |

Dependencies point one way: `cli` uses `core`; `core/runner.py` starts a `harness` session; subjects act
through `harness`; `harness` reads `robots` descriptions and drives a `backends` implementation; `tasks` are
loaded by id and reach the harness only through the hooks their YAML names.

## Tree

```
src/omakase_bench/
  cli.py                  list | doctor | run | report | compare | leaderboard | showcase | robot | suite | <task commands>
  core/
    backend.py            Backend protocol, OracleSnapshot
    frames.py             world and base frame changes
    task.py               Task, SceneSpec, Episode; a task YAML becomes these
    suite.py              Suite, suite hash, frozen suites, task lookup by id
    predicates.py         success vocabulary (its bytes are part of every suite hash)
    goal_predicates.py    `goal: module:function` success checks
    team.py               several robots in one scene: TeamSpec, turn schedule, Referee protocol
    runner.py results.py stats.py scene_logic.py
  harness/
    session.py            one robot's episode: budgets, time, trace, scoring hooks
    team_session.py       several robots: per-robot sessions, turns, referee calls, idle and working poses
    api.py                RobotAPI: every model-facing command passes through it
    docgen.py             tool specs and Python stubs generated from RobotAPI
    ee_control.py         EEController: fingertip rows, clearance checks, whole-body commands
    end_effectors.py      task-level end-effector limits
    depth.py              pixel-to-point lookups
    command_api.py        a robot's own command API: OpenAPI validation, request types, adapter protocol
    kinematics/
      model.py            the robot-only model, forward kinematics, rotations
      ik.py               IKSolver protocol, solver registry, the default damped-least-squares solver
  robots/
    spec.py               RobotSpec and load_robot: the robot.yaml schema
    d1/                   robot.yaml, d1.urdf, d1.meshes.urdf, meshes/, spec.json, PROVENANCE.md
      api/                openapi.json, PROVENANCE.md (vendored subset of the D1 daemon's API: data),
                          cameras.md (what the camera software offers a client)
    check.py              `omakase-bench robot check`: load, IK reach, stills
    arm6/                 a second robot: one 6-joint arm on a pedestal, primitives only
  backends/
    mujoco/ isaac/        simulators
    real/
      backend.py          RealBackend: lease, joint targets, measured state, cameras and depth through a
                          robot-specific driver
      http.py  ws.py      the HTTP and WebSocket transports; disabled unless enabled by option and environment,
                          no default host
      cameras.py          colour from shared-memory topics, video devices or JPEG snapshots; depth from topics
      srvd.py             the camera daemon's frame format
      calibration.py      the calibration file: lenses and mounts
      d1/                 the D1 daemon API: api.py (command lines, adapter), mirror.py (the API served by a
                          simulated robot), driver.py (joint targets to requests, state to joints), vendor_api.py
  models/                 provider adapters behind one request/reply seam
  subjects/               agents.py (tools, code), ee_agent.py, chunks.py, scripted.py, team.py
  sandbox/ report/ objects/
  tasks/
    __init__.py           registry: every task package, lookup by task id
    base.py               TaskPackage
    manipulation/
      cube_to_tray/       the template task
      touch_cube/  press_button/
      tabletop/           chess_move, stack, sort, clutter_pick
      packed_grid/
    mobile_manipulation/
      elevator_call/
    multi_robot/
      block_relay/
    games/
      chess/              rules, layout, scene, referee, player, tour, resume, cpl, video, cli
  suites/
    frozen.yaml           published suites: their hashes
    core/v1  manip/v1  grip/v1  team/v1  games/v1  drafts/v0
tools/
  check_assets.py         repository check, not part of the package
tests/                    mirrors src: core/ harness/ robots/ backends/ models/ sandbox/
                          tasks/<family>/<task>/ tools/ plus test_cli.py
docs/
  architecture.md  new_task.md  new_robot.md  real_robot.md  reference.md  decisions.md  results/
```

## Core abstractions

| Abstraction | Where | What implements it |
| --- | --- | --- |
| `Backend` | `core/backend.py` | `backends/mujoco`, `backends/isaac` |
| `RobotSpec` | `robots/spec.py` | one `robots/<name>/robot.yaml` per robot |
| `IKSolver` | `harness/kinematics/ik.py` | the default solver; others register by name |
| `EndEffectorLimits` | `harness/end_effectors.py` | one parser per end-effector type |
| `CommandAdapter` | `harness/command_api.py` | `backends/real/d1/api.py` |
| `RobotDriver` | `backends/real/backend.py` | `backends/real/d1/driver.py` |
| `Subject` | `subjects/base.py` | tools, code, ee, chunks, scripted |
| `ModelAdapter` | `models/base.py` | one adapter per provider |
| `Task` | `core/task.py` | a task YAML |
| `TaskPackage` | `tasks/base.py` | one `tasks/<family>/<task>/` package per task |
| `Referee` | `core/team.py` | a class of a multi-robot task package |
| `Suite` | `core/suite.py` | a directory under `suites/` |

The rule for all of them: the framework calls the interface, and anything specific to one robot, one
task, one solver or one simulator sits behind it.

## Tasks

A task is a package `tasks/<family>/<task>/`. Families group tasks by what the robot has to do:
`manipulation` (a fixed base and its arms), `mobile_manipulation` (the base moves as part of the task),
`multi_robot` (several robots, one agent each), `games` (multi-robot tasks ruled move by move).

A task package holds:

| Part | Form | Used for |
| --- | --- | --- |
| definition | `task.yaml` | scene, variation per seed, instruction, success, sub-goals, budget, interfaces |
| scene build | `generator: module:function` in the YAML | scenes that independent distributions cannot describe |
| success | predicates in the YAML, or `goal: module:function` | scoring from the oracle snapshot |
| referee | `referee: module:Class` in the YAML | multi-robot tasks: turn order, rulings, final result |
| observation | `interfaces:`, `ee.views`, scene cameras in the YAML | what a subject receives |
| prompt text | `instruction`, `ee.rules` in the YAML; text built by the generator | task-specific model-facing text |
| reference solution | `scripted` in the package's `PACKAGE` | the scripted upper bound, also the task's own test |
| tools | `commands` in the package's `PACKAGE` | task subcommands of the CLI |

`tasks/base.py` defines the registration record:

```python
@dataclass(frozen=True)
class TaskPackage:
    name: str                                  # "manipulation/cube_to_tray"
    summary: str
    scripted: Mapping[str, Program] = field(default_factory=dict)   # task id -> reference solution (api, session) -> None
    moves: Mapping[str, MovesFn] = field(default_factory=dict)      # task id -> pick-and-place moves, for solutions that act through ee rows
    commands: Callable[[Subparsers], None] | None = None            # adds the package's CLI commands
    directory: Path = Path(".")                                     # where task.yaml lives
```

Each package's `__init__.py` assigns one to `PACKAGE`. `tasks/__init__.py` imports every
`tasks/<family>/<task>` package and offers `packages()`, `package_for(task_id)`, `scripted_program(task_id)`,
`programs()` and `definitions()` (the `task.yaml` files). Nothing else in the repository lists tasks: the scripted
subject, the CLI and `find_task` all ask the registry.

One robot or several is a property of the definition, not of the package: a YAML with a `robots:` block
runs through `harness/team_session.py` with one subject per robot and a referee; without it, through
`harness/session.py` with one subject.

### Where a task's YAML lives

A task is developed in its package (`task.yaml`). Publishing a suite copies the YAML of each of its tasks
into `suites/<id>/v<N>/`; that copy is what the suite hash covers and it never changes afterwards. The
published tasks exist as those copies only; their packages hold the code. `find_task(id)` looks in the suite directories
(newest version first) and then in the task packages.

### Chess as one task among many

`tasks/games/chess/` contains all chess code and nothing else does; the definition is the published copy
`suites/games/v1/chess.yaml`:

| Module | Content |
| --- | --- |
| `suites/games/v1/chess.yaml` | scene, pieces, reach tables, clock, the rules text shown to the players |
| `rules.py` | legal moves, SAN, FEN, PGN |
| `layout.py` | board geometry, piece names, the reach tables |
| `scene.py` | the generator `chess_game`, the start position, the layout and reach text each player reads |
| `referee.py` | `ChessReferee`: rulings, the Fischer clocks, faults, contacts, result |
| `player.py` | the scripted player: pick and place, hand-overs, `scripted_turn` |
| `tour.py` `resume.py` `cpl.py` `video.py` | reach check, resume file, centipawn loss, video composer (were `tools/chess_*.py`) |
| `cli.py` | `omakase-bench chess tour`, `resume`, `cpl`, `video` |

The Staunton piece meshes stay in `objects/staunton.py`: `staunton` is a scene shape type any task can use
(`manip/chess_move` does).

### How the next tasks fit

**Vending machine** (`tasks/manipulation/vending_machine/`): press the button of a named product, take
the item from the delivery slot.

- `task.yaml`: fixtures for the cabinet; one `button` per product (existing articulated type); the items
  as free objects behind a `drawer` or a new articulated type `flap`; a region for the delivery slot.
- Scene behaviour "the item drops when its button is pressed" is a rule of `core/scene_logic.py`, declared
  in the YAML like `sliding_door.opens_with` today (`releases: <object>` on the button).
- Success: `all: [{pressed: button_cola}, {height_above: [cola, floor, 0.5]}, {held: cola}]`, or a
  `goal:` function in the package when the product varies per seed.
- `generator`: shuffles which product stands behind which button and writes the instruction.
- `scripted`: press with one arm, reach into the slot with the other.
- No change to `core/`, `harness/` or the backends unless the flap needs a new articulated type, which is
  one entry in `backends/mujoco/scene_mjcf.py` and one in `backends/isaac/server.py`.

**Elevator** (`tasks/mobile_manipulation/elevator/`): call the elevator, enter, select a floor, ride,
exit. `drafts/elevator_call` already covers call and enter.

- `task.yaml`: lobby and car fixtures, `button` and `sliding_door` articulated elements, regions for the
  car and the destination lobby, `interfaces: [tools, code, scripted]` because the base has to drive.
- The ride is scene behaviour: a car that moves when a floor button is pressed and the door is closed.
  That is a new articulated type `elevator_car` with its controller in `core/scene_logic.py` (the door
  controller there is the model for it).
- Success: `all: [{pressed: call_button}, {pressed: floor_2}, {base_in_region: lobby_2}]` with sub-goals
  per stage (called, entered, selected, exited).
- `scripted`: the existing `elevator_call` program extended by the in-car button and the exit drive.
- A two-robot variant (one robot holds the door) is the same package with a `robots:` block and a
  referee, in `multi_robot/`.

## Robots: description and harness

### Description (`robots/<name>/`)

A robot is a URDF plus `robot.yaml`. `robots/spec.py` reads it into `RobotSpec`; `docs/new_robot.md`
documents every field. The fields, with what used to be in code:

| Field | Meaning | Before |
| --- | --- | --- |
| `urdf`, `visuals`, `default_visual` | kinematics, collision geometry, visual variants | same |
| `base_link`, `mobile_base`, `torso_lift`, `neck` | the driven joint groups | same |
| `arms.<name>.joints`, `end_effector`, `home`, `max_joint_speed`, `leave_home_first` | arm chains | same |
| `arms.<name>.side` | `left` or `right` of the base frame, for choosing an arm by a target's y | arm names `left`/`right` assumed in code |
| `arms.<name>.self_collision_from` | first joint whose links are checked against the robot's own body | hard-coded: the third joint |
| `end_effectors.<name>…` | links, fingers, pads, opening, force, fingertip offsets | same |
| `end_effectors.<name>.tip_reach` | fingertip distance past the TCP that scripted programs plan with | constant `TIP` in `subjects/scripted.py` |
| `cameras` | link, field of view, size, depth range | same |
| `poses` | named joint poses; `home` comes from here or from `spec.json` | same |
| `collision.exempt_geoms` | name pattern of geoms left out of self-clearance | hard-coded suffix `_exempt` |
| `actuation` | position gains and force limits per joint group, base servo lag | constants in `backends/mujoco/robot_mjcf.py` |
| `api` | the robot's own command API: OpenAPI document, command-line adapter, real-backend driver, arm sides, base ceilings | adapter was implied; there was no driver |
| `ik.solver` | registered name of the robot's IK solver | the robot API always used the built-in solver |
| `defaults` | lift and neck at reset | same |

Every new field has a default equal to what the code did (`tip_reach` defaults to `fingertip_offset.open`),
`robots/d1/robot.yaml` states the D1 values explicitly, and an existing external `robot.yaml` keeps loading.

### What was D1-specific in the harness, and where it is now

| Item | Was | Is |
| --- | --- | --- |
| Arm names `left`, `right` | `Arm = Literal["left", "right"]` in the API, `"left" if y >= 0` in scripted code, prompt text "two arms" | tool schema and prompt text generated from `RobotSpec.arms`; arm choice by `arms.<name>.side` |
| Fingertip reach 11 mm | `TIP` constant | `end_effectors.<name>.tip_reach` |
| Shoulder exemption | self-clearance skips each arm's first two links | `arms.<name>.self_collision_from` |
| Torso shell exemption | geoms named `*_exempt` | `collision.exempt_geoms` |
| Actuator gains | by joint index in the arm (first four joints stiffer), base and lift numbers | `actuation` block |
| Base servo lag 0.095 s | constant measured for the D1 base | `actuation.base_lag_s` |
| Named poses | `spec.json` of the source description, plus `robot.yaml` | unchanged; `spec.json` is optional |
| Daemon command lines and their prompt text | `robot/daemon_api.py` and `/v1/...` strings in `subjects/ee_agent.py` | `backends/real/d1/api.py`, behind `CommandAdapter` |
| Default robot `d1` | `load_task` and `parse_team` fall back to `d1` | kept as the default for existing YAML; new tasks name their robot |

Not D1-specific, verified: the IK solver, `EEController`, `RobotAPI` motion code and the MuJoCo backend
read joints, chains, tool frames and limits from `RobotSpec` and the URDF only.

Three items stay:

- The generator modules of `manip`, `grip` and `team` choose `left` or `right` by a target's y, so those
  tasks need arms with those names.
- `backends/isaac/server.py` names D1 links in one contact-sensor path and sets its own drive gains; it
  cannot be exercised without Isaac Sim.
- `models/mock.py` `mock/press` picks `left` or `right` (`mock/place` reads the robot's arm names).

### IK

`harness/kinematics/ik.py`:

```python
class IKSolver(Protocol):
    def solve_ik(self, arm, position, rotation, q_current, *, pos_tol, ang_tol, iters, restarts) -> IKResult: ...

register_ik_solver(name, factory)      # factory(kinematics) -> IKSolver
ik_solver(name, kinematics)            # "dls" (DlsSolver, damped least squares) is built in
solve_near(solver, kinematics, ...)    # the solution nearest the current joints
```

`harness/kinematics/model.py` holds the robot-only model, forward kinematics and rotation helpers; it works
for any description because it is built from the URDF. `RobotAPI` uses the solver the robot names
(`robot.yaml` `ik.solver`, default `dls`); end-effector rows use it too unless the task names another
(`ee.ik`).

### The robot's own command API

Some robots accept joint-level commands over an API of their own. Three things belong to such an API and
they live in three places:

| What | Where | Why there |
| --- | --- | --- |
| The API document (OpenAPI subset) and its provenance | `robots/<name>/api/` | It is data describing the robot. `robots/d1/api/openapi.json` is the vendored subset of the D1 daemon's document; `PROVENANCE.md` beside it records source and commit; `cameras.md` describes the camera interfaces. |
| What the harness needs from any such API | `harness/command_api.py` | Checking a request against the document, the request and envelope types, and `CommandAdapter`: how a model's command lines read, how they are described in the prompt, what serves them in simulation. |
| Everything that speaks one robot's API | `backends/real/<name>/` | For the D1: the command lines and adapter (`api.py`), the API served by a simulated robot (`mirror.py`), and the driver of the real backend (`driver.py`). |

`robot.yaml` `api` names the document (`spec`), the adapter and the driver.

## Backends

A backend executes joint commands. The simulators and the real robot are the same kind of thing to the
harness; they differ in what else they can do.

**Every backend** (`core/backend.py` `Backend`): `info`, `load(robot, scene)`, `joint_positions`,
`command_joints` (the harness sends every command through `Session.command_joints`), `step` and `sim_time`
(time), `render` and `cameras`, `close`.

**Capabilities** are optional: a Protocol the backend also satisfies and a name in
`BackendInfo.capabilities`.

| Capability | What it gives | mujoco | isaac | real |
| --- | --- | --- | --- | --- |
| `reset` | build a scene and put it in the start state of a seed (`SceneReset.reset`) | yes | yes | no |
| `ground_truth` | privileged object poses (`GroundTruth.oracle`) | yes | yes | no |
| `contacts` | exact contact detection in the ground truth | yes | yes | no |
| `deterministic` | the same seed gives the same episode | yes | not stated | no |
| `depth_camera` | depth images for the `locate:` lookup | yes | no | on the robot, with a depth topic and the calibration file |
| `multi_robot`, `object_pose`, `pose_clearance`, `gripper_force_limit` | several robots in one scene, putting an object back, clearance queries, finger force caps | yes | no | no |

The real backend's joint state and cameras are not capabilities: every backend has them.

**Tasks and referees declare what they need.** `required_capabilities(task)` is the task YAML's
`requires:` list when it has one. Without it the definition implies the needs: a scene to build needs
`reset`, success predicates and referees read `ground_truth`, a `robots:` block needs `multi_robot`, and a
referee class adds its own (`ChessReferee.requires = ("contacts", "object_pose")`). The runner checks
before anything moves and raises `BackendUnsupported` naming the missing capabilities, so a task that
needs ground truth fails on the real backend with that message instead of running unjudged. A task with
`requires: []` runs on any backend; on one without ground truth its result says
`not judged: the backend has no ground truth`.

**The real backend** (`backends/real/`). For a run on hardware the bench runs on the robot's own
computer (`docs/real_robot.md`). `RealBackend` is generic. A robot-specific `RobotDriver` (`robot.yaml`
`api.driver`) turns the harness's joint targets into requests of the robot's API and that API's state
readings into joint positions; the transports send them; the camera sources deliver images and depth. The
harness therefore builds a subject's observations from the robot with the code it uses in simulation.

| Part | D1 |
| --- | --- |
| Arm lease | taken for the run (`POST /v1/arm/lease`, class `policy`), renewed every third of its lifetime on a thread (a model call outlasts the lease), named in every arm command, handed back on close. A lease another client holds refuses the run before anything is sent; a lease lost during the run stops it at the next step. |
| Motion, streamed | `ws_url` given: one WebSocket connection (`ws.py`), both arms in one `move_joints_both` request with `wait: false` per control tick. The daemon treats such requests within 250 ms of each other as a stream and slows a target on approach to a clearance margin instead of refusing it. |
| Motion, fallback | no `ws_url`: HTTP (`http.py`), one `move_joints` request per arm that changed, each checked by the daemon's motion guard as a single move. |
| Lift, neck, grippers | `slider/set_height`, `neck/cmd`, `end_effectors/{side}/set`. |
| Base | `chassis/cmd_vel` only: a change of the base target within a control period becomes the velocity that covers it, within the ceilings of `api.base_ceilings`; a motion's first velocity is taken from the measured pose, and a zero velocity follows in the first period without a new target. Only the daemon's routes; never the base's own API. |
| Joint state | measured: `GET /v1/arm/feedback`, `/v1/end_effectors`, `/v1/neck/state`, `/v1/slider/state` (slow), and the base's odometry pose from `/v1/chassis/amr/localization` where the daemon serves it. A reading is reused for 50 ms. |
| Colour | `camera_source=iceoryx`: the camera daemon's shared-memory video topics; `v4l2`: its V4L2 loopback devices; `http`: a JPEG snapshot URL per camera, for use from another host. |
| Depth | on the robot: the camera daemon's depth topics, re-drawn on the colour image with the calibrated lens (`cameras.py` `DepthTopics`). Head camera only. The depth imager is taken to sit at the colour camera unless `real.depth_to_colour` states its offset (zero for the D1: a known error of about 15 mm). A topic the camera daemon already rendered into the colour view (`depth_frame: color`) is used as it is, with the intrinsics its frames carry. |
| Calibration | `~/.config/omakase/camera_calibration.json` (schema `omakase.camera_calibration/2`, `calibration.py`): per camera the lens (five distortion models) and the mount. |
| Time | wall clock; `step` sleeps until one control period after the step before it. |

Which device or topic is which camera is configuration, not code: `robots/d1/robot.yaml` `real.cameras`
holds the on-robot defaults (taken from the camera daemon's D1 profiles and its example graph for the D1's
depth camera) and `-B camera.<name>=...`, `-B depth.<name>=...` replace them.

With a depth topic and the calibration file the backend announces `depth_camera` and offers the harness two
more things by name: `pixel_ray` (where a pixel looks, from the calibrated lens) and `camera_mount` (the
calibrated optical frame in a link of the robot). `harness/depth.py` uses them when a backend has them and
the description's ideal lens and nominal camera pose otherwise, so the `locate:` lookup is one function for
both. The backend also names the cameras that have depth (`depth_cameras`), and the prompt offers the lookup
for those views only. Without depth or calibration the capability is not announced and the lookup is left
out of the prompt.

Safety:

- It does nothing unless enabled twice: option `enable` (`-B enable=1`) and `OMAKASE_BENCH_ROBOT_ENABLE=1`.
- It has no default host: options `url`, `ws_url`, or `OMAKASE_BENCH_ROBOT_URL` and
  `OMAKASE_BENCH_ROBOT_WS_URL`.
- Every request is checked against the robot's API document before it is sent.
- With `dry_run` it prints the requests, opens no socket, device or topic and reports commanded targets.
- Closing stops the base, hands the lease back and closes devices, topics and connections, each step
  whatever the one before it did; a failure while setting up closes what was opened.
- The tests use `dry_run` or stand-ins: a loopback daemon that answers in the formats of the vendored
  document (checked against it), image files as video devices, a queue as the shared-memory layer, a
  calibration file written by the test.

What runs where on the robot: FK and IK need MuJoCo (a model, no renderer; a test runs them with the
renderer made to fail). The shared-memory binding (`iceoryx2`) has to be the camera daemon's version and
the process has to run as the camera daemon's account; both are the camera daemon's rules.

### What a subject observes, and where it comes from

| Observation | Interface | In simulation | On the real backend |
| --- | --- | --- | --- |
| Camera images (head, wrists) | `tools`/`code` `camera_image`, first view; `ee` tiled views every step | rendered by the simulator | the robot's cameras: shared-memory topics or video devices on the robot, JPEG snapshots from another host; scaled to the same size |
| Arm state: TCP position, jaw direction | `get_state`; `ee` observation row (fingertip position, yaw, tilt) | forward kinematics of the simulated joints | forward kinematics of the measured joints |
| Gripper opening | `get_state`, `open_gripper`/`close_gripper` results; `ee` row | simulated finger joints | the end effector's reported openness |
| Lift height, head angles | `get_state`; `ee` row with whole-body commands | simulated joints | `slider/state` and `neck/state` |
| Joint angles | `ee` row with joint commands | simulated joints | measured joints |
| IK feedback (`unreachable`, joint-jump refusal) | `ee` row, robot API refusals | harness IK | harness IK, the same |
| Depth lookup (`locate:`) | `ee` | depth render, ideal lens, the description's camera pose | on the robot with a depth topic and the calibration file: the depth camera's reading, the calibrated lens and mount, through the same function. The head view only: the wrist cameras publish no depth. Otherwise not offered (`ee_lookups_off` in the trace). |
| Base odometry | `get_state`; `ee` row `B` | the simulated base joints | the in-house base firmware's odometry pose (x, y, heading). With another base firmware the daemon reports no heading: the base joints then hold their targets and `base_pose` in the backend's detail says `commanded`. |
| Command refusals for clearance | robot API, `ee` whole-body commands | `pose_clearance` of the simulator | not checked by the harness (no `pose_clearance`); the robot's own motion guard refuses or slows, and a refusal is reported |
| Object list (`list_objects`) | `tools`, `code` | ground truth | **not available** (`ground_truth`); the call is refused |
| Contacts, "closed on something" beyond the opening | results of gripper calls | simulated joints | from the reported openness only |
| Success, sub-goals, referee rulings | scoring | ground truth | **not available**: a task must not require `ground_truth`; the result says "not judged" |

Sim-only, by capability: `reset`, `ground_truth`, `contacts`, `deterministic`, `pose_clearance`,
`object_pose`, `multi_robot`, `gripper_force_limit`. `depth_camera` is the one optional capability the real
backend can have.

**Chess on hardware.** `games/chess` needs `reset`, `ground_truth`, `contacts`, `object_pose` and
`multi_robot`, so it is refused on the real backend today. Running it on two robots needs each of those
replaced by something a real cell has:

- `ground_truth` for the board: a board camera above the table and a detector that reports which square
  each piece stands on (the referee compares squares, so centimetre accuracy is enough), or a human
  arbiter who confirms each position. Either becomes a capability (`board_state`) that the real backend
  announces and a `ChessReferee` variant requires instead of `ground_truth`.
- `object_pose` (the referee putting a displaced piece back): a human arbiter restores the piece, and the
  referee waits for the confirmation.
- `contacts` between the robots: the robots' own collision detection or a stop from the arbiter; a
  contact ends the game as it does in simulation.
- `reset`: the arbiter sets up the board; the seed selects nothing.

None of this is implemented.

## Suites

A suite is a directory `suites/<id>/v<N>/` with `suite.yaml` (task list, seeds) and the YAML of its tasks.
Its identity is the SHA-256 over those files, the predicate vocabulary and the code the tasks name.

`suites/frozen.yaml` records the hash of every published suite. A `Suite` reports `published` and `intact`;
a test and `omakase-bench list suites` fail when a published suite no longer has its hash.
`omakase-bench suite freeze <id>@<N> --tasks ... --seeds ...` publishes one: it writes the directory with a
copy of each task's YAML and appends the `frozen.yaml` entry. An existing version is never overwritten; a
changed task is a new suite version.

Published: `core@1`, `manip@1`, `grip@1`, `team@1`, `games@1` and `drafts@0` (tasks still being developed).
A task under development lives in its package (`task.yaml`) and runs by id; publishing copies it.
