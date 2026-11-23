# Design decisions

Each entry: the decision, the alternatives considered, and why. Newest last.

## D1. One runner, subjects declare an interface
A subject declares `tools`, `code`, `chunks` or `scripted`; one runner resets,
lets the subject drive through a `Session`, and scores from the oracle. No
`agent`/`vla` branch in the runner. `kind` is a display label only.

## D2. Success only from simulator ground truth
`Session.score()` evaluates the task's predicate on `backend.oracle()`. A
model's `finish(note)` or a program's printout is recorded as `model_claim`
and never changes the score (`tests/core/test_episodes.py::test_model_claim_never_scores`).
Latched events (a button pressed at any time, a contact at any time) are
latched by the backend, so a brief press counts even if the button springs back.

## D3. Robot = URDF + robot.yaml; D1 preset from manipulation-kit
The D1 preset vendors the composed `generated/d1/d1.urdf` of manipulation-kit (the publisher's motion
library, not public) and
`spec.json` (Apache-2.0, commit recorded in `robots/d1/PROVENANCE.md`).
`robot.yaml` names joint groups, end effectors (TCP link, jaw joints and
their open/closed values, pad links), cameras (optical frames, vertical FOV
from the nominal lens in `spec.json`) and the mobile base. A future D1 Pro or
next-generation robot is another preset directory with the same API.
Arm names are physical sides (`left` = +y).

## D4. MuJoCo model from the URDF through MuJoCo's importer
The kit's `description/mjcf.py` settings are reproduced (visuals kept,
static links not fused, mimic joints as joint equalities, visual/collision
groups). On top: collision bits (robot never collides with itself or the
floor), gravity compensation on robot bodies, high-friction pads, position
actuators per driven joint, TCP sites and cameras at optical frames.

## D5. Visual variants: full meshes by default, primitives as the light variant
`robot.yaml` lists `visuals: {mesh: d1.meshes.urdf, primitive: d1.urdf}`.
Kinematics and collision geometry are identical in both (checked when the
files were vendored). Meshes are copied unchanged from manipulation-kit-assets
and every file is listed in `assets.yaml` (sha256, source, commit, licence);
`tools/check_assets.py` (run by pytest) fails on any unlisted or changed mesh.
The two camera bodies (Gemini 305 wrist cameras, torso fisheye) are not D1
parts and stay primitive. Select with `OMAKASE_BENCH_VISUAL=primitive` or
`-B visual=primitive`.

## D6. Kinematics and IK are backend-neutral
IK runs on a robot-only MuJoCo model built from the same URDF (damped least
squares, joint limits, up to 30 random restarts, solution closest to the
current posture preferred). A backend only needs to follow joint targets, so
Isaac (and later Genesis or the real robot) gets the same Cartesian API with
no IK of its own.

## D7. Model-facing robot API (Phase 1 track: "state")
Functions: `get_state`, `list_objects`, `move_ee`, `move_ee_by`,
`open_gripper`, `close_gripper`, `go_home`, `set_lift`, `look_at`,
`move_base`, `wait`, `camera_image`. Docs for models are generated from the
signatures and docstrings (`harness/docgen.py`).
`list_objects` reports ground-truth poses and sizes, i.e. a perfect detector.
This makes Phase 1 a planning-and-control benchmark; a `vision` track that
removes `list_objects` (photos only) is planned and needs no harness change.
`move_ee` with a named approach direction ("down", "forward", ...) and a
wrist angle: for "down", a positive angle turns the jaws counter-clockwise
seen from above so an object's `yaw_deg` aligns the jaws with its faces; the
jaws are symmetric, so the API also tries `wrist + 180` when the first is out
of reach.

## D8. Code interface sandbox
Programs run in a child interpreter (`python -I`) that reads the program from
stdin and then locks itself down: an audit hook refuses `open`, `os.*`,
`subprocess.*`, `socket.*`, `ctypes` and similar events; imports are limited
to an allow-list (math, numpy, json, random, itertools, collections,
functools, statistics, time, dataclasses, typing); builtins lose `open` and
`input`; rlimits forbid new processes (`NPROC=0`), file writes (`FSIZE=0`)
and bound memory and CPU; the environment is empty and the working directory
is a fresh temporary directory. The only channel out is the robot proxy over
an inherited pipe. A network namespace (`unshare -rn`) is used when the host
allows unprivileged user namespaces; hosts that restrict them (AppArmor, for
example) run without it, and `doctor` reports which layers are active. 11 escape attempts
are in `tests/sandbox/test_sandbox.py`. This is a benchmark sandbox, not a security
boundary for hostile code: for untrusted submissions run the harness inside a
container or VM.

## D9. Timing: lockstep is official
The simulator only advances inside robot API calls, so model latency is not
scored. `--time-mode realtime` advances the world (holding the last targets)
for as long as each model call took; it is reported as reference only and the
results mark such runs unofficial.

## D10. Budgets and caps
Per episode: simulated seconds, wall seconds, robot API calls, model calls,
USD. Per run: model calls and USD (`--max-calls`, `--max-usd`). Every
exceeded cap ends the episode with `budget_exceeded:<what>`; a run-wide cap
also stops the run (`results.json` `stopped`). Provider retries are bounded
(3 attempts, exponential backoff, only for rate limits, 5xx and connection
errors). There is no unbounded loop anywhere: the tools agent nudges a
text-only reply once, then ends the episode.

## D11. Prices
Costs come from a price table (USD per million tokens) of first-party list
prices: Anthropic models, and `openai/gpt-6.2-astra` at $8 in, $40 out, $0.40
cached input, $10 for a 30-minute cache write and twice that for a 12-hour one.
A model without an entry is costed with a default marked `assumed`, which the
CLI and the subject card state; override with `-M price_in=... -M price_out=...`.
Each input token is costed at one rate (plain, cache read or cache write).

OpenAI prompt caching depends on the model generation (`models/openai.py`). From
GPT-6.2 on, top-level `instructions` follow the input and are never cached, so
the system prompt is sent as the first `developer` message; a request with a
marked prefix (the `ee` prompt) carries an explicit cache breakpoint on it and
asks for explicit mode, so that only the prefix is written (implicit mode would
bill the whole changing request as a cache write on every call). A recorded
partial game sent `instructions` and read nothing from the cache in 915k input
tokens.

## D12. Isaac backend: server next to the simulator, rebuilt per reset
Isaac Sim runs in its own Python (3.11, numpy 1.x), so the backend is a TCP
server (`backends/isaac/server.py`) and a client implementing the same
Backend Protocol. Each `reset` builds a fresh stage from the resolved scene
(robot USD cached by URDF hash), which keeps reset exact and simple; it takes
about 4 s. Articulated fixtures (buttons, doors, drawers) are small generated
URDFs; scene behaviour comes from the shared `SceneLogic`. Every control tick
is one round trip, so over a high-latency ssh tunnel (~170 ms RTT) an
episode is ~10x slower: the recommended setup runs the harness on the GPU host
(`-B host=local`).

## D13. No fake backend
MuJoCo is pip-installable, CPU-only and fast (a Phase 1 episode simulates in
well under a second without video), so tests run the real backend with mock
and scripted subjects instead of a fake one.

## D14. Phase 1 workspace
With the jaws pointing down, the D1 arm reaches about 0.48 m forward of the
base at table height, so the table's front edge is at x = 0.33 m and objects
are sampled in x in [0.38, 0.46] m. The cube and the tray are at least 0.19 m
apart so the open jaws (100 mm) clear the tray wall.

## D15. Articulated fixtures touch only the robot
Button caps, door panels, drawer fronts and hinged leaves collide with the
robot but not with walls, floor or each other (MuJoCo collision bits; in
Isaac the elevator layout keeps them physically clear: the door panels run
1 cm in front of the wall and 5 mm off the floor, and the call button sits
outside the panels' travel). Otherwise a panel sliding along a wall face
stalls on the wall or on the button and the two engines diverge.

## D16. Isaac initial state is written explicitly
Isaac Lab keeps `init_state` as defaults; the server writes the robot's and
every articulated fixture's joint state after `sim.reset()`. Without it the
robot starts at zero and the lift crawls to its target at the URDF velocity
limit (18 mm/s), which showed up as a 20-40 mm TCP offset in every reach.

## D17. Phase 1 is close to saturated for frontier models
With ground-truth `list_objects`, both tested frontier models solve
touch_cube, cube_to_tray and press_button almost always through either
interface. The elevator draft separates them. `core@2` should add mobile and
multi-step tasks and a photo-only track before a public leaderboard.

## D18. The mobile base obeys its drive type
`robot.yaml` declares `mobile_base.drive`: `differential` or `holonomic`
(the default, for presets that omit it). The simulation model still moves the
base through planar x, y and yaw joints, because a URDF cannot express a
nonholonomic constraint; the API decides which paths those joints follow.
A differential base (the D1: two driven wheels, `spec.json` `base.drive`) is
commanded with a forward speed v and a yaw rate w and zero lateral speed:
`move_base` refuses a nonzero `left_m`, runs v and w constant over one
duration (forward plus turn is an arc), keeps both within
`BASE_SPEED_FRACTION` of `max_speed` and `max_yaw_rate`, and sends the exact
unicycle pose at every control tick. After a bump it backs off along its
heading. A holonomic base keeps the straight-line interpolation of x, y and
yaw. `left_m` stays in the signature for holonomic robots and existing
callers. In MuJoCo the three base servos of a differential base share one
time constant (kv / kp = 0.095 s); with critical damping per joint, yaw
tracked about ten times faster than x and y and the measured base slid
sideways by about 3 % of its forward speed on arcs. Isaac's base drive
already uses one stiffness and damping for all three joints.

## D19. The `ee` interface: fingertip rows, IK in the harness, depth lookup

A fourth way to be a subject, for the question "can a general model control
the arms step by step from camera images alone". Every step the model gets
one text row of robot state and one image of the robot's cameras side by
side, and replies with one row of fingertip targets per arm
(`a: L x y z yaw g | R ...`, mm and degrees in the base frame, or `hold`).
There are no demonstrations and no `list_objects`.

- **Stepping** (`harness/ee_control.py`): a step moves the fingertip centre at
  most 45 mm and the yaw at most 30° from the *measured* pose towards the
  target, so a far target is repeated and a blocked arm does not wind up. The
  approach axis is always straight down.
- **IK feedback**: a step goal without a solution within 3 mm and 3° (or whose
  solution is another arm configuration, more than 1 rad from the measured
  joints) leaves the arm's joint targets unchanged; the next row carries
  `ikL: unreachable`. The gripper command still applies.
- **Pluggable solver**: the controller needs only `solve_ik` (`IKSolver`); the
  default is the backend-neutral `Kinematics`, others register under a name.
- **Depth lookup** (`harness/depth.py`): `locate: <view> <u> <v>` in a reply is
  answered with the base-frame point seen at that pixel, from the camera's
  depth image (optional backend capability `depth_camera`; MuJoCo implements
  it, ideal and noise-free), the preset's intrinsics and the camera pose from
  forward kinematics on the robot-only model, so object state cannot enter.
  Depth outside the camera's declared range returns `no depth`. Lookups take
  no step; at most two replies in a row may contain lookups only.
- **Grasp guidance**: the prompt states the pad band and how deep below an
  object's top to close (10 mm to band − jaw-tip travel − 7 mm margin), all
  computed from the preset.
- **Configuration, not source**: pad length and band, the fingertip offset
  from the TCP (jaws open and closed), the wrist angle that rows call yaw 0
  and each camera's depth range and sensor name are in `robot.yaml`; step
  limits, start poses, camera views, horizon and the task's rules are in the
  task YAML (`ee:`).
- **Prompt caching**: the instructions repeat in every request of an episode;
  the agent marks them (`Text.cache_prefix`) and the Anthropic and OpenAI
  adapters cache up to the mark. Without the mark a chess_move step with
  Opus 5.5 cost $0.021, with it $0.008.

Not modelled: depth noise and holes, depth-to-colour registration error,
calibration error of the wrist-camera pose.

## D20. Tabletop tasks with a goal per episode (`manip`)

`chess_move`, `stack`, `sort` and `clutter_pick` place several named objects
under constraints that independent per-object distributions cannot express,
and their goal changes per episode. A task may therefore name a `generator`
(`module:function`) that receives the scene, the episode's random generator
and the YAML `params` (every dimension of the cell) and returns the episode's
scene and values, and score with `goal: module:function`, which also receives
those values. `goal` is registered outside `predicates.py`, and `suite_hash`
adds the named modules only for suites that use them, so `core@1` keeps its
hash.

Grasp rule: every object a task asks to move must touch a gripper only on
the fingertip pads within the preset's pad band. The MuJoCo backend records
`(object, "<arm>_offpad")` in `contacts_ever` for any other gripper contact
(presets without `pad_band` record nothing new).

The scripted solver (`subjects/scripted_ee.py`) reads ground truth, acts
through the same rows and waits for the measured pose, so it exercises the
step limit and the IK. It solves 30/30 seeds of each task.

A full chess game between the two arms is not offered: with the gripper
pointing down, the largest square both arms reach in front of the base is
about 120 mm (kinematics only, the best of torso lifts 0 to 0.3 m), while a packed 8×8 position needs squares of at least 58 mm
(464 mm board) for the open jaws to close on a piece without sweeping its
neighbours. `chess_move` uses a three-rank section with free neighbouring
files instead.

## D21. End-effector limits per task (`gripper:`)

A task may narrow the opening range of its grippers: `max_opening_mm`,
`min_closing_mm`, `preshape_mm` (the opening of open commands and of the
episode start) and `max_force_n`, shared or per arm (`arms:`). The hardware
range stays in `robot.yaml`; `load_task` rejects task values outside it, so a
task cannot promise what the robot cannot do.

- **One enforcement point.** Every joint target leaves the harness through
  `Session.command_joints`, which clamps end-effector joints to the task's
  limits. The robot API, the `ee` controller, `execute_chunk` and the
  scripted solvers all go through it, so no interface can bypass a limit.
  Interfaces add their own feedback on top (the API result, the next `ee`
  row, the chunk result and `ChunkObservation.notes`) and state the limits
  where the model reads its commands (tool docs and stubs, the `ee` prompt,
  `ChunkObservation.limits`). Alternatives: clamping in each interface only
  (a forgotten path, e.g. a joint-space chunk, would bypass it), or a wrapper
  around the backend object (hides the backend's type from capability checks).
- **Opening, not joint values.** Limits are pad-face distances in mm, the
  quantity a task author measures. They map to finger joints linearly, the
  mapping the API already reports as `opening_m`: for the D1 the opening
  computed that way from the measured finger coordinate matches the measured
  distance of the two pad frames within 1 mm over the stroke (MuJoCo,
  10 mm steps from 100 mm to 0). A preset with a non-linear linkage would need
  a calibration table in `robot.yaml`.
- **Schema per end-effector type.** The fields above belong to
  `parallel_gripper`; `register_limit_type` lets another type (suction,
  multi-finger hand) bring its own fields, validation, clamp and prompt
  sentence. Core code only calls the `EndEffectorLimits` protocol and holds
  no gripper constants.
- **Partial openings in `ee` rows.** Under limits a row's gripper command
  may be `w<mm>`; without limits rows accept only 0 and 1 as before, so a
  `w` row in a task without limits is an invalid reply exactly as it was.
- **Force cap through a capability.** `max_force_n` caps each finger
  actuator through the optional `gripper_force_limit` backend capability
  (MuJoCo sets the actuator force range until the next reset). A backend
  without it refuses the episode rather than run it without the cap.
- **Opt-in.** Without the block the session sends targets unchanged and every
  prompt and doc is the same.
- **Start shape.** With limits the jaws are commanded to the pre-shape when
  the session starts and the episode waits until the finger joints are within
  0.3 mm of it (at most 3 s simulated): from the reset pose the D1 jaws take
  more than a second to close from 100 mm to 44 mm.

The fingertip-pad rule (D20) is independent: a narrow opening changes where
the jaws can go, not where they may touch, and off-pad contacts are recorded
as before.

## D22. Episodes start at the robot's normal home (`manip@1`, `grip@1`)

The D1 preset named `spec.json` `manipulation_home` as its arm home, a pose
used only by this harness. The arm home is now the robot's normal home
(`spec.json` `home`), so an episode starts where the robot rests, and every
new task, multi-robot tasks included, starts there unless it names another
pose of the preset (`scene.robot.home`).

- **One home for every published task.** `core@1` was first scored from
  `manipulation_home`; its scripted solutions also pass from the normal home
  (60/60), so the published suites carry no per-version start pose (D25).
  `drafts/elevator_call` starts at the normal home as well (scripted 20/20).
- **What broke from the normal home and the fixes.** With the `ee.start`
  poses removed, every ee task failed (0/20, 5 seeds): at home the jaws point
  forward, and no sequence of 45 mm / 30° steps turns them down: the wrist
  joint 7 reaches its limit, and a jaws-down pose needs about 2.8 rad of
  shoulder and wrist motion. `ee.start` is therefore kept: it is the working
  pose of the ee interface (TCP at z 1.00 m, gripper down, wrist cameras over
  the table), not a copy of `manipulation_home` (TCP at z 0.87 m, jaws turned
  inwards). Reached in one joint-space move from the normal home, the arm swept
  the table (chess 2/5, stack 2/5 with pieces and blocks knocked over: the TCP
  dips 9 cm below its home height). `ee.start_path: staged` moves the joints in
  `robot.yaml` `leave_home_first` (the shoulder pitch) first; the TCP then
  never goes below its home height. At a torso lift of 0.10 m the open jaws at
  home hang 6 cm above the table edge, where rank-1 pieces and blocks stand
  (chess 2/5 even with the staged move); the `@1` tasks use a lift of 0.15 m.
  Result: `manip@1` 120/120 and `grip@1` 30/30 scripted on seeds 0–29.
- **Nearest branch for the staged start (tasks 1.1.0).** From the normal home
  the IK solver's first solution for the jaws-down start pose turned joints 2
  and 5 by about 170° each, and back on the way home. The staged start takes
  the solution nearest the current joints (`Kinematics.solve_ik_near`:
  descents from the current joints and perturbations of them, least largest
  joint change), and the `@1` tasks moved to version 1.1.0 so that their
  suite hashes change with the behaviour. From home every jaws-down pose 1 m high still
  needs a large shoulder move (joint 2 about 140°): home points the jaws
  forward.

## D23. Tasks with several robots (`robots:`, `team@1`)

A task may hold several robots, each driven by its own agent, for games
between models (two D1s playing chess, one model each) and hand-overs.

- **Namespaced composition, not a second backend.** The backend builds every
  robot into one scene with its names prefixed `<id>/`; MuJoCo attaches each
  robot's MJCF with `MjSpec.attach` (an XML-level rename would have to know
  every attribute that references a name). Robots get their own collision bit
  (8 << k) and collide with every other robot, never with themselves, so
  contacts between robots exist in the simulation instead of passing through.
  The single-robot path builds the same XML as before (prefix ''), checked by
  byte-identical scripted final states.
- **A view and a session per robot.** `RobotView` presents the shared backend
  under the Backend Protocol with one robot's joints and cameras under their
  single-robot names and an oracle whose hands, base and contact parts are its
  own; `RobotSession` is a `Session` over that view. The robot API, the ee
  controller, the chunk interface and every agent therefore run unchanged, in
  the robot's own base frame, and every joint target still passes that robot's
  `command_joints` (its gripper limits). Alternative: robot-aware agents (every
  interface would need a robot argument, and a forgotten path could command
  another robot).
- **Turns are fresh agent runs.** A turn is one call of the agent's `run`;
  the referee's message (history, feedback) is appended to the robot's
  instruction. Agents keep no hidden state between turns, and the shared
  observation is exactly what the referee wrote, which is logged. A turn ends
  when `run` returns, at a per-turn cap, or when the referee's `turn_over`
  says so; that is how an ee agent, which otherwise runs to its horizon, ends
  a turn once its move is done.
- **Simultaneous is lockstep with threads.** Agents block on model calls, so
  each runs in its own thread. Every backend call is served by the thread that
  runs the episode (MuJoCo's renderer and the Isaac socket are not
  thread-safe), and simulated time advances one control tick only when every
  running agent waits for it. The result does not depend on model latency,
  as in single-robot lockstep runs.
- **Start and idle poses, harness transfers.** At the normal home two D1s
  0.74 m apart have their jaws 7 mm apart over the board centre, and in the
  relay layout (0.95 m) an arm left at its ee start pose touched the other
  robot's (fault on turn 2 of the first scripted run). A robot therefore has a
  `start_pose` and an `idle_pose` (named poses, default home); in `turns` mode
  the idle robots hold their idle pose and the harness moves the acting robot
  from idle to start before its turn and back after it. The moves are the
  harness's, not the model's: they are the same for every model, cost nothing
  and cannot be judged by the referee. They are checked by simulating them:
  any contact of the robot during a transfer is a `transfer_collision` fault.
  Alternative: planning collision-free paths (no planner in the core; the
  check finds a bad pose choice, which is a task-configuration error).
- **Poses are configuration with the lift.** `robot.yaml` `poses:` adds named
  poses to those of `spec.json` and may set the torso lift. `stow` does not
  serve as the D1 idle pose: at a lift of 0.10 m it puts the arm flange 40 mm
  into a 0.75 m table, and at 0.20 m its hands sit 30 mm from the opponent's
  near board corner (4.8 mm from a hand reaching there). `side_idle` (arms out
  beside the torso, jaws down at ee yaw 180, lift 0.20 m) was chosen by a
  search over TCP positions and yaws beside the torso, scored by the largest
  joint change from home and on to the jaws-down working pose, among the
  poses that keep 30 mm from the other robot reaching the far board corner
  and from the robot's own body. Its first version (TCP x 0.05, z 1.05, yaw 0
  in ee terms) was 180° away from the working pose in joints 5 and 7: the
  wrist spun on every turn. Every episode now records joint travel
  (`joint_travel`): in `team/block_relay` seed 3 the wrist joint 5 swings at
  most 85° (with the first idle pose and ee yaw 0: 174–185°, four times per
  arm), while the shoulder pitch
  (joint 2) sweeps up to 170° from home to the middle zone, one direction,
  once per move.
- **Referee in task code, generic interface.** `start`, `turn_over`,
  `after_turn` (with snapshots before and after the turn; may restore objects
  through the `object_pose` capability) and `result` (success, outcome,
  per-robot metrics). The task's `success` predicate is optional when a
  referee scores the episode; where both exist both must hold.
- **Faults and costs per robot.** Contacts between two robots' parts are
  `inter_robot_collision` faults (default policy `fail`: the episode stops
  and fails). Usage and budgets are per robot; `results.json` adds
  `per_model`, cost and calls summed per model over the robots it drove.
- **Bases.** Bases stay where the task puts them. A differential base is
  still driven only along its heading if an agent calls `move_base`.
- **Isaac.** Not implemented: the Isaac client does not announce
  `multi_robot`, and a multi-robot task is refused with that reason. The
  server would need one articulation per robot with prefixed prim paths and
  per-robot target buffers; the wire protocol already carries
  `scene.robots`.
Scripted results: `team/block_relay` 29/30 (seed 8: robot a's blue block did
not end up in the zone in three turns; each time the referee put it back).
`grip@1` 1.1.0 29/30 (seed 21: a pad displaced a neighbouring cylinder by
19 mm; 30/30 with 1.0.0, whose start turned joints 2 and 5 by half a turn).

## D24. Chess between two robots (`games@1`)

Two D1s play a full game of chess on a club-size set, one model per robot, through the `ee` interface,
with a referee that owns the rules. What made it fit, and what the harness does and does not do:

- **Set.** 45 mm squares; king 77 / 31 mm and pawn 38 / 23 mm from a retailer listing of a 3 in king
  Staunton set; queen, bishop, knight and rook heights from the FIDE height ratios scaled to that king
  (69, 57, 49, 45 mm), bases 27–29 mm. Pieces are generated from documented Staunton proportions rather
  than taken from a mesh library, so no asset licence applies; their collision shapes follow the outline
  band by band, which lets a jaw that misses the head by a millimetre touch the piece as it would a real
  one.
- **Reach, and the levers that were tried** (collision-checked IK for every square and capture slot,
  both arms, both robots; kings on every neighbouring square; fingertips from carry height, 20 mm over a
  standing king, down to a pawn's grasp height; ≥ 50 mm to the waiting robot, ≥ 10 mm arm to own body):
  gripper vertical, two robots 0.736 m apart: 58/64 per robot (the far rank's middle squares: the hand
  comes within 50 mm of the other robot's torso). Tilted grasps (−5 to +10° per square) do not change
  that count but are what reaches the far corners and the near middle. Acting lift 0.05 or 0.15 m: 58/64.
  Waiting robot's lift 0.25 or 0.30 m: 58/64. Separations 0.76–0.80 m with the robots fixed: 58–60/64.
  Robot b shifted 0.10 m sideways: 58/64. Robots on adjacent sides at 90°: the two bases overlap before
  the board is in reach. No fixed layout reached every square, so a robot moves itself: stations
  0.856 m apart, the table edge 75 mm from each lift pole, and the acting robot drives 60 mm forward
  for the board and back before it ends its turn. From its station it reaches 54/64 squares and 14/16
  slots; 60 mm forward, 64/64 and 16/16, at least 61.9 mm from the waiting robot and 64.0 mm from its
  camera housings. A sideways drift of ±30 mm of the acting robot, or ±50 mm of the waiting one, keeps
  64/64 and ≥ 61.7 mm (with the tilt chosen per square). Earlier, the harness backed the waiting robot
  off by 6 cm instead; that was dropped so that no robot is moved by anything but its own commands and
  the idle↔working transfer.
- **Self-clearance and the shoulder shell.** Arm-to-body clearance leaves out the torso's shoulder band
  (`torso_shoulder_shell_exempt`), which the D1 model exempts from its own guard because the arm bases
  pass through it; with it counted, d8 and e8 are lost (62/64).
- **Idle and working poses.** The waiting robot holds `side_idle_outward`: `side_idle` with joint 5
  turned by half a revolution and joints 6, 7 mirrored, which puts the wrist cameras on the far side of
  the jaws, facing the board. The D1 has no roll joint with that range (joint 6 spans ±60°, joint 7
  ±90°). It pairs with jaws-down working poses at yaw −45 (left) / +45 (right): from idle to working no
  joint turns by more than 80°, joint 5 by about 12°; joint 2 swings 48° out and at most 61° back, the
  way back passing through the working start pose (`robots[].return_via: ee_start`), because a direct
  joint-space return took an arm 68 mm through its own body. Measured over a scripted review game:
  joint 5 at most 45° in any phase.
- **The model decides.** The referee accepts a move with one line and no plan; which piece first, which
  arm, whether to hand a piece from one arm to the other and how to correct are the model's. The prompt
  states the physical rules (tolerances, capture slots, castling, en passant, promotion, hand-overs
  allowed, the correction allowance, the clock). It also states each arm's measured reach and the tilt
  each square needs from 60 mm forward, as data about the robot.
- **Rules of chess in the bench.** `tasks/games/chess/rules.py` (perft matches the published counts on the
  start position to depth 5, Kiwipete and positions 3–6; four recorded games replay to their final
  positions; SAN, UCI, FEN, PGN). Draws by threefold repetition and the 50-move rule are taken when the
  current position meets them, the moment a player may claim them; a library that also counts a claim
  for the move about to be made would end some games one move earlier.
- **Clock.** Fischer, 60 min + 30 s per side: model wall-clock time (API latency included) plus simulated
  execution time. Model time is what a remote model costs in a real game; execution time keeps a robot
  that dithers on the board from playing for free. The clock starts when the subject reports its arms at
  the working pose (`session.ready()`), so the harness's own move there (about 3 s a turn) is not
  charged to the player.
- **Base drift is noise, not a bias.** The noise stream is seeded per episode, robot and turn. A
  controller is built for every turn, and one stream per robot repeated the same draws each turn: over
  a scripted game White's base drifted 2.4 mm sideways per turn in one direction (19.6 mm after 8
  turns, past the 30 mm station tolerance by turn 13). With a stream per turn the same game ends 4 mm
  off.
- **Base commands.** One line drives at most `max_forward_m` and turns at most `max_turn_deg`, whatever
  its form (`drive`, `turn` or `vel`), and the path is checked for clearance every 20 mm and 2°. The
  limits were applied to none of the forms the prompt offers before, and only the half-way and end
  poses were checked.
- **Played to the end.** No move cap decides a game and nothing is adjudicated by material: a game ends
  by the rules of chess, the clock or a contact. The Fischer clock bounds its length. A backstop at 200
  moves per side is a harness stop (`*`, resumable), and the budgets are sized for it.
- **A run that dies keeps its game.** The trace is streamed to disk and synced per turn, and
  `progress.json` holds the moves and clocks after every ruled turn; `omakase-bench chess resume` reads it
  when there is no `result.json`.
- **Harness stops are not results.** A cap or an error that cuts a game short (run-wide caps, a robot's
  own budget, simulated time, a model or subject error) ends the episode with that exit reason and the
  outcome `*`; nothing is adjudicated. The task's `run_budget` states the run-wide call cap one game
  may need, and the CLI refuses to start a model on a smaller cap unless `--allow-partial` is given. A stopped
  game resumes from `start_moves` and `start_clocks`, the side to move first.
- **Faults.** A contact between the two robots decides the game: the mover loses, unless the robot it
  touched was not in its waiting position (idle pose within 5° per arm joint and 10 mm of lift, base
  within the station tolerance), in which case that robot loses; being hit while properly idle is the
  mover's fault, standing somewhere else is the hit robot's. A contact with anything else during the
  harness's own transfer is reported, not fatal: after a turn that ends with a hand among the pieces the
  transfer can touch one, and ending a game for the harness's own move would be the wrong verdict; the
  displaced piece is the mover's manipulation fault.
- **Transcript and thinking summaries.** A task opts in with `ee.transcript: true` (chess does; no other
  task's requests or traces change). The request then asks each provider for the readable summary of
  the reasoning: on Claude Opus 5.5 thinking always runs and is billed the same whether its summary is
  displayed or not, so this changes what is returned, not the cost; on GPT-6.2 the summary is generated
  alongside the reasoning that already runs. The effort settings are untouched, and the prompt is the
  same, so neither player is told to think more.
- **Robot API.** Joint-level commands are requests of the D1 daemon's HTTP API, validated against its
  OpenAPI document, served by a simulated daemon in MuJoCo and, for a robot, by an HTTP client that never
  chooses a host and needs two explicit enables (`docs/reference.md`, "The robot API"). The base noise values are
  assumptions.

Verifier (MuJoCo, scripted players through the same rows and API requests, the robot driving itself):
a king carried to every square and capture slot (`omakase-bench chess tour`), 80/80 for each robot, largest
placement error 4.7 mm;
the Opera Game 33/33 valid turns; a line with en passant, a promotion with capture and both sides
castling 32/32 (31 moves and the turn that adjourns); a review line with a hand-over between the arms for
each robot 17/17 (16 moves and the adjourning turn); no fault, no
contact between the robots.

## D25. Suites re-frozen for the first public release

Before the first public release the suites were re-frozen once, from cleaned code, and renumbered so that
every suite has one version.

- **What is published:** `core@1`, `manip@1`, `grip@1`, `team@1`, `games@1`, and `drafts@0` for tasks
  still being developed. The earlier `manip@0` and `grip@0` (the same tasks from another start pose) are
  gone; `team@0` and `games@0` became `team@1` and `games@1`. Every hash in `suites/frozen.yaml` dates
  from this pass; results recorded before it name suites and hashes that no longer exist.
- **Why once, and now:** a suite's hash covers its task files, the predicate vocabulary and the generator
  and goal modules its tasks name. Keeping the earlier hashes had required those modules to stay
  byte-identical under their first module names, a table of old import paths, a per-version start pose
  and a predicate that knew only arms called `left` and `right`. None of that had users outside the
  repository.
- **What changed with it:**
  - Generator, goal and referee modules are named by their real paths
    (`omakase_bench.tasks.manipulation.tabletop.generators:stack`).
  - `touched: [obj, any_hand]` holds for the hand of any arm, whatever it is called, so `core/touch_cube`
    is scored on any robot.
  - Every published task starts at the preset's home; a task may still name another pose
    (`scene.robot.home`).
  - The block-relay referee no longer counts a block it put back as newly placed in a later turn.
  - `games@1` holds a copy of the chess task, like every other published suite.
  - `games@1` counts a robot's allowance to correct a position in replies, as its rules text says (12
    more replies, each carried out in full). It had counted the referee's checks, of which one reply causes
    several, so a model got one or two replies.
  - A game of `games@1` that contains a promotion can be resumed: the start position is rebuilt with the
    promoted piece on the board, exchanged as the referee does it in play. The hash of `games@1` was
    recorded after these changes.
  - Manipulation faults have a price in `games@1`: what the mover puts right itself within its allowance
    (4 replies) costs nothing; a position the referee has to restore takes 1 minute from the mover's
    clock (a flag fall if that empties it). A move that ends the game by the rules ends it whatever its
    execution cost. A first version took 2 minutes and ended the game on a side's third fault; a game
    played under it ended after 14 moves on faults with the position level, so the limit is now optional
    (`fault_limit`) and chess sets none. The hash of `games@1` was recorded again with this rule.
- **From here on** a published suite does not change: a changed task or changed scoring code is a new
  suite version (`omakase-bench suite freeze`), and the test suite fails when a recorded hash no longer
  matches.

Verifier: the scripted reference solutions on every suite's own seeds (100 %), the scripted chess games
and both king tours, after the re-freeze.
