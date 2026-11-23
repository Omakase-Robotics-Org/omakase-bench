# Reference

The commands, the model adapters, the robot API, the `ee` interface, tasks with limits and several robots,
the chess task, the D1 daemon API, the backends and what was verified. For the layout see
[architecture.md](architecture.md); to add a robot or a task see [new_robot.md](new_robot.md) and
[new_task.md](new_task.md).

## Commands

| command | what it does |
|---|---|
| `list [all\|suites\|tasks\|packages\|models\|backends\|robots]` | what is available; suites show `frozen`, `open` or `CHANGED since it was published` (exit status 1) |
| `doctor [--backend isaac]` | rendering (EGL / OSMesa), ffmpeg, sandbox layers, API keys and client libraries, Isaac connection |
| `run --model a,b --interface tools,code,ee --suite core@1 \| --task t1,t2 [--seeds 0-4]` | run and write `runs/<time>_<suite>_<backend>/` |
| `report RUN` | re-render `report.html` |
| `compare RUN... [--video out.mp4] [--html out.html]` | paired statistics (exact McNemar on shared seeds), side-by-side video, table |
| `leaderboard RUN... --out site/index.html` | static leaderboard from `results.json` files |
| `showcase` | run + side-by-side video in one step |
| `robot check ROBOT [--out DIR]` | load a robot description, solve IK around each arm's home, render stills |
| `suite freeze ID@N --tasks a,b --seeds 0-19` | publish a suite: copy each task's YAML, record the hash |
| `chess tour\|resume\|cpl\|video ...` | the tools of `games/chess` (commands a task package adds) |

Useful `run` options: `--code-mode iterative`, `--backend isaac -B host=local`,
`--time-mode realtime` (reference only), `-M effort=low` (model options),
`--max-calls N --max-usd X` (run-wide caps), `--no-video`, `--robot NAME` (run single-robot tasks on
another robot).

## Models

`--model` takes `provider/model`:

| spec | adapter |
|---|---|
| `anthropic/<model>` | Anthropic Messages API (`ANTHROPIC_API_KEY`), adaptive thinking, `-M effort=` |
| `openai/<model>` | OpenAI Responses API (`OPENAI_API_KEY`), `-M reasoning=` |
| `mailbox/<dir>` | requests written to files, answers read from files: a person or another agent session answers |
| `mock/place`, `mock/press`, `mock/finish`, `mock/loop`, `mock/script:<file>` | deterministic stand-ins for tests and for trying a task without a key |
| `scripted/oracle` | oracle-informed hand-written programs (the upper bound per task) |
| `stub/hold` | chunk-interface stub that holds still |

Every model call is costed from `models/pricing.py` (first-party list prices;
a model without an entry uses a default marked `assumed`, override with
`-M price_in=… -M price_out=…`). A cap is checked before each model call, so
the call that crosses it is still made and paid for. Caps: per episode (model calls, robot API
calls, USD, simulated and wall seconds, from the task's `budget`) and per run
(`--max-calls`, `--max-usd`). An exceeded cap ends the episode with
`budget_exceeded:<what>`; retries on provider errors are bounded (3 attempts,
exponential backoff).

## The robot API

What models see is generated from the functions in `harness/api.py`:
`get_state`, `list_objects`, `move_ee`, `move_ee_by`, `open_gripper`,
`close_gripper`, `go_home`, `set_lift`, `look_at`, `move_base`, `wait`,
`camera_image`. Coordinates are metres in the robot base frame (x forward,
y left, z up). A call that cannot be carried out is refused before anything
moves, with the reason. `list_objects` reports ground-truth poses (a perfect
detector); a photo-only track is planned.

The mobile base follows the drive type declared in `robot.yaml`
(`mobile_base.drive`). The D1 is `differential` (two driven wheels): it
drives along its heading and turns, never sideways, so `move_base` refuses a
nonzero `left_m`, and `forward_m` with `turn_deg` drives an arc. A
`holonomic` base (the default when `drive` is omitted) translates in any
direction while turning. The generated docs state the drive type of the
robot in the episode.

## The ee interface

`--interface ee` makes the model the step-by-step controller of both arms.
Every step it receives one text row (joint angles, gripper commands, the
fingertip pose of each arm from forward kinematics), the last rows of the
episode and one image: the robot's cameras side by side (left wrist, head,
right wrist). It replies with one action row:

```
a: L 410 25 795 0 0 | R hold
```

= per arm the target of the fingertip centre in mm in the base frame, the yaw
in degrees and the gripper command (0 open, 1 closed), or `hold`.

- **IK in the harness, limited steps.** A step moves the fingertips at most
  45 mm and 30° towards the target from the measured pose; the model repeats a
  far target. A step goal with no IK solution leaves the arm where it is and
  the next row says `ikL: unreachable`.
- **Depth lookup.** A reply may contain up to three lines
  `locate: <view> <u> <v>`; the harness answers with the base-frame point seen
  at that pixel (`lookup head(160,120) -> x 412 y -205 z 790 mm`), computed
  from the camera's depth image, its intrinsics and its pose from the joint
  encoders. Ground truth is never read, and lookups take no step.
- **What the model is told.** The task's rules and goal, the cell layout an
  installation would know (table, board or tray poses, object dimensions),
  the gripper's pad geometry and how deep to grasp. Never where objects stand.
- **No demonstrations, no `list_objects`.**

Options a task may switch on in its `ee` block (all off by default; prompts and rows of tasks without
them are unchanged):

- **Tilted grasps** (`tilt: {max_deg}`): a row may give a tilt after the yaw (`L 520 -100 820 -45 t10 w27`);
  the gripper leans about its jaw-closing axis, jaws level, fingertips past the wrist, which reaches
  further from the robot.
- **Rows that run to their target** (`row_steps: N`): a row is repeated for up to N steps until the
  fingertips arrive, a target has no IK solution, or they stop moving.
- **Referee lines** (`say: [move, done]`): lines such as `move: Nf3` go to the referee of a multi-robot
  task, whose answer appears in the next prompt.
- **Robot API commands** (`whole_body`): joint, gripper, lift, head and base commands as requests of the
  robot daemon's API (see "The robot API"), with a base execution-noise model (`base_noise`).

Step limits, IK tolerances, start poses and camera views are per task (the
`ee:` block of the task YAML); pad geometry, fingertip offsets and each
camera's depth range are per robot (`robot.yaml`). The IK solver of the row
steps is pluggable: register one with `register_ik_solver` (`harness/kinematics/ik.py`)
and name it in the task's `ee.ik` (default `dls`, the preset's own kinematics). A backend offers depth through
the optional `depth_camera` capability; MuJoCo has it (ideal depth, no noise),
and without it lookups answer `no depth`.

## Tasks

| task | interfaces | success (simulator ground truth) |
|---|---|---|
| `core/touch_cube` | tools, code, chunks, scripted | the cube was touched by a hand, moved < 3 cm, and both hands end ≥ 10 cm away |
| `core/cube_to_tray` | tools, code, chunks, scripted | the cube rests inside the tray and no hand holds it |
| `core/press_button` | tools, code, chunks, scripted | the wall button was pressed (latched), robot pose and button position vary |
| `drafts/elevator_call` | tools, code, scripted | the call button was pressed and the whole base footprint is inside the car |
| `manip/chess_move` | ee, scripted | the named piece stands upright inside the named square of a board section (files a–h, ranks 1–3, 50 mm squares, five pieces), grippers open, no other piece moved > 10 mm or tipped |
| `manip/stack` | ee, scripted | two or three 40 mm cubes form a tower in the named order on the untouched bottom cube, at rest, grippers open |
| `manip/sort` | ee, scripted | two or three red or blue cylinders each rest in the tray of their colour, grippers open |
| `manip/clutter_pick` | ee, scripted | the named cylinder of a row with 6–12 mm gaps is lifted 100 mm, no neighbour moved > 20 mm or tipped |
| `grip/packed_grid_pick` | ee, tools, code, scripted | the named cylinder of a 3 × 3 grid is lifted 100 mm with the jaw opening limited to 44 mm, no neighbour moved > 10 mm or tipped |
| `team/block_relay` | ee, scripted | two robots in turns move their blocks into a shared zone (see "Tasks with several robots") |
| `games/chess` | ee, scripted | two robots play a game of chess with a club-size Staunton set; the referee scores it (see "Chess between two robots") |

The `manip` tasks also require every object they ask to move to be gripped on
the fingertip pads only (within the pad band of the robot preset).

Tasks are YAML (`src/omakase_bench/suites/<suite>/v<N>/`) composed from a
small predicate vocabulary (`core/predicates.py`): `inside`, `held`,
`not_held`, `touched`, `height_above`, `displaced_lt`, `hands_clear_of`,
`distance_lt`, `upright`, `pressed`, `door_open`, `base_in_region`,
`in_region`, `all`/`any`/`not`, and `python: module:function`. Tasks whose
objects and goal are drawn together per episode name a `generator` function
and score with `goal: module:function` (`tasks/manipulation/tabletop/generators.py`). A suite is
identified by the SHA-256 of its YAML files and the predicate code; results
record it, and runs with other seeds or realtime timing are marked unofficial.

## Arm home and start pose

Every episode starts with the arms at the robot's home: `robot.yaml`
`arms.<arm>.home`, for the D1 the normal home of `spec.json` (`home`). A task
may name another pose of the preset (`scene.robot.home: <pose>`). An `ee` task
then moves the arms to its working pose (`ee.start`, gripper down above the
table); with `ee.start_path: staged` the joints listed in
`arms.<arm>.leave_home_first` (D1: the shoulder pitch) move first, so the hand
rises before the wrist turns down. A single joint-space move from the D1 home
takes the jaws about 9 cm below their home height, into objects near the table
edge. The staged start takes the IK solution nearest the current joints; the
first solver solution turned joints 2 and 5 by about 170° (`manip@1` and
`grip@1` tasks 1.1.0).

Every published task starts at the preset's home. The `manip@1` and `grip@1`
tasks set the torso lift to 0.15 m (at 0.10 m the open jaws at home hang over
the near board ranks and the blocks). `spec.json` also holds
`manipulation_home`, a pose with the jaws already turned down, which a task may
name as its start. A task id without a suite (`--task manip/chess_move`)
resolves to the newest suite version that holds it.

## End-effector limits per task

A task can narrow how far its grippers open and close, so the same robot can
be tried on tasks that need a partial opening (a packed grid, a chessboard
with occupied neighbouring squares). The hardware range stays in `robot.yaml`
(`max_opening`, `force`, pads); the task's `gripper:` block must lie inside
it, or loading the task fails with the offending field and the hardware range.

```yaml
# task YAML, top level; every field optional
gripper:
  max_opening_mm: 44      # widest the jaws may open (hardware: robot.yaml max_opening)
  min_closing_mm: 0       # narrowest they close to; a close command stops there
  preshape_mm: 40         # opening of open commands and at the episode start (default: max_opening_mm)
  max_force_n: 40         # cap per jaw actuator (hardware: robot.yaml force); needs a backend
                          # with the gripper_force_limit capability (MuJoCo has it)
  arms:                   # per-arm overrides of the fields above
    right: {max_opening_mm: 30}
```

Without the block nothing changes: same docs, prompts and trajectories. With it:

- **Every joint command is clamped.** All joint targets leave the harness
  through `Session.command_joints`, which holds the finger joints inside the
  limits, whatever the interface: robot API (`tools`, `code`), `ee` rows,
  `chunks` (joint-space rows that include finger joints) and the scripted
  solvers. The jaws take the pre-shape opening before the subject starts.
- **The model is told and gets feedback.** The limits are stated in the
  `open_gripper` / `close_gripper` docs (`tools` and `code`), in the `ee`
  prompt and in `ChunkObservation.limits`. A clamp comes back as a short line:
  `open_gripper` returns `"gripper": "opening clamped to 44 mm (task limit)"`,
  an `ee` row returns `gL: opening clamped to 44 mm` in the next observation
  row, a chunk returns `"gripper": ["left: opening clamped to 44 mm"]` and the
  next `ChunkObservation.notes`.
- **Partial openings in `ee` rows.** Under limits the gripper command of a
  row may be an opening in mm, `w<mm>`: `a: L 410 25 795 0 w30 | R hold`.
  `0` opens to the pre-shape, `1` closes down to `min_closing_mm`.

The fingertip-pad rule is unchanged: grasps still have to land on the pads
within the preset's band, and off-pad contacts are recorded as before. A
narrow opening only changes where the jaws can go, not where they may touch.

The limit schema belongs to the end-effector type (`robot.yaml`
`end_effectors.<name>.type`). `parallel_gripper` is registered in
`harness/end_effectors.py`; another type (a suction cup, a multi-finger hand)
registers its own parser with `register_limit_type(type, parser)`, returning
an object that clamps its joint targets, states its limits and gives its
start shape (`EndEffectorLimits`).

`grip/packed_grid_pick` is the example: nine cylinders in a 3 × 3 grid with
columns 60–64 mm apart along the jaw axis. Fully open, the 18 mm pads reach
50–68 mm from the centre and land on a neighbour; at 44 mm they reach
22–40 mm and clear both the target and its neighbours. The scripted solver
succeeds on 30/30 seeds with the block and 0/30 without it (the pads land on
a neighbouring cylinder and the target is never gripped).

## Tasks with several robots

A task may hold several robots, each driven by its own agent (for example two
D1s facing each other across a table, one model per robot). The task lists
them under `robots:` instead of `scene.robot`:

```yaml
robots:
  - id: a                          # names its joints, cameras and contacts a/...
    preset: d1                     # robot.yaml preset; the same preset may appear twice
    base: [0, 0, 0]                # x, y (m), yaw (deg) in the world
    start_pose: home               # named pose at reset (default: the preset's home)
    idle_pose: side_idle           # held while another robot acts (default: the start pose)
    lift: 0.15                     # optional per robot: lift, neck, gripper, ee, cameras,
    agent: {model: scripted/oracle, interface: scripted}   # instruction
  - id: b
    preset: d1
    base: [0.95, 0, 180]
    gripper: {max_opening_mm: 44}  # on top of a task-wide gripper: block, for b only
    ee: {start: {left: [0.40, 0.17, 1.00, 0], right: [0.40, -0.17, 1.00, 0]}}
    cameras: [head, left_wrist]    # the cameras b's agent may use (default: all of the preset)
schedule:
  mode: turns                      # turns | simultaneous
  max_turns: 6
  turn: {sim_seconds: 40, model_calls: 30, api_calls: 60}   # caps of one turn
referee: omakase_bench.tasks.multi_robot.block_relay.relay:RelayReferee
faults: {inter_robot_collision: fail}                      # fail (default) | report
video: {camera: showcase, size: [960, 720], tiles: [a/head, b/head]}
```

- **Scene.** The backend builds the robots into one scene with namespaced
  names (`a/left_arm_joint1`, `b/head`); MuJoCo attaches each robot's MJCF
  with `MjSpec.attach` under its prefix. Robots collide with each other, not
  with themselves. Every robot starts in its `start_pose`, by default its
  normal home.
- **Poses.** A robot's named poses are those of `spec.json` plus the `poses:`
  of `robot.yaml`, which may also set the torso lift. The D1 has `side_idle`:
  arms held out beside the torso, jaws down (TCP at x 0.15, y ±0.40, z 1.00 m,
  ee yaw 180) at a lift of 0.20 m, chosen on the joint branch of home and of
  the jaws-down working pose, so that no joint turns by half a revolution on
  the way. In MuJoCo, with two D1s 0.74 m apart across a 360 mm board, one
  idling there and the other reaching the far corner of the board with its
  gripper down, the robots stay at least 42 mm apart; joint-space moves
  between home and `side_idle` keep 45 mm from the robot's own body. At the
  normal home the two robots' jaws are 7 mm apart over the board centre, and
  `stow` at a 0.10 m lift goes into the table.
- **One session per robot.** Each agent drives its robot through its own
  session over a view of the shared backend: it sees its own joints and
  cameras under the single-robot names, works in its own base frame, and every
  joint target passes its robot's `Session.command_joints`, so its gripper
  limits, IK feedback and off-pad bookkeeping apply as in a single-robot task.
  Budgets (model calls, USD, API calls) are per robot; simulated time is the
  world's.
- **Scheduling.** `turns`: the robot the referee names acts while the others
  hold their idle pose. Before the first turn every robot moves to its idle
  pose; before its turn the acting robot moves to its start pose and after it
  back to its idle pose. These transfers are harness moves (each TCP rises
  5 cm, then the arms move in joint space, the torso rising first and lowering
  last), outside the turn caps and logged; any contact of the robot with
  anything but the floor during one is a `transfer_collision` fault. A turn ends when the agent returns, when a
  per-turn cap is reached, or when the referee's `turn_over` says so (polled
  after every API call or ee step); each turn is a fresh run of the agent,
  whose instruction carries the referee's message (e.g. the move history).
  `simultaneous`: every agent runs at once in its own thread, in lockstep:
  simulated time advances one control tick only when every running agent asks
  for it, so a robot whose model is thinking holds the world.
- **Referee** (`core/team.py`, task-provided `module:Class`, built per
  episode from the episode's values and `params`): `start` names the first
  robot and its message; `turn_over` ends a turn; `after_turn` receives the
  snapshots before and after the turn, rules it valid or not, sends messages,
  puts displaced objects back (`restore`, logged; backends offer the
  `object_pose` capability) and names the next robot or ends the game;
  `result` gives success, an outcome and per-robot metrics.
- **Faults.** Contacts between parts of two robots are recorded as
  `inter_robot_collision` faults (time, robots, parts), contacts during a
  transfer as `transfer_collision`; under `fail` (the default policy) the first
  one ends the episode and fails it.
- **Binding agents.** The task's `agent:` per robot is the default; a run
  overrides it with `--model` (every robot) or per robot with
  `--bind a=anthropic/claude-opus-5-5:ee --bind b=openai/gpt-6.2-astra:ee`.
- **Joint travel.** Every episode records per arm joint the distance travelled,
  the largest one-way swing and the number of swings over 90°
  (`joint_travel`, per robot in a multi-robot episode). IK in the harness
  stays on the arm's current branch: the staged start takes the solution
  nearest the current joints, and an ee step whose solution would move a
  joint by more than `ee.limits.max_joint_step_rad` (57°) is not executed; the
  next row says `ikL: would turn joint 5 by 170 deg in one step, try another
  yaw or a nearer target`.
- **Results.** A multi-robot episode records `robots` (per robot: subject,
  interface, usage, exits, turns, the referee's metrics, its faults), `turns`,
  `faults` and `outcome`; `results.json` adds `per_model`, the cost and calls
  of each model summed over the robots it drove.
- **Backends.** MuJoCo has the `multi_robot` capability. Isaac does not yet:
  a multi-robot task on Isaac is refused before the episode starts.

`team/block_relay` (suite `team@1`) is the example: two D1s 0.95 m apart, each
with two blocks on its left; in turns, each moves one block per turn into its
half of a shared middle zone, and waits in `side_idle` during the other's turn. The referee checks each turn (one own block
placed, nothing else moved by more than 2 cm), puts back what a turn
displaced, and ends the game when all four blocks are in the zone.

```sh
uv run omakase-bench run --task team/block_relay --seeds 0-29 --no-video     # scripted, both robots
uv run omakase-bench run --task team/block_relay --seeds 0 \
  --bind a=anthropic/claude-opus-5-5:ee --bind b=openai/gpt-6.2-astra:ee --max-usd 4
```

## Chess between two robots

`games/chess` (suite `games@1`) is a game of chess between two D1s, one model per robot: robot a plays
White, robot b Black. Each robot sees only its own three cameras; the referee tells it the moves so far in
SAN, the position as FEN, the side to move, the material and both clocks. There are no demonstrations,
no object list and no piece poses.

- **Cell.** Two stations 0.856 m apart across a table (top 0.75 m); a 45 mm board (360 mm, 22 mm border)
  in the middle; a club-size set: king 77 mm on a 31 mm base, pawn 38 / 23 mm (a retailer listing of a
  3 in king set), the other heights the FIDE height ratios (K 9.5 : Q 8.5 : B 7 : N 6 : R 5.5) scaled to
  the king, the bases at club-set proportions; 16 capture slots per side beside the board on the
  mover's right; a spare queen, rook, bishop and knight per side for promotions.
- **Pieces.** Generated from documented Staunton proportions (`objects/staunton.py`): lathe meshes for
  pawn, rook (crenellated), bishop (mitre), queen (coronet) and king (cross), a horse-head silhouette on
  a turned base for the knight. Collision geometry is a stack of primitives that follows each outline
  (cylinders at the largest radius of their band, a sphere for the pawn's ball, boxes for the cross and
  the knight's head), so the jaws close on a head 10–18 mm below the top as on a real piece. The mass
  sits low, as in weighted club pieces (masses are estimates).
- **A turn.** The waiting robot holds `side_idle_outward` (arms beside the torso, wrist cameras facing
  forward) at a lift of 0.20 m. The harness moves the acting robot's arms and torso to its working pose
  (lift 0.10 m) and back afterwards; nothing else moves a robot. The model names its move
  (`move: Nf3`), carries it out with its arms, and writes `done`. From its station a robot reaches 54 of
  the 64 squares; with its base driven 60 mm forward, which it does itself through the robot API, it
  reaches all 64 and every capture slot while staying at least 50 mm from the waiting robot. A turn
  ends with the base back at its station (within 30 mm and 3°, stated in the prompt). A turn has at most
  60 steps and 80 model replies; one that ends before a legal move was named loses the game.
- **Referee lines.** `move: <SAN or UCI>` and `done` are read only as `<keyword>: <text>` or as the
  keyword alone on a line, so a sentence that begins with "Move" or "Done" is not a move or a claim.
- **Rules of the physical game, as the prompt states them.** Every piece upright within 10 mm of its
  square's centre; a captured piece upright in a free capture slot of the captor (15 mm); castling moves
  king and rook; en passant removes the pawn from its own square; a promoting pawn goes to its square and
  the referee exchanges it for a spare piece. Either arm may move any piece; a piece may be put down and
  picked up again or passed between the arms. Which piece moves first, which arm, whether to hand over
  and how to correct are the model's.
- **Referee** (`tasks/games/chess/`). Legal moves come from the bench's own rules module
  (`tasks/games/chess/rules.py`: move generation, check, castling rights, en passant, promotion, checkmate,
  stalemate, the 50-move rule, threefold repetition and insufficient material; SAN, UCI, FEN, PGN;
  checked by perft on the standard positions and by replaying recorded games). An illegal move is refused
  with the reason; the third in one turn loses. After `done` (or once the board and the hands are right)
  the referee compares every piece with the position; differences are listed and the robot has 4 more
  replies (`correction_replies`), each carried out in full; what it puts right in them costs nothing. What
  still differs is a manipulation fault: the referee puts those pieces right (logged), takes 1 minute from
  the mover's clock (`fault_clock_penalty_s`) and tells both sides the fault's number and the clock left.
  No number of faults ends a game; a task may set `fault_limit`, and a side's fault of that number then
  loses ("loss by manipulation faults").
  The game is played to its end: mate, stalemate, a draw by the rules (threefold repetition, the 50-move
  rule, insufficient material), three refused moves in a turn, a turn without a legal move, the clock
  (also when a fault penalty empties it), or a contact between the robots. Nothing is adjudicated. A backstop of the harness cuts a game off after
  200 moves per side (`params.max_moves`, `max_moves_action: stop`) as a harness stop without a result.
- **Clock.** Fischer, 60 min plus 30 s per completed move per side. A side's clock runs during its model
  calls (their wall-clock latency, API time included; a failed attempt that the adapter retries and its
  back-off are not counted) and while its robot moves (simulated time), from the moment its arms are at
  the working pose until the referee confirms the position; the harness's moves between the idle and the
  working pose are not charged. A fall of the flag loses unless the opponent has only its king, or its
  king and one knight or bishop (a simplification of "cannot mate by any series of legal moves").
- **Results.** Per robot: moves, illegal attempts, manipulation faults, knocked-over and restored pieces,
  corrections, turns with a hand-over between the arms, captures, think and execute time per move, clock
  left; the PGN. `omakase-bench chess cpl` adds, with any UCI engine on the machine (none is a
  dependency; `--engine /path/to/stockfish` for an official Stockfish release binary), the average
  centipawn loss per side, the inaccuracies, mistakes and blunders and the worst moves. `omakase-bench chess tour` repeats the reach check behind the reach table
  (a king carried to every square and capture slot by one robot). `omakase-bench chess video` cuts the game video
  (the whole game compressed to a few minutes, about a minute of highlights, or a review at a fixed
  speed) from the streamed frames and the transcript.
- **Recording and transcript.** `video.stream` writes one frame per instant with the side camera and, as
  tiles, each robot's head and both wrist cameras, the board from above and a close view of each side
  (`frames.json`: times and tile names). `ee.transcript: true` records every model call in the trace:
  the prompt text that changes per call, the provider's readable summary of the model's reasoning, the
  reply and the call's wall-clock seconds. The summaries are asked for per request (Anthropic:
  `thinking: {type: adaptive, display: summarized}`; OpenAI: `reasoning.summary: auto`); the reasoning
  itself stays the provider's opaque state. Simulated time stands still during a model call, so the
  video tool holds the picture for the call's real seconds, shows a think timer, and gives every move
  its think and execution time; compressed cuts speed the calls through and still show the real seconds.

```sh
uv run omakase-bench run --task games/chess --seeds 0-2 --no-video          # scripted players, 3 lines
uv run omakase-bench run --task games/chess --seeds 0 \
  --bind a=anthropic/claude-opus-5-5:ee --bind b=openai/gpt-6.2-astra:ee --max-calls 32160 --max-usd 20
```

**Caps and harness stops.** The task states what one game may need from the run-wide call cap
(`run_budget: {model_calls: 32160}`: the backstop's 402 turns of at most 80 model calls; a recorded game took about 13
calls per half-move). The default (`--max-calls 400`) is below that, so a run with a model does not
start until the cap is raised or `--allow-partial` is given. The spend caps (`--max-usd`, 10 by default,
and `budget.usd` per robot) are the operator's to set. A game that a cap or an error
cuts short (a run-wide cap, a robot's own budget, the simulated-time budget, a model or subject error
after the adapter's retries) is a harness stop: the episode's `exit_reason` names it, the outcome is `*`
with no result, and the clocks are those before the interrupted turn. A contact during the harness's
own move between the working and the idle pose with anything but the other robot is recorded and the
game goes on (`transfer_collision: report`), a piece it displaces being restored as a manipulation fault
of the mover.

**Contact between the robots.** A contact decides the game (`inter_robot_collision: report`, ruled by
the referee and stated in the prompt). The robot whose turn it is loses, unless the waiting robot was
not in its waiting position, in which case the waiting robot loses. The waiting position is: every arm
joint within 5° of the idle pose (`params.idle_tolerance_deg`), the lift within 10 mm of the idle height
and the base within the station tolerance (30 mm, 3°), read at the poll that finds the contact (after
every command of the mover, and after the harness's move back to the idle pose). The result records
the parts that touched, which robot was hit and why it was ruled that way (`robots.<id>.metrics.events`,
kind `contact`; `metrics.contacts`). Only the game itself produces
`1-0`, `0-1` or `1/2-1/2`: mate, a draw by the rules, three refused moves in a turn, a turn that reaches
its step limit without a legal move, the clock, or a contact between the robots.

**Resuming.** `params.start_moves` (SAN from the initial position, no promotions) and
`params.start_clocks` (seconds per robot, `clock_left_s` of the stopped run) set a copy of the task up at
a recorded position, with the captured pieces in their captors' slots and the side to move acting first.
`omakase-bench chess resume <run directory> --out resume_chess.yaml` writes that copy from a stopped
run; run it with `--task resume_chess.yaml` and the same seed. A run that died (a crash, a kill) resumes
the same way: the trace is appended entry by entry and synced after every turn, and `progress.json` in
the episode directory is replaced after every ruled turn with the moves, the clocks, the turns and each
robot's usage, so everything up to the last complete turn is kept. The physical position is rebuilt from the moves: every piece
stands on its square's centre again, with the start jitter of the seed.

## The robot API: the D1 daemon's HTTP interface

The D1 preset carries the OpenAPI document of the robot's own daemon, the HTTP API every motion of a real
D1 goes through after inverse kinematics (`robots/d1/api/openapi.json`, the subset the bench uses; source
and commit in `robots/d1/api/PROVENANCE.md`). In a task that opts in (`ee.whole_body`), the joint-level
command lines of the `ee` interface are requests of that API, sent in order:

| line | requests |
|---|---|
| `joints: L <7 deg> [vel=<0..1>]` | `POST /v1/arm/a/mode {mode: position, vel_ratio, acc_ratio}` (with `vel=`), `POST /v1/arm/a/move_joints {joints_deg, wait: true}` |
| `joints: both <7> \| <7>` | `POST /v1/arm/move_joints_both {a, b, wait: true}` |
| `joint: R <0..6> <deg>` | `POST /v1/arm/b/move_joint {joint, position, wait: true}` |
| `gripper: L <openness> [force=<0..100>]` | `POST /v1/end_effectors/a/set {openness, force_percent, wait: true}` |
| `lift: <height_m> [speed=<0..1>]` | `POST /v1/slider/set_height {height_m, speed_ratio, wait: true}` |
| `head: <pitch_rad> <yaw_rad>` | `POST /v1/neck/cmd {pitch, yaw, relative: false}` |
| `base: vel <vx> <omega> <s>` | `POST /v1/chassis/cmd_vel {vx, omega}`, `s` seconds, `POST /v1/chassis/cmd_vel {vx: 0, omega: 0}` |
| `base: drive <m>`, `base: turn <rad>` | the same at 0.1 m/s or 0.3 rad/s (conveniences; the daemon itself takes velocities) |

Arm `a` is the robot's physical left arm (`robot.yaml` `api.arm_sides`). Every request is validated
against the document before it is sent, and `tests/backends/real/d1/test_api.py` fails when the bench's requests and
the document drift apart. The answers are the daemon's envelope (`status`, `data`, `message`, and for a
failure `kind`): an openness or force outside its range is refused with 422 `unprocessable`, a lift
height outside the stroke or a ratio outside 0..1 with 400 `invalid`, a joint target outside its travel
or a joint-space path closer than the clearance margin with 409 `refused`, naming the closest pair; neck
angles and base velocities are clamped, as the daemon does. Fingertip rows (`a: L ...`) are solved by the
harness's IK and executed as joint targets, which on a robot are `move_joints_both` requests.

Two implementations serve the API: `SimDaemon` (`backends/real/d1/mirror.py`) on the MuJoCo scene and
`HttpDaemonClient` (`backends/real/http.py`) for a daemon on a robot, so a transcript of requests from the bench can be sent to a
robot unchanged. The client has no default host: it needs a URL (`OMAKASE_BENCH_ROBOT_URL`) and refuses
to start unless it is enabled both by its caller and by `OMAKASE_BENCH_ROBOT_ENABLE=1`; with `dry_run` it
prints every request and opens no socket. The daemon is unauthenticated and moves a heavy robot: reach it
only over the channel its own documentation names.

The client gives every request the time its operation states in the document (`x-timeout-seconds`: 300 s
for a waited lift move) plus 5 s, and at least `timeout_s`; a robot that does not answer is an
`unreachable` error envelope, and the command line stops there. If the wait of a `base:` line is
interrupted, the stop request that follows it is still sent. The client sends no `holder`, so it works
only while no one holds the arm lease.

Where the simulated daemon differs from a robot:

| | simulated daemon | robot daemon (per its document) |
|---|---|---|
| waiting | every request returns when the motion has ended (`wait: true`) | `wait: false` returns at once |
| `vel_ratio` | scales the joint speed of the next moves of that side; `acc_ratio` is checked and ignored | speed and acceleration |
| time of a base line | the base covers `vx * s` at its own speed limit, then stops; the simulated time that passes is that of the motion, not `s` | the velocity is held for `s` |
| one base line | at most `whole_body.max_forward_m` and `max_turn_deg` (0.10 m, 15° in `games/chess`), refused otherwise | no limit on a velocity command |
| clearance | a base, lift or joint motion that would come within `whole_body.clearance_m` (8 mm) of the table, the other robot or the robot itself is refused before it starts, checked along the path (base: every 20 mm and 2°; joints: every 2°; lift: at the target of a step of at most 0.05 m) | the daemon's own guard; it does not refuse a base motion beforehand |
| self-collision | the torso's shoulder band is left out, as the robot model's own guard leaves it out (`*_exempt` geoms) | the same exemption |
| status codes | 400 `invalid`, 409 `refused`, 422 `unprocessable` (end effector only), 404; each is one the document lists for that operation, but which condition gives which code is read from the document, not measured on a robot | |
| not modelled | the arm lease and motion fence, soft-kill latches, arm modes other than `position`, grip presets of multi-finger hands, the base's own safety and navigation stack, wheel slip | |

Fingertip rows (`a: L ...`) are not daemon requests in the simulation: the harness's IK turns each step
into joint targets, held inside the joint limits and the task's gripper limits, with a solution that
would turn a joint by more than 1 rad in one step refused. They are not checked for clearance
beforehand; a contact between the two robots is detected and ends the episode
(`faults: {inter_robot_collision: fail}`).

The base drives with an execution-noise model (`ee.base_noise`: sideways offset and heading error per
metre driven and per radian turned, drawn afresh for every motion and every turn) whose values in
`games/chess` (20 mm and 1° per metre, 2° per radian) are assumptions, not measurements of a D1.

## Backends

**MuJoCo** (default): the scene is built from the task YAML around the robot's
MJCF (MuJoCo's URDF importer with manipulation-kit's settings). A Phase 1
episode simulates in well under a second without video.

**Isaac Sim**: `backends/isaac/server.py` runs in Isaac Lab's Python on a GPU
host; the client implements the same Backend Protocol over TCP. Run the
harness on that host and let it start the server:

```sh
omakase-bench run --model scripted/oracle --suite core@1 --backend isaac -B host=local \
  -B setup='. ~/isaac-env.sh' -B python=~/isaaclab_ws/env_isaaclab/bin/python
```

`host`, `remote_dir`, `setup` and `python` name a site's machine and paths: an option left out is read
from `OMAKASE_BENCH_ISAAC_HOST`, `_REMOTE_DIR`, `_SETUP` and `_PYTHON`, and otherwise is `local`,
`~/omakase-bench`, no setup and `python`.

`-B host=<ssh destination>` also works (the client copies the package, starts
the server and forwards a port), but each control tick is a round trip, so it
is only practical on a low-latency link.

**Real robot** (`--backend real`): the bench runs on the robot's own computer, sends the joint targets
it sends a simulator and reads measured joint state, camera frames and depth back, so a subject's
observations are built by the same code. [real_robot.md](real_robot.md) has the prerequisites, the exact
command with both safety gates, every option, what is captured, how to stop, and what differs from
simulation.

| Observation | Simulation | Real robot |
| --- | --- | --- |
| Camera images | rendered | the robot's cameras (shared-memory topics, video devices, or JPEG snapshots) |
| TCP position, jaw direction, fingertip rows | FK of simulated joints | FK of measured joints |
| Gripper opening | simulated fingers | reported openness |
| Lift, head | simulated joints | measured |
| IK feedback | harness | harness |
| Depth lookup (`locate:`) | depth render | depth topic and calibration file, same function; else not offered |
| Base odometry | simulated base | the in-house base firmware's odometry; else not measured |
| Clearance refusals | the simulator's clearance query | the robot's own motion guard |
| `list_objects` | ground truth | not available: refused |
| Success | ground truth | not judged |

A task runs on the real backend only if it declares `requires: []` (or `depth_camera`); every published
task needs `reset` and `ground_truth` and is refused with a message that names them.

## Verifier outputs

| gate | command | result |
|---|---|---|
| unit and integration tests | `uv run pytest -q -W error::DeprecationWarning` | 314 passed: predicates and statistics, suite loading and hashing, the sandbox (11 escape attempts refused), budgets and retries, the model adapters against stand-in clients, scripted and mock episodes, the base drive types, the `ee` interface and gripper limits, multi-robot turns and referees, chess rules by perft and the chess referee, resume, the robot daemon API against its document, the real backend against a loopback stand-in, the asset manifest |
| scripted upper bound, MuJoCo, 3 tasks × 20 seeds | `omakase-bench run --model scripted/oracle --suite core@1 --no-video` | 60/60 (touch_cube 20/20, cube_to_tray 20/20, press_button 20/20), Wilson 95% CI 94–100% |
| scripted upper bound, Isaac Sim, 3 tasks × 20 seeds | `omakase-bench run --model scripted/oracle --suite core@1 --backend isaac -B host=local --no-video` | 60/60 (20/20 each) on Isaac Sim 5.1.0 / Isaac Lab 0.54.2, RTX 6000 Ada; gap to MuJoCo 0 pt; about 10 s wall per episode with the harness on the GPU host |
| elevator draft, MuJoCo, 20 seeds | `omakase-bench run --model scripted/oracle --task drafts/elevator_call --seeds 0-19 --no-video` | 20/20 on MuJoCo; 5/5 on Isaac (`--seeds 0-4 --backend isaac -B host=local`, with mp4s) |
| scripted upper bound of the ee tasks, MuJoCo, 4 tasks × 30 seeds | `omakase-bench run --model scripted/oracle --suite manip@1 --seeds 0-29 --no-video` | 120/120 (30/30 each), Wilson 95% CI 97–100% |
| scripted, packed grid with and without the gripper limit, MuJoCo, 30 seeds | `omakase-bench run --model scripted/oracle --suite grip@1 --seeds 0-29 --no-video` | 29/30 with `max_opening_mm: 44` (seed 21: a pad displaced a neighbour by 19 mm; the suite's own seeds 0–4: 5/5), Wilson 95% CI 83–99%; 0/30 with the block removed (`tests/harness/test_end_effectors.py` checks that the pads land on a neighbour) |
| two robots in turns, scripted, MuJoCo, 30 seeds | `omakase-bench run --model scripted/oracle --suite team@1 --seeds 0-29 --no-video` | 29/30 (the suite's own seeds 0–4: 5/5), Wilson 95% CI 83–99%; 0 faults; 118 of 123 turns valid (seed 8: robot a's blue block did not end up in the zone in three turns, each put back by the referee); wrist joint 5 swings at most 85°, none over 90° |
| chess between two robots, scripted players, MuJoCo | `omakase-bench run --task games/chess --seeds 0-2 --no-video`; `omakase-bench chess tour a tour_a.json` and `b` | Opera Game 33/33 valid turns, checkmate; en passant / promotion / castling line 32/32 (31 moves and the turn that adjourns); review line with a hand-over per robot 17/17 (16 moves and the adjourning turn); king tour 80/80 placements per robot, largest error 4.7 mm; no fault, no contact between the robots |

These are the scripted reference solutions on the published suites. Games played by models are in
[docs/results](results/README.md).
