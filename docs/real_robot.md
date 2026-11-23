# Running on the real robot

For a run on hardware the bench runs **on the robot's own computer**: it talks to the robot's daemon over
the loopback interface and reads the cameras locally. The harness is the one used in simulation; the
backend is `real`. Everything below is for the D1; another robot needs its own driver
(`docs/architecture.md`).

Nothing here has been run on hardware yet. The backend is tested against stand-ins only: a loopback
daemon that answers in the formats of the vendored API document, image files as video devices, a queue as
the shared-memory layer, and a calibration file written by the test.

## Prerequisites

- The robot's daemon is running and reachable on the loopback interface (HTTP, and WebSocket for streamed
  motion). It is unauthenticated: do not bind it to another address.
- The robot's camera daemon is running.
- **The process runs as the camera daemon's account.** Its shared-memory topics and video devices are
  readable by that account only; this is the camera daemon's rule, not the bench's. Another account sees
  "no frame on topic ...".
- The calibration file exists (`~/.config/omakase/camera_calibration.json` of that account, schema
  `omakase.camera_calibration/2`) if the depth lookup is wanted.
- A person is at the robot's stop control for the whole run.

Install, on the robot (aarch64 wheels exist for every dependency):

```sh
pip install 'omakase-bench[real]'            # adds the WebSocket client and OpenCV (video devices)
pip install iceoryx2==<the camera daemon's version>   # shared-memory topics; the version has to match the daemon's
```

MuJoCo is installed with the package and is needed: forward kinematics and IK run on a MuJoCo model of
the robot. Its renderer is not used, so no GL library is needed on the robot.

## The command

```sh
export OMAKASE_BENCH_ROBOT_ENABLE=1
omakase-bench run --task <task.yaml> --model <provider/model> --interface ee --no-video \
  --backend real \
  -B enable=1 \
  -B url=http://127.0.0.1:<daemon HTTP port> \
  -B ws_url=ws://127.0.0.1:<daemon WebSocket port>/v1/ws \
  -B camera_source=iceoryx \
  --max-usd 5
```

Two gates have to be open before anything is sent: the environment variable
`OMAKASE_BENCH_ROBOT_ENABLE=1` and the option `-B enable=1`. Either one alone does nothing.

There is no default host: without `url` (or `OMAKASE_BENCH_ROBOT_URL`) the backend refuses to start.
Check the requests first with `-B dry_run=1`, which prints them and opens no socket, device or topic.

The task has to declare `requires: []` (or only `depth_camera`): every published task needs a simulator's
`reset` and `ground_truth` and is refused on this backend with a message that names them.

| option | meaning | default |
| --- | --- | --- |
| `enable=1` | with `OMAKASE_BENCH_ROBOT_ENABLE=1`, lets the backend connect | off |
| `url`, `ws_url` | the daemon's HTTP origin and WebSocket address | none |
| `camera_source` | `iceoryx` (shared-memory topics) or `v4l2` (video devices) on the robot; `http` (JPEG snapshots) from another host | `http` |
| `camera.<name>` | the topic, device or URL of one camera | `robot.yaml` `real.cameras` |
| `depth.<name>` | the depth topic of one camera | `robot.yaml` `real.cameras.depth` |
| `depth=0` | no depth: the `locate:` lookup is not offered | on when a depth topic and the calibration exist |
| `depth_frame.<name>` | the grid the depth topic is on: `depth` (the depth imager's own) or `color` (already rendered into the colour camera's view, see below) | `robot.yaml` `real.depth_frame`; `depth` for the D1 head |
| `depth_to_colour.<name>` | where the depth imager sits in the colour camera's optical frame: `x,y,z` in metres, optionally followed by a quaternion `x,y,z,w` | `robot.yaml` `real.depth_to_colour`; zero for the D1 |
| `calibration` | the calibration file | `robot.yaml` `real.calibration` |
| `lease=0`, `lease_holder`, `lease_class`, `lease_ttl_s` | the arm lease; `lease_class` is `ambient`, `policy` or `operator` | on; `omakase-bench`, `policy`, 30 s |
| `dry_run=1` | print the requests, connect to nothing | off |

Which device or topic is which camera comes from the robot description (`robots/d1/robot.yaml`, block
`real.cameras`), marked there as on-robot defaults: the three video topics as the camera daemon's D1
profiles name them, the head's depth topic and video device as its example graph for the D1 depth camera
names them. The wrist cameras have no video device in those graphs; with `camera_source=v4l2` give them
one with `-B camera.left_wrist=/dev/video<N>` if the robot's graph has one.

## Depth and the `locate:` lookup

- Depth is the head camera's only: the D1's wrist cameras publish no depth topic.
- The lookup is offered when a depth topic is configured, the calibration file has that camera's lens and
  `depth=0` is not given. The prompt then names only the views that have depth (`head`). In every other
  case the lookup is left out of the prompt and the trace records `ee_lookups_off`.
- A depth topic that stops delivering frames during a run answers a lookup with `no depth`; the episode
  goes on.
- Known approximation of the default topic: the depth imager is taken to sit at the colour camera. No transform between the two
  is published and none has been measured for the D1, so `real.depth_to_colour` is zero and a looked-up
  point is off by the sensor's baseline, about 15 mm on the head camera. With a measured value, set it in
  `robot.yaml` or with `-B depth_to_colour.head=x,y,z`.

### Exact alignment: depth rendered into the colour view

The camera daemon reads the sensor's factory depth-to-colour transform but does not put it on the depth
topic, which is why the default above has to assume it. The daemon can apply the transform itself: its
`proc.cloud_to_video` node with `view: color` renders the depth camera's point cloud into the colour
camera's pixel grid, and the gray16 frames it publishes carry the colour camera's intrinsics. A graph for
the D1 head, beside the nodes the robot's graph already has:

```yaml
nodes:
  - id: head
    type: source.realsense
    params:
      serial: "<D435 serial>"
      color: { width: 640, height: 480, fps: 30 }     # needed: it brings the colour geometry and the transform
      depth: { width: 640, height: 480, fps: 30 }
  - id: head_deproject
    type: proc.deproject
  - id: head_depth_color
    type: proc.cloud_to_video
    params: { view: color, format: gray16 }
  - id: head_depth_color_topic
    type: sink.iceoryx
    params: { topic: dxr/camera/head/depth_color, kind: video }
edges:
  - { from: head.depth, to: head_deproject.in }
  - { from: head_deproject.out, to: head_depth_color.in }
  - { from: head_depth_color.out, to: head_depth_color_topic.in }
```

Run the bench on that topic with

```sh
-B depth.head=dxr/camera/head/depth_color -B depth_frame.head=color
```

In this mode the backend uses the frame as it is: a pixel of the colour image has the depth at the same
pixel, and its ray comes from the intrinsics in the frame. `depth_to_colour` does not apply (a configured
value is ignored with a warning), and the calibration file is needed only for the camera's mount; without
it the description's nominal camera pose is used. Two limits:

- The daemon projects with the colour camera's pinhole intrinsics and applies no lens distortion. The D435
  colour lens has almost none; for a lens with visible distortion the default mode with a calibrated lens
  is the better one.
- The D1 head keeps `dxr/camera/head/depth` as its default, because the deployed graphs publish that
  topic. A wrist depth camera (a D405) can be added the same way: its own source, `proc.deproject` and
  `proc.cloud_to_video` nodes, a topic, and `-B depth.<wrist>=<topic> -B depth_frame.<wrist>=color`.

## What happens during a run

1. The backend reads the arm lease. If another client holds it (a teleoperation station, another policy),
   the run is refused and nothing is sent. Otherwise it takes the lease as class `policy`, renews it every
   third of its lifetime for as long as the run lasts (also while a model thinks, which takes longer than
   the lease lives), and names itself as holder in every arm command. An operator-class client can still
   take the arms; the run then stops at its next step.
2. Joint targets from the harness go to the daemon: both arms per control period (0.1 s) in one streamed
   request over WebSocket, the lift, the neck and the grippers as their own requests, the base as a
   velocity that is set to zero in the first period without a new base target. A base motion starts from
   the measured pose, and a velocity never exceeds the ceilings of `robot.yaml` `api.base_ceilings`. The
   daemon's own motion guard checks every arm command and refuses or slows it.
3. The base pose is the odometry pose of the daemon's localization operation. Where the daemon answers
   `unavailable` (a base without the in-house firmware), the base joints report their commanded values and
   `results.json` says so (`provenance.backend.detail.base_pose: "commanded: ..."`): the pose is then not measured.
4. Joint state is read back from the daemon's state operations; camera frames from the topics or devices;
   depth from the depth topic, placed on the colour image with the calibrated lens.
5. The subject sees what it sees in simulation: camera images, the observation row computed from measured
   joints, and, with depth and calibration, the answers to `locate:` lookups.

## What is captured

The run directory is the usual one: `results.json`, `report.html`, and per episode `trace.jsonl` (every
model reply, every command and what the robot answered, lookups, `ee_lookups_off` when depth is missing)
and the prompt. `results.json` records the backend's detail: joint state `measured`, the transport, the
camera source and cameras, the depth cameras or why there is none, the lease, and whether the base pose is
measured (`provenance.backend.detail.base_pose`). There is no episode video (`--no-video`): the scene cameras a simulator renders do not exist.
Episodes are **not judged**: there is no ground truth, so `success` is false and the evidence says so.

## Stopping

- **The robot's stop control comes first.** The bench is not a safety system.
- `Ctrl-C` ends the run: the backend sends a zero base velocity if one was commanded, hands the lease
  back, and closes the cameras and connections. The arms hold their last target.
- If the process is killed outright, the lease expires by itself after its lifetime (30 s) and a base
  velocity stops when the daemon's own deadman lapses.
- Budgets end an episode as in simulation: `--max-usd`, `--max-calls`, the task's `budget`.
- A command the daemon refuses, a lost lease or a connection that drops ends the episode with
  `error:subject` and the daemon's answer in `exit_detail`; the backend then stops the base and hands the
  lease back as on `Ctrl-C`.
- Run one seed per invocation: the scene is reset by hand, and with several seeds the next episode starts
  as soon as the one before it ends.

## What differs from simulation

| | Simulation | Real robot |
| --- | --- | --- |
| Scene | built from the task YAML for a seed | whatever is in front of the robot; a seed selects nothing |
| Success, sub-goals, referee | ground truth | not judged |
| `list_objects` | ground truth | refused |
| Camera images | rendered, ideal lens | the robot's cameras, their real lenses |
| Depth lookup | exact depth, ideal lens, the description's camera pose | the depth camera's reading, the calibrated lens and mount; head camera only; the depth imager is taken to sit at the colour camera unless `depth_to_colour` is set (off by about 15 mm on the head camera) |
| Clearance checks | the harness asks the simulator before a whole-body command | the daemon's motion guard decides; a refusal comes back as the command's answer |
| Base | simulated joints, exact odometry | velocity commands; pose from the in-house base firmware's odometry, else not measured |
| Time | simulated, lockstep | wall clock; a model's thinking time passes on the robot |
| Reset between episodes | automatic | by hand |
