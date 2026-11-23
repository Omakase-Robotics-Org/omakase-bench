# Adding a robot

A robot is a folder with a URDF and a `robot.yaml`. Nothing in the harness or the backends names a
robot's joints or links, so no code changes. `robots/arm6/` is the small example (one arm, boxes only);
`robots/d1/` is the full one (two arms, mobile base, lift, neck, meshes, its own command API).

## 1. Make the folder

```
src/omakase_bench/robots/<name>/
  <name>.urdf          kinematics, collision geometry, inertias
  robot.yaml           what the URDF does not say
  meshes/              optional; primitives are enough
```

A folder anywhere else works too: every place that takes a robot name takes a directory.

The URDF needs, per moving link, an `<inertial>`, a `<collision>` and (to see it) a `<visual>`; per joint,
limits. Frames: the base frame the models work in is x forward, y left, z up from the floor. For each
camera add a link whose frame is the optical frame (z forward, x right, y down).

## 2. Write robot.yaml

The smallest useful file (`robots/arm6/robot.yaml`, shortened):

```yaml
name: arm6
version: 1
urdf: arm6.urdf
base_link: base_footprint
arms:
  arm:
    joints: [joint1, joint2, joint3, joint4, joint5, joint6]
    end_effector: gripper
    home: home
end_effectors:
  gripper:
    type: parallel_gripper
    link: tool_link
    tcp_link: tcp
    approach_axis: [1.0, 0.0, 0.0]
    fingers:
      - {joint: finger_r_joint, open: 0.0, closed: 0.04}
      - {joint: finger_l_joint, open: 0.0, closed: 0.04}
    pad_links: [finger_r_link, finger_l_link]
    max_opening: 0.080
cameras:
  head: {link: head_camera_optical, fovy: 50.0, width: 640, height: 480}
poses:
  home:
    joints: {joint1: 0.0, joint2: -1.2, joint3: 1.9, joint4: 0.0, joint5: 0.8708, joint6: 0.0}
```

### Every field

| Field | Required | Meaning |
| --- | --- | --- |
| `name`, `version` | yes | identity; recorded with every result together with a hash of the files |
| `urdf` | yes | path of the URDF, relative to the folder |
| `visuals`, `default_visual` | no | named URDF variants with the same kinematics and collisions (`mesh`, `primitive`) |
| `base_link` | no | the link whose frame is the base frame |
| `mobile_base` | no | `joints: {x, y, yaw}` (planar joints in the URDF), `drive: differential` or `holonomic`, `max_speed` (m/s), `max_yaw_rate` (rad/s), `footprint` (polygon, m). Omit for a fixed base. |
| `torso_lift` | no | `joint`, `range` (m), `speed` (m/s) |
| `neck` | no | `pan`, `tilt`: joint names |
| `arms.<name>.joints` | yes | the IK chain, base to tip |
| `arms.<name>.end_effector` | yes | a key of `end_effectors` |
| `arms.<name>.home` | yes | a pose name, or one number per joint |
| `arms.<name>.max_joint_speed` | no | rad/s, default 1.0 |
| `arms.<name>.side` | no | `left` or `right`: which side of the base the arm serves when a program picks an arm by a target's y. Default: the arm's name if it is `left` or `right`, else none (the arm serves both sides). |
| `arms.<name>.self_collision_from` | no | first joint whose links are checked against the robot's own body. Links before it sit at the mount and touch the body by construction. Default: the third joint. |
| `arms.<name>.leave_home_first` | no | joints that move first when the arm leaves home in stages |
| `end_effectors.<name>.type` | yes | `parallel_gripper` |
| `…link`, `tcp_link` | yes | the gripper's base link; the link at the tool centre point, midway between the fingertip pads |
| `…approach_axis` | yes | the direction the fingers point, in the TCP frame |
| `…fingers` | yes | per driven finger joint its `open` and `closed` value. The first finger's joint axis is the jaw-closing axis. |
| `…pad_links` | yes | the links that may touch an object; they get high friction and count as "held" when all touch it |
| `…max_opening` | yes | m between the pads when open |
| `…force` | no | N (or N m) per finger actuator, default 20 |
| `…pad_length`, `pad_band` | no | length of a pad from the jaw tip, and the part of it a grasp may use (m) |
| `…fingertip_offset` | no | `{open, closed}`: distance of the fingertip centre past the TCP (m) |
| `…tip_reach` | no | the fingertip distance past the TCP that scripted programs plan with; default `fingertip_offset.open` |
| `…yaw_zero_deg` | no | wrist angle that end-effector rows call yaw 0 |
| `cameras.<name>` | no | `link` (optical frame), `fovy` (deg), `width`, `height`, `depth_range` (m), `description` |
| `poses.<name>` | no | `joints: {joint: value}` and optionally `lift`. `home` is where episodes start. |
| `collision.exempt_geoms` | no | name pattern of collision geoms left out of the self-clearance check, default `*_exempt` |
| `actuation` | no | position servos of the simulated joints as `[kp, force limit]`: `base_linear`, `base_yaw`, `lift`, `arm_proximal`, `arm_distal`, `other`; `arm_proximal_joints` (how many joints of an arm count as proximal), `finger_kp`, `base_lag_s` |
| `ik.solver` | no | registered name of the IK solver, default `dls` |
| `api` | no | the robot's own command API: `spec` (OpenAPI document), `adapter`, `driver`, and what they read. See `docs/architecture.md`. |
| `real` | no | on-robot defaults of the real backend: `cameras` (`v4l2` devices, `iceoryx` topics, `depth` topics per camera name), `calibration` (file), `lease`. See `docs/real_robot.md`. |
| `defaults` | no | `lift` and `neck: {pan, tilt}` at reset |

### Named poses

`home` is required in practice: every arm starts there. Choose it with the gripper above the workspace
and away from joint limits; `robot check` prints where the TCP is. Tasks can name other poses for a
robot's start and idle configuration (`scene.robot.home`, `robots[].start_pose`, `idle_pose`).

### IK

The default solver (`dls`, damped least squares on the kinematic model) works from the chain in
`arms.<name>.joints` and the joint limits of the URDF. To use another, register it and name it:

```python
from omakase_bench.harness.kinematics.ik import register_ik_solver
register_ik_solver("mine", lambda kinematics: MySolver(kinematics))   # needs solve_ik(arm, position, rotation, q_current, ...)
```

```yaml
ik:
  solver: mine
```

### Collision and safety

- Robot links collide with fixtures and objects, never with the floor; arms are checked against the
  robot's own body from `self_collision_from` on.
- The end-effector interface refuses a step that would turn a joint too far at once or come within the
  task's clearance of anything; the limits are in the task's `ee:` block, not here.
- A task can narrow a gripper's opening and force (`gripper:` in the task YAML); the description states
  the hardware range.

## 3. Check it

```bash
uv run omakase-bench robot check <name or directory> --out /tmp/stills
```

```
robot arm6 v1: 8 joints, base fixed, lift no, neck no, cameras head, wrist, IK solver dls
  arm arm: 6 joints, side any, home TCP (0.379, 0.000, 0.934), IK reaches 54/54 targets within 0.1 m of it
  wrote /tmp/stills/arm6_outside.png
  wrote /tmp/stills/arm6_head.png
  wrote /tmp/stills/arm6_wrist.png
```

It loads the description, builds the model, solves IK for 54 poses around each arm's home with the gripper
pointing down, and renders an outside view and every camera. Exit status 1 means an arm reached nothing:
look at `approach_axis`, the finger order and the home pose first.

## 4. Run a task on it

```bash
uv run omakase-bench run --task core/cube_to_tray --model scripted/oracle --robot <name>
```

`--robot` runs a single-robot task on another robot: same scene and success check, the robot at its own
home. The scripted player reads ground truth, so a failure here is about the description (reach, gripper
geometry, friction), not about a model. Then name a model instead of `scripted/oracle`.

Tasks with a `robots:` block place their robots themselves and take no override.

## 5. Tests

`tests/robots/arm6/test_arm6.py` is the pattern: the description loads and passes the check, the
model-facing text names its arm, published tasks run on it with the scripted player.
