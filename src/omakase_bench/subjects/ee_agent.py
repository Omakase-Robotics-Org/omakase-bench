"""The ``ee`` interface: a language model as the step-by-step controller of the arms.

Every step the model receives one text row of robot state (joint angles,
gripper commands, fingertip poses from forward kinematics), the recent rows of
the episode and one image of the robot's cameras side by side. It replies with
one action row of fingertip targets (:mod:`omakase_bench.harness.ee_control`);
the harness solves the inverse kinematics, limits the step and reports targets
without a solution in the next row. There are no demonstrations and no
``list_objects``: the task states the cell layout a real installation would
know, and objects are found in the image.

With depth lookup a reply may also name pixels (``locate: <view> <u> <v>``);
the harness answers each with the base-frame point seen there
(:mod:`omakase_bench.harness.depth`). Lookups take no step.

Task YAML, ``ee`` block (all optional but ``views``):

    horizon      steps per episode (default 40)
    start        arm -> TCP x, y, z (m) and yaw (deg) taken before the first step
    views        view name -> camera, in image order
    view_width   pixels per view (default 320; the height is 3/4 of it)
    window       episode rows shown (default 8, 0 = all)
    lookups      depth lookup on (default true)
    rules        what the task demands beyond the episode's instruction (prepended to it)
    limits       EELimits fields (step size, yaw step, IK tolerances, step time)
    ik           name of the IK solver of the row steps (default ``dls``; ``ee_control.register_ik_solver``)
    grasp        depth_min and margin (m) of the grasp-depth guidance
    tilt         {max_deg: N}: rows may tilt the gripper about its jaw-closing axis (``t<deg>``)
    row_steps    a row is repeated for up to this many steps until the fingertips arrive (default 1)
    say          keywords of referee lines (``move: e4``, ``done``) a reply may contain; the session's
                 referee answers them in the next prompt (multi-robot tasks with a referee)
    whole_body   {base, lift, head: true, max_forward_m, max_turn_deg, max_lift_step_m, clearance_m}: lines
                 ``base: <mm> [<deg>]``, ``lift: <mm>``, ``head: <pan> <tilt>`` move the rest of the robot
    base_noise   {lateral_per_m, heading_deg_per_m, heading_deg_per_rad}: execution noise of base motions
    transcript   true: every model call's changing prompt text (episode rows, referee answers, the request)
                 and the provider's readable summary of the model's reasoning go into the trace with the
                 reply and the call's wall-clock seconds (default false: the reply and the seconds only)
"""
from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..harness.session import Session
from ..models.base import Image, Message, ModelAdapter, Request, Text
from ..harness import depth
from ..harness.ee_control import (EEController, EELimits, ParseError, WholeBody, arm_keys, parse_row, parse_rows)
from .agents import _account, _model_failure
from .base import SubjectCard, price_detail

SYSTEM = ("You are the low-level controller of the arms of a mobile manipulator. Every step you receive the "
          "robot state, the episode so far and the current camera image, and you output the next action row "
          "in exactly the specified format.")
MAX_LOOKUPS = 3            # locate lines answered per reply
MAX_LOOKUP_ONLY = 2        # consecutive replies that may contain lookups and no action row
LOOKUPS_SHOWN = 12         # lookup results kept in the prompt (most recent)
MAX_PARSE_FAILURES = 3     # consecutive unusable replies that end the episode
MAX_SAY_ONLY = 4           # consecutive replies with referee lines and no action row
BAR = 4                    # px, black bar between views


@dataclass(frozen=True)
class EEConfig:
    horizon: int = 40
    start: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    #: reach ``start`` in stages (robot.yaml ``leave_home_first``) instead of one joint-space move
    start_staged: bool = False
    views: Mapping[str, str] = field(default_factory=dict)
    view_width: int = 320
    window: int = 8
    lookups: bool = True
    rules: str = ""
    limits: EELimits = field(default_factory=EELimits)
    #: grasp-depth guidance: the shallowest depth that still carries an object, and the margin
    #: kept from the pad band for position error of a camera-derived height (m)
    grasp_depth_min: float = 0.010
    grasp_margin: float = 0.007
    #: largest tilt of tilted grasps (deg); 0: the gripper always points straight down
    tilt_max_deg: float = 0.0
    #: steps a row may take until its target is reached
    row_steps: int = 1
    #: keywords of the lines a reply may address to the referee
    say: tuple[str, ...] = ()
    #: base, torso and head commands the task allows (none by default)
    body: WholeBody = field(default_factory=WholeBody)
    #: record every call's changing prompt text and the model's thinking summary in the trace
    transcript: bool = False

    @property
    def view_height(self) -> int:
        return self.view_width * 3 // 4

    @classmethod
    def from_task(cls, raw: Mapping[str, Any], robot) -> "EEConfig":
        d = dict(raw.get("ee") or {})
        grasp = d.pop("grasp", {}) or {}
        views = d.pop("views", None) or {c: c for c in robot.cameras}
        unknown = set(views.values()) - set(robot.cameras)
        if unknown:
            raise ValueError(f"ee.views names unknown camera(s) {sorted(unknown)}")
        return cls(horizon=int(d.pop("horizon", 40)),
                   start={a: tuple(float(v) for v in p) for a, p in (d.pop("start", {}) or {}).items()},
                   start_staged=d.pop("start_path", "joint") == "staged",
                   views=views, view_width=int(d.pop("view_width", 320)), window=int(d.pop("window", 8)),
                   lookups=bool(d.pop("lookups", True)), rules=" ".join(str(d.pop("rules", "")).split()), limits=EELimits.from_mapping(d.pop("limits", None)),
                   grasp_depth_min=float(grasp.get("depth_min", 0.010)), grasp_margin=float(grasp.get("margin", 0.007)),
                   tilt_max_deg=float((d.pop("tilt", None) or {}).get("max_deg", 0.0)),
                   row_steps=int(d.pop("row_steps", 1)), say=tuple(str(k).lower() for k in d.pop("say", ()) or ()),
                   body=WholeBody.from_mapping(d.pop("whole_body", None), d.pop("base_noise", None)),
                   transcript=bool(d.pop("transcript", False)))


def _mm(v: float) -> int:
    return int(round(v * 1000))


def _png(img: np.ndarray) -> bytes:
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(img).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_AXES = {(0, 1): "away from the robot (+x)", (0, -1): "towards the robot (-x)", (1, 1): "the robot's left (+y)",
         (1, -1): "the robot's right (-y)", (2, 1): "up (+z)", (2, -1): "down (-z)"}


def _direction(v: np.ndarray, flatten: bool) -> str:
    """The base-frame axis a camera image direction points along; on the ground plane for a camera
    that looks down, where the image shows a floor or table."""
    v = np.array(v, float)
    if flatten:
        v[2] = 0.0
    i = int(np.argmax(np.abs(v)))
    return _AXES[(i, 1 if v[i] > 0 else -1)]


class EEAgent:
    kind = "agent"
    interface = "ee"

    def __init__(self, model: ModelAdapter, spec: str):
        self.model, self.spec = model, spec
        self.name = f"{spec}:ee"

    def card(self) -> SubjectCard:
        return SubjectCard(self.name, self.kind, self.interface, model=self.spec,
                           detail={"action": "fingertip targets, one row per step", "demonstrations": 0,
                                   "max_lookups_per_reply": MAX_LOOKUPS, **price_detail(self.model)})

    # --- prompt -------------------------------------------------------------------------------
    def _wrist_cameras(self, session: Session) -> dict[str, str]:
        """camera -> arm, for cameras mounted on an end effector."""
        kin, robot = session.api.kin, session.robot
        out = {}
        for name, cam in robot.cameras.items():
            b = kin.model.body(cam.link).id
            chain = set()
            while b > 0:
                chain.add(kin.model.body(b).name)
                b = int(kin.model.body_parentid[b])
            for arm in robot.arms:
                if robot.hand_of(arm).link in chain:
                    out[name] = arm
        return out

    def _image_text(self, session: Session, cfg: EEConfig) -> str:
        kin, robot = session.api.kin, session.robot
        kin.set_q(session.backend.joint_positions())
        wrist = self._wrist_cameras(session)
        parts = []
        for view, cam in cfg.views.items():
            _, rot = depth.camera_pose(kin, robot, cam)
            down = abs(rot[2, 2]) > 0.3
            where = (f"the camera on the {wrist[cam]} gripper, looking along its approach direction"
                     if cam in wrist else f"the {cam} camera")
            parts.append(f"'{view}' = {where}{f' ({robot.cameras[cam].description})' if robot.cameras[cam].description else ''}; "
                         f"the top of the image is {_direction(-rot[:, 1], down)} and image left is "
                         f"{_direction(-rot[:, 0], down)}")
        return (f"Camera image: {len(cfg.views)} views of {cfg.view_width}x{cfg.view_height} pixels side by side, "
                "separated by black bars, from left to right: " + ". ".join(parts) + ". The directions are those "
                "of the start pose; a wrist view turns with the gripper's yaw.")

    def _lookup_text(self, session: Session, cfg: EEConfig) -> str:
        views = depth.depth_views(session.backend, cfg.views)
        ranges = "; ".join(f"{view}: {_mm(r[0])}-{_mm(r[1])} mm" for view, cam in views.items()
                           if (r := session.robot.cameras[cam].depth_range))
        first = next(iter(views))
        return (
            f"Depth lookup: the cameras are depth cameras. Besides the action line, a reply may contain up to "
            f"{MAX_LOOKUPS} lines 'locate: <view> <u> <v>' (view = {', '.join(views)}; u right, v down, in pixels "
            f"from the top-left corner of that view's {cfg.view_width}x{cfg.view_height} image). For each, the harness "
            "reads that camera's depth at the pixel of the current image and converts it to the base frame using the "
            "camera's intrinsics and its pose from the joint encoders. The result is the surface point seen at that "
            f"pixel and appears in the next prompt, for example 'lookup {first}(150,90) -> x 412 y -205 z 790 mm'. A "
            "pixel on the top face of an object returns a point on the top face. 'no depth' means the pixel is outside "
            f"the view or outside the sensor's range from the camera ({ranges}). Lookups take no time step. A reply "
            f"with locate lines and no action line leaves the robot where it is; at most {MAX_LOOKUP_ONLY} such "
            "replies in a row are accepted. Earlier lookup results stay listed; a point stays valid until the object "
            "moves.")

    def prefix(self, session: Session, cfg: EEConfig) -> str:
        """Instructions for the episode. Every number is from the robot preset or the task's cell
        layout; object positions are not stated."""
        robot, kin = session.robot, session.api.kin
        q = session.backend.joint_positions()
        arms = list(robot.arms)
        keys = arm_keys(arms)
        hand = robot.hand_of(arms[0])
        n = len(robot.arms[arms[0]].joints)
        lim = cfg.limits
        fixed = ["The mobile base stays where it is"] if robot.mobile_base else []
        if robot.lift_joint:
            fixed.append(f"the torso lift at {_mm(q[robot.lift_joint])} mm")
        if robot.neck:
            fixed.append("the head at " + ", ".join(f"{k} {round(q[j], 2) + 0.0:.2f} rad" for k, j in robot.neck.items()))
        rest = ", ".join(fixed) + ". " if fixed else ""
        if cfg.body.any:                    # what a robot API command can move is not said to stay put
            state = [x.replace("lift at", "lift is at") for x in fixed[1 if robot.mobile_base else 0:]]
            rest = "At the first step " + " and ".join(state) + ". " if state else ""
        row_names = " / ".join(f"q{k}" for k in keys.values())
        lines = [
            "Task: " + " ".join(x for x in (cfg.rules, session.episode.instruction) if x),
            f"Robot: {robot.name}, a mobile manipulator. " + rest
            + f"You command the {' and '.join(arms)} arms ({n} revolute joints each, J1..J{n}) and their parallel "
            f"grippers ({_mm(hand.max_opening)} mm stroke, fingertip pads). Use the arm that suits the object; the "
            "other arm may stay where it is.",
            "Base frame: all positions are in mm in the robot base frame. Origin on the floor at the centre of the "
            "mobile base; x forward, y to the robot's left, z up.",
        ]
        if session.episode.values.get("layout"):
            lines.append(str(session.episode.values["layout"]))
        if session.gripper:
            grip = ("Grippers: they always point straight down. The fingertip centre is the midpoint between the two "
                    f"jaw tips. The hardware opens the pad faces up to {_mm(hand.max_opening)} mm apart; this task "
                    "limits them: " + "; ".join(self._limit_text(arm, lim) for arm, lim in session.gripper.items())
                    + ". ")
        else:
            grip = ("Grippers: they always point straight down. The fingertip centre is the midpoint between the two "
                    f"jaw tips. Open, the pad faces are {_mm(hand.max_opening)} mm apart; closed, they meet. ")
        if hand.pad_band > 0:
            grip += (f"The pads are the last {_mm(hand.pad_length)} mm of each jaw; a grasp is on the pads when the "
                     f"object touches the jaws only within {_mm(hand.pad_band)} mm of the jaw tips. ")
        grip += ("Yaw is the gripper's rotation about the vertical in degrees, counter-clockwise seen from above; at "
                 "yaw 0 (the start pose) the jaws open and close along y.")
        if cfg.tilt_max_deg:
            grip += (f" Tilted grasps: after the yaw a row may give a tilt t<deg> (up to {int(cfg.tilt_max_deg)} deg, "
                     "for example 't25'): the gripper then leans about its jaw-closing axis, the jaws stay level and the "
                     "fingertips point past the wrist, away from the robot (perpendicular to the jaws, on the +x side; "
                     "negative values lean towards the robot). The position is still that of the fingertip centre. A "
                     "tilt reaches further from the robot; without it the gripper points straight down. The "
                     "observation row shows the measured tilt after the yaw.")
        lines.append(grip)
        travel = hand.fingertip_offset[1] - hand.fingertip_offset[0]
        if hand.pad_band > 0:
            lo_d, hi_d = _mm(cfg.grasp_depth_min), _mm(hand.pad_band - travel - cfg.grasp_margin)
            lines.append(
                f"Grasp depth: closing lowers the jaw tips by up to {_mm(travel)} mm, so the contact reaches higher "
                "up the jaws than the open fingertip position suggests, and a position taken from the cameras can be "
                "several mm off. To pick up a standing object from above, start closing with the fingertip centre "
                f"{lo_d} to {hi_d} mm below the top of the object (about {(lo_d + hi_d) // 2} mm), not at its "
                f"mid-height. Deeper than that, the contact leaves the {_mm(hand.pad_band)} mm band and the grasp does "
                "not count.")
        lines.append(
            f"Observation row: {row_names} = arm joint angles J1..J{n} in integer degrees; g = gripper commands "
            "(0 open, 1 closed); p = x y z yaw of the fingertip centre, computed from the measured joints; each "
            f"suffixed with the arm's letter ({', '.join(f'{k} = {a}' for a, k in keys.items())}). 'ik<letter>: "
            "unreachable' means the previous target of that arm had no inverse-kinematics solution and the arm did "
            "not move.")
        lines.append(self._image_text(session, cfg))
        if cfg.lookups:
            lines.append(self._lookup_text(session, cfg))
        if cfg.body.any:
            lines.append(self._body_text(session, cfg))
        if cfg.say:
            lines.append(
                "Referee lines: a reply may also contain lines addressed to the referee, each starting with one of "
                + ", ".join(f"'{k}:'" for k in cfg.say) + " followed by its text. The referee's answers appear in the "
                "next prompt and stay listed for the rest of the turn. A reply with referee lines and no action line "
                f"leaves the robot where it is; at most {MAX_SAY_ONLY} such replies in a row are accepted.")
        example = " | ".join(f"{k} hold" if i else f"{k} 400 170 900 0 0" for i, k in enumerate(keys.values()))
        lo = [kin._range[j][0] for j in robot.arms[arms[0]].joints]
        hi = [kin._range[j][1] for j in robot.arms[arms[0]].joints]
        limits = " ".join(f"J{i + 1}[{int(round(math.degrees(a)))},{int(round(math.degrees(b)))}]"
                          for i, (a, b) in enumerate(zip(lo, hi)))
        lines += [
            "Action row: a: " + " | ".join(f"{k} <x> <y> <z> <yaw> <g>" for k in keys.values()) + "  = for each arm "
            "the target position of its fingertip centre (mm), the target yaw (deg) and the gripper command (0 open, "
            "1 closed). Any arm's five values may be replaced by the word hold, which keeps that arm and its gripper "
            f"as they are, for example 'a: {example}'.",
            f"The harness solves the inverse kinematics. Each step moves the fingertip centre at most "
            f"{_mm(lim.max_step_m)} mm and the yaw at most {int(lim.max_yaw_step_deg)} deg towards the target, so a far "
            "target takes several steps: " + (
                "repeat the target until the p values show it has been reached. " if cfg.row_steps <= 1 else
                f"the harness repeats a row for up to {cfg.row_steps} steps until the fingertips arrive (within 3 mm "
                "and 3 deg), the target has no IK solution, or the fingertips stop moving (for example against an "
                "object); the next row shows where they are. ")
            + "The gripper command applies in the same step; the jaws need one step to open or close fully.",
            f"Each step lasts {lim.step_seconds:g} s. Joint limits (deg): {limits}.",
            (f"A turn lasts at most {cfg.horizon} steps." if cfg.say else
             f"An episode lasts at most {cfg.horizon} steps and ends as soon as the task succeeds."),
            "No demonstrations are available; work from the task description, the camera image and the numbers above.",
            "",
        ]
        if session.gripper:
            first = keys[arms[0]]
            width_example = example.replace(f"{first} 400 170 900 0 0", f"{first} 400 170 900 0 w30")
            lines.insert(-5, (
                "Jaw opening: with this task's gripper limits the gripper command may also be an opening in mm written "
                f"w<mm>, for example 'a: {width_example}'. An opening outside the limits is clamped and the next "
                f"observation row says so, for example 'g{first}: opening clamped to "
                f"{_mm(session.gripper[arms[0]].max_opening)} mm'."))
        return "\n".join(lines)

    @staticmethod
    def _body_text(session: Session, cfg: EEConfig) -> str:
        from ..harness.command_api import command_adapter
        b, robot = cfg.body, session.robot
        adapter = command_adapter(robot)
        if adapter is None:
            raise ValueError(f"ee.whole_body needs a command API, and robot {robot.name} has none (robot.yaml api)")
        sides, lines = adapter.describe(robot, b, arm_keys(list(robot.arms)))
        text = ("Robot API commands: besides the action line a reply may contain these lines; each is sent as requests "
                f"of the robot's HTTP API, in order, before the action line ({sides}): " + "; ".join(lines) + ". "
                f"Every motion is also checked before it starts: one that would bring the robot within "
                f"{_mm(b.clearance_m)} mm of the table, the other robot, its own body or its other arm is refused, "
                "naming the closest pair; contact with the pieces is not checked. Between commands every joint holds "
                "its last target. The next observation row lists each request and the API's answer.")
        if b.base:
            text += (" Positions stay in your base frame, which moves with the base: after driving 0.05 m forward, a "
                     "point that was at x 400 is at x 350. The observation row shows B = the odometry of the base "
                     "(commanded motion since step t00: forward mm, left mm, heading deg).")
            if b.lateral_per_m or b.heading_deg_per_m or b.heading_deg_per_rad:
                text += (f" Driving is not exact: each metre driven shifts the base sideways by about "
                         f"{_mm(b.lateral_per_m)} mm and turns it by about {b.heading_deg_per_m:g} deg (one standard "
                         f"deviation), each radian of turning by about {b.heading_deg_per_rad:g} deg, which the odometry "
                         "does not show; the cameras and depth lookups do.")
        if b.lift or b.head:
            text += " The observation row also shows the lift (m) and the head pitch and yaw (rad)."
        return text

    @staticmethod
    def _limit_text(arm: str, lim) -> str:
        text = (f"the {arm} jaws open at most {_mm(lim.max_opening)} mm and close to no less than "
                f"{_mm(lim.min_closing)} mm, g 0 opens them to {_mm(lim.preshape)} mm")
        if lim.min_closing > 0:
            text += f" and g 1 closes them to {_mm(lim.min_closing)} mm unless an object stops them first"
        if lim.max_force is not None:
            text += f", each jaw pushing with at most {lim.max_force:g} N"
        return text

    @staticmethod
    def obs_row(controller: EEController, unreachable: list[str]) -> str:
        q = controller._q()
        keys = arm_keys(controller.arms)
        parts = []
        for arm in controller.arms:
            parts += [f"q{keys[arm]} {' '.join(map(str, controller.joints_deg(arm, q)))}",
                      f"g{keys[arm]} {controller.gripper_cmd[arm]}"]
        for arm in controller.arms:
            pos, yaw = controller.tip_pose(arm, q)
            tilt = f" t{int(round(controller.tilt(arm, q)))}" if controller.max_tilt_deg else ""
            parts.append(f"p{keys[arm]} {' '.join(str(_mm(v)) for v in pos)} {int(round(yaw))}{tilt}")
        parts += [f"ik{keys[arm]}: {controller.jumps.get(arm, 'unreachable')}" for arm in controller.arms
                  if arm in unreachable]
        parts += controller.notes
        b, spec = controller.body, controller.spec
        if b.base:
            o = controller.odometry
            parts.append(f"B {_mm(o[0])} {_mm(o[1])} {int(round(math.degrees(o[2])))}")
        if b.lift and spec.lift_joint:
            parts.append(f"lift {q[spec.lift_joint]:.3f}")
        if b.head and spec.neck:
            parts.append(f"head {-q[spec.neck['tilt']]:.2f} {q[spec.neck['pan']]:.2f}" if {"tilt", "pan"} <= set(spec.neck)
                         else "head " + " ".join(f"{q[j]:.2f}" for j in spec.neck.values()))
        parts += getattr(controller, "body_notes", [])
        return " | ".join(parts)

    @staticmethod
    def tiled(session: Session, cfg: EEConfig) -> np.ndarray:
        """The camera views side by side at ``view_width`` x ``view_height``, separated by black bars."""
        bar = np.zeros((cfg.view_height, BAR, 3), np.uint8)
        views = []
        for i, cam in enumerate(cfg.views.values()):
            if i:
                views.append(bar)
            views.append(session.backend.render(cam, cfg.view_width, cfg.view_height))
        return np.concatenate(views, axis=1)

    @staticmethod
    def body_commands(controller: EEController, text: str) -> list[str]:
        """Carry out the robot-API command lines of a reply in order (``joints:``, ``joint:``, ``gripper:``,
        ``lift:``, ``head:``, ``base:``) on the simulated daemon; one feedback line per request."""
        from ..harness.command_api import command_adapter
        adapter = command_adapter(controller.spec)
        return adapter.run(controller, text) if adapter is not None else []

    @staticmethod
    def said(text: str, keywords: tuple[str, ...]) -> list[tuple[str, str]]:
        """(keyword, text) of the referee lines of a reply, in order: ``<keyword>: <text>``, or the keyword
        alone on its line. A sentence that merely begins with a keyword ("Move the knight ...") is none."""
        if not keywords:
            return []
        pat = re.compile(r"^[ \t]*(" + "|".join(re.escape(k) for k in keywords)
                         + r")[ \t]*(?::[ \t]*([^\n]*)|[.!]?[ \t]*)$", re.IGNORECASE | re.MULTILINE)
        return [(m.group(1).lower(), (m.group(2) or "").strip()) for m in pat.finditer(text)]

    @staticmethod
    def suffix(history: list[tuple[str, str | None]], obs: str, lookups: list[tuple[int, str]],
               cfg: EEConfig, referee: list[tuple[int, str]] | None = None) -> tuple[str, str]:
        """(episode rows, the request that follows the image)."""
        def row(t: int, o: str, a: str | None) -> str:
            return f"t{t:02d} | {o} -> " + ("(invalid reply, held position)" if a is None else f"a: {a}")

        lines = ["### current episode"]
        start = 0
        if cfg.window and len(history) > cfg.window:
            start = len(history) - cfg.window
            lines.append(f"(t00..t{start - 1:02d} omitted)")
        lines += [row(t, o, a) for t, (o, a) in list(enumerate(history))[start:]]
        lines.append(f"t{len(history):02d} | {obs} ->")
        ask = ""
        if referee:
            ask += "Referee answers this turn (step of your line):\n" + "\n".join(
                f"t{t:02d} {text}" for t, text in referee) + "\n"
        if lookups:
            ask += ("Depth lookups so far (step at which the image was taken):\n"
                    + "\n".join(f"t{t:02d} {text}" for t, text in lookups[-LOOKUPS_SHOWN:]) + "\n")
        if cfg.lookups:
            ask += (f"You may first write up to {MAX_LOOKUPS} lines 'locate: <view> <u> <v>'; to wait for their "
                    "results before moving, stop after them. Then output the single next action line, starting "
                    "with 'a:'.")
        else:
            ask += "Reply with the single next action line only, starting with 'a:'."
        return "\n".join(lines) + "\n", ask

    # --- episode --------------------------------------------------------------------------------
    def run(self, session: Session) -> str:
        session.model_ids = set()
        cfg = EEConfig.from_task(session.task.raw, session.robot)
        lookup_views = depth.depth_views(session.backend, cfg.views) if cfg.lookups else {}
        if cfg.lookups and not lookup_views:
            # no depth images for these views: the lookup is not offered rather than answered with nothing
            cfg = replace(cfg, lookups=False)
            session.log("ee_lookups_off", reason="the backend has no depth_camera capability"
                        if not hasattr(session.backend, "render_depth") else "none of the task's views has a depth camera")
        controller = EEController(session, cfg.limits)
        controller.max_tilt_deg = cfg.tilt_max_deg
        controller.body = cfg.body
        controller.body_notes = []
        controller.move_to_start(cfg.start, cfg.start_staged)
        session.ready()
        self.controller, self.session = controller, session
        prefix = self.prefix(session, cfg)
        session.artifacts = {"prompt_prefix.txt": prefix}
        history: list[tuple[str, str | None]] = []
        lookups: list[tuple[int, str]] = []
        unreachable: list[str] = []
        referee: list[tuple[int, str]] = []
        tilt = bool(cfg.tilt_max_deg)
        lookup_only = failures = say_only = 0
        try:
            while len(history) < cfg.horizon:
                obs = self.obs_row(controller, unreachable)
                rows, ask = self.suffix(history, obs, lookups, cfg, referee) if cfg.say else \
                    self.suffix(history, obs, lookups, cfg)
                t = len(history)
                parts = (Text(prefix, cache_prefix=True), Text(rows),
                         Image(_png(self.tiled(session, cfg)), "image/png", label=f"Current camera image (t{t:02d}):"),
                         Text(ask))
                session.before_model_call()
                reply = self.model.respond(Request(SYSTEM, [Message("user", parts)], (), max_calls=0,
                                                   thinking_summary=cfg.transcript))
                _account(session, reply, prompt=rows + ask if cfg.transcript else "")
                failure = _model_failure(reply)
                if failure:
                    raise failure
                asked = depth.parse_lookups(reply.text, lookup_views, MAX_LOOKUPS) if cfg.lookups else []
                for view, u, v in asked:
                    r = depth.lookup(session.backend, session.api.kin, session.robot, cfg.views, view, u, v,
                                     cfg.view_width, cfg.view_height)
                    lookups.append((t, r.text()))
                    session.log("lookup", t=t, view=view, u=u, v=v, point_mm=r.point_mm)
                controller.body_notes = self.body_commands(controller, reply.text) if cfg.body.any else []
                lines = self.said(reply.text, cfg.say)
                for keyword, text in lines:
                    answer = session.say(keyword, text)
                    if answer:
                        referee.append((t, answer))
                try:
                    row = parse_rows(reply.text, controller.arms, 1, controller.widths, tilt)[0]
                except ParseError as exc:
                    if (lines or controller.body_notes) and say_only < MAX_SAY_ONLY and "no action rows" in str(exc):
                        say_only += 1
                        continue
                    if asked and lookup_only < MAX_LOOKUP_ONLY and "no action rows" in str(exc):
                        lookup_only += 1
                        continue
                    session.log("invalid_reply", t=t, error=str(exc))
                    history.append((obs, None))
                    failures += 1
                    if failures >= MAX_PARSE_FAILURES:
                        return "parse_failures"
                    controller.hold()
                    unreachable = []
                    continue
                lookup_only = failures = say_only = 0
                history.append((obs, row))
                parsed = parse_row(row, controller.arms, controller.widths, tilt)
                if cfg.row_steps > 1:
                    unreachable, _ = controller.step_until(parsed, cfg.row_steps)
                else:
                    unreachable = controller.step(parsed)
            return "horizon"
        finally:
            session.artifacts["ee_summary.json"] = json.dumps(
                {"steps": len(history), "ik_unreachable": controller.unreachable_total, "lookups": len(lookups)})
