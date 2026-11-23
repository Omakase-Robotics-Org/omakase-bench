"""The ``chunks`` interface for vision-language-action policies: observation in,
a chunk of joint-position targets out. Only the interface and a stub ship in
this version; real policies (remote inference servers, local checkpoints)
implement :class:`ChunkPolicy`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ..harness.session import Session
from .base import SubjectCard


@dataclass(frozen=True)
class ChunkSpec:
    """What a policy consumes and produces."""
    joints: tuple[str, ...]                 # action layout: absolute joint-position targets
    cameras: Mapping[str, tuple[int, int]]  # camera -> (height, width)
    horizon: int = 50                       # rows the policy predicts
    execute: int = 25                       # rows executed before the next query
    hz: float = 30.0


@dataclass(frozen=True)
class ChunkObservation:
    instruction: str
    joint_positions: Mapping[str, float]
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    sim_time: float = 0.0
    #: the task's end-effector limits as a sentence ('' = the hardware range)
    limits: str = ""
    #: clamps applied to the previous chunk, e.g. "left: opening clamped to 40 mm"
    notes: tuple[str, ...] = ()


class ChunkPolicy(Protocol):
    name: str

    def spec(self, robot) -> ChunkSpec: ...
    def act(self, obs: ChunkObservation) -> np.ndarray: ...


class HoldPolicy:
    """Predicts the current joint positions: the robot holds still. A plumbing check and a floor."""
    name = "stub/hold"

    def spec(self, robot) -> ChunkSpec:
        joints = tuple(j for a in robot.arms.values() for j in a.joints)
        return ChunkSpec(joints=joints, cameras={"head": (240, 320)}, horizon=10, execute=10, hz=10.0)

    def act(self, obs: ChunkObservation) -> np.ndarray:
        row = [obs.joint_positions[j] for j in self._joints]
        return np.tile(row, (10, 1))


class ChunkSubject:
    kind = "vla"
    interface = "chunks"

    def __init__(self, policy: Any, max_chunks: int = 20):
        self.policy, self.max_chunks = policy, max_chunks
        self.name = policy.name

    def card(self) -> SubjectCard:
        return SubjectCard(self.name, self.kind, self.interface, detail={"max_chunks": self.max_chunks})

    def run(self, session: Session) -> str:
        from ..harness.end_effectors import describe_limits
        spec = self.policy.spec(session.robot)
        self.policy._joints = spec.joints
        limits = describe_limits(session.gripper)
        notes: tuple[str, ...] = ()
        for _ in range(self.max_chunks):
            q = session.backend.joint_positions()
            images = {c: session.backend.render(c, w, h) for c, (h, w) in spec.cameras.items()}
            obs = ChunkObservation(session.episode.instruction, q, images, session.backend.sim_time(), limits, notes)
            chunk = np.asarray(self.policy.act(obs))[: spec.execute]
            notes = tuple(session.execute_chunk(chunk, list(spec.joints), spec.hz).get("gripper", ()))
        return "chunks_exhausted"
