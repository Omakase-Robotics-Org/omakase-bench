"""Inverse kinematics behind one interface.

A solver is anything with :class:`IKSolver`'s ``solve_ik``. Solvers register by name; a robot names its
default in ``robot.yaml`` (``ik.solver``) and a task may name another for its end-effector rows
(``ee.ik``). ``dls`` is the damped least-squares solver below, which runs on the robot-only model of
:class:`~omakase_bench.harness.kinematics.model.Kinematics` and so works for any robot description.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .model import Kinematics


@dataclass(frozen=True)
class IKResult:
    ok: bool
    q: tuple[float, ...]
    position_error: float      # m
    angle_error: float         # rad
    reason: str = ""


class IKSolver(Protocol):
    def solve_ik(self, arm: str, position: Sequence[float], rotation: np.ndarray | None,
                 q_current: Mapping[str, float], *, pos_tol: float = ..., ang_tol: float = ...,
                 iters: int = ..., restarts: int = ...) -> IKResult: ...


class DlsSolver:
    """Damped least squares on the site Jacobian, from several seeds; joint limits are clamped each step."""

    def __init__(self, kin: "Kinematics"):
        self.kin = kin

    def solve_ik(self, arm: str, position: Sequence[float], rotation: np.ndarray | None,
                 q_current: Mapping[str, float], *, pos_tol: float = 0.004,
                 ang_tol: float = 0.05, iters: int = 200, restarts: int = 30,
                 seeds: Sequence[np.ndarray] | None = None) -> IKResult:
        kin = self.kin
        spec_arm = kin.spec.arms[arm]
        joints = spec_arm.joints
        site = f"{spec_arm.end_effector}_tcp"
        target = np.array(position, float)
        lo = np.array([kin._range[j][0] for j in joints])
        hi = np.array([kin._range[j][1] for j in joints])
        dofs = [kin._dofadr[j] for j in joints]
        rng = np.random.default_rng(0)
        if seeds is None:
            seeds = [np.array([q_current.get(j, 0.0) for j in joints]), np.array(spec_arm.home)]
            seeds += [rng.uniform(lo * 0.8, hi * 0.8) for _ in range(restarts)]
        else:
            seeds = [np.asarray(v, float) for v in seeds]
        best: IKResult | None = None
        jacp = np.zeros((3, kin.model.nv))
        jacr = np.zeros((3, kin.model.nv))
        kin.set_q(q_current)
        site_id = kin.model.site(site).id
        for seed in seeds:
            q = np.clip(seed, lo, hi)
            for _ in range(iters):
                for j, v in zip(joints, q):
                    kin.data.qpos[kin._qadr[j]] = v
                mujoco.mj_kinematics(kin.model, kin.data)
                mujoco.mj_comPos(kin.model, kin.data)
                pos = kin.data.site_xpos[site_id]
                err_p = target - pos
                rows = [err_p]
                mujoco.mj_jacSite(kin.model, kin.data, jacp, jacr, site_id)
                jac = [jacp[:, dofs]]
                err_r = np.zeros(3)
                if rotation is not None:
                    cur = kin.data.site_xmat[site_id].reshape(3, 3)
                    r_err = rotation @ cur.T
                    quat = np.zeros(4)
                    mujoco.mju_mat2Quat(quat, r_err.flatten())
                    if quat[0] < 0:
                        quat = -quat
                    err_r = 2.0 * quat[1:]
                    rows.append(0.3 * err_r)
                    jac.append(0.3 * jacr[:, dofs])
                e = np.concatenate(rows)
                jm = np.vstack(jac)
                if np.linalg.norm(err_p) < pos_tol * 0.25 and np.linalg.norm(err_r) < ang_tol * 0.25:
                    break
                lam = 1e-3 + 0.05 * float(np.dot(e, e))
                dq = jm.T @ np.linalg.solve(jm @ jm.T + lam * np.eye(jm.shape[0]), e)
                step = np.max(np.abs(dq))
                if step > 0.3:
                    dq *= 0.3 / step
                q = np.clip(q + dq, lo, hi)
            for j, v in zip(joints, q):
                kin.data.qpos[kin._qadr[j]] = v
            mujoco.mj_kinematics(kin.model, kin.data)
            pos = kin.data.site_xpos[site_id]
            pe = float(np.linalg.norm(target - pos))
            ae = 0.0
            if rotation is not None:
                cur = kin.data.site_xmat[site_id].reshape(3, 3)
                ae = float(np.arccos(np.clip((np.trace(rotation @ cur.T) - 1) / 2, -1, 1)))
            ok = pe <= pos_tol and ae <= ang_tol
            cand = IKResult(ok, tuple(float(v) for v in q), pe, ae)
            if best is None or (ok and not best.ok) or (ok == best.ok and self._cost(cand, seeds[0]) < self._cost(best, seeds[0])):
                best = cand
            if ok and np.allclose(seed, seeds[0]):
                break
        kin.set_q(q_current)
        assert best is not None
        if not best.ok:
            return IKResult(False, best.q, best.position_error, best.angle_error,
                            f"unreachable: closest pose is {best.position_error * 1000:.0f} mm and "
                            f"{np.degrees(best.angle_error):.0f} deg from the target")
        return best

    def solve_ik_near(self, arm: str, position: Sequence[float], rotation: np.ndarray | None,
                      q_current: Mapping[str, float], *, tries: int = 24, **tolerances) -> IKResult:
        """The solution nearest the current joints (least largest joint change), from descents seeded at
        the current joints and at random perturbations of them, so that the arm stays on its current
        branch instead of flipping a joint by half a turn."""
        kin = self.kin
        joints = kin.spec.arms[arm].joints
        lo = np.array([kin._range[j][0] for j in joints])
        hi = np.array([kin._range[j][1] for j in joints])
        now = np.array([q_current.get(j, 0.0) for j in joints])
        rng = np.random.default_rng(0)
        best: IKResult | None = None
        closest: IKResult | None = None
        for k in range(tries + 1):
            seed = now if k == 0 else np.clip(now + rng.normal(0.0, 0.2 + 0.8 * k / tries, len(joints)), lo, hi)
            res = self.solve_ik(arm, position, rotation, q_current, seeds=[seed], **tolerances)
            if not res.ok:
                closest = res if closest is None or res.position_error < closest.position_error else closest
                continue
            if best is None or np.max(np.abs(np.array(res.q) - now)) < np.max(np.abs(np.array(best.q) - now)):
                best = res
        return best if best is not None else closest

    @staticmethod
    def _cost(r: IKResult, q0: np.ndarray) -> float:
        return r.position_error * 10 + r.angle_error + 0.05 * float(np.abs(np.array(r.q) - q0).sum())


#: name -> factory(kinematics)
IK_SOLVERS: dict[str, Callable[["Kinematics"], IKSolver]] = {"dls": DlsSolver}


def register_ik_solver(name: str, factory: Callable[["Kinematics"], IKSolver]) -> None:
    IK_SOLVERS[name] = factory


def ik_solver(name: str, kin: "Kinematics") -> IKSolver:
    if name not in IK_SOLVERS:
        raise ValueError(f"unknown IK solver {name!r}; registered: {', '.join(sorted(IK_SOLVERS))}")
    return IK_SOLVERS[name](kin)


def solve_near(solver: IKSolver, kin: "Kinematics", arm: str, position: Sequence[float], rotation: np.ndarray | None,
               q_current: Mapping[str, float], **options) -> IKResult:
    """The solution nearest the current joints, by the solver's own method when it has one."""
    near = getattr(solver, "solve_ik_near", None) or DlsSolver(kin).solve_ik_near
    return near(arm, position, rotation, q_current, **options)
