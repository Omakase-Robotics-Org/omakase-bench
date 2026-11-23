"""Scene behaviour shared by every backend: latched button presses, lamps, door controllers.

A backend reports raw joint displacements each control tick; this class turns
them into the articulated state the oracle reports and into door targets the
backend must follow, so every engine shows the same behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .task import Articulated, SceneSpec


@dataclass
class _Latch:
    pressed_at: float | None = None


@dataclass
class SceneLogic:
    scene: SceneSpec
    latches: dict[str, _Latch] = field(default_factory=dict)

    def reset(self) -> None:
        self.latches = {a.name: _Latch() for a in self.scene.articulated if a.type == "button"}

    def button_pressed(self, a: Articulated, q: float) -> bool:
        travel = float(a.params.get("travel", 0.008))
        return q >= float(a.params.get("press_threshold", 0.6)) * travel

    def update(self, t: float, q: dict[str, float]) -> None:
        for a in self.scene.articulated:
            if a.type == "button" and self.button_pressed(a, q.get(a.name, 0.0)):
                latch = self.latches[a.name]
                if latch.pressed_at is None:
                    latch.pressed_at = t

    def pressed_ever(self, name: str) -> bool:
        latch = self.latches.get(name)
        return latch is not None and latch.pressed_at is not None

    def door_target(self, a: Articulated, t: float) -> float:
        """Fraction open (0..1) a powered door is commanded to."""
        trigger = a.params.get("opens_with")
        if not trigger:
            return float(a.params.get("open", 0.0))
        latch = self.latches.get(trigger)
        if latch is None or latch.pressed_at is None:
            return 0.0
        dt = t - latch.pressed_at - float(a.params.get("delay", 2.0))
        if dt < 0:
            return 0.0
        hold = float(a.params.get("hold_open", 1e9))
        speed = float(a.params.get("speed", 0.25))     # fraction per second
        if dt < hold:
            return min(1.0, dt * speed)
        return max(0.0, 1.0 - (dt - hold) * speed)

    def lamp(self, a: Articulated, t: float) -> bool:
        """A call button stays lit from the press until the door it calls is fully open."""
        if a.type != "button" or not self.pressed_ever(a.name):
            return False
        for door in self.scene.articulated:
            if door.params.get("opens_with") == a.name and self.door_target(door, t) >= 1.0:
                return False
        return True
