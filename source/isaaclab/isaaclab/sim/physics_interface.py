# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics interface for SimulationContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim._impl.newton_backend import NewtonBackend
from isaaclab.sim._impl.physics_backend import PhysicsBackend

from .interface import Interface

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


def set_prim_attr(prim: Usd.Prim, attr_name: str, value, value_type) -> None:
    """Set or create a prim attribute."""
    attr = prim.GetAttribute(attr_name)
    if attr is None or not attr.IsValid():
        attr = prim.CreateAttribute(attr_name, value_type)
    attr.Set(value)


class PhysicsInterface(Interface):
    """Manages USD physics scene and delegates to physics backends."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize physics scene and backends.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)
        self._sim.settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self.physics_dt = self._sim.cfg.dt
        self.rendering_dt = self._sim.cfg.dt * self._sim.cfg.render_interval
        self.physics_prim_path = self._sim.cfg.physics_prim_path
        self.device = self._sim.cfg.device
        self.backend = "torch"

        # Pre-create gravity tensor to avoid torch heap corruption issues (torch 2.1+)
        self._gravity_tensor = torch.tensor(self._sim.cfg.gravity, dtype=torch.float32, device=self._sim.cfg.device)

        self._init_usd_physics_scene()
        self._backends: list[PhysicsBackend] = [NewtonBackend(sim_context)]

    @property
    def backends(self) -> list[PhysicsBackend]:
        """List of active physics backends."""
        return self._backends

    def _init_usd_physics_scene(self) -> None:
        """Create USD physics scene with timestep and gravity."""
        # Create or get physics scene prim

        if not self._sim.stage.GetPrimAtPath(self.physics_prim_path).IsValid():
            UsdPhysics.Scene.Define(self._sim.stage, self.physics_prim_path)

        self.physics_scene_prim: Usd.Prim = self._sim.stage.GetPrimAtPath(self.physics_prim_path)
        self.physics_scene = UsdPhysics.Scene(self.physics_scene_prim)

        # Configure timestep
        set_prim_attr(
            self.physics_scene_prim,
            "physxScene:timeStepsPerSecond",
            int(1.0 / self._sim.cfg.dt),
            Sdf.ValueTypeNames.Int
        )
        self._sim.stage.SetTimeCodesPerSecond(1 / self._sim.cfg.dt)

        # Configure gravity based on stage up-axis
        up_axis = UsdGeom.GetStageUpAxis(self._sim.stage)
        gravity_magnitude = abs(self._sim.cfg.gravity[2])
        gravity_dir = {
            "Z": Gf.Vec3f(0.0, 0.0, -1.0 if self._sim.cfg.gravity[2] < 0 else 1.0),
            "Y": Gf.Vec3f(0.0, -1.0 if self._sim.cfg.gravity[1] < 0 else 1.0, 0.0),
        }.get(up_axis, Gf.Vec3f(-1.0 if self._sim.cfg.gravity[0] < 0 else 1.0, 0.0, 0.0))
        self.physics_scene.CreateGravityDirectionAttr().Set(gravity_dir)
        self.physics_scene.CreateGravityMagnitudeAttr().Set(gravity_magnitude)

    def reset(self, soft: bool) -> None:
        """Reset all physics backends.

        Args:
            soft: If True, skip full reinitialization.
        """
        for backend in self._backends:
            backend.reset(soft)

    def forward(self) -> None:
        """Update articulation kinematics on all backends."""
        for backend in self._backends:
            backend.forward()

    def step(self, render: bool = True) -> None:
        """Step all physics backends.

        Args:
            render: Unused, for interface compatibility.
        """
        for backend in self._backends:
            backend.step()

    def close(self) -> None:
        """Clean up all physics backends."""
        for backend in self._backends:
            backend.close()
        self._backends.clear()
