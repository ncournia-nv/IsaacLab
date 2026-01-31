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

from isaaclab.sim._impl.newton_manager import NewtonManager

from .utils import bind_physics_material

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


def set_prim_attr(prim: Usd.Prim, attr_name: str, value, value_type) -> None:
    """Set or create a prim attribute."""
    attr = prim.GetAttribute(attr_name)
    if attr is None or not attr.IsValid():
        attr = prim.CreateAttribute(attr_name, value_type)
    attr.Set(value)


class PhysicsInterface:
    """Manages USD physics scene and Newton physics engine for SimulationContext."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize physics scene and configure Newton.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context
        self._sim.settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)

        # Store config values
        self.physics_dt = self._sim.cfg.dt
        self.rendering_dt = self._sim.cfg.dt * self._sim.cfg.render_interval
        self.physics_prim_path = self._sim.cfg.physics_prim_path
        self.device = self._sim.cfg.device
        self.backend = "torch"

        # Pre-create gravity tensor to avoid torch heap corruption issues (torch 2.1+)
        self._sim._gravity_tensor = torch.tensor(
            self._sim.cfg.gravity, dtype=torch.float32, device=self._sim.cfg.device
        )

        self._init_usd_scene()
        # self._init_physx()
        self._init_newton()

    def _init_usd_scene(self) -> None:
        """Create USD physics scene with gravity and timestep."""
        stage = self._sim.stage
        cfg = self._sim.cfg

        # Create physics scene prim
        prim: Usd.Prim = stage.GetPrimAtPath(self.physics_prim_path)
        if not prim.IsValid():
            self._physics_scene = UsdPhysics.Scene.Define(stage, self.physics_prim_path)
            prim = stage.GetPrimAtPath(self.physics_prim_path)
        else:
            self._physics_scene = UsdPhysics.Scene(prim)

        # Set timestep
        set_prim_attr(prim, "physxScene:timeStepsPerSecond", int(1.0 / cfg.dt), Sdf.ValueTypeNames.Int)
        stage.SetTimeCodesPerSecond(1 / cfg.dt)

        # Set gravity based on stage up-axis
        up_axis = UsdGeom.GetStageUpAxis(stage)
        gravity_magnitude = abs(cfg.gravity[2])
        if up_axis == "Z":
            gravity_dir = Gf.Vec3f(0.0, 0.0, -1.0 if cfg.gravity[2] < 0 else 1.0)
        elif up_axis == "Y":
            gravity_dir = Gf.Vec3f(0.0, -1.0 if cfg.gravity[1] < 0 else 1.0, 0.0)
        else:
            gravity_dir = Gf.Vec3f(-1.0 if cfg.gravity[0] < 0 else 1.0, 0.0, 0.0)
        self._physics_scene.CreateGravityDirectionAttr().Set(gravity_dir)
        self._physics_scene.CreateGravityMagnitudeAttr().Set(gravity_magnitude)

        # Store references
        self._physics_scene_prim = prim
        self._sim.physics_scene = prim
        self._sim._physics_scene = self._physics_scene

    def _init_physx(self) -> None:
        """Configure PhysX device, material, and detach for Newton-only simulation."""
        cfg = self._sim.cfg
        prim = self._physics_scene_prim

        # Configure device (CPU/GPU)
        if "cuda" in self.device:
            parsed = self.device.split(":")
            if len(parsed) == 1:
                device_id = self._sim.settings.get("/physics/cudaDevice", 0)
                if device_id < 0:
                    self._sim.settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
                self.device = f"cuda:{device_id}"
                self._sim.device = self.device
            else:
                self._sim.settings.set_int("/physics/cudaDevice", int(parsed[1]))
            self._sim.settings.set_bool("/physics/suppressReadback", True)
            set_prim_attr(prim, "physxScene:broadphaseType", "GPU", Sdf.ValueTypeNames.Token)
            set_prim_attr(prim, "physxScene:enableGPUDynamics", True, Sdf.ValueTypeNames.Bool)
        elif self.device.lower() == "cpu":
            self._sim.settings.set_bool("/physics/suppressReadback", False)
            set_prim_attr(prim, "physxScene:broadphaseType", "MBP", Sdf.ValueTypeNames.Token)
            set_prim_attr(prim, "physxScene:enableGPUDynamics", False, Sdf.ValueTypeNames.Bool)
        else:
            raise ValueError(f"Unsupported device: {self.device}")

        # Create default physics material
        material_path = f"{self.physics_prim_path}/defaultMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(self.physics_prim_path, material_path)

        # Detach PhysX (for Newton-only simulation)
        try:
            import omni.physx
            from omni.physics.stageupdate import get_physics_stage_update_node_interface
            omni.physx.get_physx_simulation_interface().detach_stage()
            get_physics_stage_update_node_interface().detach_node()
        except Exception:
            pass

    def _init_newton(self) -> None:
        """Configure Newton physics engine."""
        cfg = self._sim.cfg
        NewtonManager.set_simulation_dt(cfg.dt)
        NewtonManager._gravity_vector = cfg.gravity

        # Extract newton params from config
        to_dict = getattr(cfg, "to_dict", None)
        params = to_dict() if callable(to_dict) else {}
        newton_cfg = params.get("newton_cfg", {}) if isinstance(params, dict) else {}
        NewtonManager.set_solver_settings(dict(newton_cfg) if newton_cfg else {})

        # the usd clone mainly play the role to update fabric for omniverse
        NewtonManager._clone_physics_only = "omniverse" not in self._sim._visualizer_interface._visualizers_str

    def reset(self, soft: bool) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            NewtonManager.start_simulation()
            NewtonManager.initialize_solver()

    def forward(self) -> None:
        """Update articulation kinematics without stepping physics."""
        NewtonManager.forward_kinematics()

    def step(self) -> None:
        """Step physics simulation."""
        if self._sim.is_playing():
            NewtonManager.step()

    def close(self) -> None:
        """Clean up Newton physics resources."""
        NewtonManager.clear()
