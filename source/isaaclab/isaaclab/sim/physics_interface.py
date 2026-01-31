# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics interface for SimulationContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim._impl.newton_manager import NewtonManager

from .utils import bind_physics_material

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class PhysicsInterface:
    """Manages USD physics scene and NewtonManager lifecycle for SimulationContext."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize the physics interface.

        Args:
            sim_context: The simulation context this interface belongs to.
        """
        self._sim: Any = sim_context

        # step 0: set physics settings
        self._sim.settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self._physics_scene = None
        self._physics_scene_prim = None
        self._newton_params = self._extract_newton_params()

        # step 1: initialize parameters
        self.physics_dt = self._sim.cfg.dt
        self.rendering_dt = self._sim.cfg.dt * self._sim.cfg.render_interval
        self.backend = "torch"
        self.physics_prim_path = self._sim.cfg.physics_prim_path
        self.device = self._sim.cfg.device

        # create a tensor for gravity
        # note: this line is needed to create a "tensor" in the device to avoid issues with torch 2.1 onwards.
        #   the issue is with some heap memory corruption when torch tensor is created inside the asset class.
        #   you can reproduce the issue by commenting out this line and running the test `test_articulation.py`.
        self._sim._gravity_tensor = torch.tensor(
            self._sim.cfg.gravity, dtype=torch.float32, device=self._sim.cfg.device
        )

        # step 3: initialize_physics_scene
        physics_scene_prim = self._sim.stage.GetPrimAtPath(self.physics_prim_path)
        if not physics_scene_prim.IsValid():
            self._physics_scene = UsdPhysics.Scene.Define(self._sim.stage, self.physics_prim_path)
            physics_scene_prim = self._sim.stage.GetPrimAtPath(self.physics_prim_path)
        else:
            self._physics_scene = UsdPhysics.Scene(physics_scene_prim)

        # Set physics dt (time steps per second) using string attribute name
        self._set_physx_scene_attr(
            physics_scene_prim, "physxScene:timeStepsPerSecond", int(1.0 / self._sim.cfg.dt), Sdf.ValueTypeNames.Int
        )
        self._sim.stage.SetTimeCodesPerSecond(1 / self._sim.cfg.dt)

        # Set gravity on the physics scene
        up_axis = UsdGeom.GetStageUpAxis(self._sim.stage)
        gravity_magnitude = abs(self._sim.cfg.gravity[2])  # Get magnitude from z-component
        if up_axis == "Z":
            gravity_dir = Gf.Vec3f(0.0, 0.0, -1.0 if self._sim.cfg.gravity[2] < 0 else 1.0)
        elif up_axis == "Y":
            gravity_dir = Gf.Vec3f(0.0, -1.0 if self._sim.cfg.gravity[1] < 0 else 1.0, 0.0)
        else:
            gravity_dir = Gf.Vec3f(-1.0 if self._sim.cfg.gravity[0] < 0 else 1.0, 0.0, 0.0)

        gravity_direction_attr = self._physics_scene.CreateGravityDirectionAttr()
        if gravity_direction_attr is None:
            raise RuntimeError("Failed to create gravity direction attribute.")
        gravity_direction_attr.Set(gravity_dir)
        gravity_magnitude_attr = self._physics_scene.CreateGravityMagnitudeAttr()
        if gravity_magnitude_attr is None:
            raise RuntimeError("Failed to create gravity magnitude attribute.")
        gravity_magnitude_attr.Set(gravity_magnitude)

        self._physics_scene_prim = physics_scene_prim
        self._sim.physics_scene = physics_scene_prim
        self._sim._physics_scene = self._physics_scene

        self.set_physics_sim_device()
        self.configure_newton()
        self.create_default_physics_material()
        self.detach_physx_stage()
        # Disable USD cloning if we are not rendering or using RTX sensors
        # Octi: Somehow this is needed, ther maybe mechanism in newton to auto import usdrt and cloen
        self.update_clone_physics_only()

    def _extract_newton_params(self) -> dict:
        to_dict = getattr(self._sim.cfg, "to_dict", None)
        sim_params = to_dict() if callable(to_dict) else None
        if not sim_params or not isinstance(sim_params, dict):
            return {}
        newton_params = sim_params.get("newton_cfg")
        if newton_params is None:
            return {}
        if isinstance(newton_params, dict):
            return newton_params
        return dict(newton_params)

    def _set_physx_scene_attr(self, prim: Usd.Prim, attr_name: str, value, value_type) -> None:
        """Helper to set a PhysX scene attribute using string-based attribute names.

        Args:
            prim: The physics scene prim.
            attr_name: The full attribute name (e.g., "physxScene:timeStepsPerSecond").
            value: The value to set.
            value_type: The Sdf.ValueTypeNames type for the attribute.
        """
        attr = prim.GetAttribute(attr_name)
        if attr is None or not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, value_type)
        if attr is None:
            raise RuntimeError(f"Failed to create attribute '{attr_name}' on prim '{prim.GetPath()}'.")
        attr.Set(value)

    def set_physics_sim_device(self) -> None:
        """Sets the physics simulation device."""
        if "cuda" in self.device:
            parsed_device = self.device.split(":")
            if len(parsed_device) == 1:
                device_id = self._sim.settings.get("/physics/cudaDevice", 0)
                if device_id < 0:
                    self._sim.settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
                # resolve "cuda" to "cuda:N" for torch.cuda.set_device compatibility
                self.device = f"cuda:{device_id}"
                self._sim.device = self.device
            else:
                self._sim.settings.set_int("/physics/cudaDevice", int(parsed_device[1]))
            self._sim.settings.set_bool("/physics/suppressReadback", True)
            # Set GPU physics settings using string attribute names
            self._set_physx_scene_attr(
                self._sim.physics_scene, "physxScene:broadphaseType", "GPU", Sdf.ValueTypeNames.Token
            )
            self._set_physx_scene_attr(
                self._sim.physics_scene, "physxScene:enableGPUDynamics", True, Sdf.ValueTypeNames.Bool
            )
        elif self.device.lower() == "cpu":
            self._sim.settings.set_bool("/physics/suppressReadback", False)
            # Set CPU physics settings using string attribute names
            self._set_physx_scene_attr(
                self._sim.physics_scene, "physxScene:broadphaseType", "MBP", Sdf.ValueTypeNames.Token
            )
            self._set_physx_scene_attr(
                self._sim.physics_scene, "physxScene:enableGPUDynamics", False, Sdf.ValueTypeNames.Bool
            )
        else:
            raise Exception(f"Device {self.device} is not supported.")

    def configure_newton(self) -> None:
        NewtonManager.set_simulation_dt(self._sim.cfg.dt)
        NewtonManager._gravity_vector = self._sim.cfg.gravity
        NewtonManager.set_solver_settings(self._newton_params)

    def create_default_physics_material(self) -> None:
        # create the default physics material
        # this material is used when no material is specified for a primitive
        material_path = f"{self.physics_prim_path}/defaultMaterial"
        self._sim.cfg.physics_material.func(material_path, self._sim.cfg.physics_material)
        # bind the physics material to the scene
        bind_physics_material(self.physics_prim_path, material_path)

    def detach_physx_stage(self) -> None:
        try:
            import omni.physx
            from omni.physics.stageupdate import get_physics_stage_update_node_interface

            physx_sim_interface = omni.physx.get_physx_simulation_interface()
            physx_sim_interface.detach_stage()
            get_physics_stage_update_node_interface().detach_node()
        except Exception:
            pass

    def update_clone_physics_only(self) -> None:
        render_mode = self._sim._visualizer_interface.render_mode
        NewtonManager._clone_physics_only = render_mode in (
            self._sim._visualizer_interface.RenderMode.NO_GUI_OR_RENDERING,
            self._sim._visualizer_interface.RenderMode.NO_RENDERING,
        )

    def set_gravity(self, gravity_vector: tuple[float, float, float]) -> None:
        NewtonManager._gravity_vector = gravity_vector

    def forward_kinematics(self) -> None:
        NewtonManager.forward_kinematics()

    def start_simulation(self) -> None:
        NewtonManager.start_simulation()

    def initialize_solver(self) -> None:
        NewtonManager.initialize_solver()

    def step(self) -> None:
        NewtonManager.step()

    def render(self) -> None:
        render_fn = getattr(NewtonManager, "render", None)
        if callable(render_fn):
            render_fn()
        else:
            logger.debug("Newton render requested, but NewtonManager.render is unavailable.")

    def clear(self) -> None:
        NewtonManager.clear()
