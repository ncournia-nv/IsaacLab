# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX physics backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Sdf

from .physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim.simulation_context import SimulationContext


def _set_prim_attr(prim: "Usd.Prim", attr_name: str, value, value_type) -> None:
    """Set or create a prim attribute."""
    attr = prim.GetAttribute(attr_name)
    if attr is None or not attr.IsValid():
        attr = prim.CreateAttribute(attr_name, value_type)
    attr.Set(value)


class PhysXBackend(PhysicsBackend):
    """PhysX physics backend for Omniverse simulation."""

    def __init__(self, sim_context: "SimulationContext", physics_scene_prim: "Usd.Prim"):
        """Initialize and configure PhysX backend.

        Args:
            sim_context: Parent simulation context.
            physics_scene_prim: USD prim for the physics scene.
        """
        super().__init__(sim_context)
        self._physics_scene_prim = physics_scene_prim
        self._configure_device()
        self._create_material()
        self._detach_physx()

    def _configure_device(self) -> None:
        """Configure PhysX for CPU or GPU simulation."""
        cfg = self._sim.cfg
        prim = self._physics_scene_prim
        device = cfg.device

        if "cuda" in device:
            parsed = device.split(":")
            if len(parsed) == 1:
                device_id = self._sim.settings.get("/physics/cudaDevice", 0)
                if device_id < 0:
                    self._sim.settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
                self._sim.device = f"cuda:{device_id}"
            else:
                self._sim.settings.set_int("/physics/cudaDevice", int(parsed[1]))
            self._sim.settings.set_bool("/physics/suppressReadback", True)
            _set_prim_attr(prim, "physxScene:broadphaseType", "GPU", Sdf.ValueTypeNames.Token)
            _set_prim_attr(prim, "physxScene:enableGPUDynamics", True, Sdf.ValueTypeNames.Bool)
        elif device.lower() == "cpu":
            self._sim.settings.set_bool("/physics/suppressReadback", False)
            _set_prim_attr(prim, "physxScene:broadphaseType", "MBP", Sdf.ValueTypeNames.Token)
            _set_prim_attr(prim, "physxScene:enableGPUDynamics", False, Sdf.ValueTypeNames.Bool)
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _create_material(self) -> None:
        """Create default physics material."""
        from isaaclab.sim.utils import bind_physics_material

        cfg = self._sim.cfg
        physics_prim_path = cfg.physics_prim_path
        material_path = f"{physics_prim_path}/defaultMaterial"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(physics_prim_path, material_path)

    def _detach_physx(self) -> None:
        """Detach PhysX simulation (for Newton-only mode)."""
        try:
            import omni.physx
            from omni.physics.stageupdate import get_physics_stage_update_node_interface

            omni.physx.get_physx_simulation_interface().detach_stage()
            get_physics_stage_update_node_interface().detach_node()
        except Exception:
            pass

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation."""
        pass  # PhysX reset handled by Omniverse

    def forward(self) -> None:
        """Update kinematics without stepping physics."""
        pass  # PhysX forward handled by Omniverse

    def step(self) -> None:
        """Step physics simulation."""
        pass  # PhysX step handled by Omniverse timeline

    def close(self) -> None:
        """Clean up PhysX resources."""
        pass  # PhysX cleanup handled by Omniverse
