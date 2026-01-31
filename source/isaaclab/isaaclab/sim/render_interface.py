# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render interface for SimulationContext."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import flatdict
import toml

if TYPE_CHECKING:
    from .simulation_context import SimulationContext
    from .visualizer_interface import VisualizerInterface

logger = logging.getLogger(__name__)


class RenderInterface:
    """Manages rendering configuration and lifecycle for SimulationContext."""

    def __init__(self, sim_context: "SimulationContext", visualizer_interface: "VisualizerInterface | None" = None):
        self._sim = sim_context
        self._visualizer_interface = visualizer_interface
        self.apply_render_settings_from_cfg()

    def attach_visualizer(self, visualizer_interface: "VisualizerInterface") -> None:
        self._visualizer_interface = visualizer_interface

    def apply_render_settings_from_cfg(self):
        """Sets rtx settings specified in the RenderCfg."""
        rendering_setting_name_mapping = {
            "enable_translucency": "/rtx/translucency/enabled",
            "enable_reflections": "/rtx/reflections/enabled",
            "enable_global_illumination": "/rtx/indirectDiffuse/enabled",
            "enable_dlssg": "/rtx-transient/dlssg/enabled",
            "enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
            "dlss_mode": "/rtx/post/dlss/execMode",
            "enable_direct_lighting": "/rtx/directLighting/enabled",
            "samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
            "enable_shadows": "/rtx/shadows/enabled",
            "enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
        }

        not_carb_settings = ["rendering_mode", "carb_settings", "antialiasing_mode"]

        rendering_mode = self._sim.cfg.render_cfg.rendering_mode
        if rendering_mode is not None:
            supported_rendering_modes = ["performance", "balanced", "quality"]
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes {supported_rendering_modes}."
                )

            import carb

            repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
            preset_filename = os.path.join(repo_path, f"apps/rendering_modes/{rendering_mode}.kit")
            with open(preset_filename) as file:
                preset_dict = toml.load(file)
            preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

            for key, value in preset_dict.items():
                key = "/" + key.replace(".", "/")
                self._sim.settings.set(key, value)

        for key, value in vars(self._sim.cfg.render_cfg).items():
            if value is None or key in not_carb_settings:
                continue
            if key not in rendering_setting_name_mapping:
                raise ValueError(
                    f"'{key}' in RenderCfg not found. Note: internal 'rendering_setting_name_mapping' dictionary might"
                    " need to be updated."
                )
            key = rendering_setting_name_mapping[key]
            self._sim.settings.set(key, value)

        carb_settings = self._sim.cfg.render_cfg.carb_settings
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")
                elif "." in key:
                    key = "/" + key.replace(".", "/")
                if self._sim.settings.get(key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.general_parameters does not map to a carb setting.")
                self._sim.settings.set(key, value)

        if self._sim.cfg.render_cfg.antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=self._sim.cfg.render_cfg.antialiasing_mode)
            except Exception:
                pass

        render_mode = self._sim.settings.get("/rtx/rendermode")
        if render_mode is not None and render_mode.lower() == "raytracedlighting":
            self._sim.settings.set("/rtx/rendermode", "RaytracedLighting")

    def has_rtx_sensors(self) -> bool:
        """Returns whether the simulation has any RTX-rendering related sensors."""
        return self._sim.settings.get("/isaaclab/render/rtx_sensors")

    def is_fabric_enabled(self) -> bool:
        """Returns whether the fabric interface is enabled."""
        return self._sim._fabric_iface is not None

    def load_fabric_interface(self):
        """Loads the fabric interface if enabled."""
        if self._sim.cfg.use_fabric:
            from omni.physxfabric import get_physx_fabric_interface

            # acquire fabric interface
            self._sim._fabric_iface = get_physx_fabric_interface()
            if hasattr(self._sim._fabric_iface, "force_update"):
                # The update method in the fabric interface only performs an update if a physics step has occurred.
                # However, for rendering, we need to force an update since any element of the scene might have been
                # modified in a reset (which occurs after the physics step) and we want the renderer to be aware of
                # these changes.
                self._sim._update_fabric = self._sim._fabric_iface.force_update
            else:
                # Needed for backward compatibility with older Isaac Sim versions
                self._sim._update_fabric = self._sim._fabric_iface.update

    def render(self, mode: int | None = None):
        if self._visualizer_interface is None:
            return
        self._visualizer_interface.render(mode)

    def get_rendering_dt(self) -> float:
        if self._visualizer_interface is not None:
            ov_dt = self._visualizer_interface.get_rendering_dt()
            if ov_dt is not None:
                return ov_dt
        return self._sim.cfg.dt

    def update_scene_data(self) -> None:
        if self._visualizer_interface is None:
            return
        self._visualizer_interface.update_scene_data()

    def on_play(self) -> None:
        if self._visualizer_interface is None:
            return
        self._visualizer_interface.on_play()

    def on_stop(self) -> None:
        if self._visualizer_interface is None:
            return
        self._visualizer_interface.on_stop()

    def set_camera_view(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        if self._visualizer_interface is None:
            return
        self._visualizer_interface.set_camera_view(eye, target, camera_prim_path)
