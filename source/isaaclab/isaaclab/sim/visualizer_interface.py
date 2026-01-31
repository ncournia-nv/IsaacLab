# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer interface for SimulationContext."""

from __future__ import annotations

import enum
import logging
import os
from typing import TYPE_CHECKING

from isaaclab.visualizers import NewtonVisualizerCfg, OVVisualizerCfg, RerunVisualizerCfg, Visualizer

from .scene_data_provider import SceneDataProvider
from .utils import raise_callback_exception_if_any

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class RenderMode(enum.IntEnum):
    """Rendering modes controlling viewport/UI update frequency.

    - NO_GUI_OR_RENDERING (-1): Complete headless, nothing updated
    - NO_RENDERING (0): UI updated at reduced rate
    - PARTIAL_RENDERING (1): UI + cameras updated
    - FULL_RENDERING (2): UI + cameras + viewports updated
    """

    NO_GUI_OR_RENDERING = -1
    NO_RENDERING = 0
    PARTIAL_RENDERING = 1
    FULL_RENDERING = 2


class VisualizerInterface:
    """Manages visualizer lifecycle and rendering for SimulationContext."""

    # Expose RenderMode as class attribute for backwards compatibility
    RenderMode = RenderMode

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize visualizer interface.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context
        self.dt = self._sim.cfg.dt

        # Visualizer state
        visualizers = self.settings.get("/isaaclab/visualizer") or ""
        self._visualizers_str = [v.strip() for v in visualizers.split(",") if v.strip()]
        self._visualizers: list[Visualizer] = []
        self._visualizer_step_counter = 0
        self._scene_data_provider: SceneDataProvider | None = None
        # Detect render flags
        self._offscreen_render = bool(self.settings.get("/isaaclab/render/offscreen"))
        self._render_viewport = bool(self.settings.get("/isaaclab/render/active_viewport"))
        self._rtx_sensors = bool(self.settings.get("/isaaclab/render/rtx_sensors", False))
        self._has_gui = bool(self.settings.get("/isaaclab/visualizer"))

        if not self._has_gui and not self._offscreen_render:
            self.render_mode = RenderMode.NO_GUI_OR_RENDERING
        elif not self._has_gui and self._offscreen_render:
            self.render_mode = RenderMode.PARTIAL_RENDERING
        else:
            self.render_mode = RenderMode.FULL_RENDERING

    # -- Properties --

    @property
    def settings(self):
        return self._sim.settings

    @property
    def device(self) -> str:
        return self._sim.device

    @property
    def stage(self):
        return self._sim.stage

    @property
    def visualizers(self) -> list[Visualizer]:
        return self._visualizers

    @property
    def scene_data_provider(self) -> SceneDataProvider | None:
        return self._scene_data_provider

    def has_gui(self) -> bool:
        return self._has_gui

    # -- Visualizer Initialization --

    def _create_default_visualizer_configs(self, requested: list[str]) -> list:
        """Create default configs for requested visualizer types."""
        configs = []
        type_map = {"newton": NewtonVisualizerCfg, "rerun": RerunVisualizerCfg, "omniverse": OVVisualizerCfg}

        for viz_type in requested:
            if viz_type in type_map:
                try:
                    configs.append(type_map[viz_type]())
                except Exception as e:
                    logger.error(f"Failed to create default config for '{viz_type}': {e}")
            else:
                logger.warning(f"Unknown visualizer type '{viz_type}'. Valid: {list(type_map.keys())}")

        return configs

    def initialize_visualizers(self) -> None:
        """Initialize visualizers based on --visualizer flag."""
        if not self._visualizers_str:
            if self._has_gui or self._offscreen_render:
                logger.info("No visualizers specified via --visualizer flag.")
            return

        # Get or create visualizer configs
        cfg_list = self._sim.cfg.visualizer_cfgs
        if cfg_list is None:
            visualizer_cfgs = self._create_default_visualizer_configs(self._visualizers_str)
        else:
            visualizer_cfgs = cfg_list if isinstance(cfg_list, list) else [cfg_list]
            visualizer_cfgs = [c for c in visualizer_cfgs if c.visualizer_type in self._visualizers_str]

            if not visualizer_cfgs:
                logger.info(f"Creating default configs for: {self._visualizers_str}")
                visualizer_cfgs = self._create_default_visualizer_configs(self._visualizers_str)

        if not visualizer_cfgs:
            return

        # Create scene data provider
        self._scene_data_provider = SceneDataProvider(visualizer_cfgs)

        # Initialize each visualizer
        for cfg in visualizer_cfgs:
            try:
                visualizer = cfg.create_visualizer()
                scene_data = self._build_scene_data(cfg)
                visualizer.initialize(scene_data)
                self._visualizers.append(visualizer)
                logger.info(f"Initialized: {type(visualizer).__name__} ({cfg.visualizer_type})")
            except Exception as e:
                logger.error(f"Failed to init '{cfg.visualizer_type}': {e}")

    def _build_scene_data(self, cfg) -> dict:
        """Build scene data dict for visualizer initialization."""
        if cfg.visualizer_type in ("newton", "rerun"):
            return {"scene_data_provider": self._scene_data_provider}
        elif cfg.visualizer_type == "omniverse":
            return {"usd_stage": self._sim.stage, "simulation_context": self._sim}
        return {}

    # -- Unified Interface Methods --

    def forward(self) -> None:
        """Sync scene data and step all active visualizers.

        Args:
            dt: Time step in seconds (0.0 for kinematics-only).
        """
        if self._scene_data_provider:
            self._scene_data_provider.update()

        if not self._visualizers:
            return

    def step(self, render: bool = True) -> None:
        """Step visualizers and optionally render.

        Args:
            render: Whether to render after stepping.
        """
        # Keep UI responsive while paused
        while not self._sim.is_playing():
            self.render()

        self.forward()

        if render:
            self.render()

    def reset(self, soft: bool) -> None:
        """Reset visualizers (warmup renders on hard reset)."""
        self.settings.set_bool("/app/player/playSimulations", False)
        self._disable_app_control_on_stop_handle = not soft

        if not soft:
            for _ in range(2):
                self.render()
            if not self._visualizers:
                self.initialize_visualizers()

        self._disable_app_control_on_stop_handle = False

    def close(self) -> None:
        """Close all visualizers and clean up."""
        if self._app_control_on_stop_handle:
            self._app_control_on_stop_handle.unsubscribe()
            self._app_control_on_stop_handle = None

        for viz in self._visualizers:
            try:
                viz.close()
            except Exception as e:
                logger.error(f"Error closing {type(viz).__name__}: {e}")

        self._visualizers.clear()
        logger.info("All visualizers closed")

    def on_play(self) -> None:
        """Handle OV timeline on simulation start.
        Octi: this is not called at all in newton branch for all visualizers
        """
        pass

    def on_stop(self) -> None:
        """Handle OV timeline on simulation stop.
        Octi: this is not called at all in newton branch for all visualizers
        """
        pass

    def render(self) -> bool:
        """Render the scene (OV mode only).

        Args:
            mode: Render mode to set, or None to keep current.

        Returns:
            True if rendered, False if not in OV mode.
        """
        raise_callback_exception_if_any()

        self._visualizer_step_counter += 1
        to_remove = []

        for viz in self._visualizers:
            try:
                if not viz.is_running():
                    to_remove.append(viz)
                    continue

                # Block while training paused
                while viz.is_training_paused() and viz.is_running():
                    viz.step(0.0, state=None)

                viz.step(self.get_rendering_dt() or self.dt, state=None)
            except Exception as e:
                logger.error(f"Error stepping {type(viz).__name__}: {e}")
                to_remove.append(viz)

        for viz in to_remove:
            try:
                viz.close()
                self._visualizers.remove(viz)
                logger.info(f"Removed: {type(viz).__name__}")
            except Exception as e:
                logger.error(f"Error closing visualizer: {e}")

        return True

    def get_rendering_dt(self) -> float:
        """Get rendering dt for OV mode."""
        if "omniverse" not in self._visualizers_str:
            return self.dt

        def _from_frequency():
            freq = self.settings.get("/app/runLoops/main/rateLimitFrequency")
            return 1.0 / freq if freq else 0

        if self.settings.get("/app/runLoops/main/rateLimitEnabled"):
            return _from_frequency()

        try:
            import omni.kit.loop._loop as omni_loop
            runner = omni_loop.acquire_loop_interface()
            return runner.get_manual_step_size() if runner.get_manual_mode() else _from_frequency()
        except Exception:
            return _from_frequency()

    def set_camera_view(self, eye: tuple, target: tuple, camera_prim_path: str = "/OmniverseKit_Persp") -> None:
        """Set viewport camera position (OV visualizer only)."""
        for viz in self._visualizers:
            is_ov = getattr(getattr(viz, "cfg", None), "visualizer_type", None) == "omniverse"
            if is_ov and hasattr(viz, "set_camera_view"):
                viz.set_camera_view(eye, target)
                return

        logger.debug("No Omniverse visualizer found - set_camera_view has no effect.")
