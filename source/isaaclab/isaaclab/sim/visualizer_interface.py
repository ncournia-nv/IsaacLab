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
        self._visualizers: list[Visualizer] = []
        self._visualizer_step_counter = 0
        self._scene_data_provider: SceneDataProvider | None = None

        # Viewport state
        self._viewport_context = None
        self._viewport_window = None
        self._render_throttle_counter = 0
        self._render_throttle_period = 5

        # App control
        self._disable_app_control_on_stop_handle = False
        self._app_control_on_stop_handle = None

        self._init_render_mode()

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

    # -- Initialization --

    def _init_render_mode(self) -> None:
        """Initialize render mode based on GUI/rendering settings."""
        self.settings.set_bool("/app/player/playSimulations", False)

        # Detect GUI mode
        local_gui = self.settings.get("/app/window/enabled") or False
        livestream_gui = self.settings.get("/app/livestream/enabled") or False
        xr_gui = self.settings.get("/app/xr/enabled") or False
        self._has_gui = local_gui or livestream_gui or xr_gui

        # Detect render flags
        self._offscreen_render = bool(self.settings.get("/isaaclab/render/offscreen"))
        self._render_viewport = bool(self.settings.get("/isaaclab/render/active_viewport"))

        # Set render mode
        if not self._has_gui and not self._offscreen_render:
            self.render_mode = RenderMode.NO_GUI_OR_RENDERING
        elif not self._has_gui and self._offscreen_render:
            self.render_mode = RenderMode.PARTIAL_RENDERING
        else:
            self.render_mode = RenderMode.FULL_RENDERING
            self._init_viewport()

        # Disable viewport for offscreen-only rendering
        if not self._render_viewport and self._offscreen_render:
            self._disable_viewport()

    def _init_viewport(self) -> None:
        """Acquire viewport context for GUI mode."""
        try:
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True
            self._viewport_window = ui.Workspace.get_window("Viewport")
        except (ImportError, AttributeError):
            pass

    def _disable_viewport(self) -> None:
        """Disable viewport updates."""
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().updates_enabled = False
        except (ImportError, AttributeError):
            pass

    def set_render_mode(self, mode: int) -> None:
        """Change the current render mode."""
        if not self._has_gui:
            logger.warning(f"Cannot change render mode without GUI. Using: {self.render_mode}")
            return

        if mode == self.render_mode:
            return

        if mode == RenderMode.FULL_RENDERING:
            self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
            self._viewport_window.visible = True  # pyright: ignore [reportOptionalMemberAccess]
        elif mode in (RenderMode.PARTIAL_RENDERING, RenderMode.NO_RENDERING):
            if self._viewport_context:
                self._viewport_context.updates_enabled = False
                self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
            if mode == RenderMode.NO_RENDERING:
                self._render_throttle_counter = 0
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

        self.render_mode = mode

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

    def _get_requested_visualizers(self) -> list[str]:
        """Parse --visualizer flag and filter for headless mode."""
        requested_str = self.settings.get("/isaaclab/visualizer") or ""
        requested = [v.strip() for v in requested_str.split(",") if v.strip()]

        if not requested:
            return []

        # Filter GUI-dependent visualizers in headless mode
        if not self._has_gui and not self._offscreen_render:
            headless_compatible = [v for v in requested if v in ("newton", "rerun")]
            if len(headless_compatible) < len(requested):
                logger.info(f"Headless mode: filtering {requested} to {headless_compatible}")
            return headless_compatible

        return requested

    def initialize_visualizers(self) -> None:
        """Initialize visualizers based on --visualizer flag."""
        requested = self._get_requested_visualizers()

        if not requested:
            if self._has_gui or self._offscreen_render:
                logger.info("No visualizers specified via --visualizer flag.")
            return

        # Get or create visualizer configs
        cfg_list = self._sim.cfg.visualizer_cfgs
        if cfg_list is None:
            visualizer_cfgs = self._create_default_visualizer_configs(requested)
        else:
            visualizer_cfgs = cfg_list if isinstance(cfg_list, list) else [cfg_list]
            visualizer_cfgs = [c for c in visualizer_cfgs if c.visualizer_type in requested]

            if not visualizer_cfgs:
                logger.info(f"Creating default configs for: {requested}")
                visualizer_cfgs = self._create_default_visualizer_configs(requested)

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

    def forward(self, dt: float = 0.0) -> None:
        """Sync scene data and step all active visualizers.

        Args:
            dt: Time step in seconds (0.0 for kinematics-only).
        """
        if self._scene_data_provider:
            self._scene_data_provider.update()

        if not self._visualizers:
            return

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

                viz.step(dt, state=None)
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

    def step(self, render: bool = True) -> None:
        """Step visualizers and optionally render.

        Args:
            render: Whether to render after stepping.
        """
        # Keep UI responsive while paused
        while not self._sim.is_playing():
            self.render(mode=None)

        self.forward(self.get_rendering_dt() or self.dt)

        if render:
            self.render(mode=None)

    def reset(self, soft: bool) -> None:
        """Reset visualizers (warmup renders on hard reset)."""
        self.settings.set_bool("/app/player/playSimulations", False)
        self._disable_app_control_on_stop_handle = not soft

        if not soft:
            for _ in range(2):
                self.render(mode=None)
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

    # -- Omniverse-Specific Methods --

    def has_omniverse_visualizer(self) -> bool:
        """Check if Omniverse visualizer is active or requested."""
        if int(os.environ.get("LAUNCH_OV_APP", 0)) == 1:
            return True

        for viz in self._visualizers:
            if getattr(getattr(viz, "cfg", None), "visualizer_type", None) == "omniverse":
                return True
            if type(viz).__name__ == "OVVisualizer":
                return True

        requested = self.settings.get("/isaaclab/visualizer") or ""
        if "omniverse" in requested:
            return self._has_gui

        return False

    def on_play(self) -> None:
        """Handle OV timeline on simulation start."""
        if self.has_omniverse_visualizer():
            import omni.kit.app
            import omni.timeline
            omni.timeline.get_timeline_interface().play()
            omni.timeline.get_timeline_interface().commit()
            self.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

    def on_stop(self) -> None:
        """Handle OV timeline on simulation stop."""
        if self.has_omniverse_visualizer():
            import omni.kit.app
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
            self.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

    def render(self, mode) -> bool:
        """Render the scene (OV mode only).

        Args:
            mode: Render mode to set, or None to keep current.

        Returns:
            True if rendered, False if not in OV mode.
        """
        if not self.has_omniverse_visualizer():
            return False

        import omni.kit.app
        raise_callback_exception_if_any()

        if mode is not None:
            self.set_render_mode(mode)

        if self.render_mode == RenderMode.NO_GUI_OR_RENDERING:
            pass
        elif self.render_mode == RenderMode.NO_RENDERING:
            self._render_throttle_counter += 1
            if self._render_throttle_counter % self._render_throttle_period == 0:
                self._render_throttle_counter = 0
                self.settings.set_bool("/app/player/playSimulations", False)
                omni.kit.app.get_app().update()
        else:
            self.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

        # Restore CUDA device after app.update()
        if "cuda" in self.device:
            import torch
            torch.cuda.set_device(self.device)

        return True

    def get_rendering_dt(self) -> float | None:
        """Get rendering dt for OV mode."""
        if not self.has_omniverse_visualizer():
            return None

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
