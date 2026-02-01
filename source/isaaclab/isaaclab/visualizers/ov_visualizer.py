# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
import enum
from typing import Any
import omni.kit.app
from pxr import UsdGeom

from .ov_visualizer_cfg import OVVisualizerCfg
from .visualizer import Visualizer

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


class OVVisualizer(Visualizer):
    """Omniverse visualizer using Isaac Sim viewport.

    Renders USD stage with VisualizationMarkers and LivePlots.
    Can attach to existing app or launch standalone.
    """

    def __init__(self, cfg: OVVisualizerCfg):
        super().__init__(cfg)
        self.cfg: OVVisualizerCfg = cfg

        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._simulation_context = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._simulation_app_running = False

        self._viewport_context = None
        self._viewport_window = None
        self._render_throttle_counter = 0
        self._render_throttle_period = 5

        # App control
        self._disable_app_control_on_stop_handle = False
        self._app_control_on_stop_handle = None
        self._current_render_mode = RenderMode.NO_GUI_OR_RENDERING

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize OV visualizer."""
        if self._is_initialized:
            logger.warning("[OVVisualizer] Already initialized.")
            return

        usd_stage = None
        if scene_data is not None:
            usd_stage = scene_data.get("usd_stage")
            self._simulation_context = scene_data.get("simulation_context")

        if usd_stage is None:
            raise RuntimeError("OV visualizer requires a USD stage.")

        # Build metadata from simulation context if available
        metadata = {}
        if self._simulation_context is not None:
            # Try to get num_envs from the simulation context's scene if available
            num_envs = 0
            if hasattr(self._simulation_context, "scene") and self._simulation_context.scene is not None:
                if hasattr(self._simulation_context.scene, "num_envs"):
                    num_envs = self._simulation_context.scene.num_envs

            # Detect physics backend (could be extended to check actual backend type)
            physics_backend = "newton"  # Default for now, could be made more sophisticated

            metadata = {
                "num_envs": num_envs,
                "physics_backend": physics_backend,
                "env_prim_pattern": "/World/envs/env_{}",  # Standard pattern
            }

            self._simulation_context.settings.set_bool("/app/player/playSimulations", False)
            self._offscreen_render = bool(self._simulation_context.settings.get("/isaaclab/render/offscreen"))
            self._render_viewport = bool(self._simulation_context.settings.get("/isaaclab/render/active_viewport"))
            self._has_gui = bool(self._simulation_context.settings.get("/isaaclab/render/active_viewport"))
            # Set render mode
            if not self._has_gui and not self._offscreen_render:
                self.render_mode = RenderMode.NO_GUI_OR_RENDERING
            elif not self._has_gui and self._offscreen_render:
                self.render_mode = RenderMode.PARTIAL_RENDERING
            else:
                self.render_mode = RenderMode.FULL_RENDERING

            # enable viewport updates if GUI is enabled
            if self._has_gui:
                try:
                    import omni.ui as ui
                    from omni.kit.viewport.utility import get_active_viewport
                    self._viewport_context = get_active_viewport()
                    self._viewport_context.updates_enabled = True
                    self._viewport_window = ui.Workspace.get_window("Viewport")
                except (ImportError, AttributeError):
                    pass

            # Disable viewport for offscreen-only rendering
            if not self._render_viewport and self._offscreen_render:
                try:
                    from omni.kit.viewport.utility import get_active_viewport
                    get_active_viewport().updates_enabled = False
                except (ImportError, AttributeError):
                    pass

        self._ensure_simulation_app()
        self._setup_viewport(usd_stage, metadata)

        num_envs = metadata.get("num_envs", 0)
        physics_backend = metadata.get("physics_backend", "unknown")
        logger.info(f"[OVVisualizer] Initialized ({num_envs} envs, {physics_backend} physics)")

        self._is_initialized = True

    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualizer (no-op for OV - USD stage is auto-synced by Newton)."""
        if not self._is_initialized:
            return
        self._sim_time += dt
        self._step_counter += 1
        if self._current_render_mode != self.render_mode:
            self.set_render_mode(self.render_mode)
            self._current_render_mode = self.render_mode
        self._simulation_context.settings.set_bool("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()

        # Restore CUDA device after app.update()
        if "cuda" in self._simulation_context.device:
            import torch
            torch.cuda.set_device(self._simulation_context.device)

    def close(self) -> None:
        """Clean up visualizer resources."""
        if not self._is_initialized:
            return

        # Note: We don't close the SimulationApp here as it's managed by AppLauncher
        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False

    def is_running(self) -> bool:
        """Check if visualizer is running."""
        if self._simulation_app is not None:
            return self._simulation_app.is_running()
        return self._simulation_app_running

    def is_training_paused(self) -> bool:
        """Check if training is paused (always False for OV)."""
        return False

    def supports_markers(self) -> bool:
        # Should we add marker configuration, or let the env itself handle it.
        """Supports markers via USD prims."""
        return True

    def supports_live_plots(self) -> bool:
        """Supports live plots via Isaac Lab UI.

        When enable_live_plots=True in OVVisualizerCfg:
        - Automatically enables all manager visualizers (checkboxes checked)
        - Keeps plot frames expanded by default
        - Plots appear in the IsaacLab window docked to the right of the viewport
        """
        return True

    def get_rendering_dt(self) -> float | None:
        """Get rendering dt based on OV rate limiting settings."""
        settings = self._simulation_context.settings

        def _from_frequency():
            freq = settings.get("/app/runLoops/main/rateLimitFrequency")
            return 1.0 / freq if freq else None

        if settings.get("/app/runLoops/main/rateLimitEnabled"):
            return _from_frequency()

        try:
            import omni.kit.loop._loop as omni_loop
            runner = omni_loop.acquire_loop_interface()
            return runner.get_manual_step_size() if runner.get_manual_mode() else _from_frequency()
        except Exception:
            return _from_frequency()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set the viewport camera position and target.

        This method positions the viewport camera at the specified eye location and orients it to look at
        the target location. It uses the active camera of the viewport managed by this visualizer.

        Args:
            eye: Camera position in world coordinates (x, y, z).
            target: Target/look-at position in world coordinates (x, y, z).

        Example:

        .. code-block:: python

            # Set camera to look at the origin from position (2.5, 2.5, 2.5)
            >>> visualizer._visualizer_interface.set_camera_view(
            ...     eye=[2.5, 2.5, 2.5],
            ...     target=[0.0, 0.0, 0.0],
            ... )
        """
        if not self._is_initialized:
            logger.warning("[OVVisualizer] Cannot set camera view - visualizer not initialized.")
            return

        self._set_viewport_camera(tuple(eye), tuple(target))

    def set_render_mode(self, mode: int) -> None:
        """Set the render mode."""
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

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_simulation_app(self) -> None:
        """Ensure Isaac Sim app is running."""
        try:
            # Check if omni.kit.app is available (indicates Isaac Sim is running)
            import omni.kit.app

            # Get the running app instance
            app = omni.kit.app.get_app()
            if app is None or not app.is_running():
                raise RuntimeError(
                    "[OVVisualizer] No Isaac Sim app is running. "
                    "OV visualizer requires Isaac Sim to be launched via AppLauncher before initialization. "
                    "Ensure your script calls AppLauncher before creating the environment."
                )

            # Try to get SimulationApp instance for headless check
            try:
                from isaacsim import SimulationApp

                # Check various ways SimulationApp might store its instance
                sim_app = None
                if hasattr(SimulationApp, "_instance") and SimulationApp._instance is not None:
                    sim_app = SimulationApp._instance
                elif hasattr(SimulationApp, "instance") and callable(SimulationApp.instance):
                    sim_app = SimulationApp.instance()

                if sim_app is not None:
                    self._simulation_app = sim_app

                    # Check if running in headless mode
                    if self._simulation_app.config.get("headless", False):
                        logger.warning(
                            "[OVVisualizer] Running in headless mode. "
                            "OV visualizer requires GUI mode (launch with --headless=False) to create viewports."
                        )
                    else:
                        logger.info("[OVVisualizer] Using existing Isaac Sim app instance.")
                else:
                    # App is running but we couldn't get SimulationApp instance
                    # This is okay - we can still use omni APIs
                    
                    logger.info("[OVVisualizer] Isaac Sim app is running (via omni.kit.app).")
                self._simulation_app_running = True
            except ImportError:
                # SimulationApp not available, but omni.kit.app is running
                logger.info("[OVVisualizer] Using running Isaac Sim app (SimulationApp module not available).")

        except ImportError as e:
            raise ImportError(
                f"[OVVisualizer] Could not import omni.kit.app: {e}. Isaac Sim may not be installed or not running."
            )

    def _setup_viewport(self, usd_stage, metadata: dict) -> None:
        """Setup viewport with camera and window size."""
        try:
            import omni.kit.viewport.utility as vp_utils
            from omni.ui import DockPosition

            # Create new viewport or use existing
            if self.cfg.create_viewport and self.cfg.viewport_name:
                # Map dock position string to enum
                dock_position_map = {
                    "LEFT": DockPosition.LEFT,
                    "RIGHT": DockPosition.RIGHT,
                    "BOTTOM": DockPosition.BOTTOM,
                    "SAME": DockPosition.SAME,
                }
                dock_pos = dock_position_map.get(self.cfg.dock_position.upper(), DockPosition.SAME)

                # Create new viewport with proper API
                self._viewport_window = vp_utils.create_viewport_window(
                    name=self.cfg.viewport_name,
                    width=self.cfg.window_width,
                    height=self.cfg.window_height,
                    position_x=50,
                    position_y=50,
                    docked=True,
                )

                logger.info(f"[OVVisualizer] Created viewport '{self.cfg.viewport_name}'")

                # Dock the viewport asynchronously (needs to wait for window creation)
                asyncio.ensure_future(self._dock_viewport_async(self.cfg.viewport_name, dock_pos))

                # Create dedicated camera for this viewport
                if self._viewport_window:
                    self._create_and_assign_camera(usd_stage)
            else:
                # Use existing viewport by name, or fall back to active viewport
                if self.cfg.viewport_name:
                    self._viewport_window = vp_utils.get_viewport_window_by_name(self.cfg.viewport_name)

                    if self._viewport_window is None:
                        logger.warning(
                            f"[OVVisualizer] Viewport '{self.cfg.viewport_name}' not found. "
                            "Using active viewport instead."
                        )
                        self._viewport_window = vp_utils.get_active_viewport_window()
                    else:
                        logger.info(f"[OVVisualizer] Using existing viewport '{self.cfg.viewport_name}'")
                else:
                    self._viewport_window = vp_utils.get_active_viewport_window()
                    logger.info("[OVVisualizer] Using existing active viewport")

            if self._viewport_window is None:
                logger.warning("[OVVisualizer] Could not get/create viewport.")
                return

            # Get viewport API for camera control
            self._viewport_api = self._viewport_window.viewport_api

            # Set camera pose (uses existing camera if not created above)
            self._set_viewport_camera(self.cfg.camera_position, self.cfg.camera_target)

            logger.info(f"[OVVisualizer] Viewport configured (size: {self.cfg.window_width}x{self.cfg.window_height})")

        except ImportError as e:
            logger.warning(f"[OVVisualizer] Viewport utilities unavailable: {e}")
        except Exception as e:
            logger.error(f"[OVVisualizer] Error setting up viewport: {e}")

    async def _dock_viewport_async(self, viewport_name: str, dock_position) -> None:
        """Dock viewport window asynchronously after it's created.

        Args:
            viewport_name: Name of the viewport window to dock.
            dock_position: DockPosition enum value for where to dock.
        """
        try:
            import omni.kit.app
            import omni.ui

            # Wait for the viewport window to be created in the workspace
            viewport_window = None
            for i in range(10):  # Try up to 10 frames
                viewport_window = omni.ui.Workspace.get_window(viewport_name)
                if viewport_window:
                    logger.info(f"[OVVisualizer] Found viewport window '{viewport_name}' after {i} frames")
                    break
                await omni.kit.app.get_app().next_update_async()

            if not viewport_window:
                logger.warning(
                    f"[OVVisualizer] Could not find viewport window '{viewport_name}' in workspace for docking."
                )
                return

            # Get the main viewport to dock relative to
            main_viewport = omni.ui.Workspace.get_window("Viewport")
            if not main_viewport:
                # Try alternative viewport names
                for alt_name in ["/OmniverseKit/Viewport", "Viewport Next"]:
                    main_viewport = omni.ui.Workspace.get_window(alt_name)
                    if main_viewport:
                        break

            if main_viewport and main_viewport != viewport_window:
                # Dock the new viewport relative to the main viewport
                viewport_window.dock_in(main_viewport, dock_position, 0.5)

                # Wait a frame for docking to complete
                await omni.kit.app.get_app().next_update_async()

                # Make the new viewport the active/focused tab
                # Try multiple methods to ensure it becomes active
                viewport_window.focus()
                viewport_window.visible = True

                # Wait another frame and focus again (sometimes needed for tabs)
                await omni.kit.app.get_app().next_update_async()
                viewport_window.focus()

                logger.info(
                    f"[OVVisualizer] Docked viewport '{viewport_name}' at position {self.cfg.dock_position} and set as"
                    " active"
                )
            else:
                logger.info(
                    f"[OVVisualizer] Could not find main viewport for docking. Viewport '{viewport_name}' will remain"
                    " floating."
                )

        except Exception as e:
            logger.warning(f"[OVVisualizer] Error docking viewport: {e}")

    def _create_and_assign_camera(self, usd_stage) -> None:
        """Create a dedicated camera for this viewport and assign it."""
        try:
            # Create camera prim path based on viewport name
            camera_path = f"/World/Cameras/{self.cfg.viewport_name}_Camera"

            # Check if camera already exists
            camera_prim = usd_stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid():
                # Create camera prim
                UsdGeom.Camera.Define(usd_stage, camera_path)
                logger.info(f"[OVVisualizer] Created camera: {camera_path}")
            else:
                logger.info(f"[OVVisualizer] Using existing camera: {camera_path}")

            # Assign camera to viewport
            if self._viewport_api:
                self._viewport_api.set_active_camera(camera_path)
                logger.info(f"[OVVisualizer] Assigned camera '{camera_path}' to viewport '{self.cfg.viewport_name}'")

        except Exception as e:
            logger.warning(f"[OVVisualizer] Could not create/assign camera: {e}. Using default camera.")

    def _set_viewport_camera(self, position: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        """Set viewport camera position and target using Isaac Sim utilities."""
        if self._viewport_api is None:
            return

        try:
            # Import Isaac Sim viewport utilities
            import isaacsim.core.utils.viewports as vp_utils

            # Get the camera prim path for this viewport
            camera_path = self._viewport_api.get_active_camera()
            if not camera_path:
                camera_path = "/OmniverseKit_Persp"  # Default camera

            # Use Isaac Sim utility to set camera view
            vp_utils._visualizer_interface.set_camera_view(
                eye=list(position), target=list(target), camera_prim_path=camera_path, viewport_api=self._viewport_api
            )

            logger.info(f"[OVVisualizer] Camera set: pos={position}, target={target}, camera={camera_path}")

        except Exception as e:
            logger.warning(f"[OVVisualizer] Could not set camera: {e}")
