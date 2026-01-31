# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer interface for SimulationContext."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from isaaclab.visualizers import NewtonVisualizerCfg, OVVisualizerCfg, RerunVisualizerCfg, Visualizer

from .scene_data_provider import SceneDataProvider

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class VisualizerInterface:
    """Manages visualizer lifecycle for SimulationContext.
    
    This class handles initialization, stepping, and cleanup of visualizers.
    It delegates to the SimulationContext for settings and state access.
    """

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize the visualizer interface.
        
        Args:
            sim_context: The simulation context this interface belongs to.
        """
        self._sim = sim_context
        
        # initialize visualizers and scene data provider
        self._visualizers: list[Visualizer] = []
        self._visualizer_step_counter = 0
        self._scene_data_provider: SceneDataProvider | None = None

    @property
    def visualizers(self) -> list[Visualizer]:
        """Get the list of active visualizers."""
        return self._visualizers

    @property
    def scene_data_provider(self) -> SceneDataProvider | None:
        """Get the scene data provider."""
        return self._scene_data_provider

    def _create_default_visualizer_configs(self, requested_visualizers: list[str]) -> list:
        """Create default visualizer configurations for requested visualizer types.

        This method creates minimal default configurations for visualizers when none are defined
        in the simulation config. Each visualizer is created with all default parameters.

        Args:
            requested_visualizers: List of visualizer type names (e.g., ['newton', 'rerun', 'omniverse']).

        Returns:
            List of default visualizer config instances.
        """
        default_configs = []

        for viz_type in requested_visualizers:
            try:
                if viz_type == "newton":
                    # Create default Newton visualizer config
                    default_configs.append(NewtonVisualizerCfg())
                elif viz_type == "rerun":
                    # Create default Rerun visualizer config
                    default_configs.append(RerunVisualizerCfg())
                elif viz_type == "omniverse":
                    # Create default Omniverse visualizer config
                    default_configs.append(OVVisualizerCfg())
                else:
                    logger.warning(
                        f"[SimulationContext] Unknown visualizer type '{viz_type}' requested. "
                        "Valid types: 'newton', 'rerun', 'omniverse'. Skipping."
                    )
            except Exception as e:
                logger.error(f"[SimulationContext] Failed to create default config for visualizer '{viz_type}': {e}")

        return default_configs

    def initialize_visualizers(self) -> None:
        """Initialize visualizers based on the --visualizer command-line flag.

        This method creates and initializes visualizers only when explicitly requested via
        the --visualizer flag. It supports:
        - Single visualizer: --visualizer rerun
        - Multiple visualizers: --visualizer rerun newton omniverse
        - No visualizers: omit the --visualizer flag (default behavior)

        If visualizer configs are defined in SimulationCfg.visualizer_cfgs, they will be used.
        Otherwise, default configs with all default parameters will be automatically created.

        Note:
            - If --headless is specified, NO visualizers will be initialized (headless takes precedence).
            - If --visualizer is not specified, NO visualizers will be initialized.
            - If --visualizer is specified but no configs exist, default configs are created automatically.
            - Only visualizers specified via --visualizer will be initialized, even if
              multiple visualizer configs are present in the simulation config.
        """

        # Check if specific visualizers were requested via command-line flag
        requested_visualizers_str = self._sim.settings.get("/isaaclab/visualizer")
        if requested_visualizers_str is None:
            requested_visualizers_str = ""

        # Parse comma-separated visualizer list
        requested_visualizers = [v.strip() for v in requested_visualizers_str.split(",") if v.strip()]

        # If no visualizers were requested via --visualizer flag, skip initialization
        if not requested_visualizers:
            # Skip if no GUI and no offscreen rendering (true headless mode)
            if not self._sim._has_gui and not self._sim._offscreen_render:
                return
            logger.info(
                "[SimulationContext] No visualizers specified via --visualizer flag. "
                "Skipping visualizer initialization. Use --visualizer <type> to enable visualizers."
            )
            return

        # If in true headless mode (no GUI, no offscreen rendering) but visualizers were requested,
        # filter out visualizers that require GUI (like omniverse)
        if not self._sim._has_gui and not self._sim._offscreen_render:
            # Only non-GUI visualizers (rerun, newton) can run in headless mode
            non_gui_visualizers = [v for v in requested_visualizers if v in ["rerun", "newton"]]
            if not non_gui_visualizers:
                logger.warning(
                    "[SimulationContext] Headless mode enabled but only GUI-dependent visualizers "
                    f"(like 'omniverse') were requested: {requested_visualizers}. "
                    "Skipping all visualizer initialization."
                )
                return
            if len(non_gui_visualizers) < len(requested_visualizers):
                logger.info(
                    "[SimulationContext] Headless mode enabled. Filtering visualizers from "
                    f"{requested_visualizers} to {non_gui_visualizers} (excluding GUI-dependent visualizers)."
                )
            requested_visualizers = non_gui_visualizers

        # Handle different input formats
        visualizer_cfgs = []
        if self._sim.cfg.visualizer_cfgs is not None:
            if isinstance(self._sim.cfg.visualizer_cfgs, list):
                visualizer_cfgs = self._sim.cfg.visualizer_cfgs
            else:
                visualizer_cfgs = [self._sim.cfg.visualizer_cfgs]

        # If no visualizer configs are defined but visualizers were requested, create default configs
        if len(visualizer_cfgs) == 0:
            logger.info(
                "[SimulationContext] No visualizer configs found in simulation config. "
                f"Creating default configs for requested visualizers: {requested_visualizers}"
            )
            visualizer_cfgs = self._create_default_visualizer_configs(requested_visualizers)
        else:
            # Filter visualizers based on --visualizer flag
            original_count = len(visualizer_cfgs)

            # Filter to only requested visualizers
            visualizer_cfgs = [cfg for cfg in visualizer_cfgs if cfg.visualizer_type in requested_visualizers]

            if len(visualizer_cfgs) == 0:
                available_types = [
                    cfg.visualizer_type
                    for cfg in (
                        self._sim.cfg.visualizer_cfgs
                        if isinstance(self._sim.cfg.visualizer_cfgs, list)
                        else [self._sim.cfg.visualizer_cfgs]
                    )
                    if cfg.visualizer_type is not None
                ]
                logger.warning(
                    f"[SimulationContext] Visualizer(s) {requested_visualizers} requested via --visualizer flag, "
                    "but no matching visualizer configs were found in simulation config. "
                    f"Available visualizer types: {available_types}"
                )
                return
            elif len(visualizer_cfgs) < original_count:
                logger.info(
                    f"[SimulationContext] Visualizer(s) {requested_visualizers} specified via --visualizer flag. "
                    f"Filtering {original_count} configs to {len(visualizer_cfgs)} matching visualizer(s)."
                )

        # Create scene data provider with visualizer configs
        # Provider will determine which backends are active
        if visualizer_cfgs:
            self._scene_data_provider = SceneDataProvider(visualizer_cfgs)

        # Create and initialize each visualizer
        for viz_cfg in visualizer_cfgs:
            try:
                visualizer = viz_cfg.create_visualizer()

                # Build scene data dict with only what this visualizer needs
                scene_data = {}

                # Newton and Rerun visualizers only need scene_data_provider
                if viz_cfg.visualizer_type in ("newton", "rerun"):
                    scene_data["scene_data_provider"] = self._scene_data_provider

                # OV visualizer needs USD stage and simulation context
                elif viz_cfg.visualizer_type == "omniverse":
                    scene_data["usd_stage"] = self._sim.stage
                    scene_data["simulation_context"] = self._sim

                # Initialize visualizer with minimal required data
                visualizer.initialize(scene_data)
                self._visualizers.append(visualizer)
                logger.info(f"Initialized visualizer: {type(visualizer).__name__} (type: {viz_cfg.visualizer_type})")

            except Exception as e:
                logger.error(
                    f"Failed to initialize visualizer '{viz_cfg.visualizer_type}' ({type(viz_cfg).__name__}): {e}"
                )

    def step_visualizers(self, dt: float) -> None:
        """Update all active visualizers.

        This method steps all initialized visualizers and updates their state.
        It also handles visualizer pause states and removes closed visualizers.

        Args:
            dt: Time step in seconds.
        """
        if not self._visualizers:
            return

        self._visualizer_step_counter += 1

        # Update visualizers and check if any should be removed
        visualizers_to_remove = []

        for visualizer in self._visualizers:
            try:
                # Check if visualizer is still running
                if not visualizer.is_running():
                    visualizers_to_remove.append(visualizer)
                    continue

                # Handle training pause - block until resumed
                while visualizer.is_training_paused() and visualizer.is_running():
                    # Visualizers fetch backend-specific state themselves
                    visualizer.step(0.0, state=None)

                # Always call step to process events, even if rendering is paused
                # The visualizer's step() method handles pause state internally
                visualizer.step(dt, state=None)

            except Exception as e:
                logger.error(f"Error stepping visualizer '{type(visualizer).__name__}': {e}")
                visualizers_to_remove.append(visualizer)

        # Remove closed visualizers
        for visualizer in visualizers_to_remove:
            try:
                visualizer.close()
                self._visualizers.remove(visualizer)
                logger.info(f"Removed visualizer: {type(visualizer).__name__}")
            except Exception as e:
                logger.error(f"Error closing visualizer: {e}")

    def close_visualizers(self) -> None:
        """Close all active visualizers and clean up resources."""
        for visualizer in self._visualizers:
            try:
                visualizer.close()
            except Exception as e:
                logger.error(f"Error closing visualizer '{type(visualizer).__name__}': {e}")

        self._visualizers.clear()
        logger.info("All visualizers closed")

    def has_omniverse_visualizer(self) -> bool:
        """Returns whether the Omniverse visualizer is enabled.

        This checks both the configuration (before initialization) and the active visualizers
        (after initialization) to determine if the Omniverse visualizer will be or is active.

        Returns:
            True if the Omniverse visualizer is requested or active, False otherwise.
        """

        # Check LAUNCH_OV_APP environment variable (useful for tests that need Omniverse)
        launch_app_env = int(os.environ.get("LAUNCH_OV_APP") or 0)
        if launch_app_env == 1:
            return True

        # First, check if already initialized visualizers include OVVisualizer
        for visualizer in self._visualizers:
            # Check if visualizer has visualizer_type attribute set to "omniverse"
            if hasattr(visualizer, "cfg") and hasattr(visualizer.cfg, "visualizer_type"):
                if visualizer.cfg.visualizer_type == "omniverse":
                    return True
            # Alternative: check the class name
            if type(visualizer).__name__ == "OVVisualizer":
                return True

        # If not initialized yet, check the configuration/settings
        requested_visualizers_str = self._sim.settings.get("/isaaclab/visualizer")
        if requested_visualizers_str:
            requested_visualizers = [v.strip() for v in requested_visualizers_str.split(",") if v.strip()]
            if "omniverse" in requested_visualizers:
                # Only return True if we have a GUI (omniverse requires GUI)
                return self._sim._has_gui

        return False

    def set_camera_view(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        """Set the location and target of the viewport camera in the stage.

        This method sets the camera view by calling the OVVisualizer's set_camera_view method.
        If no OVVisualizer is active, this method has no effect.

        Args:
            eye: The location of the camera eye.
            target: The location of the camera target.
            camera_prim_path: The path to the camera primitive in the stage. Defaults to
                "/OmniverseKit_Persp". Note: This parameter is ignored as the camera path
                is determined by the active viewport.
        """
        # Find the Omniverse visualizer and call its set_camera_view method
        for visualizer in self._visualizers:
            if hasattr(visualizer, "cfg") and hasattr(visualizer.cfg, "visualizer_type"):
                if visualizer.cfg.visualizer_type == "omniverse":
                    if hasattr(visualizer, "set_camera_view"):
                        visualizer.set_camera_view(eye, target)
                        return
            # Alternative: check the class name
            if type(visualizer).__name__ == "OVVisualizer":
                if hasattr(visualizer, "set_camera_view"):
                    visualizer.set_camera_view(eye, target)
                    return

        logger.debug("No Omniverse visualizer found - set_camera_view has no effect.")

    def update_scene_data(self) -> None:
        """Update scene data provider (syncs fabric transforms if needed)."""
        if self._scene_data_provider:
            self._scene_data_provider.update()

    def on_play(self) -> None:
        """Called when simulation starts - handles OV timeline."""
        if self.has_omniverse_visualizer():
            import omni.kit.app
            import omni.timeline

            omni.timeline.get_timeline_interface().play()
            omni.timeline.get_timeline_interface().commit()
            self._sim.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

    def on_stop(self) -> None:
        """Called when simulation stops - handles OV timeline."""
        # this only applies for omniverse mode
        if self.has_omniverse_visualizer():
            import omni.kit.app
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
            self._sim.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

    def render(self, mode) -> bool:
        """Handle rendering.
        
        Args:
            mode: The rendering mode.
            
        Returns:
            True if rendering was handled, False otherwise.
        """
        import builtins

        # pass if omniverse is not running
        if not self.has_omniverse_visualizer():
            return False

        import omni.kit.app

        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
        # check if we need to change the render mode
        if mode is not None:
            self._sim.set_render_mode(mode)
        # render based on the render mode
        if self._sim.render_mode == self._sim.RenderMode.NO_GUI_OR_RENDERING:
            # we never want to render anything here (this is for complete headless mode)
            pass
        elif self._sim.render_mode == self._sim.RenderMode.NO_RENDERING:
            # throttle the rendering frequency to keep the UI responsive
            self._sim._render_throttle_counter += 1
            if self._sim._render_throttle_counter % self._sim._render_throttle_period == 0:
                self._sim._render_throttle_counter = 0
                # here we don't render viewport so don't need to flush fabric data
                # note: we don't call super().render() anymore because they do flush the fabric data
                self._sim.settings.set_bool("/app/player/playSimulations", False)
                omni.kit.app.get_app().update()
        else:
            # manually flush the fabric data to update Hydra textures
            self._sim.forward()
            # render the simulation
            # note: we don't call super().render() anymore because they do above operation inside
            #  and we don't want to do it twice. We may remove it once we drop support for Isaac Sim 2022.2.
            self._sim.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

        # app.update() may be changing the cuda device, so we force it back to our desired device here
        if "cuda" in self._sim.device:
            import torch
            torch.cuda.set_device(self._sim.device)

        return True

    def get_rendering_dt(self) -> float | None:
        """Get the current rendering dt for OV mode.
        
        Returns:
            The rendering dt if OV mode, None otherwise.
        """
        if not self.has_omniverse_visualizer():
            return None

        if self._sim.stage is None:
            raise Exception("There is no stage currently opened")

        # Helper function to get dt from frequency
        def _get_dt_from_frequency():
            frequency = self._sim.settings.get("/app/runLoops/main/rateLimitFrequency")
            return 1.0 / frequency if frequency else 0

        if self._sim.settings.get("/app/runLoops/main/rateLimitEnabled"):
            return _get_dt_from_frequency()

        try:
            import omni.kit.loop._loop as omni_loop

            _loop_runner = omni_loop.acquire_loop_interface()
            if _loop_runner.get_manual_mode():
                return _loop_runner.get_manual_step_size()
            else:
                return _get_dt_from_frequency()
        except Exception:
            return _get_dt_from_frequency()
