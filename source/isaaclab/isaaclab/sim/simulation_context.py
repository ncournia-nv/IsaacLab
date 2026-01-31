# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import gc
import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pxr import UsdUtils

import isaaclab.sim.utils.stage as stage_utils

from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim.utils import create_new_stage_in_memory, raise_callback_exception_if_any
from .physics_interface import PhysicsInterface
from .render_interface import RenderInterface
from .simulation_cfg import SimulationCfg
from .visualizer_interface import VisualizerInterface
from .spawners import DomeLightCfg, GroundPlaneCfg

# import logger
logger = logging.getLogger(__name__)


class SimulationContext:
    """Controls simulation lifecycle including physics stepping and rendering.

    This singleton class manages:

    * Physics configuration (time-step, solver parameters via :class:`isaaclab.sim.SimulationCfg`)
    * Simulation state (play, pause, step, stop)
    * Rendering and visualization

    The singleton instance can be accessed using the ``instance()`` class method.

    .. attention::
        Only the PyTorch backend is supported. All data structures are ``torch.Tensor`` objects.
    """

    # Singleton instance
    _instance: "SimulationContext | None" = None

    def __new__(cls, cfg: SimulationCfg | None = None):
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def instance(cls) -> "SimulationContext | None":
        """Get the singleton instance, or None if not created."""
        return cls._instance

    def __init__(self, cfg: SimulationCfg | None = None):
        """Initialize the simulation context.

        Args:
            cfg: Simulation configuration. Defaults to None (uses default config).
        """
        # Skip initialization if already initialized (singleton pattern)
        if self._initialized:
            return

        # store input
        self.cfg = SimulationCfg() if cfg is None else cfg
        self.device = self.cfg.device

        # get existing stage or create new one in memory
        stage_cache = UsdUtils.StageCache.Get()
        all_stages = stage_cache.GetAllStages() if stage_cache.Size() > 0 else []
        self.stage = all_stages[0] if all_stages else create_new_stage_in_memory()

        # acquire settings interface
        self.settings = SettingsManager.instance()

        # Initialize Interfaces
        self._visualizer_interface: VisualizerInterface = VisualizerInterface(self)
        self._render_interface: RenderInterface = RenderInterface(self)
        self._physics_interface: PhysicsInterface = PhysicsInterface(self)

        # define a global variable to store the exceptions raised in the callback stack
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None

        self._is_playing = False
        self._initialized = True

    def set_setting(self, name: str, value: Any):
        """Set a Carbonite setting value.

        Args:
            name: The setting name (e.g., "/physics/cudaDevice").
            value: The value to set (bool, int, float, str, list, or tuple).
        """
        # Route through typed setters for correctness and consistency for common scalar types.
        if isinstance(value, bool):
            self.settings.set_bool(name, value)
        elif isinstance(value, int):
            self.settings.set_int(name, value)
        elif isinstance(value, float):
            self.settings.set_float(name, value)
        elif isinstance(value, str):
            self.settings.set_string(name, value)
        elif isinstance(value, (list, tuple)):
            self.settings.set(name, value)
        else:
            raise ValueError(f"Unsupported value type for setting '{name}': {type(value)}")

    def get_setting(self, name: str) -> Any:
        """Get a Carbonite setting value."""
        return self.settings.get(name)

    def forward(self) -> None:
        """Update kinematics and sync scene data without stepping physics."""
        self._physics_interface.forward_kinematics()
        self._visualizer_interface.step_visualizers(0.0)

    def reset(self, soft: bool = False):
        """Reset the simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        raise_callback_exception_if_any()
        self._physics_interface.reset(soft)
        self._visualizer_interface.reset(soft)
        self._is_playing = True

    def step(self, render: bool = True):
        """Step physics, update visualizers, and optionally render.

        Args:
            render: Whether to render the scene after stepping. Defaults to True.
        """
        raise_callback_exception_if_any()

        # Keep UI responsive while paused
        while not self._is_playing:
            self._visualizer_interface.render(mode=None)

        self._physics_interface.step_simulation()
        self._visualizer_interface.step_visualizers(self.cfg.dt)  # Sync + step first
        if render:
            self._visualizer_interface.render(mode=None)  # Display last

    def is_playing(self) -> bool:
        """Returns True if simulation is playing."""
        return self._is_playing

    def play(self):
        """Start the simulation."""
        self._visualizer_interface.on_play()
        self._is_playing = True

    def stop(self):
        """Stop the simulation."""
        self._visualizer_interface.on_stop()
        self._is_playing = False

    def render(self, mode: int | None = None):
        """Refresh rendering components (viewports, UI elements).

        Args:
            mode: Render mode. Defaults to None (use current mode).
        """
        self._visualizer_interface.render(mode)

    def get_physics_dt(self) -> float:
        """Returns the physics time step."""
        return self._physics_interface.physics_dt

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step."""
        ov_dt = self._visualizer_interface.get_rendering_dt()
        if ov_dt is not None:
            return ov_dt
        return self.cfg.dt

    def clear_all_callbacks(self) -> None:
        """Clear all callbacks and trigger garbage collection."""
        gc.collect()

    def clear_instance(self):
        """Clean up resources and clear the singleton instance."""
        # clear the callback
        if hasattr(self, "_app_control_on_stop_handle") and self._app_control_on_stop_handle is not None:
            self._app_control_on_stop_handle.unsubscribe()
            self._app_control_on_stop_handle = None
        # close all visualizers
        self._visualizer_interface.close_visualizers()
        # clear stage references
        if hasattr(self, "stage"):
            self.stage = None
        # reset initialization flag
        self._initialized = False
        # clear the singleton instance
        type(self)._instance = None
        self._physics_interface.clear()


@contextmanager
def build_simulation_context(
    create_new_stage: bool = True,
    gravity_enabled: bool = True,
    device: str = "cuda:0",
    dt: float = 0.01,
    sim_cfg: SimulationCfg | None = None,
    add_ground_plane: bool = False,
    add_lighting: bool = False,
    auto_add_lighting: bool = False,
) -> Iterator[SimulationContext]:
    """Context manager to build a simulation context with the provided settings.

    This function facilitates the creation of a simulation context and provides flexibility in configuring various
    aspects of the simulation, such as time step, gravity, device, and scene elements like ground plane and
    lighting.

    If :attr:`sim_cfg` is None, then an instance of :class:`SimulationCfg` is created with default settings, with parameters
    overwritten based on arguments to the function.

    An example usage of the context manager function:

    ..  code-block:: python

        with build_simulation_context() as sim:
             # Design the scene

             # Play the simulation
             sim.reset()
             while sim.is_playing():
                 sim.step()

    Args:
        create_new_stage: Whether to create a new stage. Defaults to True.
        gravity_enabled: Whether to enable gravity in the simulation. Defaults to True.
        device: Device to run the simulation on. Defaults to "cuda:0".
        dt: Time step for the simulation: Defaults to 0.01.
        sim_cfg: :class:`isaaclab.sim.SimulationCfg` to use for the simulation. Defaults to None.
        add_ground_plane: Whether to add a ground plane to the simulation. Defaults to False.
        add_lighting: Whether to add a dome light to the simulation. Defaults to False.
        auto_add_lighting: Whether to automatically add a dome light to the simulation if the simulation has a GUI.
            Defaults to False. This is useful for debugging tests in the GUI.

    Yields:
        The simulation context to use for the simulation.

    """
    try:
        if create_new_stage:
            stage_utils.create_new_stage()

        if sim_cfg is None:
            # Construct one and overwrite the dt, gravity, and device
            sim_cfg = SimulationCfg(dt=dt)
            sim_cfg.gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
            sim_cfg.device = device
        # Construct simulation context
        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            # Ground-plane
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim._visualizer_interface.has_gui()):
            # Lighting
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1), enable_color_temperature=True, color_temperature=5500, intensity=10000
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        # Only stop programmatically in headless mode - with GUI, the app manages its own lifecycle
        if not sim._visualizer_interface.has_gui():
            sim.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()
        raise_callback_exception_if_any()
