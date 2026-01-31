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

# import omni.physx
# import omni.usd
# from isaacsim.core.api.simulation_context import SimulationContext as _SimulationContext
# from isaacsim.core.simulation_manager import SimulationManager
# from isaacsim.core.utils.viewports import set_camera_view
# from isaacsim.core.version import get_version
# from omni.physics.stageupdate import get_physics_stage_update_node_interface
from pxr import UsdUtils

import isaaclab.sim.utils.stage as stage_utils

# Import settings manager for both Omniverse and standalone modes
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim.utils import create_new_stage_in_memory
from .physics_interface import PhysicsInterface
from .render_interface import RenderInterface
from .simulation_cfg import SimulationCfg
from .visualizer_interface import VisualizerInterface
from .spawners import DomeLightCfg, GroundPlaneCfg

# import logger
logger = logging.getLogger(__name__)


class SimulationContext:
    """A class to control simulation-related events such as physics stepping and rendering.

    The simulation context helps control various simulation aspects. This includes:

    * configure the simulator with different settings such as the physics time-step, the number of physics substeps,
      and the physics solver parameters (for more information, see :class:`isaaclab.sim.SimulationCfg`)
    * playing, pausing, stepping and stopping the simulation
    * adding and removing callbacks to different simulation events such as physics stepping, rendering, etc.

    This class implements a singleton pattern to ensure only one simulation context exists at a time.
    The singleton instance can be accessed using the ``instance()`` class method.

    The simulation context is a singleton object. This means that there can only be one instance
    of the simulation context at any given time. Therefore, it is not possible to create multiple
    instances of the simulation context. Instead, the simulation context can be accessed using the
    ``instance()`` method.

    .. attention::
        Since we only support the `PyTorch <https://pytorch.org/>`_ backend for simulation, the
        simulation context is configured to use the ``torch`` backend by default. This means that
        all the data structures used in the simulation are ``torch.Tensor`` objects.

    The simulation context can be used in two different modes of operations:

    1. **Standalone python script**: In this mode, the user has full control over the simulation and
       can trigger stepping events synchronously (i.e. as a blocking call). In this case the user
       has to manually call :meth:`step` step the physics simulation and :meth:`render` to
       render the scene.
    2. **Omniverse extension**: In this mode, the user has limited control over the simulation stepping
       and all the simulation events are triggered asynchronously (i.e. as a non-blocking call). In this
       case, the user can only trigger the simulation to start, pause, and stop. The simulation takes
       care of stepping the physics simulation and rendering the scene.

    Based on above, for most functions in this class there is an equivalent function that is suffixed
    with ``_async``. The ``_async`` functions are used in the Omniverse extension mode and
    the non-``_async`` functions are used in the standalone python script mode.
    """

    # Singleton instance
    _instance: "SimulationContext | None" = None

    def __new__(cls, cfg: SimulationCfg | None = None):
        """Enforce singleton pattern by returning existing instance if available.

        Args:
            cfg: The configuration of the simulation. Ignored if instance already exists.

        Returns:
            The singleton instance of SimulationContext.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def instance(cls) -> "SimulationContext | None":
        """Get the singleton instance of the simulation context.

        Returns:
            The singleton instance if it exists, None otherwise.
        """
        return cls._instance

    def __init__(self, cfg: SimulationCfg | None = None):
        """Creates a simulation context to control the simulator.

        Args:
            cfg: The configuration of the simulation. Defaults to None,
                in which case the default configuration is used.
        """
        # Skip initialization if already initialized (singleton pattern)
        if self._initialized:
            return

        # store input
        if cfg is None:
            cfg = SimulationCfg()
        # check that the config is valid
        cfg.validate()
        self.cfg = cfg

        # get existing stage or create new one in memory
        stage_cache = UsdUtils.StageCache.Get()
        all_stages = stage_cache.GetAllStages() if stage_cache.Size() > 0 else []
        self.stage = all_stages[0] if all_stages else create_new_stage_in_memory()
        self.device = self.cfg.device
        # acquire settings interface
        # Use settings manager (works in both Omniverse and standalone modes)
        self.settings = SettingsManager.instance()

        # Initialize visualizer interface early (needed for RenderMode access)
        self._visualizer_interface: VisualizerInterface = VisualizerInterface(self)
        # Initialize render interface for rendering configuration
        self._render_interface: RenderInterface = RenderInterface(self)
        # Initialize physics interface early to configure physics and Newton
        self._physics_interface: PhysicsInterface = PhysicsInterface(self)

        # override enable scene querying if rendering is enabled
        # this is needed for some GUI features
        if self._visualizer_interface.has_gui():
            self.cfg.enable_scene_query_support = True
        # read isaac sim version (this includes build tag, release tag etc.)
        # note: we do it once here because it reads the VERSION file from disk and is not expected to change.
        # self._isaacsim_version = get_version()

        # define a global variable to store the exceptions raised in the callback stack
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None

        # flag for skipping prim deletion callback
        # when stage in memory is attached
        self._skip_next_prim_deletion_callback_fn = False

        self._is_playing = False
        self.physics_sim_view = None

        # Mark as initialized (singleton pattern)
        self._initialized = True

    def set_setting(self, name: str, value: Any):
        """Set simulation settings using the Carbonite SDK.

        .. note::
            If the input setting name does not exist, it will be created. If it does exist, the value will be
            overwritten. Please make sure to use the correct setting name.

            To understand the settings interface, please refer to the
            `Carbonite SDK <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/settings.html>`_
            documentation.

        Args:
            name: The name of the setting.
            value: The value of the setting.
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
        """Read the simulation setting using the Carbonite SDK.

        Args:
            name: The name of the setting.

        Returns:
            The value of the setting.
        """
        return self.settings.get(name)

    def forward(self) -> None:
        """Updates articulation kinematics and scene data for rendering."""
        self._physics_interface.forward_kinematics()
        # Update scene data provider (syncs fabric transforms if needed)
        self._visualizer_interface.update_scene_data()

    """
    Operations - Override (standalone)
    """

    def reset(self, soft: bool = False):
        # # check if we need to raise an exception that was raised in a callback
        # if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
        #     exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
        #     builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        #     raise exception_to_raise

        self._physics_interface.reset(soft)
        self._visualizer_interface.reset(soft)
        self._is_playing = True

    def step(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                    If set to False, the scene is not rendered and only the physics simulation is stepped.
        """
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise

        # check if the simulation timeline is paused. in that case keep stepping until it is playing
        if not self.is_playing():
            # step the simulator (but not the physics) to have UI still active
            while not self.is_playing():
                self._visualizer_interface.render(mode=None)
                # meantime if someone stops, break out of the loop
                if self.is_stopped():
                    break
            # need to do one step to refresh the app
            # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
            #   without this the app becomes unresponsive.
            # FIXME: This steps physics as well, which we is not good in general.
            import omni.kit.app

            self.settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()

        # step the simulation
        if self.stage is None:
            raise Exception("There is no stage currently opened, init_stage needed before calling this func")

        self._physics_interface.step_simulation()
        if render:
            self._visualizer_interface.render(mode=None)
        self._visualizer_interface.step_visualizers(self.cfg.dt)

    def step_warp(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                    If set to False, the scene is not rendered and only the physics simulation is stepped.
        """

        self._physics_interface.step_simulation()
        if render:
            self._visualizer_interface.render(mode=None)
        if self.cfg.enable_newton_rendering:
            self._physics_interface.render()

    def is_playing(self) -> bool:
        """Checks if the simulation is playing.

        Returns:
            True if the simulation is playing, False otherwise.
        """
        return self._is_playing

    def play(self):
        """Starts the simulation."""
        self._visualizer_interface.on_play()
        self._is_playing = True

    def stop(self):
        """Stops the simulation."""
        self._visualizer_interface.on_stop()
        self._is_playing = False

    def render(self, mode: int | None = None):
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
        self._visualizer_interface.render(mode)

    def get_physics_dt(self) -> float:
        """Returns the physics time step.

        Returns:
            The physics time step.
        """
        return self._physics_interface.physics_dt

    def get_rendering_dt(self) -> float:
        """Get the current rendering dt

        Raises:
            Exception: if there is no stage currently opened

        Returns:
            float: current rendering dt

        Example:

        .. code-block:: python

            >>> simulation_context.get_rendering_dt()
            0.016666666666666666
        """
        ov_dt = self._visualizer_interface.get_rendering_dt()
        if ov_dt is not None:
            return ov_dt
        return self.cfg.dt

    """
    Initialization/Destruction - Override.
    """

    def clear_all_callbacks(self) -> None:
        """Clear all callbacks which were added using any ``add_*_callback`` method

        Example:

        .. code-block:: python

            >>> simulation_context.clear_render_callbacks()
        """
        # self._physics_callback_functions = dict()
        # self._physics_functions = dict()
        # self._stage_callback_functions = dict()
        # self._timeline_callback_functions = dict()
        # self._render_callback_functions = dict()
        gc.collect()
        return

    def clear_instance(self):
        """Clear the simulation context and clean up resources.

        This method should be called when you want to destroy the simulation context
        and create a new one with different settings.
        """
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

    """
    Helper Functions
    """


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

            # Set up gravity
            if gravity_enabled:
                sim_cfg.gravity = (0.0, 0.0, -9.81)
            else:
                sim_cfg.gravity = (0.0, 0.0, 0.0)

            # Set device
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
                color=(0.1, 0.1, 0.1),
                enable_color_temperature=True,
                color_temperature=5500,
                intensity=10000,
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        yield sim

    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        if not sim._visualizer_interface.has_gui():
            # Stop simulation only if we aren't rendering otherwise the app will hang indefinitely
            sim.stop()

        # Clear the stage
        sim.clear_all_callbacks()
        sim.clear_instance()
        # check if we need to raise an exception that was raised in a callback
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
