# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import enum
import gc
import logging
import numpy as np
import os
import toml
import torch
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import flatdict

# import omni.physx
# import omni.usd
# from isaacsim.core.api.simulation_context import SimulationContext as _SimulationContext
# from isaacsim.core.simulation_manager import SimulationManager
# from isaacsim.core.utils.viewports import set_camera_view
# from isaacsim.core.version import get_version
# from omni.physics.stageupdate import get_physics_stage_update_node_interface
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

import isaaclab.sim.utils.stage as stage_utils

# Import settings manager for both Omniverse and standalone modes
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.sim.utils import create_new_stage_in_memory
from .simulation_cfg import SimulationCfg
from .visualizer_interface import VisualizerInterface
from .spawners import DomeLightCfg, GroundPlaneCfg
from .utils import bind_physics_material

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

    class RenderMode(enum.IntEnum):
        """Different rendering modes for the simulation.

        Render modes correspond to how the viewport and other UI elements (such as listeners to keyboard or mouse
        events) are updated. There are three main components that can be updated when the simulation is rendered:

        1. **UI elements and other extensions**: These are UI elements (such as buttons, sliders, etc.) and other
           extensions that are running in the background that need to be updated when the simulation is running.
        2. **Cameras**: These are typically based on Hydra textures and are used to render the scene from different
           viewpoints. They can be attached to a viewport or be used independently to render the scene.
        3. **Viewports**: These are windows where you can see the rendered scene.

        Updating each of the above components has a different overhead. For example, updating the viewports is
        computationally expensive compared to updating the UI elements. Therefore, it is useful to be able to
        control what is updated when the simulation is rendered. This is where the render mode comes in. There are
        four different render modes:

        * :attr:`NO_GUI_OR_RENDERING`: The simulation is running without a GUI and off-screen rendering flag is disabled,
          so none of the above are updated.
        * :attr:`NO_RENDERING`: No rendering, where only 1 is updated at a lower rate.
        * :attr:`PARTIAL_RENDERING`: Partial rendering, where only 1 and 2 are updated.
        * :attr:`FULL_RENDERING`: Full rendering, where everything (1, 2, 3) is updated.

        .. _Viewports: https://docs.omniverse.nvidia.com/extensions/latest/ext_viewport.html
        """

        NO_GUI_OR_RENDERING = -1
        """The simulation is running without a GUI and off-screen rendering is disabled."""
        NO_RENDERING = 0
        """No rendering, where only other UI elements are updated at a lower rate."""
        PARTIAL_RENDERING = 1
        """Partial rendering, where the simulation cameras and UI elements are updated."""
        FULL_RENDERING = 2
        """Full rendering, where all the simulation viewports, cameras and UI elements are updated."""

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

        # create or get stage using USD core APIs
        if self.cfg.create_stage_in_memory:
            # Create new stage in memory using USD core API
            self._initial_stage = create_new_stage_in_memory()
        else:
            # Try to get existing stage from USD StageCache
            stage_cache = UsdUtils.StageCache.Get()
            if stage_cache.Size() > 0:
                all_stages = stage_cache.GetAllStages()
                if all_stages:
                    self._initial_stage = all_stages[0]
                else:
                    raise RuntimeError("No USD stage found in StageCache. Please create a stage first.")
            else:
                # No stage exists, try omni.usd as fallback
                try:
                    import omni.usd

                    self._initial_stage = omni.usd.get_context().get_stage()
                except (ImportError, AttributeError):
                    # if we need to create a new stage outside of omni.usd, we have to do it in memory with USD core APIs
                    self._initial_stage = create_new_stage_in_memory()
                    # raise RuntimeError("No USD stage is currently open. Please create a stage first.")

        # Store stage reference for easy access
        self.stage = self._initial_stage

        # acquire settings interface
        # Use settings manager (works in both Omniverse and standalone modes)
        self.settings = SettingsManager.instance()

        # apply carb physics settings
        # SimulationManager._clear()
        self._apply_physics_settings()

        # note: we read this once since it is not expected to change during runtime
        # read flag for whether a local GUI is enabled
        self._local_gui = (
            self.settings.get("/app/window/enabled") if self.settings.get("/app/window/enabled") is not None else False
        )
        # read flag for whether livestreaming GUI is enabled
        self._livestream_gui = (
            self.settings.get("/app/livestream/enabled")
            if self.settings.get("/app/livestream/enabled") is not None
            else False
        )
        # read flag for whether XR GUI is enabled
        self._xr_gui = (
            self.settings.get("/app/xr/enabled") if self.settings.get("/app/xr/enabled") is not None else False
        )

        # read flag for whether the Isaac Lab viewport capture pipeline will be used,
        # casting None to False if the flag doesn't exist
        # this flag is set from the AppLauncher class
        self._offscreen_render = bool(self.settings.get("/isaaclab/render/offscreen"))
        # read flag for whether the default viewport should be enabled
        self._render_viewport = bool(self.settings.get("/isaaclab/render/active_viewport"))
        # flag for whether any GUI will be rendered (local, livestreamed or viewport)
        self._has_gui = self._local_gui or self._livestream_gui or self._xr_gui

        # apply render settings from render config
        self._apply_render_settings_from_cfg()

        # store the default render mode
        if not self._has_gui and not self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.NO_GUI_OR_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        elif not self._has_gui and self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = self.RenderMode.PARTIAL_RENDERING
            # set viewport context to None
            self._viewport_context = None
            self._viewport_window = None
        else:
            # note: need to import here in case the UI is not available (ex. headless mode)
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport

            # set default render mode
            # note: this can be changed by calling the `set_render_mode` function
            self.render_mode = self.RenderMode.FULL_RENDERING
            # acquire viewport context
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
            # acquire viewport window
            # TODO @mayank: Why not just use get_active_viewport_and_window() directly?
            self._viewport_window = ui.Workspace.get_window("Viewport")
            # counter for periodic rendering
            self._render_throttle_counter = 0
            # rendering frequency in terms of number of render calls
            self._render_throttle_period = 5

        # check the case where we don't need to render the viewport
        # since render_viewport can only be False in headless mode, we only need to check for offscreen_render
        if not self._render_viewport and self._offscreen_render:
            # disable the viewport if offscreen_render is enabled
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().updates_enabled = False

        # override enable scene querying if rendering is enabled
        # this is needed for some GUI features
        if self._has_gui:
            self.cfg.enable_scene_query_support = True
        # set up flatcache/fabric interface (default is None)
        # this is needed to flush the flatcache data into Hydra manually when calling `render()`
        # ref: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html
        # note: need to do this here because super().__init__ calls render and this variable is needed
        self._fabric_iface = None
        # read isaac sim version (this includes build tag, release tag etc.)
        # note: we do it once here because it reads the VERSION file from disk and is not expected to change.
        # self._isaacsim_version = get_version()

        # create a tensor for gravity
        # note: this line is needed to create a "tensor" in the device to avoid issues with torch 2.1 onwards.
        #   the issue is with some heap memory corruption when torch tensor is created inside the asset class.
        #   you can reproduce the issue by commenting out this line and running the test `test_articulation.py`.
        self._gravity_tensor = torch.tensor(self.cfg.gravity, dtype=torch.float32, device=self.cfg.device)

        # define a global variable to store the exceptions raised in the callback stack
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None

        self._disable_app_control_on_stop_handle = False

        # initialize visualizer interface
        self._visualizer_interface = VisualizerInterface(self)
        # flag for skipping prim deletion callback
        # when stage in memory is attached
        self._skip_next_prim_deletion_callback_fn = False

        # flatten out the simulation dictionary
        sim_params = self.cfg.to_dict()
        if sim_params is not None:
            if "newton_cfg" in sim_params:
                newton_params = sim_params.pop("newton_cfg")

        # initialize parameters here
        self.physics_dt = self.cfg.dt
        self.rendering_dt = self.cfg.dt * self.cfg.render_interval
        self.backend = "torch"
        self.physics_prim_path = self.cfg.physics_prim_path
        self.device = self.cfg.device

        # initialize physics scene
        physics_scene_prim = self.stage.GetPrimAtPath(self.cfg.physics_prim_path)
        if not physics_scene_prim.IsValid():
            self._physics_scene = UsdPhysics.Scene.Define(self.stage, self.cfg.physics_prim_path)
            physics_scene_prim = self.stage.GetPrimAtPath(self.cfg.physics_prim_path)
        else:
            self._physics_scene = UsdPhysics.Scene(physics_scene_prim)

        # Set physics dt (time steps per second) using string attribute name
        self._set_physx_scene_attr(
            physics_scene_prim, "physxScene:timeStepsPerSecond", int(1.0 / self.cfg.dt), Sdf.ValueTypeNames.Int
        )
        self.stage.SetTimeCodesPerSecond(1 / self.cfg.dt)

        # Set gravity on the physics scene
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        gravity_magnitude = abs(self.cfg.gravity[2])  # Get magnitude from z-component
        if up_axis == "Z":
            gravity_dir = Gf.Vec3f(0.0, 0.0, -1.0 if self.cfg.gravity[2] < 0 else 1.0)
        elif up_axis == "Y":
            gravity_dir = Gf.Vec3f(0.0, -1.0 if self.cfg.gravity[1] < 0 else 1.0, 0.0)
        else:
            gravity_dir = Gf.Vec3f(-1.0 if self.cfg.gravity[0] < 0 else 1.0, 0.0, 0.0)

        self._physics_scene.CreateGravityDirectionAttr().Set(gravity_dir)
        self._physics_scene.CreateGravityMagnitudeAttr().Set(gravity_magnitude)

        # Store physics scene prim reference
        self.physics_scene = physics_scene_prim

        # process device
        self._set_physics_sim_device()

        self._is_playing = False
        self.physics_sim_view = None

        self.settings.set_bool("/app/player/playSimulations", False)
        NewtonManager.set_simulation_dt(self.cfg.dt)
        NewtonManager._gravity_vector = self.cfg.gravity
        NewtonManager.set_solver_settings(newton_params)

        # create the default physics material
        # this material is used when no material is specified for a primitive
        material_path = f"{self.cfg.physics_prim_path}/defaultMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        # bind the physics material to the scene
        bind_physics_material(self.cfg.physics_prim_path, material_path)
        try:
            import omni.physx
            from omni.physics.stageupdate import get_physics_stage_update_node_interface

            physx_sim_interface = omni.physx.get_physx_simulation_interface()
            physx_sim_interface.detach_stage()
            get_physics_stage_update_node_interface().detach_node()
        except Exception:
            pass
        # Disable USD cloning if we are not rendering or using RTX sensors
        NewtonManager._clone_physics_only = (
            self.render_mode == self.RenderMode.NO_GUI_OR_RENDERING or self.render_mode == self.RenderMode.NO_RENDERING
        )

        # Mark as initialized (singleton pattern)
        self._initialized = True

    def _set_physx_scene_attr(self, prim: Usd.Prim, attr_name: str, value, value_type) -> None:
        """Helper to set a PhysX scene attribute using string-based attribute names.

        Args:
            prim: The physics scene prim.
            attr_name: The full attribute name (e.g., "physxScene:timeStepsPerSecond").
            value: The value to set.
            value_type: The Sdf.ValueTypeNames type for the attribute.
        """
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, value_type)
        attr.Set(value)

    def _set_physics_sim_device(self) -> None:
        """Sets the physics simulation device."""
        if "cuda" in self.device:
            parsed_device = self.device.split(":")
            if len(parsed_device) == 1:
                device_id = self.settings.get("/physics/cudaDevice", 0)
                if device_id < 0:
                    self.settings.set_int("/physics/cudaDevice", 0)
                    device_id = 0
                # resolve "cuda" to "cuda:N" for torch.cuda.set_device compatibility
                self.device = f"cuda:{device_id}"
            else:
                self.settings.set_int("/physics/cudaDevice", int(parsed_device[1]))
            self.settings.set_bool("/physics/suppressReadback", True)
            # Set GPU physics settings using string attribute names
            self._set_physx_scene_attr(self.physics_scene, "physxScene:broadphaseType", "GPU", Sdf.ValueTypeNames.Token)
            self._set_physx_scene_attr(
                self.physics_scene, "physxScene:enableGPUDynamics", True, Sdf.ValueTypeNames.Bool
            )
        elif self.device.lower() == "cpu":
            self.settings.set_bool("/physics/suppressReadback", False)
            # Set CPU physics settings using string attribute names
            self._set_physx_scene_attr(self.physics_scene, "physxScene:broadphaseType", "MBP", Sdf.ValueTypeNames.Token)
            self._set_physx_scene_attr(
                self.physics_scene, "physxScene:enableGPUDynamics", False, Sdf.ValueTypeNames.Bool
            )
        else:
            raise Exception(f"Device {self.device} is not supported.")

    def _apply_physics_settings(self):
        """Sets various carb physics settings."""
        # enable hydra scene-graph instancing
        # note: this allows rendering of instanceable assets on the GUI
        self.settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)

    def _apply_render_settings_from_cfg(self):
        """Sets rtx settings specified in the RenderCfg."""

        # define mapping of user-friendly RenderCfg names to native carb names
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

        # set preset settings (same behavior as the CLI arg --rendering_mode)
        rendering_mode = self.cfg.render_cfg.rendering_mode
        if rendering_mode is not None:
            # check if preset is supported
            supported_rendering_modes = ["performance", "balanced", "quality"]
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes {supported_rendering_modes}."
                )

            # parse preset file
            import carb

            repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
            preset_filename = os.path.join(repo_path, f"apps/rendering_modes/{rendering_mode}.kit")
            with open(preset_filename) as file:
                preset_dict = toml.load(file)
            preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

            # set presets
            for key, value in preset_dict.items():
                key = "/" + key.replace(".", "/")  # convert to carb setting format
                self.settings.set(key, value)

        # set user-friendly named settings
        for key, value in vars(self.cfg.render_cfg).items():
            if value is None or key in not_carb_settings:
                # skip unset settings and non-carb settings
                continue
            if key not in rendering_setting_name_mapping:
                raise ValueError(
                    f"'{key}' in RenderCfg not found. Note: internal 'rendering_setting_name_mapping' dictionary might"
                    " need to be updated."
                )
            key = rendering_setting_name_mapping[key]
            self.settings.set(key, value)

        # set general carb settings
        carb_settings = self.cfg.render_cfg.carb_settings
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")  # convert from python variable style string
                elif "." in key:
                    key = "/" + key.replace(".", "/")  # convert from .kit file style string
                if self.settings.get(key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.general_parameters does not map to a carb setting.")
                self.settings.set(key, value)

        # set denoiser mode
        if self.cfg.render_cfg.antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=self.cfg.render_cfg.antialiasing_mode)
            except Exception:
                pass

        # WAR: Ensure /rtx/renderMode RaytracedLighting is correctly cased.
        render_mode = self.settings.get("/rtx/rendermode")
        if render_mode is not None and render_mode.lower() == "raytracedlighting":
            self.settings.set("/rtx/rendermode", "RaytracedLighting")

    """
    Operations - New.
    """

    def has_gui(self) -> bool:
        """Returns whether the simulation has a GUI enabled.

        True if the simulation has a GUI enabled either locally or live-streamed.
        """
        return self._has_gui


    def has_rtx_sensors(self) -> bool:
        """Returns whether the simulation has any RTX-rendering related sensors.

        This function returns the value of the simulation parameter ``"/isaaclab/render/rtx_sensors"``.
        The parameter is set to True when instances of RTX-related sensors (cameras or LiDARs) are
        created using Isaac Lab's sensor classes.

        True if the simulation has RTX sensors (such as USD Cameras or LiDARs).

        For more information, please check `NVIDIA RTX documentation`_.

        .. _NVIDIA RTX documentation: https://developer.nvidia.com/rendering-technologies
        """
        return self.settings.get("/isaaclab/render/rtx_sensors")

    def is_fabric_enabled(self) -> bool:
        """Returns whether the fabric interface is enabled.

        When fabric interface is enabled, USD read/write operations are disabled. Instead all applications
        read and write the simulation state directly from the fabric interface. This reduces a lot of overhead
        that occurs during USD read/write operations.

        For more information, please check `Fabric documentation`_.

        .. _Fabric documentation: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html
        """
        return self._fabric_iface is not None

    """
    Operations - New utilities.
    """

    def set_render_mode(self, mode: RenderMode):
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the different render modes.

        .. note::
            When no GUI is available (locally or livestreamed), we do not need to choose whether the viewport
            needs to render or not (since there is no GUI). Thus, in this case, calling the function will not
            change the render mode.

        Args:
            mode (RenderMode): The rendering mode. If different than SimulationContext's rendering mode,
            SimulationContext's mode is changed to the new mode.

        Raises:
            ValueError: If the input mode is not supported.
        """
        # check if mode change is possible -- not possible when no GUI is available
        if not self._has_gui:
            logger.warning(
                f"Cannot change render mode when GUI is disabled. Using the default render mode: {self.render_mode}."
            )
            return
        # check if there is a mode change
        # note: this is mostly needed for GUI when we want to switch between full rendering and no rendering.
        if mode != self.render_mode:
            if mode == self.RenderMode.FULL_RENDERING:
                # display the viewport and enable updates
                self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
                self._viewport_window.visible = True  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == self.RenderMode.PARTIAL_RENDERING:
                # hide the viewport and disable updates
                self._viewport_context.updates_enabled = False  # pyright: ignore [reportOptionalMemberAccess]
                self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == self.RenderMode.NO_RENDERING:
                # hide the viewport and disable updates
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = False  # pyright: ignore [reportOptionalMemberAccess]
                    self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
                # reset the throttle counter
                self._render_throttle_counter = 0
            else:
                raise ValueError(f"Unsupported render mode: {mode}! Please check `RenderMode` for details.")
            # update render mode
            self.render_mode = mode

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
        NewtonManager.forward_kinematics()
        # Update scene data provider (syncs fabric transforms if needed)
        self._visualizer_interface.update_scene_data()

    def get_initial_stage(self) -> Usd.Stage:
        """Returns stage handle used during scene creation.

        Returns:
            The stage used during scene creation.
        """
        return self._initial_stage

    """
    Operations - Override (standalone)
    """

    def reset(self, soft: bool = False):
        self.settings.set_bool("/app/player/playSimulations", False)
        self._disable_app_control_on_stop_handle = True
        # # check if we need to raise an exception that was raised in a callback
        # if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
        #     exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
        #     builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        #     raise exception_to_raise

        if not soft:
            # if not self.is_stopped():
            #     self.stop()
            NewtonManager.start_simulation()
            # self.play()
            NewtonManager.initialize_solver()
            self._is_playing = True

        # app.update() may be changing the cuda device in reset, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)
        # enable kinematic rendering with fabric
        if self.physics_sim_view:
            self.physics_sim_view._backend.initialize_kinematic_bodies()
        # perform additional rendering steps to warm up replicator buffers
        # this is only needed for the first time we set the simulation
        if not soft:
            for _ in range(2):
                self.render()

        # Initialize visualizers after simulation is set up (only on first reset)
        if not soft and not self._visualizer_interface.visualizers:
            self._visualizer_interface.initialize_visualizers()

        self._disable_app_control_on_stop_handle = False

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
                self.render()
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

        if render:
            # physics dt is zero, no need to step physics, just render
            if self.is_playing():
                NewtonManager.step()
            if self.get_physics_dt() == 0:  # noqa: SIM114
                SimulationContext.render(self)
            # rendering dt is zero, but physics is not, call step and then render
            elif self.get_rendering_dt() == 0 and self.get_physics_dt() != 0:  # noqa: SIM114
                SimulationContext.render(self)
            else:
                import omni.kit.app

                self.settings.set_bool("/app/player/playSimulations", False)
                omni.kit.app.get_app().update()
        else:
            if self.is_playing():
                NewtonManager.step()

        # Update visualizers
        self._visualizer_interface.step_visualizers(self.cfg.dt)

        # app.update() may be changing the cuda device in step, so we force it back to our desired device here
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

    def step_warp(self, render: bool = True):
        """Steps the simulation.

        .. note::
            This function blocks if the timeline is paused. It only returns when the timeline is playing.

        Args:
            render: Whether to render the scene after stepping the physics simulation.
                    If set to False, the scene is not rendered and only the physics simulation is stepped.
        """

        if render:
            # physics dt is zero, no need to step physics, just render
            if self.is_playing():
                NewtonManager.step()
            if self.get_physics_dt() == 0:  # noqa: SIM114
                SimulationContext.render(self)
            # rendering dt is zero, but physics is not, call step and then render
            elif self.get_rendering_dt() == 0 and self.get_physics_dt() != 0:  # noqa: SIM114
                SimulationContext.render(self)
            else:
                self._app.update()
        else:
            if self.is_playing():
                NewtonManager.step()

        # Use the NewtonManager to render the scene if enabled
        if self.cfg.enable_newton_rendering:
            NewtonManager.render()

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

    def render(self, mode: RenderMode | None = None):
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
        return self.cfg.dt

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
        if hasattr(self, "_initial_stage"):
            self._initial_stage = None
        if hasattr(self, "stage"):
            self.stage = None
        # reset initialization flag
        self._initialized = False
        # clear the singleton instance
        type(self)._instance = None
        NewtonManager.clear()

    """
    Helper Functions
    """

    def _set_additional_physics_params(self):
        """Sets additional physical parameters that are not directly supported by the parent class."""
        # -- Gravity
        gravity = np.asarray(self.cfg.gravity)

        # Avoid division by zero
        gravity_direction = gravity

        NewtonManager._gravity_vector = gravity_direction

        # create the default physics material
        # this material is used when no material is specified for a primitive
        # check: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-materials
        material_path = f"{self.cfg.physics_prim_path}/defaultMaterial"
        self.cfg.physics_material.func(material_path, self.cfg.physics_material)
        # bind the physics material to the scene
        bind_physics_material(self.cfg.physics_prim_path, material_path)

    def _load_fabric_interface(self):
        """Loads the fabric interface if enabled."""
        if self.cfg.use_fabric:
            from omni.physxfabric import get_physx_fabric_interface

            # acquire fabric interface
            self._fabric_iface = get_physx_fabric_interface()
            if hasattr(self._fabric_iface, "force_update"):
                # The update method in the fabric interface only performs an update if a physics step has occurred.
                # However, for rendering, we need to force an update since any element of the scene might have been
                # modified in a reset (which occurs after the physics step) and we want the renderer to be aware of
                # these changes.
                self._update_fabric = self._fabric_iface.force_update
            else:
                # Needed for backward compatibility with older Isaac Sim versions
                self._update_fabric = self._fabric_iface.update


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
                NewtonManager._gravity_vector = (0.0, 0.0, -9.81)
            else:
                sim_cfg.gravity = (0.0, 0.0, 0.0)
                NewtonManager._gravity_vector = (0.0, 0.0, 0.0)

            # Set device
            sim_cfg.device = device

        # Construct simulation context
        sim = SimulationContext(sim_cfg)

        if add_ground_plane:
            # Ground-plane
            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)

        if add_lighting or (auto_add_lighting and sim.has_gui()):
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
        if not sim.has_gui():
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
