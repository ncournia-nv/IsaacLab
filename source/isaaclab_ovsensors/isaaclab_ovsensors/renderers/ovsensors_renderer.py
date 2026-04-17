# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ovsensors renderer backend for Isaac Lab.

Wraps ovsensors.Context to provide GPU-resident sensor output
via DLPack tensors — eliminating CPU copies in the hot path.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# Make ovsensors importable when running outside the installed environment.
_ovsensors_python = "/home/horde/ovsensors/ovsensors/python"
if _ovsensors_python not in sys.path:
    sys.path.insert(0, _ovsensors_python)
os.environ.setdefault("OVSENSORS_LIB_PATH", "/home/horde/ovsensors/ovsensors/build/lib")

try:
    from isaaclab.renderers.base_renderer import BaseRenderer
except ModuleNotFoundError:
    # Fallback for standalone use (no IsaacLab installation).
    from abc import ABC, abstractmethod

    class BaseRenderer(ABC):  # type: ignore[no-redef]
        """Minimal stand-in when isaaclab is not installed."""

        @abstractmethod
        def prepare_stage(self, stage, num_envs): ...

        @abstractmethod
        def create_render_data(self, sensor): ...

        @abstractmethod
        def set_outputs(self, render_data, output_data): ...

        @abstractmethod
        def update_transforms(self): ...

        @abstractmethod
        def update_camera(self, render_data, positions, orientations, intrinsics): ...

        @abstractmethod
        def render(self, render_data): ...

        @abstractmethod
        def read_output(self, render_data, camera_data): ...

        @abstractmethod
        def cleanup(self, render_data): ...


class OvsensorsRenderData:
    """Opaque render data object for OvsensorsRenderer."""

    def __init__(self):
        self.sensor_handles: list[int] = []
        self.prim_paths: list[str] = []
        # output name -> torch.Tensor or np.ndarray
        self.output_tensors: dict[str, Any] = {}
        self._last_outputs = None  # SensorOutputs from last step_sync call


class OvsensorsRendererCfg:
    """Configuration for OvsensorsRenderer.

    Can be used in place of the generic RendererCfg when constructing
    OvsensorsRenderer directly (without going through the Renderer factory).
    """

    renderer_type: str = "ovsensors"
    backend: str = "mock"  # "mock", "vulkan", "ovrtx", "ovrtx-ipc"
    width: int = 64
    height: int = 64
    usd_path: str | None = None
    num_cameras: int = 1

    def __init__(
        self,
        backend: str = "mock",
        width: int = 64,
        height: int = 64,
        usd_path: str | None = None,
        num_cameras: int = 1,
    ):
        self.renderer_type = "ovsensors"
        self.backend = backend
        self.width = width
        self.height = height
        self.usd_path = usd_path
        self.num_cameras = num_cameras


class OvsensorsRenderer(BaseRenderer):
    """Isaac Lab renderer backend wrapping ovsensors.Context.

    Key feature: GPU-resident output via DLPack — zero CPU copies when
    using a CUDA-capable backend (vulkan, ovrtx). The mock backend returns
    CPU tensors for testing without a GPU.

    Usage:
        cfg = OvsensorsRendererCfg(backend="mock")
        renderer = OvsensorsRenderer(cfg)
    """

    def __init__(self, cfg):
        # Accept either OvsensorsRendererCfg or a generic RendererCfg.
        self.cfg = cfg
        backend = getattr(cfg, "backend", "mock")
        width = getattr(cfg, "width", 64)
        height = getattr(cfg, "height", 64)
        self._width = width
        self._height = height

        from ovsensors import Context, ContextConfig

        self._ctx = Context(
            ContextConfig(
                backends=[backend],
                default_backend=backend,
            )
        )
        self._backend_name = backend
        self._usd_handle: int | None = None
        self._sensor_counter = 0

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Load USD stage into ovsensors (no-op if usd_path is not set or doesn't exist).

        Args:
            stage: USD stage to prepare, or None if not applicable.
            num_envs: Number of environments.
        """
        usd_path = getattr(self.cfg, "usd_path", None)
        if usd_path and os.path.exists(usd_path):
            self._usd_handle = self._ctx.add_usd_sync(usd_path)

    def create_render_data(self, sensor: Any) -> OvsensorsRenderData:
        """Create render data for one camera sensor.

        Args:
            sensor: The camera sensor or config dict. Inspected for
                ``num_instances`` to determine how many sensor handles to create.

        Returns:
            An :class:`OvsensorsRenderData` with pre-allocated sensor handles.
        """
        from ovsensors import SensorSpec

        data = OvsensorsRenderData()
        # sensor may be an Isaac Lab SensorBase or a plain config object.
        num_cameras = getattr(sensor, "num_instances", None) or getattr(self.cfg, "num_cameras", 1)
        for i in range(num_cameras):
            prim_path = f"/World/OvsensorsCamera_{self._sensor_counter}_{i}"
            handle = self._ctx.create_sensor(
                SensorSpec(
                    prim_path=prim_path,
                    kind="camera",
                    backend=self._backend_name,
                )
            )
            data.sensor_handles.append(handle)
            data.prim_paths.append(prim_path)
        self._sensor_counter += 1
        return data

    def set_outputs(self, render_data: OvsensorsRenderData, output_data: dict) -> None:
        """Store output buffer references for writing during render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            output_data: Dictionary mapping output names to pre-allocated tensors.
        """
        render_data.output_tensors = output_data

    def update_transforms(self) -> None:
        """Update scene transforms before rendering.

        No-op for ovsensors — transforms are pushed via :meth:`update_camera`.
        """

    def update_camera(
        self,
        render_data: OvsensorsRenderData,
        positions,
        orientations,
        intrinsics,
    ) -> None:
        """Push camera poses to ovsensors via write_attribute_sync.

        Builds a QUAT_T tensor of shape ``(N, 7)`` = ``[qx, qy, qz, qw, tx, ty, tz]``
        and writes it synchronously.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            positions: Camera positions [m] in world frame, shape ``(N, 3)``.
            orientations: Camera orientations as quaternions (x, y, z, w), shape ``(N, 4)``.
            intrinsics: Camera intrinsic matrices, shape ``(N, 3, 3)``.
        """
        if not render_data.prim_paths:
            return
        try:
            from ovsensors import Semantic

            # Handle torch tensors transparently.
            if hasattr(positions, "cpu"):
                positions = positions.cpu().numpy()
            if hasattr(orientations, "cpu"):
                orientations = orientations.cpu().numpy()

            n = len(render_data.prim_paths)
            quat_t = np.zeros((n, 7), dtype=np.float32)
            quat_t[:, :4] = np.asarray(orientations[:n]).reshape(n, 4)  # qx, qy, qz, qw
            quat_t[:, 4:] = np.asarray(positions[:n]).reshape(n, 3)  # tx, ty, tz

            # Pass a contiguous numpy array — Context._prepare_write calls
            # DLTensor.from_dlpack which consumes the numpy __dlpack__ protocol.
            self._ctx.write_attribute_sync(
                prim_paths=render_data.prim_paths,
                attr="xformOp:transform",
                tensor=quat_t,
                semantic=Semantic.XFORM_QUAT_T,
            )
            # Keep array alive across the call (already done by from_dlpack, but belt-and-suspenders).
            self._quat_t_keepalive = quat_t
        except Exception:
            # Non-fatal — camera will render at last known pose.
            pass

    def render(self, render_data: OvsensorsRenderData) -> None:
        """Step ovsensors and cache the outputs.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
        """
        if not render_data.sensor_handles:
            return
        # Destroy previous outputs before stepping.
        if render_data._last_outputs is not None:
            render_data._last_outputs.destroy()
            render_data._last_outputs = None
        render_data._last_outputs = self._ctx.step_sync(
            render_data.sensor_handles, delta_time=1.0 / 60.0
        )

    def read_output(self, render_data: OvsensorsRenderData, camera_data: Any) -> None:
        """Read rendered outputs from ovsensors into the camera data container.

        Uses DLPack for zero-copy access on CUDA backends. Falls back to
        zero-filled arrays when outputs are unavailable or stale.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            camera_data: A :class:`~isaaclab.sensors.camera.camera_data.CameraData`
                instance, or any object with an ``output`` dict attribute.
        """
        outputs = render_data._last_outputs
        if outputs is None:
            return

        from ovsensors import Device

        rgb_frames = []
        depth_frames = []

        def _map_to_array(tensor):
            if _HAS_CUPY and tensor.device.device_type.value == 2:  # kDLCUDA
                return cp.from_dlpack(tensor)
            return np.from_dlpack(tensor).copy()

        for sensor_out in outputs:
            if not sensor_out.metadata.is_fresh:
                rgb_frames.append(np.zeros((self._height, self._width, 4), dtype=np.uint8))
                depth_frames.append(np.zeros((self._height, self._width, 1), dtype=np.float32))
                continue

            # LdrColor -> "rgb" output via GPU path (Device.CUDA — zero-copy on CUDA backends).
            try:
                with sensor_out.map("LdrColor", device=Device.CUDA) as m:
                    rgb_frames.append(_map_to_array(m.tensor))
            except Exception:
                rgb_frames.append(np.zeros((self._height, self._width, 4), dtype=np.uint8))

            # Depth output via GPU path.
            try:
                with sensor_out.map("Depth", device=Device.CUDA) as m:
                    depth_frames.append(_map_to_array(m.tensor))
            except Exception:
                depth_frames.append(np.zeros((self._height, self._width, 1), dtype=np.float32))

        def _stack_frames(frames):
            if not frames:
                return None
            if _HAS_CUPY and any(hasattr(f, "device") for f in frames):
                return cp.stack([cp.asarray(f) for f in frames])
            return np.stack([np.asarray(f) for f in frames])

        # Write into pre-allocated output_tensors when present.
        if rgb_frames and "rgb" in render_data.output_tensors:
            rgb_stack = _stack_frames(rgb_frames)
            out = render_data.output_tensors["rgb"]
            if hasattr(out, "copy_"):
                import torch
                cpu = rgb_stack.get() if hasattr(rgb_stack, "get") else np.asarray(rgb_stack)
                out.copy_(torch.from_numpy(cpu))
            else:
                render_data.output_tensors["rgb"] = rgb_stack

        if depth_frames and "depth" in render_data.output_tensors:
            d_stack = _stack_frames(depth_frames)
            out = render_data.output_tensors["depth"]
            if hasattr(out, "copy_"):
                import torch
                cpu = d_stack.get() if hasattr(d_stack, "get") else np.asarray(d_stack)
                out.copy_(torch.from_numpy(cpu))
            else:
                render_data.output_tensors["depth"] = d_stack

        # Populate CameraData.output if present (Isaac Lab sensor interface).
        output_dict = getattr(camera_data, "output", None)
        if output_dict is not None:
            if rgb_frames:
                output_dict["rgb"] = _stack_frames(rgb_frames)
            if depth_frames:
                output_dict["depth"] = _stack_frames(depth_frames)

    def cleanup(self, render_data: OvsensorsRenderData) -> None:
        """Destroy sensor handles and release resources.

        Args:
            render_data: The render data object to clean up, or ``None``.
        """
        if render_data is None:
            return
        if render_data._last_outputs is not None:
            render_data._last_outputs.destroy()
            render_data._last_outputs = None
        for handle in render_data.sensor_handles:
            self._ctx.destroy_sensor(handle)
        render_data.sensor_handles.clear()
        render_data.prim_paths.clear()

    def destroy(self) -> None:
        """Tear down the ovsensors Context."""
        if self._usd_handle is not None:
            self._ctx.remove_usd_sync(self._usd_handle)
            self._usd_handle = None
        self._ctx.destroy()

    def reset_environment(self, env_id: int) -> None:
        """Signal episode boundary for env_id (flushes TAA/denoiser history).

        Args:
            env_id: Environment handle returned by :meth:`~ovsensors.Context.get_base_environment_handle`
                or :meth:`~ovsensors.Context.clone_environment_sync`.
        """
        try:
            self._ctx.reset_environment_sync(env_id)
        except Exception:
            pass
