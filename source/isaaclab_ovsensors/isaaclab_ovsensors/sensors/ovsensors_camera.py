# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvsensorsCamera — standalone camera sensor using ovsensors.Context.

Can be used without an IsaacSim simulation context.
Demonstrates the RL loop pattern:

1. Create camera
2. Clone environment N times
3. Write attribute transforms each step
4. step_sync -> GPU tensors via DLPack
5. reset_environment at episode boundaries
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make ovsensors importable when running outside the installed environment.
_ovsensors_python = "/home/horde/ovsensors/ovsensors/python"
if _ovsensors_python not in sys.path:
    sys.path.insert(0, _ovsensors_python)
os.environ.setdefault("OVSENSORS_LIB_PATH", "/home/horde/ovsensors/ovsensors/build/lib")


class OvsensorsCamera:
    """Multi-environment camera sensor backed by ovsensors.

    Example (RL-style with 8 environments):

    .. code-block:: python

        cam = OvsensorsCamera(num_envs=8, width=64, height=64, backend="mock")
        cam.initialize("/World/Camera")
        env_handles = cam.clone_environments("/World", num_envs=8)

        for step in range(1000):
            cam.write_transforms(poses)   # shape [N, 7] QUAT_T
            outputs = cam.step()          # dict of np.ndarray

            for done_idx in done_env_ids:
                cam.reset_environment(env_handles[done_idx])

        cam.destroy()
    """

    def __init__(
        self,
        num_envs: int = 1,
        width: int = 64,
        height: int = 64,
        backend: str = "mock",
    ):
        """Create an OvsensorsCamera.

        Args:
            num_envs: Number of parallel environments (cameras).
            width: Image width in pixels.
            height: Image height in pixels.
            backend: ovsensors backend — ``"mock"``, ``"vulkan"``, ``"ovrtx"``,
                or ``"ovrtx-ipc"``.
        """
        from ovsensors import Context, ContextConfig

        self._ctx = Context(ContextConfig(backends=[backend], default_backend=backend))
        self._backend = backend
        self._width = width
        self._height = height
        self._num_envs = num_envs
        self._sensor_handles: list[int] = []
        self._env_handles: list[int] = []
        self._prim_path_prefix: str = "/World/Camera"
        self._xform_binding = None

    def initialize(self, prim_path_prefix: str = "/World/Camera") -> None:
        """Create sensor handles, one per environment.

        Args:
            prim_path_prefix: Base USD prim path; individual cameras are
                named ``<prefix>_0``, ``<prefix>_1``, ...
        """
        from ovsensors import SensorSpec

        self._prim_path_prefix = prim_path_prefix
        for i in range(self._num_envs):
            h = self._ctx.create_sensor(
                SensorSpec(
                    prim_path=f"{prim_path_prefix}_{i}",
                    kind="camera",
                    backend=self._backend,
                )
            )
            self._sensor_handles.append(h)

    def clone_environments(self, base_path: str, num_envs: int) -> list[int]:
        """Clone the base environment ``num_envs - 1`` times.

        Returns handles for all environments (base + clones).

        Args:
            base_path: USD path of the base environment to clone.
            num_envs: Total number of environments including the base.

        Returns:
            List of environment handles of length ``num_envs``.
        """
        if num_envs <= 1:
            base_handle = self._ctx.get_base_environment_handle(base_path)
            self._env_handles = [base_handle]
            return self._env_handles

        clone_handles = self._ctx.clone_environment_sync(base_path, num_clones=num_envs - 1)
        base_handle = self._ctx.get_base_environment_handle(base_path)
        self._env_handles = [base_handle] + list(clone_handles)
        return self._env_handles

    def write_transforms(self, poses: np.ndarray) -> None:
        """Write QUAT_T poses to all cameras.

        Passes the array directly to ``bind_attribute.write_sync`` which uses
        ``DLTensor.from_dlpack`` internally for zero-copy transfer.

        Args:
            poses: Array of shape ``(N, 7)`` with columns
                ``[qx, qy, qz, qw, tx, ty, tz]`` [rad, m].
        """
        from ovsensors import Semantic

        arr = np.ascontiguousarray(poses, dtype=np.float32)

        if self._xform_binding is None:
            prim_paths = [f"{self._prim_path_prefix}_{i}" for i in range(self._num_envs)]
            self._xform_binding = self._ctx.bind_attribute(
                prim_paths=prim_paths,
                attr="xformOp:transform",
                semantic=Semantic.XFORM_QUAT_T,
            )

        # AttributeBinding.write_sync accepts any object with __dlpack__ (numpy arrays qualify).
        self._xform_binding.write_sync(arr)
        # Keep array alive until the write completes (belt-and-suspenders).
        self._poses_keepalive = arr

    def step(self, delta_time: float = 1 / 60) -> dict[str, np.ndarray]:
        """Step all cameras and return their outputs.

        Args:
            delta_time: Simulation time step [s].

        Returns:
            Dictionary with keys ``"LdrColor"`` (shape ``(N, H, W, 4)``, uint8)
            and ``"Depth"`` (shape ``(N, H, W, 1)``, float32).
        """
        outputs = self._ctx.step_sync(self._sensor_handles, delta_time=delta_time)
        color_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []

        for sensor_out in outputs:
            try:
                with sensor_out.map("LdrColor") as m:
                    color_frames.append(m.tensor.numpy())
            except Exception:
                color_frames.append(
                    np.zeros((self._height, self._width, 4), dtype=np.uint8)
                )
            try:
                with sensor_out.map("Depth") as m:
                    depth_frames.append(m.tensor.numpy())
            except Exception:
                depth_frames.append(
                    np.zeros((self._height, self._width, 1), dtype=np.float32)
                )

        outputs.destroy()

        n = self._num_envs
        return {
            "LdrColor": (
                np.stack(color_frames)
                if color_frames
                else np.zeros((n, self._height, self._width, 4), dtype=np.uint8)
            ),
            "Depth": (
                np.stack(depth_frames)
                if depth_frames
                else np.zeros((n, self._height, self._width, 1), dtype=np.float32)
            ),
        }

    def reset_environment(self, env_handle: int) -> None:
        """Signal an episode boundary to flush temporal rendering state.

        Args:
            env_handle: Environment handle from :meth:`clone_environments`.
        """
        self._ctx.reset_environment_sync(env_handle)

    def destroy(self) -> None:
        """Release all resources."""
        if self._xform_binding is not None:
            self._xform_binding.unbind()
            self._xform_binding = None
        for h in self._sensor_handles:
            self._ctx.destroy_sensor(h)
        self._sensor_handles.clear()
        self._ctx.destroy()
