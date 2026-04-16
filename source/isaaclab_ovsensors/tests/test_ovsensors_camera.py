# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvsensorsCamera."""

import numpy as np
import pytest

from isaaclab_ovsensors.sensors import OvsensorsCamera


@pytest.fixture
def camera():
    """Yield a single-env OvsensorsCamera with mock backend."""
    cam = OvsensorsCamera(num_envs=1, width=64, height=64, backend="mock")
    yield cam
    cam.destroy()


# ---------------------------------------------------------------------------
# test_camera_creates
# ---------------------------------------------------------------------------


def test_camera_creates():
    """OvsensorsCamera instantiates with the mock backend without errors."""
    cam = OvsensorsCamera(num_envs=1, width=32, height=32, backend="mock")
    assert cam is not None
    cam.destroy()


# ---------------------------------------------------------------------------
# test_camera_initialize
# ---------------------------------------------------------------------------


def test_camera_initialize(camera):
    """initialize creates the expected number of sensor handles."""
    camera.initialize("/World/Camera")
    assert len(camera._sensor_handles) == 1


# ---------------------------------------------------------------------------
# test_camera_step
# ---------------------------------------------------------------------------


def test_camera_step(camera):
    """step returns dicts with correctly shaped arrays for a single env."""
    camera.initialize("/World/Camera")
    outputs = camera.step()
    assert "LdrColor" in outputs
    assert "Depth" in outputs
    assert outputs["LdrColor"].shape == (1, 64, 64, 4)
    assert outputs["Depth"].shape == (1, 64, 64, 1)


# ---------------------------------------------------------------------------
# test_camera_clone_environments
# ---------------------------------------------------------------------------


def test_camera_clone_environments():
    """clone_environments returns the requested number of handles."""
    cam = OvsensorsCamera(num_envs=4, width=32, height=32, backend="mock")
    cam.initialize("/World/Camera")
    handles = cam.clone_environments("/World", num_envs=4)
    assert len(handles) == 4
    cam.destroy()


# ---------------------------------------------------------------------------
# test_camera_reset_environment
# ---------------------------------------------------------------------------


def test_camera_reset_environment():
    """reset_environment does not raise and handles remain valid afterward."""
    cam = OvsensorsCamera(num_envs=2, width=32, height=32, backend="mock")
    cam.initialize("/World/Camera")
    handles = cam.clone_environments("/World", num_envs=2)
    # Reset the first environment — should not raise.
    cam.reset_environment(handles[0])
    # Sensor handles must still be intact.
    assert len(cam._sensor_handles) == 2
    cam.destroy()


# ---------------------------------------------------------------------------
# test_camera_write_transforms
# ---------------------------------------------------------------------------


def test_camera_write_transforms():
    """write_transforms accepts a (N, 7) float32 array without raising."""
    cam = OvsensorsCamera(num_envs=3, width=32, height=32, backend="mock")
    cam.initialize("/World/Camera")
    poses = np.zeros((3, 7), dtype=np.float32)
    poses[:, 3] = 1.0  # unit quaternion w=1
    # Should not raise.
    cam.write_transforms(poses)
    cam.destroy()


# ---------------------------------------------------------------------------
# test_camera_multi_env_step
# ---------------------------------------------------------------------------


def test_camera_multi_env_step():
    """4-env step returns tensors with leading dimension 4."""
    cam = OvsensorsCamera(num_envs=4, width=64, height=64, backend="mock")
    cam.initialize("/World/Camera")
    outputs = cam.step()
    assert outputs["LdrColor"].shape == (4, 64, 64, 4)
    assert outputs["Depth"].shape == (4, 64, 64, 1)
    cam.destroy()
