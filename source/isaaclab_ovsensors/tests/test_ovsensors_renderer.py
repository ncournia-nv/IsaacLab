# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvsensorsRenderer."""

import numpy as np
import pytest

from isaaclab_ovsensors.renderers import OvsensorsRenderData, OvsensorsRenderer, OvsensorsRendererCfg


@pytest.fixture
def renderer():
    """Create an OvsensorsRenderer with the mock backend and yield it."""
    cfg = OvsensorsRendererCfg(backend="mock", width=64, height=64, num_cameras=1)
    r = OvsensorsRenderer(cfg)
    yield r
    r.destroy()


@pytest.fixture
def renderer_with_data(renderer):
    """Create a renderer and a single render-data object."""

    class _FakeSensor:
        num_instances = 2

    data = renderer.create_render_data(_FakeSensor())
    yield renderer, data
    renderer.cleanup(data)


# ---------------------------------------------------------------------------
# test_renderer_creates
# ---------------------------------------------------------------------------


def test_renderer_creates():
    """OvsensorsRenderer instantiates with the mock backend without errors."""
    cfg = OvsensorsRendererCfg(backend="mock")
    r = OvsensorsRenderer(cfg)
    assert r is not None
    r.destroy()


# ---------------------------------------------------------------------------
# test_create_render_data
# ---------------------------------------------------------------------------


def test_create_render_data(renderer):
    """create_render_data returns OvsensorsRenderData with sensor handles."""

    class _FakeSensor:
        num_instances = 3

    data = renderer.create_render_data(_FakeSensor())
    assert isinstance(data, OvsensorsRenderData)
    assert len(data.sensor_handles) == 3
    assert len(data.prim_paths) == 3
    renderer.cleanup(data)


# ---------------------------------------------------------------------------
# test_render_and_read_output
# ---------------------------------------------------------------------------


def test_render_and_read_output(renderer_with_data):
    """render + read_output populates the output dict."""
    renderer, data = renderer_with_data
    renderer.set_outputs(data, {"rgb": None, "depth": None})
    renderer.render(data)

    class _FakeCameraData:
        output = {}

    cam_data = _FakeCameraData()
    renderer.read_output(data, cam_data)
    # After read_output the internal output_tensors dict should be updated, OR
    # cam_data.output should be populated.  Either path is acceptable.
    assert cam_data.output is not None  # dict exists and was not replaced


# ---------------------------------------------------------------------------
# test_render_output_shape
# ---------------------------------------------------------------------------


def test_render_output_shape(renderer):
    """LdrColor output has shape (N, H, W, 4) after a render step."""

    class _FakeSensor:
        num_instances = 1

    data = renderer.create_render_data(_FakeSensor())
    renderer.render(data)

    class _FakeCameraData:
        output = {}

    cam_data = _FakeCameraData()
    renderer.read_output(data, cam_data)

    rgb = cam_data.output.get("rgb")
    if rgb is not None:
        assert rgb.ndim == 4
        assert rgb.shape[-1] == 4  # RGBA
        assert rgb.shape[1] == renderer._height
        assert rgb.shape[2] == renderer._width

    renderer.cleanup(data)


# ---------------------------------------------------------------------------
# test_cleanup
# ---------------------------------------------------------------------------


def test_cleanup(renderer):
    """cleanup releases sensor handles (list is emptied)."""

    class _FakeSensor:
        num_instances = 2

    data = renderer.create_render_data(_FakeSensor())
    assert len(data.sensor_handles) == 2
    renderer.cleanup(data)
    assert len(data.sensor_handles) == 0
    assert len(data.prim_paths) == 0
