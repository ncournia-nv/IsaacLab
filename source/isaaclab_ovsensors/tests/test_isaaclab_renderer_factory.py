# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test: verify that the Renderer factory resolves 'ovsensors' to OvsensorsRenderer.

This test exercises the FactoryBase dynamic-import path end-to-end:
  Renderer(cfg)  ->  import isaaclab_ovsensors.renderers  ->  OvsensorsRenderer
"""

import os
import sys

import pytest

# Ensure isaaclab_ovsensors is on sys.path before the factory tries to import it.
_pkg_root = os.path.join(os.path.dirname(__file__), "..")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# isaaclab must also be importable.
_isaaclab_src = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "isaaclab"
)
if os.path.isdir(_isaaclab_src) and _isaaclab_src not in sys.path:
    sys.path.insert(0, _isaaclab_src)


def test_factory_resolves_ovsensors():
    """Renderer(OvsensorsRendererCfg()) returns an OvsensorsRenderer instance."""
    try:
        from isaaclab.renderers.renderer import Renderer
    except ImportError as exc:
        pytest.skip(f"isaaclab not importable in this environment: {exc}")

    from isaaclab_ovsensors.renderers import OvsensorsRenderer, OvsensorsRendererCfg

    cfg = OvsensorsRendererCfg(backend="mock")
    r = Renderer(cfg)
    assert isinstance(r, OvsensorsRenderer), (
        f"Expected OvsensorsRenderer, got {type(r).__name__}"
    )
    r.destroy()
