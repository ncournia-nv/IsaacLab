# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer interface for SimulationContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab.renderer import OVRTXRenderer, OVRTXRendererCfg, RendererBase

from .interface import Interface

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class RendererInterface(Interface):
    """Manages rendering backends for SimulationContext.

    Renderers in this interface are for simulation-level settings (like RTX config).
    Asset-managed renderers (like NewtonWarpRenderer for cameras) are managed by
    their respective assets, not through this interface.
    """

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize renderer interface with default RTX renderer.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)
        # Initialize RTX renderer for settings
        rtx_cfg = OVRTXRendererCfg()
        rtx_renderer = OVRTXRenderer(rtx_cfg, sim_context)
        self._renderers: list[RendererBase] = [rtx_renderer]

    @property
    def renderers(self) -> list[RendererBase]:
        """List of active renderers."""
        return self._renderers

    def reset(self, soft: bool = False) -> None:
        """Reset all renderers (no-op for settings renderers)."""
        for renderer in self._renderers:
            renderer.reset()

    def forward(self) -> None:
        """Update all renderers (no-op for settings renderers)."""
        pass

    def step(self, render: bool = True) -> None:
        """Step all renderers (no-op for settings renderers)."""
        for renderer in self._renderers:
            renderer.step()

    def close(self) -> None:
        """Clean up all renderers."""
        for renderer in self._renderers:
            renderer.close()
        self._renderers.clear()

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step."""
        return self._sim.get_rendering_dt()
