# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer interface for SimulationContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab.sim._impl.renderer import Renderer
from isaaclab.sim._impl.rtx_renderer import RTXRenderer

from .interface import Interface

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class RendererInterface(Interface):
    """Manages rendering backends for SimulationContext."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize renderer interface with default RTX renderer.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)
        self._renderers: list[Renderer] = [RTXRenderer(sim_context)]

    @property
    def renderers(self) -> list[Renderer]:
        """List of active renderers."""
        return self._renderers

    def reset(self, soft: bool = False) -> None:
        """Reset all renderers."""
        for renderer in self._renderers:
            renderer.reset(soft)

    def forward(self) -> None:
        """Update all renderers."""
        for renderer in self._renderers:
            renderer.forward()

    def step(self, render: bool = True) -> None:
        """Step all renderers.

        Args:
            render: Whether to render.
        """
        for renderer in self._renderers:
            renderer.step(render)

    def close(self) -> None:
        """Clean up all renderers."""
        for renderer in self._renderers:
            renderer.close()
        self._renderers.clear()

    def get_rendering_dt(self) -> float:
        """Returns the rendering time step."""
        return self._sim.get_rendering_dt()
