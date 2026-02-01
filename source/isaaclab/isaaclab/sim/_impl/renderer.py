# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base renderer class for RendererInterface backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class Renderer(ABC):
    """Abstract base class for renderer backends."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize renderer.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context

    @abstractmethod
    def reset(self, soft: bool = False) -> None:
        """Reset renderer state."""
        pass

    @abstractmethod
    def forward(self) -> None:
        """Update renderer state."""
        pass

    @abstractmethod
    def step(self, render: bool = True) -> None:
        """Step renderer."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up renderer resources."""
        pass
