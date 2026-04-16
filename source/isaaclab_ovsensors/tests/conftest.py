# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for isaaclab_ovsensors tests.

Sets up sys.path and environment variables so that both ovsensors and
isaaclab_ovsensors are importable without prior installation.
"""

import os
import sys

# -- ovsensors Python package -------------------------------------------------
_ovsensors_python = "/home/horde/ovsensors/ovsensors/python"
if _ovsensors_python not in sys.path:
    sys.path.insert(0, _ovsensors_python)

os.environ.setdefault("OVSENSORS_LIB_PATH", "/home/horde/ovsensors/ovsensors/build/lib")

# -- isaaclab_ovsensors package -----------------------------------------------
_pkg_root = os.path.join(os.path.dirname(__file__), "..")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
