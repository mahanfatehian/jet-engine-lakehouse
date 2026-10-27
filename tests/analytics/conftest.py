"""Makes the harness modules of tests/analytics importable by their unit tests.

pytest runs with --import-mode=importlib (pyproject.toml), which puts no test directory on
sys.path, while the harness modules import each other by bare name, as they do when the
integration scripts run them. Appended, so they never shadow a component module.
"""

from __future__ import annotations

import sys
from pathlib import Path

HARNESS_DIR = str(Path(__file__).resolve().parent)
if HARNESS_DIR not in sys.path:
    sys.path.append(HARNESS_DIR)
