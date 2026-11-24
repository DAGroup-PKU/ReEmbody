"""Explicit filesystem dependencies, derived from this repository location.

RSSR_PROJECT_ROOT overrides the sibling project root. RSSR_SOURCE_CLOCK_ROOT
selects the sibling Blender source-clock package. RSSR_SHARED_RUNTIME_ROOT
selects the sibling rssr_augment dependency. RSSR_FCL_DEPS may point to an
existing python-fcl installation; this library never installs dependencies.
"""
from pathlib import Path
import os
SIM2_REFINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("RSSR_PROJECT_ROOT", SIM2_REFINE_ROOT.parent)).expanduser().resolve()
SOURCE_CLOCK_ROOT = Path(os.environ.get("RSSR_SOURCE_CLOCK_ROOT", PROJECT_ROOT / "sim2-real2/blender")).expanduser().resolve()
SHARED_RUNTIME_ROOT = Path(os.environ.get("RSSR_SHARED_RUNTIME_ROOT", PROJECT_ROOT / "sim1-sim2/s1_short_refine_v2")).expanduser().resolve()
STATEALIAS_REFERENCE_PACKAGE = Path(os.environ.get("RSSR_STATEALIAS_REFERENCE_PACKAGE", Path(__file__).resolve().parent / "statealias_reference")).expanduser().resolve()
FCL_DEPS = Path(os.environ.get("RSSR_FCL_DEPS", PROJECT_ROOT / "work/short_refine_v2/analysis/user_collision_review/deps")).expanduser().resolve()
