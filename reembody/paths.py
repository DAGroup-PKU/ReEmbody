"""Portable path and tool resolution for the ReEmbody data pipeline.

All machine-specific locations can be overridden via environment variables:
  - REEMBODY_THIRD_PARTY_DIR       (contains franka_description/ and mujoco_menagerie/)
  - REEMBODY_FRANKA_DESCRIPTION_DIR
  - REEMBODY_MUJOCO_MENAGERIE_DIR
  - REEMBODY_S1_MJCF_DIR           (optional Astribot S1 MJCF assets)
  - REEMBODY_S1_TASK_ASSETS_DIR    (defaults to <repo>/assets/s1_task_objects)
  - REEMBODY_FR3_SITE_ASSETS_DIR   (defaults to <repo>/assets/fr3_duo_site)
  - REEMBODY_SOURCE_CASES_DIR      (defaults to <repo>/examples)
  - REEMBODY_FFMPEG                (defaults to PATH lookup for `ffmpeg`)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES_DIR = REPO_ROOT / "stages"
ASSETS_DIR = REPO_ROOT / "assets"
EXAMPLES_DIR = REPO_ROOT / "examples"

THIRD_PARTY_DIR = Path(
    os.environ.get("REEMBODY_THIRD_PARTY_DIR", str(REPO_ROOT / "third_party"))
)
FRANKA_DESCRIPTION_DIR = Path(
    os.environ.get(
        "REEMBODY_FRANKA_DESCRIPTION_DIR",
        str(THIRD_PARTY_DIR / "franka_description"),
    )
)
MUJOCO_MENAGERIE_DIR = Path(
    os.environ.get(
        "REEMBODY_MUJOCO_MENAGERIE_DIR",
        str(THIRD_PARTY_DIR / "mujoco_menagerie"),
    )
)
S1_TASK_ASSETS_DIR = Path(
    os.environ.get(
        "REEMBODY_S1_TASK_ASSETS_DIR",
        str(ASSETS_DIR / "s1_task_objects"),
    )
)
FR3_SITE_ASSETS_DIR = Path(
    os.environ.get(
        "REEMBODY_FR3_SITE_ASSETS_DIR",
        str(ASSETS_DIR / "fr3_duo_site"),
    )
)


def resolve_franka_description() -> Path:
    """Return the path to the pinned `franka_description` repository."""
    if "REEMBODY_FRANKA_DESCRIPTION_DIR" in os.environ:
        return Path(os.environ["REEMBODY_FRANKA_DESCRIPTION_DIR"])
    base = Path(os.environ.get("REEMBODY_THIRD_PARTY_DIR", str(REPO_ROOT / "third_party")))
    return base / "franka_description"


def resolve_mujoco_menagerie() -> Path:
    """Return the path to the `mujoco_menagerie` repository."""
    if "REEMBODY_MUJOCO_MENAGERIE_DIR" in os.environ:
        return Path(os.environ["REEMBODY_MUJOCO_MENAGERIE_DIR"])
    base = Path(os.environ.get("REEMBODY_THIRD_PARTY_DIR", str(REPO_ROOT / "third_party")))
    return base / "mujoco_menagerie"


def resolve_s1_mjcf() -> Path:
    """Return the path to optional Astribot S1 MJCF robot assets."""
    if "REEMBODY_S1_MJCF_DIR" in os.environ:
        return Path(os.environ["REEMBODY_S1_MJCF_DIR"])
    base = Path(os.environ.get("REEMBODY_THIRD_PARTY_DIR", str(REPO_ROOT / "third_party")))
    return base / "astribot_s1_mjcf"


def resolve_source_cases() -> Path:
    """Return the path containing Sim1 source case directories."""
    if "REEMBODY_SOURCE_CASES_DIR" in os.environ:
        return Path(os.environ["REEMBODY_SOURCE_CASES_DIR"])
    return EXAMPLES_DIR


def resolve_s1_task_assets() -> Path:
    """Return the directory containing measured task object XMLs/definitions."""
    return Path(
        os.environ.get(
            "REEMBODY_S1_TASK_ASSETS_DIR",
            str(ASSETS_DIR / "s1_task_objects"),
        )
    )


def resolve_fr3_site_assets() -> Path:
    """Return the directory containing FR3 Duo Blender site assets."""
    return Path(
        os.environ.get(
            "REEMBODY_FR3_SITE_ASSETS_DIR",
            str(ASSETS_DIR / "fr3_duo_site"),
        )
    )


def ffmpeg_bin() -> str:
    """Locate the `ffmpeg` executable."""
    explicit = os.environ.get("REEMBODY_FFMPEG") or os.environ.get("RSSR_FFMPEG")
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / "ffmpeg"
    if candidate.is_file():
        return str(candidate)
    return "ffmpeg"


def ensure_stage_paths(*stage_subpaths: str) -> None:
    """Insert repository and stage subdirectories onto `sys.path` once."""
    paths = [REPO_ROOT] + [REPO_ROOT / sp for sp in stage_subpaths]
    for path in reversed(paths):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
