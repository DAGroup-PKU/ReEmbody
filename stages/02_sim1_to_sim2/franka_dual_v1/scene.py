"""Build the shared MuJoCo task environment XML from a reconstructed Sim1 source case."""

from __future__ import annotations

import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "rssr_augment/vendor/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))
import rssr_engineered_assets  # noqa: E402
import rssr_scene  # noqa: E402
from pilot_physics import _convert_exact_worktable_meshes_to_boxes  # noqa: E402

COBOT_BODIES = ("left_base", "right_base")
COBOT_CAMERA = "rssr_head"
PHYSICS_DT = 1.0 / 1200  # 40 substeps per 30 Hz frame


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engineered_asset_config(case_dir: Path) -> dict:
    asset = case_dir / "asset"
    return dict(
        engineered_task_asset=str(asset),
        engineered_task_asset_sha256=sha256(asset / "manifest.json"),
    )


def environment_xml(source_scene: Path, scene: dict) -> str:
    """Strip source cobot arms and rebuild task objects for target-embodiment attachment."""
    case_dir = source_scene.parent.parent
    root = ET.parse(source_scene).getroot()
    world = root.find("worldbody")
    drop = set(COBOT_BODIES) | {o["id"] for o in scene["objects"]}
    for body in list(world.findall("body")):
        if body.get("name") in drop:
            world.remove(body)
    for cam in list(world.findall("camera")):
        if cam.get("name") == COBOT_CAMERA:
            world.remove(cam)
    for act in list(root.findall("actuator")):
        root.remove(act)
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        if "piper" in (mesh.get("file") or ""):
            asset.remove(mesh)
    rssr_scene.append_objects(root, scene)
    rssr_engineered_assets.apply(root, engineered_asset_config(case_dir), "sim2", scene)
    _convert_exact_worktable_meshes_to_boxes(root)
    return ET.tostring(root, encoding="unicode")
