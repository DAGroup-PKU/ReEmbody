"""Build a dual-arm Franka MuJoCo scene from one RSSR cobot source case.

Table, floor, light and physics options (except the timestep) come from the case's Sim1 scene
(`source_reference/scene.xml`); its Piper arms, actuators, meshes, cobot head camera and object
bodies are removed. Task objects are built as the S1 Sim2 scenes build them (S1 runtime modules,
imported read-only): `rssr_scene.append_objects` from `source_scene.json`, then the case's
engineered task asset (`asset/`, `rssr_engineered_assets.apply`, kind sim2: cabinet open face,
articulation limit solver, tapered baskets, ...), then exact worktable meshes become boxes
(`pilot_physics._convert_exact_worktable_meshes_to_boxes`). Nine cup cases
(stack_cups_ep000041-49) have no objects in their source_reference XML. Two MuJoCo Menagerie Panda arms with the Franka Hand
are attached with the prefixes `left_` / `right_` at the RoboTwin layout (layout.py), and
three cameras are added: `head` (world), `left_wrist`, `right_wrist` (on each hand). A placed drawer case
(`<case>@<placement>`) builds the same scene with the cabinet and the block moved (placement.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .layout import HANDS, SELECTED, TCP_OFFSET, WRIST_CAM_POS, WRIST_CAM_XYAXES, Layout
from .placement import case_scene, receipt, source_case

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from reembody.paths import resolve_mujoco_menagerie, resolve_source_cases  # noqa: E402

RUNTIME = Path(__file__).resolve().parents[1] / 'rssr_augment/vendor/runtime'
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))
import rssr_engineered_assets  # noqa: E402
import rssr_scene  # noqa: E402
from pilot_physics import _convert_exact_worktable_meshes_to_boxes  # noqa: E402

MENAGERIE = resolve_mujoco_menagerie() / 'franka_emika_panda'
PANDA_XML = MENAGERIE / 'panda.xml'
CASES = resolve_source_cases()
COBOT_BODIES = ('left_base', 'right_base')
COBOT_CAMERA = 'rssr_head'
FINGER_OPEN = 0.04
PHYSICS_DT = 1.0 / 1200  # 40 steps per 30 Hz frame (the source scenes use 1 ms)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _panda_spec() -> mujoco.MjSpec:
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for mesh in panda.meshes:
        mesh.file = str((MENAGERIE / 'assets' / mesh.file).resolve(strict=True))
    panda.meshdir = ''
    for light in list(panda.lights):
        panda.delete(light)
    for key in list(panda.keys):
        panda.delete(key)
    for body in panda.bodies:
        body.gravcomp = 1.0  # Franka's controller compensates gravity
    return panda


def _xyaxes(cam, xyaxes):
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = list(xyaxes)
    return cam


def engineered_asset_config(case_dir: Path) -> dict:
    asset = case_dir / 'asset'
    return dict(engineered_task_asset=str(asset), engineered_task_asset_sha256=sha256(asset / 'manifest.json'))


def environment_xml(source_scene: Path, scene: dict) -> str:
    case_dir = source_scene.parent.parent
    root = ET.parse(source_scene).getroot()
    world = root.find('worldbody')
    drop = set(COBOT_BODIES) | {o['id'] for o in scene['objects']}
    for body in list(world.findall('body')):
        if body.get('name') in drop:
            world.remove(body)
    for cam in list(world.findall('camera')):
        if cam.get('name') == COBOT_CAMERA:
            world.remove(cam)
    for act in list(root.findall('actuator')):
        root.remove(act)
    asset = root.find('asset')
    for mesh in list(asset.findall('mesh')):
        if 'piper' in (mesh.get('file') or ''):
            asset.remove(mesh)
    rssr_scene.append_objects(root, scene)
    rssr_engineered_assets.apply(root, engineered_asset_config(case_dir), 'sim2', scene)
    _convert_exact_worktable_meshes_to_boxes(root)
    return ET.tostring(root, encoding='unicode')


def build_spec(source_scene: Path, layout: Layout, scene: dict | None = None) -> mujoco.MjSpec:
    source_scene = Path(source_scene)
    if scene is None:
        scene = json.loads((source_scene.parent.parent / 'source_scene.json').read_text())
    spec = mujoco.MjSpec.from_string(environment_xml(source_scene, scene))
    spec.modelname = 'rssr_franka_dual_v1'
    for hand in HANDS:
        frame = spec.worldbody.add_frame(pos=layout.base_pos(hand), quat=layout.base_quat())
        frame.attach_body(_panda_spec().body('link0'), f'{hand}_', '')
        hb = spec.body(f'{hand}_hand')
        hb.add_site(name=f'{hand}_tcp', pos=TCP_OFFSET, size=[0.004, 0, 0], rgba=[1, 0, 0, 0.0], group=4)
        _xyaxes(hb.add_camera(name=f'{hand}_wrist', pos=WRIST_CAM_POS, fovy=layout.wrist_fovy_deg),
                WRIST_CAM_XYAXES)
    pos, xyaxes = layout.head_pose()
    _xyaxes(spec.worldbody.add_camera(name='head', pos=pos, fovy=layout.head_fovy()), xyaxes)
    spec.option.timestep = PHYSICS_DT
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    return spec


def home_qpos(m: mujoco.MjModel, base_qpos: np.ndarray, layout: Layout = Layout()) -> np.ndarray:
    q = base_qpos.copy()
    for hand in HANDS:
        for i, v in enumerate(layout.arm_start_q(hand), 1):
            q[m.joint(f'{hand}_joint{i}').qposadr[0]] = v
        for f in (1, 2):
            q[m.joint(f'{hand}_finger_joint{f}').qposadr[0]] = FINGER_OPEN
    return q


def home_ctrl(m: mujoco.MjModel, layout: Layout = Layout()) -> np.ndarray:
    c = np.zeros(m.nu)
    for hand in HANDS:
        for i, v in enumerate(layout.arm_start_q(hand), 1):
            c[m.actuator(f'{hand}_actuator{i}').id] = v
        c[m.actuator(f'{hand}_actuator8').id] = 255.0
    return c


def build_case(case: str, out: Path, layout: Layout = Layout()) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    case_dir = CASES / source_case(case)
    source_scene = case_dir / 'source_reference/scene.xml'
    scene_json = case_scene(case)
    spec = build_spec(source_scene, layout, scene_json)
    m = spec.compile()
    d = mujoco.MjData(m)
    q0 = home_qpos(m, d.qpos, layout)
    spec.add_key(name='home', qpos=q0, ctrl=home_ctrl(m, layout))
    m = spec.compile()
    xml = spec.to_xml()
    (out / 'scene.xml').write_text(xml)
    check = mujoco.MjModel.from_xml_path(str(out / 'scene.xml'))
    manifest = dict(
        schema='rssr.franka_dual.scene.v1', case=case, layout=layout.to_json(),
        source_scene=str(source_scene), source_scene_sha256=sha256(source_scene),
        source_scene_json_sha256=sha256(case_dir / 'source_scene.json'),
        panda_xml=str(PANDA_XML), panda_xml_sha256=sha256(PANDA_XML),
        object_builder={f: sha256(RUNTIME / f) for f in ('rssr_scene.py', 'rssr_engineered_assets.py', 'pilot_physics.py')},
        engineered_asset=engineered_asset_config(case_dir),
        mujoco_version=mujoco.__version__, scene_xml_sha256=sha256(out / 'scene.xml'),
        nq=int(check.nq), nu=int(check.nu), bindings=scene_json['bindings'],
        objects=[dict(id=o['id'], shape=o['shape'], mobility=o['mobility']) for o in scene_json['objects']],
        drawer_placement=receipt(case))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    print(json.dumps(build_case(a.case, Path(a.out), SELECTED), indent=2))


if __name__ == '__main__':
    main()
