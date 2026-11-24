"""Export one dual-arm Franka episode for Blender: visual geometry once, world poses per rendered frame.

Rendered: Menagerie Panda/Hand visual meshes (geom group 2, with their material colours), task objects and the
worktable. Not rendered: collision geoms, sites, task regions (the real table has no marker), the MuJoCo floor
(the renderer supplies its own ground). Every `every`-th 30 Hz frame is exported (sim_dart renders every 3rd and
holds it in between). Cameras: head, left_wrist, right_wrist (MuJoCo camera poses and vertical FOV).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

CAMS = ('head', 'left_wrist', 'right_wrist')


def visual_geoms(m):
    keep = []
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        name = m.geom(g).name
        robot = body.startswith(('left_', 'right_'))
        if robot and m.geom_group[g] != 2:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or 'region' in name or body == 'world':
            continue
        if not robot and m.geom_rgba[g, 3] < 0.5 and m.geom_matid[g] < 0:
            continue
        keep.append(g)
    return keep


def geom_record(m, g):
    t = int(m.geom_type[g])
    rgba = (m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]).astype(float).tolist()
    rec = dict(id=g, name=m.geom(g).name or f'geom_{g}', body=m.body(m.geom_bodyid[g]).name, rgba=rgba,
               size=m.geom_size[g].astype(float).tolist())
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        rec.update(type='mesh', verts=m.mesh_vert[va:va + vn].astype(float).round(6).tolist(),
                   faces=m.mesh_face[fa:fa + fn].astype(int).tolist())
    else:
        rec['type'] = {mujoco.mjtGeom.mjGEOM_BOX: 'box', mujoco.mjtGeom.mjGEOM_CYLINDER: 'cylinder',
                       mujoco.mjtGeom.mjGEOM_SPHERE: 'sphere', mujoco.mjtGeom.mjGEOM_CAPSULE: 'capsule'}[t]
    return rec


def export(scene_xml: Path, qpos: np.ndarray, out: Path, every: int = 3, cams=CAMS):
    out.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    geoms = visual_geoms(m)
    (out / 'geometry.json').write_text(json.dumps([geom_record(m, g) for g in geoms]))
    frames = np.arange(0, len(qpos), every)
    pos = np.zeros((len(frames), len(geoms), 3))
    quat = np.zeros((len(frames), len(geoms), 4))
    cpos = np.zeros((len(frames), len(cams), 3))
    cquat = np.zeros((len(frames), len(cams), 4))
    cids = [m.camera(c).id for c in cams]
    for i, k in enumerate(frames):
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        pos[i] = d.geom_xpos[geoms]
        for j, g in enumerate(geoms):
            mujoco.mju_mat2Quat(quat[i, j], d.geom_xmat[g])
        cpos[i] = d.cam_xpos[cids]
        for j, c in enumerate(cids):
            mujoco.mju_mat2Quat(cquat[i, j], d.cam_xmat[c])
    np.savez_compressed(out / 'poses.npz', frames=frames, geom_pos=pos, geom_quat=quat, cam_pos=cpos, cam_quat=cquat,
                        n_frames=len(qpos))
    (out / 'cameras.json').write_text(json.dumps(dict(names=list(cams), fovy_deg=[float(m.cam_fovy[c]) for c in cids],
                                                      every=every)))
    return dict(geoms=len(geoms), rendered_frames=len(frames))
