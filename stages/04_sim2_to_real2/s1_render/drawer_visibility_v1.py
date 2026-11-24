"""Per-frame head/wrist visibility of drawer task targets (drawer-task copy of
sim2-real2/s1_v8_visibility_v1/visibility_core_v1.py, which is left unchanged).

Targets: the active handle bar, the active drawer interior floor (the "drawer
opening"), and the payload block.  MuJoCo segmentation at the native camera
mounts/fovy at 224x126 (the 16:9 raster the Blender renderer draws before the
black padding).  Head/torso/chassis/wheel geoms are hidden from the head camera,
as in the Blender renderers.  A second, wide-FOV silhouette render gives the
fraction of each target's projection that falls inside the real image.
Read-only with respect to all inputs.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402

CAMS = ('head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd')
W, H = 224, 126
WIDE = 3  # wide render covers 3x the tangent half-angles
TARGETS = ('handle', 'opening', 'block')
TARGET_GEOMS = {'handle': ('bottom_handle_bar_geom',), 'opening': ('bottom_drawer_bottom',),
                'block': ('red_block_geom',)}
HEAD_OCCLUDER = ('astribot_head_', 'astribot_torso_', 'chassis_base', 'wheel_')
CATS = ('target', 'other_target', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper',
        'cabinet', 'table', 'body', 'background')
G_ROBOT, G_HEADOCC, G_HIDDEN = 0, 1, 5
G_TARGET = {'handle': 2, 'opening': 3, 'block': 4}


class Scene:
    def __init__(self, scene_path, cams=CAMS, camera_override=None):
        m = mujoco.MjModel.from_xml_path(str(scene_path))
        if camera_override:
            for name, spec in camera_override.items():
                c = m.camera(name).id
                m.cam_pos[c] = spec['pos']
                q = np.asarray(spec['quat'], float)
                m.cam_quat[c] = q / np.linalg.norm(q)
                m.cam_fovy[c] = spec['fovy']
        self.m, self.d = m, mujoco.MjData(m)
        body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or '' for g in range(m.ngeom)]
        name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in range(m.ngeom)]
        self.cat_of_geom = np.full(m.ngeom, CATS.index('background'), np.int16)
        self.target_of_geom = np.full(m.ngeom, -1, np.int16)
        group = np.full(m.ngeom, G_HIDDEN, np.int32)
        cabinet_bodies = {'drawer_cabinet', 'bottom_drawer', 'bottom_handle_bar', 'bottom_handle_post_a',
                          'bottom_handle_post_b', 'middle_drawer_front', 'middle_handle', 'top_drawer_front',
                          'top_handle', 'lowest_drawer_front', 'lowest_handle'}
        for g in range(m.ngeom):
            b, n = body[g], name[g]
            visual = m.geom_group[g] in (0, 1, 2) and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            if not visual or 'region' in n or n.startswith('rssr_room') or b == 'world':
                continue
            for k, t in enumerate(TARGETS):
                if n in TARGET_GEOMS[t]:
                    self.target_of_geom[g] = k
                    group[g] = G_TARGET[t]
            if self.target_of_geom[g] >= 0:
                continue
            if b.startswith('astribot_arm_left'):
                cat = 'left_arm'
            elif b.startswith('astribot_gripper_left'):
                cat = 'left_gripper'
            elif b.startswith('astribot_arm_right'):
                cat = 'right_arm'
            elif b.startswith('astribot_gripper_right'):
                cat = 'right_gripper'
            elif b in cabinet_bodies:
                cat = 'cabinet'
            elif b.startswith('worktable'):
                cat = 'table'
            else:
                cat = 'body'
            self.cat_of_geom[g] = CATS.index(cat)
            group[g] = G_HEADOCC if any(s in b for s in HEAD_OCCLUDER) else G_ROBOT
        for t in TARGETS:
            if not np.any(self.target_of_geom == TARGETS.index(t)):
                raise ValueError(f'target geom missing: {t}')
        self.group = group
        m.geom_group[:] = group
        m.geom_rgba[:, 3] = np.where(group != G_HIDDEN, 1.0, m.geom_rgba[:, 3])
        self.cam_names = cams
        self.cam_ids = [m.camera(c).id for c in cams]
        self.fovy = np.array([m.cam_fovy[c] for c in self.cam_ids])
        self.pads = {h: [m.geom(f'astribot_gripper_{h}_geom_{p}').id for p in ('L112', 'R112')]
                     for h in ('left', 'right')}
        self.slide = m.jnt_qposadr[m.joint('bottom_drawer_slide').id]
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W * WIDE)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H * WIDE)
        self.renderer = mujoco.Renderer(m, H, W)
        self.renderer.enable_segmentation_rendering()
        self.wide = mujoco.Renderer(m, H * WIDE, W * WIDE)
        self.wide.enable_segmentation_rendering()
        self.opts = {}
        for key, groups in (('full_head', (G_ROBOT, 2, 3, 4)), ('full_wrist', (G_ROBOT, G_HEADOCC, 2, 3, 4))):
            self.opts[key] = self._opt(groups)
        for t in TARGETS:
            self.opts[t] = self._opt((G_TARGET[t],))

    @staticmethod
    def _opt(groups):
        o = mujoco.MjvOption()
        o.geomgroup[:] = 0
        for g in groups:
            o.geomgroup[g] = 1
        o.sitegroup[:] = 0
        o.tendongroup[:] = 0
        return o

    def set_state(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)

    def _seg(self, renderer, cam, key):
        renderer.update_scene(self.d, camera=self.cam_ids[cam], scene_option=self.opts[key])
        s = renderer.render()
        return np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)

    def frame(self, cam):
        """Full geom-id image, per-target in-image silhouettes and in-frame fractions."""
        full = self._seg(self.renderer, cam, 'full_head' if self.cam_names[cam] == 'head_rgbd' else 'full_wrist')
        sils, inframe = [], []
        c = self.cam_ids[cam]
        base = float(self.m.cam_fovy[c])
        for t in TARGETS:
            sils.append(self._seg(self.renderer, cam, t) >= 0)
            self.m.cam_fovy[c] = np.degrees(2 * np.arctan(WIDE * np.tan(np.radians(base) / 2)))
            wide = self._seg(self.wide, cam, t) >= 0
            self.m.cam_fovy[c] = base
            total = wide.sum()
            inner = wide[H:2 * H, W:2 * W].sum()
            inframe.append(inner / total if total else np.nan)
        return full, sils, np.array(inframe)

    def close(self):
        self.renderer.close()
        self.wide.close()


def measure(scene_path, qpos, frames=None, cams=('head_rgbd',), camera_override=None):
    sc = Scene(scene_path, cams, camera_override)
    frames = np.arange(len(qpos)) if frames is None else np.asarray(frames)
    T, C, K = len(frames), len(cams), len(TARGETS)
    sil = np.zeros((T, C, K), np.int32)
    vis = np.zeros((T, C, K), np.int32)
    inframe = np.full((T, C, K), np.nan)
    occ = np.zeros((T, C, K, len(CATS)), np.int32)
    proj = np.full((T, C, K, 3), np.nan)
    slide = np.zeros(T)
    pads = np.zeros((T, 2, 3))
    target_pos = np.zeros((T, K, 3))
    cam_pose = np.zeros((T, C, 12))
    gid = {t: [sc.m.geom(n).id for n in TARGET_GEOMS[t]] for t in TARGETS}
    for i, f in enumerate(frames):
        sc.set_state(qpos[f])
        slide[i] = sc.d.qpos[sc.slide]
        pads[i] = [sc.d.geom_xpos[sc.pads[h]].mean(0) for h in ('left', 'right')]
        for k, t in enumerate(TARGETS):
            target_pos[i, k] = sc.d.geom_xpos[gid[t]].mean(0)
        for ci in range(C):
            c = sc.cam_ids[ci]
            R = sc.d.cam_xmat[c].reshape(3, 3)
            cam_pose[i, ci] = np.r_[sc.d.cam_xpos[c], R.ravel()]
            fpx = 0.5 * H / np.tan(np.radians(sc.m.cam_fovy[c]) / 2)
            for k in range(K):
                pc = R.T @ (target_pos[i, k] - sc.d.cam_xpos[c])
                z = -pc[2]
                if z > 0:
                    proj[i, ci, k] = (W / 2 + fpx * pc[0] / z, H / 2 - fpx * pc[1] / z, z)
            full, sils, fr = sc.frame(ci)
            inframe[i, ci] = fr
            tgt = np.where(full >= 0, sc.target_of_geom[np.clip(full, 0, None)], -1)
            cat = np.where(full >= 0, sc.cat_of_geom[np.clip(full, 0, None)], CATS.index('background'))
            for k in range(K):
                vis[i, ci, k] = (tgt == k).sum()
                s = sils[k]
                sil[i, ci, k] = s.sum()
                if s.any():
                    cc = np.where(tgt[s] == k, 0, np.where(tgt[s] >= 0, 1, cat[s]))
                    occ[i, ci, k] = np.bincount(cc, minlength=len(CATS))
    sc.close()
    return dict(frames=frames, sil=sil, vis=vis, inframe=inframe, occ=occ, proj=proj, slide=slide,
                pads=pads, target_pos=target_pos, cam_pose=cam_pose, fovy=sc.fovy,
                cams=np.array(cams), targets=np.array(TARGETS), cats=np.array(CATS))
