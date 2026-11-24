"""Wrist-camera visibility and head-camera arm occlusion from MuJoCo segmentation renders.

Same definitions as sim2-real2/s1_v8_visibility_v1/visibility_core_v1.py, at the dual-arm Franka
camera mounts (layout.py, D435 320x240):
  * wrist visibility of object o in hand h's wrist camera = number of pixels whose front-most geom
    belongs to o (robot visual geoms, table and all task objects are occluders);
  * head occlusion of object o = |pixels of o's silhouette rendered alone that show a robot geom in
    the full render| / |silhouette|. Robot collision geoms, the floor, task regions and sites are not
    rendered.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402

from .layout import D435_H, D435_W, HANDS  # noqa: E402
from .robot import DualFranka  # noqa: E402

G_SCENE, G_TARGET, G_HIDDEN = 0, 2, 5


class VisionProbe:
    def __init__(self, m: mujoco.MjModel, width=D435_W, height=D435_H, robot=None):
        self.m = m
        self.robot = DualFranka(m) if robot is None else robot
        body = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        name = [m.geom(g).name for g in range(m.ngeom)]
        self.geom_body = np.array(m.geom_bodyid)
        group = np.full(m.ngeom, G_HIDDEN, np.int32)
        self.is_robot = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            robot = body[g].startswith(tuple(f'{h}_' for h in HANDS))
            if robot:
                if m.geom_group[g] == 2:  # Menagerie visual meshes
                    group[g] = G_SCENE
                    self.is_robot[g] = True
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or 'region' in name[g] or body[g] == 'world':
                continue
            group[g] = G_SCENE
        self.group = group
        m.geom_group[:] = group
        self.renderer = mujoco.Renderer(m, height, width)
        self.renderer.enable_segmentation_rendering()
        self.opt_full = self._opt((G_SCENE, G_TARGET))
        self.opt_target = self._opt((G_TARGET,))
        self.cams = {'head': m.camera('head').id, **{f'{h}_wrist': m.camera(f'{h}_wrist').id for h in HANDS}}

    @staticmethod
    def _opt(groups):
        o = mujoco.MjvOption()
        o.geomgroup[:] = 0
        for g in groups:
            o.geomgroup[g] = 1
        o.sitegroup[:] = 0
        return o

    def _seg(self, d, cam, opt):
        self.renderer.update_scene(d, camera=self.cams[cam], scene_option=opt)
        s = self.renderer.render()
        return np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)

    def _body_geoms(self, bodies):
        """Geoms of the named bodies (with descendants); names starting with 'geom:' select one geom."""
        sel = np.zeros(self.m.ngeom, bool)
        for b in bodies:
            if b.startswith('geom:'):
                sel[self.m.geom(b[5:]).id] = True
        bodies = [b for b in bodies if not b.startswith('geom:')]
        ids = [self.m.body(b).id for b in bodies]
        # include descendants (e.g. a drawer box and its handle)
        for g in range(self.m.ngeom):
            b = self.geom_body[g]
            while b > 0:
                if b in ids:
                    sel[g] = True
                    break
                b = self.m.body_parentid[b]
        return sel & (self.group != G_HIDDEN)

    def visible_pixels(self, d, cam: str, bodies) -> int:
        full = self._seg(d, cam, self.opt_full)
        sel = self._body_geoms(bodies)
        return int(sel[full[full >= 0]].sum())

    def occlusion(self, d, cam: str, bodies) -> dict:
        sel = self._body_geoms(bodies)
        full = self._seg(d, cam, self.opt_full)
        self.m.geom_group[:] = np.where(sel, G_TARGET, G_HIDDEN)
        try:
            alone = self._seg(d, cam, self.opt_target) >= 0
        finally:
            self.m.geom_group[:] = self.group
        sil = int(alone.sum())
        shown = full[alone]
        visible = int(sel[shown[shown >= 0]].sum())
        by_robot = int(self.is_robot[shown[shown >= 0]].sum())
        return dict(silhouette_px=sil, visible_px=visible, robot_occluded_px=by_robot,
                    robot_occluded_fraction=(by_robot / sil) if sil else None)
