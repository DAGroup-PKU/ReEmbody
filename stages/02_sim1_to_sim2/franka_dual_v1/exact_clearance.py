"""Exact robot clearances, independent of MuJoCo's narrowphase (2026-11-05: MuJoCo's contact distance between the FR3
collision meshes and the arch capsules read 10.46 cm at a pose whose true distance is 8.89 cm, and planning pushed the
arms into such poses).

Every robot collision geom (left_* / right_* bodies) is sampled densely on its surface once, in its own frame:
meshes per triangle at SPACING (vertices, plus interior points of every triangle), boxes and cylinders on their faces.
Per pose the samples are put in the world and
  * arch_distance: the smallest distance from any sample to the arch's capsule segments (point-segment distance minus
    the tube radius, exact for the capsules); its closest sample and direction for a gradient;
  * arm_gap: the smallest distance between a left and a right sample (KD tree), the two links 0 on the shared mount
    left out (as duo_robot.DualFR3.arm_clearance).
With 4 mm spacing the samples are within 2.8 mm of every surface point, so a distance between flat faces is
over-read by at most ~0.05 mm at 9 cm, and by at most 2.8 mm only where two corners meet.

    t1_retarget/bin/python -m franka_dual_v1.exact_clearance <run dir> [...]   -> per run: min arch distance and arm gap

(copied from result/franka_fr3_duo_site_v9_20261105_attempt_01/code/exact_clearance.py, 2026-11-05)
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

SPACING = 0.008  # validated: 8, 4 and 3 mm give the same distances to 0.01 mm on the v8 cups run
THIN = 0.003
AXIS_STEP = 0.003  # arch_distance: axis points per segment; the nearest-sample error is < AXIS_STEP^2 / (8 d), < 0.03 mm at 4 cm


def _tri_samples(v, f, spacing):
    out = [v]
    T = v[f]
    edge = np.max([np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 0], axis=1),
                   np.linalg.norm(T[:, 2] - T[:, 1], axis=1)], axis=0)
    for a, b, c in T[edge > spacing]:  # triangles smaller than the spacing are covered by their vertices
        n = int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing))
        if n < 2:
            continue
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        keep = (i + j) <= n
        u, w = i[keep] / n, j[keep] / n
        out.append(a + u[:, None] * (b - a) + w[:, None] * (c - a))
    return np.unique(np.round(np.vstack(out), 5), axis=0)


def _box_samples(h, spacing):
    pts = []
    for ax in range(3):
        o = [k for k in range(3) if k != ax]
        g1 = np.linspace(-h[o[0]], h[o[0]], max(2, int(np.ceil(2 * h[o[0]] / spacing)) + 1))
        g2 = np.linspace(-h[o[1]], h[o[1]], max(2, int(np.ceil(2 * h[o[1]] / spacing)) + 1))
        A, B = np.meshgrid(g1, g2)
        for s in (-1, 1):
            p = np.zeros((A.size, 3))
            p[:, o[0]], p[:, o[1]], p[:, ax] = A.ravel(), B.ravel(), s * h[ax]
            pts.append(p)
    return np.vstack(pts)


def _cyl_samples(r, hl, spacing):
    na = max(12, int(np.ceil(2 * np.pi * r / spacing)))
    nz = max(2, int(np.ceil(2 * hl / spacing)) + 1)
    t = np.linspace(0, 2 * np.pi, na, endpoint=False)
    z = np.linspace(-hl, hl, nz)
    T, Z = np.meshgrid(t, z)
    side = np.c_[r * np.cos(T.ravel()), r * np.sin(T.ravel()), Z.ravel()]
    caps = []
    for rr in np.linspace(0, r, max(2, int(np.ceil(r / spacing)) + 1)):
        k = max(1, int(np.ceil(2 * np.pi * rr / spacing)))
        tt = np.linspace(0, 2 * np.pi, k, endpoint=False)
        for zz in (-hl, hl):
            caps.append(np.c_[rr * np.cos(tt), rr * np.sin(tt), np.full(k, zz)])
    return np.vstack([side] + caps)


class Exact:
    def __init__(self, m, spacing=SPACING):
        self.m = m
        body = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self.gname = [m.geom(g).name or body[g] for g in range(m.ngeom)]
        coll = (m.geom_contype | m.geom_conaffinity) > 0
        self.geoms, self.local, self.side = [], [], []
        fixed = m.body_weldid[m.geom_bodyid] == 0
        self.fixed = []
        for g in range(m.ngeom):
            if not coll[g] or not body[g].startswith(('left_', 'right_')):
                continue
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                p = _tri_samples(v, f, spacing)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                p = _box_samples(m.geom_size[g], spacing)
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                p = _cyl_samples(m.geom_size[g][0], m.geom_size[g][1], spacing)
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, hl = m.geom_size[g][0], m.geom_size[g][1]
                p = np.vstack([_cyl_samples(r, hl, spacing), _cyl_samples(r, 0.0, spacing) * [1, 1, 0] + [0, 0, hl],
                               _cyl_samples(r, 0.0, spacing) * [1, 1, 0] - [0, 0, hl]])
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = m.geom_size[g][0]
                k = max(50, int(4 * np.pi * r * r / spacing ** 2))
                u = np.random.default_rng(0).normal(size=(k, 3))
                p = r * u / np.linalg.norm(u, axis=1, keepdims=True)
            else:
                raise ValueError(f'unsampled robot geom type {t} ({self.gname[g]})')
            # one sample per THIN-sized cell (the dense gripper meshes have far more vertices than that; the arm links,
            # sampled at `spacing`, are left as they are): thinned to 8 mm cells the arm gap near the 2F-85 coupler was
            # over-read by 2.6 mm, to 3 mm by < 0.3 mm
            _, keep = np.unique(np.floor(p / THIN).astype(np.int64), axis=0, return_index=True)
            p = p[np.sort(keep)]
            self.geoms.append(g)
            self.local.append(p)
            self.side.append(0 if body[g].startswith('left_') else 1)
            self.fixed.append(bool(fixed[g]))
        self.side = np.array(self.side)
        self.fixed = np.array(self.fixed)
        # bounding spheres and KD trees in each geom's own frame (built once): pairs whose spheres are farther apart than
        # the best distance so far are skipped
        self.c_local = [P.mean(0) for P in self.local]
        self.r = np.array([np.linalg.norm(P - c, axis=1).max() for P, c in zip(self.local, self.c_local)])
        self.tree = [cKDTree(P) for P in self.local]
        arch = [g for g in range(m.ngeom) if body[g] == 'site_support_arch']
        self.arch = arch
        self.counts = {self.gname[g]: len(p) for g, p in zip(self.geoms, self.local)}

    def world(self, d, idx=None):
        idx = range(len(self.geoms)) if idx is None else idx
        return [(i, self.local[i] @ d.geom_xmat[self.geoms[i]].reshape(3, 3).T + d.geom_xpos[self.geoms[i]]) for i in idx]

    def centers(self, d):
        return np.array([d.geom_xmat[g].reshape(3, 3) @ c + d.geom_xpos[g] for g, c in zip(self.geoms, self.c_local)])

    def arch_segments(self, d):
        segs = []
        for c in self.arch:
            p, ax = d.geom_xpos[c], d.geom_xmat[c].reshape(3, 3)[:, 2]
            segs.append((p, ax, self.m.geom_size[c][1], self.m.geom_size[c][0], c))
        return segs

    def arch_distance(self, d, side=None, cap=np.inf):
        """(distance, robot geom name, arch geom name, closest robot point, unit direction from the arch to it, robot geom
        id); np.inf and Nones when nothing is within `cap`. Vectorised (2026-11-06): the bounds of every geom against
        every arch segment at once, then the exact distance only for geoms whose bound is under the best so far."""
        none = (np.inf, None, None, None, None, None)
        if not self.arch:
            return none
        if not hasattr(self, '_arch_idx'):
            self._arch_idx = np.array(self.arch)
            self._arch_hl = self.m.geom_size[self._arch_idx, 1].copy()
            self._arch_r = self.m.geom_size[self._arch_idx, 0].copy()
            self._geom_idx = np.array(self.geoms)
            self._c_local = np.array(self.c_local)
        ai = self._arch_idx
        P0 = d.geom_xpos[ai]                                   # segment centres (n_s, 3)
        AX = d.geom_xmat[ai].reshape(-1, 3, 3)[:, :, 2]        # segment axes
        hl, r = self._arch_hl, self._arch_r
        sel = np.arange(len(self.geoms)) if side is None else np.flatnonzero(self.side == side)
        gi = self._geom_idx[sel]
        Rg = d.geom_xmat[gi].reshape(-1, 3, 3)
        C = np.einsum('gij,gj->gi', Rg, self._c_local[sel]) + d.geom_xpos[gi]          # geom centres (n_g, 3)
        V = C[:, None, :] - P0[None, :, :]                                            # (n_g, n_s, 3)
        t = np.clip(np.einsum('gsk,sk->gs', V, AX), -hl, hl)
        LB = np.linalg.norm(V - t[:, :, None] * AX[None], axis=2) - r[None] - self.r[sel][:, None]
        lbg = LB.min(1)
        best = (cap, None, None, None, None, None)
        for k in np.argsort(lbg):
            if lbg[k] >= best[0]:
                break
            i = sel[k]
            Rk, pk = Rg[k], d.geom_xpos[gi[k]]
            for sgi in np.flatnonzero(LB[k] < best[0]):
                if LB[k, sgi] >= best[0]:
                    continue
                # the segment in the geom's frame, sampled every AXIS_STEP; nearest surface sample of each axis point
                pl = Rk.T @ (P0[sgi] - pk)
                al = Rk.T @ AX[sgi]
                n = max(2, int(np.ceil(2 * hl[sgi] / AXIS_STEP)) + 1)
                Q = pl[None] + np.linspace(-hl[sgi], hl[sgi], n)[:, None] * al[None]
                dq, jq = self.tree[i].query(Q)
                q = int(np.argmin(dq))
                if dq[q] - r[sgi] < best[0]:
                    Pw = Rk @ self.local[i][jq[q]] + pk                               # that surface sample, world
                    tt = float(np.clip((Pw - P0[sgi]) @ AX[sgi], -hl[sgi], hl[sgi]))
                    vec = Pw - (P0[sgi] + tt * AX[sgi])
                    dist = float(np.linalg.norm(vec)) - r[sgi]
                    if dist < best[0]:
                        best = (dist, self.gname[self.geoms[i]], self.m.geom(int(ai[sgi])).name, Pw,
                                vec / np.linalg.norm(vec), self.geoms[i])
        return none if best[1] is None else best

    def arm_gap(self, d, cap=np.inf):
        """(distance, left geom, right geom) between the arms, the two fixed links 0 left out; np.inf beyond `cap`."""
        C = self.centers(d)
        L = np.flatnonzero(self.side == 0)
        R = np.flatnonzero(self.side == 1)
        pairs = []
        for i in L:
            for j in R:
                if self.fixed[i] and self.fixed[j]:
                    continue
                pairs.append((float(np.linalg.norm(C[i] - C[j])) - self.r[i] - self.r[j], i, j))
        pairs.sort(key=lambda x: x[0])
        best = (cap, None, None)
        for lb, i, j in pairs:
            if lb >= best[0]:
                break
            gi, gj = self.geoms[i], self.geoms[j]
            Pw = self.local[i] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
            Pj = (Pw - d.geom_xpos[gj]) @ d.geom_xmat[gj].reshape(3, 3)  # into geom j's frame
            near = np.linalg.norm(Pj - self.c_local[j], axis=1) <= self.r[j] + best[0]
            if not near.any():
                continue
            dist, _ = self.tree[j].query(Pj[near], distance_upper_bound=best[0])
            k = float(dist.min())
            if k < best[0]:
                best = (k, self.gname[gi], self.gname[gj])
        if best[1] is None:
            return (np.inf, None, None)
        return best


def measure_run(run, step=1):
    run = Path(run)
    m = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    z = np.load(run / 'episode.npz')
    labels = [str(x) for x in z['labels']] + [str(z['labels'][-1])]
    E = Exact(m)
    d = mujoco.MjData(m)
    arch = {0: (np.inf,), 1: (np.inf,)}
    gap = (np.inf,)
    for k in range(0, len(z['qpos']), step):
        d.qpos[:] = z['qpos'][k]
        mujoco.mj_kinematics(m, d)
        for s in (0, 1):
            a = E.arch_distance(d, s)
            if a[0] < arch[s][0]:
                arch[s] = (a[0], round(k / 30, 2), labels[k], a[1], a[2])
        g = E.arm_gap(d)
        if g[0] < gap[0]:
            gap = (g[0], round(k / 30, 2), labels[k], g[1], g[2])
    fmt = lambda v: dict(min_cm=round(v[0] * 100, 2), t_s=v[1], stage=v[2], pair=f'{v[3]} - {v[4]}')  # noqa: E731
    return dict(arch_left=fmt(arch[0]), arch_right=fmt(arch[1]), arm_gap=fmt(gap), step=step)


if __name__ == '__main__':
    for r in sys.argv[1:]:
        print(r, json.dumps(measure_run(r)), flush=True)
