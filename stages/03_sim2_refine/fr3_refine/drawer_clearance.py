"""Clearance between a hand or arm and the cabinet and drawer in a drawer run (2026-11-07; user: "抽屉任务左臂和抽屉还是发生了碰撞").

The planner allows the block hand to touch the bottom drawer during the insertion (S.allow), so the run's own checks say
nothing about that pair. Here every robot collision geom's surface samples (exact_clearance.Exact) are measured per frame
against every collision geom of the cabinet bodies: analytic signed distance to boxes, surface samples for the rest.

    t1_retarget/bin/python drawer_clearance.py <run dir> [...]   -> per run: closest approach per side, stage and pair
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2'))
from franka_dual_v1.exact_clearance import Exact, _box_samples, _cyl_samples  # noqa: E402

CAB_WORDS = ('cabinet', 'drawer', 'handle')


def cabinet_geoms(m):
    out = []
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        if b.startswith(('left_', 'right_')) or not any(w in b for w in CAB_WORDS):
            continue
        if (m.geom_contype[g] | m.geom_conaffinity[g]) > 0:
            out.append(g)
    return out


def box_signed(P, c, R, h):
    q = (P - c) @ R
    d = np.abs(q) - h
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(d.max(axis=1), 0.0)
    return outside + inside


def measure(run, step=1):
    run = Path(run)
    m = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    d = mujoco.MjData(m)
    z = np.load(run / 'episode.npz', allow_pickle=True)
    rec = json.loads((run / 'record.json').read_text())
    stages = rec['measurements']['stages']
    E = Exact(m)
    cab = cabinet_geoms(m)
    names = {g: m.geom(g).name or m.body(m.geom_bodyid[g]).name for g in cab}
    bodies = {g: m.body(m.geom_bodyid[g]).name for g in cab}
    nonbox = {}
    for g in cab:
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER or t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            nonbox[g] = _cyl_samples(m.geom_size[g][0], m.geom_size[g][1], 0.003)
        elif t != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f'unexpected cabinet geom type {t}: {names[g]}')
    stage_at = lambda k: next((s['label'] for s in stages if s['t0'] * 30 - 1e-6 <= k < s['t1'] * 30 - 1e-6), stages[-1]['label'])
    best = {}  # (side, cabinet body) -> closest (dist, frame, robot geom, cabinet geom, stage)
    for k in range(0, len(z['qpos']), step):
        d.qpos[:] = z['qpos'][k]
        mujoco.mj_kinematics(m, d)
        for i, g_r in enumerate(E.geoms):
            if E.fixed[i]:
                continue
            P = E.local[i] @ d.geom_xmat[g_r].reshape(3, 3).T + d.geom_xpos[g_r]
            side = 'left' if E.side[i] == 0 else 'right'
            for g in cab:
                c, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
                # bound first: most pairs are far apart
                rb = np.linalg.norm(m.geom_size[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0.2
                key = (side, bodies[g])
                cur = best.get(key, (np.inf,))[0]
                if np.linalg.norm(P.mean(0) - c) - E.r[i] - rb > min(cur, 0.25):
                    continue
                if g in nonbox:
                    dist = float(cKDTree(nonbox[g] @ R.T + c).query(P)[0].min())
                else:
                    dist = float(box_signed(P, c, R, m.geom_size[g]).min())
                if dist < cur:
                    best[key] = (round(dist, 4), k, E.gname[g_r], names[g], stage_at(k))
    return {f'{a}|{b}': v for (a, b), v in sorted(best.items())}


if __name__ == '__main__':
    for r in sys.argv[1:]:
        print(Path(r).name)
        for k, v in measure(r).items():
            print(' ', k, v)
