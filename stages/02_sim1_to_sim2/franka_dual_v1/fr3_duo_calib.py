"""Camera-alignment trajectories for the real FR3 Duo (2026-11-03): one per task, no contact with any object.

Each trajectory starts and ends at a home pose and visits two viewing poses, holding still at each (the photo
points): the hands over the objects they would grasp (drawer: the handle hand in front of the handle), then over
where the task puts them (stacking: the upper hand over the stack region, the base hand back home; drawer: the block
hand in front of the drawer, the handle hand aside); the hand not needed at a view waits aside (further out and higher
than home). Between holds the joints move on half-cosine profiles,
<= 0.4 rad/s per joint. Every 30 Hz frame is checked in the full FR3 Duo scene (fr3_duo.build_full: mount, cover,
head, both arms with the Robotiq 2F-85): every robot geom >= 4 cm from the table, objects, cabinet and the
mount/head (each arm's own base links excepted), left/right >= 5 cm, joints >= 0.15 rad inside the FR3v2 limits.
The trajectory is then replayed with the position servos at 1/1200 s and must not touch anything.

The viewing targets are geometric (TCP 15-20 cm from the object along a tilted approach); this is a calibration
motion, not task data.

    python -m franka_dual_v1.fr3_duo_calib --out <fresh dir> [--dist 0.20 --height 0.025]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from . import fr3_duo as F  # noqa: E402
from .placement import CASES  # noqa: E402

TASK_CASES = {'stack_cups': 'stack_cups_ep000002', 'stack_blocks': 'stack_blocks_ep000037',
              'drawer': 'drawer_ep000038'}
FPS = 30
V_MAX = 0.4
ENV_CLEAR, ARM_CLEAR, JOINT_MARGIN = 0.04, 0.05, 0.15
HOLD_START, HOLD_VIEW, HOLD_END, MIN_MOVE = 2.0, 3.0, 2.0, 5.0
OPEN_M = 0.085
CHECK_MARGIN = 0.08


class Checker:
    """Clearances from a copy of the model whose robot geoms report contacts within CHECK_MARGIN."""

    def __init__(self, m):
        self.m = copy.deepcopy(m)
        mm = self.m
        names = [mm.body(mm.geom_bodyid[g]).name for g in range(mm.ngeom)]
        self.arm = np.array([n.split('_')[0] if n.startswith(('left_', 'right_')) else '' for n in names])
        self.collides = (mm.geom_contype | mm.geom_conaffinity) > 0
        self.robot = (self.arm != '') & self.collides
        self.base = np.array([any(n.endswith(s) for s in ('fr3v2_link0', 'fr3v2_link1', '_base')) and
                              n.startswith(('left_', 'right_')) and '_g_' not in n for n in names])
        self.mount = np.array([n == 'fr3_duo_mount_body' for n in names])
        self.names = names
        mm.geom_margin[self.robot] = CHECK_MARGIN
        self.d = mujoco.MjData(mm)
        self.qadr = {h: np.array([mm.joint(f'{h}_fr3v2_joint{i}').qposadr[0] for i in range(1, 8)]) for h in F.HANDS}
        self.lo = {h: np.array([mm.jnt_range[mm.joint(f'{h}_fr3v2_joint{i}').id, 0] for i in range(1, 8)]) for h in F.HANDS}
        self.hi = {h: np.array([mm.jnt_range[mm.joint(f'{h}_fr3v2_joint{i}').id, 1] for i in range(1, 8)]) for h in F.HANDS}

    def __call__(self, qpos, ignore_arm=None):
        mm, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_forward(mm, d)
        env, arm = (CHECK_MARGIN, None), (CHECK_MARGIN, None)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot[g1], self.robot[g2]
            if not (r1 or r2) or (ignore_arm and ignore_arm in (self.arm[g1], self.arm[g2])):
                continue
            if r1 and r2:
                if self.arm[g1] == self.arm[g2] or (self.base[g1] and self.base[g2]):
                    continue
                if c.dist < arm[0]:
                    arm = (c.dist, (self.names[g1], self.names[g2]))
                continue
            rg, og = (g1, g2) if r1 else (g2, g1)
            if self.mount[og] and self.base[rg]:
                continue
            if c.dist < env[0]:
                env = (c.dist, (self.names[rg], self.names[og]))
        margin = min(float(np.min(np.r_[qpos[self.qadr[h]] - self.lo[h], self.hi[h] - qpos[self.qadr[h]]]))
                     for h in F.HANDS if h != ignore_arm)
        return dict(env=env, arm=arm, joint_margin=margin)

    def ok(self, r):
        return r['env'][0] >= ENV_CLEAR and r['arm'][0] >= ARM_CLEAR and r['joint_margin'] >= JOINT_MARGIN


def approach_to(target, from_xy, lean_deg, standoff, closing_hint=(1.0, 0.0, 0.0)):
    """TCP pose `standoff` back from `target` along an approach tipped `lean_deg` from straight down toward the target
    as seen from `from_xy`."""
    f = np.r_[target[:2] - from_xy, 0.0]
    f /= np.linalg.norm(f)
    th = np.radians(lean_deg)
    a = np.cos(th) * np.array([0, 0, -1.0]) + np.sin(th) * f
    y = np.cross(a, np.cross(closing_hint, a))
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(a, [0, 1.0, 0])
    y /= np.linalg.norm(y)
    return target - standoff * a, np.c_[np.cross(y, a), y, a]


class Planner:
    def __init__(self, spec, case):
        self.m = spec.compile()
        self.spec = spec
        self.chk = Checker(self.m)
        self.ik = F.DuoIK(self.m)
        self.q0 = mujoco.MjData(self.m).qpos.copy()
        self.rng = np.random.default_rng(0)
        self.case = case

    def arm_q(self, qpos, h):
        return qpos[self.ik.arms[h]['qadr']]

    def with_arm(self, qpos, h, q):
        out = qpos.copy()
        out[self.ik.arms[h]['qadr']] = q
        return out

    def best_ik(self, hand, pos, rot, qpos_other, n_seeds=48, near=None, ignore_arm=None, top=1):
        """IK solutions from random seeds; keep the one with the largest clearance (env, arm, joint margin) and, when
        `near` is given, prefer small joint motion from it."""
        a = self.ik.arms[hand]
        seeds = ([near] if near is not None else []) + [self.rng.uniform(a['lo'], a['hi']) for _ in range(n_seeds)]
        best = None
        self.why = dict(ik=0, env=0, arm=0, joint=0)
        for s in seeds:
            q, pe, re = self.ik.solve(hand, pos, rot, [s])
            if pe > 0.003 or re > 2.0:
                self.why['ik'] += 1
                continue
            qpos = self.with_arm(qpos_other, hand, q)
            r = self.chk(qpos, ignore_arm)
            if not self.chk.ok(r):
                self.why['env'] += r['env'][0] < ENV_CLEAR
                self.why['arm'] += r['arm'][0] < ARM_CLEAR
                self.why['joint'] += r['joint_margin'] < JOINT_MARGIN
                self.last_bad = r
                continue
            score = min(r['env'][0], CHECK_MARGIN) + min(r['arm'][0], CHECK_MARGIN) + 0.2 * min(r['joint_margin'], 0.5)
            if near is not None:
                score -= 0.05 * np.abs(q - near).max()
            best = (best or []) + [(score, q)]
        if not best:
            return None if top == 1 else []
        best.sort(key=lambda x: -x[0])
        uniq = []
        for sc, q in best:
            if all(np.abs(q - u).max() > 0.05 for u in uniq):
                uniq.append(q)
        return uniq[0] if top == 1 else uniq[:top]

    def segment(self, qa, qb):
        dq = np.abs(qb - qa).max()
        T = max(MIN_MOVE, np.pi / 2 * dq / V_MAX)
        n = int(np.ceil(T * FPS))
        s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
        return qa + s[:, None] * (qb - qa)

    def transition(self, qa, cands):
        """First clear motion from qa to one of the candidate configurations: both arms together, or one arm after
        the other."""
        last = None
        for qb in cands:
            moves = [[qa, qb]]
            for h in F.HANDS:
                moves.append([qa, self.with_arm(qa, h, self.arm_q(qb, h)), qb])
            for via in moves:
                seg = np.concatenate([self.segment(a, b) for a, b in zip(via[:-1], via[1:])])
                ok, info = self.path_ok(seg)
                if ok:
                    return qb, seg, len(via) - 2
                last = info
        return None, None, last

    def path_ok(self, frames):
        worst = None
        for k, q in enumerate(frames):
            r = self.chk(q)
            if not self.chk.ok(r):
                return False, dict(frame=k, **r)
            if worst is None or r['env'][0] < worst['env'][0]:
                worst = r
        return True, worst


def mount_frame(place, p):
    """World point -> mount frame (x forward to the table, y left, z up) relative to the mounting point."""
    p0, R0 = place.mount_pose()
    return (R0.T @ (np.asarray(p) - p0)).round(4).tolist()


def plan_task(task, case, place):
    scene = json.loads((CASES / case / 'source_scene.json').read_text())
    spec = F.build_full(CASES / case, scene, place)
    P = Planner(spec, case)
    objs = {o['id']: o for o in scene['objects']}
    b = scene['bindings']
    base_xy = {h: place.arm_base(h)[0][:2] for h in F.HANDS}
    table = F.TABLE_TOP
    # home: hands 0.40 m ahead of the mounting point (over the table), 0.18 m to the side, 0.33 m above the table,
    # fingers pointing 45 deg down toward the workspace
    p0, R0 = place.mount_pose()
    home_q = P.q0.copy()
    for h, side in (('left', 1), ('right', -1)):
        pos = p0 + R0 @ np.array([0.40, side * 0.18, 0.0]) + [0, 0, table + 0.33 - p0[2]]
        a = R0 @ np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5)])
        y = R0 @ np.array([0.0, 1.0, 0.0])
        rot = np.c_[np.cross(y, a), y, a]
        q = P.best_ik(h, pos, rot, home_q, n_seeds=96, ignore_arm='right' if h == 'left' else None)
        if q is None:
            raise RuntimeError(f'{task}: no clear home for {h}: {P.why} {getattr(P, "last_bad", None)}')
        home_q = P.with_arm(home_q, h, q)
    views = []
    if b.get('stack_hands'):
        hb, hu = b['stack_hands']['base'], b['stack_hands']['upper']
        base, upper = b['targets']
        def top(o):
            p = np.array(objs[o]['position_m'], float)
            return p + [0, 0, objs[o]['size_m'][2] / 2]
        v1 = {hb: (top(base), 25.0, 0.15, f'over {base}'), hu: (top(upper), 25.0, 0.15, f'over {upper}')}
        region = np.array(objs[b['target_region']]['position_m'], float)
        v2 = {hu: (region + [0, 0, 0.08], 25.0, 0.15, f"over {b['target_region']}"), hb: None}
        views = [('grasp_targets', v1), ('place_target', v2)]
    else:
        blk = b['targets'][0]
        bar = np.array(objs['bottom_handle_bar']['position_m'], float)
        bh = 'left' if objs[blk]['position_m'][0] > bar[0] else 'right'
        dh = 'right' if bh == 'left' else 'left'
        pb = np.array(objs[blk]['position_m'], float) + [0, 0, 0.025]
        v1 = {bh: (pb, 25.0, 0.15, f'over {blk}'), dh: ('handle', bar, 'in front of bottom_handle_bar')}
        front = np.array(objs['bottom_drawer']['position_m'], float) + [0, objs['bottom_drawer']['size_m'][1] / 2 + 0.07, 0.05]
        v2 = {bh: (front, 45.0, 0.25, 'in front of / above the bottom drawer'), dh: None}
        views = [('grasp_targets', v1), ('place_target', v2)]
    seq = [('home', home_q, 'home', None)]
    cur = home_q
    for name, v in views:
        per_hand, desc = {}, {}
        for h in F.HANDS:
            spec_h = v.get(h)
            side = 1.0 if h == 'left' else -1.0
            if spec_h is None:  # out of the other hand's way: further out and higher than home
                pos = p0 + R0 @ np.array([0.30, side * 0.40, 0.0]) + [0, 0, table + 0.42 - p0[2]]
                a = R0 @ np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5)])
                y = R0 @ np.array([0.0, 1.0, 0.0])
                spec_h = ('pose', pos, np.c_[np.cross(y, a), y, a], 'aside (out and up)')
            if isinstance(spec_h[0], str) and spec_h[0] == 'pose':
                pos, rot, text = spec_h[1:]
            elif isinstance(spec_h[0], str) and spec_h[0] == 'handle':
                bar, text = spec_h[1], spec_h[2]
                th = np.radians(30.0)
                a = np.array([0.0, -np.cos(th), -np.sin(th)])
                y = np.array([1.0, 0, 0])
                rot = np.c_[np.cross(y, a), y, a]
                pos = bar - 0.20 * a
            else:
                tgt, lean, standoff, text = spec_h
                pos, rot = approach_to(tgt, base_xy[h], lean, standoff)
            other = 'right' if h == 'left' else 'left'
            sols = P.best_ik(h, pos, rot, cur, n_seeds=64, near=P.arm_q(cur, h), ignore_arm=other, top=6)
            if not sols:
                raise RuntimeError(f'{task}: {name}: no clear pose for {h} ({text}): {P.why}')
            per_hand[h] = sols
            desc[h] = text
        cands = []
        for ql in per_hand['left']:
            for qr in per_hand['right']:
                q = P.with_arm(P.with_arm(cur, 'left', ql), 'right', qr)
                if P.chk.ok(P.chk(q)):
                    cands.append(q)
        if not cands:
            raise RuntimeError(f'{task}: {name}: the two arms\' poses are not clear together')
        qb, seg, info = P.transition(cur, cands)
        if qb is None:
            raise RuntimeError(f'{task}: home/{name}: no clear motion: {info}')
        seq.append((name, qb, desc, seg))
        cur = qb
    qb, seg, info = P.transition(cur, [home_q])
    if qb is None:
        raise RuntimeError(f'{task}: back home: no clear motion: {info}')
    seq.append(('home_end', home_q, 'home', seg))
    frames, holds = [], []

    def hold(q, secs, label):
        n = int(round(secs * FPS))
        holds.append(dict(label=label, t_start=round(len(frames) / FPS, 3), t_end=round((len(frames) + n - 1) / FPS, 3),
                          photo_t=round((len(frames) + n // 2) / FPS, 3)))
        frames.extend([q] * n)
    hold(seq[0][1], HOLD_START, 'home')
    for nb, qb, db, seg in seq[1:]:
        frames.extend(seg)
        hold(qb, HOLD_END if nb == 'home_end' else HOLD_VIEW, nb)
    frames = np.array(frames)
    clear = [P.chk(q) for q in frames]
    summary = dict(min_env_clearance_m=round(min(c['env'][0] for c in clear), 4),
                   closest_env_pair=min(clear, key=lambda c: c['env'][0])['env'][1],
                   min_arm_clearance_m=round(min(c['arm'][0] for c in clear), 4),
                   min_joint_margin_rad=round(min(c['joint_margin'] for c in clear), 3),
                   max_joint_speed_rad_s=round(float(np.abs(np.diff(frames, axis=0)).max() * FPS), 3))
    rep, track = replay(P, frames)
    photo = [int(round(hd['photo_t'] * FPS)) for hd in holds]
    rep['tracking_err_at_photo_points_rad'] = [round(float(track[k]), 5) for k in photo]
    return P, spec, scene, seq, frames, holds, summary, rep


def replay(P, frames):
    """Position servos at 1/1200 s, 40 steps per 30 Hz frame, grippers commanded open; robot contacts and object
    motion are recorded."""
    m = P.m
    d = mujoco.MjData(m)
    d.qpos[:] = frames[0]
    mujoco.mj_forward(m, d)
    acts = {h: [m.actuator(f'{h}_fr3v2_joint{i}').id for i in range(1, 8)] for h in F.HANDS}
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    obj0 = {m.body(m.jnt_bodyid[j]).name: d.xpos[m.jnt_bodyid[j]].copy() for j in free}
    robot = P.chk.robot
    steps = int(round(1.0 / (FPS * m.opt.timestep)))
    track, touches = [], 0
    for q in frames:
        for h in F.HANDS:
            d.ctrl[acts[h]] = q[P.ik.arms[h]['qadr']]
            d.ctrl[m.actuator(f'{h}_g_fingers_actuator').id] = 0.0
        for _ in range(steps):
            mujoco.mj_step(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (robot[g1] or robot[g2]) and not (P.chk.mount[g1] or P.chk.mount[g2]) and c.dist < 0:
                    if not (robot[g1] and robot[g2] and P.chk.arm[g1] == P.chk.arm[g2]):
                        touches += 1
        track.append(max(np.abs(d.qpos[P.ik.arms[h]['qadr']] - q[P.ik.arms[h]['qadr']]).max() for h in F.HANDS))
    moved = {n: float(np.linalg.norm(d.xpos[m.body(n).id] - p)) for n, p in obj0.items()}
    return dict(max_tracking_err_while_moving_rad=round(float(max(track)), 4), robot_contact_steps=touches,
                object_moved_m={k: round(v, 5) for k, v in moved.items()},
                warnings=int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)), np.array(track)


def write_task(out, task, case, place, P, spec, scene, seq, frames, holds, summary, rep):
    out.mkdir(parents=True, exist_ok=False)
    m = P.m
    (out / 'scene.xml').write_text(spec.to_xml())
    cols = ['time_s'] + [f'{h}_fr3v2_joint{i}' for h in F.HANDS for i in range(1, 8)] + [f'{h}_gripper_width_m' for h in F.HANDS]
    with (out / 'trajectory_30hz.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, q in enumerate(frames):
            w.writerow([f'{k / FPS:.4f}'] + [f'{q[P.ik.arms[h]["qadr"][i]]:.6f}' for h in F.HANDS for i in range(7)] +
                       [f'{OPEN_M:.3f}'] * 2)
    d = mujoco.MjData(m)
    photos = []
    for hd in holds:
        k = int(round(hd['photo_t'] * FPS))
        d.qpos[:] = frames[k]
        mujoco.mj_forward(m, d)
        photos.append(dict(hd, frame=k, joints={h: np.round(frames[k][P.ik.arms[h]['qadr']], 6).tolist() for h in F.HANDS},
                           tcp={h: dict(world=np.round(d.site_xpos[m.site(f'{h}_tcp').id], 4).tolist(),
                                        mount_frame=mount_frame(place, d.site_xpos[m.site(f'{h}_tcp').id]))
                                for h in F.HANDS}))
    targets = {s[0]: s[2] for s in seq}
    objects = [dict(id=o['id'], shape=o['shape'], size_m=o['size_m'], yaw_rad=o.get('yaw_rad', 0.0),
                    world=np.round(o['position_m'], 4).tolist(), mount_frame=mount_frame(place, o['position_m']),
                    from_table_edge_m=round(F.TABLE_EDGE_Y - o['position_m'][1], 4),
                    from_midline_m=round(o['position_m'][0], 4))
               for o in scene['objects'] if o['mobility'] == 'free' or o['shape'] in ('cabinet', 'region')]
    p0, R0 = place.mount_pose()
    meta = dict(task=task, source_case=case, fps=FPS, frames=len(frames), duration_s=round(len(frames) / FPS, 3),
                columns=cols, view_targets=targets, photo_points=photos, safety=summary, sim_replay=rep,
                placement=dict(mounting_point_world=np.round(p0, 4).tolist(), dist_behind_table_edge_m=place.dist_behind_edge,
                               height_above_table_m=place.height_above_table, table_top_m=F.TABLE_TOP,
                               base_height_above_floor_m=round(F.TABLE_TOP + place.height_above_table, 4)),
                objects=objects)
    (out / 'trajectory.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dist', type=float, default=0.20)
    ap.add_argument('--height', type=float, default=0.025)
    ap.add_argument('--tasks', default='stack_cups,stack_blocks,drawer')
    a = ap.parse_args()
    place = F.DuoPlacement(a.dist, a.height)
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for task in a.tasks.split(','):
        case = TASK_CASES[task]
        r = plan_task(task, case, place)
        meta = write_task(a.out / task, task, case, place, *r)
        res[task] = dict(case=case, duration_s=meta['duration_s'], safety=meta['safety'], sim_replay=meta['sim_replay'])
        print(task, json.dumps(res[task], default=float), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(res, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
