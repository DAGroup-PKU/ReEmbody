"""Run one task controller on one FR3 Duo variation, measure it and decide admission (2026-11-03).

A controller is an ordinary Python module with `run(S)` (franka_dual_v1/controllers/). It reads measured state from
the session S (object poses, joints, TCP, gripper width, drawer slide) and issues motion commands; each command is
planned from the state measured when it is issued and executed at once in physics (30 Hz frames, 1/1200 s steps,
FR3v2 position servos, Robotiq 2F-85 force-limited), so the next command sees its outcome. No LLM is called.

Commands (all return after execution):
  S.move_joints({hand: q}, label)        joint-space half-cosine move of one or both arms, collision-checked; when the
                                         simultaneous move is not clear, the arms move one after the other
  S.line({hand: (pos, R)}, label, ...)   straight TCP line (position half-cosine, orientation slerp), per-frame local
                                         IK from the commanded joints
  S.grip({hand: width}, label, s)        gripper width ramp (0 = close with the force limit)
  S.together({hand: plan}, label, ...)   per-hand joint plans (S.plan_line / S.joint_plan, joined in time) executed at
                                         the same time, the two arms' path checked together first
  S.hold(s, label)
  S.ik(hand, pos, [R...], allow=...)     collision-free IK (no motion)
Every 30 Hz frame records the commanded 16-value action (per arm 7 joint targets + gripper width) and the label of the
command that produced it.

Measurements (fed back for revision; all stored): the frozen RSSR contract's predicates with margins, hard
constraints, left/right arm clearance per frame, joint-limit margins, joint speed, tracking error (commanded vs
measured joints), contact events per object and hand, idle spans, state-action conflicts, wrist-camera visibility of
each grasped object before its first contact (recorded, not gated: user 2026-11-03) and head-camera visibility of
every object.

Admission (paper Sec. 3.3): (1) the contract is achieved and held to the end, (2) the hard constraints hold at every
step (the contract's, plus for the FR3 Duo: measured joint speed <= 2 rad/s and commanded-vs-measured joint error
<= 0.3 rad, so no collision jerk), (3) an independent replay of the actions from the reset state reproduces the run bit for bit, (4) learnable:
each object grasped at most once (targets exactly once), no state-action ambiguity, left/right clearance >= 2 cm in
every frame, no idle segment.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

from . import contract as C  # noqa: E402
from . import duo_scene as DS  # noqa: E402
from . import fr3_duo as F  # noqa: E402
from .duo_robot import DualFR3  # noqa: E402
from .servo import ARM_CLEARANCE_M, Executor, digest, replay  # noqa: E402

FPS = 30
SETTLE_S = 2.0
HANDS = F.HANDS
V_MAX = 0.6  # rad/s, joint-space moves
STEP_MAX = 0.025  # rad per 30 Hz frame for any commanded joint (0.75 rad/s)
PLAN_MARGIN = 0.06  # contact margin of the planning model
ENV_CLEAR, ARM_CLEAR, JOINT_MARGIN = 0.01, 0.025, 0.025  # joint margin just under the IK limit margin (0.03)
ARM_GAP_BUFFER = 0.005  # planning keeps this much more than a layout's duo_site.ARM_GAP_M
SUPPORT_BUFFER = 0.002  # planning keeps this much more than a layout's duo_site.support_clearance_m (2026-11-06: 5 -> 2 mm for
# the v2 drawer; admission measures the executed frames exactly, so a tighter plan only risks a rejected run; v2 cups and
# blocks were made with 5 mm)
IK_POS_TOL, IK_ROT_TOL = 0.003, 3.0
LINE_POS_TOL = 0.005  # per-frame IK position tolerance along straight lines (old Franka gate)
OPEN_M = 0.085
HELD_PEN_TOL = 0.001  # a held object may touch but not penetrate other objects in a plan
SOFT_W, SOFT_TOL_DEG = 0.15, 30.0  # rotation about a line's soft axis: weight and tolerance
# learnability
IDLE_S = 1.0
IDLE_EPS = 1e-4
AMBIG_Q, AMBIG_W, AMBIG_OBJ, AMBIG_MIN_STEP, AMBIG_GAP_S = 0.05, 0.005, 0.01, 0.004, 1.0
CLOSED_CMD_M = 0.02  # a gripper command below this width is a closing (all grasps command 0)
# robot safety (FR3 Duo hard constraints added to the contract's): a collision shows as a jerk of the measured joints
# 0.3 rad: lag under drawer load reaches 0.16 rad at < 1 rad/s; the one collision seen had 2.2 rad at 8 rad/s
MAX_JOINT_SPEED, MAX_TRACK_ERR = 2.0, 0.3  # rad/s (FR3 joint speed limits 2.6-3.1), rad


class PlanError(RuntimeError):
    pass


class Clearance:
    """Planning check on a copy of the model: robot collision geoms report contacts within PLAN_MARGIN."""

    def __init__(self, m, margin=PLAN_MARGIN, support_clear=None, exact=None, exact_gap=False):
        self.m = copy.deepcopy(m)
        mm = self.m
        self.margin = margin
        # exact (exact_clearance.Exact on the same geoms, duo_site.EXACT_LAYOUTS): the arch distance (and with exact_gap
        # the arm-to-arm distance) is measured on surface samples; MuJoCo's contacts for those pairs are not used
        self.exact, self.exact_gap = exact, exact_gap
        names = [mm.body(mm.geom_bodyid[g]).name for g in range(mm.ngeom)]
        self.names = names
        self.arm = np.array([n.split('_')[0] if n.startswith(('left_', 'right_')) else '' for n in names])
        collides = (mm.geom_contype | mm.geom_conaffinity) > 0
        self.robot = (self.arm != '') & collides
        self.base = np.array([n.startswith(('left_', 'right_')) and '_g_' not in n and
                              n.endswith(('fr3v2_link0', 'fr3v2_link1', '_base')) for n in names])
        self.mount = np.array([n == 'fr3_duo_mount_body' for n in names])
        # site supports (duo_site NAME_V4: the camera arch): planning keeps SUPPORT_CLEARANCE_M to them, not ENV_CLEAR
        from .duo_site import SUPPORT_CLEARANCE_M
        sup = np.array([n.startswith('site_support') for n in names])
        # support_clear (duo_site.support_clearance_m, when a layout keeps more than SUPPORT_CLEARANCE_M): planning keeps
        # it + SUPPORT_BUFFER, and the supports get a contact margin that reaches beyond it
        keep = SUPPORT_CLEARANCE_M if support_clear is None else support_clear + SUPPORT_BUFFER
        self.keep = np.where(sup, keep - ENV_CLEAR, 0.0)
        self.sup, self.keep_sup = sup, keep - ENV_CLEAR
        mm.geom_margin[self.robot] = margin
        if support_clear is not None:
            mm.geom_margin[sup] = keep + 0.03
        self.d = mujoco.MjData(mm)
        self.geom_body = mm.geom_bodyid.copy()

    def __call__(self, qpos, allow=None, attached=None):
        """allow: {hand: set of body names its geoms may touch}; attached: {hand: (body id, free-joint qpos address,
        TCP site id, object pose in the TCP frame (p, R))}: a held object moves with its hand and must not penetrate
        anything (by more than HELD_PEN_TOL) except its hand. Returns (env (dist, pair), arm (dist, pair))."""
        mm, d = self.m, self.d
        d.qpos[:] = qpos
        held = {}
        if attached:
            mujoco.mj_kinematics(mm, d)
            for h, (b, adr, site, (pr, Rr)) in attached.items():
                pt, Rt = d.site_xpos[site], d.site_xmat[site].reshape(3, 3)
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, (Rt @ Rr).reshape(9))
                d.qpos[adr:adr + 3] = pt + Rt @ pr
                d.qpos[adr + 3:adr + 7] = q
                held[b] = h
        mujoco.mj_forward(mm, d)
        env, arm = (self.margin, None), (self.margin, None)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = self.robot[g1], self.robot[g2]
            if held and not (r1 or r2):
                b1, b2 = self.geom_body[g1], self.geom_body[g2]
                if (b1 in held or b2 in held) and b1 != b2 and c.dist < -HELD_PEN_TOL:
                    pen = float(c.dist) + HELD_PEN_TOL - ENV_CLEAR  # reported below ENV_CLEAR: a violation
                    if pen < env[0]:
                        env = (pen, (self.names[g1], self.names[g2], 'held object penetrates'))
                continue
            if not (r1 or r2):
                continue
            if r1 and r2:
                if self.arm[g1] == self.arm[g2] or (self.base[g1] and self.base[g2]) or self.exact_gap:
                    continue
                if c.dist < arm[0]:
                    arm = (float(c.dist), (self.names[g1], self.names[g2]))
                continue
            rg, og = (g1, g2) if r1 else (g2, g1)
            if self.mount[og] and self.base[rg]:
                continue
            if self.exact is not None and self.sup[og]:
                continue
            if allow and self.names[og] in allow.get(self.arm[rg], ()):
                continue
            dist = float(c.dist) - self.keep[og]
            if dist < env[0]:
                env = (dist, (self.names[rg], self.names[og]))
        if self.exact is not None:
            a = self.exact.arch_distance(d, cap=self.keep_sup + env[0])
            if a[0] - self.keep_sup < env[0]:
                env = (a[0] - self.keep_sup, (a[1], a[2], 'exact'))
            if self.exact_gap:
                g = self.exact.arm_gap(d, cap=arm[0])
                if g[0] < arm[0]:
                    arm = (g[0], (g[1], g[2], 'exact'))
        return env, arm


def _hc(n):
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)


class Session:
    def __init__(self, v: DS.Variation):
        self.v = v
        self.m, self.scene, self.scene_sha, self.xml = DS.build(v)
        self.objects = {o['id']: o for o in self.scene['objects']}
        self.ids = [o['id'] for o in self.scene['objects']]
        self.bindings = self.scene['bindings']
        # left/right arm clearance planning keeps: duo_site.ARM_GAP_M (+ ARM_GAP_BUFFER) where a layout sets one for the
        # task, else ARM_CLEAR; the planning model's margin and the measured clearance then reach beyond it
        from . import duo_site
        self.arm_gap = duo_site.ARM_GAP_M.get((v.layout, v.task))
        self.arm_clear = ARM_CLEAR if self.arm_gap is None else self.arm_gap + ARM_GAP_BUFFER
        # arm-to-arch clearance planning and admission keep (duo_site.support_clearance_m; 5 cm before NAME_V7)
        self.support_clear = duo_site.support_clearance_m(v.layout)
        sc = None if self.support_clear == duo_site.SUPPORT_CLEARANCE_M else self.support_clear
        self.robot = DualFR3(self.m) if self.arm_gap is None else DualFR3(self.m, clear_margin=self.arm_gap + 0.05)
        self.ex = Executor(self.m, self.ids, self.robot)
        self.d = self.ex.reset()
        self.prev = self.robot.action_from_state(self.d)
        self.frames = [self.ex.snapshot(self.d)]
        self.trace = [self.ex.trace_row(self.d)]
        self.audit = dict(robot_table_penetration_m=0.0, joint_limit_excess=-np.inf, finger_coupling_m=0.0,
                          solver_warnings=0)
        self.actions, self.labels, self.notes = [], [], []
        self.exact = None
        if v.layout in duo_site.EXACT_LAYOUTS:
            from .exact_clearance import Exact
            self.exact = Exact(self.m)
        self.clear = Clearance(self.m, PLAN_MARGIN if self.arm_gap is None else max(PLAN_MARGIN, self.arm_clear + 0.03),
                               support_clear=sc, exact=self.exact, exact_gap=self.exact is not None and self.arm_gap is not None)
        self.ik_solver = F.DuoIK(self.m)
        self.rng = np.random.default_rng(0)
        self.home = {h: DS.home_q(v, h) for h in HANDS}
        self.place = DS.place_of(v)
        self.allow = {h: set() for h in HANDS}  # bodies each hand may touch now (set by the controller)
        self.attached = {}  # hand -> held object carried along in plan checks (attach / detach)

    # ---- measured state ------------------------------------------------------------------------
    def pose(self, body):
        b = self.m.body(body).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def q(self, hand):
        return self.robot.arm_q(self.d, hand)

    def q_cmd(self, hand):
        k = HANDS.index(hand)
        return self.prev[8 * k:8 * k + 7].copy()

    def tcp(self, hand):
        a = self.robot.arms[hand]
        return self.d.site_xpos[a.tcp].copy(), self.d.site_xmat[a.tcp].reshape(3, 3).copy()

    def width(self, hand):
        return self.robot.width(self.d, hand)

    def slide(self, joint):
        return float(self.d.qpos[self.m.joint(joint).qposadr[0]])

    def contacts(self, hand, body):
        """(left side, right side) finger contact with the body now."""
        bid = self.m.body(body).id
        sides = self.robot.finger_bodies(hand)
        hit = [False, False]
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist > 0:
                continue
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for mine, other in ((b1, b2), (b2, b1)):
                if other == bid:
                    for s in (0, 1):
                        if mine in sides[s]:
                            hit[s] = True
        return tuple(hit)

    def t(self):
        return len(self.actions) / FPS

    def note(self, text):
        self.notes.append(dict(t=round(self.t(), 3), note=text))

    # ---- kinematics ----------------------------------------------------------------------------
    def fk(self, hand, q):
        return self.robot.fk(hand, q, self.d.qpos)

    def _qpos_with(self, qs):
        qp = self.d.qpos.copy()
        for h, q in qs.items():
            qp[self.robot.arms[h].qadr] = q
        return qp

    def attach(self, hand, obj):
        """From now on, plan checks carry obj with the hand at its currently measured pose relative to the TCP."""
        b = self.m.body(obj).id
        adr = int(self.m.jnt_qposadr[self.m.body_jntadr[b]])
        pt, Rt = self.tcp(hand)
        po, Ro = self.pose(obj)
        self.attached[hand] = (b, adr, self.robot.arms[hand].tcp, (Rt.T @ (po - pt), Rt.T @ Ro))

    def detach(self, hand):
        self.attached.pop(hand, None)

    def check(self, qs, allow=None):
        env, arm = self.clear(self._qpos_with(qs), self.allow if allow is None else allow, self.attached)
        jm = min(float(np.min(np.r_[q - self.robot.arms[h].lo, self.robot.arms[h].hi - q])) for h, q in qs.items())
        return env, arm, jm

    def ok(self, env, arm, jm):
        return env[0] >= ENV_CLEAR and arm[0] >= self.arm_clear and jm >= JOINT_MARGIN

    def ik(self, hand, pos, rots, near=None, n_seeds=24, other=None):
        """Collision-free IK solution for any of the rotations; returns (q, R) closest to `near` (default: the
        commanded joints), or raises PlanError with the reasons."""
        return self.ik_many(hand, pos, rots, near, n_seeds, other, k=1)[0]

    def ik_many(self, hand, pos, rots, near=None, n_seeds=24, other=None, k=4):
        """Up to k distinct (> 0.2 rad apart) collision-free IK solutions [(q, R)], closest to `near` first."""
        near = self.q_cmd(hand) if near is None else np.asarray(near)
        other_q = {h: self.q_cmd(h) for h in HANDS if h != hand} if other is None else other
        a = self.robot.arms[hand]
        seeds = [near, self.home[hand]] + [self.rng.uniform(a.lo, a.hi) for _ in range(n_seeds)]
        found, why = [], dict(ik=0, env=None, arm=None, joint=0)
        for R in rots:
            for s in seeds:
                q, pe, re = self.ik_solver.solve(hand, np.asarray(pos, float), np.asarray(R, float), [s])
                if pe > IK_POS_TOL or re > IK_ROT_TOL:
                    why['ik'] += 1
                    continue
                env, arm, jm = self.check({hand: q, **other_q})
                if not self.ok(env, arm, jm):
                    if env[0] < ENV_CLEAR and (why['env'] is None or env[0] > why['env'][0]):
                        why['env'] = env
                    if arm[0] < self.arm_clear and (why['arm'] is None or arm[0] > why['arm'][0]):
                        why['arm'] = arm
                    why['joint'] += jm < JOINT_MARGIN
                    continue
                found.append((float(np.abs(q - near).max()), q, np.asarray(R, float)))
        if not found:
            raise PlanError(f'{hand} IK at {np.round(pos, 3).tolist()}: {why}')
        found.sort(key=lambda x: x[0])
        out = []
        for _, q, R in found:
            if all(np.abs(q - u).max() > 0.2 for u, _ in out):
                out.append((q, R))
            if len(out) >= k:
                break
        return out

    def ik_local_soft(self, hand, pos, rot, q_init, q_post, soft_axis, iters=40, damping=0.02, null_gain=0.1,
                      max_iter_step=0.05, margin=0.03):
        """Local damped least squares like robot.ik_local, with the rotation about the world axis `soft_axis` weighted
        SOFT_W (the hand may turn about it). Returns (q, pos err m, rot err deg about the other axes, deg about it)."""
        a = self.robot.arms[hand]
        m, w = self.m, self.robot._work
        f = np.asarray(soft_axis, float) / np.linalg.norm(soft_axis)
        P = np.eye(3) - (1 - SOFT_W) * np.outer(f, f)
        lo, hi = a.lo + margin, a.hi - margin
        q = np.clip(np.asarray(q_init, float), lo, hi)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        w.qpos[:] = self.d.qpos
        for _ in range(iters):
            w.qpos[a.qadr] = q
            mujoco.mj_kinematics(m, w)
            mujoco.mj_comPos(m, w)
            ep = pos - w.site_xpos[a.tcp]
            er = P @ Rotation.from_matrix(rot @ w.site_xmat[a.tcp].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(m, w, jp, jr, a.tcp)
            J = np.r_[jp[:, a.dadr], 0.3 * P @ jr[:, a.dadr]]
            e = np.r_[ep, 0.3 * er]
            JJt = J @ J.T + damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))
            dq += null_gain * (np.eye(7) - Jpinv @ J) @ (np.asarray(q_post) - q)
            n = np.abs(dq).max()
            if n > max_iter_step:
                dq *= max_iter_step / n
            q = np.clip(q + dq, lo, hi)
        w.qpos[a.qadr] = q
        mujoco.mj_kinematics(m, w)
        rv = Rotation.from_matrix(rot @ w.site_xmat[a.tcp].reshape(3, 3).T).as_rotvec()
        about = float(np.degrees(abs(rv @ f)))
        perp = float(np.degrees(np.linalg.norm(rv - (rv @ f) * f)))
        return q, float(np.linalg.norm(w.site_xpos[a.tcp] - pos)), perp, about

    # ---- motion -------------------------------------------------------------------------------
    def _execute(self, plan, label, grips=None):
        """plan: {hand: (n, 7) joint targets}; grips: {hand: (n,) width}; other values held."""
        n = max([len(p) for p in plan.values()] + [len(g) for g in (grips or {}).values()] + [0])
        for k in range(n):
            a = self.prev.copy()
            for i, h in enumerate(HANDS):
                if h in plan and len(plan[h]):
                    a[8 * i:8 * i + 7] = plan[h][min(k, len(plan[h]) - 1)]
                if grips and h in grips and len(grips[h]):
                    a[8 * i + 7] = grips[h][min(k, len(grips[h]) - 1)]
            self._step(a, label)

    def _step(self, a, label):
        a = np.asarray(a, float)
        ex, d = self.ex, self.d
        n = ex.steps_per_frame
        for s in range(1, n + 1):
            self.robot.apply_action(d, self.prev + (a - self.prev) * (s / n))
            mujoco.mj_step(self.m, d)
            ex.step_audit(d, self.audit)
            if s % ex.trace_stride == 0:
                self.trace.append(ex.trace_row(d))
        self.prev = a
        self.actions.append(a)
        self.labels.append(label)
        self.frames.append(ex.snapshot(d))

    def _joint_segment(self, qa, qb, v_max=V_MAX, min_s=0.6):
        dq = float(np.abs(qb - qa).max())
        T = max(min_s, np.pi / 2 * dq / v_max)
        n = max(1, int(np.ceil(T * FPS)))
        return qa + _hc(n)[:, None] * (qb - qa)

    def _path_ok(self, plan, allow=None, others=None):
        n = max(len(p) for p in plan.values())
        worst = None
        for k in range(0, n, 2):
            qs = {h: (plan[h][min(k, len(plan[h]) - 1)] if h in plan else
                      (others or {}).get(h, self.q_cmd(h))) for h in HANDS}
            env, arm, jm = self.check(qs, allow)
            if not self.ok(env, arm, jm):
                return False, dict(t=round(k / FPS, 2), env=env, arm=arm, joint_margin=round(jm, 3))
        return True, worst

    def move_joints(self, targets, label, v_max=V_MAX, allow=None):
        qa = {h: self.q_cmd(h) for h in HANDS}
        qb = {h: np.asarray(targets.get(h, qa[h]), float) for h in HANDS}
        movers = [h for h in HANDS if np.abs(qb[h] - qa[h]).max() > 1e-6]
        if not movers:
            return
        T = np.pi / 2 * max(np.abs(qb[h] - qa[h]).max() for h in movers) / v_max
        candidates = [({h: self._joint_segment(qa[h], qb[h], v_max, min_s=max(0.6, T)) for h in movers}, None)]
        if len(movers) == 2:
            for first, second in (('left', 'right'), ('right', 'left')):
                p1 = self._joint_segment(qa[first], qb[first], v_max)
                p2 = self._joint_segment(qa[second], qb[second], v_max)
                candidates.append(({first: np.r_[p1, np.repeat(p1[-1:], len(p2), 0)],
                                    second: np.r_[np.repeat(qa[second][None], len(p1), 0), p2]}, first))
        last = None
        for plan, first in candidates:
            ok, info = self._path_ok(plan, allow)
            if ok:
                if first:
                    self.note(f'{label}: arms moved one after the other ({first} first)')
                self._execute(plan, label)
                return
            last = info
        raise PlanError(f'{label}: no clear joint path: {last}')

    def plan_line(self, goals, speed=0.10, min_s=0.5, allow=None, start=None, check=True, label='line', soft=None):
        """Joint plans {hand: (n, 7)} for straight TCP lines to goals {hand: (pos, R)} from `start` joints (default: the
        commanded joints), executed together; raises PlanError. Nothing is executed. soft: {hand: world axis} about
        which that hand may turn (up to SOFT_TOL_DEG) to stay reachable."""
        start = start or {}
        q0 = {h: np.asarray(start.get(h, self.q_cmd(h)), float) for h in HANDS}
        n = 0
        starts = {}
        for h, (p, R) in goals.items():
            p0, R0 = self.fk(h, q0[h])
            starts[h] = (p0, R0)
            ang = Rotation.from_matrix(np.asarray(R) @ R0.T).magnitude()
            T = max(min_s, np.linalg.norm(np.asarray(p) - p0) / speed, ang / 0.6)
            n = max(n, int(np.ceil(T * FPS)))
        n_nominal = n
        for _ in range(4):
            if n > max(3 * n_nominal, n_nominal + 3 * FPS):
                raise PlanError(f'{label}: line would need {n / FPS:.1f} s (nominal {n_nominal / FPS:.1f} s): it passes '
                                'near a singularity or a joint limit')
            plans, worst_step = {}, 0.0
            for h, (p, R) in goals.items():
                p0, R0 = starts[h]
                s = _hc(n)
                slerp = Slerp([0, 1], Rotation.from_matrix([R0, np.asarray(R)]))
                q = q0[h].copy()
                post = q.copy()
                out = np.zeros((n, 7))
                for k in range(n):
                    pk = p0 + s[k] * (np.asarray(p) - p0)
                    Rk = slerp([s[k]]).as_matrix()[0]
                    if soft and h in soft:
                        qn, pe, re, about = self.ik_local_soft(h, pk, Rk, q, post, soft[h])
                        if pe > LINE_POS_TOL or re > IK_ROT_TOL or about > SOFT_TOL_DEG:
                            raise PlanError(f'{label}: {h} line leaves the reachable set at {k / n:.0%} '
                                            f'({pe * 1000:.1f} mm, {re:.1f} deg, {about:.0f} deg about the soft axis)')
                        worst_step = max(worst_step, float(np.abs(qn - q).max()))
                        out[k] = q = qn
                        continue
                    qn, pe, re = self.robot.ik_local(h, pk, Rk, q, self.d.qpos, post, iters=30)
                    if pe > LINE_POS_TOL or re > IK_ROT_TOL:
                        # near a joint limit the damped step stalls: bounded least squares from the same joints
                        qs, pe2, re2 = self.ik_solver.solve(h, pk, Rk, [q])
                        if pe2 <= LINE_POS_TOL and re2 <= IK_ROT_TOL and np.abs(qs - q).max() <= 4 * STEP_MAX:
                            qn, pe, re = qs, pe2, re2
                    if pe > LINE_POS_TOL or re > IK_ROT_TOL:
                        raise PlanError(f'{label}: {h} line leaves the reachable set at {k / n:.0%} '
                                        f'({pe * 1000:.1f} mm, {re:.1f} deg)')
                    worst_step = max(worst_step, float(np.abs(qn - q).max()))
                    out[k] = q = qn
                plans[h] = out
            if worst_step <= STEP_MAX:
                break
            n = int(np.ceil(n * worst_step / STEP_MAX * 1.1))
        if check:
            ok, info = self._path_ok(plans, allow, others={h: q0[h] for h in HANDS if h not in goals})
            if not ok:
                raise PlanError(f'{label}: line not clear: {info}')
        return plans

    def line(self, goals, label, speed=0.10, min_s=0.5, allow=None, check=True, soft=None):
        """goals: {hand: (pos, R)}. Straight TCP lines from the commanded TCP poses, executed together."""
        self._execute(self.plan_line(goals, speed, min_s, allow, check=check, label=label, soft=soft), label)

    def joint_plan(self, hand, q, v_max=V_MAX, start=None):
        """Half-cosine joint plan (n, 7) of one arm from `start` (default: its commanded joints) to q; nothing is
        executed or checked."""
        q0 = self.q_cmd(hand) if start is None else np.asarray(start, float)
        return self._joint_segment(q0, np.asarray(q, float), v_max)

    def together(self, plans, label, grips=None, allow=None, delay=None):
        """Execute per-hand joint plans {hand: (n, 7)} at the same time; a hand whose plan ends first holds its last
        target, delay {hand: s} starts that hand's plan later. grips {hand: width} ramps the gripper over the first
        0.4 s. The two arms' joint path is checked together first; raises PlanError (nothing executed)."""
        plans = {h: np.asarray(p, float) for h, p in plans.items() if len(p)}
        for h, s in (delay or {}).items():
            if h in plans and s > 0:
                plans[h] = np.r_[np.repeat(self.q_cmd(h)[None], int(round(s * FPS)), 0), plans[h]]
        if not plans:
            return
        ok, info = self._path_ok(plans, allow)
        if not ok:
            raise PlanError(f'{label}: arms not clear together: {info}')
        g = None
        if grips:
            n = int(np.ceil(0.4 * FPS))
            g = {h: self.prev[8 * HANDS.index(h) + 7] + _hc(n) * (w - self.prev[8 * HANDS.index(h) + 7])
                 for h, w in grips.items()}
        self._execute(plans, label, g)

    def run_plans(self, plans, label):
        """Execute joint plans made by plan_line (one after the other, in order)."""
        for p in plans:
            self._execute(p, label)

    def grip(self, widths, label, duration=0.6):
        n = max(1, int(np.ceil(duration * FPS)))
        g = {}
        for h, w in widths.items():
            w0 = self.prev[8 * HANDS.index(h) + 7]
            g[h] = w0 + _hc(n) * (w - w0)
        self._execute({}, label, g)

    def hold(self, duration, label):
        self._execute({}, label, {h: np.full(max(1, int(round(duration * FPS))), self.prev[8 * i + 7])
                                  for i, h in enumerate(HANDS)})

    # ---- result --------------------------------------------------------------------------------
    def finish(self, settle_s=SETTLE_S):
        ex, d = self.ex, self.d
        for s in range(1, int(round(settle_s / self.m.opt.timestep)) + 1):
            self.robot.apply_action(d, self.prev)
            mujoco.mj_step(self.m, d)
            ex.step_audit(d, self.audit)
            if s % ex.trace_stride == 0:
                self.trace.append(ex.trace_row(d))
        frames = self.frames
        roll = {k: np.array([f[k] for f in frames]) for k in frames[0]}
        roll['action'] = np.array(self.actions).reshape(len(self.actions), -1)
        roll['free_bodies'] = np.array(ex.free_bodies)
        roll['slide_joints'] = np.array(ex.slides)
        tr = {k: np.array([r[k] for r in self.trace]) for k in self.trace[0] if k != 'articulation'}
        tr['articulation_open_fraction'] = {j: np.array([r['articulation'][j] for r in self.trace]) for j in ex.slides}
        return dict(frames=roll, trace=tr, audit=self.audit)


# ---- measurement and admission ----------------------------------------------------------------------
def _spans(mask):
    out, start = [], None
    for i, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def grasp_events(S, res):
    """Grasps per object and hand: each closing of that hand's gripper (a maximal run of frames commanding a width
    below CLOSED_CMD_M) during which both finger sides touch the object. Contact flicker while the fingers stay closed
    is one grasp; opening and closing on the object again is a second one. Returns {'hand:object': [(t0, t1), ...]}."""
    tr, f = res['trace'], res['frames']
    fs = tr['finger_contact_sides']  # (T, obj, hand, side) at 120 Hz
    tt = tr['time_s']
    out = {}
    for hi, h in enumerate(HANDS):
        closed = f['action'][:, 8 * hi + 7] < CLOSED_CMD_M
        for a, b in _spans(closed):
            sel = (tt > a / FPS) & (tt <= b / FPS + 0.5)  # contact may be made right after the command ends
            for oi, o in enumerate(S.ids):
                both = fs[sel, oi, hi, 0].astype(bool) & fs[sel, oi, hi, 1].astype(bool)
                if both.sum() >= 12:  # >= 0.1 s of two-sided contact
                    out.setdefault(f'{h}:{o}', []).append((round(a / FPS, 3), round(b / FPS, 3)))
    return out


def idle_spans(res):
    a = res['frames']['action']
    still = np.all(np.abs(np.diff(a, axis=0)) < IDLE_EPS, axis=1)
    return [(round(s / FPS, 3), round(e / FPS, 3)) for s, e in _spans(still) if (e - s) / FPS > IDLE_S]


def ambiguity(res):
    """Pairs of frames >= AMBIG_GAP_S apart with similar state (joints within AMBIG_Q, widths within AMBIG_W, free
    objects within AMBIG_OBJ) whose commanded joint motions point in opposite directions."""
    f = res['frames']
    q = np.concatenate([f['state'][:-1, 0:7], f['state'][:-1, 16:23]], 1)
    w = f['action'][:, [7, 15]]
    obj = f['objects'][:-1, :, :3].reshape(len(q), -1)
    act = np.concatenate([f['action'][:, 0:7], f['action'][:, 8:15]], 1) - q
    mag = np.linalg.norm(act, axis=1)
    gap = int(AMBIG_GAP_S * FPS)
    worst = None
    idx = np.flatnonzero(mag > AMBIG_MIN_STEP)
    for i in idx:
        j = idx[idx > i + gap]
        if not len(j):
            continue
        sim = ((np.abs(q[j] - q[i]).max(1) < AMBIG_Q) & (np.abs(w[j] - w[i]).max(1) < AMBIG_W) &
               (np.abs(obj[j] - obj[i]).max(1) < AMBIG_OBJ))
        if not sim.any():
            continue
        jj = j[sim]
        cos = act[jj] @ act[i] / (mag[jj] * mag[i])
        k = int(np.argmin(cos))
        if cos[k] < -0.5 and (worst is None or cos[k] < worst[2]):
            worst = (round(i / FPS, 3), round(jj[k] / FPS, 3), round(float(cos[k]), 3))
    return worst


def wrist_view(S, res, vp, grasps):
    """Visible pixels of each grasped object in the grasping hand's wrist camera from t = 0 to its first contact."""
    out = {}
    d = mujoco.MjData(S.m)
    fs = res['trace']['contacts']
    for hand, obj in grasps:
        oi = S.ids.index(obj)
        c = fs[:, oi, HANDS.index(hand)].astype(bool)
        if not c.any():
            out[f'{hand}:{obj}'] = dict(first_contact_s=None)
            continue
        k = min(int(np.floor(res['trace']['time_s'][int(np.argmax(c))] * FPS + 1e-9)), len(res['frames']['qpos']) - 1)
        px = []
        for j in range(0, k + 1):
            d.qpos[:] = res['frames']['qpos'][j]
            mujoco.mj_forward(S.m, d)
            px.append(vp.visible_pixels(d, f'{hand}_wrist', [obj]))
        px = np.array(px)
        out[f'{hand}:{obj}'] = dict(first_contact_s=round(k / FPS, 3), min_px=int(px.min()),
                                    frames_visible=round(float((px > 0).mean()), 3),
                                    px_every_half_s=px[::FPS // 2].astype(int).tolist())
    return out


def head_view(S, res, vp, objects):
    d = mujoco.MjData(S.m)
    out = {}
    n = len(res['frames']['qpos'])
    ks = list(range(0, n, FPS // 2))
    for o in objects:
        vis = []
        for j in ks:
            d.qpos[:] = res['frames']['qpos'][j]
            mujoco.mj_forward(S.m, d)
            oc = vp.occlusion(d, 'head', [o])
            vis.append(oc['visible_px'] / oc['silhouette_px'] if oc['silhouette_px'] else 0.0)
        vis = np.array(vis)
        out[o] = dict(mean_visible=round(float(vis.mean()), 3), min_visible=round(float(vis.min()), 3),
                      frames_fully_hidden=int((vis == 0).sum()), sampled_every_s=0.5)
    return out


V1_MARGIN = 0.05  # every commanded joint at least this far inside the first-generation FR3 ranges (NAME_V4)


def support_clearance(S, res):
    """(min distance, time, geom pair) between the robot's collision geoms and the site support geoms over the
    executed frames, or None without supports. Measured as contact distances with a 0.3 m margin on the supports
    (mj_geomDistance on the FR3 meshes now and then returns 0 for pairs 13 cm apart)."""
    m = copy.deepcopy(S.m)
    names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
    sup = np.array([n.startswith('site_support') for n in names])
    if not sup.any():
        return None
    rob = np.array([n.startswith(('left_', 'right_')) for n in names])
    m.geom_margin[:] = 0.0
    m.geom_margin[sup] = 0.3
    d = mujoco.MjData(m)
    best = (np.inf, 0.0, None)
    for k, qp in enumerate(res['frames']['qpos']):
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (sup[g1] and rob[g2]) or (sup[g2] and rob[g1]):
                if c.dist < best[0]:
                    r, o = (g2, g1) if sup[g1] else (g1, g2)
                    best = (float(c.dist), k / FPS, f'{m.geom(r).name or names[r]} - {m.geom(o).name}')
    return round(best[0], 4), round(best[1], 3), best[2]


def exact_clearances(S, res):
    """Exact (min distance, time, pair) to the camera arch and between the arms over the executed frames
    (exact_clearance.Exact; duo_site.EXACT_LAYOUTS)."""
    d = mujoco.MjData(S.m)
    arch, gap = (np.inf, 0.0, None), (np.inf, 0.0, None)
    for k, qp in enumerate(res['frames']['qpos']):
        d.qpos[:] = qp
        mujoco.mj_kinematics(S.m, d)
        a = S.exact.arch_distance(d, cap=arch[0])
        if a[0] < arch[0]:
            arch = (a[0], k / FPS, f'{a[1]} - {a[2]}')
        g = S.exact.arm_gap(d, cap=gap[0])
        if g[0] < gap[0]:
            gap = (g[0], k / FPS, f'{g[1]} - {g[2]}')
    return ((round(arch[0], 4), round(arch[1], 3), arch[2]), (round(gap[0], 4), round(gap[1], 3), gap[2]))


def joint_limit_gate(f):
    """FR3 v2 position-based velocity limits (1x, as libfranka applies them) and the margin to the first-generation
    FR3 ranges, on the 30 Hz commanded targets with the measured start as row 0 (fr3_limits)."""
    from . import fr3_limits as L
    st, a = f['state'], f['action']
    Q = np.vstack([np.r_[st[0, 0:7], st[0, 16:23]], np.c_[a[:, 0:7], a[:, 8:15]]])
    r = L.check(Q, speeds=(1.0,))
    marg = []
    for c in range(14):
        lo, hi = L.FR3_V1[c % 7]
        marg.append((float(min((Q[:, c] - lo).min(), (hi - Q[:, c]).min())), f"{L.HANDS[c // 7]}_J{c % 7 + 1}"))
    m0 = min(marg)
    return {'pbv_violations_x1.0': r['pbv_violations_x1.0'], 'pbv_first_violation': r['pbv_first_violation'],
            'min_margin_to_fr3_v2_rad': r['min_pos_margin_rad'], 'min_margin_to_fr3_v1_rad': round(m0[0], 4),
            'min_margin_to_fr3_v1_joint': m0[1], 'max_speed_rad_s': r['max_speed_rad_s']}


def measure(S, res, task_grasps, view=True):
    ex = S.ex
    hard = ex.hard_constraints(res)
    contract = C._contracts.compile_contract(S.scene['task_id'], S.scene, S.bindings)
    tr = res['trace']
    arrays = dict(time_s=tr['time_s'], object_position=tr['object_position'],
                  object_quaternion=tr['object_quaternion'], contacts=tr['contacts'],
                  object_contacts=tr['object_contacts'], finger_contact_sides=tr['finger_contact_sides'],
                  gripper_opening=tr['gripper_opening'], articulation_open_fraction=tr['articulation_open_fraction'])
    ev = C._contracts.evaluate(contract, S.scene, arrays, {k: hard[k] for k in contract['hard_constraints']})
    f = res['frames']
    q = f['state']
    margins = {}
    for i, h in enumerate(HANDS):
        a = S.robot.arms[h]
        qa = q[:, 16 * i:16 * i + 7]
        margins[h] = round(float(np.min(np.minimum(qa - a.lo, a.hi - qa))), 4)
    arm_q = np.concatenate([q[:, 0:7], q[:, 16:23]], 1)
    cmd = np.concatenate([f['action'][:, 0:7], f['action'][:, 8:15]], 1)
    track = np.abs(arm_q[1:] - cmd).max(1)
    clear = f['arm_clearance_m']
    grasps = grasp_events(S, res)
    targets = S.bindings['targets']
    regrasp = {k: v for k, v in grasps.items() if len(v) > 1}
    per_obj = {}
    for k, v in grasps.items():
        per_obj[k.split(':', 1)[1]] = per_obj.get(k.split(':', 1)[1], 0) + len(v)
    missing = [t for t in targets if per_obj.get(t, 0) == 0]
    multi = sorted(o for o, n in per_obj.items() if n > 1)
    idle = idle_spans(res)
    amb = ambiguity(res)
    m = dict(
        contract=dict(task_success=bool(ev['task_success']), failed_predicates=ev['failed_predicates'],
                      hard_failures=ev['hard_failures'],
                      predicates={p['id']: dict(passed=bool(p['passed']), **{k: v for k, v in p['measurements'].items()
                                                if isinstance(v, (int, float, bool)) or v is None})
                                  for p in ev['predicates']}),
        hard=hard, audit={k: (v if isinstance(v, dict) else float(v)) for k, v in res['audit'].items()},
        min_arm_clearance_m=round(float(clear.min()), 4),
        min_arm_clearance_t=round(float(np.argmin(clear) / FPS), 3),
        min_joint_limit_margin_rad=margins,
        max_joint_speed_rad_s=round(float(np.abs(np.diff(arm_q, axis=0)).max() * FPS), 3),
        max_tracking_err_rad=round(float(track.max()), 4),
        p95_tracking_err_rad=round(float(np.quantile(track, 0.95)), 4),
        grasps=grasps, idle_spans=idle, state_action_conflict=amb,
        duration_s=round(len(f['action']) / FPS, 3),
        physics_steps=int(len(f['action']) * ex.steps_per_frame + round(SETTLE_S / S.m.opt.timestep)))
    if view:
        from .metrics import VisionProbe
        vp = VisionProbe(S.m, *DS.view_size(S.v), robot=S.robot)
        m['wrist_view'] = wrist_view(S, res, vp, task_grasps)
        m['head_view'] = head_view(S, res, vp, [o for o in S.ids if S.objects[o]['mobility'] == 'free'])
    fails = []
    if not ev['task_success']:
        fails.append('contract: ' + ', '.join(ev['failed_predicates'] + ev['hard_failures']))
    if not all(hard[k] for k in contract['hard_constraints']):
        fails.append('hard: ' + ', '.join(k for k in contract['hard_constraints'] if not hard[k]))
    if m['max_joint_speed_rad_s'] > MAX_JOINT_SPEED or m['max_tracking_err_rad'] > MAX_TRACK_ERR:
        v = np.abs(np.diff(arm_q, axis=0)).max(1) * FPS
        k = int(np.argmax(v))
        fails.append(f"hard: robot jerked (joint speed {m['max_joint_speed_rad_s']:.2f} rad/s, tracking error "
                     f"{m['max_tracking_err_rad']:.3f} rad; fastest at t={k / FPS:.2f}s): a collision")
    if clear.min() < ARM_CLEARANCE_M:
        fails.append(f'learnable: arm clearance {clear.min() * 1000:.1f} mm at t={np.argmin(clear) / FPS:.2f}s')
    if S.arm_gap is not None and clear.min() < S.arm_gap:
        fails.append(f'hard: the arms {clear.min() * 1000:.0f} mm apart at t={np.argmin(clear) / FPS:.2f}s '
                     f'(< {S.arm_gap * 1000:.0f} mm, duo_site.ARM_GAP_M)')
    sup = support_clearance(S, res)
    if sup is not None and S.exact is not None:
        # duo_site.EXACT_LAYOUTS: the arch and arm-to-arm gates use exact distances on the executed frames (MuJoCo's
        # contact values stay recorded as min_support_clearance_mujoco_m / min_arm_clearance_m)
        m['min_support_clearance_mujoco_m'] = sup[0]
        sup, gap = exact_clearances(S, res)
        m['min_arm_gap_exact_m'], m['min_arm_gap_exact_t'], m['min_arm_gap_exact_pair'] = gap
        if S.arm_gap is not None and gap[0] < S.arm_gap:
            fails.append(f'hard: the arms {gap[0] * 1000:.0f} mm apart (exact) at t={gap[1]:.2f}s ({gap[2]}, '
                         f'< {S.arm_gap * 1000:.0f} mm)')
    if sup is not None:
        m['min_support_clearance_m'], m['min_support_clearance_t'], m['min_support_clearance_pair'] = sup
        if sup[0] < S.support_clear:
            fails.append(f'hard: arm {sup[0] * 1000:.0f} mm from the camera support ({sup[2]}) at t={sup[1]:.2f}s '
                         f'(< {S.support_clear * 1000:.0f} mm)')
    if S.v.layout is not None:
        from . import duo_site
        if S.v.layout in duo_site.SITE_V6:
            lim = joint_limit_gate(f)
            m['fr3_limits'] = lim
            if lim['pbv_violations_x1.0']:
                fails.append(f"hard: FR3 v2 position-based velocity limit exceeded at 1x {lim['pbv_violations_x1.0']} "
                             f"(first {lim['pbv_first_violation']})")
            if lim['min_margin_to_fr3_v1_rad'] < V1_MARGIN:
                fails.append(f"hard: {lim['min_margin_to_fr3_v1_rad']:.3f} rad from the first-generation FR3 range "
                             f"({lim['min_margin_to_fr3_v1_joint']}), < {V1_MARGIN} rad")
    if missing:
        fails.append(f'learnable: never grasped {missing}')
    if multi:
        fails.append(f'learnable: grasped more than once {multi} ({regrasp})')
    if idle:
        fails.append(f'learnable: idle {idle}')
    if amb:
        fails.append(f'learnable: state-action conflict between t={amb[0]}s and t={amb[1]}s (cos {amb[2]})')
    return m, fails


def load_controller(path: Path):
    spec = importlib.util.spec_from_file_location(f'controller_{abs(hash(str(path)))}', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(controller: Path, v: DS.Variation, out: Path | None = None, view=True, save_episode=False,
            options: dict | None = None) -> dict:
    """options: overrides of the controller's OPTIONS (the refine stage's rule settings), stored in the record."""
    t0 = time.time()
    rec = dict(variation=v.to_json(), controller=str(controller))
    if options:
        rec['options'] = options
    try:
        S = Session(v)
    except Exception as e:  # noqa: BLE001
        rec.update(status='scene_error', error=repr(e))
        return rec
    rec['scene_sha256'] = S.scene_sha
    mod = load_controller(Path(controller))
    if options:
        mod.OPTIONS = {**getattr(mod, 'OPTIONS', {}), **options}
    try:
        mod.run(S)
        rec['status'] = 'completed'
    except PlanError as e:
        rec.update(status='plan_error', error=str(e))
    except Exception as e:  # noqa: BLE001
        rec.update(status='controller_error', error=repr(e), traceback=traceback.format_exc()[-2000:])
    rec['notes'] = S.notes
    if not S.actions:
        rec.update(admitted=False, fails=[f"{rec['status']}: {rec.get('error', 'no motion')}"], wall_s=round(time.time() - t0, 1))
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'record.json').write_text(json.dumps(rec, indent=1, default=_js) + '\n')
        return rec
    res = S.finish()
    meas, fails = measure(S, res, mod.grasps(S) if hasattr(mod, 'grasps') else [], view=view)
    if rec['status'] != 'completed':
        fails.insert(0, f"{rec['status']}: {rec.get('error')}")
    rp = replay(S.m, S.ids, res['frames']['action'], digest(res), SETTLE_S, robot_cls=DualFR3)
    meas['replay'] = rp
    if not rp['bit_exact']:
        fails.append('replay differs')
    labels = S.labels
    meas['stages'] = [dict(label=labels[a], t0=round(a / FPS, 3), t1=round(b / FPS, 3))
                      for a, b in _label_spans(labels)]
    rec.update(measurements=meas, fails=fails, admitted=not fails, wall_s=round(time.time() - t0, 1))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'record.json').write_text(json.dumps(rec, indent=1, default=_js) + '\n')
        if save_episode or rec['admitted']:
            f = res['frames']
            np.savez_compressed(out / 'episode.npz', time=f['time_s'], state=f['state'], action=f['action'],
                                qpos=f['qpos'], qvel=f['qvel'], tcp=f['tcp'], objects=f['objects'],
                                slides=f['slides'], arm_clearance_m=f['arm_clearance_m'],
                                labels=np.array(labels), free_bodies=f['free_bodies'])
            (out / 'scene.xml').write_text(S.xml)
    return rec


def _label_spans(labels):
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i))
            start = i
    return out


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)
