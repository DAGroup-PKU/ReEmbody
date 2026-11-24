"""Keep an arm off the camera arch with its redundancy (2026-11-05, drawer: "优化一下动作模式"): the FR3 has one joint
more than a hand pose needs, so for a fixed TCP position and orientation the elbow can still swing about the
shoulder-wrist line. After each IK solve of the chosen hand, its joints are stepped in the null space of the TCP
Jacobian along the gradient of its distance to the arch (every arm-to-arch contact closer than TARGET, weighted by how
much closer), then put back on the exact pose; a step is kept only if the arm's distance to the arch grows.

Nothing in the rules changes: the planner still keeps duo_site.SUPPORT_CLEARANCE_M (5 cm) to the arch, admission and
the export checks are unchanged. Only the joint solutions the planner picks change.

v9 (10 cm arch rule): the push goes on to PUSH_EXTRA beyond the layout's clearance; when the planner's IK search finds
no clear solution, its raw solutions are pushed first and checked after (pushed_search); on duo_site.EXACT_LAYOUTS the
distance and its gradient come from exact_clearance (S.exact), not from MuJoCo's mesh contacts.

    install(S, hand[, lines])   (in a controller's run(S)): wraps S.ik_local_soft, S.robot.ik_local and S.ik_many
    for `hand`
"""
import copy

import mujoco
import numpy as np

TARGET = 0.10  # push while the arm is closer than this to the arch (m)
STEP = 0.01  # largest joint step per relaxation step of a key pose (rad)
STEP_LINE = 0.004  # per line waypoint (lines advance at most duo_run.STEP_MAX = 0.025 rad per frame)
ITERS_LINE = 1  # relaxation steps per line waypoint (the posture drifts outward along the line)
ITERS_POSE = 60  # relaxation steps for a key pose (IK solution)
MARGIN = 0.3
PUSH_EXTRA = 0.02  # v9: push until this far beyond the layout's arch clearance (tried 0.03)


class ArchAway:
    def __init__(self, S, hand):
        self.S, self.hand = S, hand
        self.m = copy.deepcopy(S.m)
        m = self.m
        names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self.arch = np.array([n == 'site_support_arch' for n in names])
        self.mine = np.array([n.startswith(hand + '_') for n in names]) & ((m.geom_contype | m.geom_conaffinity) > 0)
        m.geom_margin[:] = 0.0
        m.geom_margin[self.arch] = MARGIN
        self.d = mujoco.MjData(m)
        self.a = S.robot.arms[hand]
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        self.stats = dict(poses=0, line_points=0, raised_cm=0.0)
        # v9: push until 3 cm beyond the layout's arch clearance (duo_site.support_clearance_m; TARGET for 5 cm)
        self.target = max(TARGET, getattr(S, 'support_clear', 0.05) + PUSH_EXTRA)
        self.line_mode = 'push'  # 'push' or 'posture' (along lines; a controller may switch it per stage)
        self.iters_line = ITERS_LINE

    def distance(self, q):
        """(min arm-to-arch distance, gradient of the soft objective over the hand's 7 joints)."""
        m, d, a = self.m, self.d, self.a
        d.qpos[:] = self.S.d.qpos
        for h, qq in self.S.robot.arms.items():
            if h != self.hand:
                d.qpos[qq.qadr] = self.S.q_cmd(h)
        d.qpos[a.qadr] = q
        if getattr(self.S, 'exact', None) is not None:
            # v9 (duo_site.EXACT_LAYOUTS): the exact distance (surface samples against the arch capsules) and its closest
            # point; MuJoCo's mesh contact distances read up to 1.6 cm too far at some poses, and the push found those
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = self.S.exact.arch_distance(d, 0 if self.hand == 'left' else 1, cap=self.target + 0.05)  # beyond: no push
            g = np.zeros(7)
            if e[0] < self.target:
                mujoco.mj_jac(m, d, self.jp, None, e[3], int(m.geom_bodyid[e[5]]))
                g = (self.target - e[0]) * (self.jp[:, a.dadr].T @ e[4])
            return e[0], g
        mujoco.mj_forward(m, d)
        dmin, g = np.inf, np.zeros(7)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.mine[g1] and self.arch[g2]:
                r, away = g1, -c.frame[:3]
            elif self.mine[g2] and self.arch[g1]:
                r, away = g2, c.frame[:3]
            else:
                continue
            dist = float(c.dist)
            dmin = min(dmin, dist)
            if dist < self.target:
                mujoco.mj_jac(m, d, self.jp, None, c.pos, int(m.geom_bodyid[r]))
                g += (self.target - dist) * (self.jp[:, a.dadr].T @ away)
        return dmin, g

    def relax(self, q, pos, R, iters, step=STEP):
        """q moved along the null space of the TCP pose (pos, R) away from the arch; never closer than it was."""
        S, a, m, d = self.S, self.a, self.m, self.d
        q = np.asarray(q, float).copy()
        d0, g = self.distance(q)
        best = d0
        lo, hi = a.lo + 0.03, a.hi - 0.03
        for _ in range(iters):
            if best >= self.target or not np.any(g):
                break
            mujoco.mj_jacSite(m, d, self.jp, self.jr, a.tcp)
            J = np.r_[self.jp[:, a.dadr], self.jr[:, a.dadr]]
            N = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
            dq = N @ g
            n = np.abs(dq).max()
            if n < 1e-9:
                break
            qt = np.clip(q + dq * (step / n), lo, hi)
            qt, pe, re = S.robot.ik_local_base(self.hand, pos, R, qt, S.d.qpos, qt, iters=10)
            if pe > 5e-4 or re > 0.3:
                break
            dt, gt = self.distance(qt)
            if dt <= best:
                break
            q, best, g = qt, dt, gt
        self.stats['raised_cm'] = max(self.stats['raised_cm'], (best - d0) * 100 if np.isfinite(d0) else 0.0)
        return q


def install(S, hand, lines=True, holding=True):
    """lines=False: only the key poses (ik_many) are relaxed; the hand's lines are planned as before. lines='posture':
    along lines the hand keeps its current posture (null-space target = the previous waypoint) but is not pushed.
    holding=False: key poses are left as solved while the hand carries an object (S.attached)."""
    A = ArchAway(S, hand)
    A.line_mode = 'posture' if lines == 'posture' else 'push'
    if not hasattr(S.robot, 'ik_local_base'):
        S.robot.ik_local_base = S.robot.ik_local  # the planner's own solver (relax uses it, never a wrapper)
    soft0, local0, many0 = S.ik_local_soft, S.robot.ik_local, S.ik_many

    # Along lines the wrapped hand keeps its current posture (null-space target = the previous waypoint, not the line's
    # start joints, which would pull the elbow back each waypoint) and drifts STEP_LINE per waypoint away from the arch.
    # A waypoint whose relaxed solution is less exact than the plain one keeps the plain one.
    def ik_local(h, pos, R, q_init, qpos, q_post, iters=30, **kw):
        if h != hand:
            return local0(h, pos, R, q_init, qpos, q_post, iters=iters, **kw)
        plain = local0(h, pos, R, q_init, qpos, q_init, iters=iters, **kw)
        if plain[1] > 1e-3 or A.line_mode == 'posture':
            return plain
        A.stats['line_points'] += 1
        q = A.relax(plain[0], pos, R, A.iters_line, STEP_LINE)
        out = local0(h, pos, R, q, qpos, q, iters=5, **kw)
        return out if out[1] <= max(plain[1], 1e-4) and out[2] <= max(plain[2], 0.05) else plain

    def ik_local_soft(h, pos, R, q_init, q_post, soft_axis, **kw):
        if h != hand:
            return soft0(h, pos, R, q_init, q_post, soft_axis, **kw)
        plain = soft0(h, pos, R, q_init, q_init, soft_axis, **kw)
        if plain[1] > 1e-3 or A.line_mode == 'posture':
            return plain
        A.stats['line_points'] += 1
        R_now = S.fk(h, plain[0])[1]  # the soft solve may have turned the hand about the axis: keep that turn
        q = A.relax(plain[0], pos, R_now, A.iters_line, STEP_LINE)
        out = soft0(h, pos, R, q, q, soft_axis, iters=5)
        return out if out[1] <= max(plain[1], 1e-4) and out[2] <= max(plain[2], 0.05) and out[3] <= plain[3] + 0.5 else plain

    def pushed_search(h, pos, rots, near, n_seeds, other, k):
        """v9: the planner's IK search with every raw solution pushed off the arch before the clearance check (with a
        10 cm rule the planner rejects poses at 10.2 cm that the push would carry past it)."""
        from franka_dual_v1 import duo_run as RUN
        near = S.q_cmd(h) if near is None else np.asarray(near)
        other_q = {o: S.q_cmd(o) for o in S.robot.arms if o != h} if other is None else other
        a = S.robot.arms[h]
        seeds = [near, S.home[h]] + [S.rng.uniform(a.lo, a.hi) for _ in range(n_seeds)]
        found = []
        for R in rots:
            for sd in seeds:
                q, pe, re = S.ik_solver.solve(h, np.asarray(pos, float), np.asarray(R, float), [sd])
                if pe > RUN.IK_POS_TOL or re > RUN.IK_ROT_TOL:
                    continue
                A.stats['pushed_search'] = A.stats.get('pushed_search', 0) + 1
                qr = A.relax(q, np.asarray(pos, float), S.fk(h, q)[1], ITERS_POSE)
                if S.ok(*S.check({h: qr, **other_q})):
                    found.append((float(np.abs(qr - near).max()), qr, np.asarray(R, float)))
        found.sort(key=lambda x: x[0])
        out = []
        for _, q, R in found:
            if all(np.abs(q - q2).max() > 0.2 for q2, _ in out):
                out.append((q, R))
            if len(out) >= k:
                break
        return out

    def ik_many(h, pos, rots, near=None, n_seeds=24, other=None, k=4):
        try:
            out = many0(h, pos, rots, near, n_seeds, other, k)
        except Exception:  # noqa: BLE001
            if h != hand or (not holding and h in S.attached):
                raise
            out = pushed_search(h, pos, rots, near, n_seeds, other, k)
            if not out:
                raise
            return out
        if h != hand or (not holding and h in S.attached):
            return out
        res = []
        for q, R in out:
            A.stats['poses'] += 1
            qr = A.relax(q, np.asarray(pos, float), S.fk(h, q)[1], ITERS_POSE)
            env, arm, jm = S.check({h: qr, **{o: S.q_cmd(o) for o in S.robot.arms if o != h}} if other is None
                                   else {h: qr, **other})
            res.append((qr if S.ok(env, arm, jm) else q, R))
        return res

    if lines:
        S.robot.ik_local = ik_local
        S.ik_local_soft = ik_local_soft
    S.ik_many = ik_many
    S.arch_away = A
    S.arch_away_all = {**getattr(S, 'arch_away_all', {}), hand: A}
    return A
