"""Drawer on the FR3 Duo with Robotiq 2F-85 (controller written by Claude Code, claude-opus-5-5, 2026-11-03).

Contract order: the block starts outside the bottom drawer -> the drawer is opened to >= 80 % of its slide -> the
block is inside it, resting on it -> the drawer is closed to <= 10 % and held -> block and drawer stable, released.
Hands: the block hand is the one on the block's side of the handle, the handle hand the other (as gates.roles).
The handle is grasped once and held from the pull to the push. Every target is computed from poses measured when the
stage starts.

Stages (each hand's grasp is chosen among candidates whose whole path - pre-grasp, straight reach, and the lift or the
pull - is planned collision-free before anything moves; several IK branches per candidate):
  1. handle hand: its whole hold is planned backwards from the hardest pose - IK at the pulled-open pose, the straight
     pull back to the bar and the reach back to a pre-grasp 10 cm in front of the bar, executed reversed (one IK branch
     throughout); it moves to the pre-grasp (approach toward the cabinet, level, else pitched
     down 15 deg, fingers closing across the bar from above and below, pinch point 6 mm short of the bar centre so
     the fingertips stay off the drawer face); block hand to 10 cm back along its approach from the block. The block
     hand approaches leaned 25 deg (or 15/35) so that, once turned to face the cabinet, its gripper body stays on the
     robot side of the middle drawer front; it closes across the face pair whose turn to the insertion pose is
     reachable;
  2. both straight in and close; the block hand backs the block out 12 cm along its approach (up and toward the robot);
  3. block hand: carries the block to a waiting pose over its own side (out of the pulling arm's way); handle hand:
     pulls the drawer straight out to 88 % of its slide (again if the drawer is measured below 85 %) and keeps holding
     the handle. The pull, the push and the block hand's lines into the drawer may turn the hand about the vertical
     through its pinch point (soft axis, up to 30 deg) to stay reachable: the pads stay on the bar's top and bottom
     faces and the block stays level;
  4. block hand: turns to face the cabinet and moves (joint space, the held block included in the collision check) to a
     staging pose 20 cm in front of the cabinet face (14 or 10 cm if the line in is not reachable), then straight in over the exposed part of the opened drawer (the
     first reachable of 8, 13, 3, 18 cm from its centre toward the block hand's side) and down, lowering the block
     until its bottom is 5 mm above the drawer floor, opens, backs out along its approach (else straight up) and moves
     aside (straight line, else joint move, else it stays above the drawer front);
  (before the block goes in, the drawer is measured and pulled out again if it has crept below 85 %;)
  5. handle hand: pushes the drawer back until it is 3 mm from closed (again, up to twice, while it is measured more
     than 5 % open), holds 0.3 s, opens and backs off 8 cm (site_v6: 4 cm, then sideways).

duo_drawer_v2d (2026-11-07; user, after a replay on the real robot: "抽屉任务左臂和抽屉还是发生了碰撞，你不要把夹爪离抽屉太近，而是
在拉开抽屉上方约3cm松开夹爪让方块掉进去"; duo_drawer_v2r.py otherwise): replay of drawer_v2_train_0127 touched the drawer; in the
simulation its fingers were 0.9 mm from the drawer's side wall inside the drawer, and MuJoCo reported contact there, because the
planner let the block hand touch the drawer (S.allow). Now the block hand never goes into the drawer: it stops with the block's
bottom DROP_CLEAR above the rim and opens, and the block falls in. The drawer is no longer allowed to it, and the block hand and arm
keep CAB_KEEP from the cabinet and the drawer, on exact surface samples (CabinetGuard; before: 1 cm from MuJoCo's contacts).

duo_drawer_v2r (2026-11-06, the v2 drawer's refine controller; duo_drawer_v2.py otherwise): OPTIONS['overlap'] lets the
two hands move at the same time where their joint paths are clear together (the refine stage's R2): the block hand's
approach during the handle hand's approach route, its move to the waiting pose during the pull, its move aside during
the push; where the joint path is not clear together, the moves run one after the other as before.

duo_drawer_v2 (2026-11-06, v2 data on fr3_site_20261105d, 6 cm arch rule): ARCH_OK 7 cm and ARCH_GOOD 9 cm (+1 and +3
cm over the rule, as v8b/v8c had for 5 cm; the v9 file had 11 / 13 cm for the 10 cm rule); when a waypoint ('via') route
keeps ARCH_OK, the Cartesian routes along the cabinet are not planned at all (they are 3x longer and never chosen then);
the waypoint search covers the VIA_GRASPS best-ranked handle grasps with VIA_SEEDS random IK seeds and stops at the first
grasp with a clear one (planning time: the IK solves of the full search took most of a > 30 min run).

v2 data refine options (2026-11-06; OPTIONS, the refine stage's rules, defaults = the Sim2 program):
  speed: every planned TCP line's speed and every joint move's speed limit times `speed` (lines keep the planner's
         0.025 rad per frame limit; joint moves capped at 0.72 rad/s, 0.5 rad/s while a hand holds the block);
  grip_s: every gripper action at most this long (s);
  short_exit: after the push the handle hand only frees its fingers from the bar (no move aside); the episode ends.

v9 (2026-11-05, duo_site fr3_site_20261105c, user: "拱架立柱也保证10cm安全距离"): the planner keeps 10 cm (+5 mm) to the
camera arch; the approach routes compete on length from ARCH_OK = 11 cm, and handle grasps are ranked up to ARCH_GOOD =
13 cm (v8c: 6 and 8 cm, for the 5 cm rule).

v8c (2026-11-05, S1 cabinet): one more kind of approach route, 'via': two joint moves through a waypoint beside and
above the pre-grasp (VIA_OFFSETS: out to the hand's own side, up, toward the robot), with the grasp orientation or the
start orientation; it competes with the Cartesian routes on length as in v8b.

v8b (2026-11-05, S1 cabinet): the handle hand's approach route is the shortest planned one that keeps ARCH_OK (6 cm)
from the camera arch (v8a: the one farthest from it, 19 s along the cabinet's side); every route and handle grasp is
planned before choosing. Failed direct joint approaches are noted.

S1 cabinet revision (2026-11-05, duo_site fr3_site_20261105b, user: "抽屉建模错了，参考我之前在S1任务的抽屉和方块的建模，复用"):
the cabinet and block of the S1 drawer task (35 x 26.5 x 35 cm, four 8.75 cm tiers, 1.56 cm tall handle bar). The handle
hand opens only OPEN_BAR = 3.5 cm (at 6 cm the open fingers reached the fixed drawer fronts above and below the handle)
and both hands open before anything is planned, so plans are checked with the fingers as they will be; the insertion
points follow from the drawer's inner width; the block hand's waiting pose is given in the mount frame (WAIT_MOUNT: the
desk, and with it the sim world, is 15 cm closer to the robot on this layout).

Factory gripper revision (2026-11-05, duo_site fr3_site_20261105a): the way in from the side may also turn the held
block 45 or 90 deg about the vertical (SIDE_TURNS), and first moves out of the cabinet's footprint to the opened
drawer's depth (first in to SIDE_GAP outside the cabinet's side wall) before going across; approach_over plans every route and handle grasp and takes the one
whose arm stays farthest from the camera arch (the first at least ARCH_GOOD away); handle grasps are ranked by their
arm's distance to the arch over the whole hold.

Site_v6 revision 3 (2026-11-05): when no staged way into the drawer is clear (on site_v6 the staging pose in front of
the cabinet falls onto the opened drawer), the block hand comes in from the side: up above the rim, straight across,
down. Handle grasps: up to HANDLE_TRIES candidates per approach route.

Site_v6 revision 2 (2026-11-05, user: "手完成任务后不要后退这样会碰到拱架，向侧方向移动即可"): after placing the block the
block hand rises straight out of the drawer and moves straight out to its own side; after the push the handle hand backs
off only the 4 cm that free its fingers from the bar and moves straight out to its own side.

Site_v6 revision (2026-11-04, duo_site fr3_site_20261104c): there the user's start pose leaves the handle hand beside the
cabinet and below its top, and no direct joint path reaches the pre-grasp. If none does, the handle hand takes a planned
Cartesian route (approach_over: back along the cabinet's side, else over its top), then the block hand moves to its
pre-grasp.
"""
import mujoco
import numpy as np

HANDS = ('left', 'right')
DROP_CLEAR = 0.03  # the block's bottom above the drawer rim when the hand opens
LOWER = 0.03  # the last stretch down to that height is straight
Y_SHIFTS = (0.015, 0.03, 0.0)  # release point from the middle of the exposed drawer, toward the robot (the block stays 3.5 cm inside the walls)
CAB_KEEP = 0.03  # block hand and arm to the cabinet and the drawer (exact)
CAB_WORDS = ('cabinet', 'drawer', 'handle')
PRE = 0.10
LIFT = 0.12
OPEN_BLOCK = 0.075
OPEN_BAR = 0.035  # S1 cabinet: the handle bar is 1.56 cm tall, the fixed fronts start 5.1 cm above and below its centre
PULL_FRACTION = 0.88
PULL_MIN = 0.85
BAR_SHORT = 0.006
# level first: pitched grasps squeeze the bar out of the 2F-85 (drawer creeps back 2.5-16 mm/s at 30 deg, <= 0.75 at
# 15 deg, none level; sweep of 2026-11-03)
BAR_PITCHES = (0.0, 15.0)
LEANS = (25.0, 15.0, 35.0)
INSERT_SIDES = (0.08, 0.13, 0.03, 0.18)  # insertion points: offset from the drawer centre toward the block hand
INSERT_WALL_GAP = 0.012  # S1 cabinet: insertion points keep the block this far inside the drawer's side walls
# turns of the held block about the vertical for the way in from the side (a cube: a quarter turn looks the same);
# +-45 / +-90 added 2026-11-05 for the factory gripper mounting, where the wrist cannot hold +-20 along the way
SIDE_TURNS = (0, 20, -20, 45, -45, 90, -90)
SIDE_GAP = 0.07  # factory revision: the way in from the side passes this far outside the cabinet's side wall (TCP)
SIDE_PUSH_ITERS = 3  # factory revision: arch_away steps per line point on the way in from the side
BOTH_ORDERS = False  # turned handle grasps with finger order -1 too (tried 2026-11-05: the block slipped, off)
ACROSS_OFFSET = 0.0  # the way in from the side goes across this far off the cabinet face (tried 0.03: closer to the arch)
BAR_YAWS = (0.0, 20.0, -20.0, 35.0, -35.0)  # the handle hand may face the bar turned about the vertical
GAP = 0.005
STAGE_DISTS = (0.20, 0.14, 0.10)  # staging pose in front of the cabinet face: the straight line in is shorter
UP = np.array([0.0, 0.0, 1.0])  # soft axis: the hand may turn about the vertical through its pinch point
# block-hand waiting TCP during the pull: |x|, y, height above the table (factory revision: y -0.50 -> -0.36, in front
# of the cabinet face, so the way into the drawer needs no forward move beside the cabinet)
WAIT = (0.42, -0.36, 0.30)
WAIT_MOUNT = (0.682, 0.42, 0.30)  # S1 cabinet revision: WAIT in the mount frame (forward, |left|, height above the table)
OVER_TOP = 0.12  # site_v6 approach over the cabinet: handle hand this far above the cabinet top
HANDLE_TRIES = 24  # handle grasp candidates (IK branches, yaws) tried on each route (site_v6: 12; factory revision: 24)
ARCH_GOOD = 0.09  # v2 (6 cm rule; v9 0.13, v8c 0.08). factory revision: an approach route this far from the camera arch is taken without trying more
VIA_OFFSETS = ((0.12, 0.12, 0.0), (0.18, 0.10, 0.0), (0.12, 0.20, 0.05), (0.25, 0.15, 0.05), (0.08, 0.25, 0.08),
               (0.15, 0.28, 0.0), (0.05, 0.32, 0.05))  # v2: two higher waypoints
VIA_LIFTS = (0.0, 0.10, 0.18)  # v2: first straight up this far from the start pose (it starts behind the cabinet top)
VIA_GRASPS, VIA_SEEDS = 8, 4  # v2: waypoint routes for the 8 best-ranked handle grasps, 4 random IK seeds each (v8c: 24, 24)
ARCH_OK = 0.07  # v2 (6 cm rule; v9 0.11, v8b 0.06). v8b: approach routes at least this far from the camera arch compete on length (the gate is 5 cm)
REVERSAL_COS = -0.3  # factory revision: approach routes whose joints turn back at a corner more than this are skipped


OPTIONS = dict(speed=1.0, grip_s=None, short_exit=False, overlap=False)
V_CAP, V_CAP_HELD = 0.72, 0.5


class CabinetGuard:
    """Distance from one arm's robot geoms to every collision geom of the cabinet bodies, on exact_clearance's surface samples
    (signed distance to boxes; the handles' cylinders as their boxes). A geom's bounding sphere skips far pairs."""

    def __init__(self, S, hand):
        from franka_dual_v1.exact_clearance import Exact
        self.m = S.m
        self.d = mujoco.MjData(S.m)
        self.E = S.clear.exact if getattr(S.clear, 'exact', None) is not None else Exact(S.m)
        side = 0 if hand == 'left' else 1
        self.idx = [i for i in range(len(self.E.geoms)) if self.E.side[i] == side and not self.E.fixed[i]]
        m = S.m
        self.cab, half = [], []
        for g in range(m.ngeom):
            b = m.body(m.geom_bodyid[g]).name
            if b.startswith(('left_', 'right_')) or not any(w in b for w in CAB_WORDS) or not (m.geom_contype[g] | m.geom_conaffinity[g]):
                continue
            t, sz = m.geom_type[g], m.geom_size[g]
            half.append(sz.copy() if t == mujoco.mjtGeom.mjGEOM_BOX else np.array([sz[0], sz[0], sz[1] + (sz[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)]))
            self.cab.append(g)
        self.half = np.array(half)

    def nearest(self, qpos, stop_below=-np.inf):
        """(distance, robot geom, cabinet geom); stops at the first distance under stop_below."""
        d, E = self.d, self.E
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, d)
        c = d.geom_xpos[self.cab]
        R = d.geom_xmat[self.cab].reshape(-1, 3, 3)
        best = (np.inf, None, None)
        for i in self.idx:
            g = E.geoms[i]
            Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            q = np.einsum('jki,jk->ji', R, (Rg @ E.c_local[i] + pg) - c)
            lb = np.linalg.norm(np.maximum(np.abs(q) - self.half, 0.0), axis=1) - E.r[i]
            near = np.flatnonzero(lb < best[0])
            if not len(near):
                continue
            P = E.local[i] @ Rg.T + pg
            for j in near[np.argsort(lb[near])]:
                if lb[j] >= best[0]:
                    break
                dd = np.abs((P - c[j]) @ R[j]) - self.half[j]
                v = float((np.linalg.norm(np.maximum(dd, 0.0), axis=1) + np.minimum(dd.max(axis=1), 0.0)).min())
                if v < best[0]:
                    best = (v, E.gname[g], self.m.geom(self.cab[j]).name)
                    if v < stop_below:
                        return best
        return best


def install_guard(S, hand):
    """Every path and IK check (S.check) of the planner also keeps CAB_KEEP between this arm and the cabinet."""
    guard = CabinetGuard(S, hand)
    check0 = S.check

    def check(qs, allow=None):
        env, arm, jm = check0(qs, allow)
        if hand in qs and S.ok(env, arm, jm):
            v = guard.nearest(S._qpos_with(qs), CAB_KEEP)
            if v[0] < CAB_KEEP:
                env = (0.0, (v[1], v[2], f'cabinet guard {v[0]:.3f}'))
        return env, arm, jm

    S.check = check


def install_refine(S):
    """The refine options' speed and gripper timing, applied to the planner's own calls (each scaled once: S.line goes
    through S.plan_line)."""
    k = float(OPTIONS.get('speed', 1.0))
    gs = OPTIONS.get('grip_s')
    if isinstance(gs, (list, tuple)):  # the cups' (open, close, release) form: the longest
        gs = max(gs)
    if k == 1.0 and not gs:
        return
    pl, mj, jp, gr = S.plan_line, S.move_joints, S.joint_plan, S.grip
    from franka_dual_v1.duo_run import V_MAX

    def cap(hands, v):
        held = any(h in S.attached for h in hands)
        return min(v * k, V_CAP_HELD if held else V_CAP)

    def plan_line(goals, speed=0.10, min_s=0.5, *a, **kw):
        # v2r: a line that cannot be planned faster (the planner rejects lines stretched far beyond their nominal time
        # near joint limits, and the faster nominal time stretches more) is planned at its Sim2 speed instead
        try:
            return pl(goals, speed * k, min_s / k, *a, **kw)
        except Exception:  # noqa: BLE001
            if k == 1.0:
                raise
            return pl(goals, speed, min_s, *a, **kw)

    def move_joints(targets, label, v_max=V_MAX, allow=None):
        return mj(targets, label, v_max=cap(list(targets), v_max), allow=allow)

    def joint_plan(hand, q, v_max=V_MAX, start=None):
        return jp(hand, q, cap([hand], v_max), start)

    def grip(widths, label, duration=0.6):
        return gr(widths, label, min(duration, gs) if gs else duration)
    S.plan_line, S.move_joints, S.joint_plan, S.grip = plan_line, move_joints, joint_plan, grip


def roles(S):
    blk = S.bindings['targets'][0]
    bar, _ = S.pose('bottom_handle_bar')
    c, _ = S.pose(blk)
    bh = 'left' if c[0] > bar[0] else 'right'
    dh = 'right' if bh == 'left' else 'left'
    return blk, bh, dh


def grasps(S):
    blk, bh, dh = roles(S)
    return [(dh, 'bottom_handle_bar'), (bh, blk)]


def top_down(u):
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([u[0], u[1], 0.0]) / np.linalg.norm(u[:2])
    return np.c_[np.cross(y, z), y, z]


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def handle_rot(pitch_deg, sign, pull_dir):
    """Approach toward the cabinet (against pull_dir), pitched down; closing axis across the bar (bar along x)."""
    th = np.radians(pitch_deg)
    a = -np.cos(th) * pull_dir + np.array([0.0, 0.0, -np.sin(th)])
    a /= np.linalg.norm(a)
    y = sign * np.cross(a, [1.0, 0.0, 0.0])
    y /= np.linalg.norm(y)
    return np.c_[np.cross(y, a), y, a]


def insert_rot(lean_deg, sign, axis):
    """Block-hand rotation at insertion: closing along +-x (parallel to the cabinet face), approach down and tipped
    lean_deg toward the cabinet (against the opening direction `axis`)."""
    th = np.radians(lean_deg)
    a = np.array([0.0, 0.0, -np.cos(th)]) - np.sin(th) * axis
    y = np.array([sign, 0.0, 0.0])
    return np.c_[np.cross(y, a), y, a]


def approach_over(S, dh, bh, hopts, bopts, why):
    """Handle hand to its pre-grasp by a planned Cartesian route, the whole route checked before anything moves:
      side: straight back along the cabinet's side (start orientation, may turn about the vertical) to the pre-grasp's
            distance from the robot, lowered on the way by 2/3, 0.8, 0.9 or all of the way down to the pre-grasp height
            (keeping the shoulder and elbow off the camera arch), else at its start height or 1/3 down, then straight to
            the pre-grasp turning to the grasp;
      over: straight up to OVER_TOP above the cabinet top, across at that height to above the pre-grasp, straight down
            turning to the grasp;
    each ending with the joint correction onto the planned pre-grasp. Then the block hand to its pre-grasp (joint
    move). Returns (ho, bo) or None."""
    cab = S.objects['drawer_cabinet']
    top = S.pose('drawer_cabinet')[0][2] + cab['size_m'][2] / 2
    p0, R0 = S.tcp(dh)
    hside = 1.0 if dh == 'left' else -1.0
    fr = {'side': 0.0, 'side_low': 1 / 3, 'side_05': 0.5, 'side_06': 0.6, 'side_lower': 2 / 3, 'side_075': 0.75,
          'side_08': 0.8, 'side_09': 0.9, 'side_10': 1.0}
    routes = ('side_lower', 'side_08', 'side_09', 'side_10', 'side', 'side_low', 'side_05', 'side_06', 'side_075',
              'out_side_lower', 'out_side_06', 'out_side_08', 'over')
    feasible = []  # v8b: (min arm-to-arch distance over the plan, route, ho, plan) of every clear route and grasp
    # v8c: two joint moves through a waypoint beside and above the pre-grasp
    lifts = {}  # v2: the straight lift out of the start pose that may come first (planned once per height)
    for lift in VIA_LIFTS:
        if lift:
            try:
                lifts[lift] = S.plan_line({dh: (p0 + np.array([0.0, 0.0, lift]), R0)}, speed=0.10, label='approach_lift')[dh]
            except Exception as e:  # noqa: BLE001
                why.append(f'lift {lift}: ' + str(e)[:80])
    for ho in hopts[:VIA_GRASPS]:
        if any(f[0] >= ARCH_OK for f in feasible):  # v2: the first grasp with a clear waypoint route is enough
            break
        p_pre, R_pre = S.fk(dh, ho[1])
        for (k, (lat, up, back)), lift in [(kv, lf) for lf in VIA_LIFTS for kv in enumerate(VIA_OFFSETS)
                                           if lf == 0.0 or lf in lifts]:
            if any(f[0] >= ARCH_OK for f in feasible):
                break
            via = p_pre + np.array([hside * lat, back, up])  # world: +x is the robot's left, +y toward the robot
            try:
                sols = S.ik_many(dh, via, [R_pre, R0], near=ho[1], n_seeds=VIA_SEEDS, k=2)
            except Exception as e:  # noqa: BLE001
                why.append(f'via{k}: ' + str(e)[:80])
                continue
            for q_via, _ in sols:
                try:
                    q_start = lifts[lift][-1] if lift else S.q_cmd(dh)
                    parts = ([lifts[lift]] if lift else []) + [S.joint_plan(dh, q_via, start=q_start),
                                                               S.joint_plan(dh, ho[1], start=q_via)]
                    plan = np.concatenate(parts)
                    ok, info = S._path_ok({dh: plan})
                    if not ok:
                        raise RuntimeError(f'not clear: {info}')
                    turn = reversal(parts)
                    if turn < REVERSAL_COS:
                        raise RuntimeError(f'turns back at the waypoint (cos {turn:.2f})')
                except Exception as e:  # noqa: BLE001
                    why.append(f'via{k}: ' + str(e)[:80])
                    continue
                clear = arch_min(S, dh, plan)
                feasible.append((clear, f'via{k}_lift{int(lift * 100)}', ho, plan))
                S.note(f'approach_via{k}_lift{int(lift * 100)}: clear, {len(plan) / 30:.1f} s, {clear * 100:.1f} cm from the arch')
    if not feasible:
        S.note('approach via a waypoint: none clear: ' + '; '.join(why[-4:]))
    elif any(f[0] >= ARCH_OK for f in feasible):  # v2: a clear waypoint route: the Cartesian routes are not needed
        routes = ()
    for route in routes:
        for ho in hopts[:HANDLE_TRIES]:
            p_pre, _ = S.fk(dh, ho[1])
            if route.startswith(('side', 'out_side')):
                f = fr[route.replace('out_', '')]
                z1 = p0[2] + f * (p_pre[2] - p0[2])
                legs = [(np.r_[p0[0], p_pre[1], z1], R0, {dh: UP}), (p_pre, ho[2], {dh: UP})]
                if route.startswith('out_'):  # first 6 cm straight out to the hand's own side, away from the cabinet
                    o = np.r_[p0[0] + hside * 0.06, p0[1], p0[2]]
                    legs = [(o, R0, {dh: UP}), (np.r_[o[0], p_pre[1], z1], R0, {dh: UP}), (p_pre, ho[2], {dh: UP})]
            else:
                hz = max(p0[2], top + OVER_TOP)
                legs = [(np.r_[p0[:2], hz], R0, None), (np.r_[p_pre[:2], hz], R0, {dh: UP}), (p_pre, ho[2], {dh: UP})]
            try:
                q, parts = S.q_cmd(dh), []
                for pos, R, soft in legs:
                    seg = S.plan_line({dh: (pos, R)}, speed=0.10, start={dh: q}, label=f'approach_{route}', soft=soft)[dh]
                    parts.append(seg)
                    q = seg[-1]
                parts.append(S.joint_plan(dh, ho[1], start=q))
                plan = np.concatenate(parts)
                ok, info = S._path_ok({dh: plan})
                if not ok:
                    raise RuntimeError(f'not clear: {info}')
                turn = reversal(parts)
                if turn < REVERSAL_COS:  # joints turning back at a corner: similar states, opposite motions
                    raise RuntimeError(f'turns back at a corner (cos {turn:.2f})')
            except Exception as e:  # noqa: BLE001
                S.note(f'approach_{route}: ' + str(e)[:300])
                why.append(f'{route}: ' + str(e)[:100])
                continue
            clear = arch_min(S, dh, plan)
            feasible.append((clear, route, ho, plan))
            S.note(f'approach_{route}: clear, {len(plan) / 30:.1f} s, {clear * 100:.1f} cm from the arch')
    if not feasible:
        return None
    ok = [f for f in feasible if f[0] >= ARCH_OK]
    clear, route, ho, plan = min(ok, key=lambda f: len(f[3])) if ok else max(feasible, key=lambda f: f[0])
    if OPTIONS.get('overlap'):  # v2r: the block hand's approach during the handle hand's route
        for bo in bopts[:4]:
            try:
                S.together({dh: plan, bh: S.joint_plan(bh, bo[1])}, f'approach_{route}')
            except Exception as e:  # noqa: BLE001
                S.note('approach overlap: ' + str(e)[:200])
                continue
            S.note(f'handle hand approached by the {route} route ({clear * 100:.1f} cm from the arch), block hand with it')
            return ho, bo
    S.together({dh: plan}, f'approach_{route}')
    for bo in bopts[:4]:
        try:
            S.move_joints({bh: bo[1]}, 'approach')
        except Exception as e:  # noqa: BLE001
            S.note('approach block hand: ' + str(e)[:300])
            why.append('block hand: ' + str(e)[:100])
            continue
        S.note(f'handle hand approached by the {route} route ({clear * 100:.1f} cm from the arch)')
        return ho, bo
    return None


def reversal(parts, k=10):
    """Smallest cosine between the joint motion over the last k rows of a segment and the first k of the next."""
    worst = 1.0
    for u_seg, w_seg in zip(parts[:-1], parts[1:]):
        if len(u_seg) < 2 or len(w_seg) < 2:
            continue
        u = u_seg[-1] - u_seg[-min(k, len(u_seg))]
        w = w_seg[min(k, len(w_seg)) - 1] - w_seg[0]
        if np.linalg.norm(u) > 1e-3 and np.linalg.norm(w) > 1e-3:
            worst = min(worst, float(u @ w / np.linalg.norm(u) / np.linalg.norm(w)))
    return worst


def arch_min(S, hand, plan):
    """Smallest distance of the hand's arm to the camera arch over a joint plan (every 3rd row and the last)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
    import arch_away
    A = getattr(S, 'arch_away_all', {}).get(hand) or arch_away.ArchAway(S, hand)
    rows = list(plan[::3]) + [plan[-1]]
    return min(A.distance(q)[0] for q in rows)


def run(S):
    install_refine(S)
    blk, bh, dh = roles(S)
    install_guard(S, bh)
    S.allow = {bh: {blk}, dh: {'bottom_handle_bar', 'bottom_handle_post_a', 'bottom_handle_post_b', 'bottom_drawer'}}
    dr = S.objects['bottom_drawer']
    axis = np.asarray(dr['joint']['axis'], float)  # opening direction (toward the robot)
    slide_range = dr['joint']['range'][1]
    joint = dr['joint']['name']

    pull = PULL_FRACTION * slide_range - S.slide(joint)
    # S1 cabinet revision: open first, so that every plan below is checked with the fingers as they will be
    S.grip({dh: OPEN_BAR, bh: OPEN_BLOCK}, 'preopen', 0.4)
    # handle candidates: (cost, q_pre, R, reach plan end) with reach and pull planned
    bar, _ = S.pose('bottom_handle_bar')
    g_bar = bar + BAR_SHORT * axis
    hopts, why = [], []
    # planned backwards from the hardest pose: IK at the pulled-open pose, the straight pull back to the bar and the
    # reach back to the pre-grasp are planned from there and executed reversed (one IK branch for the whole hold)
    for pitch in BAR_PITCHES:
        # turned grasps only with finger order +1: with order -1 the hand twisted on the bar near the end of the pull and
        # jammed at ~70 % (rounds 5-7, all four right-side cabinets; with +1 all four were admitted)
        # factory revision: turned grasps with either finger order (with the gripper not rotated on the flange, the
        # wrist reaches only some of the +1 ones); admission still catches a hand that twists and jams on the bar
        for sign, yaw_deg in [(sg, yw) for yw in BAR_YAWS for sg in (1.0, -1.0) if yw == 0 or sg > 0 or BOTH_ORDERS]:
            R0 = rot_z(np.radians(yaw_deg)) @ handle_rot(pitch, sign, axis)
            try:
                ends = S.ik_many(dh, g_bar + pull * axis, [R0], k=4)
            except Exception as e:  # noqa: BLE001
                why.append(str(e)[:100])
                continue
            for q_end, R in ends:
                try:
                    try:  # a fixed orientation first: a hand turning during the pull can twist on the bar and jam
                        back = S.plan_line({dh: (g_bar, R)}, speed=0.06, start={dh: q_end}, label='pull')
                    except Exception:  # noqa: BLE001
                        back = S.plan_line({dh: (g_bar, R)}, speed=0.06, start={dh: q_end}, label='pull',
                                           soft={dh: UP})
                    pull_plan = back[dh][::-1]
                    back = S.plan_line({dh: (g_bar - PRE * R[:, 2], R)}, speed=0.06, start={dh: pull_plan[0]},
                                       label='reach', soft={dh: UP})
                    reach_plan = back[dh][::-1]
                except Exception as e:  # noqa: BLE001
                    why.append(str(e)[:100])
                    continue
                q_pre = reach_plan[0]
                hopts.append((float(np.abs(q_pre - S.q_cmd(dh)).max()) + 0.5 * (pitch > 0) + 0.01 * abs(yaw_deg),
                              q_pre, R, reach_plan, pull_plan, (pitch, yaw_deg, sign)))
        # level grasps (turned or not) first: ranking 15 deg pitched grasps ahead of turned ones (round 6) pushed
        # joints past their limits and blocked the block hand's way out (14/20 -> 4/20)
        if hopts:
            break
    if not hopts:
        raise RuntimeError(f'handle: no grasp with a clear reach and pull: {why[:3]}')
    # factory revision: rank by the arm's distance to the camera arch over the whole hold (reach and pull; the push is the
    # pull reversed), candidates at least ARCH_GOOD away first, then by joint distance as before
    clear = [min(arch_min(S, dh, h[3]), arch_min(S, dh, h[4])) for h in hopts]
    order = sorted(range(len(hopts)), key=lambda i: (-min(clear[i], ARCH_GOOD), hopts[i][0]))
    hopts = [hopts[i] for i in order]
    S.note('handle grasps by arch distance (cm): ' + ', '.join(f'{clear[i] * 100:.1f}' for i in order[:HANDLE_TRIES]))
    # block candidates: R_pick = Rz(phi) R_ins with the closing axis on a face normal of the block
    c, Rc = S.pose(blk)
    yaw = np.arctan2(Rc[1, 0], Rc[0, 0])
    g = c + [0, 0, 0.005]
    bopts = []
    for lean in LEANS:
        for sign in (1.0, -1.0):
            R_ins = insert_rot(lean, sign, axis)
            for k in range(4):
                phi = (yaw + k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
                R_pick = rot_z(phi) @ R_ins
                a_pick = R_pick[:, 2]
                try:
                    sols = S.ik_many(bh, g - PRE * a_pick, [R_pick], k=2)
                except Exception as e:  # noqa: BLE001
                    why.append(str(e)[:100])
                    continue
                for q_pre, _ in sols:
                    try:
                        rp = S.plan_line({bh: (g, R_pick)}, speed=0.06, start={bh: q_pre}, label='reach')
                        S.plan_line({bh: (g - LIFT * a_pick, R_pick)}, speed=0.08, start={bh: rp[bh][-1]},
                                    label='lift')
                    except Exception as e:  # noqa: BLE001
                        why.append(str(e)[:100])
                        continue
                    bopts.append((float(np.abs(q_pre - S.q_cmd(bh)).max()) + 0.3 * abs(phi), q_pre, R_pick, phi))
        if bopts:
            break
    if not bopts:
        raise RuntimeError(f'block: no leaned grasp with a clear reach and lift: {why[-3:]}')
    bopts.sort(key=lambda x: x[0])

    chosen = None
    for ho in hopts[:HANDLE_TRIES]:
        for bo in bopts[:4]:
            try:
                S.move_joints({dh: ho[1], bh: bo[1]}, 'approach')
            except Exception as e:  # noqa: BLE001
                why.append(str(e)[:100])
                if len(why) < 400 and sum('direct approach' in n['note'] for n in S.notes) < 6:
                    S.note('direct approach: ' + str(e)[:200])
                continue
            chosen = (ho, bo)
            break
        if chosen:
            break
    if chosen is None:
        chosen = approach_over(S, dh, bh, hopts, bopts, why)
    if chosen is None:
        raise RuntimeError(f'approach: no clear joint path for any grasp pair: {why[-3:]}')
    (_, _, Rh, reach_plan, pull_plan, hk), (_, _, Rb, phi) = chosen
    S.note(f'handle grasp: pitch {hk[0]:.0f} deg, turned {hk[1]:.0f} deg about the vertical, finger order {hk[2]:+.0f}')
    reach = {dh: reach_plan, **S.plan_line({bh: (g, Rb)}, speed=0.06, label='reach')}
    ok, info = S._path_ok(reach)
    if ok:
        S.run_plans([reach], 'reach')
    else:
        S.note(f'reach: hands one after the other ({info})')
        S.run_plans([{dh: reach_plan}, {bh: reach[bh]}], 'reach')
    S.grip({dh: 0.0, bh: 0.0}, 'close', 0.6)
    S.line({bh: (g - LIFT * Rb[:, 2], Rb)}, 'lift', speed=0.08)
    S.attach(bh, blk)  # plan checks carry the block with the hand from now on
    # the block hand waits out of the pulling arm's way, over its own side
    side = 1.0 if bh == 'left' else -1.0
    pw, Rw = S.tcp(bh)
    try:
        p0m, R0m = S.place.mount_pose()
        p_wait = p0m + R0m @ np.array([WAIT_MOUNT[0], side * WAIT_MOUNT[1], 0.0])
        p_wait[2] = 0.75 + WAIT_MOUNT[2]
        q_wait, _ = S.ik(bh, p_wait, [Rw])
        A = getattr(S, 'arch_away_all', {}).get(bh)
        if A is not None:  # factory revision: the waiting pose with the elbow pushed off the camera arch
            q_try = A.relax(q_wait, p_wait, S.fk(bh, q_wait)[1], 80)
            if S.ok(*S.check({bh: q_try, dh: S.q_cmd(dh)})):
                q_wait = q_try
        pulled = False
        if OPTIONS.get('overlap'):  # v2r: to the waiting pose during the pull
            try:
                S.together({bh: S.joint_plan(bh, q_wait), dh: pull_plan}, 'pull')
                pulled = True
            except Exception as e:  # noqa: BLE001
                S.note('pull overlap: ' + str(e)[:200])
        if not pulled:
            S.move_joints({bh: q_wait}, 'block_wait')
    except Exception as e:  # noqa: BLE001
        S.note(f'block hand stays where it lifted ({str(e)[:100]})')
        pulled = False

    # pull open; pull again if the drawer lags
    if not pulled:
        S.run_plans([{dh: pull_plan}], 'pull')

    def repull(label):
        s_now = S.slide(joint)
        if s_now >= PULL_MIN * slide_range:
            return
        p_h, R_h = S.tcp(dh)
        S.note(f'{label}: drawer at {s_now / slide_range:.0%}: pulling again')
        try:
            S.line({dh: (p_h + (PULL_FRACTION * slide_range - s_now) * axis, R_h)}, 'pull_again', speed=0.04,
                   soft={dh: UP})
        except Exception as e:  # noqa: BLE001
            S.note(f'{label}: pulling again not possible ({str(e)[:100]})')

    for _ in range(2):
        s_now = S.slide(joint)
        if s_now >= PULL_MIN * slide_range:
            break
        p_h, R_h = S.tcp(dh)
        S.note(f'drawer at {s_now / slide_range:.0%} after the pull: pulling again')
        S.line({dh: (p_h + (PULL_FRACTION * slide_range - s_now) * axis, R_h)}, 'pull_again', speed=0.04,
               soft={dh: UP})

    # block into the opened drawer
    d_pos, _ = S.pose('bottom_drawer')
    mf = S.objects['middle_drawer_front']
    mf_pos, _ = S.pose('middle_drawer_front')
    front_y = mf_pos[1] + np.sign(axis[1]) * mf['size_m'][1] / 2  # face of the cabinet (toward the robot)
    half_d = dr['size_m'][1] / 2 - dr['wall_m']
    lo = front_y + np.sign(axis[1]) * 0.035
    hi = d_pos[1] + np.sign(axis[1]) * (half_d - 0.035)
    side = 1.0 if bh == 'left' else -1.0
    floor_z = d_pos[2] - dr['size_m'][2] / 2 + dr['wall_m']
    half = S.objects[blk]['size_m'][2] / 2
    # S1 cabinet revision: only offsets that keep the block INSERT_WALL_GAP inside the drawer's side walls
    max_dx = dr['size_m'][0] / 2 - dr['wall_m'] - S.objects[blk]['size_m'][0] / 2 * np.sqrt(2) - INSERT_WALL_GAP
    sides = [dx for dx in INSERT_SIDES if dx <= max_dx] + ([round(max_dx, 3)] if max_dx < max(INSERT_SIDES) else [])
    rim_z = d_pos[2] + dr['size_m'][2] / 2
    # the block's centre at release; first a little toward the robot: with the hand tipped toward the cabinet its fingers near the handle above
    targets = [np.array([d_pos[0] + side * dx, (lo + hi) / 2 + np.sign(axis[1]) * dy, rim_z + DROP_CLEAR + half]) for dy in Y_SHIFTS for dx in sides]
    p_tcp, R_tcp = S.tcp(bh)
    cb, _ = S.pose(blk)
    off = cb - p_tcp
    sy = np.sign(axis[1])
    done, why = False, []
    for target, deg, stage_d in [(t_, d_, sd) for sd in STAGE_DISTS for t_ in targets for d_ in (0, 20, -20)]:
        if done:
            break
        Rz = rot_z(np.radians(deg)) @ rot_z(-phi)
        R = Rz @ R_tcp
        goal = target - Rz @ off
        above = np.r_[goal[:2], goal[2] + LOWER]
        # staging pose in free space 20 cm in front of the cabinet face, at the height of the pose above the drawer
        stage = np.r_[above[0], front_y + sy * stage_d - (Rz @ off)[1], max(above[2], p_tcp[2])]
        try:
            sols = S.ik_many(bh, stage, [R], k=4)
        except Exception as e:  # noqa: BLE001
            why.append(f'{deg}: {e}'[:160])
            continue
        for q_stage, _ in sols:
            try:
                p1 = S.plan_line({bh: (above, R)}, speed=0.10, start={bh: q_stage}, label='insert_transport',
                                 soft={bh: UP})
                R2 = S.fk(bh, p1[bh][-1])[1]  # keep the orientation the line in ended with (no turning back)
                p2 = S.plan_line({bh: (goal, R2)}, speed=0.07, start={bh: p1[bh][-1]}, label='insert_lower',
                                 soft={bh: UP})
                S.move_joints({bh: q_stage}, 'insert_transport')
            except Exception as e:  # noqa: BLE001
                why.append(f'{deg}: {e}'[:160])
                continue
            repull('before the block goes in')
            # in and down without a pause between: a pause at the corner leaves similar states with opposite joint
            # motions on both sides of it (state-action conflict check)
            S.run_plans([p1], 'insert_transport')
            S.run_plans([p2], 'insert_lower')
            done = True
            break
    if not done:
        # site_v6 (2026-11-05): in from the side instead: straight up until the block clears the drawer rim, straight
        # across (sideways) to above the insertion point, down to above it and on down as before; planned in full first
        p_now, _ = S.tcp(bh)
        A = getattr(S, 'arch_away_all', {}).get(bh)
        if A is not None:  # factory revision: on this way in the block hand's elbow is pushed off the camera arch
            A.line_mode, A.iters_line = 'push', SIDE_PUSH_ITERS
        cab = S.objects['drawer_cabinet']
        cab_c, _ = S.pose('drawer_cabinet')
        for target in targets:
            for deg in SIDE_TURNS:
                Rz = rot_z(np.radians(deg)) @ rot_z(-phi)
                R = Rz @ R_tcp
                goal = target - Rz @ off
                above = np.r_[goal[:2], goal[2] + LOWER]
                z_top = max(above[2] + 0.02, p_now[2])
                # factory revision: before going across, out of the cabinet's footprint to the opened drawer's depth (the
                # waiting pose is beside the cabinet, deeper than its front: straight across, the block hit its side)
                # first in to just outside the cabinet's side wall (the elbow stays in, off the camera arch), then forward
                x_wall = cab_c[0] + side * (cab['size_m'][0] / 2 + SIDE_GAP)
                x_in = x_wall if side * p_now[0] > side * x_wall else p_now[0]
                # across ACROSS_OFFSET off the cabinet face (turned a quarter, the gripper's coupler passed within 1 cm
                # of the drawer fronts above), then down and back onto the point above the drawer
                y_x = above[1] + sy * ACROSS_OFFSET
                # the forward leg (index 2) is a joint move to its end pose with the elbow pushed off the camera arch
                # (along a straight line the posture drifted toward the arch faster than arch_away could push it)
                legs = [(np.r_[p_now[:2], z_top], R_tcp, 0.10), (np.r_[x_in, p_now[1], z_top], R_tcp, 0.10),
                        (np.r_[x_in, y_x, z_top], R_tcp, None), (np.r_[above[0], y_x, z_top], R, 0.10), (above, R, 0.08)]
                try:
                    q, plans, last = S.q_cmd(bh), [], p_now
                    for pos, RR, spd in legs:
                        if np.linalg.norm(pos - last) < 0.005:
                            continue
                        if spd is None:
                            qj, pe, re = S.robot.ik_local(bh, pos, RR, q, S.d.qpos, q, iters=60)
                            if pe > 1e-3 or re > 1.0:
                                raise RuntimeError(f'forward leg: no IK ({pe * 1000:.1f} mm, {re:.1f} deg)')
                            if A is not None:
                                qj = A.relax(qj, pos, RR, 80)
                            seg = {bh: S.joint_plan(bh, qj, start=q)}
                            ok, info = S._path_ok(seg)
                            if not ok:
                                raise RuntimeError(f'forward leg not clear: {info}')
                            plans.append(seg)
                            q, last = qj, pos
                            continue
                        seg = S.plan_line({bh: (pos, RR)}, speed=spd, start={bh: q}, label='insert_transport',
                                          soft={bh: UP})
                        plans.append(seg)
                        q, last = seg[bh][-1], pos
                    R2 = S.fk(bh, q)[1]
                    p2 = S.plan_line({bh: (goal, R2)}, speed=0.07, start={bh: q}, label='insert_lower', soft={bh: UP})
                except Exception as e:  # noqa: BLE001
                    why.append(f'side {deg}: {e}'[:160])
                    S.note(f'insert from the side at {np.round(target, 3).tolist()}, {deg} deg: {str(e)[:220]}')
                    continue
                repull('before the block goes in')
                S.run_plans(plans, 'insert_transport')
                S.run_plans([p2], 'insert_lower')
                S.note(f'block in from the side ({deg} deg)')
                done = True
                break
            if done:
                break
    if not done:
        raise RuntimeError(f'insert: no clear path into the drawer at {np.round(targets[0], 3).tolist()} or the other '
                           f'insertion points: {why[:3]} ... {why[-3:]}')
    S.detach(bh)
    S.grip({bh: OPEN_BLOCK}, 'insert_release', 0.5)
    # out of the drawer: straight up above the rim (site_v6 revision 2026-11-05, user: no move back toward the robot
    # after the task, where the camera arch is), else back along the leaned approach as before
    p, R = S.tcp(bh)
    a_ins = R[:, 2]
    out_d = max(0.10, (rim_z + 0.10 - p[2]) / max(1e-3, -a_ins[2])) + 0.06  # +6 cm: far enough that the move aside
    # cannot bring the arm back near a state of the retreat (state-action conflict check)
    done = False
    for goal in (np.r_[p[:2], max(p[2] + 0.14, rim_z + 0.16)], np.r_[p[:2], max(p[2] + 0.08, rim_z + 0.12)],
                 p - out_d * a_ins, p - 0.06 * a_ins):
        try:
            S.line({bh: (goal, R)}, 'block_retreat', speed=0.12, soft={bh: UP})
            done = True
            break
        except Exception as e:  # noqa: BLE001
            S.note(f'block_retreat to {np.round(goal, 3).tolist()} not possible ({str(e)[:80]})')
    if not done:
        raise RuntimeError('block_retreat: no way out of the drawer')
    S.allow[bh] = set()
    p_out, R = S.tcp(bh)
    # aside: straight out to the hand's own side at the distance in front of the robot it is at now, no lower
    aside_p = np.array([side * max(0.40, side * p_out[0]), p_out[1], max(p_out[2], 0.75 + 0.30)])
    pushed = False
    if OPTIONS.get('overlap'):  # v2r: the block hand moves aside during the push
        try:
            p_h, R_h = S.tcp(dh)
            s_now = S.slide(dr['joint']['name'])
            S.allow[dh] |= {blk, 'bottom_drawer'}  # the drawer and the block in it move with the pushing hand
            pa = S.plan_line({bh: (aside_p, R)}, speed=0.12, label='block_aside', soft={bh: UP})[bh]
            pp = S.plan_line({dh: (p_h - (s_now - 0.003) * axis, R_h)}, speed=0.06, label='push', soft={dh: UP})[dh]
            S.together({bh: pa, dh: pp}, 'push')
            pushed = True
        except Exception as e:  # noqa: BLE001
            S.note('push overlap: ' + str(e)[:200])
    try:
        if pushed:
            raise StopIteration
        S.line({bh: (aside_p, R)}, 'block_aside', speed=0.12, soft={bh: UP})
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            q, _ = S.ik(bh, aside_p, [R, top_down(np.array([1.0, 0, 0]))])
            S.move_joints({bh: q}, 'block_aside')
        except Exception as e2:  # noqa: BLE001
            S.note(f'block hand stays above the drawer ({str(e)[:60]}; {str(e2)[:60]})')

    # push closed, release the handle
    p_h, R_h = S.tcp(dh)
    s_now = S.slide(dr['joint']['name'])
    S.allow[dh] |= {blk, 'bottom_drawer'}  # the drawer and the block in it move with the pushing hand
    if not pushed:
        S.line({dh: (p_h - (s_now - 0.003) * axis, R_h)}, 'push', speed=0.06, soft={dh: UP})
    for _ in range(2):  # the bar may have slipped along the pads: push again by what is still open
        s_now = S.slide(dr['joint']['name'])
        if s_now <= 0.05 * slide_range:
            break
        S.note(f'drawer still {s_now / slide_range:.0%} open after the push: pushing again')
        p_h, R_h = S.tcp(dh)
        try:
            S.line({dh: (p_h - (s_now - 0.003) * axis, R_h)}, 'push_again', speed=0.04, soft={dh: UP})
        except Exception as e:  # noqa: BLE001
            S.note(f'pushing again not possible ({str(e)[:100]})')
            break
    S.hold(0.3, 'push_hold')
    S.grip({dh: OPEN_BAR}, 'handle_release', 0.5)
    # site_v6 revision (2026-11-05, user: after the task move sideways, not back toward the robot): back only the 4 cm
    # the fingers need to clear the bar, then straight out to the hand's own side
    p_h, R_h = S.tcp(dh)
    S.line({dh: (p_h + 0.04 * axis, R_h)}, 'handle_clear', speed=0.08)
    hside = 1.0 if dh == 'left' else -1.0
    p_h, R_h = S.tcp(dh)
    if OPTIONS.get('short_exit'):
        return
    try:
        S.line({dh: (np.array([hside * max(0.40, hside * p_h[0] + 0.12), p_h[1], p_h[2] + 0.03]), R_h)}, 'handle_aside',
               speed=0.10, soft={dh: UP})
    except Exception as e:  # noqa: BLE001
        S.note(f'handle_aside not possible ({str(e)[:100]}); the hand stays clear of the bar')
