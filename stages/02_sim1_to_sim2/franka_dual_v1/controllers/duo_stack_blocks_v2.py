"""Stack blocks on the FR3 Duo with Robotiq 2F-85, v2 (written by Claude Code, claude-opus-5-5, 2026-11-06): the
stack-cups controller's structure and refine rules (duo_stack_cups.py, the refine stage of sim2-refine/tasks/
fr3_stack_cups/refine.py) with the block controller's grasp (duo_stack_blocks.py): top-down, fingers across two
opposite faces, pinch point 5 mm above the block centre; the closest feasible face pair and finger order.

Contract order: both blocks lifted by their bound hands -> base block inside the stack region -> upper block resting
on the base block -> both stable and released. Every target is computed from poses measured when the stage starts.

Stages:
  1. both hands to a pre-grasp `pre` above their block (top-down), straight down to the grasp, close, lift 10 cm;
  2. upper hand: to its waiting pose out to its own side ('outward'), or straight to a staging point beside and above the
     stack region ('stage', the refine stage's no-detour rule; with the 10 cm arm gap the first of STAGE_SIDES that
     leaves the base carry clear), or it keeps its block at the lift pose ('hold');
  3. base hand: carries the block over the region centre (the hand may turn about the vertical), lowers it until its
     bottom is 3 mm above the table, opens, rises;
  4. upper hand: carries its block over the measured base block, lowers it until its bottom is 3 mm above the base top,
     opens, rises (and moves aside with final_aside).
OPTIONS (the refine stage's rules): upper_wait, overlap (the upper hand's staging during the base carry, the base hand's
move aside during the upper carry), final_aside, speed (lines and joints, joints capped at 0.72 rad/s, 0.5 rad/s with a
block in the hand), grip_s, preopen_in_approach, pre. The defaults are the Sim2 program (duo_stack_blocks.py's stages).
"""
import mujoco
import numpy as np

HANDS = ('left', 'right')
PINCH_ABOVE_CENTRE = 0.005
OPEN_BLOCK = 0.075  # opening before the descent: the 5 cm block with 12 mm on each side
V_HELD = 0.35  # rad/s for joint-space moves with a cup in the hand (faster moves swing it in the fingers)
LIFT = 0.10
OPEN = 0.075
GAP = 0.003  # a placed block's bottom this far above its support at release
# fixed waypoints in the robot's mount frame (forward, |left| from the mounting point; height above the table), so the
# controller does not depend on where the robot stands in the world (v10)
WAIT = (0.40, 0.38, 0.25)  # waiting TCP of the upper hand (site_v6: [1:] only, straight out sideways)
ASIDE = (0.40, 0.40, 0.30)  # where a hand goes after releasing its cup (site_v6: [1:] only, straight out sideways)
STAGE_SIDE, STAGE_ABOVE_RIM = 0.15, 0.10  # staging cup centre: beside the region centre toward its hand, bottom above the base rim
STAGE_SIDES = (0.15, 0.20, 0.25, 0.30)  # arm-gap revision: staging distances tried in order (the first is STAGE_SIDE)
TABLE_Z = 0.75
V_JOINT, V_JOINT_CAP, V_HELD_CAP = 0.6, 0.72, 0.5

OPTIONS = dict(upper_wait='outward', overlap=False, final_aside=True, speed=1.0, grip_s=(0.5, 0.6, 0.5),
               preopen_in_approach=False, pre=0.10)


def opt(k):
    return OPTIONS[k]


def sp(v):
    """A line speed scaled by the refine rule."""
    return v * opt('speed')


def vj(held=False):
    return min(V_HELD * opt('speed'), V_HELD_CAP) if held else min(V_JOINT * opt('speed'), V_JOINT_CAP)


def grasps(S):
    b = S.bindings
    base, upper = b['targets']
    h = b['stack_hands']
    return [(h['base'], base), (h['upper'], upper)]


def top_down(u):
    """TCP rotation: approach straight down, closing along the horizontal unit u."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([u[0], u[1], 0.0]) / np.linalg.norm(u[:2])
    return np.c_[np.cross(y, z), y, z]


def half_h(S, obj):
    return S.objects[obj]['size_m'][2] / 2


def face_grasps(S, hand, cube):
    """Feasible grasps [(cost, pre-grasp pos, grasp pos, R, q_pre)]: top-down, fingers across one of the cube's two
    face pairs (both finger orders), pinch point PINCH_ABOVE_CENTRE above its centre, cheapest (joint motion) first."""
    c, Rc = S.pose(cube)
    g = c + [0, 0, PINCH_ABOVE_CENTRE]
    yaw = np.arctan2(Rc[1, 0], Rc[0, 0])
    S.allow[hand] = {cube}
    pre = opt('pre')
    out = []
    for k in range(4):
        u = np.array([np.cos(yaw + k * np.pi / 2), np.sin(yaw + k * np.pi / 2), 0.0])
        try:
            q_pre, R = S.ik(hand, g + [0, 0, pre], [top_down(u)])
        except Exception:  # noqa: BLE001
            continue
        out.append((float(np.abs(q_pre - S.q_cmd(hand)).max()), g + [0, 0, pre], g, R, q_pre))
    if not out:
        raise RuntimeError(f'{hand}: no feasible face grasp of {cube}')
    return sorted(out, key=lambda x: x[0])


def approach_and_descend(S, cup_of):  # cup_of: hand -> its block
    """Try grasp pairs (cheapest total first): the joint-space approach of both hands, the straight descent and the
    lift are planned before anything moves; the first pair whose whole path is clear is executed."""
    cands = {h: face_grasps(S, h, cup_of[h])[:4] for h in HANDS}
    pairs = sorted(((a, b) for a in cands['left'] for b in cands['right']), key=lambda ab: ab[0][0] + ab[1][0])
    why = []
    for gl, gr in pairs:
        pick = {'left': gl, 'right': gr}
        try:
            dn = S.plan_line({h: (pick[h][2], pick[h][3]) for h in HANDS}, speed=sp(0.06),
                             start={h: pick[h][4] for h in HANDS}, label='descend')
            S.plan_line({h: (pick[h][2] + [0, 0, LIFT], pick[h][3]) for h in HANDS}, speed=sp(0.08),
                        start={h: dn[h][-1] for h in HANDS}, label='lift')
        except Exception as e:  # noqa: BLE001
            why.append(str(e)[:100])
            continue
        try:
            if opt('preopen_in_approach'):
                S.together({h: S.joint_plan(h, pick[h][4], vj()) for h in HANDS}, 'approach',
                           grips={h: OPEN_BLOCK for h in HANDS})
            else:
                S.move_joints({h: pick[h][4] for h in HANDS}, 'approach', v_max=vj())
        except Exception as e:  # noqa: BLE001
            why.append(str(e)[:100])
            continue
        S.line({h: (pick[h][2], pick[h][3]) for h in HANDS}, 'descend', speed=sp(0.06))
        return pick
    raise RuntimeError(f'no clear approach and descent: {why[:3]}')


def upright(z_axis):
    """Smallest rotation taking the unit vector z_axis to the world vertical."""
    z = np.asarray(z_axis, float) / np.linalg.norm(z_axis)
    v = np.cross(z, [0.0, 0.0, 1.0])
    s_, c_ = np.linalg.norm(v), z[2]
    if s_ < 1e-9:
        return np.eye(3)
    k = v / s_
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s_ * K + (1 - c_) * K @ K


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def plan_carry(S, hand, cup, target_center, label, assume=None):
    """Plans that bring the held cup's centre over target_center: the hand may turn about the vertical (0, +-30, ...,
    +-90 deg); for each turn, in order of joint motion, the straight carry at the current height (or the target's
    approach height, when higher) and the descent to 5 cm above the target are planned (collision-checked), and so is
    the final 5 cm down. Returns (carry plans [{hand: (n, 7)}], (goal, R, final plan)) for the first clear turn, or
    (None, (goal, R, q_above)) for a joint-space move above the target when no straight carry is clear. assume: joints
    of the other arm to plan against (where it will be: the refine stage's overlap; default its commanded joints)."""
    assume = assume or {}
    p_tcp, R_tcp = S.tcp(hand)
    c, _ = S.pose(cup)
    off = c - p_tcp
    cands = []
    for deg in (0, 30, -30, 60, -60, 90, -90):
        Rz = rot_z(np.radians(deg))
        goal = np.asarray(target_center) - Rz @ off
        above = goal + [0, 0, 0.05]
        try:
            q, R = S.ik(hand, above, [Rz @ R_tcp], other=assume or None)
        except Exception:  # noqa: BLE001
            continue
        cands.append((float(np.abs(q - S.q_cmd(hand)).max()), q, R, goal, above))
    if not cands:
        raise RuntimeError(f'{label}: no reachable pose above the target')
    cands.sort(key=lambda x: x[0])
    why = []
    for _, q_above, R, goal, above in cands:
        level = np.r_[above[:2], max(above[2], p_tcp[2])]
        try:
            p1 = S.plan_line({hand: (level, R)}, speed=sp(0.12), start=dict(assume), label=f'{label}_transport')
            plans = [p1]
            if level[2] > above[2] + 1e-6:
                plans.append(S.plan_line({hand: (above, R)}, speed=sp(0.08), start={**assume, hand: p1[hand][-1]},
                                         label=f'{label}_transport'))
            p3 = S.plan_line({hand: (goal, R)}, speed=sp(0.05), start={**assume, hand: plans[-1][hand][-1]},
                             label=f'{label}_lower')
        except Exception as e:  # noqa: BLE001
            why.append(str(e)[:120])
            continue
        return plans, (goal, R, p3)
    S.note(f'{label}: no straight carry clear ({why[:2]}); joint-space move')
    _, q_above, R, goal, above = cands[0]
    return None, (goal, R, q_above)


def lower_corrected(S, hand, cup, target_center, R, p3, label):
    """The last 5 cm down: the held cup is measured again; the hand turns it back upright and the goal is corrected for
    slip during the carry (the planned descent if the corrected one is not possible)."""
    c_now, Rc = S.pose(cup)
    Rcorr = upright(Rc[:, 2])  # turns the measured cup axis back to vertical (the cup is rigid in the hand)
    goal2 = np.asarray(target_center) - Rcorr @ (c_now - S.tcp(hand)[0])  # object relative to the measured hand
    try:
        S.line({hand: (goal2, Rcorr @ R)}, f'{label}_lower', speed=sp(0.05))
    except Exception as e:  # noqa: BLE001
        S.note(f'{label}: corrected descent not possible ({str(e)[:80]}); planned descent')
        S.run_plans([p3], f'{label}_lower')


def carry_and_lower(S, hand, cup, target_center, label, other=None):
    """Move the held cup so its centre ends at target_center. other: (hand, joint plan) of the other arm, executed at
    the same time as the carry when the two arms' path is clear together (the refine stage's overlap), else after it."""
    plans, (goal, R, last) = plan_carry(S, hand, cup, target_center, label,
                                       assume={other[0]: other[1][-1]} if other is not None else None)
    if plans is None:
        if other is not None:
            S.together({other[0]: other[1]}, other[2])
        S.move_joints({hand: last}, f'{label}_transport', v_max=vj(held=True))
        S.line({hand: (goal, R)}, f'{label}_lower', speed=sp(0.05))
        return
    if other is not None:
        seq = np.concatenate([p[hand] for p in plans])
        try:
            S.together({hand: seq, other[0]: other[1]}, f'{label}_transport')
        except Exception as e:  # noqa: BLE001
            # one after the other, the other arm first (its plan ends where the carry was planned against)
            S.note(f'{label}: not clear together with {other[2]} ({str(e)[:80]}); one after the other')
            S.together({other[0]: other[1]}, other[2])
            S.run_plans(plans, f'{label}_transport')
    else:
        S.run_plans(plans, f'{label}_transport')
    lower_corrected(S, hand, cup, target_center, R, last, label)


def mount_point(S, fwd, left, above_table):
    p0, R0 = S.place.mount_pose()
    p = np.round(p0 + R0 @ np.array([fwd, left, 0.0]), 12)
    return np.array([p[0], p[1], TABLE_Z + above_table])


def plan_straight_or_joint(S, hand, pos, label, rots, v_max=None, speed=0.10):
    """Joint plan of a straight TCP line to pos keeping the hand's orientation (it may turn about the vertical: cups
    are symmetric about their axis); if that line is not reachable or not clear, a joint-space plan to a collision-free
    IK solution with any of `rots`. Nothing is executed."""
    _, R = S.tcp(hand)
    try:
        return S.plan_line({hand: (pos, R)}, speed=speed, label=label, soft={hand: np.array([0.0, 0.0, 1.0])})[hand]
    except Exception as e:  # noqa: BLE001
        S.note(f'{label}: straight move not possible ({str(e)[:80]}); joint-space move')
    q, _ = S.ik(hand, pos, [R] + list(rots))
    return S.joint_plan(hand, q, v_max or vj())


def straight_or_joint(S, hand, pos, label, rots, v_max=None, speed=0.10):
    S.together({hand: plan_straight_or_joint(S, hand, pos, label, rots, v_max, speed)}, label)


def retreat(S, hand):
    """Rise 8 cm (5 cm if 8 is not reachable; may still touch the released object)."""
    p, R = S.tcp(hand)
    try:
        S.line({hand: (p + [0, 0, 0.08], R)}, f'{hand}_retreat', speed=sp(0.10))
    except Exception as e:  # noqa: BLE001
        S.note(f'{hand}_retreat: 8 cm not reachable ({str(e)[:80]}): 5 cm')
        S.line({hand: (p + [0, 0, 0.05], R)}, f'{hand}_retreat', speed=sp(0.10))


def sideways(S, hand, left, above_table):
    """A point straight out to the hand's own side of where its TCP is now: the same distance in front of the robot
    (site_v6 revision 2026-11-05, user: after its task a hand moves sideways, never back toward the robot, where the
    camera arch is), at least `left` from the robot's midline, `above_table` above the table."""
    side = 1.0 if hand == 'left' else -1.0
    p0, R0 = S.place.mount_pose()
    p, _ = S.tcp(hand)
    fwd, lat, _ = R0.T @ (p - p0)
    return mount_point(S, fwd, side * max(left, side * lat), above_table)


def aside_plan(S, hand):
    """Joint plan from the commanded joints to a point out to the hand's own side, 0.30 m above the table."""
    return plan_straight_or_joint(S, hand, sideways(S, hand, ASIDE[1], ASIDE[2]), f'{hand}_aside',
                                  [top_down(np.array([1.0, 0, 0]))], speed=sp(0.10))


def aside(S, hand):
    """Retreat, then to a point over the hand's own side, touching nothing."""
    retreat(S, hand)
    S.allow[hand] = set()  # clear of the released object now: nothing may be touched on the way aside
    S.together({hand: aside_plan(S, hand)}, f'{hand}_aside')


def wait_outward(S, hand):
    straight_or_joint(S, hand, sideways(S, hand, WAIT[1], WAIT[2]), f'{hand}_wait', [],
                      v_max=vj(held=True), speed=sp(0.08))


def stage_plan(S, hand, cup, region_xy, base_top, side_dist=STAGE_SIDE):
    """Joint plan carrying the held cup straight to the staging point: its centre STAGE_SIDE beside the region centre
    toward the hand's own side (along the mount's left axis) with its bottom STAGE_ABOVE_RIM above the base cup's rim."""
    side = 1.0 if hand == 'left' else -1.0
    _, R0 = S.place.mount_pose()
    c, _ = S.pose(cup)
    p_tcp, _ = S.tcp(hand)
    centre = np.r_[np.asarray(region_xy) + side * side_dist * R0[:2, 1], base_top + STAGE_ABOVE_RIM + half_h(S, cup)]
    return plan_straight_or_joint(S, hand, p_tcp + (centre - c), f'{hand}_stage', [], v_max=vj(held=True),
                                  speed=sp(0.10))


def choose_stage(S, hu, hb, upper, base, region_xy, base_top, target):
    """Arm-gap revision: the upper hand's staging plan at the first of STAGE_SIDES from which the base cup's carry has a
    reachable, clear pose above its target with the upper arm at the staging point (nothing is executed)."""
    why = []
    for d in STAGE_SIDES:
        try:
            st = stage_plan(S, hu, upper, region_xy, base_top, d)
            plan_carry(S, hb, base, target, 'base', assume={hu: st[-1]})
        except Exception as e:  # noqa: BLE001
            why.append(f'{d:.2f}: {str(e)[:80]}')
            continue
        S.note(f'{hu}_stage: {d * 100:.0f} cm beside the stack region')
        return st
    raise RuntimeError(f'{hu}_stage: no staging distance leaves the base carry clear: {why}')


def run(S):
    b = S.bindings
    base, upper = b['targets']
    hb, hu = b['stack_hands']['base'], b['stack_hands']['upper']
    cup_of = {hb: base, hu: upper}
    t_open, t_close, t_release = opt('grip_s')
    S.allow = {h: {cup_of[h]} for h in HANDS}
    if not opt('preopen_in_approach'):
        S.grip({h: OPEN_BLOCK for h in HANDS}, 'preopen', t_open)
    pick = approach_and_descend(S, cup_of)
    S.grip({h: 0.0 for h in HANDS}, 'close', t_close)
    S.line({h: (pick[h][2] + [0, 0, LIFT], pick[h][3]) for h in HANDS}, 'lift', speed=sp(0.08))

    # base cup into the region; the upper hand waits (v10: outward) or goes to its staging point
    region, _ = S.pose(b['target_region'])
    target = np.r_[region[:2], TABLE_Z + half_h(S, base) + GAP]
    base_top = TABLE_Z + 2 * half_h(S, base)
    other = None
    if opt('upper_wait') == 'outward':
        wait_outward(S, hu)
    elif opt('upper_wait') == 'stage':
        st = choose_stage(S, hu, hb, upper, base, region[:2], base_top, target)
        if opt('overlap'):
            other = (hu, st, f'{hu}_stage')
        else:
            S.together({hu: st}, f'{hu}_stage')
    carry_and_lower(S, hb, base, target, 'base', other=other)
    S.grip({hb: OPEN}, 'base_release', t_release)
    S.allow[hb] = {base}
    retreat(S, hb)
    S.allow[hb] = set()

    # upper cup into the base cup; the base hand moves aside first, or (overlap) while the upper hand carries
    cb, _ = S.pose(base)
    top_b = cb[2] + half_h(S, base)
    S.allow[hu] = {upper, base}
    target_u = np.r_[cb[:2], top_b + GAP + half_h(S, upper)]
    if opt('overlap'):
        carry_and_lower(S, hu, upper, target_u, 'upper', other=(hb, aside_plan(S, hb), f'{hb}_aside'))
    else:
        S.together({hb: aside_plan(S, hb)}, f'{hb}_aside')
        carry_and_lower(S, hu, upper, target_u, 'upper')
    S.grip({hu: OPEN}, 'upper_release', t_release)
    if opt('final_aside'):
        aside(S, hu)
    else:
        retreat(S, hu)
