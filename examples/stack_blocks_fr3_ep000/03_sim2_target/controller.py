"""Complete stack-blocks controller refinement: API05 wrist-first gaze/grasp and bounded legal scheduling of the released base hand's aside with the upper carry."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2/franka_dual_v1'
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import arch_away

_spec = importlib.util.spec_from_file_location('stack_blocks_baseline_gaze_refine', HERE / 'controllers' / 'duo_stack_blocks_v2.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

HANDS = ('left', 'right')
STEP_MAX = 0.025
FPS = 30
OPTIONS = dict(upper_wait='stage', overlap=True, final_aside=False, speed=1.6,
               grip_s=(0.4, 0.4, 0.4), preopen_in_approach=True, pre=0.07)


def grasps(S):
    return base.grasps(S)


def _camera_local(S, hand):
    cam = int(S.robot.arms[hand].wrist_cam)
    p, R = S.tcp(hand)
    eye = np.asarray(S.d.cam_xpos[cam], float).copy()
    C = np.asarray(S.d.cam_xmat[cam], float).reshape(3, 3)
    return R.T @ (eye - p), R.T @ (-C[:, 2])


def _camera_pose(S, hand, q, eye_local, forward_local):
    p, R = S.fk(hand, q)
    return p + R @ eye_local, R @ forward_local


def _patch_world(S, hand):
    patch = S.gaze_patches[hand]['point_object_m']
    c, R = S.pose(S.gaze_patches[hand]['object'])
    return c + R @ np.asarray(patch, float)


def _wrist_look_q(S, hand, eye_local, forward_local):
    q0 = S.q_cmd(hand).copy()
    arm = S.robot.arms[hand]
    target = _patch_world(S, hand)
    lo, hi = arm.lo[3:] + 0.03, arm.hi[3:] - 0.03

    def residual(x):
        q = q0.copy()
        q[3:] = x
        eye, fwd = _camera_pose(S, hand, q, eye_local, forward_local)
        v = target - eye
        v /= max(float(np.linalg.norm(v)), 1e-9)
        return np.r_[8.0 * np.cross(fwd, v), 0.025 * (x - q0[3:])]

    sol = least_squares(residual, np.clip(q0[3:], lo, hi), bounds=(lo, hi), max_nfev=100,
                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
    q = q0.copy()
    q[3:] = sol.x
    return q


def _rotation_toward(R, optical_tcp, eye, target):
    v = np.asarray(target, float) - np.asarray(eye, float)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    f = R @ optical_tcp
    return Rotation.align_vectors([v], [f])[0].as_matrix() @ R


def _cartesian_gaze_path(S, hand, p_goal, R_goal, start_q, label, target):
    eye_local, forward_local = _camera_local(S, hand)
    p0, R0 = S.fk(hand, start_q)
    distance = float(np.linalg.norm(np.asarray(p_goal) - p0))
    angle = float(Rotation.from_matrix(np.asarray(R_goal) @ R0.T).magnitude())
    n = max(8, int(np.ceil(max(distance / 0.15, angle / 0.6) * 30)))
    for _ in range(5):
        timing = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, np.asarray(R_goal)])))
        q = np.asarray(start_q, float).copy()
        rows = []
        worst_step = 0.0
        for u in timing:
            p = p0 + u * (np.asarray(p_goal, float) - p0)
            Rnom = slerp([float(u)]).as_matrix()[0]
            eye = p + Rnom @ eye_local
            Rlook = _rotation_toward(Rnom, forward_local, eye, target)
            w = 0.45 * np.sin(np.pi * float(u)) ** 2
            delta = Rotation.from_matrix(Rlook @ Rnom.T).as_rotvec()
            Rwant = Rotation.from_rotvec(w * delta).as_matrix() @ Rnom
            qn, pe, re = S.robot.ik_local(hand, p, Rwant, q, S.d.qpos, q, iters=40,
                                           max_iter_step=0.035, margin=0.03)
            if pe > 0.005 or re > 3.0:
                raise RuntimeError(f'{label}: continuous gaze IK residual {pe:.4f}m/{re:.2f}deg')
            worst_step = max(worst_step, float(np.max(np.abs(qn - q))))
            rows.append(qn.copy())
            q = qn
        if worst_step <= STEP_MAX + 1e-9:
            path = np.asarray(rows, float)
            ok, info = S._path_ok({hand: path}, allow=S.allow,
                                  others={h: S.q_cmd(h) for h in HANDS if h != hand})
            if not ok:
                raise RuntimeError(f'{label}: original path check failed: {info}')
            return path
        n *= int(np.ceil(worst_step / STEP_MAX * 1.15))
    raise RuntimeError(f'{label}: could not satisfy original STEP_MAX with continuous Cartesian samples')


def _paired(paths, label, S):
    n = max(len(paths[h]) for h in HANDS)
    joined = {h: np.vstack([paths[h], np.repeat(paths[h][-1][None, :], n - len(paths[h]), axis=0)])
              for h in HANDS}
    ok, info = S._path_ok(joined, allow=S.allow)
    if not ok:
        raise RuntimeError(f'{label}: paired original path check failed: {info}')
    return joined


def approach_and_descend(S, cup_of):
    candidates = {h: base.face_grasps(S, h, cup_of[h])[:4] for h in HANDS}
    pairs = sorted(((l, r) for l in candidates['left'] for r in candidates['right']),
                   key=lambda ab: ab[0][0] + ab[1][0])
    look = {}
    for h in HANDS:
        eye_local, forward_local = _camera_local(S, h)
        look[h] = _wrist_look_q(S, h, eye_local, forward_local)
    look_plans = {h: S.joint_plan(h, look[h], v_max=base.vj()) for h in HANDS}
    ok, info = S._path_ok(look_plans, allow=S.allow)
    if not ok:
        raise RuntimeError(f'wrist-first paired orientation failed original path check: {info}')
    S.together(look_plans, 'wrist_look')

    errors = []
    for left, right in pairs:
        pick = {'left': left, 'right': right}
        try:
            paths = {h: _cartesian_gaze_path(S, h, pick[h][1], pick[h][3], S.q_cmd(h),
                                              h + '_gaze_approach', _patch_world(S, h)) for h in HANDS}
            paired = _paired(paths, 'paired gaze approach', S)
            if OPTIONS['preopen_in_approach']:
                S.together(paired, 'approach', grips={h: base.OPEN_BLOCK for h in HANDS})
            else:
                S.together(paired, 'approach')
            desc = {h: _cartesian_gaze_path(S, h, pick[h][2], pick[h][3], S.q_cmd(h),
                                             h + '_gaze_descend', _patch_world(S, h)) for h in HANDS}
            S.together(_paired(desc, 'paired gaze descent', S), 'descend')
            return pick
        except Exception as e:
            errors.append(str(e)[:140])
            if any(label in ('approach', 'descend') for label in S.labels[-1:]):
                raise
    raise RuntimeError(f'no clear continuous gaze-aware top-down grasp pair after wrist look: {errors[:4]}')


def _delayed_pair_paths(S, carrier, carry, mover, aside, delay_frames):
    """Build the exact full-time-indexed paths that together(delay=...) checks and executes."""
    n = int(delay_frames)
    q0 = np.asarray(S.q_cmd(carrier), float)
    delayed = np.concatenate((np.repeat(q0[None, :], n, axis=0), np.asarray(carry, float)), axis=0)
    return {carrier: delayed, mover: np.asarray(aside, float)}


def _select_paired_schedule(S, carrier, carry, mover, aside, delays):
    """Pre-execution only: validate full delayed paths with hold-at-end semantics and select the shortest legal span."""
    best = None
    for delay_frames in sorted(set(int(x) for x in delays if int(x) >= 0)):
        joined = _delayed_pair_paths(S, carrier, carry, mover, aside, delay_frames)
        ok, info = S._path_ok(joined, allow=S.allow)
        if not ok:
            continue
        span = max(len(joined[carrier]), len(joined[mover]))
        score = (span, delay_frames)
        if best is None or score < best[0]:
            best = (score, delay_frames, joined)
    if best is None:
        return None
    return best[1], best[2]


def _delay_candidates(aside_frames):
    n = max(0, int(aside_frames))
    vals = set(range(0, n + 1, 3))
    vals.update(range(0, min(n, 18) + 1))
    vals.add(n)
    return sorted(vals)


def _short_aside_plan(S, hand):
    """Try bounded nearer own-side endpoints, retaining the exact original endpoint as last resort."""
    side = 1.0 if hand == 'left' else -1.0
    p0, R0 = S.place.mount_pose()
    p, _ = S.tcp(hand)
    fwd, lat, _ = R0.T @ (p - p0)
    specs = [(d, base.mount_point(S, fwd, side * max(d, side * lat), base.ASIDE[2]))
             for d in (0.24, 0.28, 0.32, 0.36)]
    candidates = []
    why = []
    other = next(h for h in HANDS if h != hand)
    for d, goal in specs:
        try:
            path = base.plan_straight_or_joint(S, hand, goal, f'{hand}_aside_short',
                                                [base.top_down(np.array([1.0, 0, 0]))], speed=base.sp(0.10))
            ok, info = S._path_ok({hand: path}, allow=S.allow, others={other: S.q_cmd(other)})
            if ok:
                candidates.append((d, path))
            else:
                why.append(f'{d:.2f}: {info}')
        except Exception as e:
            why.append(f'{d:.2f}: {str(e)[:80]}')
    try:
        original = original_aside_plan(S, hand)
        ok, info = S._path_ok({hand: original}, allow=S.allow, others={other: S.q_cmd(other)})
        if ok:
            candidates.append((base.ASIDE[1], original))
        else:
            why.append(f'original: {info}')
    except Exception as e:
        why.append(f'original: {str(e)[:100]}')
    if not candidates:
        raise RuntimeError(f'{hand}_aside: no individually checked short or original plan: {why}')
    candidates.sort(key=lambda item: (len(item[1]), item[0]))
    if not hasattr(S, '_aside_candidate_plans'):
        S._aside_candidate_plans = {}
    S._aside_candidate_plans[hand] = candidates
    S.note(f'{hand}_aside: planned {len(candidates)} own-side endpoints; shortest {candidates[0][0]:.2f} m')
    return candidates[0][1]


def _carry_sequence(plans, hand):
    return np.concatenate([p[hand] for p in plans], axis=0)


def _checked_carry_and_lower(S, hand, cup, target_center, label, other=None):
    if other is None or other[2].endswith('_stage'):
        return original_carry_and_lower(S, hand, cup, target_center, label, other)

    mover, supplied_aside, aside_label = other
    alternatives = getattr(S, '_aside_candidate_plans', {}).get(mover, [(base.ASIDE[1], supplied_aside)])
    planned = []
    planning_errors = []
    for endpoint, aside_path in alternatives:
        try:
            plans, result = base.plan_carry(S, hand, cup, target_center, label,
                                             assume={mover: np.asarray(aside_path)[-1]})
            goal, R, last = result
            planned.append((endpoint, aside_path, plans, goal, R, last))
        except Exception as e:
            planning_errors.append(f'{endpoint:.2f}: {str(e)[:100]}')

    legal = []
    for endpoint, aside_path, plans, goal, R, last in planned:
        if plans is None:
            continue
        seq = _carry_sequence(plans, hand)
        choice = _select_paired_schedule(S, hand, seq, mover, aside_path,
                                         _delay_candidates(len(aside_path)))
        if choice is not None:
            delay_frames, _ = choice
            span = max(len(seq) + delay_frames, len(aside_path))
            legal.append(((span, delay_frames, len(aside_path), endpoint), endpoint, aside_path,
                          plans, goal, R, last, seq, delay_frames))

    if legal:
        legal.sort(key=lambda x: x[0])
        _, endpoint, aside_path, plans, goal, R, last, seq, delay_frames = legal[0]
        # Single execution call, deliberately outside all planning/retry scopes. Any execution error propagates.
        delay = {hand: delay_frames / float(FPS)} if delay_frames else None
        S.together({hand: seq, mover: aside_path}, f'{label}_transport', delay=delay)
        base.lower_corrected(S, hand, cup, target_center, R, last, label)
        S.note(f'{label}: checked paired aside/carry; aside endpoint {endpoint:.2f} m, delay {delay_frames} frames')
        return

    # No candidate pair passed the original full-path check. Fall back to the original endpoint and serial execution;
    # this is planned before execution and does not catch/retry an execution failure.
    original_entry = next((x for x in alternatives if abs(x[0] - base.ASIDE[1]) < 1e-9), None)
    if original_entry is None:
        original_path = original_aside_plan(S, mover)
    else:
        original_path = original_entry[1]
    fallback = next((x for x in planned if x[1] is original_path or
                     (len(x[1]) == len(original_path) and np.array_equal(x[1], original_path))), None)
    if fallback is None:
        plans, result = base.plan_carry(S, hand, cup, target_center, label,
                                         assume={mover: np.asarray(original_path)[-1]})
        goal, R, last = result
    else:
        _, _, plans, goal, R, last = fallback

    if plans is None:
        S.together({mover: original_path}, aside_label)
        S.move_joints({hand: last}, f'{label}_transport', v_max=base.vj(held=True))
        S.line({hand: (goal, R)}, f'{label}_lower', speed=base.sp(0.05))
        return
    S.together({mover: original_path}, aside_label)
    S.run_plans(plans, f'{label}_transport')
    base.lower_corrected(S, hand, cup, target_center, R, last, label)
    S.note(f'{label}: paired schedules unavailable; original-endpoint serial fallback')


# Frozen API05 source functions and task mechanics; only the released-base aside planner and
# the base-aside/upper-carry scheduling branch are replaced.
original_aside_plan = base.aside_plan
original_carry_and_lower = base.carry_and_lower
base.aside_plan = _short_aside_plan
base.carry_and_lower = _checked_carry_and_lower


def run(S):
    base.OPTIONS = OPTIONS
    b = S.bindings
    base_obj, upper_obj = b['targets']
    hm = b['stack_hands']
    cup_of = {hm['base']: base_obj, hm['upper']: upper_obj}
    S.gaze_patches = {}
    for h in HANDS:
        obj = cup_of[h]
        half = float(S.objects[obj]['size_m'][2]) / 2.0
        S.gaze_patches[h] = {'object': obj, 'point_object_m': [0.0, 0.0, half], 'active': True}
    for h in HANDS:
        arch_away.install(S, h)
    base.approach_and_descend = approach_and_descend

    grip0 = S.grip
    def grip_with_gaze_release(widths, label, duration=0.6):
        out = grip0(widths, label, duration)
        if label == 'base_release':
            S.gaze_patches[hm['base']]['active'] = False
        elif label == 'upper_release':
            S.gaze_patches[hm['upper']]['active'] = False
        return out
    S.grip = grip_with_gaze_release
    result = base.run(S)
    S.note('API05 wrist-first gaze/grasp retained; complete paired schedules checked before one execution call')
    S.note('arch_away: ' + '; '.join(f'{h} {S.arch_away_all[h].stats}' for h in S.arch_away_all))
    return result
