"""Copied-state, contact-aware arm target for a selected box contact event.

The event agent and RuleAgent choose the object, hand and contact region. This
robot skill converts that one current event into a finite-surface arm pose.
It never writes a scene state or bypasses online native hard monitoring.
"""
from __future__ import annotations

import hashlib

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .arm_path_planner import (_robot_body_ids, _robot_scene_penetration,
                               _table_penetration)
from .executor import desired_rotation


def plan_contact_pose(model: mujoco.MjModel, obs: dict, action) -> dict:
    if action.event != 'contact' or action.hand not in ('left', 'right'):
        return {'status': 'not_applicable'}
    obj, hand = action.object_id, action.hand
    geometry = obs.get('object_geometry', {}).get(obj, {})
    if geometry.get('shape') != 'box':
        return {'status': 'unsupported_shape'}
    if any(obs['finger_contact'][obj][hand]):
        return {'status': 'already_touching'}
    q0 = np.asarray(obs['robot_qpos'], dtype=float)
    if q0.shape != (model.nq,) or not np.isfinite(q0).all():
        raise ValueError('Copied contact planner lacks complete native qpos')
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    object_geoms = [int(geom) for geom in range(model.ngeom)
                    if int(model.geom_bodyid[geom]) == model.body(obj).id]
    if not object_geoms:
        return {'status': 'no_collision_geometry'}
    pads = [model.geom(f'astribot_gripper_{hand}_geom_{suffix}').id
            for suffix in ('L112', 'R112')]
    joints = [model.joint(f'astribot_arm_{hand}_joint_{index}')
              for index in range(1, 8)]
    qa = np.asarray([int(j.qposadr[0]) for j in joints])
    bounds = np.asarray([model.jnt_range[j.id] for j in joints])
    lower, upper = bounds[:, 0] + .07, bounds[:, 1] - .07
    if np.any(lower >= upper):
        raise ValueError('Contact arm has no safe joint interval')
    current = q0[qa].copy()
    # A native box may be represented by an equivalent convex mesh. The
    # frozen scene geometry gives its dimensions in the object body frame;
    # MuJoCo still checks the actual mesh during copied-state path screening.
    center = np.asarray(obs['object_position_m'][obj], dtype=float)
    rotation = np.asarray(obs['object_rotation_world'][obj], dtype=float)
    size = np.asarray(geometry['size_m'], dtype=float) / 2
    if size.shape != (3,) or np.any(size <= 0) or not np.isfinite(size).all():
        return {'status': 'invalid_box_dimensions'}
    relative = rotation.T @ (np.asarray(obs['finger_center_m'][hand]) - center)
    face_axis = int(np.argmax(np.abs(relative) / (size + .005)))
    face_sign = 1. if relative[face_axis] >= 0 else -1.
    normal = rotation[:, face_axis] * face_sign
    local_offset = rotation.T @ np.asarray(action.target_offset_m, dtype=float)
    local_offset[face_axis] = 0.
    for axis in range(3):
        if axis != face_axis:
            local_offset[axis] = np.clip(local_offset[axis],
                                         -size[axis], size[axis])
    tangential = rotation @ local_offset
    object_support = float(np.sum(np.abs(rotation.T @ normal) * size))

    def residual(arm: np.ndarray) -> np.ndarray:
        data.qpos[:] = q0
        data.qpos[qa] = arm
        mujoco.mj_forward(model, data)
        pad_support = max(float(np.sum(np.abs(
            data.geom_xmat[pad].reshape(3, 3).T @ normal) *
            model.geom_size[pad])) for pad in pads)
        target = center + tangential + normal * (object_support + pad_support + .002)
        midpoint = data.geom_xpos[pads].mean(axis=0)
        return midpoint - target

    robot_bodies = _robot_body_ids(model)

    def checked_path(target_arm: np.ndarray) -> tuple[bool, float, int]:
        count = max(21, int(np.ceil(np.linalg.norm(target_arm - current) / .025)))
        worst = 0.
        invalid = 0
        for fraction in np.linspace(0., 1., count):
            data.qpos[:] = q0
            data.qpos[qa] = current + fraction * (target_arm - current)
            mujoco.mj_forward(model, data)
            penetration = max(_table_penetration(model, data),
                              _robot_scene_penetration(model, data, robot_bodies))
            worst = max(worst, penetration)
            invalid += penetration > .003
            if invalid:
                break
        return not invalid, float(worst), count

    seed = int.from_bytes(hashlib.sha256(
        np.asarray(current, dtype=np.float64).tobytes() +
        np.asarray(center, dtype=np.float64).tobytes() +
        np.asarray(action.target_offset_m, dtype=np.float64).tobytes()
    ).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    starts = [np.clip(current, lower, upper)]
    starts += [np.clip(current + rng.normal(size=7) *
                       (upper - lower) * .12, lower, upper) for _ in range(3)]
    # Contact pose is a finite surface, so different arm configurations can
    # reach the same point with very different fixture clearance. Keep this
    # deterministic and independent of task names or historical trajectories.
    starts += [rng.uniform(lower, upper) for _ in range(12)]
    candidates = []
    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper),
                               max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        error = float(np.linalg.norm(residual(result.x)))
        if error > .006:
            candidates.append({'error_m': error, 'path_valid': False,
                               'path_samples': 0, 'worst_penetration_m': None})
            continue
        valid, worst, samples = checked_path(result.x)
        candidates.append({'error_m': error, 'path_valid': valid,
                           'path_samples': samples,
                           'worst_penetration_m': worst,
                           'arm_qpos_rad': result.x.tolist()})
    feasible = [row for row in candidates if row['path_valid']]
    if not feasible:
        return {'status': 'no_certified_static_path', 'face_axis': face_axis,
                'face_sign': face_sign, 'candidate_count': len(candidates),
                'best_error_m': min(row['error_m'] for row in candidates),
                'random_seed': seed}
    # Prefer a route with margin to unrelated scene geometry. The immutable
    # 3 mm hard gate remains unchanged; this ranking is a planning objective.
    best = min(feasible, key=lambda row: (
        row['worst_penetration_m'],
        row['error_m'] + .002 * np.linalg.norm(
            np.asarray(row['arm_qpos_rad']) - current)))
    # Refine a position-only surface proposal to the full gripper pose.
    # The native coupled waist is one additional variable. Every candidate
    # is checked against the original joint ranges and sampled geometry;
    # retain the position-only proposal if no coupled path is viable.
    from .immutable_hard_observation import HardSnapshotter
    waist = [model.joint(f'astribot_torso_joint_{index}') for index in (1, 2)]
    wqa = np.asarray([int(j.qposadr[0]) for j in waist])
    coupled_sum = float(q0[wqa].sum())
    waist_ranges = np.asarray([model.jnt_range[j.id] for j in waist])
    waist_lo = max(float(waist_ranges[0, 0]) + .07,
                   coupled_sum - float(waist_ranges[1, 1]) + .07)
    waist_hi = min(float(waist_ranges[0, 1]) - .07,
                   coupled_sum - float(waist_ranges[1, 0]) - .07)
    gripper_body = model.body(f'astribot_gripper_{hand}_base').id
    rotation_target = desired_rotation(obj, obs)
    starting_full = np.r_[current, q0[wqa[0]]]
    full_lower = np.r_[lower, waist_lo]
    full_upper = np.r_[upper, waist_hi]
    full_rows = []
    if waist_lo < waist_hi and np.all(starting_full >= full_lower) and np.all(
            starting_full <= full_upper):
        snapshotter = HardSnapshotter(model)

        def full_error(x, requested_rotation):
            data.qpos[:] = q0
            data.qpos[qa] = x[:7]
            data.qpos[wqa[0]] = x[7]
            data.qpos[wqa[1]] = coupled_sum - x[7]
            mujoco.mj_forward(model, data)
            pad_support = max(float(np.sum(np.abs(
                data.geom_xmat[pad].reshape(3, 3).T @ normal) *
                model.geom_size[pad])) for pad in pads)
            target = center + tangential + normal * (
                object_support + pad_support + .002)
            midpoint = data.geom_xpos[pads].mean(axis=0)
            actual_rotation = data.xmat[gripper_body].reshape(3, 3)
            angular = Rotation.from_matrix(
                requested_rotation @ actual_rotation.T).as_rotvec()
            return midpoint-target, angular

        def full_residual(x, requested_rotation):
            position, angle = full_error(x, requested_rotation)
            return np.r_[position, .06*angle, .0007*(x-starting_full)]

        best_arm = np.asarray(best['arm_qpos_rad'])
        data.qpos[:] = q0
        data.qpos[qa] = best_arm
        mujoco.mj_forward(model, data)
        starting_rotation = data.xmat[gripper_body].reshape(3, 3).copy()
        rotation_delta = Rotation.from_matrix(
            rotation_target @ starting_rotation.T).as_rotvec()
        starts_full = [np.r_[best_arm, q0[wqa[0]]]]
        starts_full += [np.r_[best_arm, np.clip(
            q0[wqa[0]] + delta, waist_lo, waist_hi)]
            for delta in (-.15, .15)]
        for fraction_of_rotation in (.25, .5, .75, 1.):
            requested_rotation = (Rotation.from_rotvec(
                fraction_of_rotation*rotation_delta).as_matrix() @
                starting_rotation)
            for start_full in starts_full:
                solved = least_squares(full_residual, np.clip(start_full,
                    full_lower, full_upper), args=(requested_rotation,),
                    bounds=(full_lower, full_upper), max_nfev=120,
                    ftol=1e-6, xtol=1e-6, gtol=1e-6)
                position_error, rotation_error = full_error(
                    solved.x, requested_rotation)
                actual_rotation = data.xmat[gripper_body].reshape(3, 3)
                canonical_angle = float(Rotation.from_matrix(
                    rotation_target @ actual_rotation.T).magnitude())
                pos_norm = float(np.linalg.norm(position_error))
                angle_norm = float(np.linalg.norm(rotation_error))
                count = max(31, int(np.ceil(np.linalg.norm(
                    solved.x-starting_full)/.025)))
                worst = 0.
                static_valid = True
                if pos_norm <= .006 and angle_norm <= .15:
                    for fraction in np.linspace(0., 1., count):
                        x = starting_full + fraction*(solved.x-starting_full)
                        data.qpos[:] = q0
                        data.qpos[qa] = x[:7]
                        data.qpos[wqa[0]] = x[7]
                        data.qpos[wqa[1]] = coupled_sum-x[7]
                        mujoco.mj_forward(model, data)
                        hard_sample = snapshotter(data)
                        worst = max(worst, float(hard_sample[
                            'robot_scene_penetration_m']), float(hard_sample[
                            'cross_arm_penetration_m']), float(hard_sample[
                            'robot_table_penetration_m']))
                        if not hard_sample['all_hard_passed']:
                            static_valid = False
                            break
                else:
                    static_valid = False
                full_rows.append({'position_error_m': pos_norm,
                                  'rotation_error_rad': angle_norm,
                                  'canonical_rotation_error_rad': canonical_angle,
                                  'rotation_fraction': fraction_of_rotation,
                                  'path_valid': static_valid,
                                  'worst_penetration_m': worst,
                                  'sampled_path_points': count,
                                  'target': solved.x.tolist()})
    full_feasible = [row for row in full_rows if row['path_valid']]
    full_best = min(full_feasible, key=lambda row: (
        row['canonical_rotation_error_rad'], row['worst_penetration_m'] +
        .01*np.linalg.norm(np.asarray(row['target'])-starting_full))) if full_feasible else None
    plan = {'status': 'planned', 'object_id': obj, 'hand': hand,
            'object_center_m': center.tolist(), 'face_axis': face_axis,
            'face_sign': face_sign, 'normal_world': normal.tolist(),
        'torso_qpos_at_plan_rad': [float(q0[model.joint(
            f'astribot_torso_joint_{index}').qposadr[0]]) for index in (1, 2)],
        'candidate_count': len(candidates), 'random_seed': seed,
        'clearance_first_candidate_count': sum(
            row['path_valid'] and row['worst_penetration_m'] <= .0005
            for row in candidates),
            'position_error_m': best['error_m'],
            'sampled_path_points': best['path_samples'],
            'max_sampled_robot_scene_penetration_m': best['worst_penetration_m'],
            'target_arm_qpos_rad': best['arm_qpos_rad'],
            'full_pose_candidate_count': len(full_rows),
            'full_pose_feasible_count': len(full_feasible),
            'full_pose_candidate_summary': [{key: row[key] for key in (
                'position_error_m','rotation_error_rad',
                'canonical_rotation_error_rad','rotation_fraction',
                'path_valid','worst_penetration_m','target')} for row in full_rows],
            'planning_mode': 'position_surface',
            'claim_limit': 'Static copied-state proposal; online physics and native hard constraints remain binding.'}
    if full_best is not None:
        plan.update(planning_mode='full_pose_coupled_waist',
                    target_arm_qpos_rad=full_best['target'][:7],
                    target_waist_qpos_rad=[full_best['target'][7],
                                           coupled_sum-full_best['target'][7]],
                    position_error_m=full_best['position_error_m'],
                    rotation_error_rad=full_best['canonical_rotation_error_rad'],
                    sampled_path_points=full_best['sampled_path_points'],
                    max_sampled_robot_scene_penetration_m=full_best[
                        'worst_penetration_m'])
    return plan
