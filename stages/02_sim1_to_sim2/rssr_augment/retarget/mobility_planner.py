"""Robot capability probe for S1 planar-base reposition events.

This proposes reachable base/arm poses from a copied MuJoCo state. It does
not select a task rule, issue a robot command, write live simulation state,
or certify physical task success. The RuleAgent must choose whether to use
an offered reposition event from actual online evidence.
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


BASE_NAMES = ('astribot_chassis_x', 'astribot_chassis_y',
              'astribot_chassis_zrot')


def _table_penetration(model, data):
    largest = 0.
    for contact in data.contact:
        names = (model.geom(int(contact.geom1)).name or '',
                 model.geom(int(contact.geom2)).name or '')
        if any(name.startswith('worktable') for name in names) and any(
                'astribot_' in name or name.startswith(('left_', 'right_'))
                for name in names):
            largest = max(largest, -float(contact.dist))
    return largest


def _robot_body_ids(model):
    """Find the complete articulated robot by hierarchy, not scene names."""
    root = model.body('chassis_base').id
    result = set()
    for body in range(model.nbody):
        current = body
        while current and current != root:
            current = int(model.body_parentid[current])
        if current == root:
            result.add(body)
    return result


def _robot_scene_penetration(model, data, robot_bodies):
    """Measure any robot contact with the independent scene during base travel."""
    largest = 0.
    for contact in data.contact:
        body_a = int(model.geom_bodyid[int(contact.geom1)])
        body_b = int(model.geom_bodyid[int(contact.geom2)])
        if (body_a in robot_bodies) != (body_b in robot_bodies):
            largest = max(largest, -float(contact.dist))
    return largest


def _base_path_penetration(model, data, original, base_qa, arm_qa,
                           final_base, final_arm):
    """Screen base travel against all scene bodies; then screen the arm sweep."""
    initial_base = original[base_qa]
    initial_arm = original[arm_qa]
    robot_bodies = _robot_body_ids(model)
    results = []
    for order in itertools.permutations(range(3)):
        point = initial_base.copy()
        samples = []
        base_scene_samples = []
        for axis in order:
            start = point[axis]
            for fraction in np.linspace(0., 1., 21)[1:]:
                point[axis] = start + fraction * (final_base[axis] - start)
                data.qpos[:] = original
                data.qpos[base_qa] = point
                mujoco.mj_forward(model, data)
                samples.append(_table_penetration(model, data))
                base_scene_samples.append(_robot_scene_penetration(
                    model, data, robot_bodies))
            point[axis] = final_base[axis]
        for fraction in np.linspace(0., 1., 41)[1:]:
            data.qpos[:] = original
            data.qpos[base_qa] = final_base
            data.qpos[arm_qa] = initial_arm + fraction * (final_arm - initial_arm)
            mujoco.mj_forward(model, data)
            samples.append(_table_penetration(model, data))
        results.append({'order': [BASE_NAMES[axis] for axis in order],
                        'max_table_penetration_m': max(samples),
                        'poses_over_3mm': sum(value > .003 for value in samples),
                        'sampled_poses': len(samples),
                        'max_base_travel_scene_penetration_m': max(base_scene_samples),
                        'base_travel_poses_scene_over_3mm': sum(
                            value > .003 for value in base_scene_samples),
                        'base_travel_sampled_poses': len(base_scene_samples)})
    results.sort(key=lambda row: (row['base_travel_poses_scene_over_3mm'],
                                  row['poses_over_3mm'],
                                  row['max_table_penetration_m']))
    return results[0]


def propose_reposition(model, current_qpos, hand, target_points_world,
                       target_rotation_world, *, max_path_checks=24,
                       search_profile='local'):
    """Return geometry-derived feasible first-contact base poses.

    `model` must be a private planning copy of the candidate S1 scene.
    `target_points_world` is a mapping of contact-region labels to measured
    world points. The method is deliberately unaware of case or scene names.
    """
    current_qpos = np.asarray(current_qpos, dtype=float)
    target_rotation_world = np.asarray(target_rotation_world, dtype=float)
    if current_qpos.shape != (model.nq,) or not np.isfinite(current_qpos).all():
        raise ValueError('Planner requires one finite copied physical state')
    if hand not in ('left', 'right') or target_rotation_world.shape != (3, 3):
        raise ValueError('Planner hand or target rotation invalid')
    if not target_points_world:
        raise ValueError('Planner has no measured contact region')
    data = mujoco.MjData(model)
    data.qpos[:] = current_qpos
    mujoco.mj_forward(model, data)
    base_joints = [model.joint(name) for name in BASE_NAMES]
    base_qa = np.asarray([int(j.qposadr[0]) for j in base_joints])
    base_va = np.asarray([int(j.dofadr[0]) for j in base_joints])
    arm_joints = [model.joint(f'astribot_arm_{hand}_joint_{i}')
                  for i in range(1, 8)]
    arm_qa = np.asarray([int(j.qposadr[0]) for j in arm_joints])
    arm_bounds = np.asarray([model.jnt_range[j.id] for j in arm_joints])
    lower, upper = arm_bounds[:, 0] + .07, arm_bounds[:, 1] - .07
    initial_base = current_qpos[base_qa]
    initial_arm = current_qpos[arm_qa]
    body = model.body(f'astribot_gripper_{hand}_base').id
    pads = [model.geom(f'astribot_gripper_{hand}_geom_{suffix}').id
            for suffix in ('L112', 'R112')]
    chassis = model.body('chassis_base').id
    base_world = data.xpos[chassis].copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros_like(jacp)
    mujoco.mj_jacBody(model, data, jacp, jacr, chassis)
    slide_jac = jacp[:2, base_va[:2]]
    if abs(float(np.linalg.det(slide_jac))) < .5:
        raise ValueError('S1 planar slide axes cannot span table plane')
    centers = {label: np.asarray(point, dtype=float)
               for label, point in target_points_world.items()}
    if any(point.shape != (3,) or not np.isfinite(point).all()
           for point in centers.values()):
        raise ValueError('Contact region must be a finite 3D point')
    side = np.sign(np.mean([point[0] for point in centers.values()]) - base_world[0])
    side = 1. if side >= 0 else -1.
    if search_profile == 'local':
        x_shifts = (0., side * .15, side * .30)
        y_shifts = (0., -.10, -.20, -.30)
        yaw_shifts = (0., -side * np.deg2rad(25.),
                      -side * np.deg2rad(45.))
    elif search_profile == 'expanded':
        # A robot capability fallback in world coordinates, independent of
        # task labels. Joint limits and collision checks still apply.
        x_shifts = (0., side * .15, side * .30, side * .45)
        y_shifts = (0., -.10, -.20, -.30, -.40, .10)
        yaw_shifts = (0., -side * np.deg2rad(25.),
                      -side * np.deg2rad(45.),
                      side * np.deg2rad(25.),
                      side * np.deg2rad(45.))
    else:
        raise ValueError('Unknown base reach search profile')
    start_seeds = (initial_arm, np.clip((lower + upper) / 2, lower, upper))
    fitted = []
    static_fit_diagnostics = []
    checked = 0
    initial_collision_rejected = 0
    for x_shift in x_shifts:
        for y_shift in y_shifts:
            world_shift = np.array([x_shift, y_shift])
            slide_delta = np.linalg.solve(slide_jac, world_shift)
            for yaw_shift in yaw_shifts:
                base = initial_base + np.r_[slide_delta, yaw_shift]
                if any(base[i] < model.jnt_range[j.id, 0] + .005 or
                       base[i] > model.jnt_range[j.id, 1] - .005
                       for i, j in enumerate(base_joints)):
                    continue
                data.qpos[:] = current_qpos
                data.qpos[base_qa] = base
                mujoco.mj_forward(model, data)
                checked += 1
                if _table_penetration(model, data) > .003:
                    initial_collision_rejected += 1
                    continue
                for label, target in centers.items():
                    best = None
                    for seed in start_seeds:
                        data.qpos[:] = current_qpos
                        data.qpos[base_qa] = base

                        def residual(arm):
                            data.qpos[arm_qa] = arm
                            mujoco.mj_forward(model, data)
                            position = data.geom_xpos[pads].mean(axis=0)
                            rotation = data.xmat[body].reshape(3, 3)
                            angular = Rotation.from_matrix(
                                target_rotation_world @ rotation.T).as_rotvec()
                            return np.r_[position - target, .08 * angular]

                        solution = least_squares(
                            residual, np.clip(seed, lower, upper),
                            bounds=(lower, upper), max_nfev=70,
                            ftol=1e-6, xtol=1e-6, gtol=1e-6)
                        error = residual(solution.x)
                        position_error = float(np.linalg.norm(error[:3]))
                        rotation_error = float(np.linalg.norm(error[3:]) / .08)
                        penetration = _table_penetration(model, data)
                        score = (position_error + .08 * rotation_error +
                                 10. * penetration)
                        row = {'contact_region': label,
                               'base_joint_target': base.tolist(),
                               'base_world_shift_xy_m': world_shift.tolist(),
                               'base_yaw_shift_rad': float(yaw_shift),
                               'arm_joint_target': solution.x.tolist(),
                               'position_error_m': position_error,
                               'rotation_error_rad': rotation_error,
                               'table_penetration_m': penetration,
                               'score': score}
                        if best is None or score < best['score']:
                            best = row
                    static_fit_diagnostics.append(best)
                    if (best['position_error_m'] <= .02 and
                            best['rotation_error_rad'] <= .25 and
                            best['table_penetration_m'] <= .003):
                        fitted.append(best)
    refinement_attempts = 0
    if search_profile == 'expanded' and not fitted:
        # Refine robot poses that nearly meet the same geometric hand target.
        # The neighborhood is tied to measured fit error, never a task name.
        near_fits = sorted(static_fit_diagnostics,
                           key=lambda row: row['score'])[:3]
        for near in near_fits:
            if (near['position_error_m'] > .06 or
                    near['rotation_error_rad'] > .6 or
                    near['table_penetration_m'] > .003):
                continue
            target = centers[near['contact_region']]
            for degrees in (-15., -10., -5., 5., 10., 15.):
                base = np.asarray(near['base_joint_target']).copy()
                base[2] += np.deg2rad(degrees)
                if any(base[i] < model.jnt_range[j.id, 0] + .005 or
                       base[i] > model.jnt_range[j.id, 1] - .005
                       for i, j in enumerate(base_joints)):
                    continue
                data.qpos[:] = current_qpos
                data.qpos[base_qa] = base
                mujoco.mj_forward(model, data)
                if _table_penetration(model, data) > .003:
                    continue
                refinement_attempts += 1

                def residual_refined(arm):
                    data.qpos[arm_qa] = arm
                    mujoco.mj_forward(model, data)
                    position = data.geom_xpos[pads].mean(axis=0)
                    rotation = data.xmat[body].reshape(3, 3)
                    angular = Rotation.from_matrix(
                        target_rotation_world @ rotation.T).as_rotvec()
                    return np.r_[position - target, .08 * angular]

                seed = np.clip(np.asarray(near['arm_joint_target']),
                               lower, upper)
                solution = least_squares(
                    residual_refined, seed, bounds=(lower, upper),
                    max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                error = residual_refined(solution.x)
                position_error = float(np.linalg.norm(error[:3]))
                rotation_error = float(np.linalg.norm(error[3:]) / .08)
                penetration = _table_penetration(model, data)
                world_shift = data.xpos[chassis][:2] - base_world[:2]
                row = {'contact_region': near['contact_region'],
                       'base_joint_target': base.tolist(),
                       'base_world_shift_xy_m': world_shift.tolist(),
                       'base_yaw_shift_rad': float(base[2]-initial_base[2]),
                       'arm_joint_target': solution.x.tolist(),
                       'position_error_m': position_error,
                       'rotation_error_rad': rotation_error,
                       'table_penetration_m': penetration,
                       'score': position_error + .08*rotation_error +
                                10.*penetration,
                       'source': 'measured_near_fit_yaw_refinement'}
                static_fit_diagnostics.append(row)
                if (position_error <= .02 and rotation_error <= .25 and
                        penetration <= .003):
                    fitted.append(row)
    fitted.sort(key=lambda row: (np.linalg.norm(row['base_world_shift_xy_m']) +
                                 .12 * abs(row['base_yaw_shift_rad']) +
                                 row['score']))
    path_checked = []
    accepted = []
    for row in fitted[:max_path_checks]:
        path = _base_path_penetration(
            model, data, current_qpos, base_qa, arm_qa,
            np.asarray(row['base_joint_target']),
            np.asarray(row['arm_joint_target']))
        path_checked.append({**row, 'path': path})
        if (path['poses_over_3mm'] == 0 and
                path['base_travel_poses_scene_over_3mm'] == 0):
            accepted.append(path_checked[-1])
        if len(accepted) >= 3:
            break
    fixed_base_static = int(sum(
        np.linalg.norm(row['base_world_shift_xy_m']) < 1e-8 and
        abs(row['base_yaw_shift_rad']) < 1e-8 for row in fitted))
    rejected_paths = sorted(
        (row for row in path_checked if row['path']['poses_over_3mm'] or
         row['path']['base_travel_poses_scene_over_3mm']),
        key=lambda row: (
            row['path']['base_travel_poses_scene_over_3mm'],
            row['path']['poses_over_3mm'],
            row['path']['max_table_penetration_m']))
    return {'schema': 'rssr.s1_mobile_capability_probe.v1',
            'hand': hand, 'search_profile': search_profile,
            'base_poses_checked': checked,
            'initial_table_collision_rejected': initial_collision_rejected,
            'near_fit_refinement_attempts': refinement_attempts,
            'fixed_base_static_pose_count': fixed_base_static,
            'static_pose_count': len(fitted),
            'best_static_fit_diagnostics': sorted(
                static_fit_diagnostics,
                key=lambda row: row['score'])[:12],
            'path_checked_count': len(path_checked),
            'path_valid_count': len(accepted),
            'path_valid_candidates': accepted,
            'best_rejected_path_summaries': [
                {'contact_region': row['contact_region'],
                 'base_joint_target': row['base_joint_target'],
                 'position_error_m': row['position_error_m'],
                 'rotation_error_rad': row['rotation_error_rad'],
                 'path': row['path']}
                for row in rejected_paths[:6]],
            'robot_physics_steps': 0,
            'claim_limit': 'Copied-state kinematics only; candidate is not a driven trajectory or task success.'}
