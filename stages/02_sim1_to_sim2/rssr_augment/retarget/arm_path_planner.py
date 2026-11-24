"""Find a bounded arm transit waypoint in a copied MuJoCo state.

This is a robot skill proposal, independent of task names and RuleAgent
policy. It validates static samples only; the live executor must still check
every physics step and the native task.
"""
from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np


def _table_penetration(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    largest = 0.
    for contact in data.contact:
        names = (model.geom(int(contact.geom1)).name or '',
                 model.geom(int(contact.geom2)).name or '')
        if any(name.startswith('worktable') for name in names) and any(
                'astribot_' in name or name.startswith(('left_', 'right_'))
                for name in names):
            largest = max(largest, -float(contact.dist))
    return largest


def _robot_body_ids(model: mujoco.MjModel) -> set[int]:
    root = model.body('chassis_base').id
    result = set()
    for body in range(model.nbody):
        current = body
        while current and current != root:
            current = int(model.body_parentid[current])
        if current == root:
            result.add(body)
    return result


def _robot_scene_penetration(model: mujoco.MjModel,
                             data: mujoco.MjData,
                             robot_bodies: set[int]) -> float:
    largest = 0.
    for contact in data.contact:
        first = int(model.geom_bodyid[int(contact.geom1)])
        second = int(model.geom_bodyid[int(contact.geom2)])
        if (first in robot_bodies) != (second in robot_bodies):
            largest = max(largest, -float(contact.dist))
    return largest


def find_one_waypoint(model: mujoco.MjModel, source_qpos: np.ndarray,
                      base_qpos_indices: np.ndarray,
                      arm_qpos_indices: np.ndarray,
                      target_base: np.ndarray,
                      target_arm: np.ndarray,
                      *, max_attempts: int = 300,
                      max_penetration_m: float = .003) -> dict:
    """Screen two straight joint legs with one copied-state waypoint.

    The caller must validate the base travel and endpoint IK separately.
    The returned waypoint is never a command by itself.
    """
    if not 1 <= max_attempts <= 2000:
        raise ValueError('Waypoint sample budget must be 1..2000')
    source_qpos = np.asarray(source_qpos, dtype=float)
    base_qpos_indices = np.asarray(base_qpos_indices, dtype=int)
    arm_qpos_indices = np.asarray(arm_qpos_indices, dtype=int)
    target_base = np.asarray(target_base, dtype=float)
    target_arm = np.asarray(target_arm, dtype=float)
    if (source_qpos.shape != (model.nq,) or
            base_qpos_indices.shape != (3,) or
            arm_qpos_indices.ndim != 1 or len(arm_qpos_indices) == 0 or
            target_base.shape != (3,) or
            target_arm.shape != (len(arm_qpos_indices),) or
            not all(np.isfinite(array).all() for array in
                    (source_qpos, target_base, target_arm))):
        raise ValueError('Arm path copied-state dimensions differ')
    data = mujoco.MjData(model)
    robot_bodies = _robot_body_ids(model)
    baseline = source_qpos.copy()
    baseline[base_qpos_indices] = target_base
    start_arm = baseline[arm_qpos_indices].copy()
    lower = []
    upper = []
    for address in arm_qpos_indices:
        joint = next((j for j in range(model.njnt)
                      if int(model.jnt_qposadr[j]) == int(address)), None)
        if joint is None or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('Arm path requires named scalar hinge joints')
        lower.append(float(model.jnt_range[joint, 0]) + .07)
        upper.append(float(model.jnt_range[joint, 1]) - .07)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if np.any(target_arm < lower - 1e-8) or np.any(target_arm > upper + 1e-8):
        raise ValueError('Arm target outside bounded command range')

    def screen(first: np.ndarray, second: np.ndarray,
               samples: int) -> tuple[int, float, float]:
        count, deepest_table, deepest_scene = 0, 0., 0.
        for fraction in np.linspace(0., 1., samples):
            data.qpos[:] = baseline
            data.qpos[arm_qpos_indices] = first + fraction * (second - first)
            mujoco.mj_forward(model, data)
            table = _table_penetration(model, data)
            scene = _robot_scene_penetration(model, data, robot_bodies)
            count += max(table, scene) > max_penetration_m
            deepest_table = max(deepest_table, table)
            deepest_scene = max(deepest_scene, scene)
        return int(count), float(deepest_table), float(deepest_scene)

    direct = screen(start_arm, target_arm, 101)
    if direct[0] == 0:
        return {'status': 'direct', 'waypoint_arm_qpos': None,
                'attempts': 0, 'direct_101_samples': direct,
                'first_leg_101_samples': direct,
                'second_leg_101_samples': (0, 0., 0.),
                'max_penetration_m': max_penetration_m,
                'physics_steps': 0}
    payload = (np.asarray(start_arm, dtype=np.float64).tobytes() +
               np.asarray(target_arm, dtype=np.float64).tobytes() +
               np.asarray(target_base, dtype=np.float64).tobytes())
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    center = (start_arm + target_arm) / 2
    width = upper - lower
    seeds = [model.qpos0[arm_qpos_indices], (lower + upper) / 2,
             start_arm + .25 * (target_arm - start_arm), center,
             start_arm + .75 * (target_arm - start_arm)]
    per_scale = math.ceil(max(0, max_attempts - len(seeds)) / 4)
    for scale in (.10, .20, .35, .50):
        for _ in range(per_scale):
            seeds.append(np.clip(center + rng.normal(0., scale, len(center)) * width,
                                 lower, upper))
    for index, candidate in enumerate(seeds[:max_attempts], start=1):
        waypoint = np.clip(candidate, lower, upper)
        first = screen(start_arm, waypoint, 21)
        if first[0] != 0:
            continue
        second = screen(waypoint, target_arm, 21)
        if second[0] != 0:
            continue
        dense_first = screen(start_arm, waypoint, 101)
        dense_second = screen(waypoint, target_arm, 101)
        if dense_first[0] == dense_second[0] == 0:
            return {'status': 'waypoint',
                    'waypoint_arm_qpos': waypoint.tolist(),
                    'attempts': index,
                    'random_seed': seed,
                    'direct_101_samples': direct,
                    'first_leg_101_samples': dense_first,
                    'second_leg_101_samples': dense_second,
                    'max_penetration_m': max_penetration_m,
                    'physics_steps': 0}
    return {'status': 'unresolved', 'waypoint_arm_qpos': None,
            'attempts': min(len(seeds), max_attempts),
            'random_seed': seed,
            'direct_101_samples': direct,
            'max_penetration_m': max_penetration_m,
            'physics_steps': 0}
