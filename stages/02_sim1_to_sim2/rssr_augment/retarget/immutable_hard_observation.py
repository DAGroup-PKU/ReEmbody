"""Read-only measurements of native and model-integrity fidelity gates.

The precompiled reader does not call forward/step or write model/data. Feed its
snapshots to HardAccumulator at *every physical step*, including the initial
state. A sample reconstructed from an old 100 Hz rollout is a diagnostic only;
it cannot recover unrecorded every-step maxima or solver-warning history.
"""
from __future__ import annotations

import math
import numpy as np
from .collision_integrity import ContactIntegrityMonitor

HARD_KEYS = (
    'robot_table_penetration_le_3mm',
    'limited_joints_within_2mrad_or_2mm',
    'solver_no_warnings',
    'waist_coupling_within_1mrad',
    'gripper_coupling_within_2mrad',
    'cross_arm_penetration_le_3mm',
    'robot_scene_penetration_le_3mm',
    'chassis_world_position_fixed_10nm',
    'chassis_world_orientation_fixed_1e8',
)


class HardSnapshotter:
    """Cache native model indices once; subsequent measurements are read-only."""
    def __init__(self, model):
        import mujoco
        self.model = model
        self.waist_sum = float(model.numeric_data[int(model.numeric('rssr_s1_measured_waist_sum_rad').adr[0])])
        limited = np.flatnonzero(
            np.asarray(model.jnt_limited, dtype=bool) &
            np.isin(model.jnt_type, (mujoco.mjtJoint.mjJNT_HINGE,
                                    mujoco.mjtJoint.mjJNT_SLIDE)))
        self.limit_qa = np.asarray(model.jnt_qposadr[limited], dtype=int).copy()
        self.lo = np.asarray(model.jnt_range[limited, 0], dtype=float).copy()
        self.hi = np.asarray(model.jnt_range[limited, 1], dtype=float).copy()
        self.limited_names = tuple(model.joint(int(j)).name or '' for j in limited)
        self.waist_qa = np.array([int(model.joint(
            'astribot_torso_joint_' + str(i)).qposadr[0]) for i in (1, 2)])
        rows = []
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            a, b = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
            if b < 0 or 'gripper' not in (model.joint(a).name or ''):
                continue
            rows.append((int(model.jnt_qposadr[a]), int(model.jnt_qposadr[b]),
                         np.asarray(model.eq_data[i, :5], dtype=float).copy()))
        self.couplings = tuple(rows)
        self.qpos0 = np.asarray(model.qpos0, dtype=float).copy()
        names = tuple(model.geom(i).name or '' for i in range(model.ngeom))
        # These are the frozen native geometry classifications, not task IDs.
        self.table = np.array([n.startswith('worktable') for n in names])
        self.robot = np.array(['astribot_' in n or n.startswith(('left_', 'right_'))
                               for n in names])
        self.contact_integrity = ContactIntegrityMonitor(model)
        self.chassis_id = int(model.body('chassis_base').id)
        if (int(model.body_parentid[self.chassis_id]) != 0 or
                int(model.body_jntnum[self.chassis_id]) != 0):
            raise ValueError('chassis_base must be rigidly attached to the world')
        self.chassis_pos0 = np.asarray(model.body_pos[self.chassis_id], dtype=float).copy()
        self.chassis_quat0 = np.asarray(model.body_quat[self.chassis_id], dtype=float).copy()

    def __call__(self, data):
        q = np.asarray(data.qpos)
        valid = bool(np.isfinite(q).all() and np.isfinite(data.qvel).all()
                     and np.isfinite(data.qacc).all() and math.isfinite(float(data.time)))
        excess = np.maximum(0., np.maximum(self.lo - q[self.limit_qa],
                                           q[self.limit_qa] - self.hi))
        max_excess = float(excess.max(initial=0.))
        coupling = []
        for a, b, poly in self.couplings:
            relative = float(q[b] - self.qpos0[b])
            expected = sum(float(poly[k]) * relative ** k for k in range(5))
            coupling.append(abs(float(q[a] - self.qpos0[a]) - expected))
        grip = float(max(coupling, default=0.))
        waist = abs(float(q[self.waist_qa].sum() - self.waist_sum))
        penetration = 0.
        for c in data.contact:
            if not math.isfinite(float(c.dist)):
                valid = False
                continue
            a, b = int(c.geom1), int(c.geom2)
            if (self.table[a] and self.robot[b]) or (self.table[b] and self.robot[a]):
                penetration = max(penetration, -float(c.dist))
        warnings = int(sum(int(w.number) for w in data.warning))
        integrity = self.contact_integrity.observe(data)
        chassis_pos = np.asarray(data.xpos[self.chassis_id], dtype=float)
        chassis_quat = np.asarray(data.xquat[self.chassis_id], dtype=float)
        chassis_translation = float(np.linalg.norm(chassis_pos - self.chassis_pos0))
        chassis_z = float(abs(chassis_pos[2] - self.chassis_pos0[2]))
        chassis_orientation = float(min(np.linalg.norm(chassis_quat - self.chassis_quat0),
                                        np.linalg.norm(chassis_quat + self.chassis_quat0)))
        valid = valid and all(math.isfinite(v) for v in
                              (max_excess, grip, waist, penetration,
                               integrity['cross_arm_penetration_m'],
                               integrity['robot_scene_penetration_m'],
                               chassis_translation, chassis_z, chassis_orientation))
        checks = dict(zip(HARD_KEYS, (
            penetration <= .003, max_excess <= .002, warnings == 0,
            waist <= .001, grip <= .002,
            integrity['cross_arm_penetration_m'] <= .003,
            integrity['robot_scene_penetration_m'] <= .003,
            chassis_translation <= 1e-8 and chassis_z <= 1e-8,
            chassis_orientation <= 1e-8)))
        checks = {k: bool(valid and v) for k, v in checks.items()}
        return {'time_s': float(data.time),
                'limited_joint_excess_native': max_excess,
                'gripper_coupling_residual_rad': grip,
                'waist_coupling_residual_rad': waist,
                'robot_table_penetration_m': float(penetration),
                'cross_arm_penetration_m': integrity['cross_arm_penetration_m'],
                'robot_scene_penetration_m': integrity['robot_scene_penetration_m'],
                'chassis_world_translation_m': chassis_translation,
                'chassis_world_z_displacement_m': chassis_z,
                'chassis_world_quaternion_distance': chassis_orientation,
                'solver_warning_count': warnings,
                'hard_constraints': checks,
                'all_hard_passed': bool(all(checks.values())),
                'measurements_valid': bool(valid),
                'scope': 'current physical state; not historical maxima'}


def hard_snapshot(model, data):
    """Convenient one-off diagnostic; live use should cache HardSnapshotter."""
    return HardSnapshotter(model)(data)


class HardAccumulator:
    """Latch failures and maxima from an explicitly supplied physical stream."""
    def __init__(self):
        self.checks = {k: True for k in HARD_KEYS}
        self.maxima = {}
        self.valid = True
        self.count = 0
        self.first_time = self.last_time = None

    def observe(self, snapshot):
        now = float(snapshot['time_s'])
        valid = bool(snapshot['measurements_valid'] and math.isfinite(now)
                     and (self.last_time is None or now >= self.last_time))
        self.valid = self.valid and valid
        for k in HARD_KEYS:
            self.checks[k] = bool(self.checks[k] and self.valid
                                  and snapshot['hard_constraints'].get(k) is True)
        for k in ('limited_joint_excess_native', 'gripper_coupling_residual_rad',
                  'waist_coupling_residual_rad', 'robot_table_penetration_m',
                  'cross_arm_penetration_m', 'robot_scene_penetration_m',
                  'solver_warning_count', 'chassis_world_translation_m',
                  'chassis_world_z_displacement_m',
                  'chassis_world_quaternion_distance'):
            value = snapshot[k]
            self.maxima[k] = max(self.maxima.get(k, value), value)
        self.count += 1
        if self.first_time is None:
            self.first_time = now
        self.last_time = now
        return {**snapshot, **self.maxima, 'hard_constraints': dict(self.checks),
                'all_hard_passed': bool(self.valid and all(self.checks.values())),
                'measurements_valid': self.valid, 'physical_observations': self.count,
                'first_observation_time_s': self.first_time,
                'scope': 'cumulative supplied physical observations; caller must supply every step'}
