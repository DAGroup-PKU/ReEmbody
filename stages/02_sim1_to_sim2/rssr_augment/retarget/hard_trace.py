"""Persist all trusted physical-step measurements for independent verification."""
from pathlib import Path
import numpy as np
from .immutable_hard_observation import HARD_KEYS

METRICS = ('limited_joint_excess_native', 'gripper_coupling_residual_rad',
           'waist_coupling_residual_rad', 'robot_table_penetration_m',
           'cross_arm_penetration_m', 'robot_scene_penetration_m',
           'solver_warning_count', 'chassis_world_translation_m',
           'chassis_world_z_displacement_m',
           'chassis_world_quaternion_distance')


def save_hard_trace(path, records):
    if not records:
        raise ValueError('No physical hard observations')
    path = Path(path)
    arrays = {k: np.asarray([r[k] for r in records]) for k in ('time_s', *METRICS)}
    arrays['measurements_valid'] = np.asarray([r['measurements_valid'] for r in records], dtype=bool)
    arrays['hard_constraints'] = np.asarray([[r['hard_constraints'][k] for k in HARD_KEYS]
                                            for r in records], dtype=bool)
    arrays['hard_constraint_names'] = np.asarray(HARD_KEYS, dtype='U64')
    np.savez_compressed(path, **arrays)
