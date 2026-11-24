"""Isolated S1 virtual-planar-base adapter for the v80 physical backend.

The frozen v80 runtime is not edited. This development adapter makes the
three upstream MJCF chassis joints explicit and maps the frozen reference
columns by joint name. It is not a wheel-drive model or an approved score.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np

_runtime = Path(__file__).resolve().parents[1] / 'cobot150_runtime_v80_task_completion_20260923'
if str(_runtime) not in sys.path:
    sys.path.insert(0, str(_runtime))
import pilot_physics as _base


aperture = _base.aperture
_original_target_model = _base.target_model
_original_commands = _base.commands
_lock = threading.RLock()
_JOINTS = (
    ('astribot_chassis_x', 'slide', '1 0 0', '-0.5 0.5', '20000'),
    ('astribot_chassis_y', 'slide', '0 1 0', '-0.5 0.5', '20000'),
    ('astribot_chassis_zrot', 'hinge', '0 0 1', '-1.2 1.2', '5000'),
)


def target_model(config):
    root = _original_target_model(config)
    chassis = root.find("./worldbody/body[@name='chassis_base']")
    if chassis is None or chassis.findall('joint'):
        raise ValueError('Unexpected S1 chassis structure for mobile adapter')
    for index, (name, kind, axis, bounds, _) in enumerate(_JOINTS):
        chassis.insert(index, ET.Element('joint', {
            'name': name, 'type': kind, 'axis': axis, 'range': bounds,
            'limited': 'true', 'armature': '0.1', 'damping': '10'}))
    actuators = ET.SubElement(root, 'actuator')
    for name, _, _, _, gain in _JOINTS:
        ET.SubElement(actuators, 'position', {'name': name, 'joint': name,
                                             'kp': gain, 'forcelimited': 'true',
                                             'forcerange': '-5000 5000'})
    return root


@contextmanager
def _override_backend(*, command_mapper=None):
    # v80 resolves these helpers as module globals inside build/rollout.
    # Serialize this temporary override; full campaign cases run in separate
    # processes, while native and replay use this same deterministic wrapper.
    with _lock:
        old_target, old_commands = _base.target_model, _base.commands
        _base.target_model = target_model
        if command_mapper is not None:
            _base.commands = command_mapper
        try:
            yield
        finally:
            _base.target_model, _base.commands = old_target, old_commands


def build(config, kind, folder, scene=None):
    if kind != 'sim2':
        return _base.build(config, kind, folder, scene)
    with _override_backend():
        return _base.build(config, kind, folder, scene)


def map_reference_commands(config, kind, model, reference):
    names = tuple(config.get('_reference_robot_joint_names', ()))
    if not names or len(set(names)) != len(names):
        raise ValueError('Mobile reference joint-name binding missing')
    if reference is None or reference.ndim != 2 or reference.shape[1] != len(names):
        raise ValueError('Mobile reference columns do not match joint names')
    times, targets, gripper = _original_commands(config, kind, model)
    if targets.shape[0] != reference.shape[0]:
        raise ValueError('Mobile reference trajectory length changed')
    for column, name in enumerate(names):
        joint = model.joint(name)
        targets[:, int(joint.qposadr[0])] = reference[:, column]
    return times, targets, gripper


def rollout(config, kind, folder, scene, target_qpos=None, stop_s=None):
    if kind != 'sim2':
        return _base.rollout(config, kind, folder, scene,
                             target_qpos=target_qpos, stop_s=stop_s)
    def commands(config, kind, model):
        return map_reference_commands(config, kind, model, target_qpos)

    with _override_backend(command_mapper=commands):
        # The frozen runtime's target_qpos argument writes qpos[:, :30].
        # Apply it above by named joints and keep that old positional path off.
        return _base.rollout(config, kind, folder, scene,
                             target_qpos=None, stop_s=stop_s)
