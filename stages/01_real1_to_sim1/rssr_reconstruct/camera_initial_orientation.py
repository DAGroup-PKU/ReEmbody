"""First-frame cross-embodiment optical consistency, never hardware calibration.

Source orientation is a current-episode visual estimate. It supplies a necessary
relative consistency check; RGB comparison with Real1 is still mandatory.
Only saved initial states are read. Later wrist world trajectories may differ.
"""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from pilot_camera_agent import CAMERAS, _spec, _transform, apply_camera_edit


POLICY = {'name': 'initial_optical_consistency_v1',
          'max_so3_difference_degrees': 45., 'max_table_up_roll_difference_degrees': 30.,
          'minimum_projected_up_norm': .25,
          'scope': 'same-role initial optical orientation in shared table coordinates',
          'later_wrist_world_pose_equality_required': False,
          'source_reference': 'same_case_estimated_sim1_rig_not_measured_calibration'}


def capture_initial_context(loaded, profiles, table_frames=None):
    import mujoco
    parents = {}
    for kind, (model, data, rollout, mapping) in loaded.items():
        if len(mapping) < 1 or not 0 <= int(mapping[0]) < len(rollout['qpos']):
            raise ValueError('Initial camera check requires a saved first-frame state')
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        table_from_world = np.linalg.inv(_transform((table_frames or {}).get(kind, np.eye(4))))
        try:
            index = int(mapping[0])
            data.qpos[:] = rollout['qpos'][index]
            data.qvel[:] = rollout['qvel'][index]
            data.time = float(rollout['time_s'][index])
            mujoco.mj_forward(model, data)
            parents[kind] = {}
            for view, name in CAMERAS.items():
                body = int(model.cam_bodyid[model.camera(name).id])
                parent_name = model.body(body).name if body else 'world'
                if profiles[kind][name]['parent_body'] != parent_name:
                    raise ValueError('Profile parent does not match the saved-state model')
                world_from_parent = np.eye(4)
                if body:
                    world_from_parent[:3, :3] = data.xmat[body].reshape(3, 3)
                    world_from_parent[:3, 3] = data.xpos[body]
                parents[kind][view] = {'parent_body': parent_name,
                    'table_from_parent': (table_from_world @ world_from_parent).tolist()}
        finally:
            data.qpos[:], data.qvel[:], data.time = before
            mujoco.mj_forward(model, data)
    return {'schema': 'rssr.initial_camera_context.v1', 'parents': parents,
            'saved_initial_state_only': True, 'dynamics_steps': 0}


def table_camera_rotation(profiles, context, kind, view, local_rotation=None):
    entry = context['parents'][kind][view]
    spec = profiles[kind][CAMERAS[view]]
    if spec['parent_body'] != entry['parent_body']:
        raise ValueError('Initial orientation context has another parent')
    parent = _transform(entry['table_from_parent'])[:3, :3]
    return parent @ (_spec(spec)[1] if local_rotation is None else local_rotation)


def comparison(profiles, context, view, target_local_rotation=None):
    source = table_camera_rotation(profiles, context, 'sim1', view)
    target = table_camera_rotation(profiles, context, 'sim2', view, target_local_rotation)
    vector = Rotation.from_matrix(source.T @ target).as_rotvec()
    angle = float(np.rad2deg(np.linalg.norm(vector)))
    source_up = source.T @ np.array([0., 0., 1.])
    target_up = target.T @ np.array([0., 0., 1.])
    enough_up = min(np.linalg.norm(source_up[:2]), np.linalg.norm(target_up[:2])) >= POLICY['minimum_projected_up_norm']
    roll = None
    if enough_up:
        a, b = np.arctan2(source_up[0], source_up[1]), np.arctan2(target_up[0], target_up[1])
        roll = float(abs(np.rad2deg((b-a+np.pi) % (2*np.pi)-np.pi)))
    passed = angle <= POLICY['max_so3_difference_degrees'] + 1e-8 and (
        roll is None or roll <= POLICY['max_table_up_roll_difference_degrees'] + 1e-8)
    return {'view': view, 'so3_difference_degrees': angle,
            'table_up_roll_difference_degrees': roll,
            'table_up_roll_observable': bool(enough_up), 'passed': bool(passed)}


def diagnose(profiles, context):
    if set(profiles) != {'sim1', 'sim2'} or set(context.get('parents', {})) != {'sim1', 'sim2'}:
        raise ValueError('Final orientation check requires both saved-state rigs')
    rows = [comparison(profiles, context, view) for view in CAMERAS]
    return {'policy': POLICY, 'passed': all(r['passed'] for r in rows), 'views': rows,
            'visual_verification_required': True, 'hardware_calibrated': False}


def residual(profiles, context, view, target_local_rotation):
    row = comparison(profiles, context, view, target_local_rotation)
    a = row['so3_difference_degrees'] / POLICY['max_so3_difference_degrees']
    b = (row['table_up_roll_difference_degrees'] or 0.) / POLICY['max_table_up_roll_difference_degrees']
    return [.05*a, 2*max(0., a-1), .05*b, 2*max(0., b-1)]


def align_source_orientation(profiles, origin, kind, view, context, table_frames=None):
    if kind != 'sim2' or view not in ('left_wrist', 'right_wrist'):
        raise ValueError('Source orientation transfer is only for a target wrist')
    desired_table = table_camera_rotation(profiles, context, 'sim1', view)
    parent = _transform(context['parents'][kind][view]['table_from_parent'])[:3, :3]
    current = _spec(profiles[kind][CAMERAS[view]])[1]
    # Express a same-case optical direction in the TARGET parent. Never copy
    # the source flange-local transform to a different robot flange.
    desired_local = parent.T @ desired_table
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        angles = Rotation.from_matrix(current.T @ desired_local).as_euler('xyz', degrees=True)
    edit = {'rotation_camera_xyz_degrees': angles.tolist()}
    result = apply_camera_edit(profiles, origin, kind, view, edit, table_frames)
    return result, {'schema': 'rssr.initial_orientation_fit.v1', 'kind': kind, 'view': view,
                    'before': comparison(profiles, context, view),
                    'after': comparison(result, context, view), 'applied_edit': edit,
                    'translation_and_fov_changed': False, 'parent_changed': False,
                    'same_case_source_orientation': True, 'historical_profile_used': False,
                    'api_requests': 0, 'render_calls': 0, 'dynamics_steps': 0,
                    'visual_verification_required': True}
