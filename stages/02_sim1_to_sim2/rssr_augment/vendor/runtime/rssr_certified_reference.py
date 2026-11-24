"""Use the actual certified Sim1 model and commands as target grasp goals."""
import pathlib
import numpy as np
import mujoco
from common import read, write, sha

def reference(config):
    from pilot_physics import hand_pose
    proof = None
    if config.get('source_success_receipt'):
        from rssr_source_gate import require
        proof = require(config, read(config['frozen_scene_path']))
        folder = pathlib.Path(proof['report']).parent
    elif config.get('source_reference_artifact'):
        # A failed Sim1 replay may still contain a valid physical command
        # trace. It is usable as a geometric reference, but is not admitted as
        # a Sim1 success proof.
        folder = pathlib.Path(config['source_reference_artifact'])
        if not (folder/'scene.xml').exists() or not (folder/'command_targets.npz').exists():
            raise ValueError('Source reference artifact is incomplete')
    else:
        raise ValueError('Certified reference needs a source artifact')
    raw = np.load(pathlib.Path(config['import_case']) / 'source_action_state.npz')
    commands = np.load(folder / 'command_targets.npz')
    times, qpos = commands['time_s'], commands['qpos']
    if not np.array_equal(times, raw['timestamp'] - raw['timestamp'][0]):
        raise ValueError('Certified source timestamps do not match original action')
    model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
    data = mujoco.MjData(model)
    if qpos.shape != (len(times), model.nq):
        raise ValueError('Certified source commands/model shape mismatch')
    contract_path = pathlib.Path(config['model_root']) / 'source/scene/task_frame_contract.json'
    contract = read(contract_path)
    ref = {'source_time_s': times.copy()}
    poses = {s: [] for s in ('left', 'right')}
    max_error = 0.
    for qi in qpos:
        data.qpos[:] = qi
        mujoco.mj_forward(model, data)
        for side in poses:
            position, rotation = hand_pose(model, data, 'sim1', side)
            chain = contract['chains'][side]
            grasp = np.eye(4)
            # Translation is already the model's distal grasp site in world;
            # neither source registration nor flange offset is applied twice.
            grasp[:3, 3] = position
            grasp[:3, :3] = rotation @ np.asarray(chain['source_reference_from_grasp'])[:3, :3]
            offset = np.asarray(chain['target_site_from_grasp'])
            target = grasp @ np.linalg.inv(offset)
            max_error = max(max_error, float(np.max(np.abs(target @ offset - grasp))))
            poses[side].append(target)
    for side, index in [('left', 13), ('right', 6)]:
        p = np.asarray(poses[side])
        ref[side + '_target_position'] = p[:, :3, 3]
        ref[side + '_target_rotation'] = p[:, :3, :3]
        ref['source_' + side + '_gripper'] = raw['action'][:, index].copy()
        if not np.allclose(np.clip(raw['action'][:, index]/1.05, 0, 1), commands[side+'_gripper_command'], atol=1e-10):
            raise ValueError('Certified gripper commands differ from original source')
    if max_error > 1e-10:
        raise ValueError('Source/target grasp conversion failed roundtrip')
    write(pathlib.Path(config['run_root']) / 'reference_audit.json', {
        'schema': 'rssr.certified_source_reference.v1', 'frames': len(times),
        'source_model_sha256': sha(folder/'scene.xml'),
        'command_targets_sha256': sha(folder/'command_targets.npz'),
        'frame_contract_sha256': sha(contract_path),
        'source_registration': config.get('source_base_world_shift_m'),
        'reference_type': 'commanded Sim1 FK in the certified scene world frame',
        'timestamps_preserved': True, 'source_actions_modified': False,
        'grasp_transform_roundtrip_max_abs_error': max_error,
        'physical_target_success_not_implied_by_reference': True})
    return ref
