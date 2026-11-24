"""Versioned engineering camera mounts, independent of episodes and learning.

One offline build fixes camera roll in the asset's qpos0 reference posture.
The optical axis, position and FOV are preserved. Runtime never levels a wrist
to gravity: its fixed local transform follows its physical parent. This is an
unmeasured virtual rig, not recovered Real1 calibration.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from common import read, sha
from pilot_camera_agent import (CAMERAS, DEFAULT_PARENTS, _spec, _transform,
                                _rotation, digest)

POLICY = 'asset_reference_posture_fixed_mount_v1'
SCHEMA = 'rssr.camera_mount_config.v1'


def chain_signature(model, kind):
    """Only mounting-chain geometry; exclude objects, states and joint limits."""
    out = {}
    for name in CAMERAS.values():
        cid = model.camera(name).id
        bid = int(model.cam_bodyid[cid])
        parent = model.body(bid).name if bid else 'world'
        if parent != DEFAULT_PARENTS[kind][name] or int(model.cam_mode[cid]) != 0:
            raise ValueError('Unexpected camera parent/mode: '+kind+'/'+name)
        bodies = []
        while bid:
            joints = []
            for jid in range(model.njnt):
                if int(model.jnt_bodyid[jid]) == bid:
                    joints.append({'name':model.joint(jid).name,
                                   'type':int(model.jnt_type[jid]),
                                   'pos':model.jnt_pos[jid].tolist(),
                                   'axis':model.jnt_axis[jid].tolist()})
            bodies.append({'name':model.body(bid).name,
                           'position':model.body_pos[bid].tolist(),
                           'quaternion':model.body_quat[bid].tolist(), 'joints':joints})
            bid = int(model.body_parentid[bid])
        out[name] = {'parent':parent, 'chain':bodies}
    return digest(out)


def upright_reference(rotation_world, table_up, table_forward):
    """Roll-only normalization at ONE model reference pose, never per frame."""
    old = _rotation(rotation_world)
    backward = old[:, 2]
    up = np.asarray(table_up, float)
    up = up-backward*np.dot(up, backward)
    fallback = np.linalg.norm(up) < 1e-6
    if fallback:
        up = np.asarray(table_forward, float)
        up = up-backward*np.dot(up, backward)
    if np.linalg.norm(up) < 1e-6:
        raise ValueError('Reference up and forward cannot define image roll')
    up /= np.linalg.norm(up)
    return _rotation(np.column_stack((np.cross(up, backward), up, backward))), fallback


def build(models, asset_files, table_frames=None):
    import mujoco
    if set(models) != {'sim1', 'sim2'}:
        raise ValueError('Both canonical embodiment models required')
    entries, receipts, signatures = {}, {}, {}
    tables = {k:_transform((table_frames or {}).get(k,np.eye(4))) for k in models}
    for kind, model in models.items():
        signatures[kind] = chain_signature(model, kind)
        data = mujoco.MjData(model)  # qpos0, NEVER a source episode or rollout
        mujoco.mj_forward(model, data)
        entries[kind], receipts[kind] = {}, {}
        for role, name in CAMERAS.items():
            cid = model.camera(name).id; bid = int(model.cam_bodyid[cid])
            old = data.cam_xmat[cid].reshape(3,3).copy()
            world, fallback = upright_reference(old, tables[kind][:3,2], tables[kind][:3,1])
            parent_r = data.xmat[bid].reshape(3,3)
            spec = {'parent_body':DEFAULT_PARENTS[kind][name],
                    'position_in_parent_m':model.cam_pos[cid].tolist(),
                    'rotation_in_parent':(parent_r.T@world).tolist(),
                    'fovy_degrees':float(model.cam_fovy[cid]),
                    'basis':POLICY, 'hardware_calibrated':False,
                    'provenance':{'reference_pose':'canonical_asset_qpos0',
                                  'episode_or_library_input':False}}
            # External entries are stored in the table frame, transformed only
            # when the table frame is placed in a simulation world.
            if role == 'head':
                spec['position_in_parent_m'] = (tables[kind][:3,:3].T@(data.cam_xpos[cid]-tables[kind][:3,3])).tolist()
                spec['rotation_in_parent'] = (tables[kind][:3,:3].T@world).tolist()
            _spec(spec); entries[kind][name] = spec
            receipts[kind][name] = {'reference_world_rotation_before':old.tolist(),
                'reference_world_rotation_after':world.tolist(),
                'optical_axis_preserved':bool(np.allclose(old[:,2],world[:,2],atol=1e-10)),
                'near_vertical_forward_fallback':bool(fallback)}
        receipts[kind]['reference_qpos0'] = model.qpos0.tolist()
    entries['sim2'][CAMERAS['head']] = deepcopy(entries['sim1'][CAMERAS['head']])
    config = {'schema':SCHEMA, 'policy':POLICY, 'cameras':entries,
              'chain_signatures':signatures, 'reference_pose_receipts':receipts,
              'asset_files':{str(Path(p).resolve()):sha(p) for p in asset_files},
              'head_storage_frame':'table', 'wrist_storage_frame':'own_parent',
              'hardware_calibrated':False, 'real1_extrinsics_available':False,
              'per_episode_fitting':False, 'per_frame_leveling':False}
    validate(config)
    return config


def validate(config):
    if config.get('schema') != SCHEMA or config.get('policy') != POLICY:
        raise ValueError('Unsupported fixed camera configuration')
    if (config.get('hardware_calibrated') is not False or
            config.get('per_episode_fitting') is not False or config.get('per_frame_leveling') is not False):
        raise ValueError('This policy requires unmeasured fixed engineering mounts')
    if config.get('head_storage_frame') != 'table' or config.get('wrist_storage_frame') != 'own_parent':
        raise ValueError('Wrong camera storage frame')
    if set(config['cameras']) != {'sim1','sim2'}:
        raise ValueError('Both embodiment profiles required')
    for kind, entries in config['cameras'].items():
        if set(entries) != set(CAMERAS.values()):raise ValueError('Three named cameras required')
        if len(config['chain_signatures'][kind]) != 64:raise ValueError('Missing mount chain identity')
        for name, spec in entries.items():
            _spec(spec)
            if spec['parent_body'] != DEFAULT_PARENTS[kind][name]:raise ValueError('Wrong camera parent')
    if config['cameras']['sim1'][CAMERAS['head']] != config['cameras']['sim2'][CAMERAS['head']]:
        raise ValueError('External camera must be shared in table coordinates')
    return config


def load(path):
    config = validate(read(path))
    for asset, identity in config['asset_files'].items():
        if sha(asset) != identity:raise ValueError('Fixed camera source asset changed: '+asset)
    return config


def profiles_for(config, models, table_frames=None):
    validate(config)
    out = {}
    for kind, model in models.items():
        if chain_signature(model,kind) != config['chain_signatures'][kind]:
            raise ValueError('Embodiment mounting chain differs from frozen camera config: '+kind)
        out[kind] = deepcopy(config['cameras'][kind])
        table = _transform((table_frames or {}).get(kind,np.eye(4)))
        head = out[kind][CAMERAS['head']]; p,r,_ = _spec(head)
        head['position_in_parent_m'] = (table[:3,:3]@p+table[:3,3]).tolist()
        head['rotation_in_parent'] = (table[:3,:3]@r).tolist()
    return out


def diagnose_mounts(loaded, profiles, rendered):
    """Check rendered world poses against parent FK without editing cameras."""
    import mujoco
    rows = []
    for kind, (model, _, rollout, indices) in loaded.items():
        data = mujoco.MjData(model)
        for frame, views in rendered[kind].items():
            data.qpos[:] = rollout['qpos'][int(indices[int(frame)])]
            mujoco.mj_forward(model,data)
            for role, item in views.items():
                name = CAMERAS[role]; p,r,_ = _spec(profiles[kind][name])
                cid = model.camera(name).id; bid = int(model.cam_bodyid[cid])
                parent_r = data.xmat[bid].reshape(3,3)
                pe = float(np.linalg.norm(np.asarray(item['camera_position_world_m'])-(data.xpos[bid]+parent_r@p)))
                re = float(np.max(np.abs(np.asarray(item['camera_rotation_world'])-parent_r@r)))
                rows.append({'kind':kind,'view':role,'frame':int(frame),
                             'position_error_m':pe,'rotation_matrix_error':re,
                             'passed':bool(pe<1e-8 and re<1e-8)})
    return {'passed':bool(rows) and all(r['passed'] for r in rows), 'views':rows,
            'policy':POLICY, 'runtime_world_leveling':False,
            'wrist_world_orientation_may_change':True, 'real1_alignment_certified':False}
