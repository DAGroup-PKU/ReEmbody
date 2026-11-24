"""Declared initial geometry/head variants; no object writes during execution."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from .common import stable_seed

HEAD_NAMES = ('astribot_head_joint_1', 'astribot_head_joint_2')

def sample_profile(case_id, index, settings):
    rng = np.random.default_rng(stable_seed(settings['seed'], case_id, index, 'physical'))
    def draw(key):
        return float(rng.uniform(*settings[key]))
    return {'schema': 'rssr.lyc_dev_v2.physical_profile.v1', 'case_id': case_id,
            'variant_index': index, 'seed': stable_seed(settings['seed'], case_id, index, 'physical'),
            'head_delta_rad': [draw('head_yaw_delta_rad'), draw('head_pitch_delta_rad')],
            'table_height_delta_m': draw('table_height_delta_m'),
            'table_length_scale': draw('table_length_scale'), 'table_width_scale': draw('table_width_scale'),
            'object_seed': stable_seed(settings['seed'], case_id, index, 'objects'),
            'free_object_xy_delta_m': settings['free_object_xy_delta_m'],
            'requires_new_physical_execution': True}

def augment_scene(scene, config, profile):
    if profile is None:
        return scene, None
    allowed = {'schema', 'case_id', 'variant_index', 'seed', 'head_delta_rad',
               'table_height_delta_m', 'table_length_scale', 'table_width_scale',
               'object_seed', 'free_object_xy_delta_m', 'requires_new_physical_execution'}
    if set(profile) != allowed or profile['requires_new_physical_execution'] is not True:
        raise ValueError('Undeclared augmentation fields')
    height = float(profile['table_height_delta_m'])
    head = np.asarray(profile['head_delta_rad'], dtype=float)
    if (not np.isfinite(height) or abs(height) > .05 or head.shape != (2,) or
            not np.isfinite(head).all() or np.any(np.abs(head) > .3) or
            not all(.8 <= profile[k] <= 1.2 for k in ('table_length_scale', 'table_width_scale'))):
        raise ValueError('Physical augmentation outside declared conservative envelope')
    lo, hi = profile['free_object_xy_delta_m']
    if not np.isfinite([lo, hi]).all() or not -.05 <= lo <= hi <= .05:
        raise ValueError('Object position randomization exceeds initial-scene envelope')
    if np.any(head != 0.): raise ValueError('No head jitter authorized in measured S1 campaign')
    result = deepcopy(scene)
    rng = np.random.default_rng(profile['object_seed'])
    offsets = {};placed=[];rejections={}
    from .measured_s1 import reference
    setup=reference();center=np.asarray(setup['table_center_xy_m']);half=np.asarray(setup['table_half_xy_m'])
    for obj in result['objects']:
        # Scene positions are world coordinates, including articulated children.
        # Shift fixtures/regions together so table-relative task roles stay intact.
        obj['position_m'][2] += height
        if obj['mobility'] == 'free' and not obj.get('parent_id'):
            radius=float(max(obj['size_m'][:2])/2)
            for attempt in range(64):
                delta=rng.uniform(lo,hi,2);candidate=np.asarray(obj['position_m'][:2])+delta
                if np.all(np.abs(candidate-center)+radius+.005<=half) and all(np.linalg.norm(candidate-prior)>radius+r+.005 for prior,r in placed):break
            else:raise ValueError('No supported nonoverlapping randomized cup layout')
            placed.append((candidate,radius));rejections[obj['id']]=attempt
            obj['position_m'][:2] = (np.asarray(obj['position_m'][:2]) + delta).tolist()
            offsets[obj['id']] = delta.tolist()
    config['_lyc_physical_profile'] = deepcopy(profile)
    # Physical head target, not render-only camera pitch or relabeling.
    config['_lyc_head_target_rad'] = head.tolist()
    return result, {**profile, 'free_object_offsets_xy_m': offsets,
                    'initial_layout_rejections': rejections, 'source_scene': deepcopy(scene), 'augmented_scene': result,
                    'object_shapes_masses_friction_unchanged': True,
                    'task_bindings_unchanged': result['bindings'] == scene['bindings'],
                    'physical_success_verified': False}

def modify_table(root, profile):
    table = root.find(".//body[@name='worktable']")
    if table is None:
        raise ValueError('Native table not found')
    top = table.find(".//geom[@name='worktable_top']")
    if top is None or top.get('type', 'sphere') != 'box':
        raise ValueError('Table augmentation needs the exact native box representation')
    size = np.fromstring(top.get('size'), sep=' ')
    pos = np.fromstring(top.get('pos', '0 0 0'), sep=' ')
    original = size.copy()
    size[:2] *= [profile['table_length_scale'], profile['table_width_scale']]
    dz = profile['table_height_delta_m']
    pos[2] += dz
    def text(values):
        return ' '.join(format(float(x), '.17g') for x in values)
    top.set('size', text(size)); top.set('pos', text(pos))
    for geom in table.findall('.//geom'):
        if not geom.get('name', '').startswith('worktable_leg_'):
            continue
        p = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
        s = np.fromstring(geom.get('size'), sep=' ')
        if s.shape != (3,) or p.shape != (3,):
            raise ValueError('Unsupported native table leg')
        p[:2] += np.sign(p[:2]) * (size[:2] - original[:2])
        p[2] += dz / 2; s[2] += dz / 2
        if np.any(s <= 0):
            raise ValueError('Augmentation invalidates tabletop supports')
        geom.set('pos', text(p)); geom.set('size', text(s))

def install_backend(backend):
    if getattr(backend, '_lyc_augmented', False):
        return
    original_model, original_commands = backend.target_model, backend.commands
    def target_model(config):
        root = original_model(config)
        if config.get('_lyc_physical_profile'):
            modify_table(root, config['_lyc_physical_profile'])
        return root
    def commands(config, kind, model):
        times, targets, gripper = original_commands(config, kind, model)
        if kind == 'sim2' and '_lyc_head_target_rad' in config:
            for name, target in zip(HEAD_NAMES, config['_lyc_head_target_rad']):
                joint = model.joint(name)
                if not model.jnt_limited[joint.id] or not model.jnt_range[joint.id, 0] + .01 <= target <= model.jnt_range[joint.id, 1] - .01:
                    raise ValueError('Head target outside native joint range')
                targets[:, int(joint.qposadr[0])] = target
        return times, targets, gripper
    backend.target_model, backend.commands = target_model, commands
    backend._lyc_augmented = True
