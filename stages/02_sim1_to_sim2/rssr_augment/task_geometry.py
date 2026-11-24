"""Bind appearance and review metadata to the actual corrected execution model."""
from copy import deepcopy
from pathlib import Path
import numpy as np

from .common import BRANCH, read, file_sha

TRUTH = BRANCH / 'config/task_geometry_truth_v8.json'


def drawer_structure(scene):
    """Check task semantics independently of misleading legacy body names."""
    truth = read(TRUTH)['drawer']
    objects = {obj['id']: obj for obj in scene['objects']}
    levels = truth['level_body_names_from_top']
    modeled_levels = [name for name, obj in objects.items()
                      if name.endswith('_drawer_front') or obj.get('shape') == 'open_box']
    if any(name not in objects for name in levels):
        raise ValueError('Drawer model must contain four observed levels')
    if set(modeled_levels) != set(levels):
        raise ValueError('Drawer model has extra or unidentified drawer levels')
    ordered = sorted(levels, key=lambda name: objects[name]['position_m'][2], reverse=True)
    if ordered != levels or ordered.index(truth['active_body']) + 1 != truth['active_level_from_top']:
        raise ValueError('Active drawer must be the second level from the bottom')
    active = objects[truth['active_body']]
    if (active['mobility'] != 'slide' or active['joint']['name'] != truth['active_joint']
            or scene['bindings']['articulation'] != truth['active_joint']
            or scene['bindings']['containers'] != [truth['active_body']]
            or any(objects[name]['mobility'] != 'static' for name in levels if name != truth['active_body'])):
        raise ValueError('Drawer articulation/task target differs from the observed active level')
    return {**truth, 'truth_file': str(TRUTH), 'truth_sha256': file_sha(TRUTH)}


def execution_reference_scene(row):
    """Read the scene actually executed, without rewriting any frozen evidence."""
    receipt_path = Path(row['source_receipt']).parent / 'physical_augmentation.json'
    if not receipt_path.is_file():
        raise ValueError('Executed physical augmentation metadata is required')
    receipt = read(receipt_path)
    if receipt['case_id'] != row['case_id'] or receipt.get('task_bindings_unchanged') is not True:
        raise ValueError('Executed scene metadata belongs to a different task')
    scene = deepcopy(receipt['augmented_scene'])
    if file_sha(row['scene']) != row['scene_sha256']:
        raise ValueError('Executed scene XML bytes changed')
    import mujoco
    model = mujoco.MjModel.from_xml_path(row['scene'])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for obj in scene['objects']:
        body = model.body(obj['id'])
        if not np.allclose(data.xpos[body.id], obj['position_m'], rtol=0, atol=1e-8):
            raise ValueError('Executed scene body does not match augmented metadata: ' + obj['id'])
        joint = obj.get('joint')
        if joint:
            actual = model.joint(joint['name'])
            if (int(model.jnt_bodyid[actual.id]) != body.id
                    or int(model.jnt_type[actual.id]) != int(mujoco.mjtJoint.mjJNT_SLIDE)
                    or not np.allclose(model.jnt_axis[actual.id], joint.get('axis', [0, 1, 0]), rtol=0, atol=1e-8)
                    or not np.allclose(model.jnt_range[actual.id], joint['range'], rtol=0, atol=1e-8)):
                raise ValueError('Executed articulation differs from scene metadata')
        elif int(model.body_jntnum[body.id]) and obj['mobility'] == 'static':
            raise ValueError('Fixed task object unexpectedly has an articulation: ' + obj['id'])
    return scene, {'path': str(receipt_path), 'sha256': file_sha(receipt_path),
                   'scene_xml': row['scene'], 'scene_xml_sha256': row['scene_sha256'],
                   'all_object_initial_centers_checked': True}


def bind_execution_scene(row):
    result = deepcopy(row)
    scene, provenance = execution_reference_scene(row)
    result['source_reference_scene'] = scene
    result['source_reference_scene_provenance'] = provenance
    if row['family'] == 'drawer':
        result['task_geometry_truth'] = drawer_structure(scene)
    return result


def verify_drawer_model(row):
    if row['family'] != 'drawer':
        return None
    scene, provenance = execution_reference_scene(row)
    return {**drawer_structure(scene), 'execution_scene': provenance}
