"""URDF-backed Piper wrist camera prior, without changing any model or rollout.

This is a mounting *prior*, not Cobot hardware calibration. The repository's
Piper-L D435 variant has the same +Z tool / Y jaw-plane convention as the
imported source MuJoCo flange, but its arm parameters are not identical.
Its URDF terminates at camera_link and omits an optical frame; the explicit
ROS camera-link convention below is therefore an assumption to be verified
against Real1 by the camera agent, not an observed extrinsic calibration.
"""
import hashlib
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


PRIOR_URDF = (
    'source/models/trackc/piper_isaac_sim/piper_l_description/urdf/'
    'piper_l_description_d435.urdf'
)
# Columns: optical right, optical down, optical forward expressed in a ROS
# camera_link (+X forward, +Y left, +Z up). MuJoCo/OpenGL then flips Y and Z.
CAMERA_LINK_FROM_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
OPTICAL_FROM_OPENGL = np.diag([1., -1., -1.])


def fixed_chain(urdf_path, parent='Link6', child='camera_link'):
    """Compose only fixed URDF joints, returning parent-from-child and evidence."""
    root = ET.parse(urdf_path).getroot()
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    chain = []
    node = child
    seen = set()
    while node != parent:
        if node in seen or node not in by_child:
            raise ValueError('No unambiguous fixed camera chain to ' + parent)
        seen.add(node)
        joint = by_child[node]
        if joint.get('type') != 'fixed':
            raise ValueError('Camera mounting chain contains a moving joint')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        transform[:3, 3] = xyz
        chain.append((transform, {'joint': joint.get('name'), 'parent': joint.find('parent').get('link'),
                                  'child': node, 'xyz_m': xyz.tolist(), 'rpy_rad': rpy.tolist()}))
        node = joint.find('parent').get('link')
    composed = np.eye(4)
    for transform, _ in reversed(chain):
        composed = composed @ transform
    return composed, [item for _, item in reversed(chain)]


def _check_source_flange(model, side):
    """Reject a different flange convention instead of silently transplanting it."""
    name = side + '_wrist_rgbd'
    camera = model.camera(name).id
    parent = int(model.cam_bodyid[camera])
    expected = side + '_link6'
    if model.body(parent).name != expected:
        raise ValueError(name + ' is not attached to ' + expected)
    joint = model.joint(side + '_joint6').id
    if not np.allclose(model.jnt_axis[joint], [0, 0, 1], atol=1e-5):
        raise ValueError('Unexpected source wrist rotation axis')
    jaws = []
    for index in [7, 8]:
        body = model.body(side + '_link' + str(index)).id
        if int(model.body_parentid[body]) != parent:
            raise ValueError('Unexpected source jaw parent')
        axis = model.jnt_axis[model.joint(side + '_joint' + str(index)).id]
        rotation = Rotation.from_quat(model.body_quat[body][[1, 2, 3, 0]]).as_matrix()
        opening_axis = rotation @ axis
        if not np.allclose(np.abs(opening_axis), [0, 1, 0], atol=1e-4):
            raise ValueError('Source jaw plane differs from the Piper-L mounting prior')
        if model.body_pos[body][2] <= .1:
            raise ValueError('Expected grasp geometry along positive flange Z')
        jaws.append({'body': model.body(body).name, 'opening_axis_in_flange': opening_axis.tolist()})
    return {'camera': name, 'parent_body': expected, 'parent_id': parent,
            'flange_convention_checked': True, 'jaw_axes': jaws,
            'exact_robot_model_match': False}


def source_wrist_prior(config, model, fovy_degrees=None):
    """Return (entries, provenance) compatible with camera_agent.apply_profile.

    This function does not mutate model. The caller may retain a known intrinsic
    via fovy_degrees, or inherit each existing model camera's current FOV. The
    URDF supplies no intrinsic calibration. Use only for source Sim1; never
    transplant Piper camera mounts to S1.
    """
    path = pathlib.Path(config['model_root']) / PRIOR_URDF
    transform, chain = fixed_chain(path)
    rotation = transform[:3, :3] @ CAMERA_LINK_FROM_OPTICAL @ OPTICAL_FROM_OPENGL
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError('Improper camera rotation')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = [_check_source_flange(model, side) for side in ['left', 'right']]
    provenance = {
        'schema': 'rssr.source_camera_prior.v1', 'urdf': str(path), 'urdf_sha256': digest,
        'urdf_parent': 'Link6', 'urdf_child': 'camera_link', 'fixed_joints': chain,
        'parent_from_camera_link': transform.tolist(),
        'camera_link_from_optical': CAMERA_LINK_FROM_OPTICAL.tolist(),
        'optical_from_opengl': OPTICAL_FROM_OPENGL.tolist(),
        'optical_frame_explicit_in_urdf': False,
        'optical_frame_assumption': 'camera_link +X forward +Y left +Z up; optical +Z forward +X right +Y down',
        'hardware_calibrated': False, 'exact_robot_model_match': False,
        'basis': 'Piper-L D435 fixed mounting prior with compatible distal flange convention',
        'source_model_checks': checks,
        'limitations': ['Piper-L arm parameters differ from imported source robot',
                       'Actual dataset camera model/mount identity is not established',
                       'URDF camera_link has no RGB lens offset or optical-frame joint',
                       'FOV is an inherited or caller-provided prior, not a URDF measurement'],
    }
    entries = {}
    for side in ['left', 'right']:
        name = side + '_wrist_rgbd'
        fovy = float(model.cam_fovy[model.camera(name).id]) if fovy_degrees is None else float(fovy_degrees)
        if not 20 <= fovy <= 110:
            raise ValueError('Invalid camera FOV')
        entries[name] = {
            'parent_body': side + '_link6', 'position_in_parent_m': transform[:3, 3].tolist(),
            'rotation_in_parent': rotation.tolist(), 'fovy_degrees': fovy,
            'basis': provenance['basis'], 'hardware_calibrated': False,
            'prior_urdf_sha256': digest, 'optical_frame_assumed': True,
        }
    return entries, provenance


if __name__ == '__main__':
    import argparse
    import json
    import mujoco
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--fovy', type=float)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.scene)
    profiles, evidence = source_wrist_prior({'model_root': args.model_root}, model, args.fovy)
    print(json.dumps({'profiles': profiles, 'provenance': evidence}, indent=2))
