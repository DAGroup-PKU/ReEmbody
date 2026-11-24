"""Reuse an accepted fixed virtual camera rig; never refit it per episode.

Settings: camera_alignment.fixed_profile_from is a completed base run's
camera_alignment directory; fixed_profile_code_root is its sealed code directory.
The base must have its replay manifest and episode bundle. Older accepted profiles
without rig_signature are supported by hashing their verified frozen model assets.
Only p/R/FOV and parent identities are reused, never physics/video/acceptance.
Sim1 physics/task visibility is diagnostic; Sim2 physics/visibility is required.
"""
import copy
import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

from common import read, write, sha
from camera_rig_check import CAMERAS, PARENTS, _spec, source_event_indices


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array(value):
    return np.asarray(value).tolist()


def rig_signature(loaded):
    """Hash robot geometry/kinematic chains, not layouts or fitted camera poses.

    Robot roots are the world children containing the declared wrist parents.
    Object/table/floor bodies are excluded. Camera p/R/FOV are deliberately not
    part of the identity: they are the separately frozen values being inherited.
    Compiled mesh vertices/faces are included, independent of global mesh IDs.
    """
    if set(loaded) != {'sim1', 'sim2'}:
        raise ValueError('Rig signature needs both source and target models')
    kinds = {}
    for kind, values in loaded.items():
        m = values[0]
        cameras, roots = [], set()
        for name in CAMERAS.values():
            cid = m.camera(name).id
            bid = int(m.cam_bodyid[cid])
            parent = m.body(bid).name if bid else 'world'
            if parent != PARENTS[kind][name] or int(m.cam_mode[cid]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
                raise ValueError('Unexpected fixed camera parent/mode: ' + kind + '/' + name)
            cameras.append({'name': name, 'parent': parent, 'mode': int(m.cam_mode[cid])})
            if bid:
                while int(m.body_parentid[bid]):
                    bid = int(m.body_parentid[bid])
                roots.add(bid)
        body_ids = set(roots)
        for bid in range(1, m.nbody):
            if int(m.body_parentid[bid]) in body_ids:
                body_ids.add(bid)
        bodies, joints, geoms = [], [], []
        for bid in sorted(body_ids):
            bodies.append({'name': m.body(bid).name,
                           'parent': m.body(int(m.body_parentid[bid])).name or 'world',
                           'position': _array(m.body_pos[bid]), 'quaternion': _array(m.body_quat[bid])})
        for jid in range(m.njnt):
            if int(m.jnt_bodyid[jid]) in body_ids:
                joints.append({'name': m.joint(jid).name, 'body': m.body(int(m.jnt_bodyid[jid])).name,
                               'type': int(m.jnt_type[jid]), 'position': _array(m.jnt_pos[jid]),
                               'axis': _array(m.jnt_axis[jid]), 'range': _array(m.jnt_range[jid]),
                               'limited': bool(m.jnt_limited[jid])})
        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) not in body_ids:
                continue
            row = {'name': m.geom(gid).name, 'body': m.body(int(m.geom_bodyid[gid])).name,
                   'type': int(m.geom_type[gid]), 'position': _array(m.geom_pos[gid]),
                   'quaternion': _array(m.geom_quat[gid]), 'size': _array(m.geom_size[gid])}
            if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[gid])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                row['mesh'] = {'name': m.mesh(mid).name,
                               'vertices': _digest(_array(m.mesh_vert[va:va+vn])),
                               'faces': _digest(_array(m.mesh_face[fa:fa+fn]))}
            geoms.append(row)
        kinds[kind] = {'camera_parents': cameras, 'robot_root_bodies': sorted(m.body(b).name for b in roots),
                       'bodies': bodies, 'joints': joints, 'geoms': geoms}
    payload = {'schema': 'rssr.robot_camera_rig_signature.v1', 'kinds': kinds}
    return dict(payload, sha256=_digest(payload))


def fixed_entries(cameras):
    """Extract only fixed transforms; unknown inherited behavior is discarded."""
    if not isinstance(cameras, dict) or set(cameras) != {'sim1', 'sim2'}:
        raise ValueError('Expected two embodiments in frozen camera profile')
    result = {}
    for kind in ('sim1', 'sim2'):
        if set(cameras[kind]) != set(CAMERAS.values()):
            raise ValueError('Expected exactly three camera profiles per embodiment')
        result[kind] = {}
        for name, entry in cameras[kind].items():
            p, r, f = _spec(entry)
            if entry.get('parent_body') != PARENTS[kind][name]:
                raise ValueError('Inherited camera side/parent mismatch')
            result[kind][name] = {'parent_body': entry['parent_body'],
                                  'position_in_parent_m': p.tolist(),
                                  'rotation_in_parent': r.tolist(), 'fovy_degrees': f,
                                  'hardware_calibrated': False}
    return result


def _freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen camera input changed: ' + str(path))
    else:
        write(path, value)


def _physics_required(selection, kind):
    trial = pathlib.Path(selection['trial'])
    report = read(trial/'report.json')
    if report.get('scene_sha256') != sha(trial/'scene.xml'):
        raise ValueError('Physics report/model hash mismatch: ' + kind)
    if kind == 'sim2' and (report.get('task_success') is not True or
                          not isinstance(report.get('criteria'), dict) or not report['criteria'] or
                          not all(v is True for v in report['criteria'].values())):
        raise ValueError('Target physical task has not passed')
    return report


def _verify_base(folder, code_root):
    """Read and verify accepted base provenance; never rewrite that run."""
    root = folder.parent
    result = read(folder/'result.json')
    if (result.get('passed') is not True or result.get('visual_passed') is not True or
        result.get('rig_integrity_passed') is not True or result.get('task_visibility_passed') is not True):
        raise ValueError('Base fixed camera rig was not accepted')
    if result.get('profile_sha256') != sha(folder/'camera_profile.json'):
        raise ValueError('Base camera profile hash mismatch')
    from camera_refinement import visual_pass
    if not visual_pass(result.get('verdict')):
        raise ValueError('Base does not contain six accepted K3 camera roles')
    profile = read(folder/'camera_profile.json')
    frozen = read(root/'frozen_run.json')
    for name, digest in frozen['code'].items():
        path = code_root/name
        if not path.resolve().is_relative_to(code_root.resolve()) or sha(path) != digest:
            raise ValueError('Base sealed code mismatch: ' + name)
    for name, digest in frozen['inputs'].items():
        if sha(pathlib.Path(frozen['config']['import_case'])/name) != digest:
            raise ValueError('Base source input changed: ' + name)
    inputs = read(folder/'input.json')
    manifest = read(root/'replay_manifest.json')
    selections = {kind: read(root/kind/'selection.json') for kind in ('sim1', 'sim2')}
    assets = {}
    reports = {}
    with zipfile.ZipFile(root/'RSSR_episode_bundle.zip') as archive:
        for kind, selection in selections.items():
            trial = pathlib.Path(selection['trial'])
            if not trial.resolve().is_relative_to(root.resolve()):
                raise ValueError('Base selection points outside its completed run')
            for name, digest in inputs['stages'][kind].items():
                if sha(trial/name) != digest:
                    raise ValueError('Base camera physics changed: ' + kind + '/' + name)
            reports[kind] = _physics_required(selection, kind)
            for name in ('report.json', 'scene.xml', 'rollout.npz', 'command_targets.npz'):
                if hashlib.sha256(archive.read(kind+'/'+name)).hexdigest() != sha(trial/name):
                    raise ValueError('Base archived physics changed: ' + kind + '/' + name)
            scene = trial/'scene.xml'
            if manifest['scenes'][kind]['original_xml_sha256'] != sha(scene):
                raise ValueError('Replay manifest scene changed')
            expected = {manifest['assets'][a]['sha256'] for a in manifest['scenes'][kind]['assets']}
            xml = ET.parse(scene).getroot()
            if xml.findall('.//include'):
                raise ValueError('Flatten model includes before fixed rig reuse')
            compiler = xml.find('compiler')
            found = set()
            for node in xml.findall('asset/mesh') + xml.findall('asset/texture'):
                if not node.get('file'):
                    continue
                path = pathlib.Path(node.get('file'))
                if not path.is_absolute():
                    prefix = compiler.get('meshdir' if node.tag == 'mesh' else 'texturedir', '') if compiler is not None else ''
                    path = scene.parent/prefix/path
                digest = sha(path)
                if digest not in expected:
                    raise ValueError('Base mesh/texture not in the frozen replay asset set')
                found.add(digest)
                assets[str(path)] = digest
            if found != expected:
                raise ValueError('Base replay asset set differs from model assets')
    provenance = {'camera_result_sha256': sha(folder/'result.json'),
                  'camera_profile_sha256': sha(folder/'camera_profile.json'),
                  'frozen_run_sha256': sha(root/'frozen_run.json'),
                  'replay_manifest_sha256': sha(root/'replay_manifest.json'),
                  'sealed_code_sha256': _digest(frozen['code']), 'sealed_code_root': str(code_root),
                  'verified_assets': assets, 'report_sha256': {k: sha(pathlib.Path(s['trial'])/'report.json') for k, s in selections.items()}}
    return profile, frozen['config'], selections, provenance


def run_fixed(config, budget, root):
    """Verify a new case with an inherited fixed rig, without fitting or repair.

    This function contains one existing bounded K3 verification call. Tests mock
    that helper; importing the module or computing signatures never calls APIs.
    """
    from camera_refinement import (models, native_check, reference_frames, observations,
                                   render_samples, sheets, visual_pass, camera_decide)
    from camera_render_fit import evaluate_rendered_camera
    root = pathlib.Path(root)
    folder = root/'camera_alignment'
    folder.mkdir(parents=True, exist_ok=True)
    settings = config['camera_alignment']
    if config.get('source_task_success_required', False):
        raise ValueError('Fixed-profile protocol uses Sim1 diagnostic / Sim2 required task scope')
    base = pathlib.Path(settings['fixed_profile_from']).resolve()
    if base == folder.resolve():
        raise ValueError('A case cannot inherit its own camera acceptance')
    base_code = pathlib.Path(settings['fixed_profile_code_root']).resolve()
    inherited, base_config, base_selections, provenance = _verify_base(base, base_code)
    profiles = fixed_entries(inherited['cameras'])
    origins = fixed_entries(inherited['origin_cameras'])
    selections = {k: read(root/k/'selection.json') for k in ('sim1', 'sim2')}
    for kind, selection in selections.items():
        _physics_required(selection, kind)
    reference = pathlib.Path(config['import_case'])/'real1_condition.mp4'
    mapping_path = pathlib.Path(config['import_case'])/'frame_action_mapping.npz'
    evidence_path = root/'source_construction/evidence/action_evidence.json'
    loaded = base_loaded = None
    try:
        loaded = models(config, selections)
        base_loaded = models(base_config, base_selections)
        signature = rig_signature(loaded)
        base_signature = rig_signature(base_loaded)
        recorded = inherited.get('rig_signature')
        if recorded is not None and recorded != base_signature:
            raise ValueError('Recorded base rig signature differs from its verified models')
        if signature != base_signature:
            raise ValueError('New case uses a different robot/camera kinematic or geometry rig')
        provenance['rig_signature_source'] = 'recorded_profile' if recorded is not None else 'verified_frozen_models_fallback'
        provenance['base_rig_signature_sha256'] = base_signature['sha256']
        with np.load(mapping_path) as mapping:
            count = len(mapping['source_frame_index'])
            events = source_event_indices(read(evidence_path)['events'], mapping['source_time_s'])
        heldout = np.round(np.array([.09, .27, .45, .63, .87])*(count-1)).astype(int).tolist()
        if len(set(heldout)) != 5:
            raise ValueError('Episode too short for five distinct verification frames')
        inputs = {'real1': sha(reference), 'settings': settings, 'fixed_profile_provenance': provenance,
                  'frame_mapping_sha256': sha(mapping_path), 'action_evidence_sha256': sha(evidence_path),
                  'task_contract_sha256': sha(root/'task_contract.json'),
                  'stages': {k: {name: sha(pathlib.Path(s['trial'])/name)
                                 for name in ('scene.xml', 'rollout.npz', 'report.json')}
                             for k, s in selections.items()}}
        _freeze(folder/'input.json', inputs)
        _freeze(folder/'native_event_mapping.json', events)
        _freeze(folder/'split.json', {'train': [], 'heldout': heldout, 'selected_before_optimization': True,
                                     'no_per_case_optimization': True})
        profile = {'schema': 'rssr.camera_profile.v2', 'cameras': profiles, 'origin_cameras': origins,
                   'rig_signature': signature, 'hardware_calibrated': False,
                   'world_or_parent_fixed_for_entire_episode': True,
                   'role': 'inherited fixed virtual rig; new case independently verified',
                   'fixed_profile_provenance': provenance}
        _freeze(folder/'camera_profile.json', profile)
        profile_hash = sha(folder/'camera_profile.json')
        _freeze(folder/'profile_frozen_before_validation.json', {'sha256': profile_hash,
                                                               'training_only_history': [],
                                                               'new_case_fitted_frames': []})
        if (folder/'result.json').exists():
            cached = read(folder/'result.json')
            if cached.get('profile_sha256') != profile_hash:
                raise ValueError('Cached inherited camera profile changed')
            for name, digest in cached['evidence_sha256'].items():
                if sha(folder/name) != digest:
                    raise ValueError('Cached new-case camera evidence changed')
            return cached
        rig, facts = native_check(config, root, loaded, profiles, origins, selections, heldout)
        if rig['task_visibility'].get('scope') != 'sim2_only':
            raise ValueError('Native checker has not implemented Sim2-only task visibility scope')
        write(folder/'rig_report.json', rig)
        references = reference_frames(reference, heldout)
        rows = observations(loaded, selections, references, heldout)
        reports, bbox_passed = {}, True
        for kind in profiles:
            reports[kind] = {}
            for role, name in CAMERAS.items():
                report = evaluate_rendered_camera(loaded[kind], name, rows[kind][role], profiles[kind][name])
                before = evaluate_rendered_camera(loaded[kind], name, rows[kind][role], origins[kind][name])
                report['before_median_px'] = before['median_px']
                report['no_regression'] = bool(report['median_px'] is not None and before['median_px'] is not None
                                             and report['median_px'] <= before['median_px']+2)
                report['passed'] = bool(report['passed'] and report['no_regression'])
                report['diagnostic_only'] = True
                reports[kind][role] = report
                bbox_passed = bool(bbox_passed and report['passed'])
        write(folder/'geometry_report.json', reports)
        write(folder/'heldout_observations.json', rows)
        rendered = render_samples(loaded, profiles, heldout)
        verdict = camera_decide(config, budget, folder/'verification',
                                (pathlib.Path(__file__).parent/'prompts/camera_alignment.txt').read_text(),
                                {'task': read(root/'task_contract.json')['task'],
                                 'camera_constraint': 'unchanged inherited fixed rig; no case-specific fitting',
                                 'calibration_status': 'virtual fixed rig, not measured hardware extrinsics',
                                 'native_sim_visibility': facts, 'source_task_success_required': False},
                                sheets(folder, references, rendered, heldout, 'heldout'))
        visual = visual_pass(verdict)
        result = {'schema': 'rssr.camera_alignment.v3', 'passed': bool(rig['passed'] and visual),
                  'reference_bbox_passed': bbox_passed, 'visual_passed': visual,
                  'rig_integrity_passed': rig['rig_integrity']['passed'],
                  'task_visibility_passed': rig['task_visibility']['passed'], 'task_visibility_scope': 'sim2_only',
                  'acceptance_protocol': 'fixed_rig_reuse_role_consistency_v1',
                  'reference_bbox_is_calibration_ground_truth': False,
                  'acceptance_scope': 'same fixed virtual rig, six camera roles, target task visibility',
                  'hardware_calibrated': False, 'profile': str(folder/'camera_profile.json'),
                  'profile_sha256': profile_hash, 'native_event_mapping_sha256': sha(folder/'native_event_mapping.json'),
                  'verdict': verdict, 'physics_and_actions_modified': False,
                  'fitted_frames': [], 'heldout_frames': heldout, 'inherited_acceptance': False,
                  'fixed_profile_from': str(base), 'new_case_camera_parameters_modified': False,
                  'state': 'fixed_rig_and_role_verified' if rig['passed'] and visual else 'camera_match_not_verified',
                  'evidence_sha256': {name: sha(folder/name) for name in
                                     ('native_event_mapping.json', 'rig_report.json', 'geometry_report.json', 'heldout_observations.json')}}
        for kind, selection in selections.items():
            for name, digest in inputs['stages'][kind].items():
                if sha(pathlib.Path(selection['trial'])/name) != digest:
                    raise ValueError('Fixed camera verification changed physical evidence')
        write(folder/'result.json', result)
        return result
    finally:
        for values in (loaded, base_loaded):
            if values:
                for m, d, z, ix in values.values():
                    z.close()
