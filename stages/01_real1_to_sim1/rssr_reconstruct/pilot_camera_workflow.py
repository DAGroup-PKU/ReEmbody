"""Actual camera-stage wrapper for the new RSSR pilot, using its shared ledger.

Only current-episode raw video, freshly built geometry and saved physical states
are read. Sim1 task success is never an input. No old camera fit or old recipe
entry point is imported; source mounts remain explicit engineering priors.
"""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from common import read, write, sha
from rssr_evidence_cache import load as load_evidence, save as save_evidence
from pilot_api import call, propose
from protocol_decision import DecisionOutputExhausted
from camera_initial_orientation import (POLICY as INITIAL_ORIENTATION_POLICY,
    capture_initial_context, diagnose as diagnose_initial_orientation)
from camera_clearance import POLICY as CAMERA_CLEARANCE_POLICY
from camera_interaction import POLICY as CAMERA_INTERACTION_POLICY, validate_saved_contacts
from camera_landmarks import LANDMARK_PROMPT, validate_landmark, BOX_SIGNS
from pilot_camera_agent import (CAMERAS, digest, model_profiles, initialize_profiles,
    apply_camera_edit, diagnose_camera_profiles, visual_verdict_passed,
    render_profile_frames)


def _landmarks(config, root, scene):
    case = Path(config['import_case'])
    images = [case/(role+'_first.jpg') for role in CAMERAS]
    contract = {'images': {role: sha(path) for role, path in zip(CAMERAS, images)},
                'objects': [{'id': o['id'], 'shape': o['shape'], 'rgba': o.get('rgba')}
                            for o in scene['objects']],
                'task': config['task']}
    folder = root/'camera_reference_observations'
    if (folder/'input.json').exists():
        old = read(folder/'input.json')
        # Geometry positions may change during reconstruction, but this cached
        # observation only identifies real-image semantic objects and pixels.
        if old != contract:
            raise ValueError('Real1 camera reference images or object identities changed')
        if (folder/'landmarks.json').exists():
            return read(folder/'landmarks.json')
        if not config.get('autonomous_recovery'):
            raise RuntimeError('Landmark request incomplete; recover shared API artifact before retry')
    write(folder/'input.json', contract)
    try:
        response = call(config, folder/'decision', LANDMARK_PROMPT,
                        {'task': config['task'], 'image_order': list(CAMERAS),
                         'objects': contract['objects']}, images, stage='camera.observe')
    except DecisionOutputExhausted as error:
        from rssr_errors import StageUnresolved
        raise StageUnresolved('camera','No landmarks within the fixed model-output budget') from error
    views = response.get('views', [])
    if not isinstance(views, list) or len(views) != 3 or not all(isinstance(v, dict) for v in views) or {v.get('view') for v in views} != set(CAMERAS):
        raise ValueError('Landmark response requires three distinct actual views')
    objects = {o['id'] for o in scene['objects']}
    normalized = {}
    for view in views:
        rows = view.get('landmarks', [])
        if not isinstance(rows, list):
            raise ValueError('Landmark list required')
        known = {}
        for row in rows:
            if not isinstance(row, dict) or row.get('object_id') not in objects or row['object_id'] in known:
                raise ValueError('Unknown or duplicated landmark object')
            known[row['object_id']] = validate_landmark(row)
        normalized[view['view']] = {
            oid: known.get(oid, {'object_id': oid, 'visible': False,
                                'center_xy_normalized': None, 'reason': 'not identified in image'})
            for oid in sorted(objects)}
    output = {'views': normalized, 'observed_source_frame_index': 0,
              'estimated_2d_landmarks': True, 'measured_calibration': False,
              'unknown_coordinates_inferred': False, 'input_sha256': digest(contract)}
    write(folder/'landmarks.json', output)
    return output


def _load(config, scene, folder, final, selections):
    from object_physics import build, commands
    if scene.get('schema')=='rssr.scene.v2':
        from pilot_physics import build, commands
    imported = Path(config['import_case'])
    with np.load(imported/'frame_action_mapping.npz') as z:
        source_indices = np.asarray(z['source_frame_index'], int)
        source_times = np.asarray(z['source_time_s'], float)
    if final and not selections:
        raise ValueError('Final camera QA requires current Sim1 and Sim2 physical selections')
    frames = sorted(set([0, (len(source_indices)-1)//2, len(source_indices)-1])) if final else [0]
    loaded, provenance, file_hashes, mapping = {}, {}, {}, {}
    stages = ('sim1', 'sim2') if final else ('sim1',)
    for stage in stages:
        if final:
            trial = Path(selections[stage]['trial'])
            model = mujoco.MjModel.from_xml_path(str(trial/'scene.xml'))
            rollout = np.load(trial/'rollout.npz')
            try:
                validate_saved_contacts(rollout, [o['id'] for o in scene['objects']])
            except Exception:
                rollout.close()
                raise
            # Existing sample selection only: no interpolation, resampling or
            # forcing source timestamps onto a different executed trajectory.
            indices = np.maximum(np.searchsorted(rollout['time_s'], source_times, side='right')-1, 0)
            file_hashes[stage] = {str(trial/name): sha(trial/name) for name in ('scene.xml', 'rollout.npz')}
            times = np.asarray(rollout['time_s'])[indices[frames]]
        else:
            trial = folder/'initial_source_state'
            model = build(config, stage, trial, scene)
            _, command_q, _ = commands(config, stage, model)
            rollout = {'qpos': command_q[:1].copy(), 'qvel': np.zeros((1, model.nv)), 'time_s': np.array([0.])}
            indices = np.array([0])
            times = np.array([0.])
            file_hashes[stage] = {str(trial/'scene.xml'): sha(trial/'scene.xml')}
        loaded[stage] = (model, mujoco.MjData(model), rollout, indices)
        provenance[stage] = {'kind': 'model_asset', 'artifact_sha256': sha(trial/'scene.xml'),
                             'contains_learned_fit': False,
                             'basis': 'fresh case geometry and declared model mounting prior'}
        mapping[stage] = {'source_frame_index': source_indices[frames].tolist(),
                          'source_time_s': source_times[frames].tolist(),
                          'simulation_time_s': times.tolist(), 'sampling_tolerance_s': .02}
    return loaded, provenance, file_hashes, mapping, frames, source_indices, source_times


def _real_reference_images(config, frame_indices, source_indices, folder):
    case = read(Path(config['import_case'])/'case.json')
    folder.mkdir(parents=True, exist_ok=True)
    images = {frame: {} for frame in frame_indices}
    for role in CAMERAS:
        cap = cv2.VideoCapture(case['source_videos'][role])
        if not cap.isOpened():
            raise RuntimeError('Cannot open current raw Real1 '+role+' video')
        try:
            for frame in frame_indices:
                raw = int(source_indices[frame])
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError('Missing raw Real1 frame '+str(raw)+' in '+role)
                path = folder/f'real1_{raw:06d}_{role}.png'
                if not cv2.imwrite(str(path), cv2.resize(bgr, (480, 360))):
                    raise RuntimeError('Failed to save Real1 reference image')
                images[frame][role] = path
        finally:
            cap.release()
    return images


def _sheets(references, rendered, frames, source_indices, source_times, folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    stages = ['real1']+list(rendered)
    for role in CAMERAS:
        # One role per image: compare Real1/Sim1/Sim2 beside each other and
        # follow the same view through time. Pixel evidence is unchanged.
        sheet = np.zeros((390*len(frames), 480*len(stages), 3), np.uint8)
        for i, frame in enumerate(frames):
            for j, stage in enumerate(stages):
                path = references[frame][role] if stage == 'real1' else rendered[stage][frame][role]['image']
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError('Missing current camera evidence image')
                sheet[390*i+30:390*i+390, 480*j:480*(j+1)] = cv2.resize(image, (480, 360))
                label = f'{stage.upper()}/{role} source_frame={source_indices[frame]} t={source_times[frame]:.3f}'
                cv2.putText(sheet, label, (480*j+5, 390*i+20), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1)
        path = folder/f'view_{role}.jpg'
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError('Failed to save visual evidence sheet')
        paths.append(path)
    return paths


def _observations(loaded, scene, landmarks, frames, source_indices):
    observations = []
    for stage, (model, data, rollout, mapping) in loaded.items():
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        try:
            for frame in frames:
                index = int(mapping[frame])
                data.qpos[:] = rollout['qpos'][index]
                data.qvel[:] = rollout['qvel'][index]
                mujoco.mj_forward(model, data)
                for role, name in CAMERAS.items():
                    cid = model.camera(name).id
                    parent = int(model.cam_bodyid[cid])
                    for obj in scene['objects']:
                        reference = landmarks['views'][role][obj['id']]
                        # Missing reference observations remain unknown; they
                        # never become fabricated camera calibration points.
                        if frame == 0 and not reference['visible']:
                            continue
                        point = data.xpos[model.body(obj['id']).id].copy()
                        local = (point-data.xpos[parent]) @ data.xmat[parent].reshape(3, 3) if parent else point
                        row = {'kind': stage, 'view': role, 'object_id': obj['id'],
                               'source_frame_index': int(source_indices[frame]),
                               'point_in_parent_m': local.tolist(), 'image_size_wh': [480, 360],
                               'static_landmark': False}
                        if frame == 0:
                            if reference.get('bbox_xyxy_normalized') is not None:
                                body = model.body(obj['id']).id
                                corners = (BOX_SIGNS*np.asarray(obj['size_m'])/2) @ data.xmat[body].reshape(3, 3).T + point
                                row['bounds_in_parent_m'] = ((corners-data.xpos[parent]) @ data.xmat[parent].reshape(3, 3) if parent else corners).tolist()
                                row['bbox_xyxy'] = (np.asarray(reference['bbox_xyxy_normalized'])*[480, 360, 480, 360]).tolist()
                                row['evidence_type'] = 'observed_visible_extent_not_hidden_center'
                            else:
                                row['pixel_xy'] = (np.array(reference['center_xy_normalized'])*[480, 360]).tolist()
                        observations.append(row)
        finally:
            data.qpos[:], data.qvel[:], data.time = before
            mujoco.mj_forward(model, data)
    return observations


def _score(report, verdict, stages):
    from rssr_camera_learning import visual_checks
    bad_visual = sum(value is not True for value in visual_checks(verdict,stages).values())
    missing = sum(not v['has_initial_visible_landmark'] or not v['projection_correspondences'] for v in report['views'])
    projections = [v['median_normalized_projection_error'] for v in report['views'] if v['median_normalized_projection_error'] is not None]
    failed_projection = sum(v > .15 for v in projections)
    score = (1e6*len(report['rig']['errors'])+1e5*len(report['errors'])+1e4*missing+
             5000*failed_projection+1000*bad_visual+100*(float(np.mean(projections)) if projections else 1.))
    score+=5000*len(report.get('raster_visibility',{}).get('issues',[]))
    for row in report.get('initial_optical_orientation',{}).get('views',[]):
        score += 10000*int(not row['passed']) + row['so3_difference_degrees']
    # A malformed top-level verdict never gets the same zero visual penalty as
    # a fully inspectable complete verdict, even if individual flags look true.
    if not visual_verdict_passed(verdict, stages) and bad_visual == 0:
        score += 1000
    return float(score)


def run(config, knowledge, root, scene, profiles=None, final=False, selections=None, scene_review=None):
    root = Path(root)
    fixed_mount_path = config.get('fixed_camera_config')
    fixed_mount = None
    if fixed_mount_path:
        from camera_mount_config import load as load_mount_config
        fixed_mount = load_mount_config(fixed_mount_path)
    scene_hash = digest(scene)
    folder = root/'camera_alignment' if final else root/'camera_source'/scene_hash[:16]
    folder.mkdir(parents=True, exist_ok=True)
    settings = {'scene_sha256': scene_hash, 'final': bool(final),
                'scene_review_sha256':digest(scene_review),
                'source_profiles_sha256': digest(profiles),
                'mapping_sha256': sha(Path(config['import_case'])/'frame_action_mapping.npz'),
                'case_metadata_sha256': sha(Path(config['import_case'])/'case.json'),
                'source_actions_sha256': sha(Path(config['import_case'])/'source_action_state.npz'),
                'initial_target_model_sha256': sha(Path(config['import_case'])/'sim2_robot_only_scene.xml'),
                'task': config['task'],
                'physical_selections': {kind: {name: sha(Path(value['trial'])/name)
                                               for name in ('scene.xml', 'rollout.npz')}
                                        for kind, value in (selections or {}).items()} if final else None,
                'max_candidates': 1 if fixed_mount else int(config.get('max_camera_agent_rounds', 4)),
                'initial_orientation_policy':('fixed_mount_parent_fk' if fixed_mount else INITIAL_ORIENTATION_POLICY) if final else None,
                'visual_layout':'one_role_per_sheet_columns_real1_sim1_sim2_rows_time_v1',
                'render_policy':'rgb_and_ids_hide_diagnostic_sites_v1',
                'visibility_search_policy':'first_middle_last_saved_states_soft_occlusion_v1',
                'camera_clearance_policy':CAMERA_CLEARANCE_POLICY,
                'camera_interaction_policy':CAMERA_INTERACTION_POLICY,
                'learning_feedback_policy':'named_camera_feedback_v1',
                'visual_review_policy':'only_after_required_numeric_checks_pass'}
    if fixed_mount:
        settings.update(fixed_camera_config_sha256=sha(fixed_mount_path),
                        camera_policy=fixed_mount['policy'],per_case_camera_adjustment=False)
    if settings['max_candidates'] < 1:
        raise ValueError('Camera candidate budget must be positive')
    if (folder/'input.json').exists():
        if read(folder/'input.json') != settings:
            raise ValueError('Camera workflow inputs changed in an existing candidate directory')
        if (folder/'result.json').exists():
            previous = read(folder/'result.json')
            if sha(folder/'camera_profile.json') != previous['profile_sha256']:
                raise ValueError('Completed camera profile changed')
            return previous
        if not config.get('autonomous_recovery'):
            raise RuntimeError('Incomplete camera workflow; recover rather than duplicate billed requests')
    write(folder/'input.json', settings)
    landmarks = _landmarks(config, root, scene)
    loaded, provenance, file_hashes, mapping, frames, source_indices, source_times = _load(config, scene, folder, final, selections)
    try:
        raw = model_profiles({k: value[0] for k, value in loaded.items()}, provenance)
        # A same-case source registration is allowed to carry forward into
        # target QA. Its inputs/profiles are hashed above. Target mounts always
        # come from the fresh target asset, never another episode's fit.
        if profiles is not None and not fixed_mount:
            supplied = profiles.get('cameras', profiles)
            if 'sim1' in supplied:
                for name in CAMERAS.values():
                    if supplied['sim1'][name]['parent_body'] != raw['sim1'][name]['parent_body']:
                        raise ValueError('Current source profile changed the raw parent binding')
                raw['sim1'] = deepcopy(supplied['sim1'])
        if fixed_mount:
            from camera_mount_config import profiles_for
            current = profiles_for(fixed_mount,{k:v[0] for k,v in loaded.items()},config.get('world_from_table'))
            if profiles is not None:
                supplied=profiles.get('cameras',profiles)
                if supplied.get('sim1') != current['sim1']:
                    raise ValueError('Source camera differs from frozen embodiment mounting')
            initialization={'policy':fixed_mount['policy'],'config_sha256':sha(fixed_mount_path),
                            'hardware_calibrated':False,'episode_fitting':False,
                            'reference_posture':'canonical_asset_qpos0_not_episode_initial_state'}
        else:
            current, initialization = initialize_profiles(raw, config.get('world_from_table'))
        origin = deepcopy(current)
        initial_orientation = capture_initial_context(loaded,current,config.get('world_from_table')) if final and not fixed_mount else None
        if final:write(folder/'initial_orientation_context.json',initial_orientation)
        write(folder/'initial_profiles.json', current)
        write(folder/'initialization.json', initialization)
        write(folder/'physical_input_hashes.json', file_hashes)
        refs = _real_reference_images(config, frames, source_indices, folder/'real1_references')
        obs = _observations(loaded, scene, landmarks, frames, source_indices)
        write(folder/'observations.json', {'rows': obs, 'frame_mapping': mapping, 'landmarks': landmarks})
        history, best_report, best_score, best_sheets = [], None, None, []
        latest_rejected = None
        passed = False
        stages = tuple(loaded)
        verify_prompt = (Path(__file__).parent/'prompts/pilot_camera_verify.txt').read_text(encoding='utf-8')
        if fixed_mount:
            verify_prompt += ('\nFIXED EMBODIMENT MOUNT POLICY: the rig was fixed once from canonical model assets. '
                'Do not propose any edits. The head is upright in the table frame. Wrist image roll follows '
                'the wrist, so do not require world-upright wrist images or matching wrist world rotations '
                'between embodiments at the first frame or later. Still report observable Real1 view-role, '
                'perspective, FOV and occlusion mismatches honestly; a valid fixed mount is not proof of '
                'Real1 calibration or alignment. Preserve all unresolved evidence.\n')
        if not final:
            verify_prompt += '\nThis call is SOURCE-ONLY INITIALIZATION: inspect exactly the three Sim1 views at frame zero. Return only those three entries. Do not claim temporal or Sim2 verification.\n'
        tool_contract = {'tool': 'camera_patch', 'template': {'kind': '$stage', 'view': '$view', 'edit': {'translation_camera_m': ['$dx', 0, 0]}},
                         'available_stages': list(stages), 'views': list(CAMERAS),
                         'allowed_edit_fields': ['translation_camera_m', 'rotation_camera_xyz_degrees', 'fovy_delta_degrees'],
                         'projection_fit_alternative': {'edit':{'fit_projection':True},
                             'meaning':'Agent may select this instead of explicit deltas. Fixed local solver fits observed correspondences for the selected view, within the same edit bounds. No hidden-center guesses, no rendering, no imported camera fit; rendered verification still required.'},
                         'visibility_fit_alternative': {'edit':{'fit_visibility':True},
                             'meaning':'Agent may select a bounded local numerical search after coarse registration. Initial projections and visible object extents stay constrained. First, middle and last saved states guide usable workspace and less robot occlusion, with one fixed mount for every state. Saved hand contacts identify interacting objects: the tool rejects sampled contact objects outside their wrist-camera frustum and prefers actual visible object pixels. Maximum 901 candidate evaluations and 2703 ID renders (source-only 901), with exact candidate caching; no dynamics or API calls. Dynamic object pixels need not equal Real1. K3 must still verify occlusion and visible interaction; numeric fitting alone does not certify consistency.',
                             'local_rotation_range_degrees':30,'same_translation_and_fov_bounds':True,
                             'no_saved_case_specific_parameters':True},
                         'atomic_multi_view_alternative': {'edits':[
                             {'kind':'sim1','view':'left_wrist','edit':{'fit_projection':True}},
                             {'kind':'sim1','view':'right_wrist','edit':{'fit_projection':True}}]},
                         'atomic_policy':'One to six edits in one candidate and one visual review. A view may appear once; head appears once across both models. All edits pass the same bounds or the entire candidate is rejected. Use fitting for projection errors; use visual evidence to correct roll/occlusion that a partial box cannot determine.',
                         'translation_step_per_axis_m': {'head': .15, 'wrist': .04},
                         'rotation_xyz_step_degrees': 180, 'fovy_step_degrees': 20, 'fovy_range_degrees': [20, 110],
                         'cumulative_translation_from_stage_origin_m': {'head': .9, 'wrist': .20},
                         'head_edit_affects_both_stages_in_table_frame': True,
                         'wrist_policy': 'fixed relative to own embodiment and own side; not equal world trajectory',
                         'immutable': ['camera_parent', 'objects', 'robot_states', 'actions', 'time_map', 'success_predicates', 'crop', 'mirror'],
                         'sim1_task_success_required': False,
                         'lens_clearance_policy':CAMERA_CLEARANCE_POLICY,
                         'contact_object_visibility_policy':CAMERA_INTERACTION_POLICY,
                         'pixel_equality_required': 'initial object layout only; not dynamic objects or robot silhouettes'}
        if final:
            tool_contract['initial_optical_orientation_policy']=INITIAL_ORIENTATION_POLICY
            tool_contract['source_orientation_alternative']={
                'kind':'sim2','view':'left_wrist or right_wrist','edit':{'align_source_orientation':True},
                'meaning':'Use the current same-case Sim1 optical direction in shared table coordinates, transformed into the target wrist parent at the saved first frame. Changes only a fixed target camera rotation. It does not copy a Cobot-local mounting to S1 or constrain later world trajectories. Correct an invalid source rig first; this tool is not Real1 calibration or visual acceptance.'}
        context = {'phase': 'final_rig' if final else 'source_initialization',
                   'task_family': config.get('task_family', 'rigid_container_and_block'),
                   'calibration': 'estimated_fixed_mount', 'roles': list(CAMERAS)}
        for index in range(settings['max_candidates']):
            destination = folder/f'candidate_{index:02d}'
            candidate, trace = deepcopy(current), None
            before_score = best_score
            if index:
                payload = {'task': config['task'], 'tool_contract': tool_contract, 'camera_profiles': current,
                           'diagnostics': best_report, 'history': history, 'remaining_candidates': settings['max_candidates']-index,
                           'first_frame_landmarks': landmarks,
                           'instructions': 'Only cameras may change. Sim1 task success is not required. Later object motion may differ; preserve view role and visible interaction, not robot pixel shape.'}
                if config.get('stage_protocol'):
                    payload.update(camera_origin=origin,frame_mapping=mapping,camera_phase=context['phase'])
                decision_images = list(best_sheets)
                payload['image_order'] = ['best_candidate_sheet_'+str(i) for i in range(len(best_sheets))]
                if latest_rejected is not None:
                    payload['latest_rejected_camera'] = deepcopy(latest_rejected)
                    rejected_images = latest_rejected['report']['images']
                    decision_images.extend(rejected_images)
                    payload['image_order'].extend('rejected_candidate_sheet_'+str(i) for i in range(len(rejected_images)))
                try:
                    patch, trace = propose(config, knowledge, 'camera', destination/'proposal', payload, decision_images)
                    if trace.get('stop'):
                        history.append({'candidate': index, 'stopped': True, 'trace': trace})
                        break
                    write(destination/'patch.json', {'patch': patch, 'trace': trace})
                    tool_identity = {'settings':settings,'before':current,'origin':origin,
                                     'observations':obs,'patch':patch,'trace':trace}
                    cached_tool = load_evidence(destination/'tool_evidence.json', tool_identity)
                    if cached_tool is None:
                        from camera_patch_tool import apply_patch
                        candidate,tool_receipt=apply_patch(current,origin,obs,patch,config.get('world_from_table'),
                            visibility_context={'loaded':loaded,'landmarks':landmarks,'frame_mapping':mapping,
                                                'scene_object_ids':[o['id'] for o in scene['objects']],
                                                'initial_orientation':initial_orientation})
                        write(destination/'camera_tool_receipt.json',tool_receipt)
                        save_evidence(destination/'tool_evidence.json', tool_identity,
                                      {'candidate':candidate}, [destination/'camera_tool_receipt.json'])
                    else:
                        candidate = cached_tool['candidate']
                except DecisionOutputExhausted as error:
                    history.append({'candidate':index,'stopped':True,'method_budget_exhausted':str(error)})
                    break
                except (ValueError, KeyError, TypeError) as error:
                    history.append({'candidate': index, 'accepted': False, 'tool_error': str(error), 'trace': trace})
                    write(destination/'rejected.json', history[-1])
                    continue
            write(destination/'profile_before.json', current)
            write(destination/'profile_proposed.json', candidate)
            visual_identity = {'settings':settings,'candidate':candidate,'origin':origin,
                               'landmarks':landmarks,'frame_mapping':mapping}
            evidence = load_evidence(destination/'render_evidence.json', visual_identity)
            if evidence is None:
                numerical = diagnose_camera_profiles(candidate, origin, obs, mapping, config.get('world_from_table'))
                if scene_review is not None:
                    numerical['source_scene_review']={
                        'camera_issues':deepcopy(scene_review.get('camera_issues',[])),
                        'scene_sha256':scene_hash,'source_profiles_sha256':digest(profiles),
                        'scope':'Earlier scene review of the starting source rig. Inspect current images to resolve these concerns; never silently treat them as resolved.'}
                rendered = render_profile_frames(loaded, candidate, frames, destination/'renders',
                                                 object_ids=[o['id'] for o in scene['objects']])
                from camera_visibility import diagnose as diagnose_visibility
                numerical['raster_visibility']=diagnose_visibility(rendered,landmarks)
                numerical['numerical_passed']=bool(numerical['numerical_passed'] and numerical['raster_visibility']['passed'])
                if fixed_mount:
                    from camera_mount_config import diagnose_mounts
                    numerical['fixed_mount']=diagnose_mounts(loaded,candidate,rendered)
                    numerical['numerical_passed']=bool(numerical['numerical_passed'] and numerical['fixed_mount']['passed'])
                elif final:
                    numerical['initial_optical_orientation']=diagnose_initial_orientation(candidate,initial_orientation)
                    numerical['numerical_passed']=bool(numerical['numerical_passed'] and numerical['initial_optical_orientation']['passed'])
                numerical['evidence_layout']={'sheet_order':list(CAMERAS),
                    'columns':['real1']+list(stages),
                    'rows_source_frame_index':source_indices[frames].tolist(),
                    'one_camera_role_per_sheet':True}
                sheets = _sheets(refs, rendered, frames, source_indices, source_times, destination/'sheets')
                save_evidence(destination/'render_evidence.json', visual_identity,
                              {'numerical':numerical,'sheets':list(map(str,sheets))}, sheets)
            else:
                numerical, sheets = evidence['numerical'], [Path(p) for p in evidence['sheets']]
            review_performed=bool(numerical['numerical_passed'] and (not final or numerical['temporal_evidence_sufficient']))
            review_exhausted = None
            try:
                verdict = call(config, destination/'verification', verify_prompt,
                           {'task': config['task'], 'available_stages': list(stages), 'source_frame_indices': source_indices[frames].tolist(),
                            'source_task_success_required': False, 'diagnostics': numerical,
                            'not_measured_calibration': True,
                                **({'frame_mapping':mapping,'camera_phase':context['phase']} if config.get('stage_protocol') else {})}, sheets, stage='camera.review') if review_performed else None
            except DecisionOutputExhausted as error:
                verdict = None
                review_exhausted = str(error)
                write(destination/'review_budget_exhausted.json',{'reason':review_exhausted,
                      'visual_verdict':None,'passed':False,'not_a_physical_success':True})
            if not review_performed:
                write(destination/'verification_skipped.json',{
                    'api_requests':0,'visual_verdict':None,'passed':False,
                    'reason':'Required numeric camera checks already fail; rendered sheets remain available to the repair Agent.'})
            visual = visual_verdict_passed(verdict, stages)
            success = bool(numerical['numerical_passed'] and visual and (not final or numerical['temporal_evidence_sufficient']))
            score = _score(numerical, verdict, stages)
            report = {'numerical': numerical, 'visual_verdict': verdict, 'visual_passed': visual,
                      'visual_review_performed':review_performed,
                      'visual_review_budget_exhausted':review_exhausted,
                      'passed': success, 'score': score, 'images': list(map(str, sheets)),
                      'source_task_success_required': False, 'physics_and_actions_modified': False}
            from rssr_camera_learning import feedback,admission
            measured_feedback=feedback(best_report,report,stages)
            report['learning_feedback']=measured_feedback
            write(destination/'report.json', report)
            accepted = best_score is None or (score < best_score and measured_feedback['nonregression']) or success
            history.append({'candidate': index, 'score': score, 'accepted': accepted, 'passed': success,
                            'trace': trace, 'before_score': before_score, 'camera_feedback':measured_feedback,
                            'report': str(destination/'report.json')})
            write(destination/'acceptance.json', history[-1])
            if trace is not None:
                knowledge.record('camera', config['case_id'], trace, [float(before_score)], [score], context, success,
                                 admission_evidence=admission(measured_feedback))
            if accepted:
                current, best_report, best_score, best_sheets = candidate, report, score, sheets
                latest_rejected = None
            else:
                latest_rejected = {'candidate':index, 'accepted':False,
                                   'profiles':deepcopy(candidate), 'report':deepcopy(report),
                                   'score_delta_from_best':float(score-best_score),
                                   'new_edits_apply_to':'camera_profiles_best_not_rejected'}
            write(folder/'progress.json', {'history': history, 'best_score': best_score, 'complete': False})
            if success:
                passed = True
                break
            if review_exhausted:
                history.append({'candidate':index,'stopped':True,'method_budget_exhausted':review_exhausted})
                break
        profile = {'schema': 'rssr.camera_profile.v2', 'cameras': current, 'origin_cameras': origin,
                   'hardware_calibrated': False, 'world_or_parent_fixed_for_entire_episode': True,
                   'role': 'same-case current-run registration; not hardware calibration'}
        if fixed_mount:
            profile.update(role='fixed embodiment engineering rig; not hardware calibration',
                           fixed_camera_config_sha256=sha(fixed_mount_path),camera_policy=fixed_mount['policy'],
                           fixed_across_episodes=True,per_frame_leveling=False)
        write(folder/'camera_profile.json', profile)
        for paths in file_hashes.values():
            for path, checksum in paths.items():
                if sha(path) != checksum:
                    raise RuntimeError('Camera workflow modified its physical input '+path)
        result = {'passed': passed, 'profiles': current, 'report': best_report,
                  'history': history, 'final': bool(final), 'state': 'camera_verified' if passed else 'camera_not_verified',
                  'profile': str(folder/'camera_profile.json'), 'profile_sha256': sha(folder/'camera_profile.json'),
                  'physics_and_actions_modified': False, 'source_task_success_required': False,
                  'old_profile_imported': False, 'same_case_source_profile_inherited': profiles is not None,
                  'temporal_verification': bool(final and best_report and best_report['numerical']['timing_passed']
                                                and best_report['numerical']['temporal_evidence_sufficient']),
                  'hardware_calibrated': False}
        if fixed_mount:
            result.update(camera_policy=fixed_mount['policy'],fixed_camera_config_sha256=sha(fixed_mount_path),
                          camera_adjustment_api_requests=0,camera_library_updates=False,
                          fixed_across_episodes=True,same_case_source_profile_inherited=False)
        write(folder/'result.json', result)
        return result
    finally:
        for _, _, rollout, _ in loaded.values():
            if hasattr(rollout, 'close'):
                rollout.close()
