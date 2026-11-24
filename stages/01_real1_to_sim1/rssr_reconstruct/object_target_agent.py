"""Frozen object-feedback repair loop for the supported box/container task.

The LLM proposes bounded action/layout changes. Deterministic physics checks
decide progress and success. Materials, controller gains and success criteria
are not inference-time repair knobs. No rendering or shared-memory updates.
"""
import copy
import json
import pathlib
import numpy as np
from common import read, write, sha
from object_scene_tools import target_scene, apply_scene_delta
from object_retarget import repair
from object_physics import rollout
from object_decision import decide

CODE = pathlib.Path(__file__).resolve().parent
HARD = ('initial_cube_outside_container', 'max_robot_table_penetration_le_3mm',
        'joint_limits_within_2mrad', 'solver_no_warnings', 'waist_coupling_within_1mrad')
TASK = ('container_lifted_by_left', 'cube_lifted_by_right', 'cube_inside_container',
        'cube_supported_by_container', 'released', 'stable', 'container_on_table',
        'container_hold_phase_complete', 'container_left_contact_maintained',
        'container_bilateral_grasp_maintained', 'container_no_unsupported_early_table_return')


def vector(value, limit, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.max(np.abs(a)) > limit:
        raise ValueError('Invalid bounded three-vector: '+name)
    return a.tolist()


def validate_correction(value):
    allowed = {'hand_offsets_m', 'vertical_grasp_mix', 'vertical_grasp_mix_by_hand',
               'phase_hand_offsets_m'}
    if not isinstance(value, dict) or set(value)-allowed:
        raise ValueError('Unknown action correction field')
    offsets = value.get('hand_offsets_m')
    if not isinstance(offsets, dict) or set(offsets) != {'left', 'right'}:
        raise ValueError('Specify both complete hand offsets')
    out = {'hand_offsets_m': {s: vector(offsets[s], .16, s) for s in offsets}}
    mix = value.get('vertical_grasp_mix', 0.)
    per_hand = value.get('vertical_grasp_mix_by_hand', {})
    if not isinstance(per_hand, dict) or set(per_hand)-{'left', 'right'}:
        raise ValueError('Unknown hand orientation field')
    for v in [mix, *per_hand.values()]:
        if type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError('Orientation interpolation must be in [0,1]')
    out.update(vertical_grasp_mix=float(mix), vertical_grasp_mix_by_hand=copy.deepcopy(per_hand))
    phases = value.get('phase_hand_offsets_m', {})
    if not isinstance(phases, dict) or set(phases)-{'left', 'right'}:
        raise ValueError('Unknown phase-offset hand')
    out['phase_hand_offsets_m'] = {}
    for side, points in phases.items():
        if not isinstance(points, dict) or set(points)-{'grasp', 'release'}:
            raise ValueError('Only grasp and release phase corrections are supported')
        out['phase_hand_offsets_m'][side] = {}
        for phase, delta in points.items():
            d = vector(delta, .08, side+'/'+phase)
            vector(np.asarray(offsets[side])+d, .16, 'total phase correction')
            out['phase_hand_offsets_m'][side][phase] = d
    return out


def proposal(base_scene, base_correction, initial_scene, decision):
    if not isinstance(decision, dict) or set(decision)-{'reason', 'scene_patch', 'correction'}:
        raise ValueError('Unknown repair decision field')
    if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        raise ValueError('Repair must cite its diagnosis')
    patch = decision.get('scene_patch', {})
    if not isinstance(patch, dict) or set(patch)-{'object_xy_shift_m', 'basket_yaw_delta_rad'}:
        raise ValueError('Inference cannot edit material, shape, mass, table or controller')
    shifts = patch.get('object_xy_shift_m', {})
    names = {o['id'] for o in initial_scene['objects']}
    if not isinstance(shifts, dict) or set(shifts)-names:
        raise ValueError('Unknown scene object')
    for name, xy in shifts.items():
        a = np.asarray(xy, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all() or np.max(np.abs(a)) > .04:
            raise ValueError('Layout proposal exceeds 4 cm per axis')
    yaw = patch.get('basket_yaw_delta_rad', 0.)
    if type(yaw) not in (int, float) or not np.isfinite(yaw) or abs(yaw) > .3:
        raise ValueError('Invalid basket yaw proposal')
    scene = apply_scene_delta(base_scene, patch)
    initial = {o['id']: o for o in initial_scene['objects']}
    for obj in scene['objects']:
        if np.linalg.norm(np.asarray(obj['position_m'])-initial[obj['id']]['position_m']) > .12+1e-10:
            raise ValueError('Cumulative object relocation exceeds 12 cm')
        da = (obj.get('yaw_rad', 0.)-initial[obj['id']].get('yaw_rad', 0.)+np.pi)%(2*np.pi)-np.pi
        if abs(da) > .6+1e-10:
            raise ValueError('Cumulative object rotation exceeds 0.6 rad')
    correction = validate_correction(decision.get('correction', base_correction))
    return scene, correction


def objective(report, scene=None):
    criteria = report.get('criteria', {})
    missing = [k for k in HARD+TASK if type(criteria.get(k)) is not bool]
    if missing:
        raise ValueError('Incomplete physical task report: '+','.join(missing))
    hard = [k for k in HARD if not criteria[k]]
    failed = [k for k in TASK if not criteria[k]]
    # These residuals guide progress only; they NEVER replace the success gate.
    hold = report.get('container_hold', {})
    gaps = min(10., float(hold.get('maximum_unheld_span_s', 10.)))
    bilateral = float(hold.get('bilateral_contact_fraction', 0.))
    motion = max(report.get('last_second_motion_m', [1.]), default=1.)
    residual = gaps/.1 + max(0., .9-bilateral)/.9 + min(10., float(motion)/.003)
    placement = 0.
    if scene is not None and 'cube_final_container_relative_m' in report:
        objects = {o['id']: o for o in scene['objects']}
        basket = objects[scene['container_id']]; cube = objects[scene['movable_id']]
        relative = np.asarray(report['cube_final_container_relative_m'], float)
        desired_z = -basket['size_m'][2]/2+basket['wall_m']+cube['size_m'][2]/2
        placement = float(np.linalg.norm(relative-np.array([0., 0., desired_z]))/.05)
        residual += placement
    if not np.isfinite(residual):
        raise ValueError('Nonfinite physical objective')
    return {'hard_failures': hard, 'task_failures': failed,
            'rank': [len(hard), len(failed), residual],
            'passed': not hard and not failed and report.get('task_success') is True,
            'placement_center_residual_in_5cm_units': placement,
            'basis': 'hard constraints, then task predicates, then numeric holding/stability residuals'}


def diagnostic(report):
    result = {k: v for k, v in report.items() if k != 'state_trace'}
    trace = report.get('state_trace', [])
    ix = np.unique(np.linspace(0, len(trace)-1, min(12, len(trace))).round().astype(int)) if trace else []
    keep=set(ix)|{i for i,row in enumerate(trace) if row.get('source_events') or row.get('contact_diagnostic_events')}
    result['state_trace'] = [trace[i] for i in sorted(keep)]
    return result


def state_feedback(folder, scene, ik_plan=None, contact_queries=()):
    """Read FK and object state at source events; do not render or step physics."""
    import mujoco
    from object_physics import hand_pose
    folder = pathlib.Path(folder)
    m = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); d = mujoco.MjData(m)
    from rssr_tracking_feedback import load_tracking,ik_residual_at,servo_residual
    tracks=load_tracking(ik_plan);nominal=mujoco.MjData(m)
    with np.load(folder/'rollout.npz') as z, np.load(folder/'command_targets.npz') as commands:
        times = z['time_s']
        selected = set(np.linspace(0, len(times)-1, 6).round().astype(int).tolist())
        event_labels={};preclosure={}
        for side in ['left', 'right']:
            from rssr_physical_feedback import grasp_events
            events=grasp_events(commands['time_s'],commands[side+'_gripper_command'])
            for cycle,event in enumerate(events):
                from rssr_preclosure_feedback import preclosure_motion
                close_index=int(np.argmin(np.abs(times-event['closed_s'])))
                preclosure.setdefault(close_index,[]).append(preclosure_motion(z,scene,side,cycle,events))
                for phase,stamp in [('closed',event['closed_s']),
                                    ('holding',(event['closed_s']+event['release_s'])/2),
                                    ('release',event['release_s'])]:
                    index=int(np.argmin(np.abs(times-stamp)))
                    selected.add(index)
                    event_labels.setdefault(index,[]).append({'hand':side,'cycle':cycle,'phase':phase})
        diagnostic_events={}
        for query in contact_queries:
            index=int(np.argmin(np.abs(times-query['time_s'])))
            selected.add(index);diagnostic_events.setdefault(index,[]).append(query)
        rows = []
        for index in sorted(selected):
            d.qpos[:] = z['qpos'][index]; d.qvel[:] = z['qvel'][index]; mujoco.mj_forward(m, d)
            poses = {s: hand_pose(m, d, 'sim2', s) for s in ['left', 'right']}
            hands = {s: poses[s][0] for s in poses}
            positions = {o['id']: z['object_position'][index, i] for i, o in enumerate(scene['objects'])}
            from rssr_contact_feedback import distances
            rows.append({'time_s': float(times[index]),
                'source_events':event_labels.get(index,[]),
                'object_surface_distance_m':distances(m,d,scene['objects'],hands),
                'object_positions_m': {k: v.tolist() for k, v in positions.items()},
                'hand_positions_m': {k: v.tolist() for k, v in hands.items()},
                'hand_rotations': {s: poses[s][1].tolist() for s in poses},
                'object_quaternions_wxyz': {o['id']: z['object_quaternion'][index, i].tolist() for i, o in enumerate(scene['objects'])},
                'object_minus_hand_m': {s: {k: (v-hands[s]).tolist() for k, v in positions.items()} for s in hands},
                'hand_table_contacts': z['contacts'][index].tolist(),
                'opening_m': z['gripper_opening'][index].tolist()})
            if event_labels.get(index) or diagnostic_events.get(index):
                rows[-1]['ik_tracking']=ik_residual_at(tracks,float(times[index]))
                rows[-1]['servo_tracking']=servo_residual(m,d,nominal,commands,float(times[index]))
            if diagnostic_events.get(index):
                from rssr_hold_feedback import contact_geometry
                from rssr_grasp_geometry import component_proximity
                from pilot_physics import aperture
                rows[-1]['contact_diagnostic_events']=diagnostic_events[index]
                observations=[]
                for query in diagnostic_events[index]:
                    side=query['hand'];oi=next(i for i,o in enumerate(scene['objects']) if o['id']==query['object'])
                    item=contact_geometry(m,d,query['object'],side,poses[side][1][:,0])
                    frames=(ik_plan or {}).get('grasp_frame_definitions',{})
                    item['component_proximity']=component_proximity(m,d,query['object'],side,frames.get(side,{}).get('site_from_grasp_rotation'))
                    item.update(predicate=query['predicate'],
                        finger_contact_sides=z['finger_contact_sides'][index,oi,('left','right').index(side)].astype(bool).tolist(),
                        commanded_opening_m=aperture(m,nominal,side),actual_opening_m=aperture(m,d,side),
                        source_normalized_opening=float(np.interp(times[index],commands['time_s'],commands[side+'_gripper_command'])))
                    observations.append(item)
                rows[-1]['holding_contact_geometry']=observations
            if index in preclosure:rows[-1]['preclosure_motion']=preclosure[index]
    return {'state_trace': rows, 'basis': 'native saved physical states and FK at every source grasp cycle; surface distances are analytic geometry proxies, not contact success', 'render_calls': 0}


def select_target(config, budget, root, semantics, source):
    root = pathlib.Path(root)
    scene, correction = target_scene(config, root/'target_construction', source['scene'])
    initial = copy.deepcopy(scene)
    correction = validate_correction(correction)
    maximum = int(config.get('max_target_agent_rounds', 8))
    if not 1 <= maximum <= 12:
        raise ValueError('Object agent budget must be 1..12 candidate slots')
    contract = {'schema': 'rssr.object_retarget_contract.v1', 'scope': 'box_into_held_open_container',
                'task': semantics['task'], 'hard_criteria': list(HARD), 'task_criteria': list(TASK),
                'max_candidate_slots': maximum, 'initial_scene': initial,
                'immutable_during_inference': ['success_criteria', 'robot_model', 'controller',
                    'object_shape', 'object_size', 'object_mass', 'contact_material', 'source_action'],
                'source_task_success_required': False, 'shared_memory_updated': False,
                'source_action_sha256': sha(pathlib.Path(config['import_case'])/'source_action_state.npz'),
                'initial_trajectory_sha256': sha(pathlib.Path(config['import_case'])/'agent_trajectory_full.npz')}
    contract['implementation_sha256'] = {name: sha(CODE/name) for name in [
        'object_target_agent.py', 'object_retarget.py', 'object_scene_tools.py', 'object_physics.py',
        'object_decision.py', 'prompts/object_target_repair.txt']}
    dest = root/'object_retarget_contract.json'
    if dest.exists() and read(dest) != contract:
        raise ValueError('Frozen object task contract changed')
    write(dest, contract)
    best = None
    history = []
    seen = set()
    prompt = (CODE/'prompts/object_target_repair.txt').read_text(encoding='utf8')
    for i in range(maximum):
        if any(sha(CODE/name) != digest for name, digest in contract['implementation_sha256'].items()):
            raise RuntimeError('Object agent implementation changed during inference')
        for name, field in [('source_action_state.npz', 'source_action_sha256'),
                            ('agent_trajectory_full.npz', 'initial_trajectory_sha256')]:
            if sha(pathlib.Path(config['import_case'])/name) != contract[field]:
                raise RuntimeError('Object agent source changed during inference')
        trial = root/'sim2'/f'trial_{i:02d}'
        trial.mkdir(parents=True, exist_ok=True)
        if i:
            decision = decide(config, budget, trial/'decision', prompt,
                {'contract': contract, 'best_scene': best['scene'], 'best_correction': best['correction'],
                 'best_report': diagnostic(best['report']), 'best_objective': best['objective'],
                 'history': history, 'remaining_candidate_slots': maximum-i,
                 'source_task_success_required': False})
            try:
                scene, correction = proposal(best['scene'], best['correction'], initial, decision)
            except (ValueError, TypeError, AssertionError) as error:
                note = {'trial': i, 'status': 'invalid_proposal', 'reason': str(error), 'proposal': decision}
                history.append(note); write(trial/'outcome.json', note)
                continue
        identity = json.dumps([scene, correction], sort_keys=True, allow_nan=False)
        if identity in seen:
            note = {'trial': i, 'status': 'duplicate_proposal', 'proposal': decision}
            history.append(note); write(trial/'outcome.json', note)
            continue
        seen.add(identity)
        write(trial/'scene_parameters.json', scene); write(trial/'correction.json', correction)
        if (trial/'repaired_commands.npz').exists():
            with np.load(trial/'repaired_commands.npz') as data:
                q = data['qpos']
            ik = read(trial/'repair_report.json')
        else:
            try:
                q, ik = repair(config, trial, correction)
            except (ValueError, AssertionError) as error:
                if best is None:
                    raise
                note = {'trial': i, 'status': 'tool_rejected', 'reason': str(error), 'correction': correction}
                history.append(note); write(trial/'outcome.json', note)
                continue
        physics = trial/'physics'
        report = read(physics/'report.json') if (physics/'report.json').exists() else rollout(config, 'sim2', physics, scene, q)
        feedback = state_feedback(physics, scene)
        write(trial/'state_feedback.json', feedback)
        report = dict(report, ik_correction=ik, state_trace=feedback['state_trace'])
        score = objective(report, scene)
        improved = best is None or tuple(score['rank']) < tuple(best['objective']['rank'])
        candidate = {'trial': str(physics), 'scene': copy.deepcopy(scene), 'correction': copy.deepcopy(correction),
                     'report': report, 'objective': score}
        if improved or score['passed']:
            best = candidate
        note = {'trial': i, 'status': 'passed' if score['passed'] else ('improved' if improved else 'rolled_back'),
                'objective': score, 'correction': correction,
                'scene_positions_m': {o['id']: o['position_m'] for o in scene['objects']}}
        history.append(note); write(trial/'outcome.json', note)
        write(root/'object_target_progress.json', {'contract_sha256': sha(dest), 'history': history,
              'best_trial': best['trial'], 'best_objective': best['objective'], 'shared_memory_updated': False})
        print(json.dumps({'stage': 'object_retarget', **note}), flush=True)
        if score['passed']:
            candidate.update(acceptance_basis='frozen object/contact task predicates and embodiment constraints',
                             retarget_origin='v22 initialization + autonomous bounded object-feedback agent',
                             shared_memory_updated=False, inference_code_modified=False)
            write(root/'sim2/selection.json', candidate)
            return candidate
    write(root/'object_target_result.json', {'status': 'budget_exhausted', 'best_trial': best['trial'],
          'best_objective': best['objective'], 'history': history, 'task_success': False})
    raise RuntimeError('object_target_not_solved_within_frozen_budget')
