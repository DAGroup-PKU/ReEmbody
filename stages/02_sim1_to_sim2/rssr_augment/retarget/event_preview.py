"""Bounded, isolated native-physics preview of one selected loaded event.

The source MuJoCo session is paused while this function runs. The branch owns
only its new MjData. Its outcome is evidence for the online agents; it is never
counted as a task result, and it cannot modify the source physical state.
"""
from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np

from .executor import (control_step, overhead_approach, side_corridor_approach,
                       front_staging_transport)
from .immutable_hard_observation import HardAccumulator, HardSnapshotter
from .session import _observation, robot_command_indices


LOADED_EVENTS = frozenset(('hold', 'lift', 'transport', 'align', 'open_drawer',
                           'close_drawer', 'push_close'))
PREVIEW_EVENTS = LOADED_EVENTS | {'approach', 'contact', 'grasp', 'release'}


def horizon_for(event: str) -> float | None:
    """Spend long previews only where slip takes time to become observable."""
    if event in ('open_drawer', 'close_drawer', 'push_close'):
        return 5.0
    if event == 'transport':
        return 2.5
    if event in ('lift', 'align'):
        return .15
    if event == 'approach':
        return 2.5
    if event in ('contact', 'grasp'):
        # These transitions can hit a robot joint limit before a durable
        # bilateral grasp appears.  Screen the copied branch with the same
        # every-step native hard observer used for loaded motion.
        return 1.0
    return None


def _source_digest(state: dict, action, contract=None) -> str:
    digest = hashlib.sha256()
    for key in ('qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart', 'last_q',
                'loaded_contact_anchor_local_m',
                'feature_hold_anchor_local_m',
                'contact_settle_anchor_qpos'):
        value = state.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        digest.update(key.encode())
        digest.update(array.shape.__repr__().encode())
        digest.update(array.tobytes())
    digest.update(json.dumps({'time_s': float(state['time_s']),
                              'pull_offset_m': state.get('pull_offset_m'),
                              'stop_progress_at': state.get('stop_progress_at'),
                              'hold_anchors': state.get('hold_anchors'),
                              'feature_hold_object_id': state.get('feature_hold_object_id'),
                              'feature_hold_hand': state.get('feature_hold_hand'),
                              'feature_hold_axis_joint': state.get('feature_hold_axis_joint'),
                              'feature_hold_pull_offset_m': state.get('feature_hold_pull_offset_m'),
                              'support_probe_object_id': state.get('support_probe_object_id'),
                              'support_lock_waist': state.get('support_lock_waist'),
                              'contact_settle_anchor_key': state.get('contact_settle_anchor_key'),
                              'contact_settle_last_touch_t': state.get('contact_settle_last_touch_t'),
                              'contact_ik_plan': state.get('contact_ik_plan'),
                              'contract_id': None if contract is None else contract.get('contract_id'),
                              'action': action.__dict__},
                             sort_keys=True, default=lambda value: np.asarray(value).tolist()).encode())
    return digest.hexdigest()


def evaluate(model: mujoco.MjModel, servo_class, aperture, scene: dict,
             state: dict, action, *, duration_s: float, contract=None,
             return_final_state: bool = False):
    if action.event not in PREVIEW_EVENTS:
        raise ValueError('Event has no physical preview')
    if action.hand not in ('left', 'right') or action.object_id is None:
        raise ValueError('Loaded event lacks a manipulator and target')
    if not .1 <= duration_s <= 30.0:
        raise ValueError('Preview horizon outside fixed budget')
    data = mujoco.MjData(model)
    for key in ('qpos', 'qvel', 'ctrl', 'act'):
        value = np.asarray(state[key], dtype=float)
        if value.shape != getattr(data, key).shape:
            raise ValueError('Copied native state shape mismatch: ' + key)
        getattr(data, key)[:] = value
    data.time = float(state['time_s'])
    mujoco.mj_forward(model, data)
    if state.get('qacc_warmstart') is not None:
        warmstart = np.asarray(state['qacc_warmstart'], dtype=float)
        if warmstart.shape != data.qacc_warmstart.shape:
            raise ValueError('Copied native warmstart shape mismatch')
        data.qacc_warmstart[:] = warmstart
    source_time = float(data.time)
    source_arrays = {key: np.asarray(state[key]).copy()
                     for key in ('qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart')
                     if key in state}
    acts = [(a, int(model.jnt_qposadr[int(model.actuator_trnid[a, 0])]),
             int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])]))
            for a in range(model.nu)]
    servo = servo_class(model, acts, 'sim2')
    reader, hard = HardSnapshotter(model), HardAccumulator()
    active_gripper_geoms = set()
    for geom_id in range(model.ngeom):
        if not (model.geom(geom_id).name or '').startswith('astribot_gripper_'):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        while body_id:
            if (model.body(body_id).name or '').startswith(
                    f'astribot_arm_{action.hand}_'):
                active_gripper_geoms.add(geom_id)
                break
            body_id = int(model.body_parentid[body_id])
    names = tuple(row['id'] for row in scene['objects'])
    geometry = {row['id']: {key: row[key] for key in (
        'shape', 'size_m', 'wall_m', 'parent_id', 'role', 'mobility',
        'joint', 'yaw_rad') if key in row} for row in scene['objects']}
    initial = _observation(model, data, names, aperture)
    initial['object_geometry'] = geometry
    def approach_remaining(observation):
        if action.event != 'approach':
            return None
        if observation.get('object_geometry',{}).get(action.object_id,{}).get('shape')=='cup':
            from dataclasses import replace
            selected=replace(action,target_offset_m=(0.,0.,0.),recovery_mode='overhead')
            return float(overhead_approach(selected, observation)[1])
        if action.recovery_mode == 'overhead':
            return float(overhead_approach(action, observation)[1])
        if action.recovery_mode == 'side_corridor':
            return float(side_corridor_approach(action, observation)[1])
        return float(np.linalg.norm(
            np.asarray(observation['finger_center_m'][action.hand]) -
            np.asarray(observation['object_position_m'][action.object_id]) -
            np.asarray(action.target_offset_m)))
    initial_approach_remaining = approach_remaining(initial)
    if action.object_id not in initial['finger_contact']:
        raise ValueError('Preview target absent from scene observation')
    initially_bilateral = bool(all(initial['finger_contact'][
        action.object_id][action.hand]))
    if action.event in LOADED_EVENTS and not initially_bilateral:
        raise ValueError('Loaded preview requires measured bilateral contact')
    selected_joint = (geometry.get(action.target_id, {}).get('joint', {})
                      .get('name') if action.target_id else None)
    start_fraction = (float(initial['articulation_fraction'][selected_joint])
                      if selected_joint in initial['articulation_fraction'] else None)
    dependent_joint = {}
    if contract is not None:
        for predicate in contract['predicates']:
            if (predicate['kind'] == 'open_insert_close' and
                    predicate['object'] == action.object_id and
                    predicate['joint'] in initial['articulation_fraction']):
                dependent_joint[predicate['joint']] = {
                    'required_open_fraction': float(predicate['open_fraction']),
                    'start_fraction': float(initial['articulation_fraction'][predicate['joint']]),
                    'minimum_fraction': float(initial['articulation_fraction'][predicate['joint']])}
    start_object_z = float(initial['object_position_m'][action.object_id][2])
    first_relative = (initial['object_rotation_world'][action.object_id].T @
                      (initial['finger_center_m'][action.hand] -
                       initial['object_position_m'][action.object_id]))
    inactive = 'right' if action.hand == 'left' else 'left'
    anchor = state.get('hold_anchors', {}).get(inactive)
    if anchor is None:
        anchor = {'position': initial['finger_center_m'][inactive].copy(),
                  'rotation': initial['hand_rotation_world'][inactive].copy()}
    # The loaded-cup skill also holds its active hand's action-start rotation.
    # Preserve both copied anchors so previews issue the same robot targets
    # as ContinuousSession; retaining only the bystander changes the skill.
    hold_anchors = {hand: {
        'position': np.asarray(value['position']).copy(),
        'rotation': np.asarray(value['rotation']).copy()}
        for hand, value in state.get('hold_anchors', {}).items()}
    hold_anchors[inactive] = {'position': np.asarray(anchor['position']).copy(),
                              'rotation': np.asarray(anchor['rotation']).copy()}
    support_feature = state.get('feature_hold_object_id')
    support_probe = state.get('support_probe_object_id', support_feature)
    if support_probe is not None and support_probe not in initial['finger_contact']:
        raise ValueError('Support probe feature absent from observation')
    if state.get('support_lock_waist') and support_feature is None:
        raise ValueError('Waist support lock requires a bound support feature')
    if support_feature is not None:
        if (state.get('feature_hold_hand') != inactive or
                support_feature not in initial['object_position_m'] or
                state.get('feature_hold_axis_joint') not in initial['articulation_axis_world']):
            raise ValueError('Invalid inactive-hand feature support binding')
        support_anchor = np.asarray(state['feature_hold_anchor_local_m'], dtype=float)
        if support_anchor.shape != (3,) or not np.isfinite(support_anchor).all():
            raise ValueError('Invalid feature support anchor')
        support_pull = float(state.get('feature_hold_pull_offset_m', .006))
        if not 0 < support_pull <= .02:
            raise ValueError('Feature support pull outside envelope')
    target = np.asarray(state['desired_qpos'], dtype=float).copy()
    velocity = np.asarray(state['desired_qvel'], dtype=float).copy()
    if target.shape != (model.nq,) or velocity.shape != (model.nq,):
        raise ValueError('Preview source command has wrong shape')
    robot_qa, scene_qa, base_qa = robot_command_indices(model)
    last_q = state.get('last_q')
    if last_q is not None:
        target[robot_qa] = np.asarray(last_q)[robot_qa]
        velocity[robot_qa] = 0.
    waist_qa = np.array([model.joint(f'astribot_torso_joint_{i}').qposadr[0]
                         for i in (1, 2)])
    initial_waist = np.asarray(state['torso_initial_qpos'], dtype=float)
    if initial_waist.shape != (2,):
        raise ValueError('Preview lacks native waist anchor')
    timestep = float(model.opt.timestep)
    nstep = int(round(duration_s/timestep))
    bilateral_steps = bilateral_run = longest_bilateral = 0
    first_persistent_loss_s = None
    loss_start_s = None
    blocking_steps = blocking_run = longest_blocking = 0
    support_bilateral_steps = 0
    first_persistent_blocking_s = None
    blocking_start_s = None
    blocking_object_ids: set[str] = set()
    maximum_blocking_penetration_m = 0.
    staged_transport = (action.event == 'transport' and
                        action.recovery_mode in ('front_clearance', 'front_align'))
    initial_stage_remaining = (float(front_staging_transport(action, initial)[1])
                               if staged_transport else None)
    start_progress = (-initial_stage_remaining if staged_transport else
                      start_fraction if start_fraction is not None else start_object_z)
    late_progress = start_progress
    final_progress = start_progress
    final = initial
    trace = []
    settle_anchor = (None if state.get('contact_settle_anchor_qpos') is None
                     else np.asarray(state['contact_settle_anchor_qpos']).copy())
    settle_key = state.get('contact_settle_anchor_key')
    settle_last_touch_t = state.get('contact_settle_last_touch_t')
    cached_obs = None
    observed_at = -float('inf')
    for step in range(nstep):
        # Match the native rollout's per-step forward pass before servo.apply.
        # Reusing stale bias/contact state changes free-object dynamics even
        # under identical robot commands and invalidates branch forecasts.
        mujoco.mj_forward(model, data)
        if cached_obs is None or float(data.time)-observed_at >= .01-1e-9:
            cached_obs = _observation(model, data, names, aperture)
            cached_obs['object_geometry'] = geometry
            observed_at = float(data.time)
        obs = cached_obs
        obs['torso_initial_qpos'] = initial_waist
        obs['hold_anchors'] = hold_anchors
        if action.event == 'contact':
            fingers = obs['finger_contact'][action.object_id][action.hand]
            selected_key = (action.object_id, action.hand)
            cup_contact=geometry.get(action.object_id,{}).get('shape')=='cup'
            contact_for_anchor=(all(fingers) if cup_contact else any(fingers))
            if len(fingers) == 2 and contact_for_anchor:
                if settle_anchor is None or settle_key != selected_key:
                    settle_anchor = data.qpos.copy()
                    settle_key = selected_key
                settle_last_touch_t = float(data.time)
                obs['contact_settle_anchor_qpos'] = settle_anchor
            elif (not cup_contact and len(fingers) == 2 and settle_anchor is not None and
                  settle_key == selected_key and settle_last_touch_t is not None and
                  float(data.time)-float(settle_last_touch_t) <= .4):
                obs['contact_settle_anchor_qpos'] = settle_anchor
            else:
                settle_anchor = None
                settle_key = None
                settle_last_touch_t = None
                obs.pop('contact_settle_anchor_qpos', None)
            if state.get('contact_ik_plan') is not None:
                obs['contact_ik_plan'] = state['contact_ik_plan']
        if state.get('support_lock_waist'):
            obs['support_lock_waist'] = True
        if support_feature is not None:
            axis = np.asarray(obs['articulation_axis_world'][
                state['feature_hold_axis_joint']], dtype=float)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            hold_anchors[inactive]['position'] = (
                np.asarray(obs['object_position_m'][support_feature]) +
                np.asarray(obs['object_rotation_world'][support_feature]) @
                support_anchor + support_pull * axis)
        if state.get('loaded_contact_anchor_local_m') is not None:
            obs['loaded_contact_anchor_local_m'] = np.asarray(
                state['loaded_contact_anchor_local_m'], dtype=float)
            obs['pull_offset_m'] = float(state.get('pull_offset_m', .005))
        prior_scene_target = target[scene_qa].copy()
        q, dq = control_step(action, obs, target, velocity, model)
        if not np.array_equal(q[scene_qa], prior_scene_target):
            raise ValueError('Preview changed a scene command')
        if last_q is not None:
            prior_t = state.get('last_t')
            dt = (timestep if step or prior_t is None else
                  float(data.time) - float(prior_t))
            dt = timestep if dt <= 0 else dt
            q[robot_qa] = np.clip(q[robot_qa], target[robot_qa]-4.*timestep,
                                  target[robot_qa]+4.*timestep)
            if len(base_qa):
                q[base_qa] = target[base_qa]
                dq[base_qa] = 0.
        q[waist_qa[1]] = float(initial_waist.sum()) - q[waist_qa[0]]
        servo.apply(data, q, dq)
        mujoco.mj_step(model, data)
        hard.observe(reader(data))
        target, velocity = q, dq
        last_q = q
        final = _observation(model, data, names, aperture)
        final['object_geometry'] = geometry
        if support_probe is not None:
            support_bilateral_steps += int(all(final['finger_contact'][support_probe][inactive]))
        for joint, row in dependent_joint.items():
            row['minimum_fraction'] = min(row['minimum_fraction'],
                                          float(final['articulation_fraction'][joint]))
        paired = bool(all(final['finger_contact'][action.object_id][action.hand]))
        blocked = False
        bilateral_steps += int(paired)
        bilateral_run = bilateral_run+1 if paired else 0
        longest_bilateral = max(longest_bilateral, bilateral_run)
        if action.event in LOADED_EVENTS:
            if paired:
                loss_start_s = None
            elif loss_start_s is None:
                loss_start_s = float(data.time)
            if (loss_start_s is not None and first_persistent_loss_s is None and
                    float(data.time)-loss_start_s >= .10):
                first_persistent_loss_s = float(loss_start_s-source_time)
        if action.event == 'approach':
            blockers = [row for row in final['gripper_scene_contacts'][action.hand]
                        if row['object_id'] != action.object_id]
            blocked = bool(blockers and not any(
                final['finger_contact'][action.object_id][action.hand]))
            blocking_steps += int(blocked)
            blocking_run = blocking_run + 1 if blocked else 0
            longest_blocking = max(longest_blocking, blocking_run)
            blocking_object_ids.update(row['object_id'] for row in blockers)
            maximum_blocking_penetration_m = max(
                maximum_blocking_penetration_m,
                *(float(row['penetration_m']) for row in blockers), 0.)
            if blocked and blocking_start_s is None:
                blocking_start_s = float(data.time)
            if not blocked:
                blocking_start_s = None
            if (blocking_start_s is not None and
                    first_persistent_blocking_s is None and
                    float(data.time)-blocking_start_s >= .04):
                first_persistent_blocking_s = float(blocking_start_s-source_time)
        final_progress = (-float(front_staging_transport(action, final)[1])
                          if staged_transport else
                          float(final['articulation_fraction'][selected_joint])
                          if start_fraction is not None else
                          float(final['object_position_m'][action.object_id][2]))
        if float(data.time-source_time) <= duration_s-.5:
            late_progress = final_progress
        if step % max(1, int(round(.5/timestep))) == 0 or step == nstep-1:
            contacts = []
            maximum_noncarried_force_n = 0.
            for index, contact in enumerate(data.contact):
                if contact.dist > 0:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                if wrench[0] < 1.:
                    continue
                for active, other in ((int(contact.geom1), int(contact.geom2)),
                                      (int(contact.geom2), int(contact.geom1))):
                    if active not in active_gripper_geoms:
                        continue
                    other_body = int(model.geom_bodyid[other])
                    if (other_body != model.body(action.object_id).id and
                            not (model.body(other_body).name or '').startswith('astribot_')):
                        maximum_noncarried_force_n = max(
                            maximum_noncarried_force_n, float(wrench[0]))
                contacts.append({
                    'geoms': [model.geom(int(contact.geom1)).name,
                              model.geom(int(contact.geom2)).name],
                    'normal_force_n': float(wrench[0]),
                    'penetration_m': max(0.,-float(contact.dist))})
            trace.append({'elapsed_s': round(float(data.time-source_time), 4),
                          'progress': final_progress,
                          'object_position_m': np.asarray(
                              final['object_position_m'][action.object_id]).tolist(),
                          'max_noncarried_gripper_force_n': maximum_noncarried_force_n,
                          'support_bilateral_contact': (
                              bool(all(final['finger_contact'][support_probe][inactive]))
                              if support_probe is not None else None),
                          'support_gripper_opening_m': (
                              float(final['gripper_opening_m'][inactive])
                              if support_probe is not None else None),
                          'support_anchor_error_m': (
                              float(np.linalg.norm(
                                  np.asarray(hold_anchors[inactive]['position']) -
                                  np.asarray(final['finger_center_m'][inactive])))
                              if support_probe is not None else None),
                          'support_orientation_error_rad': (
                              float(np.arccos(np.clip((np.trace(
                                  np.asarray(hold_anchors[inactive]['rotation']).T @
                                  np.asarray(final['hand_rotation_world'][inactive])) - 1.)/2.,
                                  -1., 1.)))
                              if support_probe is not None else None),
                          'support_joint_margin_rad': (
                              float(final['manipulator_joint_margin_rad'][inactive])
                              if support_probe is not None else None),
                          'torso_qpos_rad': (
                              np.asarray(final['robot_qpos'][waist_qa]).tolist()
                              if support_probe is not None else None),
                          'bilateral_contact': paired,
                          'non_target_contact': blocked,
                          'finger_normal_force_n': final[
                              'finger_normal_force_n'][action.object_id][action.hand],
                          'gripper_opening_m': final['gripper_opening_m'][action.hand],
                          'top_contact_pairs': sorted(contacts,
                              key=lambda row: row['normal_force_n'],
                              reverse=True)[:8],
                          'hard_passed': bool(hard.valid and
                                              all(hard.checks.values()))})
        if not hard.valid or not all(hard.checks.values()):
            break
        if (state.get('stop_progress_at') is not None and
                action.event == 'open_drawer' and
                final_progress >= float(state['stop_progress_at'])):
            break
        if first_persistent_loss_s is not None:
            break
        if first_persistent_blocking_s is not None:
            break
    final_relative = (final['object_rotation_world'][action.object_id].T @
                      (final['finger_center_m'][action.hand] -
                       final['object_position_m'][action.object_id]))
    elapsed = float(data.time-source_time)
    late_interval = min(.5, elapsed)
    late_rate = ((final_progress-late_progress)/late_interval
                 if late_interval > 0 else 0.)
    normalized_risk = None
    late_loaded_obstruction = False
    late_noncarried_force_n = 0.
    late_object_displacement_m = None
    if action.event in ('lift', 'transport', 'align') and len(trace) >= 3:
        late = trace[-3:]
        late_noncarried_force_n = max(
            float(row['max_noncarried_gripper_force_n']) for row in late)
        late_object_displacement_m = float(np.linalg.norm(
            np.asarray(late[-1]['object_position_m']) -
            np.asarray(late[0]['object_position_m'])))
        late_loaded_obstruction = bool(
            sum(float(row['max_noncarried_gripper_force_n']) >= 30.
                for row in late) >= 2 and late_object_displacement_m < .01)
    if start_fraction is not None and action.event in ('open_drawer', 'close_drawer'):
        nominal_range = geometry[action.target_id]['joint']['range']
        travel = abs(float(nominal_range[1]-nominal_range[0]))
        expected_rate = min(float(action.max_speed_m_s), .025)/max(travel, 1e-9)
        signed_rate = late_rate if action.event == 'open_drawer' else -late_rate
        normalized_risk = bool(signed_rate < .20*expected_rate)
    elif action.event in ('lift', 'transport', 'align'):
        normalized_risk = bool(not all(final['finger_contact'][
            action.object_id][action.hand]) and
            bilateral_steps/max(1, hard.count) < .5 or late_loaded_obstruction or
            (staged_transport and -final_progress > .008 and
             final_progress - start_progress <
             min(.008, .20 * max(initial_stage_remaining, 0.))))
    elif action.event == 'approach':
        remaining = approach_remaining(final)
        normalized_risk = bool(first_persistent_blocking_s is not None or
                               initial_approach_remaining - remaining <
                               min(.008, .20 * initial_approach_remaining))
    dependency_regression = any(
        row['start_fraction'] >= row['required_open_fraction'] and
        row['minimum_fraction'] < row['required_open_fraction'] - .005
        for row in dependent_joint.values())
    at_risk = bool(not hard.valid or not all(hard.checks.values()) or
                   first_persistent_loss_s is not None or normalized_risk or
                   dependency_regression)
    result = {'schema': 'rssr.isolated_loaded_event_preview.v1',
            'source_sha256': _source_digest(state, action, contract),
            'source_time_s': source_time, 'duration_requested_s': duration_s,
            'simulated_physics_steps': hard.count,
            'simulated_elapsed_s': elapsed, 'event': action.event,
            'object_id': action.object_id, 'hand': action.hand,
            'initial_bilateral_contact': initially_bilateral,
            'support_bilateral_fraction': (support_bilateral_steps/max(1, hard.count)
                                           if support_probe is not None else None),
            'ending_support_bilateral_contact': (
                bool(all(final['finger_contact'][support_probe][inactive]))
                if support_probe is not None else None),
            'bilateral_fraction': bilateral_steps/max(1, hard.count),
            'ending_bilateral_contact': bool(all(final['finger_contact'][
                action.object_id][action.hand])),
            'longest_bilateral_contact_s': longest_bilateral*timestep,
            'first_persistent_contact_loss_s': first_persistent_loss_s,
            'first_persistent_non_target_contact_s': first_persistent_blocking_s,
            'non_target_contact_fraction': blocking_steps/max(1, hard.count),
            'longest_non_target_contact_s': longest_blocking*timestep,
            'non_target_object_ids': sorted(blocking_object_ids),
            'maximum_non_target_penetration_m': maximum_blocking_penetration_m,
            'start_progress': start_progress, 'end_progress': final_progress,
            'initial_stage_remaining_m': initial_stage_remaining,
            'final_stage_remaining_m': (-final_progress if staged_transport else None),
            'initial_object_position_m': np.asarray(
                initial['object_position_m'][action.object_id]).tolist(),
            'final_object_position_m': np.asarray(
                final['object_position_m'][action.object_id]).tolist(),
            'final_target_position_m': (np.asarray(
                final['object_position_m'][action.target_id]).tolist()
                if action.target_id in final['object_position_m'] else None),
            'initial_approach_remaining_m': initial_approach_remaining,
            'final_approach_remaining_m': approach_remaining(final),
            'dependent_articulation': dependent_joint,
            'predicted_dependency_regression': dependency_regression,
            'reached_requested_progress_stop': bool(
                state.get('stop_progress_at') is not None and
                final_progress >= float(state['stop_progress_at'])),
            'last_half_second_progress_rate': late_rate,
            'late_loaded_obstruction': late_loaded_obstruction,
            'late_noncarried_gripper_force_n': late_noncarried_force_n,
            'late_object_displacement_m': late_object_displacement_m,
            'feature_frame_pad_displacement_m': (final_relative-first_relative).tolist(),
            'predicted_hard_passed': bool(hard.valid and all(hard.checks.values())),
            'hard_constraints': dict(hard.checks), 'hard_maxima': hard.maxima,
            'predicted_risk': at_risk,
            'trace': trace,
            'source_state_modified': bool(any(
                not np.array_equal(value, np.asarray(state[key]))
                for key, value in source_arrays.items())),
            'claim_limit': 'Isolated copied-state forecast; native task and exact replay remain binding.'}
    if not return_final_state:
        return result
    continuation = dict(state)
    continuation.update({
        'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(),
        'ctrl': data.ctrl.copy(), 'act': data.act.copy(),
        'qacc_warmstart': data.qacc_warmstart.copy(),
        'time_s': float(data.time), 'desired_qpos': target.copy(),
        'desired_qvel': velocity.copy(), 'last_q': last_q.copy(),
        'last_t': float(data.time) - timestep,
        'hold_anchors': {hand: {
            'position': np.asarray(anchor['position']).copy(),
            'rotation': np.asarray(anchor['rotation']).copy()}
            for hand, anchor in hold_anchors.items()},
        'torso_initial_qpos': initial_waist.copy()})
    return result, continuation
