"""Immutable physical prerequisites for mutable RSSR event rules.

No task/case/object-name dispatch; bindings come from the frozen contract and
trusted base event catalogue. Rules may change scheduling, never these checks.
The caller must attach a current/cumulative trusted ``immutable_hard`` snapshot.
Missing hard measurements permit only bounded hold, never a claimed success.
"""
from __future__ import annotations

import math
import numpy as np
from .contact_geometry import contact_clearance_m, CONTACT_START_CLEARANCE_M

HARD_KEYS = (
    'robot_table_penetration_le_3mm', 'limited_joints_within_2mrad_or_2mm',
    'solver_no_warnings', 'waist_coupling_within_1mrad',
    'gripper_coupling_within_2mrad',
    'cross_arm_penetration_le_3mm', 'robot_scene_penetration_le_3mm',
    'chassis_world_position_fixed_10nm',
    'chassis_world_orientation_fixed_1e8',
)
ALLOWED = {'approach', 'contact', 'grasp', 'lift', 'transport', 'align',
           'open_drawer', 'close_drawer', 'push_close', 'release', 'clearance',
           'retreat', 'hold'}
CARRY = {'lift', 'transport', 'align', 'open_drawer', 'close_drawer', 'clearance'}
RELATIONS = {'inside_region', 'inside', 'stable_supported_on', 'cup_nested'}


def _action(event):
    return event.get('action', event) if isinstance(event, dict) else getattr(event, 'action', event)


def _get(action, name, default=None):
    return action.get(name, default) if isinstance(action, dict) else getattr(action, name, default)


def guard(event, obs, contract, events):
    """Return ``(allowed: bool, reasons: list[str])``; never mutate inputs."""
    reasons = []
    a = _action(event)
    kind = _get(a, 'event')
    hand, obj, target = (_get(a, k) for k in ('hand', 'object_id', 'target_id'))
    if kind not in ALLOWED:
        return False, ['unknown_or_reference_action']
    try:
        speed = float(_get(a, 'max_speed_m_s', .04))
        offset = np.asarray(_get(a, 'target_offset_m', (0., 0., 0.)), dtype=float)
        fraction = _get(a, 'gripper_fraction')
        bystander_clearance = _get(a, 'bystander_clearance_m')
        support_pull = _get(a, 'support_pull_m')
        if not math.isfinite(speed) or not 0 < speed <= .1:
            reasons.append('speed_outside_immutable_envelope')
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > .20):
            reasons.append('offset_outside_immutable_envelope')
        if fraction is not None and (not math.isfinite(float(fraction)) or not 0 <= float(fraction) <= 1):
            reasons.append('gripper_fraction_outside_immutable_envelope')
        if bystander_clearance is not None:
            if (kind not in ('open_drawer', 'close_drawer') or
                    not math.isfinite(float(bystander_clearance)) or
                    not 0 < float(bystander_clearance) <= .25 or
                    len(obs.get('finger_center_m', {})) < 2):
                reasons.append('bystander_clearance_outside_immutable_envelope')
        if support_pull is not None and (
                kind != 'transport' or not math.isfinite(float(support_pull)) or
                not 0. < float(support_pull) <= .02):
            reasons.append('support_pull_outside_immutable_envelope')
        if hand not in obs.get('finger_center_m', {}):
            reasons.append('unknown_manipulator')
        if kind == 'hold':
            if obj is not None or target is not None:
                reasons.append('hold_has_object_or_target')
            return not reasons, reasons
        hard = obs.get('immutable_hard')
        if not isinstance(hard, dict) or hard.get('measurements_valid') is not True:
            reasons.append('trusted_hard_measurements_missing_or_invalid')
        elif (abs(float(hard['time_s']) - float(obs['time_s'])) > 1e-7 or
              not all(hard.get('hard_constraints', {}).get(k) is True for k in HARD_KEYS)):
            reasons.append('trusted_hard_measurements_stale_or_failed')
        positions = obs['object_position_m']
        if obj not in positions or target is not None and target not in positions:
            reasons.append('unknown_object_or_target')
            return False, reasons
        for name in (obj, target):
            if name is not None and (np.asarray(positions[name]).shape != (3,) or
                                     not np.isfinite(positions[name]).all()):
                reasons.append('nonfinite_or_invalid_object_position')
        if not np.isfinite(obs['finger_center_m'][hand]).all():
            reasons.append('nonfinite_finger_position')
        base_actions = [_action(e) for e in events]
        matching = [b for b in base_actions if _get(b, 'event') == kind
                    and _get(b, 'object_id') == obj and _get(b, 'target_id') == target]
        if not matching:
            reasons.append('action_binding_outside_frozen_catalogue')
        baseline_hands = {_get(b, 'hand') for b in matching}
        if bystander_clearance is not None and not any(
                _get(b, 'hand') == hand and
                _get(b, 'bystander_clearance_m') == bystander_clearance
                for b in matching):
            reasons.append('bystander_clearance_outside_frozen_catalogue')
        if support_pull is not None:
            if not any(_get(b, 'hand') == hand and
                       _get(b, 'recovery_mode') == _get(a, 'recovery_mode') and
                       _get(b, 'support_pull_m') == support_pull
                       for b in matching):
                reasons.append('support_pull_outside_frozen_catalogue')
            inactive = next((h for h in obs.get('finger_center_m', {})
                             if h != hand), None)
            geometry = obs.get('object_geometry', {})
            features = [name for name, meta in geometry.items()
                        if meta.get('parent_id') == target and
                        meta.get('mobility') == 'static']
            if inactive is None or not any(all(obs.get('finger_contact', {})
                    .get(name, {}).get(inactive, (False, False)))
                    for name in features):
                reasons.append('support_pull_requires_bilateral_feature_contact')
        if hand not in baseline_hands and kind != 'release':
            reasons.append('hand_outside_frozen_task_binding')
        if kind == 'reposition_base':
            target_base = np.asarray(_get(a, 'base_joint_target'), dtype=float)
            start_base = np.asarray(_get(a, 'planned_base_qpos'), dtype=float)
            start_arm = np.asarray(_get(a, 'planned_arm_qpos'), dtype=float)
            start_object = np.asarray(_get(a, 'planned_object_position_m'), dtype=float)
            current_object = np.asarray(positions[obj], dtype=float)
            current_base = np.asarray(obs['base_joint_qpos'], dtype=float)
            current_arm = np.asarray(obs['arm_joint_qpos'][hand], dtype=float)
            axis_order = _get(a, 'base_axis_order')
            certificate = _get(a, 'base_certificate_sha256')
            if (target is not None or any(array.shape != shape or
                    not np.isfinite(array).all() for array, shape in (
                        (target_base, (3,)), (start_base, (3,)),
                        (start_arm, (7,)), (start_object, (3,)),
                        (current_object, (3,)), (current_base, (3,)),
                        (current_arm, (7,)))) or
                    sorted(axis_order or ()) != [0, 1, 2] or
                    not isinstance(certificate, str) or len(certificate) != 64 or
                    any(char not in '0123456789abcdef' for char in certificate)):
                reasons.append('invalid_base_capability_certificate')
            elif not any(
                    np.array_equal(target_base, np.asarray(
                        _get(b, 'base_joint_target'), dtype=float)) and
                    tuple(axis_order) == tuple(_get(b, 'base_axis_order') or ()) and
                    np.array_equal(start_base, np.asarray(
                        _get(b, 'planned_base_qpos'), dtype=float)) and
                    np.array_equal(start_arm, np.asarray(
                        _get(b, 'planned_arm_qpos'), dtype=float)) and
                    np.array_equal(start_object, np.asarray(
                        _get(b, 'planned_object_position_m'), dtype=float)) and
                    certificate == _get(b, 'base_certificate_sha256')
                    for b in matching):
                reasons.append('base_target_outside_certified_catalogue')
            else:
                if (np.max(np.abs(current_base - start_base)) > .025 or
                        np.max(np.abs(current_arm - start_arm)) > .04 or
                        np.linalg.norm(current_object - start_object) > .01):
                    reasons.append('base_path_certificate_stale')
            if any(any(bool(value) for value in hands.values())
                   for hands in obs['hand_contact'].values()) or any(
                       obs.get('gripper_scene_contacts', {}).get(h, [])
                       for h in obs['finger_center_m']):
                reasons.append('base_reposition_requires_clear_hands')
            return not reasons, list(dict.fromkeys(reasons))
        if kind in ('clearance', 'retreat') and _get(a, 'recovery_mode') not in {
                _get(b, 'recovery_mode') for b in matching}:
            reasons.append('recovery_mode_outside_frozen_catalogue')
        if kind in ('approach', 'transport') and _get(a, 'recovery_mode') not in {
                _get(b, 'recovery_mode') for b in matching
                if _get(b, 'hand') == hand}:
            reasons.append('approach_route_outside_frozen_catalogue')
        predicates = contract['predicates']
        live = obs['live_predicates']

        def row(p):
            return live[p['id']]

        def touching(o, h):
            return bool(obs['hand_contact'][o][h])

        def secure(o, h):
            return (all(obs['finger_contact'][o][h]) and
                    float(obs['bilateral_contact_seconds'][o][h]) >= .1)

        def external_support(o):
            # Native sensor exposes environmental table contact as the only
            # support token outside the scene object namespace. No name dispatch.
            return bool(set(obs['support_contact'][o]) - set(positions))

        if kind in CARRY and not secure(obj, hand):
            reasons.append('bilateral_contact_not_persistent_0p1s')
        if kind == 'push_close' and not any(obs['finger_contact'][obj][hand]):
            reasons.append('push_requires_measured_finger_contact')
        if kind in ('clearance', 'retreat'):
            other = [c for c in obs.get('gripper_scene_contacts', {}).get(hand, [])
                     if c.get('object_id') != obj]
            partial_target = (kind == 'retreat' and not secure(obj, hand) and
                any(c.get('object_id') == obj for c in
                    obs.get('gripper_scene_contacts', {}).get(hand, [])))
            active_contacts = other or ([c for c in
                obs.get('gripper_scene_contacts', {}).get(hand, [])
                if c.get('object_id') == obj] if partial_target else [])
            if not active_contacts:
                reasons.append('no_measured_recovery_contact')
            elif any(np.asarray(c.get('away_world')).shape != (3,) or
                     not np.isfinite(c['away_world']).all() or
                     abs(np.linalg.norm(c['away_world'])-1.) > .01 for c in active_contacts):
                reasons.append('invalid_contact_away_direction')
            if kind == 'retreat' and secure(obj, hand):
                reasons.append('retreat_requires_unsecured_target')
        if kind == 'contact':
            distance = contact_clearance_m(obs, obj, hand, offset)
            if not math.isfinite(distance) or distance > CONTACT_START_CLEARANCE_M:
                reasons.append('proposed_contact_region_not_reached')
        for p in predicates:
            if p['kind'] == 'base_placed_before_stack' and obj == p['upper']:
                relation = next((q for q in predicates if q['kind'] == 'inside_region'
                                 and q['object'] == p['base'] and q['region'] == p['region']), None)
                placed = (relation is not None and
                          float(row(relation)['instantaneous_margin']) >= 0 and
                          external_support(p['base']))
                if not placed:
                    reasons.append('contract_base_not_placed_before_upper')
            if p['kind'] != 'open_insert_close':
                continue
            fraction_now = float(obs['articulation_fraction'][p['joint']])
            if not math.isfinite(fraction_now):
                reasons.append('nonfinite_articulation_fraction')
            if (kind in ('transport', 'align') and obj == p['object']
                    and target == p['container'] and fraction_now < float(p['open_fraction'])):
                reasons.append('contract_container_not_open_before_insertion')
            if kind in ('close_drawer', 'push_close') and target == p['container']:
                if row(p).get('measurements', {}).get('inserted_while_open') is not True:
                    reasons.append('contract_insertion_not_measured_before_close')
                if kind == 'push_close':
                    payload_contact = obs['hand_contact'].get(p['object'], {})
                    if any(bool(value) for value in payload_contact.values()):
                        reasons.append('push_before_payload_release')
                    inside = next((q for q in predicates if q['kind'] == 'inside'
                                   and q['object'] == p['object'] and
                                   q.get('container') == p['container']), None)
                    if (inside is not None and
                            float(row(inside)['instantaneous_margin']) < 0):
                        reasons.append('push_before_payload_inside')
                    if fraction_now <= float(p['closed_fraction']):
                        reasons.append('push_after_articulation_closed')
        if kind == 'release':
            if hand not in baseline_hands and not touching(obj, hand):
                reasons.append('alternate_release_hand_not_currently_touching')
            relation = next((p for p in predicates if p['kind'] in RELATIONS
                             and p['object'] == obj), None)
            correct_support = False
            geometry_ready = False
            if relation is not None:
                support = relation.get('support', relation.get('container'))
                correct_support = (support in obs['support_contact'][obj]
                                   if support is not None else external_support(obj))
                geometry_ready = float(row(relation)['instantaneous_margin']) >= 0
                destination = next((b for b in base_actions
                                    if _get(b, 'object_id') == obj
                                    and _get(b, 'event') in ('transport', 'align')), None)
                if destination is not None:
                    goal = (np.asarray(positions[_get(destination, 'target_id')]) +
                            np.asarray(_get(destination, 'target_offset_m', (0., 0., 0.))))
                    delta = np.asarray(positions[obj]) - goal
                    geometry_ready = bool(geometry_ready and abs(delta[2]) <= .006
                                          and np.linalg.norm(delta[:2]) <= .008)
            else:
                # A released articulation handle is bound by trusted close
                # event's target, not a hardcoded handle object name.
                closers = [b for b in base_actions if _get(b, 'event') == 'close_drawer'
                           and _get(b, 'object_id') == obj]
                for b in closers:
                    p = next((p for p in predicates if p['kind'] == 'open_insert_close'
                              and p['container'] == _get(b, 'target_id')), None)
                    if p is not None:
                        geometry_ready |= bool(
                            row(p).get('measurements', {}).get('inserted_while_open') is True and
                            float(obs['articulation_fraction'][p['joint']]) <= float(p['closed_fraction']))
            if not (geometry_ready or correct_support):
                reasons.append('release_without_frozen_geometry_or_correct_target_support')
        # A current lift is required for carrying toward a destination; a
        # latched historical lift does not prove the object remains lifted.
        if kind in ('transport', 'align'):
            lift = next((p for p in predicates if p['kind'] == 'lift_with_contact'
                         and p['object'] == obj), None)
            if lift is not None and float(row(lift).get('measurements', {}).get('lift_m', -1)) < float(lift['height_m']):
                reasons.append('contract_object_not_currently_lifted')
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        reasons.append('missing_or_invalid_trusted_measurement')
    return not reasons, list(dict.fromkeys(reasons))
