"""Shared contract-derived seed scheduler; runtime rules may change soft gates.

Bindings come from contracts and event actions, never case IDs or object names.
This module neither changes native predicates nor certifies task success.
"""
from collections import Counter
from dataclasses import replace
import numpy as np
from .events import buffered_open_fraction
from .progress_guard import ordered_relation_prerequisite_warning
from .contact_geometry import contact_clearance_m, CONTACT_START_CLEARANCE_M

RELATION_KINDS = {'inside_region', 'inside', 'stable_supported_on', 'cup_nested'}
FAILURE_REASONS = {'stagnation', 'event_timeout', 'contact_lost',
                   'protected_predicate_lost', 'non_target_contact',
                   'articulation_regression', 'target_nonpad_contact',
                   'ordered_prerequisite_at_risk'}


def base_certificate_retry_available(event, history):
    """Count failed uses of the current measured proposal, not an older pose."""
    if event.action.event != 'reposition_base':
        return False
    same_certificate_failures = sum(
        row['event_id'] == event.event_id and
        row.get('base_certificate_sha256') ==
            event.action.base_certificate_sha256 and
        row.get('stop_reason') in FAILURE_REASONS
        for row in history)
    return same_certificate_failures < 2


def episode_memory(history):
    failures = Counter()
    last = {}
    achieved = set()
    collision_modes = Counter()
    collision_objects = {}
    for row in history:
        event = row['event_id']
        reason = row.get('stop_reason')
        if reason in FAILURE_REASONS:
            failures[event] += 1
            last[event] = {'time_s': row['time_s'], 'reason': reason,
                           'signed_margin': row.get('signed_margin')}
        if reason == 'event_reached':
            achieved.add(event)
        if reason == 'non_target_contact':
            hand = row.get('selected_hand')
            target = row.get('selected_object_id')
            blockers = sorted({c['object_id'] for c in
                               row.get('gripper_scene_contacts', {}).get(hand, [])
                               if c.get('object_id') != target})
            collision_modes[event] += 1
            collision_objects.setdefault(event, Counter()).update(blockers)
    frequent_collisions = [
        {'event_id': event, 'count': count,
         'contacted_object_ids': dict(collision_objects[event].most_common(4)),
         'last_time_s': last[event]['time_s']}
        for event, count in collision_modes.most_common(6)]
    return {'failed_attempts': {k: {'count': v, **last[k]} for k, v in sorted(failures.items())},
            'frequent_collision_modes': frequent_collisions,
            'measured_event_completions': sorted(achieved),
            'history_records_consumed': len(history),
            'scope': 'same ongoing episode; no cross-case knowledge or success labels'}


def decision_context(events, obs, contract, history=()):
    """Return feasible events and an explicit account of blocked physical facts."""
    predicates = contract['predicates']
    live = obs['live_predicates']
    by_id = {p['id']: p for p in predicates}
    relations = {p['object']: p for p in predicates if p['kind'] in RELATION_KINDS}
    lifts = {p['object']: p for p in predicates if p['kind'] == 'lift_with_contact'}
    operations = [p for p in predicates if p['kind'] == 'open_insert_close']
    orders = [p for p in predicates if p['kind'] == 'base_placed_before_stack']
    positions = obs['object_position_m']
    destinations = {e.action.object_id: e for e in events
                    if e.action.event in ('transport', 'align')}
    holds = [e for e in events if e.action.event == 'hold']
    memory = episode_memory(history)
    blocked = []
    feasible = []

    def row(pred):
        return live[pred['id']]

    def touching(obj):
        return [h for h, value in obs.get('hand_contact', {}).get(obj, {}).items() if value]

    def secure(obj, hand):
        return (all(obs['finger_contact'][obj][hand]) and
                obs.get('bilateral_contact_seconds', {}).get(obj, {}).get(hand, 0.) >= .1)

    def historically_lifted(obj):
        return obj not in lifts or row(lifts[obj])['passed']

    def goal_geometry(obj):
        p = relations.get(obj)
        return p is not None and row(p)['instantaneous_margin'] >= 0 and historically_lifted(obj)

    def supported(obj):
        p = relations.get(obj)
        contacts = obs['support_contact'].get(obj, ())
        if p is None:
            return bool(contacts)
        support = p.get('support', p.get('container'))
        return support in contacts if support is not None else bool(contacts)

    def settling(obj):
        """Support may be established before a finite native window is complete."""
        if not supported(obj) or not historically_lifted(obj):
            return False
        p = relations.get(obj)
        if p is None:
            return False
        r = row(p)
        if r['instantaneous_margin'] >= 0:
            return True
        # Native measurement names describe constraint components and units.
        # Only vertical support settling is deferred; confinement/tilt remain.
        m = r.get('measurements', {})
        confinement = {k: v for k, v in m.items()
                       if k not in {'floor_clearance_m', 'height_clearance_m'}}
        return bool(confinement) and all(np.isfinite(v) and v >= 0 for v in confinement.values())

    def parked(obj, hand):
        delta = np.asarray(obs['finger_center_m'][hand]) - np.asarray(positions[obj])
        return delta[2] >= .10 and np.linalg.norm(delta[:2]) >= .17

    def dependencies(obj):
        reasons = []
        for p in orders:
            if p['upper'] != obj:
                continue
            base = p['base']
            if not goal_geometry(base) or not supported(base):
                reasons.append({'fact': 'predecessor_placed_and_supported', 'object': base})
            if touching(base):
                reasons.append({'fact': 'predecessor_released', 'object': base})
            hand = lifts[base]['hand'] if base in lifts else destinations[base].action.hand
            if not parked(base, hand):
                reasons.append({'fact': 'predecessor_hand_withdrawn', 'object': base, 'hand': hand})
        for p in operations:
            if p['object'] != obj:
                continue
            measured = row(p)['measurements']
            if (not measured.get('inserted_while_open', False) and
                    obs['articulation_fraction'][p['joint']] < p['open_fraction']):
                reasons.append({'fact': 'container_open', 'container': p['container'], 'joint': p['joint']})
        return reasons

    def articulation(event):
        a = event.action
        p = next((p for p in operations if p['container'] == a.target_id), None)
        if p is None:
            return None
        measured = row(p)['measurements']
        fraction = obs['articulation_fraction'][p['joint']]
        inserted = bool(measured.get('inserted_while_open', False))
        released = not touching(p['object'])
        needs_open = not inserted and fraction < buffered_open_fraction(p['open_fraction'])
        needs_close = inserted and released and fraction > p['closed_fraction'] and goal_geometry(p['object'])
        closed = inserted and released and fraction <= p['closed_fraction']
        return p, needs_open, needs_close, closed

    actuators = {}
    for e in events:
        if e.action.event in ('open_drawer', 'close_drawer', 'push_close'):
            actuators[e.action.object_id] = articulation(e)

    hard_failed = (obs['min_joint_margin_native'] < -.002 or
                   obs['max_robot_table_penetration_m'] > .003)
    if hard_failed or all(r['passed'] for r in live.values()):
        return holds, {'memory': memory, 'blocked_events': [],
                       'reason': 'irreversible_hard_failure' if hard_failed else 'all_live_predicates_measured'}

    for e in events:
        a = e.action
        kind, obj, hand = a.event, a.object_id, a.hand
        reasons = []
        if kind in ('reference', 'hold'):
            continue
        act = actuators.get(obj)
        if kind == 'reposition_base':
            # This is a robot capability proposal, bound to a currently
            # feasible pickup event. It must not move a held task object.
            if any(any(bool(value) for value in hands.values())
                   for hands in obs['hand_contact'].values()):
                reasons.append({'fact': 'all_hands_clear_for_base_travel'})
        elif kind == 'retreat':
            # A measured non-target collision can require moving an empty
            # hand before any task dependency becomes satisfiable.
            pass
        elif act is not None:
            _, needs_open, needs_close, closed = act
            if kind == 'release':
                if not closed:
                    reasons.append({'fact': 'articulation_sequence_closed'})
            elif kind == 'open_drawer':
                operation = act[0]
                required_open = (buffered_open_fraction(operation['open_fraction'])
                                 if e.event_id.split(':rule:', 1)[0].endswith(':buffered')
                                 else operation['open_fraction'])
                if (row(operation)['measurements'].get('inserted_while_open', False) or
                        obs['articulation_fraction'][operation['joint']] >= required_open):
                    reasons.append({'fact': 'opening_currently_needed'})
            elif kind in ('close_drawer', 'push_close'):
                if not needs_close:
                    reasons.append({'fact': 'insertion_released_and_closing_needed'})
            elif not (needs_open or needs_close):
                reasons.append({'fact': 'articulation_contact_recovery_needed'})
        else:
            dependencies_now = dependencies(obj)
            if kind in ('clearance', 'retreat'):
                # Clearance is a bounded collision-recovery motion. A moving
                # container may have partly closed during the collision; keep
                # clearance available so the grasped object can separate
                # before the articulation is reopened. The immutable guard
                # still requires secure grasp and real noncarried contact.
                dependencies_now = [reason for reason in dependencies_now
                                    if reason['fact'] != 'container_open']
            reasons.extend(dependencies_now)
            if kind not in ('release', 'retreat') and settling(obj):
                reasons.append({'fact': 'support_has_replaced_hand_support'})

        contact = secure(obj, hand)
        if kind == 'reposition_base':
            if contact:
                reasons.append({'fact': 'target_contact_not_yet_established'})
        elif kind in ('approach', 'contact'):
            if any(c['object_id'] != obj for c in
                   obs.get('gripper_scene_contacts', {}).get(hand, [])):
                reasons.append({'fact': 'non_target_gripper_contact_requires_retreat'})
            distance = contact_clearance_m(obs, obj, hand, a.target_offset_m)
            if contact:
                reasons.append({'fact': 'grasp_already_secure'})
            if kind == 'contact' and distance > CONTACT_START_CLEARANCE_M:
                reasons.append({'fact': 'contact_surface_reached',
                                'surface_clearance_m': distance})
            if kind == 'approach' and distance < .008:
                reasons.append({'fact': 'approach_already_reached'})
        elif kind == 'grasp':
            distance = contact_clearance_m(obs, obj, hand)
            if contact or not (any(obs['finger_contact'][obj][hand]) or
                               distance < CONTACT_START_CLEARANCE_M):
                reasons.append({'fact': 'near_or_partial_contact_needs_stabilization'})
        elif kind == 'clearance':
            if not contact:
                reasons.append({'fact': 'bilateral_contact_persistent',
                                'object': obj, 'hand': hand})
            extra = [c['object_id'] for c in
                     obs.get('gripper_scene_contacts', {}).get(hand, [])
                     if c['object_id'] != obj]
            if not extra:
                reasons.append({'fact': 'noncarried_scene_contact_present'})
        elif kind == 'retreat':
            if contact:
                reasons.append({'fact': 'target_grasp_already_secure'})
            extra = [c['object_id'] for c in
                     obs.get('gripper_scene_contacts', {}).get(hand, [])
                     if c['object_id'] != obj]
            partial_target = (not contact and any(
                c['object_id'] == obj for c in
                obs.get('gripper_scene_contacts', {}).get(hand, [])))
            if not extra and not partial_target:
                reasons.append({'fact': 'measured_recovery_contact_present'})
        elif kind == 'push_close':
            if not any(obs['finger_contact'][obj][hand]):
                reasons.append({'fact': 'at_least_one_finger_contacts_articulation_feature',
                                'object': obj, 'hand': hand})
        elif kind in ('lift', 'transport', 'align', 'open_drawer', 'close_drawer'):
            if not contact:
                reasons.append({'fact': 'bilateral_contact_persistent', 'object': obj, 'hand': hand})
            if kind in ('transport', 'align'):
                warning = ordered_relation_prerequisite_warning(e, contract, obs)
                if warning is not None:
                    reasons.append({'fact': 'ordered_prerequisite_reserve', **warning})
            if kind in ('lift', 'transport', 'align'):
                lift_event = next((x for x in events if x.action.event == 'lift' and x.action.object_id == obj), None)
                if obj in lifts:
                    lifted_now = row(lifts[obj]).get('measurements', {}).get('lift_m', 0.) >= lifts[obj]['height_m']
                elif lift_event is not None and lift_event.action.target_id is not None:
                    lifted_now = positions[obj][2] >= positions[lift_event.action.target_id][2] + lift_event.action.target_offset_m[2]
                elif lift_event is not None:
                    # A contract without an explicit lift predicate can still
                    # demand placement. Require secure grasp and measured loss
                    # of environmental support before transport.
                    lifted_now = contact and not obs['support_contact'].get(obj)
                else:
                    lifted_now = False
                if kind == 'lift' and lifted_now:
                    reasons.append({'fact': 'lift_currently_satisfied'})
                if kind in ('transport', 'align') and not lifted_now:
                    relation = relations.get(obj)
                    relation_margin = (float(row(relation)['instantaneous_margin'])
                                       if relation is not None else -np.inf)
                    relation_target = (relation.get('support', relation.get('container'))
                                       if relation is not None else None)
                    # The geometric lift target is an intermediate waypoint,
                    # not a native prerequisite when no lift predicate exists.
                    # A carried object that is already close to its native
                    # destination may need to continue toward that relation
                    # rather than return to the earlier lift waypoint.
                    relation_tolerance = (float(relation.get('clearance_tolerance_m', 0.))
                                          if relation is not None else 0.)
                    near_relation = (obj not in lifts and secure(obj, hand) and
                        relation_target == a.target_id and
                        np.isfinite(relation_margin) and
                        -.5 * float(a.max_speed_m_s) - relation_tolerance <=
                        relation_margin < 0.)
                    if not near_relation:
                        reasons.append({'fact': 'object_currently_lifted'})
        elif kind == 'release' and act is None:
            d = destinations.get(obj)
            ready = False
            if d is not None and goal_geometry(obj):
                target_z = (positions[d.action.target_id][2] +
                            d.action.target_offset_m[2])
                # A planar native relation can pass while the object is still
                # high above its destination. Keep the shared vertical arrival
                # condition, and let the frozen relation decide horizontal fit.
                ready = abs(positions[obj][2] - target_z) <= .006
            if not (ready or settling(obj)):
                reasons.append({'fact': 'destination_ready_or_supported_settling'})
            if not touching(obj) and parked(obj, hand):
                reasons.append({'fact': 'release_and_withdrawal_already_complete'})
        else:
            if kind != 'release':
                reasons.append({'fact': 'unsupported_event_semantics', 'event': kind})
        if reasons:
            blocked.append({'event_id': e.event_id, 'unmet': reasons})
        else:
            feasible.append(e)

    # Release the hand that actually contacts the supported object, including
    # contact by the other arm; native hand-specific lift requirements remain.
    for obj in relations:
        if not settling(obj) or dependencies(obj):
            continue
        template = next((e for e in events if e.action.event == 'release' and e.action.object_id == obj), None)
        if template is None:
            continue
        for hand in touching(obj):
            if hand == template.action.hand:
                continue
            feasible.append(replace(template, event_id=template.event_id + ':actual_' + hand,
                                    action=replace(template.action, hand=hand)))

    goal_objects = set(relations)
    terminal_articulations = all(articulation(e)[3] for e in events
                                if e.action.event == 'close_drawer')
    if (goal_objects and all(settling(obj) for obj in goal_objects)
            and not any(touching(obj) for obj in goal_objects) and terminal_articulations):
        feasible.extend(holds)

    # Structured episode memory preserves failed modes beyond the last 6 rows.
    failed = memory['failed_attempts']

    def renewed_effect_progress(event):
        """A changed measured effect can justify retrying a failed actuator.

        The retry counter prevents repeated commands in the same state. It
        should not permanently suppress an actuator after another action has
        moved its native effect closer to the contract threshold.
        """
        if event.action.event not in ('open_drawer', 'close_drawer', 'push_close'):
            return False
        previous = failed.get(event.event_id, {}).get('signed_margin')
        state = actuators.get(event.action.object_id)
        if previous is None or state is None:
            return False
        operation = state[0]
        fraction = obs['articulation_fraction'][operation['joint']]
        opening_target = (buffered_open_fraction(operation['open_fraction'])
                          if event.event_id.split(':rule:', 1)[0].endswith(':buffered')
                          else operation['open_fraction'])
        margin = (fraction - opening_target
                  if event.action.event == 'open_drawer'
                  else operation['closed_fraction'] - fraction)
        return margin >= previous + .015

    def renewed_grasp_after_contact_loss(event):
        """Allow one actuator retry when a genuinely new hold has stabilized."""
        if event.action.event not in ('open_drawer', 'close_drawer'):
            return False
        previous = failed.get(event.event_id, {})
        if previous.get('reason') != 'contact_lost':
            return False
        obj, hand = event.action.object_id, event.action.hand
        if not secure(obj, hand):
            return False
        age = float(obs.get('bilateral_contact_seconds', {}).get(obj, {}).get(hand, 0.))
        now = float(obs['time_s'])
        # The failed actuator must predate this uninterrupted contact episode.
        # Require extra dwell beyond the native 0.1 s grasp gate so a noisy
        # single contact frame does not reset the retry limit.
        return age >= .5 and now - age > float(previous['time_s']) + 1e-3

    alternatives = [e for e in feasible if
                    failed.get(e.event_id, {}).get('count', 0) < 2 or
                    renewed_effect_progress(e) or
                    renewed_grasp_after_contact_loss(e) or
                    base_certificate_retry_available(e, history)]
    if alternatives:
        feasible = alternatives
    unique = {e.event_id: e for e in feasible}
    dead_end = not unique
    if dead_end:
        # Bounded observation exposes a dead end without historical commands.
        unique = {e.event_id: e for e in holds}
    return list(unique.values()), {'memory': memory, 'blocked_events': blocked,
                                  'dead_end': dead_end, 'goal_predicates': list(by_id),
                                  'reason': 'no_feasible_recovery' if dead_end else 'measured_physical_preconditions'}
