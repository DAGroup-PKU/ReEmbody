"""Prevent a soft rule from repeatedly reissuing a completed physical event.

This uses only same-episode measured completion and the current physical
state. If a grasp is lost or an object falls below its lift requirement, the
event becomes eligible again. No task, object, or case name is special-cased.
"""
import numpy as np
from .events import signed_margin


def _template_id(event_id):
    return event_id.split(':rule:', 1)[0]


def _still_complete(event, obs, contract):
    action = event.action
    obj, hand = action.object_id, action.hand
    if action.event == 'grasp':
        return (all(obs['finger_contact'][obj][hand]) and
                float(obs['bilateral_contact_seconds'][obj][hand]) >= .1)
    if action.event == 'lift':
        lift = next((p for p in contract['predicates']
                     if p['kind'] == 'lift_with_contact' and p['object'] == obj), None)
        if not all(obs['finger_contact'][obj][hand]):
            return False
        if lift is not None:
            return (float(obs['live_predicates'][lift['id']]['measurements'].get('lift_m', -1.)) >=
                    float(lift['height_m']))
        # A target-relative lift has an exact measured completion condition
        # even when the native contract has no standalone lift predicate.
        # Re-offer only if the object falls below that target or the grasp is
        # lost. This uses the same geometry as the physical event margin.
        target = action.target_id
        positions = obs['object_position_m']
        if target is None or obj not in positions or target not in positions:
            return False
        return (float(positions[obj][2]) >=
                float(positions[target][2]) + float(action.target_offset_m[2]))
    if action.event in ('approach', 'contact', 'transport', 'align',
                        'release', 'clearance', 'retreat', 'open_drawer', 'close_drawer',
                        'push_close', 'reposition_base'):
        measured = dict(obs)
        for name in ('object_position_m', 'finger_center_m',
                     'hand_rotation_world', 'object_rotation_world'):
            measured[name] = {key: np.asarray(value, dtype=float)
                              for key, value in obs[name].items()}
        return signed_margin(event, measured, None, contract) >= 0.0
    return False


def filter_completed_reoffers(events, obs, history, contract):
    """Return eligible events and measured reasons for filtered reoffers."""
    completed = {(_template_id(row['event_id']), row.get('selected_object_id'),
                  row.get('selected_hand')) for row in history
                 if row.get('stop_reason') == 'event_reached'}
    eligible, rejected = [], []
    for event in events:
        key = (_template_id(event.event_id), event.action.object_id,
               event.action.hand)
        if key in completed and _still_complete(event, obs, contract):
            rejected.append({'event_id': event.event_id,
                             'reason': 'same_episode_event_reached_and_still_complete'})
        else:
            eligible.append(event)
    return eligible, rejected
