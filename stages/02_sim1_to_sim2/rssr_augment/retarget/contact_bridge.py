"""Measured missing transition from a reached approach to physical contact."""
import numpy as np

from .executor import desired_rotation


def contact_bridge_gaps(available, obs, all_events):
    """Describe reachable contact templates without choosing a rule for them."""
    gaps = []
    for approach in available:
        action = approach.action
        if action.event != 'approach' or action.object_id is None or action.hand is None:
            continue
        obj, hand = action.object_id, action.hand
        if obs['hand_contact'].get(obj, {}).get(hand, False):
            continue
        target = (np.asarray(obs['object_position_m'][obj]) +
                  np.asarray(action.target_offset_m))
        distance = float(np.linalg.norm(np.asarray(obs['finger_center_m'][hand]) - target))
        current = np.asarray(obs['hand_rotation_world'][hand])
        desired = desired_rotation(obj, obs)
        rotation_error = float(np.arccos(np.clip(
            (np.trace(desired.T @ current) - 1.) / 2., -1., 1.)))
        # Match the immutable contact-event distance gate, so the bridge can
        # be offered before an approach pose is declared fully reached.
        if distance > .012 or rotation_error > .15:
            continue
        paired = [event for event in available
                  if event.action.event == 'contact' and
                  event.action.object_id == obj and event.action.hand == hand and
                  np.allclose(event.action.target_offset_m,
                              action.target_offset_m, atol=1e-9, rtol=0.)]
        if paired:
            continue
        templates = [event for event in all_events
                     if event.action.event == 'contact' and
                     event.action.object_id == obj and event.action.hand == hand]
        if not templates:
            continue
        template = min(templates, key=lambda event: float(np.linalg.norm(
            np.asarray(event.action.target_offset_m) -
            np.asarray(action.target_offset_m))))
        delta = np.asarray(action.target_offset_m) - np.asarray(
            template.action.target_offset_m)
        gaps.append({'approach_event_id': approach.event_id,
                     'contact_template_event_id': template.event_id,
                     'object_id': obj, 'hand': hand,
                     'approach_target_offset_m': list(action.target_offset_m),
                     'contact_offset_delta_m': delta.tolist(),
                     'rule_offset_within_envelope': bool(
                         np.max(np.abs(delta)) <= .025 and
                         np.linalg.norm(delta) <= .04),
                     'finger_to_approach_target_m': distance,
                     'hand_rotation_error_rad': rotation_error,
                     'physical_contact_observed': False})
    return sorted(gaps, key=lambda row: (
        not row['rule_offset_within_envelope'],
        row['finger_to_approach_target_m'], row['approach_event_id']))
