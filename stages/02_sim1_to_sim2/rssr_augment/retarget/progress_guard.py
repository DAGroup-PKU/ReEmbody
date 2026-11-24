"""Protect measured progress on a still-needed articulated operation.

The guard reads native predicates and current contacts. It does not prescribe
a scene rule, move an object, change physics, or change success thresholds.
"""
import numpy as np

from .executor import _target
from .events import buffered_open_fraction


def ordered_relation_prerequisite_warning(event, contract, obs):
    """Report a measured temporal precondition at risk during relation motion.

    An ordered task can require a relation to become true while a mechanism
    remains past its threshold. This pauses the *current* short motion for the
    agent to reconsider; it does not choose the next event or change the task.
    """
    action = event.action
    if action.event not in ('transport', 'align') or not action.target_id:
        return None
    for predicate in contract['predicates']:
        if (predicate.get('kind') != 'open_insert_close' or
                predicate.get('object') != action.object_id or
                predicate.get('container') != action.target_id):
            continue
        native = obs['live_predicates'][predicate['id']]
        if native.get('measurements', {}).get('inserted_while_open', False):
            return None
        required = float(predicate['open_fraction'])
        reserve = buffered_open_fraction(required)
        actual = float(obs['articulation_fraction'][predicate['joint']])
        if actual >= reserve:
            return None
        relation = next((row for row in contract['predicates']
                         if row.get('kind') == 'inside' and
                         row.get('object') == action.object_id and
                         row.get('container') == action.target_id), None)
        if relation is not None:
            native_margin = float(obs['live_predicates'][relation['id']]
                                  ['instantaneous_margin'])
            # Intervene near a possible relation crossing. Earlier staging
            # can remain useful even with a small mechanism reserve.
            lookahead_m = .5 * float(action.max_speed_m_s) + .02
            if native_margin < -lookahead_m:
                return None
        else:
            native_margin = None
        return {'predicate_id': predicate['id'],
                'joint': predicate['joint'],
                'required_fraction': required,
                'buffered_fraction': reserve,
                'measured_fraction': actual,
                'native_relation_margin_m': native_margin,
                'native_inserted_while_open': False}
    return None


def active_slide_predicates(contract, obs):
    for predicate in contract['predicates']:
        if predicate['kind'] != 'open_insert_close':
            continue
        measured = obs['live_predicates'][predicate['id']].get('measurements', {})
        if not measured.get('inserted_while_open', False):
            yield predicate


def update_peak_and_check(contract, obs, peaks, last_fractions,
                          regression_latched=None, *, tolerance=.01):
    """Return a regression reason without resetting the running simulation."""
    regression = False
    if regression_latched is None:
        regression_latched = {}
    for predicate in active_slide_predicates(contract, obs):
        joint = predicate['joint']
        fraction = float(obs['articulation_fraction'][joint])
        peak = max(float(peaks.get(joint, fraction)), fraction)
        peaks[joint] = peak
        previous = float(last_fractions.get(joint, fraction))
        last_fractions[joint] = fraction
        if fraction >= peak - tolerance/2:
            regression_latched[joint] = False
        elif (peak >= .02 and fraction < peak - tolerance and
              fraction < previous - 5e-5 and
              not regression_latched.get(joint, False)):
            regression_latched[joint] = True
            regression = True
    return 'articulation_regression' if regression else None


def backward_loaded_recontact(event, contract, obs, peaks, *, tolerance_m=.003):
    """Screen a wrist command that would push a contacted slider backward."""
    action = event.action
    # A contact event may use a copied-state clearance IK pose rather than
    # the direct Cartesian target. Its actual motion is monitored below by
    # the native articulation regression trigger, so do not screen it using
    # the unused direct target.
    if action.event != 'approach' or not action.hand or not action.object_id:
        return None
    for predicate in active_slide_predicates(contract, obs):
        if event.source_predicate != predicate['id']:
            continue
        if float(peaks.get(predicate['joint'], 0.)) < .02:
            continue
        geometry = obs.get('object_geometry', {})
        target = geometry.get(action.object_id, {})
        contact_objects = {action.object_id, target.get('parent_id')}
        touching = any(row['object_id'] in contact_objects for row in
                       obs.get('gripper_scene_contacts', {}).get(action.hand, []))
        if not touching:
            continue
        axis = np.asarray(obs['articulation_axis_world'][predicate['joint']], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        target_point = np.asarray(_target(action, obs), dtype=float)
        hand_point = np.asarray(obs['finger_center_m'][action.hand], dtype=float)
        projection = float(np.dot(target_point - hand_point, axis))
        if projection < -tolerance_m:
            return {'predicate_id': predicate['id'], 'joint': predicate['joint'],
                    'hand': action.hand, 'projected_backwards_m': projection,
                    'measured_open_fraction': float(
                        obs['articulation_fraction'][predicate['joint']]),
                    'peak_open_fraction': float(peaks[predicate['joint']])}
    return None
