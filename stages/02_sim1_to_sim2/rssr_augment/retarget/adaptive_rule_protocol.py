"""Bounded runtime RuleAgent protocol for the event_v27 executor.

Model-generated conditions are data, never Python or executable source code.
Native safety/task preconditions must be supplied by an independent guard.
"""
from dataclasses import replace
import hashlib
import json
import math
import numpy as np


SCHEMA = 'rssr.rule_patch.v1'
OBS_ROOTS = {'time_s', 'live_predicates', 'finger_contact',
             'finger_normal_force_n', 'hand_contact',
             'bilateral_contact_seconds', 'support_contact', 'articulation_fraction',
             'gripper_opening_m', 'object_position_m', 'object_velocity_m_s',
             'min_joint_margin_native', 'arm_torso_min_joint_margin_rad',
             'gripper_min_joint_margin_rad',
             'max_robot_table_penetration_m'}
OBS_ROOTS.add('immutable_hard')
OBS_ROOTS.add('finger_center_m')
OBS_ROOTS.add('object_geometry')
OBS_ROOTS.add('articulation_axis_world')
OBS_ROOTS.add('hand_rotation_world')
OBS_ROOTS.add('object_rotation_world')
OBS_ROOTS.add('manipulator_joint_margin_rad')
OBS_ROOTS.add('limiting_manipulator_joint')
OBS_ROOTS.add('gripper_scene_contacts')
OBS_ROOTS.add('gripper_scene_contact_object_ids')
OBS_ROOTS.add('base_joint_qpos')
OBS_ROOTS.add('arm_joint_qpos')
OBS_ROOTS.add('torso_max_physical_excursion_rad')
OBS_ROOTS.add('torso_max_command_excursion_rad')
EXPECTED_ROOTS = {'live_predicates', 'finger_contact',
                  'finger_normal_force_n', 'hand_contact',
                  'bilateral_contact_seconds', 'articulation_fraction',
                  'gripper_opening_m', 'object_position_m', 'object_velocity_m_s',
                  'finger_center_m', 'min_joint_margin_native',
                  'arm_torso_min_joint_margin_rad',
                  'gripper_min_joint_margin_rad',
                  'manipulator_joint_margin_rad',
                  'base_joint_qpos',
                  'torso_max_physical_excursion_rad',
                  'max_robot_table_penetration_m'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def lookup(obs, path):
    if not isinstance(path, list) or not path or path[0] not in OBS_ROOTS or len(path) > 7:
        raise ValueError('Unapproved observation path')
    value = obs
    for key in path:
        if isinstance(value, dict) and isinstance(key, str):
            value = value[key]
        elif isinstance(value, (tuple, list)) or hasattr(value, 'shape'):
            if type(key) is not int or key < 0:
                raise ValueError('Invalid vector index')
            value = value[key]
        else:
            raise ValueError('Path crosses a scalar')
    return value


def _number(value):
    value = _scalar(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return float(value)


def _scalar(value):
    """Normalize a NumPy scalar exactly as the pinned JSON observation does."""
    return value.item() if isinstance(value, np.generic) else value


def expression(expr, obs, depth=0, budget=None):
    if budget is None:
        budget = [64]
    budget[0] -= 1
    if depth > 5 or budget[0] < 0 or not isinstance(expr, dict):
        raise ValueError('Condition AST exceeds bounds')
    if len(expr) == 1 and next(iter(expr)) in ('all', 'any', 'not'):
        op = next(iter(expr))
        if op == 'not':
            return not expression(expr[op], obs, depth + 1, budget)
        children = expr[op]
        if not isinstance(children, list) or not children or len(children) > 16:
            raise ValueError('Invalid boolean children')
        # Validate every child, including branches that would short circuit.
        values = [expression(child, obs, depth + 1, budget) for child in children]
        return all(values) if op == 'all' else any(values)
    if set(expr) != {'path', 'op', 'value'}:
        raise ValueError('Invalid comparison fields')
    left, right, op = _scalar(lookup(obs, expr['path'])), _scalar(expr['value']), expr['op']
    if op in ('lt', 'le', 'gt', 'ge'):
        left, right = _number(left), _number(right)
        return {'lt': left < right, 'le': left <= right,
                'gt': left > right, 'ge': left >= right}[op]
    if op in ('eq', 'ne'):
        if type(right) not in (bool, str, int, float) or type(left) not in (bool, str, int, float):
            raise ValueError('Equality requires scalar values')
        # JSON has one spelling for an integral zero while native measurements
        # commonly contain 0.0. Treat finite int/float scalars as the same
        # numeric domain; keep booleans and strings type strict.
        if type(left) in (int, float) and type(right) in (int, float):
            left, right = _number(left), _number(right)
        elif type(left) is not type(right):
            raise ValueError('Equality types differ')
        return left == right if op == 'eq' else left != right
    if op == 'contains' and isinstance(left, (list, tuple)) and isinstance(right, str):
        return right in left
    raise ValueError('Unsupported comparison')


def expected_met(expected, before, after):
    a, b = (_scalar(lookup(source, expected['path'])) for source in (before, after))
    kind, value = expected['change'], _scalar(expected['value'])
    if kind == 'becomes':
        if type(value) not in (bool, str) or type(a) is not type(value) or type(b) is not type(value):
            raise ValueError('Invalid state transition')
        if a == value:
            raise ValueError('Expected transition already holds at exposure')
        return b == value
    delta = _number(value)
    if delta <= 0:
        raise ValueError('Expected delta must be positive')
    a, b = _number(a), _number(b)
    if kind == 'increase_by':
        return b - a >= delta
    if kind == 'decrease_by':
        return a - b >= delta
    raise ValueError('Unsupported expected change')


def validate_patch(patch, obs, events, parent_sha256, evidence_sha256):
    required = {'schema', 'parent_sha256', 'evidence_sha256', 'reason', 'rules'}
    if set(patch) != required or patch['schema'] != SCHEMA:
        raise ValueError('Invalid rule patch schema')
    if patch['parent_sha256'] != parent_sha256 or patch['evidence_sha256'] != evidence_sha256:
        raise ValueError('Stale revision or unbound observation evidence')
    if not isinstance(patch['reason'], str) or not patch['reason'].strip():
        raise ValueError('Missing evidence-grounded reason')
    if not isinstance(patch['rules'], list) or not 1 <= len(patch['rules']) <= 3:
        raise ValueError('Invalid rule count')
    catalog = {e.event_id: e for e in events}
    ids = set()
    for rule in patch['rules']:
        if set(rule) != {'id', 'when', 'offer', 'defer', 'expected', 'observe_s', 'ttl_s'}:
            raise ValueError('Invalid rule fields')
        if not isinstance(rule['id'], str) or not rule['id'] or rule['id'] in ids:
            raise ValueError('Invalid or duplicated rule ID')
        ids.add(rule['id'])
        expression(rule['when'], obs)
        observe, ttl = _number(rule['observe_s']), _number(rule['ttl_s'])
        if not .5 <= observe <= ttl <= 8.:
            raise ValueError('Invalid simulated-time observation bounds')
        if not isinstance(rule['offer'], list) or len(rule['offer']) > 6:
            raise ValueError('Invalid offers')
        if not isinstance(rule['defer'], list) or len(rule['defer']) > 6:
            raise ValueError(f"Invalid deferrals in rule {rule['id']}: maximum 6 event IDs")
        if not rule['offer'] and not rule['defer']:
            raise ValueError('Rule has no candidate effect')
        if any(event_id not in catalog or catalog[event_id].action.event == 'reference' for event_id in rule['defer']):
            raise ValueError('Unknown deferred event')
        for offer in rule['offer']:
            if set(offer) != {'event_id', 'priority', 'offset_delta_m', 'hand'}:
                raise ValueError('Invalid parameterized offer')
            if offer['event_id'] not in catalog:
                raise ValueError('Unknown action template')
            action = catalog[offer['event_id']].action
            if action.event == 'reference':
                raise ValueError('Historical reference commands are excluded')
            if abs(_number(offer['priority'])) > 10:
                raise ValueError('Invalid priority')
            delta = offer['offset_delta_m']
            if not isinstance(delta, list) or len(delta) != 3:
                raise ValueError('Invalid contact offset')
            values = [_number(x) for x in delta]
            if max(abs(x) for x in values) > .025 or sum(x*x for x in values) > .04**2:
                raise ValueError('Contact proposal exceeds immutable envelope')
            if action.event in ('transport', 'align'):
                # Let RuleAgent retain measured clearance during a carry.
                # A signed horizontal shift or downward correction requires
                # other geometric evidence and is outside this safe template.
                if values[0] or values[1] or not 0. <= values[2] <= .025:
                    raise ValueError(
                        f"Rule {rule['id']} offer {offer['event_id']}: "
                        'carry offset must be [0,0,dz] with 0<=dz<=0.025m')
            elif any(values) and action.event not in ('approach', 'contact'):
                raise ValueError(
                    f"Rule {rule['id']} offer {offer['event_id']}: "
                    'offset_delta_m must be [0,0,0] outside approach/contact/carry')
            if offer['hand'] is not None:
                available_manipulators = {e.action.hand for e in events
                                          if e.action.hand is not None}
                if offer['hand'] not in available_manipulators:
                    raise ValueError('Unknown proposed manipulator')
                if offer['hand'] != action.hand and action.event != 'release':
                    paired = any(e.action.event == action.event and
                                 e.action.object_id == action.object_id and
                                 e.action.target_id == action.target_id and
                                 e.action.hand == offer['hand'] for e in events)
                    if not paired:
                        raise ValueError('Proposed hand is outside contract-compiled capabilities')
        ex = rule['expected']
        if set(ex) != {'path', 'change', 'value'}:
            raise ValueError('Invalid measurable expectation')
        if not isinstance(ex['path'], list) or not ex['path'] or ex['path'][0] not in EXPECTED_ROOTS:
            raise ValueError(
                f"Rule {rule['id']} expected.path {ex['path']!r}: "
                'must name a live physical measurement')
        # Reject a vacuous expectation now; recheck at actual effect exposure.
        expected_met(ex, obs, obs)
    return digest(patch)


def apply_patch(patch, obs, base_events, all_events, immutable_guard, *, activated_time_s):
    """Guard is mandatory and applies to base and model-proposed actions alike.

    The guard returns (allowed, immutable rejection reasons). It must validate
    native causal requirements and physical action preconditions independently
    of mutable rules. An AST passing schema checks is not a safety certificate.
    """
    if not callable(immutable_guard):
        raise ValueError('Independent immutable guard is required')
    catalog = {e.event_id: e for e in all_events}
    chosen = {e.event_id: (0., e) for e in base_events if e.action.event != 'reference'}
    effect_owners = {}
    deferred = {}
    inactive_missing_paths = []
    for rule in patch['rules']:
        elapsed = obs['time_s'] - activated_time_s
        if elapsed < 0:
            raise ValueError('Rule activation is from the future')
        if elapsed >= rule['ttl_s']:
            continue
        try:
            condition = expression(rule['when'], obs)
        except (KeyError, IndexError):
            # A contact list or other variable-length observation may shrink
            # after rule activation. Treat the whole condition as inactive;
            # do not evaluate `not missing` as true or alter the hard guard.
            inactive_missing_paths.append(rule['id'])
            continue
        if not condition:
            continue
        for event_id in rule['defer']:
            chosen.pop(event_id, None)
            deferred[event_id] = rule['id']
        for i, offer in enumerate(rule['offer']):
            template = catalog[offer['event_id']]
            a = template.action
            offset = tuple(float(x) + y for x, y in zip(a.target_offset_m, offer['offset_delta_m']))
            if any(abs(x) > .20 for x in offset):
                raise ValueError('Combined offset exceeds executor envelope')
            action = replace(a, target_offset_m=offset, hand=offer['hand'] or a.hand)
            event_id = template.event_id if action == a else f'{template.event_id}:rule:{rule["id"]}:{i}'
            event = replace(template, event_id=event_id, action=action)
            chosen[event_id] = (offer['priority'], event)
            effect_owners[event_id] = rule['id']
    accepted, rejected = [], []
    for priority, event in sorted(chosen.values(), key=lambda pair: (-pair[0], pair[1].event_id)):
        allowed, reasons = immutable_guard(event, obs)
        if allowed:
            accepted.append(event)
        else:
            rejected.append({'event_id': event.event_id, 'immutable_reasons': reasons})
    return accepted, {'offered_event_rule_owners': effect_owners,
                      'deferred_event_rule_owners': deferred,
                      'immutable_rejections': rejected,
                      'inactive_missing_observation_path_rules':
                          inactive_missing_paths,
                      'dead_end': not accepted}


class RevisionMonitor:
    """Episode-local feedback; rollback changes future policy, never physics."""
    def __init__(self, patch, activated_time_s):
        self.patch = patch
        self.activated_time_s = activated_time_s
        self.exposures = {}
        self.outcomes = {}

    def selected(self, event_id, application, obs):
        rule_id = application['offered_event_rule_owners'].get(event_id)
        if rule_id is None or rule_id in self.exposures:
            return None
        rule = next(r for r in self.patch['rules'] if r['id'] == rule_id)
        expected_met(rule['expected'], obs, obs)
        before = json.loads(json.dumps(obs, default=lambda x: x.tolist()))
        self.exposures[rule_id] = {'before': before, 'time_s': obs['time_s'], 'event_id': event_id}
        return {'kind': 'actual_rule_exposure', 'rule_id': rule_id, 'event_id': event_id,
                'time_s': obs['time_s'], 'before_sha256': digest(before)}

    def observe(self, obs, irreversible_hard_failure=False, protected_relation_lost=False):
        results = []
        for rule in self.patch['rules']:
            exposure = self.exposures.get(rule['id'])
            if irreversible_hard_failure or protected_relation_lost:
                status = 'rollback_for_measured_regression'
            elif rule['id'] in self.outcomes:
                status = self.outcomes[rule['id']]
            elif obs['time_s'] - self.activated_time_s >= rule['ttl_s']:
                status = 'expired_without_verified_effect'
            elif exposure is not None:
                elapsed = obs['time_s'] - exposure['time_s']
                if elapsed < 0:
                    raise ValueError('Simulation time moved backwards')
                if elapsed <= rule['observe_s'] and expected_met(rule['expected'], exposure['before'], obs):
                    status = 'observed_expected_change'
                elif elapsed >= rule['observe_s']:
                    status = 'rollback_for_missing_expected_change'
                else:
                    status = 'awaiting_measured_effect'
            else:
                status = ('expired_without_exposure' if obs['time_s'] - self.activated_time_s >= rule['ttl_s']
                          else 'not_exposed')
            if status not in ('awaiting_measured_effect', 'not_exposed'):
                self.outcomes[rule['id']] = status
            results.append({'rule_id': rule['id'], 'status': status, 'time_s': obs['time_s'],
                            'native_success_certified': False})
        return results
