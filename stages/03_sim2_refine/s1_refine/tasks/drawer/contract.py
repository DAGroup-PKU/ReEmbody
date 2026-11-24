"""Derive drawer bindings from unchanged native predicates, never cup rules."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
import math
import numpy as np

@dataclass(frozen=True)
class DrawerBindings:
    operation_id: str
    payload: str
    container: str
    joint: str
    handle: str
    open_fraction: float
    closed_fraction: float
    articulation_hold_s: float
    final_stability_s: float
    final_release_s: float

    def packet(self):
        from dataclasses import asdict
        return asdict(self)


def derive_bindings(contract, scene, choose_feature):
    operations = [p for p in contract['predicates'] if p['kind'] == 'open_insert_close']
    if len(operations) != 1:
        raise ValueError('Drawer refine requires exactly one native open_insert_close predicate')
    p = operations[0]
    objects = {o['id']: o for o in scene['objects']}
    payload, container = p['object'], p['container']
    if payload not in objects or container not in objects:
        raise ValueError('Native drawer bindings are absent from the scene')
    joint = objects[container].get('joint', {})
    if joint.get('name') != p['joint'] or objects[container].get('mobility') != 'slide':
        raise ValueError('Native task must bind the measured drawer slide')
    if not any(q['kind'] == 'inside' and q['object'] == payload and q['container'] == container
               for q in contract['predicates']):
        raise ValueError('Native payload containment predicate is required')
    stable = [q for q in contract['predicates'] if q['kind'] == 'stable']
    released = [q for q in contract['predicates'] if q['kind'] == 'released']
    if not stable or not released or not any(payload in q['objects'] for q in released):
        raise ValueError('Native final stability and payload release predicates are required')
    vals = [float(p[k]) for k in ('open_fraction', 'closed_fraction', 'hold_s')]
    if not all(math.isfinite(v) for v in vals) or not 0 <= vals[1] < vals[0] <= 1 or vals[2] <= 0:
        raise ValueError('Invalid native drawer articulation thresholds')
    return DrawerBindings(p['id'], payload, container, p['joint'],
        choose_feature(container, scene), *vals,
        max(float(q.get('window_s', 0)) for q in stable),
        max(float(q.get('window_s', 0)) for q in released))


def select_hands(bindings, contract, observation):
    """Choose distinct permitted hands once, using the actual initial distances."""
    hands = tuple(sorted(observation['finger_center_m']))
    positions = observation['object_position_m']
    def allowed(obj, hand):
        bound = {p['hand'] for p in contract['predicates']
                 if p['kind'] == 'lift_with_contact' and p['object'] == obj}
        return not bound or hand in bound
    choices = []
    for handle_hand, payload_hand in permutations(hands, 2):
        if allowed(bindings.handle, handle_hand) and allowed(bindings.payload, payload_hand):
            distances = [float(np.linalg.norm(np.asarray(observation['finger_center_m'][hand])
                               - np.asarray(positions[obj]))) for obj, hand in
                         ((bindings.handle, handle_hand), (bindings.payload, payload_hand))]
            if all(math.isfinite(v) for v in distances):
                choices.append((sum(distances), max(distances), handle_hand, payload_hand, distances))
    if not choices:
        raise ValueError('No distinct hand assignment satisfies the native bindings')
    _, _, handle_hand, payload_hand, distances = min(choices)
    return {'handle': handle_hand, 'payload': payload_hand,
            'initial_hand_target_distances_m': distances,
            'selection': 'minimum total initial distance among distinct contract-permitted hands'}
