"""Compile bounded action bindings from contract and scene structure.

This module contains no RSSR case ID, object name, success label, or replay
command. It only enumerates which existing robot primitives are permitted.
An unknown physical skill is a capability gap, never an invented command.
"""
from __future__ import annotations

import math

import numpy as np

REHANDABLE = {'approach', 'contact', 'grasp', 'lift', 'transport', 'align',
              'open_drawer', 'close_drawer', 'push_close', 'release'}


def explicit_object_hands(contract: dict, manipulators: tuple[str, ...]) -> dict[str, str]:
    """Any hand named by a frozen predicate is immutable for that object."""
    binding = {}
    for predicate in contract['predicates']:
        obj, hand = predicate.get('object'), predicate.get('hand')
        if hand is None:
            continue
        if not isinstance(obj, str) or hand not in manipulators:
            raise ValueError('Contract binds an unavailable manipulator')
        if obj in binding and binding[obj] != hand:
            raise ValueError('Conflicting frozen hand constraints for object')
        binding[obj] = hand
    return binding


def other_hand_allowed(action, contract: dict, manipulators: tuple[str, ...]) -> bool:
    obj = action.object_id
    return (action.event in REHANDABLE and isinstance(obj, str)
            and obj not in explicit_object_hands(contract, manipulators)
            and len(manipulators) > 1)


def articulated_feature(obj: str, objects: dict[str, dict]) -> bool:
    """A graspable child fixture rigidly attached to a moving scene object."""
    child = objects.get(obj, {})
    parent = objects.get(child.get('parent_id'), {})
    return bool(child.get('role') == 'fixture' and parent.get('joint')
                and child.get('shape') in {'box', 'cylinder', 'capsule'})


def articulation_axis_world(container: dict) -> np.ndarray:
    joint = container.get('joint')
    if not isinstance(joint, dict) or joint.get('type') != 'slide':
        raise ValueError('Capability gap: no supported slide articulation')
    axis = np.asarray(joint.get('axis'), dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < .99:
        raise ValueError('Capability gap: articulation axis is invalid')
    yaw = float(container.get('yaw_rad', 0.))
    if not math.isfinite(yaw):
        raise ValueError('Capability gap: articulation yaw is invalid')
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ axis
    return world / np.linalg.norm(world)


def choose_articulation_feature(container_id: str, scene: dict) -> str:
    """Choose the most bar-like graspable child of an articulated container.

    A scene with no supported child is rejected explicitly. This is a frozen
    geometric capability heuristic; it is not a case-specific repair rule.
    """
    objects = {row['id']: row for row in scene['objects']}
    container = objects[container_id]
    articulation_axis_world(container)
    candidates = [row for row in scene['objects']
                  if row.get('parent_id') == container_id
                  and articulated_feature(row['id'], objects)]
    if not candidates:
        raise ValueError('Capability gap: articulated object has no graspable child fixture')

    def score(row):
        size = np.asarray(row['size_m'], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.min(size) <= 0:
            raise ValueError('Capability gap: grasp feature geometry is invalid')
        elongation = float(np.max(size) / np.median(size))
        return (elongation, float(np.max(size)), row['id'])

    return max(candidates, key=score)['id']


def derive_pickup_objects(contract: dict) -> set[str]:
    """Objects whose frozen relational goal may require robot motion."""
    result = set()
    for predicate in contract['predicates']:
        if predicate['kind'] in {'inside_region', 'inside',
                                 'stable_supported_on', 'cup_nested'}:
            result.add(predicate['object'])
    return result
