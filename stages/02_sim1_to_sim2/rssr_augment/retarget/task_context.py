"""Ground RuleAgent task understanding in the frozen contract and scene.

This is an evidence packet, not a task-specific rule table.  The model must
interpret the relations and choose soft rules; the native contract remains the
only success definition.  Missing executor skills are reported explicitly.
"""
from __future__ import annotations

from .scene_profile import scene_signature


def build_task_context(contract: dict, scene: dict, events: list,
                       source_task_text: str | None = None) -> dict:
    objects = {row['id']: row for row in scene['objects']}
    if len(objects) != len(scene['objects']):
        raise ValueError('Duplicate scene object identity')
    signature, binding = scene_signature(contract, scene)
    event_by_predicate: dict[str, list[str]] = {}
    for event in events:
        event_by_predicate.setdefault(event.source_predicate, []).append(event.event_id)
    nodes = []
    for predicate in contract['predicates']:
        names = {key: predicate[key] for key in
                 ('object', 'container', 'region', 'support', 'base', 'upper')
                 if key in predicate}
        names.update({f'objects_{index}': name for index, name in
                      enumerate(predicate.get('objects', []))})
        for name in names.values():
            if name not in objects:
                raise ValueError('Task predicate names an unknown object')
        nodes.append({'predicate_id': predicate['id'], 'kind': predicate['kind'],
                      'relations': names,
                      'role_bindings': {key: binding['object_bindings'][name]
                                        for key, name in names.items()},
                      'native_parameters': {key: value for key, value in predicate.items()
                                            if key not in ('id', 'kind', 'object', 'objects',
                                                           'container', 'region', 'support',
                                                           'base', 'upper')},
                      'available_event_ids': event_by_predicate.get(predicate['id'], [])})
    return {'schema': 'rssr.task_context.v1',
            'source_task_text': source_task_text,
            'source_text_is_success_authority': False,
            'scene_signature': signature,
            'scene_roles': {'object_bindings': binding['object_bindings'],
                            'joint_bindings': binding['joint_bindings']},
            'object_geometry': [{key: row.get(key) for key in
                                 ('id', 'shape', 'role', 'size_m', 'parent_id', 'joint',
                                  'mobility', 'support_id')}
                                for row in scene['objects']],
            'native_predicate_nodes': nodes,
            'available_physical_primitives': sorted({event.action.event for event in events
                                                      if event.action.event != 'reference'}),
            'event_coverage_is_success_proof': False,
            'rule_scope': 'soft candidate selection and bounded contact offsets only'}
