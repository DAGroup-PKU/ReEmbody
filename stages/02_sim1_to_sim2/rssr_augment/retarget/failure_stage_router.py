"""Select one physical dependency for a bounded RuleAgent revision.

The router reads native predicate *kinds* and measured state. It has no task
names, object names, robot morphology assumptions, or executable rules.
The RuleAgent remains responsible for inventing and revising scene rules.
"""

from __future__ import annotations


def route_stage(contract: dict, observation: dict, trigger: dict) -> dict:
    live = observation.get('live_predicates', {})
    predicates = contract.get('predicates', [])
    unmet = [p for p in predicates if not live.get(p['id'], {}).get('passed', False)]
    reason = str(trigger.get('reason') or '')
    if not observation.get('immutable_hard', {}).get('all_hard_passed', True):
        stage, focus = 'safety', None
    elif reason in ('gripper_coupling_warning', 'non_target_contact',
                    'protected_predicate_lost'):
        stage, focus = 'safety', None
    else:
        ordered = []
        for predicate in unmet:
            kind = predicate['kind']
            measured = live.get(predicate['id'], {}).get('measurements', {})
            if kind == 'open_insert_close':
                if not measured.get('opened', False):
                    ordered.append((0, 'articulation', predicate['id']))
                elif not measured.get('inserted_while_open', False):
                    ordered.append((2, 'transfer', predicate['id']))
                else:
                    ordered.append((3, 'articulation', predicate['id']))
            elif kind == 'lift_with_contact':
                ordered.append((1, 'contact', predicate['id']))
            elif kind in ('inside', 'inside_region', 'cup_nested',
                          'stable_supported_on', 'base_placed_before_stack'):
                ordered.append((2, 'transfer', predicate['id']))
            elif kind in ('stable', 'released', 'object_contact'):
                ordered.append((4, 'settle', predicate['id']))
        if ordered:
            _, stage, focus = min(ordered, key=lambda item: item[0])
        else:
            stage, focus = ('verify', None) if not unmet else ('contact', unmet[0]['id'])
        if reason in ('contact_lost', 'target_contact_started',
                      'target_nonpad_contact',
                      'measured_gripper_closure_without_target_contact'):
            stage = 'contact'
    return {'schema': 'rssr.failure_stage_route.v1',
            'specialist': stage,
            'focus_predicate_id': focus,
            'unmet_predicate_ids': [p['id'] for p in unmet],
            'routing_basis': 'native_predicate_kinds_and_live_physics',
            'trigger_reason': reason}
