"""Try at most one same-family learned recipe before another paid decision.

Selection is a fixed Agent policy, not an assertion that the recipe transfers.
The normal retarget evaluator must still execute and score this candidate.
"""
import copy
from pathlib import Path

from common import read, write
from pilot_ik import edit
from pilot_learning import validate_operator, instantiate
from rssr_knowledge import clean_operator
from rssr_protocol import digest


def first_probe(config, knowledge, folder, payload):
    folder = Path(folder)
    path = folder/'experience_probe.json'
    identity = digest({'case_id': config['case_id'], 'mode': config.get('mode', 'train'),
                       'task_family': config['task_family'], 'evidence': payload})
    if path.exists():
        receipt = read(path)
        body = dict(receipt)
        signature = body.pop('receipt_sha256')
        if digest(body) != signature or body['input_sha256'] != identity:
            raise RuntimeError('Experience probe evidence or receipt changed')
        # A concurrent update cannot change a previously selected or skipped
        # probe. Re-enter the ordinary executor using the original decision.
        return (receipt['patch'], receipt['trace']) if receipt['selected'] else None
    snapshot = knowledge.snapshot('retarget', include_referenced=True)
    operators = {o['id']: o for o in snapshot['operators']}
    body = {'schema': 'rssr.experience.probe.v1', 'input_sha256': identity,
            'knowledge_version': snapshot['version'], 'selected': False,
            'selection_policy': 'successful_composed_program_then_same_family_recipe_v2',
            'transfer_verified_here': False, 'api_requests': 0,
            'counts_against_normal_candidate_budget': True, 'skipped': []}
    experiences = sorted(snapshot['experiences'], key=lambda e:bool(e.get('context',{}).get('compiled_program_policy')), reverse=True)
    for experience in experiences:
        context = experience.get('context', {})
        if (context.get('task_family') != config['task_family'] or
                context.get('phase') != 'object_task' or not experience.get('improved_cases')):
            continue
        raw = operators.get(experience['operator_id'])
        training_success_candidate=(config.get('mode','train')=='train' and
            context.get('compiled_program_policy') and experience.get('successful_cases') and
            raw is not None and raw.get('admission_state')=='candidate')
        if raw is None or (not raw.get('effect_validated') and not training_success_candidate):
            continue
        try:
            if context.get('compiled_program_policy'):
                from rssr_success_recipe import compatibility, POLICY
                if context['compiled_program_policy'] != POLICY or not experience.get('successful_cases'):
                    raise ValueError('Unknown or unverified compiled-program policy')
                if context.get('compatibility') != compatibility(config,payload['scene']):
                    raise ValueError('Composed program does not match current robot/task/event structure')
            operator = validate_operator(clean_operator(raw), 'retarget')
            if operator['id'] != raw['id']:
                raise ValueError('Operator identity changed')
            arguments = copy.deepcopy(experience['arguments'])
            patch = instantiate(operator, arguments)
            candidate = edit(payload['best_program'], patch)
            if digest(candidate) == digest(edit(payload['best_program'], {})):
                raise ValueError('Recipe makes no change to the best program')
        except (ValueError, KeyError, TypeError) as error:
            body['skipped'].append({'experience_id': experience['id'], 'reason': str(error)})
            continue
        trace = {'operator_id': operator['id'], 'operator_definition': operator,
                 'arguments': arguments, 'knowledge_version': snapshot['version'],
                 'mode': config.get('mode', 'train'), 'case': config['case_id'],
                 'diagnosis': 'Same-family previously improved recipe; transfer is unverified.',
                 'hypothesis': 'Test this bounded recipe once using the unchanged physical predicates.',
                 'decision_source': 'fixed_agent_experience_probe',
                 'experience_id': experience['id'], 'experience_context':copy.deepcopy(context)}
        body.update(selected=True, patch=patch, trace=trace,
                    candidate_sha256=digest(candidate))
        break
    if not body['selected'] and snapshot.get('trajectory_priors'):
        from rssr_v22_bridge import prior_probe
        inherited=prior_probe(config,snapshot,payload)
        if inherited is not None:
            patch,trace,candidate=inherited
            body.update(selected=True,patch=patch,trace=trace,candidate_sha256=digest(candidate),
                        inherited_trajectory_only=True,object_task_transfer_verified=False)
    write(path, dict(body, receipt_sha256=digest(body)))
    return (body['patch'], body['trace']) if body['selected'] else None
