"""Read-only, query-specific outcomes of already executed recipe evidence.

Never writes a library entry, assumes missing preconditions, or executes a
recipe. A resolved predicate is a past conditional observation, not transfer.
"""
import copy
from collections import defaultdict
from pilot_learning import digest

POLICY = 'current_failure_resolution_from_executed_evidence_v1'


def outcome(recipe, experience, query, load_evidence):
    same_task = experience['context'].get('task_family') == query.get('task_family')
    hard = set(query.get('unresolved_hard_predicates', []))
    task = set(query.get('unresolved_task_predicates', [])) if same_task else set()
    target = {('hard', p) for p in hard} | {('task', p) for p in task}
    # Hard feasibility first; physical improvement is not task success.
    prioritized = {('hard', p) for p in hard} if hard else target
    resolved = defaultdict(set)
    full_cases, observed_cases, success_cases = set(), set(), set()
    missing = excluded = maximum = 0
    for identity in set(recipe.get('evidence_ids', [])):
        item = load_evidence(identity)
        if item is None:
            missing += 1
            continue
        if digest(item) != identity or item.get('stage') != 'retarget':
            raise ValueError('Retarget evidence identity/stage mismatch')
        proof = item.get('admission_evidence') or {}
        fields = ('hard_before', 'hard_after', 'task_before', 'task_after')
        if not all(isinstance(proof.get(k), list) for k in fields):
            missing += 1
            continue
        hb, ha, tb, ta = (set(proof[k]) for k in fields)
        if (not ha <= hb or not ta <= tb or proof.get('scene_unverified') or
                proof.get('experiment', {}).get('diagnostic_only') or
                (proof.get('evidence_policy') == 'interaction_v17' and
                 proof.get('hypothesis_verified') is not True)):
            excluded += 1
            continue
        case = item['case']; observed_cases.add(case)
        gained = {('hard', p) for p in hb - ha} | {('task', p) for p in tb - ta}
        current = gained & target
        for key in current:
            resolved[key].add(case)
        # Do not combine successes at different times into one full resolution.
        maximum = max(maximum, len(current & prioritized))
        if prioritized and prioritized <= gained:
            full_cases.add(case)
        if item.get('passed') is True and not ha and not ta:
            success_cases.add(case)
    return {
        'policy': POLICY,
        'scope': 'past conditional executions of this exact recipe; current transfer unverified',
        'observed_demonstrations': len(observed_cases),
        'missing_named_evidence_count': missing,
        'excluded_evidence_count': excluded,
        'current_failure_resolutions': [
            {'kind': k[0], 'predicate': k[1], 'supporting_demonstrations': len(v)}
            for k, v in sorted(resolved.items())],
        'priority': 'hard' if hard else 'task',
        'priority_failure_count': len(prioritized),
        'maximum_priority_failures_resolved_together': maximum,
        'all_priority_failures_resolved_demonstrations': len(full_cases),
        'selected_recipe_task_success_demonstrations': len(success_cases),
    }


def outcome_rank(value):
    return (value.get('maximum_priority_failures_resolved_together', 0),
            value.get('all_priority_failures_resolved_demonstrations', 0),
            value.get('selected_recipe_task_success_demonstrations', 0))


def prepare(experience, query, load_evidence, accept_recipe=None):
    """Select an existing recipe for this query without changing stored defaults."""
    result = copy.deepcopy(experience)
    if load_evidence is None or not query or not any(k in query for k in
            ('unresolved_hard_predicates', 'unresolved_task_predicates')):
        return result
    variants = result.get('recipe_variants', [])
    if not variants:
        return result
    choices = []
    for recipe in variants:
        # A past success cannot silently restore a recipe that subsequent
        # Agent evidence retired. Restoration belongs to the lifecycle writer.
        if recipe.get('retrieval_retired'):continue
        if accept_recipe is not None and not accept_recipe(dict(result,arguments=recipe['arguments'])):
            continue
        summary = outcome(recipe, result, query, load_evidence)
        score = outcome_rank(summary) + (len(recipe.get('successful_cases', [])),
                len(recipe.get('improved_cases', [])), -recipe['best_after'][-1])
        choices.append((score, recipe, summary))
    if not choices:return result
    _, selected, summary = max(choices, key=lambda row: row[0])
    result['arguments'] = copy.deepcopy(selected['arguments'])
    result['transfer_verified'] = len(selected.get('improved_cases', [])) >= 2
    result['selected_recipe_successful_cases'] = copy.deepcopy(selected.get('successful_cases', []))
    result['selected_recipe_improved_case_count'] = len(selected.get('improved_cases', []))
    result['retrieval_outcomes'] = summary
    return result
