"""Expose one actually observed full model state for optional Agent replay.

Query-only derivation from admitted evidence. No bank writes, selected defaults,
new experience, simulation, or task-success inference are performed here.
"""
import copy
from pilot_learning import digest
from rssr_model_tools import edit_profile


def observed_options(experiences,query,load_evidence):
    q=query or {};wanted=set(q.get('unresolved_hard_predicates',[]))
    catalog=q.get('model_reuse_catalog');model=q.get('model_contract_sha256')
    # A full observed override is directly representable as model_patch only
    # from an empty current profile. Do not accidentally retain extra settings.
    if (q.get('model_profile')!={} or not wanted or not model or
        not catalog or load_evidence is None):return []
    groups={};seen=set()
    for exp in experiences:
        if (exp.get('admission_state')!='active' or exp.get('retrieval_retired') or
            exp['context'].get('evidence_scope')!='model_health' or
            exp['context'].get('model_contract_sha256')!=model):continue
        recipe=next((r for r in exp.get('recipe_variants',[])
                     if r['arguments']==exp['arguments'] and not r.get('retrieval_retired')),None)
        if recipe is None:continue
        for identity in set(recipe.get('evidence_ids',[])):
            if identity in seen:continue
            seen.add(identity);item=load_evidence(identity)
            if item is None:continue
            if digest(item)!=identity or item.get('stage')!='retarget':
                raise ValueError('Model-state evidence identity/stage mismatch')
            context=item.get('context',{});p=item.get('admission_evidence') or {}
            if context.get('model_contract_sha256')!=model or context.get('evidence_scope')!='model_health':continue
            if (p.get('evidence_policy')!='interaction_v17' or p.get('hypothesis_verified') is not True or
                p.get('scene_unverified') or p.get('experiment',{}).get('diagnostic_only')):continue
            if not all(isinstance(p.get(k),list) for k in ('hard_before','hard_after','task_before','task_after')):continue
            if not set(p['hard_after'])<=set(p['hard_before']) or not set(p['task_after'])<=set(p['task_before']):continue
            passed=wanted-set(p['hard_after'])
            if not passed:continue
            execution=p.get('execution_context') or {};profile=execution.get('observed_result_model_profile')
            if not isinstance(profile,dict) or not profile:continue
            try:
                if edit_profile({},profile,catalog)!=profile:continue
            except (ValueError,TypeError,KeyError):continue
            group=groups.setdefault(digest(profile),{'model_patch':copy.deepcopy(profile),'records':[]})
            action=execution.get('observed_result_action_program')
            comparison=('unknown' if not isinstance(action,dict) or 'action_program' not in q else
                        'equal' if action==q['action_program'] else 'different')
            scene=execution.get('observed_result_scene_sha256')
            scene_comparison=('unknown' if scene is None or 'scene_sha256' not in q else
                              'equal' if scene==q['scene_sha256'] else 'different')
            group['records'].append({'case':item['case'],'evidence_id':identity,
                'current_hard_failures_absent_together':sorted(passed),
                'current_vs_observed_action':comparison,
                'current_vs_observed_scene':scene_comparison,
                'observed_action_program':copy.deepcopy(action),
                'observed_scene_sha256':execution.get('observed_result_scene_sha256')})
    candidates=[]
    for key,group in groups.items():
        rows=group['records'];cases={r['case'] for r in rows}
        if len(cases)<2:continue
        all_clear={r['case'] for r in rows if set(r['current_hard_failures_absent_together'])==wanted}
        equal={r['case'] for r in rows if r['current_vs_observed_action']=='equal'}
        maximum=max(len(r['current_hard_failures_absent_together']) for r in rows)
        rank=(maximum,len(all_clear),len(equal),len(cases),key)
        representative=max(rows,key=lambda r:(len(r['current_hard_failures_absent_together']),
            r['current_vs_observed_action']=='equal',r['observed_action_program'] is not None,r['case']))
        candidate={'id':'observed_model_'+key[:16],'model_patch':group['model_patch'],
            'observed_demonstrations':len(cases),'all_current_hard_absent_demonstrations':len(all_clear),
            'maximum_current_hard_absent_together':maximum,
            'current_action_equal_demonstrations':len(equal),
            'conditional_execution_example':copy.deepcopy(representative),
            'scope':'Optional full observed model override, not a newly validated recipe. Available only from an empty current model profile. Counts describe conditional past executions, not transfer or a causal effect of this full patch. Agent must choose whether to propose it with its own measured experiment and run normal physics; action and scene are not copied.'}
        candidates.append((rank,candidate))
    return [max(candidates,key=lambda pair:pair[0])[1]] if candidates else []
