"""Evidence-gated operator families; parameter recipes do not create abilities.

Deterministic bookkeeping only: no LLM calls or additional simulation rollouts.
Original executed definitions remain in evidence/aliases for exact recovery.
"""
import copy
from pilot_learning import validate_operator, instantiate, digest

POLICY = 'two_demonstrations_structural_families_v1'
ACTIVE_LIMIT = 24
CANDIDATE_CONTEXT_LIMIT = 2


def definition(op):
    return {k:copy.deepcopy(op[k]) for k in
            ('name','mechanism','preconditions','tool','parameters','template')}


def configure(state):
    if state.get('operator_policy'):
        if state['operator_policy'] != POLICY:
            raise ValueError('Unknown operator admission policy')
        return state
    state['operator_policy'] = POLICY
    state['operator_registry'] = {}
    state['operator_aliases'] = {}
    state['candidate_operators'] = []
    state['archived_operators'] = []
    state['observation_ids'] = []
    # Initial imports are explicitly pinned priors, not newly proven abilities.
    state['pinned_operator_ids'] = [o['id'] for o in state['operators']]
    if len(state['pinned_operator_ids']) > ACTIVE_LIMIT:
        raise ValueError('Initial operator library exceeds active limit')
    return state


def canonicalize(operator, arguments, stage):
    """Factor numeric values into arguments while retaining structural selectors.

This is structural equivalence, not a claim of physical/semantic equivalence.
Cycle/seed selectors, boolean switches, object/hand keys, enum constants, and
solver definition code remain distinct. Tool execution still enforces bounds.
"""
    patch = instantiate(operator, arguments)
    params, args = {}, {}
    def walk(source, value, path=()):
        if 'definition' in path:
            return copy.deepcopy(source)
        if isinstance(value, dict):
            return {k:walk(source[k],value[k],path+(k,)) for k in sorted(value)}
        if isinstance(value, list):
            return [walk(source[i],v,path+(i,)) for i,v in enumerate(value)]
        if type(value) in (int,float) and (not path or path[-1] not in ('cycle','seed_index')):
            name='p'+str(len(params))
            spec=operator['parameters'].get(source[1:]) if isinstance(source,str) and source.startswith('$') else None
            params[name]=copy.deepcopy(spec) if spec and set(spec)=={'min','max'} else {'min':value,'max':value}
            args[name]=value
            return '$'+name
        # Enum-valued placeholders retain their domain and repeated-use binding.
        if isinstance(source,str) and source.startswith('$'):
            name='p'+str(len(params));params[name]=copy.deepcopy(operator['parameters'][source[1:]])
            args[name]=value
            return '$'+name
        return copy.deepcopy(value)
    template=walk(operator['template'],patch)
    if len(params)>32:
        # Large composed programs stay exact recipes, rather than inventing a
        # broad high-dimensional parameterized capability.
        template=copy.deepcopy(operator['template']);params=copy.deepcopy(operator['parameters']);args=copy.deepcopy(arguments)
    body=definition(operator);body.update(parameters=params,template=template)
    result=validate_operator(body,stage)
    key=stage+'_family_'+digest({'tool':body['tool'],'template':template,
          'parameter_types':{k:'numeric' if 'min' in v else ['enum',v['enum']] for k,v in params.items()}})[:16]
    if instantiate(result,args)!=patch:
        raise RuntimeError('Operator factoring changed the executed patch')
    return key,result,args


def hard_safe(item):
    proof=item.get('admission_evidence') or {}
    if item['stage']=='camera' and proof.get('evidence_policy')=='named_camera_feedback_v1':
        return proof.get('established_check_nonregression') is True
    if item['stage']!='retarget':
        return True
    if 'hard_before' in proof and 'hard_after' in proof:
        return set(proof['hard_after']) <= set(proof['hard_before'])
    # Historical scores without named predicates certify non-regression only
    # when the resulting candidate has no hard failures at all.
    return item['after'][0] == 0


def all_definitions(state):
    values={o['id']:o for k in ('operators','candidate_operators','archived_operators') for o in state.get(k,[])}
    values.update(state.get('operator_aliases',{}))
    return values


def rank_family(family):
    return (len(family['successful_cases']),len(family['improved_cases']),-len(family['failed_cases']))


def refresh(state):
    lifecycle=state.get('lifecycle_registry',{})
    pinned_all={o['id']:o for key in ('operators','archived_operators') for o in state.get(key,[]) if o['id'] in state['pinned_operator_ids']}
    pinned=[o for oid,o in pinned_all.items() if not lifecycle.get(oid,{}).get('retired')]
    eligible=[f for f in state['operator_registry'].values() if len(f['improved_cases'])>=2 and not lifecycle.get(f['family_id'],{}).get('retired')]
    # Prefer established families on ties; deterministic ordering prevents
    # unrelated timing changes from gratuitously replacing equal candidates.
    eligible.sort(key=lambda f:(rank_family(f),f.get('status')=='active',f['family_id']),reverse=True)
    active={f['family_id'] for f in eligible[:max(0,ACTIVE_LIMIT-len(pinned))]}
    pools={'active':[],'candidate':[],'archived':[]}
    for family in state['operator_registry'].values():
        retired=lifecycle.get(family['family_id'],{}).get('retired')
        status='retired' if retired else 'active' if family['family_id'] in active else 'archived' if len(family['improved_cases'])>=2 else 'candidate'
        family['status']=status
        op=copy.deepcopy(family['operator'])
        op.update(family_id=family['family_id'],admission_state=status,
                  validated_demonstrations=len(family['improved_cases']),effect_validated=status=='active')
        pools['archived' if retired else status].append(op)
    pools['archived'] += [dict(o,admission_state='retired',effect_validated=False) for oid,o in pinned_all.items() if lifecycle.get(oid,{}).get('retired')]
    state['operators']=pinned+pools['active']
    state['candidate_operators']=pools['candidate']
    state['archived_operators']=pools['archived']
    for exp in state['experiences']:
        exp['admission_state']='retired' if exp.get('retrieval_retired') else 'active' if exp['operator_id'] in {o['id'] for o in state['operators']} else 'candidate'


def execution_example_rank(value):
    # Examples explain execution conditions, not the recipe's success rate.
    # Frequency alone would permanently crowd out richer new observations.
    # This does not change admission, evidence counts or claims of transfer.
    fields=('control_model_profile','observed_result_model_profile',
            'control_action_program','observed_result_action_program',
            'control_scene_sha256','observed_result_scene_sha256')
    known=sum(k in value for k in fields)
    return (-known,-len(value.get('cases',[])),-value.get('last_observation_index',0),value['context_id'])


def update_experience(state,item,identity,operator,arguments,family_id):
    context=item['context'];case=item['case']
    key=digest({'operator_family':family_id,'context':context})
    exp=next((e for e in state['experiences'] if e['id']==key),None)
    if exp is None:
        exp={'id':key,'stage':item['stage'],'operator_id':operator['id'],'family_id':family_id,
             'mechanism':operator['mechanism'],'preconditions':operator['preconditions'],
             'context':copy.deepcopy(context),'arguments':copy.deepcopy(arguments),'uses':0,
             'improved_cases':[],'successful_cases':[],'evidence_ids':[],'recipe_variants':[]}
        state['experiences'].append(exp)
    exp['operator_id']=operator['id'];exp['uses']+=1
    level=(item.get('admission_evidence') or {}).get('evidence_level')
    if level:exp['evidence_levels']=sorted(set(exp.get('evidence_levels',[])+[level]))
    exp['improved_cases']=sorted(set(exp['improved_cases']+[case]))
    if item['passed']:exp['successful_cases']=sorted(set(exp['successful_cases']+[case]))
    exp['evidence_ids'].append(identity)
    recipe=next((v for v in exp['recipe_variants'] if v['arguments']==arguments),None)
    if recipe is None:
        recipe={'arguments':copy.deepcopy(arguments),'improved_cases':[],'successful_cases':[],
                'best_after':item['after'],'evidence_ids':[]}
        exp['recipe_variants'].append(recipe)
    recipe['improved_cases']=sorted(set(recipe['improved_cases']+[case]))
    if item['passed']:recipe['successful_cases']=sorted(set(recipe['successful_cases']+[case]))
    recipe['best_after']=min(recipe['best_after'],item['after']);recipe['evidence_ids'].append(identity)
    execution=(item.get('admission_evidence') or {}).get('execution_context')
    if execution is not None:
        examples=recipe.setdefault('execution_examples',[])
        # Same settings under more demonstrations add support, not duplicate
        # capabilities or arbitrarily many prerequisite snapshots.
        key=digest(execution)
        sample=next((v for v in examples if v['context_id']==key),None)
        if sample is None:
            sample=dict(copy.deepcopy(execution),context_id=key,cases=[],evidence_ids=[])
            examples.append(sample)
        sample['cases']=sorted(set(sample['cases']+[case]));sample['evidence_ids'].append(identity)
        sample['last_observation_index']=len(recipe['evidence_ids'])
        examples.sort(key=execution_example_rank)
        del examples[3:]
    exp['recipe_variants'].sort(key=lambda v:(-len(v['successful_cases']),-len(v['improved_cases']),v['best_after'],digest(v['arguments'])))
    exp['recipe_variants']=exp['recipe_variants'][:3]
    best=exp['recipe_variants'][0];exp['arguments']=copy.deepcopy(best['arguments'])
    # Mechanism-level generality and identical-recipe transfer are different.
    exp['transfer_verified']=len(best['improved_cases'])>=2
    exp['selected_recipe_successful_cases']=best['successful_cases']


def observe(state,item,identity):
    """Called inside the bank's sole writer lock, after an actual evaluation."""
    configure(state)
    if item['context'].get('compiled_program_policy'):
        from rssr_success_recipe import POLICY as PROGRAM_POLICY
        if item['stage']!='retarget' or item['context']['compiled_program_policy']!=PROGRAM_POLICY:
            raise ValueError('Unknown composed-program experience policy')
    raw=item['trace']['operator_definition'];raw_id=raw['id'];case=item['case']
    beneficial=item['improved'] and hard_safe(item)
    proof=item.get('admission_evidence') or {}
    interaction=proof.get('evidence_policy') in ('interaction_v16','interaction_v17')
    learning=proof.get('evidence_policy')=='interaction_v17'
    if learning:
        # A measured model-health gain may be tied on the task rank. It is
        # stored in its own evidence scope, never counted as interaction gain.
        if proof.get('evidence_level')=='model_health':
            beneficial=bool(hard_safe(item) and proof.get('hypothesis_verified'))
        beneficial=bool(beneficial and proof.get('hypothesis_verified') and not proof.get('exploration_only') and
                        not proof.get('experiment',{}).get('diagnostic_only'))
    observed_family=raw_id if raw_id in state['pinned_operator_ids'] else canonicalize(raw,item['trace']['arguments'],item['stage'])[0]
    if item['context'].get('compiled_program_policy'):
        observed_family=item['stage']+'_program_'+digest(item['context'])[:16]
    if interaction:
        beneficial=bool(beneficial and proof.get('evidence_level')!='numeric' and
            not proof.get('scene_unverified') and set(proof['task_after'])<=set(proof['task_before']))
        key=digest({'family_id':observed_family,'context':item['context']})
        stats=state.setdefault('attempt_statistics',{}).setdefault(key,{
            'family_id':observed_family,'context':copy.deepcopy(item['context']),
            'attempt_count':0,'interaction_gain_count':0,'task_success_count':0,'numeric_only_count':0})
        stats['attempt_count']+=1;stats['interaction_gain_count']+=int(beneficial and proof.get('evidence_level')!='model_health')
        if learning:
            stats['model_health_gain_count']=stats.get('model_health_gain_count',0)+int(beneficial and proof.get('evidence_level')=='model_health')
            stats['prediction_matched_count']=stats.get('prediction_matched_count',0)+int(proof.get('hypothesis_verified',False))
        stats['task_success_count']+=int(item['passed']);stats['numeric_only_count']+=int(proof.get('evidence_level')=='numeric')
    if not beneficial:
        negative={'id':identity,'context':item['context'],'operator_id':raw_id,
                  'operator_definition':raw,'arguments':item['trace']['arguments'],
                  'before':item['before'],'after':item['after'],'case':case,
                  'admission_rejection':'no_verified_interaction_gain_or_regression' if interaction else 'no_improvement_or_hard_constraint_regression',
                  'evidence_level':proof.get('evidence_level'), 'scene_unverified':proof.get('scene_unverified',False)}
        if learning:
            negative.update(experiment=proof.get('experiment'),diagnosis=item['trace'].get('diagnosis'),hypothesis=item['trace'].get('hypothesis'))
            if proof.get('exploration_only'):
                negative.update(admission_rejection='unresolved_incumbent_regression_in_search_branch',search_branch=proof.get('search_branch'))
        state['negative_examples']=(state['negative_examples']+[negative])[-120:]
        family=state['operator_registry'].get(observed_family)
        if family is not None:
            family['failed_cases']=sorted(set(family['failed_cases']+[case]));refresh(state)
        return
    if raw_id in state['pinned_operator_ids']:
        operator=all_definitions(state)[raw_id]
        operator['effect_validated']=True
        update_experience(state,item,identity,operator,item['trace']['arguments'],raw_id)
        return
    family_id,operator,arguments=canonicalize(raw,item['trace']['arguments'],item['stage'])
    if item['context'].get('compiled_program_policy'):
        # Preserve one exact successful recipe per compatibility class; its
        # second demonstration is evaluated by the existing training probe.
        family_id=item['stage']+'_program_'+digest(item['context'])[:16]
        operator=validate_operator(definition(raw),item['stage']);arguments=item['trace']['arguments']
    family=state['operator_registry'].get(family_id)
    if family is None:
        family={'family_id':family_id,'operator':operator,'improved_cases':[],
                'successful_cases':[],'failed_cases':[],'evidence_ids':[],'status':'candidate'}
        state['operator_registry'][family_id]=family
    else:
        previous=family['operator']
        if item['context'].get('compiled_program_policy') and previous['id']!=operator['id']:
            return
        state['operator_aliases'][previous['id']]=copy.deepcopy(previous)
        body=definition(previous)
        # Grow a family's numeric domain only to include a newly measured
        # successful value, never an untested proposal's wider declared range.
        for key,value in arguments.items():
            spec=body['parameters'][key]
            if 'min' in spec:
                spec['min']=min(spec['min'],value);spec['max']=max(spec['max'],value)
        operator=validate_operator(body,item['stage']);family['operator']=operator
        for exp in state['experiences']:
            if exp.get('family_id')==family_id:exp['operator_id']=operator['id']
    state['operator_aliases'][raw_id]=copy.deepcopy(raw)
    family['improved_cases']=sorted(set(family['improved_cases']+[case]))
    if item['passed']:family['successful_cases']=sorted(set(family['successful_cases']+[case]))
    family['evidence_ids'].append(identity)
    update_experience(state,item,identity,operator,arguments,family_id)
    refresh(state)


def snapshot(state,mode,query,transient,load_evidence=None):
    from rssr_evidence_retrieval import prepare,outcome_rank,POLICY as RETRIEVAL_POLICY
    family=(query or {}).get('task_family');case=(query or {}).get('case_id')
    definitions=all_definitions(state)
    def unchanged_model_recipe(e):
        current=(query or {}).get('model_profile')
        if current is None:return False
        op=definitions.get(e['operator_id'])
        if op is None:return False
        patch=instantiate(op,e['arguments'])
        if not patch.get('model_patch') or set(patch)-{'model_patch','experiment'}:return False
        def subset(delta,base):
            if isinstance(delta,dict):
                return isinstance(base,dict) and all(k in base and subset(v,base[k]) for k,v in delta.items())
            return delta==base
        return subset(patch['model_patch'],current)
    def relevant(e):
        if e.get('retrieval_retired'):return False
        if unchanged_model_recipe(e):return False
        context=e['context']
        if (query or {}).get('model_contract_sha256') and context.get('model_contract_sha256') not in (None,query['model_contract_sha256']):return False
        return (family is None or context['task_family']==family or
            (context.get('evidence_scope')=='model_health' and context.get('model_contract_sha256') and
             context['model_contract_sha256']==(query or {}).get('model_contract_sha256')))
    active=copy.deepcopy(state['operators']);ids={o['id'] for o in active}
    prepared=[prepare(e,query,load_evidence,lambda v:not unchanged_model_recipe(v))
              for e in state['experiences']]
    experiences=[e for e in prepared if e['operator_id'] in ids and relevant(e)]
    phase=(query or {}).get('repair_phase')
    health_needed=(query or {}).get('model_health_unresolved')
    def rank(e):
        recipe=next((v for v in e.get('recipe_variants',[]) if v['arguments']==e['arguments']),None)
        e['execution_examples']=copy.deepcopy(recipe.get('execution_examples',[])[:2]) if recipe else []
        for example in e['execution_examples']:
            # Query-only comparison; missing old conditions remain unknown.
            example['current_control_comparison']={}
            for label,saved,current in (
                    ('model','control_model_profile','model_profile'),
                    ('action','control_action_program','action_program'),
                    ('scene','control_scene_sha256','scene_sha256')):
                example['current_control_comparison'][label]=(
                    'unknown' if saved not in example or current not in (query or {}) else
                    'equal' if example[saved]==query[current] else 'different')
        key=digest({'family_id':e.get('family_id',e['operator_id']),'context':e['context']})
        stats=state.get('attempt_statistics',{}).get(key,{})
        e.update({k:stats[k] for k in ('attempt_count','interaction_gain_count','task_success_count','numeric_only_count','model_health_gain_count','prediction_matched_count') if k in stats})
        attempts=stats.get('attempt_count',0)
        health=e['context'].get('evidence_scope')=='model_health'
        gain_key='model_health_gain_count' if health else 'interaction_gain_count'
        rate=stats.get(gain_key,0)/attempts if attempts else 0.
        # Numerical model recipes are not tied to grasp/placement phases.
        scope_match=health==health_needed if type(health_needed) is bool else False
        return (scope_match,*outcome_rank(e.get('retrieval_outcomes',{})),len(e['successful_cases']),
                not health and phase is not None and e['context'].get('repair_phase')==phase,
                rate,e['context'].get('task_family')==family,len(e['improved_cases']),e['uses'])
    experiences.sort(key=rank,reverse=True)
    candidates=[]
    if mode=='train':
        for op in state['candidate_operators']:
            rec=state['operator_registry'][op['family_id']]
            exp=[e for e in prepared if e['operator_id']==op['id'] and
                 relevant(e)]
            if exp and case not in rec['improved_cases']:
                candidates.append((len(rec['improved_cases']),op,exp))
        if load_evidence and any(e.get('retrieval_outcomes') for _,_,exps in candidates for e in exps):
            candidates.sort(key=lambda v:(max(rank(e) for e in v[2]),v[0],v[1]['id']),reverse=True)
        else:
            candidates.sort(key=lambda v:(-v[0],v[1]['id']))
    selected=candidates[:CANDIDATE_CONTEXT_LIMIT]
    active.extend(copy.deepcopy(v[1]) for v in selected)
    # Candidates carry visible probation status. They consume ordinary agent
    # decisions; no dedicated validation requests are scheduled.
    probation=[]
    for _,op,exp in selected:
        # Reserve a visible recipe for each advertised candidate. A full pool
        # of active experiences must not silently erase its transfer evidence.
        best=max(exp,key=rank)
        value=copy.deepcopy(best);value['admission_state']='candidate'
        probation.append(value)
    # One mechanism must not fill all eight slots with different task labels.
    # After family diversity, remaining slots can show useful context variants.
    diverse=[];deferred=[];seen=set()
    for e in experiences:
        key=e.get('family_id',e['operator_id'])
        (deferred if key in seen else diverse).append(e);seen.add(key)
    limit=8-len(probation)
    experiences=(diverse+deferred)[:limit]+probation
    known={o['id'] for o in active}
    from rssr_library_lifecycle import retired_ids,notes
    retired=retired_ids(state)
    active.extend(copy.deepcopy(o) for i,o in transient.items() if i not in known and i not in retired)
    result={'version':digest(state),'operators':active,'experiences':experiences,
            'negative_examples':[e for e in state['negative_examples'] if family is None or e['context']['task_family']==family][-8:],
            'mode':mode,'admission_policy':{'name':POLICY,'active_limit':ACTIVE_LIMIT,
                'minimum_distinct_demonstrations':2,'candidate_context_limit':CANDIDATE_CONTEXT_LIMIT,
                'retrieval_policy':RETRIEVAL_POLICY if load_evidence else 'mechanism_diversity_and_reserved_probation_recipes_v1',
                'candidate_is_verified':False}}
    from rssr_model_reuse import observed_options
    reuse=observed_options(experiences,query,load_evidence)
    if reuse:result['observed_model_reuse_options']=reuse
    if state.get('lifecycle_policy'):result['library_lifecycle']=notes(state)
    return result
