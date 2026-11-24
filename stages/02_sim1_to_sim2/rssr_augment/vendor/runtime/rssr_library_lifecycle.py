"""Agent-owned reversible library maintenance driven only by executed evidence.

No hand-authored repair entries. Removal means removal from retrieval, with an
immutable audit trail; an in-flight old snapshot can still resolve its tool.
"""
import copy
from pilot_learning import digest
from rssr_operator_policy import canonicalize,hard_safe

POLICY='evidence_revision_v1'
WINDOW=12


def context_key(context):
    return digest({k:context.get(k) for k in ('task_family','phase','repair_phase','model_contract_sha256','evidence_scope')})


def distinct_recent(rows):
    # One episode cannot manufacture repeated transfer evidence by retrying.
    latest={}
    for r in rows:latest[r['case']]=r
    return list(latest.values())


def update(state,item,identity):
    """Called by the learner, under its writer lock, after observe()."""
    if state.get('lifecycle_policy') not in (None,POLICY):raise ValueError('Unknown library lifecycle')
    state['lifecycle_policy']=POLICY
    raw=item['trace']['operator_definition'];raw_id=raw['id']
    family_id,args=None,None
    if raw_id in state['pinned_operator_ids']:
        family_id=raw_id;args=item['trace']['arguments']
    else:
        family_id,_,args=canonicalize(raw,item['trace']['arguments'],item['stage'])
    if item['context'].get('compiled_program_policy'):
        family_id=item['stage']+'_program_'+digest(item['context'])[:16]
    # observe() is the only admission authority. Predictions alone don't count.
    positive=any(identity in e.get('evidence_ids',[]) for e in state['experiences'])
    proof=item.get('admission_evidence') or {}
    verified_observation=proof.get('evidence_level')!='numeric' and not proof.get('scene_unverified') and not proof.get('exploration_only')
    row={'evidence_id':identity,'case':item['case'],'context':context_key(item['context']),
         'arguments_id':digest(args),'positive':positive,'passed':bool(item['passed']),
         'hard_regression':not hard_safe(item),'verified_observation':verified_observation,
         'operator_id':raw_id}
    rec=state.setdefault('lifecycle_registry',{}).setdefault(family_id,{
        'recent':[],'retired':False,'recovery_cases':[],'operator_ids':[],
        'context_states':{},'recipe_states':{}})
    if any(x['evidence_id']==identity for x in rec['recent']):return []
    rec['recent']=(rec['recent']+[row])[-WINDOW:]
    family=state.get('operator_registry',{}).get(family_id,{})
    canonical_id=family.get('operator',{}).get('id',raw_id)
    rec['operator_ids']=sorted(set(rec['operator_ids']+[raw_id,canonical_id]))
    changes=[]

    def assess(target,rows,kind,key):
        recent=distinct_recent(rows)
        gains=sum(r['positive'] for r in recent)
        bad=sum(r['hard_regression'] for r in recent)
        # Numeric-only exploration and unreviewed layout changes are not proof
        # that a transferable mechanism is ineffective.
        failures=sum(not r['positive'] and r['verified_observation'] for r in recent)
        if target.get('retired'):
            if row['positive'] and (kind=='family' or key==row.get('context' if kind=='context' else 'arguments_id')):
                target['recovery_cases']=sorted(set(target.get('recovery_cases',[])+[row['case']]))
            if len(target.get('recovery_cases',[]))>=2:
                target.update(retired=False,recovery_cases=[])
                changes.append({'action':'restore','scope':kind,'key':key,'reason':'two_new_distinct_demonstrations_verified'})
        elif gains==0 and (failures>=4 or bad>=2):
            target.update(retired=True,recovery_cases=[],retired_at_evidence=identity,
                          reason='recent_distinct_failures' if failures>=4 else 'repeated_hard_regression')
            changes.append({'action':'retire','scope':kind,'key':key,'reason':target['reason'],
                            'supporting_evidence_ids':[r['evidence_id'] for r in recent]})

    # Clear stale recovery evidence after a new hard regression.
    if row['hard_regression']:rec['recovery_cases']=[]
    assess(rec,rec['recent'],'family',family_id)
    for field,kind,value in [('context_states','context',row['context']),('recipe_states','recipe',row['arguments_id'])]:
        target=rec[field].setdefault(value,{'retired':False,'recovery_cases':[]})
        if row['hard_regression']:target['recovery_cases']=[]
        assess(target,[r for r in rec['recent'] if r['context' if kind=='context' else 'arguments_id']==value],kind,value)

    # Revise a recalled recipe using only already executed alternatives.
    for exp in state['experiences']:
        if exp.get('family_id',exp['operator_id'])!=family_id:continue
        context=rec['context_states'].get(context_key(exp['context']),{})
        exp['retrieval_retired']=bool(rec['retired'] or context.get('retired'))
        variants=exp.get('recipe_variants',[])
        usable=[r for r in variants if not rec['recipe_states'].get(digest(r['arguments']),{}).get('retired')]
        for r in variants:r['retrieval_retired']=r not in usable
        if variants and not usable:exp['retrieval_retired']=True
        if usable:
            def rank(r):
                history=distinct_recent([x for x in rec['recent'] if x['arguments_id']==digest(r['arguments'])])
                rate=sum(x['positive'] for x in history)/len(history) if history else 0.
                return (rate,len(r['successful_cases']),len(r['improved_cases']))
            winner=max(usable,key=rank)
            if exp['arguments']!=winner['arguments']:
                changes.append({'action':'revise_recipe','scope':'experience','key':exp['id'],
                    'previous_arguments_sha256':digest(exp['arguments']),'next_arguments_sha256':digest(winner['arguments']),
                    'reason':'measured_alternative_with_better_recent_evidence'})
            exp['arguments']=copy.deepcopy(winner['arguments'])
            exp['transfer_verified']=len(winner['improved_cases'])>=2
            exp['selected_recipe_successful_cases']=winner['successful_cases']
    return changes


def retired_ids(state):
    return {oid for f in state.get('lifecycle_registry',{}).values() if f.get('retired') for oid in f['operator_ids']}


def notes(state):
    registry=state.get('lifecycle_registry',{})
    return {'policy':POLICY,'window':WINDOW,'changes_are_agent_evidence_driven':True,
        'retired_families':[{'family_id':key,'reason':v.get('reason'),'recovery_distinct_demonstrations':len(v.get('recovery_cases',[]))}
                            for key,v in registry.items() if v.get('retired')][-8:],
        'retirement_is_not_a_proof_of_universal_failure':True}
