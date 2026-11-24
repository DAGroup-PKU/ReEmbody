"""One case-local exploration branch; never an accepted result or a skill.

The Agent may repair a real, hard-safe candidate that gained task predicates but
regressed others. Final selection and positive learning keep the incumbent's
original non-regression requirements. No additional rollout budget is granted.
"""
import copy
from object_target_agent import diagnostic
from rssr_learning_rules import metrics

POLICY='bounded_hard_safe_branch_v1'
MAX_BRANCH_EVALUATIONS=2


def differences(incumbent,candidate):
    old=incumbent['objective'];new=candidate['objective']
    return {'gained_task_predicates':sorted(set(old['task_failures'])-set(new['task_failures'])),
            'regressed_task_predicates':sorted(set(new['task_failures'])-set(old['task_failures']))}


def eligible(incumbent,candidate,experiment):
    delta=differences(incumbent,candidate)
    return bool(not incumbent['objective']['hard_failures'] and not candidate['objective']['hard_failures']
        and not candidate['objective']['passed'] and experiment.get('prediction_matched')
        and not experiment.get('diagnostic_only')
        and delta['gained_task_predicates'] and delta['regressed_task_predicates'])


def observation(incumbent,branch,used):
    if branch is None or used>=MAX_BRANCH_EVALUATIONS:return None
    report=diagnostic(branch['report'])
    report['learning_state']={'model_profile':copy.deepcopy(branch.get('model_profile',{})),
                              'measured_metrics':metrics(branch['report'])}
    return {'policy':POLICY,'trial':branch['trial'],'incumbent_trial':incumbent['trial'],
        'program':copy.deepcopy(branch['program']),'scene':copy.deepcopy(branch['scene']),
        'model_profile':copy.deepcopy(branch.get('model_profile',{})),
        'objective':copy.deepcopy(branch['objective']),'report':report,
        **differences(incumbent,branch),'remaining_branch_evaluations':MAX_BRANCH_EVALUATIONS-used,
        'accepted':False,'positive_transfer_evidence':False,
        'semantics':'Optional base_candidate=exploration in the decision envelope, outside the operator template. '
            'Patch one layer of this exact executed state and forecast its measured metrics. '
            'The full task reruns from the initial state; no rollout-state continuation or extra evaluation. '
            'It cannot replace the incumbent or supply positive experience until incumbent named constraints no longer regress.'}


def select(trace,incumbent,branch,used,patch):
    choice=(trace or {}).get('base_candidate','incumbent')
    if choice=='incumbent':return incumbent,False
    if choice!='exploration':raise ValueError('Unknown base_candidate decision')
    if branch is None:raise ValueError('No eligible exploration candidate in this decision')
    if used>=MAX_BRANCH_EVALUATIONS:raise ValueError('Case exploration branch budget exhausted')
    if patch.get('diagnostic_probe'):raise ValueError('Isolation probes require the accepted incumbent control')
    if not eligible(incumbent,branch,{'prediction_matched':True}):
        raise ValueError('Exploration candidate is no longer eligible against this incumbent')
    return branch,True


def bind_proof(proof,control,incumbent,current):
    result=copy.deepcopy(proof)
    debt=differences(incumbent,current)['regressed_task_predicates']
    hard=set(current['objective']['hard_failures'])-set(incumbent['objective']['hard_failures'])
    result['search_branch']={'policy':POLICY,'control_trial':control['trial'],
        'incumbent_trial':incumbent['trial'],'unresolved_incumbent_task_regressions':debt,
        'unresolved_incumbent_hard_regressions':sorted(hard)}
    result['exploration_only']=bool(debt or hard)
    return result
