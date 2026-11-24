"""Measure hypotheses before admitting experience; no embedded repair recipes."""
import copy
import math
from rssr_interaction import admission, diagnosis

SCOPES = {
    'hard_count':{'action','scene','model'}, 'task_count':{'action','scene','model'},
    'joint_excess':{'action','scene','model'}, 'coupling_residual':{'model'},
    'ik_error':{'action'}, 'servo_error':{'action','model'},
    'relation_violation':{'action','scene'},
    'holding_gap':{'action','scene'}, 'holding_bilateral':{'action','scene'},
}
MINIMUM_GAIN={k:(1. if k.endswith('_count') else .0001) for k in SCOPES}
MINIMUM_GAIN.update(holding_gap=.01,holding_bilateral=.01)


def metric_family(key):
    return ('coupling_residual' if key.startswith('coupling/') else
            'joint_excess' if key.startswith('joint/') else
            'holding_gap' if key.startswith('holding/') and key.endswith('/gap_excess_s') else
            'holding_bilateral' if key.startswith('holding/') and key.endswith('/bilateral_deficit') else
            'relation_violation' if key.startswith('relation/') else key)


def model_health_metric(key):
    # Clearing a named hard constraint through a model-only experiment is
    # physical-model evidence too. The choice of a count versus a continuous
    # prediction must not turn the same verified repair into a negative.
    return metric_family(key) in ('joint_excess','coupling_residual','hard_count')


def evidence_scope(experiment):
    # Mechanism applicability is separate from whether this execution also
    # finished the task. A numerical repair does not become layout-specific
    # solely because it removes the last hard failure of one episode.
    return 'model_health' if model_health_metric(experiment['metric']) and experiment['edited_layer']==['model'] else 'interaction'


def metrics(report):
    objective=report.get('objective',{})
    errors=diagnosis(report)['closed_event_median_errors_m']
    relations={p['id']:float(p['measurements']['maximum_relation_violation_m']) for p in
             report.get('task_evaluation',{}).get('predicates',[]) if
             'maximum_relation_violation_m' in p.get('measurements',{}) and not p['kind'].startswith('initial_')}
    margins=list(relations.values())
    values={k:float(v) for k,v in {
        'hard_count':len(objective.get('hard_failures',[])),
        'task_count':len(objective.get('task_failures',[])),
        'joint_excess':report.get('max_joint_limit_violation_rad'),
        'coupling_residual':report.get('model_health',{}).get('max_coupling_residual_rad'),
        'ik_error':errors['ik'], 'servo_error':errors['servo'],
        'relation_violation':max(margins) if margins else None,
    }.items() if v is not None and math.isfinite(v)}
    for prefix,entries in [('coupling',report.get('model_health',{}).get('per_coupling_residual_rad',{})),
                          ('joint',report.get('joint_limit_audit',{}).get('per_joint_excess_native_units',{})),
                          ('relation',relations)]:
        values.update({prefix+'/'+key:float(value) for key,value in entries.items() if math.isfinite(value)})
    for p in report.get('task_evaluation',{}).get('predicates',[]):
        if p['kind']!='hold_source_interval':continue
        for name,field in [('gap_excess_s','hold_contact_gap_excess_s'),('bilateral_deficit','hold_bilateral_fraction_deficit')]:
            value=p.get('measurements',{}).get(field)
            if value is not None and math.isfinite(value):values['holding/'+p['id']+'/'+name]=float(value)
    return values


def retain_relation_progress(before,after,experiment):
    """Keep measured within-case geometry progress; never admit a task skill.

The task success gate and experience admission stay unchanged. Every existing
named geometric relation and established predicate must remain non-regressing.
"""
    return _retain_interaction_progress(before,after,experiment,'relation/')


def retain_holding_progress(before,after,experiment):
    return _retain_interaction_progress(before,after,experiment,'holding/')


def _retain_interaction_progress(before,after,experiment,prefix):
    if (not experiment['metric'].startswith(prefix) or
            not experiment.get('prediction_matched') or experiment.get('diagnostic_only')):
        return False
    bo,ao=before.get('objective',{}),after.get('objective',{})
    # A completed task already uses the success branch. Do not label it as
    # partial progress or later count it as evidence that this exception helped.
    if ao.get('passed') is True or after.get('task_success') is True:
        return False
    if (bo.get('hard_failures') or ao.get('hard_failures') or
            not set(ao.get('task_failures',[]))<=set(bo.get('task_failures',[]))):
        return False
    a,b=metrics(before),metrics(after)
    keys=[k for k in a if k.startswith(('relation/','holding/'))]
    return bool(keys and all(k in b and b[k]<=a[k]+.0001 for k in keys))


def layer(patch):
    layers=set()
    if patch.get('model_patch'):layers.add('model')
    if patch.get('scene_patch'):layers.add('scene')
    if set(patch)-{'experiment','model_patch','scene_patch','diagnostic_probe'}:layers.add('action')
    return layers


def validate_experiment(patch,before):
    spec=patch.get('experiment')
    if not isinstance(spec,dict) or set(spec)!={'metric','minimum_improvement'}:
        raise ValueError('experiment requires metric and minimum_improvement before execution')
    key=spec['metric'];minimum=spec['minimum_improvement']
    family=metric_family(key)
    if key not in metrics(before) or family not in SCOPES:raise ValueError('Hypothesis metric unmeasured')
    if type(minimum) not in (int,float) or not math.isfinite(minimum) or minimum<MINIMUM_GAIN[family]:
        raise ValueError('Prediction below declared measurable improvement floor: '+str(MINIMUM_GAIN[family]))
    if patch.get('diagnostic_probe'):
        if set(patch)!={'diagnostic_probe','experiment'}:raise ValueError('Isolation probe cannot combine edits')
    else:
        selected=layer(patch)
        if len(selected)!=1:raise ValueError('One causal layer per experiment: action OR scene OR model')
        if not selected<=SCOPES[family]:raise ValueError('Selected tool cannot directly affect the predicted metric')
    return copy.deepcopy(spec)


def validate_probe(value,used):
    from rssr_model_tools import PROBES
    if not isinstance(value,str):
        raise ValueError('diagnostic_probe must be a single JSON string, not an array/object; allowed values: '+', '.join(PROBES))
    if value not in PROBES:raise ValueError('Unknown diagnostic_probe string: '+value)
    if value in used:raise ValueError('This diagnostic probe was already executed; use its measured comparison in hypothesis_ledger')
    if len(used)>=2:raise ValueError('Both diagnostic probes were already measured; no unused probe remains')


def evidence(before,after,spec,patch):
    a,b=metrics(before),metrics(after);key=spec['metric']
    delta=a[key]-b[key] if key in b else None
    matched=delta is not None and delta>=spec['minimum_improvement']
    return {'metric':key,'before':a[key],'after':b.get(key),'decrease':delta,
            'minimum_improvement':spec['minimum_improvement'],'prediction_matched':bool(matched),
            'edited_layer':sorted(layer(patch)),
            'diagnostic_only':bool(patch.get('diagnostic_probe')),
            'interpretation':'measured conditional effect; not proof of a unique cause'}


def observed_execution_context(before,after,control_scene=None,observed_scene=None):
    """Keep actual control conditions; never infer minimal causal prerequisites."""
    context={'policy':'observed_execution_conditions_v2',
             'scope':'observed settings for this executed pair; not independently validated minimal prerequisites or a transferable complete solution'}
    if isinstance(before.get('model_profile'),dict) and isinstance(after.get('model_profile'),dict):
        context.update(control_model_profile=copy.deepcopy(before['model_profile']),
                       observed_result_model_profile=copy.deepcopy(after['model_profile']),
                       default_model_control_observed=not before['model_profile'])
    a=before['ik'].get('program') if isinstance(before.get('ik'),dict) else None
    b=after['ik'].get('program') if isinstance(after.get('ik'),dict) else None
    if isinstance(a,dict) and isinstance(b,dict):
        context.update(control_action_program=copy.deepcopy(a),
                       observed_result_action_program=copy.deepcopy(b))
    if isinstance(control_scene,dict) and isinstance(observed_scene,dict):
        from rssr_protocol import digest
        context.update(control_scene_sha256=digest(control_scene),
                       observed_result_scene_sha256=digest(observed_scene))
    return context if len(context)>2 else None


def proof(before,after,scene_changed,experiment,*,control_scene=None,observed_scene=None):
    result=admission(before,after,scene_changed)
    result.update(evidence_policy='interaction_v17',experiment=experiment)
    health=model_health_metric(experiment['metric']) and experiment['edited_layer']==['model']
    if health and result['evidence_level']=='numeric' and experiment['prediction_matched']:
        result['evidence_level']='model_health'
    result['hypothesis_verified']=bool(experiment['prediction_matched'] and not experiment['diagnostic_only'])
    execution=observed_execution_context(before,after,control_scene,observed_scene)
    if execution is not None:result['execution_context']=execution
    if health:
        a,b=metrics(before),metrics(after)
        result['global_model_health_nonregression']=all(k in a and k in b and b[k]<=a[k]+.0001 for k in ('coupling_residual','joint_excess'))
        result['hypothesis_verified'] &= result['global_model_health_nonregression']
    return result


def contract():
    return {'experiment':{'metric':list(SCOPES)+['coupling/<reported equality id>','joint/<reported joint name>','relation/<reported predicate id>','holding/<reported predicate id>/gap_excess_s','holding/<reported predicate id>/bilateral_deficit'],'minimum_improvement':'positive number in native metric units; predict a decrease before evaluation'},
            'scope_by_metric':{k:sorted(v) for k,v in SCOPES.items()},
            'minimum_measurable_gain':MINIMUM_GAIN,
            'rules':['Every candidate names one measured outcome and predicted improvement.',
                     'One edit layer per iteration; same source actions, goal predicates and evaluation budget.',
                     'Rejected/invalid/probe outcomes remain evidence, never task success.',
                     'An advertised hard-safe exploration candidate is only a case-local alternative control. The Agent may explicitly select it for at most two ordinary evaluations; unresolved incumbent regressions block final selection and positive experience. Default patches still use the incumbent.',
                     'Two failed predictions with the same edit structure require a different mechanism or an unused isolation probe.',
                     'Only matched hypotheses with non-regressing named hard/task predicates can create candidates.',
                     'A selected task success under an edited scene remains a pending observation until the existing final scene review passes for that exact executed scene. It then enters the same admission rules with bound execution and review evidence; failed or unknown reviews cannot supply positive experience.',
                     'Model-health evidence is labeled separately and cannot be called a task skill.',
                     'A model-only hard_count prediction can establish model-health evidence when a named hard failure is cleared; both measured coupling and joint residuals and named hard/task predicates must not regress. Missing health measurements cannot certify non-regression.',
                     'Measured named-relation or holding progress can retain a within-case branch without admitting a task skill; hard/task predicates and all measured relation/holding residuals must not regress.',
                     'At least two distinct demonstrations validate a family before activation; numeric parameter variants share the family.']}


def structure(patch):
    # Diagnostic/metric metadata does not create a new action mechanism.
    def walk(v):
        if isinstance(v,dict):return {k:walk(x) for k,x in sorted(v.items())}
        if isinstance(v,list):return [walk(x) for x in v]
        return '<numeric>' if type(v) in (int,float) else v
    return walk({k:v for k,v in patch.items() if k not in ('experiment','diagnostic_probe')})


def enforce_no_repeat(patch,ledger):
    if patch.get('diagnostic_probe'):return
    key=structure(patch)
    failures=[x for x in ledger if x.get('structure')==key and not x.get('diagnostic_only')]
    if len(failures)>=2 and all(not x.get('prediction_matched') for x in failures[-2:]):
        # A new discriminating probe since the last failure permits one retest.
        last=ledger.index(failures[-1])
        if not any(x.get('diagnostic_only') for x in ledger[last+1:]):
            raise ValueError('Two failed predictions for this mechanism: change mechanism or obtain new isolation evidence')
