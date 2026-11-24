"""Bounded compatibility interpreter for inherited v22 solver operators.

Only solver fields already exposed by the RSSR retarget tool can be changed.
Expressions are parsed, never evaluated as Python. Prior trajectory evidence
does not certify the new controller or object task.
"""
import ast
import copy
import math

DEFAULTS={'upstream_weight':0.,'ik_nullspace_gain':0.,'ik_nullspace_target':'none',
          'ik_damping':.05,'ik_warm_start':'prev','ik_joint_limit_margin':0.,
          'ik_pos_weight':1.,'ik_rot_weight':1.}
SHARED=set(DEFAULTS)-{'ik_pos_weight','ik_rot_weight'}
CHAIN={'ik_pos_weight','ik_rot_weight'}


def expression(text,c,p):
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (str,int,float,bool):return n.value
        if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in ('c','p'):
            key=visit(n.slice)
            if not isinstance(key,str):raise ValueError('Operator lookup must use a named scalar')
            return (c if n.value.id=='c' else p)[key]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            v=visit(n.operand);return v if isinstance(n.op,ast.UAdd) else -v
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            return a/b
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and not n.keywords:
            args=[visit(a) for a in n.args]
            if n.func.id=='abs' and len(args)==1:return abs(args[0])
            if n.func.id=='clip' and len(args)==3:return max(args[1],min(args[2],args[0]))
        if isinstance(n,ast.BoolOp):
            values=[bool(visit(v)) for v in n.values]
            if isinstance(n.op,ast.And):return all(values)
            if isinstance(n.op,ast.Or):return any(values)
        if isinstance(n,ast.Compare):
            values=[visit(n.left),*[visit(v) for v in n.comparators]]
            for a,op,b in zip(values,n.ops,values[1:]):
                if isinstance(op,ast.Eq):ok=a==b
                elif isinstance(op,ast.NotEq):ok=a!=b
                elif isinstance(op,ast.Gt):ok=a>b
                elif isinstance(op,ast.GtE):ok=a>=b
                elif isinstance(op,ast.Lt):ok=a<b
                elif isinstance(op,ast.LtE):ok=a<=b
                else:raise ValueError('Unsupported operator comparison')
                if not ok:return False
            return True
        raise ValueError('Unsupported inherited operator expression')
    if not isinstance(text,str) or len(text)>1000:raise ValueError('Invalid inherited expression')
    try:return visit(ast.parse(text,mode='eval').body)
    except (SyntaxError,ZeroDivisionError,OverflowError,TypeError) as error:
        raise ValueError('Invalid inherited solver expression') from error


def compact_definition(raw):
    params={}
    for name,spec in raw['parameters'].items():
        params[name]=({'enum':copy.deepcopy(spec['choices'])} if 'choices' in spec
                      else {'min':spec['min'],'max':spec['max']})
    return {k:copy.deepcopy(raw[k]) for k in ('id','scope','requires','updates')}|{'parameters':params}


def validate_call(call):
    if set(call)!={'definition','scope','parameters'}:raise ValueError('Invalid inherited solver call')
    op=call['definition'];args=call['parameters'];scope=call['scope']
    if set(op)!={'id','scope','requires','updates','parameters'} or op['scope'] not in ('shared','chain'):
        raise ValueError('Only shared/chain solver definitions can be inherited')
    allowed=SHARED if op['scope']=='shared' else CHAIN
    if not op['updates'] or set(op['updates'])-allowed:raise ValueError('Inherited operator changes an unavailable field')
    if scope not in (('shared','all') if op['scope']=='shared' else ('left','right','all')):
        raise ValueError('Inherited operator scope mismatch')
    if set(args)!=set(op['parameters']):raise ValueError('Inherited parameters must exactly match')
    for name,rule in op['parameters'].items():
        value=args[name]
        if set(rule)=={'enum'}:
            if value not in rule['enum']:raise ValueError('Inherited enum outside domain')
        elif set(rule)=={'min','max'}:
            if type(value) not in (int,float) or not math.isfinite(value) or not rule['min']<=value<=rule['max']:
                raise ValueError('Inherited numeric parameter outside domain')
        else:raise ValueError('Invalid inherited parameter domain')
    expression(op['requires'],DEFAULTS,args)
    for formula in op['updates'].values():expression(formula,DEFAULTS,args)
    return op,args,scope


def apply_calls(program,calls):
    if not isinstance(calls,list) or not 1<=len(calls)<=64:raise ValueError('One to 64 inherited solver calls required')
    out=copy.deepcopy(program)
    for call in calls:
        op,args,scope=validate_call(call)
        targets=['shared'] if op['scope']=='shared' else ['left','right'] if scope=='all' else [scope]
        for side in targets:
            effective=DEFAULTS|out.get('ik',{})
            if side!='shared':effective=effective|out['arms'][side]
            if not expression(op['requires'],effective,args):raise ValueError('Inherited operator precondition failed')
            values={k:expression(v,effective,args) for k,v in op['updates'].items()}
            if all(values[k]==effective[k] for k in values):raise ValueError('Inherited solver call has no effect')
            (out['ik'] if side=='shared' else out['arms'][side]).update(values)
    return out


def prior_probe(config,snapshot,payload):
    from pathlib import Path
    from common import sha
    from pilot_learning import validate_operator
    from pilot_ik import edit
    priors=snapshot.get('trajectory_priors',[])
    for prior in sorted(priors,key=lambda p:(-p['trajectory_success_count'],p['id'])):
        root=Path(config['model_root'])/'source/scene'
        required=prior['required_assets']
        if any(sha(root/name)!=identity for name,identity in required.items()):continue
        patch=copy.deepcopy(prior['patch'])
        try:
            candidate=edit(payload['best_program'],patch)
            operator=validate_operator({'name':'Inherited v22 trajectory recipe '+prior['id'],
                'mechanism':prior['title'][:1000],
                'preconditions':'Matching robot/task-frame assets. Kinematic prior only; object task and controller execution must be measured anew.',
                'tool':'retarget_patch','parameters':{},'template':patch},'retarget')
        except (ValueError,KeyError,TypeError):continue
        trace={'operator_id':operator['id'],'operator_definition':operator,'arguments':{},
               'knowledge_version':snapshot['version'],'case':config['case_id'],'mode':config.get('mode','train'),
               'diagnosis':'Initialize with inherited v22 solver experience; object outcome is unverified.',
               'hypothesis':'Reuse solver parameters, recompute the current trajectory, then check contact and task predicates.',
               'decision_source':'inherited_v22_trajectory_recipe','experience_id':prior['id'],
               'prior_evidence_level':'constrained_trajectory_success','object_task_transfer_verified':False}
        return patch,trace,candidate
    return None
