"""Run-local generated operators and evidence. Never import a historical bank."""
import contextlib, copy, fcntl, hashlib, json, math, pathlib, re, time
from common import read, write

STAGES = ('reconstruction', 'camera', 'retarget')
TOOLS = {'reconstruction': 'scene_patch', 'camera': 'camera_patch', 'retarget': 'retarget_patch'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def validate_operator(op, stage):
    if stage not in STAGES or not isinstance(op, dict): raise ValueError('Unknown operator stage')
    allowed = {'id','name','mechanism','preconditions','tool','parameters','template'}
    if set(op)-allowed or op.get('tool') != TOOLS[stage]: raise ValueError('Unsupported operator tool/field')
    for key in ('name','mechanism','preconditions'):
        if not isinstance(op.get(key), str) or not 1 <= len(op[key]) <= 1400: raise ValueError('Operator needs '+key)
    params = op.get('parameters')
    if not isinstance(params, dict) or len(params)>32: raise ValueError('Invalid operator parameters')
    for name, spec in params.items():
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', name) or not isinstance(spec,dict): raise ValueError('Invalid parameter')
        if set(spec)=={'enum'}:
            if not isinstance(spec['enum'],list) or not spec['enum'] or len(spec['enum'])>20 or any(type(v) not in (str,int,float) for v in spec['enum']): raise ValueError('Invalid enum')
        elif set(spec)=={'min','max'}:
            if any(type(spec[k]) not in (int,float) or not math.isfinite(spec[k]) for k in spec) or spec['min']>spec['max']: raise ValueError('Invalid numeric bounds')
        else: raise ValueError('Parameters use min/max or enum')
    if not isinstance(op.get('template'),dict) or not op['template']: raise ValueError('Nonempty JSON template required')
    def walk(v, depth=0):
        if depth>10: raise ValueError('Template too deep')
        if isinstance(v,dict):
            if any(not isinstance(k,str) or len(k)>100 for k in v): raise ValueError('Invalid template key')
            for x in v.values(): walk(x,depth+1)
        elif isinstance(v,list):
            if len(v)>100: raise ValueError('Template too large')
            for x in v: walk(x,depth+1)
        elif isinstance(v,str) and v.startswith('$'):
            if v[1:] not in params: raise ValueError('Unknown template parameter')
        elif type(v) not in (str,int,float,bool,type(None)): raise ValueError('JSON only')
        elif isinstance(v,float) and not math.isfinite(v): raise ValueError('Nonfinite template')
    walk(op['template'])
    if len(json.dumps(op))>10000: raise ValueError('Operator too large')
    result=copy.deepcopy(op);result['stage']=stage
    result['id']=stage+'_'+digest({'tool':op['tool'],'parameters':params,'template':op['template']})[:16]
    return result

def instantiate(op, args):
    if not isinstance(args,dict) or set(args)!=set(op['parameters']): raise ValueError('Arguments must exactly match operator parameters')
    for key,spec in op['parameters'].items():
        value=args[key]
        if 'enum' in spec:
            if value not in spec['enum']: raise ValueError('Unknown categorical argument '+key)
        elif type(value) not in (int,float) or not math.isfinite(value) or not spec['min']<=value<=spec['max']: raise ValueError('Argument outside declared range '+key)
    def fill(v):
        if isinstance(v,dict): return {k:fill(x) for k,x in v.items()}
        if isinstance(v,list): return [fill(x) for x in v]
        return args[v[1:]] if isinstance(v,str) and v.startswith('$') else v
    return fill(op['template'])

class Knowledge:
    def __init__(self, root):
        self.root=pathlib.Path(root);self.root.mkdir(parents=True,exist_ok=True)
    @contextlib.contextmanager
    def locked(self):
        with (self.root/'knowledge.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);yield
    def initialize(self):
        with self.locked():
            p=self.root/'knowledge.json'
            if not p.exists():
                write(p,{'schema':'rssr.empty_learning.v1','operators':[],'experiences':[],'evidence':[]})
                write(self.root/'initialization.json',{'at':time.time(),'operators':0,'experiences':0,'old_bank_imported':False,'old_operators_imported':False})
    def snapshot(self,stage,include_referenced=False):
        state=read(self.root/'knowledge.json')
        ops=[o for o in state['operators'] if o['stage']==stage]
        exp=[e for e in state['experiences'] if e['stage']==stage]
        exp.sort(key=lambda e:(len(e['improved_cases']),e['uses']),reverse=True)
        selected=ops[-24:]
        if include_referenced:
            needed={e['operator_id'] for e in exp[:8]};present={o['id'] for o in selected}
            selected += [o for o in ops if o['id'] in needed-present]
        return {'version':digest(state),'operators':selected,'experiences':exp[:8]}
    def resolve(self,stage,decision,case):
        if not isinstance(decision,dict) or not decision.get('diagnosis') or not decision.get('hypothesis'): raise ValueError('Measured diagnosis and hypothesis required')
        if decision.get('stop') is True: return None,{'stop':True,'diagnosis':decision['diagnosis']}
        with self.locked():
            p=self.root/'knowledge.json';state=read(p)
            if decision.get('operator_proposal'):
                op=validate_operator(decision['operator_proposal'],stage)
                known=next((o for o in state['operators'] if o['id']==op['id']),None)
                if known:op=known
                else:
                    op.update(created_case=case,created_at=time.time(),effect_validated=False)
                    state['operators'].append(op);write(p,state)
            else:
                op=next((o for o in state['operators'] if o['stage']==stage and o['id']==decision.get('operator_id')),None)
                if op is None:raise ValueError('Choose supplied operator or propose a new definition')
            args=decision.get('arguments',{});patch=instantiate(op,args)
        return patch,{'operator_id':op['id'],'arguments':args,'operator_definition':op,'diagnosis':decision['diagnosis'],'hypothesis':decision['hypothesis'],'knowledge_version':digest(state)}
    def record(self,stage,case,trace,before,after,context,passed=False):
        if not trace or not trace.get('operator_id'):return
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in [*before,*after]):raise ValueError('Finite measured scores required')
        item={'stage':stage,'case':case,'trace':trace,'before':list(before),'after':list(after),'context':context,'passed':bool(passed),'improved':tuple(after)<tuple(before)}
        item['id']=digest(item)
        with self.locked():
            p=self.root/'knowledge.json';state=read(p)
            if any(e['id']==item['id'] for e in state['evidence']):return
            state['evidence'].append(item)
            # Keep failures as evidence, never as verified reusable success.
            if item['improved']:
                op=next(o for o in state['operators'] if o['id']==trace['operator_id']);op['effect_validated']=True
                key=digest({'stage':stage,'operator':op['id'],'context':context})
                exp=next((e for e in state['experiences'] if e['id']==key),None)
                if exp is None:
                    exp={'id':key,'stage':stage,'operator_id':op['id'],'mechanism':op['mechanism'],'preconditions':op['preconditions'],'context':context,'arguments':trace['arguments'],'improved_cases':[],'successful_cases':[],'evidence_ids':[],'uses':0}
                    state['experiences'].append(exp)
                exp['improved_cases']=sorted(set(exp['improved_cases']+[case]));exp['uses']+=1;exp['evidence_ids'].append(item['id'])
                if passed:exp['successful_cases']=sorted(set(exp['successful_cases']+[case]))
                exp['transfer_verified']=len(exp['improved_cases'])>=2
            write(p,state);write(self.root/'versions'/(digest(state)+'.json'),state)
