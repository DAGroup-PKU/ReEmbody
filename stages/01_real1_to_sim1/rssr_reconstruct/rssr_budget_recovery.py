"""Resume sealed cases after an explicit change to API spending authorization.

The original method modules and request identities stay intact. Only the request
accounting class is replaced in the recovery interpreter. Completed outcomes,
learning entries, decisions and physical results are never fabricated or edited.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def validate_authorization(path,root,limit):
    if not path:raise ValueError('Explicit request-budget authorization required')
    path=Path(path).resolve();value=read(path)
    if (value.get('schema')!='rssr.request-budget.authorization.v1' or
        Path(value.get('run_root','')).resolve()!=Path(root).resolve() or
        value.get('source')!='user' or value.get('kind')!='k3' or
        'cumulative_limit' not in value or value['cumulative_limit']!=limit or
        not value.get('user_instruction')):
        raise ValueError('Request-budget authorization does not match this continuation')
    return {'path':str(path),'sha256':sha(path)}


def budget_error(result):
    return (result.get('status')=='execution_error' and result.get('error_type')=='RuntimeError'
            and result.get('error')=='k3_budget_exhausted')


def verify_plan(plan_path):
    plan=read(plan_path);ref=plan.get('request_budget_authorization')
    if not ref or sha(ref['path'])!=ref['sha256']:raise ValueError('Budget authorization missing or changed')
    validate_authorization(ref['path'],plan['run_root'],plan['request']['execution']['max_k3_requests'])
    revision=read(plan['revision_file'])
    if revision['request_id']!=digest(plan['request']):raise ValueError('Recovery plan identity changed')
    for path,identity in revision['files'].items():
        if sha(path)!=identity:raise ValueError('Recovery implementation or dependency changed: '+path)
    here=Path(__file__).resolve()
    if revision['files'].get(str(here))!=sha(here):raise ValueError('Recovery coordinator is not sealed')
    return plan


def owner_for(plan,row):
    root=Path(plan['run_root']).resolve();folder=root/'cases'/row['case_id']
    if folder.resolve().parent!=root/'cases':raise ValueError('Case path outside campaign')
    stored=read(folder/'request.json');identity=stored['campaign_request_id']
    expected={'schema':'rssr.case.request.v1','row':row,'mode':plan['request']['mode'],
              'campaign_request_id':identity}
    if stored!=expected:raise ValueError('Case row or mode changed')
    known=[(plan['parent_request'],root/'frozen.json')]
    for ancestor in plan.get('ancestor_phases',[]):
        if sha(ancestor['plan'])!=ancestor['plan_sha256']:raise ValueError('Ancestor plan changed')
        old=read(ancestor['plan']);known.append((old['request'],Path(old['revision_file'])))
    for request,revision_path in known:
        if digest(request)!=identity:continue
        revision=read(revision_path)
        if 'request_id' in revision and revision['request_id']!=identity:raise ValueError('Owner revision mismatch')
        for path,expected_sha in revision['files'].items():
            if sha(path)!=expected_sha:raise ValueError('Original frozen dependency changed: '+path)
        modules=[Path(p) for p in revision['files'] if Path(p).name=='rssr_campaign.py']
        if len(modules)!=1:raise ValueError('Original method implementation is ambiguous')
        return request,modules[0].parent,revision_path
    raise ValueError('Case is not owned by a sealed predecessor')


def protected_cache(folder):
    patterns=('**/operator_resolution.json','**/wire_response.json',
              '**/physics/report.json','**/physics/rollout.npz')
    return {str(p.relative_to(folder)):sha(p) for pattern in patterns for p in folder.glob(pattern)}


def recover_one(plan_path,row):
    """Fresh-process entry: keep old imports, overlay accounting, then resume."""
    plan=verify_plan(plan_path);root=Path(plan['run_root']).resolve();phase=Path(plan_path).parent
    request,code,revision=owner_for(plan,row);folder=root/'cases'/row['case_id']
    dest=phase/'budget_recovery'/row['case_id'];dest.mkdir(parents=True,exist_ok=True)
    result_path=folder/'result.json';start=dest/'start.json';finish=dest/'finish.json'
    if finish.exists():
        receipt=read(finish)
        if sha(result_path)!=receipt['result_sha256']:raise ValueError('Recovered result changed')
        return receipt
    # Load only the new accounting class under a separate module identity.
    spec=importlib.util.spec_from_file_location('_authorized_budget_accounting',Path(__file__).with_name('common.py'))
    accounting=importlib.util.module_from_spec(spec);spec.loader.exec_module(accounting)
    sys.path.insert(0,str(code))
    import common as original_common
    if Path(original_common.__file__).resolve()!=code/'common.py':raise ValueError('Original common module was not selected')
    limit=plan['request']['execution']['max_k3_requests']
    class AuthorizedBudget(accounting.Budget):
        def __init__(self,budget_root,limits):
            if Path(budget_root).resolve()!=root:raise ValueError('Recovery attempted a different request ledger')
            super().__init__(budget_root,dict(limits,k3=limit))
    original_common.Budget=AuthorizedBudget
    import rssr_campaign as campaign
    if Path(campaign.__file__).resolve()!=code/'rssr_campaign.py':raise ValueError('Original campaign module was not selected')
    with original_common.run_lock(folder):
        if not start.exists():
            previous=campaign.saved_case_result(request,row)
            if previous is None or not budget_error(previous):raise ValueError('Only a recorded budget interruption may start recovery')
            if limit is not None and read(root/'budget.json')['used'].get('k3',0)>=limit:
                raise ValueError('No newly authorized request capacity')
            archive=folder/'execution_attempts'/f"{previous.get('execution_attempt',1):02d}.json"
            write(start,{'schema':'rssr.budget.recovery.start.v1','at':time.time(),'case_id':row['case_id'],
                'owner_request_id':digest(request),'method_code_root':str(code),'owner_revision_sha256':sha(revision),
                'accounting_adapter_sha256':sha(Path(__file__).with_name('common.py')),
                'budget_authorization':plan['request_budget_authorization'],'effective_k3_limit':limit,
                'old_result_sha256':sha(result_path),'archive':str(archive),'request_sha256':sha(folder/'request.json'),
                'protected_cached_artifacts':protected_cache(folder),'case_identity_changed':False})
        baseline=read(start);archive=Path(baseline['archive'])
        if result_path.exists() and sha(result_path)==baseline['old_result_sha256']:
            archive.parent.mkdir(parents=True,exist_ok=True)
            if archive.exists():
                if sha(archive)!=baseline['old_result_sha256']:raise ValueError('Interruption archive collision')
            else:
                with archive.open('xb') as f:
                    f.write(result_path.read_bytes());f.flush();os.fsync(f.fileno())
            result_path.unlink()
        if not archive.exists() or sha(archive)!=baseline['old_result_sha256']:
            raise ValueError('Original interruption was not preserved')
    result=campaign.run_case(request,row)
    while result.get('error_type')=='DecisionUnavailable' and result.get('execution_attempt',1)<3:
        result=campaign.run_case(request,row)
    if sha(folder/'request.json')!=baseline['request_sha256']:raise RuntimeError('Recovery changed original case identity')
    for relative,identity in baseline['protected_cached_artifacts'].items():
        if sha(folder/relative)!=identity:raise RuntimeError('Recovery changed a cached decision or physical result: '+relative)
    receipt={'schema':'rssr.budget.recovery.finish.v1','at':time.time(),'case_id':row['case_id'],
        'owner_request_id':digest(request),'method_code_root':str(code),'status':result['status'],
        'sim2_task_success':result.get('sim2_task_success'),'execution_attempt':result.get('execution_attempt'),
        'result_sha256':sha(result_path),'start_receipt_sha256':sha(start),
        'original_case_identity_preserved':True,'original_cached_artifacts_unchanged':True,
        'effective_k3_limit':limit,'same_cumulative_ledger':True}
    write(finish,receipt);return receipt


def recover_budget_cases(plan_path,rows):
    plan=verify_plan(plan_path);root=Path(plan['run_root']);phase=Path(plan_path).parent
    jobs=[]
    for row in rows:
        folder=root/'cases'/row['case_id'];result=folder/'result.json';start=phase/'budget_recovery'/row['case_id']/'start.json'
        if (result.exists() and budget_error(read(result))) or start.exists():jobs.append(row)
    if not jobs:return []
    def invoke(row):
        dest=phase/'budget_recovery'/row['case_id'];dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.log').open('a') as log:
            subprocess.run([plan['request']['runtime']['python'],'-u',str(Path(__file__).resolve()),
                '--plan',str(plan_path),'--case',row['case_id']],stdout=log,stderr=subprocess.STDOUT,check=True)
        return read(dest/'finish.json')
    receipts=[]
    workers=plan['request']['execution']['active_cases']
    write(phase/'status.json',{'stage':'recovering_budget_interrupted','at':time.time(),'planned_recovery':len(jobs),'completed_recovery':0,'workers':workers})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(invoke,row):row for row in jobs}
        for future in concurrent.futures.as_completed(pending):
            receipts.append(future.result())
            write(phase/'status.json',{'stage':'recovering_budget_interrupted','at':time.time(),
                'planned_recovery':len(jobs),'completed_recovery':len(receipts),'workers':workers})
    write(phase/'budget_recovery/summary.json',{'cases':receipts,'completed':len(receipts),'reset_library':False})
    return receipts


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--case',required=True)
    args=parser.parse_args();plan=read(args.plan)
    rows=read(plan['request']['dataset_manifest'])['rows'];row=next(r for r in rows if r['case_id']==args.case)
    print(json.dumps(recover_one(args.plan,row),ensure_ascii=False))
