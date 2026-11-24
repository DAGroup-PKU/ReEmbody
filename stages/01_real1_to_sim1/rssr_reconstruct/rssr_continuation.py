"""Append raw training rows to one live campaign at a sealed phase boundary.

Keeps original cases, library, budget and frozen code intact. New cases receive
their own versioned request; old outcomes remain bound to their original request.
"""
import argparse
import collections
import concurrent.futures
import copy
import multiprocessing
import os
import pathlib
import subprocess
import json
import time
import fcntl
from common import read,write,sha,run_lock
from rssr_protocol import digest
from rssr_dataset import validate_manifest
from rssr_campaign import preflight,verify_frozen,runtime_config,run_case,saved_case_result,summary
from rssr_knowledge import KnowledgeBank
from pilot_campaign import file_manifest


def process_alive(pid,start_ticks):
    try:
        data=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return data[0]!='Z' and data[19]==str(start_ticks)


def verify_revision(path,request):
    revision=read(path)
    if revision['request_id']!=digest(request):raise ValueError('Continuation request changed')
    if pathlib.Path(path).resolve().parent.parent.parent!=pathlib.Path(request['run_root']).resolve():
        raise ValueError('Continuation revision must be kept inside the same run')
    for file,identity in revision['files'].items():
        if sha(file)!=identity:raise RuntimeError('Continuation code/model/input changed: '+file)


def check_superset(parent_manifest,full_manifest):
    original=validate_manifest(parent_manifest);full=validate_manifest(full_manifest)
    if full_manifest['split']!='train' or len(full)!=1500:raise ValueError('Goal requires all 1500 training rows')
    lookup={r['case_id']:r for r in full}
    if collections.Counter(r['task_id'] for r in full)!=dict.fromkeys([f'di{i}' for i in range(1,8)]+[f'dm{i}' for i in range(1,9)],100):
        raise ValueError('Goal requires all 15 task families, 100 episodes each')
    if any(lookup.get(r['case_id'])!=r for r in original):raise ValueError('Continuation changed an existing raw episode')
    return original,full


def goal_gate(results,expected=1500):
    unique={r['case_id']:r for r in results}
    if len(unique)!=len(results):raise ValueError('Duplicate training denominator')
    successes=sum(r.get('sim2_task_success') is True for r in results)
    complete=len(unique)==expected and all(r['status']=='completed' for r in results)
    return {'training_rows_required':expected,'accounted_rows':len(unique),'sim2_successes':successes,
            'sim2_success_rate_all_1500':successes/expected,'all_training_complete':complete,
            'strictly_above_70_percent':bool(complete and successes/expected>.70),
            'inference_required':10,'camera_correctness_also_required_in_inference':True,
            'real2_allowed_now':False,'next_stage':'frozen_inference10' if complete and successes/expected>.70 else 'continue_training_or_rule_repair'}


def prepare(run_root,manifest_path,verification_path,request_cap=10480,phase_number=1,authorization_path=None,fixed_camera_config=None):
    root=pathlib.Path(run_root).resolve();verify_frozen(root)
    parent=read(root/'request.json');original,rows=check_superset(read(parent['dataset_manifest']),read(manifest_path))
    verification=read(verification_path);vrows=validate_manifest(verification)
    if len(vrows)!=10 or verification['split']=='train':raise ValueError('Exactly ten fixed inference rows required')
    if request_cap is not None and (type(request_cap) is not int or request_cap<=0):
        raise ValueError('Invalid request cap')
    authorization=None
    if request_cap is None or request_cap>10480:
        from rssr_budget_recovery import validate_authorization
        authorization=validate_authorization(authorization_path,root,request_cap)
    if type(phase_number) is not int or not 1<=phase_number<=9999:raise ValueError('Invalid continuation phase')
    ancestors=[]
    for number in range(1,phase_number):
        previous=root/'continuations'/f'{number:04d}'
        if not (previous/'launch.json').exists():raise ValueError('A preceding continuation was not launched')
        sealed=read(previous/'plan.json');verify_revision(sealed['revision_file'],sealed['request'])
        if sealed['run_root']!=str(root):raise ValueError('Predecessor belongs to another campaign')
        check_superset(read(sealed['request']['dataset_manifest']),read(manifest_path))
        ancestors.append({'request':sealed['request'],'plan':str(previous/'plan.json'),'plan_sha256':sha(previous/'plan.json')})
    phase=root/'continuations'/f'{phase_number:04d}';phase.mkdir(parents=True,exist_ok=True)
    predecessor=ancestors[-1]['request'] if ancestors else parent
    predecessor_dir=root/'continuations'/f'{phase_number-1:04d}' if ancestors else root
    request=copy.deepcopy(predecessor)
    request['dataset_manifest']=str(pathlib.Path(manifest_path).resolve())
    request['runtime'].update(library_lifecycle='evidence_revision_v1',continuation_revision_file=str(phase/'revision.json'))
    request['execution']['max_k3_requests']=request_cap
    if fixed_camera_config:
        request['runtime']['fixed_camera_config']=str(pathlib.Path(fixed_camera_config).resolve())
    preflight(request)
    plan={'schema':'rssr.continuation.plan.v1','run_root':str(root),'parent_request':parent,
          'parent_request_sha256':sha(root/'request.json'),'parent_launch':read(predecessor_dir/'launch.json'),
          'ancestor_phases':ancestors,'predecessor_request_id':digest(predecessor),
          'predecessor_response':str(predecessor_dir/'response.json'),
          'request':request,'revision_file':str(phase/'revision.json'),
          'manifest_sha256':sha(manifest_path),'verification_manifest':str(pathlib.Path(verification_path).resolve()),
          'verification_manifest_sha256':sha(verification_path),'existing_case_ids':[r['case_id'] for r in original],
          'planned':1500,'no_new_run_or_library_reset':True,'global_request_cap':request_cap}
    if authorization:plan['request_budget_authorization']=authorization
    path=phase/'plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Continuation plan already sealed')
    write(path,plan)
    # Seal this code while the predecessor continues unchanged.
    files=file_manifest(pathlib.Path(__file__).parent,runtime_config(request))
    files[str(pathlib.Path(manifest_path).resolve())]=sha(manifest_path)
    files[str(pathlib.Path(verification_path).resolve())]=sha(verification_path)
    if authorization:files[authorization['path']]=authorization['sha256']
    revision={'schema':'rssr.continuation.revision.v1','request_id':digest(request),'files':files,
              'parent_frozen_sha256':sha(root/'frozen.json'),'rule_change':'Explicitly authorized cumulative API accounting without a numeric ceiling; retain all per-case iteration, timeout, concurrency and retry protections. Resume budget-interrupted cases with their original frozen method implementation, cached decisions and physics, while archiving interruption results and applying only the new accounting adapter. No learning-policy, success-criterion, library-entry or task-parameter changes.',
              'all_prior_artifacts_preserved':True}
    if fixed_camera_config:
        revision['rule_change']=('Replace per-episode camera fitting with a sealed embodiment mounting configuration. '
            'One asset-qpos0 roll normalization preserves optical axis, position and FOV; no world leveling of moving wrists. '
            'Keep projection, raster occlusion, timing and Real1 visual checks, and verify fixed parent-relative FK. '
            'Do not require source/target wrists to share first-frame world orientation. Camera stage uses one candidate, '
            'no proposal API and no camera bank writes. Preserve physics task criteria, all libraries and prior results. '
            'Configuration is an unmeasured engineering prior, not hardware calibration.')
    revision_path=phase/'revision.json'
    if revision_path.exists() and read(revision_path)!=revision:raise ValueError('Continuation code changed after sealing')
    write(revision_path,revision);write(root/'training_goal_plan.json',{'plan':str(path),'planned':1500,'rate_gate':'>0.70','inference_rows':10,'continue_same_run':True})
    return path


def restore_cases(plan,rows):
    """Validate each outcome against the sealed request that actually ran it."""
    root=pathlib.Path(plan['run_root']);request=plan['request']
    known=[plan['parent_request']]+[a['request'] for a in plan.get('ancestor_phases',[])]+[request]
    registry={digest(r):r for r in known}
    for ancestor in plan.get('ancestor_phases',[]):
        if sha(ancestor['plan'])!=ancestor['plan_sha256']:raise ValueError('Ancestor plan changed')
        prior=read(ancestor['plan']);verify_revision(prior['revision_file'],prior['request'])
    results=[];queue=collections.deque();owners=set()
    for row in rows:
        path=root/'cases'/row['case_id']/'request.json'
        if path.exists():
            identity=read(path).get('campaign_request_id');owner=registry.get(identity)
            if owner is None:raise ValueError('Case request is not in the sealed campaign history: '+row['case_id'])
            old=saved_case_result(owner,row)
            if old is not None:
                if old.get('error_type')=='DecisionUnavailable' and old.get('execution_attempt',1)<3 and identity==digest(request):
                    queue.append(row)
                    continue
                results.append(old)
                if identity!=digest(request):owners.add(row['case_id'])
                continue
            if identity!=digest(request):raise RuntimeError('A prior-phase case is unfinished; preserve and explicitly resume its frozen request: '+row['case_id'])
        elif row['case_id'] in plan['existing_case_ids']:
            raise RuntimeError('An original case is missing')
        queue.append(row)
    return results,queue,owners


def recover_prior_decisions(plan,rows,phase):
    """Finish bounded transport recovery using each predecessor's own code.

    The supervisor holds the training lock and all predecessor workers ended.
    This is continuation of cached decisions, never a new experiment or reset.
    """
    root=pathlib.Path(plan['run_root']);current=digest(plan['request'])
    registry={digest(plan['parent_request']):(plan['parent_request'],root/'frozen.json')}
    for item in plan.get('ancestor_phases',[]):
        if sha(item['plan'])!=item['plan_sha256']:raise ValueError('Ancestor plan changed')
        prior=read(item['plan']);verify_revision(prior['revision_file'],prior['request'])
        registry[digest(prior['request'])]=(prior['request'],pathlib.Path(prior['revision_file']))
    receipts=[]
    for row in rows:
        folder=root/'cases'/row['case_id'];path=folder/'result.json'
        if not path.exists():continue
        identity=read(folder/'request.json')['campaign_request_id']
        if identity==current:continue
        if identity not in registry:raise ValueError('Unsealed prior case cannot be recovered')
        owner,revision_path=registry[identity]
        result=saved_case_result(owner,row)
        while result is not None and result.get('error_type')=='DecisionUnavailable' and result.get('execution_attempt',1)<3:
            cap=owner['execution']['max_k3_requests']
            if cap is not None and read(root/'budget.json')['used'].get('k3',0)>=cap:break
            revision=read(revision_path)
            for file,identity_hash in revision['files'].items():
                if sha(file)!=identity_hash:raise ValueError('Predecessor frozen dependency changed: '+file)
            modules=[pathlib.Path(p) for p in revision['files'] if pathlib.Path(p).name=='rssr_campaign.py']
            if len(modules)!=1:raise ValueError('Cannot uniquely locate predecessor implementation')
            code=modules[0].parent
            if sha(modules[0])!=revision['files'][str(modules[0])]:raise ValueError('Predecessor code changed')
            attempt=result.get('execution_attempt',1);dest=pathlib.Path(phase)/'prior_recovery'/row['case_id'];dest.mkdir(parents=True,exist_ok=True)
            # Fresh interpreter prevents importing the new revision for an old
            # request. run_case retains original inputs and archived attempts.
            script="import json,sys; b=json.load(sys.stdin); sys.path.insert(0,b['code']); from rssr_campaign import run_case; run_case(b['request'],b['row'])"
            with (dest/f'attempt_{attempt+1:02d}.log').open('a') as log:
                subprocess.run([owner['runtime']['python'],'-c',script],
                    input=json.dumps({'code':str(code),'request':owner,'row':row}),text=True,
                    stdout=log,stderr=subprocess.STDOUT,cwd=code,check=True)
            result=saved_case_result(owner,row)
            if result is None or result.get('execution_attempt',1)<=attempt:raise RuntimeError('Prior recovery made no recorded attempt')
            receipt={'case_id':row['case_id'],'request_id':identity,'code_root':str(code),
                     'execution_attempt':result['execution_attempt'],'status':result['status'],
                     'result_sha256':sha(path),'same_case_and_ledger':True}
            write(dest/f'attempt_{result["execution_attempt"]:02d}.json',receipt);receipts.append(receipt)
    return receipts


def execute(plan_path):
    plan=read(plan_path);root=pathlib.Path(plan['run_root']);phase=pathlib.Path(plan_path).parent
    request=plan['request'];prior=plan['parent_launch']
    with (phase/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Never restart or kill a process just because observation is slow.
        while process_alive(prior['pid'],prior['start_ticks']):
            write(phase/'status.json',{'stage':'waiting_for_existing_workers','observed_live_pid':prior['pid'],'at':time.time()})
            time.sleep(15)
        receipt=pathlib.Path(plan.get('predecessor_response',str(root/'response.json')))
        # Phase 0001 predates per-phase terminal receipts.
        if not receipt.exists() and plan.get('ancestor_phases'):receipt=root/'continuation_response.json'
        if not receipt.exists():raise RuntimeError('Predecessor stopped without terminal receipt; explicit resume required')
        terminal=read(receipt)
        if plan.get('predecessor_request_id') and terminal.get('request_id')!=plan['predecessor_request_id']:
            raise ValueError('Predecessor terminal receipt belongs to a different request')
        with run_lock(root):
            verify_frozen(root);verify_revision(plan['revision_file'],request)
            if sha(root/'request.json')!=plan['parent_request_sha256']:raise ValueError('Original campaign request changed')
            original,rows=check_superset(read(plan['parent_request']['dataset_manifest']),read(request['dataset_manifest']))
            if plan.get('request_budget_authorization'):
                from rssr_budget_recovery import recover_budget_cases
                recover_budget_cases(plan_path,rows)
            recover_prior_decisions(plan,rows,phase)
            if not (phase/'entry.json').exists():
                write(phase/'entry.json',{'at':time.time(),'parent_terminal':terminal,
                    'library_sha256':{s:sha(root/'learning'/(s+'.json')) for s in ('reconstruction','camera','retarget')},
                    'budget_used':read(root/'budget.json')['used'],'library_initialized_again':False})
            results,queue,owners=restore_cases(plan,rows)
            write(root/'status.json',{'stage':'continuing_training','pid':os.getpid(),'planned':1500,'at':time.time()})
            write(phase/'request.json',request)
            write(phase/'status.json',{'stage':'training','at':time.time(),'terminal':len(results),'planned':1500,'inflight':0})
            # An external rules review can request a clean boundary; already
            # running episodes finish with their immutable code and snapshots.
            def next_row():
                if (phase/'pause_after_inflight.json').exists():return None
                if (phase/'rule_review_needed.json').exists():return None
                new_results=[r for r in results if r['case_id'] not in owners]
                window=new_results[-30:]
                if len(window)==30 and sum(r.get('sim2_task_success') is True for r in window)<=3:
                    write(phase/'rule_review_needed.json',{'reason':'at_least_90_percent_failure_in_30_recent_cases',
                        'case_ids':[r['case_id'] for r in window],'at':time.time(),
                        'automatic_library_edits_are_not_human_experiences':True})
                    return None
                cap=request['execution']['max_k3_requests']
                if cap is not None and read(root/'budget.json')['used'].get('k3',0)>=cap:return None
                return queue.popleft() if queue else None
            with concurrent.futures.ProcessPoolExecutor(max_workers=request['execution']['active_cases'],
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                pending={}
                for _ in range(request['execution']['active_cases']):
                    row=next_row()
                    if row:pending[pool.submit(run_case,request,row)]=row
                while pending:
                    done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        row=pending.pop(future)
                        try:r=future.result()
                        except Exception as e:r={'case_id':row['case_id'],'status':'execution_error','error':str(e),'sim2_task_success':None,'rssr_to_sim2_passed':None}
                        if r.get('error_type')=='DecisionUnavailable' and r.get('execution_attempt',1)<3:
                            queue.appendleft(row)
                        else:results.append(r)
                        progress=summary(request,rows,results);progress['goal_gate']=goal_gate(results)
                        write(root/'continuation_progress.json',progress)
                        write(phase/'status.json',{'stage':'training','at':time.time(),'terminal':len(results),'planned':1500,'inflight':len(pending)})
                        row=next_row()
                        if row:pending[pool.submit(run_case,request,row)]=row
            verify_revision(plan['revision_file'],request);verify_frozen(root)
            result=summary(request,rows,results);gate=goal_gate(results);result['goal_gate']=gate
            if gate['all_training_complete']:
                bank=KnowledgeBank(root/'learning','train')
                coverage={'expected_case_ids':[r['case_id'] for r in rows],'processed_case_ids':[r['case_id'] for r in results],
                          'execution_error_case_ids':[],'unsupported_case_ids':[],'all_training_data_traversed':True,
                          'manifest_id':read(request['dataset_manifest'])['manifest_id'],'goal_gate':gate,
                          'continuation_revision_sha256':sha(plan['revision_file'])}
                result['checkpoint']=str(bank.export(root/'checkpoint_full1500',coverage))
            result['status']='ready_for_inference10' if gate['strictly_above_70_percent'] else 'training_complete_below_gate' if gate['all_training_complete'] else 'rule_review_needed' if (phase/'rule_review_needed.json').exists() else 'paused_after_inflight' if (phase/'pause_after_inflight.json').exists() else 'incomplete_budget_or_execution'
            write(phase/'response.json',result)
            write(root/'continuation_response.json',result);write(root/'goal_gate.json',gate)
            write(phase/'status.json',{'stage':result['status'],'at':time.time(),'terminal':len(results),'planned':1500})
            return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-root');p.add_argument('--manifest');p.add_argument('--verification');p.add_argument('--plan')
    a=p.parse_args()
    if a.prepare:print(prepare(a.run_root,a.manifest,a.verification))
    elif a.plan:execute(a.plan)
    else:p.error('Choose --prepare or --plan')


if __name__=='__main__':main()
