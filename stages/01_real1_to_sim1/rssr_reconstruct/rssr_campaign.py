"""JSON-only train/infer orchestration, frozen code and auditable bank lifecycle.

Physical backend coverage is checked before credentials or paid calls. Real2 is
not reachable from this module. A full-data campaign never filters unsupported
tasks down to a convenient subset.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import collections
import concurrent.futures
import contextlib
import copy
import json
import multiprocessing
import pathlib
import time
import traceback

from common import read, write, sha, load_credentials, run_lock
from rssr_protocol import parse, digest, Draft202012Validator
from rssr_dataset import validate_manifest
from rssr_knowledge import KnowledgeBank
from rssr_tasks import coverage
from rssr_errors import StageUnresolved
from pilot_learning import STAGES


def request_schema():
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object',
            'additionalProperties': False, 'required': ['schema','mode','run_root','dataset_manifest','runtime','knowledge','execution'],
            'properties': {
                'schema': {'const': 'rssr.campaign.request.v1'}, 'mode': {'enum': ['train','infer']},
                'run_root': {'type': 'string', 'minLength': 1}, 'dataset_manifest': {'type': 'string', 'minLength': 1},
                'runtime': {'type': 'object'},
                'knowledge': {'type': 'object', 'additionalProperties': False, 'required': ['initialization','imports'],
                              'properties': {'initialization': {'enum': ['empty','selective','frozen']},
                                             'imports': {'type': 'object','additionalProperties': False,
                                                         'properties': {s:{'type':'string','minLength':1} for s in STAGES}}}},
                'execution': {'type': 'object', 'additionalProperties': False,
                              'required': ['active_cases','render_sim_videos','max_k3_requests','run_real2'],
                              'properties': {'active_cases': {'type':'integer','minimum':1,'maximum':20},
                                             'render_sim_videos': {'type':'boolean'},
                                             'max_k3_requests': {'type':['integer','null'],'minimum':1,
                                                                 'description':'Cumulative request ceiling; null explicitly removes the ceiling while retaining accounting.'},
                                             'run_real2': {'const':False}}}}}


def validate_request(request):
    Draft202012Validator(request_schema()).validate(request)
    know = request['knowledge']
    if request['mode'] == 'infer' and (know['initialization'] != 'frozen' or set(know['imports']) != set(STAGES)):
        raise ValueError('Inference requires a frozen checkpoint for all three stages')
    if request['mode'] == 'train':
        if know['initialization'] not in ('empty','selective'):
            raise ValueError('Training uses empty or selective initialization')
        if (know['initialization'] == 'empty') != (not know['imports']):
            raise ValueError('Empty training cannot load any library; selective training must name its imports')
    runtime = request['runtime']
    forbidden = {'initialization_response_cache','resume_physics_from','resume_video_from','resume_semantics_from',
                 'retarget_interface','old_knowledge_allowed','old_results_allowed','import_case',
                 'run_root','batch_root','case_id','task','task_family','api_budget_root','stage_protocol',
                 'max_k3_requests','max_video_creates','run_real2','scene_backend',
                 'autonomous_recovery','k3_max_attempts_per_decision_total','api_identity'}
    if forbidden & set(runtime):
        raise ValueError('Runtime may not override identity/budget or import historical outputs: '+str(sorted(forbidden & set(runtime))))
    required = {'model_root','env_file','python','ffmpeg','ffprobe','seed','k3_concurrency','k3_timeout_s',
                'k3_attempts_per_decision','object_decision_max_tokens','max_scene_agent_rounds',
                'max_camera_agent_rounds','max_target_agent_rounds'}
    if not required <= set(runtime):
        raise ValueError('Missing runtime fields: '+str(sorted(required-set(runtime))))
    for field in required-{'model_root','env_file','python','ffmpeg','ffprobe'}:
        if type(runtime[field]) is not int or runtime[field] < (0 if field == 'seed' else 1):
            raise ValueError('Invalid integer runtime setting: '+field)
    if runtime['k3_concurrency'] > 20 or runtime['k3_attempts_per_decision'] > 3:
        raise ValueError('API concurrency/retry limit exceeded')
    if runtime.get('llm_api', 'chat_completions') not in ('chat_completions', 'responses'):
        raise ValueError('Unsupported LLM API transport')
    if 'retarget_experience_probe' in runtime and type(runtime['retarget_experience_probe']) is not bool:
        raise ValueError('retarget_experience_probe must be boolean')
    if 'interaction_revision' in runtime and runtime['interaction_revision']!='v16':
        raise ValueError('Unknown interaction repair revision')
    if 'learning_revision' in runtime:
        from rssr_operator_policy import POLICY
        if runtime['learning_revision']!='v17' or runtime.get('interaction_revision')!='v16' or runtime.get('operator_admission_policy')!=POLICY:
            raise ValueError('v17 learning requires v16 interactions and evidence-gated operator admission')
    if {'model_profile','_diagnostic_probe'} & set(runtime):raise ValueError('Model profiles must be learned by the Agent, not injected into a request')
    if runtime.get('library_lifecycle'):
        from rssr_library_lifecycle import POLICY
        if runtime['library_lifecycle']!=POLICY:raise ValueError('Unknown library lifecycle')
    if 'operator_admission_policy' in runtime:
        from rssr_operator_policy import POLICY
        if runtime['operator_admission_policy']!=POLICY:raise ValueError('Unknown operator admission policy')
    if 'k3_native_schema_stages' in runtime:
        from rssr_protocol import STAGES as JSON_STAGES
        stages=runtime['k3_native_schema_stages']
        if not isinstance(stages,list) or any(not isinstance(s,str) or s not in JSON_STAGES for s in stages) or len(stages)!=len(set(stages)):
            raise ValueError('Invalid native JSON schema stages')
    return request


def runtime_config(request):
    config=dict(request['runtime'], batch_root=request['run_root'], stage_protocol='rssr.stage.v1',
                max_k3_requests=request['execution']['max_k3_requests'], max_video_creates=0,
                run_real2=False, source_task_success_required=False, mode=request['mode'],scene_backend='rssr.scene.v2',
                autonomous_recovery=True,k3_max_attempts_per_decision_total=6)
    frozen=pathlib.Path(request['run_root'])/'frozen.json'
    if frozen.exists() and read(frozen).get('api_identity'):
        config['api_identity']=read(frozen)['api_identity']
    return config


def preflight(request):
    request = validate_request(request)
    if request['runtime'].get('fixed_camera_config'):
        from camera_mount_config import load as load_mount_config
        load_mount_config(request['runtime']['fixed_camera_config'])
    manifest = read(request['dataset_manifest'])
    rows = validate_manifest(manifest)
    if (request['mode'] == 'train') != (manifest['split'] == 'train'):
        raise ValueError('Dataset split and run mode disagree')
    backend = coverage(rows)
    return {'schema':'rssr.campaign.preflight.v1', 'request_id':digest(request),
            'mode':request['mode'], 'planned':len(rows), 'task_counts':dict(collections.Counter(r['task_id'] for r in rows)),
            'backend_coverage':backend, 'can_execute':backend['all_supported'],
            'cold_start':request['knowledge']['initialization']=='empty',
            'knowledge_updates':request['mode']=='train', 'real2_run':False, 'api_requests':0,
            'manifest_id':manifest['manifest_id'],
            'verification_overlap_policy':manifest.get('overlap_policy')}


def verify_frozen(root):
    frozen = read(pathlib.Path(root)/'frozen.json')
    for path, identity in frozen['files'].items():
        if sha(path) != identity:
            raise RuntimeError('Frozen code/model/input changed: '+path)


def verify_new_run_root(root):
    """Only a previously sealed run can resume generated data or a mutable bank.

    A newly named campaign may retain accounting/launcher receipts, but must
    never silently adopt prepopulated episode outputs or a prior mutable bank.
    This guards accidental contamination; it is not a security attestation.
    """
    root = pathlib.Path(root)
    if (root/'frozen.json').exists():
        return
    reserved = ('cases', 'learning', 'checkpoint', 'response.json', 'progress.json')
    present = [name for name in reserved if (root/name).exists()]
    if present:
        raise ValueError('New campaign contains prior generated artifacts; use a fresh run directory: '+','.join(present))


def stage_receipt(root, stage, data, assets=()):
    body = {'schema':'rssr.stage.result.v1','stage':stage,'data':data,
            'assets':[{'path':str(p),'sha256':sha(p)} for p in assets]}
    body['result_id'] = digest(body)
    write(pathlib.Path(root)/'stage_results'/(stage+'.json'), body)
    return body


def run_case(request, row):
    root=pathlib.Path(request['run_root'])/'cases'/row['case_id']
    root.mkdir(parents=True,exist_ok=True)
    # stdout is the machine-readable CLI response. Tool/model progress belongs
    # in the per-case log, not interspersed with the final JSON envelope.
    with (root/'execution.log').open('a',encoding='utf8',buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return _run_case(request,row)


def saved_case_result(request,row):
    root=pathlib.Path(request['run_root'])/'cases'/row['case_id']
    path=root/'result.json'
    if not path.exists():return None
    expected={'schema':'rssr.case.request.v1','row':row,'mode':request['mode'],
              'campaign_request_id':digest(request)}
    if not (root/'request.json').exists() or read(root/'request.json')!=expected:
        raise ValueError('Saved case result belongs to a different request: '+row['case_id'])
    result=read(path)
    if (result.get('schema')!='rssr.case.response.v1' or result.get('case_id')!=row['case_id'] or
        result.get('mode')!=request['mode'] or result.get('task_id')!=row['task_id']):
        raise ValueError('Saved case result identity mismatch: '+row['case_id'])
    return result


def restored_work(request,rows):
    """Collect completed work before consulting the remaining API budget.

    Resuming with an exhausted budget must not overwrite already completed
    coverage with an empty summary or bill for rereading those results.
    """
    results=[];pending=collections.deque()
    for row in rows:
        previous=saved_case_result(request,row)
        retryable=previous and previous.get('error_type')=='DecisionUnavailable' and previous.get('execution_attempt',1)<3
        if previous is not None and not retryable:results.append(previous)
        else:pending.append(row)
    return results,pending


def final_acceptance(reconstruction, target, camera):
    # Source registration is an intermediate proposal. Final camera QA checks
    # all six Sim1/Sim2 views on the final scene and executed trajectories.
    # An earlier failed mount must not veto its later verified repair.
    return bool(reconstruction['objective']['passed'] and target['passed'] and
                camera.get('passed') and camera.get('final') and camera.get('temporal_verification'))


def _run_case(request, row):
    root = pathlib.Path(request['run_root'])/'cases'/row['case_id']
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        result_path = root/'result.json'
        if result_path.exists():
            previous=saved_case_result(request,row)
            if previous.get('error_type')!='DecisionUnavailable' or previous.get('execution_attempt',1)>=3:
                return previous
            write(root/'execution_attempts'/f"{previous.get('execution_attempt',1):02d}.json",previous)
            result_path.unlink() # immutable archive above retains the failed result
        execution_attempt=1+len(list((root/'execution_attempts').glob('*.json')))
        config = dict(runtime_config(request), run_root=str(root), import_case=str(root/'inputs'),
                      case_id=row['case_id'], task=row['task'], task_family=row['task_family'])
        write(root/'request.json', {'schema':'rssr.case.request.v1','row':row,'mode':request['mode'],
                                  'campaign_request_id':digest(request)})
        write(root/'config.json', config)
        started = time.time()
        stages = {}
        bank = KnowledgeBank(pathlib.Path(request['run_root'])/'learning', request['mode'],
                             {'task_family':row['task_family'],'case_id':row['case_id']},
                             maintenance_policy=config.get('library_lifecycle'))
        initial_bank = {s:sha(bank.root/(s+'.json')) for s in STAGES}
        def status(stage):
            verify_frozen(request['run_root'])
            if config.get('continuation_revision_file'):
                from rssr_continuation import verify_revision
                verify_revision(config['continuation_revision_file'],request)
            write(root/'status.json', {'stage':stage,'at':time.time(),'pid':os.getpid()})
        try:
            status('preparing')
            # Hash every raw asset just before it is consumed; selection did not
            # judge source task success or use any prior retarget result.
            validate_manifest({'manifest_id':digest({'rows':[row]}),'rows':[row]}, verify_assets=True)
            load_credentials(config['env_file'])
            from pilot_prepare import prepare
            from pilot_scene import initialize, reconstruct
            from pilot_camera_workflow import run as cameras
            from pilot_physics import rollout
            from pilot_target import run as target_run
            prepare(config,row)
            status('scene.initialize')
            scene = initialize(config,root/'reconstruction',bank)
            stages['scene.initialize'] = stage_receipt(root,'scene.initialize',{'scene':scene})['result_id']
            status('camera.source')
            source_camera = cameras(config,bank,root,scene,final=False)
            stages['camera.source'] = stage_receipt(root,'camera.source',source_camera)['result_id']
            status('reconstruction')
            reconstruction = reconstruct(config,bank,root/'reconstruction/refinement',scene,source_camera['profiles']['sim1'])
            bank.record_scene_prior(config['case_id'],reconstruction,
                                    {'task_family':row['task_family'],'phase':'accepted_initial_scene'})
            scene = reconstruction['scene']
            stages['reconstruction'] = stage_receipt(root,'reconstruction',reconstruction)['result_id']
            status('retarget')
            target = target_run(config,bank,root/'sim2',scene)
            stages['retarget'] = stage_receipt(root,'retarget',target,[pathlib.Path(target['trajectory'])])['result_id']
            scene=target['scene']
            if config.get('learning_revision')=='v17':config=dict(config,model_profile=target.get('model_profile',{}))
            source_report={'task_success':None};source=None
            camera={'passed':False,'final':False,'temporal_verification':False,'skipped_reason':'sim2_task_failed'}
            # Final source replay and camera repair are useful only after Sim2
            # succeeds. They must consume exactly the selected final scene.
            if config.get('interaction_revision')!='v16' or target['passed']:
                from rssr_errors import CandidateInvalid
                from protocol_decision import DecisionOutputExhausted
                if target.get('scene_modified_by_retarget'):
                    status('scene.final_review')
                    from pilot_scene import review_fixed_scene
                    try:
                        reconstruction=review_fixed_scene(config,root/'reconstruction/final_review',scene,source_camera['profiles']['sim1'])
                    except DecisionOutputExhausted as error:
                        reconstruction={'scene':scene,'objective':{'passed':False},'verdict':{},'review_unresolved':str(error)}
                    stages['scene.final_review']=stage_receipt(root,'scene.final_review',reconstruction)['result_id']
                    if config.get('learning_revision')=='v17':
                        from rssr_deferred_scene_learning import release_reviewed_success
                        release_reviewed_success(config,bank,target,root/'stage_results/scene.final_review.json')
                status('source_replay')
                physical=root/'sim1/physics'
                try:
                    source_report=read(physical/'report.json') if (physical/'report.json').exists() else rollout(config,'sim1',physical,scene)
                    source={'trial':str(physical),'scene':scene,'report':source_report,'task_success_required':False}
                    write(root/'sim1/selection.json',source)
                except CandidateInvalid as error:
                    source_report={'task_success':None,'invalid_source_replay':str(error),'evidence':error.evidence}
                    write(root/'sim1/replay_failure.json',source_report)
                    camera.update(skipped_reason='source_replay_numerically_invalid')
                if source is not None:
                    status('camera.final')
                    camera=cameras(config,bank,root,scene,profiles=source_camera['profiles'],final=True,
                                   selections={'sim1':source,'sim2':target},scene_review=reconstruction['verdict'])
            stages['camera.final'] = stage_receipt(root,'camera.final',camera)['result_id']
            if request['execution']['render_sim_videos'] and source is not None and camera.get('final'):
                status('rendering_sim_only')
                from object_render import render
                for kind, selection in [('sim1',source),('sim2',target)]:
                    render(dict(config,camera_alignment={'enabled':True}),kind,selection['trial'],root/(kind+'.mp4'))
            success = final_acceptance(reconstruction,target,camera)
            result = {'status':'completed','reconstruction_passed':bool(reconstruction['objective']['passed']),
                      'source_camera_passed':bool(source_camera['passed']),
                      'source_camera_initialization_is_final_gate':False,
                      'acceptance_policy':'final_scene_physics_and_joint_temporal_camera_review_v2',
                      'sim1_task_success':source_report['task_success'],'sim1_success_required':False,
                      'sim2_task_success':bool(target['passed']),'camera_passed':bool(camera['passed']),
                      'final_camera_skipped_reason':camera.get('skipped_reason'),
                      'rssr_to_sim2_passed':success,'stage_results':stages}
        except StageUnresolved as error:
            result={'status':'completed','method_failure_stage':error.stage,'reason':str(error),
                    'reconstruction_passed':False,'source_camera_passed':None,'sim2_task_success':None,
                    'camera_passed':None,'rssr_to_sim2_passed':False,'stage_results':stages}
        except Exception as error:
            result = {'status':'execution_error','error_type':type(error).__name__,'error':str(error),
                      'traceback':traceback.format_exc(),'rssr_to_sim2_passed':None,'stage_results':stages}
        finally:
            if request['mode']=='infer' and initial_bank != {s:sha(bank.root/(s+'.json')) for s in STAGES}:
                raise RuntimeError('Inference mutated a frozen stage library')
        result.update(schema='rssr.case.response.v1',case_id=row['case_id'],task_id=row['task_id'],
                      mode=request['mode'],real2_run=False,wall_seconds=time.time()-started,
                      cold_start=request['knowledge']['initialization']=='empty',
                      library_updates=request['mode']=='train',execution_attempt=execution_attempt)
        write(result_path,result)
        write(root/'status.json',{'stage':result['status'],'at':time.time()})
        return result


def summary(request, rows, results):
    complete = [r for r in results if r['status']=='completed']
    errors = [r['case_id'] for r in results if r['status']!='completed']
    successes = sum(r.get('rssr_to_sim2_passed') is True for r in complete)
    return {'schema':'rssr.campaign.response.v1','request_id':digest(request),'mode':request['mode'],
            'planned':len(rows),'terminal':len(results),'completed':len(complete),'execution_errors':errors,
            'successes':successes,'success_rate_all_requested':successes/len(rows),
            'success_rate_completed':successes/len(complete) if complete else None,
            'rate_is_final':len(results)==len(rows),'real2_run':False,
            'cold_start':request['knowledge']['initialization']=='empty',
            'stage_successes':{s:sum(r.get(s) is True for r in complete) for s in
                               ('reconstruction_passed','source_camera_passed','sim2_task_success','camera_passed')},
            'cases':results}


def execute(request):
    report = preflight(request)
    if not report['can_execute']:
        raise ValueError('Full task coverage is incomplete; no case was started: '+','.join(report['backend_coverage']['missing']))
    root = pathlib.Path(request['run_root'])
    root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        verify_new_run_root(root)
        from rssr_api_identity import capture_api_identity
        model_identity=capture_api_identity(request)
        from pilot_campaign import file_manifest
        manifest = read(request['dataset_manifest'])
        rows = validate_manifest(manifest,verify_assets=True)
        code=pathlib.Path(__file__).parent
        files = file_manifest(code,runtime_config(request))
        files.update({str(p):sha(p) for p in (code/'schemas').rglob('*.json')})
        files[str(pathlib.Path(request['dataset_manifest']).resolve())] = sha(request['dataset_manifest'])
        for checkpoint in request['knowledge']['imports'].values():
            cp = pathlib.Path(checkpoint).resolve()
            files[str(cp)] = sha(cp)
            for entry in read(cp)['stages'].values():
                path = (cp.parent/entry['path']).resolve()
                files[str(path)] = sha(path)
        frozen = {'schema':'rssr.frozen.campaign.v1','request':request,'manifest_id':manifest['manifest_id'],
                  'files':files,'no_human_case_edits':True,'no_old_trajectory_or_camera_import':True,
                  'backend_coverage':report['backend_coverage'],'api_identity':model_identity}
        dest = root/'frozen.json'
        if dest.exists() and read(dest)!=frozen:
            raise ValueError('Frozen campaign changed; create another revision')
        write(dest,frozen)
        bank = KnowledgeBank.initialize(root/'learning',request['mode'],request['knowledge']['imports'],
                                       admission_policy=request['runtime'].get('operator_admission_policy'))
        initial_receipt = root/'initial_state.json'
        if not initial_receipt.exists():
            if any((root/'cases').glob('*')):
                raise ValueError('Cannot retroactively certify an initial state after cases have started')
            initial_states = {s:read(bank.root/(s+'.json')) for s in STAGES}
            counts = {s:{k:len(state[k]) for k in ('operators','experiences','negative_examples','training_cases','inherited_training_cases','events')}
                      for s,state in initial_states.items()}
            for s,state in initial_states.items():
                counts[s]['scene_priors']=len(state.get('scene_priors',[]))
                counts[s]['trajectory_priors']=len(state.get('trajectory_priors',[]))
                counts[s].update(evidence_count=state['evidence_count'],revision=state['revision'])
            if request['knowledge']['initialization']=='empty' and any(n for stage in counts.values() for n in stage.values()):
                raise ValueError('Cold-start receipt requires all three banks to be actually empty')
            write(initial_receipt, {'schema':'rssr.initial_state.v1','request_id':digest(request),
                                   'initialization':request['knowledge'],'counts':counts,
                                   'stage_sha256':{s:sha(bank.root/(s+'.json')) for s in STAGES},
                                   'generated_case_directories_at_start':len(list((root/'cases').glob('*'))),
                                   'frozen_campaign_sha256':sha(dest)})
        write(root/'request.json',request)
        write(root/'status.json',{'stage':'running','pid':os.getpid(),'at':time.time()})
        results,queue = restored_work(request,rows)
        # Submission is bounded: new work is scheduled when one case finishes.
        # API concurrency is independently limited by shared pilot_api slots.
        with concurrent.futures.ProcessPoolExecutor(max_workers=request['execution']['active_cases'],
                mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            def next_row():
                budget=root/'budget.json'
                cap=request['execution']['max_k3_requests']
                if cap is not None and budget.exists() and read(budget).get('used',{}).get('k3',0)>=cap:
                    return None
                return queue.popleft() if queue else None
            for _ in range(request['execution']['active_cases']):
                row = next_row()
                if row: pending[pool.submit(run_case,request,row)] = row
            while pending:
                done,_ = concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    try: result = future.result()
                    except Exception as error:
                        result = {'case_id':row['case_id'],'status':'execution_error','error':str(error),'rssr_to_sim2_passed':None}
                    if result.get('error_type')=='DecisionUnavailable' and result.get('execution_attempt',1)<3:
                        queue.append(row)
                    else:results.append(result)
                    write(root/'progress.json',summary(request,rows,results))
                    row = next_row()
                    if row: pending[pool.submit(run_case,request,row)] = row
        verify_frozen(root)
        final = summary(request,rows,results)
        if request['mode']=='train' and len(results)==len(rows) and not final['execution_errors']:
            coverage_report = {'expected_case_ids':[r['case_id'] for r in rows],
                               'processed_case_ids':[r['case_id'] for r in results],
                               'execution_error_case_ids':[], 'unsupported_case_ids':[],
                               'manifest_id':manifest['manifest_id'], 'all_training_data_traversed':True,
                               'case_successes':final['successes'],'frozen_code_sha256':sha(root/'frozen.json')}
            final['checkpoint'] = str(bank.export(root/'checkpoint',coverage_report))
        final['status'] = 'incomplete_budget' if len(results)!=len(rows) else ('complete' if not final['execution_errors'] else 'incomplete_execution_errors')
        write(root/'response.json',final)
        write(root/'status.json',{'stage':final['status'],'at':time.time()})
        return final


def main(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    request = parse(pathlib.Path(args.request).read_text(encoding='utf8'))
    if mode and request['mode'] != mode:
        raise ValueError('Entrypoint and request mode disagree')
    result = preflight(request) if args.preflight_only else execute(request)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
