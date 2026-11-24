"""RSSR: frozen retarget -> K3 condition design -> Seedance -> K3 repair loop.

There are no per-case human review hooks. A failed verifier never becomes a pass
because the budget is exhausted. The best rejected candidate remains inspectable.
"""
import argparse,concurrent.futures,json,os,pathlib,shutil,sys,time,zipfile,subprocess,fcntl
from common import read,write,sha,event,load_credentials,Budget,run_lock,seal,segments,semantic_accept,verifier_action,inspection_times,CRITERIA
from prepare import prepare,hashes
from media import clip,probe,image_at,head_reference,format_check,concatenate
from k3 import K3,K3DecisionError
from quality import scan,repair_constraints
from adapter import pack_t,unpack_t
MediaHost = Seedance = install_tunnel = HTTP = sanitized = None  # legacy video-gen backend omitted

CODE=pathlib.Path(__file__).resolve().parent
PROMPTS={k:(CODE/'prompts'/(f'{k}_v2.txt' if k in ['generate','verifier'] else f'{k}.txt')).read_text(encoding='utf8') for k in ['planner','generate','verifier']}

def valid_times(values,start,end):
    return [float(t) for t in values if isinstance(t,(int,float)) and start<=float(t)<end][:2]

def contract_for(plan,start,end):
    return {k:plan.get(k) for k in ['objects','initial_state','final_state','appearance_profile']} | {
        'events':[e for e in plan.get('events',[]) if isinstance(e,dict) and isinstance(e.get('time_s'),(int,float)) and e.get('time_s',0)<end and e.get('end_s',e.get('time_s',0))>=start],
        'times_are_global_seconds':True}

def score(report):return sum(report.get('criteria',{}).get(k,{}).get('pass') is True for k in CRITERIA)

def execution_stages(cases,gate_ids):
    gate=set(gate_ids)
    if gate-{i['case_id'] for i in cases}:raise ValueError('Unknown initial quality-gate case')
    return [[i for i in cases if i['case_id'] in gate],[i for i in cases if i['case_id'] not in gate]] if gate else [cases]

def verify(k3,folder,*args,**kwargs):
    try:return k3.decision(folder,*args,**kwargs)
    except K3DecisionError:
        result={'video_visible':False,'criteria':{},'violations':[], 'uncertainties':['Verifier unavailable after bounded retries'],
          'task_success_visible':None,'infrastructure_error':True}
        write(pathlib.Path(folder)/'unavailable.json',result)
        return result

def run_segment(c,spec,index,plan,previous,k3,video,config,external_repair='',force=False):
    root=pathlib.Path(config['run_root']);folder=c/'segments'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=True);fps=config['fps']
    start=spec['start']/fps;end=spec['end']/fps
    real1=folder/'real1.mp4';sim2=folder/'sim2.mp4'
    clip(c/'real1_condition.mp4',real1,spec['start'],spec['end'],config);clip(c/'sim2_robot_only.mp4',sim2,spec['start'],spec['end'],config)
    motion_condition=pack_t(sim2,folder/'sim2_t_layout.mp4',config) if config.get('internal_layout')=='t' else sim2
    write(folder/'segment.json',spec|{'global_start_s':start,'global_end_s':end,'no_retarget_change':True})
    best=None;repair=external_repair;references=valid_times(plan.get('reference_times_s',[]),start,end);contract=contract_for(plan,start,end)
    contract['camera_geometry_caution']=read(c/'camera_geometry.json')
    prior=read(folder/'selection.json') if (folder/'selection.json').exists() else None
    if prior and not force:return prior
    existing=max([int(p.name.split('_')[-1]) for p in folder.glob('attempt_*') if p.is_dir()] or [0]) if force else 0
    if force:best=prior
    dense=[float(e['time_s'])-start for e in contract['events'] if isinstance(e.get('time_s'),(float,int))]
    for attempt in range(1,config['max_video_attempts_per_segment']+1):
        if attempt<=existing:continue
        a=folder/f'attempt_{attempt:02d}';a.mkdir(parents=True,exist_ok=True)
        if (a/'outcome.json').exists():
            outcome=read(a/'outcome.json')
            if outcome.get('path') and (best is None or outcome.get('score',-1)>best.get('score',-1)):best=outcome
            if outcome.get('accepted'):write(folder/'selection.json',outcome);return outcome
            repair=outcome.get('repair','');references=valid_times(outcome.get('reference_times_s',[]),start,end);continue
        reference_paths=[image_at(motion_condition,0,a/'sim2_structure.jpg')]
        reference_labels=['IMAGE 1：Sim2 本段首帧，锁定三视角结构、S1姿态和相机；其中缺失物体需按Real1补回。']
        reference_paths.append(image_at(motion_condition,(spec['end']-spec['start'])//2,a/'s1_identity.jpg'))
        reference_labels.append('IMAGE 2：同一Sim2中段S1本体身份和三个独立机位参考；只锁定本体身份，各时刻姿态仍逐帧服从VIDEO 1。')
        chosen=sorted(set([start,*references,min(end-1/fps,(start+end)/2)]))[:4]
        if config.get('initial_scene_only'):chosen=[0.0]
        if config.get('dynamic_real1_repair_reference') and attempt>1 and references:
            chosen=sorted(set([*chosen,references[0]]))
        for t in chosen:
            reference_source=c/'real1_condition.mp4' if config.get('initial_scene_only') else real1
            reference_index=round(t*fps) if config.get('initial_scene_only') else min(spec['end']-spec['start']-1,round((t-start)*fps))
            p=head_reference(reference_source,reference_index,a/f'real1_reference_{len(reference_paths)}.jpg');reference_paths.append(p)
            reference_labels.append(f'IMAGE {len(reference_paths)}：源Real1在全局t={t:.3f}s的头部场景观察。仅提供实验室外观、物体身份和该时刻状态，不能把此图当成任何目标机位，不复制其中黑色机器人。')
        if previous and previous.get('path'):
            prev_index=spec['start']-previous['spec']['start']
            prior_video=pathlib.Path(previous['path'])
            if config.get('internal_layout')=='t':prior_video=pack_t(prior_video,a/'previous_t.mp4',config)
            p=image_at(prior_video,prev_index,a/'continuity.jpg');reference_paths.append(p)
            reference_labels.append(f'IMAGE {len(reference_paths)}：上一段生成视频在本段起点的同一时刻，用于延续材质、照明、物体身份及布局。若与冻结动作/Real1状态矛盾，先遵守两段原始视频。')
        prompt=PROMPTS['generate'].format(task=read(c/'case.json')['task'],start_s=start,end_s=end,contract=json.dumps(contract,ensure_ascii=False),references='\n'.join(reference_labels),repair=str(repair)[:3000] or '首次生成：逐项遵守上述运动、物体与跨视角契约。')
        if config.get('internal_layout')=='t':prompt=prompt.replace('三视角横排画布','三视角T布局画布').replace('三个窗口从左至右始终对应 VIDEO 1 的 HEAD、LEFT_WRIST、RIGHT_WRIST','三个窗口始终为：上方大画面 HEAD，左下 LEFT_WRIST，右下 RIGHT_WRIST')+'\n保留VIDEO 1完整T布局、三个不同机位的分界和宽高；上方头部相机并非Real1参考图的相机，不得复制参考图构图。初始场景图只说明物体身份及环境材质；事件发生后按观察记录更新物体状态，不把初态复制到终态。'
        seed=config['seed']+int(c.name[:2])*10000+index*100+attempt
        generated=video.generate(a,motion_condition,real1,reference_paths,prompt,seed)
        if generated['status'] in ['uncertain_create','poll_deadline']:
            # Resume this exact attempt by task ID. Never issue another paid POST here.
            write(folder/'pending.json',generated);raise RuntimeError(generated['status'])
        if generated['status']!='succeeded':
            outcome={'accepted':False,'status':'generation_failed','provider':generated,'attempt':attempt,'score':-1,'repair':'保留原任务与动作条件；重新执行同一视频编辑任务。'};write(a/'outcome.json',outcome);continue
        candidate=pathlib.Path(generated['path']);fmt=format_check(candidate,motion_condition,config)
        # Some encoders append one extra trailing frame. Dropping that frame preserves time.
        if not fmt['passed'] and fmt.get('generated',{}).get('frames')==spec['end']-spec['start']+1 and fmt.get('checks',{}).get('fps') and fmt.get('checks',{}).get('aspect'):
            aligned=a/'real2_aligned.mp4';clip(candidate,aligned,0,spec['end']-spec['start'],config);candidate=aligned
            fmt=format_check(candidate,motion_condition,config);fmt['dropped_extra_trailing_frame']=True
        write(a/'format_check.json',fmt)
        if not fmt['passed']:
            if not fmt.get('checks',{}).get('frame_count'):
                video.frame_contract_failed.set()
                write(root/'provider_frame_contract_failure.json',{'attempt':str(a),'format':fmt,'action':'stop_new_paid_creates_without_padding_or_time_warp'})
                raise RuntimeError('Provider frame count differs from calibrated window')
            outcome={'accepted':False,'status':'format_rejected','attempt':attempt,'score':-1,'repair':'必须保持视频1的原始24 FPS、准确时长、三视角完整画布；禁止改速、增加停顿或改变宽高比。'};write(a/'outcome.json',outcome);repair=outcome['repair'];continue
        if config.get('internal_layout')=='t':candidate=unpack_t(candidate,a/'real2_horizontal.mp4',config)
        text=f'本段完整视频的全局时间范围是 {start:.3f}..{end:.3f}s，图像标签为本段局部时间。任务：'+read(c/'case.json')['task']+'。只验证本段阶段，不强迫提前完成全任务。独立检查所有可见动作与物体关系。'
        if spec['start']==0 and spec['end']==read(c/'render_manifest.json')['frames']:
            text+='\n本段覆盖完整演示，必须核对完整任务终态；只有在Real1和Real2均实际可见对应完成状态时，task_success_visible才为true。'
        text+='\n冻结相机的工具中心投影（不是物体可见性证明，不能代替实际像素证据）：'+json.dumps(contract['camera_geometry_caution'],ensure_ascii=False)
        videos={'Real1':real1,'Sim2':sim2,'Real2':candidate}
        automatic=scan(videos,a/'pixel_quality.json',fps,event_times=dense)
        text+='\n全帧扫描需要核查的异常：'+json.dumps({'jumps':automatic['suspicious_jumps'][:4],'duplicate_views':automatic['possible_duplicate_views']},ensure_ascii=False)
        report=verify(k3,a/'verification',PROMPTS['verifier'],text,videos,maximum=16,rate=2,explicit_indices=automatic['inspection_indices'])
        verification_path=a/'verification/result.json'
        for inspection in range(config.get('max_additional_inspections',1)):
            if report.get('infrastructure_error') or not automatic['passed'] or verifier_action(report,config['verifier_confidence_min'])!='inspect':break
            times,intervals=inspection_times(report,(spec['end']-spec['start']-1)/fps,14)
            indices=sorted(set(round(t*fps) for t in times));inspect_folder=a/f'inspection_{inspection+1:02d}'
            event(root,'additional_evidence_requested',case=c.name,segment=index,attempt=attempt,intervals=intervals,frames=len(indices))
            report=verify(k3,inspect_folder,PROMPTS['verifier'],text+'\n本次是补充取证，不是新生成。优先检查前次指出的证据缺口区间（局部秒）：'+json.dumps(intervals)+'。不预设视频正确或错误。基于本次可见帧重新判定全部项目；依然看不清则保留uncertainties，不能提高置信度冒充直接证据。',videos,maximum=16,rate=2,explicit_indices=indices)
            verification_path=inspect_folder/'result.json'
        accepted=automatic['passed'] and semantic_accept(report,config['verifier_confidence_min'])
        # The verifier returns local reference times; convert them to global times.
        times=[start+t for t in valid_times(report.get('reference_times_s',[]),0,end-start)]
        outcome={'accepted':accepted,'status':'model_verifier_passed' if accepted else 'model_verifier_rejected','path':str(candidate),'attempt':attempt,'score':score(report),'repair':repair_constraints(report,automatic),'reference_times_s':times,'verification':str(verification_path),'spec':spec,'sha256':sha(candidate),'not_physical_ground_truth':True,'pixel_quality_passed':automatic['passed']}
        write(a/'outcome.json',outcome)
        if best is None or outcome['score']>best.get('score',-1):best=outcome
        event(root,'segment_verdict',case=c.name,segment=index,attempt=attempt,accepted=accepted,passed_criteria=outcome['score'])
        if accepted:write(folder/'selection.json',outcome);return outcome
        if report.get('infrastructure_error') or (automatic['passed'] and verifier_action(report,config['verifier_confidence_min'])=='inspect'):
            outcome['status']='verifier_unavailable' if report.get('infrastructure_error') else 'evidence_insufficient';write(a/'outcome.json',outcome);write(folder/'selection.json',outcome);return outcome
        repair=outcome['repair'];references=times
    if best is None:best={'accepted':False,'status':'no_valid_candidate','spec':spec,'score':-1}
    write(folder/'selection.json',best);return best

def run_case(item,config,k3,video):
    root=pathlib.Path(config['run_root']);c=root/item['case_id'];final=c/'final';final.mkdir(exist_ok=True)
    if (final/'result.json').exists():return read(final/'result.json')
    if not read(c/'preparation.json').get('ready'):return {'case':c.name,'status':'input_failed','accepted':False}
    meta=read(c/'render_manifest.json');specs=segments(meta['frames'],config['segment_frames'],config['minimum_segment_frames'],config['overlap_frames'],config.get('fixed_length_windows',False))
    plan=k3.decision(c/'planning',PROMPTS['planner'],'完整真实演示与冻结的S1轨迹，任务：'+item['task'],{'Real1':c/'real1_condition.mp4','Sim2':c/'sim2_robot_only.mp4'},maximum=16,rate=1)
    if plan.get('video_visible') is not True:raise RuntimeError('K3 cannot observe visual input')
    selected=[]
    for i,spec in enumerate(specs):
        candidate=run_segment(c,spec,i,plan,selected[-1] if selected else None,k3,video,config);selected.append(candidate)
        if not candidate.get('path'):break
    if len(selected)!=len(specs) or not all(s.get('path') for s in selected):
        result={'case':c.name,'status':'incomplete_generation','accepted':False,'segments':selected};write(final/'result.json',result);return result
    if len(specs)==1:
        # The single segment IS the complete episode. Reuse the verdict on the
        # exact same bytes; a second identical audit adds no independent evidence.
        s=selected[0];candidate=pathlib.Path(s['path'])
        if sha(candidate)!=s['sha256']:raise RuntimeError('Verified candidate bytes changed')
        audit=read(s['verification']) if pathlib.Path(s['verification']).exists() else {}
        fmt=format_check(candidate,c/'sim2_robot_only.mp4',config)
        accepted=fmt['passed'] and s.get('accepted') is True and audit.get('task_success_visible') is True and plan.get('source_task_completed') is True
        deliverable=final/'real2_three_view.mp4';shutil.copy2(candidate,deliverable)
        shutil.copy2(c/'frame_action_mapping.npz',final/'nominal_action_mapping.npz')
        result={'case':c.name,'status':'model_verifier_passed' if accepted else s['status'],'accepted':accepted,'video':str(deliverable),'sha256':sha(deliverable),'task_success_visible_by_k3':audit.get('task_success_visible'),'physical_task_success':None,'hardware_ready':False,'action_labels':'nominal Sim2 FK trajectory; pixel-to-action equivalence not metrically established','semantic_verification':'K3 timestamped frame sequence, not ground truth','full_episode':True,'segments':selected,'final_audit':s['verification'],'full_audit_reuses_identical_single_segment':True,'automatic_human_case_edits':False}
        write(final/'result.json',result);event(root,'case_finished',case=c.name,status=result['status']);return result
    # Full-sequence audit also examines seams. One bounded automatic downstream repair sweep.
    for sweep in range(2):
        output=final/f'real2_full_revision_{sweep}.mp4'
        concatenate([(pathlib.Path(s['path']),spec['drop_prefix']) for s,spec in zip(selected,specs)],output,config)
        fmt=format_check(output,c/'sim2_robot_only.mp4',config);write(final/f'format_{sweep}.json',fmt)
        seam_frames=[s['start']+s['drop_prefix'] for s in specs[1:]]
        videos={'Real1':c/'real1_condition.mp4','Sim2':c/'sim2_robot_only.mp4','Real2':output}
        automatic=scan(videos,final/f'pixel_quality_{sweep}.json',config['fps'],seam_frames)
        audit=verify(k3,final/f'audit_{sweep}',PROMPTS['verifier'],'现在检查完整任务而非片段。任务：'+item['task']+'。重点检查真实分段边界、场景/本体身份连续性和任务终态。必须在Real1中看到对应完成状态，并在Real2中实际出现，task_success_visible才能为true。全帧扫描异常：'+json.dumps(automatic['suspicious_jumps'][:4]),videos,maximum=16,rate=2,explicit_indices=automatic['inspection_indices'])
        audit_path=final/f'audit_{sweep}/result.json'
        if not audit.get('infrastructure_error') and automatic['passed'] and verifier_action(audit,config['verifier_confidence_min'])=='inspect' and config.get('max_additional_inspections',1):
            extra_times,intervals=inspection_times(audit,(meta['frames']-1)/config['fps'],14)
            extra_indices=sorted(set(round(t*config['fps']) for t in extra_times))
            audit=verify(k3,final/f'audit_{sweep}_inspection',PROMPTS['verifier'],'这是完整任务的补充取证。任务：'+item['task']+'。优先核实证据不足的时间区间：'+json.dumps(intervals)+'。检查完整任务终态和跨段一致性；不因之前的判定推定正确，仍不清楚就保留不确定。',videos,maximum=16,rate=2,explicit_indices=extra_indices)
            audit_path=final/f'audit_{sweep}_inspection/result.json'
        accepted=fmt['passed'] and automatic['passed'] and all(s.get('accepted') for s in selected) and semantic_accept(audit,config['verifier_confidence_min']) and audit.get('task_success_visible') is True and plan.get('source_task_completed') is True
        if accepted or sweep==1:break
        violations=audit.get('violations',[]);times=[v.get('time_s') for v in violations if isinstance(v,dict) and isinstance(v.get('time_s'),(int,float))]
        if not times:break
        earliest=max(0,min(times));begin=next((i for i,s in enumerate(specs) if s['end']/config['fps']>earliest),0)
        changed=False
        for i in range(begin,len(specs)):
            previous=selected[i-1] if i else None
            repaired=run_segment(c,specs[i],i,plan,previous,k3,video,config,external_repair=repair_constraints(audit,automatic),force=True)
            if repaired.get('path') and repaired.get('path')!=selected[i].get('path'):selected[i]=repaired;changed=True
        if not changed:break
    deliverable=final/'real2_three_view.mp4';shutil.copy2(output,deliverable)
    shutil.copy2(c/'frame_action_mapping.npz',final/'nominal_action_mapping.npz')
    result={'case':c.name,'status':'model_verifier_passed' if accepted else 'generated_but_rejected','accepted':accepted,'video':str(deliverable),'sha256':sha(deliverable),'task_success_visible_by_k3':audit.get('task_success_visible'),'physical_task_success':None,'hardware_ready':False,'action_labels':'nominal Sim2 FK trajectory; pixel-to-action equivalence not metrically established','semantic_verification':'K3 timestamped frame sequence, not ground truth','full_episode':True,'segments':selected,'final_audit':str(audit_path),'automatic_human_case_edits':False}
    write(final/'result.json',result);event(root,'case_finished',case=c.name,status=result['status']);return result

def package(root,results):
    with zipfile.ZipFile(root/'rssr10_deliverables.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in ['summary.json','frozen_run.json','budget.json','selection.json']:
            if (root/name).exists():z.write(root/name,name)
        for r in results:
            c=root/r['case'];f=c/'final'
            if not f.exists():continue
            for p in f.rglob('*'):
                if not p.is_file() or any(part.startswith('frames') for part in p.parts) or p.name.startswith(('join_','concat.')) or p.name.startswith('real2_full_revision'):continue
                z.write(p,str(p.relative_to(root)))
            for p in c.glob('segments/*/attempt_*/prompt.txt'):z.write(p,str(p.relative_to(root)))
            for p in c.glob('segments/*/attempt_*/verification/result.json'):z.write(p,str(p.relative_to(root)))
            for p in c.glob('segments/*/attempt_*/inspection_*/result.json'):z.write(p,str(p.relative_to(root)))
            if (c/'planning/result.json').exists():z.write(c/'planning/result.json',str((c/'planning/result.json').relative_to(root)))

def recover_prior_downloads(config,video):
    """Fetch already-paid tasks from the previous infrastructure revision; no new POST."""
    prior=pathlib.Path(config['import_root'])
    for receipt in prior.glob('*/segments/*/attempt_*/create_response.json'):
        folder=receipt.parent;target=folder/'real2.mp4'
        if target.exists():continue
        task_id=read(receipt).get('id')
        if not task_id:continue
        try:
            status,record=video.call('GET',config['ark_endpoint']+'/'+task_id)
            write(folder/'recovered_provider_status.json',sanitized(record))
            if status!=200 or record.get('status')!='succeeded':continue
            url=(record.get('content') or {}).get('video_url')
            if not url:continue
            with HTTP.open(url,timeout=180) as stream,target.with_suffix('.part').open('wb') as f:
                while data:=stream.read(1024*1024):f.write(data)
            target.with_suffix('.part').replace(target)
            write(folder/'download.json',{'task_id':task_id,'sha256':sha(target),'recovered_without_new_create':True,'usage':record.get('usage')})
        except Exception:continue

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,default=CODE/'config.json');p.add_argument('--prepare-only',action='store_true');p.add_argument('--key-stdin',action='store_true');p.add_argument('--wait-for-regression',action='store_true');a=p.parse_args();config=read(a.config)
    root=pathlib.Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
    if a.key_stdin:os.environ['ARK_API_KEY']=sys.stdin.readline().strip()
    with run_lock(root):
        if config.get('wait_for_import_run_to_finish'):
            event(root,'waiting_for_prior_run',prior=config['import_root'])
            with (pathlib.Path(config['import_root'])/'run.lock').open('a') as previous:
                fcntl.flock(previous,fcntl.LOCK_EX)
        load_credentials(config['env_file']);selection=prepare(config,CODE)
        if a.prepare_only:return
        if a.wait_for_regression:
            deadline=time.time()+1500
            event(root,'waiting_for_inflight_regression')
            while not (root/'regression/result.json').exists():
                if time.time()>deadline:raise RuntimeError('Regression preflight did not finish; no new generation submitted')
                time.sleep(3)
        if config.get('require_regression_pass'):
            if not (root/'regression/result.json').exists():
                subprocess.run([config['python'],str(CODE/'regression.py')],check=True)
            if read(root/'regression/result.json').get('passed') is not True:
                raise RuntimeError('Revised verifier must reject the known-bad regression before new paid generation')
        frozen=seal(root,CODE,config,selection,hashes(config,selection));event(root,'run_sealed',sha256=frozen)
        if not os.environ.get('ARK_API_KEY'):raise RuntimeError('ARK_API_KEY must be provided by environment or private stdin')
        raise RuntimeError('Legacy tunnel video-gen backend is disabled in open-source pipeline')
        if config.get('budget_carry_from') and not (root/'budget.json').exists():
            shutil.copy2(config['budget_carry_from'],root/'budget.json')
            write(root/'budget_carry_provenance.json',{'source':config['budget_carry_from'],'sha256':sha(config['budget_carry_from']),'limits_not_reset':True})
        budget=Budget(root,{'video_gen':config.get('max_video_creates',0),'k3':config['max_k3_requests']});k3=K3(config,budget)
        # Native video was tested against this gateway and was not visible; frames were verified.
        write(root/'visual_transport.json',{'mode':config['video_input_mode'],'native_video_probe_visible':False,'image_probe_visible':True,'model':k3.model,'not_every_video_frame_observed':True})
        results=[]
        with MediaHost(config) as host:
            video=Seedance(config,budget,host,os.environ['ARK_API_KEY'])
            recover_prior_downloads(config,video)
            gate=set(config.get('initial_gate_cases',[]))
            stages=execution_stages(selection['cases'],gate)
            for stage_index,items in enumerate(stages):
                if stage_index and not all(r.get('accepted') is True for r in results):
                    results.extend({'case':i['case_id'],'status':'not_run_initial_quality_gate_failed','accepted':False} for i in items)
                    event(root,'quality_gate_stopped_expansion',remaining=len(items))
                    break
                with concurrent.futures.ThreadPoolExecutor(max_workers=config['case_concurrency']) as pool:
                    futures={pool.submit(run_case,item,config,k3,video):item for item in items}
                    for f in concurrent.futures.as_completed(futures):
                        item=futures[f]
                        try:result=f.result()
                        except Exception as e:
                            result={'case':item['case_id'],'status':'pipeline_error','accepted':False,'error_type':type(e).__name__,'reason':str(e)[:160]};event(root,'case_error',**result)
                        results.append(result);write(root/'summary.json',{'completed_cases':len(results),'planned_cases':len(selection['cases']),'accepted_by_k3':sum(r.get('accepted') is True for r in results),'results':results,'all_actions_unchanged':True})
            write(root/'summary.json',{'completed_cases':sum(r['status']!='not_run_initial_quality_gate_failed' for r in results),'planned_cases':len(selection['cases']),'accepted_by_k3':sum(r.get('accepted') is True for r in results),'results':results,'all_actions_unchanged':True})
            recover_prior_downloads(config,video)
        package(root,results);event(root,'run_finished',cases=len(results),accepted=sum(r.get('accepted') is True for r in results))

if __name__=='__main__':main()
