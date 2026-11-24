"""Fixed-mount camera fitting and bounded K3 repair, separate from physics.

Only training images enter repair. Final held-out images are opened once after
the chosen profile is frozen. No robot/object state is an optimization variable.
"""
import copy,itertools,pathlib,json,os
import cv2,mujoco,numpy as np
from scipy.spatial.transform import Rotation
from common import read,write,sha
from camera_agent import CAMERAS,initial_profile,apply_profile,reference_frames,sampled_states,camera_decide
from camera_box_fit import detect_boxes,fit_boxes


def models(config,selections):
    from object_render import configure_cameras
    result={}
    for kind,selection in selections.items():
        m=mujoco.MjModel.from_xml_path(str(pathlib.Path(selection['trial'])/'scene.xml'))
        configure_cameras(m,kind);z,ix=sampled_states(config,selection)
        result[kind]=(m,mujoco.MjData(m),z,ix)
    return result


def render_samples(loaded,profiles,indices):
    result={}
    for kind,(m,d,z,ix) in loaded.items():
        apply_profile(m,profiles[kind]);renderer=mujoco.Renderer(m,height=360,width=480);result[kind]={}
        try:
            for i in indices:
                d.qpos[:]=z['qpos'][ix[i]];d.qvel[:]=z['qvel'][ix[i]];mujoco.mj_forward(m,d)
                result[kind][i]={}
                for role,name in CAMERAS.items():
                    renderer.update_scene(d,camera=name);result[kind][i][role]=renderer.render().copy()
        finally:
            if hasattr(renderer,'close'):renderer.close()
            del renderer
    return result


def sheets(folder,references,rendered,indices,prefix):
    paths=[]
    for i in indices:
        sheet=np.zeros((1170,1440,3),np.uint8)
        for row,(title,images) in enumerate([('REAL1',references[i]),('SIM1',rendered['sim1'][i]),('SIM2',rendered['sim2'][i])]):
            for col,role in enumerate(CAMERAS):
                x=col*480;y=row*390;sheet[y+30:y+390,x:x+480]=images[role]
                cv2.putText(sheet,f'{title}/{role} t={i/24:.2f}',(x+8,y+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
        p=folder/f'{prefix}_{i:04d}.jpg';cv2.imwrite(str(p),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR));paths.append(p)
    return paths


def observations(loaded,selections,references,indices):
    result={}
    for kind,(m,d,z,ix) in loaded.items():
        result[kind]={role:[] for role in CAMERAS}
        objects=selections[kind]['scene']['objects'];names=[o['id'] for o in objects]
        for i in indices:
            d.qpos[:]=z['qpos'][ix[i]];mujoco.mj_forward(m,d)
            for role,name in CAMERAS.items():
                cid=m.camera(name).id;bid=int(m.cam_bodyid[cid]);detected=detect_boxes(references[i][role])
                for obj,detection in detected.items():
                    if obj not in names:continue
                    j=names.index(obj);spec=objects[j]
                    corners=np.array(list(itertools.product([-1,1],repeat=3)))*np.array(spec['size_m'])/2
                    rotation=Rotation.from_quat(z['object_quaternion'][ix[i],j][[1,2,3,0]]).as_matrix()
                    world=corners@rotation.T+z['object_position'][ix[i],j]
                    parent=(world-d.xpos[bid])@d.xmat[bid].reshape(3,3) if bid else world
                    result[kind][role].append(dict(detection,frame=i,object=obj,corners_in_parent=parent.tolist()))
    return result


def visual_pass(verdict):
    if not isinstance(verdict,dict):return False
    expected={(k,v) for k in ['sim1','sim2'] for v in CAMERAS};views=verdict.get('views',[])
    if not isinstance(views,list) or not all(isinstance(v,dict) for v in views):return False
    return bool(verdict.get('video_visible') is True and len(views)==6 and
        {(r.get('kind'),r.get('view')) for r in views}==expected and
        all(r.get('pass') is True and type(r.get('confidence')) in (int,float) and .8<=r['confidence']<=1 and isinstance(r.get('reason'),str) and bool(r['reason'].strip()) for r in views)
        and isinstance(verdict.get('unresolved'),list) and not verdict['unresolved'])


def apply_patches(profiles,origin,verdict):
    if not isinstance(verdict,dict) or not isinstance(verdict.get('views'),list):raise ValueError('Invalid repair schema')
    new=copy.deepcopy(profiles);seen=set()
    for item in verdict.get('views',[]):
        if not isinstance(item,dict):raise ValueError('Invalid camera repair item')
        key=(item.get('kind'),item.get('view'))
        if key in seen or key[0] not in new or key[1] not in CAMERAS:raise ValueError('Invalid camera repair key')
        seen.add(key)
        if item.get('pass') is True:continue
        spec=new[key[0]][CAMERAS[key[1]]];base=origin[key[0]][CAMERAS[key[1]]]
        patch=item.get('patch',{})
        if not isinstance(patch,dict):raise ValueError('Invalid camera patch object')
        try:
            dp=np.array(patch.get('translation_camera_m',[0,0,0]),float)
            dr=np.array(patch.get('rotation_camera_xyz_degrees',[0,0,0]),float);df=float(patch.get('fovy_delta_degrees',0))
        except (TypeError,ValueError) as error:raise ValueError('Invalid camera patch numbers') from error
        limit=.12 if key[1]=='head' else .035
        if dp.shape!=(3,) or dr.shape!=(3,) or not np.isfinite(dp).all() or not np.isfinite(dr).all() or not np.isfinite(df) or np.max(abs(dp))>limit or np.max(abs(dr))>20 or abs(df)>15:raise ValueError('Camera repair outside step limits')
        R=np.array(spec['rotation_in_parent']);position=np.array(spec['position_in_parent_m'])+R@dp
        cumulative=.9 if key[1]=='head' else .20
        if np.linalg.norm(position-np.array(base['position_in_parent_m']))>cumulative:raise ValueError('Camera repair outside cumulative installation bounds')
        spec['position_in_parent_m']=position.tolist();spec['rotation_in_parent']=(R@Rotation.from_euler('xyz',dr,degrees=True).as_matrix()).tolist()
        spec['fovy_degrees']=float(np.clip(spec['fovy_degrees']+df,20,110));spec['basis']='fixed_camera_box_fit_and_bounded_K3_training_repair'
    if len(seen)!=6:raise ValueError('Camera repair requires all six views')
    return new


def native_check(config,root,loaded,profiles,origin,selections,indices):
    from camera_rig_check import source_event_indices,validate_rig
    evidence=read(pathlib.Path(root)/'source_construction/evidence/action_evidence.json')
    with np.load(pathlib.Path(config['import_case'])/'frame_action_mapping.npz') as mapping:
        events=source_event_indices(evidence['events'],mapping['source_time_s'])
    saved=pathlib.Path(root)/'camera_alignment/native_event_mapping.json'
    if not saved.exists() or read(saved)!=events:raise ValueError('Native visibility events were not frozen before camera optimization')
    report=validate_rig(loaded,profiles,origin,selections,{kind:events['indices'] for kind in loaded},
                        diagnostic_indices={kind:indices for kind in loaded})
    if not config.get('source_task_success_required',False):
        # User-defined scope: task completion/visibility belongs to Sim2.
        # Preserve source diagnostics, but do not turn its task outcome into a gate.
        report['all_stages_visibility_diagnostic']=copy.deepcopy(report['task_visibility'])
        target_events=[r for r in report['task_visibility']['events'] if r['kind']=='sim2']
        target_errors=[e for e in report['task_visibility']['errors'] if e.startswith('sim2:')]
        report['task_visibility']=dict(report['task_visibility'],scope='sim2_only',events=target_events,
            errors=target_errors,passed=bool(target_events and not target_errors and all(r['passed'] for r in target_events)))
        report['passed']=bool(report['rig_integrity']['passed'] and report['task_visibility']['passed'])
    report['source_event_mapping']=events
    facts={kind:[{'frame':f['frame_index'],'views':{role:{group:round(v['fraction'],3) for group,v in view['coverage'].items()}
                  for role,view in f['views'].items()}} for f in value['frames'] if f['frame_index'] in indices]
           for kind,value in report['kinds'].items()}
    facts['failed_task_visibility']=[r for r in report['task_visibility']['events'] if not r['passed']]
    facts['rig_errors']=report['rig_integrity']['errors']
    facts['visibility_errors']=report['task_visibility']['errors']
    return report,facts


def run(config,budget,root):
    root=pathlib.Path(root);folder=root/'camera_alignment';folder.mkdir(exist_ok=True)
    settings=config['camera_alignment'];reference=pathlib.Path(config['import_case'])/'real1_condition.mp4'
    selections={k:read(root/k/'selection.json') for k in ['sim1','sim2']}
    inputs={'real1':sha(reference),'settings':settings,'frame_mapping_sha256':sha(pathlib.Path(config['import_case'])/'frame_action_mapping.npz'),
            'action_evidence_sha256':sha(root/'source_construction/evidence/action_evidence.json'),
            'stages':{k:{n:sha(pathlib.Path(s['trial'])/n) for n in ['scene.xml','rollout.npz']} for k,s in selections.items()}}
    if (folder/'input.json').exists() and read(folder/'input.json')!=inputs:raise RuntimeError('Camera inputs changed')
    write(folder/'input.json',inputs)
    if (folder/'result.json').exists():
        cached=read(folder/'result.json')
        if sha(folder/'camera_profile.json')!=cached['profile_sha256']:raise RuntimeError('Cached camera profile changed')
        if settings.get('policy')=='fixed_rig_role_consistency_v4' and sha(folder/'native_event_mapping.json')!=cached['native_event_mapping_sha256']:raise RuntimeError('Cached camera event mapping changed')
        return cached
    with np.load(pathlib.Path(config['import_case'])/'frame_action_mapping.npz') as z:
        count=len(z['source_frame_index'])
        if settings.get('policy')=='fixed_rig_role_consistency_v4':
            from camera_rig_check import source_event_indices
            events=source_event_indices(read(root/'source_construction/evidence/action_evidence.json')['events'],z['source_time_s'])
            event_path=folder/'native_event_mapping.json'
            if event_path.exists() and read(event_path)!=events:raise ValueError('Frozen native camera events changed')
            if not event_path.exists():write(event_path,events)
    # Different indices from r18; validation images are never passed to repair.
    train=np.round(np.array([0,.18,.36,.54,.72,1])*(count-1)).astype(int).tolist()
    heldout=np.round(np.array([.09,.27,.45,.63,.87])*(count-1)).astype(int).tolist()
    write(folder/'split.json',{'train':train,'heldout':heldout,'selected_before_optimization':True})
    references=reference_frames(reference,train);loaded=models(config,selections)
    profiles={kind:initial_profile(values[0]) for kind,values in loaded.items()}
    if settings.get('source_urdf_camera_prior',False):
        from source_camera_prior import source_wrist_prior
        entries,evidence=source_wrist_prior(config,loaded['sim1'][0]);profiles['sim1'].update(entries)
        write(folder/'source_camera_prior.json',evidence)
    origin=copy.deepcopy(profiles)
    # First fix discrete orientation; numerical local fitting starts from a
    # K3-selected orientation instead of searching through inverted silhouettes.
    rendered=render_samples(loaded,profiles,[train[0],train[2],train[-1]])
    orientation_system=(pathlib.Path(__file__).parent/'prompts/camera_orientation_plan.txt').read_text()
    orientation_payload={'allowed_roll_degrees':[0,90,180,270],'training_frames':[train[0],train[2],train[-1]]}
    orientation_images=sheets(folder,references,rendered,[train[0],train[2],train[-1]],'orientation')
    plan=None
    if settings.get('reuse_orientation_from'):
        source=pathlib.Path(settings['reuse_orientation_from']);prior_input=read(source/'input.json')
        source_images=list(prior_input['images']);equivalence=[]
        for old,new in zip(source_images,orientation_images):
            a=cv2.imread(old);b=cv2.imread(str(new))
            if a is None or b is None or a.shape!=b.shape:equivalence.append({'equivalent':False});continue
            delta=np.abs(a.astype(float)-b.astype(float))
            equivalence.append({'mean_abs_u8':float(delta.mean()),'max_abs_u8':float(delta.max()),'equivalent':bool(delta.mean()<=.001 and delta.max()<=8)})
        previous_inputs=read(source.parent/'input.json')
        same_physics=previous_inputs['real1']==inputs['real1'] and previous_inputs['stages']==inputs['stages']
        if (prior_input['system']==orientation_system and prior_input['payload']==orientation_payload and prior_input['model']==os.environ.get('KIMI_MODEL','Kimi-K3') and
            same_physics and len(equivalence)==len(orientation_images) and all(e['equivalent'] for e in equivalence)):
            plan=read(source/'result.json');write(folder/'orientation'/'result.json',plan)
            write(folder/'orientation'/'import.json',{'source':str(source),'input_sha256':sha(source/'input.json'),
                'result_sha256':sha(source/'result.json'),'same_system_payload_model_and_physics':True,
                'image_equivalence':equivalence,'note':'subpixel GPU/JPEG raster differences; not a relaxed camera acceptance threshold','new_api_calls':0})
    if plan is None:
        plan=camera_decide(config,budget,folder/'orientation',orientation_system,orientation_payload,orientation_images)
    seen=set()
    for item in plan.get('views',[]):
        key=(item.get('kind'),item.get('view'));roll=item.get('roll_degrees')
        if key in seen or key[0] not in profiles or key[1] not in CAMERAS or roll not in [0,90,180,270]:raise ValueError('Invalid orientation plan')
        seen.add(key);spec=profiles[key[0]][CAMERAS[key[1]]]
        spec['rotation_in_parent']=(np.array(spec['rotation_in_parent'])@Rotation.from_euler('z',roll,degrees=True).as_matrix()).tolist()
    if len(seen)!=6 or plan.get('video_visible') is not True:raise ValueError('Incomplete orientation plan')
    rows=observations(loaded,selections,references,train);write(folder/'training_observations.json',rows)
    fit_reports={}
    for kind in profiles:
        fit_reports[kind]={}
        for role,name in CAMERAS.items():
            limit=.7 if role=='head' else .16
            from camera_render_fit import fit_visible_camera
            profiles[kind][name],fit_reports[kind][role]=fit_visible_camera(loaded[kind],kind,role,name,rows[kind][role],profiles[kind][name],limit)
    write(folder/'initial_fit.json',{'profiles':profiles,'reports':fit_reports})
    if settings.get('reuse_training_repair_from'):
        previous=pathlib.Path(settings['reuse_training_repair_from'])
        previous_inputs=read(previous.parent/'input.json');previous_split=read(previous.parent/'split.json')
        if previous_inputs['real1']!=inputs['real1'] or previous_inputs['stages']!=inputs['stages'] or previous_split['train']!=train or previous_split['heldout']!=heldout:
            raise ValueError('Training camera warm start has different data or split')
        if (previous.parent/'heldout_observations.json').exists() or (previous.parent/'verification').exists():
            raise ValueError('Warm start already exposed heldout; cannot reuse this validation split')
        before=read(previous/'profile_before.json');proposal=read(previous/'result.json')
        profiles=apply_patches(before,origin,proposal)
        for kind in profiles:apply_profile(loaded[kind][0],profiles[kind])
        write(folder/'training_profile_import.json',{'source':str(previous),'profile_before_sha256':sha(previous/'profile_before.json'),
            'result_sha256':sha(previous/'result.json'),'same_physics_and_split':True,'heldout_exposed':False,'profiles':profiles})
    prompt=(pathlib.Path(__file__).parent/'prompts/camera_repair.txt').read_text();history=[]
    role_protocol=settings.get('policy')=='fixed_rig_role_consistency_v4'
    critique_frames=[train[0],train[2],train[-1]]
    for round_id in range(int(settings.get('max_repair_rounds',4))):
        dest=folder/f'repair_{round_id:02d}';dest.mkdir(exist_ok=True)
        rendered=render_samples(loaded,profiles,critique_frames)
        native_facts=None
        if role_protocol:
            check,native_facts=native_check(config,root,loaded,profiles,origin,selections,critique_frames)
            write(dest/'native_rig_check.json',check)
        compact_history=[{'views':[{'kind':v.get('kind'),'view':v.get('view'),'pass':v.get('pass'),'reason':v.get('reason'),'patch':v.get('patch')} for v in h.get('views',[])],
                          'tool_rejection':h.get('tool_rejection')} for h in history[-1:]]
        verdict=camera_decide(config,budget,dest,prompt,{'task':read(root/'task_contract.json')['task'],
            'fovy_degrees':{k:{r:profiles[k][n]['fovy_degrees'] for r,n in CAMERAS.items()} for k in profiles},
            'round':round_id,'last_step':compact_history,'hardware_calibrated':False,'native_sim_visibility':native_facts,
            'task_success_required_only_for':'sim2','sim1_object_execution_is_diagnostic_only':True,
            'no_changes_to_objects_or_actions':True},sheets(dest,references,rendered,critique_frames,'training'))
        write(dest/'profile_before.json',profiles);history.append(verdict)
        if visual_pass(verdict) and (not role_protocol or check['passed']):break
        if round_id+1<int(settings.get('max_repair_rounds',4)):
            try:profiles=apply_patches(profiles,origin,verdict)
            except ValueError as error:
                # Invalid LLM edits are fed back within the bounded loop; the
                # current valid fixed profile stays intact.
                rejection={'tool_rejection':str(error),'profile_unchanged':True}
                write(dest/'patch_rejected.json',rejection);history.append(rejection)
    profile={'schema':'rssr.camera_profile.v2','cameras':profiles,'origin_cameras':origin,'hardware_calibrated':False,
        'world_or_parent_fixed_for_entire_episode':True,'role':'fixed virtual camera visual registration; not hardware calibration'}
    write(folder/'camera_profile.json',profile)
    # Freeze the candidate BEFORE loading held-out reference images.
    profile_hash=sha(folder/'camera_profile.json');write(folder/'profile_frozen_before_validation.json',{'sha256':profile_hash,'training_only_history':history})
    heldref=reference_frames(reference,heldout);heldrows=observations(loaded,selections,heldref,heldout)
    from camera_box_fit import evaluate_boxes
    from camera_render_fit import evaluate_rendered_camera
    reports={};numerical=True
    for kind in profiles:
        reports[kind]={}
        for role,name in CAMERAS.items():
            report=evaluate_rendered_camera(loaded[kind],name,heldrows[kind][role],profiles[kind][name])
            before=evaluate_rendered_camera(loaded[kind],name,heldrows[kind][role],origin[kind][name])
            report['before_median_px']=before['median_px']
            report['no_regression']=bool(report['median_px'] is not None and before['median_px'] is not None and report['median_px']<=before['median_px']+2)
            report['passed']=bool(report['passed'] and report['no_regression'])
            report['full_box_projection_diagnostic']=evaluate_boxes(heldrows[kind][role],profiles[kind][name]);reports[kind][role]=report
            numerical=numerical and report['passed']
    write(folder/'geometry_report.json',reports);write(folder/'heldout_observations.json',heldrows)
    rig=None;native_facts=None
    if role_protocol:
        rig,native_facts=native_check(config,root,loaded,profiles,origin,selections,heldout)
        write(folder/'rig_report.json',rig)
    rendered=render_samples(loaded,profiles,heldout)
    verdict=camera_decide(config,budget,folder/'verification',(pathlib.Path(__file__).parent/'prompts/camera_alignment.txt').read_text(),
        {'task':read(root/'task_contract.json')['task'],'camera_constraint':'fixed parent-relative cameras; robot, objects and actions untouched',
         'calibration_status':'unmeasured visual fit to approximate geometry','native_sim_visibility':native_facts,
         'task_success_required_only_for':'sim2','sim1_object_execution_is_diagnostic_only':True},sheets(folder,heldref,rendered,heldout,'heldout'))
    visual=visual_pass(verdict)
    accepted=bool((rig['passed'] if role_protocol else numerical) and visual)
    result={'schema':'rssr.camera_alignment.v3' if role_protocol else 'rssr.camera_alignment.v2','passed':accepted,'reference_bbox_passed':bool(numerical),'visual_passed':visual,
        'acceptance_protocol':settings['policy'],'rig_integrity_passed':rig['rig_integrity']['passed'] if rig else None,
        'native_event_mapping_sha256':sha(folder/'native_event_mapping.json') if role_protocol else None,
        'task_visibility_passed':rig['task_visibility']['passed'] if rig else None,
        'reference_bbox_is_calibration_ground_truth':False,
        'acceptance_scope':'fixed virtual rig, corresponding camera roles and native task visibility; not measured extrinsics' if role_protocol else 'legacy proxy bbox and K3 views',
        'hardware_calibrated':False,'profile':str(folder/'camera_profile.json'),'profile_sha256':profile_hash,'verdict':verdict,
        'physics_and_actions_modified':False,'fitted_frames':train,'heldout_frames':heldout,
        'state':'fixed_rig_and_role_verified' if accepted and role_protocol else ('visual_match_verified' if accepted else 'camera_match_not_verified')}
    for m,d,z,ix in loaded.values():z.close()
    for kind,s in selections.items():
        for name,digest in inputs['stages'][kind].items():
            if sha(pathlib.Path(s['trial'])/name)!=digest:raise RuntimeError('Camera agent modified physics')
    write(folder/'result.json',result);print(json.dumps({'camera_subagent':result['state'],'reference_bbox_passed':numerical,'rig_passed':rig['passed'] if rig else None,'visual_passed':visual}),flush=True)
    return result
