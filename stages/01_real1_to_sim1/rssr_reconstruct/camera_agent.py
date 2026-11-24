"""Small RSSR subagent: fixed-pose numerical proposal, held-out K3 acceptance."""
import pathlib
import cv2
import mujoco
import numpy as np
from common import read,write,sha
from object_decision import decide
from camera_fit import color_centers,fit_camera

CAMERAS = {'head':'rssr_head','left_wrist':'left_wrist_rgbd','right_wrist':'right_wrist_rgbd'}

def camera_decide(config,budget,folder,system,payload,images):
    if config.get('camera_reasoning_effort'):
        from camera_decision import decide as camera_transport
        return camera_transport(config,budget,folder,system,payload,images)
    settings=dict(config,object_decision_max_tokens=config.get('camera_decision_max_tokens',8192),k3_timeout_s=config.get('camera_timeout_s',config['k3_timeout_s']),k3_attempts_per_decision=1)
    return decide(settings,budget,folder,system,payload,images)


def apply_profile(model, entries):
    from scipy.spatial.transform import Rotation
    for name, spec in entries.items():
        cid=model.camera(name).id;bid=int(model.cam_bodyid[cid]);parent=model.body(bid).name if bid else 'world'
        if parent != spec['parent_body']:
            raise ValueError('Camera parent changed: '+name)
        p=np.array(spec['position_in_parent_m']);r=np.array(spec['rotation_in_parent'])
        if p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Invalid camera transform')
        if not np.allclose(r.T@r,np.eye(3),atol=1e-6) or abs(np.linalg.det(r)-1)>1e-6:raise ValueError('Camera must have proper rotation')
        if not 20<=spec['fovy_degrees']<=110:raise ValueError('Invalid camera FOV')
        model.cam_pos[cid]=p;model.cam_quat[cid]=Rotation.from_matrix(r).as_quat()[[3,0,1,2]];model.cam_fovy[cid]=spec['fovy_degrees']


def initial_profile(model):
    from scipy.spatial.transform import Rotation
    out={}
    for name in CAMERAS.values():
        cid=model.camera(name).id;bid=int(model.cam_bodyid[cid]);q=model.cam_quat[cid]
        out[name]={'parent_body':model.body(bid).name if bid else 'world',
                   'position_in_parent_m':model.cam_pos[cid].tolist(),
                   'rotation_in_parent':Rotation.from_quat(q[[1,2,3,0]]).as_matrix().tolist(),
                   'fovy_degrees':float(model.cam_fovy[cid]),'hardware_calibrated':False}
    return out


def reference_frames(video,indices):
    cap=cv2.VideoCapture(str(video));out={}
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=cap.read()
            if not ok:raise RuntimeError('Missing Real1 camera reference frame')
            if frame.shape[:2]!=(630,1470):raise RuntimeError('Unknown Real1 camera panel layout')
            out[i]={role:cv2.cvtColor(frame[135:495,j*495:j*495+480],cv2.COLOR_BGR2RGB) for j,role in enumerate(CAMERAS)}
    finally:cap.release()
    return out


def sampled_states(config,selection):
    z=np.load(pathlib.Path(selection['trial'])/'rollout.npz')
    with np.load(pathlib.Path(config['import_case'])/'frame_action_mapping.npz') as mapping:
        ix=np.maximum(np.searchsorted(z['time_s'],mapping['source_time_s'],side='right')-1,0)
    return z,ix


def orientation_plan(config,budget,folder,selections,references,indices):
    """K3 selects one discrete roll per camera using training frames only."""
    from object_render import configure_cameras
    from scipy.spatial.transform import Rotation
    renders={}
    for kind,selection in selections.items():
        model=mujoco.MjModel.from_xml_path(str(pathlib.Path(selection['trial'])/'scene.xml'))
        configure_cameras(model,kind);model.site_rgba[:,3]=0;state=mujoco.MjData(model)
        z,ix=sampled_states(config,selection);renderer=mujoco.Renderer(model,height=360,width=480);renders[kind]={}
        try:
            for i in indices:
                state.qpos[:]=z['qpos'][ix[i]];mujoco.mj_forward(model,state);renders[kind][i]={}
                for role,name in CAMERAS.items():
                    renderer.update_scene(state,camera=name);renders[kind][i][role]=renderer.render().copy()
        finally:
            if hasattr(renderer,'close'):renderer.close()
            del renderer;z.close()
    sheets=[]
    for i in indices:
        sheet=np.zeros((1170,1440,3),np.uint8)
        for row,(title,images) in enumerate([('REAL1',references[i]),('SIM1',renders['sim1'][i]),('SIM2',renders['sim2'][i])]):
            for col,role in enumerate(CAMERAS):
                x=col*480;y=row*390;sheet[y+30:y+390,x:x+480]=images[role]
                cv2.putText(sheet,f'{title}/{role} t={i/24:.2f}',(x+8,y+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
        path=folder/f'planning_{i:04d}.jpg';cv2.imwrite(str(path),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR));sheets.append(path)
    plan=camera_decide(config,budget,folder/'planning',(pathlib.Path(__file__).parent/'prompts/camera_orientation_plan.txt').read_text(),{'allowed_roll_degrees':[0,90,180,270],'training_frames':indices},sheets)
    expected={(k,v) for k in ['sim1','sim2'] for v in CAMERAS};views=plan.get('views',[])
    if plan.get('video_visible') is not True or len(views)!=6 or {(r.get('kind'),r.get('view')) for r in views}!=expected:raise RuntimeError('Invalid camera orientation plan')
    for r in views:
        if r.get('roll_degrees') not in [0,90,180,270]:raise RuntimeError('Unbounded camera orientation proposal')
    return {(r['kind'],r['view']):r for r in views}


def run_camera_agent(config,budget,root):
    if config.get('camera_alignment',{}).get('fixed_profile_from'):
        from camera_fixed_profile import run_fixed
        return run_fixed(config,budget,root)
    if config.get('camera_alignment',{}).get('policy') in ['fixed_mount_box_fit_K3_repair_v2','fixed_mount_visible_fit_K3_repair_v3','fixed_rig_role_consistency_v4']:
        from camera_refinement import run
        return run(config,budget,root)
    root=pathlib.Path(root);folder=root/'camera_alignment';folder.mkdir(exist_ok=True)
    settings=config.get('camera_alignment',{})
    reference=pathlib.Path(config['import_case'])/'real1_condition.mp4'
    selections={k:read(root/k/'selection.json') for k in ['sim1','sim2']}
    inputs={'real1':sha(reference),'settings':settings,'stages':{k:{n:sha(pathlib.Path(s['trial'])/n) for n in ['scene.xml','rollout.npz']} for k,s in selections.items()}}
    if (folder/'input.json').exists() and read(folder/'input.json')!=inputs:raise RuntimeError('Camera alignment inputs changed')
    write(folder/'input.json',inputs)
    if (folder/'result.json').exists():return read(folder/'result.json')
    with np.load(pathlib.Path(config['import_case'])/'frame_action_mapping.npz') as mapping:count=len(mapping['source_frame_index'])
    indices=np.linspace(0,count-1,11).round().astype(int).tolist();train=set(indices[::2]);heldout=indices[1::2]
    references=reference_frames(reference,indices)
    observed={i:{v:color_centers(im) for v,im in views.items()} for i,views in references.items()}
    write(folder/'observations.json',{'train_frames':sorted(train),'heldout_frames':heldout,'detections':observed,'method':'largest unclipped saturated color component; proxy correspondences, not measured calibration'})
    planning_indices=[indices[0],indices[4],indices[-1]]
    plan=orientation_plan(config,budget,folder,selections,references,planning_indices)
    profiles={};reports={};sim_images={};numerical=True
    # This backend is intentionally small: colored rigid blocks. Other tasks need a measured profile or another detector.
    for kind,selection in selections.items():
        model=mujoco.MjModel.from_xml_path(str(pathlib.Path(selection['trial'])/'scene.xml'))
        from object_render import configure_cameras
        configure_cameras(model,kind);model.site_rgba[:,3]=0
        state=mujoco.MjData(model);z,ix=sampled_states(config,selection)
        initial=initial_profile(model);candidate={};reports[kind]={}
        object_names=[o['id'] for o in selection['scene']['objects']]
        for role,name in CAMERAS.items():
            from scipy.spatial.transform import Rotation
            proposal=plan[(kind,role)];initial[name]['rotation_in_parent']=(np.array(initial[name]['rotation_in_parent'])@Rotation.from_euler('z',proposal['roll_degrees'],degrees=True).as_matrix()).tolist()
            initial[name]['orientation_proposal']=proposal
            cid=model.camera(name).id;bid=int(model.cam_bodyid[cid]);rows=[]
            for i in indices:
                state.qpos[:]=z['qpos'][ix[i]];mujoco.mj_forward(model,state)
                for obj,pixel in observed[i][role].items():
                    if obj not in object_names:continue
                    position=z['object_position'][ix[i],object_names.index(obj)]
                    local=state.xmat[bid].reshape(3,3).T@(position-state.xpos[bid]) if bid else position
                    rows.append({'frame':i,'object':obj,'point_in_parent':local.tolist(),'pixel':pixel})
            limit=float(settings.get('head_translation_limit_m',.7) if bid==0 else settings.get('source_wrist_translation_limit_m',.12) if kind=='sim1' else settings.get('target_wrist_translation_limit_m',.05))
            candidate[name],report=fit_camera([r for r in rows if r['frame'] in train],[r for r in rows if r['frame'] not in train],initial[name],bid==0,limit)
            reports[kind][role]=report;numerical=numerical and report['passed']
        profiles[kind]=candidate;apply_profile(model,candidate)
        renderer=mujoco.Renderer(model,height=360,width=480);sim_images[kind]={}
        try:
            for i in heldout:
                state.qpos[:]=z['qpos'][ix[i]];mujoco.mj_forward(model,state)
                sim_images[kind][i]={}
                for role,name in CAMERAS.items():
                    renderer.update_scene(state,camera=name);sim_images[kind][i][role]=renderer.render().copy()
        finally:
            if hasattr(renderer,'close'):renderer.close()
            del renderer;z.close()
    profile={'schema':'rssr.camera_profile.v1','cameras':profiles,'hardware_calibrated':False,'world_or_parent_fixed_for_entire_episode':True,'role':'visual alignment estimate; not deployment calibration'}
    write(folder/'camera_profile.json',profile);write(folder/'geometry_report.json',reports)
    sheets=[]
    for i in heldout:
        sheet=np.zeros((3*390,3*480,3),np.uint8)
        for row,(title,views) in enumerate([('REAL1',references[i]),('SIM1',sim_images['sim1'][i]),('SIM2',sim_images['sim2'][i])]):
            for col,role in enumerate(CAMERAS):
                x=col*480;y=row*390;sheet[y+30:y+390,x:x+480]=views[role]
                cv2.putText(sheet,f'{title} / {role} / {i/24:.2f}s',(x+8,y+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
        p=folder/f'heldout_{i:04d}.jpg';cv2.imwrite(str(p),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR));sheets.append(p)
    system=(pathlib.Path(__file__).parent/'prompts/camera_alignment.txt').read_text()
    try:
        verdict=camera_decide(config,budget,folder/'verification',system,{'task':read(root/'task_contract.json')['task'],'camera_constraint':'one fixed parent-relative transform per view per embodiment; no scene/action changes','calibration_status':'unmeasured proxy visual fit'},sheets)
    except RuntimeError:
        verdict={'video_visible':False,'views':[],'unresolved':['K3 camera verification unavailable']}
    expected={(k,v) for k in ['sim1','sim2'] for v in CAMERAS};views=verdict.get('views',[])
    visual=bool(verdict.get('video_visible') is True and len(views)==6 and {(v.get('kind'),v.get('view')) for v in views}==expected and all(v.get('pass') is True and isinstance(v.get('confidence'),(int,float)) and v['confidence']>=.8 for v in views) and not verdict.get('unresolved'))
    result={'schema':'rssr.camera_alignment.v1','passed':bool(numerical and visual),'numerical_passed':bool(numerical),'visual_passed':visual,'hardware_calibrated':False,'profile':str(folder/'camera_profile.json'),'verdict':verdict,'physics_and_actions_modified':False,'fitted_frames':sorted(train),'heldout_frames':heldout,'state':'visual_match_verified' if numerical and visual else 'camera_match_not_verified'}
    write(folder/'result.json',result);print(__import__('json').dumps({'camera_subagent':result['state'],'numerical_passed':bool(numerical),'visual_passed':visual}),flush=True)
    return result
