"""Visual/source-action scene reconstruction. Sim1 task success is diagnostic."""
import copy,pathlib
import numpy as np,mujoco
from rssr_protocol import ValidationError
from common import read,write,sha
from pilot_api import call,propose
from object_physics import action_evidence,build,commands,rollout

CRITERIA=('object_identity','relative_layout','geometry_scale','table_support')
CONTRACT={'tool':'scene_patch','fields':{'object_xyz_shift_m':'object ID -> [dx,dy,dz], each axis <=.04 m; z recomputed from table support','object_yaw_delta_rad':'object ID -> radians, step <=.3','object_size_scale':'object ID -> [sx,sy,sz] scale, each .9..1.1'},'cumulative_bounds':'centres within .15m of visual initialization, size within .5..1.5 of initialization; fixed table/mass/friction; Sim1 task success not objective'}
CONTRACT['fields']['object_roll_pitch_delta_rad']='rssr.scene.v2 only: object ID -> [roll_delta,pitch_delta], per axis <=.35 radians. Final tilt in [-pi,pi]; table height follows rotated geometry.'

def validate_scene(scene):
    if isinstance(scene,dict) and scene.get('schema')=='rssr.scene.v2':
        from rssr_scene import validate
        return validate(scene)
    if not isinstance(scene,dict) or set(scene)!={'container_id','movable_id','table_shift_z_m','objects'}:raise ValueError('Invalid scene fields')
    if scene['container_id']!='basket' or scene['movable_id']!='blue_cube' or scene['table_shift_z_m']!=0:raise ValueError('Unsupported task/table')
    ids=[]
    for obj in scene['objects']:
        if set(obj)-{'id','shape','position_m','size_m','yaw_rad','rgba','mass_kg','wall_m'}:raise ValueError('Unexpected object field')
        name=obj['id'];ids.append(name)
        if name not in ['basket','blue_cube','red_cube','green_cube']:raise ValueError('Unknown object identity')
        p=np.asarray(obj['position_m'],float);size=np.asarray(obj['size_m'],float);rgba=np.asarray(obj['rgba'],float)
        if p.shape!=(3,) or size.shape!=(3,) or not np.isfinite([*p,*size,obj['yaw_rad']]).all() or np.any(size<.015) or np.any(size>.45):raise ValueError('Invalid object geometry')
        if rgba.shape!=(4,) or np.any(rgba<0) or np.any(rgba>1):raise ValueError('Invalid rgba')
        if not -.58<=p[0]<=.58 or not -1.13<=p[1]<=-.37:raise ValueError('Object outside tabletop')
        if abs(p[2]-(.75+size[2]/2+.0003))>.003:raise ValueError('Object initial height must respect support')
        expected_shape='open_box' if name=='basket' else 'box'
        if obj['shape']!=expected_shape or obj['mass_kg']!=(.12 if name=='basket' else .04):raise ValueError('Shape/mass contract mismatch')
        if name=='basket' and not .004<=obj.get('wall_m',0)<=.015:raise ValueError('Basket wall invalid')
    if len(ids)!=len(set(ids)) or not {'basket','blue_cube'}<=set(ids):raise ValueError('Missing/duplicate objects')
    return copy.deepcopy(scene)

def scene_edit(scene,initial,patch):
    if not isinstance(patch,dict) or set(patch)-set(CONTRACT['fields']):raise ValueError('Unknown scene tool field')
    out=copy.deepcopy(scene);base={o['id']:o for o in initial['objects']};known=set(base)
    for name,value in patch.items():
        if not isinstance(value,dict) or set(value)-known:raise ValueError('Unknown object in patch')
    for obj in out['objects']:
        name=obj['id'];shift=np.asarray(patch.get('object_xyz_shift_m',{}).get(name,[0,0,0]),float)
        scale=np.asarray(patch.get('object_size_scale',{}).get(name,[1,1,1]),float)
        yaw=patch.get('object_yaw_delta_rad',{}).get(name,0.)
        tilt=np.asarray(patch.get('object_roll_pitch_delta_rad',{}).get(name,[0.,0.]),float)
        if shift.shape!=(3,) or not np.isfinite(shift).all() or abs(shift).max()>.04:raise ValueError('Scene shift bound')
        if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any(scale<.9) or np.any(scale>1.1):raise ValueError('Scene scale bound')
        if type(yaw) not in (int,float) or not np.isfinite(yaw) or abs(yaw)>.3:raise ValueError('Scene yaw bound')
        if tilt.shape!=(2,) or not np.isfinite(tilt).all() or abs(tilt).max()>.35:raise ValueError('Scene tilt bound')
        if tilt.any():
            if scene.get('schema')!='rssr.scene.v2':raise ValueError('Tilt requires generic scene geometry')
            obj['roll_pitch_rad']=(np.asarray(obj.get('roll_pitch_rad',[0.,0.]))+tilt).tolist()
        obj['position_m']=(np.array(obj['position_m'])+shift).tolist();obj['size_m']=(np.array(obj['size_m'])*scale).tolist()
        obj['yaw_rad']+=yaw
        if scene.get('schema')!='rssr.scene.v2':obj['position_m'][2]=.7503+obj['size_m'][2]/2
        elif obj.get('support_id')=='table':
            from rssr_scene import lowest_geometry_offset
            obj['position_m'][2]=.7503-lowest_geometry_offset(obj)
        if np.linalg.norm(np.array(obj['position_m'])-base[name]['position_m'])>.15:raise ValueError('Cumulative centre shift exceeded')
        ratio=np.array(obj['size_m'])/base[name]['size_m']
        if np.any(ratio<.5) or np.any(ratio>1.5):raise ValueError('Cumulative scale exceeded')
    if out==scene:raise ValueError('Scene edit has no effective geometric change')
    return validate_scene(out)

def initial_images(config):return [pathlib.Path(config['import_case'])/(r+'_first.jpg') for r in ['head','left_wrist','right_wrist']]

def frame0(config,scene,folder,profile=None,kind='sim1'):
    import cv2
    builder=build;commander=commands
    if scene.get('schema')=='rssr.scene.v2':
        from pilot_physics import build as builder,commands as commander
    folder=pathlib.Path(folder);m=builder(config,kind,folder,scene);d=mujoco.MjData(m);t,q,g=commander(config,kind,m);d.qpos[:]=q[0]
    if profile:
        from camera_agent import apply_profile
        apply_profile(m,profile)
    m.site_rgba[:,3]=0;mujoco.mj_forward(m,d);renderer=mujoco.Renderer(m,height=360,width=480);paths=[]
    try:
        for name in ['rssr_head','left_wrist_rgbd','right_wrist_rgbd']:
            renderer.update_scene(d,camera=name);p=folder/(name+'.jpg')
            if not cv2.imwrite(str(p),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Source scene frame image could not be written')
            paths.append(p)
    finally:
        from pilot_camera_agent import _release_renderer
        _release_renderer(renderer)
    bad=[]
    for ct in d.contact:
        if ct.dist < -.003:
            names=[m.geom(ct.geom1).name or '',m.geom(ct.geom2).name or '']
            if any(any(n.startswith(o['id']) for o in scene['objects']) for n in names):bad.append({'geoms':names,'penetration_m':float(-ct.dist)})
    return paths,{'initial_penetrations':bad,'objects':scene['objects'],'model_sha256':sha(folder/'scene.xml')}

def scene_score(verdict,numeric):
    failed=[];uncertainty=0.
    for key in CRITERIA:
        row=verdict.get('criteria',{}).get(key,{})
        c=row.get('confidence',0)
        if type(c) not in (int,float) or not np.isfinite(c) or not 0<=c<=1:raise ValueError('Invalid visual confidence')
        if row.get('pass') is not True or c<.75 or not row.get('evidence'):failed.append(key)
        uncertainty+=1-c
    rank=[len(numeric['initial_penetrations']),len(failed),uncertainty]
    return {'passed':rank[:2]==[0,0],'rank':rank,'failed':failed,'scope':'approximate visual scene, not metric ground truth; no Sim1 task gate'}

def initialize(config,folder,knowledge=None):
    folder=pathlib.Path(folder);dest=folder/'initial_scene.json'
    if dest.exists():return read(dest)
    evidence=action_evidence
    generic=config.get('scene_backend')=='rssr.scene.v2'
    if generic:
        from pilot_physics import action_evidence as evidence
    ev=evidence(config,'sim1',folder/'source_action_geometry')
    if config.get('interaction_revision')=='v16':
        from rssr_interaction import capabilities
        ev['target_capabilities']=capabilities(config)
    prompt=(pathlib.Path(__file__).parent/'prompts'/('rssr_scene_initial.txt' if generic else 'pilot_scene_initial.txt')).read_text(encoding='utf8')
    payload={'task':config['task'],'action_geometry_hint':ev,'table':{'top_z_m':.75,'size_m':[1.2,.8,.05]}}
    if knowledge is not None:
        payload['knowledge']=knowledge.snapshot('reconstruction',include_referenced=True)
    if generic:payload.update(scene_contract='rssr.scene.v2',task_id=config['task_family'])
    previous=folder/'initial_decision/full_input.json'
    if config.get('stage_protocol') and previous.exists():
        saved=read(previous)['payload'];old=dict(saved);old.pop('knowledge',None)
        current=dict(payload);current.pop('knowledge',None)
        if old!=current:raise ValueError('Scene initialization evidence changed during recovery')
        if 'knowledge' in saved:payload['knowledge']=saved['knowledge']
    from protocol_decision import DecisionOutputExhausted
    try:
        response=call(config,folder/'initial_decision',prompt,payload,initial_images(config),stage='scene.initialize')
    except DecisionOutputExhausted as error:
        from rssr_errors import StageUnresolved
        raise StageUnresolved('reconstruction','No complete scene within the fixed model-output budget') from error
    if response.get('supported') is not True:
        if generic:
            from rssr_errors import StageUnresolved
            raise StageUnresolved('reconstruction','Visual scene unresolved: '+str(response.get('reason')))
        raise ValueError('Unsupported visual task: '+str(response.get('reason')))
    scene=validate_scene(response['scene']);write(dest,scene);return scene

def reconstruct(config,knowledge,root,initial,profile=None):
    from protocol_decision import DecisionOutputExhausted
    root=pathlib.Path(root);best=None;history=[];scene=copy.deepcopy(initial);trace=None
    if (root/'selection.json').exists():return read(root/'selection.json')
    for i in range(config['max_scene_agent_rounds']):
        trial=root/f'trial_{i:02d}';trial.mkdir(parents=True,exist_ok=True)
        if i:
            try:patch,trace=propose(config,knowledge,'reconstruction',trial/'decision',{'best_scene':best['scene'],'best_verdict':best['verdict'],'best_numeric':best['numeric'],'history':history,'tool_contract':CONTRACT},initial_images(config))
            except DecisionOutputExhausted as e:
                history.append({'round':i,'stopped':True,'method_budget_exhausted':str(e)});break
            except (ValueError,TypeError,KeyError,ValidationError) as e:
                history.append({'round':i,'invalid_operator':str(e),'decision_path':str(trial/'decision')});write(trial/'outcome.json',history[-1]);continue
            if patch is None:break
            try:scene=scene_edit(best['scene'],initial,patch)
            except (ValueError,TypeError,KeyError,ValidationError) as e:
                history.append({'round':i,'invalid_proposal':str(e),'trace':trace});write(trial/'outcome.json',history[-1]);continue
        from rssr_evidence_cache import load as load_evidence, save as save_evidence
        evidence_identity={'scene':scene,'profile':profile,
                           'source_images':{str(p):sha(p) for p in initial_images(config)},
                           'source_action_sha256':sha(pathlib.Path(config['import_case'])/'source_action_state.npz')}
        evidence=load_evidence(trial/'render_evidence.json',evidence_identity)
        if evidence is None:
            images,numeric=frame0(config,scene,trial/'frame0',profile)
            save_evidence(trial/'render_evidence.json',evidence_identity,
                          {'images':list(map(str,images)),'numeric':numeric}, images)
        else:
            images=[pathlib.Path(p) for p in evidence['images']];numeric=evidence['numeric']
        prompt=(pathlib.Path(__file__).parent/'prompts/pilot_scene_review.txt').read_text(encoding='utf8')
        try:
            verdict=call(config,trial/'review',prompt,{'task':config['task'],'scene':scene,'numeric':numeric,'image_order':['Real1 HEAD','Real1 LEFT_WRIST','Real1 RIGHT_WRIST','Sim1 HEAD','Sim1 LEFT_WRIST','Sim1 RIGHT_WRIST']},initial_images(config)+images,stage='scene.review')
        except DecisionOutputExhausted as error:
            if best is None:
                from rssr_errors import StageUnresolved
                raise StageUnresolved('reconstruction','No visual scene verdict within the fixed model-output budget') from error
            history.append({'round':i,'stopped':True,'method_budget_exhausted':str(error)});break
        score=scene_score(verdict,numeric);candidate={'scene':scene,'verdict':verdict,'numeric':numeric,'objective':score,'trial':str(trial)}
        if best and trace:knowledge.record('reconstruction',config['case_id'],trace,best['objective']['rank'],score['rank'],config.get('task_family','di1')+':first_frame_layout',score['passed'])
        if best is None or tuple(score['rank'])<tuple(best['objective']['rank']):best=copy.deepcopy(candidate)
        history.append({'round':i,'objective':score,'trace':trace});write(trial/'outcome.json',candidate)
        if score['passed']:break
    best['history']=history;write(root/'selection.json',best);return best

def review_fixed_scene(config,root,scene,profile=None):
    """Revalidate a changed final scene; never change it after physical execution."""
    root=pathlib.Path(root);dest=root/'selection.json'
    if dest.exists():
        saved=read(dest)
        if saved['scene']!=scene:raise ValueError('Final scene identity changed')
        return saved
    from rssr_evidence_cache import load,save
    identity={'scene':scene,'profile':profile,
              'source_images':{str(p):sha(p) for p in initial_images(config)},
              'source_action_sha256':sha(pathlib.Path(config['import_case'])/'source_action_state.npz')}
    evidence=load(root/'render_evidence.json',identity)
    if evidence is None:
        images,numeric=frame0(config,scene,root/'frame0',profile)
        evidence={'images':list(map(str,images)),'numeric':numeric}
        save(root/'render_evidence.json',identity,evidence,images)
    else:images=[pathlib.Path(p) for p in evidence['images']];numeric=evidence['numeric']
    prompt=(pathlib.Path(__file__).parent/'prompts/pilot_scene_review.txt').read_text(encoding='utf8')
    verdict=call(config,root/'review',prompt,{'task':config['task'],'scene':scene,'numeric':numeric,
        'image_order':['Real1 HEAD','Real1 LEFT_WRIST','Real1 RIGHT_WRIST','Sim1 HEAD','Sim1 LEFT_WRIST','Sim1 RIGHT_WRIST']},
        initial_images(config)+images,stage='scene.review')
    result={'scene':scene,'verdict':verdict,'numeric':numeric,'objective':scene_score(verdict,numeric),
            'trial':str(root),'immutable_final_scene_review':True}
    write(dest,result);return result
