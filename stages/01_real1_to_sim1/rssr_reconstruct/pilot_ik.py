"""Fresh default/random IK baseline; learned changes are explicit JSON fields."""
import copy, hashlib, pathlib, sys, time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from common import read,write,sha
from object_physics import target_model, build, hand_pose, LINK
from object_retarget import phase_offsets
from object_target_agent import validate_correction
from rssr_ik_initialization import POLICIES, initial_guess

NUMERIC={'ik_damping':(.001,.5),'ik_nullspace_gain':(0.,1.),'ik_joint_limit_margin':(0.,.3),
         'upstream_weight':(0.,2.),'ik_max_iters':(20,200),'ik_tol':(.1,5.),'ik_pos_weight':(.1,4.),'ik_rot_weight':(0.,4.)}
CATEGORICAL={'ik_nullspace_target':['none','init','mid','prev'],'ik_warm_start':['prev','init']}

def merge(a,b):
    out=copy.deepcopy(a)
    for k,v in b.items():out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else copy.deepcopy(v)
    return out

def initial_program():
    return {'ik':{'ik_joint_limit_margin':.02},'arms':{'left':{},'right':{}},'seed_index':0,'initialization':'model_default',
            'correction':{'hand_offsets_m':{'left':[0,0,0],'right':[0,0,0]},'vertical_grasp_mix':0.}}

def edit(program,patch):
    if isinstance(patch,dict) and 'solver_operations' in patch:
        from rssr_v22_bridge import apply_calls
        ordinary={k:v for k,v in patch.items() if k!='solver_operations'}
        base=edit(program,ordinary)
        return edit(apply_calls(base,patch['solver_operations']),{})
    if not isinstance(patch,dict) or set(patch)-{'ik','arms','seed_index','initialization','correction','event_corrections','time_knots','vertical_alignment_frame','grasp_frame_offsets_m'}:raise ValueError('Retarget edits only IK/arm weights/seed/initialization/correction/source events/time knots/declared frames and offsets; scene fixed')
    out=merge(program,patch)
    if 'event_corrections' in patch:
        from rssr_action_events import merge_patch
        out['event_corrections']=merge_patch(program.get('event_corrections',{}),patch['event_corrections'])
    if type(out['seed_index']) is not int or not 0<=out['seed_index']<=65535:raise ValueError('Seed index out of range')
    if out['initialization'] not in POLICIES:raise ValueError('Unknown IK initialization policy')
    from rssr_orientation_tools import FRAMES
    if out.get('vertical_alignment_frame','tool_site_legacy') not in FRAMES:raise ValueError('Unknown vertical alignment frame')
    if 'grasp_frame_offsets_m' in out:
        from rssr_local_offsets import validate
        out['grasp_frame_offsets_m']=validate(out['grasp_frame_offsets_m'])
    for k,v in out['ik'].items():
        if k in NUMERIC:
            if type(v) not in (float,int) or not np.isfinite(v) or not NUMERIC[k][0]<=v<=NUMERIC[k][1]:raise ValueError('IK outside tool bounds '+k)
            if k=='ik_max_iters' and int(v)!=v:raise ValueError('Integer iterations required')
        elif k in CATEGORICAL:
            if v not in CATEGORICAL[k]:raise ValueError('Unknown IK category')
        else:raise ValueError('Unknown IK field '+k)
    if set(out['arms'])!={'left','right'}:raise ValueError('Both arm configurations required')
    for spec in out['arms'].values():
        if set(spec)-{'ik_pos_weight','ik_rot_weight'}:raise ValueError('Arm-specific edits only weights')
        for k,v in spec.items():
            if type(v) not in (float,int) or not np.isfinite(v) or not NUMERIC[k][0]<=v<=NUMERIC[k][1]:raise ValueError('Arm weight out of range')
    out['correction']=validate_correction(out['correction'])
    if 'event_corrections' in out:
        from rssr_action_events import validate
        out['event_corrections']=validate(out['event_corrections'])
    if 'time_knots' in out:
        from rssr_time_knots import validate
        out['time_knots']=validate(out['time_knots'])
    return out

def reference(config):
    src=np.load(pathlib.Path(config['import_case'])/'source_action_state.npz')
    if config.get('interaction_revision')=='v16':
        # One geometry convention for reconstruction, source replay and target
        # goals. Observation.state remains evidence, not a second TCP source.
        from pilot_physics import build,commands,hand_pose
        folder=pathlib.Path(config['import_case'])/'source_action_reference'
        cache=folder/'reference.npz'
        if cache.exists():
            with np.load(cache) as z:return {k:z[k].copy() for k in z.files}
        m=build(config,'sim1',folder);d=mujoco.MjData(m);t,q,g=commands(config,'sim1',m)
        contract=read(pathlib.Path(config['model_root'])/'source/scene/task_frame_contract.json')
        ref={'source_time_s':t}
        for side in ['left','right']:
            poses=[]
            for qi in q:
                d.qpos[:]=qi;mujoco.mj_forward(m,d);pos,rot=hand_pose(m,d,'sim1',side)
                # Piper grasp axes include the declared flange-to-grasp
                # rotation. The tool-site translation is the actual FK site.
                grasp=np.eye(4);grasp[:3,3]=pos
                grasp[:3,:3]=rot@np.asarray(contract['chains'][side]['source_reference_from_grasp'])[:3,:3]
                poses.append(grasp@np.linalg.inv(np.asarray(contract['chains'][side]['target_site_from_grasp'])))
            poses=np.asarray(poses);ref[side+'_target_position']=poses[:,:3,3];ref[side+'_target_rotation']=poses[:,:3,:3]
            ref['source_'+side+'_gripper']=src['action'][:,13 if side=='left' else 6]
        np.savez_compressed(cache,**ref);return ref
    contract=read(pathlib.Path(config['model_root'])/'source/scene/task_frame_contract.json')
    world=np.asarray(contract['world_from_task']);ref={};t=src['timestamp']-src['timestamp'][0]
    for side,offset,gindex in [('left',20,13),('right',7,6)]:
        c=contract['chains'][side];poses=np.tile(np.eye(4),(len(t),1,1))
        poses[:,:3,3]=src['state'][:,offset:offset+3];poses[:,:3,:3]=Rotation.from_euler('xyz',src['state'][:,offset+3:offset+6]).as_matrix()
        mapped=world@np.asarray(c['task_from_source_base'])@poses@np.asarray(c['source_reference_from_grasp'])@np.linalg.inv(np.asarray(c['target_site_from_grasp']))
        ref[side+'_target_position']=mapped[:,:3,3];ref[side+'_target_rotation']=mapped[:,:3,:3]
        ref['source_'+side+'_gripper']=src['action'][:,gindex]
    ref['source_time_s']=t;return ref

def plan(config,folder,program):
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'plan.json').exists():
        rec=read(folder/'plan.json')
        if rec['program']!=program or sha(folder/'trajectory.npz')!=rec['sha256']:raise ValueError('Saved plan mismatch')
        if rec.get('goal_tracking') and sha(rec['goal_tracking']['path'])!=rec['goal_tracking']['sha256']:
            raise ValueError('Saved IK target/FK evidence changed')
        return rec
    sys.path.append(str(pathlib.Path(config['model_root'])/'source'))
    from agent_retarget.trackc.ik_dualarm import CoupledChainIK
    from agent_retarget.trackc.robots import astribot_s1_robot
    from agent_retarget.config import RetargetConfig
    program=edit(initial_program(),program);ref=reference(config);cfg=RetargetConfig()
    defaults={k:getattr(cfg,k) for k in [*NUMERIC,*CATEGORICAL]}
    for k,v in program['ik'].items():setattr(cfg,k,int(v) if k=='ik_max_iters' else v)
    arms=[copy.deepcopy(cfg),copy.deepcopy(cfg)]
    for i,s in enumerate(['left','right']):
        for k,v in program['arms'][s].items():setattr(arms[i],k,v)
    # The dynamics asset folds fixed torso joints 3/4 into body transforms.
    # The declared IK model keeps all four coordinates; the constrained solver
    # fixes 3/4 to the same angles. Exported upstream_q retains that explicit form.
    ik_model=pathlib.Path(config['model_root'])/'source/scene/s1_tabletop.xml'
    m=mujoco.MjModel.from_xml_path(str(ik_model));d=mujoco.MjData(m);robot=astribot_s1_robot()
    solver=CoupledChainIK(m,d,robot.upstream_joint_names,[(robot.left_arm.joint_names,'arm_left_tool'),(robot.right_arm.joint_names,'arm_right_tool')])
    lo,hi,_=solver.limits(RetargetConfig())
    seed=int.from_bytes(hashlib.sha256(f"rssr-pilot|{config['seed']}|{config['case_id']}|{program['seed_index']}".encode()).digest()[:8],'big')
    rng=np.random.Generator(np.random.PCG64(seed))
    q0=initial_guess(solver.q_default(),lo,hi,program['initialization'],rng)
    alignment_frame=program.get('vertical_alignment_frame','tool_site_legacy')
    grasp_contract=read(pathlib.Path(config['model_root'])/'source/scene/task_frame_contract.json')
    solver.q_default=lambda:q0.copy();previous=q0.copy();qs=[];errors=[];t=ref['source_time_s'];c=program['correction'];goals={}
    for s in ['left','right']:
        offsets=np.asarray(c['hand_offsets_m'][s])+phase_offsets(t,np.clip(ref['source_'+s+'_gripper']/1.05,0,1),c.get('phase_hand_offsets_m',{}).get(s,{}))
        from rssr_action_events import corrections
        event_shift,event_rotation=corrections(t,np.clip(ref['source_'+s+'_gripper']/1.05,0,1),program.get('event_corrections',{}).get(s,[]))
        if program.get('time_knots'):
            from rssr_time_knots import corrections as knot_corrections
            knot_shift,knot_rotation=knot_corrections(t,program['time_knots'].get(s,[]))
            event_shift=event_shift+knot_shift
            event_rotation=event_rotation+knot_rotation
            if np.max(np.abs(event_rotation))>35.+1e-10:
                raise ValueError('Combined event/time-knot world rotation exceeds 35 degrees per axis')
        offsets=offsets+event_shift
        if np.max(np.abs(offsets))>.16+1e-10:raise ValueError('Combined source-event and global displacement exceeds .16m')
        rot=ref[s+'_target_rotation'].copy();mix=c.get('vertical_grasp_mix_by_hand',{}).get(s,c['vertical_grasp_mix'])
        from rssr_orientation_tools import vertical_alignment
        conversion=np.asarray(grasp_contract['chains'][s]['target_site_from_grasp'])[:3,:3]
        rot=vertical_alignment(rot,mix,alignment_frame,conversion)
        rot=Rotation.from_euler('xyz',event_rotation,degrees=True).as_matrix()@rot
        from rssr_local_offsets import goals as local_goals
        pos=local_goals(ref[s+'_target_position'],offsets,rot,conversion,program.get('grasp_frame_offsets_m',{}).get(s))
        goals[s]=(pos,rot)
    site_positions={s:[] for s in ['left','right']};site_rotations={s:[] for s in site_positions}
    started=time.monotonic()
    for k in range(len(t)):
        solver._continuity_reference=qs[-1] if qs else None
        target=[(goals[s][0][k],goals[s][1][k]) for s in ['left','right']]
        q,_=solver.solve_frame(target,previous,cfg,arms);qs.append(q.copy())
        previous=q.copy() if cfg.ik_warm_start=='prev' else q0.copy()
        e=[]
        for i,(p,r) in enumerate(target):
            ap,ar=solver._site_pose(i);e.extend([np.linalg.norm(p-ap)*1000,np.rad2deg(Rotation.from_matrix(r@ar.T).magnitude())])
            side=['left','right'][i];site_positions[side].append(ap.copy());site_rotations[side].append(ar.copy())
        errors.append(e)
    qs=np.array(qs);z=dict(ref,upstream_q=qs[:,:4],left_q=qs[:,4:11],right_q=qs[:,11:18]);path=folder/'trajectory.npz';np.savez_compressed(path,**z)
    full=np.tile(m.qpos0,(len(t),1))
    for s in ['left','right']:
        for j in range(1,8):full[:,m.joint(f'astribot_arm_{s}_joint_{j}').qposadr[0]]=z[s+'_q'][:,j-1]
    for j in range(1,5):full[:,m.joint(f'astribot_torso_joint_{j}').qposadr[0]]=z['upstream_q'][:,j-1]
    # Geometry-based gripper command mapping is handled by the physics backend.
    np.savez_compressed(folder/'robot_qpos.npz',qpos=full)
    tracking=folder/'goal_tracking.npz'
    np.savez_compressed(tracking,time_s=t,
        **{s+'_target_position':goals[s][0] for s in goals},
        **{s+'_target_rotation':goals[s][1] for s in goals},
        **{s+'_planned_site_position':np.asarray(site_positions[s]) for s in goals},
        **{s+'_planned_site_rotation':np.asarray(site_rotations[s]) for s in goals})
    rec={'program':program,'defaults':defaults,'random_seed':str(seed),'initial_guess':q0.tolist(),'trajectory':str(path),'sha256':sha(path),'mean_errors_left_mm_deg_right_mm_deg':np.mean(errors,axis=0).tolist(),'max_adjacent_joint_step_rad':float(abs(np.diff(qs,axis=0)).max()),'wall_seconds':time.monotonic()-started,'old_learned_trajectory_imported':False,'source_reference':'recorded source end-effector states mapped by frozen estimated task/TCP contract; source action remains unchanged'}
    if config.get('interaction_revision')=='v16':rec['source_reference']='recorded source joint-action FK; same source grasp-site geometry as scene initialization; estimated bases, not hardware calibration'
    rec['goal_tracking']={'path':str(tracking),'sha256':sha(tracking),'frame':'world','site_names':{s:'arm_'+s+'_tool' for s in goals}}
    rec['grasp_frame_definitions']={s:{'site_from_grasp_rotation':np.asarray(grasp_contract['chains'][s]['target_site_from_grasp'])[:3,:3].tolist(),
        'basis':'declared robot frame contract, not measured hardware calibration'} for s in goals}
    write(folder/'plan.json',rec);return rec
