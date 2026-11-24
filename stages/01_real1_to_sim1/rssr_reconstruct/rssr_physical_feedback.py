"""Connect generic task contracts to saved MuJoCo state; no visual task oracle."""
import pathlib
import numpy as np
from common import read,write,sha
from rssr_task_contracts import compile_contract,evaluate
from rssr_protocol import digest


def grasp_events(time_s,command):
    t=np.asarray(time_s);g=np.asarray(command);events=[];inside=False;start=None;armed=False
    if t.ndim!=1 or g.shape!=t.shape or len(t)<2 or not np.isfinite([t,g]).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid source event time/command')
    for i,v in enumerate(g):
        # Initial closed posture is not an observed grasp. Arm only after an
        # observed open command; the next closure is a genuine transition.
        if not inside and v>.3:armed=True
        if not inside and armed and v<.1:inside=True;start=i;armed=False
        elif inside and v>.3:
            events.append({'closed_index':start,'release_index':i,'closed_s':float(t[start]),'release_s':float(t[i])})
            inside=False
            armed=True
    if inside:events.append({'closed_index':start,'release_index':len(g)-1,'closed_s':float(t[start]),'release_s':float(t[-1])})
    return events


def evaluate_rollout(config,kind,folder,model,arrays,scene,time_s,grip,report):
    from rssr_model_tools import enabled,COUPLING_TOLERANCE_RAD
    contract=compile_contract(scene['task_id'],scene,scene['bindings'],model_health_required=enabled(config))
    folder=pathlib.Path(folder)
    if config.get('task_contract') and read(config['task_contract'])!=contract:
        raise ValueError('Physical rollout changed frozen target task contract')
    path=folder/'task_contract.json'
    if path.exists() and read(path)!=contract:raise ValueError('Saved physics task contract differs')
    write(path,contract)
    events={side:grasp_events(time_s,values) for side,values in grip.items()}
    data=dict(arrays)
    # Missing grasp is explicitly unsuccessful, not a reason to invent a phase.
    data['source_grasp_intervals']={s:ev[0] if ev else {'closed_s':float(time_s[-1]),'release_s':float(time_s[-1])} for s,ev in events.items()}
    data['articulation_open_fraction']={}
    for o in scene['objects']:
        if o.get('joint'):
            j=o['joint'];qa=model.joint(j['name']).qposadr[0];lo,hi=j['range']
            data['articulation_open_fraction'][j['name']]=(arrays['qpos'][:,qa]-lo)/(hi-lo)
    hard={'robot_table_penetration_le_3mm':report['max_robot_table_penetration_m']<=.003,
          'limited_joints_within_2mrad_or_2mm':report['max_joint_limit_violation_rad']<=.002,
          'solver_no_warnings':not any(report['warnings']),
          'waist_coupling_within_1mrad':report.get('max_waist_sum_residual_rad',0.)<=.001}
    if enabled(config):hard['gripper_coupling_within_2mrad']=report['model_health']['max_coupling_residual_rad']<=COUPLING_TOLERANCE_RAD
    evaluation=evaluate(contract,scene,data,hard)
    write(folder/'task_evaluation.json',evaluation)
    write(folder/'source_grasp_events.json',events)
    np.savez_compressed(folder/'task_state.npz',time_s=arrays['time_s'],
                        **{'articulation_'+k:v for k,v in data['articulation_open_fraction'].items()})
    criteria={**hard,**{p['id']:p['passed'] for p in evaluation['predicates']}}
    # Continuous approach/placement distance guides repair without replacing
    # any predicate. Desired offsets are relative to actual moving containers.
    ids=[o['id'] for o in scene['objects']];pos=arrays['object_position'];b=scene['bindings'];residual=0.;relations=[]
    for i,obj in enumerate(b['targets']):
        reference=b['containers'][min(i,len(b['containers'])-1)] if b['containers'] else b['target_region']
        if reference:
            delta=pos[-1,ids.index(obj)]-pos[-1,ids.index(reference)]
            residual+=float(np.linalg.norm(delta))/.05
            relations.append({'object':obj,'reference':reference,'final_world_delta_m':delta.tolist()})
    for p in evaluation['predicates']:
        if p['passed']:continue
        m=p['measurements']
        residual+=float(m.get('maximum_contact_gap_s',0.))/.1
        residual+=1-float(m.get('fraction',m.get('contact_fraction',1.)))
    source_progress = None
    if kind == 'sim1':
        from rssr_source_progress import progress
        source_progress = progress(evaluation, contract, residual)
        residual = source_progress['residual']
    objective={'passed':evaluation['task_success'],'rank':[*evaluation['rank'],residual],
               'hard_failures':evaluation['hard_failures'],'task_failures':evaluation['failed_predicates'],
               'basis':'frozen task predicates, then continuous numerical repair feedback'}
    if source_progress is not None:
        objective['source_progress'] = source_progress
        objective['basis'] = 'unchanged hard/task failure counts, then normalized measured deficits of failed source predicates'
        from rssr_hard_progress import with_hard_progress
        objective = with_hard_progress(objective, report)
    return {'task_success':evaluation['task_success'],'criteria':criteria,'objective':objective,
            'task_evaluation':evaluation,'task_contract_id':contract['contract_id'],
            'source_grasp_events':events,'final_relations':relations,
            'final_positions_m':dict(zip(ids,pos[-1].tolist())),
            'lift_m':dict(zip(ids,(pos[:,:,2].max(axis=0)-pos[0,:,2]).tolist())),
            'contact_samples':{n:arrays['contacts'][:,i].sum(axis=0).tolist() for i,n in enumerate(ids)},
            'articulation_final_open_fraction':{k:float(v[-1]) for k,v in data['articulation_open_fraction'].items()}}
