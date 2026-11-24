"""Deterministic feasibility and repair context; never substitutes for rollout."""
import copy
import pathlib
import numpy as np
from common import read, write


def capabilities(config):
    dest=pathlib.Path(config['import_case'])/'target_capabilities.json'
    if dest.exists():return read(dest)
    import mujoco
    from pilot_physics import build,aperture,LINK,GRIPPER_COMMAND_MARGIN_RAD
    m=build(config,'sim2',dest.parent/'capability_model');d=mujoco.MjData(m)
    result={'basis':'declared model geometry; not hardware calibration','hands':{},
            'source_max_opening_m':.095,'force_limits_changed':False}
    for hand in ('left','right'):
        lo,hi=m.jnt_range[m.joint('astribot_gripper_'+hand+'_joint_L1').id]
        values=[]
        for a in np.linspace(lo+GRIPPER_COMMAND_MARGIN_RAD,hi-GRIPPER_COMMAND_MARGIN_RAD,80):
            for suffix,sign in LINK.items():d.qpos[m.joint('astribot_gripper_'+hand+'_joint_'+suffix).qposadr[0]]=sign*a
            mujoco.mj_forward(m,d);values.append(aperture(m,d,hand))
        result['hands'][hand]={'minimum_commanded_opening_m':float(min(values)),
                               'maximum_commanded_opening_m':float(max(values))}
    write(dest,result);return result


def feasibility(scene,caps):
    issues=[];objects={o['id']:o for o in scene['objects']}
    # Push-only targets do not require an enclosing grasp.
    targets=[] if scene['task_id']=='dm4' else scene['bindings']['targets']
    opening=min(v['maximum_commanded_opening_m'] for v in caps['hands'].values())
    for name in targets:
        o=objects[name]
        if o['shape'] in ('box','sphere') and min(o['size_m'])>opening:
            issues.append({'object':name,'reason':'minimum_extent_exceeds_declared_opening',
                           'minimum_extent_m':min(o['size_m']),'maximum_opening_m':opening})
    return {'issues':issues,'parallel_enclosing_grasp_compatible':not issues,
            'scope':'coarse geometry screening, not a proof of grasp success or impossibility'}


def scene_patch(scene,initial,patch):
    """Only bounded uncertain free-object geometry; task/reference roles fixed."""
    from pilot_scene import scene_edit
    if not isinstance(patch,dict) or set(patch)-{'object_xyz_shift_m','object_yaw_delta_rad','object_size_scale'}:
        raise ValueError('Target scene repair permits position/yaw/isotropic scale only')
    known={o['id']:o for o in initial['objects']}
    for field,values in patch.items():
        if not isinstance(values,dict):raise ValueError('Scene field must map object IDs to values')
        for name,value in values.items():
            o=known.get(name)
            if o is None or o['mobility']!='free' or o['role'] in ('grain','target_region') or name in initial['bindings']['excluded']:
                raise ValueError('Cannot move fixtures, regions, particles or excluded objects')
            if field=='object_size_scale':
                v=np.asarray(value,float)
                if v.shape!=(3,) or not np.isfinite(v).all() or not np.allclose(v,[v[0]]*3,atol=1e-10):
                    raise ValueError('Only isotropic reconstruction scale correction allowed')
    out=scene_edit(scene,initial,patch)
    for o in out['objects']:
        base=known[o['id']]
        if np.linalg.norm(np.asarray(o['position_m'])-base['position_m'])>.10+1e-10:
            raise ValueError('Target scene repair exceeds frozen 10 cm uncertainty')
        ratio=np.asarray(o['size_m'])/base['size_m']
        if np.any(ratio<.9-1e-10) or np.any(ratio>1.1+1e-10):
            raise ValueError('Target scale exceeds frozen +/-10 percent uncertainty')
    if out['bindings']!=initial['bindings']:raise ValueError('Task roles changed')
    return out


def diagnosis(report):
    predicates=report.get('task_evaluation',{}).get('predicates',[])
    failed={p['kind'] for p in predicates if not p['passed']}
    layer='placement'
    if failed & {'lift_with_contact','hand_contact_observed'}:layer='grasp'
    elif 'hold_source_interval' in failed:layer='holding'
    elif 'open_insert_close' in failed:layer='articulation'
    elif 'particle_transfer' in failed:layer='pouring'
    elif failed & {'stable','released'}:layer='release'
    if report.get('task_success'):layer='complete'
    errors={'ik':[],'servo':[]}
    for row in report.get('state_trace',[]):
        for event in row.get('source_events',[]):
            if event['phase']!='closed':continue
            hand=event['hand']
            ik=row.get('ik_tracking',{}).get('hands',{}).get(hand,{}).get('position_error_m')
            servo=row.get('servo_tracking',{}).get(hand,{}).get('position_error_m')
            if ik is not None:errors['ik'].append(ik)
            if servo is not None:errors['servo'].append(servo)
    return {'interaction_phase':layer,'failed_predicate_kinds':sorted(failed),
            'closed_event_median_errors_m':{k:float(np.median(v)) if v else None for k,v in errors.items()},
            'semantics':'diagnostic separation only; large IK error is not proof of geometric unreachability'}


def admission(before,after,scene_changed=False):
    bp={p['id']:p for p in before.get('task_evaluation',{}).get('predicates',[])}
    gained=[p for p in after.get('task_evaluation',{}).get('predicates',[])
            if p['passed'] and p['id'] in bp and not bp[p['id']]['passed']]
    kinds={p['kind'] for p in gained}
    level='task_success' if after.get('task_success') else 'holding' if 'hold_source_interval' in kinds else 'contact' if kinds & {'lift_with_contact','hand_contact_observed'} else 'task_progress' if gained else 'numeric'
    return {'hard_before':before['objective']['hard_failures'],'hard_after':after['objective']['hard_failures'],
            'evidence_policy':'interaction_v16','evidence_level':level,
            'task_before':before['objective']['task_failures'],'task_after':after['objective']['task_failures'],
            'gained_predicates':[p['id'] for p in gained],'scene_unverified':scene_changed}
