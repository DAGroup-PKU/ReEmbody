"""Distinguish IK residuals from physical servo error at native action events.

These are diagnostic quantities, not alternate task success criteria. The
solver's tool site is explicitly distinct from the finger grasp centre.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from common import sha


def load_tracking(plan):
    if not plan or not plan.get('goal_tracking'):
        return None  # Older development artifacts have no per-frame IK targets.
    entry=plan['goal_tracking']
    if sha(entry['path'])!=entry['sha256']:
        raise ValueError('IK target/FK evidence changed')
    with np.load(entry['path'],allow_pickle=False) as z:
        tracks={k:z[k] for k in z.files}
    times=tracks['time_s']
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid IK evidence timestamps')
    for side in ('left','right'):
        for name,shape in [('target_position',(len(times),3)),('planned_site_position',(len(times),3)),
                           ('target_rotation',(len(times),3,3)),('planned_site_rotation',(len(times),3,3))]:
            value=tracks[side+'_'+name]
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Invalid IK evidence '+side+'_'+name)
    return tracks


def ik_residual_at(tracks,time_s):
    if tracks is None:
        return {'available':False,'reason':'per-frame IK evidence was not saved by this revision'}
    index=int(np.argmin(abs(tracks['time_s']-time_s)))
    result={'available':True,'source_index':index,'source_time_s':float(tracks['time_s'][index]),
            'time_difference_from_physics_s':float(tracks['time_s'][index]-time_s),'hands':{}}
    for side in ('left','right'):
        target=tracks[side+'_target_position'][index]
        planned=tracks[side+'_planned_site_position'][index]
        rotation=tracks[side+'_target_rotation'][index]@tracks[side+'_planned_site_rotation'][index].T
        result['hands'][side]={'site':'arm_'+side+'_tool','target_position_m':target.tolist(),
            'planned_position_m':planned.tolist(),'target_minus_planned_m':(target-planned).tolist(),
            'position_error_m':float(np.linalg.norm(target-planned)),
            'orientation_error_degrees':float(np.rad2deg(Rotation.from_matrix(rotation).magnitude()))}
    return result


def servo_residual(model,actual,nominal,commands,time_s):
    import mujoco
    times=commands['time_s'];index=int(np.clip(np.searchsorted(times,time_s,side='right')-1,0,len(times)-2))
    alpha=float(np.clip((time_s-times[index])/(times[index+1]-times[index]),0,1))
    nominal.qpos[:]=(1-alpha)*commands['qpos'][index]+alpha*commands['qpos'][index+1]
    nominal.qvel[:]=0
    mujoco.mj_forward(model,nominal)  # Read-only FK, never a dynamics state update.
    result={}
    for side in ('left','right'):
        site=model.site('arm_'+side+'_tool').id
        desired=nominal.site_xpos[site];observed=actual.site_xpos[site]
        result[side]={'site':'arm_'+side+'_tool','commanded_position_m':desired.tolist(),
                     'actual_position_m':observed.tolist(),
                     'command_minus_actual_m':(desired-observed).tolist(),
                     'position_error_m':float(np.linalg.norm(desired-observed))}
    return result
