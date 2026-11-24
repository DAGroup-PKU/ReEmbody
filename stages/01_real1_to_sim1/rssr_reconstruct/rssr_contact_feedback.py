"""Continuous approach evidence for repair before the first physical contact.

These distances order otherwise equivalent failed candidates. They never turn a
failed contact, task, joint or collision predicate into success.
"""
import copy
import numpy as np


def solid_distance(point, kind, size):
    point=np.asarray(point,float);size=np.asarray(size,float)
    if point.shape!=(3,) or size.shape!=(3,) or not np.isfinite([*point,*size]).all():
        raise ValueError('Finite local geometry coordinates required')
    if kind=='sphere':return float(max(0.,np.linalg.norm(point)-size[0]))
    if kind=='cylinder':
        q=np.array([np.linalg.norm(point[:2])-size[0],abs(point[2])-size[1]])
        return float(np.linalg.norm(np.maximum(q,0.)))
    if kind=='capsule':
        q=point.copy();q[2]-=np.clip(q[2],-size[1],size[1])
        return float(max(0.,np.linalg.norm(q)-size[0]))
    if kind=='box':return float(np.linalg.norm(np.maximum(np.abs(point)-size,0.)))
    raise ValueError('Unsupported analytic collision primitive')


def distances(model,data,objects,hand_points):
    import mujoco
    types={int(getattr(mujoco.mjtGeom,'mjGEOM_'+name.upper())):name
           for name in ('sphere','cylinder','capsule','box')}
    output={hand:{} for hand in hand_points}
    for obj in objects:
        body=model.body(obj['id']).id
        geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==body and model.geom_contype[g]]
        for hand,point in hand_points.items():
            values=[]
            for geom in geoms:
                local=(point-data.geom_xpos[geom])@data.geom_xmat[geom].reshape(3,3)
                kind=types.get(int(model.geom_type[geom]))
                if kind:
                    values.append(solid_distance(local,kind,model.geom_size[geom]))
            if values:
                output[hand][obj['id']]=min(values)
            else:
                # Non-primitive objects retain an explicit coarse oriented-box
                # proxy. It is feedback only, never a collision/contact oracle.
                local=(point-data.xpos[body])@data.xmat[body].reshape(3,3)
                output[hand][obj['id']]=solid_distance(local,'box',np.asarray(obj['size_m'])/2)
    return output


def with_approach_objective(objective,contract,feedback):
    current=copy.deepcopy(objective)
    if current['passed']:
        return current
    required={}
    for predicate in contract['predicates']:
        if predicate['kind']=='lift_with_contact':
            required.setdefault(predicate['object'],set()).add(predicate['hand'])
        elif predicate['kind']=='hand_contact_observed':
            required.setdefault(predicate['object'],set()).update(('left','right'))
        elif predicate['kind']=='inside':
            required.setdefault(predicate['object'],set())
    rows=feedback['state_trace'];evidence=[]
    for obj,hands in required.items():
        choices=[]
        for hand in hands or ('left','right'):
            at_close=[r for r in rows if any(e['hand']==hand and e['phase']=='closed'
                                            for e in r.get('source_events',[]))]
            for row in at_close or rows:
                value=row.get('object_surface_distance_m',{}).get(hand,{}).get(obj)
                if value is None:continue
                value=float(value)
                if not np.isfinite(value) or value<0:raise ValueError('Invalid approach evidence')
                choices.append((value,hand,row['time_s']))
        if choices:
            value,hand,time=min(choices)
            evidence.append({'object':obj,'hand':hand,'source_time_s':time,'surface_distance_m':value})
    residual=sum(row['surface_distance_m'] for row in evidence)/.05
    current['rank'][-1]+=residual
    current['approach_residual_in_5cm_units']=residual
    current['approach_evidence']=evidence
    current['approach_is_success_evidence']=False
    return current
