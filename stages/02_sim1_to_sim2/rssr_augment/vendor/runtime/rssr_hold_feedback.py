"""Observed holding/contact timing; no state modification or repair recipe."""
import numpy as np


def timing_diagnostics(t,mask,hand_contact,finger_sides,predicate,opening=None):
    from pilot_physics import _true_intervals
    times=np.asarray(t,float)[mask]
    hand=np.asarray(hand_contact,bool)[mask]
    sides=np.asarray(finger_sides,bool)[mask]
    if not len(times) or sides.shape!=(len(times),2):
        raise ValueError('Complete two-sided holding observations required')
    bilateral=sides.all(axis=1);any_pad=sides.any(axis=1)
    acquired=np.flatnonzero(bilateral)
    loss=np.flatnonzero(~bilateral & (np.arange(len(times))>acquired[0])) if len(acquired) else []
    first_loss=int(loss[0]) if len(loss) else None
    gaps=sorted(_true_intervals(times,~bilateral),key=lambda r:r['duration_s'],reverse=True)
    event_indices={0:'hold_window_start'}
    if gaps:
        # Inspect the dominant interruption rather than a one-sample flicker.
        # The first observed loss remains separately reported above the trace.
        onset=int(np.argmin(abs(times-gaps[0]['start_s'])))
        if onset>0:event_indices.setdefault(onset-1,'before_longest_bilateral_gap')
        event_indices.setdefault(onset,'longest_bilateral_gap_start')
    result={
        'phase_complete':True,'sampled_window_s':[float(times[0]),float(times[-1])],
        'contact_gap_limit_s':predicate['max_contact_gap_s'],
        'minimum_bilateral_fraction':predicate['minimum_bilateral_fraction'],
        'any_hand_contact_fraction':float(hand.mean()),
        'finger_side_contact_fractions':sides.mean(axis=0).tolist(),
        'single_finger_side_fraction':float((sides.sum(axis=1)==1).mean()),
        'other_hand_parts_only_fraction':float((hand & ~any_pad).mean()),
        'bilateral_contact_observed':bool(len(acquired)),
        'first_bilateral_contact_s':float(times[acquired[0]]) if len(acquired) else None,
        'first_bilateral_loss_after_acquisition_s':float(times[first_loss]) if first_loss is not None else None,
        'maximum_bilateral_gap_s':max((g['duration_s'] for g in gaps),default=0.),
        'longest_bilateral_gaps':gaps[:2],
        'diagnostic_time_queries':[{'time_s':float(times[i]),'reason':reason}
                                   for i,reason in sorted(event_indices.items())],
        'basis':'saved contact samples inside the unchanged source-commanded hold window; contact loss is an observation, not a causal diagnosis'}
    if opening is not None:
        values=np.asarray(opening,float)[mask]
        if values.shape!=(len(times),) or not np.isfinite(values).all():
            raise ValueError('Invalid observed holding aperture')
        result['observed_opening_range_m']=[float(values.min()),float(values.max())]
    return result


def event_queries(report,contract):
    definitions={p['id']:p for p in contract.get('predicates',[])}
    queries=[]
    for row in report.get('task_evaluation',{}).get('predicates',[]):
        if row['passed']:continue
        definition=definitions.get(row['id'])
        if definition is None:continue
        if row['kind']=='hold_source_interval':
            points=row.get('measurements',{}).get('diagnostic_time_queries',[])
        elif row['kind']=='lift_with_contact':
            # These are already sampled source-closure states, not new rollouts
            # or invented task events. Inspect geometry even without contact.
            points=[{'time_s':event['closed_s'],'reason':'failed_lift_source_closure'}
                    for event in report.get('source_grasp_events',{}).get(definition['hand'],[])]
        else:continue
        for point in points:
            queries.append(dict(point,predicate=row['id'],object=definition['object'],hand=definition['hand']))
    return queries


def contact_geometry(model,data,obj,hand,closing_axis):
    """Read recomputed contacts at saved qpos; no stepping or force inference."""
    axis=np.asarray(closing_axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-5:
        raise ValueError('Unit closing axis required')
    root=model.body(obj).id;descendants={root}
    for body in range(model.nbody):
        parent=body
        while parent and parent not in descendants:parent=int(model.body_parentid[parent])
        if parent in descendants:descendants.add(body)
    pairs={};count=0
    for ct in data.contact:
        if ct.dist>0:continue
        geoms=[int(ct.geom1),int(ct.geom2)]
        names=[model.geom(g).name or '' for g in geoms]
        belongs=[int(model.geom_bodyid[g]) in descendants for g in geoms]
        if sum(belongs)!=1:continue
        robot=1-belongs.index(True);body=model.body(int(model.geom_bodyid[geoms[robot]])).name or ''
        if hand not in names[robot] and hand not in body:continue
        if 'gripper' not in names[robot] and not body.endswith(('link7','link8')):continue
        normal=np.asarray(ct.frame[:3],float);count+=1
        key=tuple(names);value={'geoms':names,'point_world_m':np.asarray(ct.pos).tolist(),
            'normal_world':normal.tolist(),'penetration_m':float(-ct.dist),
            'absolute_normal_closing_axis_alignment':float(abs(np.dot(normal,axis)))}
        if key not in pairs or value['penetration_m']>pairs[key]['penetration_m']:pairs[key]=value
    ordered=sorted(pairs.values(),key=lambda x:-x['penetration_m'])
    return {'object':obj,'hand':hand,'closing_axis_world':axis.tolist(),
            'closing_axis_basis':'declared grasp-frame local x aperture axis of this target backend',
            'contact_count':count,'distinct_geom_pairs':len(ordered),'contacts':ordered[:8],
            'scope':'unsigned normal/closing-axis alignment is geometry, not evidence of force closure or a prescribed rotation'}
