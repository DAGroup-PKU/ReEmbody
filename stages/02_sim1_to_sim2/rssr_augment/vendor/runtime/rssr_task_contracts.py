"""Frozen task predicates for the 15 Cobot families, independent of the LLM.

The scene agent supplies object identities once, before target execution. This
compiler fixes required relations and thresholds; a repair cannot redefine its
own success. The evaluator consumes numerical state, contacts and articulation
measurements. This module alone does not implement the required MuJoCo assets.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

from rssr_protocol import digest
from rssr_dataset import TASK_IDS

HELD = {'di1','di2','di3','di4','dm1','dm2'}
STABLE_M = .003
SUPPORT_CONTACT_FRACTION = .8
GEOMETRY_POLICY = 'sphere_analytic_v1'


def compile_contract(task_id, scene, bindings, model_health_required=False):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown task family')
    objects = {o['id']:o for o in scene['objects']}
    if len(objects) != len(scene['objects']):
        raise ValueError('Duplicate scene object IDs')
    allowed = {'targets','containers','excluded','particles','target_region','articulation','stack_hands'}
    if set(bindings)-allowed:
        raise ValueError('Unknown semantic binding')
    groups = {k:list(bindings.get(k,[])) for k in ('targets','containers','excluded','particles')}
    for key, ids in groups.items():
        if len(ids)!=len(set(ids)) or any(i not in objects for i in ids):
            raise ValueError('Unknown/duplicate bound object: '+key)
    if set(groups['targets']) & set(groups['excluded']):
        raise ValueError('Target and excluded objects overlap')
    if set(groups['targets']) & set(groups['containers']):
        raise ValueError('An object cannot contain itself')
    targets, containers = groups['targets'], groups['containers']
    predicates = []
    def add(kind, **fields):
        predicates.append(dict(id=f'{kind}_{len(predicates):02d}',kind=kind,**fields))
    def stable_release(ids):
        add('stable', objects=ids, window_s=1., max_motion_m=STABLE_M)
        add('released', objects=ids, window_s=1.)
    def contain(obj, container):
        add('initial_outside',object=obj,container=container)
        add('inside',object=obj,container=container,clearance_tolerance_m=.002)
        add('object_contact',object=obj,support=container,window_s=.2,minimum_fraction=SUPPORT_CONTACT_FRACTION)
    def count(target_count, container_count):
        if len(targets)!=target_count or len(containers)!=container_count:
            raise ValueError('Binding count disagrees with task '+task_id)
    if task_id in HELD:
        count(1,1)
        add('lift_with_contact',object=containers[0],hand='left',height_m=.02)
        add('hold_source_interval',object=containers[0],hand='left',acquisition_grace_s=.5,
            max_contact_gap_s=.10,minimum_bilateral_fraction=.90)
        add('lift_with_contact',object=targets[0],hand='right',height_m=.025)
        contain(targets[0],containers[0])
        add('table_contact',object=containers[0],window_s=.2,minimum_fraction=SUPPORT_CONTACT_FRACTION)
        stable_release(targets+containers)
    elif task_id == 'di5':
        count(2,1)
        if len(groups['excluded']) != 1:
            raise ValueError('Exactly one unselected cube is required')
        for obj,hand in zip(targets,('left','right')):
            add('lift_with_contact',object=obj,hand=hand,height_m=.025)
            contain(obj,containers[0])
        add('outside',objects=groups['excluded'],container=containers[0])
        stable_release(targets+containers+groups['excluded'])
    elif task_id == 'di6':
        count(2,2)
        for obj,container,hand in zip(targets,containers,('left','right')):
            add('lift_with_contact',object=obj,hand=hand,height_m=.025)
            contain(obj,container)
        stable_release(targets+containers)
    elif task_id == 'di7':
        if not targets or len(containers)!=1 or len(groups['excluded'])!=1:
            raise ValueError('Non-red targets, one red excluded object and one basket required')
        for obj in targets:contain(obj,containers[0])
        add('outside',objects=groups['excluded'],container=containers[0])
        stable_release(targets+containers+groups['excluded'])
    elif task_id == 'dm3':
        if not targets or len(containers)!=1:
            raise ValueError('Bottles and one tray required')
        for obj in targets:
            add('initial_not_upright_supported',object=obj,support=containers[0],maximum_tilt_degrees=10.,
                xy_tolerance_m=.002,z_tolerance_m=.005)
            add('hand_contact_observed',object=obj)
            add('upright',object=obj,maximum_tilt_degrees=10.)
            add('supported_on',object=obj,support=containers[0],xy_tolerance_m=.002,z_tolerance_m=.005)
            add('object_contact',object=obj,support=containers[0],window_s=.2,minimum_fraction=SUPPORT_CONTACT_FRACTION)
        stable_release(targets+containers)
    elif task_id == 'dm4':
        count(2,0)
        region=bindings.get('target_region')
        if region not in objects:raise ValueError('A reconstructed target region is required')
        for obj in targets:
            add('initial_outside_region',object=obj,region=region)
            add('inside_region',object=obj,region=region,tolerance_m=.002)
            add('hand_contact_observed',object=obj)
        stable_release(targets)
    elif task_id == 'dm5':
        count(1,1)  # target is the pouring cup; container is the receiving bowl
        if len(groups['particles'])<12:
            raise ValueError('Pouring needs at least twelve explicit free particles')
        add('particles_initially_inside',objects=groups['particles'],container=targets[0],minimum_fraction=.95)
        add('particle_transfer',objects=groups['particles'],container=containers[0],minimum_fraction=.90)
        add('lift_with_contact',object=containers[0],hand='left',height_m=.02)
        add('hand_contact_observed',object=targets[0])
        stable_release(targets+containers+groups['particles'])
    elif task_id in ('dm6','dm8'):
        count(1,1)
        articulation=bindings.get('articulation')
        if not isinstance(articulation,str) or not articulation:
            raise ValueError('Named drawer/lid articulation required')
        contain(targets[0],containers[0])
        add('open_insert_close',object=targets[0],container=containers[0],joint=articulation,
            open_fraction=.8,closed_fraction=.1,hold_s=.2)
        stable_release(targets+containers)
    elif task_id == 'dm7':
        count(2,0)  # first bound target is the right-hand base; second is the upper block
        region=bindings.get('target_region')
        if region not in objects:raise ValueError('Stacking requires a target region')
        base,upper=targets
        hands=bindings.get('stack_hands',{'base':'right','upper':'left'})
        if set(hands)!={'base','upper'} or set(hands.values())!={'left','right'}:
            raise ValueError('Stack hand assignments must bind distinct left/right hands')
        for obj,hand in [(base,hands['base']),(upper,hands['upper'])]:
            add('lift_with_contact',object=obj,hand=hand,height_m=.025)
        add('inside_region',object=base,region=region,tolerance_m=.002)
        cup_pair=objects[base]['shape']=='cup' and objects[upper]['shape']=='cup'
        if cup_pair:
            add('cup_nested',object=upper,support=base,radial_tolerance_m=.006,
                minimum_depth_m=.015,maximum_tilt_degrees=15.,floor_penetration_tolerance_m=.0005)
        else:
            add('stable_supported_on',object=upper,support=base,center_margin_m=.005,z_tolerance_m=.005)
        add('object_contact',object=upper,support=base,window_s=.2,minimum_fraction=SUPPORT_CONTACT_FRACTION)
        add('base_placed_before_stack',base=base,upper=upper,region=region,
            support_policy='cup_nesting_v2_contact_tolerance' if cup_pair else 'uniform_box_center_margin_v1',
            **({'floor_penetration_tolerance_m':.0005} if cup_pair else {}))
        stable_release(targets)
    for obj in groups['excluded']:
        add('preserved',object=obj,max_displacement_m=.02)
    body={'schema':'rssr.task.contract.v1','task_id':task_id,'scene_sha256':digest(scene),
          'bindings':copy.deepcopy(bindings),'predicates':predicates,
          'geometry_evaluation':GEOMETRY_POLICY,
          'sim1_success_required':False,'objects_forced_allowed':False,
          'hard_constraints':['robot_table_penetration_le_3mm','limited_joints_within_2mrad_or_2mm',
                              'solver_no_warnings','waist_coupling_within_1mrad']}
    if model_health_required:
        body['hard_constraints'].append('gripper_coupling_within_2mrad')
        body['model_fidelity_policy']='v17; simulation assumption, not hardware calibration'
    body['contract_id']=digest(body)
    return body


def evaluate(contract, scene, arrays, hard_constraints):
    """No rendering, learned thresholds, or optimistic handling of missing state."""
    body=dict(contract);identity=body.pop('contract_id')
    if digest(body)!=identity or digest(scene)!=contract['scene_sha256']:
        raise ValueError('Frozen task or scene changed')
    geometry_policy=contract.get('geometry_evaluation','oriented_box_legacy')
    if geometry_policy not in ('oriented_box_legacy',GEOMETRY_POLICY):
        raise ValueError('Unknown frozen geometry evaluator')
    if set(hard_constraints)!=set(contract['hard_constraints']) or any(type(v) is not bool for v in hard_constraints.values()):
        raise ValueError('Every frozen robot constraint needs an explicit boolean measurement')
    objects={o['id']:o for o in scene['objects']};ids=list(objects);index={n:i for i,n in enumerate(ids)}
    t=np.asarray(arrays['time_s'],float);positions=np.asarray(arrays['object_position'],float)
    quats=np.asarray(arrays['object_quaternion'],float)
    contacts=np.asarray(arrays['contacts'])
    pairs=np.asarray(arrays['object_contacts'])
    n=len(t);count=len(ids)
    if n<2 or np.any(np.diff(t)<=0) or positions.shape!=(n,count,3) or quats.shape!=(n,count,4) or contacts.shape!=(n,count,3) or pairs.shape!=(n,count,count):
        raise ValueError('Incomplete or mismatched numerical trace')
    if not np.isfinite(positions).all() or not np.isfinite(quats).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite numerical trace')
    if np.max(np.abs(np.linalg.norm(quats,axis=-1)-1))>.001:
        raise ValueError('Invalid physical quaternion')
    rotations=Rotation.from_quat(quats[:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,count,3,3)
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    def window(seconds):
        if t[-1]-t[0]<seconds:raise ValueError('Trace shorter than required stability/contact interval')
        return t>=t[-1]-seconds
    def corners_local(obj,container):
        oi,ci=index[obj],index[container]
        if analytic_sphere(obj):
            # A sphere's support is rotation invariant. These reference-frame
            # extrema are exact for axial bounds, not physical box corners.
            center=np.einsum('tji,tj->ti',rotations[:,ci],positions[:,oi]-positions[:,ci])
            return center[:,None,:]+signs*objects[obj]['size_m'][0]/2
        corners=signs*np.asarray(objects[obj]['size_m'])/2
        world=np.einsum('tij,kj->tki',rotations[:,oi],corners)+positions[:,oi,None,:]
        return np.einsum('tji,tkj->tki',rotations[:,ci],world-positions[:,ci,None,:])
    def analytic_sphere(obj):
        return geometry_policy==GEOMETRY_POLICY and objects[obj].get('shape') in ('sphere','grain')
    def radial_extent(obj,container,corners):
        if analytic_sphere(obj):
            oi,ci=index[obj],index[container]
            center=np.einsum('tji,tj->ti',rotations[:,ci],positions[:,oi]-positions[:,ci])
            return np.linalg.norm(center[:,:2],axis=-1)+objects[obj]['size_m'][0]/2
        return np.linalg.norm(corners[:,:,:2],axis=-1).max(axis=1)
    def inside(obj,container,tolerance=.002):
        corners=corners_local(obj,container);size=np.array(objects[container]['size_m']);wall=objects[container].get('wall_m',0.)
        horizontal=np.all(np.abs(corners[:,:,:2]) <= size[:2]/2-wall+tolerance,axis=(1,2))
        if objects[container].get('shape') in ('cup','bowl'):
            horizontal=radial_extent(obj,container,corners)<=size[0]/2-wall+tolerance
        return (horizontal &
                (corners[:,:,2].min(axis=1)>=-size[2]/2+wall-.004) &
                (corners[:,:,2].max(axis=1)<=size[2]/2+.003))
    def region_contains(obj,region,tolerance=.002):
        corners=corners_local(obj,region);size=np.asarray(objects[region]['size_m'])
        return np.all(np.abs(corners[:,:,:2])<=size[:2]/2+tolerance,axis=(1,2))
    def supported(obj,support,xy=.002,z=.005):
        corners=corners_local(obj,support);size=np.asarray(objects[support]['size_m'])
        return (np.all(np.abs(corners[:,:,:2])<=size[:2]/2+xy,axis=(1,2)) &
                (np.abs(corners[:,:,2].min(axis=1)-size[2]/2)<=z))
    def stable_support(obj,support,margin=.005,z=.005):
        # Both dm7 cubes are uniform boxes: their body centre is their COM.
        # Contact, release and one-second stability remain separate requirements.
        corners=corners_local(obj,support);size=np.asarray(objects[support]['size_m'])
        oi,bi=index[obj],index[support]
        center=np.einsum('tji,tj->ti',rotations[:,bi],positions[:,oi]-positions[:,bi])
        clearance=size[:2]/2-margin-np.abs(center[:,:2])
        height=z-np.abs(corners[:,:,2].min(axis=1)-size[2]/2)
        return (clearance.min(axis=1)>=0)&(height>=0),clearance,height
    def cup_nesting(upper,base,radial_tolerance=.006,minimum_depth=.015,maximum_tilt=15.,floor_tolerance=0.):
        ui,bi=index[upper],index[base]
        relative=np.einsum('tji,tj->ti',rotations[:,bi],positions[:,ui]-positions[:,bi])
        base_size=np.asarray(objects[base]['size_m']);upper_size=np.asarray(objects[upper]['size_m'])
        interior_radius=base_size[0]/2-objects[base]['wall_m']
        radial_clearance=interior_radius-upper_size[0]/2+radial_tolerance-np.linalg.norm(relative[:,:2],axis=1)
        # A smaller tapered-cup proxy can enter the receiving cup.  The upper
        # bottom must remain above the lower floor, with visible overlap.
        depth=base_size[2]/2-(relative[:,2]-upper_size[2]/2)
        floor_clearance=base_size[2]-objects[base]['wall_m']-depth
        overlap_clearance=depth-minimum_depth
        axes=np.einsum('tij,tj->ti',rotations[:,bi].transpose(0,2,1),rotations[:,ui,:,2])
        tilt=np.degrees(np.arccos(np.clip(axes[:,2],-1,1)))
        tilt_clearance=maximum_tilt-tilt
        # MuJoCo finite contact allows microscopic compression at a supported
        # floor. A zero-penetration inequality rejects even ordinary resting
        # contact. The tolerance is explicit in each new frozen contract;
        # legacy contracts retain strict zero, and independent contact,
        # release, tilt, depth and stability predicates remain required.
        state=(radial_clearance>=0)&(floor_clearance>=-floor_tolerance)&(overlap_clearance>=0)&(tilt_clearance>=0)
        return state,{'radial_clearance_m':radial_clearance,'floor_clearance_m':floor_clearance,
                      'overlap_clearance_m':overlap_clearance,'tilt_clearance_degrees':tilt_clearance}
    def placement_measurements(obj,reference,xy,z=None,initial=False):
        corners=corners_local(obj,reference);size=np.asarray(objects[reference]['size_m'])
        horizontal=size[:2]/2+xy-np.abs(corners[:,:,:2]).max(axis=1)
        violation=np.maximum(0.,-horizontal.min(axis=1))
        use=np.array([0]) if initial else np.flatnonzero(window(.2))
        measured={'reference_frame':'reference_object_local','reference_object':reference,
            'object':obj,'evaluation_window_source_time_s':[float(t[use[0]]),float(t[use[-1]])],
            'final_reference_axes_in_world':rotations[-1,index[reference]].tolist(),
            'final_object_extent_min_m':corners[-1].min(axis=0).tolist(),
            'final_object_extent_max_m':corners[-1].max(axis=0).tolist(),
            'final_signed_xy_clearance_m':horizontal[-1].tolist(),
            'measurement_basis':'same bounding extents, frame, tolerance and window as the frozen predicate'}
        if z is not None:
            gap=corners[:,:,2].min(axis=1)-size[2]/2
            clearance=z-np.abs(gap)
            violation=np.maximum(violation,np.maximum(0.,-clearance))
            measured.update(final_signed_bottom_to_support_top_m=float(gap[-1]),
                            final_signed_support_height_clearance_m=float(clearance[-1]))
        measured['maximum_relation_violation_m']=float(violation[use].max())
        return measured
    results=[]
    for predicate in contract['predicates']:
        kind=predicate['kind'];obj=predicate.get('object');oi=index[obj] if obj else None
        measured={};passed=False
        if kind in ('initial_outside','inside'):
            c=predicate['container'];corners=corners_local(obj,c);size=np.array(objects[c]['size_m']);wall=objects[c].get('wall_m',0.)
            horizontal=size[:2]/2-wall+.002-np.abs(corners[:,:,:2]).max(axis=1)
            radial=(size[0]/2-wall+.002-radial_extent(obj,c,corners)) if objects[c].get('shape') in ('cup','bowl') else horizontal.min(axis=1)
            bottom=corners[:,:,2].min(axis=1)-(-size[2]/2+wall-.004)
            top=size[2]/2+.003-corners[:,:,2].max(axis=1)
            use=np.array([0]) if kind=='initial_outside' else np.flatnonzero(window(.2))
            violation=np.maximum(0.,-np.minimum.reduce([radial,bottom,top]))
            measured={'reference_frame':'container_local','container':c,
                'final_object_extent_min_m':corners[-1].min(axis=0).tolist(),'final_object_extent_max_m':corners[-1].max(axis=0).tolist(),
                'final_signed_xy_clearance_m':horizontal[-1].tolist(),'final_signed_bottom_clearance_m':float(bottom[-1]),
                'final_signed_top_clearance_m':float(top[-1]),'maximum_relation_violation_m':float(violation[use].max()),
                'final_container_up_axis_world':rotations[-1,index[c],:,2].tolist(),
                'final_container_tilt_degrees':float(np.degrees(np.arccos(np.clip(rotations[-1,index[c],2,2],-1,1)))),
                'final_below_exterior_floor':bool(corners[-1,:,2].max()<=-size[2]/2+.002)}
            states=inside(obj,predicate['container'],predicate.get('clearance_tolerance_m',.002))
            passed=not bool(states[0]) if kind=='initial_outside' else bool(states[window(.2)].all())
        elif kind=='outside':
            passed=all(not bool(inside(o,predicate['container'])[window(.2)].any()) for o in predicate['objects'])
        elif kind in ('inside_region','initial_outside_region'):
            states=region_contains(obj,predicate['region'],predicate.get('tolerance_m',.002))
            passed=not bool(states[0]) if kind.startswith('initial_') else bool(states[window(.2)].all())
            measured=placement_measurements(obj,predicate['region'],predicate.get('tolerance_m',.002),
                                            initial=kind.startswith('initial_'))
        elif kind=='supported_on':
            passed=bool(supported(obj,predicate['support'],predicate['xy_tolerance_m'],predicate['z_tolerance_m'])[window(.2)].all())
            measured=placement_measurements(obj,predicate['support'],predicate['xy_tolerance_m'],predicate['z_tolerance_m'])
        elif kind=='stable_supported_on':
            states,clearance,height=stable_support(obj,predicate['support'],predicate['center_margin_m'],predicate['z_tolerance_m'])
            use=window(.2);passed=bool(states[use].all())
            measured={'support_policy':'uniform_box_center_margin_v1',
                'minimum_center_margin_clearance_m':float(clearance[use].min()),
                'minimum_height_clearance_m':float(height[use].min()),
                'maximum_relation_violation_m':float(max(0.,-clearance[use].min(),-height[use].min())),
                'full_footprint_alignment_diagnostic':placement_measurements(obj,predicate['support'],.002,.005),
                'contact_release_and_stability_required_separately':True}
        elif kind=='cup_nested':
            states,margins=cup_nesting(obj,predicate['support'],predicate['radial_tolerance_m'],
                predicate['minimum_depth_m'],predicate['maximum_tilt_degrees'],predicate.get('floor_penetration_tolerance_m',0.))
            use=window(.2);passed=bool(states[use].all())
            measured={'support_policy':'cup_nesting_v1',
                'minimum_radial_clearance_m':float(margins['radial_clearance_m'][use].min()),
                'minimum_floor_clearance_m':float(margins['floor_clearance_m'][use].min()),
                'floor_penetration_tolerance_m':predicate.get('floor_penetration_tolerance_m',0.),
                'floor_penetration_excess_m':float(max(0.,-margins['floor_clearance_m'][use].min()-predicate.get('floor_penetration_tolerance_m',0.))),
                'minimum_overlap_clearance_m':float(margins['overlap_clearance_m'][use].min()),
                'maximum_relation_violation_m':float(max(0.,-margins['radial_clearance_m'][use].min(),
                    -margins['floor_clearance_m'][use].min()-predicate.get('floor_penetration_tolerance_m',0.),-margins['overlap_clearance_m'][use].min()))}
        elif kind=='upright':
            tilt=np.degrees(np.arccos(np.clip(rotations[:,oi,2,2],-1,1)))
            measured={'maximum_final_tilt_degrees':float(tilt[window(.2)].max())}
            passed=measured['maximum_final_tilt_degrees']<=predicate['maximum_tilt_degrees']
        elif kind=='initial_not_upright_supported':
            initial_tilt=float(np.degrees(np.arccos(np.clip(rotations[0,oi,2,2],-1,1))))
            initial_support=bool(supported(obj,predicate['support'],predicate['xy_tolerance_m'],predicate['z_tolerance_m'])[0])
            measured={'initial_tilt_degrees':initial_tilt,'initially_supported_on_target':initial_support}
            passed=not (initial_support and initial_tilt<=predicate['maximum_tilt_degrees'])
        elif kind=='lift_with_contact':
            lift=positions[:,oi,2]-positions[0,oi,2];hand=('left','right').index(predicate['hand'])
            hand_contact=contacts[:,oi,hand].astype(bool)
            measured={'max_lift_m':float(lift.max()),
                      'max_lift_with_hand_contact_m':float(lift[hand_contact].max()) if hand_contact.any() else 0.0}
            passed=bool(np.any((lift>=predicate['height_m']) & hand_contact))
        elif kind=='hand_contact_observed':
            passed=bool(contacts[:,oi,:2].any())
        elif kind in ('table_contact','object_contact'):
            hits=contacts[:,oi,2] if kind=='table_contact' else pairs[:,oi,index[predicate['support']]]
            measured={'contact_fraction':float(np.mean(hits[window(predicate['window_s'])].astype(bool)))}
            passed=measured['contact_fraction']>=predicate['minimum_fraction']
        elif kind=='stable':
            subset=positions[window(predicate['window_s'])][:,[index[o] for o in predicate['objects']]]
            measured={'maximum_motion_m':float(np.linalg.norm(np.ptp(subset,axis=0),axis=-1).max())}
            passed=measured['maximum_motion_m']<=predicate['max_motion_m']
        elif kind=='released':
            passed=not bool(contacts[window(predicate['window_s'])][:,[index[o] for o in predicate['objects']],:2].any())
        elif kind=='preserved':
            measured={'maximum_displacement_m':float(np.linalg.norm(positions[:,oi]-positions[0,oi],axis=-1).max())}
            passed=measured['maximum_displacement_m']<=predicate['max_displacement_m']
        elif kind in ('particles_initially_inside','particle_transfer'):
            states=np.stack([inside(o,predicate['container']) for o in predicate['objects']],axis=1)
            fraction=float(states[0].mean()) if kind=='particles_initially_inside' else float(states[window(.2)].all(axis=0).mean())
            measured={'fraction':fraction};passed=fraction>=predicate['minimum_fraction']
        elif kind=='hold_source_interval':
            hand=('left','right').index(predicate['hand']);command=arrays['source_grasp_intervals'][predicate['hand']]
            start=float(command['closed_s'])+predicate['acquisition_grace_s'];end=float(command['release_s'])
            mask=(t>=start)&(t<end)
            if end-start<.2 or t[-1]<end or not mask.any():
                measured={'phase_complete':False}
            else:
                from pilot_physics import _true_intervals
                gaps=_true_intervals(t[mask],~contacts[mask,oi,hand].astype(bool))
                max_gap=max((g['duration_s'] for g in gaps),default=0.)
                bilateral=np.asarray(arrays['finger_contact_sides'])[mask,oi,hand].all(axis=-1).mean()
                measured={'maximum_contact_gap_s':float(max_gap),'bilateral_fraction':float(bilateral)}
                passed=max_gap<=predicate['max_contact_gap_s']+1e-8 and bilateral>=predicate['minimum_bilateral_fraction']
                from rssr_hold_feedback import timing_diagnostics
                measured.update(timing_diagnostics(t,mask,contacts[:,oi,hand],
                    np.asarray(arrays['finger_contact_sides'])[:,oi,hand],predicate,
                    np.asarray(arrays['gripper_opening'])[:,hand] if 'gripper_opening' in arrays else None))
                measured['hold_contact_gap_excess_s']=max(0.,float(max_gap)-predicate['max_contact_gap_s'])
                measured['hold_bilateral_fraction_deficit']=max(0.,predicate['minimum_bilateral_fraction']-float(bilateral))
        elif kind=='open_insert_close':
            fraction=np.asarray(arrays['articulation_open_fraction'][predicate['joint']],float)
            if fraction.shape!=(n,) or not np.isfinite(fraction).all():raise ValueError('Missing articulation trace')
            states=inside(obj,predicate['container'])
            insertions=np.flatnonzero(states & ~np.r_[False,states[:-1]])
            first=int(insertions[0]) if len(insertions) else -1
            before=np.any(fraction[:max(0,first)]>=predicate['open_fraction'])
            during=first>=0 and fraction[first]>=predicate['open_fraction']
            final=bool(np.all(fraction[window(predicate['hold_s'])]<=predicate['closed_fraction']))
            measured={'first_insertion_s':float(t[first]) if first>=0 else None,'opened_before_insertion':bool(before),
                      'open_during_insertion':bool(during),'closed_at_end':final,
                      'maximum_open_fraction_before_insertion':float(fraction[:first].max()) if first>0 else None,
                      'open_fraction_at_insertion':float(fraction[first]) if first>=0 else None,
                      'maximum_final_open_fraction':float(fraction[window(predicate['hold_s'])].max())}
            passed=first>0 and before and during and final
        elif kind=='base_placed_before_stack':
            base,upper=predicate['base'],predicate['upper'];bi,ui=index[base],index[upper]
            placement=region_contains(base,predicate['region']) & contacts[:,bi,2].astype(bool)
            support_states=(cup_nesting(upper,base,floor_tolerance=predicate.get('floor_penetration_tolerance_m',0.))[0] if predicate.get('support_policy') in ('cup_nesting_v1','cup_nesting_v2_contact_tolerance')
                else stable_support(upper,base)[0] if predicate.get('support_policy')=='uniform_box_center_margin_v1'
                else supported(upper,base))
            stack=support_states & pairs[:,ui,bi].astype(bool)
            p=np.flatnonzero(placement);s=np.flatnonzero(stack)
            passed=bool(len(p) and len(s) and p[0]<s[0])
            measured={'base_placement_observed':bool(len(p)),'stack_contact_observed':bool(len(s)),
                'first_base_placed_s':float(t[p[0]]) if len(p) else None,
                'first_upper_supported_contact_s':float(t[s[0]]) if len(s) else None,
                'order_observable':bool(len(p) and len(s)),
                'basis':'first qualifying simulator states; a missing event is not a reversed order'}
        else:
            raise ValueError('Unknown task predicate '+kind)
        results.append({'id':predicate['id'],'kind':kind,'passed':bool(passed),'measurements':measured})
    hard_failed=[name for name,passed in hard_constraints.items() if not passed]
    failed=[p['id'] for p in results if not p['passed']]
    return {'schema':'rssr.task.evaluation.v1','contract_id':identity,'task_id':contract['task_id'],
            'geometry_evaluation':geometry_policy,
            'task_success':not hard_failed and not failed,'hard_constraints':hard_constraints,
            'predicates':results,'failed_predicates':failed,'hard_failures':hard_failed,
            'rank':[len(hard_failed),len(failed)],'state_source':'simulator_numeric_trace','render_calls':0}
