"""Agent-invoked local camera search using saved states and actual ID renders.

This is a fixed numerical tool, not a learned camera profile or a visual
verifier. It never steps dynamics, changes the scene or consumes API calls.
"""
from copy import deepcopy
import time
import numpy as np
from scipy.optimize import minimize
from camera_visibility import (raster_metrics, diagnose, MAX_NORMALIZED_BOX_ERROR,
    MIN_INITIAL_WORKSPACE_COVERAGE, MAX_INITIAL_ROBOT_COVERAGE)
from pilot_camera_agent import (CAMERAS, apply_camera_edit, diagnose_camera_profiles,
                               _id_renderer, _release_renderer, camera_scene_option)
from camera_clearance import inspect as camera_clearance
from camera_interaction import capture as contact_context, inspect as interaction_visibility


def _loss(projection_error, raster_row, x):
    extent=[c['normalized_visible_extent_error'] for c in raster_row['checks']]
    extent_error=float(np.median(extent)) if extent else 2.
    missing=sum(reason.startswith(('reference_object_not_raster_visible',
                                  'reference_object_heavily_occluded')) for reason in raster_row['errors'])
    workspace=raster_row['workspace_pixel_fraction']
    value=(10*max(0.,projection_error-MAX_NORMALIZED_BOX_ERROR)+
           10*max(0.,extent_error-MAX_NORMALIZED_BOX_ERROR)+
           10*max(0.,MIN_INITIAL_WORKSPACE_COVERAGE-workspace)+
           10*max(0.,raster_row['robot_pixel_fraction']-MAX_INITIAL_ROBOT_COVERAGE)+
           2*missing+.25*projection_error+.5*extent_error+
           .3*max(0.,.75-workspace)+.002*float(np.linalg.norm(x)))
    # The .75 preference is a soft search term, not a new acceptance rule.
    return float(value),extent_error


def _sample_frames(mapping):
    """Bounded saved-state samples, including both endpoints; no interpolation."""
    if len(mapping) < 1:
        raise ValueError('No saved camera frame mapping')
    return sorted({0, (len(mapping)-1)//2, len(mapping)-1})


def _temporal_loss(metrics):
    # These are soft search preferences only. No pixel equality, forced
    # object visibility or new acceptance thresholds are imposed after t=0.
    rows = []
    for frame, metric in metrics:
        width, height = metric['image_size_wh']
        workspace = min(1., metric['table_pixel_fraction'] +
                        sum(o['pixels'] for o in metric['objects'].values())/(width*height))
        robot = metric['robot_pixel_fraction']
        value = (5*max(0., MIN_INITIAL_WORKSPACE_COVERAGE-workspace) +
                 5*max(0., robot-MAX_INITIAL_ROBOT_COVERAGE) +
                 2*robot + .3*max(0., .75-workspace))
        rows.append({'frame': frame, 'workspace_fraction': workspace,
                     'robot_fraction': robot, 'soft_loss': float(value)})
    values = [r['soft_loss'] for r in rows]
    return (float(.5*max(values)+.5*np.mean(values)) if values else 0.), rows


def _search(evaluate,bounds):
    baseline=evaluate(np.zeros(7));best=deepcopy(baseline);attempts=1
    def objective(x):
        nonlocal best,attempts
        row=evaluate(x);attempts+=1
        if row is None:return 100.
        if row['objective']<best['objective']:best=deepcopy(row)
        return row['objective']
    starts=[np.zeros(7)]
    for axis,sign in [(3,1),(3,-1),(4,1),(4,-1)]:
        x=np.zeros(7);x[axis]=sign*.5;starts.append(x)
    for initial in starts:
        minimize(objective,initial,method='Powell',bounds=bounds,
                 options={'maxfev':180,'xtol':.015,'ftol':.0001})
    return baseline,best,attempts


def fit_visibility(profiles,origin,observations,kind,view,context,table_frames=None):
    import mujoco
    from camera_agent import apply_profile
    if kind not in profiles or view not in CAMERAS:
        raise ValueError('Unknown camera to fit')
    if not isinstance(context,dict) or not {'loaded','landmarks','frame_mapping'}<=set(context):
        raise ValueError('Visibility fit requires saved physical state and current image observations')
    if not any(r['kind']==kind and r['view']==view and r['source_frame_index']==0 and
               ('bbox_xyxy' in r or 'pixel_xy' in r) for r in observations):
        raise ValueError('Visibility fit cannot invent a first-frame correspondence')
    model,data,rollout,mapping=context['loaded'][kind]
    frames = _sample_frames(mapping)
    if any(not 0<=int(mapping[frame])<len(rollout['qpos']) for frame in frames):
        raise ValueError('Missing mapped saved state for visibility fit')
    saved=(model.cam_pos.copy(),model.cam_quat.copy(),model.cam_fovy.copy(),
                               data.qpos.copy(),data.qvel.copy(),float(data.time))
    shift=.15 if view=='head' else .04
    scale=np.array([shift]*3+[30.]*3+[20.])
    fov=profiles[kind][CAMERAS[view]]['fovy_degrees']
    bounds=[(-1.,1.)]*6+[(max(-1.,(20.-fov)/20.),min(1.,(110.-fov)/20.))]
    objects=list(context['landmarks']['views'][view])
    contact_order=context.get('scene_object_ids')
    if 'contacts' in rollout and (not isinstance(contact_order,list) or len(contact_order)!=len(set(contact_order)) or set(contact_order)!=set(objects)):
        raise ValueError('Contact-aware fitting requires the immutable scene object order, not landmark sorting')
    renderer=None;render_calls=0;cache_hits=0;cache={};frame_states=[];started=time.monotonic()
    scene_option=camera_scene_option()
    def evaluate(x):
        nonlocal render_calls,cache_hits
        key=tuple(float(v) for v in x)
        if key in cache:
            cache_hits+=1
            return cache[key]
        edit={'translation_camera_m':(x[:3]*scale[:3]).tolist(),
              'rotation_camera_xyz_degrees':(x[3:6]*30.).tolist(),
              'fovy_delta_degrees':float(x[6]*20.)}
        try:
            candidate=(deepcopy(profiles) if np.max(np.abs(x))<1e-10 else
                       apply_camera_edit(profiles,origin,kind,view,edit,table_frames))
        except ValueError:
            cache[key]=None
            return None
        report=diagnose_camera_profiles(candidate,origin,observations,context['frame_mapping'],table_frames)
        projection=next(r for r in report['views'] if r['kind']==kind and r['view']==view)
        error=projection['median_normalized_projection_error']
        projection_error=float(error) if error is not None else 2.
        apply_profile(model,candidate[kind])
        metrics=[]
        for frame,frame_data,contact in frame_states:
            # Only fixed camera parameters vary during this search. Saved
            # body poses, contacts and geometry remain identical across all
            # candidates; recomputing forward dynamics is unnecessary.
            mujoco.mj_camlight(model,frame_data)
            renderer.update_scene(frame_data,camera=CAMERAS[view],scene_option=scene_option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            metric=raster_metrics(model,renderer.render(),objects);render_calls+=1
            metric['camera_clearance']=camera_clearance(model,frame_data,CAMERAS[view])
            metric['interaction_visibility']=interaction_visibility(model,frame_data,CAMERAS[view],contact,metric)
            metrics.append((frame,metric))
        metric=metrics[0][1]
        raster=diagnose({kind:{0:{view:{'raster_visibility':metric}}}},context['landmarks'])
        row=raster['views'][0];value,extent=_loss(projection_error,row,x)
        temporal,temporal_rows=_temporal_loss(metrics[1:])
        value+=temporal
        interactions=[m['interaction_visibility'] for _,m in metrics]
        value+=sum(r['soft_loss'] for r in interactions)
        interaction_ok=all(not r['available'] or r['passed'] for r in interactions)
        clearances=[m['camera_clearance'] for _,m in metrics]
        for c in clearances:
            distance=c['minimum_signed_distance_m']
            value+=100 if distance is None else 100*max(0.,c['minimum_required_clearance_m']-distance)/.01
        initial_context=context.get('initial_orientation')
        orientation_ok=True
        if initial_context is not None and kind=='sim2':
            from camera_initial_orientation import comparison,residual as orientation_residual
            from pilot_camera_agent import _spec
            orientation_ok=comparison(candidate,initial_context,view)['passed']
            value+=10*float(np.linalg.norm(orientation_residual(candidate,initial_context,view,
                                 _spec(candidate[kind][CAMERAS[view]])[1])))
        result={'objective':value,'projection_error':projection_error,'raster_extent_error':extent,
                'workspace_fraction':row['workspace_pixel_fraction'],'robot_fraction':metric['robot_pixel_fraction'],
                'raster_errors':row['errors'],'initial_orientation_passed':orientation_ok,
                'selected_view_numeric_passed':bool(projection['projection_passed'] and raster['passed'] and orientation_ok and interaction_ok and all(c['passed'] for c in clearances)),
                'interaction_visibility_by_frame':[{'frame':f,**m['interaction_visibility']} for f,m in metrics],
                'camera_clearance_by_frame':[{'frame':f,**m['camera_clearance']} for f,m in metrics],
                'temporal_soft_loss':temporal,'temporal_samples':temporal_rows,
                'edit':edit}
        cache[key]=result
        return result
    try:
        renderer=_id_renderer(model,360,480);renderer.enable_segmentation_rendering()
        for frame in frames:
            index=int(mapping[frame])
            frame_data=mujoco.MjData(model)
            frame_data.qpos[:]=rollout['qpos'][index];frame_data.qvel[:]=rollout['qvel'][index]
            if 'time_s' in rollout:frame_data.time=float(rollout['time_s'][index])
            mujoco.mj_forward(model,frame_data)
            frame_states.append((frame,frame_data,contact_context(model,frame_data,rollout,index,contact_order or objects,view)))
        baseline,best,attempts=_search(evaluate,bounds)
        if best['objective']>=baseline['objective']-1e-7:
            raise ValueError('No bounded visibility improvement; use other camera evidence or edit')
        candidate=apply_camera_edit(profiles,origin,kind,view,best['edit'],table_frames)
        if not best['initial_orientation_passed']:
            raise ValueError('Visibility fit still violates initial orientation; align the source optical direction first')
        return candidate,{'schema':'rssr.camera_visibility_fit.receipt.v1','kind':kind,'view':view,
                          'baseline':baseline,'best':best,'applied_edit':best['edit'],
                          'search_attempts':attempts,'render_calls':render_calls,'max_search_attempts':901,
                          'sampled_video_frames':frames,'max_id_renders':901*len(frames),
                          'exact_candidate_cache_hits':cache_hits,'temporal_loss_is_soft_only':True,
                          'saved_state_forward_evaluations':len(frame_states),
                          'candidate_camera_update':'mj_camlight_on_saved_state',
                          'wall_seconds':time.monotonic()-started,'api_requests':0,'dynamics_steps':0,
                          'historical_profile_used':False,'physics_and_actions_modified':False,
                          'visual_verification_required':True}
    finally:
        try:
            if renderer is not None:_release_renderer(renderer)
        finally:
            model.cam_pos[:],model.cam_quat[:],model.cam_fovy[:]=saved[:3]
            data.qpos[:],data.qvel[:],data.time=saved[3:]
            mujoco.mj_forward(model,data)
