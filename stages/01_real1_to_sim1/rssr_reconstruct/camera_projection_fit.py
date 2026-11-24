"""Agent-invoked bounded camera fit from this case's observed correspondences.

Numerical fitting is a fixed tool, not an imported profile or camera verifier.
It does not render, move objects, change actions, or claim visual correctness.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from camera_landmarks import project_visible_bounds
from pilot_camera_agent import CAMERAS, _spec, apply_camera_edit


def fit_projection(profiles,origin,observations,kind,view,table_frames=None,*,initial_context=None):
    if kind not in profiles or view not in CAMERAS:
        raise ValueError('Unknown camera to fit')
    rows=[r for r in observations if r['kind']==kind and r['view']==view
          and ('bbox_xyxy' in r or 'pixel_xy' in r)]
    if not rows:raise ValueError('No observed correspondences; camera fit cannot invent landmarks')
    p,r,f=_spec(profiles[kind][CAMERAS[view]])
    shift=.15 if view=='head' else .04
    low=np.r_[[-shift]*3,[-180.]*3,max(-20.,20.-f)]
    high=np.r_[[shift]*3,[180.]*3,min(20.,110.-f)]
    started=time.monotonic();evaluations=0
    orientation_guard = initial_context is not None and kind == 'sim2'
    def residual(x,regularize=True):
        nonlocal evaluations
        evaluations+=1
        pos=p+r@x[:3];rot=r@Rotation.from_euler('xyz',x[3:6],degrees=True).as_matrix()
        values=[]
        for row in rows:
            width,height=row['image_size_wh'];scale=max(width,height)
            focal=height/(2*np.tan(np.deg2rad(f+x[6])/2))
            if 'bbox_xyxy' in row:
                corners=(np.asarray(row['bounds_in_parent_m'])-pos)@rot
                box=project_visible_bounds(corners,width,height,focal)
                values.extend([2.]*4 if box is None else (box-np.asarray(row['bbox_xyxy']))/scale)
            else:
                point=(np.asarray(row['point_in_parent_m'])-pos)@rot;depth=-point[2]
                uv=np.array([width/2+focal*point[0]/max(depth,.005),height/2-focal*point[1]/max(depth,.005)])
                values.extend((uv-np.asarray(row['pixel_xy']))/scale)
                values.append(max(0.,.005-depth)*10.)
        if orientation_guard:
            from camera_initial_orientation import residual as orientation_residual
            values.extend(orientation_residual(profiles,initial_context,view,rot))
        if regularize:
            # Resolve weakly observed directions toward the declared mount.
            # No real hardware calibration is inferred from this prior.
            values.extend(.005*x[:3]/shift)
            values.extend(.0005*x[3:6]/180.)
            values.append(.001*x[6]/20.)
        return np.asarray(values)
    baseline=float(np.linalg.norm(residual(np.zeros(7),False)))
    best=None;starts=[]
    # Fixed, case-independent orientation hypotheses; all real-valued edits
    # still obey the same per-step and cumulative bounds as manual primitives.
    for axis,angle in [(2,0),(2,90),(2,-90),(2,180),(0,90),(0,-90),(1,90),(1,-90)]:
        x=np.zeros(7);x[3+axis]=angle
        starts.append(x)
    if orientation_guard:
        from camera_initial_orientation import table_camera_rotation
        parent=np.asarray(initial_context['parents'][kind][view]['table_from_parent'])[:3,:3]
        desired=parent.T@table_camera_rotation(profiles,initial_context,'sim1',view)
        seed=np.zeros(7);seed[3:6]=Rotation.from_matrix(r.T@desired).as_euler('xyz',degrees=True)
        starts[0]=seed
    attempts=[]
    for seed in starts:
        solution=least_squares(residual,seed,bounds=(low,high),max_nfev=60,
                               diff_step=1e-4,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        x=solution.x;score=float(np.linalg.norm(residual(x,False)))
        edit={'translation_camera_m':x[:3].tolist(),
              'rotation_camera_xyz_degrees':x[3:6].tolist(),'fovy_delta_degrees':float(x[6])}
        try:
            candidate=apply_camera_edit(profiles,origin,kind,view,edit,table_frames)
            if orientation_guard:
                from camera_initial_orientation import comparison
                if not comparison(candidate,initial_context,view)['passed']:
                    raise ValueError('Projection fit violates initial optical orientation consistency')
        except ValueError as error:
            attempts.append({'score':score,'valid':False,'reason':str(error)})
            continue
        attempts.append({'score':score,'valid':True})
        if best is None or score<best[0]:best=(score,edit,candidate)
    if best is None or best[0]>=baseline-1e-7:
        raise ValueError('No bounded projection improvement; use other camera evidence or edit')
    return best[2],{'schema':'rssr.camera_fit.receipt.v1','kind':kind,'view':view,
                   'baseline_residual':baseline,'fitted_residual':best[0],'applied_edit':best[1],
                   'starts':attempts,'residual_evaluations':evaluations,
                   'wall_seconds':time.monotonic()-started,'render_calls':0,'api_requests':0,
                   'historical_profile_used':False,'visual_verification_required':True,
                   'initial_orientation_guard':orientation_guard,
                   'residual_includes_initial_orientation':orientation_guard,
                   'physics_and_actions_modified':False}
