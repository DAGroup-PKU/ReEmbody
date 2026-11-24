"""HOST calibrated object-region geometry with denser target coverage.

The old nine box-corner/centre rays can miss a visible thin handle. Positive
visibility here still requires an actual MuJoCo first hit on the target body;
we sample its actual declared bounds more densely, without changing the scene,
camera calibration, physical gates, or requiring a fixed centroid/patch.
"""
import numpy as np,mujoco
import drawer_numerical_tools as NT
DR=NT.DR

def predicted_state(S,qs,slide_follow=None):
    # Unchanged NT helper provides scratch arm FK and attached-block prediction.
    # Its optional rigid sliding extrapolation is omitted here: at a hard stop,
    # actual tracking/compliance may make that affine model exceed the range by
    # fractions of a millimetre before the first legal pulling command.
    d=NT.predicted_data(S,qs,slide_follow=None);info=None
    if slide_follow is not None:
        h,obj,joint=slide_follow['hand'],slide_follow['object'],slide_follow['joint']
        contacts=S.contacts(h,obj)
        if len(contacts)!=2 or not all(contacts):raise DR.PlanError('Sliding camera prediction requires actual bilateral target contacts')
        ji=S.m.joint(joint).id;adr=int(S.m.jnt_qposadr[ji]);axis=np.asarray(S.objects['bottom_drawer']['joint']['axis'],float);axis/=np.linalg.norm(axis)
        current=float(S.slide(joint));actual_TCP=S.tcp(h)[0];future_TCP=d.site_xpos[S.robot.arms[h].tcp];raw=float(current+axis@(future_TCP-actual_TCP));lo,hi=S.m.jnt_range[ji]
        projected=float(np.clip(raw,lo,hi));d.qpos[adr]=projected;mujoco.mj_forward(S.m,d)
        info=dict(author='HOST_scratch_prediction_only',actual_slide_m=current,unclamped_predicted_slide_m=raw,projected_predicted_slide_m=projected,projection_m=projected-raw,original_articulation_range_m=[float(lo),float(hi)],actual_range_and_state_and_physics_unchanged=True)
    return d,info

def facing(S,qs,h,obj,*,slide_follow=None):
    d,slide_info=predicted_state(S,qs,slide_follow);ci=int(S.robot.arms[h].wrist_cam)
    eye=d.cam_xpos[ci];C=d.cam_xmat[ci].reshape(3,3);bi=S.m.body(obj).id
    center=d.xpos[bi]
    geoms=np.flatnonzero(S.m.geom_bodyid==bi)
    if len(geoms)!=1 or S.m.geom_type[geoms[0]]!=mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError('This drawer surface audit requires one actual native box geom per target')
    gi=int(geoms[0]);gc=d.geom_xpos[gi];R=d.geom_xmat[gi].reshape(3,3);half=S.m.geom_size[gi]
    # Dense length/face samples use actual native collision surface coordinates.
    # A 0.1% inset avoids floating-point tangent misses; first hit is still the
    # real physical surface. Interior grid nodes are excluded.
    axes=[np.linspace(-1.,1.,17 if i==int(np.argmax(half)) else 3) for i in range(3)]
    tv=np.tan(np.deg2rad(S.m.cam_fovy[ci]/2));th=tv*224/168;inside=visible=total=0;hit_counts={}
    for x in axes[0]:
      for y in axes[1]:
       for z in axes[2]:
        if max(abs(x),abs(y),abs(z))<1.:continue
        total+=1;point=gc+R@(np.array([x,y,z])*half*.999);v=C.T@(point-eye);depth=-v[2]
        if not (depth>0 and abs(v[0])<=depth*th and abs(v[1])<=depth*tv):continue
        inside+=1;gh=np.array([-1],np.int32);direction=point-eye;direction/=np.linalg.norm(direction)
        mujoco.mj_ray(S.m,d,eye,direction,None,1,-1,gh)
        body=S.m.body(int(S.m.geom_bodyid[gh[0]])).name if gh[0]>=0 else None
        hit_counts[str(body)]=hit_counts.get(str(body),0)+1
        visible+=int(gh[0]>=0 and S.m.geom_bodyid[gh[0]]==bi)
    v=(center-eye)/np.linalg.norm(center-eye)
    return dict(probes=total,actual_target_geom=S.m.geom(gi).name,inside_raw_FOV=inside,unoccluded_target_probes=visible,center_axis_angle_deg=float(np.degrees(np.arccos(np.clip(v@(-C[:,2]),-1,1)))),first_hit_body_counts=hit_counts,predicted_only=True,sliding_prediction=slide_info)

def audit(S,plans,targets,*,start=None,slide_follow=None):
    n=max((len(p) for p in plans.values()),default=0);q0={h:np.asarray((start or {}).get(h,S.q_cmd(h)),float) for h in DR.HANDS};rows=[]
    for k in range(n):
        qs={h:(plans[h][min(k,len(plans[h])-1)] if h in plans else q0[h]) for h in DR.HANDS}
        rows.append(dict(frame=k,by_hand={h:facing(S,qs,h,obj,slide_follow=slide_follow) for h,obj in targets.items()}))
    return dict(author='HOST_dense_calibrated_region_geometry_not_Luna_strategy',command_frames=n,rows=rows,all_target_regions_in_FOV=all(v['inside_raw_FOV']>0 for r in rows for v in r['by_hand'].values()),all_target_regions_visible=all(v['unoccluded_target_probes']>0 for r in rows for v in r['by_hand'].values()),qualification='Actual native box surface/length probes, calibrated FOV and first-hit body identity; visible means at least one sampled target region visible each frame, not complete unocclusion. Predicted geometry only, not new RGB or physics admission.')
