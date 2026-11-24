"""HOST numeric rigid-camera-space route tool; no simulator camera changes.

Caller chooses camera-origin waypoints, gaze cone, speed, and endpoint. The
actual fixed TCP-camera extrinsics produce real TCP goals solved by original
robot IK. All seven joints, exact endpoint, original check and command bounds
remain physical. No execution or task-strategy choice occurs in this module.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import drawer_numerical_tools as NT
import drawer_region_geometry_v2 as RG
from joint_region_path import retime_new_plan
DR=NT.DR;G=NT.G

def cone_camera(C,eye,target,cone_deg):
    f=-C[:,2];v=G.unit(target-eye);angle=np.arccos(np.clip(f@v,-1,1));excess=max(angle-np.radians(cone_deg),0.)
    if excess<=1e-9:return C
    axis=np.cross(f,v)
    if np.linalg.norm(axis)<1e-9:axis=np.cross(f,C[:,0])
    return Rotation.from_rotvec(G.unit(axis)*excess).as_matrix()@C

def plan(S,h,obj,q_goal,*,waypoints=None,camera_speed=.30,angular_speed=.8,cone_deg=20.,min_s=.4,null_gain=.12,force_nullspace_iterations=False,preserve_source_joint_branch=True):
    if h in S.attached:raise DR.PlanError('Camera-space free approach tool cannot alter an existing grasp')
    q0=S.q_cmd(h);qgoal=np.asarray(q_goal,float);ci=int(S.robot.arms[h].wrist_cam)
    d0=NT.predicted_data(S,{h:q0});d1=NT.predicted_data(S,{h:qgoal})
    p0,R0=S.fk(h,q0);eye0=d0.cam_xpos[ci].copy();C0=d0.cam_xmat[ci].reshape(3,3).copy()
    eye_local=R0.T@(eye0-p0);camera_local=R0.T@C0
    eye1=d1.cam_xpos[ci].copy();C1=d1.cam_xmat[ci].reshape(3,3).copy();target,_=S.pose(obj)
    nodes=[eye0,*[np.asarray(p,float) for p in (waypoints or [])],eye1];distances=np.linalg.norm(np.diff(nodes,axis=0),axis=1);total=float(sum(distances));rotation=float(Rotation.from_matrix(C1@C0.T).magnitude())
    duration=max(float(min_s),total/float(camera_speed),rotation/float(angular_speed));n=max(2,int(np.ceil(duration*30)))
    cumulative=np.r_[0.,np.cumsum(distances)];interpolation=Slerp([0.,1.],Rotation.from_matrix(np.array([C0,C1])))
    q=q0.copy();path=[];max_pe=max_re=0.
    for u in DR._hc(n):
        distance=u*total;segment=min(int(np.searchsorted(cumulative,distance,side='right')-1),len(nodes)-2);segment=max(segment,0)
        frac=(distance-cumulative[segment])/max(distances[segment],1e-10);eye=nodes[segment]+frac*(nodes[segment+1]-nodes[segment]);C=interpolation([u]).as_matrix()[0]
        if u<1.-1e-10:C=cone_camera(C,eye,target,cone_deg)
        R=C@camera_local.T;p=eye-R@eye_local
        extra={'tol_m':0.,'tol_rad':0.} if force_nullspace_iterations else {}
        q,pe,re=S.robot.ik_local(h,p,R,q,S.d.qpos,(1-u)*q0+u*qgoal,iters=80,null_gain=null_gain,**extra)
        max_pe=max(max_pe,pe);max_re=max(max_re,re);path.append(q.copy())
        if pe>DR.LINE_POS_TOL or re>DR.IK_ROT_TOL:
            return np.asarray(path),dict(qualified_geometry=False,reason='native IK pose tolerance',position_error_m=pe,rotation_error_deg=re,frames_solved=len(path)),None
    # Source reach plans are bound to the exact pregrasp branch. Complete the
    # newly planned path with a native bounded joint correction, then re-audit.
    before_bridge_q=q.copy();before_bridge_frames=len(path);bridge=[]
    if preserve_source_joint_branch and np.max(np.abs(q-qgoal))>1e-9:
        bridge=NT.joint_segment(S,h,qgoal,v_max=.72,start=q);path.extend(bridge)
    path=retime_new_plan(q0,path,min(DR.STEP_MAX*.92,.72/30));report=RG.audit(S,{h:path},{h:obj});ok,info=S._path_ok({h:path},S.allow)
    step=float(np.max(np.abs(np.diff(np.vstack([q0,path]),axis=0))))
    summary=dict(author='HOST_rigid_camera_space_numerics_not_Luna_policy',motion_s=len(path)/30,frames=len(path),camera_origin_path_length_m=total,nominal_duration_s=duration,before_bridge_frames=before_bridge_frames,before_bridge_q=before_bridge_q.tolist(),bridge_frames=len(bridge),maximum_exact_endpoint_correction_rad=float(np.max(np.abs(before_bridge_q-qgoal))),exact_endpoint=bool(np.array_equal(path[-1],qgoal)),source_joint_branch_retained=bool(preserve_source_joint_branch),max_joint_step_rad=step,maximum_IK_position_error_m=max_pe,maximum_IK_rotation_error_deg=max_re,all_target_regions_in_FOV=report['all_target_regions_in_FOV'],sampled_target_region_visible_each_frame=report['all_target_regions_visible'],zero_target_ray_frames=sum(r['by_hand'][h]['unoccluded_target_probes']==0 for r in report['rows']),original_path_ok=bool(ok),original_failure=info,qualified_geometry=bool(ok and report['all_target_regions_in_FOV'] and step<=DR.STEP_MAX+1e-8))
    return path,summary,report
