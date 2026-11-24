#!/usr/bin/env python3
"""HOST numerical joint-arc projection, with no task-strategy authorship claim.

Input is a caller-selected legal endpoint and solver parameters. The tool may
change all seven joints along a natural arc, while retaining the exact endpoint,
calibrated target-region tracking, original joint/frame limits and path guards.
No control execution, model changes, or task/physics-gate overrides.
"""
from pathlib import Path
import sys,numpy as np
from scipy.optimize import least_squares
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import drawer_numerical_tools as NT
import drawer_region_geometry_v2 as RG
G=NT.G;DR=NT.DR

def retime_new_plan(q0,path,step):
    points=np.vstack([np.asarray(q0,float),np.asarray(path,float)])
    edges=np.max(np.abs(np.diff(points,axis=0)),axis=1);arc=np.r_[0.,np.cumsum(edges)]
    keep=np.r_[True,np.diff(arc)>1e-12];points=points[keep];arc=arc[keep]
    if len(arc)==1:return points[-1:].copy()
    count=max(1,int(np.ceil(arc[-1]/step)));samples=np.linspace(0.,arc[-1],count+1)[1:]
    out=np.stack([np.interp(samples,arc,points[:,j]) for j in range(7)],axis=1);out[-1]=np.asarray(path[-1],float)
    return out

def plan(S,h,obj,q_goal,*,v_max=.72,cone_deg=24.,q_weight=1.,smooth_weight=.12,los_margin=.006,front_margin=.030,time_multiplier=1.,max_nfev=35,clearance_weight=100.,clearance_buffer=.004,nominal_path=None,position_weight=0.):
    q0=S.q_cmd(h);qgoal=np.asarray(q_goal,float);a=S.robot.arms[h]
    if h in S.attached:raise DR.PlanError('Free-hand camera projection cannot rotate a held camera/object off its actual grasp')
    raw=NT.joint_segment(S,h,qgoal,v_max=v_max) if nominal_path is None else np.asarray(nominal_path,float)
    if raw.ndim!=2 or raw.shape[1]!=7 or not np.array_equal(raw[-1],qgoal):raise ValueError('Numerical nominal path must preserve supplied exact endpoint')
    n=max(len(raw),int(np.ceil(len(raw)*time_multiplier)));nominal_points=np.vstack([q0,raw])
    eyeL,fL=G.mount(S,h);target,_=S.pose(obj)
    cab_c,cab_R=S.pose('drawer_cabinet');cab_half=np.asarray(S.objects['drawer_cabinet']['size_m'])/2
    axis=G.unit(np.asarray(S.objects['bottom_drawer']['joint']['axis'],float))
    front=cab_c+axis*float(np.abs(cab_R.T@axis)@cab_half);ray_u=np.linspace(.02,.99,50)[:,None]
    path=[];last=q0.copy()
    def exact_distances(q):
        qs={hh:S.q_cmd(hh) for hh in DR.HANDS};qs[h]=q
        env,arm,jm=S.check(qs,S.allow)
        return np.array([env[0],arm[0],jm])
    thresholds=np.array([DR.ENV_CLEAR+clearance_buffer,S.arm_clear+clearance_buffer,DR.JOINT_MARGIN])
    for k in range(n):
        t=(k+1)/n*len(raw);lo=min(int(t),len(raw)-1);fraction=t-lo
        nominal=nominal_points[lo]+fraction*(nominal_points[lo+1]-nominal_points[lo]);pnom,_=S.fk(h,nominal)
        if k==n-1:
            q=qgoal.copy()
        else:
            reference=nominal.copy();dist=exact_distances(reference);gradient=np.zeros((3,7))
            if np.any(dist<thresholds+.012):
                for j in range(7):
                    plus=reference.copy();plus[j]+=.001
                    gradient[:,j]=(exact_distances(plus)-dist)/.001
            def residual(q):
                p,R=S.fk(h,q);eye=p+R@eyeL;f=R@fL;v=G.unit(target-eye)
                dot=np.clip(f@v,-1,1);angle=np.arccos(dot)
                gaze=np.cross(f,v)*(max(angle-np.radians(cone_deg),0.)/max(np.sin(angle),1e-8))
                points=eye+ray_u*(target-eye);bq=np.abs((points-cab_c)@cab_R)-cab_half
                sdf=np.linalg.norm(np.maximum(bq,0.),axis=1)+np.minimum(bq.max(axis=1),0.)
                occlusion=np.maximum(los_margin-sdf,0.)
                front_penalty=max(0.,front_margin-float((eye-front)@axis)) if obj=='bottom_handle_bar' else 0.
                # A local finite-difference model accelerates numerical search;
                # unchanged native exact checks still determine admission below.
                clearance=np.maximum(thresholds-(dist+gradient@(q-reference)),0.)
                return np.r_[12*gaze,q_weight*(q-nominal),smooth_weight*(q-last),position_weight*(p-pnom),100*occlusion,80*front_penalty,clearance_weight*clearance]
            seed=np.clip(nominal,a.lo+.035,a.hi-.035)
            q=least_squares(residual,seed,bounds=(a.lo+.035,a.hi-.035),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7).x
        path.append(q.copy());last=q
    cap=.5 if h in S.attached else .72
    p=retime_new_plan(q0,path,min(DR.STEP_MAX*.92,cap/30))
    worst=float(np.max(np.abs(np.diff(np.vstack([q0,p]),axis=0))))
    report=RG.audit(S,{h:p},{h:obj})
    ok,info=S._path_ok({h:p},S.allow)
    result=dict(author='HOST_numeric_joint_arc_tool_not_Luna_strategy',motion_s=len(p)/30,command_frames=len(p),max_joint_command_step_rad=worst,endpoint_exact=bool(np.array_equal(p[-1],qgoal)),all_target_regions_in_FOV=report['all_target_regions_in_FOV'],all_target_regions_visible=report['all_target_regions_visible'],original_path_ok=bool(ok),original_failure=info,minimum_FOV_probes=min(v['inside_raw_FOV'] for r in report['rows'] for v in r['by_hand'].values()),minimum_visible_probes=min(v['unoccluded_target_probes'] for r in report['rows'] for v in r['by_hand'].values()),qualified_geometry=bool(worst<=DR.STEP_MAX+1e-8 and report['all_target_regions_in_FOV'] and ok),visibility_is_separate_evidence_not_full_object_unocclusion_gate=True)
    return p,result,report
