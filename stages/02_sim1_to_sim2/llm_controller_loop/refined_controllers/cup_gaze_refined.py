"""API13 minimal schedule repair: synchronized paired approach retiming and post-release empty-hand aside retiming."""
import importlib.util, sys
import numpy as np
from scipy.optimize import least_squares

from pathlib import Path
_FR3 = Path(__file__).resolve().parents[2] / 'franka_dual_v1'
BASE = str(_FR3 / 'controllers/duo_stack_cups.py')
CODE = str(_FR3)
if CODE not in sys.path: sys.path.insert(0, CODE)
import arch_away
_spec=importlib.util.spec_from_file_location('stack_cups_source_pinned',BASE)
B=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)
H=('left','right'); FPS=30.; STEP_MAX=.025
CAM_T=np.array([.075,0.,-.1188])
CAM_R=np.array([[-.8455898165843033,0.,.5338331781454938],[0.,1.,0.],[-.5338331781454938,0.,-.8455898165843033]])
TAN_V=np.tan(np.deg2rad(58.)/2); TAN_H=TAN_V*4./3.
OPTIONS=dict(upper_wait='stage',overlap=True,final_aside=False,speed=1.6,grip_s=(.4,.4,.4),preopen_in_approach=True,pre=.07)

def _unit(x):
 x=np.asarray(x,float); return x/max(float(np.linalg.norm(x)),1e-12)
def _cam(p,R): return np.asarray(p)+np.asarray(R)@CAM_T, np.asarray(R)@CAM_R
def _camera_xyz(eye,Rc,point): return np.asarray(Rc).T@(np.asarray(point,float)-np.asarray(eye,float))
def _in_fov(cam_xyz):
 x,y,z=np.asarray(cam_xyz,float); d=-z
 return bool(d>1e-8 and abs(x)<=TAN_H*d and abs(y)<=TAN_V*d)
def _optical(Rc): return -np.asarray(Rc)[:,2]
def _cup_local_bounds(S,cup):
 m=S.m; bid=m.body(cup).id; allp=[]; mj=B.mujoco
 for g in range(m.ngeom):
  if m.geom_bodyid[g]!=bid: continue
  R=np.zeros(9); mj.mju_quat2Mat(R,m.geom_quat[g]); R=R.reshape(3,3)
  if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
   k=m.geom_dataid[g]; v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
  elif m.geom_type[g]==mj.mjtGeom.mjGEOM_BOX:
   v=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
  else: continue
  allp.append(m.geom_pos[g]+v@R.T)
 if not allp: raise RuntimeError(f'{cup}: no supported collision vertices')
 z=np.concatenate(allp)[:,2]; return float(z.min()),float(z.max())
def _target_samples(S,cup):
 height,r0,r1=B.cup_profile(S,cup); lo,hi=_cup_local_bounds(S,cup)
 if abs((hi-lo)-height)>2e-4: raise RuntimeError(f'{cup}: profile/bounds disagree')
 return np.asarray([[ (r0+(r1-r0)*f)*np.cos(a),(r0+(r1-r0)*f)*np.sin(a),lo+height*f]
  for f in (.08,.30,.55,.80,.98) for a in np.arange(16)*np.pi/8],float)
def _visible_fraction(S,h,q,cup,pts):
 p,R=S.fk(h,np.asarray(q)); eye,Rc=_cam(p,R); obj,Ro=S.pose(cup)
 world=np.asarray(obj)[:,None]+np.asarray(Ro)@np.asarray(pts).T
 cam=(np.asarray(Rc).T@(world-np.asarray(eye)[:,None])).T
 return float(np.mean([_in_fov(x) for x in cam]))
def _gaze_row(obj,point): return {'object':str(obj),'point_object_m':[float(x) for x in np.asarray(point,float)],'active':True}
def _bind(S,picks,cup_of):
 rows={}
 for h in H:
  c,R=S.pose(cup_of[h]); local=np.asarray(R).T@(np.asarray(picks[h][2],float)-np.asarray(c))
  rows[h]=_gaze_row(cup_of[h],local)
 S.gaze_patches=rows
 return rows

def retime_joint_curve(q_start,curve,vmax):
 """Retiming by joint-space max-coordinate arc length; keep physical path/endpoints, enforce frame and velocity caps."""
 q0=np.asarray(q_start,float); a=np.asarray(curve,float)
 if a.ndim!=2 or a.shape[1]!=len(q0) or not len(a): raise ValueError('expected nonempty NxD curve')
 pts=np.vstack((q0,a)); ds=np.max(np.abs(np.diff(pts,axis=0)),axis=1); ss=np.r_[0.,np.cumsum(ds)]
 keep=np.r_[True,np.diff(ss)>1e-12]; ss=ss[keep]; pts=pts[keep]
 if ss[-1]<1e-12: return a[-1:]
 vmax=float(vmax); n=max(2,int(np.ceil((ss[-1]/vmax+.22)*FPS)))
 while True:
  T=n/FPS; tr=min(.18,T/2); t=np.linspace(0.,T,n+1)
  v=np.minimum(1.,np.minimum(t/max(tr,1e-9),(T-t)/max(tr,1e-9)))
  arc=np.r_[0.,np.cumsum((v[:-1]+v[1:])*.5/FPS)]
  if arc[-1]<=1e-12: n+=1; continue
  s=arc[1:]/arc[-1]*ss[-1]
  out=np.column_stack([np.interp(s,ss,pts[:,j]) for j in range(pts.shape[1])]); out[-1]=a[-1]
  inc=np.max(np.abs(np.diff(np.vstack((q0,out)),axis=0)),axis=1)
  if np.max(inc)<=STEP_MAX+1e-10 and np.max(inc)*FPS<=vmax+1e-8: return out
  n+=max(1,int(np.ceil(n*max(np.max(inc)/STEP_MAX,np.max(inc)*FPS/vmax)*1.03)))
def _retime_pair(curves,qstart,vmax=.72):
 """Pad original left/right curves to common length, retime the combined 14-D curve once, split afterward."""
 n=max(len(curves[h]) for h in H); padded={}
 for h in H:
  a=np.asarray(curves[h],float)
  if a.ndim!=2 or a.shape[1]!=7 or not len(a): raise ValueError('paired hand curve must be nonempty Nx7')
  padded[h]=np.vstack((a,np.repeat(a[-1:],n-len(a),axis=0)))
 q0=np.concatenate([np.asarray(qstart[h],float) for h in H]); pair=np.concatenate([padded[h] for h in H],axis=1)
 out=retime_joint_curve(q0,pair,vmax)
 return {h:out[:,7*i:7*(i+1)].copy() for i,h in enumerate(H)}
def _turn_goal(S,h,q0,target):
 arm=S.robot.arms[h]; lo=np.asarray(arm.lo,float)+.03; hi=np.asarray(arm.hi,float)-.03
 lo[:4]=np.maximum(lo[:4],q0[:4]); hi[:4]=np.minimum(hi[:4],q0[:4]); lo[4:]=np.maximum(lo[4:],q0[4:]-1.1); hi[4:]=np.minimum(hi[4:],q0[4:]+1.1)
 if np.any(lo[4:]>=hi[4:]): raise RuntimeError('camera wrist bounds unavailable')
 def fun(x):
  q=q0.copy(); q[4:]=x; p,R=S.fk(h,q); eye,Rc=_cam(p,R); ray=_unit(target-eye); f=_unit(_optical(Rc))
  return np.r_[4*np.cross(f,ray),2*(1.-np.dot(f,ray)),.035*(x-q0[4:])]
 z=least_squares(fun,q0[4:],bounds=(lo[4:],hi[4:]),max_nfev=100,xtol=2e-5,ftol=2e-5,gtol=2e-5)
 q=q0.copy(); q[4:]=z.x; return q
def _select(S,cup_of):
 qnow={h:np.asarray(S.q_cmd(h),float) for h in H}; candidates={}; surfaces={}
 for h in H:
  surfaces[h]=_target_samples(S,cup_of[h]); opts=B.wall_grasps(S,h,cup_of[h])[:8]; per=[]
  for item in opts:
   _,pre,g,R,qpre=item; qt=_turn_goal(S,h,qnow[h],np.asarray(g,float))
   per.append((item,qt,S.joint_plan(h,qt,v_max=min(.72,B.vj()),start=qnow[h]),S.joint_plan(h,qpre,v_max=B.vj(),start=qt)))
  candidates[h]=per
 pairs=[]
 for l in candidates['left']:
  for r in candidates['right']:
   pair={'left':l,'right':r}; starts={h:pair[h][0][4] for h in H}
   try:
    dn=S.plan_line({h:(pair[h][0][2],pair[h][0][3]) for h in H},speed=B.sp(.06),start=starts,label='descend')
    S.plan_line({h:(pair[h][0][2]+[0,0,B.LIFT],pair[h][0][3]) for h in H},speed=B.sp(.08),start={h:dn[h][-1] for h in H},label='lift')
   except Exception: continue
   turns={h:pair[h][2] for h in H}; aps={h:pair[h][3] for h in H}
   if not S._path_ok(turns)[0] or not S._path_ok(aps)[0]: continue
   vals=[]
   for h in H:
    seq=list(turns[h])+list(aps[h])+list(dn[h]); vf=[_visible_fraction(S,h,q,cup_of[h],surfaces[h]) for q in seq]
    vals.append((float(np.mean(vf)) if vf else 0.,float(np.quantile(vf,.1)) if vf else 0.))
   score=(min(v[1] for v in vals),sum(v[0] for v in vals),-sum(float(np.max(np.abs(pair[h][1]-qnow[h]))) for h in H))
   pairs.append((score,pair))
 if not pairs: raise RuntimeError('no collision-clear top-down grasp pair')
 pairs.sort(key=lambda x:x[0],reverse=True); return pairs[0][1]
def _initial_turn(S,pair):
 q0={h:np.asarray(S.q_cmd(h),float) for h in H}; plans={h:retime_joint_curve(q0[h],pair[h][2],min(.72,B.vj())) for h in H}
 ok,detail=S._path_ok(plans)
 if not ok: raise RuntimeError('camera turn aggregate path blocked: '+str(detail))
 S.together(plans,'initial_wrist_camera_turn')
def _approach_selected(S,pair,cup_of,body_names):
 if set(body_names)!=set(H) or any(str(body_names[h])!=str(cup_of[h]) for h in H): raise RuntimeError('approach callback binding mismatch')
 q0={h:np.asarray(S.q_cmd(h),float) for h in H}
 plans=_retime_pair({h:pair[h][3] for h in H},q0,.72)
 ok,detail=S._path_ok(plans)
 if not ok: raise RuntimeError('retimed paired approach blocked: '+str(detail))
 if B.opt('preopen_in_approach'): S.together(plans,'approach',grips={h:B.OPEN_WALL for h in H})
 else: S.move_joints({h:pair[h][0][4] for h in H},'approach',v_max=B.vj())
 picks={h:pair[h][0] for h in H}; S.line({h:(picks[h][2],picks[h][3]) for h in H},'descend',speed=B.sp(.06))
 return picks
def _approach_callback(pair,cup_of):
 def callback(S,body_names): return _approach_selected(S,pair,cup_of,body_names)
 return callback
def run(S):
 b=S.bindings; base,upper=b['targets']; hb,hu=b['stack_hands']['base'],b['stack_hands']['upper']; cup_of={hb:base,hu:upper}
 if set(cup_of)!=set(H): raise RuntimeError('unexpected bound stack hands')
 old_options=dict(B.OPTIONS); old_approach=B.approach_and_descend; old_move=S.move_joints; old_grip=S.grip; old_together=S.together; old_aside=B.ASIDE
 B.OPTIONS.update(OPTIONS); B.ASIDE=(.40,.25,.30)
 for h in H: arch_away.install(S,h,lines='posture')
 S.allow={h:{cup_of[h]} for h in H}
 try:
  pair=_select(S,cup_of); _bind(S,{h:pair[h][0] for h in H},cup_of); _initial_turn(S,pair)
  def faster_held_move(targets,label,v_max=.6,allow=None):
   if not ('transport' in label and any(x in label for x in ('base','upper'))): return old_move(targets,label,v_max=v_max,allow=allow)
   q0={h:np.asarray(S.q_cmd(h),float) for h in H}; qb={h:np.asarray(targets.get(h,q0[h]),float) for h in H}; moving=[h for h in H if np.max(np.abs(qb[h]-q0[h]))>1e-7]
   if not moving:return
   vmax=min(.5,float(v_max)); plans={h:retime_joint_curve(q0[h],S.joint_plan(h,qb[h],v_max=vmax,start=q0[h]),vmax) for h in moving}
   ok,detail=S._path_ok(plans,allow)
   if not ok: raise RuntimeError(f'{label}: retimed held joint path blocked: {detail}')
   S.together(plans,label,allow=allow)
  def retimed_postrelease_aside(plans,label,grips=None,allow=None,delay=None):
   if label==f'{hb}_aside' and not grips and not delay:
    q0={h:np.asarray(S.q_cmd(h),float) for h in plans}; plans={h:retime_joint_curve(q0[h],plans[h],.72) for h in plans}
    ok,detail=S._path_ok(plans,allow)
    if not ok: raise RuntimeError(f'{label}: retimed empty-hand path blocked: {detail}')
   return old_together(plans,label,grips=grips,allow=allow,delay=delay)
  def tracked_grip(widths,label,duration):
   result=old_grip(widths,label,duration)
   if label in ('base_release','upper_release'):
    h=hb if label=='base_release' else hu; S.gaze_patches[h]['active']=False
   return result
  S.move_joints=faster_held_move; S.together=retimed_postrelease_aside; S.grip=tracked_grip; B.approach_and_descend=_approach_callback(pair,cup_of)
  return B.run(S)
 finally:
  B.OPTIONS.clear(); B.OPTIONS.update(old_options); B.ASIDE=old_aside; B.approach_and_descend=old_approach
  S.move_joints=old_move; S.together=old_together; S.grip=old_grip
def grasps(S): return B.grasps(S)
