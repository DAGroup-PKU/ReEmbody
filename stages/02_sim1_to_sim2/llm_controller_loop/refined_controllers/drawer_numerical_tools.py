"""HOST-authored numerical execution tools; strategies and choices belong to Luna.

This module provides no autonomous task policy. It generates candidate catalogs,
solves bounded Cartesian/gaze paths and validates requested plans with the
unchanged native task guard and checked executor. No mj_step/state/gate changes.
"""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import geometric_look_catalog as G
from franka_dual_v1 import duo_run as DR
AUTHOR='HOST_NUMERICAL_TOOL_not_Luna_strategy'
BASE=G.base

def prepare_task_checks(S):
 """Install the original source CabinetGuard and source contact-allow map once."""
 blk,bh,dh=BASE.roles(S)
 S.allow={bh:{blk},dh:{'bottom_handle_bar','bottom_handle_post_a','bottom_handle_post_b','bottom_drawer'}}
 if not getattr(S,'_host_original_drawer_guard',False):
  BASE.install_guard(S,bh);S._host_original_drawer_guard=True
 return blk,bh,dh

def initial_catalog(S,count=16):
 """All numerical region-facing and native-path-safe pairs, without selecting policy."""
 blk,bh,dh=prepare_task_checks(S)
 rows={h:G.candidate_rows(S,h,blk if h==bh else 'bottom_handle_bar',count) for h in DR.HANDS}
 good={h:[r for r in rows[h] if r['original_single_path_ok'] and r['facing']['unoccluded_target_ray_probes']>0] for h in DR.HANDS}
 pairs=[]
 for l in good['left']:
  for r in good['right']:
   plans={'left':l['path'],'right':r['path']};ok,info=S._path_ok(plans,S.allow)
   if ok:pairs.append(dict(candidate_ids={'left':l['id'],'right':r['id']},plans=plans,motion_s=max(len(l['path']),len(r['path']))/30,maximum_TCP_displacement_m=max(np.linalg.norm(l['natural_TCP_displacement_m']),np.linalg.norm(r['natural_TCP_displacement_m'])),endpoint_facing={'left':l['facing'],'right':r['facing']}))
 return dict(author=AUTHOR,rows=rows,pairs=pairs)

def checked_together(S,plans,label,*,native_together,grips=None,allow=None,delay=None):
 """Validate actual delayed pair; delegate scalar widths to native ramp semantics."""
 p={h:np.asarray(x,float) for h,x in plans.items() if len(x)}
 for h,s in (delay or {}).items():
  if h in p and s>0:p[h]=np.r_[np.repeat(S.q_cmd(h)[None],int(round(s*30)),0),p[h]]
 for h,x in p.items():
  if not np.isfinite(x).all() or x.ndim!=2 or x.shape[1]!=7:raise DR.PlanError('nonfinite/invalid requested joint path')
  step=np.max(np.abs(np.diff(np.vstack([S.q_cmd(h),x]),axis=0)))
  if step>DR.STEP_MAX+1e-8:raise DR.PlanError(f'{label}: requested path increment {step} exceeds native limit')
 if p:
  ok,info=S._path_ok(p,allow)
  if not ok:raise DR.PlanError(f'{label}: original final aggregate path check failed: {info}')
 return native_together(p,label,grips=grips,allow=allow,delay=None)

def joint_segment(S,h,q,*,v_max,start=None):
 """Native timing, loaded/free physical cap retained. No task route choice."""
 cap=.5 if h in S.attached else .72
 q0=S.q_cmd(h) if start is None else np.asarray(start,float)
 p=S._joint_segment(q0,np.asarray(q,float),min(float(v_max),cap))
 if np.max(np.abs(np.diff(np.vstack([q0,p]),axis=0)))>DR.STEP_MAX+1e-8:raise DR.PlanError('native joint segment unexpectedly exceeds STEP_MAX')
 return p

def predicted_data(S,qs,slide_follow=None):
 """Scratch FK carrying native attached objects; optional actual-contact sliding prediction."""
 d=mujoco.MjData(S.m);d.qpos[:]=S._qpos_with(qs);mujoco.mj_forward(S.m,d)
 for h,(b,adr,site,(pr,Rr)) in S.attached.items():
  pt=d.site_xpos[site];Rt=d.site_xmat[site].reshape(3,3);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,(Rt@Rr).reshape(9))
  d.qpos[adr:adr+3]=pt+Rt@pr;d.qpos[adr+3:adr+7]=quat
 if slide_follow:
  h,obj,joint=slide_follow['hand'],slide_follow['object'],slide_follow['joint']
  if not all(S.contacts(h,obj)):raise DR.PlanError('sliding held-target prediction requires actual bilateral contact')
  ji=S.m.joint(joint).id;adr=int(S.m.jnt_qposadr[ji]);axis=np.asarray(S.objects['bottom_drawer']['joint']['axis'],float);axis/=np.linalg.norm(axis)
  p0,_=S.tcp(h);pp=d.site_xpos[S.robot.arms[h].tcp];qslide=float(S.slide(joint)+axis@(pp-p0));lo,hi=S.m.jnt_range[ji]
  if qslide<lo-1e-6 or qslide>hi+1e-6:raise DR.PlanError('requested sliding prediction exceeds original articulation range')
  d.qpos[adr]=np.clip(qslide,lo,hi)
 mujoco.mj_forward(S.m,d);return d

def region_facing(S,qs,h,obj,*,slide_follow=None):
 """Exact calibrated FOV and actual scene-ray probes on scratch predicted state."""
 d=predicted_data(S,qs,slide_follow);ci=int(S.robot.arms[h].wrist_cam);eye=d.cam_xpos[ci];C=d.cam_xmat[ci].reshape(3,3);f=-C[:,2]
 bi=S.m.body(obj).id;c=d.xpos[bi];R=d.xmat[bi].reshape(3,3);sz=np.asarray(S.objects[obj]['size_m'],float)
 locals=[np.zeros(3)]+[np.array([i,j,k])*sz/2*.98 for i in (-1,1) for j in (-1,1) for k in (-1,1)]
 tv=np.tan(np.deg2rad(S.m.cam_fovy[ci]/2));th=tv*224/168;inside=visible=0
 for loc in locals:
  point=c+R@loc;v=C.T@(point-eye);depth=-v[2];infov=depth>0 and abs(v[0])<=depth*th and abs(v[1])<=depth*tv
  inside+=int(infov)
  if infov:
   gh=np.array([-1],np.int32);mujoco.mj_ray(S.m,d,eye,G.unit(point-eye),None,1,-1,gh)
   if gh[0]>=0 and S.m.geom_bodyid[gh[0]]==bi:visible+=1
 v=G.unit(c-eye);return dict(center_axis_angle_deg=float(np.degrees(np.arccos(np.clip(v@f,-1,1)))),probes=len(locals),inside_raw_FOV=inside,unoccluded_target_probes=visible,predicted_only=True)

def audit_facing(S,plans,targets,*,start=None,slide_follow=None):
 """Numerical sampled command FK evidence; no physical/rendered admission claim."""
 n=max((len(p) for p in plans.values()),default=0);q0={h:np.asarray((start or {}).get(h,S.q_cmd(h)),float) for h in DR.HANDS};rows=[]
 for k in range(n):
  qs={h:(plans[h][min(k,len(plans[h])-1)] if h in plans else q0[h]) for h in DR.HANDS}
  rows.append(dict(frame=k,by_hand={h:region_facing(S,qs,h,obj,slide_follow=slide_follow) for h,obj in targets.items()}))
 return dict(author=AUTHOR,command_frames=n,rows=rows,all_target_regions_in_FOV=all(v['inside_raw_FOV']>0 for row in rows for v in row['by_hand'].values()),all_target_regions_visible=all(v['unoccluded_target_probes']>0 for row in rows for v in row['by_hand'].values()),qualification='sampled predicted FK/FOV/ray geometry; original physics and actual rendered visibility still required')

def _cone_rotation(p,R,eye_local,forward_local,target,max_angle_deg):
 out=np.asarray(R,float).copy();limit=np.deg2rad(float(max_angle_deg))
 for _ in range(8):
  eye=p+out@eye_local;f=out@forward_local;v=G.unit(target-eye);angle=np.arccos(np.clip(f@v,-1,1))
  if angle<=limit+1e-8:break
  axis=np.cross(f,v)
  if np.linalg.norm(axis)<1e-9:axis=np.cross(f,[1.,0,0])
  if np.linalg.norm(axis)<1e-9:axis=np.cross(f,[0.,1.,0])
  out=Rotation.from_rotvec(G.unit(axis)*(angle-limit)).as_matrix()@out
 return out

def plan_region_line(S,h,obj,p_goal,R_goal,*,start=None,speed=.12,min_s=.4,max_center_angle_deg=24.,preserve_end_rotation=True,allow=None,other_start=None,slide_follow=None,require_unoccluded=True):
 """Numerical line solver for a MODEL-specified waypoint and target.

 First native line is kept verbatim if actual object-region probes stay visible.
 Otherwise minimally correct its camera cone and re-solve Cartesian IK while
 retaining the original target position and prescribed legal endpoint rotation.
 Native STEP_MAX resampling and final original complete pair checks apply.
 """
 qstart=S.q_cmd(h) if start is None else np.asarray(start,float);others={hh:np.asarray((other_start or {}).get(hh,S.q_cmd(hh)),float) for hh in DR.HANDS if hh!=h};st={**others,h:qstart}
 raw=S.plan_line({h:(np.asarray(p_goal,float),np.asarray(R_goal,float))},speed=speed,min_s=min_s,allow=allow,start=st,check=False,label='host_numeric_model_requested_region_line')[h]
 report=audit_facing(S,{h:raw},{h:obj},start=st,slide_follow=slide_follow)
 admission_key='all_target_regions_visible' if require_unoccluded else 'all_target_regions_in_FOV'
 if report[admission_key]:
  ok,info=S._path_ok({h:raw},allow,others=others)
  if not ok:raise DR.PlanError(f'native region-visible line collision: {info}')
  return raw,report
 if h in S.attached or slide_follow:
  raise DR.PlanError("held camera/object geometry has world occlusion; model must choose a different route, not rotate off the grasp")
 eyeL,fL=G.mount(S,h);p0,R0=S.fk(h,qstart);c0,_=S.pose(obj)
 # Resample native nominal Cartesian/rotation path, never just jump to independently solved gaze joints.
 for multiplier in (1,2,4,8):
  n=max(2,len(raw)*multiplier);q=qstart.copy();path=[];worst=0.
  for k,u in enumerate(DR._hc(n)):
   pos=p0+u*(np.asarray(p_goal)-p0)
   rv=Rotation.from_matrix(np.asarray(R_goal)@R0.T).as_rotvec();Rn=Rotation.from_rotvec(rv*u).as_matrix()@R0
   RR=Rn if preserve_end_rotation and k==n-1 else _cone_rotation(pos,Rn,eyeL,fL,c0,max_center_angle_deg)
   qn,pe,re=S.robot.ik_local(h,pos,RR,q,S.d.qpos,q,iters=60)
   if pe>DR.LINE_POS_TOL or re>DR.IK_ROT_TOL:raise DR.PlanError(f'numerical gaze-line IK leaves original reachable tolerances ({pe}m,{re}deg)')
   worst=max(worst,float(np.max(np.abs(qn-q))));path.append(qn.copy());q=qn
  if worst>DR.STEP_MAX+1e-8:continue
  p=np.asarray(path);report=audit_facing(S,{h:p},{h:obj},start=st,slide_follow=slide_follow)
  if not report[admission_key]:raise DR.PlanError('model-requested waypoint/rotation fails requested region-FOV/visibility; numerical rotation cannot lawfully fix it')
  ok,info=S._path_ok({h:p},allow,others=others)
  if not ok:raise DR.PlanError(f'final numerical gaze-line original pathcheck failed: {info}')
  return p,report
 raise DR.PlanError('continuous gaze line cannot meet immutable per-frame STEP_MAX after resampling')


def load_exact_initial_pairs(S,catalog_dir,*,catalog_sha256):
 """Load only an exact-variation, source-bound host numerical catalog chosen by model."""
 d=Path(catalog_dir);p=d/'feasible_pair_catalog.json'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=catalog_sha256:raise ValueError('host catalog hash changed')
 provenance=json.loads((d/'provenance.json').read_text())
 if provenance['full_source_variation']!=S.v.to_json():return None
 if float(S.d.time)!=0. or len(S.actions)!=0:raise ValueError('initial catalog requires original t0 before commands')
 if hashlib.sha256(S.xml.encode()).hexdigest()!=provenance['source_scene_sha256']:raise ValueError('cached geometry scene differs from actual exact initial scene')
 rows=json.loads(p.read_text());out=[]
 for row in rows:
  plans={h:np.asarray(q,float) for h,q in row['plans_q'].items()}
  ok,info=S._path_ok(plans,S.allow)
  if not ok:raise DR.PlanError('cached geometry no longer passes original task check: '+str(info))
  out.append(dict(candidate_ids=row['candidate_ids'],plans=plans,motion_s=row['issued_motion_clock_s'],numerical_only_not_physics_admission=True))
 return out

def load_exact_initial_catalog(S,catalog_dir,*,artifact_hashes):
 """Load ALL source-bound numerical pairs; Luna chooses, checked_together rechecks selected path."""
 d=Path(catalog_dir)
 names=('provenance.json','left_candidates.json','right_candidates.json','paired_candidates.json')
 if set(artifact_hashes)!=set(names):raise ValueError('complete exact catalog artifact hashes required')
 for name in names:
  if hashlib.sha256((d/name).read_bytes()).hexdigest()!=artifact_hashes[name]:raise ValueError('host numeric catalog artifact changed: '+name)
 provenance=json.loads((d/'provenance.json').read_text())
 if provenance['full_source_variation']!=S.v.to_json():return None
 if float(S.d.time)!=0. or len(S.actions)!=0:raise ValueError('initial catalog requires original t0 before commands')
 if hashlib.sha256(S.xml.encode()).hexdigest()!=provenance['source_scene_sha256']:raise ValueError('cached geometry scene differs from actual exact initial scene')
 rows={h:{r['id']:r for r in json.loads((d/f'{h}_candidates.json').read_text())} for h in DR.HANDS};out=[]
 for pair in json.loads((d/'paired_candidates.json').read_text()):
  if not pair['original_paired_path_ok']:continue
  chosen={h:rows[h][pair['candidate_ids'][h]] for h in DR.HANDS}
  out.append(dict(candidate_ids=pair['candidate_ids'],plans={h:np.asarray(chosen[h]['path'],float) for h in DR.HANDS},motion_s=pair['issued_motion_clock_s'],optimistic_joint_step_only_lower_bound_s=pair['optimistic_joint_step_only_lower_bound_s'],maximum_TCP_displacement_m=max(np.linalg.norm(chosen[h]['natural_TCP_displacement_m']) for h in DR.HANDS),endpoint_facing={h:chosen[h]['facing'] for h in DR.HANDS},numerical_only_not_physics_admission=True))
 return out
