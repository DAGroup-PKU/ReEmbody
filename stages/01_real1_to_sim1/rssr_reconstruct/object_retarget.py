"""Temporary RSSR postprocessor for a frozen kinematic Retarget API result.

The imported v22 policy is untouched. Candidate corrections use task-relative
contact geometry, are checked by dynamics, and exported as new target actions.
"""
import pathlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from object_physics import build,commands,hand_pose
from common import write

def phase_offsets(time_s,gripper_command,points):
 """Smooth source-event correction; no case IDs, frame edits or retiming."""
 ts=np.asarray(time_s,float);g=np.asarray(gripper_command,float);out=np.zeros((len(ts),3))
 if not points:return out
 if len(ts)<2 or len(g)!=len(ts) or np.any(np.diff(ts)<=0) or not np.isfinite(ts).all():raise ValueError('Invalid phase clock')
 if set(points)-{'grasp','release'}:raise ValueError('Unknown contact phase')
 values={}
 for name in ['grasp','release']:
  v=np.asarray(points.get(name,[0,0,0]),float)
  if v.shape!=(3,) or not np.isfinite(v).all() or np.max(np.abs(v))>.08:raise ValueError('Invalid phase correction')
  values[name]=v
 closed=np.flatnonzero(g<.1)
 if not len(closed):raise ValueError('Phase repair needs a source closure event')
 close=int(closed[0]);pre=np.flatnonzero(g[:close]>.8);approach=int(pre[-1]) if len(pre) else 0
 opened=np.flatnonzero(g[close:]>.3);release=close+int(opened[0]) if len(opened) else len(ts)-1
 approach_time=min(float(ts[approach]),max(float(ts[0]),float(ts[close])-.75))
 anchors={float(ts[0]):np.zeros(3),approach_time:np.zeros(3),float(ts[close]):values['grasp'],float(ts[release]):values['release']}
 if release<len(ts)-1:anchors[float(ts[-1])]=np.zeros(3)
 ordered=sorted(anchors)
 for i in range(len(ordered)-1):
  a,b=ordered[i:i+2];mask=(ts>=a)&(ts<=b);u=(ts[mask]-a)/(b-a);w=(u*u*(3-2*u))[:,None]
  out[mask]=(1-w)*anchors[a]+w*anchors[b]
 return out


def repair(config,folder,correction):
 folder=pathlib.Path(folder);m=build(config,'sim2',folder/'ik_model');d=mujoco.MjData(m);ts,q,grip=commands(config,'sim2',m);out=q.copy();errors=[]
 offsets=correction.get('hand_offsets_m',{'left':[0,0,0],'right':[0,0,0]});mix=float(correction.get('vertical_grasp_mix',0))
 assert 0<=mix<=1 and all(np.max(np.abs(offsets[s]))<=.16 for s in offsets)
 phases={s:phase_offsets(ts,grip[s],correction.get('phase_hand_offsets_m',{}).get(s,{})) for s in ['left','right']}
 if any(np.max(np.abs(phases[s]+np.asarray(offsets[s])))>.16+1e-10 for s in phases):raise ValueError('Total phase correction exceeds 0.16 m')
 for f,original in enumerate(q):
  d.qpos[:]=original;mujoco.mj_forward(m,d);goals={}
  for side in ['left','right']:
   p,r=hand_pose(m,d,'sim2',side);gp=p+np.asarray(offsets[side])+phases[side][f];x=r[:,0].copy();x[2]=0;x/=np.linalg.norm(x);z=np.array([0,0,-1.]);y=np.cross(z,x);vr=np.c_[x,y,z]
   side_mix=float(correction.get('vertical_grasp_mix_by_hand',{}).get(side,mix));assert 0<=side_mix<=1
   gr=Slerp([0,1],Rotation.from_matrix(np.stack([r,vr])))(side_mix).as_matrix();goals[side]=(gp,gr)
  for side,(gp,gr) in goals.items():
   ids=[m.joint(f'astribot_arm_{side}_joint_{j}').id for j in range(1,8)];qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];lo=m.jnt_range[ids,0]+.01;hi=m.jnt_range[ids,1]-.01
   base=m.body('astribot_gripper_'+side+'_base').id
   for _ in range(90):
    mujoco.mj_forward(m,d);p,r=hand_pose(m,d,'sim2',side);ep=gp-p;er=Rotation.from_matrix(gr@r.T).as_rotvec()
    if np.linalg.norm(ep)<.0004 and np.linalg.norm(er)<.008:break
    jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,p,base);J=np.vstack([jp[:,va],.025*jr[:,va]]);e=np.r_[ep,.025*er];reg=.00001
    dq=np.linalg.solve(J.T@J+reg*np.eye(7),J.T@e+reg*.03*(original[qa]-d.qpos[qa]));d.qpos[qa]=np.clip(d.qpos[qa]+np.clip(dq,-.15,.15),lo,hi)
   mujoco.mj_forward(m,d);p,r=hand_pose(m,d,'sim2',side);errors.append([float(np.linalg.norm(gp-p)),float(np.linalg.norm(Rotation.from_matrix(gr@r.T).as_rotvec()))])
  out[f]=d.qpos
 err=np.array(errors);report={'method':'bounded_cartesian_contact_frame_correction_of_v22_output','max_position_error_m':float(err[:,0].max()),'p95_position_error_m':float(np.quantile(err[:,0],.95)),'max_orientation_error_rad':float(err[:,1].max()),'max_joint_change_rad':float(abs(out-q).max()),'max_adjacent_joint_step_rad':float(abs(np.diff(out,axis=0)).max()),'correction':correction,'v22_modified':False,'object_states_used':True}
 np.savez_compressed(folder/'repaired_commands.npz',time_s=ts,qpos=out,imported_qpos=q);write(folder/'repair_report.json',report);return out,report
