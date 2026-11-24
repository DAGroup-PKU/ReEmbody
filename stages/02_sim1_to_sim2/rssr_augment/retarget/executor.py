"""Functional-point servo with bounded coupled waist and fixed chassis."""
import numpy as np
from .session import EventAction
from .capability_compiler import articulated_feature
GRIPPER_SIGNS={'L1':1,'L11':1,'L2':-1,'R1':1,'R11':-1,'R2':1}
MOVE_EVENTS={'approach','contact','lift','transport','align','open_drawer','close_drawer','push_close','release','clearance','retreat'}
CANONICAL=np.array([[1.,0,0],[0,0,-1.],[0,1.,0]])
def _feature(obj,obs):
 return articulated_feature(obj,obs.get('object_geometry',{}))
def _axis(target,obs):
 meta=obs.get('object_geometry',{}).get(target,{})
 name=meta.get('joint',{}).get('name')
 if name not in obs.get('articulation_axis_world',{}):raise ValueError('Capability gap: target articulation axis missing')
 return obs['articulation_axis_world'][name]
def desired_rotation(obj,obs=None):
 if _feature(obj,obs or {}):
  local=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])@CANONICAL@np.diag([-1.,-1.,1.])
  return obs['object_rotation_world'][obj]@local
 shape=(obs or {}).get('object_geometry',{}).get(obj,{}).get('shape')
 if shape=='box':
  c,s=np.cos(np.pi/12),np.sin(np.pi/12)
  return np.array([[1.,0,0],[0,c,-s],[0,s,c]])@CANONICAL
 return CANONICAL

def _arm_indices(model,hand):
 joints=[model.joint(f'astribot_arm_{hand}_joint_{i}') for i in range(1,8)]
 return np.array([j.qposadr[0] for j in joints]),np.array([j.dofadr[0] for j in joints])

def recovery_target(action,obs):
 hand=np.asarray(obs['finger_center_m'][action.hand],dtype=float)
 other=[c for c in obs.get('gripper_scene_contacts',{}).get(action.hand,[])
        if c['object_id']!=action.object_id]
 if (not other and action.event=='retreat' and
     any(c['object_id']==action.object_id for c in
         obs.get('gripper_scene_contacts',{}).get(action.hand,[]))):
  other=[c for c in obs.get('gripper_scene_contacts',{}).get(action.hand,[])
         if c['object_id']==action.object_id]
 if not other:return hand.copy()
 contact=max(other,key=lambda c:c['penetration_m'])
 away=np.asarray(contact['away_world'],dtype=float)
 mode=action.recovery_mode
 if mode=='away':direction=away
 elif mode=='up':
  up=np.array([0.,0.,1.]);direction=up-np.dot(up,away)*away
  if np.linalg.norm(direction)<1e-6:direction=away
  else:direction=away+direction/np.linalg.norm(direction)
 elif mode in ('tangent_a','tangent_b'):
  direction=np.cross(np.array([0.,0.,1.]),away)
  if np.linalg.norm(direction)<1e-6:direction=np.cross(np.array([0.,1.,0.]),away)
  if mode=='tangent_b':direction=-direction
  # Tangential escape also needs positive separation from the measured
  # contact. A purely tangential servo can keep loading a joint stop.
  direction=away+direction/max(np.linalg.norm(direction),1e-12)
 else:raise ValueError('Unknown contact clearance mode')
 direction=direction/max(np.linalg.norm(direction),1e-12)
 return hand+.04*direction

def overhead_approach(action, obs):
 """A bounded, object-relative staging route for an open gripper.

 The optional route lifts, translates above the target, then descends. It
 uses fresh measured geometry at every control step and never edits physics.
 The RuleAgent and execution agent decide whether to offer and select it.
 """
 hand=np.asarray(obs['finger_center_m'][action.hand],dtype=float)
 target=(np.asarray(obs['object_position_m'][action.object_id],dtype=float)+
         np.asarray(action.target_offset_m,dtype=float))
 size=np.asarray(obs.get('object_geometry',{}).get(action.object_id,{}).get(
     'size_m',(0.,0.,0.)),dtype=float)
 height=float(size[2]) if size.shape==(3,) and np.isfinite(size).all() else 0.
 clearance=min(.12,max(.045,height+.035))
 tolerance=(.003 if obs.get('object_geometry',{}).get(action.object_id,{}).get('shape')=='cup' else .018)
 staging_z=float(target[2]+clearance)
 planar=float(np.linalg.norm(hand[:2]-target[:2]))
 cup=obs.get('object_geometry',{}).get(action.object_id,{}).get('shape')=='cup'
 # An 8mm closed-loop lateral drift during descent must pause/center, not
 # repeatedly lift the hand back to the initial staging height.
 lift_threshold=(.03 if cup else tolerance)
 if planar>lift_threshold and hand[2]<staging_z-.008:
  waypoint=np.array([hand[0],hand[1],staging_z])
  remaining=(staging_z-hand[2])+planar+clearance
 elif planar>tolerance:
  waypoint=np.array([target[0],target[1],float(hand[2]) if cup else max(staging_z,float(hand[2]))])
  remaining=planar+max(clearance,float(hand[2]-target[2]))
 else:
  waypoint=target
  remaining=float(np.linalg.norm(hand-target))
 return waypoint,remaining

def side_corridor_approach(action, obs):
 """Stage an open hand outside a measured fixture's lateral boundary."""
 fixture=action.target_id
 geometry=obs.get('object_geometry',{}).get(fixture,{})
 size=np.asarray(geometry.get('size_m',()),dtype=float)
 if (size.shape!=(3,) or not np.isfinite(size).all() or
     np.any(size<=0) or geometry.get('mobility')!='static'):
  raise ValueError('Side corridor requires a bounded static fixture')
 hand=np.asarray(obs['finger_center_m'][action.hand],dtype=float)
 target=np.asarray(obs['object_position_m'][action.object_id],dtype=float)
 center=np.asarray(obs['object_position_m'][fixture],dtype=float)
 rotation=np.asarray(obs['object_rotation_world'][fixture],dtype=float)
 hand_local=rotation.T@(hand-center)
 target_local=rotation.T@(target-center)
 side=(1. if target_local[0]>.01 else -1. if target_local[0]<-.01
       else 1. if hand_local[0]>=0 else -1.)
 opening=float(obs['gripper_opening_m'][action.hand])
 margin=float(np.clip(.5*opening+.03,.06,.11))
 boundary=size[0]/2+margin
 waypoint_local=hand_local.copy()
 waypoint_local[0]=side*max(side*hand_local[0],boundary)
 waypoint=center+rotation@waypoint_local
 return waypoint,float(np.linalg.norm(waypoint-hand))

def front_staging_transport(action, obs):
 """Stage a held object outside a measured container opening."""
 container=action.target_id
 geometry=obs.get('object_geometry',{}).get(container,{})
 payload=obs.get('object_geometry',{}).get(action.object_id,{})
 joint=(geometry.get('joint') or {}).get('name')
 size=np.asarray(geometry.get('size_m',()),dtype=float)
 payload_size=np.asarray(payload.get('size_m',()),dtype=float)
 if (joint not in obs.get('articulation_axis_world',{}) or
     size.shape!=(3,) or payload_size.shape!=(3,) or
     not np.isfinite(size).all() or not np.isfinite(payload_size).all() or
     np.any(size<=0) or np.any(payload_size<=0)):
  raise ValueError('Front staging requires bounded articulated geometry')
 axis=np.asarray(obs['articulation_axis_world'][joint],dtype=float)
 axis/=max(float(np.linalg.norm(axis)),1e-12)
 if abs(float(axis[2]))>.5:
  raise ValueError('Opening axis is not suitable for horizontal staging')
 center=np.asarray(obs['object_position_m'][container],dtype=float)
 object_pos=np.asarray(obs['object_position_m'][action.object_id],dtype=float)
 hand=np.asarray(obs['finger_center_m'][action.hand],dtype=float)
 rotation=np.asarray(obs['object_rotation_world'][container],dtype=float)
 object_rotation=np.asarray(obs['object_rotation_world'][action.object_id],dtype=float)
 half_depth=.5*float(np.sum(np.abs(rotation.T@axis)*size))
 payload_half_depth=.5*float(np.sum(np.abs(object_rotation.T@axis)*payload_size))
 opening=float(obs['gripper_opening_m'][action.hand])
 forward_clearance=float(np.clip(.5*opening+.04,.06,.11))
 front_depth=half_depth+payload_half_depth+forward_clearance
 up=np.array([0.,0.,1.])
 rim_height=(float(center[2])+
             .5*float(np.sum(np.abs(rotation.T@up)*size))+
             .5*float(np.sum(np.abs(object_rotation.T@up)*payload_size))+.03)
 if action.recovery_mode=='front_clearance':
  current_depth=float(np.dot(object_pos-center,axis))
  waypoint=object_pos+max(0.,front_depth-current_depth)*axis
  waypoint[2]=max(float(waypoint[2]),rim_height)
 elif action.recovery_mode=='front_align':
  waypoint=center+front_depth*axis
  waypoint[2]=rim_height
 else:
  raise ValueError('Unknown front staging mode')
 return waypoint+(hand-object_pos),float(np.linalg.norm(waypoint-object_pos))

def _target(action,obs):
 hand=obs['finger_center_m'][action.hand];obj=obs['object_position_m'][action.object_id]
 offset=np.asarray(action.target_offset_m)
 if action.event in ('clearance','retreat'):
  pinned=obs.get('recovery_target_m')
  return np.asarray(pinned,dtype=float) if pinned is not None else recovery_target(action,obs)
 if action.event in ('approach','contact','grasp'):
  if (action.event in ('approach','contact','grasp') and
      obs.get('object_geometry',{}).get(action.object_id,{}).get('shape')=='cup'):
   from dataclasses import replace
   return overhead_approach(replace(action,target_offset_m=(0.,0.,0.)),obs)[0]
  if action.event=='approach' and action.recovery_mode=='side_corridor':
   return side_corridor_approach(action,obs)[0]
  if action.event=='approach' and action.recovery_mode=='overhead':
   return overhead_approach(action,obs)[0]
  if action.event=='approach' and _feature(action.object_id,obs):
   rotation=obs['hand_rotation_world'][action.hand]
   error=float(np.arccos(np.clip((np.trace(desired_rotation(action.object_id,obs).T@rotation)-1)/2,-1.,1.)))
   if error>.15:
    parent=obs['object_geometry'][action.object_id]['parent_id']
    return obj+offset+.094*_axis(parent,obs)+np.array([0.,0.,.08])
  return obj+offset
 if action.event=='lift':
  if action.target_id is not None:
   return np.array([hand[0],hand[1],obs['object_position_m'][action.target_id][2]+offset[2]+.003+(hand-obj)[2]])
  return hand+np.array([0.,0.,.04])
 if action.event=='release':
  if _feature(action.object_id,obs):
   parent=obs['object_geometry'][action.object_id]['parent_id']
   return obj+.10*_axis(parent,obs)
  if hand[2]-obj[2]<.06:return np.array([hand[0],hand[1],obj[2]+.13])
  sign=1. if action.hand=='left' else -1.
  return obj+np.array([sign*.22,.04,.16])
 if action.event in ('transport','align'):
  if action.event=='transport' and action.recovery_mode in ('front_clearance','front_align'):
   return front_staging_transport(action,obs)[0]
  target=obs['object_position_m'][action.target_id]+offset
  geometry=obs.get('object_geometry',{})
  source_geometry=geometry.get(action.object_id,{})
  target_geometry=geometry.get(action.target_id,{})
  # Transit above a physical support object using measured world-space
  # bounds. Container insertion and abstract region targets retain their
  # existing contract-compiled paths below.
  if (source_geometry.get('size_m') is not None and
      target_geometry.get('size_m') is not None and
      target_geometry.get('role') not in ('container','target_region') and
      target_geometry.get('mobility') not in ('slide','hinge')):
   source_half=float(np.sum(np.abs(obs['object_rotation_world'][action.object_id][2,:])*np.asarray(source_geometry['size_m'])/2))
   target_half=float(np.sum(np.abs(obs['object_rotation_world'][action.target_id][2,:])*np.asarray(target_geometry['size_m'])/2))
   clear=float(obs['object_position_m'][action.target_id][2]+target_half+source_half+.020)
   if np.linalg.norm((target-obj)[:2])>.003:
    if obj[2]<clear-.004:target[:2]=obj[:2]
    target[2]=max(float(target[2]),clear)
   return target+(hand-obj)
  if np.linalg.norm((target-obj)[:2])>.035:target[2]=max(float(obj[2]),float(target[2]+.07))
  return target+(hand-obj)
 axis=_axis(action.target_id,obs)
 anchor=obs.get('loaded_contact_anchor_local_m')
 if action.event in ('open_drawer','close_drawer') and anchor is not None:
  anchor=np.asarray(anchor,dtype=float)
  if anchor.shape!=(3,) or not np.isfinite(anchor).all():
   raise ValueError('Invalid feature-relative contact anchor')
  pull=float(obs.get('pull_offset_m',.005))
  if not np.isfinite(pull) or not 0.<pull<=.02:
   raise ValueError('Invalid bounded contact pull offset')
  obj_rotation=obs['object_rotation_world'][action.object_id]
  direction=1. if action.event=='open_drawer' else -1.
  return (obj+obj_rotation@anchor+direction*pull*axis)
 return hand+(1 if action.event=='open_drawer' else -1)*.025*axis

def _angular(current,target):
 v=sum(np.cross(current[:,i],target[:,i]) for i in range(3))*1.5
 n=np.linalg.norm(v)
 return v*min(1.,.6/max(n,1e-12))
def _bounded_velocity(jac,goal,actual,bounds,margins=None):
 available=np.ones(len(actual),dtype=bool);v=np.zeros(len(actual))
 margins=(np.full(len(actual),.08) if margins is None else
          np.asarray(margins,dtype=float))
 for _ in range(len(actual)+1):
  active=jac[:,available];v[:]=0
  v[available]=active.T@np.linalg.solve(active@active.T+.015**2*np.eye(6),goal)
  # Leave room for physical tracking error near any limited joint. The
  # controller may otherwise command an in-range target while the robot
  # drifts through the hard boundary during a long hold.
  blocked=((actual<bounds[:,0]+margins)&(v<0))|((actual>bounds[:,1]-margins)&(v>0))
  if not np.any(blocked&available):break
  available&=~blocked
 return np.clip(v,-1.2,1.2)
def _command(q,dq,qa,v,actual,bounds,dt,clip_margins=None):
 new=np.clip(q[qa]+v*dt,actual-.18,actual+.18)
 clip_margins=(np.full(len(actual),.07) if clip_margins is None else
               np.asarray(clip_margins,dtype=float))
 new=np.clip(new,bounds[:,0]+clip_margins,bounds[:,1]-clip_margins)
 dq[qa]=np.clip((new-q[qa])/dt,-1.2,1.2);q[qa]=new

def contact_loaded_open_angle(requested,actual,contact_count,lower,upper):
 """Leave joint travel and limit target lead while physical contact loads a gripper."""
 span=upper-lower
 target=max(requested,actual-.07*span) if requested < actual else requested
 return max(target,lower+.08*span) if contact_count > 0 and requested < lower+.08*span else target

def control_step(action,obs,q,dq,model):
 if not 0<action.max_speed_m_s<=.1 or not np.isfinite(action.target_offset_m).all() or np.max(np.abs(action.target_offset_m))>.20:raise ValueError('Invalid motion envelope')
 if action.event=='reposition_base':
  raise ValueError('S1 chassis translation and yaw are disabled')
  if action.hand not in ('left','right') or action.object_id is None or action.target_id is not None:
   raise ValueError('Invalid base event binding')
  target=np.asarray(action.base_joint_target,dtype=float)
  order=action.base_axis_order
  if target.shape!=(3,) or not np.isfinite(target).all() or sorted(order or ())!=[0,1,2]:
   raise ValueError('Invalid certified base target or axis order')
  q=q.copy();dq=dq.copy();dt=float(model.opt.timestep)
  joints=[model.joint(name) for name in ('astribot_chassis_x','astribot_chassis_y',
                                         'astribot_chassis_zrot')]
  qa=np.asarray([int(j.qposadr[0]) for j in joints])
  actual=np.asarray(obs['base_joint_qpos'],dtype=float)
  if actual.shape!=(3,) or not np.isfinite(actual).all():
   raise ValueError('Invalid measured base state')
  for axis in order:
   tolerance=.004 if axis<2 else .01
   if abs(actual[axis]-target[axis])<=tolerance and abs(q[qa[axis]]-target[axis])<=tolerance:
    continue
   lo,hi=model.jnt_range[joints[axis].id]
   if not lo+.005<=target[axis]<=hi-.005:
    raise ValueError('Base target violates joint travel margin')
   speed=action.max_speed_m_s if axis<2 else .25
   previous=float(q[qa[axis]])
   q[qa[axis]]=previous+np.clip(target[axis]-previous,-speed*dt,speed*dt)
   dq[qa[axis]]=(q[qa[axis]]-previous)/dt
   break
  # The certified base path held both arms at their initial joint positions.
  # Do not add a hidden world-frame hand compensation command here.
  return q,dq
 if action.event not in MOVE_EVENTS|{'grasp','hold'} or action.hand not in ('left','right'):raise ValueError('Unknown motion event or hand')
 q=q.copy();dq=dq.copy();dt=float(model.opt.timestep)
 waist=[model.joint(f'astribot_torso_joint_{i}') for i in (1,2)]
 wqa=np.asarray([int(j.qposadr[0]) for j in waist])
 wva=np.asarray([int(j.dofadr[0]) for j in waist])
 if obs.get('support_lock_waist',False):
  # A measured second-hand support contact may need the shared waist held
  # while the active manipulator moves. Preserve the native coupled posture.
  q[wqa]=obs['robot_qpos'][wqa]
  dq[wqa]=0.
 initial_waist=np.asarray(obs['torso_initial_qpos'],dtype=float)
 coupling=float(initial_waist.sum())
 waist_lo=max(float(model.jnt_range[waist[0].id,0])+.07,
              coupling-float(model.jnt_range[waist[1].id,1])+.07)
 waist_hi=min(float(model.jnt_range[waist[0].id,1])-.07,
              coupling-float(model.jnt_range[waist[1].id,0])-.07)
 if not waist_lo<waist_hi:raise ValueError('Coupled waist has no permitted motion')
 waist_velocity=0.
 move=action.event in MOVE_EVENTS
 cup_pickup=(action.event in ('contact','grasp') and
             obs.get('object_geometry',{}).get(action.object_id,{}).get('shape')=='cup')
 cup_position_error=float(np.linalg.norm(np.asarray(obs['finger_center_m'][action.hand])-
     np.asarray(obs['object_position_m'].get(action.object_id,obs['finger_center_m'][action.hand]))))
 cup_rotation_error=float(np.arccos(np.clip((np.trace(desired_rotation(action.object_id,obs).T@
     obs['hand_rotation_world'][action.hand])-1.)/2.,-1.,1.)))
 cup_aligned=(cup_position_error<=.004 and cup_rotation_error<=.10)
 if cup_pickup:move=True
 target_fingers=(obs.get('finger_contact',{}).get(action.object_id,{})
                 .get(action.hand,(False,False)))
 one_sided_settle=(action.event=='contact' and
                   ((len(target_fingers)==2 and sum(target_fingers)==1) or
                    'contact_settle_anchor_qpos' in obs))
 if cup_pickup and not all(target_fingers):one_sided_settle=False
 if cup_pickup and all(target_fingers):move=False
 if one_sided_settle:
  # A contacting pad supplies a physical anchor. Pause wrist and
  # coupled waist motion while the other finger closes.
  move=False
  arm_qa,_=_arm_indices(model,action.hand)
  anchor=np.asarray(obs.get('contact_settle_anchor_qpos',obs['robot_qpos']))
  q[arm_qa]=anchor[arm_qa];dq[arm_qa]=0.
  q[wqa[0]]=anchor[wqa[0]]
  q[wqa[1]]=coupling-q[wqa[0]];dq[wqa]=0.
 if action.event=='release':
  supported=bool(obs['support_contact'][action.object_id]) or _feature(action.object_id,obs)
  move=supported and (not all(obs['finger_contact'][action.object_id][action.hand]) or obs['gripper_opening_m'][action.hand]>.075)
 if move:
  qa,va=_arm_indices(model,action.hand)
  joints=[model.joint(f'astribot_arm_{action.hand}_joint_{i}') for i in range(1,8)]
  bounds=np.array([model.jnt_range[j.id] for j in joints])
  actual=obs['robot_qpos'][qa]
  plan=obs.get('contact_ik_plan',{})
  plan_waist=np.asarray(plan.get('torso_qpos_at_plan_rad',()),dtype=float)
  use_plan=(action.event=='contact' and plan.get('status')=='planned' and
            plan.get('object_id')==action.object_id and
            plan.get('hand')==action.hand and
            plan_waist.shape==(2,) and
            (plan.get('planning_mode')=='full_pose_coupled_waist' or
             np.max(np.abs(obs['robot_qpos'][wqa]-plan_waist))<=.01) and
            np.linalg.norm(np.asarray(obs['object_position_m'][action.object_id])-
                           np.asarray(plan['object_center_m']))<=.005)
  if use_plan:
   target=np.asarray(plan['target_arm_qpos_rad'],dtype=float)
   if target.shape!=(7,) or not np.isfinite(target).all():
    raise ValueError('Contact IK target is invalid')
   full_pose=plan.get('planning_mode')=='full_pose_coupled_waist'
   if full_pose:
    target_waist=np.asarray(plan['target_waist_qpos_rad'],dtype=float)
    if (target_waist.shape!=(2,) or not np.isfinite(target_waist).all() or
        abs(target_waist.sum()-coupling)>.001):
     raise ValueError('Contact IK waist target is invalid')
    qa=np.r_[qa,wqa[0]]
    actual=np.r_[actual,obs['robot_qpos'][wqa[0]]]
    bounds=np.vstack([bounds,[waist_lo,waist_hi]])
    target=np.r_[target,target_waist[0]]
   v=np.clip(2.*(target-actual),-.8,.8)
   if full_pose:v[-1]=np.clip(v[-1],-.25,.25)
   jac=obs['hand_jacobian'][action.hand]
   linear=jac[:,va]@v[:7]
   if full_pose:linear+=(jac[:,wva[0]]-jac[:,wva[1]])*v[-1]
   speed=float(np.linalg.norm(linear))
   if speed>action.max_speed_m_s:
    v*=action.max_speed_m_s/speed
  else:
   error=_target(action,obs)-obs['finger_center_m'][action.hand]
   linear=error*3.;norm=np.linalg.norm(linear)
   speed_limit=action.max_speed_m_s
   # A load-bearing hand can lose its grasp when a carry starts with the same
   # speed as a free-space approach. Reduce only the selected short event;
   # live contact and the native hard monitor remain authoritative.
   if (action.event in ('lift','transport','align','clearance') and
       action.object_id in obs['finger_contact'] and
       all(obs['finger_contact'][action.object_id][action.hand])):
    speed_limit=min(speed_limit,.04 if action.event=='transport' else .02)
   # Contact-loaded articulation can slip when the wrist outruns the native
   # joint. This uses only the measured contact and selected primitive.
   if (action.event in ('open_drawer','close_drawer') and
       action.object_id in obs['finger_contact'] and
       any(obs['finger_contact'][action.object_id][action.hand])):
    speed_limit=min(speed_limit,.025)
   if norm>speed_limit:linear*=speed_limit/norm
   # Preserve the measured hand orientation while transporting a loaded cup.
   # Rotating toward the free-space canonical pose after closure shears the
   # tapered wall through the finger pads and can eject a first valid grasp.
   # This changes only robot motion targets; native contacts/forces stay intact.
   rotation_target=desired_rotation(action.object_id,obs)
   if (action.event in ('lift','transport','align') and
       obs.get('object_geometry',{}).get(action.object_id,{}).get('shape')=='cup' and
       all(obs['finger_contact'][action.object_id][action.hand])):
    rotation_target=np.asarray(obs.get('hold_anchors',{}).get(action.hand,{}).get(
        'rotation',obs['hand_rotation_world'][action.hand]))
   angular=(np.zeros(3) if action.event in ('clearance','retreat') else
            _angular(obs['hand_rotation_world'][action.hand],rotation_target))
   jp=obs['hand_jacobian'][action.hand];jr=obs['hand_rotation_jacobian'][action.hand]
   arm_jac=np.vstack([jp[:,va],.12*jr[:,va]])
   goal=np.r_[linear,.12*angular]
   arm_v=_bounded_velocity(arm_jac,goal,actual,bounds)
   need_waist=(np.linalg.norm(jp[:,va]@arm_v-linear)>.004 and
               not obs.get('support_lock_waist',False))
   if need_waist:
    jac=np.vstack([np.column_stack([jp[:,va],jp[:,wva[0]]-jp[:,wva[1]]]),
                   .12*np.column_stack([jr[:,va],jr[:,wva[0]]-jr[:,wva[1]]])])
    qa=np.r_[qa,wqa[0]]
    bounds=np.vstack([bounds,[waist_lo,waist_hi]])
    actual=obs['robot_qpos'][qa]
    v=_bounded_velocity(jac,goal,actual,bounds,
                        margins=np.r_[np.full(7,.08),.005])
    v[-1]=np.clip(v[-1],-.25,.25)
   else:
    v=arm_v
  _command(q,dq,qa,v,actual,bounds,dt,
           clip_margins=(np.r_[np.full(7,.07),0.] if use_plan and
                         plan.get('planning_mode')=='full_pose_coupled_waist'
                         else np.r_[np.full(7,.07),.005] if not use_plan and
                         need_waist else None))
  if use_plan and plan.get('planning_mode')=='full_pose_coupled_waist':
   q[wqa[1]]=coupling-q[wqa[0]]
   dq[wqa[1]]=-dq[wqa[0]]
   waist_velocity=float(dq[wqa[0]])
  if not use_plan and need_waist:
   q[wqa[1]]=coupling-q[wqa[0]]
   dq[wqa[1]]=-dq[wqa[0]]
   waist_velocity=float(dq[wqa[0]])
 # Keep the unused hand at its world pose while the active arm moves.
 inactive='right' if action.hand=='left' else 'left'
 anchor=obs.get('hold_anchors',{}).get(inactive)
 if anchor is not None:
  qa,va=_arm_indices(model,inactive);jp=obs['hand_jacobian'][inactive];jr=obs['hand_rotation_jacobian'][inactive]
  jac=np.vstack([jp[:,va],.12*jr[:,va]])
  position=(np.asarray(anchor['position'])-obs['finger_center_m'][inactive])*3.
  n=np.linalg.norm(position)
  if n>.08:position*=.08/n
  angular=_angular(obs['hand_rotation_world'][inactive],np.asarray(anchor['rotation']))
  joints=[model.joint(f'astribot_arm_{inactive}_joint_{i}') for i in range(1,8)]
  bounds=np.array([model.jnt_range[j.id] for j in joints]);actual=obs['robot_qpos'][qa]
  shared=np.r_[jp[:,wva[0]]-jp[:,wva[1]],
               .12*(jr[:,wva[0]]-jr[:,wva[1]])]*waist_velocity
  v=_bounded_velocity(jac,np.r_[position,.12*angular]-shared,actual,bounds)
  _command(q,dq,qa,v,actual,bounds,dt)
 if action.event in ('grasp','release') or action.gripper_fraction is not None:
  fraction=0. if action.event=='grasp' else 1. if action.event=='release' else action.gripper_fraction
  if fraction is None or not 0<=fraction<=1:raise ValueError('Invalid gripper fraction')
  angle=.02+(1.-fraction)*.89
  if action.event=='approach' and _feature(action.object_id,obs):
   # Reserve distance from the open joint stop while the wrist approaches.
   angle=max(.08,angle)
  if action.event=='contact' and _feature(action.object_id,obs) and not all(obs['finger_contact'][action.object_id][action.hand]):
   qa=int(model.joint(f'astribot_gripper_{action.hand}_joint_L1').qposadr[0])
   rate=.60 if one_sided_settle else .15
   angle=min(.91,float(q[qa])+rate*dt)
   if not one_sided_settle:
    angle=min(angle,float(obs['robot_qpos'][qa])+.015)
  if cup_pickup and not all(target_fingers):
   # Center and orient with open jaws before the first intentional closure.
   # Close gradually only after the actual midpoint reaches the cup axis.
   qa=int(model.joint(f'astribot_gripper_{action.hand}_joint_L1').qposadr[0])
   angle=(min(.91,float(q[qa])+.30*dt) if cup_aligned else .02)
  if fraction<.5 and action.object_id is not None and all(obs['finger_contact'][action.object_id][action.hand]):
   # Command a small closure error after physical contact. Native actuator
   # force cap and collision dynamics are unchanged; no state or force write.
   angle=float(obs['robot_qpos'][model.joint(f'astribot_gripper_{action.hand}_joint_L1').qposadr[0]])+.006
  proximal=model.joint(f'astribot_gripper_{action.hand}_joint_L1')
  lower,upper=model.jnt_range[proximal.id]
  actual=float(obs['robot_qpos'][proximal.qposadr[0]])
  angle=contact_loaded_open_angle(
   angle,actual,len(obs.get('gripper_scene_contacts',{}).get(action.hand,[])),
   float(lower),float(upper))
  for suffix,sign in GRIPPER_SIGNS.items():
   joint=model.joint(f'astribot_gripper_{action.hand}_joint_{suffix}');qa=int(joint.qposadr[0]);lo,hi=model.jnt_range[joint.id]
   q[qa]=np.clip(sign*angle,lo+.02,hi-.02);dq[qa]=0
 return q,dq
