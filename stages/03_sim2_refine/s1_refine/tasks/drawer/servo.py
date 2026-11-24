"""Drawer-only robot action policy; native physics and actuator model are untouched.

Copied native control_step with one declared robot-policy change: the loaded
open/close translational speed cap is 0.06m/s rather than 0.025m/s. All original
10cm/s hard action envelope, contact logic, coupling, joint and force bounds
remain. This module is used only by the new DrawerSession._control override.
"""
import numpy as np
from rssr_augment.retarget.executor import (
    GRIPPER_SIGNS, MOVE_EVENTS, _angular, _arm_indices, _bounded_velocity,
    _command, _feature, _target, contact_loaded_open_angle, desired_rotation,
)
DRAWER_LOADED_SPEED_M_S = .06
NATIVE_EXECUTOR_SHA256 = "aad5b9e0d3fe4859d1b36f9bb809c61ce203362b4e6392b376ff96f8904d2a0f"

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
    speed_limit=min(speed_limit,DRAWER_LOADED_SPEED_M_S)
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
