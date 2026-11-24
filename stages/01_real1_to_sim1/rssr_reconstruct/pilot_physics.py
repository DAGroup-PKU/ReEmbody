"""Pilot copy of the declared physical tool, with every-step all-joint auditing.

Source geometry is reconstructed from action contact opportunities; object sizes
are visual priors, not measured geometry. This backend supports box-in-open-box.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,itertools,json,pathlib,time,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from common import read,write,sha
from rssr_errors import CandidateInvalid

SIGNS=np.array(list(itertools.product([-1.,1.],repeat=3)))
LINK={'L1':1,'L11':1,'L2':-1,'R1':1,'R11':-1,'R2':1}
GRIPPER_COMMAND_MARGIN_RAD=.02


def guarded_gripper_angles(angles,limits):
 # A fixed position-command stop avoids servo overshoot at the mechanical
 # joint stops. This changes the desired command, never executed qpos, joint
 # bounds, contact forces or the success tolerance. Contact still sets the
 # achieved opening under the existing finite 2 Nm force cap.
 lo,hi=np.asarray(limits,float)
 if not np.isfinite([lo,hi]).all() or hi-lo<=2*GRIPPER_COMMAND_MARGIN_RAD:
  raise ValueError('Gripper limits cannot accommodate the declared command guard')
 value=np.asarray(angles,float)
 if not np.isfinite(value).all():raise ValueError('Nonfinite gripper command')
 return np.clip(value,lo+GRIPPER_COMMAND_MARGIN_RAD,hi-GRIPPER_COMMAND_MARGIN_RAD)
def vec(x):return ' '.join(f'{v:.10g}' for v in np.asarray(x).ravel())
def quat(r):q=Rotation.from_matrix(r).as_quat();return vec(np.r_[q[3],q[:3]])
def cam(parent,name,pos,look):
 p=np.asarray(pos);f=np.asarray(look)-p;f/=np.linalg.norm(f);r=np.cross(f,[0,0,1]);r/=np.linalg.norm(r);u=np.cross(r,f)
 ET.SubElement(parent,'camera',name=name,pos=vec(p),quat=quat(np.c_[r,u,-f]),fovy='55')

def source_model(config):
 root=pathlib.Path(config['model_root']);src=root/'source/models/trackc/piper_isaac_sim/piper_description/mujoco_model/piper_description.xml'
 original=ET.parse(src).getroot();contract=read(root/'source/scene/task_frame_contract.json')
 x=ET.Element('mujoco',model='rssr_source_proxy');ET.SubElement(x,'compiler',angle='radian')
 ET.SubElement(x,'option',timestep='.002',integrator='implicitfast',iterations='100',tolerance='1e-9')
 asset=ET.SubElement(x,'asset');world=ET.SubElement(x,'worldbody');acts=ET.SubElement(x,'actuator')
 for mesh in original.findall('asset/mesh'):
  e=copy.deepcopy(mesh);e.set('file',str((src.parent/e.get('file')).resolve()));asset.append(e)
 for side in ['left','right']:
  tf=np.array(contract['world_from_task'])@np.array(contract['chains'][side]['task_from_source_base'])
  mount=ET.SubElement(world,'body',name=side+'_base',pos=vec(tf[:3,3]),quat=quat(tf[:3,:3]))
  for element in original.find('worldbody'):
   e=copy.deepcopy(element)
   for node in e.iter():
    if node.get('name'):node.set('name',side+'_'+node.get('name'))
    if node.tag=='geom':
     node.set('name',side+'_'+node.get('mesh','base')+'_geom');node.set('rgba','.09 .10 .11 1');node.set('contype','1');node.set('conaffinity','6')
     node.set('friction','1 .01 .002');node.set('condim','4');node.set('solref','.006 1')
    if node.tag=='joint':
     node.set('damping','2')
     if node.get('name').endswith(('joint7','joint8')):
      node.set('damping','10');node.set('range','0 .0475' if node.get('name').endswith('joint7') else '-.0475 0')
   mount.append(e)
  for element in original.find('actuator'):
   a=copy.deepcopy(element);a.set('name',side+'_'+a.get('name'));a.set('joint',side+'_'+a.get('joint'))
   if a.get('name').endswith(('joint7','joint8')):a.set('kp','1000');a.set('forcerange','-12 12')
   else:
    # Fixed control adapter, independent of this episode's task outcome.
    a.set('kp','1000');a.set('forcerange','-100 100')
   acts.append(a)
  flange=mount.find(".//body[@name='"+side+"_link6']")
  ET.SubElement(flange,'site',name='arm_'+side+'_tool',pos='0 0 .133669403578')
  # Estimated source wrist extrinsics, explicitly not image-calibrated.
  ET.SubElement(flange,'camera',name=side+'_wrist_rgbd',pos='0 .04 .07',xyaxes='1 0 0 0 -1 0',fovy='60')
 table=copy.deepcopy(ET.parse(root/'source/scene/s1_tabletop.xml').getroot().find(".//body[@name='worktable']"));world.append(table)
 ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .1',rgba='.5 .5 .5 1',contype='2',conaffinity='13')
 cam(world,'rssr_head',[0,-.32,1.45],[0,-.735,.78]);return x

def target_model(config):
 src=pathlib.Path(config['import_case'])/'sim2_robot_only_scene.xml';x=ET.parse(src).getroot()
 # Imported mesh paths are already absolute. Preserve waist constraints/model.
 x.find('option').set('integrator','implicitfast');x.find('option').set('timestep','.002')
 for group in x.findall('actuator'):
  for a in group:
   if 'gripper' in a.get('name',''):
    a.set('forcerange','-2 2');a.set('biasprm','0 -500 -1')
 # The inherited display model used 5 Nm s/rad at each tiny driving finger
 # joint: with a finite 2 Nm tendon cap it took several seconds to close.
 # Use one declared small-link servo prior for both hands, then contact decides
 # the achieved aperture. These are simulation parameters, not hardware claims.
 for j in x.iter('joint'):
  if 'gripper' in j.get('name',''):
   j.set('damping','.05');j.set('armature','.0002')
 for g in x.iter('geom'):
  if 'gripper' in g.get('name',''):
   g.set('friction','1 .01 .002');g.set('condim','4');g.set('solref','.006 1')
 coupled=any({e.get('joint1'),e.get('joint2')}=={'astribot_torso_joint_1','astribot_torso_joint_2'} for e in x.findall('./equality/joint'))
 if not coupled:
  eq=ET.SubElement(x,'equality');ET.SubElement(eq,'joint',name='rssr_waist_coupling',joint1='astribot_torso_joint_1',joint2='astribot_torso_joint_2',polycoef='-.28403 -1 0 0 0',solref='.004 1',solimp='.99 .999 .001')
 return x

def build(config,kind,folder,scene=None):
 expected=config.get('mujoco_version_expected')
 if expected and mujoco.__version__!=expected:raise ValueError('MuJoCo version differs from frozen experiment contract')
 if scene and scene.get('schema')=='rssr.scene.v2':
  from rssr_scene import build as generic_build
  return generic_build(config,kind,folder,scene)
 x=source_model(config) if kind=='sim1' else target_model(config);world=x.find('worldbody')
 visual=x.find('visual')
 if visual is None:visual=ET.SubElement(x,'visual')
 glob=visual.find('global')
 if glob is None:glob=ET.SubElement(visual,'global')
 glob.set('offwidth','640');glob.set('offheight','480')
 ET.SubElement(world,'light',pos='0 -.7 2.5',dir='0 0 -1',diffuse='.7 .7 .7')
 for g in x.iter('geom'):
  if g.get('name')=='worktable_top':g.set('rgba','.88 .88 .86 1')
  if g.get('name','').startswith('physical_robot_'):g.set('rgba','0 0 0 0')
 if scene:
  dz=float(scene.get('table_shift_z_m',0))
  table=world.find("body[@name='worktable']");p=np.fromstring(table.get('pos'),sep=' ');p[2]+=dz;table.set('pos',vec(p))
  for obj in scene['objects']:
   b=ET.SubElement(world,'body',name=obj['id'],pos=vec(obj['position_m']),euler=vec([0,0,obj.get('yaw_rad',0)]));ET.SubElement(b,'freejoint',name=obj['id']+'_free')
   friction=np.asarray(obj.get('contact_friction',[1.,.01,.002]),dtype=float)
   contact_dim=obj.get('contact_condim',4)
   if type(contact_dim) is not int or contact_dim not in (4,6):raise ValueError('Object contact_condim must be 4 or 6')
   # Declared simulation contact prior, not a measured material coefficient.
   # The repair agent may propose values within these fixed physical bounds;
   # source actions, servo limits, object masses and free-body dynamics stay fixed.
   if friction.shape!=(3,) or not np.isfinite(friction).all() or np.any(friction<np.array([.3,.003,0.])) or np.any(friction>np.array([2.,.02,.005])):
    raise ValueError('Object contact_friction outside declared simulation bounds')
   # condim=6 additionally enables the bounded rolling-friction coefficient;
   # it remains a finite contact wrench, never a kinematic grasp constraint.
   size=np.asarray(obj['size_m']);attrs=dict(type='box',contype='4',conaffinity='15',condim=str(contact_dim),friction=vec(friction),solref='.006 1',solimp='.95 .99 .001',rgba=vec(obj['rgba']))
   if obj['shape']=='box':ET.SubElement(b,'geom',name=obj['id']+'_geom',size=vec(size/2),mass=str(obj.get('mass_kg',.04)),**attrs)
   elif obj['shape']=='open_box':
    sx,sy,h=size;t=float(obj['wall_m']);mass=obj.get('mass_kg',.12)
    parts=[('bottom',[0,0,-h/2+t/2],[sx/2,sy/2,t/2]),('xplus',[sx/2-t/2,0,0],[t/2,sy/2,h/2]),('xminus',[-sx/2+t/2,0,0],[t/2,sy/2,h/2]),('yplus',[0,sy/2-t/2,0],[sx/2-t,t/2,h/2]),('yminus',[0,-sy/2+t/2,0],[sx/2-t,t/2,h/2])]
    for name,p,s in parts:ET.SubElement(b,'geom',name=obj['id']+'_'+name,pos=vec(p),size=vec(s),mass=str(mass/5),**attrs)
   else:raise ValueError('Unsupported shape')
 folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'scene.xml';ET.indent(x);path.write_text(ET.tostring(x,encoding='unicode'))
 return mujoco.MjModel.from_xml_path(str(path))

def commands(config,kind,m):
 src=np.load(pathlib.Path(config['import_case'])/'source_action_state.npz')
 z=np.load(pathlib.Path(config['import_case'])/'agent_trajectory_full.npz') if kind=='sim2' else None
 ts=z['source_time_s'] if z is not None else src['timestamp']-src['timestamp'][0];n=len(ts)
 d=mujoco.MjData(m);q=np.tile(d.qpos,(n,1));grip={s:np.clip(src['action'][:,13 if s=='left' else 6]/1.05,0,1) for s in ['left','right']}
 for j in range(m.njnt):
  name=m.joint(j).name;qa=m.jnt_qposadr[j]
  if kind=='sim1' and name.startswith(('left_joint','right_joint')):
   side=name.split('_')[0];num=int(name.split('joint')[1]);off=7 if side=='left' else 0
   if num<=6:q[:,qa]=src['action'][:,off+num-1]
   else:q[:,qa]=.0475*grip[side]*(1 if num==7 else -1)
  elif kind=='sim2':
   if name.startswith('astribot_arm_'):q[:,qa]=z[name.split('_')[2]+'_q'][:,int(name[-1])-1]
   elif name.startswith('astribot_torso_joint_'):q[:,qa]=z['upstream_q'][:,int(name[-1])-1]
 # Inverse target linkage aperture mapping is geometry based.
 if kind=='sim2':
  angles=np.linspace(0,.93,187);gaps=[]
  for a in angles:
   for s in LINK:d.qpos[m.joint('astribot_gripper_left_joint_'+s).qposadr[0]]=LINK[s]*a
   mujoco.mj_forward(m,d);gaps.append(aperture(m,d,'left'))
  for side in ['left','right']:
   width=.095 if config.get('interaction_revision')=='v16' else .07
   from rssr_model_tools import enabled
   if enabled(config):width*=config.get('model_profile',{}).get('gripper_opening_scale',1.)
   # Source opening is 2 * .0475 m. Saturate at the target's geometric
   # opening, then apply the same existing angle guard. No joint limit or
   # force cap is changed and achieved opening remains contact determined.
   av=np.interp(width*grip[side],gaps[::-1],angles[::-1])
   av=guarded_gripper_angles(av,m.jnt_range[m.joint('astribot_gripper_'+side+'_joint_L1').id])
   for suffix,sg in LINK.items():q[:,m.joint(f'astribot_gripper_{side}_joint_{suffix}').qposadr[0]]=sg*av
 return ts,q,grip

def aperture(m,d,side):
 b=m.body('astribot_gripper_'+side+'_base').id;rot=d.xmat[b].reshape(3,3);pos=d.xpos[b];vs=[]
 for suffix in ['L112','R112']:
  g=m.geom('astribot_gripper_'+side+'_geom_'+suffix).id;v=(SIGNS*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];vs.append((v-pos)@rot)
 return float(vs[0][:,0].min()-vs[1][:,0].max())

def hand_pose(m,d,kind,side):
 if kind=='sim2':
  gids=[m.geom('astribot_gripper_'+side+'_geom_'+v).id for v in ['L112','R112']];p=np.mean(d.geom_xpos[gids],axis=0);r=d.xmat[m.body('astribot_gripper_'+side+'_base').id].reshape(3,3)
 else:
  # The Piper finger meshes extend back from their joint frames. Using 0.19m
  # incorrectly placed a contact anchor beyond the tips. Use the model's
  # established distal grasp site, then let finite contact determine grasping.
  site=m.site('arm_'+side+'_tool').id;b=m.body(side+'_link6').id;r=d.xmat[b].reshape(3,3);p=d.site_xpos[site].copy()
 return p,r

def action_evidence(config,kind,folder):
 m=build(config,kind,folder);ts,q,grip=commands(config,kind,m);d=mujoco.MjData(m);paths={s:[] for s in grip};rots={s:[] for s in grip}
 for qi in q:
  d.qpos[:]=qi;mujoco.mj_forward(m,d)
  for side in grip:p,r=hand_pose(m,d,kind,side);paths[side].append(p.copy());rots[side].append(r.copy())
 paths={k:np.asarray(v) for k,v in paths.items()};events={}
 for side,g in grip.items():
  from rssr_physical_feedback import grasp_events
  cycles=grasp_events(ts,g)
  if not cycles:
   if config.get('scene_backend')=='rssr.scene.v2':
    events[side]={'grasp_observed':False,'initial_center_m':paths[side][0].tolist(),'final_center_m':paths[side][-1].tolist()}
    continue
   raise ValueError('No closure event for '+side)
  close=cycles[0]['closed_index'];release=cycles[0]['release_index'];pre=np.flatnonzero(g[:close]>.8);approach=int(pre[-1]) if len(pre) else 0
  events[side]={'approach_index':approach,'closed_index':close,'release_index':release,'approach_s':float(ts[approach]),'closed_s':float(ts[close]),'release_s':float(ts[release]),'contact_center_at_close_m':paths[side][close].tolist(),'rotation_at_close':rots[side][close].tolist(),'contact_center_at_release_m':paths[side][release].tolist(),'lift_m':float(paths[side][close:release+1,2].max()-paths[side][close,2])}
  events[side]['cycles']=[dict(e,contact_center_at_close_m=paths[side][e['closed_index']].tolist(),contact_center_at_release_m=paths[side][e['release_index']].tolist()) for e in cycles]
 evidence={'kind':kind,'task':read(pathlib.Path(config['import_case'])/'case.json')['task'],'events':events,'time_s':ts[::30].tolist(),'contact_center_samples':{s:paths[s][::30].tolist() for s in paths},'source_action_fixed':kind=='sim1','render_calls':0,'geometry_registration':'inherited estimated source base registration; not measured calibration'}
 write(pathlib.Path(folder)/'action_evidence.json',evidence);np.savez_compressed(pathlib.Path(folder)/'kinematic_reference.npz',time_s=ts,qpos=q,**{s+'_center':v for s,v in paths.items()},**{s+'_rotation':v for s,v in rots.items()});return evidence

def rollout(config,kind,folder,scene,target_qpos=None,stop_s=None):
 folder=pathlib.Path(folder);m=build(config,kind,folder,scene);ts,targets,grip=commands(config,kind,m)
 if target_qpos is not None:targets[:,:30]=target_qpos
 d=mujoco.MjData(m);d.qpos[:]=targets[0];mujoco.mj_forward(m,d);vel=np.gradient(targets,ts,axis=0)
 acts=[];armqa=[]
 for a in range(m.nu):
  if m.actuator_trntype[a]!=mujoco.mjtTrn.mjTRN_JOINT:continue
  j=int(m.actuator_trnid[a,0]);qa=int(m.jnt_qposadr[j]);va=int(m.jnt_dofadr[j]);acts.append((a,qa,va))
  if 'gripper' not in m.joint(j).name and not (kind=='sim1' and m.joint(j).name.endswith(('joint7','joint8'))):armqa.append(qa)
 from rssr_servo import JointServo
 servo=JointServo(m,acts,kind)
 objs=scene['objects'];names=[o['id'] for o in objs];bids=[m.body(n).id for n in names];geombody={g:i for i,b in enumerate(bids) for g in range(m.ngeom) if m.geom_bodyid[g]==b}
 records={k:[] for k in ['time_s','qpos','qvel','ctrl','object_position','object_quaternion','object_velocity','contacts','object_contacts','finger_contact_sides','robot_table_penetration','joint_tracking_error','gripper_opening']}
 worst_contact=None
 limited=[j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)]
 limit_qa=m.jnt_qposadr[limited];limit_lo=m.jnt_range[limited,0];limit_hi=m.jnt_range[limited,1]
 every_step_joint_excess=np.zeros(len(limited))
 from rssr_joint_diagnostics import JointLimitTrace
 joint_trace=JointLimitTrace(m,limited)
 from rssr_model_tools import enabled,HealthAudit
 health=HealthAudit(m) if enabled(config) else None
 if health is not None:health.observe(m,d)
 # Keep the original 100 Hz state-report clock across precision experiments.
 # Every physical step is still audited, and action/video clocks are unchanged.
 record_stride=max(1,int(round(.01/m.opt.timestep)))
 duration=float(ts[-1] if stop_s is None else stop_s);steps=int(np.ceil(duration/m.opt.timestep));started=time.monotonic();lastindex=0
 for step in range(steps+1):
  t=float(d.time);i=min(len(ts)-2,max(0,int(np.searchsorted(ts,t,side='right')-1)));alpha=np.clip((t-ts[i])/(ts[i+1]-ts[i]),0,1)
  desired=(1-alpha)*targets[i]+alpha*targets[i+1];dv=(1-alpha)*vel[i]+alpha*vel[i+1];mujoco.mj_forward(m,d)
  servo.apply(d,desired,dv)
  if step%record_stride==0 or step==steps:
   hit=np.zeros((len(names),3),dtype=np.int8);pairs=np.zeros((len(names),len(names)),dtype=np.int8);fingers=np.zeros((len(names),2,2),dtype=np.int8);pen=0.
   for ct in d.contact:
    if ct.dist>0:continue
    gs=[int(ct.geom1),int(ct.geom2)];gn=[m.geom(g).name or '' for g in gs]
    if any(n.startswith('worktable') for n in gn) and any('astribot_' in n or n.startswith(('left_','right_')) for n in gn):
     pen=max(pen,-float(ct.dist))
     if worst_contact is None or -ct.dist>worst_contact['penetration_m']:worst_contact={'time_s':t,'geoms':gn,'penetration_m':float(-ct.dist)}
    if all(g in geombody for g in gs):pairs[geombody[gs[0]],geombody[gs[1]]]=1;pairs[geombody[gs[1]],geombody[gs[0]]]=1
    for g,other in [(gs[0],gs[1]),(gs[1],gs[0])]:
     if g not in geombody:continue
     oi=geombody[g];n=m.geom(other).name or '';bi=m.geom_bodyid[other];bn=m.body(bi).name or ''
     if n.startswith('worktable'):hit[oi,2]=1
     for si,s in enumerate(['left','right']):
      if s in n or s in bn:
       if 'gripper' in n or bn.endswith(('link7','link8')):
        hit[oi,si]=1
        if ('L112' in n or bn.endswith('link7')):fingers[oi,si,0]=1
        if ('R112' in n or bn.endswith('link8')):fingers[oi,si,1]=1
   opens=[aperture(m,d,s) if kind=='sim2' else float(d.qpos[m.joint(s+'_joint7').qposadr[0]]-d.qpos[m.joint(s+'_joint8').qposadr[0]]) for s in ['left','right']]
   vals=[t,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.xpos[bids].copy(),d.xquat[bids].copy(),d.cvel[bids,3:].copy(),hit,pairs,fingers,pen,float(np.max(np.abs(d.qpos[armqa]-desired[armqa]))),opens]
   for k,v in zip(records,vals):records[k].append(v)
  previous_time=float(d.time)
  mujoco.mj_step(m,d)
  if health is not None:health.observe(m,d)
  try:
   check_physics_step(previous_time,float(d.time),d.qpos,d.qvel,d.qacc,[int(w.number) for w in d.warning])
  except CandidateInvalid as error:
   if health is not None:error.evidence['model_health']=health.report()
   arrays={k:np.array(v) for k,v in records.items()}
   np.savez_compressed(folder/'failed_rollout.npz',**arrays)
   write(folder/'physics_failure.json',{'reason':str(error),'evidence':error.evidence,'step':step,'no_success_evaluation':True})
   raise
  step_joint_excess=np.maximum(limit_lo-d.qpos[limit_qa],d.qpos[limit_qa]-limit_hi)
  every_step_joint_excess=np.maximum(every_step_joint_excess,step_joint_excess)
  joint_trace.observe(m,d,step_joint_excess,desired,previous_time)
 arrays={k:np.array(v) for k,v in records.items()};np.savez_compressed(folder/'rollout.npz',**arrays);np.savez_compressed(folder/'command_targets.npz',time_s=ts,qpos=targets,**{s+'_gripper_command':g for s,g in grip.items()})
 generic=scene.get('schema')=='rssr.scene.v2'
 if generic:
  report={'criteria':{},'max_robot_table_penetration_m':float(arrays['robot_table_penetration'].max()),
          'max_joint_tracking_error_rad':float(arrays['joint_tracking_error'].max())}
 else:
  hold_contract=container_hold_contract(ts,grip['left'])
  write(folder/'container_hold_contract.json',hold_contract)
  report=metrics(m,arrays,scene,hold_contract)
 report.update(kind=kind,wall_seconds=time.monotonic()-started,simulation_seconds=d.time,objects_forced=False,object_weld_used=False,robot_qpos_forced_during_execution=False,render_calls=0,warnings=[int(w.number) for w in d.warning],scene_sha256=sha(folder/'scene.xml'))
 if kind=='sim2':report['gripper_command_adapter']={'position_margin_rad':GRIPPER_COMMAND_MARGIN_RAD,
     'normalized_source_opening_scale_m':.095 if config.get('interaction_revision')=='v16' else .07,
     'force_cap_nm':2.,'executed_qpos_clipped':False,'joint_limits_changed':False,
     'contact_determines_achieved_opening':True,'hardware_calibrated':False}
 report['state_trace']=[{'time_s':round(float(arrays['time_s'][k]),3),'positions_m':np.round(arrays['object_position'][k],4).tolist(),'hand_table_contacts':arrays['contacts'][k].tolist(),'opening_m':np.round(arrays['gripper_opening'][k],4).tolist(),'table_penetration_m':round(float(arrays['robot_table_penetration'][k]),5)} for k in range(0,len(arrays['time_s']),50)]
 report['worst_robot_table_contact']=worst_contact
 # A visually plausible outcome cannot override solver or embodiment failures.
 joint_error=0.
 for jid in range(m.njnt):
  name=m.joint(jid).name or ''
  if m.jnt_type[jid]!=mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[jid] or 'gripper' in name:continue
  qa=m.jnt_qposadr[jid];lo,hi=m.jnt_range[jid];joint_error=max(joint_error,float(np.max(np.maximum(lo-arrays['qpos'][:,qa],arrays['qpos'][:,qa]-hi))))
 joint_error=max(joint_error,float(every_step_joint_excess.max(initial=0)))
 report['max_joint_limit_violation_rad']=max(0.,joint_error)
 report['joint_limit_audit']={'scope':'every physical step, all limited hinge and slide joints including gripper; qpos never clipped','timestep_s':float(m.opt.timestep),'tolerance':.002,'per_joint_excess_native_units':{m.joint(j).name:float(every_step_joint_excess[i]) for i,j in enumerate(limited)},'native_units':'rad for hinge, metre for slide'}
 report['joint_limit_audit']['by_scope']={scope:{m.joint(j).name:float(every_step_joint_excess[i]) for i,j in enumerate(limited) if ((m.joint(j).name or '').startswith(('astribot_','left_','right_')))==(scope=='robot')} for scope in ('robot','objects')}
 report['joint_limit_audit']['temporal_diagnostics']=joint_trace.report()
 report['criteria']['joint_limits_within_2mrad']=joint_error<=.002
 report['criteria']['solver_no_warnings']=not any(report['warnings'])
 if health is not None:
  report['model_health']=health.report()
  report['model_profile']=config.get('model_profile',{})
 if kind=='sim2':
  qa=[m.joint('astribot_torso_joint_'+str(j)).qposadr[0] for j in [1,2]];residual=float(np.max(np.abs(arrays['qpos'][:,qa].sum(axis=1)+.28403)));report['max_waist_sum_residual_rad']=residual;report['criteria']['waist_coupling_within_1mrad']=residual<=.001
 report['task_success']=all(report['criteria'].values())
 if generic:
  from rssr_physical_feedback import evaluate_rollout
  report.update(evaluate_rollout(config,kind,folder,m,arrays,scene,ts,grip,report))
 if config.get('_diagnostic_probe'):
  report.update(diagnostic_only=True,diagnostic_intervention=config['_diagnostic_probe'],task_success=False)
  if 'objective' in report:report['objective']['passed']=False
 write(folder/'report.json',report);return report


def check_physics_step(before,after,qpos,qvel,qacc,warnings):
 """Fail a candidate immediately; never manufacture a monotonic trace."""
 finite=all(np.isfinite(v).all() for v in (qpos,qvel,qacc))
 if not finite or not np.isfinite(after) or after<=before or any(warnings):
  raise CandidateInvalid('Physics instability or simulator time reset',
      {'previous_time_s':before,'time_s':after,'finite_state':bool(finite),'warnings':list(warnings)})

def container_hold_contract(time_s,left_gripper_command):
 """Derive the required hold interval from the immutable source command.

 Both embodiments share source command timing. A fixed .5 s acquisition grace
 allows finite servo closure; the hold requirement lasts until the commanded
 release (> .3 normalized opening). Contact chatter up to .10 s is tolerated.
 This is declared before physical outcomes and does not select a lucky interval.
 """
 ts=np.asarray(time_s,dtype=float);g=np.asarray(left_gripper_command,dtype=float)
 closed=np.flatnonzero(g<.1)
 if len(ts)!=len(g) or len(ts)<2 or not len(closed):raise ValueError('No valid source container grasp command')
 close=int(closed[0]);opened=np.flatnonzero(g[close:]>.3)
 release=close+int(opened[0]) if len(opened) else len(g)-1
 start=float(ts[close]+.5);end=float(ts[release])
 if end-start<.2:raise ValueError('Source container hold command interval is too short')
 return {'schema':'rssr.container_hold_contract.v1','basis':'immutable_source_left_gripper_command',
         'close_index':close,'release_index':release,'closed_s':float(ts[close]),'release_s':end,
         'required_start_s':start,'required_end_s':end,'acquisition_grace_s':.5,
         'maximum_contact_gap_s':.10,'maximum_unsupported_table_span_s':.10,
         'minimum_bilateral_contact_fraction':.90,'release_command_observed':bool(len(opened))}


def _true_intervals(time_s,mask):
 """Intervals of sampled failures, including their one-sample duration."""
 ts=np.asarray(time_s,dtype=float);mask=np.asarray(mask,dtype=bool)
 if not len(ts):return []
 dt=float(np.median(np.diff(ts))) if len(ts)>1 else 0.
 changes=np.diff(np.r_[False,mask,False].astype(np.int8));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
 return [{'start_s':float(ts[a]),'end_s':float(ts[b-1]+dt),'duration_s':float(ts[b-1]-ts[a]+dt)} for a,b in zip(starts,ends)]


def container_hold_metrics(a,container_index,contract):
 """Task-phase holding checks; lift-once plus final containment is insufficient."""
 if contract is None:
  return {'available':False,'reason':'missing_source_action_hold_contract'}, {'container_hold_contract_available':False}
 t=np.asarray(a['time_s']);window=(t>=contract['required_start_s'])&(t<contract['required_end_s'])
 complete=bool(len(t) and t[-1]>=contract['required_end_s'] and np.any(window))
 if not np.any(window):
  return {'available':True,'contract':contract,'complete':False,'reason':'hold_phase_not_simulated'}, {'container_hold_phase_complete':False}
 left=a['contacts'][window,container_index,0].astype(bool);table=a['contacts'][window,container_index,2].astype(bool)
 bilateral=np.all(a['finger_contact_sides'][window,container_index,0].astype(bool),axis=-1)
 gaps=_true_intervals(t[window],~left);unsupported=_true_intervals(t[window],table&~left)
 maxgap=max([row['duration_s'] for row in gaps],default=0.);maxunsupported=max([row['duration_s'] for row in unsupported],default=0.)
 fraction=float(np.mean(bilateral))
 diagnostic={'available':True,'contract':contract,'complete':complete,'samples':int(window.sum()),
             'left_contact_fraction':float(np.mean(left)),'bilateral_contact_fraction':fraction,
             'maximum_unheld_span_s':maxgap,'maximum_table_without_left_span_s':maxunsupported,
             'unheld_intervals':gaps,'table_without_left_intervals':unsupported,
             'table_contact_fraction':float(np.mean(table)),
             'scope':'holding during source-commanded grasp; supported placement before opening is allowed only while hand contact persists'}
 criteria={'container_hold_phase_complete':complete,
           'container_left_contact_maintained':maxgap<=contract['maximum_contact_gap_s']+1e-8,
           'container_bilateral_grasp_maintained':fraction>=contract['minimum_bilateral_contact_fraction'],
           'container_no_unsupported_early_table_return':maxunsupported<=contract['maximum_unsupported_table_span_s']+1e-8}
 return diagnostic,criteria


def metrics(m,a,scene,hold_contract=None):
 names=[o['id'] for o in scene['objects']];bi=names.index(scene['container_id']);ci=names.index(scene['movable_id']);basket=scene['objects'][bi];cube=scene['objects'][ci]
 pp=a['object_position'];time_s=a['time_s'];last=time_s>time_s[-1]-1;rot=Rotation.from_quat(a['object_quaternion'][-1,bi][[1,2,3,0]]).as_matrix();local=rot.T@(pp[-1,ci]-pp[-1,bi]);size=np.array(basket['size_m']);half=np.array(cube['size_m'])/2;t=basket['wall_m']
 cube_rot=Rotation.from_quat(a['object_quaternion'][-1,ci][[1,2,3,0]]).as_matrix();corners=(SIGNS*half)@cube_rot.T@rot+local
 inside=bool(np.all(np.abs(corners[:,:2])<size[:2]/2-t+.002) and np.min(corners[:,2])>=-size[2]/2+t-.004 and np.max(corners[:,2])<=size[2]/2+.003)
 stable=float(np.max(np.linalg.norm(np.ptp(pp[last],axis=0),axis=-1)))<.003
 lift=np.max(pp[:,:,2],axis=0)-pp[0,:,2];released=not bool(a['contacts'][last,:,:2].any());initial_outside=abs(pp[0,ci,0]-pp[0,bi,0])>size[0]/2 or abs(pp[0,ci,1]-pp[0,bi,1])>size[1]/2
 left_lift=bool(np.any((pp[:,bi,2]-pp[0,bi,2]>.02)&a['contacts'][:,bi,0].astype(bool)));right_lift=bool(np.any((pp[:,ci,2]-pp[0,ci,2]>.025)&a['contacts'][:,ci,1].astype(bool)))
 support=bool(a['contacts'][-20:,bi,2].any());cube_support=bool(a.get('object_contacts',np.zeros((len(time_s),len(names),len(names))))[-20:,bi,ci].any())
 rules={'initial_cube_outside_container':bool(initial_outside),'container_lifted_by_left':left_lift,'cube_lifted_by_right':right_lift,'cube_inside_container':inside,'cube_supported_by_container':cube_support,'released':released,'stable':stable,'container_on_table':support,'max_robot_table_penetration_le_3mm':float(a['robot_table_penetration'].max())<=.003}
 holding,holding_rules=container_hold_metrics(a,bi,hold_contract);rules.update(holding_rules)
 return {'task_success':all(rules.values()),'criteria':rules,'container_hold':holding,'lift_m':dict(zip(names,lift.tolist())),'final_positions_m':dict(zip(names,pp[-1].tolist())),'cube_final_container_relative_m':local.tolist(),'max_robot_table_penetration_m':float(a['robot_table_penetration'].max()),'max_joint_tracking_error_rad':float(a['joint_tracking_error'].max()),'contact_samples':{n:a['contacts'][:,i].sum(axis=0).tolist() for i,n in enumerate(names)},'bilateral_finger_contact_samples':{n:np.all(a['finger_contact_sides'][:,i],axis=-1).sum(axis=0).tolist() for i,n in enumerate(names)},'last_second_motion_m':np.linalg.norm(np.ptp(pp[last],axis=0),axis=-1).tolist()}
