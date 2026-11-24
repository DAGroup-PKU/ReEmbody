"""Measured S1 posture on restored official S1 joints; declared scene priors."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .common import BRANCH,read

import os as _os
SOURCE=Path(_os.environ.get('REEMBODY_S1_MJCF_DIR', str(Path(__file__).resolve().parents[3] / 'third_party/astribot_s1_mjcf')))
CUP=dict(top_diameter_m=.08,bottom_diameter_m=.0585,height_m=.09,wall_m=.0015,mass_kg=.06,segments=32,
         height_bottom_diameter_wall_mass_are_priors=True)
LINK={'L1':1,'L11':1,'L2':-1,'R1':1,'R11':-1,'R2':1}
def vec(v):return ' '.join(format(float(x),'.17g') for x in np.asarray(v).ravel())
@lru_cache(None)
def reference():return read(BRANCH/'config/measured_alignment.json')
def pose():return read(reference()['pose_path'])
def joint_values():return {'astribot_'+k:v for k,v in pose()['joints_rad'].items()}
_NOMINAL_GRIPPER_ANGLES={}
def initial_pose(model):
 d=mujoco.MjData(model)
 for name,value in joint_values().items():
  joint=model.joint(name)
  if not model.jnt_range[joint.id,0]<=value<=model.jnt_range[joint.id,1]:raise ValueError('Snapshot outside original S1 joint limits: '+name)
  d.qpos[int(joint.qposadr[0])]=value
 # Nominal SDK endpoints are recorded by the snapshot (0 open,100 closed).
 # Convert the opening fraction through the original S1 geometric gap curve;
 # this is a documented simulator assumption, not hardware angle calibration.
 from pilot_physics import aperture
 for side in ('left','right'):
  if side not in _NOMINAL_GRIPPER_ANGLES:
    angles=np.linspace(.02,.91,179);gaps=[]
    for angle in angles:
     for suffix,sign in LINK.items():d.qpos[int(model.joint(f'astribot_gripper_{side}_joint_{suffix}').qposadr[0])]=sign*angle
     mujoco.mj_forward(model,d);gaps.append(aperture(model,d,side))
    opening_fraction=1.-float(pose()['gripper_sdk_0_100'][side])/100.
    if not 0.<=opening_fraction<=1.:raise ValueError('SDK nominal opening outside endpoints')
    target_gap=float(gaps[-1])+opening_fraction*(float(gaps[0])-float(gaps[-1]))
    angle=float(np.interp(target_gap,np.asarray(gaps)[::-1],angles[::-1]))
    _NOMINAL_GRIPPER_ANGLES[side]=angle
  angle=_NOMINAL_GRIPPER_ANGLES[side]
  for suffix,sign in LINK.items():d.qpos[int(model.joint(f'astribot_gripper_{side}_joint_{suffix}').qposadr[0])]=sign*angle
 return d.qpos.copy()
def apply_table(root,config):
 ref=reference();table=root.find(".//body[@name='worktable']");top=table.find(".//geom[@name='worktable_top']")
 table.set('pos',vec([*ref['table_center_xy_m'],0.]));size=np.fromstring(top.get('size'),sep=' ');size[:2]=ref['table_half_xy_m']
 dz=float(config.get('_lyc_physical_profile',{}).get('table_height_delta_m',0.));height=ref['tabletop_z_m']+dz
 top.set('size',vec(size));top.set('pos',vec([0.,0.,height-size[2]]))
 for g in table.iter('geom'):
  if not g.get('name','').startswith('worktable_leg_'):continue
  old=np.fromstring(g.get('pos'),sep=' ');s=np.fromstring(g.get('size'),sep=' ');s[2]=(height-2*size[2])/2
  g.set('size',vec(s));g.set('pos',vec([*(np.sign(old[:2])*(size[:2]-.07)),s[2]]))
def apply_robot(root):
 ref=reference();original=ET.parse(SOURCE/'model_with_effector/astribot_s1_model_with_gripper.xml').getroot()
 for i in (3,4):
  name=f'astribot_torso_link_{i}';target=root.find(f".//body[@name='{name}']");native=original.find(f".//body[@name='{name}']")
  for key in ('pos','quat'):
   if native.get(key) is None:target.attrib.pop(key,None)
   else:target.set(key,native.get(key))
  target.insert(0,deepcopy(native.find('joint')))
 acts=root.find('actuator');native=ET.parse(SOURCE/'actuators/actuator_s1_actuator_chassis_fixed_for_gripper.xml').getroot()
 for i in (3,4):
  a=deepcopy(native.find(f".//*[@name='astribot_torso_joint_{i}']"));a.attrib.pop('class',None)
  # Same inherited torso position servo and finite effort envelope.
  proto=root.find(".//actuator/*[@name='astribot_torso_joint_1']")
  if proto is not None:
   a=deepcopy(proto);a.set('name',f'astribot_torso_joint_{i}');a.set('joint',f'astribot_torso_joint_{i}')
  acts.append(a)
 values=joint_values();s=values['astribot_torso_joint_1']+values['astribot_torso_joint_2']
 for e in root.findall('equality/joint'):
  if {e.get('joint1'),e.get('joint2')}=={'astribot_torso_joint_1','astribot_torso_joint_2'}:e.set('polycoef',vec([s,-1.,0.,0.,0.]))
 custom=root.find('custom')
 if custom is None:custom=ET.SubElement(root,'custom')
 ET.SubElement(custom,'numeric',name='rssr_s1_measured_waist_sum_rad',data=str(s))
 base=root.find(".//body[@name='chassis_base']");base.set('pos',vec(ref['robot_base_world_position_m']));base.attrib.pop('euler',None);yaw=ref['robot_base_world_yaw_rad'];base.set('quat',vec([np.cos(yaw/2),0,0,np.sin(yaw/2)]))
 alignment=Path(ref['camera_alignment_path'])
 if alignment.exists():
  calibration=read(alignment)
  for name,values in calibration.get('cameras',{}).items():
   cam=root.find(f".//camera[@name='{name}']")
   if cam is None:raise ValueError('Unknown camera '+name)
   for key in ('pos','quat','fovy'):
    if key in values:cam.set(key,str(values[key]) if key=='fovy' else vec(values[key]))
def normalize_scene(scene):
 result=deepcopy(scene);ref=reference();cups=[o for o in result['objects'] if o['shape']=='cup']
 if len(cups)!=2:raise ValueError('Expected two cups')
 old_height=float(cups[0]['position_m'][2])-float(cups[0]['size_m'][2])/2-.0003
 for obj in result['objects']:
  obj['position_m'][2]+=ref['tabletop_z_m']-old_height
  if obj['shape']=='region':obj['rgba'][-1]=0.
 for i,obj in enumerate(cups):
  obj['size_m']=[CUP['top_diameter_m']]*2+[CUP['height_m']];obj['wall_m']=CUP['wall_m'];obj['mass_kg']=CUP['mass_kg']
  obj['position_m']=[*ref['cup_centers_xy_m'][i],ref['tabletop_z_m']+CUP['height_m']/2+.0003]
  obj['evidence']='User 8 cm tapered top diameter; inherited 9 cm height and 5.85 cm bottom/wall/mass priors, identical cups.'
 region=next(o for o in result['objects'] if o['id']==result['bindings']['target_region']);region['position_m']=[*ref['goal_center_xy_m'],ref['tabletop_z_m']+.0018]
 return result
def add_cup_geometry(asset,body,obj):
 spec=CUP;n=spec['segments'];h=spec['height_m'];wall=spec['wall_m'];attrs=dict(contype='4',conaffinity='15',condim='4',friction='1 .01 .002',solref='.006 1',solimp='.95 .99 .001',rgba=vec(obj['rgba']))
 ET.SubElement(body,'geom',name=obj['id']+'_bottom',type='cylinder',size=vec([spec['bottom_diameter_m']/2,wall/2]),pos=vec([0,0,-h/2+wall/2]),mass=str(spec['mass_kg']*.2),**attrs)
 for i in range(n):
  angles=[2*math.pi*i/n,2*math.pi*(i+1)/n];vertices=[]
  for z in (-h/2,h/2):
   outer=(spec['bottom_diameter_m']+(z+h/2)/h*(spec['top_diameter_m']-spec['bottom_diameter_m']))/2
   for radius in (outer,outer-wall):
    for angle in angles:vertices.append([radius*math.cos(angle),radius*math.sin(angle),z])
  name=obj['id']+f'_wall_{i:02d}';ET.SubElement(asset,'mesh',name=name+'_mesh',vertex=vec(vertices));ET.SubElement(body,'geom',name=name,type='mesh',mesh=name+'_mesh',mass=str(spec['mass_kg']*.8/n),**attrs)
def install_backend(backend):
 if getattr(backend,'_s1_measured_installed',False):return
 original_model=backend.target_model;original_commands=backend.commands
 def target_model(config):
  root=original_model(config);apply_robot(root);apply_table(root,config);return root
 def commands(config,kind,model):
  if kind!='sim2':return original_commands(config,kind,model)
  src=np.load(Path(config['import_case'])/'source_action_state.npz');times=src['timestamp']-src['timestamp'][0]
  initial=initial_pose(model)
  overrides=config.get('_s1_initial_robot_qpos_by_name')
  if overrides is not None:
   expected={model.joint(j).name for j in range(model.njnt) if (model.joint(j).name or '').startswith('astribot_')}
   if set(overrides)!=expected:raise ValueError('Aligned start must declare every named scalar robot joint')
   for name,value in overrides.items():
    joint=model.joint(name);qa=int(joint.qposadr[0]);value=float(value)
    if not np.isfinite(value) or not model.jnt_range[joint.id,0]<=value<=model.jnt_range[joint.id,1]:raise ValueError('Aligned joint outside original limits: '+name)
    if '_arm_' not in name and value!=float(initial[qa]):raise ValueError('Aligned experiment may perturb only arm joints: '+name)
    initial[qa]=value
  targets=np.tile(initial,(len(times),1));grip={s:np.clip(src['action'][:,13 if s=='left' else 6]/1.05,0,1) for s in ['left','right']}
  return times,targets,grip
 backend.target_model=target_model;backend.commands=commands;backend._s1_measured_installed=True
 import rssr_scene
 original_geometry=rssr_scene.add_geometry
 def add_geometry(asset,body,obj):
  if obj['shape']=='cup':return add_cup_geometry(asset,body,obj)
  return original_geometry(asset,body,obj)
 rssr_scene.add_geometry=add_geometry
