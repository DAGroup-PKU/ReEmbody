"""Deterministic tools exposed to the RSSR scene and target repair agents."""
import copy,pathlib
import numpy as np,mujoco
from common import read,write
from object_physics import build,commands,action_evidence

def action_scene(config,folder,semantics):
 folder=pathlib.Path(folder);ev=action_evidence(config,'sim1',folder/'evidence');m=build(config,'sim1',folder/'model');ts,q,g=commands(config,'sim1',m);d=mujoco.MjData(m);src=np.load(pathlib.Path(config['import_case'])/'source_action_state.npz');lowest=[]
 for side in ['left','right']:
  k=ev['events'][side]['closed_index'];d.qpos[:]=q[k];mujoco.mj_forward(m,d)
  for n in [side+'_link7_geom',side+'_link8_geom']:
   gid=m.geom(n).id;mi=m.geom_dataid[gid];v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];lowest.append(float(v[:,2].min()))
 table=float(np.clip(min(lowest)-.004,.60,.75));L=np.array(ev['events']['left']['contact_center_at_close_m']);R=np.array(ev['events']['right']['contact_center_at_close_m']);normal=np.array(ev['events']['left']['rotation_at_close'])[:,1];normal[2]=0;normal/=np.linalg.norm(normal)
 rk=ev['events']['right']['closed_index'];side=float(np.clip(.095*float(np.median(src['state'][rk:rk+15,6]))/1.05,.035,.060))
 width=float(semantics.get('container_width_m',.16));depth=float(semantics.get('container_depth_m',.20));wall=float(semantics.get('wall_m',.012));assert .12<=width<=.24 and .14<=depth<=.28 and .004<=wall<=.015
 h=float(np.clip(max(L[2]-table+.014,side+wall+.005),.035,.10));center=L-normal*(width-wall)/2;center[2]=table+h/2+.0003
 cube={'id':'blue_cube','shape':'box','position_m':[float(R[0]),float(R[1]),table+side/2+.0003],'size_m':[side]*3,'yaw_rad':float(np.arctan2(ev['events']['right']['rotation_at_close'][1][1],ev['events']['right']['rotation_at_close'][0][1])),'rgba':[.01,.24,.85,1.],'mass_kg':.04}
 basket={'id':'basket','shape':'open_box','position_m':center.tolist(),'size_m':[width,depth,h],'wall_m':wall,'yaw_rad':float(np.arctan2(normal[1],normal[0])),'rgba':[.43,.25,.10,1.],'mass_kg':.12}
 scene={'container_id':'basket','movable_id':'blue_cube','table_shift_z_m':table-.75,'objects':[basket,cube]}
 for i,color in enumerate(semantics.get('stationary_cube_colors',[])):
  assert color in ['red','green'];obj=copy.deepcopy(cube);obj['id']=color+'_cube';obj['position_m'][0]-=.11*(i+1);obj['rgba']=[.75,.025,.02,1.] if color=='red' else [.015,.38,.05,1.];scene['objects'].append(obj)
 write(folder/'construction.json',{'scene':scene,'action_evidence':ev,'physical_finger_lowest_z_m':lowest,'size_source':'Piper aperture prior + recorded obstructed gripper feedback; geometry estimate, not true metric reconstruction','source_action_fixed':True,'render_calls':0});return scene

def target_scene(config,folder,source_scene):
 folder=pathlib.Path(folder);ev=action_evidence(config,'sim2',folder/'evidence');scene=copy.deepcopy(source_scene);L=np.array(ev['events']['left']['contact_center_at_close_m']);R=np.array(ev['events']['right']['contact_center_at_close_m']);normal=np.array(ev['events']['left']['rotation_at_close'])[:,0];normal[2]=0;normal/=np.linalg.norm(normal)
 basket=next(o for o in scene['objects'] if o['id']=='basket');cube=next(o for o in scene['objects'] if o['id']=='blue_cube');old=np.array(cube['position_m']);table=.75;scene['table_shift_z_m']=0.
 bp=L-normal*(basket['size_m'][0]-basket['wall_m'])/2;bp[2]=table+basket['size_m'][2]/2+.0003;basket['position_m']=bp.tolist();basket['yaw_rad']=float(np.arctan2(normal[1],normal[0]));cube['position_m']=[float(R[0]),float(R[1]),table+cube['size_m'][2]/2+.0003];cube['yaw_rad']=float(np.arctan2(ev['events']['right']['rotation_at_close'][1][0],ev['events']['right']['rotation_at_close'][0][0]))
 for obj in scene['objects']:
  if obj['id'] not in ['basket','blue_cube']:obj['position_m']=(np.array(obj['position_m'])-old+np.array(cube['position_m'])).tolist()
 correction={'hand_offsets_m':{'left':[0,0,float(table+basket['size_m'][2]-.015-L[2])],'right':[0,0,float(cube['position_m'][2]-R[2])]},'vertical_grasp_mix':0.}
 return scene,correction

def apply_scene_delta(scene,patch):
 scene=copy.deepcopy(scene)
 assert set(patch)<={'object_xy_shift_m','table_z_shift_m','basket_yaw_delta_rad','object_contact_friction','object_contact_condim'}
 friction=patch.get('object_contact_friction',{})
 if not isinstance(friction,dict) or set(friction)-{o['id'] for o in scene['objects']}:raise ValueError('Unknown contact material object')
 condim=patch.get('object_contact_condim',{})
 if not isinstance(condim,dict) or set(condim)-{o['id'] for o in scene['objects']}:raise ValueError('Unknown contact model object')
 if any(type(v) is not int or v not in [4,6] for v in condim.values()):raise ValueError('Contact dimension must be 4 or 6')
 dz=float(patch.get('table_z_shift_m',0));assert abs(dz)<=.025;scene['table_shift_z_m']+=dz
 for o in scene['objects']:
  shift=patch.get('object_xy_shift_m',{}).get(o['id'],[0,0]);assert len(shift)==2 and np.max(np.abs(shift))<=.04;o['position_m'][:2]=(np.array(o['position_m'][:2])+shift).tolist();o['position_m'][2]+=dz
  if o['id']=='basket':
   da=float(patch.get('basket_yaw_delta_rad',0));assert abs(da)<=.3;o['yaw_rad']+=da
  if o['id'] in friction:
   f=np.array(friction[o['id']],float)
   if f.shape!=(3,) or not np.isfinite(f).all() or np.any(f<np.array([.3,.003,0.])) or np.any(f>np.array([2.,.02,.005])):raise ValueError('Contact friction outside material prior')
   o['contact_friction']=f.tolist()
  if o['id'] in condim:o['contact_condim']=condim[o['id']]
 return scene
