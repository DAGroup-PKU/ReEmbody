"""Render only frozen physical rollouts; source indices remain auditable."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import pathlib,subprocess,json
import numpy as np,mujoco,cv2
from scipy.spatial.transform import Rotation
from common import write,read,sha
from media import probe,encode_args

def configure_cameras(m,kind,profile=None):
 overrides={}
 m.site_rgba[:,3]=0  # The same diagnostic visibility in preview and final video.
 if kind=='sim1':
  # The source asset omits wrist camera mounts. Use a declared flange-relative
  # camera outside the gripper housing, aimed at the model's distal grasp site.
  # This changes only rendering; dynamics and all physical states are unchanged.
  # Piper closes along flange Y; mounting along Y puts the lens behind a
  # finger. Offset along flange X so both jaws remain on the sides of the view.
  p=np.array([-.09,0.,.065]);f=np.array([0.,0.,.145])-p;f/=np.linalg.norm(f);right=np.array([0.,1.,0.]);up=np.cross(-f,right);r=np.c_[right,up,-f];q=Rotation.from_matrix(r).as_quat()
  for side in ['left','right']:
   cid=m.camera(side+'_wrist_rgbd').id;m.cam_pos[cid]=p;m.cam_quat[cid]=q[[3,0,1,2]];m.cam_fovy[cid]=70
   overrides[side+'_wrist_rgbd']={'position_in_flange_m':p.tolist(),'rotation_in_flange':r.tolist(),'fovy_degrees':70,'basis':'unmeasured exterior wrist camera prior'}
 if profile is not None:
  from camera_agent import apply_profile
  entries=profile['cameras'][kind];apply_profile(m,entries)
  for name,spec in entries.items():
   overrides[name]=dict(spec,position_in_flange_m=spec['position_in_parent_m'],rotation_in_flange=spec['rotation_in_parent'])
 return overrides

def render(config,kind,trial,output):
 trial=pathlib.Path(trial);output=pathlib.Path(output);z=np.load(trial/'rollout.npz');mapping=np.load(pathlib.Path(config['import_case'])/'frame_action_mapping.npz');ts=mapping['source_time_s'];indices=np.searchsorted(z['time_s'],ts,side='right')-1;indices=np.maximum(indices,0)
 assert len(np.unique(indices))==len(indices)
 camera_path=pathlib.Path(config['run_root'])/'camera_alignment/camera_profile.json'
 profile=read(camera_path) if config.get('camera_alignment',{}).get('enabled') and camera_path.exists() else None
 m=mujoco.MjModel.from_xml_path(str(trial/'scene.xml'));camera_overrides=configure_cameras(m,kind,profile);d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=360,width=480)
 raw=output.with_suffix('.raw.mp4');writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),24,(1470,630));assert writer.isOpened()
 cameras=['rssr_head','left_wrist_rgbd','right_wrist_rgbd'];camera_positions=[];camera_rotations=[]
 for f,k in enumerate(indices):
  d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];mujoco.mj_forward(m,d);frame=np.zeros((630,1470,3),np.uint8);cp=[];cr=[]
  for j,c in enumerate(cameras):
   renderer.update_scene(d,camera=c);im=renderer.render();frame[135:495,j*495:j*495+480]=cv2.cvtColor(im,cv2.COLOR_RGB2BGR);cv2.putText(frame,c.upper(),(j*495+8,115),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
   cid=m.camera(c).id;cp.append(d.cam_xpos[cid].copy());cr.append(d.cam_xmat[cid].reshape(3,3).copy())
  writer.write(frame);camera_positions.append(cp);camera_rotations.append(cr)
  if f%160==0:print(json.dumps({'render':kind,'frame':f,'total':len(indices)}),flush=True)
 writer.release()
 if hasattr(renderer,'close'):renderer.close()
 del renderer
 subprocess.run(encode_args(config)+['-i',str(raw),'-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
 robot_joints=[j for j in range(m.njnt) if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE];qa=m.jnt_qposadr[robot_joints];va=m.jnt_dofadr[robot_joints]
 np.savez_compressed(output.with_suffix('.npz'),video_frame_index=np.arange(len(indices)),video_pts_s=np.arange(len(indices))/24,source_frame_index=mapping['source_frame_index'],source_time_s=ts,simulation_sample_index=indices,simulation_time_s=z['time_s'][indices],qpos=z['qpos'][indices],qvel=z['qvel'][indices],ctrl=z['ctrl'][indices],robot_qpos=z['qpos'][indices][:,qa],robot_qvel=z['qvel'][indices][:,va],robot_joint_names=np.array([m.joint(j).name for j in robot_joints]),actuator_names=np.array([m.actuator(j).name for j in range(m.nu)]),gripper_opening_m=z['gripper_opening'][indices],object_position=z['object_position'][indices],object_quaternion=z['object_quaternion'][indices],contacts=z['contacts'][indices],camera_position=camera_positions,camera_rotation=camera_rotations)
 write(output.with_suffix('.json'),{'kind':kind,'frames':len(indices),'fps':24,'physics_report':str(trial/'report.json'),'source_indices_sha256':sha(pathlib.Path(config['import_case'])/'frame_action_mapping.npz'),'sampling':'monotonic selection of existing physics samples at preserved source indices; no frame interpolation or duplication','camera_calibration':'estimated source head/wrist; target model cameras; not measured cross-view registration','camera_overrides':camera_overrides,'model_sha256':sha(trial/'scene.xml'),'rollout_sha256':sha(trial/'rollout.npz'),'mujoco_version':mujoco.__version__,'joints':[{'name':m.joint(j).name,'qpos_address':int(m.jnt_qposadr[j]),'dof_address':int(m.jnt_dofadr[j]),'type':int(m.jnt_type[j])} for j in range(m.njnt)],'actuators':[m.actuator(j).name for j in range(m.nu)],'cameras':cameras})

def overview(config,root,real2):
 root=pathlib.Path(root);paths=[root/'real1.mp4',root/'sim1.mp4',root/'sim2.mp4',pathlib.Path(real2)];labels=['REAL1 | SOURCE RECORDING','SIM1 | SOURCE ACTION PROXY','SIM2 | TARGET PHYSICS','REAL2 | GENERATED CANDIDATE'];frames=[probe(p,config)['frames'] for p in paths];assert len(set(frames))==1
 args=encode_args(config)
 for p in paths:args+=['-i',str(p)]
 graph=[]
 for i,label in enumerate(labels):graph.append(f'[{i}:v]crop=1470:400:0:100,drawtext=text=\'{label}\':x=10:y=5:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.8[v{i}]')
 graph.append('[v0][v1][v2][v3]vstack=inputs=4[out]');out=root/'RSSR_3x4.mp4'
 subprocess.run(args+['-filter_complex',';'.join(graph),'-map','[out]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True);return out
