"""Object-free S1 kinematic rendering and matching Real1 reference clips."""
import os,argparse,itertools,json,pathlib,shutil,subprocess,hashlib,xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco,cv2
from PIL import Image,ImageDraw
import os, shutil
R=pathlib.Path(os.environ.get('REEMBODY_RECONSTRUCT_RUN_ROOT', '.'))
FF=os.environ.get('REEMBODY_FFMPEG') or shutil.which('ffmpeg') or 'ffmpeg'
SIGNS=np.array(list(itertools.product([-1.,1.],repeat=3)))
LINK={'L1':1,'L11':1,'L2':-1,'R1':1,'R11':-1,'R2':1}
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def vec(a):return ' '.join(map(str,np.asarray(a).ravel()))
def gripper(m,d,s,a):
 for n,sign in LINK.items():d.qpos[m.joint(f'astribot_gripper_{s}_joint_{n}').qposadr[0]]=sign*a
def gap(m,d):
 b=m.body('astribot_gripper_left_base').id;rot=d.xmat[b].reshape(3,3);pos=d.xpos[b];vs=[]
 for name in ['L112','R112']:
  g=m.geom('astribot_gripper_left_geom_'+name).id;world=(SIGNS*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];vs.append((world-pos)@rot)
 return float(vs[0][:,0].min()-vs[1][:,0].max())
def build(case):
 source=R/'implementation_revisions/physical_constraints_17/s1_tabletop_physical.xml';tree=ET.parse(source);x=tree.getroot()
 assert not x.findall('.//freejoint'),'Unexpected movable objects in robot-only source model'
 comp=x.find('compiler');md=pathlib.Path(comp.get('meshdir','')) if comp is not None else pathlib.Path('')
 for e in x.findall('./asset/mesh'):
  p=pathlib.Path(e.get('file'));e.set('file',str(p if p.is_absolute() else (source.parent/md/p).resolve()))
 if comp is not None:comp.set('meshdir','')
 visual=x.find('visual')
 if visual is None:visual=ET.SubElement(x,'visual')
 glob=visual.find('global')
 if glob is None:glob=ET.SubElement(visual,'global')
 glob.set('offwidth','640');glob.set('offheight','480')
 for g in x.iter('geom'):
  if (g.get('name') or '').startswith('physical_robot_'):g.set('rgba','0 0 0 0')
  if g.get('name')=='worktable_top':g.set('rgba','.88 .88 .86 1')
  if g.get('type')=='plane':g.attrib.pop('material',None);g.set('rgba','.48 .50 .52 1')
 pos=np.array([0,-.32,1.45]);look=np.array([0,-.735,.78]);fw=look-pos;fw/=np.linalg.norm(fw);right=np.cross(fw,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,fw);q=np.empty(4);mujoco.mju_mat2Quat(q,np.column_stack([right,up,-fw]).ravel())
 world=x.find('worldbody');ET.SubElement(world,'camera',name='rssr_head',pos=vec(pos),quat=vec(q),fovy='55')
 path=case/'sim2_robot_only_scene.xml';tree.write(path,encoding='unicode');return mujoco.MjModel.from_xml_path(str(path))
def encode(raw,dest):
 subprocess.run([FF,'-hide_banner','-loglevel','error','-y','-i',str(raw),'-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);p.add_argument('--full',action='store_true');a=p.parse_args();c=a.case;meta=json.loads((c/'case.json').read_text());response=json.loads((c/'response.json').read_text());assert response['status']=='trajectory_passed'
 shutil.copy2(response['artifacts']['joint_trajectory_path'],c/'agent_trajectory_full.npz');z=dict(np.load(c/'agent_trajectory_full.npz'));src=dict(np.load(c/'source_action_state.npz'));ts=z['source_time_s'];assert len(ts)==len(src['timestamp'])
 m=build(c);d=mujoco.MjData(m);assert not np.any(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
 aa=np.linspace(0,.93,187);gaps=[]
 for av in aa:gripper(m,d,'left',av);mujoco.mj_forward(m,d);gaps.append(gap(m,d))
 assert np.all(np.diff(gaps)<0)
 n=len(ts);span=min(450,n);gs=np.c_[z['source_left_gripper'],z['source_right_gripper']];movement=np.max(-np.diff(gs,axis=0),axis=1);ev=np.flatnonzero((movement>.04)&(np.arange(n-1)>30));center=int(ev[0]) if len(ev) else n//2
 if a.full:span=n;start=0
 else:start=int(np.clip(center-60,0,max(0,n-span)))
 count=int(np.floor(span*24/30));idx=start+np.floor(np.arange(count)*30/24).astype(int);assert len(set(idx))==len(idx) and idx[-1]<n and count>=96
 cameras=['rssr_head','left_wrist_rgbd','right_wrist_rgbd'];roles=['head','left_wrist','right_wrist'];caps=[cv2.VideoCapture(meta['source_videos'][r]) for r in roles]
 assert all(cap.isOpened() for cap in caps)
 for cap in caps:cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx[0]))
 renderer=mujoco.Renderer(m,height=480,width=640);raw1=c/'real1_raw_encode.mp4';raw2=c/'sim2_raw_encode.mp4';wr=[cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),24,(1470,630)) for p in [raw1,raw2]];assert all(w.isOpened() for w in wr)
 qpos=[];cpos=[];crot=[];gripangles=[];sheets=[];previous=int(idx[0])-1
 for f,k in enumerate(idx):
  for side in ['left','right']:
   for j in range(1,8):d.qpos[m.joint(f'astribot_arm_{side}_joint_{j}').qposadr[0]]=z[side+'_q'][k,j-1]
  for j in range(1,5):
   jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'astribot_torso_joint_{j}')
   if jid>=0:d.qpos[m.jnt_qposadr[jid]]=z['upstream_q'][k,j-1]
  angles=[]
  for side in ['left','right']:
   angle=float(np.interp(.07*np.clip(z['source_'+side+'_gripper'][k]/1.05,0,1),np.asarray(gaps)[::-1],aa[::-1]));gripper(m,d,side,angle);angles.append(angle)
  d.qvel[:]=0;mujoco.mj_forward(m,d);one=[];two=[];cp=[];cr=[]
  for cap,cam in zip(caps,cameras):
   for _ in range(int(k)-previous-1):assert cap.grab()
   ok,frame=cap.read();assert ok,f'Source frame missing {k}';one.append(cv2.cvtColor(cv2.resize(frame,(480,360)),cv2.COLOR_BGR2RGB))
   renderer.update_scene(d,camera=cam);two.append(cv2.resize(renderer.render(),(480,360)));cid=m.camera(cam).id;cp.append(d.cam_xpos[cid].copy());cr.append(d.cam_xmat[cid].reshape(3,3).copy())
  canv=[]
  for views,w in zip([one,two],wr):
   out=Image.new('RGB',(1470,630),'black');draw=ImageDraw.Draw(out)
   for j,(im,role) in enumerate(zip(views,roles)):
    x=2+j*492;out.paste(Image.fromarray(im),(x,135));draw.text((x+10,111),role.upper(),fill='white')
   w.write(cv2.cvtColor(np.array(out),cv2.COLOR_RGB2BGR));canv.append(out)
  if f in set(np.linspace(0,count-1,5).astype(int).tolist()):sheets.append((canv[0].resize((735,315)),canv[1].resize((735,315))))
  qpos.append(d.qpos.copy());cpos.append(cp);crot.append(cr);gripangles.append(angles);previous=int(k)
  if f%96==0:print(json.dumps({'case':c.name,'render_frame':f,'frames':count}),flush=True)
 for w in wr:w.release()
 for cap in caps:cap.release()
 if callable(getattr(renderer,'close',None)):renderer.close()
 for raw,dest in [(raw1,c/'real1_condition.mp4'),(raw2,c/'sim2_robot_only.mp4')]:encode(raw,dest)
 sheet=Image.new('RGB',(1470,315*len(sheets)))
 for i,(one,two) in enumerate(sheets):sheet.paste(one,(0,i*315));sheet.paste(two,(735,i*315))
 sheet.save(c/'input_contact_sheet.jpg')
 np.savez_compressed(c/'frame_action_mapping.npz',video_frame_index=np.arange(count),video_pts_s=np.arange(count)/24,source_frame_index=idx,source_time_s=ts[idx],left_q=z['left_q'][idx],right_q=z['right_q'][idx],upstream_q=z['upstream_q'][idx],source_left_gripper=z['source_left_gripper'][idx],source_right_gripper=z['source_right_gripper'][idx],target_gripper_angle_rad=gripangles,nominal_qpos=qpos,camera_position_world_m=cpos,camera_rotation_world_opengl=crot)
 report={'case_id':c.name,'task':meta['task'],'frames':count,'fps':24,'duration_s':count/24,'source_start_frame':int(idx[0]),'source_end_frame':int(idx[-1]),'source_start_s':float(ts[idx[0]]),'source_end_s':float(ts[idx[-1]]),'clip_selection':'first substantial gripper closure minus 2 seconds, clamped to source bounds; maximum 15 seconds','full_episode':span==n,
  'render_revision':3 if a.full else 2,'frame_policy':'30 to 24 fps by dropping original source/trajectory indices; no interpolation or repeated frames','objects_in_sim2':False,'free_joints_in_sim2':0,'kinematic_render_only':True,'physical_execution_verified':False,'task_success':None,'camera_calibration':'Head estimated fixed view in front of S1 head; wrists from S1 model. Cross-embodiment camera matching is approximate.',
  'gripper_mapping':'Model geometry: source 0..1.05 -> 0..70mm opening -> S1 linkage angle; visual configuration only, not hardware command.',
  'interface_provenance':response['provenance'],'trajectory_sha256':hashlib.sha256((c/'agent_trajectory_full.npz').read_bytes()).hexdigest(),'videos':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [c/'real1_condition.mp4',c/'sim2_robot_only.mp4']}}
 if a.full:report['clip_selection']='full episode, original frame indices downsampled from 30 to 24 FPS'
 write(c/'render_manifest.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
