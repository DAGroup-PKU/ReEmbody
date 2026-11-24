"""Read-only camera cues from frozen kinematics, never invented object geometry."""
import numpy as np
from common import write

def project(point,position,rotation,fovy,aspect=4/3):
    p=np.asarray(rotation).reshape(3,3).T@(np.asarray(point)-np.asarray(position))
    depth=-float(p[2])
    if depth<=0:return {'in_front':False,'center_in_frame':False}
    tangent=np.tan(np.radians(fovy)/2)
    xy=[float(p[0]/depth/(tangent*aspect)),float(p[1]/depth/tangent)]
    return {'in_front':True,'center_in_frame':all(abs(v)<=1 for v in xy),'normalized_xy_up_positive':[round(v,4) for v in xy],'depth_m':round(depth,4)}

def camera_cues(case):
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(case/'sim2_robot_only_scene.xml'));d=mujoco.MjData(m)
    with np.load(case/'frame_action_mapping.npz') as z:d.qpos[:]=z['nominal_qpos'][0]
    mujoco.mj_forward(m,d)
    rows=[]
    for side in ['left','right']:
        cid=m.camera(side+'_wrist_rgbd').id;sid=m.site('arm_'+side+'_tool').id
        rows.append({'view':side+'_wrist','tool_site':'arm_'+side+'_tool',**project(d.site_xpos[sid],d.cam_xpos[cid],d.cam_xmat[cid],m.cam_fovy[cid])})
    result={'source':'Frozen MuJoCo tool sites and wrist cameras at first frame; rigid attachment relative projection.',
      'normalized_image_bounds':[-1,1],'own_tool_projection':rows,
      'limits':'Tool center only, not object center/extent/occlusion. Outside tool center does NOT prove the held object is invisible. Never move a camera or object just to make it visible; unresolved object visibility stays unknown.'}
    write(case/'camera_geometry.json',result);return result
