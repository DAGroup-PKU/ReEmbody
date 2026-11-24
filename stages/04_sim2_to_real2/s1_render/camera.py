"""Fixed camera-mount adjustments composed with native per-frame kinematics."""
import math
import numpy as np

CAMERAS=('head_rgbd','left_wrist_rgbd','right_wrist_rgbd')


def transform(position,rotation):
    out=np.eye(4);out[:3,:3]=np.asarray(rotation,float).reshape(3,3);out[:3,3]=np.asarray(position,float)
    return out


def quaternion_rotation(wxyz):
    q=np.asarray(wxyz,float);assert q.shape==(4,) and np.isfinite(q).all()
    q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def camera_adjustment(spec):
    """Delta is in the native camera's local frame and is identical at all times."""
    if 'matrix_in_camera' in spec:
        delta=np.asarray(spec['matrix_in_camera'],float)
    else:
        delta=transform(spec.get('position_in_camera_m',[0,0,0]),
            quaternion_rotation(spec.get('quaternion_in_camera_wxyz',[1,0,0,0])))
    assert delta.shape==(4,4) and np.isfinite(delta).all()
    assert np.allclose(delta[3],[0,0,0,1]) and np.allclose(delta[:3,:3].T@delta[:3,:3],np.eye(3),atol=1e-7)
    assert abs(np.linalg.det(delta[:3,:3])-1)<1e-7
    return delta


def fovy(spec,native_fovy,reference=False):
    ref=float(spec.get('reference_fovy_deg',native_fovy))
    aspect=float(spec.get('reference_aspect',16/9))
    square=math.degrees(2*math.atan(aspect*math.tan(math.radians(ref)/2)))
    value=float(spec.get('fovy_deg',square))
    assert 1<ref<170 and 1<value<170 and aspect>0
    return ref if reference else value
