"""Explicit frame semantics for bounded orientation interpolation.

The legacy site-axis operation is preserved so old learned recipes keep their
meaning. The Agent may instead choose the grasp frame declared by the robot
contract. Neither mode selects a grasp point or prescribes a mixing value.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

FRAMES=('tool_site_legacy','declared_grasp')


def vertical_alignment(site_rotations,mix,frame='tool_site_legacy',site_from_grasp=None):
    if frame not in FRAMES:raise ValueError('Unknown vertical alignment frame')
    if type(mix) not in (int,float) or not np.isfinite(mix) or not 0<=mix<=1:
        raise ValueError('Vertical alignment mix must be in [0,1]')
    rot=np.asarray(site_rotations,float).copy()
    if rot.ndim!=3 or rot.shape[1:]!=(3,3) or not np.isfinite(rot).all():
        raise ValueError('Finite site rotation sequence required')
    conversion=None
    if frame=='declared_grasp':
        conversion=np.asarray(site_from_grasp,float)
        if (conversion.shape!=(3,3) or not np.isfinite(conversion).all() or
            not np.allclose(conversion.T@conversion,np.eye(3),atol=1e-6,rtol=0) or
            not np.isclose(np.linalg.det(conversion),1.,atol=1e-6,rtol=0)):
            raise ValueError('Proper declared site-from-grasp rotation required')
    if mix:
        for k,r in enumerate(rot):
            observed=r if conversion is None else r@conversion
            x=observed[:,0].copy();x[2]=0
            if np.linalg.norm(x)<1e-8:x=np.array([1.,0.,0.])
            x/=np.linalg.norm(x);z=np.array([0.,0.,-1.]);vertical=np.c_[x,np.cross(z,x),z]
            goal=vertical if conversion is None else vertical@conversion.T
            rot[k]=Slerp([0,1],Rotation.from_matrix(np.stack([r,goal])))(mix).as_matrix()
    return rot
