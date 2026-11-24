"""Check native moving geometry against the fixed task-facing head camera."""
from __future__ import annotations
import itertools
import math
import numpy as np
from .measured_s1 import CUP

def project(model,data,points):
    camera=model.camera('head_rgbd').id
    local=(np.asarray(points)-data.cam_xpos[camera])@data.cam_xmat[camera].reshape(3,3)
    depth=-local[:,2]
    tangent=math.tan(math.radians(float(model.cam_fovy[camera]))/2)
    uv=local[:,:2]/np.maximum(depth[:,None],1e-9)/tangent
    uv[:,0]/=(16/9)
    return uv,depth


class HeadViewAudit:
    def __init__(self,model,scene):
        self.model=model;self.count=0;self.maximum_extent=0.;self.minimum_depth=math.inf
        self.all_visible=True;self.all_aims_hit_table=True
        self.rows=[]
        for obj in scene['objects']:
            # Logical transparent target regions have no visible surface.
            if obj['shape']=='region':continue
            if obj['shape']=='cup':
                local=[(diameter/2*math.cos(a),diameter/2*math.sin(a),z)
                       for diameter,z in ((CUP['bottom_diameter_m'],-CUP['height_m']/2),
                                          (CUP['top_diameter_m'],CUP['height_m']/2))
                       for a in np.linspace(0,2*math.pi,32,endpoint=False)]
            else:local=np.array(list(itertools.product((-1,1),repeat=3)))*np.array(obj['size_m'])/2
            self.rows.append((model.body(obj['id']).id,np.asarray(local)))

    def observe(self,data):
        cloud=np.concatenate([local@data.xmat[body].reshape(3,3).T+data.xpos[body] for body,local in self.rows])
        uv,depth=project(self.model,data,cloud);extent=float(np.max(np.abs(uv)))
        self.maximum_extent=max(self.maximum_extent,extent);self.minimum_depth=min(self.minimum_depth,float(depth.min()))
        self.all_visible=bool(self.all_visible and np.all(depth>.05) and extent<=.98)
        camera=self.model.camera('head_rgbd').id;top=self.model.geom('worktable_top').id
        direction=-data.cam_xmat[camera].reshape(3,3)[:,2]
        z=data.geom_xpos[top,2]+self.model.geom_size[top,2]
        distance=(z-data.cam_xpos[camera,2])/direction[2] if direction[2]<-.01 else -1
        hit=data.cam_xpos[camera]+distance*direction
        inside=bool(distance>0 and np.all(np.abs(hit[:2]-data.geom_xpos[top,:2])<=self.model.geom_size[top,:2]))
        self.all_aims_hit_table=bool(self.all_aims_hit_table and inside);self.count+=1

    def report(self):
        return dict(schema='rssr.s1.reference_head_view_audit.v1',physical_observations=self.count,
            all_task_geometry_in_view=self.all_visible,optical_axis_always_hits_table=self.all_aims_hit_table,
            maximum_normalized_extent=self.maximum_extent,minimum_camera_depth_m=self.minimum_depth,
            same_fixed_camera_mount_for_whole_trajectory=True,physical_head_joint_augmentation=False,reference_aspect=16/9,
            reference_vertical_fovy_deg=float(self.model.cam_fovy[self.model.camera('head_rgbd').id]),
            output_square_policy='preserve reference horizontal FOV; renderer expands square vertical coverage',
            geometry_scope='physical task objects; non-rendered logical regions excluded',
            real_hardware_extrinsics_calibrated=False,
            passed=bool(self.count and self.all_visible and self.all_aims_hit_table))
