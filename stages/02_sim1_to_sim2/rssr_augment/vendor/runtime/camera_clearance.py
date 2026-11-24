"""Lens clearance from primitive collision solids at the saved physical pose.

This detects cameras inside task objects/table/colliders. Meshes and calibration
are not certified by this check. No geometry, camera or physical state is edited.
"""
import numpy as np

MIN_LENS_CLEARANCE_M = .003
POLICY = {'name': 'primitive_camera_clearance_v1', 'minimum_clearance_m': MIN_LENS_CLEARANCE_M,
          'scope': 'visible collidable primitives at sampled saved states',
          'renderer_near_clip_sphere_also_required': True,
          'mesh_clearance_certified': False}


def signed_distance(local, size, kind):
    import mujoco
    p=np.asarray(local,dtype=float);size=np.asarray(size,dtype=float)
    if kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):return float(np.linalg.norm(p)-size[0])
    if kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(np.linalg.norm([p[0],p[1],max(0.,abs(p[2])-size[1])])-size[0])
    if kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        d=np.array([np.linalg.norm(p[:2])-size[0],abs(p[2])-size[1]])
    elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):d=np.abs(p)-size
    elif kind==int(mujoco.mjtGeom.mjGEOM_PLANE):return float(p[2])
    else:return None
    return float(np.linalg.norm(np.maximum(d,0))+min(float(np.max(d)),0.))


def inspect(model,data,camera_name):
    cid=model.camera(camera_name).id
    position=data.cam_xpos[cid]
    required=max(MIN_LENS_CLEARANCE_M,float(model.vis.map.znear*model.stat.extent))
    nearest=[];unsupported=0;checked=0
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]) or model.geom_rgba[g,3]<=0:
            continue
        local=(position-data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3,3)
        distance=signed_distance(local,model.geom_size[g],int(model.geom_type[g]))
        if distance is None:
            unsupported+=1
            continue
        checked+=1
        if len(nearest)<4 or distance<nearest[-1]['distance_m']:
            nearest.append({'geom':model.geom(g).name or str(g),
                            'body':model.body(int(model.geom_bodyid[g])).name,
                            'distance_m':distance})
            nearest=sorted(nearest,key=lambda row:row['distance_m'])[:4]
    minimum=min((r['distance_m'] for r in nearest),default=None)
    return {'policy':POLICY,'passed':minimum is not None and minimum>=required,
            'minimum_required_clearance_m':required,
            'minimum_signed_distance_m':minimum,'nearest':nearest,
            'checked_primitives':checked,'unsupported_collision_geometries':unsupported,
            'complete_geometry_clearance_certified':unsupported==0 and checked>0}
