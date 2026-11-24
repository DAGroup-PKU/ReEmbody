"""Read-only component geometry at saved states, never a grasp prescription."""
import numpy as np


def closest_solid_point(point, kind, size):
    """Euclidean projection onto a closed analytic solid in its local frame.

    An interior point projects to itself. This is not signed penetration,
    finger-to-object clearance, a surface normal, or a force-closure test.
    """
    p=np.asarray(point,float);s=np.asarray(size,float)
    if p.shape!=(3,) or s.shape!=(3,) or not np.isfinite([*p,*s]).all():
        raise ValueError('Finite three-dimensional primitive coordinates required')
    if kind=='box':return np.clip(p,-s,s)
    if kind=='sphere':return p*min(1.,s[0]/max(np.linalg.norm(p),1e-30))
    if kind=='cylinder':
        q=p.copy();q[:2]*=min(1.,s[0]/max(np.linalg.norm(p[:2]),1e-30))
        q[2]=np.clip(q[2],-s[1],s[1]);return q
    if kind=='capsule':
        center=np.array([0.,0.,np.clip(p[2],-s[1],s[1])]);delta=p-center
        return center+delta*min(1.,s[0]/max(np.linalg.norm(delta),1e-30))
    raise ValueError('Unsupported analytic collision primitive')


def component_proximity(model,data,obj,hand,site_from_grasp_rotation=None):
    """Use current FK only; fixed names bind this backend's two distal pads."""
    import mujoco
    types={int(getattr(mujoco.mjtGeom,'mjGEOM_'+k.upper())):k
           for k in ('box','sphere','cylinder','capsule')}
    root=model.body(obj).id
    descendants={root}
    for body in range(model.nbody):
        parent=body
        while parent and parent not in descendants:parent=int(model.body_parentid[parent])
        if parent in descendants:descendants.add(body)
    geoms=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    body=model.body('astribot_gripper_'+hand+'_base').id
    rotation=data.xmat[body].reshape(3,3)
    site=model.site('arm_'+hand+'_tool').id;origin=data.site_xpos[site]
    grasp=None
    if site_from_grasp_rotation is not None:
        conversion=np.asarray(site_from_grasp_rotation,float)
        if (conversion.shape!=(3,3) or not np.isfinite(conversion).all() or
            not np.allclose(conversion.T@conversion,np.eye(3),atol=1e-6,rtol=0) or
            not np.isclose(np.linalg.det(conversion),1.,atol=1e-6,rtol=0)):
            raise ValueError('Proper site-from-grasp rotation required for geometry feedback')
        grasp=data.site_xmat[site].reshape(3,3)@conversion
    pads=[]
    for suffix in ('L112','R112'):
        gid=model.geom('astribot_gripper_'+hand+'_geom_'+suffix).id
        point=data.geom_xpos[gid];choices=[]
        for geom in geoms:
            kind=types.get(int(model.geom_type[geom]))
            if kind is None:continue
            r=data.geom_xmat[geom].reshape(3,3);center=data.geom_xpos[geom]
            local=r.T@(point-center)
            closest=center+r@closest_solid_point(local,kind,model.geom_size[geom])
            vector=closest-point
            choices.append({'geom':model.geom(geom).name or str(geom),'primitive':kind,
                            'point_to_solid_distance_m':float(np.linalg.norm(vector)),
                            'pad_point_to_solid_world_m':vector.tolist(),
                            'pad_point_to_solid_hand_frame_m':(rotation.T@vector).tolist()})
            if grasp is not None:choices[-1]['pad_point_to_solid_grasp_frame_m']=(grasp.T@vector).tolist()
        choices.sort(key=lambda row:(row['point_to_solid_distance_m'],row['geom']))
        pads.append({'geom':model.geom(gid).name,'center_world_m':point.tolist(),
                     'tool_site_to_pad_center_hand_frame_m':(rotation.T@(point-origin)).tolist(),
                     'nearest_components':choices[:2]})
        if grasp is not None:pads[-1]['tool_site_to_pad_center_grasp_frame_m']=(grasp.T@(point-origin)).tolist()
    result={'hand':hand,'object':obj,'hand_frame_body':model.body(body).name,
            'hand_frame_axes_world':rotation.tolist(),'tool_site':model.site(site).name,
            'tool_site_world_m':origin.tolist(),'pads':pads,
            'object_collision_component_count':len(geoms),
            'unsupported_components':[model.geom(g).name or str(g) for g in geoms
                                      if int(model.geom_type[g]) not in types],
            'scope':'Exact analytic point-to-solid projections from distal pad centers, not finger clearances or contact/grasp success. Each pad chooses its nearest components independently; no grasp site, repair direction, or parameter is prescribed. Interior points have zero distance. Unsupported shapes are listed, not replaced by a solid object bounding box.'}
    if grasp is not None:result['declared_grasp_frame_axes_world']=grasp.tolist()
    return result
