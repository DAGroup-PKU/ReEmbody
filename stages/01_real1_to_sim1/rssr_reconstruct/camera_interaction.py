"""Observe the interacting objects using saved contacts, not a hand-written task pose.

Frustum tests use conservative collision-geometry bounding spheres. Rendered ID
pixels remain the evidence of actual visibility; a sphere in view is not proof
that an object is unoccluded. No source pixel path is imposed on target motion.
"""
import numpy as np

POLICY = {'name': 'saved_contact_object_frustum_v1',
          'contact_columns': ['left', 'right', 'table'],
          'minimum_visible_pixels_for_search': 16,
          'preferred_contact_object_image_fraction': .01,
          'hard_gate': 'recorded hand-contact objects must intersect their camera frustum',
          'occlusion_is_separate': True, 'dynamic_pixel_equality_required': False}


def validate_saved_contacts(rollout, object_ids):
    """Final physical QA requires explicit contact evidence, even if all zero."""
    if 'contacts' not in rollout:
        raise ValueError('Final camera QA requires saved physical contact evidence')
    contacts = np.asarray(rollout['contacts'])
    if (len(object_ids) != len(set(object_ids)) or
            contacts.shape != (len(rollout['qpos']), len(object_ids), 3) or
            not np.isfinite(contacts).all() or np.any(contacts < 0)):
        raise ValueError('Saved camera contact evidence does not match the scene object order')


def capture(model, data, rollout, index, object_ids, view):
    if 'contacts' not in rollout:
        return {'available': False, 'reason': 'no_saved_contact_array', 'objects': []}
    contacts=np.asarray(rollout['contacts'][index])
    if len(object_ids)!=len(set(object_ids)) or contacts.shape != (len(object_ids),3) or not np.isfinite(contacts).all() or np.any(contacts<0):
        raise ValueError('Saved camera contact evidence does not match the scene object order')
    columns={'left_wrist':[0], 'right_wrist':[1], 'head':[0,1]}[view]
    rows=[]
    for i,name in enumerate(object_ids):
        if not np.any(contacts[i,columns]>0):continue
        body=model.body(name).id
        geoms=np.flatnonzero(model.geom_bodyid==body)
        spheres=[{'center_world_m':data.geom_xpos[g].tolist(), 'radius_m':float(model.geom_rbound[g])}
                 for g in geoms if model.geom_rgba[g,3]>0]
        rows.append({'object_id':name, 'contact_hands':[h for h,c in zip(('left','right'),(0,1)) if contacts[i,c]>0],
                     'spheres':spheres})
    return {'available':True,'simulation_sample_index':int(index),'objects':rows}


def inspect(model, data, camera_name, context, raster):
    if not context['available']:
        return {'policy':POLICY,'available':False,'passed':None,'objects':[], 'soft_loss':0.,
                'reason':context['reason']}
    cid=model.camera(camera_name).id
    width,height=raster['image_size_wh']
    preferred_pixels=max(POLICY['minimum_visible_pixels_for_search'],
                         POLICY['preferred_contact_object_image_fraction']*width*height)
    tan_y=float(np.tan(np.deg2rad(model.cam_fovy[cid])/2));tan_x=tan_y*width/height
    near=float(model.vis.map.znear*model.stat.extent)
    far=float(model.vis.map.zfar*model.stat.extent)
    rotation=data.cam_xmat[cid].reshape(3,3);position=data.cam_xpos[cid]
    rows=[]
    for obj in context['objects']:
        violations=[]
        for sphere in obj['spheres']:
            point=(np.asarray(sphere['center_world_m'])-position) @ rotation
            depth=-float(point[2]);radius=sphere['radius_m']
            # Signed distances to the six frustum planes, normalized to metres.
            violations.append(max(0., near-depth-radius, depth-far-radius,
                (abs(point[0])-depth*tan_x)/np.sqrt(1+tan_x*tan_x)-radius,
                (abs(point[1])-depth*tan_y)/np.sqrt(1+tan_y*tan_y)-radius))
        gap=float(min(violations)) if violations else None
        pixels=int(raster['objects'].get(obj['object_id'],{}).get('pixels',0))
        in_frustum=bool(gap is not None and gap<=1e-9)
        # Prefer seeing the contacted object even when its bounding sphere is
        # already in view. This never moves the object to satisfy the camera.
        # Sixteen pixels detect presence but do not provide a useful view of
        # an interaction. Keep a continuous preference for a larger visible
        # patch; this is not forced equality to source object pixels.
        loss=(20. if gap is None else 20*gap/.05)+3*max(0.,1-pixels/preferred_pixels)
        rows.append({'object_id':obj['object_id'],'contact_hands':obj['contact_hands'],
                     'frustum_intersection_possible':in_frustum,'frustum_gap_m':gap,
                     'visible_pixels':pixels,'raster_visible':pixels>=POLICY['minimum_visible_pixels_for_search'],
                     'preferred_visible_pixels':preferred_pixels,
                     'soft_loss':float(loss)})
    return {'policy':POLICY,'available':True,'passed':all(r['frustum_intersection_possible'] for r in rows),
            'objects':rows,'soft_loss':sum(r['soft_loss'] for r in rows),
            'actual_occlusion_still_requires_review':True}
