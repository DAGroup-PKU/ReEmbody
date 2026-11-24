"""Actual raster visibility; projected geometry alone cannot see occlusion."""
import numpy as np

MAX_INITIAL_ROBOT_COVERAGE=.55
MIN_INITIAL_WORKSPACE_COVERAGE=.45
MIN_OBJECT_PIXELS=16
MIN_REFERENCE_BOX_FILL=.03
MAX_NORMALIZED_BOX_ERROR=.15


def raster_metrics(model,segmentation,object_ids):
    import mujoco
    import cv2
    seg=np.asarray(segmentation)
    if seg.ndim!=3 or seg.shape[2]!=2:raise ValueError('MuJoCo object ID/type segmentation required')
    valid=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[:,:,0]>=0)&(seg[:,:,0]<model.ngeom)
    ids=np.where(valid,seg[:,:,0],-1);height,width=ids.shape
    robots=[];table=[];object_geoms={}
    for g in range(model.ngeom):
        body=model.body(int(model.geom_bodyid[g])).name or ''
        name=model.geom(g).name or ''
        if body.startswith(('left_','right_','astribot_')) or name.startswith('physical_robot_'):robots.append(g)
        if body=='worktable' or name=='worktable_top':table.append(g)
    for name in object_ids:
        bid=model.body(name).id
        object_geoms[name]=np.flatnonzero(model.geom_bodyid==bid)
    objects={}
    for name,geoms in object_geoms.items():
        mask=np.isin(ids,geoms)
        # ID-colour edge antialiasing can create isolated pixels labelled as
        # another valid geometry. Do not let them expand a box across a frame.
        total=int(mask.sum())
        if total:
            _,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
            keep=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=max(4,total*.001))
            mask=np.isin(labels,keep[keep!=0])
        y,x=np.nonzero(mask);count=len(x)
        objects[name]={'pixels':count,'pixel_fraction':count/(width*height),
                       'raw_segment_pixels':total,
                       'bbox_xyxy':None if not count else [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]}
    return {'image_size_wh':[width,height],'robot_pixel_fraction':float(np.isin(ids,robots).mean()),
            'table_pixel_fraction':float(np.isin(ids,table).mean()),'objects':objects,
            'background_pixel_fraction':float((ids<0).mean()),
            'basis':'MuJoCo rendered object IDs with actual depth occlusion; not unoccluded projection'}


def diagnose(rendered,landmarks):
    rows=[];issues=[]
    for kind,frames in rendered.items():
        for frame,views in frames.items():
            for view,image in views.items():
                metric=image.get('raster_visibility')
                if metric is None:
                    issues.append({'kind':kind,'view':view,'frame':int(frame),'reason':'missing_raster_evidence'})
                    continue
                row={'kind':kind,'view':view,'frame':int(frame),**metric,'initial':int(frame)==0}
                errors=[];checks=[]
                clearance=metric.get('camera_clearance')
                if clearance is not None and not clearance['passed']:
                    errors.append('camera_intersects_or_too_close_to_collision_solid')
                interaction=metric.get('interaction_visibility')
                if interaction is not None and interaction['available'] and not interaction['passed']:
                    errors.extend('contact_object_outside_camera_frustum:'+r['object_id']
                                  for r in interaction['objects'] if not r['frustum_intersection_possible'])
                if int(frame)==0:
                    if metric['robot_pixel_fraction']>MAX_INITIAL_ROBOT_COVERAGE:
                        errors.append('robot_blocks_most_of_initial_image')
                    width,height=metric['image_size_wh'];scale=max(width,height)
                    workspace=min(1.,metric['table_pixel_fraction']+sum(o['pixels'] for o in metric['objects'].values())/(width*height))
                    row['workspace_pixel_fraction']=workspace
                    if workspace<MIN_INITIAL_WORKSPACE_COVERAGE:
                        errors.append('insufficient_initial_table_and_object_view')
                    for name,reference in landmarks['views'][view].items():
                        if not reference['visible']:continue
                        actual=metric['objects'].get(name)
                        if actual is None or actual['pixels']<MIN_OBJECT_PIXELS or actual['bbox_xyxy'] is None:
                            errors.append('reference_object_not_raster_visible:'+name);continue
                        box=np.asarray(actual['bbox_xyxy'],float)
                        if reference.get('bbox_xyxy_normalized') is not None:
                            expected=np.asarray(reference['bbox_xyxy_normalized'])*[width,height,width,height]
                            area=max(1.,float(np.prod(expected[2:]-expected[:2])))
                            error=float(np.linalg.norm(box-expected)/2/scale)
                            if actual['pixels']/area<MIN_REFERENCE_BOX_FILL:
                                errors.append('reference_object_heavily_occluded:'+name)
                        elif reference.get('center_xy_normalized') is not None:
                            expected=np.asarray(reference['center_xy_normalized'])*[width,height]
                            error=float(np.linalg.norm((box[:2]+box[2:])/2-expected)/scale)
                        else:
                            errors.append('reference_object_has_no_coordinates:'+name);continue
                        checks.append({'object_id':name,'normalized_visible_extent_error':error})
                    if not checks:errors.append('no_initial_raster_correspondence')
                    elif float(np.median([c['normalized_visible_extent_error'] for c in checks]))>MAX_NORMALIZED_BOX_ERROR:
                        errors.append('visible_extent_disagrees_with_reference')
                row.update(checks=checks,errors=errors)
                rows.append(row)
                if errors:issues.append({'kind':kind,'view':view,'frame':int(frame),'reasons':errors})
    if not rows:issues.append({'reason':'no_raster_evidence'})
    return {'passed':not issues,'issues':issues,'views':rows,
            'thresholds':{'max_initial_robot_pixel_fraction':MAX_INITIAL_ROBOT_COVERAGE,
                          'min_initial_table_and_object_pixel_fraction':MIN_INITIAL_WORKSPACE_COVERAGE,
                          'min_object_pixels':MIN_OBJECT_PIXELS,'min_reference_box_fill':MIN_REFERENCE_BOX_FILL,
                          'max_normalized_extent_error':MAX_NORMALIZED_BOX_ERROR},
            'temporal_visibility_certified':False,'dynamic_frames_reviewed_by_k3':True}
