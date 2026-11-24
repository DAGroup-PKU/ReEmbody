#!/usr/bin/env python3
"""Read-only FK audit of FR3 Duo grasps against the actual exported Blender rig.

Does not simulate, import a controller, change a scene, render, or contact HF.
Checks are explicit candidate engineering criteria, not retroactive task labels.
The 30 Hz native states cannot establish continuous visibility or contact forces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_DATASET = Path(__file__).resolve().parents[3] / 'examples'
DEFAULT_RENDERER = Path(__file__).resolve().parents[3] / 'stages/04_sim2_to_real2/fr3_site_render/render_site_v2.py'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def rotation(q):
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    return r.reshape(3, 3)


def angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=-1)/np.maximum(n, 1e-15), -1, 1)))


def first(mask):
    a = np.flatnonzero(mask)
    return int(a[0]) if len(a) else None


def stats(x):
    x = np.asarray(x, float)
    if not len(x):
        return None
    return dict(min=float(x.min()), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def nearest_triangle_point(point, triangles):
    """Nearest point on an explicit triangle soup; no external mesh queries."""
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab,ac=b-a,c-a
    ap=point-a
    aa=np.einsum('ij,ij->i',ab,ab);bb=np.einsum('ij,ij->i',ab,ac);cc=np.einsum('ij,ij->i',ac,ac)
    d=np.einsum('ij,ij->i',ap,ab);e=np.einsum('ij,ij->i',ap,ac)
    det=aa*cc-bb*bb
    valid=abs(det)>1e-20
    u=np.zeros(len(a));v=np.zeros(len(a))
    u[valid]=(d[valid]*cc[valid]-e[valid]*bb[valid])/det[valid]
    v[valid]=(e[valid]*aa[valid]-d[valid]*bb[valid])/det[valid]
    inside=valid&(u>=0)&(v>=0)&(u+v<=1)
    candidates=[(a+u[:,None]*ab+v[:,None]*ac)[inside]]
    for begin,end in ((a,b),(b,c),(c,a)):
        edge=end-begin
        w=np.clip(np.einsum('ij,ij->i',point-begin,edge)/np.maximum(np.einsum('ij,ij->i',edge,edge),1e-30),0,1)
        candidates.append(begin+w[:,None]*edge)
    points=np.concatenate(candidates)
    return points[int(np.argmin(np.linalg.norm(points-point,axis=1)))]


def nearest_object_surface(m,d,obj,point):
    """Diagnostic nearest collision-surface patch, in the actual object's frame.

    This anchors an aim point to an existing surface, not to an invented
    camera mount or free-floating spatial target. It is not a new grasp plan.
    """
    bid=m.body(obj).id
    best=None
    for gid in range(m.ngeom):
        if int(m.geom_bodyid[gid])!=bid or not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
            continue
        R=d.geom_xmat[gid].reshape(3,3);pos=d.geom_xpos[gid]
        p=R.T@(point-pos)
        kind=int(m.geom_type[gid]);size=m.geom_size[gid]
        if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(m.geom_dataid[gid]);va=int(m.mesh_vertadr[mid]);vn=int(m.mesh_vertnum[mid]);fa=int(m.mesh_faceadr[mid]);fn=int(m.mesh_facenum[mid])
            vertices=m.mesh_vert[va:va+vn].astype(float);faces=m.mesh_face[fa:fa+fn]
            near=nearest_triangle_point(p,vertices[faces])
        elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):
            near=np.clip(p,-size,size)
            if np.all(abs(p)<=size):
                axis=int(np.argmin(size-abs(p)));near[axis]=size[axis]*(1 if p[axis]>=0 else -1)
        elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radial=p[:2]/max(np.linalg.norm(p[:2]),1e-15)*size[0]
            wall=np.r_[radial,np.clip(p[2],-size[1],size[1])]
            capxy=p[:2]*min(1.,size[0]/max(np.linalg.norm(p[:2]),1e-15))
            cap=np.r_[capxy,size[1]*(1 if p[2]>=0 else -1)]
            near=min((wall,cap),key=lambda v:np.linalg.norm(v-p))
        else:
            continue
        world=pos+R@near;distance=float(np.linalg.norm(world-point))
        if best is None or distance<best['distance_from_contact_tcp_m']:
            local=d.xmat[bid].reshape(3,3).T@(world-d.xpos[bid])
            best=dict(geom_name=m.geom(gid).name,geom_id=gid,object_local_m=local.tolist(),world_at_contact_m=world.tolist(),
                      distance_from_contact_tcp_m=distance,source='nearest actual native collision triangle/box/cylinder surface to measured contact TCP')
    return best


def camera_projection(point, eye, R, fovy, aspect=4/3):
    # Renderer consumes cam_quat directly (Blender -Z optical axis, +Y up).
    local = np.einsum('...ji,...j->...i', R, point-eye)
    depth = -local[..., 2]
    tangent = np.tan(np.deg2rad(fovy)/2)
    uv = local[..., :2] / np.maximum(depth[..., None], 1e-15) / tangent
    uv[..., 0] /= aspect
    return uv, depth


def first_sustained(mask, t, duration):
    start = None
    for i, good in enumerate(mask):
        start = i if good and start is None else start if good else None
        if start is not None and t[i]-t[start] >= duration-1e-8:
            return start
    return None


def render_binding(meta_row, run, render_source, renderer):
    source_json = render_source/'source.json'
    sm = read(source_json)
    receipt_path = Path(meta_row['render_receipt']) if meta_row else None
    rr = read(receipt_path) if receipt_path else None
    config = receipt_path.parent.parent/'config.json' if receipt_path else None
    camera_json = render_source/'cameras.json'
    cams = read(camera_json)
    checks = dict(native_inputs_match_render_source=all(
        Path(p).is_file() and sha(p) == h for p, h in sm.get('inputs', {}).items()),
        nonempty_native_input_binding=bool(sm.get('inputs')),
        render_source_episode_matches_this_run=sm.get('inputs', {}).get(str(run/'episode.npz')) == sha(run/'episode.npz'),
        exported_pose_hash_matches_source=sm.get('artifacts', {}).get('poses.npz') == sha(render_source/'poses.npz'),
        camera_spec_hash_matches_source=sm.get('artifacts', {}).get('cameras.json') == sha(camera_json))
    if rr:
        checks.update(published_episode_hash_matches=meta_row.get('episode_sha256') == sha(run/'episode.npz'),
                      published_record_hash_matches=meta_row.get('record_sha256') == sha(run/'record.json'),
                      published_render_receipt_hash_matches=meta_row.get('render_receipt_sha256') == sha(receipt_path),
                      render_input_source_hash_matches=any(Path(p).name == 'source.json' and h == sha(source_json)
                                                           for p, h in rr.get('input_pins', {}).items()),
                      render_input_config_hash_matches=config.is_file() and any(Path(p).name == 'config.json' and h == sha(config)
                                                           for p, h in rr.get('input_pins', {}).items()),
                      raw_raster_is_4_by_3=rr.get('raw_resolution') == [224, 168],
                      output_raster_is_224_square=rr.get('resolution') == [224, 224])
    else:
        checks['actual_render_execution_binding'] = None
    # Explicitly bind the renderer's real camera path, rather than assume XML
    # cameras and Blender mounts are interchangeable. No imported renderer runs.
    text = renderer.read_text()
    checks['renderer_consumes_exported_camera_pose_directly'] = all(s in text for s in [
        "co.location = P['cam_pos'][k, i] + off", "co.rotation_quaternion = P['cam_quat'][k, i]",
        "for name, fovy in zip(cams['names'], cams['fovy_deg']):", "cd.angle_y = math.radians(fovy)"])
    pins = {str(p):sha(p) for p in (source_json, camera_json, render_source/'poses.npz', renderer)}
    if receipt_path:
        pins.update({str(receipt_path):sha(receipt_path), str(config):sha(config)})
    return cams, sm, dict(checks=checks, verified=all(v is True for v in checks.values()),
        renderer_camera_semantics='Blender receives exported camera quaternion and position, plus the same global translation as objects. No camera-specific adjustment.',
        raster='4:3 raw224x168 then28px padding above/below; projection uses the raw sensor.', pins=pins)


def audit(run, render_source, renderer, *, meta_row=None, gaze_deg=10., tilt_deg=15., translation_m=.005,
          xy_m=.012, above_m=.035, hold_s=.1, duration_limit=20.):
    run, render_source, renderer = map(lambda p: Path(p).resolve(strict=True), (run, render_source, renderer))
    record = read(run/'record.json')
    cams, source, rig = render_binding(meta_row, run, render_source, renderer)
    m = mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
    d = mujoco.MjData(m)
    with np.load(run/'episode.npz', allow_pickle=False) as z:
        ep = {k:z[k].copy() for k in z.files}
    with np.load(render_source/'poses.npz', allow_pickle=False) as z:
        poses = {k:z[k].copy() for k in z.files}
    t, q = ep['time'], ep['qpos']
    if not (q.shape[0] == len(t) and np.isfinite(q).all() and np.all(np.diff(t)>0)):
        raise ValueError('Native state arrays are incomplete or nonfinite')
    grasps = record.get('measurements',{}).get('grasps',{})
    obj_names = [o['id'] for o in source['objects']]
    body_names = {m.body(i).name for i in range(m.nbody)}
    # Articulated cabinet parts are emitted separately by the drawer renderer,
    # and are consequently absent from source.objects (the generic free-body
    # list). Their actual model bodies remain available for honest FK metrics.
    obj_names += [key.split(':',1)[1] for key in grasps
                  if key.split(':',1)[1] not in obj_names and key.split(':',1)[1] in body_names]
    obj_names = list(dict.fromkeys(obj_names))
    names = cams['names']
    ci = {n:i for i,n in enumerate(names)}
    ids = [m.camera(n).id for n in names]
    cp, cr = np.empty((len(t),len(ids),3)), np.empty((len(t),len(ids),3,3))
    tcp = np.empty((len(t),2,3))
    axis = np.empty_like(tcp)
    object_positions = {n:np.empty((len(t),3)) for n in obj_names}
    object_rotations = {n:np.empty((len(t),3,3)) for n in obj_names}
    for k, state in enumerate(q):
        d.qpos[:] = state
        # Kinematics + camera propagation only. No stepping, force inference,
        # contact recomputation or replacement object trajectories.
        mujoco.mj_kinematics(m,d)
        mujoco.mj_camlight(m,d)
        cp[k], cr[k] = d.cam_xpos[ids], d.cam_xmat[ids].reshape(-1,3,3)
        for hi,h in enumerate(('left','right')):
            si = m.site(f'{h}_tcp').id
            tcp[k,hi], axis[k,hi] = d.site_xpos[si], d.site_xmat[si].reshape(3,3)[:,2]
        for n in obj_names:
            b = m.body(n).id
            object_positions[n][k], object_rotations[n][k] = d.xpos[b], d.xmat[b].reshape(3,3)
    # Prove rigid rig applicability on all exported states and all cameras.
    pose_match = poses['cam_pos'].shape == cp.shape and poses['cam_quat'].shape[:2] == cp.shape[:2]
    rig['checks']['exported_camera_shape_matches_native_frames'] = bool(pose_match)
    if pose_match:
        exported_r = np.array([[rotation(v) for v in row] for row in poses['cam_quat']])
        err_p = float(np.max(abs(cp-poses['cam_pos'])))
        err_r = float(np.max(abs(cr-exported_r)))
        rig['native_to_exported_camera_max_abs_position_m'] = err_p
        rig['native_to_exported_camera_max_abs_rotation'] = err_r
        rig['checks']['all_native_camera_poses_match_actual_export'] = err_p<1e-8 and err_r<1e-8
    else:
        rig['checks']['all_native_camera_poses_match_actual_export'] = False
    rig['verified'] = all(v is True for v in rig['checks'].values())
    # Separately bind actual saved objects and TCP samples to the same FK.
    free = ep['free_bodies'].tolist()
    object_errors = {n:float(np.max(abs(object_positions[n]-ep['objects'][:,free.index(n),:3])))
                     for n in obj_names if n in free}
    tcp_error = float(max(np.max(abs(tcp[:,hi]-ep['tcp'][:,7*hi:7*hi+3])) for hi in range(2)))
    manip_duration = float(t[-1]-t[0])
    steps = record.get('measurements',{}).get('physics_steps')
    full_duration = float(steps*m.opt.timestep) if isinstance(steps,int) else None
    output = dict(schema='rssr.fr3_duo.camera_topdown_fk_audit.v1', native=str(run), variation=record.get('variation'),
        native_admitted=record.get('admitted'), rig=rig,
        clocks=dict(recorded_motion_s=manip_duration, physical_step_s=float(m.opt.timestep),
                    recorded_motion_le_limit=manip_duration<=duration_limit+1e-8, physics_steps=steps,
                    full_physics_s=full_duration, unexported_terminal_physics_s=None if full_duration is None else full_duration-manip_duration,
                    full_physics_le_limit=None if full_duration is None else full_duration<=duration_limit+1e-8,
                    limit_s=duration_limit, native_state_count=len(t), maximum_recorded_gap_s=float(np.diff(t).max())),
        kinematic_binding=dict(saved_tcp_max_position_error_m=tcp_error, saved_free_object_max_position_error_m=object_errors,
                               actual_state_not_action_used=True),
        criteria=dict(gaze_axis_error_deg=gaze_deg, topdown_axis_error_deg=tilt_deg,
                      first_tcp_translation_m=translation_m, overhead_xy_error_m=xy_m,
                      overhead_clearance_above_object_top_m=above_m, sustained_alignment_s=hold_s,
                      status='engineering diagnostics only; these provisional numerical settings are not user-specified hard gates and do not establish requested-goal admission'),
        grasps={}, simulation_steps=0,
        limitations=['Geometry/FOV is not rendered-pixel visibility; occlusion is not checked.',
                     'States are recorded at30Hz; no continuous-time camera or approach claim.',
                     'Saved contact intervals are read as evidence; no forces are reconstructed.',
                     'Final settle is not exported in episode.npz; full time is derived separately from actual physics step count.'])
    for key, intervals in grasps.items():
        h, obj = key.split(':',1)
        if h not in ('left','right') or obj not in object_positions:
            output['grasps'][key] = dict(status='unknown',reason='Target missing from actual render-source objects')
            continue
        hi, camera = ('left','right').index(h), ci[f'{h}_wrist']
        center = object_positions[obj]
        spec = next((o for o in source['objects'] if o['id']==obj),dict(id=obj,shape=None))
        size = spec.get('size_m')
        half = float(size[2])/2 if isinstance(size,list) and len(size)==3 else None
        # Rim/top center is computed in the moving object frame. Report both
        # centroid and top aim so the API sees what target definition changes.
        top = None if half is None else center+object_rotations[obj][:,:,2]*half
        forward = -cr[:,camera,:,2]
        aim = {label:angle(forward, point-cp[:,camera]) for label,point in [('centroid',center),('top',top)] if point is not None}
        uv, depth = camera_projection(center,cp[:,camera],cr[:,camera],cams['fovy_deg'][camera])
        tilt = angle(axis[:,hi],np.array([0.,0.,-1.]))
        rows=[]
        for interval in intervals:
            contact_s=float(interval[0])
            end_i=min(int(np.searchsorted(t,contact_s)),len(t)-1)
            pre=np.arange(end_i+1)
            angles=aim['centroid']
            align_i=first_sustained((angles[:end_i+1]<=gaze_deg)&(depth[:end_i+1]>.005),t[:end_i+1],hold_s)
            move_i=first(np.linalg.norm(tcp[:end_i+1,hi]-tcp[0,hi],axis=1)>translation_m)
            rel=tcp[:,hi]-center
            topz = None if top is None else top[:,2]
            # Wall-pinch grasps intentionally offset TCP from the object
            # centroid. Test a vertical approach above the measured grasp XY,
            # while also recording the centroid offset; do not invent a center
            # grasp requirement for that task.
            contact_offset_xy = rel[end_i,:2].copy()
            grasp_xy_error = np.linalg.norm(rel[:end_i+1,:2]-contact_offset_xy,axis=1)
            above_i=None if topz is None else first((grasp_xy_error<=xy_m)&
                          (tcp[:end_i+1,hi,2]-topz[:end_i+1]>=above_m)&(tilt[:end_i+1]<=tilt_deg))
            descent=None
            if above_i is not None:
                tail=np.arange(above_i,end_i+1)
                fall=float(tcp[above_i,hi,2]-tcp[end_i,hi,2])
                relxy=rel[tail,:2]
                drift=float(np.max(np.linalg.norm(relxy-relxy[0],axis=1)))
                rise=float(np.maximum(np.diff(tcp[tail,hi,2]),0).sum())
                descent=dict(start_s=float(t[above_i]),end_s=float(t[end_i]),vertical_drop_m=fall,
                             maximum_horizontal_drift_m=drift,cumulative_upward_motion_m=rise,
                             maximum_approach_axis_tilt_deg=float(tilt[tail].max()),
                             passed=fall>=.025 and drift<=.02 and rise<=.005 and bool((tilt[tail]<=tilt_deg).all()))
            close_i=None
            if 'action' in ep and len(ep['action'])==len(t)-1:
                width=ep['action'][:,8*hi+7]
                crossings=np.flatnonzero((width<.02)&np.r_[True,width[:-1]>=.02])
                # Match the transition that caused this object's saved
                # contact interval. Initial .085 -> .035 preshaping never
                # crosses .02 and cannot count as loaded grasp closure.
                nearby=[int(i) for i in crossings if contact_s-.75<=t[i]<=contact_s+.10]
                if len(nearby)==1:
                    threshold_i=nearby[0]
                    start=threshold_i
                    while start>0 and width[start]-width[start-1]<-1e-6:
                        start-=1
                    close_i=min(start+1,len(width)-1)
                elif len(nearby)>1:
                    close_i=None  # ambiguous transitions are not guessed
            camera_rows={label:dict(initial_deg=float(a[0]),precontact_deg=stats(a[pre]),at_first_contact_deg=float(a[end_i])) for label,a in aim.items()}
            d.qpos[:]=q[end_i];mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
            patch=nearest_object_surface(m,d,obj,tcp[end_i,hi])
            patch_metrics=None
            if patch:
                target=center+np.einsum('tij,j->ti',object_rotations[obj],np.array(patch['object_local_m']))
                patch_angles=angle(forward,target-cp[:,camera])
                puv,pdepth=camera_projection(target,cp[:,camera],cr[:,camera],cams['fovy_deg'][camera])
                patch_metrics=dict(patch=patch,aim_error_deg=dict(initial=float(patch_angles[0]),precontact=stats(patch_angles[pre]),at_contact=float(patch_angles[end_i])),
                                   projected_in_raw_frame_fraction=float(np.mean((np.max(abs(puv[pre]),axis=1)<=1)&(pdepth[pre]>.005))),
                                   initial_uv=puv[0].tolist(),at_contact_uv=puv[end_i].tolist(),
                                   diagnostic_only=True,not_a_new_grasp_proposal=True,
                                   target_selection='post-hoc diagnostic anchored to measured source contact; a new controller must select and freeze its own legal object patch before execution')
            bbox=None
            if isinstance(size,list) and len(size)==3:
                vertices=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])*np.array(size)/2
                corners=center[:,None,:]+np.einsum('tij,nj->tni',object_rotations[obj],vertices)
                buv,bdepth=camera_projection(corners,cp[:,camera,None,:],cr[:,camera,None,:,:],cams['fovy_deg'][camera])
                seen=(np.max(abs(buv),axis=2)<=1)&(bdepth>.005)
                bbox=dict(precontact_all8corners_in_frame_fraction=float(np.mean(np.all(seen[pre],axis=1))),
                          precontact_anycorner_in_frame_fraction=float(np.mean(np.any(seen[pre],axis=1))),
                          at_contact_visible_corner_count=int(seen[end_i].sum()),occlusion_checked=False,
                          definition='size_m oriented box corners, not pixel silhouette')
            tracking=None if align_i is None else bool((angles[align_i:end_i+1]<=gaze_deg).all())
            rows.append(dict(saved_contact_interval_s=interval, first_actual_recorded_contact_sample_s=float(t[end_i]),
                camera=cams['names'][camera],aim_error=camera_rows,
                object_bound_surface_patch_aim=patch_metrics,object_oriented_bbox_projection=bbox,
                centroid_projected_in_raw_frame_fraction=float(np.mean((np.max(abs(uv[pre]),axis=1)<=1)&(depth[pre]>.005))),
                initial_centroid_uv=uv[0].tolist(), at_contact_centroid_uv=uv[end_i].tolist(),
                initial_camera_alignment_s=None if align_i is None else float(t[align_i]),
                first_tcp_translation_s=None if move_i is None else float(t[move_i]),
                aligned_before_translation=None if move_i is None else False if align_i is None else align_i<move_i,
                camera_tracks_until_contact=tracking,
                first_detected_close_ramp_s=None if close_i is None else float(t[close_i]),
                close_matching='unique command crossing below .02m within[-.75,+.10]s of this saved object contact; backtrack contiguous narrowing ramp',
                overhead_achieved_s=None if above_i is None else float(t[above_i]),
                overhead_reference=dict(measured_contact_tcp_minus_object_xy_m=contact_offset_xy.tolist(),
                                        test='TCP above the measured contact XY, not necessarily above the object centroid'),
                above_then_descend=descent,
                approach_axis_tilt_deg=dict(initial=float(tilt[0]),at_contact=float(tilt[end_i]),precontact=stats(tilt[pre])),
                topdown_before_closure=None if close_i is None else bool(tilt[close_i]<=tilt_deg),
                axis_definition='actual left_tcp/right_tcp site local+Z, documented FR3 approach axis'))
        output['grasps'][key]=dict(status='measured',object_shape=spec.get('shape'),
            target_geometry_scope=('articulated drawer body; its origin is not the grasped handle and its gaze is diagnostic only'
                                   if obj=='bottom_drawer' else 'actual model body origin; handle bar origin is its geometric midpoint'
                                   if obj=='bottom_handle_bar' else 'actual object body centroid'),episodes=rows)
    output['inputs']={str(p):sha(p) for p in [run/'record.json',run/'episode.npz',run/'scene.xml',Path(__file__).resolve()]}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--task',choices=('cup','block','draw','v2_draw_new'))
    p.add_argument('--episode-index',type=int,default=0)
    p.add_argument('--run',type=Path)
    p.add_argument('--render-source',type=Path)
    p.add_argument('--renderer',type=Path,default=DEFAULT_RENDERER)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    row=None
    if args.task:
        rows=[json.loads(s) for s in (args.dataset/args.task/'meta/source_cases.jsonl').read_text().splitlines()]
        selected=[r for r in rows if r['episode_index']==args.episode_index]
        if len(selected)!=1:raise ValueError('episode-index must select exactly one source row')
        row=selected[0]
        args.run=Path(row['run'])
        args.render_source=Path(row['render_receipt']).parent.parent/'source'
    if not args.run or not args.render_source:raise ValueError('Specify --task or both --run and --render-source')
    if args.output.exists():raise FileExistsError('Fresh output required: '+str(args.output))
    result=audit(args.run,args.render_source,args.renderer,meta_row=row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),rig_verified=result['rig']['verified'],clocks=result['clocks'],
        grasp_keys=list(result['grasps']),simulation_steps=0)))


if __name__=='__main__':
    main()
