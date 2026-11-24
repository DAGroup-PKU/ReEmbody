"""Camera tools for the empty-library RSSR pilot; no transport or historical fits.

The camera is a fixed, parent-relative virtual rig. External cameras share the
table reference frame; each wrist camera follows its own embodiment's wrist.
Camera proposals never change a robot/object state, action, time map or success
predicate. All fitting decisions and optional generated operators come from the
caller's current-run Agent/library; this module supplies bounded geometry tools.

``run_camera_pilot`` is deliberately callback based. ``evaluate(profiles)`` must
render the CURRENT episode and return numerical + visual evidence. ``decide``
uses the shared API ledger. This module cannot issue network requests itself.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CAMERAS = {'head': 'rssr_head', 'left_wrist': 'left_wrist_rgbd',
           'right_wrist': 'right_wrist_rgbd'}
STAGES = ('sim1', 'sim2')
DEFAULT_PARENTS = {
    'sim1': {'rssr_head': 'world', 'left_wrist_rgbd': 'left_link6', 'right_wrist_rgbd': 'right_link6'},
    'sim2': {'rssr_head': 'world', 'left_wrist_rgbd': 'arm_left_tool_cam', 'right_wrist_rgbd': 'arm_right_tool_cam'}}
FIELDS = {'parent_body', 'position_in_parent_m', 'rotation_in_parent',
          'fovy_degrees', 'basis', 'hardware_calibrated', 'provenance'}
EDIT_FIELDS = {'translation_camera_m', 'rotation_camera_xyz_degrees',
               'fovy_delta_degrees'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _finite(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + ' has invalid shape or non-finite values')
    return a


def _rotation(value):
    r = _finite(value, (3, 3), 'rotation')
    if not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-6) or abs(np.linalg.det(r)-1) > 1e-6:
        raise ValueError('Camera rotation must be proper SO(3), never a reflection')
    return r


def _transform(value):
    t = _finite(value, (4, 4), 'table transform')
    _rotation(t[:3, :3])
    if not np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError('Table transform must be rigid SE(3)')
    return t


def _spec(spec):
    if not isinstance(spec, dict) or set(spec)-FIELDS:
        raise ValueError('Unknown or dynamic camera fields; only one fixed mounting is allowed')
    if not isinstance(spec.get('parent_body'), str) or not spec['parent_body']:
        raise ValueError('Camera parent body is required')
    p = _finite(spec['position_in_parent_m'], (3,), 'position')
    r = _rotation(spec['rotation_in_parent'])
    f = spec['fovy_degrees']
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or not 20 <= f <= 110:
        raise ValueError('Vertical FOV must be a finite scalar in [20,110] degrees')
    return p, r, float(f)


def _euler_xyz(degrees):
    x, y, z = np.deg2rad(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quaternion_matrix(q):
    w, x, y, z = _finite(q, (4,), 'quaternion')
    n = w*w+x*x+y*y+z*z
    if n < 1e-16:
        raise ValueError('Zero camera quaternion')
    return np.array([[1-2*(y*y+z*z)/n, 2*(x*y-z*w)/n, 2*(x*z+y*w)/n],
                     [2*(x*y+z*w)/n, 1-2*(x*x+z*z)/n, 2*(y*z-x*w)/n],
                     [2*(x*z-y*w)/n, 2*(y*z+x*w)/n, 1-2*(x*x+y*y)/n]])


def model_profiles(models, provenance, expected_parents=None):
    """Read raw scene cameras, without configure_cameras or fitted-profile import.

    The caller supplies asset lineage per stage: ``kind`` is ``model_asset`` or
    ``measured_calibration``, ``artifact_sha256`` identifies the fresh scene or
    underlying model, and ``contains_learned_fit`` MUST be explicitly false.
    This records declared provenance; it is not independent proof of an asset's
    history. The campaign must hash/check the model manifest as well.
    """
    out = {}
    bindings = DEFAULT_PARENTS if expected_parents is None else expected_parents
    for kind in models:
        if kind not in STAGES:
            raise ValueError('Unknown simulation stage')
        evidence = provenance[kind]
        if evidence.get('kind') not in ('model_asset', 'measured_calibration') or evidence.get('contains_learned_fit') is not False:
            raise ValueError('Fresh raw model/mount provenance is required; historical fits are forbidden')
        artifact = evidence.get('artifact_sha256', '')
        if not isinstance(artifact, str) or len(artifact) != 64 or any(c not in '0123456789abcdef' for c in artifact):
            raise ValueError('Model provenance requires a SHA256 digest')
        m = models[kind]
        out[kind] = {}
        for role, name in CAMERAS.items():
            cid = m.camera(name).id
            bid = int(m.cam_bodyid[cid])
            if int(m.cam_mode[cid]) != 0:
                raise ValueError('Tracking/target camera modes are forbidden')
            parent = m.body(bid).name if bid else 'world'
            if (role == 'head') != (parent == 'world'):
                raise ValueError('External view must be world-fixed; wrist view must be body-mounted')
            if parent != bindings[kind][name]:
                raise ValueError(kind + '/' + role + ': camera does not match the declared wrist/side binding')
            out[kind][name] = {
                'parent_body': parent, 'position_in_parent_m': m.cam_pos[cid].tolist(),
                'rotation_in_parent': _quaternion_matrix(m.cam_quat[cid]).tolist(),
                'fovy_degrees': float(m.cam_fovy[cid]),
                'basis': evidence['kind'], 'hardware_calibrated': evidence['kind'] == 'measured_calibration',
                'provenance': deepcopy(evidence)}
    return out


def _release_renderer(renderer):
    """Release MuJoCo 2.3.7 and current Renderer resources explicitly.

    2.3.7 has no Renderer.close/__del__, but owns MjrContext.free and
    GLContext.free. Release the framebuffer context before its GL context.
    New versions expose close(), which remains the preferred public API.
    """
    close = getattr(renderer, 'close', None)
    if callable(close):
        close()
        return
    context = getattr(renderer, '_gl_context', None)
    render_context = getattr(renderer, '_mjr_context', None)
    try:
        make_current = getattr(context, 'make_current', None)
        if callable(make_current):
            make_current()
        free = getattr(render_context, 'free', None)
        if callable(free):
            free()
    finally:
        # Discard the already-freed MjrContext and release GL even if explicit
        # framebuffer cleanup fails. Both attributes exist in the audited 2.3.7.
        if hasattr(renderer, '_mjr_context'):
            renderer._mjr_context = None
        free = getattr(context, 'free', None)
        if callable(free):
            free()
        if hasattr(renderer, '_gl_context'):
            renderer._gl_context = None


def _id_renderer(model, height, width):
    """ID colours must not blend into other valid geometry IDs at edges."""
    import mujoco
    samples = int(model.vis.quality.offsamples)
    try:
        model.vis.quality.offsamples = 0
        return mujoco.Renderer(model, height=height, width=width)
    finally:
        model.vis.quality.offsamples = samples


def camera_scene_option():
    """Hide nonphysical diagnostic sites, preserving every visible mesh/geom.

    Scene previews and final video already suppress sites. Use the same view
    for RGB checks and depth-aware IDs without mutating model site colours.
    """
    import mujoco
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    return option


def render_profile_frames(loaded, profiles, source_frame_indices, output_dir,
                          *, width=480, height=360, object_ids=None):
    """Render current saved states with a proposed rig; restore everything.

    ``loaded[kind] = (model, data, rollout, video_to_rollout_indices)`` is the
    existing camera renderer convention. Rollouts may contain only the initial
    state for a source-only reconstruction check. No dynamics are stepped and
    no object/robot pose is invented. Returns per-stage/frame/view image paths
    and camera world transforms for integration with the shared visual Agent.
    """
    import cv2
    import mujoco
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    scene_option = camera_scene_option()
    for kind, entries in profiles.items():
        model, data, rollout, mapping = loaded[kind]
        saved = (model.cam_pos.copy(), model.cam_quat.copy(), model.cam_fovy.copy(),
                 data.qpos.copy(), data.qvel.copy(), float(data.time))
        renderer = None
        id_renderer = None
        result[kind] = {}
        try:
            for name, spec in entries.items():
                p, r, f = _spec(spec)
                cid = model.camera(name).id
                bid = int(model.cam_bodyid[cid])
                parent = model.body(bid).name if bid else 'world'
                if parent != spec['parent_body'] or int(model.cam_mode[cid]) != 0:
                    raise ValueError('Scene camera parent or fixed mode changed')
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, r.reshape(9))
                model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid] = p, quat, f
            renderer = mujoco.Renderer(model, height=height, width=width)
            if object_ids is not None:
                id_renderer = _id_renderer(model, height, width)
                id_renderer.enable_segmentation_rendering()
            for frame in source_frame_indices:
                if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(mapping):
                    raise ValueError('Requested source frame is not in the saved frame mapping')
                index = int(mapping[frame])
                if not 0 <= index < len(rollout['qpos']):
                    raise ValueError('Mapped simulation sample does not exist')
                data.qpos[:] = rollout['qpos'][index]
                data.qvel[:] = rollout['qvel'][index]
                if 'time_s' in rollout:
                    data.time = float(rollout['time_s'][index])
                mujoco.mj_forward(model, data)
                result[kind][int(frame)] = {}
                for role, name in CAMERAS.items():
                    renderer.update_scene(data, camera=name, scene_option=scene_option)
                    rgb = renderer.render().copy()
                    visibility=None
                    if object_ids is not None:
                        from camera_visibility import raster_metrics
                        id_renderer.update_scene(data,camera=name,scene_option=scene_option)
                        # A separate unblended ID framebuffer preserves the
                        # original RGB renderer's antialiasing and appearance.
                        id_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                        id_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                        visibility=raster_metrics(model,id_renderer.render(),object_ids)
                        from camera_clearance import inspect as camera_clearance
                        visibility['camera_clearance']=camera_clearance(model,data,name)
                        from camera_interaction import capture as contact_context, inspect as interaction_visibility
                        visibility['interaction_visibility']=interaction_visibility(model,data,name,
                            contact_context(model,data,rollout,index,object_ids,role),visibility)
                        visibility['id_framebuffer_multisampling']=0
                    path = root/f'{kind}_{frame:06d}_{role}.png'
                    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError('Camera render image could not be written')
                    cid = model.camera(name).id
                    result[kind][int(frame)][role] = {
                        'image': str(path), 'simulation_sample_index': index,
                        'diagnostic_sites_visible': False,
                        'camera_position_world_m': data.cam_xpos[cid].tolist(),
                        'camera_rotation_world': data.cam_xmat[cid].reshape(3, 3).tolist()}
                    if visibility is not None:
                        result[kind][int(frame)][role]['raster_visibility']=visibility
        finally:
            try:
                try:
                    if id_renderer is not None:
                        _release_renderer(id_renderer)
                finally:
                    if renderer is not None:
                        _release_renderer(renderer)
            finally:
                model.cam_pos[:], model.cam_quat[:], model.cam_fovy[:] = saved[:3]
                data.qpos[:], data.qvel[:], data.time = saved[3:]
                mujoco.mj_forward(model, data)
    _save(root/'render_manifest.json', result)
    return result


def initialize_profiles(raw_profiles, table_frames=None):
    """Use current raw mounts and one shared external camera in table coordinates.

    ``table_frames[kind]`` is world-from-table. Wrist transforms are never copied
    from Cobot to S1. Parent identities in raw_profiles become immutable.
    Returns profiles plus a provenance/initialization receipt.
    """
    stages = tuple(raw_profiles)
    if not stages or set(stages)-set(STAGES):
        raise ValueError('At least one of sim1/sim2 is required')
    tables = {k: _transform((table_frames or {}).get(k, np.eye(4))) for k in stages}
    profiles = deepcopy(raw_profiles)
    for kind in stages:
        if set(profiles[kind]) != set(CAMERAS.values()):
            raise ValueError('Exactly head, left wrist and right wrist cameras are required')
        for role, name in CAMERAS.items():
            spec = profiles[kind][name]
            _spec(spec)
            if (role == 'head') != (spec['parent_body'] == 'world'):
                raise ValueError('Incorrect camera role/parent binding')
        if profiles[kind][CAMERAS['left_wrist']]['parent_body'] == profiles[kind][CAMERAS['right_wrist']]['parent_body']:
            raise ValueError('Left and right wrist views must follow distinct wrists')
    source = 'sim1' if 'sim1' in profiles else 'sim2'
    p, r, f = _spec(profiles[source][CAMERAS['head']])
    tc = np.eye(4)
    tc[:3, :3], tc[:3, 3] = r, p
    table_camera = np.linalg.inv(tables[source]) @ tc
    for target_kind in stages:
        if target_kind == source:
            continue
        target = tables[target_kind] @ table_camera
        head = profiles[target_kind][CAMERAS['head']]
        head.update(position_in_parent_m=target[:3, 3].tolist(),
                    rotation_in_parent=target[:3, :3].tolist(), fovy_degrees=f,
                    basis='shared_external_camera_in_table_frame', hardware_calibrated=False)
    receipt = {'schema': 'rssr.pilot_camera_initialization.v1',
               'input_profile_sha256': digest(raw_profiles), 'initialized_profile_sha256': digest(profiles),
               'world_from_table': {k: t.tolist() for k, t in tables.items()},
               'historical_fitted_profile_loaded': False,
               'wrist_policy': 'each_camera_fixed_to_own_embodiment_wrist',
               'head_policy': 'one_common_camera_in_table_coordinates',
               'hardware_calibrated': False}
    return profiles, receipt


def apply_camera_edit(profiles, origin, kind, view, edit, table_frames=None):
    """Apply an Agent-selected bounded primitive; not a learned operator itself.

    The parent campaign may wrap this primitive in generated named operators.
    Translation/rotation are expressed in the current camera local frame. Head
    edits affect both stages through the common table frame. No hidden clipping.
    """
    if kind not in profiles or kind not in STAGES or view not in CAMERAS or not isinstance(edit, dict) or not edit or set(edit)-EDIT_FIELDS:
        raise ValueError('Invalid camera stage/view/edit')
    out = deepcopy(profiles)
    name = CAMERAS[view]
    spec = out[kind][name]
    p, r, f = _spec(spec)
    dp = _finite(edit.get('translation_camera_m', [0, 0, 0]), (3,), 'camera translation')
    dr = _finite(edit.get('rotation_camera_xyz_degrees', [0, 0, 0]), (3,), 'camera rotation')
    df = edit.get('fovy_delta_degrees', 0)
    if isinstance(df, bool) or not isinstance(df, (int, float)) or not math.isfinite(df):
        raise ValueError('Invalid FOV delta')
    if np.max(np.abs(dp)) > (.15 if view == 'head' else .04) or np.max(np.abs(dr)) > 180 or abs(df) > 20:
        raise ValueError('Camera edit exceeds one-step search bounds')
    spec.update(position_in_parent_m=(p+r@dp).tolist(),
                rotation_in_parent=(r@_euler_xyz(dr)).tolist(), fovy_degrees=f+df,
                basis='current_run_agent_proposal', hardware_calibrated=False)
    _spec(spec)
    if view == 'head':
        tables = {k: _transform((table_frames or {}).get(k, np.eye(4))) for k in profiles}
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = _spec(spec)[1], _spec(spec)[0]
        common = np.linalg.inv(tables[kind]) @ transform
        for other in profiles:
            target = tables[other] @ common
            out[other][name].update(position_in_parent_m=target[:3, 3].tolist(),
                                    rotation_in_parent=target[:3, :3].tolist(),
                                    fovy_degrees=spec['fovy_degrees'], basis='current_run_agent_proposal',
                                    hardware_calibrated=False)
    report = validate_profiles(out, origin, table_frames)
    if not report['passed']:
        raise ValueError('; '.join(report['errors']))
    # Metadata is not a real camera edit.
    changed = any(not np.allclose(_spec(profiles[s][n])[0], _spec(out[s][n])[0]) or
                  not np.allclose(_spec(profiles[s][n])[1], _spec(out[s][n])[1]) or
                  _spec(profiles[s][n])[2] != _spec(out[s][n])[2]
                  for s in profiles for n in CAMERAS.values())
    if not changed:
        raise ValueError('Camera edit has no geometric effect')
    return out


def validate_profiles(profiles, origin, table_frames=None):
    errors = []
    stages = tuple(profiles)
    tables = {k: _transform((table_frames or {}).get(k, np.eye(4))) for k in stages}
    external = {}
    try:
        if not stages or set(stages)-set(STAGES) or set(origin) != set(stages):
            raise ValueError('Current stages must match the immutable origin')
        for kind in stages:
            if set(profiles[kind]) != set(CAMERAS.values()):
                raise ValueError('All three camera roles are required')
            for role, name in CAMERAS.items():
                p, r, f = _spec(profiles[kind][name])
                op, _, _ = _spec(origin[kind][name])
                parent = profiles[kind][name]['parent_body']
                if parent != origin[kind][name]['parent_body'] or (role == 'head') != (parent == 'world'):
                    raise ValueError(kind + '/' + role + ': changed or invalid parent')
                if np.linalg.norm(p-op) > (.9 if role == 'head' else .20) + 1e-10:
                    raise ValueError(kind + '/' + role + ': cumulative mounting bound exceeded')
                if role == 'head':
                    t = np.eye(4)
                    t[:3, :3], t[:3, 3] = r, p
                    external[kind] = (np.linalg.inv(tables[kind]) @ t, f)
        if len(stages) == 2 and (not np.allclose(external['sim1'][0], external['sim2'][0], atol=1e-6, rtol=0) or abs(external['sim1'][1]-external['sim2'][1]) > 1e-8):
            raise ValueError('External cameras disagree in the shared table frame')
    except (ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return {'passed': not errors, 'errors': errors,
            'wrist_world_pose_equality_required': False, 'robot_pixel_shape_equality_required': False,
            'hardware_calibrated': False}


def diagnose_camera_profiles(profiles, origin, observations, frame_mapping, table_frames=None):
    """Deterministic rig, projection and timing diagnostics with explicit unknowns.

    Observations contain kind, view, source_frame_index, point_in_parent_m,
    image_size_wh, optional pixel_xy and static_landmark. For Sim2 only first
    frame/static landmarks may constrain pixel agreement; moving objects at
    later frames are visibility diagnostics because Sim2 may repair the action.
    Occlusion needs rendering and is intentionally not certified here.

    Mapping contains equal-length source_frame_index/source_time_s and each
    stage's simulation_time_s; stage time_offset_s is an explicit warmup offset.
    """
    rig = validate_profiles(profiles, origin, table_frames)
    errors, mappings, details, by_view = [], {}, [], {}
    for kind in profiles:
        try:
            mapping = frame_mapping[kind]
            frames = np.asarray(mapping['source_frame_index'])
            source = np.asarray(mapping['source_time_s'], dtype=float)
            actual = np.asarray(mapping['simulation_time_s'], dtype=float)
            if frames.ndim != 1 or len(frames) < 1 or frames.dtype.kind not in 'iu' or not (len(frames) == len(source) == len(actual)):
                raise ValueError('Frame mapping arrays must have equal positive length')
            if source.ndim != 1 or actual.ndim != 1 or not np.isfinite(source).all() or not np.isfinite(actual).all() or np.any(np.diff(frames) <= 0) or np.any(np.diff(source) <= 0) or np.any(np.diff(actual) <= 0):
                raise ValueError('Frame mapping must be finite, increasing, and without duplicates')
            offset = float(mapping.get('time_offset_s', 0))
            tolerance = float(mapping.get('sampling_tolerance_s', .02))
            if not math.isfinite(offset) or not 0 <= tolerance <= .1:
                raise ValueError('Invalid declared time offset/tolerance')
            residual = float(np.max(np.abs(actual-offset-source)))
            if residual > tolerance:
                raise ValueError('Simulation/reference timing mismatch')
            mappings[kind] = {'first': int(frames[0]), 'frames': set(map(int, frames)),
                              'time_residual_s': residual}
        except (ValueError, KeyError, TypeError) as error:
            errors.append(kind + ': ' + str(error))
    if set(mappings) == set(STAGES):
        for field in ('source_frame_index', 'source_time_s'):
            if not np.array_equal(frame_mapping['sim1'][field], frame_mapping['sim2'][field]):
                errors.append('Sim1/Sim2 reference frame identities or timestamps differ')
    for observation in observations:
        row = {k: observation.get(k) for k in ('kind', 'view', 'source_frame_index', 'object_id')}
        try:
            kind, view, frame = observation['kind'], observation['view'], observation['source_frame_index']
            if kind not in STAGES or view not in CAMERAS or frame not in mappings.get(kind, {}).get('frames', set()):
                raise ValueError('Observation role or source frame is invalid')
            p, r, f = _spec(profiles[kind][CAMERAS[view]])
            point = _finite(observation['point_in_parent_m'], (3,), 'object point')
            width, height = _finite(observation['image_size_wh'], (2,), 'image dimensions')
            if min(width, height) < 1:
                raise ValueError('Invalid image dimensions')
            local = (point-p) @ r
            depth = float(-local[2])
            focal = height/(2*np.tan(np.deg2rad(f)/2))
            uv = np.array([width/2+focal*local[0]/max(depth, 1e-9),
                           height/2-focal*local[1]/max(depth, 1e-9)])
            visible = bool(depth > .005 and 0 <= uv[0] < width and 0 <= uv[1] < height)
            match = bool(kind == 'sim1' or frame == mappings[kind]['first'] or observation.get('static_landmark') is True)
            error = None
            if match and 'pixel_xy' in observation:
                error = float(np.linalg.norm(uv-_finite(observation['pixel_xy'], (2,), 'observed pixel')))
            if 'bbox_xyxy' in observation:
                from camera_landmarks import project_visible_bounds
                expected = _finite(observation['bbox_xyxy'], (4,), 'observed visible bounds')
                if np.any(expected < 0) or np.any(expected > [width, height, width, height]) or expected[0] >= expected[2] or expected[1] >= expected[3]:
                    raise ValueError('Invalid observed bounding rectangle')
                corners = _finite(observation['bounds_in_parent_m'], (8, 3), 'object bounding corners')
                projected_box = project_visible_bounds((corners-p) @ r, width, height, focal)
                visible = projected_box is not None
                row['projected_bbox_xyxy'] = None if projected_box is None else projected_box.tolist()
                row['evidence_type'] = 'visible_extent'
                if match:
                    # Missing silhouette remains a large error; it never becomes
                    # a spurious match by clipping its center into the image.
                    error = float(np.linalg.norm(projected_box-expected)) if visible else float(2*max(width, height))
            row.update(projected_pixel_xy=uv.tolist(), depth_m=depth, in_frustum=visible,
                       compared_pixel_position=match and error is not None,
                       projection_error_px=error, normalized_projection_error=None if error is None else error/max(width, height))
            by_view.setdefault((kind, view), []).append(row)
        except (ValueError, KeyError, TypeError) as error:
            row['error'] = str(error)
            errors.append('observation: ' + str(error))
        details.append(row)
    views = []
    for kind in profiles:
        for view in CAMERAS:
            rows = by_view.get((kind, view), [])
            matched = [r for r in rows if r.get('normalized_projection_error') is not None]
            first = [r for r in rows if r['source_frame_index'] == mappings.get(kind, {}).get('first')]
            views.append({'kind': kind, 'view': view, 'observations': len(rows),
                          'first_frame_observations': len(first),
                          'has_initial_visible_landmark': any(r['in_frustum'] for r in first),
                          'in_frustum_fraction': None if not rows else sum(r['in_frustum'] for r in rows)/len(rows),
                          'projection_correspondences': len(matched),
                          'median_normalized_projection_error': None if not matched else float(np.median([r['normalized_projection_error'] for r in matched]))})
            current=views[-1];failures=[]
            if not current['has_initial_visible_landmark']:failures.append('missing_initial_visible_landmark')
            if not matched:failures.append('missing_projection_correspondence')
            elif current['median_normalized_projection_error']>.15:failures.append('projection_error_above_threshold')
            current.update(projection_passed=not failures,projection_failure_reasons=failures,
                           max_median_normalized_projection_error=.15)
    sufficient = all(v['has_initial_visible_landmark'] and v['projection_correspondences'] > 0 for v in views)
    projected = sufficient and all(v['median_normalized_projection_error'] <= .15 for v in views)
    return {'schema': 'rssr.pilot_camera_diagnostics.v1', 'rig': rig,
            'timing_passed': not errors, 'errors': errors, 'views': views, 'observations': details,
            'projection_evidence_sufficient': sufficient, 'coarse_projection_passed': bool(projected),
            'numerical_passed': bool(rig['passed'] and not errors and projected),
            'temporal_evidence_sufficient': bool(mappings and all(len(m['frames']) > 1 for m in mappings.values())),
            'occlusion_verified': False, 'visual_orientation_verified': False,
            'hardware_calibrated': False, 'task_success_verified': False,
            'requires_rendered_visual_verification': True}


def visual_verdict_passed(verdict, stages=STAGES):
    """All six roles need orientation/FOV/visibility evidence, not robot likeness."""
    if not isinstance(verdict, dict) or verdict.get('images_visible') is not True:
        return False
    views = verdict.get('views')
    if not isinstance(views, list) or len(views) != len(stages)*3 or not all(isinstance(v, dict) for v in views):
        return False
    if {(v.get('kind'), v.get('view')) for v in views} != {(k, r) for k in stages for r in CAMERAS}:
        return False
    required = ('orientation_consistent', 'fov_adequate', 'role_correct', 'key_objects_visible')
    return bool(not verdict.get('unresolved', ['missing']) and all(
        all(v.get(field) is True for field in required) and
        isinstance(v.get('reason'), str) and v['reason'].strip()
        for v in views))


def run_camera_pilot(initial, knowledge_snapshot, decide, evaluate, output_dir,
                     *, run_id, case_id, table_frames=None, max_candidates=5,
                     operator_apply=None, record_evidence=None):
    """Bounded current-run loop, with no old-profile or fixed-repair imports.

    ``decide(payload, folder)`` returns {kind,view,edit,reason}, or a generated
    ``operator_id``/parameters which ``operator_apply(decision,snapshot)`` maps
    to {kind,view,edit}. The caller owns operator creation, validation, storage
    and the API ledger. An empty operator list is valid: the first decision may
    propose a new operator through that callback. Direct edit decisions remain
    explicitly recorded as primitive proposals, not pre-existing operators.

    ``evaluate(profiles, folder)`` returns {numerical_passed, visual_verdict,
    score, ...}; lower score is better. Missing visual evidence never passes.
    It must evaluate frozen physical rollouts and fresh rendered camera views.
    This loop records before/after evidence; it never edits the shared library.
    The caller can promote accepted evidence after checking run identity.
    """
    if knowledge_snapshot.get('run_id', run_id) != run_id:
        raise ValueError('Knowledge snapshot belongs to another run')
    experiences = knowledge_snapshot.get('experiences', knowledge_snapshot.get('experience'))
    if not isinstance(knowledge_snapshot.get('operators'), list) or not isinstance(experiences, list):
        raise ValueError('Explicit operator and experience lists are required, including when empty')
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 1:
        raise ValueError('Positive candidate budget required')
    root = Path(output_dir)
    profiles, initialization = initialize_profiles(initial, table_frames)
    origin = deepcopy(profiles)
    inputs = {'run_id': run_id, 'case_id': case_id, 'initialization': initialization,
              'knowledge_snapshot_sha256': digest(knowledge_snapshot), 'max_candidates': max_candidates}
    if (root/'input.json').exists():
        if json.loads((root/'input.json').read_text(encoding='utf-8')) != inputs:
            raise ValueError('Camera pilot resume input changed')
        if (root/'result.json').exists():
            result = json.loads((root/'result.json').read_text(encoding='utf-8'))
            if digest(json.loads((root/'camera_profile.json').read_text(encoding='utf-8'))) != result['profile_sha256']:
                raise ValueError('Saved camera result profile changed')
            return result
        raise RuntimeError('Incomplete camera pilot requires explicit recovery; refusing duplicate decisions')
    _save(root/'input.json', inputs)
    _save(root/'knowledge_snapshot.json', knowledge_snapshot)
    _save(root/'initial_profiles.json', profiles)
    history, evidence, best_score, best_report = [], [], float('inf'), None
    success = False
    for index in range(max_candidates):
        folder = root/f'candidate_{index:02d}'
        before = deepcopy(profiles)
        decision = None
        if index:
            payload = {'run_id': run_id, 'case_id': case_id, 'profiles': profiles,
                       'diagnostics': best_report, 'history': history,
                       'operators': knowledge_snapshot['operators'], 'experiences': experiences,
                       'candidate_budget_remaining': max_candidates-index,
                       'camera_roles': CAMERAS, 'source_task_success_required': False,
                       'physics_and_actions_mutable': False}
            decision = decide(payload, folder)
            _save(folder/'decision.json', decision)
            if decision.get('stop') is True:
                history.append({'candidate': index, 'stop': True, 'reason': decision.get('reason')})
                break
            try:
                edit = operator_apply(decision, knowledge_snapshot) if operator_apply else decision
                candidate = apply_camera_edit(profiles, origin, edit['kind'], edit['view'], edit['edit'], table_frames)
            except (ValueError, KeyError, TypeError) as error:
                history.append({'candidate': index, 'accepted': False, 'tool_error': str(error), 'decision': decision})
                _save(folder/'rejection.json', history[-1])
                continue
        else:
            candidate = profiles
        _save(folder/'profile_before.json', before)
        _save(folder/'profile_proposed.json', candidate)
        report = evaluate(candidate, folder)
        _save(folder/'diagnostics.json', report)
        score = report.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Camera evaluator must return a finite numeric score')
        rig = validate_profiles(candidate, origin, table_frames)
        passed = bool(rig['passed'] and report.get('numerical_passed') is True and visual_verdict_passed(report.get('visual_verdict'), tuple(profiles)))
        accepted = bool(rig['passed'] and (score < best_score or passed))
        row = {'candidate': index, 'accepted': accepted, 'passed': passed, 'score': score,
               'before_profile_sha256': digest(before), 'after_profile_sha256': digest(candidate),
               'decision': decision, 'diagnostics': str(folder/'diagnostics.json')}
        history.append(row)
        before_report = best_report
        if accepted:
            profiles, best_report, best_score = candidate, report, float(score)
        if decision is not None:
            evidence.append(dict(row, run_id=run_id, case_id=case_id, stage='camera',
                                 active_success=passed, evidence_type='verified_camera_repair' if passed else 'camera_repair_attempt'))
            if record_evidence is not None:
                record_evidence(evidence[-1], before_report, report)
        _save(folder/'acceptance.json', row)
        if passed:
            success = True
            break
    profile = {'schema': 'rssr.camera_profile.v2', 'cameras': profiles, 'origin_cameras': origin,
               'hardware_calibrated': False, 'world_or_parent_fixed_for_entire_episode': True,
               'role': 'current-run visual registration, not hardware calibration'}
    _save(root/'camera_profile.json', profile)
    _save(root/'evidence.json', evidence)
    result = {'schema': 'rssr.pilot_camera_result.v1', 'run_id': run_id, 'case_id': case_id,
              'passed': success, 'history': history, 'best_diagnostics': best_report,
              'profile': str(root/'camera_profile.json'), 'profile_sha256': digest(profile),
              'evidence': str(root/'evidence.json'), 'hardware_calibrated': False,
              'physics_and_actions_modified': False, 'historical_profiles_imported': False,
              'state': 'camera_verified' if success else 'camera_not_verified'}
    _save(root/'result.json', result)
    return result
