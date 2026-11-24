"""Read-only evidence for a fixed virtual RSSR camera rig (protocol v1).

This proves parent-relative mounting and native task visibility, NOT agreement
with measured Real1/S1 extrinsics, hardware calibration, or physical task success.
Real1 object boxes are deliberately not an acceptance input. They remain separate
cross-domain diagnostics. The six-view K3 >= .8 check is also a separate gate.

``loaded[kind] = (MjModel, MjData, saved_rollout, video_to_rollout_indices)``;
profiles/origins use camera_agent.initial_profile's three named-camera schema.
``eventindices[kind]`` maps approach/transport/release/terminal to video indices.
Each required object must be visible in at least one camera at EACH event index.
Extra diagnostic_indices do not affect this visibility gate. No image is saved,
no reference video is opened, and no mj_step or mutable library is used.
"""
import copy
import hashlib
import itertools
import json
import numbers

import mujoco
import numpy as np


SCHEMA = 'rssr.fixed_virtual_camera_rig.v1'
CAMERAS = {'head': 'rssr_head', 'left_wrist': 'left_wrist_rgbd',
           'right_wrist': 'right_wrist_rgbd'}
PARENTS = {
    'sim1': {'rssr_head': 'world', 'left_wrist_rgbd': 'left_link6',
             'right_wrist_rgbd': 'right_link6'},
    'sim2': {'rssr_head': 'world', 'left_wrist_rgbd': 'arm_left_tool_cam',
             'right_wrist_rgbd': 'arm_right_tool_cam'},
}
PHASES = ('approach', 'transport', 'release', 'terminal')
WIDTH, HEIGHT, MIN_VISIBLE_PIXELS = 480, 360, 10
SO3_TOLERANCE, FK_TOLERANCE = 1e-6, 1e-8
_DYNAMIC_KEYS = {'frames', 'keyframes', 'trajectory', 'per_frame', 'poses',
                 'lookat', 'look_at', 'target', 'crop', 'mirror', 'zoom',
                 'position_sequence', 'rotation_sequence', 'fovy_sequence'}


def _hash_arrays(*values):
    digest = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _json_hash(value):
    try:
        return hashlib.sha256(json.dumps(value, sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return None


def source_event_indices(events, source_time_s):
    """Map fixed source action events to existing video frames, without resampling.

    Input is action_evidence['events'] (or the whole evidence dictionary), with
    left/right approach_s, closed_s and release_s. Transport is the prescribed
    midpoint of closed->release; terminal is the last frame. Nearest-frame ties
    select the earlier frame. Out-of-range events fail instead of being clipped.
    Returns ``{'indices': {phase:[...]}, 'mapping': [...], ...}`` for both sims.
    """
    if isinstance(events, dict) and 'events' in events:
        events = events['events']
    t = np.asarray(source_time_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Source video times must be finite and strictly increasing')
    if not isinstance(events, dict) or not {'left', 'right'}.issubset(events):
        raise ValueError('Both source hand event records are required')
    indices = {p: [] for p in PHASES}
    mapping = []
    for hand in ('left', 'right'):
        e = events[hand]
        times = [float(e[name]) for name in ('approach_s', 'closed_s', 'release_s')]
        if not np.isfinite(times).all() or not t[0] <= times[0] <= times[1] <= times[2] <= t[-1]:
            raise ValueError('Source hand events must be ordered within the video interval')
        for phase, value in [('approach', times[0]),
                             ('transport', .5 * (times[1] + times[2])),
                             ('release', times[2])]:
            frame = int(np.argmin(abs(t - value)))
            indices[phase].append(frame)
            mapping.append({'hand': hand, 'phase': phase, 'source_event_s': value,
                            'frame_index': frame, 'mapped_source_s': float(t[frame])})
    indices['terminal'] = [len(t) - 1]
    for phase in PHASES:
        indices[phase] = sorted(set(indices[phase]))
    mapping.append({'phase': 'terminal', 'frame_index': len(t) - 1,
                    'mapped_source_s': float(t[-1])})
    return {'indices': indices, 'mapping': mapping,
            'policy': 'source_action_events_nearest_existing_frame_no_interpolation',
            'source_time_sha256': _hash_arrays(t)}


def _events(value, count):
    if not isinstance(value, dict) or set(value) != set(PHASES):
        raise ValueError('Event indices must specify exactly ' + ', '.join(PHASES))
    result = {}
    for phase in PHASES:
        frames = value[phase]
        if not isinstance(frames, (list, tuple)) or not frames:
            raise ValueError('Each event phase needs at least one frame')
        if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or
               not 0 <= int(i) < count for i in frames):
            raise ValueError('Event frame outside the saved video mapping')
        result[phase] = sorted(set(map(int, frames)))
    if result['terminal'] != [count - 1]:
        raise ValueError('Terminal event must include exactly the actual last frame')
    return result


def _spec(value):
    if not isinstance(value, dict) or _DYNAMIC_KEYS.intersection(value):
        raise ValueError('Only one fixed camera transform is allowed; no dynamic/crop controls')
    p = np.asarray(value['position_in_parent_m'], dtype=float)
    r = np.asarray(value['rotation_in_parent'], dtype=float)
    f = value['fovy_degrees']
    if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Camera position/rotation must be finite fixed (3,)/(3,3) arrays')
    if isinstance(f, bool) or not isinstance(f, numbers.Real) or not np.isfinite(f) or not 20 <= f <= 110:
        raise ValueError('Camera FOV must be one finite scalar in [20,110] degrees')
    if not np.allclose(r.T @ r, np.eye(3), atol=SO3_TOLERANCE, rtol=0) or abs(np.linalg.det(r) - 1) > SO3_TOLERANCE:
        raise ValueError('Camera rotation must be proper SO(3), not a reflection')
    return p, r, float(f)


def _profiles(model, kind, entries, origins):
    rows, errors, parsed = [], [], {}
    names = set(CAMERAS.values())
    if not isinstance(entries, dict) or not isinstance(origins, dict) or set(entries) != names or set(origins) != names:
        return rows, ['Exactly the three named fixed camera profiles and origins are required'], parsed
    for role, name in CAMERAS.items():
        row = {'camera': name, 'role': role, 'expected_parent': PARENTS[kind][name]}
        try:
            p, r, f = _spec(entries[name])
            op, _, _ = _spec(origins[name])
            cid = model.camera(name).id
            bid = int(model.cam_bodyid[cid])
            parent = model.body(bid).name if bid else 'world'
            row.update(parent=parent, camera_mode=int(model.cam_mode[cid]))
            if parent != PARENTS[kind][name] or entries[name].get('parent_body') != parent or origins[name].get('parent_body') != parent:
                raise ValueError('Camera parent does not match the declared embodiment/side')
            if int(model.cam_mode[cid]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
                raise ValueError('Camera mode must be fixed, not track/target/look-at')
            limit = .90 if role == 'head' else .20
            distance = float(np.linalg.norm(p - op))
            row.update(displacement_from_origin_m=distance, installation_limit_m=limit,
                       fovy_degrees=f, rotation_determinant=float(np.linalg.det(r)))
            if distance > limit + 1e-12:
                raise ValueError('Fixed mount lies outside the unchanged installation bound')
            parsed[name] = (cid, bid, p, r, f)
            row['passed'] = True
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            row.update(passed=False, error=str(error))
            errors.append(name + ': ' + str(error))
        rows.append(row)
    return rows, errors, parsed


def _descendants(model, root):
    selected = {int(root)}
    for i in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[i]) in selected:
            selected.add(i)
    return selected


def _geom_groups(model, selection):
    objects = {}
    for spec in selection['scene']['objects']:
        name = spec['id']
        if name in objects:
            raise ValueError('Duplicate scene object id')
        bids = _descendants(model, model.body(name).id)
        objects[name] = np.flatnonzero(np.isin(model.geom_bodyid, list(bids)))
        if not len(objects[name]):
            raise ValueError('Scene object has no native MuJoCo geometry: ' + name)
    object_ids = set(int(g) for ids in objects.values() for g in ids)
    table, floor = set(), set()
    for i in range(model.ngeom):
        names = [model.geom(i).name or '']
        bid = int(model.geom_bodyid[i])
        while bid:
            names.append(model.body(bid).name or '')
            bid = int(model.body_parentid[bid])
        if any(n.lower().startswith('floor') for n in names):
            floor.add(i)
        elif any(n.lower().startswith(('table', 'worktable')) for n in names):
            table.add(i)
    roots = set()
    for name in ('left_wrist_rgbd', 'right_wrist_rgbd'):
        bid = int(model.cam_bodyid[model.camera(name).id])
        while int(model.body_parentid[bid]):
            bid = int(model.body_parentid[bid])
        if bid:
            roots.add(bid)
    robot_bodies = set().union(*(_descendants(model, root) for root in roots))
    robot = set(np.flatnonzero(np.isin(model.geom_bodyid, list(robot_bodies)))) - table - floor - object_ids
    table -= object_ids
    floor -= object_ids
    return objects, {'robot': sorted(map(int, robot)), 'table': sorted(table), 'floor': sorted(floor),
                     'objects': sorted(object_ids)}, sorted(model.body(r).name for r in roots)


def _coverage(seg, groups):
    geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    masks = {'background': seg[:, :, 0] == -1}
    for name, ids in groups.items():
        masks[name] = geom & np.isin(seg[:, :, 0], ids)
    masks['other'] = ~np.logical_or.reduce(list(masks.values()))
    total = int(seg.shape[0] * seg.shape[1])
    return {name: {'pixels': int(mask.sum()), 'fraction': float(mask.sum() / total)}
            for name, mask in masks.items()}


def _extent(model, data, cid, spec):
    """Coarse projected solid bounding extent; NOT a true occlusion ratio."""
    size = np.asarray(spec['size_m'], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Invalid object dimensions for clipping diagnostic')
    bid = model.body(spec['id']).id
    corners = np.array(list(itertools.product([-1., 1.], repeat=3))) * size / 2
    world = corners @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
    local = (world - data.cam_xpos[cid]) @ data.cam_xmat[cid].reshape(3, 3)
    depth = -local[:, 2]
    result = {'all_corners_behind_camera': bool(np.all(depth <= 0)),
              'near_plane_crossing': bool(np.any(depth <= 0) and np.any(depth > 0))}
    if np.any(depth <= 0):
        return dict(result, projected_bbox=None, expected_clipped_edges=None)
    focal = HEIGHT / (2 * np.tan(np.radians(model.cam_fovy[cid]) / 2))
    uv = np.column_stack([WIDTH / 2 + focal * local[:, 0] / depth,
                          HEIGHT / 2 - focal * local[:, 1] / depth])
    b = [float(uv[:, 0].min()), float(uv[:, 1].min()),
         float(uv[:, 0].max()), float(uv[:, 1].max())]
    return dict(result, projected_bbox=b,
                expected_clipped_edges=[e for e, flag in zip(('left', 'top', 'right', 'bottom'),
                                        (b[0] < 0, b[1] < 0, b[2] > WIDTH, b[3] > HEIGHT)) if flag],
                entirely_outside_image=bool(b[2] <= 0 or b[0] >= WIDTH or b[3] <= 0 or b[1] >= HEIGHT))


def validate_rig(loaded, profiles, origins, selections, eventindices, *, diagnostic_indices=None):
    """Return independent rig_integrity and task_visibility evidence, without API calls.

    ``passed`` means ONLY these two numerical gates. The caller must additionally
    require physical success and the separate six-view K3 confidence >= .8 gate.
    Before viewing camera outcomes, construct eventindices with source_event_indices
    and persist its mapping/source-time hash. Pin implementation/profile/origin and
    event hashes before final verification; this function cannot infer whether an
    arbitrary caller-supplied index really is a source action event.
    """
    report = {'schema': SCHEMA, 'passed': False, 'hardware_calibrated': False,
              'real_camera_extrinsics_verified': False,
              'scope': 'fixed_virtual_mount_and_native_event_visibility_only',
              'physics_task_success_verified': False,
              'requires_separate_six_view_K3_gate': True,
              'physics_and_actions_modified': False,
              'reference_images_opened': False, 'uses_cross_domain_object_boxes_as_gate': False,
              'thresholds': {'minimum_visible_pixels': MIN_VISIBLE_PIXELS,
                             'width': WIDTH, 'height': HEIGHT, 'fovy_degrees': [20, 110],
                             'head_installation_limit_m': .90, 'wrist_installation_limit_m': .20,
                             'SO3_atol': SO3_TOLERANCE, 'FK_atol': FK_TOLERANCE},
              'profile_sha256': _json_hash(profiles), 'origin_sha256': _json_hash(origins),
              'event_indices_sha256': _json_hash(eventindices),
              'event_indices': copy.deepcopy(eventindices),
              'event_provenance_requirement': 'caller pins source_event_indices mapping before camera inspection',
              'rig_integrity': {'passed': False, 'errors': []},
              'task_visibility': {'passed': False, 'errors': [], 'events': []},
              'kinds': {}}
    if not all(isinstance(v, dict) and set(v) == {'sim1', 'sim2'}
               for v in (loaded, profiles, origins, selections, eventindices)):
        report['rig_integrity']['errors'].append('All five inputs must contain exactly sim1 and sim2')
        return report
    diagnostic_indices = diagnostic_indices or {}
    for kind in ('sim1', 'sim2'):
        original, _, z, ix = loaded[kind]
        rows, errors, parsed = _profiles(original, kind, profiles[kind], origins[kind])
        detail = {'cameras': rows, 'frames': [], 'errors': [], 'FK_frames_checked': 0,
                  'FK_max_position_error_m': 0., 'FK_max_rotation_matrix_error': 0.}
        report['kinds'][kind] = detail
        report['rig_integrity']['errors'].extend(kind + ': ' + e for e in errors)
        if errors:
            continue
        try:
            ix = np.asarray(ix)
            qpos, qvel, times = (np.asarray(z[key]) for key in ('qpos', 'qvel', 'time_s'))
            if (ix.ndim != 1 or len(ix) < 2 or ix.dtype.kind not in 'iu' or
                np.any(np.diff(ix.astype(np.int64)) < 0) or np.any(ix < 0) or np.any(ix >= len(qpos)) or
                qpos.shape != (len(times), original.nq) or qvel.shape != (len(times), original.nv) or
                times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
                not np.isfinite(qpos[ix]).all() or not np.isfinite(qvel[ix]).all()):
                raise ValueError('Invalid/nonfinite saved state or frame-to-state mapping')
            detail['saved_states_sha256'] = _hash_arrays(qpos, qvel, times, ix)
        except (TypeError, ValueError, KeyError, IndexError) as error:
            report['rig_integrity']['errors'].append(kind + ': ' + str(error))
            continue
        try:
            events = _events(eventindices[kind], len(ix))
            diagnostics = diagnostic_indices.get(kind, [])
            if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or not 0 <= i < len(ix)
                   for i in diagnostics):
                raise ValueError('Diagnostic frame outside the saved video mapping')
            event_frames = set(i for frames in events.values() for i in frames)
            render_frames = event_frames | set(map(int, diagnostics))
            scene = selections[kind]['scene']
            required = [scene['container_id'], scene['movable_id']]
            if len(set(required)) != 2:
                raise ValueError('Distinct container_id and movable_id are required')
            specs = {o['id']: o for o in scene['objects']}
            if not set(required) <= set(specs):
                raise ValueError('Required task objects missing from scene')
        except (TypeError, ValueError, KeyError) as error:
            report['task_visibility']['errors'].append(kind + ': ' + str(error))
            events, render_frames, required = {}, set(), []
        # Both model AND data are private. No caller state is temporarily changed.
        m = copy.copy(original)
        d = mujoco.MjData(m)
        for name, (cid, bid, p, r, f) in parsed.items():
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(r).ravel())
            m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = p, quat, f
        renderer = None
        facts = {}
        # Audit every saved VIDEO state independently of object detections/rendering.
        # A missing/out-of-view object must not become a failed mounting invariant.
        try:
            for frame, index in enumerate(ix):
                d.qpos[:] = qpos[index]
                d.qvel[:] = qvel[index]
                d.time = float(times[index])
                mujoco.mj_forward(m, d)
                for name, (cid, bid, p, r, _) in parsed.items():
                    parent_r = d.xmat[bid].reshape(3, 3)
                    ep = parent_r @ p + d.xpos[bid]
                    er = parent_r @ r
                    poserr = float(np.max(abs(ep - d.cam_xpos[cid])))
                    roterr = float(np.max(abs(er - d.cam_xmat[cid].reshape(3, 3))))
                    detail['FK_max_position_error_m'] = max(detail['FK_max_position_error_m'], poserr)
                    detail['FK_max_rotation_matrix_error'] = max(detail['FK_max_rotation_matrix_error'], roterr)
                    if not np.isfinite(poserr + roterr) or max(poserr, roterr) > FK_TOLERANCE:
                        raise ValueError('Parent FK and camera world transform disagree: ' + name)
                detail['FK_frames_checked'] += 1
        except (TypeError, ValueError, RuntimeError, IndexError) as error:
            report['rig_integrity']['errors'].append(kind + ': incomplete FK audit: ' + str(error))
        try:
            if render_frames:
                objects, groups, robot_roots = _geom_groups(m, selections[kind])
                detail['geometry_classes'] = dict(groups, object_geom_ids={n: ids.tolist() for n, ids in objects.items()},
                                                  robot_root_bodies=robot_roots)
                m.vis.quality.offsamples = 0
                renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
                renderer.enable_segmentation_rendering()
                from camera_render_fit import _segmentation
            for frame in sorted(render_frames):
                index = ix[frame]
                d.qpos[:] = qpos[index]
                d.qvel[:] = qvel[index]
                d.time = float(times[index])
                mujoco.mj_forward(m, d)
                views = {}
                for role, name in CAMERAS.items():
                    renderer.update_scene(d, camera=name)
                    if renderer.scene.flags.strides != (0,):
                        for flag in ('mjRND_SKYBOX', 'mjRND_FOG', 'mjRND_SHADOW', 'mjRND_REFLECTION'):
                            renderer.scene.flags[getattr(mujoco.mjtRndFlag, flag)] = False
                    seg, readout = _segmentation(renderer)
                    view = {'camera': name, 'coverage': _coverage(seg, groups),
                            'readout': readout, 'objects': {}}
                    geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                    for obj in required:
                        mask = geom & np.isin(seg[:, :, 0], objects[obj])
                        visible = int(mask.sum())
                        ys, xs = np.nonzero(mask)
                        b = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if visible else None
                        extent = _extent(m, d, m.camera(name).id, specs[obj])
                        view['objects'][obj] = {
                            'visible_pixels': visible, 'visible': visible >= MIN_VISIBLE_PIXELS,
                            'visible_bbox': b,
                            'visible_boundary_edges': [e for e, flag in zip(('left', 'top', 'right', 'bottom'),
                                                       (b[0] == 0, b[1] == 0, b[2] == WIDTH, b[3] == HEIGHT)) if flag] if b else [],
                            'projection_diagnostic': extent,
                            'occlusion_note': 'visible silhouette is native; projected extent is a coarse box, not an occlusion ratio'}
                    views[role] = view
                facts[frame] = views
                detail['frames'].append({'frame_index': frame, 'rollout_index': int(index),
                                         'rollout_time_s': float(times[index]),
                                         'event_phases': [p for p, frames in events.items() if frame in frames],
                                         'diagnostic_only': frame not in event_frames, 'views': views})
        except (TypeError, ValueError, KeyError, RuntimeError, IndexError) as error:
            detail['errors'].append(str(error))
            report['task_visibility']['errors'].append(kind + ': native visibility unavailable: ' + str(error))
        finally:
            if renderer is not None and hasattr(renderer, 'close'):
                renderer.close()
            del renderer
        for phase, frames in events.items():
            for frame in frames:
                for obj in required:
                    supported = [role for role, v in facts.get(frame, {}).items()
                                 if v['objects'][obj]['visible']]
                    row = {'kind': kind, 'phase': phase, 'frame_index': frame,
                           'object': obj, 'visible_in': supported, 'passed': bool(supported)}
                    report['task_visibility']['events'].append(row)
        detail['caller_inputs_unchanged'] = detail['saved_states_sha256'] == _hash_arrays(qpos, qvel, times, ix)
        if not detail['caller_inputs_unchanged']:
            report['rig_integrity']['errors'].append(kind + ': caller rollout mutated during check')
    report['rig_integrity']['passed'] = not report['rig_integrity']['errors']
    visibility = report['task_visibility']
    visibility['passed'] = bool(not visibility['errors'] and visibility['events'] and
                                all(row['passed'] for row in visibility['events']) and
                                {row['kind'] for row in visibility['events']} == {'sim1', 'sim2'})
    report['passed'] = bool(report['rig_integrity']['passed'] and visibility['passed'])
    return report
