"""Select fixed camera proposals using actual MuJoCo object visibility.

Box projection is only a proposal mechanism. Selection and final evaluation
use rendered segmentation, including self-occlusion and image-boundary cuts.
This module reads saved physical states; it never steps or repairs physics.
"""
from copy import deepcopy
import ctypes
import json

import mujoco
import numpy as np

from camera_agent import apply_profile
from camera_box_fit import (_rows, _edge_residuals, _evidence, evaluate_boxes,
                            fit_boxes, MIN_MEDIAN_LIMIT_PX, P90_LIMIT_PX)


def _segmentation(renderer):
    """Use exact ID colors even with old MuJoCo/NumPy zero-stride flag views."""
    flags = renderer.scene.flags
    zero_stride_flags = flags.strides == (0,)
    if zero_stride_flags:
        # MuJoCo 2.3.7 + NumPy 2 exposes mjtByte flags[10] with stride 0.
        # The regular Renderer therefore changes only the first byte when
        # enabling SEGMENT/IDCOLOR. Rebuild the correct contiguous C-array
        # view and perform the very same rendering primitives explicitly.
        c_flags = (ctypes.c_uint8 * flags.size).from_address(flags.ctypes.data)
        physical_flags = np.ndarray((flags.size,), dtype=np.uint8, buffer=c_flags)
        saved_flags = physical_flags.copy()
        try:
            physical_flags[:] = 0
            for flag in ('mjRND_SEGMENT', 'mjRND_IDCOLOR', 'mjRND_CULL_FACE'):
                physical_flags[int(getattr(mujoco.mjtRndFlag, flag))] = 1
            renderer._gl_context.make_current()
            mujoco.mjr_render(renderer._rect, renderer.scene, renderer._mjr_context)
        finally:
            physical_flags[:] = saved_flags
    else:
        try:
            result = renderer.render().copy()
            return result, {'compatibility_readout': False, 'unknown_color_pixels': 0}
        except IndexError:
            pass
    # Read the already-rendered ID buffer and map ONLY exact IDs of scene
    # geoms. Unknown/background colors never become valid geometry IDs.
    rgb = np.empty((renderer.height, renderer.width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, renderer._rect, renderer._mjr_context)
    values = rgb.astype(np.uint32)
    ids = values[:, :, 0] + (values[:, :, 1] << 8) + (values[:, :, 2] << 16)
    result = np.full((renderer.height, renderer.width, 2), -1, np.int32)
    known = ids == 0
    for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
        if geom.segid < 0:
            continue
        mask = ids == geom.segid+1
        result[mask, 0], result[mask, 1] = geom.objid, geom.objtype
        known |= mask
    return np.flipud(result).copy(), {'compatibility_readout': True,
                                      'zero_stride_flags_repaired': zero_stride_flags,
                                      'unknown_color_pixels': int((~known).sum()),
                                      'mapped_foreground_pixels': int(np.sum(result[:, :, 0] >= 0))}


def evaluate_rendered_camera(loaded, name, rows, profile):
    """Evaluate a frozen camera against reference boxes on specified frames.

    Reference clipping uses one-sided edge constraints. Expected objects with
    fewer than 10 actually visible pixels get a 300 px error and fail the gate,
    regardless of their projected, unoccluded 3-D bounding box.
    """
    m, d, states, indices = loaded
    rows, excluded = _rows(rows)
    evidence = _evidence(rows, 3, 2, 6)
    base = {'method': 'actual_mujoco_visible_segmentation_bbox',
            'real_calibration': False, 'uses_proxy_object_geometry': True,
            'count': len(rows), 'excluded': excluded, 'evidence': evidence,
            'thresholds': {'median_px': MIN_MEDIAN_LIMIT_PX, 'p90_px': P90_LIMIT_PX,
                           'minimum_visible_pixels': 10, 'missing_error_px': 300.},
            'projection_diagnostic': evaluate_boxes(rows, profile)}
    if not rows:
        return dict(base, passed=False, reason='insufficient_correspondences',
                    median_px=None, p90_px=None, weighted_mean_px=None,
                    missing_observations=0, visible_observations=0, rows=[])
    sizes = {(row['width'], row['height']) for row in rows}
    if len(sizes) != 1:
        raise ValueError('Mixed image resolutions within one camera evaluation')
    width, height = next(iter(sizes))
    cid = m.camera(name).id
    saved_camera = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy(), float(m.cam_fovy[cid]))
    saved_state = (d.qpos.copy(), d.qvel.copy(), float(d.time))
    saved_samples = int(m.vis.quality.offsamples)
    renderer = None
    details, rendering_diagnostics = [], []
    try:
        apply_profile(m, {name: profile})
        # Multisample antialiasing blends integer ID colors at boundaries on
        # older MuJoCo builds. Segmentation needs exact integer raster labels.
        m.vis.quality.offsamples = 0
        renderer = mujoco.Renderer(m, height=height, width=width)
        renderer.enable_segmentation_rendering()
        ids = {}
        for object_name in {row['object'] for row in rows}:
            try:
                bid = m.body(object_name).id
            except KeyError:
                bid = -1
            ids[object_name] = np.flatnonzero(m.geom_bodyid == bid) if bid >= 0 else np.array([], dtype=int)
        for frame in sorted({row['frame'] for row in rows}):
            if not 0 <= int(frame) < len(indices):
                raise ValueError('Reference frame outside saved physical sequence')
            index = int(indices[int(frame)])
            d.qpos[:] = states['qpos'][index]
            d.qvel[:] = states['qvel'][index]
            if 'time_s' in states:
                d.time = float(states['time_s'][index])
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=name)
            # These decorative passes can emit regular RGB background colors
            # under old MuJoCo ID rendering, which are not valid geometry IDs.
            if renderer.scene.flags.strides != (0,):
                for flag in ('mjRND_SKYBOX', 'mjRND_FOG', 'mjRND_SHADOW', 'mjRND_REFLECTION'):
                    renderer.scene.flags[getattr(mujoco.mjtRndFlag, flag)] = False
            segmentation, rendering_note = _segmentation(renderer)
            rendering_diagnostics.append(dict(rendering_note, frame=int(frame)))
            geometry_pixels = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            for row in (item for item in rows if item['frame'] == frame):
                mask = geometry_pixels & np.isin(segmentation[:, :, 0], ids[row['object']])
                visible_pixels = int(mask.sum())
                predicted = None
                if visible_pixels >= 10:
                    ys, xs = np.nonzero(mask)
                    predicted = np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1], dtype=float)
                    residual = _edge_residuals([row], predicted[None, :])[0]
                    error = float(np.linalg.norm(residual) / np.sqrt(2.))
                    reason = 'visible_bbox_measured'
                else:
                    error = 300.
                    reason = 'expected_object_not_visibly_supported'
                details.append({'frame': int(frame), 'object': row['object'],
                                'visible_pixels': visible_pixels, 'error_px': error,
                                'observed_box': row['box'].tolist(),
                                'rendered_box': None if predicted is None else predicted.tolist(),
                                'exact_edges': row['edge_mask'].tolist(),
                                'censored_edges': row['censored_edges'].tolist(),
                                'weight': row['weight'], 'reason': reason})
    finally:
        if renderer is not None:
            if hasattr(renderer, 'close'):
                renderer.close()
            del renderer
        m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = saved_camera
        m.vis.quality.offsamples = saved_samples
        d.qpos[:], d.qvel[:], d.time = saved_state
        mujoco.mj_forward(m, d)
    errors = np.array([item['error_px'] for item in details])
    weights = np.array([item['weight'] for item in details])
    missing = sum(item['visible_pixels'] < 10 for item in details)
    median, p90 = float(np.median(errors)), float(np.quantile(errors, .9))
    passed = bool(evidence['sufficient'] and not missing and median <= MIN_MEDIAN_LIMIT_PX and p90 <= P90_LIMIT_PX)
    reason = ('insufficient_correspondences' if not evidence['sufficient'] else
              'expected_objects_not_visible' if missing else
              'rendered_geometry_thresholds_met_pending_K3' if passed else
              'rendered_geometry_gate_failed')
    return dict(base, passed=passed, reason=reason, median_px=median, p90_px=p90,
                weighted_mean_px=float(np.average(errors, weights=weights)),
                missing_observations=int(missing), visible_observations=len(details)-int(missing),
                rows=details, rendering_diagnostics=rendering_diagnostics)


def fit_visible_camera(loaded_tuple, kind, role, name, rows, initial, translation_limit):
    """Generate deterministic proposals, rank only on actual TRAIN renders.

    No held-out reference is accepted by this interface. Mount translation is
    additionally limited in Euclidean norm (wrist .20 m, world/head .90 m),
    consistent with the subsequent bounded K3 repair tool.
    """
    valid_rows, excluded = _rows(rows)
    p0 = np.asarray(initial['position_in_parent_m'], dtype=float)
    R0 = np.asarray(initial['rotation_in_parent'], dtype=float)
    norm_limit = .9 if role == 'head' else .2
    candidates = [('prior', deepcopy(initial), {'proposal': 'unchanged_fixed_mount'})]
    proposals = [('full_box_fit', initial, translation_limit),
                 ('fixed_position_box_fit', initial, 0.),
                 ('small_mount_box_fit', initial, .025)]
    for direction in (-1., 1.):
        shifted = deepcopy(initial)
        shifted['position_in_parent_m'] = (p0 + R0[:, 2]*(direction*.025)).tolist()
        proposals.append(('optical_z_minus_25mm' if direction < 0 else 'optical_z_plus_25mm', shifted, 0.))
    for label, prior, limit in proposals:
        try:
            candidate, diagnostic = fit_boxes(valid_rows, [], prior, limit)
            candidates.append((label, candidate, diagnostic))
        except (ValueError, RuntimeError) as error:
            candidates.append((label, None, {'proposal_error': str(error)}))
    reports, ranked, seen = [], [], set()
    for label, candidate, diagnostic in candidates:
        entry = {'candidate': label, 'proposal_diagnostic': diagnostic}
        if candidate is None:
            reports.append(dict(entry, rejected=True, reason='proposal_failed'))
            continue
        shift = float(np.linalg.norm(np.asarray(candidate['position_in_parent_m'])-p0))
        entry.update(profile=candidate, translation_from_initial_m=shift)
        if shift > norm_limit + 1e-9:
            reports.append(dict(entry, rejected=True, reason='cumulative_mount_translation_bound',
                                translation_norm_limit_m=norm_limit))
            continue
        key = json.dumps({k: candidate[k] for k in ('position_in_parent_m', 'rotation_in_parent', 'fovy_degrees')}, sort_keys=True)
        if key in seen:
            reports.append(dict(entry, rejected=True, reason='duplicate_camera_geometry'))
            continue
        seen.add(key)
        measured = evaluate_rendered_camera(loaded_tuple, name, valid_rows, candidate)
        entry.update(rejected=False, actual_visibility=measured)
        reports.append(entry)
        score = measured['weighted_mean_px']
        if score is not None:
            ranked.append((float(score), measured['missing_observations'], shift, len(ranked), label, candidate, measured))
    if not ranked:
        return deepcopy(initial), {'passed': False, 'reason': 'insufficient_visible_training_evidence',
                                   'selected': 'prior', 'candidates': reports, 'excluded': excluded,
                                   'selection_uses': 'training_actual_render_only'}
    best = min(ranked, key=lambda item: item[:4])
    profile = deepcopy(best[5])
    profile.update(basis='fixed_mount_candidate_selected_by_actual_training_visibility', hardware_calibrated=False)
    return profile, {'passed': False, 'reason': 'frozen_heldout_evaluation_still_required',
                     'kind': kind, 'view': role, 'selected': best[4], 'training': best[6],
                     'candidates': reports, 'excluded': excluded,
                     'translation_norm_limit_m': norm_limit,
                     'selection_uses': 'training_actual_render_only',
                     'real_calibration': False, 'physics_modified': False}
