"""Fixed-camera visual estimation from approximate 3-D object boxes.

This module never changes scene geometry or robot/object states.  A fitted
profile is a visual estimate, not hardware calibration: approximate object
dimensions/layouts can bias all estimated camera parameters.  Callers must
add a semantic/visual check; a small box residual alone is not camera proof.

Coordinates follow MuJoCo: camera +X right, +Y up and -Z forward.  Bboxes use
continuous pixel edge coordinates [left, top, right, bottom], right/bottom
exclusive for detections.  Train and held-out frame identifiers must differ.
"""
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from camera_fit import project


MAX_ROTATION_RADIANS = np.deg2rad(60.)
MIN_MEDIAN_LIMIT_PX = 35.
P90_LIMIT_PX = 75.


def detect_boxes(rgb):
    """Return conservative color-mask candidates, including rejection reasons.

    A missing key means no sizeable component.  ``geometry_reliable=False``
    means the box must not be fitted (tiny/fragmented/ambiguous mask).
    A color mask cannot certify absence of occlusion or identify semantics;
    a caller with additional evidence can set ``occluded=True`` on its rows.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image with three channels')
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    masks = {
        'blue_cube': (h >= 90) & (h <= 130) & (s >= 100) & (v >= 40),
        'red_cube': ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 40),
        'green_cube': (h >= 35) & (h <= 88) & (s >= 90) & (v >= 30),
        'basket': (h >= 8) & (h <= 30) & (s >= 55) & (v >= 35) & (v <= 235),
    }
    result = {}
    for name, mask in masks.items():
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if count < 2:
            continue
        order = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        x, y, box_w, box_h, area = [int(a) for a in stats[order[0]]]
        if area < 10:
            continue
        clipped = x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1
        fill = area / (box_w * box_h)
        ambiguous = len(order) > 1 and stats[order[1], cv2.CC_STAT_AREA] >= .5 * area
        reasons = []
        if area < 30 or min(box_w, box_h) < 5:
            reasons.append('tiny_mask')
        if fill < (.12 if name == 'basket' else .25):
            reasons.append('sparse_or_fragmented_mask')
        if ambiguous:
            reasons.append('multiple_comparable_components')
        result[name] = {
            'box': [x, y, x + box_w, y + box_h],
            'area_px': area, 'fill_fraction': float(fill),
            'width': width, 'height': height,
            'clipped': bool(clipped),
            'edge_mask': [x > 1, y > 1, x + box_w < width - 1, y + box_h < height - 1],
            'censored_edges': [x <= 1, y <= 1, x + box_w >= width - 1, y + box_h >= height - 1],
            'geometry_reliable': not reasons, 'rejection_reasons': reasons,
            'weight': 1. if name in ('blue_cube', 'basket') else .2,
            'source': 'color_mask_candidate_not_semantically_verified',
        }
    return result


def _rows(rows):
    valid, excluded = [], []
    for row in rows:
        reasons = []
        try:
            points = np.asarray(row['corners_in_parent'], dtype=float)
            box = np.asarray(row['box'], dtype=float)
            edges = np.asarray(row.get('edge_mask', [True] * 4), dtype=bool).copy()
            weight = float(row.get('weight', 1.))
            width, height = int(row.get('width', 480)), int(row.get('height', 360))
            censored = np.asarray(row.get('censored_edges', [False] * 4), dtype=bool).copy()
            if points.shape != (8, 3) or box.shape != (4,) or edges.shape != (4,):
                reasons.append('invalid_shape')
            elif not np.isfinite(points).all() or not np.isfinite(box).all():
                reasons.append('nonfinite_geometry')
            elif box[2] - box[0] < 5 or box[3] - box[1] < 5:
                reasons.append('tiny_or_invalid_box')
            if not np.isfinite(weight) or weight <= 0 or width <= 0 or height <= 0:
                reasons.append('invalid_weight_or_resolution')
            # Image-boundary truncation is censoring, not an exact full-object
            # extremum. Occlusion by another object is different and excluded.
            if box.shape == (4,) and edges.shape == (4,) and censored.shape == (4,):
                censored |= [box[0] <= 1, box[1] <= 1, box[2] >= width-1, box[3] >= height-1]
                edges &= ~censored
                if not edges.any():
                    reasons.append('no_exact_edges')
            else:
                reasons.append('invalid_edge_flags')
            if row.get('occluded', False):
                reasons.append('explicit_occlusion')
            if row.get('clipped', False) and censored.shape == (4,) and not censored.any():
                reasons.append('unspecified_clipped_edges')
            if not row.get('geometry_reliable', True):
                reasons.append('unreliable_detection')
            if 'frame' not in row or 'object' not in row:
                reasons.append('missing_frame_or_object')
        except (KeyError, TypeError, ValueError):
            reasons.append('invalid_row')
        if reasons:
            excluded.append({'frame': row.get('frame'), 'object': row.get('object'),
                             'reasons': sorted(set(reasons))})
        else:
            valid.append(dict(row, corners_in_parent=points, box=box, weight=weight,
                              width=width, height=height, edge_mask=edges,
                              censored_edges=censored))
    return valid, excluded


def _project_boxes(rows, profile):
    boxes, depths = [], []
    for row in rows:
        uv, depth = project(row['corners_in_parent'], np.asarray(profile['position_in_parent_m']),
                            np.asarray(profile['rotation_in_parent']), profile['fovy_degrees'],
                            width=row['width'], height=row['height'])
        boxes.append(np.r_[uv.min(axis=0), uv.max(axis=0)])
        depths.append(depth)
    return np.asarray(boxes).reshape(-1, 4), np.asarray(depths).reshape(-1, 8)


def _edge_residuals(rows, boxes):
    observed = np.asarray([r['box'] for r in rows])
    exact = np.asarray([r['edge_mask'] for r in rows])
    censored = np.asarray([r['censored_edges'] for r in rows])
    delta = boxes-observed
    residual = np.where(exact, delta, 0.)
    # left/top must project at or beyond their visible border; right/bottom
    # must project at or beyond theirs in the opposite direction.
    violation = np.c_[np.maximum(delta[:, :2], 0.), np.minimum(delta[:, 2:], 0.)]
    return residual + np.where(censored, violation, 0.)


def _metrics(rows, profile):
    if not rows:
        return {'count': 0, 'median_px': None, 'p90_px': None, 'in_front_fraction': 0.}
    boxes, depths = _project_boxes(rows, profile)
    observed = np.asarray([r['box'] for r in rows])
    # RMS of Euclidean displacement of top-left and bottom-right corners.
    # Report unweighted errors: low proxy weights cannot hide validation errors.
    residuals = _edge_residuals(rows, boxes)
    errors = np.linalg.norm(residuals, axis=1) / np.sqrt(2.)
    center_errors = np.linalg.norm((boxes[:, :2] + boxes[:, 2:] -
                                   observed[:, :2] - observed[:, 2:]) / 2., axis=1)
    return {
        'count': len(rows), 'median_px': float(np.median(errors)),
        'p90_px': float(np.quantile(errors, .9)),
        'in_front_fraction': float(np.mean(depths > .02)),
        'center_median_px': float(np.median(center_errors)),
        'rows': [{'frame': row['frame'], 'object': row['object'],
                  'error_px': float(err), 'minimum_depth_m': float(np.min(depth)),
                  'exact_edges': row['edge_mask'].tolist(),
                  'censored_edges': row['censored_edges'].tolist(),
                  'observed_box': row['box'].tolist(), 'predicted_box': pred.tolist()}
                 for row, err, depth, pred in zip(rows, errors, depths, boxes)],
    }


def _evidence(rows, minimum_rows, minimum_frames, minimum_edges):
    exact = np.asarray([r['edge_mask'] for r in rows], dtype=bool).reshape(-1, 4)
    return {
        'count': len(rows), 'frames': len({r['frame'] for r in rows}),
        'exact_edges': int(exact.sum()),
        'horizontal_exact_edges': int(exact[:, [0, 2]].sum()),
        'vertical_exact_edges': int(exact[:, [1, 3]].sum()),
        'sufficient': bool(len(rows) >= minimum_rows and len({r['frame'] for r in rows}) >= minimum_frames
                           and exact.sum() >= minimum_edges and exact[:, [0, 2]].sum() >= 2
                           and exact[:, [1, 3]].sum() >= 2),
    }


def evaluate_boxes(rows, profile):
    """Evaluate a frozen profile on untouched frames, without optimization.

    At least 3 observations, 2 frames, 6 exact edges and both image dimensions
    are needed. Censored edges contribute one-sided violations only. Missing
    or explicitly occluded observations cannot supply evidence.
    """
    rows, excluded = _rows(rows)
    metric = _metrics(rows, profile)
    evidence = _evidence(rows, 3, 2, 6)
    metric.update(excluded=excluded, evidence=evidence,
                  real_calibration=False, uses_proxy_object_geometry=True)
    metric['passed'] = bool(evidence['sufficient'] and metric['median_px'] <= MIN_MEDIAN_LIMIT_PX
                            and metric['p90_px'] <= P90_LIMIT_PX and metric['in_front_fraction'] == 1.)
    metric['reason'] = ('insufficient_correspondences' if not evidence['sufficient'] else
                        'geometry_thresholds_met_pending_semantic_verification' if metric['passed'] else
                        'heldout_geometry_gate_failed')
    return metric


def _rotation_vector(parameter):
    # Smooth radial map into the open 60 degree ball. The bounded optimizer
    # cannot evade the installation-rotation limit using simultaneous axes.
    norm = np.linalg.norm(parameter)
    return parameter * (MAX_ROTATION_RADIANS * np.tanh(norm) / norm if norm > 1e-12
                        else MAX_ROTATION_RADIANS)


def fit_boxes(train, heldout, initial, translation_limit):
    """Fit one constant parent-relative SE(3) camera and vertical FOV.

    Rows: frame, object, corners_in_parent (8x3), box (4), optional weight,
    width/height, geometry_reliable, clipped, occluded, edge_mask. Incomplete
    explicitly occluded rows are excluded. Image-boundary clipping gives
    one-sided residuals; unmeasured non-boundary edges are ignored.
    ``translation_limit`` is a scalar or xyz vector, in metres per axis.
    No held-out row participates in optimization or multistart selection.
    """
    train, train_excluded = _rows(train)
    heldout, heldout_excluded = _rows(heldout)
    report = {
        'training_observations': len(train), 'heldout_observations': len(heldout),
        'training_excluded': train_excluded, 'heldout_excluded': heldout_excluded,
        'uses_proxy_object_geometry': True, 'real_calibration': False,
        'method': 'constant_parent_pose_and_fovy_box_fit',
        'metric': 'rms_bbox_corner_displacement_with_one_sided_censoring_px',
        'thresholds': {'median_px': MIN_MEDIAN_LIMIT_PX, 'p90_px': P90_LIMIT_PX},
        'limitation': 'Approximate dimensions/layouts bias the visual estimate; independent semantic camera verification is required.',
    }
    train_frames = {r['frame'] for r in train}
    heldout_frames = {r['frame'] for r in heldout}
    if train_frames & heldout_frames:
        return deepcopy(initial), dict(report, passed=False, reason='train_heldout_frame_overlap')
    training_evidence = _evidence(train, 6, 3, 12)
    report['training_evidence'] = training_evidence
    if not training_evidence['sufficient']:
        return deepcopy(initial), dict(report, passed=False, reason='insufficient_correspondences')

    p0 = np.asarray(initial['position_in_parent_m'], dtype=float)
    r0 = np.asarray(initial['rotation_in_parent'], dtype=float)
    f0 = float(initial['fovy_degrees'])
    limits = np.broadcast_to(np.asarray(translation_limit, dtype=float), (3,)).copy()
    if (p0.shape != (3,) or r0.shape != (3, 3) or not np.isfinite(p0).all() or
            not np.isfinite(r0).all() or not np.allclose(r0.T @ r0, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(r0), 1., atol=1e-6) or
            not np.isfinite(limits).all() or np.any(limits < 0) or not 20. <= f0 <= 110.):
        raise ValueError('Invalid camera prior, translation bounds or SO(3) rotation')

    boxes = np.asarray([r['box'] for r in train])
    weights = np.sqrt(np.asarray([r['weight'] for r in train]))
    lower, upper = np.r_[-np.ones(3), -np.ones(3)*3., 20.], np.r_[np.ones(3), np.ones(3)*3., 110.]

    def profile_for(x):
        return dict(initial, position_in_parent_m=(p0 + limits*x[:3]).tolist(),
                    rotation_in_parent=(r0 @ Rotation.from_rotvec(_rotation_vector(x[3:6])).as_matrix()).tolist(),
                    fovy_degrees=float(x[6]))

    def residual(x):
        pred, depth = _project_boxes(train, profile_for(x))
        return np.r_[(_edge_residuals(train, pred)*weights[:, None]).ravel()/30.,
                     np.maximum(.025-depth, 0).ravel()*100.,
                     .2*x[:3], .1*_rotation_vector(x[3:6])/MAX_ROTATION_RADIANS,
                     .2*np.log(x[6]/f0)]

    starts = [np.r_[np.zeros(6), f0]]
    # Deterministic multistart, entirely defined before observing held-out data.
    for axis in range(3):
        for sign in (-1., 1.):
            start = starts[0].copy()
            start[axis+3] = sign*np.arctanh(20./60.)
            starts.append(start)
    for factor in (.75, 1.25):
        start = starts[0].copy()
        start[6] = np.clip(f0*factor, 20.001, 109.999)
        starts.append(start)
    fits = [least_squares(residual, start, bounds=(lower, upper), loss='soft_l1',
                          f_scale=1., max_nfev=250, x_scale='jac') for start in starts]
    best = min(fits, key=lambda candidate: candidate.cost)
    profile = profile_for(best.x)
    profile.update(basis='fixed_pose_fovy_visual_fit_to_approximate_object_boxes',
                   hardware_calibrated=False)
    metric = evaluate_boxes(heldout, profile)
    before = _metrics(heldout, initial)
    report.update(
        training=_metrics(train, profile), heldout=metric, before_heldout=before,
        translation_change_m=float(np.linalg.norm(limits*best.x[:3])),
        translation_change_per_axis_m=(limits*best.x[:3]).tolist(),
        translation_limit_per_axis_m=limits.tolist(),
        rotation_change_degrees=float(np.rad2deg(np.linalg.norm(_rotation_vector(best.x[3:6])))),
        fovy_before_degrees=f0, fovy_after_degrees=profile['fovy_degrees'],
        translation_optimized=bool(np.any(limits > 0)), intrinsic_optimized=True,
        multistart_count=len(starts), selection_uses='training_cost_only',
        optimizer_success=bool(best.success), total_function_evaluations=sum(x.nfev for x in fits),
        optimum_at_bounds=bool(np.any(np.minimum(best.x-lower, upper-best.x) < 1e-4)),
    )
    # Preserve the original numerical thresholds and no-regression check.
    # This is a diagnostic geometry gate, never the final camera approval.
    report['passed'] = bool(metric['passed'] and metric['median_px'] <= before['median_px'] + 2.)
    report['reason'] = ('geometry_thresholds_met_pending_semantic_verification' if report['passed'] else
                        'heldout_not_evaluated' if not heldout else metric['reason'])
    return profile, report
