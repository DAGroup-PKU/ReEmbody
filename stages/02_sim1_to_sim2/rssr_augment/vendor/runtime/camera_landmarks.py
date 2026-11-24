"""Observed image evidence, including objects truncated by the image boundary.

A visible bounding box is not a guess at an off-screen object center. Only
unoccluded silhouettes (possibly frame-truncated) can constrain this coarse fit.
Rendered review still checks occlusion, orientation and camera role separately.
"""
import itertools
import numpy as np


LANDMARK_PROMPT = '''Read the three labeled Real1 first-frame images in image_order.
Identify only supplied object IDs. Return JSON views with exactly head,
left_wrist, right_wrist. The EXACT outer shape is an array, not a name-keyed
object: {"views":[{"view":"head","landmarks":[]},
{"view":"left_wrist","landmarks":[]},{"view":"right_wrist","landmarks":[]}]}.
Fill those landmark arrays. Each landmark has object_id, visible,
center_xy_normalized, bbox_xyxy_normalized, reason. For a recognizable object
whose outline is unoccluded, report its visible bounding rectangle [xmin,ymin,
xmax,ymax] in full-image normalized coordinates; center_xy_normalized=null.
An object cut by the IMAGE BORDER is still observable: bound only its visible
part, put the truncated edge at 0 or 1, and describe truncation in reason. Do
not extend the rectangle or infer the hidden center. Partial OCCLUSION by a
gripper or another object is different: if the full outline is unknown, set
visible=false and both coordinates=null. If using an unequivocally observed
center instead of a bounding rectangle, set bbox_xyxy_normalized=null.
Never invent visibility, coordinates, or hidden outlines from task intent.
This is coarse visual registration evidence, not measured calibration or task
success. Object centers outside the image do not imply the whole object is
invisible. If nothing can be identified, return unknown, not fabricated data.'''


def validate_landmark(row):
    if type(row.get('visible')) is not bool:
        raise ValueError('Landmark visibility must be explicitly observed')
    center, box = row.get('center_xy_normalized'), row.get('bbox_xyxy_normalized')
    if not row['visible']:
        if center is not None or box is not None:
            raise ValueError('Invisible landmark cannot have inferred coordinates')
        return dict(row, center_xy_normalized=None)
    if (center is None) == (box is None):
        raise ValueError('Visible landmark needs exactly one center or visible box')
    value = np.asarray(box if box is not None else center, float)
    expected = (4,) if box is not None else (2,)
    if value.shape != expected or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError('Landmark coordinates must be finite and normalized')
    if box is not None and (box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError('Visible bounding box must have positive area')
    if not isinstance(row.get('reason'), str) or not row['reason'].strip():
        raise ValueError('Landmark requires an observation reason')
    key = 'bbox_xyxy_normalized' if box is not None else 'center_xy_normalized'
    return dict(row, **{key: value.tolist()})


BOX_SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))
BOX_EDGES = [(i, j) for i in range(8) for j in range(i+1, 8)
             if np.count_nonzero(BOX_SIGNS[i] != BOX_SIGNS[j]) == 1]


def project_visible_bounds(corners_in_camera, width, height, focal):
    """Project an oriented box, near-plane clip its edges, then clip to image.

    Returns None if the box has no visible image area. MuJoCo camera axes are
    +x right, +y up, -z forward. Input corner ordering is BOX_SIGNS ordering.
    """
    corners = np.asarray(corners_in_camera, float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError('Expected eight finite geometric bounding corners')
    near = .005
    inside = -corners[:, 2] >= near
    points = list(corners[inside])
    for i, j in BOX_EDGES:
        if bool(inside[i]) != bool(inside[j]):
            fraction = (-near-corners[i, 2])/(corners[j, 2]-corners[i, 2])
            points.append(corners[i]+fraction*(corners[j]-corners[i]))
    if not points:
        return None
    points = np.asarray(points)
    uv = np.c_[width/2+focal*points[:, 0]/(-points[:, 2]),
               height/2-focal*points[:, 1]/(-points[:, 2])]
    lo, hi = np.maximum(uv.min(axis=0), [0, 0]), np.minimum(uv.max(axis=0), [width, height])
    if np.any(hi <= lo):
        return None
    return np.r_[lo, hi]
