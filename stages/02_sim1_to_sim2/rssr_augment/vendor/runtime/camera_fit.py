"""Bounded fixed-camera fitting against approximate object correspondences.

Coordinates use MuJoCo's camera-to-parent rotation: image right +X, up +Y,
view direction -Z. Fits are visual estimates, never measured calibration.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def color_centers(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    masks = {
        'blue_cube': (h >= 90) & (h <= 130) & (s >= 100) & (v >= 40),
        'red_cube': ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 40),
        'green_cube': (h >= 35) & (h <= 88) & (s >= 90) & (v >= 30),
    }
    result = {}
    for name, mask in masks.items():
        n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype('uint8'))
        if n < 2:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, height, area = stats[k]
        # Clipped or tiny patches are not reliable 2-D object centers.
        if area < 30 or x <= 1 or y <= 1 or x+w >= rgb.shape[1]-1 or y+height >= rgb.shape[0]-1:
            continue
        result[name] = centers[k].tolist()
    return result


def project(points, position, rotation, fovy, width=480, height=360):
    xyz = (np.asarray(points)-position) @ rotation
    depth = -xyz[:, 2]
    focal = height/(2*np.tan(np.deg2rad(fovy)/2))
    uv = np.c_[width/2+focal*xyz[:, 0]/np.maximum(depth, .005),
               height/2-focal*xyz[:, 1]/np.maximum(depth, .005)]
    return uv, depth


def errors(points, pixels, profile):
    if len(points) == 0:
        return {'count': 0, 'median_px': None, 'p90_px': None, 'in_front_fraction': 0.}
    uv, depth = project(points, np.array(profile['position_in_parent_m']),
                        np.array(profile['rotation_in_parent']), profile['fovy_degrees'])
    err = np.linalg.norm(uv-pixels, axis=1)
    return {'count': len(err), 'median_px': float(np.median(err)),
            'p90_px': float(np.quantile(err, .9)), 'in_front_fraction': float(np.mean(depth > .02))}


def fit_camera(train, heldout, initial, world_fixed, translation_limit):
    points = np.array([r['point_in_parent'] for r in train]);pixels = np.array([r['pixel'] for r in train])
    validation_points = np.array([r['point_in_parent'] for r in heldout])
    validation_pixels = np.array([r['pixel'] for r in heldout])
    report = {'training_observations': len(train), 'heldout_observations': len(heldout),
              'uses_proxy_object_geometry': True, 'real_calibration': False}
    if len(train) < 6 or len(heldout) < 3 or len({r['frame'] for r in train}) < 3:
        return initial, dict(report, passed=False, reason='insufficient_correspondences')
    p0 = np.array(initial['position_in_parent_m']);r0 = np.array(initial['rotation_in_parent'])
    # Proxy layouts cannot identify real metric extrinsics. Keep installation
    # position and FOV frozen; propose only one constant image orientation.
    weights=np.array([1. if row['object']=='blue_cube' else .2 for row in train])
    planned='orientation_proposal' in initial
    lower = -np.deg2rad([20.,20.,10.]) if planned else np.array([-np.pi*1.01]*3)
    upper = -lower

    def unpack(x):
        return p0, r0 @ Rotation.from_rotvec(x).as_matrix(), initial['fovy_degrees']

    def residual(x):
        p, r, f = unpack(x);uv, depth = project(points, p, r, f)
        return np.r_[((uv-pixels)*np.sqrt(weights[:,None])).ravel()/30.,
                     np.maximum(.025-depth, 0)*100]

    fits = []
    for roll in ([0.] if planned else [0., np.pi, -np.pi/2, np.pi/2]):
        x = np.array([0.,0.,roll])
        opt = least_squares(residual, x, bounds=(lower, upper), loss='soft_l1',
                            f_scale=1., max_nfev=350)
        fits.append(opt)
    # Candidate selection uses training residual only; held-out observations are never fitted.
    best = min(fits, key=lambda r: r.cost)
    p, r, f = unpack(best.x)
    profile = dict(initial, position_in_parent_m=p.tolist(), rotation_in_parent=r.tolist(),
                   fovy_degrees=float(f), basis='fixed_camera_fit_to_proxy_correspondences',
                   hardware_calibrated=False)
    report.update(training=errors(points, pixels, profile),
                  heldout=errors(validation_points, validation_pixels, profile),
                  before_heldout=errors(validation_points, validation_pixels, initial),
                  translation_change_m=float(np.linalg.norm(p-p0)),
                  rotation_change_degrees=float(np.rad2deg(np.linalg.norm(best.x))),
                  translation_optimized=False, intrinsic_optimized=False,
                  proxy_red_green_weight=.2,
                  optimum_at_bounds=bool(np.any(np.minimum(best.x-lower, upper-best.x)<1e-4)))
    e=report['heldout'];before=report['before_heldout']
    report['passed'] = bool(e['median_px'] <= 35 and e['p90_px'] <= 75 and
                            e['in_front_fraction'] == 1. and e['median_px'] <= before['median_px']+2)
    return profile, report
