"""Fit the user's site from one RealSense D455 head-camera frame (S1 analogue: sim2-real2/s1_heightfit_v2/fit_room.py).

Observations (observations.json, pixels in the 640x480 stream): the four corners of the cloth-covered desk top, the
near tips of the two sled feet, and the two far corners of the white plate on top of the FR3 Duo mount. Model:
  * D455 colour intrinsics at 640x480: vertical FOV 65 deg (fx = fy = 376.7 px), principal point at the centre;
  * the desk is the S1 desk model, 1.2 x 0.6 m, sled feet 0.56 m either side of its centre; the feet tips' depth
    under the top is fitted;
  * the plate is the top of fr3_duo_cover (0.25 m wide, front edge 0.065 m ahead of and 0.175 m above the mounting
    point); the mounting point is 0.775 m above the floor (the user's value).
Unknowns: camera pose, desk height, feet tip depth, the mount's position and yaw on the floor (a weak 5 deg prior
keeps it square to the desk). Output: the camera pose in the mount frame, the mount behind the desk's near edge and
its lateral offset, residuals.

    python -m franka_dual_v1.site_fit --observations <observations.json> --out <d455_fit.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

W, D = 1.2, 0.6
FEET_X = 0.56
F_PX, CX, CY = 376.7, 320.0, 240.0
MOUNT_ABOVE_FLOOR = 0.775
PLATE = dict(front_x=0.065, half_width=0.125, z=0.175)


def project(P, R, t):
    q = (np.asarray(P) - t) @ R  # camera axes as columns of R: x right, y down, z forward
    return np.array([CX + F_PX * q[0] / q[2], CY + F_PX * q[1] / q[2]])


def mount_axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 1.0]], 1)  # mount x fwd, y left, z up in the desk frame


def fit(obs):
    top = {'far_left': (-W / 2, D / 2, 0), 'far_right': (W / 2, D / 2, 0), 'near_left': (-W / 2, -D / 2, 0),
           'near_right': (W / 2, -D / 2, 0)}

    def unpack(x):
        return Rot.from_rotvec(x[:3]).as_matrix(), x[3:6], x[6], x[7], x[8:10], x[10]

    def residual(x):
        R, t, h, tip, pm, yaw = unpack(x)
        M, o = mount_axes(yaw), np.array([pm[0], pm[1], MOUNT_ABOVE_FLOOR - h])
        r = []
        for k, p in obs['desk_top'].items():
            r += list(project(top[k], R, t) - p)
        for k, p in obs['feet_tips'].items():
            r += list(project((-FEET_X if k == 'left' else FEET_X, -D / 2 + tip, -h), R, t) - p)
        for k, p in obs['mount_plate_far_corners'].items():
            y = PLATE['half_width'] if k == 'left' else -PLATE['half_width']
            r += list(project(o + M @ [PLATE['front_x'], y, PLATE['z']], R, t) - p)
        r.append(yaw / np.radians(5))
        return np.array(r)

    best = None
    for pitch in (40, 50, 60):
        for ty in (-0.7, -0.9):
            x0 = np.r_[Rot.from_euler('x', -90 - pitch, degrees=True).as_rotvec(), 0, ty, 0.7, 0.76, 0.1, 0, -0.75, 0]
            lb = [-np.inf] * 6 + [0.65, -0.05, -0.5, -1.5, -0.3]
            ub = [np.inf] * 6 + [0.9, 0.35, 0.5, -0.3, 0.3]
            sol = least_squares(residual, x0, bounds=(lb, ub))
            if best is None or sol.cost < best.cost:
                best = sol
    R, t, h, tip, pm, yaw = unpack(best.x)
    M, o = mount_axes(yaw), np.array([pm[0], pm[1], MOUNT_ABOVE_FLOOR - h])
    Rm = M.T @ R
    z = Rm[:, 2]
    res = best.fun[:-1]
    return dict(schema='rssr.fr3_duo.site_fit.v1', observations=obs,
                intrinsics=dict(width=640, height=480, fx=F_PX, fy=F_PX, cx=CX, cy=CY, vfov_deg=65.0),
                camera_in_mount=dict(position_m=(M.T @ (t - o)).tolist(), R_cam_xright_ydown_zfwd=Rm.tolist(),
                                     pitch_down_deg=float(np.degrees(np.arcsin(-z[2]))),
                                     yaw_left_deg=float(np.degrees(np.arctan2(z[1], z[0])))),
                desk_height_m=float(h), feet_tip_inside_near_edge_m=float(tip),
                mount_behind_near_edge_m=float(-D / 2 - pm[1]), mount_right_of_desk_centre_m=float(pm[0]),
                mount_yaw_deg=float(np.degrees(yaw)), rms_px=float(np.sqrt(np.mean(res ** 2))),
                residuals_px=np.round(res, 2).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--observations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    r = fit(json.loads(a.observations.read_text()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1) + '\n')
    print(json.dumps({k: r[k] for k in ('rms_px', 'desk_height_m', 'mount_behind_near_edge_m',
                                        'mount_right_of_desk_centre_m', 'camera_in_mount')}))


if __name__ == '__main__':
    main()
