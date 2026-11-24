"""Shared geometry for the centred-cabinet S1 drawer layout (v1).

Native frame: robot chassis_base at (0, +0.04, 0.097); robot forward is -y,
robot left is +x, table top z = 0.80.  All layout numbers in the user-facing
spec are given relative to the robot base centre (forward = -y, left = +x).
"""
from __future__ import annotations

import numpy as np
import mujoco

BASE_XY = np.array([0.0, 0.04])
TABLE_TOP_Z = 0.80
CAB_SIZE = np.array([0.35, 0.265, 0.35])
CAB_HALF_DEPTH = CAB_SIZE[1] / 2
SLIDE_RANGE = 0.17

# Installed Astribot SDK kinematic.xml camera mounts (pi0.5/operations/
# camera_mount_alignment_20261025_attempt_01/camera/SDK_S1v2_camera_nodes.json).
SDK_RAW = {
    'head_rgbd': dict(parent='astribot_head_link_2', pos=[0.0603, -0.2139, -0.0277],
                      quat=[0.7025, -0.0295, -0.71, -0.0135], fovy=51.4175),
    'left_wrist_rgbd': dict(parent='arm_left_tool_cam',
                            pos=[0.011420996159648984, 0.058882000224149605, 0.044260118849329295],
                            quat=[0.703108425353722, -0.7109134990656752, -0.009037420129074155,
                                  -0.012604129490710232], fovy=57.0),
    'right_wrist_rgbd': dict(parent='arm_right_tool_cam', pos=[0.00772, 0.05592, 0.04319],
                             quat=[-0.7130607009975457, 0.7006910271198059, 0.013493708973317153,
                                   0.019860539389038687], fovy=57.0),
}


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def sdk_cameras():
    """SDK nominal optical centre and optical axis; image rolled 180 deg about the
    optical axis so 'up' matches the photo-fitted training mounts and real images
    (the SDK MJCF frames are ~180 deg rolled relative to them)."""
    out = {}
    for name, spec in SDK_RAW.items():
        q = np.asarray(spec['quat'], float)
        q = q / np.linalg.norm(q)
        out[name] = dict(parent=spec['parent'], pos=list(spec['pos']),
                         quat=quat_mul(q, [0.0, 0.0, 0.0, 1.0]).tolist(), fovy=spec['fovy'],
                         sdk_quat_raw=list(spec['quat']), image_roll_deg=180.0)
    return out


def apply_cameras(model, cams):
    for name, spec in cams.items():
        c = model.camera(name).id
        if model.body(model.cam_bodyid[c]).name != spec['parent']:
            raise ValueError(f'camera parent mismatch {name}')
        model.cam_pos[c] = spec['pos']
        model.cam_quat[c] = spec['quat']
        model.cam_fovy[c] = spec['fovy']


def cabinet_center_y(front_from_base_m):
    """Cabinet centre y for a cabinet front face this far ahead of the base centre."""
    return BASE_XY[1] - front_from_base_m - CAB_HALF_DEPTH


def to_base(xy):
    """Native world xy -> (forward_m, left_m) from the robot base centre."""
    xy = np.asarray(xy, float)
    return np.array([BASE_XY[1] - xy[1], xy[0] - BASE_XY[0]])


def from_base(forward_m, left_m):
    return np.array([BASE_XY[0] + left_m, BASE_XY[1] - forward_m])
