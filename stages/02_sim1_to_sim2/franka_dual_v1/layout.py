"""Dual-arm Franka layout: RoboTwin 2.0 `franka-panda` placement mapped into the RSSR source frame.

Not aligned with the real dual-arm Franka. The user approved this placeholder on
2026-11-02 until real mount and camera parameters are available.

RoboTwin (third_party/RoboTwin, envs/robot/robot.py, embodiments/franka-panda/config.yml,
env_cfg/task_config/_camera_config.yml):
  * robot_pose (0, -0.65, 0.75), quat wxyz (0.707, 0, 0, 0.707): arm x-axis points +y, toward the table;
  * two single-arm bodies, left at x - 0.4, right at x + 0.4 (embodiment_dis 0.8 m);
  * table top z 0.74, near edge y -0.35;
  * head camera at (-0.032, -0.45, 1.35), forward (0, 0.6, -0.8), left (-1, 0, 0), D435 fovy 37 deg, 320x240;
  * wrist D435 on panda_hand: hand_to_camera_mount rpy (pi, -pi/2, -pi/2), camera_joint rpy (pi/2, 0, 0)
    xyz (0.042, -0.045, 0); in the hand frame the camera sits at (0.045, 0, 0.042), looks along hand +z
    (toward the fingers), image up = hand +x.

RSSR source frame (RSSR/data/150case/cases/*/source_reference/scene.xml): the cobot faces -y, its left
arm at +x; table top z 0.75, near edge y -0.35; cobot arm bases at y -0.29 .. -0.39.

The map is RoboTwin -> RSSR = rotate 180 deg about z, then translate. Two anchors:
  * 'source_base' (default): the arm base line lands on the cobot's base line (y = -0.33), so object
    reach matches the source demonstrations;
  * 'table_edge': RoboTwin's base-to-table-edge distance (0.30 m) is kept; base line y = -0.05;
  * 'base_line': the base line at `base_line_y`.
All keep RoboTwin's arm spacing and base height above the table (+0.01 m). The head camera is either
RoboTwin's, kept at its pose relative to the arm bases (`head_spec=None`), or `head_spec` =
(forward, height above the table, lateral, fovy) from the arm base midpoint, looking at HEAD_LOOK_AT.

SELECTED (2026-11-02, evidence in result/franka_dual_luna_v1_20261102_attempt_01/layout_check/):
  * base line y = -0.19 (0.16 m behind the table edge): every IK key pose of the assigned hands is
    reachable in 50/50 cup, 50/50 block and 45/50 drawer source layouts, against 50/32/39 at the cobot
    base line (-0.33) and 49/49/37 at RoboTwin's table-edge distance (-0.05);
  * head camera 0.30 m behind the base midpoint, 0.60 m above the table (1.35 m, RoboTwin's height),
    fovy 37: chosen over 50 candidates by arm occlusion at placement. RoboTwin's pose, mapped here,
    leaves an object out of view at t = 0 in 25/150 cases and hides 38% of the place location;
  * start pose (frame 0 of every task) START_Q: canonical Panda posture (q3 = q5 = 0, q7 = pi/4), chosen
    over a joint grid by frame-0 wrist visibility (start_pose_sweep_v2); each hand's grasp target is in its
    wrist camera at t = 0 in all 150 source layouts (min 1368 px left, 1023 px right, 320x240). It replaces
    the v1 pick START_Q_V1 (a back-leaning, fully folded arm from which continuous IK to a grasp swings the
    elbow by > 1 rad). RoboTwin's home pose points the hand straight down: 245 of the 300 targets have
    0 px at t = 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

RT_BASE = np.array([0.0, -0.65, 0.75])
RT_TABLE_TOP = 0.74
RT_TABLE_NEAR_EDGE_Y = -0.35
RT_SPACING = 0.8
RT_HEAD_POS = np.array([-0.032, -0.45, 1.35])
RT_HEAD_FORWARD = np.array([0.0, 0.6, -0.8])
RT_HEAD_LEFT = np.array([-1.0, 0.0, 0.0])
D435_FOVY_DEG = 37.0
D435_W, D435_H = 320, 240
# RoboTwin franka-panda homestate (both arms)
HOME_Q = (0.0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483)

HEAD_LOOK_AT = np.array([0.0, -0.82, 0.80])  # centre of the source object and cabinet extents
SRC_TABLE_TOP = 0.75
SRC_TABLE_NEAR_EDGE_Y = -0.35
SRC_BASE_LINE_Y = -0.33

HANDS = ('left', 'right')
# Wrist camera in the menagerie `hand` body frame (same frame as URDF panda_hand).
WRIST_CAM_POS = (0.045, 0.0, 0.042)
WRIST_CAM_XYAXES = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)  # image right = hand +y, image up = hand +x
TCP_OFFSET = (0.0, 0.0, 0.1034)  # Franka Hand TCP from the flange-side hand origin


def rot_z180(v):
    v = np.asarray(v, float)
    return np.array([-v[0], -v[1], v[2]])


@dataclass(frozen=True)
class Layout:
    anchor: str = 'source_base'
    base_line_y: float | None = None
    spacing_m: float = RT_SPACING
    head_fovy_deg: float = D435_FOVY_DEG
    wrist_fovy_deg: float = D435_FOVY_DEG
    head_spec: tuple | None = None
    start_q: tuple | None = None  # ((7 left), (7 right)); None = RoboTwin HOME_Q

    def translation(self):
        rt_base = rot_z180(RT_BASE)
        dz = SRC_TABLE_TOP - RT_TABLE_TOP
        if self.anchor == 'source_base':
            return np.array([0.0, SRC_BASE_LINE_Y - rt_base[1], dz])
        if self.anchor == 'table_edge':
            return np.array([0.0, SRC_TABLE_NEAR_EDGE_Y - (-RT_TABLE_NEAR_EDGE_Y), dz])
        if self.anchor == 'base_line':
            return np.array([0.0, self.base_line_y - rt_base[1], dz])
        raise ValueError(f'unknown anchor {self.anchor}')

    def base_pos(self, hand):
        # RoboTwin: left at x - spacing/2 facing +y; after the 180 deg turn left is at +x.
        rt = RT_BASE + np.array([(-1 if hand == 'left' else 1) * self.spacing_m / 2, 0, 0])
        return rot_z180(rt) + self.translation()

    @staticmethod
    def base_quat():
        # arm x-axis along -y: rotation of -90 deg about z (wxyz)
        return np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)])

    def head_pose(self):
        if self.head_spec is not None:
            fwd, up, lat, _ = self.head_spec
            mid = 0.5 * (self.base_pos('left') + self.base_pos('right'))
            pos = np.array([mid[0] + lat, mid[1] - fwd, SRC_TABLE_TOP + up])
            z = pos - HEAD_LOOK_AT  # camera looks along -z
            z /= np.linalg.norm(z)
            x = np.cross([0, 0, 1.0], z)
            x /= np.linalg.norm(x)
            return pos, np.r_[x, np.cross(z, x)]
        pos = rot_z180(RT_HEAD_POS) + self.translation()
        fwd = rot_z180(RT_HEAD_FORWARD)
        left = rot_z180(RT_HEAD_LEFT)
        fwd = fwd / np.linalg.norm(fwd)
        up = np.cross(fwd, left)
        right = -left
        return pos, np.r_[right, up]

    def arm_start_q(self, hand):
        return tuple(HOME_Q) if self.start_q is None else tuple(self.start_q[HANDS.index(hand)])

    def head_fovy(self):
        return self.head_spec[3] if self.head_spec is not None else self.head_fovy_deg

    def to_json(self):
        pos, xyaxes = self.head_pose()
        return dict(asdict(self), translation=self.translation().tolist(),
                    base_pos={h: self.base_pos(h).tolist() for h in HANDS},
                    base_quat_wxyz=self.base_quat().tolist(), head_pos=pos.tolist(),
                    head_xyaxes=xyaxes.tolist(), wrist_cam_pos_in_hand=list(WRIST_CAM_POS),
                    wrist_cam_xyaxes_in_hand=list(WRIST_CAM_XYAXES), tcp_offset_in_hand=list(TCP_OFFSET),
                    head_fovy=self.head_fovy(), robotwin_home_q=list(HOME_Q),
                    start_q={h: list(self.arm_start_q(h)) for h in HANDS}, real_robot_aligned=False)


START_Q_V1 = ((-0.156018, -1.086295, -0.625422, -2.889184, -0.057355, 2.46903, 0.017225),
              (0.157305, -1.033193, 0.584243, -2.832365, 0.072686, 2.373286, 1.539172))
START_Q = ((-0.2, -0.05, 0.0, -1.85, 0.0, 1.85, 0.785),
           (0.1, -0.25, 0.0, -1.85, 0.0, 1.75, 0.785))
# null-space posture of the deploy IK: q3 = q5 = 0 (masked), the canonical family of START_Q
NULL_POSTURE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.785)
NULL_MASK = (0, 0, 1, 0, 1, 0, 0)
SELECTED = Layout(anchor='base_line', base_line_y=-0.19, head_spec=(-0.30, 0.60, 0.0, 37.0), start_q=START_Q)
