"""FR3 Duo kinematic layout: two FR3v2 arms on the FR3 Duo mount, the Franka head camera, Robotiq
2F-85 TCP. Kinematics only (no gripper geometry, no actuators): for reach and camera-view estimates.

Source: franka_description (github.com/frankarobotics/franka_description), commit 7aeeddc4:
  * robots/fr3_duo/fr3_duo.xacro: default arms fr3v2 / fr3v2, prefixes left / right, each arm's base = the mount
    point fr3_duo_mount_<prefix>, the head on fr3_duo_mount_mounting_point;
  * accessories/fr3_duo_mount_v0_3/kinematics.yaml: origin = mounting point + (0.037, 0, 0.05); left arm base =
    origin + (0, +0.05018, 0), rpy (-0.89334809, -0.17456074, -0.46334506); right arm mirrored (y, roll, yaw negated);
  * accessories/franka_head_v0_2/kinematics.yaml: head = mounting point + (0, 0, 0.167); head camera mounting point =
    head + (0.042464, -0.014113, 0.223526), pitch 0.7156 rad (x forward, tilted down);
  * robots/fr3v2/joint_limits.yaml: joint ranges (wider than Menagerie franka_fr3_v2's, which is used for the links:
    same kinematic chain, 0.333 / 0.316 / 0.0825 / 0.384 / 0.088 / flange 0.107);
  * the mount + cover collision meshes span x -0.114 .. 0.118 m (front), y +-0.144, z -0.031 .. 0.175 from the
    mounting point; the head reaches z 0.415.
URDF frames: x forward, y left, z up; rpy is fixed-axis (R = Rz(yaw) Ry(pitch) Rx(roll)).

Placed in the RSSR source frame (robot facing -y, table top z 0.75, near table edge y -0.35): the mounting point at
(0, -0.35 + dist_behind_edge, 0.75 + height_above_table), mount +x along world -y, so the URDF left arm is at +x as in
the cobot data. The TCP is the Robotiq 2F-85 pinch point, 0.156 m along the flange z axis (Menagerie robotiq_2f85:
base_mount 0.007 + base 0.0038 + pinch 0.145); any adapter plate between flange and gripper is not included.
Wrist cameras (RealSense D405) and the torso camera have no mount in franka_description; they are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .scene import environment_xml
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from reembody.paths import resolve_franka_description, resolve_mujoco_menagerie  # noqa: E402

DESCRIPTION = resolve_franka_description()
DESCRIPTION_COMMIT = '7aeeddc449edf8d62b594f9e36a81da53e7796f9'
FR3V2_XML = resolve_mujoco_menagerie() / 'franka_fr3_v2/fr3v2.xml'
MOUNT_ORIGIN = np.array([0.037, 0.0, 0.05])
ARM_MOUNT = {'left': (np.array([0.0, 0.05018, 0.0]), (-0.89334809, -0.17456074, -0.46334506)),
             'right': (np.array([0.0, -0.05018, 0.0]), (0.89334809, -0.17456074, 0.46334506))}
HEAD = np.array([0.0, 0.0, 0.167])
HEAD_CAM = (np.array([0.042463788936189344, -0.014113, 0.22352590079539492]), (0.0, 0.7156, 0.0))
FR3V2_LIMITS = [(-2.9007400166666666, 2.9007400166666666), (-1.8360900166666667, 1.8360900166666667),
                (-2.9007400166666666, 2.9007400166666666), (-3.077020016666667, -0.11693708333333333),
                (-2.87630335, 2.87630335), (0.43982265, 4.62163335), (-3.05083335, 3.05083335)]
MOUNT_FRONT_X = 0.118
ROBOTIQ_2F85_TCP = 0.156
# Robotiq 2F-85 on the FR3 flange: rotated about the flange axis like franka_description's default hand (rpy_ee
# 0 0 -pi/4); adapter plates are not modelled. Superseded for the real robot by FACTORY_* (DuoPlacement fields).
GRIPPER_YAW = -np.pi / 4
# The user's factory URDF (fr3_duo_factory_robotiq.zip, 2026-11-05; result/franka_fr3_duo_urdf_factory_robotiq_20261105
# _attempt_01): arm bases' origin (0.0369, 0, 0.050885) from the mounting point; the Robotiq base_link 0.017 m past
# the flange on a 0.0375 m x 0.017 m adapter cylinder, not rotated (fingers close along the flange y axis). The
# Menagerie gripper's base sits 0.0108 m past its base_mount origin and is turned -90 deg in it (its fingers on its
# own y axis, like base_link's), so base_mount turns +90 deg and moves out 0.017 - 0.0108 = 0.0062 m.
FACTORY_MOUNT_ORIGIN = (0.0369, 0.0, 0.050885)
FACTORY_GRIPPER_YAW = np.pi / 2
FACTORY_GRIPPER_OFFSET = 0.0062
FACTORY_COUPLING = (0.0375, 0.017)
# ZED Mini at 1280x720 (HD720): 85 x 54 deg (Stereolabs); the max-FOV mode is 90 x 60
ZED_MINI_FOV_DEG = (85.0, 54.0)
TABLE_TOP = 0.75
TABLE_EDGE_Y = -0.35
HANDS = ('left', 'right')


def rpy(r, p, y):
    return Rotation.from_euler('xyz', [r, p, y]).as_matrix()


@dataclass(frozen=True)
class DuoPlacement:
    dist_behind_edge: float  # mounting point behind the near table edge (m)
    height_above_table: float  # mounting point above the table top (m)
    # Defaults: franka_description fr3_duo_mount_v0_3 and the Robotiq rotated like the Franka Hand, no adapter.
    # FACTORY_* below: the user's factory URDF (2026-11-05).
    mount_origin: tuple = tuple(MOUNT_ORIGIN)  # arm bases' common origin from the mounting point (m)
    gripper_yaw: float = GRIPPER_YAW  # 2F-85 about the flange axis (rad)
    gripper_offset: float = 0.0  # 2F-85 base_mount further along the flange axis (m)
    coupling: tuple | None = None  # (radius, length) of a flange-to-gripper adapter cylinder (m)

    def mount_pose(self):
        R = Rotation.from_euler('z', -90, degrees=True).as_matrix()  # mount +x -> world -y
        return np.array([0.0, TABLE_EDGE_Y + self.dist_behind_edge, TABLE_TOP + self.height_above_table]), R

    def arm_base(self, hand):
        p, R = self.mount_pose()
        t, e = ARM_MOUNT[hand]
        return p + R @ (np.asarray(self.mount_origin) + t), R @ rpy(*e)

    def head_camera(self):
        """Position, and the camera frame (x forward = optical axis, y left, z up) in the world."""
        p, R = self.mount_pose()
        t, e = HEAD_CAM
        return p + R @ (HEAD + t), R @ rpy(*e)


def build(case_dir: Path, scene: dict, place: DuoPlacement, tcp=ROBOTIQ_2F85_TCP) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(environment_xml(case_dir / 'source_reference/scene.xml', scene))
    for hand in HANDS:
        arm = mujoco.MjSpec.from_file(str(FR3V2_XML))
        for mesh in arm.meshes:
            mesh.file = str((FR3V2_XML.parent / 'assets' / mesh.file).resolve())
        arm.meshdir = ''
        for k in list(arm.keys):
            arm.delete(k)
        for i, (lo, hi) in enumerate(FR3V2_LIMITS, 1):
            arm.joint(f'fr3v2_joint{i}').range = [lo, hi]
        p, R = place.arm_base(hand)
        q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
        frame = spec.worldbody.add_frame(pos=p, quat=q)
        frame.attach_body(arm.body('base'), f'{hand}_', '')
        spec.body(f'{hand}_fr3v2_link8').add_site(name=f'{hand}_tcp', pos=[0, 0, tcp])
    pc, Rc = place.head_camera()
    cam = spec.worldbody.add_camera(name='head', pos=pc, fovy=ZED_MINI_FOV_DEG[1])
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = list(np.r_[-Rc[:, 1], Rc[:, 2]])  # image right = -y, image up = z, looking along +x
    return spec.compile()


class DuoIK:
    def __init__(self, m: mujoco.MjModel, margin=0.03):
        self.m, self.d = m, mujoco.MjData(m)
        self.arms = {}
        for h in HANDS:
            js = [m.joint(f'{h}_fr3v2_joint{i}') for i in range(1, 8)]
            self.arms[h] = dict(qadr=np.array([j.qposadr[0] for j in js]),
                                lo=np.array([m.jnt_range[j.id, 0] for j in js]) + margin,
                                hi=np.array([m.jnt_range[j.id, 1] for j in js]) - margin,
                                tcp=m.site(f'{h}_tcp').id,
                                geoms=np.array([g for g in range(m.ngeom)
                                                if m.body(m.geom_bodyid[g]).name.startswith(f'{h}_')]))
        robot = np.zeros(m.ngeom, bool)
        for a in self.arms.values():
            robot[a['geoms']] = True
        self.robot = robot

    def solve(self, hand, pos, rot, seeds):
        a, m, d = self.arms[hand], self.m, self.d
        best = None

        def res(x):
            d.qpos[a['qadr']] = x
            mujoco.mj_kinematics(m, d)
            pe = d.site_xpos[a['tcp']] - pos
            re = Rotation.from_matrix(rot @ d.site_xmat[a['tcp']].reshape(3, 3).T).as_rotvec()
            return np.r_[pe, 0.1 * re]

        for s in seeds:
            sol = least_squares(res, np.clip(s, a['lo'], a['hi']), bounds=(a['lo'], a['hi']), max_nfev=200,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            r = res(sol.x)
            pe, re = float(np.linalg.norm(r[:3])), float(np.degrees(np.linalg.norm(r[3:]) / 0.1))
            if best is None or pe + re / 1000 < best[1] + best[2] / 1000:
                best = (sol.x.copy(), pe, re)
            if pe < 0.005 and re < 3.0:
                break
        return best

    def penetration(self, hand, q, exclude_bodies=()):
        """Deepest contact of the arm's collision geoms with anything that is not the robot (after setting q)."""
        a, m, d = self.arms[hand], self.m, self.d
        d.qpos[a['qadr']] = q
        mujoco.mj_forward(m, d)
        worst = 0.0
        ex = {m.body(b).id for b in exclude_bodies}
        mine = np.zeros(m.ngeom, bool)
        mine[a['geoms']] = True
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if mine[g1] == mine[g2] or (self.robot[g1] and self.robot[g2]):
                continue
            other = g2 if mine[g1] else g1
            if m.geom_bodyid[other] in ex:
                continue
            worst = max(worst, -c.dist)
        return worst


# ---- full scene for replay and rendering (2026-11-03, calibration trajectories) -----------------------------------
ROBOTIQ_XML = resolve_mujoco_menagerie() / 'robotiq_2f85/2f85.xml'
# Gripper mounting: GRIPPER_YAW / FACTORY_* above, per DuoPlacement. Wrist camera (RealSense D405, 1280x720, ~87 x 58
# deg): NOMINAL mount, 7.5 cm off the gripper axis on the gripper's +x side, 3 cm in front of the flange, aimed at the
# pinch point; the real holder pose is what the photographs are for.
WRIST_CAM_IN_GRIPPER = np.array([0.075, 0.0, 0.03])
D405_FOVY_DEG = 58.0
MOUNT_MESHES = (('fr3_duo_mount', 'meshes/accessories/fr3_duo_mount_v0_3/collision/fr3_duo_mount.stl', np.zeros(3)),
                ('fr3_duo_cover', 'meshes/accessories/fr3_duo_mount_v0_3/collision/fr3_duo_cover.stl',
                 np.array([0.0, 0.0, 0.068])),
                ('franka_head', 'meshes/accessories/franka_head_v0_2/collision/franka_head.stl', HEAD))


def _arm_spec():
    arm = mujoco.MjSpec.from_file(str(FR3V2_XML))
    for mesh in arm.meshes:
        mesh.file = str((FR3V2_XML.parent / 'assets' / mesh.file).resolve())
    arm.meshdir = ''
    for k in list(arm.keys):
        arm.delete(k)
    for i, (lo, hi) in enumerate(FR3V2_LIMITS, 1):
        arm.joint(f'fr3v2_joint{i}').range = [lo, hi]
    for b in arm.bodies:
        b.gravcomp = 1.0  # the FR3 controller compensates gravity
    return arm


def _gripper_spec():
    g = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))
    for mesh in g.meshes:
        mesh.file = str((ROBOTIQ_XML.parent / 'assets' / mesh.file).resolve())
    g.meshdir = ''
    for b in g.bodies:
        b.gravcomp = 1.0
    return g


def build_full(case_dir: Path, scene: dict, place: DuoPlacement, physics_dt=1.0 / 1200) -> mujoco.MjSpec:
    """Table, task objects, FR3 Duo mount + cover + head (collision meshes, static), two FR3v2 arms with position
    servos and a Robotiq 2F-85 each, TCP site = 2F-85 pinch, cameras head (ZED Mini HD720) and <hand>_wrist (D405,
    nominal mount)."""
    spec = mujoco.MjSpec.from_string(environment_xml(case_dir / 'source_reference/scene.xml', scene))
    spec.modelname = 'rssr_fr3_duo'
    p0, R0 = place.mount_pose()
    q0 = Rotation.from_matrix(R0).as_quat()[[3, 0, 1, 2]]
    mount = spec.worldbody.add_body(name='fr3_duo_mount_body', pos=p0, quat=q0)
    for name, rel, off in MOUNT_MESHES:
        spec.add_mesh(name=name, file=str(DESCRIPTION / rel))
        mount.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=off,
                       rgba=[0.92, 0.92, 0.9, 1.0] if name != 'fr3_duo_mount' else [0.15, 0.15, 0.17, 1.0])
    for hand in HANDS:
        p, R = place.arm_base(hand)
        frame = spec.worldbody.add_frame(pos=p, quat=Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]])
        spec.attach(_arm_spec(), prefix=f'{hand}_', frame=frame)
        flange = spec.body(f'{hand}_fr3v2_link8')
        gq = Rotation.from_euler('z', place.gripper_yaw).as_quat()[[3, 0, 1, 2]]
        gframe = flange.add_frame(pos=[0.0, 0.0, place.gripper_offset], quat=gq)
        spec.attach(_gripper_spec(), prefix=f'{hand}_g_', frame=gframe)
        if place.coupling:
            r, h = place.coupling
            flange.add_geom(name=f'{hand}_coupling', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0],
                            pos=[0, 0, h / 2], contype=1, conaffinity=1, group=3, rgba=[0.1, 0.1, 0.1, 1])
            flange.add_geom(name=f'{hand}_coupling_visual', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0],
                            pos=[0, 0, h / 2], contype=0, conaffinity=0, group=2, rgba=[0.1, 0.1, 0.1, 1])
        base = spec.body(f'{hand}_g_base')
        pinch = np.array([0.0, 0.0, 0.145])
        base.add_site(name=f'{hand}_tcp', pos=pinch)
        cam_pos = WRIST_CAM_IN_GRIPPER - np.array([0.0, 0.0, 0.0038])  # base frame is 3.8 mm past base_mount
        z = cam_pos - pinch  # camera looks along -z: from the camera toward the pinch point
        z /= np.linalg.norm(z)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        cam = base.add_camera(name=f'{hand}_wrist', pos=cam_pos, fovy=D405_FOVY_DEG)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = list(np.r_[x, np.cross(z, x)])
    pc, Rc = place.head_camera()
    cam = spec.worldbody.add_camera(name='head', pos=pc, fovy=ZED_MINI_FOV_DEG[1])
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = list(np.r_[-Rc[:, 1], Rc[:, 2]])
    spec.option.timestep = physics_dt
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    return spec
