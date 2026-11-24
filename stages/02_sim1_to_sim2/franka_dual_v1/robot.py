"""Joint/actuator binding, kinematics and bounded IK for the dual-arm Franka scene.

State per arm: 7 arm joints + 2 finger joints (qpos and qvel); the gripper width is the sum of the
two finger joints (0 .. 0.08 m). Action per arm: 7 absolute joint-position targets + 1 gripper
width target, ordered left then right (16 values).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .layout import HANDS

GRIPPER_MAX_WIDTH = 0.08
GRIPPER_CTRL_MAX = 255.0
LIMIT_MARGIN = 0.03  # rad kept from every arm joint limit by IK


@dataclass
class Arm:
    hand: str
    qadr: np.ndarray
    dadr: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    act: np.ndarray
    grip_act: int
    finger_qadr: np.ndarray
    tcp: int
    hand_body: int
    bodies: np.ndarray
    finger_geoms: np.ndarray
    geoms: np.ndarray
    wrist_cam: int


class DualFranka:
    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.arms: dict[str, Arm] = {}
        body_names = [m.body(b).name for b in range(m.nbody)]
        for hand in HANDS:
            joints = [m.joint(f'{hand}_joint{i}') for i in range(1, 8)]
            fingers = [m.joint(f'{hand}_finger_joint{i}') for i in (1, 2)]
            bodies = np.array([b for b, n in enumerate(body_names) if n.startswith(f'{hand}_')])
            fbodies = [m.body(f'{hand}_left_finger').id, m.body(f'{hand}_right_finger').id]
            geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies])
            self.arms[hand] = Arm(
                hand=hand,
                qadr=np.array([j.qposadr[0] for j in joints]),
                dadr=np.array([j.dofadr[0] for j in joints]),
                lo=np.array([m.jnt_range[j.id, 0] for j in joints]),
                hi=np.array([m.jnt_range[j.id, 1] for j in joints]),
                act=np.array([m.actuator(f'{hand}_actuator{i}').id for i in range(1, 8)]),
                grip_act=m.actuator(f'{hand}_actuator8').id,
                finger_qadr=np.array([j.qposadr[0] for j in fingers]),
                tcp=m.site(f'{hand}_tcp').id,
                hand_body=m.body(f'{hand}_hand').id,
                bodies=bodies,
                finger_geoms=np.array([g for g in geoms if m.geom_bodyid[g] in fbodies]),
                geoms=geoms,
                wrist_cam=m.camera(f'{hand}_wrist').id)
        self._work = mujoco.MjData(m)

    # ---- state / action -------------------------------------------------------------------
    def finger_bodies(self, hand):
        """([left finger body ids], [right finger body ids]) for the contact trace."""
        return [self.m.body(f'{hand}_left_finger').id], [self.m.body(f'{hand}_right_finger').id]

    def finger_coupling(self, d) -> float:
        return max(abs(float(np.diff(d.qpos[a.finger_qadr])[0])) for a in self.arms.values())

    def arm_q(self, d, hand):
        return d.qpos[self.arms[hand].qadr].copy()

    def width(self, d, hand):
        return float(d.qpos[self.arms[hand].finger_qadr].sum())

    def state(self, d) -> dict:
        out = {}
        for hand, a in self.arms.items():
            out[f'{hand}_arm_qpos'] = d.qpos[a.qadr].copy()
            out[f'{hand}_arm_qvel'] = d.qvel[a.dadr].copy()
            out[f'{hand}_finger_qpos'] = d.qpos[a.finger_qadr].copy()
            out[f'{hand}_gripper_width'] = np.array([d.qpos[a.finger_qadr].sum()])
            out[f'{hand}_tcp_pos'] = d.site_xpos[a.tcp].copy()
            out[f'{hand}_tcp_quat'] = _mat2quat(d.site_xmat[a.tcp])
        return out

    @staticmethod
    def state_vector(s: dict) -> np.ndarray:
        """32 values: per arm 7 q + 2 finger q + 7 qd (left then right)."""
        return np.concatenate([np.r_[s[f'{h}_arm_qpos'], s[f'{h}_finger_qpos'], s[f'{h}_arm_qvel']]
                               for h in HANDS])

    def apply_action(self, d, action: np.ndarray):
        """action: 16 = per arm 7 joint targets + gripper width (m), left then right."""
        for k, hand in enumerate(HANDS):
            a = self.arms[hand]
            seg = action[8 * k: 8 * k + 8]
            d.ctrl[a.act] = seg[:7]
            d.ctrl[a.grip_act] = np.clip(seg[7] / GRIPPER_MAX_WIDTH, 0, 1) * GRIPPER_CTRL_MAX

    def action_from_state(self, d) -> np.ndarray:
        return np.concatenate([np.r_[self.arm_q(d, h), self.width(d, h)] for h in HANDS])

    # ---- kinematics -----------------------------------------------------------------------
    def fk(self, hand, q, base_qpos):
        w = self._work
        w.qpos[:] = base_qpos
        w.qpos[self.arms[hand].qadr] = q
        mujoco.mj_kinematics(self.m, w)
        a = self.arms[hand]
        return w.site_xpos[a.tcp].copy(), w.site_xmat[a.tcp].reshape(3, 3).copy()

    def ik(self, hand, pos, rot, q_init, base_qpos, q_ref=None, rot_weight=0.1, reg=3e-4,
           margin=LIMIT_MARGIN, posture=None):
        """Bounded IK for the TCP pose. Returns (q, pos_err_m, rot_err_deg).
        posture: {joint index: (value, weight)} extra residuals that pin the redundant (elbow swivel) direction."""
        a = self.arms[hand]
        lo, hi = a.lo + margin, a.hi - margin
        q_ref = np.asarray(q_init if q_ref is None else q_ref, float)
        w = self._work
        w.qpos[:] = base_qpos

        def residual(x):
            w.qpos[a.qadr] = x
            mujoco.mj_kinematics(self.m, w)
            pe = w.site_xpos[a.tcp] - pos
            re = Rotation.from_matrix(rot @ w.site_xmat[a.tcp].reshape(3, 3).T).as_rotvec()
            extra = [w * (x[j] - v) for j, (v, w) in (posture or {}).items()]
            return np.r_[pe, rot_weight * re, reg * (x - q_ref), extra]

        x0 = np.clip(q_init, lo, hi)
        sol = least_squares(residual, x0, bounds=(lo, hi), max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        residual(sol.x)
        perr = float(np.linalg.norm(w.site_xpos[a.tcp] - pos))
        rerr = float(np.degrees(Rotation.from_matrix(rot @ w.site_xmat[a.tcp].reshape(3, 3).T).magnitude()))
        return sol.x, perr, rerr


    def ik_local(self, hand, pos, rot, q_init, base_qpos, q_post, rot_weight=0.3, damping=0.02, null_gain=0.1,
                 iters=12, max_iter_step=0.05, margin=LIMIT_MARGIN, tol_m=1e-4, tol_rad=1e-3, null_mask=None,
                 limit_zone=0.0, limit_gain=0.0):
        """Local damped-least-squares IK from q_init (continuous: never jumps to another solution branch).
        Each iteration: dq = J^T (J J^T + damping^2 I)^-1 e + null_gain * (I - J^+ J)(q_post - q), clipped to
        max_iter_step and to the joint limits minus `margin`. Returns (q, pos_err_m, rot_err_deg)."""
        a = self.arms[hand]
        m, w = self.m, self._work
        lo, hi = a.lo + margin, a.hi - margin
        q = np.clip(np.asarray(q_init, float), lo, hi)
        q_post = np.asarray(q_post, float)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        w.qpos[:] = base_qpos
        for _ in range(iters):
            w.qpos[a.qadr] = q
            mujoco.mj_kinematics(m, w)
            mujoco.mj_comPos(m, w)
            ep = pos - w.site_xpos[a.tcp]
            er = Rotation.from_matrix(rot @ w.site_xmat[a.tcp].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(ep) < tol_m and np.linalg.norm(er) < tol_rad:
                break
            mujoco.mj_jacSite(m, w, jp, jr, a.tcp)
            J = np.r_[jp[:, a.dadr], rot_weight * jr[:, a.dadr]]
            e = np.r_[ep, rot_weight * er]
            JJt = J @ J.T + damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))
            bias = (q_post - q) if null_mask is None else np.asarray(null_mask, float) * (q_post - q)
            if limit_gain:
                # push joints within limit_zone of a limit back toward the inside (null space only)
                lo_d, hi_d = q - lo, hi - q
                bias = bias + limit_gain * (np.clip(limit_zone - lo_d, 0, None) - np.clip(limit_zone - hi_d, 0, None))
            dq += null_gain * (np.eye(7) - Jpinv @ J) @ bias
            n = np.abs(dq).max()
            if n > max_iter_step:
                dq *= max_iter_step / n
            q = np.clip(q + dq, lo, hi)
        w.qpos[a.qadr] = q
        mujoco.mj_kinematics(m, w)
        perr = float(np.linalg.norm(w.site_xpos[a.tcp] - pos))
        rerr = float(np.degrees(Rotation.from_matrix(rot @ w.site_xmat[a.tcp].reshape(3, 3).T).magnitude()))
        return q, perr, rerr


def _mat2quat(xmat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, float).reshape(9))
    return q


def grasp_rotation(yaw: float, lean_deg: float = 0.0, facing=np.array([0.0, -1.0, 0.0])) -> np.ndarray:
    """TCP rotation for a top-down grasp with the fingers closing across `yaw`, tilted by `lean_deg`.

    Hand frame: z = approach (fingers point along +z), y = finger opening axis. With lean_deg = 0 the
    approach is straight down. A positive lean tips the approach toward `facing` (horizontal), i.e. the
    hand comes in from the robot side and its wrist camera (on hand +x) looks further ahead.
    """
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.cross(y, z)
    R = np.c_[x, y, z]
    if lean_deg:
        f = np.asarray(facing, float)
        axis = np.cross(f, [0.0, 0.0, 1.0])
        R = Rotation.from_rotvec(np.radians(lean_deg) * axis / np.linalg.norm(axis)).as_matrix() @ R
    return R
