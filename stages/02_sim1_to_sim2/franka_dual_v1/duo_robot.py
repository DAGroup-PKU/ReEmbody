"""FR3 Duo binding (two FR3v2 arms + Robotiq 2F-85, fr3_duo.build_full) with the DualFranka interface (robot.py), so
the servo executor, its contract trace, the audit and the bit-exact replay (servo.py) run unchanged.

Per arm: 7 arm joints and the 2F-85 as two values, the left and right driver joint angles (rad, 0 open .. 0.8
closed). Gripper width = distance between the inner faces of the two finger pads along the TCP closing axis (TCP
frame: z = approach, y = closing; the TCP site is the 2F-85 pinch point). Action per arm: 7 absolute joint targets +
1 gripper width target (m), mapped to the fingers_actuator control (0 open .. 255 closed) through WIDTH_TABLE, the
steady-state width of the free gripper at each control value in this model (`python -m franka_dual_v1.duo_robot
--calibrate` writes duo_gripper_table.json). A width below the table's closed width commands 255: the tendon force
limit (5 N in the Menagerie model) then sets the squeeze, not the position error.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from .robot import Arm, DualFranka

HANDS = ('left', 'right')
SIDE_PARTS = ('driver', 'coupler', 'spring_link', 'follower', 'pad', 'silicone_pad')
PAD_HALF_THICKNESS = 0.004  # pad_box geoms, half size along the closing axis
TABLE_PATH = Path(__file__).with_name('duo_gripper_table.json')
CLEAR_MARGIN = 0.10  # left/right clearance is reported up to this distance
CTRL_MAX = 255.0


def _load_table():
    if not TABLE_PATH.exists():
        return None
    t = json.loads(TABLE_PATH.read_text())
    w, c = np.array(t['width_m']), np.array(t['ctrl'])
    order = np.argsort(w)
    return w[order], c[order]


WIDTH_TABLE = _load_table()


class DualFR3(DualFranka):
    # The Menagerie 2F-85 coupler joints rest on their upper limit (0 rad): the limit is the linkage's mechanical stop
    # that keeps the fingers parallel (0.1 mrad beyond it at rest at every opening, up to ~9 mrad under arm
    # acceleration, measured 2026-11-03). They are left out of the joint-limit audit; every other joint is audited.
    LIMIT_AUDIT_SKIP = tuple(f'{h}_g_{s}_coupler_joint' for h in HANDS for s in ('left', 'right'))

    def __init__(self, m: mujoco.MjModel, clear_margin: float = CLEAR_MARGIN):
        self.m = m
        self.clear_margin = clear_margin  # left/right clearance is reported up to this distance
        self.arms: dict[str, Arm] = {}
        body_names = [m.body(b).name for b in range(m.nbody)]
        self._sides = {}
        self._pads = {}
        for hand in HANDS:
            joints = [m.joint(f'{hand}_fr3v2_joint{i}') for i in range(1, 8)]
            drivers = [m.joint(f'{hand}_g_{s}_driver_joint') for s in ('left', 'right')]
            bodies = np.array([b for b, n in enumerate(body_names) if n.startswith(f'{hand}_')])
            sides = [[m.body(f'{hand}_g_{s}_{p}').id for p in SIDE_PARTS] for s in ('left', 'right')]
            self._sides[hand] = sides
            self._pads[hand] = [m.geom(f'{hand}_g_{s}_pad1').id for s in ('left', 'right')]
            geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies])
            fb = set(sides[0] + sides[1])
            self.arms[hand] = Arm(
                hand=hand,
                qadr=np.array([j.qposadr[0] for j in joints]),
                dadr=np.array([j.dofadr[0] for j in joints]),
                lo=np.array([m.jnt_range[j.id, 0] for j in joints]),
                hi=np.array([m.jnt_range[j.id, 1] for j in joints]),
                act=np.array([m.actuator(f'{hand}_fr3v2_joint{i}').id for i in range(1, 8)]),
                grip_act=m.actuator(f'{hand}_g_fingers_actuator').id,
                finger_qadr=np.array([j.qposadr[0] for j in drivers]),
                tcp=m.site(f'{hand}_tcp').id,
                hand_body=m.body(f'{hand}_g_base').id,
                bodies=bodies,
                finger_geoms=np.array([g for g in geoms if m.geom_bodyid[g] in fb]),
                geoms=geoms,
                wrist_cam=m.camera(f'{hand}_wrist').id)
        self._work = mujoco.MjData(m)
        self._clear_model()

    # ---- left/right clearance ------------------------------------------------------------------
    def _clear_model(self):
        """Copy of the model in which only the robot's collision geoms collide, with a CLEAR_MARGIN contact margin.
        mj_geomDistance (servo.Executor's generic path) returns false zeros for some box-mesh pairs here (2F-85 pad vs
        FR3v2 link6, witness points 9 cm apart), so the clearance is read from the engine's narrowphase instead."""
        m = self.m
        cm = copy.deepcopy(m)
        side = np.full(m.ngeom, -1)
        for k, a in enumerate(self.arms.values()):
            side[a.geoms] = k
        coll = (m.geom_contype | m.geom_conaffinity) > 0
        robot = (side >= 0) & coll
        cm.geom_contype[~robot] = 0
        cm.geom_conaffinity[~robot] = 0
        cm.geom_margin[robot] = self.clear_margin
        self._cm, self._cd = cm, mujoco.MjData(cm)
        self._side = side
        self._fixed = m.body_weldid[m.geom_bodyid] == 0  # arm bases on the shared mount never move

    def arm_clearance(self, d) -> float:
        cm, cd = self._cm, self._cd
        cd.qpos[:] = d.qpos
        mujoco.mj_kinematics(cm, cd)
        mujoco.mj_collision(cm, cd)
        best = self.clear_margin
        side, fixed = self._side, self._fixed
        for i in range(cd.ncon):
            c = cd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if side[g1] >= 0 and side[g2] >= 0 and side[g1] != side[g2] and not (fixed[g1] and fixed[g2]):
                best = min(best, float(c.dist))
        return best

    # ---- gripper ---------------------------------------------------------------------------
    def finger_bodies(self, hand):
        return self._sides[hand]

    def finger_coupling(self, d) -> float:
        return max(abs(float(np.diff(d.qpos[a.finger_qadr])[0])) for a in self.arms.values())

    def width(self, d, hand):
        a = self.arms[hand]
        y = d.site_xmat[a.tcp].reshape(3, 3)[:, 1]
        gl, gr = self._pads[hand]
        return float(abs((d.geom_xpos[gl] - d.geom_xpos[gr]) @ y) - 2 * PAD_HALF_THICKNESS)

    def state(self, d) -> dict:
        out = super().state(d)
        for hand in HANDS:
            out[f'{hand}_gripper_width'] = np.array([self.width(d, hand)])
        return out

    @staticmethod
    def ctrl_for_width(width: float) -> float:
        if WIDTH_TABLE is None:
            raise RuntimeError(f'{TABLE_PATH} missing: run python -m franka_dual_v1.duo_robot --calibrate')
        w, c = WIDTH_TABLE
        return float(np.interp(width, w, c))

    def apply_action(self, d, action: np.ndarray):
        """action: 16 = per arm 7 joint targets + gripper width (m), left then right."""
        for k, hand in enumerate(HANDS):
            a = self.arms[hand]
            seg = action[8 * k: 8 * k + 8]
            d.ctrl[a.act] = seg[:7]
            d.ctrl[a.grip_act] = self.ctrl_for_width(seg[7])


def calibrate(scene_xml: Path, hand='left', settle_s=0.6, step=5):
    """Steady-state pad width of the free gripper for control values 0, step, ..., 255 (arms held at the home key)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    r = DualFR3.__new__(DualFR3)
    DualFR3.__init__(r, m)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
    rows = []
    for c in list(range(0, 256, step)) + [255]:
        d.ctrl[r.arms[hand].grip_act] = c
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        rows.append((float(c), r.width(d, hand), float(d.qpos[r.arms[hand].finger_qadr].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', type=Path, required=True, help='a duo_scene scene.xml with a home key')
    a = ap.parse_args()
    rows = calibrate(a.calibrate)
    TABLE_PATH.write_text(json.dumps(dict(
        schema='rssr.fr3_duo.gripper_table.v1', scene=str(a.calibrate),
        note='free Robotiq 2F-85 (Menagerie), steady state after 0.6 s per control value, from open',
        ctrl=[r[0] for r in rows], width_m=[round(r[1], 6) for r in rows],
        driver_rad=[round(r[2], 6) for r in rows]), indent=1) + '\n')
    for r in rows[::5] + [rows[-1]]:
        print(f'ctrl {r[0]:5.0f}  width {r[1] * 1000:6.2f} mm  driver {r[2]:.3f}')


if __name__ == '__main__':
    main()
