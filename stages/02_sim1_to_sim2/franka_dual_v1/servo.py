"""Execute 30 Hz absolute joint/gripper targets on the dual-arm Franka scene and record every frame.

Targets are linearly interpolated between frames at the physics step (1/1200 s; position actuators
with the Menagerie gains, gravity compensated).

Two clocks are recorded:
  * 30 Hz frames (data clock): the 32-value joint state, the 16-value action, full qpos/qvel, TCP
    poses, free-object poses, the drawer slide and the left/right arm clearance;
  * 120 Hz contract trace in the schema of the frozen S1 evaluator
    (s1_short_refine_v2/rssr_augment/vendor/runtime/rssr_task_contracts.evaluate): object poses for
    every source-scene object, contacts[obj, (left fingers, right fingers, table)], object-object
    contacts, finger_contact_sides[obj, hand, (left finger, right finger)], gripper opening and the
    articulation open fraction.
Every physics step is audited for robot-table penetration, joint limits (all limited joints) and
solver warnings. `replay` re-executes the actions in a fresh MjData and must match bit for bit.
"""
from __future__ import annotations

import hashlib

import mujoco
import numpy as np

from .layout import HANDS
from .robot import DualFranka

FPS = 30
TRACE_HZ = 120
PENETRATION_TOL_M = 0.003
JOINT_LIMIT_TOL = 0.002  # rad or m
ARM_CLEARANCE_M = 0.020
FINGER_COUPLING_TOL_M = 0.002


class Executor:
    def __init__(self, m: mujoco.MjModel, object_ids: list[str], robot=None):
        self.m = m
        self.robot = DualFranka(m) if robot is None else robot
        self.object_ids = list(object_ids)
        self.obj_bodies = np.array([m.body(o).id for o in self.object_ids])
        self.geom_obj = np.full(m.ngeom, -1)
        for i, b in enumerate(self.obj_bodies):
            self.geom_obj[m.geom_bodyid == b] = i
        self.free_bodies = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt)
                            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.slides = [m.joint(j).name for j in range(m.njnt)
                       if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and not m.joint(j).name.startswith(HANDS)]
        self.robot_geom = np.zeros(m.ngeom, bool)
        self.finger_side = np.full((m.ngeom, 2), -1)  # (hand index, finger index)
        for k, (hand, a) in enumerate(self.robot.arms.items()):
            self.robot_geom[a.geoms] = True
            for f, fbs in enumerate(self.robot.finger_bodies(hand)):
                for fb in fbs:
                    self.finger_side[m.geom_bodyid == fb] = (k, f)
        self.table_geom = np.array([m.body(m.geom_bodyid[g]).name == 'worktable' for g in range(m.ngeom)])
        coll = lambda a: np.array([g for g in a.geoms if m.geom_contype[g] or m.geom_conaffinity[g]])
        self.coll = {h: coll(a) for h, a in self.robot.arms.items()}
        # left/right pairs; geoms welded to the world on both sides (arm bases on a shared mount) never move and are
        # left out
        fixed = m.body_weldid[m.geom_bodyid] == 0
        self.clear_pairs = [(int(ga), int(gb)) for ga in self.coll['left'] for gb in self.coll['right']
                            if not (fixed[ga] and fixed[gb])]
        skip = set(getattr(self.robot, 'LIMIT_AUDIT_SKIP', ()))
        lim = [j for j in range(m.njnt) if m.jnt_limited[j] and m.joint(j).name not in skip and
               m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.lim_qadr = m.jnt_qposadr[lim]
        self.lim_names = [m.joint(j).name for j in lim]
        self.lim_lo, self.lim_hi = m.jnt_range[lim, 0], m.jnt_range[lim, 1]
        self.steps_per_frame = int(round(1.0 / (FPS * m.opt.timestep)))
        self.trace_stride = int(round(1.0 / (TRACE_HZ * m.opt.timestep)))
        if (abs(self.steps_per_frame * m.opt.timestep * FPS - 1) > 1e-6 or
                self.steps_per_frame % self.trace_stride):
            raise ValueError('physics timestep must divide the 30 Hz and 120 Hz periods')

    def reset(self, key='home'):
        d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, d, self.m.key(key).id)
        mujoco.mj_forward(self.m, d)
        return d

    # ---- per-frame measurements --------------------------------------------------------------
    def arm_clearance(self, d) -> float:
        if hasattr(self.robot, 'arm_clearance'):
            return self.robot.arm_clearance(d)
        best = np.inf
        fromto = np.zeros(6)
        for ga, gb in self.clear_pairs:
            best = min(best, mujoco.mj_geomDistance(self.m, d, ga, gb, ARM_CLEARANCE_M * 5, fromto))
        return float(best)

    def snapshot(self, d) -> dict:
        s = self.robot.state(d)
        return dict(
            time_s=float(d.time), state=self.robot.state_vector(s), qpos=d.qpos.copy(), qvel=d.qvel.copy(),
            ctrl=d.ctrl.copy(),
            tcp=np.r_[s['left_tcp_pos'], s['left_tcp_quat'], s['right_tcp_pos'], s['right_tcp_quat']],
            objects=np.array([np.r_[d.xpos[self.m.body(b).id], d.xquat[self.m.body(b).id]]
                              for b in self.free_bodies]).reshape(len(self.free_bodies), 7),
            slides=np.array([d.qpos[self.m.joint(j).qposadr[0]] for j in self.slides]),
            arm_clearance_m=self.arm_clearance(d))

    def trace_row(self, d) -> dict:
        n = len(self.object_ids)
        hit = np.zeros((n, 3), np.int8)
        pairs = np.zeros((n, n), np.int8)
        fingers = np.zeros((n, 2, 2), np.int8)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0:
                continue
            g = (int(c.geom1), int(c.geom2))
            oa, ob = self.geom_obj[g[0]], self.geom_obj[g[1]]
            if oa >= 0 and ob >= 0 and oa != ob:
                pairs[oa, ob] = pairs[ob, oa] = 1
            for gi, go in ((g[0], g[1]), (g[1], g[0])):
                oi = self.geom_obj[gi]
                if oi < 0:
                    continue
                if self.table_geom[go]:
                    hit[oi, 2] = 1
                k, f = self.finger_side[go]
                if k >= 0:
                    hit[oi, k] = 1
                    fingers[oi, k, f] = 1
        art = {j: float(d.qpos[self.m.joint(j).qposadr[0]] / self.m.jnt_range[self.m.joint(j).id, 1])
               for j in self.slides}
        return dict(time_s=float(d.time), object_position=d.xpos[self.obj_bodies].copy(),
                    object_quaternion=d.xquat[self.obj_bodies].copy(), contacts=hit, object_contacts=pairs,
                    finger_contact_sides=fingers,
                    gripper_opening=np.array([self.robot.width(d, h) for h in HANDS]), articulation=art)

    def step_audit(self, d, audit):
        pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (self.robot_geom[g1] and self.table_geom[g2]) or (self.robot_geom[g2] and self.table_geom[g1]):
                pen = max(pen, -float(c.dist))
        q = d.qpos[self.lim_qadr]
        per = np.maximum(self.lim_lo - q, q - self.lim_hi)
        excess = float(per.max())
        if excess > audit['joint_limit_excess']:
            j = int(np.argmax(per))
            audit['joint_limit_worst'] = dict(joint=self.lim_names[j], t=round(float(d.time), 3),
                                              excess=round(excess, 5), q=round(float(q[j]), 5),
                                              range=[float(self.lim_lo[j]), float(self.lim_hi[j])])
        coupling = self.robot.finger_coupling(d)
        audit['robot_table_penetration_m'] = max(audit['robot_table_penetration_m'], pen)
        audit['joint_limit_excess'] = max(audit['joint_limit_excess'], excess)
        audit['finger_coupling_m'] = max(audit['finger_coupling_m'], coupling)
        audit['solver_warnings'] = int(sum(int(w.number) for w in d.warning))

    # ---- execution ---------------------------------------------------------------------------
    def run(self, actions: np.ndarray, d=None, settle_s: float = 0.0) -> dict:
        """actions: (T, 16). Frame 0 is the reset state; frame k+1 is after executing actions[k]."""
        actions = np.asarray(actions, float)
        return self.run_online(len(actions), lambda k, d: actions[k], d, settle_s)

    def run_online(self, n_frames: int, policy, d=None, settle_s: float = 0.0) -> dict:
        """policy(k, d) -> action (16) for frame k, given the live MjData (read only). After the last frame
        the last action is held for settle_s: the hold enters the 120 Hz contract trace and the audit, not
        the 30 Hz frames (as sim_dart's SETTLE_S)."""
        d = self.reset() if d is None else d
        prev = self.robot.action_from_state(d)
        frames = [self.snapshot(d)]
        trace = [self.trace_row(d)]
        audit = dict(robot_table_penetration_m=0.0, joint_limit_excess=-np.inf, finger_coupling_m=0.0,
                     solver_warnings=0)
        n = self.steps_per_frame
        actions = []
        for k in range(n_frames):
            a = np.asarray(policy(k, d), float)
            actions.append(a)
            for s in range(1, n + 1):
                self.robot.apply_action(d, prev + (a - prev) * (s / n))
                mujoco.mj_step(self.m, d)
                self.step_audit(d, audit)
                if s % self.trace_stride == 0:
                    trace.append(self.trace_row(d))
            prev = a
            frames.append(self.snapshot(d))
        for s in range(1, int(round(settle_s / self.m.opt.timestep)) + 1):
            self.robot.apply_action(d, prev)
            mujoco.mj_step(self.m, d)
            self.step_audit(d, audit)
            if s % self.trace_stride == 0:
                trace.append(self.trace_row(d))
        roll = {k: np.array([f[k] for f in frames]) for k in frames[0]}
        roll['action'] = np.array(actions).reshape(n_frames, -1)
        roll['free_bodies'] = np.array(self.free_bodies)
        roll['slide_joints'] = np.array(self.slides)
        tr = {k: np.array([r[k] for r in trace]) for k in trace[0] if k != 'articulation'}
        tr['articulation_open_fraction'] = {j: np.array([r['articulation'][j] for r in trace]) for j in self.slides}
        return dict(frames=roll, trace=tr, audit=audit)

    def hard_constraints(self, result: dict) -> dict:
        """Booleans for the frozen contract's hard-constraint names, plus the Franka arm clearance."""
        a = result['audit']
        return {
            'robot_table_penetration_le_3mm': bool(a['robot_table_penetration_m'] <= PENETRATION_TOL_M),
            'limited_joints_within_2mrad_or_2mm': bool(a['joint_limit_excess'] <= JOINT_LIMIT_TOL),
            'solver_no_warnings': bool(a['solver_warnings'] == 0),
            # the Franka has no waist: nothing to couple
            'waist_coupling_within_1mrad': True,
            'gripper_coupling_within_2mrad': bool(a['finger_coupling_m'] <= FINGER_COUPLING_TOL_M),
        }

    def arm_clearance_ok(self, result: dict) -> bool:
        return bool(result['frames']['arm_clearance_m'].min() >= ARM_CLEARANCE_M)


def digest(result: dict) -> str:
    h = hashlib.sha256()
    f = result['frames']
    for k in ('qpos', 'qvel', 'state', 'objects'):
        h.update(np.ascontiguousarray(f[k]).tobytes())
    for k in ('object_position', 'object_quaternion', 'contacts'):
        h.update(np.ascontiguousarray(result['trace'][k]).tobytes())
    return h.hexdigest()


def replay(m: mujoco.MjModel, object_ids, actions: np.ndarray, expected: str, settle_s: float = 0.0,
           robot_cls=None) -> dict:
    got = digest(Executor(m, object_ids, None if robot_cls is None else robot_cls(m)).run(actions, settle_s=settle_s))
    return dict(expected=expected, replay=got, bit_exact=bool(got == expected))
