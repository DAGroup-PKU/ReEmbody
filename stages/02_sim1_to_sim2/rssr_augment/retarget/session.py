"""Pause a native rollout at observed events without resetting MuJoCo state.

One session owns one physics subprocess.  Only robot joint commands pass through
``JointServo.apply``; callers receive copied measurements and never receive the
mutable MuJoCo data object.  The native v80 rollout still writes its unchanged
task report and every-step hard-constraint audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Callable

import mujoco
import numpy as np

from .immutable_hard_observation import HardSnapshotter, HardAccumulator
from .hard_trace import save_hard_trace

# Robot-specific adapter declaration. Event compilation consumes this tuple;
# another robot adapter may declare one or more different IDs.
ADAPTER_MANIPULATORS = ('left', 'right')


def robot_command_indices(model):
    """Partition command coordinates by named robot joints, never by column."""
    # This S1 campaign has a stationary chassis. Reject a mobile model at the
    # physics boundary, including a model whose base actuator was left idle.
    chassis = model.body('chassis_base')
    if int(model.body_parentid[chassis.id]) != 0 or int(model.body_jntnum[chassis.id]) != 0:
        raise ValueError('S1 chassis must be world attached with no base joints')
    robot = [model.joint(j) for j in range(model.njnt)
             if (model.joint(j).name or '').startswith('astribot_')]
    if not robot or any(model.jnt_type[j.id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE) for j in robot):
        raise ValueError('Mobile robot command ownership is ambiguous')
    robot_qa = np.asarray(sorted(int(j.qposadr[0]) for j in robot))
    scene_qa = np.setdiff1d(np.arange(model.nq), robot_qa)
    base_names = ('astribot_chassis_x', 'astribot_chassis_y',
                  'astribot_chassis_zrot')
    joint_names = {model.joint(index).name for index in range(model.njnt)}
    present = [name in joint_names for name in base_names]
    if any(present) and not all(present):
        raise ValueError('Partial virtual chassis joint set')
    if any(present):
        raise ValueError('S1 chassis translation and heading are fixed')
    base_qa = (np.asarray([int(model.joint(name).qposadr[0])
                           for name in base_names], dtype=int)
               if all(present) else np.empty(0, dtype=int))
    if not set(base_qa) <= set(robot_qa):
        raise ValueError('Mobile chassis joints are not robot owned')
    return robot_qa, scene_qa, base_qa


@dataclass(frozen=True)
class EventAction:
    event: str
    hand: str | None = None
    object_id: str | None = None
    target_id: str | None = None
    max_speed_m_s: float = 0.04
    target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_fraction: float | None = None
    recovery_mode: str | None = None
    bystander_clearance_m: float | None = None
    support_pull_m: float | None = None
    # Robot-specific capability arguments. Ordinary task events leave these
    # unset; the RuleAgent still selects an event ID rather than joint values.
    base_joint_target: tuple[float, float, float] | None = None
    base_axis_order: tuple[int, int, int] | None = None
    planned_base_qpos: tuple[float, float, float] | None = None
    planned_arm_qpos: tuple[float, ...] | None = None
    planned_object_position_m: tuple[float, float, float] | None = None
    base_certificate_sha256: str | None = None


def loaded_motion_context(action: EventAction, obs: dict, scene: dict) -> dict:
    """Derive a measured feature anchor and a safe idle-hand pose for a slide."""
    if action.support_pull_m is not None:
        if action.event != 'transport' or action.target_id is None:
            raise ValueError('Support reserve requires measured transport target')
        if not 0. < float(action.support_pull_m) <= .02:
            raise ValueError('Support reserve outside bounded motion envelope')
        geometry = {row['id']: row for row in scene['objects']}
        joint = geometry[action.target_id].get('joint') or {}
        joint_name = joint.get('name')
        if joint_name not in obs.get('articulation_axis_world', {}):
            raise ValueError('Support reserve requires observed articulation')
        inactive = next((hand for hand in obs['finger_center_m']
                         if hand != action.hand), None)
        if inactive is None:
            raise ValueError('Support reserve requires a second manipulator')
        features = [row['id'] for row in scene['objects']
                    if row.get('parent_id') == action.target_id and
                    row.get('mobility') == 'static' and
                    all(obs['finger_contact'][row['id']][inactive])]
        if not features:
            raise ValueError('Support reserve requires bilateral feature contact')
        axis = np.asarray(obs['articulation_axis_world'][joint_name], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return {'bystander_hold_anchor': {inactive: {
            'position': np.asarray(obs['finger_center_m'][inactive]).copy() +
                        float(action.support_pull_m) * axis,
            'rotation': np.asarray(obs['hand_rotation_world'][inactive]).copy()}}}
    if action.event not in ('open_drawer', 'close_drawer'):
        return {}
    geometry = {row['id']: row for row in scene['objects']}
    feature = action.object_id
    if feature not in geometry or action.target_id not in geometry:
        raise ValueError('Loaded articulation target is absent')
    size = np.asarray(geometry[feature]['size_m'], dtype=float)
    anchor = (np.asarray(obs['object_rotation_world'][feature]).T @
              (np.asarray(obs['finger_center_m'][action.hand]) -
               np.asarray(obs['object_position_m'][feature])))
    result = {'loaded_contact_anchor_local_m': anchor,
              'pull_offset_m': min(.008, max(.003, .18 * float(np.min(size))))}
    clearance = action.bystander_clearance_m
    if clearance is not None:
        joint = geometry[action.target_id].get('joint') or {}
        axis = np.asarray(obs['articulation_axis_world'][joint['name']], dtype=float)
        axis /= np.linalg.norm(axis)
        inactive = next((hand for hand in obs['finger_center_m'] if hand != action.hand), None)
        if inactive is None:
            raise ValueError('Bystander clearance requires a second manipulator')
        direction = 1. if action.event == 'open_drawer' else -1.
        result['bystander_hold_anchor'] = {inactive: {
            'position': np.asarray(obs['finger_center_m'][inactive]).copy() +
                        direction * float(clearance) * axis,
            'rotation': np.asarray(obs['hand_rotation_world'][inactive]).copy()}}
    return result


def _observation(model, data, names: tuple[str, ...], aperture) -> dict:
    objects = {name: model.body(name).id for name in names}
    hand_position, finger_center, jacobian, opening = {}, {}, {}, {}
    rotation, rotation_jacobian = {}, {}
    for hand in ("left", "right"):
        body = model.body(f"astribot_gripper_{hand}_base").id
        jp = np.zeros((3, model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jacBody(model, data, jp, jr, body)
        hand_position[hand] = data.xpos[body].copy()
        tips = [model.geom(f"astribot_gripper_{hand}_geom_{suffix}").id
                for suffix in ("L112", "R112")]
        finger_center[hand] = data.geom_xpos[tips].mean(axis=0).copy()
        # The controlled point is the pad centroid, not the gripper base.
        # Include the arm rotation contribution to that world-space point.
        mujoco.mj_jac(model, data, jp, jr, finger_center[hand], body)
        jacobian[hand] = jp.copy()
        rotation[hand] = data.xmat[body].reshape(3,3).copy()
        rotation_jacobian[hand] = jr.copy()
        opening[hand] = float(aperture(model, data, hand))
    joint_names = {model.joint(index).name for index in range(model.njnt)}
    base_position = (tuple(float(data.qpos[model.joint(name).qposadr[0]])
                           for name in ('astribot_chassis_x',
                                        'astribot_chassis_y',
                                        'astribot_chassis_zrot'))
                     if 'astribot_chassis_x' in joint_names else (0.0, 0.0, 0.0))
    arm_position = {
        hand: tuple(float(data.qpos[model.joint(
            f'astribot_arm_{hand}_joint_{index}').qposadr[0]])
            for index in range(1, 8))
        for hand in ('left', 'right')}
    contact = {name: {"left": (False, False), "right": (False, False)}
               for name in names}
    # Geometric overlap can have zero solver load. Keep measured normal force
    # separate so the agent can distinguish touch from a loaded grasp.
    finger_normal_force_n = {name: {"left": [0., 0.], "right": [0., 0.]}
                             for name in names}
    hand_contact = {name: {"left": False, "right": False} for name in names}
    scene_contacts = {"left": {}, "right": {}}
    support = {name: set() for name in names}
    reverse = {body: name for name, body in objects.items()}
    # Map all collidable gripper parts by their parent arm. Imported mesh
    # names alone are unreliable: one left palm carries a right-side prefix.
    gripper_hands = {}
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name or ''
        if not geom_name.startswith('astribot_gripper_') or not model.geom_contype[geom_id]:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        while body_id:
            body_name = model.body(body_id).name or ''
            if body_name.startswith('astribot_arm_left_'):
                gripper_hands[geom_id] = 'left'
                break
            if body_name.startswith('astribot_arm_right_'):
                gripper_hands[geom_id] = 'right'
                break
            body_id = int(model.body_parentid[body_id])
    # The collidable distal mesh and the pad proxy share one physical finger
    # link. Classify contact by that link, not by the proxy geom name; otherwise
    # a real distal-finger contact is reported as a lost grasp. This maps robot
    # geometry only and leaves the native task evaluator and hard gates intact.
    distal_side_by_body = {}
    for hand in ('left', 'right'):
        for side_index, suffix in enumerate(('L112', 'R112')):
            pad = model.geom(f'astribot_gripper_{hand}_geom_{suffix}').id
            distal_side_by_body[(hand, int(model.geom_bodyid[pad]))] = side_index
    for contact_index, c in enumerate(data.contact):
        if c.dist > 0:
            continue
        for g, other in ((int(c.geom1), int(c.geom2)),
                         (int(c.geom2), int(c.geom1))):
            name = reverse.get(int(model.geom_bodyid[g]))
            if name is None:
                continue
            other_name = model.geom(other).name or ""
            other_body = model.body(int(model.geom_bodyid[other])).name or ""
            if int(model.geom_bodyid[other]) in reverse:
                support[name].add(reverse[int(model.geom_bodyid[other])])
            if other_name.startswith("worktable"):
                support[name].add("worktable")
            for hand in ("left", "right"):
                if gripper_hands.get(other) == hand:
                    hand_contact[name][hand] = True
                    a, b = contact[name][hand]
                    distal_side = distal_side_by_body.get(
                        (hand, int(model.geom_bodyid[other])))
                    a |= distal_side == 0 or other_body.endswith("link7")
                    b |= distal_side == 1 or other_body.endswith("link8")
                    contact[name][hand] = (a, b)
                    # Any collidable geom on the same distal finger body is
                    # a physically measured finger contact, not a palm hit.
                    if distal_side is not None:
                        force = np.zeros(6, dtype=float)
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        if not np.isfinite(force).all():
                            raise ValueError('Nonfinite native distal-finger contact force')
                        finger_normal_force_n[name][hand][distal_side] = max(
                            finger_normal_force_n[name][hand][distal_side],
                            max(0., float(force[0])))
                        # MuJoCo exposes contact frame rows on the vectorized
                        # data.contact array. Reading c.frame from an iterated
                        # MjContact broadcasts its first scalar in this build.
                        # frame[0:3] points geom1 -> geom2, so negate it when
                        # the robot pad is geom1.
                    normal = data.contact.frame[contact_index, :3]
                    away = -normal if other == int(c.geom1) else normal
                    norm = float(np.linalg.norm(away))
                    if norm > 1e-9 and np.isfinite(norm):
                        key = (name, other_name)
                        item = {"object_id": name,
                                "robot_geom": other_name,
                                "scene_geom": model.geom(g).name or "",
                                "away_world": (away / norm).tolist(),
                                "penetration_m": max(0., -float(c.dist))}
                        old = scene_contacts[hand].get(key)
                        if old is None or item['penetration_m'] > old['penetration_m']:
                            scene_contacts[hand][key] = item
        # Keep non-task fixtures visible to the online agent as measured
        # gripper contacts. The task object loop above cannot see bodies such
        # as the work surface because they are outside the contract object
        # namespace. No object pose or simulation state is changed here.
        for robot_geom_id, fixture_geom_id in ((int(c.geom1), int(c.geom2)),
                                               (int(c.geom2), int(c.geom1))):
            robot_geom = model.geom(robot_geom_id).name or ""
            fixture_body_id = int(model.geom_bodyid[fixture_geom_id])
            fixture_body = model.body(fixture_body_id).name or ""
            if fixture_body_id in reverse or fixture_body.startswith('astribot_'):
                continue
            for hand in ('left', 'right'):
                if gripper_hands.get(robot_geom_id) != hand:
                    continue
                normal = data.contact.frame[contact_index, :3]
                away = -normal if robot_geom_id == int(c.geom1) else normal
                norm = float(np.linalg.norm(away))
                if norm <= 1e-9 or not np.isfinite(norm):
                    continue
                fixture_geom = model.geom(fixture_geom_id).name or ''
                fixture_id = fixture_body or fixture_geom
                key = (fixture_id, robot_geom)
                item = {'object_id': fixture_id,
                        'robot_geom': robot_geom,
                        'scene_geom': fixture_geom,
                        'away_world': (away / norm).tolist(),
                        'penetration_m': max(0., -float(c.dist))}
                old = scene_contacts[hand].get(key)
                if old is None or item['penetration_m'] > old['penetration_m']:
                    scene_contacts[hand][key] = item
    articulation, articulation_fraction, articulation_axis = {}, {}, {}
    for j in range(model.njnt):
        name = model.joint(j).name or ""
        if (model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and
                not name.startswith('astribot_')):
            position = float(data.qpos[int(model.jnt_qposadr[j])])
            articulation[name] = position
            articulation_axis[name] = data.xaxis[j].copy()
            lo, hi = model.jnt_range[j]
            articulation_fraction[name] = float(np.clip((position - lo) / (hi - lo), 0, 1))
    joint_margin = float("inf")
    limiting_joint = None
    arm_torso_margin = float("inf")
    limiting_arm_torso_joint = None
    gripper_margin = float("inf")
    limiting_gripper_joint = None
    manipulator_joint_margin = {}
    limiting_manipulator_joint = {}
    for j in range(model.njnt):
        if not model.jnt_limited[j] or model.jnt_type[j] not in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        qa = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        margin = min(float(data.qpos[qa]-lo), float(hi-data.qpos[qa]))
        if margin < joint_margin:
            joint_margin = margin
            limiting_joint = model.joint(j).name
        name = model.joint(j).name or ""
        if name.startswith(("astribot_arm_", "astribot_torso_joint_")) and margin < arm_torso_margin:
            arm_torso_margin = margin
            limiting_arm_torso_joint = name
        if name.startswith('astribot_gripper_') and margin < gripper_margin:
            gripper_margin = margin
            limiting_gripper_joint = name
        # This robot adapter discovers its arms from the model and exposes
        # an arm-count-independent capability map to the RuleAgent.
        match = re.fullmatch(r"astribot_arm_(.+)_joint_\d+", name)
        if match:
            manipulator = match.group(1)
            if margin < manipulator_joint_margin.get(manipulator, float("inf")):
                manipulator_joint_margin[manipulator] = margin
                limiting_manipulator_joint[manipulator] = name
    if not np.isfinite(arm_torso_margin):
        raise ValueError("Robot arm/torso joint limits unavailable")
    if not np.isfinite(gripper_margin):
        raise ValueError("Robot gripper joint limits unavailable")
    if not manipulator_joint_margin or not all(
            np.isfinite(value) for value in manipulator_joint_margin.values()):
        raise ValueError("Robot arm joint limits unavailable")
    penetration = 0.0
    for c in data.contact:
        if c.dist > 0:
            continue
        names_pair = (model.geom(int(c.geom1)).name or "",
                      model.geom(int(c.geom2)).name or "")
        if (any(name.startswith("worktable") for name in names_pair) and
                any("astribot_" in name for name in names_pair)):
            penetration = max(penetration, -float(c.dist))
    # Bound prompt evidence by severity while retaining object IDs of every
    # actual collidable gripper component, including visual meshes.
    ranked_scene_contacts = {
        hand: sorted(items.values(), key=lambda item: (
            -item['penetration_m'], item['object_id'], item['robot_geom']))[:8]
        for hand, items in scene_contacts.items()}
    return {"time_s": float(data.time),
            # Indices in the capability executor are model qpos/dof addresses.
            # Return the full arrays so inserting base joints cannot truncate
            # a right-arm or gripper state. The model object remains private.
            "robot_qpos": data.qpos.copy(),
            "robot_qvel": data.qvel.copy(),
            "base_joint_qpos": base_position,
            "arm_joint_qpos": arm_position,
            "object_position_m": {name: data.xpos[body].copy()
                                  for name, body in objects.items()},
            "object_rotation_world": {name: data.xmat[body].reshape(3,3).copy()
                                      for name, body in objects.items()},
            "object_velocity_m_s": {name: data.cvel[body, 3:].copy()
                                     for name, body in objects.items()},
            "hand_position_m": hand_position,
            "finger_center_m": finger_center,
            "hand_jacobian": jacobian,
            "hand_rotation_world": rotation,
            "hand_rotation_jacobian": rotation_jacobian,
            "gripper_opening_m": opening,
            "finger_contact": contact,
            "finger_normal_force_n": finger_normal_force_n,
            "hand_contact": hand_contact,
            "gripper_scene_contacts": ranked_scene_contacts,
            "gripper_scene_contact_object_ids": {
                hand: sorted({item['object_id'] for item in items})
                for hand, items in ranked_scene_contacts.items()},
            "support_contact": {k: tuple(sorted(v)) for k, v in support.items()},
            "articulation_qpos": articulation,
            "articulation_axis_world": articulation_axis,
            "articulation_fraction": articulation_fraction,
            "min_joint_margin_native": joint_margin,
            "limiting_joint": limiting_joint,
            "arm_torso_min_joint_margin_rad": arm_torso_margin,
            "limiting_arm_torso_joint": limiting_arm_torso_joint,
            "gripper_min_joint_margin_rad": gripper_margin,
            "limiting_gripper_joint": limiting_gripper_joint,
            "manipulator_joint_margin_rad": manipulator_joint_margin,
            "limiting_manipulator_joint": limiting_manipulator_joint,
            "max_robot_table_penetration_m": penetration}


class ContinuousSession:
    """A single native rollout controlled by observe/advance calls.

    ``advance`` blocks until the requested short action finishes, its event
    predicate becomes true, or the episode ends.  Model calls may happen while
    this session is paused; no physical steps run during that time.
    """

    def __init__(self, pilot_physics, rssr_servo, config: dict, scene: dict,
                 output: Path, target_qpos: np.ndarray, *, stop_s: float | None = None, observer=None):
        self.pilot = pilot_physics
        self.servo_module = rssr_servo
        self.config, self.scene = config, scene
        self.output, self.target_qpos, self.stop_s = Path(output), target_qpos, stop_s
        if self.output.exists():
            raise FileExistsError(self.output)
        self.observer = observer
        self.names = tuple(row["id"] for row in scene["objects"])
        self.condition = threading.Condition()
        self.ready = False
        self.done = False
        self.error: BaseException | None = None
        self.aborted = False
        self.terminal_stop_requested = False
        self.report: dict | None = None
        self.snapshot: dict | None = None
        self.action: EventAction | None = None
        self.event_test: Callable[[dict], str | None] | None = None
        self.recovery_epoch: int | None = None
        self.recovery_start_m: np.ndarray | None = None
        self.recovery_target_m: np.ndarray | None = None
        self.contact_ik_epoch: int | None = None
        self.contact_ik_plan: dict | None = None
        self.contact_ik_plan_records: list[dict] = []
        self.remaining = 0
        self.advanced = 0
        self.reason = "initial"
        self.timestep_s: float | None = None
        self.hard_records = []
        self._hard_data = None
        self._hard_reader = None
        self._hard_accumulator = None
        self._preview_state = None
        self._preview_servo_class = None
        self.torso_start_qpos = None
        self.torso_qa = None
        self.torso_actual_max_excursion_rad = np.zeros(2)
        self.torso_command_max_excursion_rad = np.zeros(2)
        self.command_records: list[tuple[float, np.ndarray, np.ndarray, str]] = []
        self._thread: threading.Thread | None = None

    def start(self) -> dict:
        if self._thread is not None:
            raise RuntimeError("Session already started")
        self._thread = threading.Thread(target=self._run, name="rssr-continuous-physics",
                                        daemon=False)
        self._thread.start()
        return self.observe()

    def observe(self) -> dict:
        with self.condition:
            self.condition.wait_for(lambda: self.ready or self.done)
            if self.error is not None:
                raise RuntimeError("Native continuous rollout failed") from self.error
            if self.snapshot is None:
                raise RuntimeError("Rollout ended before its first observation")
            return self.snapshot

    def advance(self, action: EventAction, duration_s: float,
                event_test: Callable[[dict], str | None] | None = None,
                *, action_epoch: int | None = None) -> tuple[dict, str]:
        if not np.isfinite(duration_s) or not 0 < duration_s <= 2.0:
            raise ValueError("A short advance must last at most 2 seconds")
        with self.condition:
            if not self.ready or self.done:
                raise RuntimeError("Session is not paused at an observation")
            if action.event in ('clearance', 'retreat'):
                epoch = action_epoch if action_epoch is not None else id(action)
                if self.recovery_epoch != epoch:
                    from .executor import recovery_target
                    assert self.snapshot is not None
                    self.recovery_start_m = np.asarray(
                        self.snapshot['finger_center_m'][action.hand], dtype=float).copy()
                    self.recovery_target_m = recovery_target(action, self.snapshot)
                    self.recovery_epoch = epoch
            else:
                self.recovery_epoch = None
                self.recovery_start_m = None
                self.recovery_target_m = None
            if action.event == 'contact':
                epoch = action_epoch if action_epoch is not None else id(action)
                if self.contact_ik_epoch != epoch:
                    from .contact_pose_ik import plan_contact_pose
                    assert self.snapshot is not None
                    try:
                        plan = plan_contact_pose(self._hard_reader.model,
                                                 self.snapshot, action)
                    except Exception as error:
                        plan = {'status': 'planner_error',
                                'error_type': type(error).__name__,
                                'error': str(error)}
                    self.contact_ik_plan = plan
                    self.contact_ik_epoch = epoch
                    self.contact_ik_plan_records.append({
                        'time_s': float(self.snapshot['time_s']),
                        'action_epoch': epoch, 'event': action.event,
                        'object_id': action.object_id, 'hand': action.hand,
                        'target_offset_m': list(action.target_offset_m),
                        'plan': plan})
            else:
                self.contact_ik_epoch = None
                self.contact_ik_plan = None
            self.action, self.event_test = action, event_test
            if self.timestep_s is None:
                raise RuntimeError("Native physics timestep is unavailable")
            self.remaining = max(1, int(round(duration_s / self.timestep_s)))
            self.advanced = 0
            self.ready = False
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.ready or self.done)
            if self.error is not None:
                raise RuntimeError("Native continuous rollout failed") from self.error
            if self.snapshot is None:
                raise RuntimeError("Rollout ended without an observation")
            return self.snapshot, self.reason

    def finish(self) -> dict:
        """After the episode ends, return the native report and command trace."""
        with self.condition:
            if not self.done:
                raise RuntimeError("Episode still paused; advance it to completion")
            if self.error is not None:
                raise RuntimeError("Native continuous rollout failed") from self.error
            assert self.report is not None
            return self.report

    def request_terminal_stop(self) -> None:
        """Finish after the next audited physical step from a paused state."""
        with self.condition:
            if not self.ready or self.done:
                raise RuntimeError('Terminal stop requires a paused live session')
            self.terminal_stop_requested = True

    def preview(self, action: EventAction, duration_s: float, *, contract=None) -> dict:
        """Run a bounded branch from a paused native state without advancing it."""
        with self.condition:
            if not self.ready or self.done or self._preview_state is None:
                raise RuntimeError('Physical preview requires a paused session')
            state = dict(self._preview_state)
            model = self._hard_reader.model
            servo_class = self._preview_servo_class
            observation = self.snapshot
        if state.get('active_action_id') != id(action):
            previous_anchors = state.get('hold_anchors', {})
            state['hold_anchors'] = {hand: {
                'position': np.asarray(observation['finger_center_m'][hand]).copy(),
                'rotation': np.asarray(observation['hand_rotation_world'][hand]).copy()}
                for hand in observation['finger_center_m']}
            inactive = next((hand for hand in observation['finger_center_m']
                             if hand != action.hand), None)
            if (state.get('active_hand') == action.hand and
                    inactive in previous_anchors):
                state['hold_anchors'][inactive] = previous_anchors[inactive]
            context = loaded_motion_context(action, observation, self.scene)
            state.update({key: value for key, value in context.items()
                          if key != 'bystander_hold_anchor'})
            state['hold_anchors'].update(context.get('bystander_hold_anchor', {}))
        if action.event == 'contact':
            from .contact_pose_ik import plan_contact_pose
            try:
                state['contact_ik_plan'] = plan_contact_pose(model, observation, action)
            except Exception as error:
                state['contact_ik_plan'] = {
                    'status': 'planner_error',
                    'error_type': type(error).__name__,
                    'error': str(error)}
        else:
            state['contact_ik_plan'] = None
        from .event_preview import evaluate
        return evaluate(model, servo_class, self.pilot.aperture,
                        self.scene, state, action, duration_s=duration_s,
                        contract=contract)

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def abort(self) -> None:
        """Stop a paused rollout after an external failure without a leaked thread."""
        with self.condition:
            self.aborted = True
            self.condition.notify_all()
        self.join()

    def _run(self) -> None:
        original = self.servo_module.JointServo
        self._preview_servo_class = original
        session = self

        class EventServo(original):
            def __init__(self, model, acts, kind):
                super().__init__(model, acts, kind)
                self.model = model
                session.timestep_s = float(model.opt.timestep)
                self.last_q = None
                self.last_t = None
                self.cached_snapshot = None
                self.observed_at = -1.0
                self.hold_anchors = {}
                self.active_hand = None
                self.active_event_kind = None
                self.active_action_id = None
                self.loaded_contact_anchor_local_m = None
                self.pull_offset_m = None
                self.contact_settle_anchor_q = None
                self.contact_settle_anchor_key = None
                self.contact_settle_last_touch_t = None
                self.contact_since = {}
                self.robot_qa, self.scene_qa, self.base_qa = robot_command_indices(model)
                self.torso_qa = np.asarray([int(model.joint(
                    f'astribot_torso_joint_{index}').qposadr[0])
                    for index in (1, 2)])
                self.locked_torso = None
                session.torso_qa = self.torso_qa.copy()
                session._hard_reader = HardSnapshotter(model)
                session._hard_accumulator = HardAccumulator()

            def apply(self, data, desired, velocity):
                if self.locked_torso is None:
                    self.locked_torso = np.asarray(
                        data.qpos[self.torso_qa], dtype=float).copy()
                    session.torso_start_qpos = self.locked_torso.copy()
                session.torso_actual_max_excursion_rad = np.maximum(
                    session.torso_actual_max_excursion_rad,
                    np.abs(data.qpos[self.torso_qa] - self.locked_torso))
                if self.cached_snapshot is None or float(data.time)-self.observed_at >= .01-1e-9:
                    snapshot = _observation(self.model, data, session.names, session.pilot.aperture)
                    if session.observer is not None:
                        snapshot = session.observer(snapshot)
                    for obj,hands in snapshot['finger_contact'].items():
                        for hand,pads in hands.items():
                            key=(obj,hand)
                            if all(pads): self.contact_since.setdefault(key,float(data.time))
                            else: self.contact_since.pop(key,None)
                    snapshot['object_geometry']={row['id']:{key:row[key] for key in ('shape','size_m','wall_m','parent_id','role','mobility','joint','yaw_rad') if key in row} for row in session.scene['objects']}
                    snapshot['bilateral_contact_seconds']={obj:{hand:max(0.,float(data.time)-self.contact_since.get((obj,hand),float(data.time))) for hand in hands} for obj,hands in snapshot['finger_contact'].items()}
                    self.cached_snapshot = snapshot
                    self.observed_at = float(data.time)
                snapshot = self.cached_snapshot
                session._hard_data = data
                instant_hard = session._hard_reader(data)
                session.hard_records.append(instant_hard)
                snapshot['immutable_hard'] = session._hard_accumulator.observe(instant_hard)
                # The cumulative hard record is deliberately latched. Expose
                # this step separately so a warning can respond to the live
                # state without repeatedly firing on an old peak.
                snapshot['instantaneous_gripper_coupling_residual_rad'] = float(
                    instant_hard['gripper_coupling_residual_rad'])
                snapshot['time_s'] = float(data.time)
                snapshot['torso_max_physical_excursion_rad'] = float(np.max(
                    session.torso_actual_max_excursion_rad))
                snapshot['torso_max_command_excursion_rad'] = float(np.max(
                    session.torso_command_max_excursion_rad))
                snapshot['torso_initial_qpos'] = self.locked_torso.copy()
                if session.recovery_start_m is not None:
                    snapshot['recovery_start_m'] = session.recovery_start_m.copy()
                    snapshot['recovery_target_m'] = session.recovery_target_m.copy()
                else:
                    snapshot.pop('recovery_start_m', None)
                    snapshot.pop('recovery_target_m', None)
                with session.condition:
                    session.snapshot = snapshot
                    if session.action is not None:
                        event_reason = (session.event_test(snapshot)
                                        if session.event_test is not None and
                                        session.advanced > 0 else None)
                        if event_reason:
                            session.reason = event_reason
                            session.action = None
                        elif session.remaining <= 0:
                            session.reason = "duration"
                            session.action = None
                    if session.action is None:
                        session.snapshot = snapshot
                        session._preview_state = {
                            'qpos': data.qpos.copy(),
                            'qvel': data.qvel.copy(),
                            'ctrl': data.ctrl.copy(),
                            'act': data.act.copy(),
                            'qacc_warmstart': data.qacc_warmstart.copy(),
                            'time_s': float(data.time),
                            'desired_qpos': desired.copy(),
                            'desired_qvel': velocity.copy(),
                            'last_q': None if self.last_q is None else self.last_q.copy(),
                            'last_t': self.last_t,
                            'hold_anchors': {hand: {
                                'position': value['position'].copy(),
                                'rotation': value['rotation'].copy()}
                                for hand, value in self.hold_anchors.items()},
                            'active_action_id': self.active_action_id,
                            'active_hand': self.active_hand,
                            'contact_settle_anchor_qpos': (
                                None if self.contact_settle_anchor_q is None
                                else self.contact_settle_anchor_q.copy()),
                            'contact_settle_anchor_key': self.contact_settle_anchor_key,
                            'contact_settle_last_touch_t': self.contact_settle_last_touch_t,
                            'loaded_contact_anchor_local_m': (None if self.loaded_contact_anchor_local_m is None
                                else self.loaded_contact_anchor_local_m.copy()),
                            'pull_offset_m': self.pull_offset_m,
                            'torso_initial_qpos': self.locked_torso.copy(),
                        }
                        session.ready = True
                        session.condition.notify_all()
                        session.condition.wait_for(
                            lambda: session.action is not None or session.aborted)
                    if session.aborted:
                        raise RuntimeError("Continuous session aborted before episode completion")
                    action = session.action
                    session.remaining -= 1
                    session.advanced += 1
                assert action is not None
                if self.active_action_id != id(action):
                    previous_active_hand = self.active_hand
                    previous_anchors = self.hold_anchors
                    self.hold_anchors = {hand: {
                        'position': snapshot['finger_center_m'][hand].copy(),
                        'rotation': snapshot['hand_rotation_world'][hand].copy()}
                        for hand in snapshot['finger_center_m']}
                    # Preserve the other hand's world target across consecutive
                    # events by the same active hand. Reset when control changes
                    # hands, or an explicit support-reserve event supplies a new
                    # measured target.
                    inactive = next((hand for hand in snapshot['finger_center_m']
                                     if hand != action.hand), None)
                    if (previous_active_hand == action.hand and
                            inactive in previous_anchors):
                        self.hold_anchors[inactive] = previous_anchors[inactive]
                    context = loaded_motion_context(action, snapshot, session.scene)
                    self.loaded_contact_anchor_local_m = context.get('loaded_contact_anchor_local_m')
                    self.pull_offset_m = context.get('pull_offset_m')
                    self.hold_anchors.update(context.get('bystander_hold_anchor', {}))
                    self.active_action_id = id(action)
                    self.active_hand = (None if action.event == 'reference'
                                        else action.hand)
                self.active_event_kind=action.event
                snapshot['hold_anchors']=self.hold_anchors
                if self.loaded_contact_anchor_local_m is not None and action.event in ('open_drawer', 'close_drawer'):
                    snapshot['loaded_contact_anchor_local_m'] = self.loaded_contact_anchor_local_m
                    snapshot['pull_offset_m'] = self.pull_offset_m
                else:
                    snapshot.pop('loaded_contact_anchor_local_m', None)
                    snapshot.pop('pull_offset_m', None)
                fingers=(snapshot.get('finger_contact',{})
                         .get(action.object_id,{}).get(action.hand,(False,False)))
                settle_key=(action.object_id,action.hand)
                # A cup must finish centering before its arm pose is frozen.
                # Capturing the first unilateral touch restores a stale pose
                # as soon as bilateral contact appears and undoes centering.
                cup_contact=(snapshot.get('object_geometry',{}).get(
                    action.object_id,{}).get('shape')=='cup')
                contact_for_anchor=(all(fingers) if cup_contact else any(fingers))
                if (action.event=='contact' and len(fingers)==2 and
                        contact_for_anchor):
                    if (self.contact_settle_anchor_q is None or
                            self.contact_settle_anchor_key!=settle_key):
                        self.contact_settle_anchor_q=data.qpos.copy()
                        self.contact_settle_anchor_key=settle_key
                    self.contact_settle_last_touch_t=float(data.time)
                    snapshot['contact_settle_anchor_qpos']=self.contact_settle_anchor_q
                elif (action.event=='contact' and not cup_contact and len(fingers)==2 and
                      not any(fingers) and
                      self.contact_settle_anchor_q is not None and
                      self.contact_settle_anchor_key==settle_key and
                      self.contact_settle_last_touch_t is not None and
                      float(data.time)-self.contact_settle_last_touch_t<=.4):
                    # A brief contact sensor dropout must not restart wrist IK
                    # while the second finger is still closing.
                    snapshot['contact_settle_anchor_qpos']=self.contact_settle_anchor_q
                else:
                    self.contact_settle_anchor_q=None
                    self.contact_settle_anchor_key=None
                    self.contact_settle_last_touch_t=None
                    snapshot.pop('contact_settle_anchor_qpos',None)
                if session.contact_ik_plan is not None:
                    snapshot['contact_ik_plan'] = session.contact_ik_plan
                else:
                    snapshot.pop('contact_ik_plan', None)
                q, dq = desired.copy(), velocity.copy()
                if action.event != "reference":
                    # The selected event owns commands; the unselected arm
                    # holds its last target instead of continuing a hidden plan.
                    if self.last_q is not None:
                        q[self.robot_qa] = self.last_q[self.robot_qa]
                        dq[self.robot_qa] = 0.0
                    q, dq = session._control(action, snapshot, q, dq, self.model)
                if (q.shape != desired.shape or dq.shape != velocity.shape or
                        not np.isfinite(q).all() or not np.isfinite(dq).all() or
                        not np.array_equal(q[self.scene_qa], desired[self.scene_qa]) or
                        not np.array_equal(dq[self.scene_qa], velocity[self.scene_qa])):
                    raise ValueError("Event executor returned invalid or nonrobot commands")
                if self.last_q is not None:
                    dt = float(data.time) - self.last_t
                    if dt <= 0:
                        raise ValueError("Event executor saw nonpositive physics time step")
                    q[self.robot_qa] = np.clip(
                        q[self.robot_qa], self.last_q[self.robot_qa] - 4.0 * dt,
                        self.last_q[self.robot_qa] + 4.0 * dt)
                    # Base movement, once offered as an event, must obey a
                    # separate slow envelope even though arm joints can move
                    # faster. This is only a command bound, not proof of
                    # physical chassis tracking.
                    if len(self.base_qa) == 3:
                        q[self.base_qa[:2]] = np.clip(
                            q[self.base_qa[:2]],
                            self.last_q[self.base_qa[:2]] - .20 * dt,
                            self.last_q[self.base_qa[:2]] + .20 * dt)
                        q[self.base_qa[2]] = np.clip(
                            q[self.base_qa[2]],
                            self.last_q[self.base_qa[2]] - .35 * dt,
                            self.last_q[self.base_qa[2]] + .35 * dt)
                        dq[self.base_qa] = (q[self.base_qa] -
                                            self.last_q[self.base_qa]) / dt
                    if action.event=='reference':
                        dq[self.robot_qa] = np.clip(
                            (q[self.robot_qa]-self.last_q[self.robot_qa])/dt,
                            -4., 4.)
                # One coupled waist degree of freedom may improve measured
                # hand reach. The other torso joint follows the native S1
                # coupling while the chassis stays fixed.
                first, second = self.torso_qa
                prior = (float(self.last_q[first]) if self.last_q is not None
                         else float(self.locked_torso[0]))
                dt_waist = (float(data.time)-self.last_t if self.last_q is not None
                            else float(self.model.opt.timestep))
                requested = (float(self.locked_torso[0]) if
                             action.event == 'reference' else float(q[first]))
                first_joint = self.model.joint('astribot_torso_joint_1')
                second_joint = self.model.joint('astribot_torso_joint_2')
                coupled_sum = float(self.locked_torso.sum())
                waist_lo = max(float(self.model.jnt_range[first_joint.id, 0]) + .07,
                               coupled_sum - float(self.model.jnt_range[second_joint.id, 1]) + .07)
                waist_hi = min(float(self.model.jnt_range[first_joint.id, 1]) - .07,
                               coupled_sum - float(self.model.jnt_range[second_joint.id, 0]) - .07)
                requested = float(np.clip(requested, waist_lo, waist_hi))
                requested = float(np.clip(requested,
                    prior-.25*dt_waist, prior+.25*dt_waist))
                q[first] = requested
                q[second] = float(self.locked_torso.sum()) - requested
                dq[first] = (requested-prior)/dt_waist
                dq[second] = -dq[first]
                for index,value in enumerate(session.config['_s1_measured_torso_rad'][2:],3):
                    address=int(self.model.joint(f'astribot_torso_joint_{index}').qposadr[0]);q[address]=value;dq[address]=0.
                if '_lyc_head_target_rad' in session.config:
                    for name, target in zip(('astribot_head_joint_1', 'astribot_head_joint_2'),
                                            session.config['_lyc_head_target_rad']):
                        address = int(self.model.joint(name).qposadr[0])
                        q[address] = target
                        dq[address] = 0.
                session.torso_command_max_excursion_rad = np.maximum(
                    session.torso_command_max_excursion_rad,
                    np.abs(q[self.torso_qa] - self.locked_torso))
                self.last_q, self.last_t = q.copy(), float(data.time)
                desired[:], velocity[:] = q, dq
                session.command_records.append((float(data.time), q.copy(), dq.copy(),
                                                action.event))
                return super().apply(data, desired, velocity)

        self.servo_module.JointServo = EventServo
        try:
            self.report = self.pilot.rollout(self.config, "sim2", self.output,
                                            self.scene, target_qpos=self.target_qpos,
                                            stop_s=self.stop_s,
                                            terminal_stop=lambda: self.terminal_stop_requested)
            # apply observes before step. Observe the final step result too.
            if self._hard_data is not None:
                if self.torso_qa is not None and self.torso_start_qpos is not None:
                    self.torso_actual_max_excursion_rad = np.maximum(
                        self.torso_actual_max_excursion_rad,
                        np.abs(self._hard_data.qpos[self.torso_qa] -
                               self.torso_start_qpos))
                instant_hard = self._hard_reader(self._hard_data)
                self.hard_records.append(instant_hard)
                cumulative = self._hard_accumulator.observe(instant_hard)
                final = _observation(self._hard_reader.model, self._hard_data,
                                     self.names, self.pilot.aperture)
                if self.observer is not None:
                    final = self.observer(final)
                final['immutable_hard'] = cumulative
                final['instantaneous_gripper_coupling_residual_rad'] = float(
                    instant_hard['gripper_coupling_residual_rad'])
                final['torso_max_physical_excursion_rad'] = float(np.max(
                    self.torso_actual_max_excursion_rad))
                final['torso_max_command_excursion_rad'] = float(np.max(
                    self.torso_command_max_excursion_rad))
                final['bilateral_contact_seconds'] = self.snapshot['bilateral_contact_seconds']
                final['object_geometry'] = self.snapshot['object_geometry']
                self.snapshot = final
        except BaseException as error:
            self.error = error
        finally:
            self.servo_module.JointServo = original
            try:
                if self.hard_records:
                    save_hard_trace(self.output / 'immutable_hard_trace.npz', self.hard_records)
            except BaseException as error:
                self.error = error
            finally:
                with self.condition:
                    self.done = True
                    self.ready = False
                    self.condition.notify_all()

    def _control(self, action: EventAction, snapshot: dict, q: np.ndarray,
                 dq: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
        # The closed-loop robot controller is supplied in executor.py.  Keeping
        # this hook here makes baseline identity runs possible before admission.
        from .executor import control_step
        return control_step(action, snapshot, q, dq, model)
