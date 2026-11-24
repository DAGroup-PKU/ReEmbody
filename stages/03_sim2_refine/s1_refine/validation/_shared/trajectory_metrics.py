"""Read-only native trajectory metrics; curvature and jerk are not success gates.

Finite differences describe the sampled native state, not a hardware safety
certificate. Path excess can be required for clearance, so no uncalibrated
smoothness or path threshold is used to relabel a physical failure as success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {"max": float(np.max(values)), "p99": float(np.percentile(values, 99)),
            "p95": float(np.percentile(values, 95)), "median": float(np.median(values))}


def runs(mask, time_s):
    mask = np.asarray(mask, dtype=bool)
    t = np.asarray(time_s, dtype=float)
    if not len(t):
        return []
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [{"start_s": float(t[a]), "end_s": float(t[b - 1] + dt),
             "duration_s": float(t[b - 1] - t[a] + dt)}
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def command_cycles(time_s, desired_angle):
    """Count meaningful command closures from t=0, including outside events.

    Use the established 0.06rad close / 0.12rad reopen command thresholds.
    This deliberately counts a commanded empty closure even when no actual
    aperture change occurs. The independent force gate determines acquisition.
    """
    t, angles = np.asarray(time_s), np.asarray(desired_angle)
    rows = []
    for hi, hand in enumerate(("left", "right")):
        state = "waiting"
        trough = peak = float(angles[0, hi])
        closes, opens = [], []
        for i in range(1, len(t)):
            value = float(angles[i, hi])
            if state == "waiting":
                trough = min(trough, value)
                if value - trough >= .06 - 1e-12:
                    closes.append(float(t[i])); peak = value; state = "closing"
            else:
                peak = max(peak, value)
                if peak - value >= .12 - 1e-12:
                    opens.append(float(t[i])); trough = value; state = "waiting"
        rows.append({"hand": hand, "meaningful_close_count": len(closes),
                     "meaningful_close_s": closes, "command_reopen_s": opens})
    return {"scope": "entire native command stream, including before labelled acquisition and after release",
            "close_command_rise_rad": .06, "reopen_command_fall_rad": .12,
            "hands": rows, "exactly_one_close_each": all(x["meaningful_close_count"] == 1 for x in rows)}


def metrics(case_dir):
    case = Path(case_dir)
    receipt = json.loads((case / "receipt.json").read_text())
    native = np.load(case / "execute/rollout.npz", allow_pickle=False)
    commands = np.load(case / "issued_robot_commands.npz", allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(case / "execute/scene.xml"))
    t = native["time_s"]
    if len(t) < 4 or not np.all(np.diff(t) > 0):
        raise ValueError("Metrics require at least four strictly ordered native samples")
    arm_joints = [j for j in (model.joint(i) for i in range(model.njnt))
                  if (j.name or "").startswith(("astribot_arm_left_", "astribot_arm_right_"))]
    qa = np.asarray([int(j.qposadr[0]) for j in arm_joints])
    va = np.asarray([int(j.dofadr[0]) for j in arm_joints])
    velocity = native["qvel"][:, va]
    acceleration = np.gradient(velocity, t, axis=0)
    jerk = np.gradient(acceleration, t, axis=0)
    command_velocity = np.gradient(commands["qpos"][:, qa], commands["time_s"], axis=0)
    command_acceleration = np.gradient(command_velocity, commands["time_s"], axis=0)
    actuated = []
    for ai in range(model.nu):
        joint = model.joint(int(model.actuator_trnid[ai, 0]))
        position_index, velocity_index = int(joint.qposadr[0]), int(joint.dofadr[0])
        actual_v = native["qvel"][:, velocity_index]
        desired_v = np.gradient(commands["qpos"][:, position_index], commands["time_s"])
        actual_a = np.gradient(actual_v, t)
        desired_a = np.gradient(desired_v, commands["time_s"])
        unit = "rad" if int(model.jnt_type[joint.id]) == int(mujoco.mjtJoint.mjJNT_HINGE) else "m"
        actuated.append({"joint": joint.name, "position_unit": unit,
                         "actual_max_abs_velocity_per_s": float(np.max(np.abs(actual_v))),
                         "actual_max_abs_acceleration_per_s2": float(np.max(np.abs(actual_a))),
                         "command_max_abs_velocity_per_s": float(np.max(np.abs(desired_v))),
                         "command_max_abs_acceleration_per_s2": float(np.max(np.abs(desired_a)))})
    hand_qa = [int(model.joint(f"astribot_gripper_{hand}_joint_L1").qposadr[0])
               for hand in ("left", "right")]
    history = receipt.get("event_history", [])
    data = mujoco.MjData(model)
    tip_ids = [[model.geom(f"astribot_gripper_{hand}_geom_{suffix}").id for suffix in ("L112", "R112")]
               for hand in ("left", "right")]
    centers = np.empty((len(t), 2, 3))
    for i, qpos in enumerate(native["qpos"]):
        data.qpos[:] = qpos
        # These diagnostics require FK only. Do not reconstruct contact forces
        # from saved states; exact force evidence belongs to the fresh replay.
        mujoco.mj_kinematics(model, data)
        centers[i] = [data.geom_xpos[ids].mean(axis=0) for ids in tip_ids]
    rows, phase_duration, phase_runs = [], {}, []
    previous_time = float(t[0])
    for entry in history:
        end = float(entry["time_s"])
        kind = entry["event_id"].split(":")[0]
        hand = entry.get("selected_hand")
        obj = entry.get("selected_object_id")
        phase_duration[kind] = phase_duration.get(kind, 0.) + max(0., end - previous_time)
        key = [kind, hand, obj]
        if not phase_runs or phase_runs[-1]["phase"] != key:
            phase_runs.append({"phase": key, "start_s": previous_time, "end_s": end})
        else:
            phase_runs[-1]["end_s"] = end
        indices = np.flatnonzero((t >= previous_time - 1e-8) & (t <= end + 1e-8))
        if len(indices) > 1 and hand in ("left", "right"):
            hi = ("left", "right").index(hand)
            points = centers[indices, hi]
            path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            direct = float(np.linalg.norm(points[-1] - points[0]))
            row = {"phase": kind, "hand": hand, "object": obj,
                   "start_s": previous_time, "end_s": end,
                   "eef_path_m": path, "endpoint_displacement_m": direct,
                   "path_excess_over_endpoint_chord_m": max(0., path - direct)}
            if kind == "approach" and obj in ("cup_left_base", "cup_right_upper"):
                oi = ("cup_left_base", "cup_right_upper").index(obj)
                distance = np.linalg.norm(points - native["object_position"][indices, oi], axis=1)
                increases = np.maximum(np.diff(distance), 0.)
                row.update(eef_cup_start_distance_m=float(distance[0]),
                           eef_cup_end_distance_m=float(distance[-1]),
                           cumulative_distance_increase_m=float(increases.sum()))
            rows.append(row)
        previous_time = end
    arm_step = np.linalg.norm(np.diff(native["qpos"][:, qa], axis=0), axis=1)
    total_eef = np.linalg.norm(np.diff(centers, axis=0), axis=2).sum(axis=0)
    eef_velocity = np.gradient(centers, t, axis=0)
    slow = np.max(np.abs(velocity), axis=1) < .01
    return {"schema": "rssr.short_refine.native_motion_metrics.v1",
            "case": str(case), "duration_s": float(t[-1] - t[0]),
            "native_sample_count": len(t), "native_sample_interval_s": distribution(np.diff(t)),
            "command_count": len(commands["time_s"]),
            "command_interval_s": distribution(np.diff(commands["time_s"])),
            "whole_episode_command_cycles": command_cycles(commands["time_s"], commands["qpos"][:, hand_qa]),
            "arm_joint_velocity_abs_rad_s": distribution(np.abs(velocity)),
            "arm_joint_acceleration_abs_rad_s2": distribution(np.abs(acceleration)),
            "arm_joint_jerk_abs_rad_s3": distribution(np.abs(jerk)),
            "command_arm_joint_velocity_abs_rad_s": distribution(np.abs(command_velocity)),
            "command_arm_joint_acceleration_abs_rad_s2": distribution(np.abs(command_acceleration)),
            "per_actuated_joint_actual_and_command_motion": actuated,
            "arm_joint_path_l2_rad": float(arm_step.sum()),
            "eef_path_m": dict(zip(("left", "right"), map(float, total_eef))),
            "eef_speed_m_s": distribution(np.linalg.norm(eef_velocity, axis=-1)),
            "arm_slow_fraction_lt_0_01rad_s": float(np.mean(slow)),
            "arm_slow_runs_ge_0_2s": [r for r in runs(slow, t) if r["duration_s"] >= .2 - 1e-8],
            "labelled_phase_duration_s": phase_duration, "merged_phase_runs": phase_runs,
            "per_event_eef_path": rows,
            "scope": "Actual sampled native qpos/qvel and FK; event labels are controller annotations",
            "classification_policy": "Jerk, acceleration, path excess, slow runs and distance reversals are diagnostic only; obstacles, contact and final stability can require them. Physical acceptance uses independent task/hard/first-grasp/release/force gates."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = metrics(args.case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"duration_s": report["duration_s"], "output": str(args.out)}))


if __name__ == "__main__":
    main()
