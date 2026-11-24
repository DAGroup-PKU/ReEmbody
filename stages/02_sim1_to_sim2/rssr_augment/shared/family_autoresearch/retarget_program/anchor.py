"""Shared grasp-stage anchor rule; output is a bounded diagnostic candidate.

The rule has no case ID or success-label input. It maps a source command tool
position relative to the manipulated object's source pose into the measured
target object frame at the grasp event. It does not claim a full retarget or
physical success; a compiled native rollout must test the candidate.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from rssr_augment.shared.family_autoresearch.store import file_sha, read_json


VERSION = "grasp_anchor_translation_v2"
# The native whole-stream guard measures fingertip geometry, not the tool site.
# A 2 mm command margin keeps IK/FK conversion from crossing its 4 cm bound.
BOUND_M = .038


def _rotation_from_wxyz(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid object quaternion")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _source_rotation(obj: dict) -> np.ndarray:
    yaw = float(obj.get("yaw_rad", 0.))
    pitch_roll = obj.get("roll_pitch_rad", [0., 0.])
    if len(pitch_roll) != 2:
        raise ValueError("Source object roll/pitch must have two components")
    roll, pitch = map(float, pitch_roll)
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]]) @ \
           np.array([[cy,0,sy],[0,1.,0],[-sy,0,cy]]) @ \
           np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])


def _site_position(model: mujoco.MjModel, data: mujoco.MjData,
                   qpos: np.ndarray, hand: str) -> np.ndarray:
    if qpos.shape != (model.nq,):
        raise ValueError("Command qpos does not match model")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site(f"arm_{hand}_tool").id].copy()


def map_tool_at_stage_start(source_anchor: np.ndarray, source_rotation: np.ndarray,
                            source_tool: np.ndarray, target_anchor: np.ndarray,
                            target_rotation: np.ndarray) -> np.ndarray:
    """T_WA(target) @ inverse(T_WA(source)) @ tool(source), position part."""
    inputs = (source_anchor, source_rotation, source_tool, target_anchor, target_rotation)
    if not all(np.isfinite(value).all() for value in inputs):
        raise ValueError("Non-finite anchor transform")
    if source_rotation.shape != (3, 3) or target_rotation.shape != (3, 3):
        raise ValueError("Anchor rotations must be 3x3")
    return target_anchor + target_rotation @ source_rotation.T @ (source_tool-source_anchor)


def grasp_anchor_delta(*, source_scene: dict, source_model_path: Path,
                       source_commands: dict, target_scene: dict,
                       target_model_path: Path, target_commands: dict,
                       target_rollout: dict, grasp_events: dict,
                       task_contract: dict, target_predicate: str,
                       source_rollout: dict | None = None) -> dict:
    """Return an event-offset suggestion from observed geometry, without labels.

    The source object's initial pose is the acquisition-stage anchor. The
    target object's pose is sampled at the target grasp event. A carried object
    needs a separately planned object trajectory; this rule stops at grasp.
    """
    lifts = [p for p in task_contract.get("predicates", [])
             if p.get("kind") == "lift_with_contact" and p.get("id") == target_predicate]
    if not lifts:
        raise ValueError("No failed, bound lift predicate for grasp anchor rule")
    spec = lifts[0]
    object_id, hand = spec.get("object"), spec.get("hand")
    if hand not in ("left", "right"):
        raise ValueError("Lift predicate lacks a bound hand")
    objects_src = {obj["id"]: obj for obj in source_scene["objects"]}
    objects_dst = [obj["id"] for obj in target_scene["objects"]]
    if object_id not in objects_src or object_id not in objects_dst:
        raise ValueError("Bound object missing from source or target scene")
    if not grasp_events.get(hand):
        raise ValueError("No source grasp cycle for bound hand")
    cycle = 0
    when = float(grasp_events[hand][cycle]["closed_s"])
    source_times = np.asarray(source_commands["time_s"], dtype=float)
    target_times = np.asarray(target_commands["time_s"], dtype=float)
    physical_times = np.asarray(target_rollout["time_s"], dtype=float)
    if not all(len(t) >= 2 and np.isfinite(t).all() and np.all(np.diff(t) > 0)
               for t in (source_times, target_times, physical_times)):
        raise ValueError("Invalid source/target time base")
    si = int(np.argmin(np.abs(source_times-when)))
    ti = int(np.argmin(np.abs(target_times-when)))
    pi = int(np.argmin(np.abs(physical_times-when)))
    src_model = mujoco.MjModel.from_xml_path(str(source_model_path))
    dst_model = mujoco.MjModel.from_xml_path(str(target_model_path))
    src_tool = _site_position(src_model, mujoco.MjData(src_model),
                              np.asarray(source_commands["qpos"][si]), hand)
    dst_tool = _site_position(dst_model, mujoco.MjData(dst_model),
                              np.asarray(target_commands["qpos"][ti]), hand)
    src_obj = objects_src[object_id]
    source_anchor_basis = "scene_initial_pose"
    if source_rollout is not None:
        source_object_ids = list(objects_src)
        so = source_object_ids.index(object_id)
        sr = int(np.argmin(np.abs(np.asarray(source_rollout["time_s"])-when)))
        src_pos = np.asarray(source_rollout["object_position"][sr, so], dtype=float)
        src_rot = _rotation_from_wxyz(source_rollout["object_quaternion"][sr, so])
        source_anchor_basis = "measured_source_pose_at_grasp"
    else:
        src_pos = np.asarray(src_obj["position_m"], dtype=float)
        src_rot = _source_rotation(src_obj)
    oi = objects_dst.index(object_id)
    dst_pos = np.asarray(target_rollout["object_position"][pi, oi], dtype=float)
    dst_rot = _rotation_from_wxyz(target_rollout["object_quaternion"][pi, oi])
    target_tool = map_tool_at_stage_start(src_pos, src_rot, src_tool, dst_pos, dst_rot)
    full_delta = target_tool-dst_tool
    if not np.isfinite(full_delta).all():
        raise ValueError("Non-finite anchor mapping")
    bounded = np.clip(full_delta, -BOUND_M, BOUND_M)
    if np.max(np.abs(bounded)) < .00005:
        raise ValueError("Anchor displacement is below the allowed action resolution")
    return {
        "program_version": VERSION, "stage": "acquisition", "anchor_object": object_id,
        "hand": hand, "cycle": cycle, "stage_start_time_s": when,
        "source_anchor_basis": source_anchor_basis,
        "transform": "target_object_world @ inverse(source_object_world) @ source_tool_world",
        "source_tool_world_m": src_tool.round(6).tolist(),
        "source_anchor_world_m": src_pos.round(6).tolist(),
        "target_anchor_world_m": dst_pos.round(6).tolist(),
        "target_command_tool_world_m": dst_tool.round(6).tolist(),
        "mapped_tool_world_m": target_tool.round(6).tolist(),
        "unbounded_delta_m": full_delta.round(6).tolist(),
        "bounded_delta_m": bounded.round(6).tolist(),
        "clipped": bool(np.any(np.abs(full_delta) > BOUND_M)),
        "command": {"operation": "event_offset", "hand": hand, "cycle": cycle,
                    "event_phase": "grasp", "delta_m": bounded.round(6).tolist()},
        "claim_scope": "grasp waypoint diagnostic; full native rollout required",
    }


def instantiate(baseline_row: dict) -> dict:
    """Load hash-bound observations, then call the case-agnostic shared rule."""
    source_dir = Path(baseline_row["source_dir"])
    config_path = source_dir / "config.json"
    if file_sha(config_path) != baseline_row["file_sha256"][str(config_path)]:
        raise ValueError("Source config changed after baseline materialization")
    config = read_json(config_path)
    source_reference = Path(config["source_reference_artifact"]).resolve()
    source_scene_path = source_dir / "scene.json"
    if not source_scene_path.is_file():
        source_scene_path = source_reference.parent / "source_scene.json"
    source_model_path = source_reference / "scene.xml"
    source_command_path = source_reference / "command_targets.npz"
    source_rollout_path = source_reference / "rollout.npz"
    target_report_path = Path(baseline_row["seed_report_path"])
    target_report = read_json(target_report_path)
    target_model_path = target_report_path.parent / "scene.xml"
    target_command_path = target_report_path.parent / "command_targets.npz"
    target_rollout_path = target_report_path.parent / "rollout.npz"
    selection_path = Path(baseline_row["selection_path"])
    contract_path = source_dir / "sim2/task_contract.json"
    files = [source_scene_path, source_model_path, source_command_path,
             target_report_path, target_model_path, target_command_path,
             target_rollout_path, selection_path, contract_path]
    if source_rollout_path.is_file():
        files.append(source_rollout_path)
        source_report_path = source_reference / "report.json"
        if source_report_path.is_file():
            files.append(source_report_path)
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Anchor rule needs complete source and target traces")
    if file_sha(target_report_path) != baseline_row["file_sha256"][str(target_report_path)]:
        raise ValueError("Target report changed after baseline materialization")
    if file_sha(selection_path) != baseline_row["file_sha256"][str(selection_path)]:
        raise ValueError("Target selection changed after baseline materialization")
    with np.load(source_command_path, allow_pickle=False) as source_commands, \
         np.load(target_command_path, allow_pickle=False) as target_commands, \
         np.load(target_rollout_path, allow_pickle=False) as target_rollout:
        source_rollout = (np.load(source_rollout_path, allow_pickle=False)
                          if source_rollout_path.is_file() else None)
        try:
            if source_rollout is not None and source_report_path.is_file():
                source_report = read_json(source_report_path)
                source_ids = [obj["id"] for obj in read_json(source_scene_path)["objects"]]
                if source_rollout["object_position"].shape[1] != len(source_ids):
                    raise ValueError("Source object trace does not match source scene order")
                for index, object_id in enumerate(source_ids):
                    final = source_report.get("final_positions_m", {}).get(object_id)
                    if final is None or not np.allclose(source_rollout["object_position"][-1, index],
                                                        final, atol=1e-8, rtol=0):
                        raise ValueError("Source object order disagrees with native report")
            result = grasp_anchor_delta(
                source_scene=read_json(source_scene_path), source_model_path=source_model_path,
                source_commands=source_commands,
                target_scene=read_json(selection_path)["scene"],
                target_model_path=target_model_path, target_commands=target_commands,
                target_rollout=target_rollout,
                grasp_events=target_report["source_grasp_events"],
                task_contract=read_json(contract_path),
                target_predicate=baseline_row["target_predicate"],
                source_rollout=source_rollout,
            )
        finally:
            if source_rollout is not None:
                source_rollout.close()
    result["case_id"] = baseline_row["case_id"]
    result["evidence_sha256"] = {str(path): file_sha(path) for path in files}
    return result
