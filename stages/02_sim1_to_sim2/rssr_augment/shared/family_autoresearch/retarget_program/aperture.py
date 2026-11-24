"""Re-express a source pregrasp aperture in the target gripper geometry.

This is a grasp-stage hypothesis, not a task-success claim.  It reads measured
source opening and the actual target command geometry.  It does not depend on
case numbers, pass labels, or a per-case coordinate table.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..store import file_sha, read_json


VERSION = "pregrasp_aperture_reexpression_v1"
_CORNERS = np.asarray([[x, y, z] for x in (-1., 1.)
                       for y in (-1., 1.) for z in (-1., 1.)])


def _command_gap(model: mujoco.MjModel, data: mujoco.MjData,
                 qpos: np.ndarray, hand: str) -> float:
    """Use the same distal-pad vertex clearance as the frozen command adapter."""
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("Target command does not match the frozen model")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    body = model.body(f"astribot_gripper_{hand}_base").id
    rotation = data.xmat[body].reshape(3, 3)
    position = data.xpos[body]
    vertices = []
    for suffix in ("L112", "R112"):
        geom = model.geom(f"astribot_gripper_{hand}_geom_{suffix}").id
        world = (_CORNERS * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(3, 3).T
        world += data.geom_xpos[geom]
        vertices.append((world-position) @ rotation)
    return float(vertices[0][:, 0].min()-vertices[1][:, 0].max())


def infer_pregrasp_window(*, source_times: np.ndarray, source_opening: np.ndarray,
                          target_times: np.ndarray, target_normalized: np.ndarray,
                          target_command_gap_m: float, close_s: float,
                          object_width_m: float, duration_s: float,
                          hand: str) -> dict:
    """Infer one bounded pulse from observed morphology and an event clock."""
    for times, values in ((source_times, source_opening),
                          (target_times, target_normalized)):
        if (times.ndim != 1 or values.shape != times.shape or len(times) < 2
                or not np.isfinite(times).all() or not np.isfinite(values).all()
                or not np.all(np.diff(times) > 0)):
            raise ValueError("Invalid source or target gripper time series")
    if not all(np.isfinite(v) for v in (target_command_gap_m, close_s,
                                       object_width_m, duration_s)):
        raise ValueError("Nonfinite aperture hypothesis")
    if hand not in ("left", "right"):
        raise ValueError("Unknown grasp hand")
    sample_s = close_s-.8
    start_s, end_s, ramp_s = close_s-1.25, close_s-.05, .15
    if sample_s < source_times[0] or sample_s > min(source_times[-1], target_times[-1]):
        raise ValueError("No matched pregrasp observation")
    if start_s < .001 or end_s > duration_s-.001 or end_s-start_s < 2*ramp_s+.05:
        raise ValueError("Grasp pulse leaves the complete episode")
    source_gap = float(np.interp(sample_s, source_times, source_opening))
    current_norm = float(np.interp(sample_s, target_times, target_normalized))
    if not (.1 <= current_norm <= .9 and 0 < target_command_gap_m < object_width_m+.004
            and source_gap > object_width_m+.008
            and source_gap-target_command_gap_m >= .012):
        raise ValueError("No measured pregrasp aperture mismatch of this kind")
    # The target command adapter is nonlinear in joint angle, but its declared
    # normalized opening has an approximately linear physical-gap meaning here.
    # Limit the hypothesis to a reachable normalized command; native preflight
    # verifies its exact compiled command geometry and all-joint envelope.
    floor = round(min(.98, current_norm*(source_gap+.002)/target_command_gap_m), 4)
    if floor <= current_norm+.05:
        raise ValueError("Proposed pulse has negligible command effect")
    return {"sample_time_s": round(sample_s, 7),
            "source_actual_gap_m": round(source_gap, 7),
            "target_command_gap_m": round(target_command_gap_m, 7),
            "object_width_m": round(object_width_m, 7),
            "target_source_normalized_opening": round(current_norm, 7),
            "target_opening_floor": floor,
            "command": {"operation": "gripper_pulse", "hand": hand,
                        "start_s": round(start_s, 7), "end_s": round(end_s, 7),
                        "ramp_s": ramp_s, "opening_floor": floor}}


def instantiate(packet: dict, index_case: dict) -> dict:
    """Bind a failed block lift to source and target traces, then infer a pulse."""
    if packet.get("family") != "stack_blocks" or packet.get("stage") != "acquisition":
        raise ValueError("This adapter applies to block acquisition")
    spec = packet.get("target_spec", {})
    hand, object_id = spec.get("hand"), spec.get("object")
    if spec.get("kind") != "lift_with_contact" or hand not in ("left", "right"):
        raise ValueError("No bound failed block lift")
    report = packet["parent"]["report"]
    events = report.get("source_grasp_events", {}).get(hand, [])
    if not events:
        raise ValueError("No source grasp event for the bound hand")
    if packet["parent"]["candidate"].get("windows", {}).get(hand):
        raise ValueError("Existing gripper window requires separate attribution")
    close_s = float(events[0]["closed_s"])
    source_dir = Path(index_case["source_dir"])
    config_path = source_dir / "config.json"
    selection_path = source_dir / "sim2/selection.json"
    config, selection = read_json(config_path), read_json(selection_path)
    objects = {row["id"]: row for row in selection["scene"]["objects"]}
    obj = objects.get(object_id)
    if obj is None or obj.get("shape") != "box":
        raise ValueError("Bound block has no box geometry")
    width = float(min(obj["size_m"][:2]))
    source_root = Path(config["source_reference_artifact"])
    source_rollout_path = source_root / "rollout.npz"
    target_report_path = Path(packet["parent"]["report_path"])
    target_root = target_report_path.parent
    target_command_path = target_root / "command_targets.npz"
    target_model_path = target_root / "scene.xml"
    files = (config_path, selection_path, source_rollout_path,
             target_report_path, target_command_path, target_model_path)
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Pregrasp aperture evidence is incomplete")
    with np.load(source_rollout_path, allow_pickle=False) as src, \
         np.load(target_command_path, allow_pickle=False) as tgt:
        target_times = np.asarray(tgt["time_s"], dtype=float)
        index = int(np.argmin(np.abs(target_times-(close_s-.8))))
        model = mujoco.MjModel.from_xml_path(str(target_model_path))
        gap = _command_gap(model, mujoco.MjData(model),
                           np.asarray(tgt["qpos"][index], dtype=float), hand)
        result = infer_pregrasp_window(
            source_times=np.asarray(src["time_s"], dtype=float),
            source_opening=np.asarray(src["gripper_opening"][:, 0 if hand == "left" else 1], dtype=float),
            target_times=target_times,
            target_normalized=np.asarray(tgt[hand+"_gripper_command"], dtype=float),
            target_command_gap_m=gap, close_s=close_s,
            object_width_m=width, duration_s=float(report["simulation_seconds"]),
            hand=hand)
    result.update(program_version=VERSION, anchor_object=object_id, hand=hand,
                  stage="acquisition", source_grasp_event_s=close_s,
                  evidence_sha256={str(path): file_sha(path) for path in files},
                  claim_scope="pregrasp opening diagnostic; complete native task rollout required")
    return result
