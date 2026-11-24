"""Read-only FK contact screening for a declared grasp-stage command window."""
from __future__ import annotations

import math

import mujoco
import numpy as np


def gripper_table_distance(model: mujoco.MjModel, data: mujoco.MjData,
                           qpos: np.ndarray, hand: str) -> float:
    """Return the most negative target gripper/table contact distance, or 0."""
    if hand not in ("left", "right") or qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("Invalid grasp clearance command")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    deepest = 0.
    gripper = "astribot_gripper_" + hand
    for i in range(data.ncon):
        contact = data.contact[i]
        a = model.geom(contact.geom1).name or ""
        b = model.geom(contact.geom2).name or ""
        if ((gripper in a and "worktable" in b)
                or (gripper in b and "worktable" in a)):
            deepest = min(deepest, float(contact.dist))
    return deepest


def sample_window(model: mujoco.MjModel, stream: tuple,
                  hand: str, start_s: float, end_s: float,
                  *, step_s: float = .05) -> dict:
    """Sample the full declared stage at fixed times, using native commands."""
    times, qpos = map(np.asarray, stream[:2])
    if (times.ndim != 1 or qpos.shape != (len(times), model.nq)
            or len(times) < 2 or not np.isfinite(times).all()
            or not np.isfinite(qpos).all() or not np.all(np.diff(times) > 0)
            or not all(math.isfinite(x) for x in (start_s, end_s, step_s))
            or not times[0] <= start_s < end_s <= times[-1]
            or not .02 <= step_s <= .2):
        raise ValueError("Invalid grasp clearance stage or command clock")
    data = mujoco.MjData(model)
    count = int(math.ceil((end_s-start_s)/step_s))+1
    samples = np.linspace(start_s, end_s, count)
    minimum = 0.
    worst = float(start_s)
    for when in samples:
        index = min(len(times)-2, max(0, int(np.searchsorted(times, when, side="right")-1)))
        alpha = float(np.clip((when-times[index])/(times[index+1]-times[index]), 0., 1.))
        command = (1.-alpha)*qpos[index]+alpha*qpos[index+1]
        distance = gripper_table_distance(model, data, command, hand)
        if distance < minimum:
            minimum, worst = distance, float(when)
    return {"hand": hand, "start_s": float(start_s), "end_s": float(end_s),
            "step_s": float(step_s), "sample_count": len(samples),
            "minimum_gripper_table_signed_distance_m": minimum,
            "worst_time_s": worst}


def compare_grasp_window(model: mujoco.MjModel, candidate: tuple,
                         parent: tuple, probe: dict) -> dict:
    """Screen an asserted table-clearance mechanism before independent review."""
    if not isinstance(probe, dict) or set(probe) not in (
            {"hand", "start_s", "end_s"},
            {"hand", "start_s", "end_s", "object", "close_s"}):
        raise ValueError("Grasp clearance probe must declare hand and exact stage window")
    kwargs = {"hand": probe["hand"], "start_s": float(probe["start_s"]),
              "end_s": float(probe["end_s"])}
    before = sample_window(model, parent, **kwargs)
    after = sample_window(model, candidate, **kwargs)
    old = before["minimum_gripper_table_signed_distance_m"]
    new = after["minimum_gripper_table_signed_distance_m"]
    if old > -.005:
        raise ValueError("Parent lacks material commanded gripper/table penetration")
    if new < -.003 or new-old < .008:
        raise ValueError(f"Candidate did not clear declared table collision: {old:.5f} -> {new:.5f} m")
    alignment = None
    if "object" in probe:
        from .grasp_alignment import compare_alignment
        alignment = compare_alignment(model, candidate, parent,
            hand=probe["hand"], object_id=probe["object"], close_s=float(probe["close_s"]))
    return {"schema": "rssr.family.grasp_clearance_preflight.v1",
            "parent": before, "candidate": after,
            "improvement_m": new-old,
            "grasp_alignment": alignment,
            "static_only": True,
            "physical_contact_and_task_success_unproven": True}
