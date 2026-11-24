"""Read-only commanded grasp geometry at a frozen object pose."""
from __future__ import annotations

import mujoco
import numpy as np


def sample_alignment(model: mujoco.MjModel, stream: tuple, hand: str,
                     object_id: str, when: float) -> dict:
    times, qpos = map(np.asarray, stream[:2])
    if hand not in ("left", "right") or not object_id or not times[0] <= when <= times[-1]:
        raise ValueError("Invalid grasp alignment probe")
    i = int(np.clip(np.searchsorted(times, when, side="right")-1, 0, len(times)-2))
    alpha = float(np.clip((when-times[i])/(times[i+1]-times[i]), 0., 1.))
    data = mujoco.MjData(model)
    data.qpos[:] = (1.-alpha)*qpos[i]+alpha*qpos[i+1]
    mujoco.mj_forward(model, data)
    base = data.body(f"astribot_gripper_{hand}_base")
    tool = data.site(f"arm_{hand}_tool")
    obj = data.body(object_id)
    axis = base.xmat.reshape(3, 3)[:, 0]
    relative = obj.xpos-tool.xpos
    contacts = {"left": [], "right": []}
    for k in range(data.ncon):
        contact = data.contact[k]
        names = (model.geom(contact.geom1).name or "", model.geom(contact.geom2).name or "")
        if not any(object_id in name for name in names):
            continue
        pad = next((name for name in names if f"astribot_gripper_{hand}_geom_" in name), "")
        if "_geom_L" in pad:
            contacts["left"].append(float(contact.dist))
        elif "_geom_R" in pad:
            contacts["right"].append(float(contact.dist))
    return {"time_s": float(when), "object": object_id, "hand": hand,
            "object_minus_tool_m": relative.tolist(),
            "closing_axis_world": axis.tolist(),
            "object_closing_axis_projection_m": float(relative@axis),
            "pad_contact_counts": {side: len(values) for side, values in contacts.items()},
            "minimum_pad_object_signed_distance_m": {
                side: min(values, default=0.) for side, values in contacts.items()},
            "static_only": True}


def compare_alignment(model: mujoco.MjModel, candidate: tuple, parent: tuple,
                      *, hand: str, object_id: str, close_s: float,
                      require_bilateral: bool = True) -> dict:
    before = sample_alignment(model, parent, hand, object_id, close_s)
    after = sample_alignment(model, candidate, hand, object_id, close_s)
    preclose = sample_alignment(model, candidate, hand, object_id, close_s-.1)
    old = abs(before["object_closing_axis_projection_m"])
    new = abs(after["object_closing_axis_projection_m"])
    if old-new < .01 or new > .01:
        raise ValueError(f"Candidate has no centered grasp: {old:.5f} -> {new:.5f} m")
    if require_bilateral and not all(preclose["pad_contact_counts"].values()):
        raise ValueError("Candidate has no bilateral static pad/object contact before closure")
    return {"schema": "rssr.family.grasp_alignment_preflight.v1",
            "parent_at_close": before, "candidate_at_close": after,
            "candidate_before_close": preclose,
            "static_only": True, "achieved_contact_and_lift_unproven": True}
