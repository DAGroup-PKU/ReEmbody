"""Single-factor planar centering of a frozen target grasp command."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..grasp_alignment import sample_alignment
from ..store import file_sha


VERSION = "dominant_axis_grasp_centering_v5"


def instantiate(packet: dict, index_case: dict) -> dict:
    if packet.get("family") != "stack_blocks" or packet.get("stage") != "acquisition":
        raise ValueError("Planar centering applies to block acquisition")
    spec = packet.get("target_spec", {})
    hand, object_id = spec.get("hand"), spec.get("object")
    if spec.get("kind") != "lift_with_contact" or hand not in ("left", "right") or not object_id:
        raise ValueError("No bound failed object grasp")
    report_path = Path(packet["parent"]["report_path"])
    report = packet["parent"]["report"]
    events = report.get("source_grasp_events", {}).get(hand, [])
    if not events:
        raise ValueError("No source grasp event for bound hand")
    close_s = float(events[0]["closed_s"])
    release_s = float(events[0]["release_s"])
    command_path, scene_path = report_path.parent/"command_targets.npz", report_path.parent/"scene.xml"
    files = (report_path, command_path, scene_path)
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Missing frozen grasp command evidence")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    with np.load(command_path, allow_pickle=False) as z:
        alignment = sample_alignment(model, (z["time_s"], z["qpos"]), hand, object_id, close_s)
    projection = float(alignment["object_closing_axis_projection_m"])
    axis = np.asarray(alignment["closing_axis_world"], dtype=float)
    denominator = float(axis[:2]@axis[:2])
    if denominator < .5 or abs(projection) < .015:
        raise ValueError("No material correctable closing-axis error")
    dominant = int(np.argmax(np.abs(axis[:2])))
    offset = np.zeros(2, dtype=float)
    offset[dominant] = float(np.clip(.85*projection/axis[dominant], -.036, .036))
    start, rise, hold = close_s-1.45, close_s-1.0, close_s+.45
    end = min(close_s+1.15, release_s-.1, float(report["simulation_seconds"])-.001)
    if start < .001 or min(rise-start, hold-rise, end-hold) < .05:
        raise ValueError("No complete source grasp window")
    return {"program_version": VERSION, "hand": hand, "anchor_object": object_id,
            "stage": "acquisition", "close_s": close_s,
            "probe_start_s": round(rise, 7), "probe_end_s": round(hold, 7),
            "parent_grasp_alignment": alignment,
            "morphology_rule": "translate on dominant horizontal closing-axis component by clamp(0.85*projection/axis_component,-36,36) mm; compose the pulse through existing IK time knots; other translation axes, orientation and aperture stay frozen",
            "command": {"operation": "motion_window", "hand": hand,
                        "delta_m": [round(float(offset[0]), 3), round(float(offset[1]), 3), 0.],
                        "start_s": round(start, 7), "rise_end_s": round(rise, 7),
                        "hold_end_s": round(hold, 7), "end_s": round(end, 7)},
            "evidence_sha256": {str(p): file_sha(p) for p in files},
            "claim_scope": "single-factor planar command test; achieved contact and task success unproven"}
