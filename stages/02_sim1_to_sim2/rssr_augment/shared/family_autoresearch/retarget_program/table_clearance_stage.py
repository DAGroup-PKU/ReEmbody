"""Versioned full grasp-stage pose correction from commanded table penetration."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from ..grasp_alignment import sample_alignment
from .table_clearance import instantiate as measure_parent


VERSION = "grasp_table_clearance_stage_v3"
STAGE_WORLD_AXIS_BOUND_M = .100


def instantiate(packet: dict, index_case: dict) -> dict:
    parent = measure_parent(packet, index_case)
    depth = -float(parent["source_command_penetration_m"])
    # These are versioned robot/table morphology priors.  The depth is measured
    # from this case's command/model, not looked up from its ID or pass label.
    vertical = round(min(.065, max(.045, math.ceil(2.7*depth*1000)/1000)), 3)
    if vertical >= STAGE_WORLD_AXIS_BOUND_M:
        raise ValueError("Stage displacement exceeds independent safety boundary")
    command = parent["command"]
    report_path = Path(packet["parent"]["report_path"])
    model = mujoco.MjModel.from_xml_path(str(report_path.parent / "scene.xml"))
    with np.load(report_path.parent / "command_targets.npz", allow_pickle=False) as commands:
        stream = (commands["time_s"], commands["qpos"])
        close_s = float(packet["parent"]["report"]["source_grasp_events"][parent["hand"]][0]["closed_s"])
        alignment = sample_alignment(model, stream, parent["hand"], parent["anchor_object"], close_s)
    projection = alignment["object_closing_axis_projection_m"]
    axis = np.asarray(alignment["closing_axis_world"])
    planar_norm_sq = float(np.dot(axis[:2], axis[:2]))
    if planar_norm_sq < .5:
        raise ValueError("Closing axis is unsuitable for planar grasp centering")
    planar = .75*projection*axis[:2]/planar_norm_sq
    if np.max(np.abs(planar)) > .035:
        raise ValueError("Planar grasp centering exceeds bounded morphology prior")
    parent.update(program_version=VERSION, stage_world_axis_bound_m=STAGE_WORLD_AXIS_BOUND_M,
                  close_s=close_s, parent_grasp_alignment=alignment,
                  morphology_rule="vertical=clamp(ceil(2.7*static_depth_mm),45,65) mm; world-x pitch=20 deg; planar=0.75*closing_axis_projection*axis_xy/norm(axis_xy)^2",
                  command={"operation": "stage_motion_window", "hand": parent["hand"],
                           "delta_m": [round(float(planar[0]), 4), round(float(planar[1]), 4), vertical],
                           "rotation_degrees": [20., 0., 0.],
                           **{key: command[key] for key in
                              ("start_s", "rise_end_s", "hold_end_s", "end_s")}},
                  claim_scope="independent bounded stage preflight, then full native task rollout")
    return parent
