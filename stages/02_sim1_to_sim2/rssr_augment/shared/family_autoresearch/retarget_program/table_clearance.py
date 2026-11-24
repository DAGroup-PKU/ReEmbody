"""Shared acquisition-stage clearance rule for commanded gripper/table collision."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from ..clearance import sample_window
from ..store import file_sha


VERSION = "grasp_table_clearance_v1"


def infer_clearance_window(*, close_s: float, release_s: float,
                           duration_s: float, penetration_m: float,
                           hand: str) -> dict:
    if hand not in ("left", "right") or not all(math.isfinite(v) for v in
            (close_s, release_s, duration_s, penetration_m)):
        raise ValueError("Invalid grasp clearance event")
    if penetration_m > -.005:
        raise ValueError("No material commanded gripper/table collision")
    start = close_s-1.45
    rise_end = close_s-1.0
    hold_end = close_s+.45
    end = min(close_s+1.15, release_s-.1, duration_s-.001)
    if start < .001 or min(rise_end-start, hold_end-rise_end, end-hold_end) < .05:
        raise ValueError("No complete clearance stage within the source grasp cycle")
    displacement = math.ceil(1000*(-penetration_m+.006))/1000
    if displacement > .035:
        raise ValueError("Required clearance exceeds the bounded grasp-stage correction")
    return {
        "probe_start_s": round(rise_end, 7),
        "probe_end_s": round(hold_end, 7),
        "source_command_penetration_m": round(penetration_m, 7),
        "command": {"operation": "motion_window", "hand": hand,
                    "delta_m": [0., 0., round(displacement, 4)],
                    "start_s": round(start, 7),
                    "rise_end_s": round(rise_end, 7),
                    "hold_end_s": round(hold_end, 7),
                    "end_s": round(end, 7)},
    }


def instantiate(packet: dict, index_case: dict) -> dict:
    """Use measured source event timing and actual target command FK, no labels."""
    if packet.get("family") != "stack_blocks" or packet.get("stage") != "acquisition":
        raise ValueError("This adapter applies to block acquisition")
    spec = packet.get("target_spec", {})
    hand = spec.get("hand")
    if spec.get("kind") != "lift_with_contact" or hand not in ("left", "right"):
        raise ValueError("No bound failed block lift")
    report = packet["parent"]["report"]
    events = report.get("source_grasp_events", {}).get(hand, [])
    if not events:
        raise ValueError("No source grasp event for bound hand")
    close_s, release_s = float(events[0]["closed_s"]), float(events[0]["release_s"])
    report_path = Path(packet["parent"]["report_path"])
    command_path = report_path.parent / "command_targets.npz"
    model_path = report_path.parent / "scene.xml"
    files = (report_path, command_path, model_path)
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Frozen target command/model evidence is incomplete")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(command_path, allow_pickle=False) as commands:
        stream = (commands["time_s"], commands["qpos"])
        probe_start, probe_end = close_s-1.0, close_s+.45
        baseline = sample_window(model, stream, hand, probe_start, probe_end)
    result = infer_clearance_window(
        close_s=close_s, release_s=release_s,
        duration_s=float(report["simulation_seconds"]),
        penetration_m=baseline["minimum_gripper_table_signed_distance_m"],
        hand=hand)
    result.update(program_version=VERSION, hand=hand, stage="acquisition",
                  anchor_object=spec.get("object"),
                  baseline_static_clearance=baseline,
                  evidence_sha256={str(path): file_sha(path) for path in files},
                  claim_scope="static commanded collision hypothesis; full native rollout required")
    return result
