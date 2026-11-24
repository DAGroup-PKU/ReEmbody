import hashlib, json, sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
POLICY = {
    "schema": "rssr.drawer.luna_policy.v2",
    "initial": {
        "rank": "angle_then_motion",
        "fresh_candidate_count": 16,
        "source0_candidate_ids": {"left": "left_000", "right": "right_000"},
    },
    "handle_route": {
        "family": "front_midpoint", "front_offset_m": 0.04,
        "cone_deg": 20.0, "camera_speed": 0.30, "angular_speed": 0.8,
    },
    "block_route": {
        "family": "direct", "front_offset_m": 0.0,
        "cone_deg": 20.0, "camera_speed": 0.30, "angular_speed": 0.8,
    },
    "wait_mount": [0.82, 0.28, 0.22],
    "source_speed": 1.6,
    "grip_s": [0.4, 0.4, 0.4],
    "retime_labels": ["reach", "insert_transport", "block_retreat", "insert_lower", "handle_clear"],
    "retime_joint_speed_rad_s": 0.5,
    "insertion_route": {
        "family": "source_side", "side_gap_m": 0.12,
        "side_turns_deg": [0.0, -20.0],
    },
}

def _pinned_adapter():
    if str(ADAPTER_DIR) not in sys.path:
        sys.path.insert(0, str(ADAPTER_DIR))
    import drawer_execution_adapter as adapter
    return adapter

adapter = _pinned_adapter()

def run(S):
    return adapter.run(S, POLICY)

def grasps(S):
    return adapter.grasps(S)
