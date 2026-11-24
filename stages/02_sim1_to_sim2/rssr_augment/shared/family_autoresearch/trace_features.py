"""Hash-bound command, actual tool, contact and lift timeline from a full rollout.

These features route experiments; they never replace native task predicates.
Unavailable or inconsistent traces are explicitly unattributed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .store import file_sha, read_json


def _nearest(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times-t)))


def summarize(report: dict, report_path, contract: dict, target_predicate: str) -> dict:
    path = Path(report_path).resolve()
    physics = path.parent
    files = [path, physics / "rollout.npz", physics / "command_targets.npz",
             physics / "scene.xml", physics / "task_contract.json"]
    result = {"status": "unattributed", "first_observed_gap": "unattributed",
              "events": [], "evidence_refs": [], "evidence_sha256": {}}
    try:
        if any(not item.is_file() for item in files):
            raise FileNotFoundError("Full report/rollout/command/scene/contract bundle is incomplete")
        if read_json(path) != report:
            raise ValueError("Report differs from saved full evidence")
        physical_contract = read_json(physics / "task_contract.json")
        if {p["id"] for p in physical_contract["predicates"]} != {p["id"] for p in contract["predicates"]}:
            raise ValueError("Executed and frozen predicate sets differ")
        scene_path = physics.parent / "proposal.json"
        if not scene_path.is_file():
            # Historical pre-v80 reports may not retain a proposal; leave them
            # unattributed rather than guessing object order.
            raise FileNotFoundError("Executed proposal with scene order is absent")
        scene = read_json(scene_path)["scene"]
        object_ids = [o["id"] for o in scene["objects"]]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Duplicate executed scene object ID")
        files.append(scene_path)
        with np.load(physics / "rollout.npz", allow_pickle=False) as actual, \
             np.load(physics / "command_targets.npz", allow_pickle=False) as command:
            times = np.asarray(actual["time_s"], dtype=float)
            commands = np.asarray(command["time_s"], dtype=float)
            positions = np.asarray(actual["object_position"], dtype=float)
            contacts = np.asarray(actual["contacts"])
            if (len(times) < 2 or len(commands) < 2 or np.any(np.diff(times) <= 0)
                    or np.any(np.diff(commands) <= 0) or not np.isfinite(times).all()
                    or positions.shape != (len(times), len(object_ids), 3)
                    or contacts.shape != positions.shape):
                raise ValueError("Invalid full trace shape or time base")
            for index, object_id in enumerate(object_ids):
                final = report.get("final_positions_m", {}).get(object_id)
                if final is None or not np.allclose(positions[-1, index], final, atol=1e-8, rtol=0):
                    raise ValueError("Object order differs from terminal report")
            import mujoco
            model = mujoco.MjModel.from_xml_path(str(physics / "scene.xml"))
            data = mujoco.MjData(model)
            for spec in contract["predicates"]:
                if spec.get("kind") != "lift_with_contact":
                    continue
                hand, object_id = spec.get("hand"), spec.get("object")
                if hand not in ("left", "right") or object_id not in object_ids:
                    continue
                oi, hi = object_ids.index(object_id), ("left", "right").index(hand)
                site = model.site(f"arm_{hand}_tool").id
                for cycle, event in enumerate(report.get("source_grasp_events", {}).get(hand, [])):
                    close, release = float(event["closed_s"]), float(event["release_s"])
                    if not times[0] <= close < release <= times[-1] + np.max(np.diff(times)):
                        continue
                    ri, ci = _nearest(times, close), _nearest(commands, close)
                    tool = []
                    for qpos in (actual["qpos"][ri], command["qpos"][ci]):
                        if qpos.shape != (model.nq,):
                            raise ValueError("Tool FK qpos differs from executed model")
                        data.qpos[:] = qpos
                        mujoco.mj_forward(model, data)
                        tool.append(data.site_xpos[site].copy())
                    mask = (times >= close) & (times <= release)
                    touched = np.flatnonzero(mask & (contacts[:, oi, hi] > 0))
                    z = positions[:, oi, 2]
                    row = {"predicate": spec["id"], "object": object_id, "hand": hand,
                           "cycle": cycle, "close_s": close, "release_s": release,
                           "actual_tool_world_m": tool[0].round(6).tolist(),
                           "command_tool_world_m": tool[1].round(6).tolist(),
                           "tool_tracking_error_m": round(float(np.linalg.norm(tool[0]-tool[1])), 6),
                           "first_contact_s": float(times[touched[0]]) if len(touched) else None,
                           "last_contact_s": float(times[touched[-1]]) if len(touched) else None,
                           "contact_fraction": round(float(np.mean(contacts[mask, oi, hi] > 0)), 6),
                           "max_lift_m": round(float(np.max(z[mask]-z[0])), 6)}
                    result["events"].append(row)
        bound = [x for x in result["events"] if x["predicate"] == target_predicate]
        if bound:
            first = min(bound, key=lambda x: x["close_s"])
            if first["tool_tracking_error_m"] > .02:
                result["first_observed_gap"] = "tool_tracking_at_grasp"
            elif first["first_contact_s"] is None:
                result["first_observed_gap"] = "hand_object_contact"
            elif first["max_lift_m"] < .025:
                result["first_observed_gap"] = "lift_after_contact"
            else:
                result["first_observed_gap"] = "downstream_task"
        else:
            result["first_observed_gap"] = "no_bound_grasp_event"
        result["status"] = "complete"
        result["evidence_refs"] = [str(item) for item in files]
        result["evidence_sha256"] = {str(item): file_sha(item) for item in files}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["diagnostic"] = {"reason": type(exc).__name__, "detail": str(exc)[:300]}
    return result
