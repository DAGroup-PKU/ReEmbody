"""One compact GPT proposal when a deterministic local search has stalled.

The model never edits physics state or declares success. The caller applies the
validated delta to a copy of its incumbent, runs the normal scene/IK validators,
and measures the unchanged physical task predicates before accepting it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from rssr_augment.shared import responses_transport
from rssr_augment.common import SOURCE_ROOT


ROOT = SOURCE_ROOT
PROFILE_FILE = ROOT / "config/llm_profiles.json"
if not PROFILE_FILE.is_file():
    PROFILE_FILE = ROOT / 'sim1-sim2/archive/legacy_workspace_20260927/config/llm_profiles.json'
PROMPT_LIMIT = 16000
MAX_AXIS_M = 0.04
MAX_CUMULATIVE_OBJECT_XY_M = 0.10
MAX_ATTEMPTS = 3


class DecisionUnavailable(RuntimeError):
    """No valid model proposal was obtained; no physics trial was spent."""


PROMPT = """You are the task-level subagent for a dual-arm Sim1-to-Sim2 retarget experiment.
The normal bounded loop has stalled. Return exactly ONE JSON object, no prose.
Goal: improve the named physical task predicate while preserving passed hard
constraints. Absolute Sim1 world object coordinates are NOT a success target.
Use measured signed world-frame relation geometry, object-minus-hand position
and velocity residuals, achieved EE tracking, source grasp/release event times,
and prior physical responses. Distinguish grasp, carry, and release/placement.
Choose the least disruptive feasible lever: robot_event, robot_time_knot,
object_initial_xy, split, or gripper_window. The object lever ONLY shifts the INITIAL XY pose
of a listed free object; never set rollout qpos/qvel/ctrl or move a fixture,
change object geometry, task thresholds, or grasp/release command timing.
Return fields exactly: target_predicate, lever, diagnosis, prediction, robot,
object, gripper. robot is null for object_initial_xy or gripper_window;
otherwise it is an object with
hand (left/right), cycle (0-based source grasp cycle), phase (grasp/release),
delta_m ([world x,y,z] in metres), and time_s (null for robot_event; source
clock time for robot_time_knot). object is null for robot_event and
robot_time_knot; otherwise it is an object with object_id and delta_xy_m
([world x,y] metres). split uses robot_event or robot_time_knot behavior
according to whether time_s is null, AND an initial free-object XY shift.
gripper is null for every lever except gripper_window. For gripper_window,
robot and object are null, and gripper has hand, cycle, mode, start_s, end_s,
ramp_s, level. mode is gripper_pulse (opening floor, level 0..1) near source
closure or hold_release (opening cap, level 0..0.5) near source release.
The window must be <=1.5 s, start/end within 0.75 s of the named source event,
ramp_s 0.05..0.5 s, and end_s-start_s >= 2*ramp_s+0.05 s.
For every new robot or object edit, each axis magnitude must be <=0.04 m.
The sum of all free objects' cumulative XY displacements from their unmodified
baselines must remain <=0.10 m. Use a single failing predicate from failed_predicates.
Prediction must name the same predicate and state the expected direction of
its measured margin or truth value; diagnosis must cite signed axis, event,
and why the selected lever should help. If hard_failures remain, propose
only a task repair that does not worsen them; the caller will verify physics.
Do not claim that object-state residual reduction alone is task success."""


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value: Any, size: int, bound: float, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must have {size} numeric entries")
    result = [_finite_number(item, name) for item in value]
    if any(abs(item) > bound + 1e-12 for item in result):
        raise ValueError(f"{name} exceeds {bound} m per axis")
    return result


def _free_objects(observation: dict, allowed_object_ids: set[str]) -> dict[str, dict]:
    rows = observation.get("free_objects")
    if not isinstance(rows, list):
        raise ValueError("free_objects must be a compact list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id", "mobility", "position_m", "baseline_position_m"
        }:
            raise ValueError("free object row requires id, mobility, current and baseline pose")
        name = row["id"]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("invalid or duplicate free object id")
        if name not in allowed_object_ids or row["mobility"] != "free":
            raise ValueError("observation contains an unapproved or non-free object")
        _vector(row["position_m"], 3, 10.0, "object current position_m")
        _vector(row["baseline_position_m"], 3, 10.0, "object baseline position_m")
        result[name] = row
    return result


def validate_observation(observation: dict, allowed_object_ids: set[str], duration_s: float) -> dict:
    """Require only compact physical facts and never a full rollout/video."""
    duration_s = _finite_number(duration_s, "duration_s")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    required = {"case_id", "failed_predicates", "hard_failures", "phase",
                "source_events", "signed_relations", "object_hand_state",
                "ee_tracking", "free_objects", "recent_trials"}
    if set(observation) != required:
        raise ValueError("observation must contain only the compact physical evidence fields")
    if not isinstance(observation["case_id"], str) or not observation["case_id"]:
        raise ValueError("case_id is required")
    failures = observation["failed_predicates"]
    if not isinstance(failures, list) or not failures or any(not isinstance(x, str) or not x for x in failures):
        raise ValueError("failed_predicates must be a nonempty list of names")
    if len(set(failures)) != len(failures) or len(failures) > 24:
        raise ValueError("failed_predicates are duplicated or excessive")
    if not isinstance(observation["hard_failures"], list) or any(
            x not in failures for x in observation["hard_failures"]):
        raise ValueError("hard_failures must be named failed predicates")
    if observation["phase"] not in ("grasp", "carry", "release", "placement"):
        raise ValueError("phase must identify the failing event")
    events = observation["source_events"]
    if not isinstance(events, dict) or set(events) != {"left", "right"}:
        raise ValueError("source_events must list left and right hands")
    for hand, cycles in events.items():
        if not isinstance(cycles, list) or len(cycles) > 16:
            raise ValueError("too many source grasp cycles")
        for index, cycle in enumerate(cycles):
            if not isinstance(cycle, dict) or set(cycle) != {"closed_s", "release_s"}:
                raise ValueError("source event must have closed_s and release_s")
            close = _finite_number(cycle["closed_s"], "closed_s")
            release = _finite_number(cycle["release_s"], "release_s")
            if not 0 <= close < release <= duration_s:
                raise ValueError(f"invalid {hand} source event {index}")
    for key, limit in (("signed_relations", 24), ("object_hand_state", 16),
                       ("recent_trials", 8)):
        if not isinstance(observation[key], list) or len(observation[key]) > limit:
            raise ValueError(f"{key} must be a bounded list")
    if not isinstance(observation["ee_tracking"], dict):
        raise ValueError("ee_tracking must be an object")
    _free_objects(observation, allowed_object_ids)
    # Reject NaN and oversized evidence before either request logging or network.
    encoded = json.dumps(observation, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(PROMPT) + len(encoded) > PROMPT_LIMIT:
        raise ValueError("joint decision exceeds the compact 16k-character budget")
    return observation


def validate_proposal(proposal: dict, observation: dict,
                      allowed_object_ids: set[str], duration_s: float) -> dict:
    """Reject model output outside the four normal, bounded control levers."""
    validate_observation(observation, allowed_object_ids, duration_s)
    if not isinstance(proposal, dict) or set(proposal) != {
        "target_predicate", "lever", "diagnosis", "prediction", "robot", "object", "gripper"
    }:
        raise ValueError("proposal fields do not match the joint-action schema")
    predicate = proposal["target_predicate"]
    if predicate not in observation["failed_predicates"]:
        raise ValueError("target_predicate is not a currently failed named predicate")
    for key in ("diagnosis", "prediction"):
        value = proposal[key]
        if not isinstance(value, str) or not 8 <= len(value) <= 500:
            raise ValueError(f"{key} must be a concise, nonempty explanation")
    if predicate not in proposal["prediction"]:
        raise ValueError("prediction must name the chosen physical predicate")
    lever = proposal["lever"]
    if lever not in ("robot_event", "robot_time_knot", "object_initial_xy", "split", "gripper_window"):
        raise ValueError("unknown robot/object lever")
    robot = proposal["robot"]
    obj = proposal["object"]
    gripper = proposal["gripper"]
    if (lever in ("robot_event", "robot_time_knot", "split")) != (robot is not None):
        raise ValueError("robot edit is inconsistent with lever")
    if (lever in ("object_initial_xy", "split")) != (obj is not None):
        raise ValueError("object edit is inconsistent with lever")
    if (lever == "gripper_window") != (gripper is not None):
        raise ValueError("gripper edit is inconsistent with lever")
    if robot is not None:
        if not isinstance(robot, dict) or set(robot) != {"hand", "cycle", "phase", "delta_m", "time_s"}:
            raise ValueError("robot edit fields are invalid")
        hand, cycle, phase = robot["hand"], robot["cycle"], robot["phase"]
        if hand not in ("left", "right") or type(cycle) is not int or cycle < 0:
            raise ValueError("robot hand/cycle is invalid")
        events = observation["source_events"][hand]
        if cycle >= len(events) or phase not in ("grasp", "release"):
            raise ValueError("robot edit does not address a source event")
        delta = _vector(robot["delta_m"], 3, MAX_AXIS_M, "robot delta_m")
        if not any(delta):
            raise ValueError("robot edit must move a hand")
        time_s = robot["time_s"]
        if lever == "robot_event" or (lever == "split" and time_s is None):
            if time_s is not None:
                raise ValueError("event edit must have null time_s")
        else:
            time_s = _finite_number(time_s, "robot time_s")
            event = events[cycle]
            lo, hi = event["closed_s"], event["release_s"]
            if not 0.05 <= time_s <= duration_s - 0.05 or not lo - 0.5 <= time_s <= hi + 0.5:
                raise ValueError("time knot must be near its named source event")
            anchor = lo if phase == "grasp" else hi
            if abs(time_s - anchor) > 0.75:
                raise ValueError("time knot is too far from its named event")
    if obj is not None:
        if not isinstance(obj, dict) or set(obj) != {"object_id", "delta_xy_m"}:
            raise ValueError("object edit fields are invalid")
        free = _free_objects(observation, allowed_object_ids)
        object_id = obj["object_id"]
        if object_id not in free:
            raise ValueError("object edit does not name an approved free object")
        delta = _vector(obj["delta_xy_m"], 2, MAX_AXIS_M, "object delta_xy_m")
        if not any(delta):
            raise ValueError("object edit must move its free object")
        row = free[object_id]
        x = row["position_m"][0] + delta[0] - row["baseline_position_m"][0]
        y = row["position_m"][1] + delta[1] - row["baseline_position_m"][1]
        if math.hypot(x, y) > MAX_CUMULATIVE_OBJECT_XY_M + 1e-12:
            raise ValueError("object cumulative XY shift exceeds 0.10 m")
        total = sum(math.hypot(
            (item["position_m"][0] + (delta[0] if name == object_id else 0)
             - item["baseline_position_m"][0]),
            (item["position_m"][1] + (delta[1] if name == object_id else 0)
             - item["baseline_position_m"][1]))
            for name, item in free.items())
        if total > MAX_CUMULATIVE_OBJECT_XY_M + 1e-12:
            raise ValueError("all free objects' cumulative XY shifts exceed 0.10 m")
    if gripper is not None:
        if not isinstance(gripper, dict) or set(gripper) != {
            "hand", "cycle", "mode", "start_s", "end_s", "ramp_s", "level"
        }:
            raise ValueError("gripper window fields are invalid")
        hand, cycle, mode = gripper["hand"], gripper["cycle"], gripper["mode"]
        if hand not in ("left", "right") or type(cycle) is not int or cycle < 0:
            raise ValueError("gripper hand/cycle is invalid")
        events = observation["source_events"][hand]
        if cycle >= len(events) or mode not in ("gripper_pulse", "hold_release"):
            raise ValueError("gripper mode does not address a source event")
        start = _finite_number(gripper["start_s"], "gripper start_s")
        end = _finite_number(gripper["end_s"], "gripper end_s")
        ramp = _finite_number(gripper["ramp_s"], "gripper ramp_s")
        level = _finite_number(gripper["level"], "gripper level")
        anchor = events[cycle]["closed_s" if mode == "gripper_pulse" else "release_s"]
        if not (0 <= start < end <= duration_s and end - start <= 1.5 + 1e-12 and
                abs(start - anchor) <= 0.75 + 1e-12 and
                abs(end - anchor) <= 0.75 + 1e-12):
            raise ValueError("gripper window must stay within 0.75 s of its source event")
        if not (0.05 <= ramp <= 0.5 and end - start >= 2 * ramp + 0.05 - 1e-12):
            raise ValueError("gripper window duration/ramp violates semantic control bounds")
        upper = 1.0 if mode == "gripper_pulse" else 0.5
        if not 0 <= level <= upper:
            raise ValueError("gripper level violates pulse/hold opening bound")
    return proposal


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _profile() -> tuple[str, dict]:
    config = _read_json(PROFILE_FILE)
    name = os.environ.get("RSSR_JOINT_MODEL_PROFILE", "gpt6luna")
    profile = config["profiles"].get(name)
    if not profile or profile.get("llm_api") != "responses" or not profile.get("env_file"):
        raise RuntimeError("joint decision requires a configured Responses profile")
    return name, profile


def _client(profile: dict):
    # Use the existing profile's env-file convention, but never persist or print
    # any credential or provider error body. Importing OpenAI is delayed so
    # schema-only tests need no network client dependency.
    from openai import OpenAI

    env_file = Path(profile["env_file"])
    if not env_file.is_file():
        raise RuntimeError("configured Responses credentials file is missing")
    raw = subprocess.check_output(
        ["bash", "-c", 'set -a; source "$1"; env -0', "_", str(env_file)],
        stderr=subprocess.DEVNULL,
    )
    values = {}
    for entry in raw.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            if key in (b"KIMI_API_KEY", b"KIMI_BASE_URL"):
                values[key.decode()] = value.decode()
    if not values.get("KIMI_API_KEY") or not values.get("KIMI_BASE_URL"):
        raise RuntimeError("configured Responses credentials are incomplete")
    return OpenAI(api_key=values["KIMI_API_KEY"], base_url=values["KIMI_BASE_URL"],
                  timeout=90, max_retries=0)


def decide(request_dir: Path, observation: dict,
           allowed_object_ids: set[str], duration_s: float) -> dict:
    """Persist/replay one bounded decision; no physics or scene mutation here."""
    allowed = set(allowed_object_ids)
    validate_observation(observation, allowed, duration_s)
    name, profile = _profile()
    request_dir = Path(request_dir)
    payload = {"schema": "rssr.joint_task_subagent.request.v1", "profile": name,
               "model": profile["model"], "prompt": PROMPT,
               "observation": observation, "allowed_object_ids": sorted(allowed),
               "duration_s": float(duration_s), "max_output_tokens": 1536,
               "reasoning_effort": "medium"}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    request_path = request_dir / "request.json"
    saved_path = request_dir / "decision.json"
    if request_path.exists():
        receipt = _read_json(request_path)
        if receipt != {"sha256": digest, **payload}:
            raise RuntimeError("saved joint decision request differs from current evidence")
    else:
        _write_json(request_path, {"sha256": digest, **payload})
    if saved_path.exists():
        saved = _read_json(saved_path)
        if saved.get("request_sha256") != digest:
            raise RuntimeError("saved joint proposal has a different request identity")
        return validate_proposal(saved["decision"], observation, allowed, duration_s)
    evidence = json.dumps(observation, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    client = _client(profile)
    messages = [{"role": "developer", "content": PROMPT},
                {"role": "user", "content": evidence}]
    for attempt in range(MAX_ATTEMPTS):
        api_request = {"request_sha256": digest, "attempt": attempt + 1,
                       "model": profile["model"], "messages": messages,
                       "response_format": {"type": "json_object"},
                       "max_output_tokens": 1536, "reasoning_effort": "medium",
                       "store": False, "stream": False}
        api_request_path = request_dir / f"api_request_{attempt:02d}.json"
        if api_request_path.exists():
            if _read_json(api_request_path) != api_request:
                raise RuntimeError("saved attempt request differs from resumed conversation")
        else:
            _write_json(api_request_path, api_request)
        answer_path = request_dir / f"response_{attempt:02d}.json"
        if answer_path.exists():
            answer = _read_json(answer_path)
            if answer.get("request_sha256") != digest:
                raise RuntimeError("saved provider response has a different request identity")
        else:
            try:
                answer = responses_transport.create(
                    client, {"model": profile["model"], "max_tokens": 1536,
                             "reasoning_effort": "medium"}, messages,
                    {"type": "json_object"})
            except Exception as exc:
                _write_json(request_dir / f"transport_{attempt:02d}.json",
                            {"request_sha256": digest, "error_type": type(exc).__name__,
                             "status_code": getattr(exc, "status_code", None)})
                raise DecisionUnavailable("Responses transport unavailable; no physics action") from exc
            # Adapter output has no authorization headers. Preserve the complete
            # text/usage/provider receipt before parsing so resumes are exact.
            _write_json(answer_path, {"request_sha256": digest, **answer})
        try:
            if answer.get("finish_reason") != "stop":
                raise ValueError("provider response did not finish cleanly")
            proposal = validate_proposal(json.loads(answer["content"]),
                                         observation, allowed, duration_s)
            _write_json(saved_path, {"request_sha256": digest, "decision": proposal,
                                     "provider_response_id": answer.get("provider_response_id"),
                                     "provider_model": answer.get("provider_model"),
                                     "usage": answer.get("usage"), "attempt": attempt + 1})
            return proposal
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            error = str(exc)[:250]
            _write_json(request_dir / f"validation_{attempt:02d}.json",
                        {"request_sha256": digest, "error": error})
            messages.extend([{"role": "assistant", "content": answer.get("content", "")},
                             {"role": "user", "content":
                              "Local bounds/schema rejected that proposal: " + error +
                              ". Return one corrected JSON object for the original evidence."}])
    raise DecisionUnavailable("no locally valid joint robot/object action after bounded attempts")
