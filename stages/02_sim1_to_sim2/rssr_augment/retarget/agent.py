"""LLM agent selects only the next measurable physical event."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from rssr_augment.shared.family_autoresearch.agents import (
    DecisionUnavailable, InvalidCompleteResponse, MAX_REQUEST_CHARS,
    SCHEMA, _json, _profile, _request,
)
from rssr_augment.shared.family_autoresearch.store import Store, digest, file_sha, pinned_json

from .events import EventDefinition, signed_margin
from .relational_event_policy import episode_memory


VERSION = "rssr.retarget.unified_event_agent.v35.luna_general_full150.v127_native_step_preview_seq342"
MODEL_PROFILE = {
    "label": "GPT-6 Luna via configured NVIDIA route",
    "env_file": os.environ.get("REEMBODY_LLM_ENV_FILE", ""),
    "llm_api": "responses",
    "model": os.environ.get("REEMBODY_LLM_MODEL", "gpt-5"),
}
MAX_OUTPUT_TOKENS = 4096
PROMPT = """You are the RSSR execution agent for the supplied physical task contract.
The operator library starts empty. A RuleAgent may revise declarative rules from this campaign's prior structurally matching cases and the current episode; inspect rule_context and structured episode_memory. Use physical relations and measured state, not task-family labels or a memorized scenario recipe. Never claim a previously tried mode is new. Select only the NEXT
measurable physical event from available_events. Use available_manipulators
and measured joint margins, reach and occupancy when assigning a manipulator.
The S1 chassis has no translation or yaw freedom. The executor may use at most
coupled waist motion within the original S1 joint ranges when arm-only
measured reach is insufficient; it retains native coupling and checks every physical step. Treat a
static reach estimate as a proposal, never as a task completion claim.
If rule_context contains physical_preview_risk, an isolated copy of the
current native state predicted non-target contact, poor loaded contact,
task progress, or a hard failure for the excluded event. Treat this as a
forecast, inspect its measurements, and select another currently available
event. The live state has not advanced.
With one available manipulator, evaluate its feasible modes; with several,
compare only contract-permitted assignments. A small margin is measured risk,
not proof that another assignment can succeed. Return one short JSON object:
{"event_id": exact available ID, "reason": short evidence-grounded diagnosis,
 "expected_signal": measurable event evidence, "abandon_condition": measurable
 reason to replan}. When the contract does not bind a manipulator, choose among supplied candidates using measured reach, contact and occupied manipulators. First approach with open fingers, then contact closes the jaws. When exactly one target finger touches, contact temporarily holds the wrist while closing; inspect whether the second finger joins and all physical hard checks stay valid. A nearby open hand can choose contact without previous finger contact. Never repeatedly approach after reaching the object. Wait for stable bilateral contact before lifting. A grasp event stabilizes contact without moving the arm. If another manipulator exists, its adapter compensates shared waist motion to hold its world pose. Lift only with measured bilateral contact. Respect the contract's dependency relations. Release only when measured support geometry is reached. Do not output a trajectory, direction, Cartesian delta,
 joint target, scene edit, success claim, or invented event. For a slide
 articulation, a supplied push_close event moves along its measured joint
 axis while keeping the current gripper command; it requires real fingertip
 contact and is checked again after a short segment. Judge its use from the
 current native close relation and observed contact, never from a task name.
 The deterministic
 short-horizon executor computes robot motion from fresh Sim2 relative error and
 observes again. Use Sim1 grasp times only as event-order reference. Current
 Sim2 contact, object motion, joint margin and native predicates govern the
 choice. On stagnation or contact loss, choose a different contact mode or reestablish contact before lifting. If an unsecured gripper touches a non-target object or fixture, inspect gripper_scene_contacts and consider a guarded retreat before another approach. When a securely carried object stalls while its gripper also touches another scene object, consider the guarded clearance event if available. Contact with the intended support can be useful; judge its role from the current predicate and motion evidence. Retreat and clearance follow a measured contact direction for a short physical segment and must be checked again; neither guarantees a collision-free route. Contact modes are executor-derived geometry choices, not Cartesian planning by you. Prefer keeping native task predicates already achieved. Do not claim
 completion from a partial event. Measured RuleAgent goal_progress separates a
 local effect from movement of unfinished native predicates; use it when
 selecting the next event, while allowing measured grasp and contact steps
 that are necessary prerequisites. episode_memory.frequent_collision_modes
 names exact modes and contacted objects that repeatedly blocked progress.
 A supplied overhead approach is an optional measured lift, translate and
 descend route for an open hand. An overhead_high variant ends at a slightly
 higher geometry-scaled contact region; consider it when the center endpoint
 repeatedly contacts a nearby support. Select either only from current
 geometry and collision evidence, then inspect its next segment.
 When closing a sliding feature repeatedly loses fingertip contact, a supplied
 guarded_slow event offers the same physical relation at lower hand speed.
 A supplied side_corridor approach stages the open hand just outside a
 measured static fixture side before an ordinary object approach. Consider
 it when the direct path loads intervening geometry; check the stage in
 physics, then select the object approach separately. The candidate does
 not imply that the later route or complete task will succeed.
 For a held object approaching an articulated container, supplied
 front_clearance and front_align events use the measured opening axis and
 container bounds to stage in front before the ordinary insertion event.
 Consider them when a direct loaded path stalls or contacts intervening
 geometry. A support_reserve variant may move a second, already gripping
 manipulator a bounded distance along that axis; use it only when measured
 opening drift threatens a required dependency. Check grasp, opening and
 hard constraints after each short segment, then choose the next event.
 Compare measured articulation progress and contact persistence after use;
 slower motion is only a hypothesis and must not replace reacquiring contact.
 If a measured opening margin is near its contract threshold before a
 dependent physical transition, a buffered opening candidate may provide a
 reserve against observed drift. Choose it only while the ordered transition
 has not occurred and recheck the measured fraction afterward.
 If rule_context blocks a relation event for ordered_prerequisite_reserve,
 restore the measured prerequisite with an available guarded event before
 trying that relation again. A spatial relation alone does not satisfy an
 ordered native transition.
 When a required spatial relation has become true, inspect the other
 predicates needed for completion. If support contact or release is still
 missing, establish that measured support before another lift. If the object
 is supported and released but an ordered final transition remains false,
 prioritize its next physical event. Revisit the object's spatial motion
 only if a native relation was lost or the remaining transition requires it.
 A higher contact surface is a geometry-scaled candidate when lower contact
 misses a pad or presses neighboring geometry. During any articulated motion,
 a bystander_clearance candidate moves an inactive hand along the measured
 joint axis to reduce opposition from moving geometry. Select these only from
 measured pad contact, collision force, or isolated physical preview evidence;
 inspect joint progress, loaded contact and every-step hard checks afterward.
 Independent replay and the frozen native
 judge alone certify success. Keep every text field under 240 characters.
"""


def _vector(value):
    return np.asarray(value, dtype=float).round(5).tolist()


def compact_observation(obs: dict) -> dict:
    return {"time_s": round(obs["time_s"], 4),
            "live_predicates": obs.get("live_predicates", {}),
            "failed_live_predicates": [k for k,v in obs.get("live_predicates", {}).items() if not v["passed"]],
            "object_position_m": {k: _vector(v) for k, v in
                                  obs["object_position_m"].items()},
            "object_velocity_m_s": {k: _vector(v) for k, v in
                                    obs["object_velocity_m_s"].items()},
            "finger_center_m": {k: _vector(v) for k, v in
                                obs["finger_center_m"].items()},
            "hand_rotation_world": {k: np.asarray(v, dtype=float).round(6).tolist()
                                    for k, v in obs["hand_rotation_world"].items()},
            "object_rotation_world": {k: np.asarray(v, dtype=float).round(6).tolist()
                                      for k, v in obs["object_rotation_world"].items()},
            "finger_contact": obs["finger_contact"],
            "finger_normal_force_n": {
                obj: {hand: [round(float(value), 4) for value in pads]
                      for hand, pads in hands.items()}
                for obj, hands in obs.get('finger_normal_force_n', {}).items()},
            "bilateral_contact_seconds": obs.get("bilateral_contact_seconds", {}),
            "hand_contact": obs.get("hand_contact", {}),
            "gripper_scene_contacts": obs.get("gripper_scene_contacts", {}),
            "gripper_scene_contact_object_ids": obs.get(
                "gripper_scene_contact_object_ids", {}),
            "support_contact": obs["support_contact"],
            "gripper_opening_m": obs["gripper_opening_m"],
            "articulation_fraction": obs["articulation_fraction"],
            "min_joint_margin_native": round(obs["min_joint_margin_native"], 6),
            "base_joint_qpos": _vector(obs['base_joint_qpos']),
            "torso_max_physical_excursion_rad": round(
                obs.get('torso_max_physical_excursion_rad', 0.), 6),
            "torso_max_command_excursion_rad": round(
                obs.get('torso_max_command_excursion_rad', 0.), 6),
            "limiting_joint": obs.get("limiting_joint"),
            "arm_torso_min_joint_margin_rad": round(obs["arm_torso_min_joint_margin_rad"], 6),
            "limiting_arm_torso_joint": obs.get("limiting_arm_torso_joint"),
            "gripper_min_joint_margin_rad": round(obs["gripper_min_joint_margin_rad"], 6),
            "limiting_gripper_joint": obs.get("limiting_gripper_joint"),
            "manipulator_joint_margin_rad": {
                manipulator: round(value, 6) for manipulator, value in
                obs["manipulator_joint_margin_rad"].items()},
            "limiting_manipulator_joint": obs["limiting_manipulator_joint"],
            "immutable_hard": obs.get("immutable_hard", {}),
            "max_robot_table_penetration_m": round(
                obs["max_robot_table_penetration_m"], 6)}


def validate_selection(raw: dict, by_id: dict) -> str:
    """Validate the executable event; retain nonexecuted explanation verbatim."""
    if not isinstance(raw, dict) or set(raw) != {
            "event_id", "reason", "expected_signal", "abandon_condition"}:
        raise ValueError("K3 event selection has invalid fields")
    event_id = raw["event_id"]
    if not isinstance(event_id, str) or event_id not in by_id:
        raise ValueError("K3 event selection is outside approved catalog")
    if any(not isinstance(raw[k], str) or not raw[k].strip()
           for k in ("reason", "expected_signal", "abandon_condition")):
        raise ValueError("K3 event selection has invalid explanation fields")
    return event_id


def decide_next(case: dict, obs: dict, initial: dict,
                available: list[EventDefinition], history: list[dict],
                store: Store, output: Path, rule_context: dict | None = None) -> EventDefinition:
    if not available:
        raise ValueError("No event candidates remain")
    case_id = case["case_id"]
    packet = {"schema": "rssr.retarget.event_packet.v1",
              "case_id": case_id,
              "task_predicates": case["contract"]["predicates"],
              "scene_objects": [{k:v for k,v in o.items() if k in ("id","shape","size_m","wall_m","role","parent_id","mobility","joint")} for o in case["scene"]["objects"]],
              "rule_context": rule_context or {},
              "available_manipulators": sorted({e.action.hand for e in available
                                                   if e.action.hand is not None}),
              "episode_memory": episode_memory(history),
              "observation": compact_observation(obs),
              "event_history": [{key: row[key] for key in (
                    'event_id', 'time_s', 'signed_margin', 'delta_margin',
                    'stop_reason', 'transition_warning', 'selected_object_id',
                    'selected_hand') if key in row}
                    for row in history[-6:]],
              "event_history_full_sha256": digest(history[-6:]),
              "available_events": [dict(e.as_packet(),
                                        signed_margin=round(signed_margin(
                                            e, obs, initial, case["contract"]), 5))
                                   for e in available]}
    # Keep the complete event history on disk in the episode receipt. The
    # model already receives current physics and structured episode memory;
    # repeating full contact snapshots in six past events can exceed the
    # provider's explicit request bound and silently halt online decisions.
    def request_chars(evidence):
        request = {'schema': SCHEMA, 'role': 'next_physical_event',
                   'profile': 'gpt6luna',
                   'model': _profile(MODEL_PROFILE)['model'],
                   'prompt': PROMPT, 'evidence': evidence,
                   'max_output_tokens': MAX_OUTPUT_TOKENS,
                   'reasoning_effort': 'medium'}
        return len(_json(request))

    if request_chars(packet) > MAX_REQUEST_CHARS - 8192:
        context = packet.get('rule_context', {})
        blocked = context.get('blocked_events')
        if isinstance(blocked, list):
            context['blocked_events_full_count'] = len(blocked)
            context['blocked_events_full_sha256'] = digest(blocked)
            context['blocked_events'] = blocked[:12]
        recent = context.get('recent_rule_feedback')
        if isinstance(recent, list):
            context['recent_rule_feedback_full_count'] = len(recent)
            context['recent_rule_feedback_full_sha256'] = digest(recent)
            context['recent_rule_feedback'] = recent[-2:]
    if request_chars(packet) > MAX_REQUEST_CHARS - 8192:
        raise ValueError('Event decision evidence still exceeds compact request limit')
    output.mkdir(parents=True, exist_ok=True)
    pinned_json(output / "packet.json", packet)
    by_id = {e.event_id: e for e in available}
    last_selection_error = None
    last_rejected_event_id = None
    for attempt in range(3):
        request_dir = output / ("model_request" if attempt == 0
                                else f"schema_repair_{attempt}/model_request")
        key = "api:event_v10:" + digest(str(request_dir.resolve()))[:20]
        charged = False

        def reserve(request_sha256):
            nonlocal charged
            store.reserve("api", key, {"case_id": case_id,
                                       "role": "next_physical_event",
                                       "request_sha256": request_sha256,
                                       "output": str(request_dir)})
            charged = True

        evidence = packet if attempt == 0 else {
            **packet, "schema_repair": {
                "error": last_selection_error or "Previous response was unavailable",
                "previous_rejected_event_id": last_rejected_event_id,
                "allowed_event_ids": sorted(by_id),
                "instruction": "Choose one ID from allowed_event_ids exactly; the previous ID cannot execute now. Return all four short fields."}}
        try:
            raw, identity = _request("next_physical_event", PROMPT, evidence,
                                     request_dir, MODEL_PROFILE, reserve,
                                     max_output_tokens=MAX_OUTPUT_TOKENS)
        except (DecisionUnavailable, InvalidCompleteResponse) as error:
            if charged:
                store.settle(key, "error", str(request_dir))
            last_selection_error = type(error).__name__
            last_rejected_event_id = None
            if attempt < 2:
                continue
            raise
        if charged:
            store.settle(key, "complete", str(request_dir / "response.json"))
        try:
            event_id = validate_selection(raw, by_id)
            receipt = {"schema": "rssr.retarget.event_decision.v1",
                       "agent_version": VERSION, "case_id": case_id,
                       "event_id": event_id, "model_response_sha256":
                       file_sha(request_dir / "response.json"),
                       "request_sha256": identity, "decision": raw,
                       "packet_sha256": file_sha(output / "packet.json"),
                       "explanation_length_warnings": {
                           k: len(raw[k]) for k in
                           ("reason", "expected_signal", "abandon_condition")
                           if len(raw[k]) > 240}}
            pinned_json(output / "decision.json", receipt)
            return by_id[event_id]
        except ValueError as error:
            last_selection_error = str(error)
            last_rejected_event_id = (raw.get('event_id')
                                      if isinstance(raw, dict) else None)
            if attempt == 2:
                raise
    raise RuntimeError("K3 failed to select an approved physical event")
