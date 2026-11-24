"""Finite family repair mechanisms compiled through the existing native controls.

Catalogs use source events and the current parent, never case-number rules. A
model selects a mechanism; deterministic search chooses its next bounded member.
No routine in this module runs physics, edits a scene, or changes a judge.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FAMILIES = ("stack_cups", "stack_blocks", "drawer")


def counts_as_tested(row: dict) -> bool:
    """A local review schema error did not test the control or its mechanism."""
    if row.get("status") == "review_rejected" and not row.get("physical_completed"):
        reasons = row.get("reason") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        if any(reason == "Candidate control, identity, scope or evidence failed independent validation"
               or reason.startswith("Candidate local validation failed:")
               or reason == "Reviewer schema validation exhausted"
               for reason in reasons if isinstance(reason, str)):
            return False
    return True


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def parent_candidate(packet: dict) -> dict:
    parent = packet["parent"]
    return parent.get("candidate", parent)


def report_of(packet: dict) -> dict:
    return packet.get("parent", {}).get("report", packet.get("report", {}))


def family_of(packet: dict) -> str:
    family = {"cups": "stack_cups", "blocks": "stack_blocks"}.get(
        packet["family"], packet["family"])
    if family not in FAMILIES:
        raise ValueError("Unknown research task family")
    return family


def _duration(packet: dict) -> float:
    value = packet.get("duration_s", report_of(packet).get("simulation_seconds"))
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError("A finite episode duration is required")
    return float(value)


def target_spec(packet: dict) -> dict:
    """Bind acquisition to the contract's actual hand, never a case-ID rule."""
    if isinstance(packet.get("target_spec"), dict):
        return packet["target_spec"]
    if packet.get("source_dir"):
        path = Path(packet["source_dir"]) / "sim2/task_contract.json"
        if path.is_file():
            contract = json.loads(path.read_text(encoding="utf8"))
            return next((row for row in contract.get("predicates", [])
                         if row.get("id") == packet["target_predicate"]), {})
    return {}


def _placement_hand(packet: dict) -> str | None:
    """Prefer the target object's unique lift-contract hand, retaining both."""
    spec = target_spec(packet)
    if spec.get("hand") in ("left", "right"):
        return spec["hand"]
    obj = spec.get("object")
    if not obj:
        return None
    binding = packet.get("object_hand_bindings", {}).get(obj)
    if binding in ("left", "right"):
        return binding
    if isinstance(binding, list) and len(set(binding)) == 1 and binding[0] in ("left", "right"):
        return binding[0]
    if packet.get("source_dir"):
        path = Path(packet["source_dir"]) / "sim2/task_contract.json"
        if path.is_file():
            contract = json.loads(path.read_text(encoding="utf8"))
            hands = {row.get("hand") for row in contract.get("predicates", [])
                     if row.get("kind") == "lift_with_contact" and row.get("object") == obj
                     and row.get("hand") in ("left", "right")}
            if len(hands) == 1:
                return next(iter(hands))
    return None


def _stage(packet: dict) -> str:
    explicit = packet.get("stage", "")
    if explicit in ("model_health", "hard", "hard_constraint", "safety"):
        return "model_health"
    if packet.get("target_predicate") in report_of(packet).get("objective", {}).get("hard_failures", []):
        return "model_health"
    if explicit in ("acquisition", "grasp", "contact_establishment", "lift"):
        return "acquisition"
    if explicit in ("hold", "carry", "contact_retention"):
        return "carry"
    if explicit in ("placement", "release", "close", "closure"):
        return "placement"
    target = packet.get("target_predicate", "")
    return "acquisition" if "lift" in target or "grasp" in target else "placement"


def _window(times: list[float], duration: float) -> dict | None:
    times = [round(max(.001, min(duration - .001, t)), 7) for t in times]
    if any(b - a < .05 - 1e-9 for a, b in zip(times, times[1:])):
        return None
    return dict(zip(("start_s", "rise_end_s", "hold_end_s", "end_s"), times))


def carry_evidence(packet: dict) -> dict | None:
    """Validate the measured lost-contact episode before choosing its controls.

    The final failed task predicate may concern support, while the relevant
    contact loss belongs to another hand than the predicate's support object.
    Hand and cycle therefore come from measured stage evidence, not that name.
    """
    if _stage(packet) != "carry":
        return None
    evidence = packet.get("stage_evidence")
    if not isinstance(evidence, dict):
        return None
    hand, cycle = evidence.get("hand"), evidence.get("cycle")
    if hand not in ("left", "right") or type(cycle) is not int:
        return None
    events = report_of(packet).get("source_grasp_events", {}).get(hand, [])
    if not 0 <= cycle < len(events) or not isinstance(evidence.get("object"), str):
        return None
    fields = ("closed_s", "release_s", "last_contact_s", "contact_loss_s")
    if any(type(evidence.get(k)) not in (int, float) or not math.isfinite(evidence[k]) for k in fields):
        return None
    if any(abs(float(evidence[k]) - float(events[cycle][k])) > 1e-5 for k in ("closed_s", "release_s")):
        return None
    closed, release = float(evidence["closed_s"]), float(evidence["release_s"])
    last, loss = float(evidence["last_contact_s"]), float(evidence["contact_loss_s"])
    if not 0 <= closed <= last <= loss < release <= _duration(packet) + .05:
        return None
    return copy.deepcopy(evidence)


def _carry_members(evidence: dict, family: str, duration: float) -> list[tuple[str, str, dict]]:
    """Local, open-loop tests around observed loss; no new feedback is implied."""
    hand, cycle = evidence["hand"], evidence["cycle"]
    closed, release, loss = (float(evidence[k]) for k in ("closed_s", "release_s", "contact_loss_s"))
    stem = f"{hand}_c{cycle}_loss{loss:.4f}"
    result = []
    for lead in (.35, .55):
        start = max(closed + .05, loss - lead)
        end = min(release - .10, loss + .40)
        if start >= loss - .10 or end <= loss + .10 or end - start < .25:
            continue
        for level in (.05, .10, .20):
            result.append((f"{stem}_floor_{lead:g}_{level:g}", "carry_opening_floor",
                {"operation": "gripper_pulse", "hand": hand, "cycle": cycle,
                 "start_s": start, "end_s": end, "ramp_s": .10, "opening_floor": level}))
        # A cap can be inert if the parent is already closed. Complete compiled
        # command differences, checked by native preflight, decide applicability.
        for level in (0., .15, .30):
            result.append((f"{stem}_cap_{lead:g}_{level:g}", "carry_closing_cap",
                {"operation": "hold_release", "hand": hand, "cycle": cycle,
                 "start_s": start, "end_s": end, "ramp_s": .10, "opening_cap": level}))
        timing = _window([start, loss - .10, min(release - .25, loss + .20),
                          min(release - .10, loss + .50)], duration)
        if timing and timing["start_s"] > closed and timing["end_s"] < release:
            magnitudes = (.002, .004, .006) if family == "stack_cups" else (.005, .01, .02)
            for magnitude in magnitudes:
                for axis in (0, 1, 2):
                    for sign in (-1, 1):
                        delta = [0., 0., 0.]
                        delta[axis] = sign * magnitude
                        result.append((f"{stem}_spatial_{lead:g}_{axis}_{sign}_{magnitude:g}",
                            "carry_contact_window", {"operation": "motion_window", "hand": hand,
                                "cycle": cycle, "delta_m": delta, **timing}))
    return result


def _rebound(report: dict) -> dict | None:
    """Small dependency-free equivalent of the existing native rebound guard."""
    rows = report.get("task_evaluation", {}).get("predicates", [])
    row = next((r for r in rows if r.get("kind") == "open_insert_close"
                and not r.get("passed")), None)
    if row is None:
        return None
    m = row.get("measurements", {})
    if not (m.get("opened_before_insertion") is True
            and m.get("open_during_insertion") is True
            and m.get("closed_at_end") is False):
        return None
    insertion = m.get("first_insertion_s")
    events = report.get("source_grasp_events", {}).get("right", [])
    if type(insertion) not in (int, float) or not events:
        return None
    close = float(events[-1]["closed_s"])
    duration = float(report["simulation_seconds"])
    trace = [r for r in report.get("drawer_open_fraction_trace", [])
             if close <= float(r["time_s"]) <= duration - .2]
    if not trace:
        return None
    low = min(trace, key=lambda r: float(r["open_fraction"]))
    after = [float(r["open_fraction"]) for r in trace
             if float(r["time_s"]) > float(low["time_s"]) + .2]
    if float(low["open_fraction"]) > .4 or max(after, default=0) - float(low["open_fraction"]) < .25:
        return None
    return {"valley_time_s": float(low["time_s"]),
            "first_insertion_s": float(insertion), "closed_s": close}


def _slowdown_members(packet: dict) -> list[tuple[str, dict]]:
    rebound = _rebound(report_of(packet))
    if rebound is None:
        return []
    knots = parent_candidate(packet)["program"].get("time_knots", {}).get("right", [])
    if len(knots) > 12:
        return []
    indices = [i for i, row in enumerate(knots)
               if rebound["valley_time_s"] - .85 <= float(row["time_s"])
               <= rebound["valley_time_s"] + .15]
    if len(indices) < 2:
        return []
    peak = min(indices, key=lambda i: knots[i]["offset_m"][1])
    start = next((i for i in indices if i < peak
                  and abs(knots[i]["offset_m"][1]) < abs(knots[peak]["offset_m"][1]) - .005), None)
    if start is None or knots[peak]["offset_m"][1] >= -.005:
        return []
    members = []
    for factor in (1.5, 2.):
        old, end = float(knots[start]["time_s"]), float(knots[peak]["time_s"])
        new = end - factor * (end - old)
        protected = max(rebound["closed_s"] + .05, rebound["first_insertion_s"] + .2)
        if new <= protected or (start and new <= float(knots[start - 1]["time_s"])):
            continue
        members.append((f"drawer_slowdown_{factor:g}", {
            "operation": "local_slowdown", "hand": "right", "knot_index": start,
            "old_start_s": old, "new_start_s": new, "peak_time_s": end,
            "factor": factor, "protected_after_s": protected}))
    return members


def catalog(packet: dict, *, include_tried: bool = False) -> list[dict]:
    """Return a stable, finite action directory conditional on family/stage.

    Release holds reproduce seq109's twelve timing/level pairs for each hand.
    Precontact windows extend the seq106 timing mechanism to measured events.
    Rebound slowdown exists only with insertion and rebound evidence.
    """
    family, stage, report = family_of(packet), _stage(packet), report_of(packet)
    duration = _duration(packet)
    target = packet.get("target_predicate")
    if not isinstance(target, str) or not target:
        raise ValueError("A currently failed target_predicate is required")
    events = report.get("source_grasp_events", {})
    parent_sha = digest(parent_candidate(packet))
    rows = []

    def add(name: str, mechanism: str, command: dict, phase: str) -> None:
        signature = digest({"case_id": packet["case_id"], "parent": parent_sha,
                            "command": command})
        hand = command["hand"]
        grip = command["operation"] in ("hold_release", "gripper_pulse", "release_opening")
        decision = {"target_predicate": target,
                    "lever": "gripper_window" if grip else command["operation"],
                    "source": "family_autoresearch." + mechanism,
                    "catalog_id": name, "mechanism": mechanism,
                    "robot": None if grip else {"hand": hand, "phase": phase,
                        "cycle": command.get("cycle", 0),
                        "delta_m": command.get("delta_m", [0., 0., 0.]), "time_s": None},
                    "gripper": {"hand": hand} if grip else None,
                    "object": None,
                    "diagnosis": "Test bounded " + mechanism + " against measured source events",
                    "prediction": target + ": measure the predicted physical response"}
        rows.append({"id": name, "signature": signature, "mechanism": mechanism,
                     "parent_sha256": parent_sha, "case_id": packet["case_id"],
                     "duration_s": duration, "decision": decision, "command": command})

    # Order matches the existing seq109 catalog for deterministic release search.
    measured_carry = carry_evidence(packet)
    if stage == "carry":
        # Missing/contradictory measured evidence requires a research handoff;
        # it must never fall through to pregrasp or late release tuning.
        if measured_carry:
            for name, mechanism, command in _carry_members(measured_carry, family, duration):
                add(name, mechanism, command, "carry")
                rows[-1]["decision"]["stage_evidence"] = copy.deepcopy(measured_carry)
                if mechanism == "carry_opening_floor":
                    rows[-1]["decision"]["diagnosis"] = "Test whether a small opening floor changes contact near observed loss; it may alleviate squeeze and does not tighten grasp"
                elif mechanism == "carry_closing_cap":
                    rows[-1]["decision"]["diagnosis"] = "Test a closing cap near observed loss only if compiled commands change; an already closed parent makes this inert"
        hands = ()
    else:
        spec_hand = target_spec(packet).get("hand")
        hands = (spec_hand,) if stage == "acquisition" and spec_hand in ("left", "right") else ("right", "left")
        if stage == "placement":
            preferred = _placement_hand(packet)
            if preferred:
                hands = (preferred, "right" if preferred == "left" else "left")
    for hand in hands:
        for cycle, event in enumerate(events.get(hand, [])):
            closed, release = float(event["closed_s"]), float(event["release_s"])
            if not (0 <= closed < release <= duration + .05):
                continue
            stem = f"{hand}_c{cycle}"
            if stage == "placement":
                for before, after in ((.20, .20), (.35, .35), (.20, .50), (.35, .50)):
                    for level in (0., .25, .5):
                        start, end = release - before, release + after
                        if start < 0 or end > duration:
                            continue
                        add(f"{stem}_hold_b{int(before*100):02d}_a{int(after*100):02d}_c{int(level*100):02d}",
                            "release_timing", {"operation": "hold_release", "hand": hand,
                                "cycle": cycle, "start_s": start, "end_s": end,
                                "ramp_s": .1, "opening_cap": level}, "release")
                for level in (.5, .75, 1.):
                    add(f"{stem}_release_open_{int(level*100)}", "release_opening",
                        {"operation": "release_opening", "hand": hand, "cycle": cycle,
                         "opening_floor": level, "ramp_s": .2}, "release")
            if stage == "acquisition":
                timing = _window([closed - 1.0, closed - .35, closed + .20,
                                  min(closed + .6, release - .1)], duration)
                if timing:
                    for magnitude in (.01, .02, .03):
                        for axis in (0, 1, 2):
                            for sign in (-1, 1):
                                delta = [0., 0., 0.]
                                delta[axis] = sign * magnitude
                                add(f"{stem}_precontact_{axis}_{sign}_{magnitude:g}", "precontact_shift",
                                    {"operation": "motion_window", "hand": hand,
                                     "delta_m": delta, **timing}, "grasp")
                hold = _window([closed + .05, closed + .20,
                                min(closed + .65, release - .25), min(closed + 1., release - .1)], duration)
                if hold:
                    for magnitude in (.005, .01, .02):
                        add(f"{stem}_lift_{magnitude:g}", "contact_lift",
                            {"operation": "motion_window", "hand": hand,
                             "delta_m": [0., 0., magnitude], **hold}, "grasp")
                # Small opening floors probe contact retention; measured contacts
                # must decide their value, not an assumed object width.
                start, end = closed + .05, min(closed + .65, release - .05)
                if end - start >= .25:
                    for level in (.05, .1, .2):
                        add(f"{stem}_hold_floor_{level:g}", "contact_hold",
                            {"operation": "gripper_pulse", "hand": hand,
                             "start_s": start, "end_s": end, "ramp_s": .1,
                             "opening_floor": level}, "grasp")
            if stage == "placement":
                timing = _window([closed + .05, closed + .35, release - .10,
                                  min(release + .4, duration - .001)], duration)
                if timing:
                    magnitudes = (.002, .004, .006) if family == "stack_cups" else (.005, .01, .02)
                    for magnitude in magnitudes:
                        for axis in (0, 1, 2):
                            for sign in (-1, 1):
                                delta = [0., 0., 0.]
                                delta[axis] = sign * magnitude
                                add(f"{stem}_placement_{axis}_{sign}_{magnitude:g}", "placement_window",
                                    {"operation": "motion_window", "hand": hand,
                                     "delta_m": delta, **timing}, "release")
    if family == "drawer" and stage == "placement":
        for name, command in _slowdown_members(packet):
            add(name, "drawer_local_slowdown", command, "release")

    allowed = packet.get("allowed_scope", {})
    if isinstance(allowed, dict) and allowed.get("mechanisms") is not None:
        rows = [r for r in rows if r["mechanism"] in allowed["mechanisms"]]
    if include_tried:
        return rows
    tried = {r.get("action_signature", r.get("signature")) for r in packet.get("history", [])
             if counts_as_tested(r)}
    tried.update(packet.get("tried_action_signatures", []))
    # Legacy entries without signatures only apply when they name this parent.
    tried_ids = {r.get("action_id") for r in packet.get("history", [])
                 if counts_as_tested(r) and r.get("parent_sha256") == parent_sha}
    retired = set(packet.get("retired_mechanisms", [])) | set(packet.get("blocked_mechanisms", []))
    return [r for r in rows if r["signature"] not in tried and r["id"] not in tried_ids
            and r["mechanism"] not in retired]


def next_action(packet: dict, mechanism: str) -> dict | None:
    return next((row for row in catalog(packet) if row["mechanism"] == mechanism), None)


CONTROL_FIELDS = {
    "motion_window": {"operation", "hand", "delta_m", "start_s", "rise_end_s", "hold_end_s", "end_s"},
    "stage_motion_window": {"operation", "hand", "delta_m", "rotation_degrees",
                            "start_s", "rise_end_s", "hold_end_s", "end_s"},
    "event_offset": {"operation", "hand", "cycle", "event_phase", "delta_m"},
    "gripper_pulse": {"operation", "hand", "start_s", "end_s", "ramp_s", "opening_floor"},
    "hold_release": {"operation", "hand", "start_s", "end_s", "ramp_s", "opening_cap"},
    "release_opening": {"operation", "hand", "cycle", "ramp_s", "opening_floor"},
}


def build_action(packet: dict, command: dict) -> dict:
    """Validate one model-generated control in a closed typed DSL, never code."""
    if not isinstance(command, dict) or command.get("operation") not in CONTROL_FIELDS:
        raise ValueError("Command operation is outside the allowed control DSL")
    command = copy.deepcopy(command)
    op = command["operation"]
    required = CONTROL_FIELDS[op]
    if not required.issubset(command) or set(command) - required - {"cycle"}:
        raise ValueError("Command fields do not match the typed control schema")
    hand = command.get("hand")
    if hand not in ("left", "right"):
        raise ValueError("Command hand must be left or right")
    scope = packet.get("allowed_scope", {})
    if isinstance(scope, dict):
        if scope.get("operations") is not None and op not in scope["operations"]:
            raise ValueError("Command operation is outside task scope")
        if scope.get("hands") is not None and hand not in scope["hands"]:
            raise ValueError("Command hand is outside task scope")
    duration = _duration(packet)
    def number(key, lo, hi):
        value = command[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Command " + key + " is outside finite bounds")
        command[key] = float(value)
        return float(value)
    if "cycle" in command:
        cycle = command["cycle"]
        events = report_of(packet).get("source_grasp_events", {}).get(hand, [])
        if type(cycle) is not int or not 0 <= cycle < len(events):
            raise ValueError("Command cycle does not name a source event")
    if "delta_m" in command:
        values = command["delta_m"]
        displacement_bound = .08 if op == "stage_motion_window" else .04
        if (not isinstance(values, list) or len(values) != 3 or any(type(v) not in (int, float)
                or not math.isfinite(v) or abs(v) > displacement_bound for v in values)
                or max(abs(v) for v in values) < .00005):
            raise ValueError("Command displacement must be nonzero and inside its operation bound")
        command["delta_m"] = [float(v) for v in values]
    if op == "event_offset" and command["event_phase"] not in ("grasp", "release"):
        raise ValueError("Command event_phase must be grasp or release")
    if op in ("motion_window", "stage_motion_window"):
        times = [number(k, 0., duration) for k in ("start_s", "rise_end_s", "hold_end_s", "end_s")]
        if any(b-a < .05-1e-9 for a,b in zip(times,times[1:])):
            raise ValueError("Motion window knots must be separated by at least 50 ms")
    if op == "stage_motion_window":
        angles = command["rotation_degrees"]
        if (not isinstance(angles, list) or len(angles) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 35.
                       for v in angles)):
            raise ValueError("Stage orientation exceeds finite 35-degree axis bound")
        command["rotation_degrees"] = [float(v) for v in angles]
    if op in ("gripper_pulse", "hold_release", "release_opening"):
        ramp = number("ramp_s", .05, .5)
        number("opening_cap" if op == "hold_release" else "opening_floor", 0., .5 if op == "hold_release" else 1.)
        if op != "release_opening":
            start, end = number("start_s", 0., duration), number("end_s", 0., duration)
            if end-start < 2*ramp+.05-1e-9:
                raise ValueError("Gripper window is too short for its ramps")
    if _stage(packet) == "carry":
        measured = carry_evidence(packet)
        if measured is None:
            raise ValueError("Carry repair lacks consistent measured contact-loss evidence")
        if hand != measured["hand"] or command.get("cycle", measured["cycle"]) != measured["cycle"]:
            raise ValueError("Carry repair addresses the wrong measured hand or cycle")
        if op not in ("motion_window", "gripper_pulse", "hold_release") or not (
                measured["closed_s"] < command["start_s"] < measured["contact_loss_s"]
                < command["end_s"] < measured["release_s"]):
            raise ValueError("Carry repair must act around measured loss after closure and before release")
        command["cycle"] = measured["cycle"]
    parent_sha = digest(parent_candidate(packet))
    signature = digest({"case_id": packet["case_id"], "parent": parent_sha, "command": command})
    tried = set(packet.get("tried_action_signatures", [])) | {
        row.get("action_signature", row.get("signature")) for row in packet.get("history", [])
        if counts_as_tested(row)}
    if signature in tried:
        raise ValueError("Generated control already tested for this case and parent")
    mechanism = "generated_" + op + "_" + hand + "_" + _stage(packet)
    if mechanism in set(packet.get("retired_mechanisms", [])) | set(packet.get("blocked_mechanisms", [])):
        raise ValueError("Generated control mechanism is retired for this parent")
    grip = op in ("gripper_pulse", "hold_release", "release_opening")
    target = packet["target_predicate"]
    decision = {"target_predicate": target, "lever": "gripper_window" if grip else op,
        "source": "family_autoresearch.generated_control", "mechanism": mechanism,
        "catalog_id": "generated_" + signature[:16], "object": None,
        "robot": None if grip else {"hand": hand, "cycle": command.get("cycle", 0),
            "phase": command.get("event_phase", _stage(packet)),
            "delta_m": command.get("delta_m", [0.,0.,0.]), "time_s": None},
        "gripper": {"hand": hand} if grip else None,
        "diagnosis": "Autonomous bounded control hypothesis",
        "prediction": target + ": verify measured physical response"}
    if op == "stage_motion_window":
        decision["stage_plan"] = "bounded_stage_v1"
    return {"id": decision["catalog_id"], "signature": signature, "mechanism": mechanism,
        "parent_sha256": parent_sha, "case_id": packet["case_id"], "duration_s": duration,
        "generated": True, "dsl_version": 1, "decision": decision, "command": command}


def validate_action(packet: dict, action: dict) -> dict:
    """Independently reconstruct candidate identity/scope before model review."""
    if not isinstance(action, dict):
        raise ValueError("Missing structured action")
    if action.get("generated") is True:
        canonical = build_action(packet, action.get("command"))
    else:
        canonical = next((a for a in catalog(packet) if a["signature"] == action.get("signature")), None)
        if canonical is None:
            raise ValueError("Catalog action is absent, already tested or retired")
    for key in canonical:
        if key == "decision":
            expected = {k:v for k,v in canonical[key].items() if k not in ("diagnosis", "prediction")}
            actual = {k:v for k,v in action.get(key, {}).items() if k not in ("diagnosis", "prediction")}
            if actual != expected:
                raise ValueError("Action decision scope differs from validated command")
        elif action.get(key) != canonical[key]:
            raise ValueError("Action identity differs from validated command")
    return canonical


def mechanism_menu(packet: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in catalog(packet):
        if row["mechanism"] not in grouped:
            grouped[row["mechanism"]] = {"mechanism_id": row["mechanism"],
                "next_action": row, "remaining_members": 0}
        grouped[row["mechanism"]]["remaining_members"] += 1
    return list(grouped.values())


def _segments(knots: list[dict]) -> tuple[float, float]:
    velocities = []
    for a, b in zip(knots, knots[1:]):
        dt = float(b["time_s"]) - float(a["time_s"])
        if dt <= 0:
            raise ValueError("Retimed knots are not strictly increasing")
        velocities.append([(y - x) / dt for x, y in zip(a["offset_m"], b["offset_m"])])
    return (max((math.dist(v, [0., 0., 0.]) for v in velocities), default=0.),
            max((math.dist(a, b) for a, b in zip(velocities, velocities[1:])), default=0.))


def _knot_value(rows: list[dict], when: float, field: str, duration: float) -> list[float]:
    """Evaluate the frozen smoothstep knot program before adding a new pulse."""
    points = [{"time_s": 0., field: [0., 0., 0.]}, *rows,
              {"time_s": duration, field: [0., 0., 0.]}]
    for left, right in zip(points, points[1:]):
        a, b = float(left["time_s"]), float(right["time_s"])
        if a <= when <= b:
            u = (when-a)/(b-a)
            weight = u*u*(3.-2.*u)
            x, y = left.get(field, [0., 0., 0.]), right.get(field, [0., 0., 0.])
            return [(1.-weight)*float(x[i])+weight*float(y[i]) for i in range(3)]
    raise ValueError("Motion knot outside frozen episode")


def _composed_motion_knots(rows: list[dict], command: dict, duration: float) -> list[dict]:
    """Keep an additive motion pulse active through all pre-existing hand knots."""
    start, rise, hold, end = (float(command[key]) for key in
                              ("start_s", "rise_end_s", "hold_end_s", "end_s"))
    if len({start, rise, hold, end}) != 4 or not 0 <= start < rise < hold < end <= duration:
        raise ValueError("Invalid composed motion window")
    def pulse(when: float) -> float:
        if when <= start or when >= end:
            return 0.
        if when < rise:
            u = (when-start)/(rise-start)
            return u*u*(3.-2.*u)
        if when <= hold:
            return 1.
        u = (when-hold)/(end-hold)
        return 1.-u*u*(3.-2.*u)
    times = sorted({*[float(row["time_s"]) for row in rows], start, rise, hold, end})
    if len(times) > 12:
        raise ValueError("Composed motion window exceeds 12 time knots")
    delta = command["delta_m"]
    result = []
    for when in times:
        old = _knot_value(rows, when, "offset_m", duration)
        rotation = _knot_value(rows, when, "rotation_degrees", duration)
        weight = pulse(when)
        result.append({"time_s": when,
                       "offset_m": [old[i]+weight*float(delta[i]) for i in range(3)],
                       "rotation_degrees": rotation})
    return result


def compile_action(parent: dict, action: dict, config: dict, *,
                   duration_s: float | None = None, core=None) -> dict:
    """Compile a catalog member, preserving scene/model and returning six fields.

    The native adapter separately verifies the complete command stream against
    the 4 cm action / 16 cm cumulative bounds before any physics starts.
    """
    if core is None:
        from pipeline.joint_robot_object_v2_20260923 import runner_core as core
    parent = parent.get("candidate", parent)
    if digest(parent) != action["parent_sha256"]:
        raise ValueError("Action belongs to a different parent candidate")
    command = copy.deepcopy(action["command"])
    expected = digest({"case_id": action["case_id"], "parent": action["parent_sha256"],
                       "command": command})
    if action["signature"] != expected:
        raise ValueError("Action signature does not match its command")
    duration = float(duration_s if duration_s is not None else action["duration_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid action duration")
    if command.get("hand") not in ("left", "right"):
        raise ValueError("Action must declare one existing hand")
    decision = action["decision"]
    grip = command["operation"] in ("hold_release", "gripper_pulse", "release_opening")
    expected_lever = "gripper_window" if grip else command["operation"]
    declared = decision.get("gripper" if grip else "robot")
    if (decision.get("lever") != expected_lever or not isinstance(declared, dict)
            or declared.get("hand") != command["hand"] or decision.get("object") is not None):
        raise ValueError("Declared action scope differs from compiled control")
    program, windows = copy.deepcopy(parent["program"]), copy.deepcopy(parent.get("windows", {}))
    operation = command["operation"]
    if operation == "local_slowdown":
        knots = program.get("time_knots", {}).get(command["hand"], [])
        index = command["knot_index"]
        if type(index) is not int or not 0 <= index < len(knots) or len(knots) > 12:
            raise ValueError("Invalid existing slowdown knot")
        old, new = float(knots[index]["time_s"]), float(command["new_start_s"])
        if old != command["old_start_s"] or not command["protected_after_s"] < new < old:
            raise ValueError("Slowdown crosses insertion or source-event guard")
        before = _segments(knots)
        knots[index]["time_s"] = new
        after = _segments(knots)
        if any(b > a + 1e-9 for a, b in zip(before, after)):
            raise ValueError("Slowdown increases speed or velocity-jump proxy")
        if not 0 < float(knots[0]["time_s"]) < float(knots[-1]["time_s"]) < duration:
            raise ValueError("Slowdown leaves episode bounds")
    elif operation == "stage_motion_window":
        if decision.get("stage_plan") != "bounded_stage_v1":
            raise ValueError("Stage plan lacks the independent bounded compiler identity")
        hand = command["hand"]
        if program.get("time_knots", {}).get(hand):
            raise ValueError("Stage compiler requires an unmodified hand knot stream")
        delta = command["delta_m"]
        angles = command["rotation_degrees"]
        if (len(delta) != 3 or any(type(v) not in (int, float) or not math.isfinite(v)
                                   or abs(v) > .08 for v in delta)
                or len(angles) != 3 or any(type(v) not in (int, float) or not math.isfinite(v)
                                           or abs(v) > 35. for v in angles)):
            raise ValueError("Stage displacement or orientation exceeds compiler bounds")
        times = [command[k] for k in ("start_s", "rise_end_s", "hold_end_s", "end_s")]
        if (any(type(t) not in (int, float) or not math.isfinite(t)
                or not 0 <= t <= duration for t in times)
                or any(b-a < .05-1e-9 for a,b in zip(times,times[1:]))):
            raise ValueError("Stage window leaves the frozen source clock")
        program.setdefault("time_knots", {})[hand] = [
            {"time_s": float(t),
             "offset_m": list(delta if i in (1, 2) else [0., 0., 0.]),
             "rotation_degrees": list(angles if i in (1, 2) else [0., 0., 0.])}
            for i, t in enumerate(times)]
    else:
        allowed = {"motion_window", "event_offset", "gripper_pulse", "hold_release", "release_opening"}
        if operation not in allowed:
            raise ValueError("Unsupported family control operation")
        if "delta_m" in command:
            delta = command["delta_m"]
            if (not isinstance(delta, list) or len(delta) != 3 or any(type(v) not in (int, float)
                    or not math.isfinite(v) or abs(v) > .04 for v in delta)):
                raise ValueError("Single-action displacement exceeds finite 4 cm bound")
        if operation == "motion_window":
            times = [command[k] for k in ("start_s", "rise_end_s", "hold_end_s", "end_s")]
            if any(not 0 <= t <= duration for t in times) or any(b-a < .05-1e-9 for a,b in zip(times,times[1:])):
                raise ValueError("Motion window leaves episode bounds")
        if operation in ("gripper_pulse", "hold_release"):
            if not 0 <= command["start_s"] < command["end_s"] <= duration:
                raise ValueError("Gripper window leaves episode bounds")
        carry_gripper = (action.get("generated") or action.get("mechanism", "").startswith("carry_")) and operation in ("gripper_pulse", "hold_release")
        prior_windows = copy.deepcopy(windows.get(command["hand"], [])) if carry_gripper else []
        if carry_gripper:
            for old in prior_windows:
                old_start, old_end = float(old["start_s"]), float(old.get("end_s", duration))
                if max(old_start, float(command["start_s"])) < min(old_end, float(command["end_s"])):
                    raise ValueError("Carry intervention overlaps an existing gripper window; isolate the intended control first")
            if len(prior_windows) >= 3:
                raise ValueError("Carry intervention would exceed three preserved gripper windows")
        existing_knots = program.get("time_knots", {}).get(command["hand"], [])
        if operation == "motion_window" and existing_knots:
            program.setdefault("time_knots", {})[command["hand"]] = _composed_motion_knots(
                existing_knots, command, duration)
        else:
            program, windows = core.apply_decision(program, windows, command, duration)
        if carry_gripper:
            # Existing apply_decision intentionally replaces all same-mode
            # windows. A local carry experiment must preserve earlier/later
            # verified windows, so restore them and append only the new member.
            added = next((row for row in windows[command["hand"]]
                          if row.get("mode") == operation and row.get("start_s") == command["start_s"]
                          and row.get("end_s") == command["end_s"]), None)
            if added is None:
                raise ValueError("Native compiler did not produce the requested carry window")
            windows[command["hand"]] = prior_windows + [copy.deepcopy(added)]
    program = core.validate_program(program, {})
    core.validate_combined_ik_bounds(program, config)
    if program == parent["program"] and windows == parent.get("windows", {}):
        raise ValueError("Candidate command is a no-op")
    cost = float(parent.get("cost", 0.)) + max((abs(v) / (.08 if operation == "stage_motion_window" else .04)
                                                for v in command.get("delta_m", [])), default=0.)
    return {"program": program, "scene": copy.deepcopy(parent["scene"]),
            "windows": windows, "model_profile": copy.deepcopy(parent.get("model_profile", {})),
            "cost": cost, "decision": copy.deepcopy(action["decision"])}
