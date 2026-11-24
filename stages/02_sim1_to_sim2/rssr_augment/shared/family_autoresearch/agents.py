"""Isolated family specialists and an independent evidence-grounded reviewer.

Each invocation is one stateless Responses request with no transport retries.
Request/response receipts are content-addressed and durable; an interrupted
unknown-outcome request is not silently submitted twice. Credentials stay in
memory inside the existing credential loader and never enter a receipt.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from rssr_augment.shared import responses_transport
from rssr_augment.common import SOURCE_ROOT
from . import actions


ROOT = SOURCE_ROOT
PROFILE_FILE = ROOT / 'config/llm_profiles.json'
if not PROFILE_FILE.is_file():
    PROFILE_FILE = ROOT / 'sim1-sim2/archive/legacy_workspace_20260927/config/llm_profiles.json'
MODEL = os.environ.get("REEMBODY_LLM_MODEL", "gpt-5")
MAX_OUTPUT_TOKENS = 8192
MAX_REQUEST_CHARS = 90000
SCHEMA = "rssr.family_autoresearch.agent.v1"


class DecisionUnavailable(RuntimeError):
    """A model request failed validation or its outcome cannot safely be replayed."""


class InvalidCompleteResponse(ValueError):
    def __init__(self, identity, reason="Known provider output is not a JSON object"):
        super().__init__(reason)
        self.identity = identity


AUTONOMOUS_PROMPT = """You are the autonomous retarget researcher assigned to ONE
case_id in this packet. Own this case's diagnosis, experiments and checkpoint end
to end, using the relevant family skills. Do not assume other cases' private
context or make decisions for them. You must
finish the research step using the configured API and provided physical evidence;
there is no human or coding-agent handoff. Catalogs are optional suggestions. If
they are unsuitable or exhausted, invent one falsifiable control hypothesis and
ONE typed command using the existing bounded control DSL, not Python or new tools.
Return JSON with status ('candidate' or 'unresolved_within_allowed_controls'),
mechanism_id (catalog choice or null), hypothesis (diagnosis,prediction,falsifier,
evidence_refs), command (null for a catalog choice; otherwise one typed command),
and unresolved (null for candidate; otherwise reason,missing_capability,
required_evidence). Prediction must name target_predicate; cite supplied references.
Keep diagnosis, prediction and falsifier each under1000 characters; one concise
testable claim is sufficient. Do not copy the whole evidence packet into a field.
Allowed command schemas (all also have operation and hand left/right):
motion_window: delta_m [x,y,z],start_s,rise_end_s,hold_end_s,end_s;
event_offset: cycle,event_phase grasp/release,delta_m [x,y,z];
gripper_pulse: start_s,end_s,ramp_s,opening_floor;
hold_release: start_s,end_s,ramp_s,opening_cap;
release_opening: cycle,ramp_s,opening_floor.
Optional cycle on time windows must match a source event. All numbers finite;
each displacement axis <=0.04 m, nonzero; cumulative 0.16 m verified natively.
Times inside episode; motion knots at least0.05 s apart; ramp_s0.05..0.5;
gripper window length >=2*ramp_s+0.05; floor0..1 and cap0..0.5.
Stage carry must use measured hand/cycle and a time window around contact loss,
after established closure and before release. Opening floors do not tighten grip.
Closing caps may be no-ops; native compilation must prove actual control change.
Keep facts distinct from hypotheses; no success claim without full native judge.
Never edit scene, physics, assets, judge, or direct simulator state. Preserve prior
nonoverlapping gripper windows. Do not repeat tested commands or retired mechanisms.
Temporary task regression is admissible for hard-feasible measured stage progress.
If missing evidence or capability makes every allowed control unjustified, return
unresolved_within_allowed_controls with an explicit machine-readable reason; do
not request human intervention or pretend the case succeeded.
"""


BASE_PROMPT = """You are an autonomous RSSR Sim1-to-Sim2 research specialist.
Use only the supplied TaskPacket: previous chats and unstated measurements are
unavailable. Facts require evidence_refs from this packet. Label causal claims
as hypotheses. Sim1 is a reference, not guaranteed correct. The native frozen
v80 full-episode judge alone determines success; local margin gains, short probes
and partial trajectories are not full success. Scene/assets/physics/judge edits,
direct state writes, and changes outside allowed_scope are forbidden. Repairs
use normal bounded control: 4 cm per action and 16 cm cumulative per axis,
subsequently checked against the complete native command stream.
Choose ONE mechanism_id from mechanism_menu; its next bounded parameter member
is selected deterministically. Do not invent parameters, hand choices, new
tools, or observations. Consider prior negative evidence and actual response.
A hard-feasible branch may temporarily lose task predicates when measured
stage progress has a concrete recovery target. Do not require every old task
predicate to stay true. Never allow new hard violations.
If no listed mechanism addresses the observed failure, return mechanism_handoff
with a concrete alternative mechanism, its required evidence and minimal test;
do not silently loop through an irrelevant menu.
Return exactly JSON: {status:'candidate'|'mechanism_handoff',
mechanism_id:string|null, hypothesis:{diagnosis:string,prediction:string,
falsifier:string,evidence_refs:[string]}, handoff:null|{mechanism:string,
required_evidence:[string],minimal_test:string}}. Prediction must name the
target_predicate, expected observable change and what would disprove it.
When stage is carry, inspect stage_evidence and its provenance. A lift predicate
can have passed earlier even though the object loses hand contact and falls long
before source release. The carry menu is bound to that measured hand/cycle and
acts before/around observed contact loss. Do not claim an opening floor tightens
grasp: it imposes a minimum opening and can test relief of excessive squeeze or
other contact changes. A closing cap may have no effect when commands are already
closed; native preflight must demonstrate an actual command change before physics.
"""

FAMILY_PROMPTS = {
    "stack_cups": """You own cups end to end: acquisition, sustained bilateral
contact, transport, nesting/support, release and slip. Millimetre command shifts
can change the contact state discontinuously. For grasp failures inspect actual
pad/tool response and first bilateral contact before altering placement. Earlier
precontact positioning is distinct from a late command that the arm cannot
track. For retained cups with placement errors compare spatial placement and
release timing hypotheses. The release_timing menu reproduces the seq109
hold-release family; both spatial arms must remain exactly fixed for a gripper
experiment. That prior success is evidence for a test, not a universal rule.
""",
    "stack_blocks": """You own blocks end to end: pad clearance, bilateral
grasp, lift, hold during transport, dual-hand timing, stacking/support and release.
Distinguish no initial contact from good closure followed by later loss. Preserve
a verified grasp as an exploration branch even when it exposes an unresolved
release failure. A previous release predicate may have passed only because the
block fell early. Use measured contact and lift to challenge that explanation.
Jaw opening controls must be selected by actual response, not object width alone.
""",
    "drawer": """You own drawer insertion and closure end to end. Separate
hand-handle contact, hand-body interference, inserted-object contact and rebound.
A drawer open/insert/close failure alone does not establish rebound. The guarded
drawer_local_slowdown menu appears only after measured insertion and rebound
with an existing late-push ramp. It preserves displacement targets while changing
one ramp's timing. Release opening and hold windows can be considered only when
actual source event timing makes them relevant. A feedback idea with no reachable
trigger is not a physical test; request the missing evidence or mechanism handoff.
""",
}

REVIEW_PROMPT = """You are the independent RSSR experiment auditor, with a fresh
context and read-only evidence. Keep each findings/reasons item under1000 characters.
First inspect raw_evidence and independently
enumerate the current failures, actual contact/execution response, and missing
measurements. Then assess proposal; the specialist's diagnosis is a hypothesis,
not another source of facts. Check cited evidence, failure target, actual action
scope, parent identity, duplicate/retired mechanisms, confounded control changes,
and whether this experiment can distinguish the proposed mechanism. Judge only
design quality here; native physics will decide physical success. A motion command
does not prove achieved motion. Do not invent facts or certify success.
The candidate has already passed deterministic native compilation and full
command preflight. Inspect preflight_receipt and its cited file. Actual motion,
contact and downstream task outcome remain unknown until physical rollout.
Hard-feasible exploration can temporarily regress task predicates when there is
measured stage progress and a concrete recovery target. Do not reject merely for
losing a previously passed release predicate. Scene, physical parameters, asset,
judge and direct simulator-state edits are disallowed. Full native command checks
must gate execution, including spatial freeze for gripper-only edits.
For carry proposals independently inspect stage_evidence hand/cycle/timing and
provenance. Earlier successful lift does not prove later retention. Reject
pregrasp or release-only timing for a documented earlier loss. An opening floor
cannot be described as tightening grasp, and a closing cap may be a command no-op.
Return exactly JSON {verdict:'proceed'|'revise'|'insufficient_evidence',
reasons:[string],evidence_refs:[string],independent_findings:[string]}.
For evidence_refs, copy whole strings verbatim from the input evidence_refs
array. Do not construct paths, append path segments, or put history labels in
that array. Explain history in independent_findings instead.
Proceed means this bounded experiment is justified, not that its hypothesis is
true. If evidence is missing, identify the exact measurement needed.
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save_decision(path: Path, value: dict) -> None:
    if path.exists() and _read(path) != value:
        raise ValueError("Saved validated research decision changed")
    if not path.exists():
        _write(path, value)


def _profile(profile=None) -> dict:
    if profile is None or isinstance(profile, str):
        name = "gpt6luna" if profile is None else profile
        if name not in ("gpt6luna", "k3"):
            raise ValueError("Unknown configured research profile")
        profile = _read(PROFILE_FILE)["profiles"][name]
    if not isinstance(profile, dict) or (profile.get("model"), profile.get("llm_api")) not in (
            (MODEL, "responses"), ("Kimi-K3", "chat_completions")):
        raise ValueError("Research profile differs from configured model routes")
    return copy.deepcopy(profile)


def _clean_strings(value, name: str, limit: int = 1500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid " + name)
    return value


def _task_contract(packet: dict) -> tuple[dict, dict]:
    """Expose the full fixed task contract, checking packet/source agreement."""
    contract = packet.get("task_contract")
    path = Path(packet["source_dir"]) / "sim2/task_contract.json" if packet.get("source_dir") else None
    provenance = {"source": "packet.task_contract", "path": None, "file_sha256": None}
    if path is not None and path.is_file():
        raw = path.read_bytes()
        source = json.loads(raw)
        if contract is not None and actions.digest(contract) != actions.digest(source):
            raise ValueError("TaskPacket full contract differs from the canonical source")
        contract = source
        provenance.update(source="canonical_source_contract", path=str(path.resolve()),
                          file_sha256=hashlib.sha256(raw).hexdigest())
        pinned = {**packet.get("source_sha256", {}), **packet.get("stage_evidence_sha256", {})}
        if str(path.resolve()) in pinned and pinned[str(path.resolve())] != provenance["file_sha256"]:
            raise ValueError("Task contract differs from the packet's frozen source fingerprint")
    if contract is None:
        contract = {"predicates": []}
        provenance["source"] = "not_supplied"
    if not isinstance(contract, dict) or not isinstance(contract.get("predicates", []), list):
        raise ValueError("Full task contract must contain a predicate list")
    provenance["content_sha256"] = actions.digest(contract)
    return contract, provenance


def _refs(packet: dict) -> list[str]:
    refs = packet.get("evidence_refs", [])
    if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs):
        raise ValueError("TaskPacket evidence_refs must be paths or stable references")
    refs = list(refs)
    evidence = packet.get("stage_evidence") or {}
    stage_refs = packet.get("stage_evidence_refs", evidence.get("evidence_refs", evidence.get("refs", [])))
    if not isinstance(stage_refs, list) or any(not isinstance(ref, str) or not ref for ref in stage_refs):
        raise ValueError("Stage evidence provenance must contain stable string references")
    refs.extend(ref for ref in stage_refs if ref not in refs)
    report_path = packet["parent"].get("report_path")
    if report_path and report_path not in refs:
        refs = [*refs, str(report_path)]
    _, provenance = _task_contract(packet)
    if provenance["path"] and provenance["path"] not in refs:
        refs.append(provenance["path"])
    return refs


def validate_packet(packet: dict) -> dict:
    required = {"case_id", "family", "parent", "failures", "history", "allowed_scope",
                "remaining_budget", "evidence_refs", "target_predicate"}
    if not isinstance(packet, dict) or not required.issubset(packet):
        raise ValueError("Incomplete isolated TaskPacket")
    _clean_strings(packet["case_id"], "case_id", 200)
    actions.family_of(packet)
    if not isinstance(packet["parent"], dict) or not isinstance(actions.report_of(packet), dict):
        raise ValueError("TaskPacket needs a parent and raw report")
    candidate = actions.parent_candidate(packet)
    if not {"program", "scene", "windows", "model_profile", "cost", "decision"}.issubset(candidate):
        raise ValueError("TaskPacket parent lacks a complete native candidate")
    if not isinstance(packet["history"], list) or not isinstance(packet["remaining_budget"], dict):
        raise ValueError("TaskPacket history/budget must be structured")
    refs = _refs(packet)
    if not refs:
        raise ValueError("No evidence reference accompanies the raw report")
    report = actions.report_of(packet)
    preflight = packet.get("preflight_receipt")
    if preflight is not None:
        if not isinstance(preflight, dict) or not {"path", "sha256", "summary"}.issubset(preflight):
            raise ValueError("Preflight receipt reference is incomplete")
        path = Path(preflight["path"])
        if (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != preflight["sha256"]
                or preflight["path"] not in refs):
            raise ValueError("Preflight receipt differs from review evidence")
    trace = packet.get("trace_features")
    if trace is not None and trace.get("status") == "complete":
        hashes = trace.get("evidence_sha256", {})
        if not hashes or any(ref not in refs or not Path(ref).is_file()
                             or hashlib.sha256(Path(ref).read_bytes()).hexdigest() != expected
                             for ref, expected in hashes.items()):
            raise ValueError("Trace feature evidence differs from its saved source")
    report_path = packet["parent"].get("report_path")
    if report_path:
        path = Path(report_path)
        if not path.is_file() or actions.digest(_read(path)) != actions.digest(report):
            raise ValueError("TaskPacket raw report does not match its saved evidence")
    objective = report.get("objective", {})
    failures = list(objective.get("hard_failures", [])) + list(objective.get("task_failures", []))
    if failures and packet["target_predicate"] not in failures:
        raise ValueError("Research target is not a raw-report failure")
    _json(packet)  # reject NaN and non-JSON state before any file or network write
    return packet


def raw_evidence(packet: dict) -> dict:
    """Deterministic direct report projection, identical for specialist/auditor."""
    report = actions.report_of(packet)
    keys = ("objective", "task_evaluation", "source_grasp_events", "simulation_seconds",
            "final_relations", "final_positions_m", "lift_m", "contact_samples",
            "articulation_final_open_fraction", "drawer_open_fraction_trace",
            "gripper_limit_peak_events", "max_joint_limit_violation_rad",
            "max_robot_table_penetration_m", "max_joint_tracking_error_rad",
            "gripper_command_adapter", "runtime_audit", "judgment_version",
            "grasp_retention", "ee_tracking", "measured_event_response", "state_trace")
    projected = {key: copy.deepcopy(report[key]) for key in keys if key in report}
    # Keep traces exact: if evidence exceeds the request limit, fail visibly and
    # let the coordinator produce an explicitly selected evidence packet.
    return {"report_sha256": actions.digest(report), "report_fields": projected,
            "evidence_refs": _refs(packet)}


def _context(packet: dict) -> dict:
    stage = packet.get("stage_evidence") or {}
    stage_refs = packet.get("stage_evidence_refs", stage.get("evidence_refs", stage.get("refs", [])))
    parent = actions.parent_candidate(packet)
    contract, contract_provenance = _task_contract(packet)
    physical_fields = {"id", "shape", "role", "mobility", "position_m", "size_m", "yaw_rad",
                       "roll_pitch_rad", "wall_m", "mass_kg", "parent_id", "support_id", "joint"}
    scene_objects = [{key: copy.deepcopy(value) for key,value in row.items() if key in physical_fields}
                     for row in parent["scene"].get("objects", [])]
    object_ids = [row["id"] for row in scene_objects]
    bindings = {}
    for spec in contract.get("predicates", []):
        if spec.get("kind") == "lift_with_contact" and spec.get("object") and spec.get("hand") in ("left", "right"):
            bindings.setdefault(spec["object"], set()).add(spec["hand"])
    bindings = {obj: next(iter(hands)) if len(hands) == 1 else sorted(hands) for obj,hands in bindings.items()}
    return {"case_id": packet["case_id"], "family": actions.family_of(packet),
            "parent_sha256": actions.digest(parent),
            "current_control": {"program": copy.deepcopy(parent["program"]),
                                "windows": copy.deepcopy(parent["windows"])},
            "scene_objects": scene_objects,
            "state_trace_layout": {"positions_m_object_order": object_ids,
                "hand_table_contacts_object_order": object_ids,
                "hand_table_contacts_columns": ["left", "right", "table"],
                "opening_m_hands": ["left", "right"],
                "basis": "Frozen native v80 scene.objects order and rollout contacts/opening array layout"},
            "task_contract_specs": copy.deepcopy(contract.get("predicates", [])),
            "task_contract_bindings": copy.deepcopy(contract.get("bindings", {})),
            "object_hand_bindings": bindings,
            "context_provenance": {"parent_candidate_sha256": actions.digest(parent),
                "parent_scene_sha256": actions.digest(parent["scene"]),
                "task_contract": contract_provenance,
                "stage_evidence_sha256": copy.deepcopy(packet.get("stage_evidence_sha256", {}))},
            "stage": packet.get("stage"), "target_predicate": packet["target_predicate"],
            "target_spec": actions.target_spec(packet),
            "stage_evidence": copy.deepcopy(packet.get("stage_evidence")),
            "stage_evidence_refs": copy.deepcopy(stage_refs),
            "trace_features": copy.deepcopy(packet.get("trace_features")),
            "failures": packet["failures"], "history": packet["history"],
            "allowed_scope": packet["allowed_scope"],
            "remaining_budget": packet["remaining_budget"],
            "retired_mechanisms": sorted(set(packet.get("retired_mechanisms", []))
                                          | set(packet.get("blocked_mechanisms", []))),
            "preflight_receipt": copy.deepcopy(packet.get("preflight_receipt")),
            "shared_program_instance": copy.deepcopy(packet.get("shared_program_instance")),
            "raw_evidence": raw_evidence(packet)}


def _request(role: str, prompt: str, evidence: dict, request_dir: Path, profile,
             before_request: Callable[[str], None] | None, client=None,
             max_output_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[dict, str]:
    selected = _profile(profile)
    profile_name = "k3" if selected["llm_api"] == "chat_completions" else "gpt6luna"
    request = {"schema": SCHEMA, "role": role, "profile": profile_name,
               "model": selected["model"], "prompt": prompt, "evidence": evidence,
               "max_output_tokens": max_output_tokens,
               "reasoning_effort": "medium" if profile_name == "gpt6luna" else "low"}
    if len(_json(request)) > MAX_REQUEST_CHARS:
        raise ValueError("Research request exceeds explicit compact-evidence bound")
    identity = actions.digest(request)
    folder = Path(request_dir)
    folder.mkdir(parents=True, exist_ok=True)
    # Concurrent resumes of the same request cannot submit two calls.
    with (folder / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request_path, response_path = folder / "request.json", folder / "response.json"
        receipt = {"request_sha256": identity, **request}
        if request_path.exists() and _read(request_path) != receipt:
            raise ValueError("Saved research request hash mismatch")
        if not request_path.exists():
            _write(request_path, receipt)
        if response_path.exists():
            response = _read(response_path)
            if response.get("request_sha256") != identity:
                raise ValueError("Saved provider response hash mismatch")
        else:
            if (folder / "attempt.json").exists():
                raise DecisionUnavailable("Prior request has no reusable response; explicit new request identity required")
            if client is None:
                from rssr_augment.shared.joint_robot_object_v2_20260923.task_subagent import _client
                try:
                    client = _client(selected)
                except Exception:
                    raise DecisionUnavailable("Configured Responses client is unavailable") from None
                # The user authorized this inference destination. Validate the
                # resolved client rather than a config label: its env-file URL
                # can change without changing the profile JSON fingerprint.
                expected_host = ("llmapi.blsc.cn" if profile_name == "k3"
                                 else os.environ.get("REEMBODY_LLM_HOST", "api.openai.com"))
                if urlparse(str(getattr(client, "base_url", ""))).hostname != expected_host:
                    raise DecisionUnavailable("Configured model destination is outside the authorized host")
            # Existing loader disables retries. Also force this on injected SDK
            # clients so callback accounting always means one actual request.
            if hasattr(client, "with_options"):
                client = client.with_options(max_retries=0,
                    timeout=900 if profile_name == "k3" else 90)
            if before_request is not None:
                before_request(identity)
            _write(folder / "attempt.json", {"request_sha256": identity, "status": "submitted"})
            try:
                if profile_name == "k3":
                    completion = client.chat.completions.create(
                        model=selected["model"],
                        messages=[{"role": "system", "content": prompt},
                                  {"role": "user", "content": _json(evidence)}],
                        response_format={"type": "json_object"},
                        temperature=1., stream=False,
                        max_tokens=max_output_tokens,
                        extra_body={"reasoning_effort": "low"})
                    answer = {"content": completion.choices[0].message.content or "",
                              "finish_reason": completion.choices[0].finish_reason,
                              "provider_model": completion.model,
                              "provider_response_id": completion.id,
                              "provider_status": None,
                              "provider_api": "chat_completions",
                              "usage": completion.usage.model_dump() if completion.usage else None}
                else:
                    answer = responses_transport.create(client,
                        {"model": selected["model"], "max_tokens": max_output_tokens, "reasoning_effort": "medium"},
                        [{"role": "developer", "content": prompt},
                         {"role": "user", "content": _json(evidence)}], {"type": "json_object"})
            except Exception as exc:
                _write(folder / "error.json", {"request_sha256": identity,
                    "error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)})
                raise DecisionUnavailable("Model request failed; sanitized receipt recorded") from None
            response = {key: answer.get(key) for key in ("content", "finish_reason", "provider_model",
                "provider_response_id", "provider_status", "provider_api", "usage")}
            response["request_sha256"] = identity
            _write(response_path, response)
        if response.get("finish_reason") == "length":
            raise InvalidCompleteResponse(identity, "Provider confirmed token truncation; return shorter complete JSON")
        if response.get("finish_reason") != "stop":
            raise DecisionUnavailable("Research response is incomplete")
        try:
            value = json.loads(response["content"])
        except (TypeError, ValueError, KeyError):
            raise InvalidCompleteResponse(identity) from None
        if not isinstance(value, dict):
            raise InvalidCompleteResponse(identity)
        return value, identity


def _validate_refs(refs, packet: dict) -> list[str]:
    allowed = _refs(packet)
    if not isinstance(refs, list) or not refs:
        raise ValueError("Research conclusion needs nonempty evidence_refs copied from the input")
    bad = [ref for ref in refs if ref not in allowed]
    if bad:
        raise ValueError("Research conclusion references absent evidence: " + str(bad[:2])[:180]
                         + "; copy exact strings from input evidence_refs")
    return refs


def _validated_request(role, prompt, context, folder, profile, callback, client, validator):
    """Repair only known-complete schema failures, with at most two new calls."""
    identities, previous, error = [], None, None
    for attempt in range(3):
        current = Path(folder) if attempt == 0 else Path(folder) / f"schema_repair_{attempt:02d}"
        evidence = copy.deepcopy(context)
        if attempt:
            evidence["schema_repair"] = {"attempt": attempt, "previous_output": previous,
                "validation_error": error, "instruction": "Correct the typed schema/evidence error; do not repeat an invalid control."}
        try:
            raw, identity = _request(role, prompt, evidence, current, profile, callback, client)
        except InvalidCompleteResponse as exc:
            identity = exc.identity
            identities.append(identity)
            previous = _read(current / "response.json").get("content")
            error = str(exc)
        else:
            identities.append(identity)
            try:
                return validator(raw), identities
            except (TypeError, ValueError, KeyError) as exc:
                previous, error = raw, str(exc)[:300]
        _write(current / "validation_error.json", {"request_sha256": identity,
            "error": error, "repair_attempt": attempt})
    return None, identities


def _hypothesis(raw, packet):
    hypothesis = raw.get("hypothesis")
    if not isinstance(hypothesis, dict) or set(hypothesis) != {"diagnosis", "prediction", "falsifier", "evidence_refs"}:
        raise ValueError("Missing structured falsifiable hypothesis")
    for key in ("diagnosis", "prediction", "falsifier"):
        _clean_strings(hypothesis[key], key)
    _validate_refs(hypothesis["evidence_refs"], packet)
    if packet["target_predicate"] not in hypothesis["prediction"]:
        raise ValueError("Prediction must name the currently failed target predicate")
    return hypothesis


def _proposal(raw, packet, generated_only=False):
    if set(raw) - {"status", "mechanism_id", "hypothesis", "handoff", "command", "unresolved"}:
        raise ValueError("Unknown proposal schema fields")
    hypothesis = _hypothesis(raw, packet)
    if raw.get("status") == "mechanism_handoff":
        if generated_only:
            raise ValueError("Generate an allowed command or unresolved_within_allowed_controls; no external handoff")
        return {"status": "needs_generated_control", "hypothesis": hypothesis,
                "handoff": raw.get("handoff"), "action": None}
    if raw.get("status") == "unresolved_within_allowed_controls":
        unresolved = raw.get("unresolved")
        if not isinstance(unresolved, dict) or set(unresolved) != {"reason", "missing_capability", "required_evidence"}:
            raise ValueError("Unresolved outcome needs reason, missing_capability, required_evidence")
        _clean_strings(unresolved["reason"], "unresolved reason")
        _clean_strings(unresolved["missing_capability"], "missing capability")
        if not isinstance(unresolved["required_evidence"], list):
            raise ValueError("Required evidence must be a list")
        if raw.get("command") is not None:
            raise ValueError("Unresolved outcome cannot carry an executable command")
        return {"status": "unresolved_within_allowed_controls", "hypothesis": hypothesis,
                "action": None, "mechanism": None, "unresolved": unresolved}
    if raw.get("status") not in ("candidate", "generated_candidate"):
        raise ValueError("Invalid autonomous proposal status")
    if raw.get("command") is not None:
        action = actions.build_action(packet, raw["command"])
    else:
        if generated_only:
            raise ValueError("Catalog exhausted: provide one typed command")
        action = actions.next_action(packet, raw.get("mechanism_id"))
        if action is None:
            raise ValueError("No admissible catalog member; generate a typed command instead")
        action = copy.deepcopy(action)
    action["decision"].update(diagnosis=hypothesis["diagnosis"], prediction=hypothesis["prediction"])
    return {"status": "candidate", "mechanism": action["mechanism"], "hypothesis": hypothesis,
            "action": action, "handoff": None, "unresolved": None}


def propose(packet: dict, request_dir: Path, profile=None, *, before_request=None,
            reserve_request=None, client=None) -> dict:
    """Autonomous catalog or generated-control research; no manual handoff."""
    validate_packet(packet)
    if before_request is not None and reserve_request is not None:
        raise ValueError("Supply only one request budget callback")
    callback = before_request or reserve_request
    context = _context(packet)
    context["mechanism_menu"] = actions.mechanism_menu(packet)
    context["allowed_control_fields"] = {op: sorted(fields) for op,fields in actions.CONTROL_FIELDS.items()}
    prompt = AUTONOMOUS_PROMPT + FAMILY_PROMPTS[actions.family_of(packet)]
    result, identities = _validated_request("family_specialist", prompt, context, request_dir,
        profile, callback, client, lambda raw: _proposal(raw, packet, not context["mechanism_menu"]))
    if result and result["status"] == "needs_generated_control":
        context["rejected_catalog_hypothesis"] = result
        result, extra = _validated_request("autonomous_control_researcher", prompt, context,
            Path(request_dir) / "autonomous_generation", profile, callback, client,
            lambda raw: _proposal(raw, packet, True))
        identities.extend(extra)
    if result is None:
        result = {"status": "unresolved_within_allowed_controls", "mechanism": None,
            "action": None, "hypothesis": None, "unresolved": {
                "reason": "schema_validation_exhausted", "missing_capability": "valid bounded control proposal",
                "required_evidence": _refs(packet)}}
    result.update(request_sha256=identities[-1], request_sha256s=identities)
    _save_decision(Path(request_dir) / "decision.json", result)
    return result


def review(packet: dict, proposal: dict, request_dir: Path, profile=None, *,
           before_request=None, reserve_request=None, client=None) -> dict:
    """Independent new-context design review against the same raw metrics."""
    validate_packet(packet)
    if before_request is not None and reserve_request is not None:
        raise ValueError("Supply only one request budget callback")
    if proposal.get("status") != "candidate" or not proposal.get("action"):
        return {"verdict": "insufficient_evidence", "reasons": ["A concrete candidate is required"],
                "evidence_refs": _refs(packet), "independent_findings": [], "request_sha256": None}
    action = proposal["action"]
    try:
        actions.validate_action(packet, action)
        _hypothesis(proposal, packet)
    except (ValueError, TypeError, KeyError) as exc:
        return {"verdict": "revise", "reasons": ["Candidate local validation failed: " + str(exc)[:300]],
                "evidence_refs": _refs(packet), "independent_findings": [], "request_sha256": None,
                "request_sha256s": []}
    context = _context(packet)
    context["proposal"] = copy.deepcopy(proposal)
    # No specialist prompt, provider response, thought trace or conversation is
    # carried into this call. Raw measurements and proposal are separate fields.
    def validate_review(raw):
        if set(raw) != {"verdict", "reasons", "evidence_refs", "independent_findings"}:
            raise ValueError("Reviewer response fields invalid")
        if raw["verdict"] not in ("proceed", "revise", "insufficient_evidence"):
            raise ValueError("Reviewer verdict invalid")
        for name in ("reasons", "independent_findings"):
            if not isinstance(raw[name], list) or not raw[name]:
                raise ValueError("Reviewer must record reasons and independent findings")
            for item in raw[name]:
                _clean_strings(item, name)
        _validate_refs(raw["evidence_refs"], packet)
        return raw
    result, identities = _validated_request("independent_reviewer", REVIEW_PROMPT, context,
        request_dir, profile, before_request or reserve_request, client, validate_review)
    if result is None:
        result = {"verdict": "insufficient_evidence", "reasons": ["Reviewer schema validation exhausted"],
                  "evidence_refs": _refs(packet), "independent_findings": ["No valid independent audit obtained"]}
    result.update(request_sha256=identities[-1], request_sha256s=identities)
    _save_decision(Path(request_dir) / "decision.json", result)
    return result
