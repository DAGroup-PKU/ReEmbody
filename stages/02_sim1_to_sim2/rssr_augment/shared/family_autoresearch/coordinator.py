"""Family scheduling and resumable propose/review/preflight/physics transactions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .store import (Store, DEFAULT_LIMITS, FAMILIES, BudgetExhausted,
                    UncertainOperation, atomic_json, pinned_json, read_json,
                    digest, file_sha, utc_now)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/cobot150_family_autoresearch_v5_20260924"
PYTHON = Path(os.environ.get("REEMBODY_PHYSICS_PYTHON", sys.executable))
FOCUS = {
    "stack_cups": ("stack_cups_ep000027", "stack_cups_ep000039", "stack_cups_ep000040"),
    "stack_blocks": ("stack_blocks_ep000001", "stack_blocks_ep000010"),
    "drawer": ("drawer_ep000009", "drawer_ep000014"),
}


def implementation_identity():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += sorted((Path(__file__).parent / "retarget_program").glob("*.py"))
    native = ROOT / "pipeline/cobot150_runtime_v80_task_completion_20260923"
    paths += sorted(native.glob("*.py"))
    paths += [ROOT / "run_semantic_gpt150_task_completion_core_seq34.py",
              ROOT / "pipeline/crossfamily_stage_v58_seq104/safety.py",
              ROOT / "pipeline/responses_transport.py",
              ROOT / "config/llm_profiles.json"]
    return {str(path): file_sha(path) for path in paths}


def initialize(run, *, limits=None, profile="gpt6luna", index_from=None):
    from .evidence import build_index
    from .native import source_artifact_manifest
    run = Path(run).resolve()
    if run.parent != ROOT / "runs" or not run.name.startswith("cobot150_family_"):
        raise ValueError("Use a new RSSR/runs/cobot150_family_* directory")
    if (run / "manifest.json").exists():
        validate_identity(run)
        return Store(run)
    index = read_json(index_from) if index_from else build_index(ROOT)
    cases = index["cases"]
    if len(cases) != 150 or any(sum(c["family"] == f for c in cases) != 50 for f in FAMILIES):
        raise ValueError("Campaign requires exactly 50 cases in each of three families")
    pinned_json(run / "evidence_index.json", index)
    initial_artifacts = {}
    for case in cases:
        _, provenance = initial_candidate(case)
        origin = provenance.get("origin")
        paths = [provenance.get("baseline_trajectory_path")]
        if isinstance(origin, dict):
            paths += [origin.get("proposal_path"), origin.get("report_path")]
        initial_artifacts[case["case_id"]] = {p: file_sha(p) for p in paths if p}
    manifest = {
        "schema": "rssr.family.campaign.v1", "created_at": utc_now(),
        "goal": "150 distinct cases with full frozen native hard/task pass",
        "limits": dict(DEFAULT_LIMITS if limits is None else limits),
        "profile": profile, "python": str(PYTHON), "physics_workers": 1,
        "case_workers": 3,
        "agent_ownership": "one case per agent; family selects skills only",
        "case_limits": {"full": 4, "api": 12, "steps": 16},
        "max_agent_roles_active": 4, "max_api_concurrency": 2,
        "evidence_index_sha256": file_sha(run / "evidence_index.json"),
        "native_runtime_sha256": index["native_runtime_sha256"],
        "code_sha256": implementation_identity(),
        "authorization": {"source": "user accepted and requested implementation of three-family plan",
                          "automatic_review_within_budget": True,
                          "external_model_host": os.environ.get("REEMBODY_LLM_HOST", "api.openai.com"),
                          "external_model_approval": "User explicitly approved experiment data transfer within the 600 request budget",
                          "historical_coordination_files_unchanged": True},
        "frozen": {"scene": "v74 per-case selection", "model_profile": "v74 per-case selection",
                   "judge": "native v80 hard/task; strict_v1 separately reported",
                   "parent_world_axis_bound_m": .04, "source_world_axis_bound_m": .16},
        "source_sha256": source_artifact_manifest(cases),
        "initial_artifact_sha256": initial_artifacts,
        "case_ids": [c["case_id"] for c in cases],
        "historical_passes_are_new_success": False,
    }
    store = Store(run)
    store.initialize(manifest, manifest["case_ids"])
    write_summary(store)
    return store


def validate_identity(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if implementation_identity() != manifest["code_sha256"]:
        raise RuntimeError("Implementation changed: create a new campaign revision; old evidence stays immutable")
    if file_sha(run / "evidence_index.json") != manifest["evidence_index_sha256"]:
        raise RuntimeError("Evidence index changed")
    for path, expected in manifest["source_sha256"].items():
        if file_sha(path) != expected:
            raise RuntimeError("Frozen source changed: " + path)
    Store(run).validate_events()
    return manifest


def choose_case(state, allowed=None):
    allowed = set(allowed) if allowed else set(state["cases"])
    # Resume the exact unfinished task before allocating fresh work.
    pending = [cid for cid, c in state["cases"].items()
               if cid in allowed and c["pending_trial"]]
    if pending:
        return sorted(pending)[0]
    start = state.get("family_cursor", 0)
    for offset in range(3):
        family = FAMILIES[(start + offset) % 3]
        candidates = [(cid, c) for cid, c in state["cases"].items()
                      if cid in allowed and c["family"] == family
                      and c["status"] not in ("solved", "research_needed", "unresolved", "budget_exhausted")]
        if candidates:
            focus = FOCUS[family]
            return min(candidates, key=lambda pair: (
                not bool(pair[1].get("owner_agent")), pair[1]["visits"],
                focus.index(pair[0]) if pair[0] in focus else 99, pair[0]))[0]
    return None


def choose_parent(case):
    if not case["stable"]:
        return None
    active = [(key, branch) for key, branch in case["branches"].items()
              if branch["kind"] == "exploration" and not branch.get("retired")
              and branch.get("followups", 0) < 4]
    if active:
        # Give a newly acquired physical stage an opportunity to complete the task.
        return min(active, key=lambda pair: (pair[1].get("followups", 0), pair[1]["rank"]))[0]
    return case["stable"]


def candidate_from(value):
    if "candidate" in value:
        value = value["candidate"]
    return {"program": copy.deepcopy(value["program"]),
            "scene": copy.deepcopy(value["scene"]),
            "windows": copy.deepcopy(value.get("windows", {})),
            "model_profile": copy.deepcopy(value.get("model_profile", {})),
            "cost": float(value.get("cost", 0)),
            "decision": copy.deepcopy(value.get("decision", {"source": "baseline"}))}


def initial_candidate(index_case):
    source = read_json(Path(index_case["source_dir"]) / "sim2/selection.json")
    compatible = index_case.get("baseline_candidates", [])
    if compatible:
        selected = min(compatible, key=lambda c: (
            not c.get("passed"), len(c.get("hard_failures", [])),
            len(c.get("task_failures", [])), c["report_path"]))
        proposal = read_json(selected["proposal_path"])
        candidate = candidate_from(proposal)
        # Adopt the byte-exact frozen physical source, including harmless JSON
        # serialization roundoff. Never import a historic model override.
        candidate["scene"] = copy.deepcopy(source["scene"])
        candidate["model_profile"] = copy.deepcopy(source.get("model_profile", {}))
        trajectory = Path(selected["proposal_path"]).parent / "inputs/agent_trajectory_full.npz"
        return candidate, {"origin": selected, "baseline_trajectory_path":
                           str(trajectory) if trajectory.is_file() else None}
    return candidate_from(source), {"origin": "frozen_v74_selection"}


def target_stage(report):
    hard = report["objective"]["hard_failures"]
    failed = report["objective"]["task_failures"]
    if hard:
        return "model_health", hard[0]
    for name in failed:
        if name.startswith("lift_with_contact"):
            return "acquisition", name
    for name in ("inside_01", "open_insert_close_03"):
        if name in failed:
            return "articulation", name
    return "placement", failed[0] if failed else "complete"


def packet_for(store, index_case, parent_id, trial):
    from .actions import counts_as_tested
    from .routing import infer_stage
    from .trace_features import summarize
    case = store.read()["cases"][trial["case_id"]]
    branch = case["branches"][parent_id]
    if any(file_sha(p) != h for p, h in branch.get("output_sha256", {}).items()):
        raise ValueError("Parent evidence changed before research routing")
    report = read_json(branch["report_path"])
    contract_path = (Path(index_case["source_dir"]) / "sim2/task_contract.json").resolve()
    contract = read_json(contract_path)
    routing = infer_stage(report, contract, branch["report_path"])
    stage, target = routing["stage"], routing["target_predicate"]
    trace = summarize(report, branch["report_path"], contract, target)
    parent = {"candidate": candidate_from(read_json(branch["proposal_path"])),
              "report_path": branch["report_path"], "report": report,
              "proposal_path": branch["proposal_path"],
              "trajectory_path": branch.get("trajectory_path"), "id": parent_id}
    blocked = [x.split(":", 1)[1] for x in case["blocked_mechanisms"]
               if x.startswith(parent_id + ":")]
    target_spec = next((p for p in contract.get("predicates", []) if p.get("id") == target), {})
    packet = {"schema": "rssr.family.task_packet.v1", "case_id": trial["case_id"],
            "family": case["family"], "trial_id": trial["id"], "parent": parent,
            "agent_id": "case:" + trial["case_id"],
            "ownership": "This agent owns only this case until terminal status or checkpoint",
            "stage": stage, "target_predicate": target, "target_spec": target_spec,
            "task_contract": contract,
            "source_sha256": {str(contract_path): file_sha(contract_path)},
            "stage_evidence": routing["stage_evidence"],
            "stage_evidence_refs": routing["evidence_refs"],
            "stage_evidence_sha256": routing["evidence_sha256"],
            "trace_features": trace,
            "routing_policy": routing["routing_policy"],
            "routing_diagnostics": routing["diagnostics"],
            "source_dir": index_case["source_dir"],
            "duration_s": float(report["simulation_seconds"]),
            "failures": report["objective"], "history": case["history"][-12:],
            "tried_action_signatures": [h["action_signature"] for h in case["history"]
                                         if h.get("action_signature") and counts_as_tested(h)],
            "blocked_mechanisms": blocked, "retired_mechanisms": blocked,
            "remaining_budget": store.remaining(),
            "remaining_case_budget": store.case_remaining(trial["case_id"]),
            "evidence_refs": list(dict.fromkeys([branch["report_path"], branch["proposal_path"], str(contract_path),
                                                   *routing["evidence_refs"], *trace["evidence_refs"]])),
            "allowed_scope": {"family_algorithms": True, "scene_edits": False,
                              "physics_edits": False, "judge_edits": False,
                              "exploration_task_regression": True,
                              "hard_regression": False, "single_axis_m": .04,
                              "total_axis_m": .16}}
    from .retarget_program import attach_grasp_candidate
    return attach_grasp_candidate(packet, index_case)


def _metric_progress(before, after, target):
    """Stage evidence only; never changes native success or exports data."""
    bm = {p["id"]: p.get("measurements", {}) for p in before["task_evaluation"]["predicates"]}
    am = {p["id"]: p.get("measurements", {}) for p in after["task_evaluation"]["predicates"]}
    supported = []
    thresholds = {"max_lift_with_hand_contact_m": .005, "contact_fraction": .05,
                  "bilateral_contact_duration_s": .05}
    for key, minimum in thresholds.items():
        left, right = bm.get(target, {}).get(key), am.get(target, {}).get(key)
        if type(left) in (int, float) and type(right) in (int, float) and right-left >= minimum:
            supported.append({"metric": key, "before": left, "after": right})
    return supported


def assess_transition(before, after, target=None):
    ah = set(after["objective"]["hard_failures"])
    af = set(after["objective"]["task_failures"])
    bf = set(before["objective"]["task_failures"]) if before else set()
    bh = set(before["objective"]["hard_failures"]) if before else set()
    gains, losses = sorted(bf-af), sorted(af-bf) if before else []
    metrics = _metric_progress(before, after, target) if before and target else []
    passed = bool(after["objective"]["passed"]) and not ah and not af
    kind = ("success" if passed else "stable" if before is None else
            "stable" if ah < bh and af <= bf else
            "stable" if not ah and not losses and gains else
            "exploration" if not ah and (gains or metrics) else None)
    return {"branch_kind": kind, "gains": gains, "losses": losses,
            "stage_evidence": metrics, "hard_gains": sorted(bh-ah),
            "no_effect": not bool(gains or metrics or bh-ah),
            "rank": [len(ah), len(af)], "native_pass": passed}


def run_native(store, trial, job, mode):
    directory = store.run / "cases" / trial["case_id"] / f"trial_{trial['index']:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    job = {**job, "mode": mode}
    job_path, result_path = directory / (mode + "_job.json"), directory / (mode + "_result.json")
    pinned_json(job_path, job)
    if result_path.exists():
        result = read_json(result_path)
        if mode == "evaluate":
            key = "full:" + trial["id"]
            if key in store.read()["reservations"]:
                store.settle(key, "complete" if result.get("status") == "complete" else "error", str(result_path))
        return result
    key = "full:" + trial["id"]
    if mode == "evaluate":
        if key in store.read()["reservations"]:
            # Recovery never executes physics. A missing completion receipt ends
            # this charged attempt; a subsequent trial gets a fresh index.
            job_path = directory / "recover_job.json"
            pinned_json(job_path, {**job, "recover_only": True})
        else:
            store.reserve("full", key, {"case_id": trial["case_id"], "job": str(job_path)})
            store.update_trial(trial["id"], phase="physical_started")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(ROOT)
    timeout = min(1800., store.remaining()["wall_seconds"])
    if timeout <= 0:
        raise BudgetExhausted("Wall-clock budget exhausted")
    with (directory / (mode + ".log")).open("a") as log:
        try:
            completed = subprocess.run([str(PYTHON), "-m", "pipeline.family_autoresearch.native",
                                        "--job", str(job_path), "--result", str(result_path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            if mode == "evaluate":
                store.settle(key, "interrupted", str(directory / (mode + ".log")))
            return {"status": "tool_error", "error": "native_worker_timeout",
                    "complete_native_rollout": False}
    if not result_path.exists():
        result = {"status": "tool_error", "error": "native_worker_no_result",
                  "exit_code": completed.returncode, "complete_native_rollout": False}
    else:
        result = read_json(result_path)
    if mode == "evaluate":
        store.settle(key, "complete" if result.get("status") == "complete" else "error",
                     str(result_path))
    return result


def verify_native_result(result, store, trial):
    """Bind fresh evidence to this case/trial before considering its predicates."""
    directory = store.run / "cases" / trial["case_id"] / f"trial_{trial['index']:02d}"
    if (result.get("case_id") != trial["case_id"] or result.get("status") != "complete"
            or result.get("complete_native_rollout") is not True):
        raise ValueError("Native result does not identify this complete case")
    required = {"report_path": "physics/report.json", "proposal_path": "proposal.json",
                "trajectory_path": "inputs/agent_trajectory_full.npz",
                "guard_path": "safety/runtime_guard.json", "receipt_path": "native/receipt.json",
                "evaluation_path": "evaluation.json"}
    hashes = result.get("output_sha256", {})
    for key, relative in required.items():
        expected = (directory / relative).resolve()
        if not result.get(key) or Path(result[key]).resolve() != expected or str(expected) not in hashes:
            raise ValueError("Native result has unbound output: " + key)
    for path, expected in hashes.items():
        if not Path(path).resolve().is_relative_to(directory.resolve()) or file_sha(path) != expected:
            raise ValueError("Native output fingerprint mismatch: " + path)
    return hashes


def _api_callback(store, trial, role):
    def reserve(request_id):
        store.reserve("api", "api:" + str(request_id),
                      {"trial": trial["id"], "role": role})
    return reserve


def settle_api(store, response, path):
    request_ids = list(response.get("request_sha256s", []))
    if response.get("request_sha256"):
        request_ids.append(response["request_sha256"])
    for request_id in set(request_ids):
        if "api:" + request_id in store.read()["reservations"]:
            store.settle("api:" + request_id, "complete", str(path))


def execute_trial(store, index_case, resources=None):
    from . import agents, actions
    from .evidence import audit_report
    from .native import source_dependency_hashes
    manifest = read_json(store.run / "manifest.json")
    frozen_inputs = {}
    if "source_sha256" in manifest:
        frozen_inputs = source_dependency_hashes(index_case["case_id"])
        if any(manifest["source_sha256"].get(p) != h for p, h in frozen_inputs.items()):
            raise ValueError("Case source inputs differ from campaign initialization")
    case = store.read()["cases"][index_case["case_id"]]
    trial = store.start_trial(index_case["case_id"], choose_parent(case))
    task_dir = store.run / "tasks" / trial["id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    parent_id = trial["parent_id"]
    parent = None
    action, proposal, packet = {}, {}, None
    if parent_id is None:
        frozen_inputs.update(manifest.get("initial_artifact_sha256", {}).get(index_case["case_id"], {}))
        if any(file_sha(p) != h for p, h in frozen_inputs.items()):
            raise ValueError("Initial candidate artifacts changed")
        candidate, provenance = initial_candidate(index_case)
        baseline_path = provenance.get("baseline_trajectory_path")
        store.update_trial(trial["id"], phase="baseline", provenance=provenance)
    else:
        packet_path = task_dir / "task_packet.json"
        packet = read_json(packet_path) if packet_path.exists() else packet_for(store, index_case, parent_id, trial)
        pinned_json(packet_path, packet)
        parent = packet["parent"]
        if packet.get("shared_program_proposal"):
            proposal = copy.deepcopy(packet["shared_program_proposal"])
            pinned_json(task_dir / "shared_program_proposal.json", proposal)
        else:
            with resources.api if resources else nullcontext():
                proposal = agents.propose(packet, task_dir / "expert", manifest["profile"],
                                          reserve_request=_api_callback(store, trial, "case_specialist"))
            settle_api(store, proposal, task_dir / "expert")
        if proposal.get("status") in ("mechanism_handoff", "unresolved_within_allowed_controls"):
            atomic_json(task_dir / "research_handoff.json", {
                "task_packet": str(packet_path), "proposal": proposal,
                "requires": "retained evidence for future bounded research",
                "reason": "Case agent cannot justify a further allowed control; case remains unsolved"})
            store.finish_trial(trial["id"], "mechanism_handoff")
            store.set_case_status(trial["case_id"], "unresolved", "Model reported no justified remaining control")
            return
        action = proposal["action"]
        source_config = read_json(Path(index_case["source_dir"]) / "config.json")
        try:
            candidate = actions.compile_action(parent["candidate"], action, source_config,
                                                duration_s=packet["duration_s"])
        except ValueError as exc:
            store.finish_trial(trial["id"], "compile_rejected", observation={
                "action_id": action["id"], "mechanism": action["mechanism"],
                "action_signature": action.get("signature"), "parent_sha256": action.get("parent_sha256"),
                "parent_id": parent_id, "physical_completed": False, "reason": str(exc)[:1500]})
            return
        baseline_path = None
    candidate_path = task_dir / "candidate.json"
    pinned_json(candidate_path, candidate)
    job = {"case_id": trial["case_id"], "run_root": str(store.run),
           "index": trial["index"], "candidate": candidate,
           "expected_artifact_sha256": {**frozen_inputs, str(candidate_path): file_sha(candidate_path)}}
    if parent:
        job["parent"] = {key: parent[key] for key in (
            "proposal_path", "report_path", "trajectory_path") if parent.get(key)}
    if packet and packet.get("shared_program_instance", {}).get("program_version") == "grasp_table_clearance_stage_v3":
        instance = packet["shared_program_instance"]
        job["stage_mode"] = "bounded_stage_v1"
        job["grasp_clearance_probe"] = {
            "hand": instance["hand"], "start_s": instance["probe_start_s"],
            "end_s": instance["probe_end_s"], "object": instance["anchor_object"],
            "close_s": instance["close_s"]}
    if packet and packet.get("shared_program_instance", {}).get("program_version") == "dominant_axis_grasp_centering_v5":
        instance = packet["shared_program_instance"]
        job["grasp_alignment_probe"] = {
            "hand": instance["hand"], "object": instance["anchor_object"],
            "close_s": instance["close_s"]}
        job["grasp_table_diagnostic_probe"] = {
            "hand": instance["hand"], "start_s": instance["probe_start_s"],
            "end_s": instance["probe_end_s"]}
    if baseline_path:
        job["baseline_trajectory_path"] = baseline_path
    with resources.physics if resources else nullcontext():
        preview = run_native(store, trial, job, "preflight")
    if preview.get("status") != "validated":
        store.finish_trial(trial["id"], "preflight_rejected" if preview.get("status") == "rejected" else "tool_error",
                           observation={"action_id": action.get("id"), "mechanism": action.get("mechanism"),
                                        "action_signature": action.get("signature"),
                                        "parent_sha256": action.get("parent_sha256"),
                                        "parent_id": parent_id, "physical_completed": False,
                                        "details": preview})
        return
    if parent:
        receipt_path = preview.get("receipt_path")
        if not receipt_path or not Path(receipt_path).is_file():
            store.finish_trial(trial["id"], "tool_error", observation={
                "action_id": action.get("id"), "mechanism": action.get("mechanism"),
                "parent_id": parent_id, "physical_completed": False,
                "reason": "Validated native preflight lacks a saved receipt"})
            return
        receipt = read_json(receipt_path)
        review_packet = copy.deepcopy(packet)
        review_packet["preflight_receipt"] = {
            "path": str(Path(receipt_path).resolve()), "sha256": file_sha(receipt_path),
            "summary": {key: receipt.get(key) for key in (
                "schema", "candidate_sha256", "frozen_scene_sha256",
                "frozen_model_profile_sha256", "modified_hands",
                "max_single_action_world_axis_m", "max_total_source_world_axis_m",
                "exact_control_scope", "grasp_clearance", "grasp_alignment",
                "grasp_table_diagnostic", "safety_mode",
                "stage_world_axis_bound_m", "joint_command_audit",
                "physical_started")}}
        review_packet["evidence_refs"] = list(dict.fromkeys([
            *review_packet["evidence_refs"], review_packet["preflight_receipt"]["path"]]))
        pinned_json(task_dir / "review_packet.json", review_packet)
        with resources.api if resources else nullcontext():
            reviewed = agents.review(review_packet, proposal, task_dir / "reviewer", manifest["profile"],
                                     reserve_request=_api_callback(store, trial, "independent_reviewer"))
        settle_api(store, reviewed, task_dir / "reviewer")
        atomic_json(task_dir / "review.json", reviewed)
        if reviewed.get("verdict") != "proceed":
            store.finish_trial(trial["id"], "review_rejected", observation={
                "action_id": action["id"], "mechanism": action["mechanism"],
                "action_signature": action.get("signature"),
                "parent_sha256": action.get("parent_sha256"),
                "parent_id": parent_id, "reason": reviewed.get("reasons"),
                "preflight_receipt_sha256": review_packet["preflight_receipt"]["sha256"],
                "physical_completed": False})
            return
    with resources.physics if resources else nullcontext():
        result = run_native(store, trial, job, "evaluate")
    report_path = result.get("report_path") or result.get("evaluation", {}).get("report_path")
    if result.get("status") != "complete" or not report_path:
        store.finish_trial(trial["id"], "tool_error", observation={
            "action_id": action.get("id"), "mechanism": action.get("mechanism"),
            "action_signature": action.get("signature"),
            "parent_sha256": action.get("parent_sha256"),
            "parent_id": parent_id, "physical_completed": False, "details": result})
        return
    audit = audit_report(report_path, manifest["native_runtime_sha256"], require_guard=True)
    try:
        output_sha256 = verify_native_result(result, store, trial)
    except (ValueError, OSError) as exc:
        audit["valid"] = False
        audit.setdefault("errors", []).append(str(exc))
    atomic_json(task_dir / "result_audit.json", audit)
    if not audit["valid"]:
        store.finish_trial(trial["id"], "invalid_evidence", observation={
            "physical_completed": True, "audit": audit})
        return
    after = read_json(report_path)
    transition = assess_transition(parent["report"] if parent else None, after,
                                   packet["target_predicate"] if packet else None)
    if packet and packet.get("stage") == "carry":
        from .routing import retention_progress
        contract = read_json(Path(index_case["source_dir"]) / "sim2/task_contract.json")
        measured = retention_progress(packet.get("stage_evidence"), after, contract, report_path)
        if measured and not after["objective"]["hard_failures"]:
            transition["stage_evidence"].extend(measured)
            transition["no_effect"] = False
            if transition["branch_kind"] is None:
                transition["branch_kind"] = "exploration"
    if parent_id:
        case_now = store.read()["cases"][trial["case_id"]]
        stable_report = read_json(case_now["branches"][case_now["stable"]]["report_path"])
        stable_transition = assess_transition(stable_report, after, packet["target_predicate"])
        if transition["branch_kind"] == "stable" and stable_transition["branch_kind"] != "stable":
            transition["branch_kind"] = "exploration" if not after["objective"]["hard_failures"] else None
    branch = None
    if transition["branch_kind"]:
        physics_dir = Path(report_path).parent
        proposal_path = result.get("proposal_path") or result.get("evaluation", {}).get("proposal_path")
        if not proposal_path:
            proposal_path = str(physics_dir.parent / "proposal.json")
        branch = {"kind": transition["branch_kind"], "rank": transition["rank"],
                  "report_path": str(report_path), "report_sha256": file_sha(report_path),
                  "output_sha256": output_sha256,
                  "proposal_path": str(proposal_path),
                  "trajectory_path": str(physics_dir.parent / "inputs/agent_trajectory_full.npz"),
                  "followups": 0, "parent": parent_id}
    store.finish_trial(trial["id"], "native_passed" if transition["native_pass"] else "physical_failed",
                       branch, {"action_id": action.get("id"), "mechanism": action.get("mechanism"),
                                "action_signature": action.get("signature"),
                                "parent_sha256": action.get("parent_sha256"),
                                "parent_id": parent_id, "report_path": str(report_path),
                                "physical_completed": True, **transition})


def write_summary(store):
    state = store.read()
    manifest = read_json(store.run / "manifest.json")
    by_family = {}
    for family in FAMILIES:
        cases = [v for v in state["cases"].values() if v["family"] == family]
        by_family[family] = {"total": len(cases), "verified_success": sum(c["status"] == "solved" for c in cases),
                             "research_needed": sum(c["status"] == "research_needed" for c in cases),
                             "unresolved": sum(c["status"] == "unresolved" for c in cases),
                             "case_budget_exhausted": sum(c["status"] == "budget_exhausted" for c in cases),
                             "evaluated": sum(bool(c["branches"]) for c in cases)}
    passed, evidence_errors = [], []
    for cid, case in state["cases"].items():
        if case["status"] != "solved":
            continue
        branch = case["branches"][case["success"]["trial"]]
        hashes = branch.get("output_sha256", {})
        try:
            if not hashes or any(file_sha(p) != h for p, h in hashes.items()):
                raise ValueError("Saved success artifacts changed")
        except (OSError, ValueError):
            evidence_errors.append({"case_id": cid, "error": "success_artifact_mismatch"})
            by_family[case["family"]]["verified_success"] -= 1
            continue
        passed.append(dict(case_id=cid, **case["success"]))
    summary = {"schema": "rssr.family.summary.v1", "status": "evidence_invalid" if evidence_errors else state["status"],
               "goal_complete": len(passed) == 150, "planned_cases": 150,
               "verified_success": len(passed), "by_family": by_family,
               "attempts_charged": state["spent"], "completed_operations": state["completed"],
               "remaining": store.remaining(), "events": len(state["events"]),
               "historical_success_not_counted": True, "successes": passed,
               "evidence_errors": evidence_errors,
               "active_agents": list(state.get("active_agents", {}).values()),
               "manifest_sha256": digest(manifest)}
    atomic_json(store.run / "summary.json", summary)
    atomic_json(store.run / "dataset_eligibility.json", {
        "schema": "rssr.family.verified_dataset.v1", "cases": passed,
        "requires": "full native v80 hard/task pass, complete guard and pinned evidence"})
    return summary


class CaseResources:
    """Case ownership is separate from model/network/physics concurrency."""
    def __init__(self, max_steps=None, api_concurrency=2):
        self.api = threading.BoundedSemaphore(api_concurrency)
        self.physics = threading.BoundedSemaphore(1)
        self.stop = threading.Event()
        self.step_lock = threading.Lock()
        self.steps = 0
        self.max_steps = max_steps
        self.global_budget_reason = None

    def claim_step(self):
        with self.step_lock:
            if self.stop.is_set() or (self.max_steps is not None and self.steps >= self.max_steps):
                return False
            self.steps += 1
            return True

    def release_step(self):
        with self.step_lock:
            if self.steps <= 0:
                raise ValueError("No reserved launch step to release")
            self.steps -= 1


def _close_pending(store, case_id, outcome):
    state = store.read()
    trial_id = state["cases"][case_id]["pending_trial"]
    if trial_id:
        # Unknown provider outcomes stay charged and are never resubmitted.
        for key, value in state["reservations"].items():
            if value["status"] == "started" and (value["metadata"].get("trial") == trial_id
                    or key in ("full:" + trial_id, "short:" + trial_id)):
                store.settle(key, "interrupted", "case agent stopped; saved request receipts retained")
        store.finish_trial(trial_id, outcome, observation={"physical_completed": False})


def case_worker(store, index_case, resources, *, baseline_only=False, first_step_reserved=False):
    """A worker retains one case across proposal/review/evaluation iterations."""
    from .store import CaseBudgetExhausted
    case_id = index_case["case_id"]
    agent_id = "case:" + case_id
    with store.transaction() as box:
        case = box[0]["cases"][case_id]
        case["owner_agent"] = agent_id
        box[0].setdefault("active_agents", {})[agent_id] = {
            "agent_id": agent_id, "case_id": case_id, "family": case["family"], "started_at": utc_now()}
    store.record("case_agent_started", {"agent_id": agent_id, "case_id": case_id})
    try:
        while not resources.stop.is_set():
            current_state = store.read()
            case = current_state["cases"][case_id]
            if case["status"] in ("solved", "unresolved", "research_needed", "budget_exhausted"):
                break
            remaining = store.case_remaining(case_id)
            paid_pending = case["pending_trial"] and "full:" + case["pending_trial"] in current_state["reservations"]
            if not paid_pending and remaining.get("full") is not None and remaining["full"] <= 0:
                usage = store.case_usage(case_id)
                raise CaseBudgetExhausted(case_id, "full", usage["full"], usage["full"])
            if not case["pending_trial"] and remaining.get("steps") is not None and remaining["steps"] <= 0:
                usage = store.case_usage(case_id)
                raise CaseBudgetExhausted(case_id, "steps", usage["steps"], usage["steps"])
            if (not paid_pending and store.remaining()["full"] <= 0) or store.remaining()["wall_seconds"] <= 0:
                raise BudgetExhausted("Global physical or wall-clock budget exhausted")
            if not first_step_reserved and not resources.claim_step():
                break
            first_step_reserved = False
            print(json.dumps({"event": "case_step", "agent_id": agent_id,
                              "case_id": case_id, "global_step": resources.steps}), flush=True)
            execute_trial(store, index_case, resources=resources)
            if baseline_only:
                break
        return {"case_id": case_id, "status": store.read()["cases"][case_id]["status"]}
    except CaseBudgetExhausted as exc:
        _close_pending(store, case_id, "case_budget_exhausted")
        store.set_case_status(case_id, "budget_exhausted", str(exc))
        return {"case_id": case_id, "status": "case_budget_exhausted"}
    except BudgetExhausted as exc:
        resources.global_budget_reason = str(exc)
        resources.stop.set()
        _close_pending(store, case_id, "global_budget_exhausted")
        return {"case_id": case_id, "status": "global_budget_exhausted"}
    except Exception as exc:
        _close_pending(store, case_id, "agent_error")
        store.set_case_status(case_id, "unresolved", "Case agent error: " + type(exc).__name__)
        store.record("case_agent_error", {"case_id": case_id, "error_type": type(exc).__name__})
        return {"case_id": case_id, "status": "unresolved", "error_type": type(exc).__name__}
    finally:
        if first_step_reserved:
            resources.release_step()
        with store.transaction() as box:
            box[0].setdefault("active_agents", {}).pop(agent_id, None)
        store.record("case_agent_stopped", {"agent_id": agent_id, "case_id": case_id})


def run_campaign(store, *, cases=None, max_trials=None, baseline_only=False, workers=None):
    from .research import active_revision_guard
    manifest = validate_identity(store.run)
    index = {c["case_id"]: c for c in read_json(store.run / "evidence_index.json")["cases"]}
    if cases and not set(cases) <= set(index):
        raise ValueError("Unknown case filter")
    workers = manifest.get("case_workers", 3) if workers is None else workers
    if type(workers) is not int or not 1 <= workers <= manifest.get("case_workers", 3):
        raise ValueError("Case worker count exceeds frozen campaign limit")
    resources = CaseResources(max_trials, manifest.get("max_api_concurrency", 2))
    with active_revision_guard(store.run), (store.run / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store.set_status("running")
        # Process-local leases from a crashed invocation cannot remain active.
        with store.transaction() as box:
            box[0]["active_agents"] = {}
        visited, active = set(), {}
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="case-agent")
        try:
            while True:
                while len(active) < workers and not resources.stop.is_set():
                    if max_trials is not None and resources.steps >= max_trials:
                        break
                    state = store.read()
                    scope = [cid for cid in (cases or list(index))
                             if cid not in active.values() and cid not in visited]
                    if baseline_only:
                        scope = [cid for cid in scope if not state["cases"][cid]["branches"]]
                    if not scope:
                        break
                    case_id = choose_case(state, scope)
                    if case_id is None:
                        break
                    if not resources.claim_step():
                        break
                    visited.add(case_id)
                    with store.transaction() as box:
                        box[0]["family_cursor"] = (FAMILIES.index(index[case_id]["family"]) + 1) % 3
                    future = pool.submit(case_worker, store, index[case_id], resources,
                                         baseline_only=baseline_only, first_step_reserved=True)
                    active[future] = case_id
                if not active:
                    break
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    case_id = active.pop(future)
                    result = future.result()
                    summary = write_summary(store)
                    print(json.dumps({"event": "case_agent_result", **result,
                                      "verified_success": summary["verified_success"],
                                      "charged": summary["attempts_charged"]}), flush=True)
                if not done:
                    summary = write_summary(store)
                    print(json.dumps({"event": "heartbeat", "active_cases": sorted(active.values()),
                                      "charged": summary["attempts_charged"]}), flush=True)
            state = store.read()
            complete = all(c["status"] == "solved" for c in state["cases"].values())
            if resources.global_budget_reason:
                store.set_status("budget_exhausted", resources.global_budget_reason)
            else:
                store.set_status("complete" if complete else "checkpoint",
                                 "case agents reached terminal status or launch checkpoint")
        except KeyboardInterrupt:
            resources.stop.set()
            store.set_status("interrupted", "Saved case ownership and evidence retained")
            raise
        except Exception as exc:
            resources.stop.set()
            store.set_status("blocked", type(exc).__name__)
            print(json.dumps({"event": "blocked", "error_type": type(exc).__name__}), flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            write_summary(store)
    return write_summary(store)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "run", "validate", "handoffs", "fork"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--workers", type=int, help="Concurrent single-case agents (maximum 3); physics remains serial")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--profile", default="gpt6luna")
    parser.add_argument("--index-from", type=Path, help="Reuse an existing read-only evidence index at initialization")
    parser.add_argument("--parent-run", type=Path, help="Parent campaign for a code research revision; budgets carry forward")
    args = parser.parse_args(argv)
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.command == "fork":
        if args.parent_run is None:
            parser.error("fork requires --parent-run")
        from .research import fork_revision
        summary = write_summary(fork_revision(args.parent_run, args.run, requeue_cases=args.cases))
    elif args.command == "handoffs":
        from .research import list_work_items
        summary = {"work_items": list_work_items(args.run)}
    elif args.command == "init":
        store = initialize(args.run, profile=args.profile, index_from=args.index_from)
        summary = write_summary(store)
    elif args.command == "validate":
        validate_identity(args.run)
        summary = {"identity_valid": True, "events_valid": True}
    elif args.command == "status":
        summary = write_summary(Store(args.run))
    else:
        summary = run_campaign(Store(args.run), cases=args.cases,
                               max_trials=args.max_trials, baseline_only=args.baseline_only, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
