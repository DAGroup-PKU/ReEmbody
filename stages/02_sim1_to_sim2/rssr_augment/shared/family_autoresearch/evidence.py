"""Read-only historical evidence index and fail-closed native Sim2 audit.

Historical coverage is explicitly distinct from a new campaign's coverage.  A
valid report is not automatically a usable baseline: its physical scene and
model profile must also match the frozen v74 selection for that case.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

NATIVE_RUNTIME_SHA256 = "6047d6c0ebfc762fc3017dcbe34b4eb72a15fbf6b5aaf00d4f831ac2c51640a9"
V76_RUN = "cobot150_v76_fromzero_round01_20260922"
V74_RUN = "rssr_cobot_sim2_v74_action_centered_20260921"
HARD_CONSTRAINTS = (
    "robot_table_penetration_le_3mm", "limited_joints_within_2mrad_or_2mm",
    "solver_no_warnings", "waist_coupling_within_1mrad", "gripper_coupling_within_2mrad",
)
FAMILIES = ("drawer", "stack_blocks", "stack_cups")
CASE_RE = re.compile(r"^(drawer|stack_blocks|stack_cups)_ep\d{6}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def physically_equal(a: Any, b: Any, tolerance: float = 1e-10) -> bool:
    """Ignore only descriptive scene annotations, never physical parameters."""
    if isinstance(a, dict) and isinstance(b, dict):
        ignored = {"evidence", "head_center", "color", "rgba"}
        keys_a, keys_b = set(a) - ignored, set(b) - ignored
        return keys_a == keys_b and all(physically_equal(a[k], b[k], tolerance) for k in keys_a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(physically_equal(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance
    return type(a) is type(b) and a == b


def _metadata_errors(value: Any, prefix: str = "") -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else key
            if key in {"partial", "partial_rollout", "truncated", "safety_truncated", "probe_only", "diagnostic_only",
                       "objects_forced", "object_weld_used", "robot_qpos_forced_during_execution",
                       "object_state_forced_during_rollout", "executing_state_written_by_guard"} and item:
                errors.append(f"unsafe_or_partial:{label}")
            if key in {"stop_s", "rollout_stop_s", "short_rollout_stop_s", "diagnostic_intervention", "_diagnostic_probe"} and item is not None and item is not False:
                errors.append(f"partial_or_diagnostic:{label}")
            if key in {"complete_native_rollout", "full_rollout", "native_rollout_complete", "complete"} and item is False:
                errors.append(f"incomplete:{label}")
            if key in {"first_trigger", "safety_trigger", "abort_reason"} and item is not None and item is not False and item != "":
                errors.append(f"interrupted:{label}")
            errors.extend(_metadata_errors(item, label))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_metadata_errors(item, f"{prefix}[{index}]"))
    return errors


def _times(path: Path, key: str = "time_s") -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        times = arrays[key].astype(float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("invalid time vector")
    return times


def audit_report(report_path: Path, expected_runtime_sha: str | None = None,
                 require_guard: bool = False) -> dict:
    """Validate native evidence; no summary, agent statement or vote can pass it.

    Unguarded historical runs require matching physical and command horizons.
    New executions should set require_guard=True.  Existing contradictory guard
    or evaluation metadata always rejects a report, even if its booleans pass.
    """
    path = Path(report_path).resolve()
    result = {"valid": False, "passed": False, "complete": False, "errors": [],
              "hard_failures": [], "task_failures": [], "predicate_vector": {},
              "report_path": str(path), "report_sha256": None, "scene_sha256": None,
              "runtime_sha256": None, "guard_path": None, "guard_sha256": None,
              "evidence_fingerprints": {}, "strict_v1_passed": False}
    errors = result["errors"]
    try:
        report = _read(path)
        result["report_sha256"] = sha256(path)
        result["evidence_fingerprints"][str(path)] = result["report_sha256"]
        runtime = report.get("runtime_audit", {}).get("pilot_physics_sha256")
        result["runtime_sha256"] = runtime
        if runtime != NATIVE_RUNTIME_SHA256 or runtime != (expected_runtime_sha or NATIVE_RUNTIME_SHA256):
            errors.append("non_native_or_changed_runtime")
        if report.get("kind") != "sim2" or report.get("judgment_version") != "task_completion_v1":
            errors.append("non_native_judgment")
        for field in ("objects_forced", "object_weld_used", "robot_qpos_forced_during_execution"):
            if report.get(field) is not False:
                errors.append(f"missing_or_unsafe:{field}")
        errors.extend(_metadata_errors(report, "report"))

        evaluation = report.get("task_evaluation", {})
        hard = evaluation.get("hard_constraints", {})
        predicates = evaluation.get("predicates", [])
        task = {p["id"]: p.get("passed") for p in predicates if isinstance(p, dict) and "id" in p}
        result["predicate_vector"] = {**hard, **task}
        if set(hard) != set(HARD_CONSTRAINTS) or not task or len(task) != len(predicates):
            errors.append("missing_or_duplicate_predicates")
        if any(type(passed) is not bool for passed in [*hard.values(), *task.values()]):
            errors.append("non_boolean_predicate")
        hard_failed = sorted(name for name, passed in hard.items() if passed is not True)
        task_failed = sorted(name for name, passed in task.items() if passed is not True)
        result.update(hard_failures=hard_failed, task_failures=task_failed)
        objective = report.get("objective", {})
        raw_passed = bool(task and hard) and not hard_failed and not task_failed
        for stated, measured, label in (
            (objective.get("hard_failures"), hard_failed, "objective_hard"),
            (objective.get("task_failures"), task_failed, "objective_task"),
            (evaluation.get("hard_failures"), hard_failed, "evaluation_hard"),
            (evaluation.get("failed_predicates"), task_failed, "evaluation_task"),
        ):
            if not isinstance(stated, list) or sorted(stated) != measured:
                errors.append(f"inconsistent:{label}")
        for stated in (objective.get("passed"), evaluation.get("task_success"), report.get("task_success")):
            if stated is not raw_passed:
                errors.append("inconsistent:passed")
        if any(report.get("criteria", {}).get(k) is not v for k, v in result["predicate_vector"].items()):
            errors.append("inconsistent:criteria")
        native = report.get("judgments", {}).get("task_completion_v1", {})
        if native.get("objective") != objective or native.get("task_evaluation") != evaluation:
            errors.append("inconsistent:native_judgment")
        result["strict_v1_passed"] = report.get("judgments", {}).get("strict_v1", {}).get("objective", {}).get("passed") is True

        for filename in ("scene.xml", "task_contract.json", "rollout.npz", "command_targets.npz"):
            artifact = path.parent / filename
            if not artifact.is_file():
                errors.append(f"missing:{filename}")
            else:
                result["evidence_fingerprints"][str(artifact)] = sha256(artifact)
        scene = path.parent / "scene.xml"
        if scene.is_file():
            result["scene_sha256"] = sha256(scene)
            if result["scene_sha256"] != report.get("scene_sha256"):
                errors.append("scene_hash_mismatch")
        contract_path = path.parent / "task_contract.json"
        if contract_path.is_file():
            contract = _read(contract_path)
            if {p["id"] for p in contract.get("predicates", [])} != set(task):
                errors.append("contract_predicate_mismatch")
            if set(contract.get("hard_constraints", [])) != set(hard):
                errors.append("contract_hard_mismatch")
            if contract.get("contract_id") != evaluation.get("contract_id") or contract.get("contract_id") != report.get("task_contract_id"):
                errors.append("contract_id_mismatch")

        trial = path.parent.parent
        guard_paths = [trial / "safety/runtime_guard.json", path.parent / "runtime_guard.json"]
        metadata_paths = {trial / "evaluation.json", trial / "result.json", trial / "proposal.json", trial / "experiment.json"}
        metadata_paths.update(trial.glob("*receipt*.json"))
        for metadata_path in sorted(metadata_paths):
            if metadata_path.is_file():
                metadata = _read(metadata_path)
                errors.extend(_metadata_errors(metadata, metadata_path.name))
                for key in ("bilateral_safety_audit_path", "runtime_guard_path", "guard_path"):
                    if metadata.get(key):
                        candidate = Path(metadata[key])
                        guard_paths.append(candidate if candidate.is_absolute() else trial / candidate)
        guards = list(dict.fromkeys(p.resolve() for p in guard_paths if p.is_file()))
        if require_guard and not guards:
            errors.append("missing_runtime_guard")
        for guard_path in guards:
            guard = _read(guard_path)
            result.update(guard_path=str(guard_path), guard_sha256=sha256(guard_path))
            result["evidence_fingerprints"][str(guard_path)] = result["guard_sha256"]
            errors.extend(_metadata_errors(guard, "runtime_guard"))
            if guard.get("complete_native_rollout") is not True:
                errors.append("guard_not_complete")
            if guard.get("executing_state_written_by_guard") is not False:
                errors.append("guard_state_write_unverified")
            if guard.get("schema") == "rssr.family.bounded_stage_runtime_guard.v1":
                if (guard.get("stage_world_axis_bound_m") != .1
                        or not isinstance(guard.get("stage_safety_code_sha256"), str)
                        or len(guard["stage_safety_code_sha256"]) != 64):
                    errors.append("invalid_stage_guard_identity")
                limits = (("max_single_action_world_axis_m", .1),
                          ("max_total_source_world_axis_m", .1))
            else:
                # Preserve the existing legacy guard audit for older complete
                # native reports and intentionally minimal audit fixtures.
                limits = (("max_single_action_world_axis_m", .04),
                          ("max_total_source_world_axis_m", .16))
            for field, limit in limits:
                values = np.asarray(guard.get(field, []), dtype=float)
                if not values.size or not np.isfinite(values).all() or np.abs(values).max() > limit + 1e-9:
                    errors.append(f"guard_limit:{field}")

        try:
            measured = _times(path.parent / "rollout.npz")
            command = _times(path.parent / "command_targets.npz")
            duration = float(report.get("simulation_seconds", 0))
            dt = float(report.get("joint_limit_audit", {}).get("timestep_s", .002))
            tolerance = max(2 * dt, .002)
            if measured[0] > command[0] + tolerance or measured[-1] < command[-1] - tolerance:
                errors.append("partial_command_horizon")
            input_trajectory = trial / "inputs/agent_trajectory_full.npz"
            if input_trajectory.is_file():
                original_times = _times(input_trajectory, "source_time_s")
                if measured[0] > original_times[0] + tolerance or measured[-1] < original_times[-1] - tolerance:
                    errors.append("partial_input_trajectory_horizon")
            if not math.isfinite(duration) or duration < command[-1] - tolerance or abs(duration - measured[-1]) > .021:
                errors.append("partial_or_inconsistent_rollout_duration")
            result["simulation_seconds"] = duration
            result["complete"] = not any("partial" in e or "incomplete" in e or "interrupt" in e or "guard_not_complete" in e for e in errors)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"unverifiable_rollout:{type(exc).__name__}")
        result["valid"] = not errors
        result["passed"] = result["valid"] and result["complete"] and raw_passed
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"unreadable_or_malformed_evidence:{type(exc).__name__}:{exc}")
    result["errors"] = list(dict.fromkeys(errors))
    return result


def _stage(failures: list[str], hard: list[str]) -> str:
    if hard:
        return "hard_constraints"
    if any("lift" in name for name in failures):
        return "grasp_and_hold"
    if any(name.startswith(("inside", "initial_outside")) for name in failures):
        return "placement"
    if any("open_insert_close" in name for name in failures):
        return "drawer_closure"
    if any("released" in name for name in failures):
        return "release"
    return "placement_and_stability" if failures else "replay_verification"


def build_index(root: Path) -> dict:
    """Index every manifest case without modifying any historical artifact."""
    root = Path(root).resolve()
    summary_path = root / "runs" / V76_RUN / "summary.json"
    summary = _read(summary_path)
    manifest_path = summary_path.with_name("manifest.json")
    jobs = _read(manifest_path).get("jobs", []) if manifest_path.is_file() else summary.get("results", [])
    expected_ids = {f"{family}_ep{episode:06d}" for family in FAMILIES for episode in range(50)}
    if len(jobs) != 150 or {row["case_id"] for row in jobs} != expected_ids:
        raise ValueError("The frozen manifest must contain exactly 50 cases of each of the three families")
    results = {row["case_id"]: row for row in summary.get("results", [])}
    source_root = root.parent / "r2r/results" / V74_RUN / "cases"
    cases = {}
    for job in sorted(jobs, key=lambda row: row["case_id"]):
        case_id = job["case_id"]
        source = source_root / case_id
        selection_path = source / "sim2/selection.json"
        entry = {"case_id": case_id, "family": job.get("task", case_id.rsplit("_ep", 1)[0]),
                 "source_sim1_passed": job.get("source_sim1_passed", results.get(case_id, {}).get("source_sim1_passed")),
                 "source_dir": str(source), "source_selection_path": str(selection_path),
                 "baseline_candidates": [], "diagnostic_history": [], "initial_candidate": None,
                 "stage": "baseline_required", "priority": 0, "index_errors": []}
        try:
            selected = _read(selection_path)
            trial_physics = Path(selected["trial"])
            trajectory = trial_physics.parent / "inputs/agent_trajectory_full.npz"
            entry["initial_candidate"] = {
                "scene": selected["scene"], "program": selected["program"], "windows": {},
                "model_profile": selected.get("model_profile", selected.get("report", {}).get("model_profile", {})),
                "cost": 0., "decision": {"lever": "baseline", "source": "frozen_v74_selection"},
                "origin": "frozen_v74_selection_requires_native_replay", "proposal_path": str(selection_path),
                "trajectory_path": str(trajectory) if trajectory.is_file() else None,
                "report_path": str(trial_physics / "report.json") if (trial_physics / "report.json").is_file() else None,
                "source_selection_sha256": sha256(selection_path),
            }
            objective = selected.get("objective", selected.get("report", {}).get("objective", {}))
            entry["stage"] = _stage(objective.get("task_failures", []), objective.get("hard_failures", []))
            entry["priority"] = len(objective.get("hard_failures", [])) * 10 + len(objective.get("task_failures", []))
        except (OSError, ValueError, KeyError) as exc:
            entry["index_errors"].append(f"source_unavailable:{exc}")
        cases[case_id] = entry

    scanned, native_count = 0, 0
    historical_pass, compatible_pass = set(), set()
    for report_path in sorted((root / "runs").glob("*/cases/*/**/physics/report.json")):
        case_id = next((part for part in report_path.parts if CASE_RE.fullmatch(part)), None)
        if case_id not in cases:
            continue
        scanned += 1
        try:
            raw = report_path.read_bytes()
            if b'"runtime_audit"' not in raw:
                continue
            report = json.loads(raw)
            if report.get("runtime_audit", {}).get("pilot_physics_sha256") != NATIVE_RUNTIME_SHA256:
                continue
        except (OSError, ValueError, AttributeError):
            continue
        native_count += 1
        audit = audit_report(report_path)
        if audit["passed"]:
            historical_pass.add(case_id)
        proposal_path = report_path.parent.parent / "proposal.json"
        proposal = None
        reasons = list(audit["errors"])
        if proposal_path.is_file():
            try:
                proposal = _read(proposal_path)
            except (OSError, ValueError) as exc:
                reasons.append(f"malformed_proposal:{exc}")
        else:
            reasons.append("missing_proposal")
        entry = cases[case_id]
        initial = entry["initial_candidate"]
        if initial is None:
            reasons.append("missing_frozen_source")
        elif proposal is not None:
            if not physically_equal(proposal.get("scene"), initial["scene"]):
                reasons.append("frozen_scene_mismatch")
            if not physically_equal(proposal.get("model_profile", {}), initial["model_profile"]):
                reasons.append("frozen_model_profile_mismatch")
            if not physically_equal(report.get("model_profile", {}), proposal.get("model_profile", {})):
                reasons.append("proposal_report_model_profile_mismatch")
            if proposal.get("candidate_sha256"):
                identity = {key: proposal.get(key, {}) for key in ("program", "scene", "windows", "model_profile")}
                if _digest(identity) != proposal["candidate_sha256"]:
                    reasons.append("proposal_candidate_hash_mismatch")
            evaluation_path = report_path.parent.parent / "evaluation.json"
            if evaluation_path.is_file():
                try:
                    historical_evaluation = _read(evaluation_path)
                    for key, actual in (("report_path", report_path), ("proposal_path", proposal_path)):
                        if historical_evaluation.get(key) and Path(historical_evaluation[key]).resolve() != actual.resolve():
                            reasons.append(f"evaluation_{key}_mismatch")
                    recorded_hash = historical_evaluation.get("candidate_sha256")
                    if recorded_hash and recorded_hash != proposal.get("candidate_sha256"):
                        reasons.append("evaluation_candidate_hash_mismatch")
                except (OSError, ValueError, TypeError) as exc:
                    reasons.append(f"malformed_evaluation:{exc}")
        trajectory = report_path.parent.parent / "inputs/agent_trajectory_full.npz"
        if not trajectory.is_file():
            reasons.append("missing_executed_trajectory")
        candidate = {key: audit[key] for key in ("report_path", "report_sha256", "scene_sha256", "passed",
                     "hard_failures", "task_failures", "runtime_sha256", "complete", "predicate_vector",
                     "strict_v1_passed", "guard_path", "guard_sha256")}
        candidate.update(proposal_path=str(proposal_path) if proposal is not None else None,
                         proposal_sha256=sha256(proposal_path) if proposal is not None else None,
                         trajectory_path=str(trajectory) if trajectory.is_file() else None,
                         origin="historical_native_v80", compatible=not reasons, exclusion_reasons=reasons)
        if reasons:
            entry["diagnostic_history"].append(candidate)
        else:
            entry["baseline_candidates"].append(candidate)
            if audit["passed"]:
                compatible_pass.add(case_id)
    for entry in cases.values():
        entry["baseline_candidates"].sort(key=lambda row: (not row["passed"], len(row["hard_failures"]), len(row["task_failures"]), row["report_path"]))
    return {"schema": "rssr.family_autoresearch.evidence_index.v1", "cases": list(cases.values()),
            "native_runtime_sha256": NATIVE_RUNTIME_SHA256, "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
            "summary_sha256": sha256(summary_path), "historical_v80_pass_cases": sorted(historical_pass),
            "compatible_verified_cases": sorted(compatible_pass), "fresh_campaign_pass_cases": [],
            "coverage_basis": "Historical evidence only; no fresh campaign evaluations are implied.",
            "scan": {"reports_seen": scanned, "native_v80_reports": native_count}}
