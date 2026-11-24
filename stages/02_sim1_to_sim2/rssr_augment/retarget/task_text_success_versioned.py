"""Read-only physical success scoring for the original three task texts.

The frozen v80 evaluation is retained separately. This scorer removes
requirements absent from the source task instruction; physical quality
predicates remain visible without being added as task gates.
"""
from __future__ import annotations

from pathlib import Path

from .native_evidence import audit_report
from rssr_augment.shared.family_autoresearch.store import file_sha, read_json


REQUIRED = {
    "drawer": ("inside",),
    "stack_blocks": ("stable_supported_on", "object_contact", "stable", "released"),
    "stack_cups": ("cup_nested", "object_contact", "stable", "released"),
}


def family_for(case_id: str) -> str:
    family = case_id.rsplit("_ep", 1)[0]
    if family not in REQUIRED:
        raise ValueError(f"Unknown frozen family: {case_id}")
    return family


def score_report(report_path: Path, case_id: str) -> dict:
    """Score a completed native report without changing the original judge."""
    family = family_for(case_id)
    report = read_json(report_path)
    audit = audit_report(report_path)
    evaluation = report["task_evaluation"]
    predicates = {item["kind"]: bool(item["passed"])
                  for item in evaluation["predicates"]}
    missing = [kind for kind in REQUIRED[family] if kind not in predicates]
    required_failures = [kind for kind in REQUIRED[family]
                         if not predicates.get(kind, False)]
    hard_failures = list(evaluation["hard_failures"])
    forbidden_state_write = bool(report.get("objects_forced") or
                                 report.get("object_weld_used") or
                                 report.get("robot_qpos_forced_during_execution"))
    task_goal_met = not missing and not required_failures
    execution_valid = (bool(audit["valid"]) and not hard_failures
                       and not forbidden_state_write)
    success = task_goal_met and execution_valid
    return {
        "case_id": case_id,
        "family": family,
        "report_path": str(report_path),
        "report_sha256": file_sha(report_path),
        "native_audit_valid": bool(audit["valid"]),
        "native_audit_errors": audit["errors"],
        "legacy_task_completion_v1": bool(evaluation["task_success"]),
        "required_predicates": list(REQUIRED[family]),
        "missing_predicates": missing,
        "required_failures": required_failures,
        "task_goal_met": task_goal_met,
        "execution_valid": execution_valid,
        "quality_predicates": {kind: predicates.get(kind)
                               for kind in ("object_contact", "stable", "released")},
        "hard_failures": hard_failures,
        "forbidden_state_write": forbidden_state_write,
        "task_text_physical_success": success,
    }
