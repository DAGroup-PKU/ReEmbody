"""Versioned rejudgment of measured Sim2 reports; never changes task predicates."""
from __future__ import annotations

import copy


LIMIT = "limited_joints_within_2mrad_or_2mm"


def dual_judgment(report):
    """Return strict and task-completion views from one immutable physical trace."""
    strict_objective = copy.deepcopy(report["objective"])
    strict_task = copy.deepcopy(report["task_evaluation"])
    audit = report["joint_limit_audit"]
    excess = audit["per_joint_excess_native_units"]
    tolerance = float(audit["tolerance"])
    if not excess or tolerance != 0.002:
        raise ValueError("missing or changed every-step joint-limit audit")
    non_gripper = {name: float(value) for name, value in excess.items()
                   if "gripper" not in name}
    gripper = {name: float(value) for name, value in excess.items()
               if "gripper" in name}
    hard = copy.deepcopy(strict_task["hard_constraints"])
    if LIMIT not in hard:
        raise ValueError("strict report lacks frozen limited-joint constraint")
    hard[LIMIT] = max(non_gripper.values(), default=0.0) <= tolerance
    new_task = copy.deepcopy(strict_task)
    new_task["hard_constraints"] = hard
    new_task["hard_failures"] = [name for name, passed in hard.items() if not passed]
    new_task["task_success"] = not new_task["hard_failures"] and not new_task["failed_predicates"]
    new_task["rank"] = [len(new_task["hard_failures"]), len(new_task["failed_predicates"])]
    new_objective = copy.deepcopy(strict_objective)
    new_objective["passed"] = new_task["task_success"]
    new_objective["hard_failures"] = new_task["hard_failures"]
    new_objective["task_failures"] = new_task["failed_predicates"]
    new_objective["rank"] = [*new_task["rank"], *strict_objective["rank"][2:]]
    diagnostics = {
        "gripper_joint_excess_native_units": gripper,
        "non_gripper_joint_excess_native_units": non_gripper,
        "maximum_gripper_excess_rad": max(gripper.values(), default=0.0),
        "maximum_non_gripper_excess_native_units": max(non_gripper.values(), default=0.0),
        "manual_inspection_required": any(value > 0.1 for value in gripper.values()),
    }
    return {"strict_v1": {"objective": strict_objective, "task_evaluation": strict_task},
            "task_completion_v1": {"objective": new_objective, "task_evaluation": new_task},
            "gripper_joint_diagnostics": diagnostics}


def apply(report):
    """Produce a search view; leave the caller's original report unchanged."""
    value = copy.deepcopy(report)
    judgments = dual_judgment(value)
    value["judgments"] = judgments
    value["objective"] = judgments["task_completion_v1"]["objective"]
    value["task_evaluation"] = judgments["task_completion_v1"]["task_evaluation"]
    value["task_success"] = value["objective"]["passed"]
    value["criteria"] = {**value["criteria"], **value["task_evaluation"]["hard_constraints"]}
    value["judgment_version"] = "task_completion_v1"
    return value
