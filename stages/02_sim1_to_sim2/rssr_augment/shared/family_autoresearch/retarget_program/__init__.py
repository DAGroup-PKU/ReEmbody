"""Shared bounded programs instantiated from geometry, never case lookup tables."""
from __future__ import annotations

import copy
from pathlib import Path

from .. import actions
from ..store import file_sha
from .anchor import instantiate
from .aperture import instantiate as instantiate_aperture
from .grasp_centering import instantiate as instantiate_grasp_centering
from .table_clearance_stage import instantiate as instantiate_table_clearance_stage

ANCHOR_V2_STATUS = "diagnostic_only_after_two_full_native_failures"
APERTURE_V1_STATUS = "diagnostic_only_after_two_full_native_failures"
TABLE_CLEARANCE_STAGE_V3_STATUS = "diagnostic_only_after_two_independent_review_rejections"
GRASP_CENTERING_V5_STATUS = "active_candidate"


def attach_grasp_candidate(packet: dict, index_case: dict) -> dict:
    """Attach one falsifiable grasp probe to an acquisition TaskPacket.

    Stage routing supplies the failed task predicate. The shared transform only
    sees the bound object/hand and source/target geometry; it never reads a case
    ID, past pass label, or per-case coordinate table to compute its offset.
    """
    if packet.get("stage") != "acquisition" or packet.get("target_spec", {}).get("kind") != "lift_with_contact":
        return packet
    if packet.get("family") == "stack_blocks" and TABLE_CLEARANCE_STAGE_V3_STATUS == "active_candidate":
        try:
            program = instantiate_table_clearance_stage(packet, index_case)
            candidate_packet = copy.deepcopy(packet)
            candidate_packet["allowed_scope"]["stage_plan"] = {
                "operation": "stage_motion_window", "stage_world_axis_bound_m": .1,
                "joint_rate_bound_rad_s": 4., "frozen_source_clock": True,
                "independent_preflight_and_runtime_guard": True}
            candidate_packet["evidence_refs"] = list(dict.fromkeys([
                *candidate_packet["evidence_refs"], *program["evidence_sha256"]]))
            candidate_packet["source_sha256"] = {**candidate_packet.get("source_sha256", {}),
                                                  **program["evidence_sha256"]}
            action = actions.build_action(candidate_packet, program["command"])
            hypothesis = {
                "diagnosis": (
                    f"The frozen target command's {program['hand']} gripper intersects the worktable "
                    f"by {abs(program['source_command_penetration_m']):.4f} m during the bound "
                    "grasp stage. Finite contact can keep the actual tool above or behind that "
                    "unreachable command, explaining command-to-actual divergence. Map the "
                    "complete grasp stage to a higher, pitched, object-centered tool pose under the separate "
                    "10 cm world-axis and 4 rad/s joint-rate safety guard."),
                "prediction": (
                    packet["target_predicate"] + ": the independent stage preflight should show less than "
                    "3 mm static gripper/table penetration and at least 8 mm improvement. "
                    "At closure, object offset along the closing axis should fall below 10 mm "
                    "with bilateral static pad contact before closure. Vertical motion tests table clearance; "
                    "pitch tests gripper orientation; planar motion tests closing-axis alignment. "
                    "The full rollout should then reduce grasp-time tool-tracking error, "
                    "establish sustained contact and increase contact-qualified object lift; "
                    "all task predicates and hard failures must be audited separately."),
                "falsifier": (
                    "Preflight collision improvement below 8 mm or remaining penetration "
                    "above 3 mm or a unilateral closing geometry refutes command feasibility; unchanged actual tracking, contact "
                    "or complete task after a verified command refutes physical usefulness."),
                "evidence_refs": list(program["evidence_sha256"]),
            }
            action["decision"].update(diagnosis=hypothesis["diagnosis"],
                                      prediction=hypothesis["prediction"])
            candidate_packet["shared_program_instance"] = program
            candidate_packet["shared_program_proposal"] = {
                "status": "candidate", "program_version": program["program_version"],
                "proposal_revision": "table_clearance_stage_hypothesis_v3",
                "mechanism": action["mechanism"], "hypothesis": hypothesis,
                "action": action, "handoff": None, "unresolved": None}
            return candidate_packet
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            packet = copy.deepcopy(packet)
            packet["table_clearance_diagnostic"] = {"status": "not_applicable",
                "reason": type(exc).__name__, "detail": str(exc)[:250]}
    if packet.get("family") == "stack_blocks" and GRASP_CENTERING_V5_STATUS == "active_candidate":
        try:
            program = instantiate_grasp_centering(packet, index_case)
            candidate_packet = copy.deepcopy(packet)
            candidate_packet["evidence_refs"] = list(dict.fromkeys([
                *candidate_packet["evidence_refs"], *program["evidence_sha256"]]))
            candidate_packet["source_sha256"] = {**candidate_packet.get("source_sha256", {}),
                                                  **program["evidence_sha256"]}
            action = actions.build_action(candidate_packet, program["command"])
            projection = program["parent_grasp_alignment"]["object_closing_axis_projection_m"]
            hypothesis = {
                "diagnosis": (
                    f"At the bound {program['hand']} grasp close, the frozen target command places "
                    f"{program['anchor_object']} {projection:.4f} m from the tool along the closing axis. "
                    "The parent grasp had sparse cube contact. A single dominant-axis translation "
                    "tests whether closing-axis alignment causes the missed lift. The new offset "
                    "is composed through the existing IK knots; height, pitch, "
                    "gripper command and scene remain fixed."),
                "prediction": (
                    packet["target_predicate"] + ": the compiled target command should reduce the "
                    "closing-axis offset and preserve the unmodified control coordinates. In the "
                    "full native rollout, the cube should enter between both pads before closure, "
                    "sustain bilateral contact longer, and lift with hand contact more than the parent. "
                    "Commanded gripper/table overlap is measured separately because planar "
                    "centering may worsen it. Hard constraints and downstream task predicates "
                    "are audited separately."),
                "falsifier": (
                    "If the compiled offset does not shrink, or the achieved rollout retains "
                    "one-sided contact and no contact-qualified lift, planar centering is insufficient."),
                "evidence_refs": list(program["evidence_sha256"]),
            }
            action["decision"].update(diagnosis=hypothesis["diagnosis"],
                                      prediction=hypothesis["prediction"])
            candidate_packet["shared_program_instance"] = program
            candidate_packet["shared_program_proposal"] = {
                "status": "candidate", "program_version": program["program_version"],
                "proposal_revision": "single_factor_dominant_axis_grasp_centering_v5",
                "mechanism": action["mechanism"], "hypothesis": hypothesis,
                "action": action, "handoff": None, "unresolved": None}
            return candidate_packet
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            packet = copy.deepcopy(packet)
            packet["grasp_centering_diagnostic"] = {"status": "not_applicable",
                "reason": type(exc).__name__, "detail": str(exc)[:250]}
    if packet.get("family") == "stack_blocks" and APERTURE_V1_STATUS == "active_candidate":
        try:
            program = instantiate_aperture(packet, index_case)
            candidate_packet = copy.deepcopy(packet)
            candidate_packet["evidence_refs"] = list(dict.fromkeys([
                *candidate_packet["evidence_refs"], *program["evidence_sha256"]]))
            candidate_packet["source_sha256"] = {**candidate_packet.get("source_sha256", {}),
                                                  **program["evidence_sha256"]}
            action = actions.build_action(candidate_packet, program["command"])
            hypothesis = {
                "diagnosis": (
                    f"At {program['sample_time_s']:.3f} s before the source grasp, "
                    f"source measured opening is {program['source_actual_gap_m']:.4f} m, "
                    f"target commanded opening is {program['target_command_gap_m']:.4f} m, "
                    f"and bound box width is {program['object_width_m']:.4f} m. "
                    "The smaller target approach opening can cause unilateral contact and displacement "
                    "before closure. A time-bounded pulse tests this geometry mismatch."),
                "prediction": (
                    packet["target_predicate"] + ": during approach, target commanded gripper opening increases; measured "
                    "preclosure object displacement and one-sided contact should decrease. "
                    "At/after closure, bilateral contact and contact-qualified lift should "
                    "increase without a new hard failure. Full task success is separate."),
                "falsifier": (
                    "If compiled opening does not increase, or the verified full rollout "
                    "does not improve contact-qualified lift and downstream task predicates, "
                    "retire this pulse rather than extrapolating from command geometry."),
                "evidence_refs": list(program["evidence_sha256"]),
            }
            action["decision"].update(diagnosis=hypothesis["diagnosis"],
                                      prediction=hypothesis["prediction"])
            candidate_packet["shared_program_instance"] = program
            candidate_packet["shared_program_proposal"] = {
                "status": "candidate", "program_version": program["program_version"],
                "proposal_revision": "aperture_hypothesis_v1",
                "mechanism": action["mechanism"], "hypothesis": hypothesis,
                "action": action, "handoff": None, "unresolved": None}
            return candidate_packet
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            packet = copy.deepcopy(packet)
            packet["aperture_program_diagnostic"] = {"status": "not_applicable",
                "reason": type(exc).__name__, "detail": str(exc)[:250]}
    elif packet.get("family") == "stack_blocks":
        packet = copy.deepcopy(packet)
        packet["aperture_program_diagnostic"] = {"status": APERTURE_V1_STATUS,
            "program_version": "pregrasp_aperture_reexpression_v1",
            "reason": "Two distinct full v80 block rollouts changed aperture but gained no task predicate"}
    if ANCHOR_V2_STATUS != "active_candidate":
        packet = copy.deepcopy(packet)
        packet["shared_program_diagnostic"] = {"status": ANCHOR_V2_STATUS,
            "program_version": "grasp_anchor_translation_v2",
            "reason": "Full v80 trials for stack_blocks_ep000010 and stack_blocks_ep000015 had no task-predicate gain"}
        return packet
    source_dir = Path(index_case["source_dir"])
    report_path = Path(packet["parent"]["report_path"])
    selection_path = source_dir / "sim2/selection.json"
    config_path = source_dir / "config.json"
    contract_path = source_dir / "sim2/task_contract.json"
    try:
        row = {"case_id": packet["case_id"], "source_dir": str(source_dir),
               "seed_report_path": str(report_path), "selection_path": str(selection_path),
               "target_predicate": packet["target_predicate"],
               "file_sha256": {str(path): file_sha(path) for path in
                               (report_path, selection_path, config_path)},
               "task_contract_sha256": file_sha(contract_path)}
        program = instantiate(row)
        candidate_packet = copy.deepcopy(packet)
        candidate_packet["evidence_refs"] = list(dict.fromkeys([
            *candidate_packet["evidence_refs"], *program["evidence_sha256"]]))
        candidate_packet["source_sha256"] = {**candidate_packet.get("source_sha256", {}),
                                              **program["evidence_sha256"]}
        action = actions.build_action(candidate_packet, program["command"])
        signed_explanation = (
            "At this grasp event, mapped source-relative tool target "
            f"{program['mapped_tool_world_m']} m minus current target command "
            f"{program['target_command_tool_world_m']} m equals signed residual "
            f"{program['unbounded_delta_m']} m. Applying bounded delta "
            f"{program['bounded_delta_m']} m moves the current command toward "
            "the mapped target; clipping may leave a residual.")
        prediction = (packet["target_predicate"] + ": the anchored grasp offset should change "
                      "measured hand-object contact and lift during the bound source hold cycle; "
                      "the other hand's spatial command should remain exact")
        hypothesis = {
            "diagnosis": signed_explanation,
            "prediction": prediction,
            "falsifier": "A command not moving toward the mapped target invalidates the sign convention; no measured contact or object-lift change after verified tool response refutes this grasp hypothesis",
            "evidence_refs": [str(report_path), *[path for path in program["evidence_sha256"]
                                                if path != str(report_path)]]}
        action["decision"].update(diagnosis=hypothesis["diagnosis"],
                                  prediction=hypothesis["prediction"])
        candidate_packet["shared_program_instance"] = program
        candidate_packet["shared_program_proposal"] = {
            "status": "candidate", "program_version": program["program_version"],
            "proposal_revision": "anchor_hypothesis_v2",
            "mechanism": action["mechanism"], "hypothesis": hypothesis,
            "action": action, "handoff": None, "unresolved": None}
        return candidate_packet
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        packet = copy.deepcopy(packet)
        packet["shared_program_diagnostic"] = {"status": "not_applicable",
            "reason": type(exc).__name__, "detail": str(exc)[:250]}
        return packet
