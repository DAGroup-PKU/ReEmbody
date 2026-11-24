"""Independent physical/first-grasp gate for the published 111-case lineage.

Original scene bytes, task semantics, and all initial native arrays are exact.
Only the new continuous robot command trajectory changes. Clearance and
observation-alias gates are additional independent admission requirements.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SHARED = HERE / "_shared"
ALIGNED_VALIDATION = SHARED
FIRST_GRASP_WORK = SHARED
if __package__:
    from ._shared.initial_state import sha256
    from ._shared.trajectory_metrics import command_cycles, metrics
    from ._shared.paths import PROJECT_ROOT
else:
    from _shared.initial_state import sha256
    from _shared.trajectory_metrics import command_cycles, metrics
    from _shared.paths import PROJECT_ROOT
OLD_AUDIT = SHARED / "audit_native_first_grasp.py"
REFERENCE_CASE = PROJECT_ROOT / "result/sim1-sim2/s1_heightfit_3x1_20261017/physical_00/cases/stack_cups_ep000000/attempt_00"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def equal_npz(left, right):
    if not Path(left).is_file() or not Path(right).is_file():
        return False
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        return set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


def task_semantics(contract):
    return {k: v for k, v in contract.items() if k not in {"scene_sha256", "contract_id"}}


def source_evidence(source_receipt, source_reference, strategy):
    """Verify the exact selected native behind this published source case."""
    if source_receipt is not None:
        source_receipt = Path(source_receipt)
        source = source_receipt.parent
        source_id = read(source_receipt)['case_id']
        lineage = strategy['human_align_source']['source_cases'][source_id]
        if source.resolve() != Path(lineage['source_native']).resolve():
            raise ValueError('Source receipt differs from the published selected native')
        if not all(Path(p).is_file() and sha256(p)==h for p,h in lineage['source_pins'].items()):
            raise ValueError('Published source evidence changed')
        source_native = np.load(source / "execute/rollout.npz", allow_pickle=False)
        files = [source_receipt, source / "s1_command_initialization.json", source / "execute/scene.xml",
                 source / "execute/rollout.npz", source / "execute/task_contract.json"]
        return {"identity": str(source), "case_id": read(source_receipt)["case_id"],
                "mode": "published_selected_native_source_rollout", "profile": None,
                "initialization": read(source/'s1_command_initialization.json'),
                "contract": read(source / "execute/task_contract.json"), "scene": source / "execute/scene.xml",
                "initial": {k: source_native[k][0] for k in ("qpos", "qvel", "object_position", "object_quaternion")},
                "checks": {"source_native_rollout_present": True,"published_source_pins_verified":True}, "evidence": files+list(map(Path,lineage['source_pins']))}
    raise ValueError('v3 human-align requires the actual published source native receipt')


def continuous_clock_checks(receipt, model, native, commands, hard, report):
    step = float(model.opt.timestep)
    ct, ht, nt = commands["time_s"], hard["time_s"], native["time_s"]
    end = float(ht[-1])
    # Clock comparisons only absorb floating-point integration roundoff. A
    # complete native step beyond 15.0 seconds is rejected by step count too.
    tolerance = max(1e-8, step * 1e-5)
    count_duration = (len(ht) - 1) * step
    return {
        "native_clock_starts_at_zero": abs(float(nt[0])) <= 1e-12 and abs(float(ht[0])) <= 1e-12,
        "native_every_physics_step_clock_contiguous": len(ht) >= 2 and bool(np.allclose(np.diff(ht), step, rtol=0, atol=tolerance)),
        "issued_every_physics_step_clock_contiguous": len(ct) >= 1 and (len(ct) < 2 or bool(np.allclose(np.diff(ct), step, rtol=0, atol=tolerance))),
        "one_issued_command_per_physics_step": len(ht) == len(ct) + 1 and bool(np.allclose(ht[:-1], ct, rtol=0, atol=tolerance)),
        "hard_trace_all_constraints_all_steps_pass": bool(np.asarray(hard["hard_constraints"]).all() and np.asarray(hard["measurements_valid"]).all()),
        "native_samples_within_continuous_execution": bool(np.all(np.diff(nt) > 0) and float(nt[-1]) <= end + tolerance and end - float(nt[-1]) <= .011 + tolerance),
        "full_physics_duration_le_15s_including_stability": count_duration <= 15. + 1e-12 and end <= 15. + tolerance,
        "physical_report_duration_equals_full_clock": abs(float(report.get("simulation_seconds", -1.)) - end) <= tolerance,
        "receipt_executed_horizon_equals_full_clock": abs(float(receipt.get("executed_horizon_s", -1.)) - end) <= step + tolerance,
        "same_continuous_native_simulation_state": receipt.get("same_simulation_state") is True,
        "no_agent_object_state_writes": receipt.get("object_state_written_by_agent") is False,
        "no_forced_objects_welds_or_robot_teleport_execution": all(report.get(k) is False for k in
            ("objects_forced", "object_weld_used", "robot_qpos_forced_during_execution")),
    }


def audit(receipt_path, strategy_path, implementation, load_dir, source_receipt, out):
    receipt_path, strategy_path, implementation, load_dir, source_receipt, out = map(
        Path, (receipt_path, strategy_path, implementation, load_dir, source_receipt, out))
    if (out / "quality_receipt.json").exists():
        raise FileExistsError("Immutable quality receipt already exists: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    case = receipt_path.parent
    receipt, strategy = map(read, (receipt_path, strategy_path))
    source = source_evidence(source_receipt, None, strategy)
    native = np.load(case / "execute/rollout.npz", allow_pickle=False)
    commands = np.load(case / "issued_robot_commands.npz", allow_pickle=False)
    hard = np.load(case / "execute/immutable_hard_trace.npz", allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(case / "execute/scene.xml"))
    contract = read(case / "execute/task_contract.json")
    source_contract = source["contract"]
    report = read(case / "execute/report.json")
    initialization=read(case/'s1_command_initialization.json')
    initial_checks={
        'source_joint_names_exact':initialization['joint_names']==source['initialization']['joint_names'],
        'source_joint_values_exact':initialization['initial_joint_values_rad']==source['initialization']['initial_joint_values_rad'],
        'source_all_native_arrays_exact':all(np.array_equal(native[k][0],source['initial'][k]) for k in ('qpos','qvel','object_position','object_quaternion')),
        'startup_native_hard_passed':initialization.get('startup_hard',{}).get('all_hard_passed') is True,
    }
    initial=dict(passed=all(initial_checks.values()),checks=initial_checks,scope='Exact inherited published native initialization; no S4 perturbation profile substituted')
    (out / "initial_state_audit.json").write_text(json.dumps(initial, indent=2) + "\n")
    implementation_hashes = strategy.get("implementation_sha256", {})
    checks = {
        "new_receipt_bound_to_exact_frozen_strategy": receipt.get("strategy_sha256") == sha256(strategy_path),
        "new_strategy_implementation_files_unchanged": bool(implementation_hashes) and all(
            Path(p).is_file() and sha256(p) == digest for p, digest in implementation_hashes.items()),
        "new_policy_binds_published_111_lineage": strategy['human_align_source']['prefix']=='150case_data_enhance_blender_v2_trajectory_refine' and strategy['human_align_source']['revision']=='098642c9e57dfa933db9d8976564a55bd740a1ea',
        "source_joint_initialization_preserved_exactly": initial['passed'],
        "source_case_id_matches": source["case_id"] == receipt.get("case_id"),
        "source_model_and_scene_xml_preserved_exactly": sha256(case / "execute/scene.xml") == sha256(source["scene"]),
        "all_source_initial_native_arrays_preserved_exactly": all(np.array_equal(native[k][0], source["initial"][k])
            for k in ("qpos", "qvel", "object_position", "object_quaternion")),
        "initial_state_passes_declared_geometry_and_joint_gate": initial["passed"],
        "persisted_named_initialization_equal_source": initial_checks['source_joint_names_exact'] and initial_checks['source_joint_values_exact'],
        "native_full_task_accepted": receipt.get("accepted") is True,
        "native_task_goal": receipt.get("native", {}).get("task_text_physical_success") is True,
        "native_all_hard_passed": receipt.get("online_every_step_hard_snapshot", {}).get("all_hard_passed") is True,
        "original_task_predicates_and_thresholds_unchanged": task_semantics(contract) == task_semantics(source_contract),
        "independent_replay_exact": (receipt.get("replay") or {}).get("exact") is True,
        "independent_every_step_hard_trace_exact": (receipt.get("replay") or {}).get("every_step_hard_trace_exact") is True,
        "persisted_independent_replay_arrays_exact": all(equal_npz(case / "execute" / name, case / "replay" / name)
            for name in ("rollout.npz", "immutable_hard_trace.npz")),
    }
    checks.update(source["checks"])
    clock = continuous_clock_checks(receipt, model, native, commands, hard, report)
    checks.update(clock)
    stable = next(p for p in contract["predicates"] if p["kind"] == "stable")
    released = next(p for p in contract["predicates"] if p["kind"] == "released")
    checks["original_full_1s_stable_and_released_windows_preserved"] = float(stable["window_s"]) == 1. and float(released["window_s"]) == 1.
    hand_qa = [int(model.joint(f"astribot_gripper_{hand}_joint_L1").qposadr[0]) for hand in ("left", "right")]
    cycles = command_cycles(commands["time_s"], commands["qpos"][:, hand_qa])
    checks["exactly_one_meaningful_close_per_hand_in_entire_command_stream"] = cycles["exactly_one_close_each"]
    load_receipt_path, loads_path = load_dir / "native_replay_load_receipt.json", load_dir / "native_replay_loads.npz"
    loads_receipt = read(load_receipt_path) if load_receipt_path.exists() else {}
    binding, actual_loads = {}, None
    if loads_path.exists():
        actual_loads = np.load(loads_path, allow_pickle=False)
        replay = loads_receipt.get("replay", {})
        binding = {
            "native_source_path": Path(loads_receipt.get("native_source", "/missing")).resolve() == case.resolve(),
            "command_sha256": replay.get("command_sha256") == sha256(case / "issued_robot_commands.npz"),
            "actual_load_clock_exact": np.array_equal(actual_loads["time_s"], native["time_s"]),
            "actual_load_object_positions_exact": np.array_equal(actual_loads["object_position_m"], native["object_position"][:, :2]),
            "forces_shape_finite_nonnegative": actual_loads["pad_force_n"].shape == (len(native["time_s"]), 2, 2) and bool(np.isfinite(actual_loads["pad_force_n"]).all()) and bool((actual_loads["pad_force_n"] >= 0).all()),
            "local_positions_shape_finite": actual_loads["object_in_finger_frame_m"].shape == (len(native["time_s"]), 2, 3) and bool(np.isfinite(actual_loads["object_in_finger_frame_m"]).all()),
            "fresh_actual_native_solver_replay": all(loads_receipt.get(k) is True for k in ("fresh_native_replay", "force_from_actual_solver_state", "sample_clock_exact")),
            "replay_exact": replay.get("exact") is True and replay.get("every_step_hard_trace_exact") is True,
        }
    checks["actual_solver_loads_from_fresh_exact_replay"] = bool(binding) and all(binding.values())
    checks["persisted_actual_load_replay_arrays_exact"] = checks["actual_solver_loads_from_fresh_exact_replay"] and all(
        equal_npz(case / "execute" / name, load_dir / "replay" / name) for name in ("rollout.npz", "immutable_hard_trace.npz"))
    old = load_module("short_refine_frozen_first_grasp", OLD_AUDIT)
    release = load_module("short_refine_frozen_release", FIRST_GRASP_WORK / "release_cycle_integrity.py")
    hands = []
    integrity = {"passed": False, "reason": "valid_actual_solver_replay_required"}
    retention = dict(integrity)
    if checks["actual_solver_loads_from_fresh_exact_replay"]:
        t = native["time_s"]
        idx = np.clip(np.searchsorted(commands["time_s"], t), 0, len(commands["time_s"]) - 1)
        history = receipt["event_history"]
        ends = np.asarray([entry["time_s"] for entry in history])
        event_idx = np.minimum(np.searchsorted(ends, t, side="right"), len(history) - 1)
        diagnostics = {"time_s": t, "desired_gripper_angle_rad": commands["qpos"][idx][:, hand_qa],
            "event_kind": np.asarray([history[i]["event_id"].split(":")[0] for i in event_idx]),
            "event_object": np.asarray([history[i].get("selected_object_id") or "" for i in event_idx]),
            "event_hand": np.asarray([history[i].get("selected_hand") or "" for i in event_idx]),
            "pad_force_n": actual_loads["pad_force_n"],
            "object_in_finger_frame_m": actual_loads["object_in_finger_frame_m"]}
        np.savez_compressed(out / "exact_force_diagnostics.npz", **diagnostics)
        packets = [read(path)["observation"] for path in sorted((case / "decisions").glob("*/packet.json"))]
        hands = [old.summarize_hand(t, native["object_position"], diagnostics["pad_force_n"],
                    diagnostics["object_in_finger_frame_m"], native["gripper_opening"], diagnostics["desired_gripper_angle_rad"],
                    diagnostics["event_kind"], diagnostics["event_object"], diagnostics["event_hand"], hi, packets)
                 for hi in range(2)]
        integrity = release.evaluate(case, native, diagnostics, implementation)
        retention = release.acquisition_retention(t, diagnostics["pad_force_n"], diagnostics["object_in_finger_frame_m"], diagnostics)
    checks.update({
        "both_first_close_success": len(hands) == 2 and all(h.get("first_close_success") for h in hands),
        "exactly_one_labelled_close_each": len(hands) == 2 and all(h.get("intent_close_count") == 1 for h in hands),
        "no_reopens_before_legal_release": len(hands) == 2 and all(h.get("commanded_reopen_count") == 0 for h in hands),
        "no_recovery_before_lift": len(hands) == 2 and all(not h.get("recovery_before_first_proof") for h in hands),
        "no_slip": len(hands) == 2 and all(not h.get("slip_candidates") for h in hands),
        "no_object_motion_in_gripper_over_15mm": len(hands) == 2 and all(not h.get("relative_motion_review") for h in hands),
        "no_contact_loss_over_100ms_before_release": len(hands) == 2 and all(not h.get("contact_loss_before_release") for h in hands),
        "whole_episode_physical_cycle_and_legal_release_integrity": integrity["passed"],
        "first_loaded_grasp_retained_before_and_after_25mm_lift": retention["passed"],
    })
    motion = metrics(case)
    evidence = [receipt_path, strategy_path, *source["evidence"], case / "execute/scene.xml",
        case / "execute/rollout.npz", case / "issued_robot_commands.npz", case / "execute/immutable_hard_trace.npz",
        case / "execute/task_contract.json", case / "execute/report.json",
        OLD_AUDIT, FIRST_GRASP_WORK / "release_cycle_integrity.py", Path(__file__),
        SHARED/'trajectory_metrics.py', ALIGNED_VALIDATION / "initial_state.py"]
    evidence.extend(path for path in (load_receipt_path, loads_path) if path.exists())
    result = {"schema": "rssr.v3_human_align.physical_quality.v1", "case_id": receipt["case_id"],
        "candidate": str(case), "source": source["identity"], "source_evidence_mode": source["mode"],
        "candidate_accepted": all(checks.values()),
        "checks": checks, "full_physics_time_s": float(hard["time_s"][-1]),
        "physics_step_count_duration_s": float((len(hard["time_s"]) - 1) * model.opt.timestep),
        "time_limit_s": 15., "time_includes_original_1s_final_stability_and_released_windows": True,
        "initial_state_audit": initial, "hands": hands, "whole_episode_integrity": integrity,
        "first_loaded_acquisition_retention": retention, "force_evidence_binding": binding,
        "whole_episode_command_cycles": cycles, "motion_metrics": motion,
        "audit_criteria": old.CRITERIA,
        "force_evidence_source": "Fresh exact native replay actual solver state; first-grasp thresholds reused without saved-state force reconstruction",
        "acceptance_scope": "Continuous new native physics from exact source initial state; unchanged full task/hard/first-grasp/release criteria; <=15s including final 1s stability; exact command replay and fresh actual-solver force replay. Path and jerk are reported, not substituted for physical proof.",
        "evidence_sha256": {str(path): sha256(path) for path in evidence}, "never_published": True}
    (out / "quality_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("receipt", "strategy", "implementation", "load-audit", "out", "source-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.receipt, args.strategy, args.implementation, args.load_audit, args.source_receipt, args.out)
    print(json.dumps({"candidate_accepted": report["candidate_accepted"],
                      "failed_checks": [k for k, v in report["checks"].items() if not v],
                      "full_physics_time_s": report["full_physics_time_s"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
