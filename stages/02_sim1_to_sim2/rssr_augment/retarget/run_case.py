"""Run one Luna event policy in continuous Sim2 and independently replay commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from rssr_augment.shared.family_autoresearch.store import Store, atomic_json, digest, file_sha, pinned_json
from .task_text_success_versioned import score_report as legacy_score

def score_report(path, case_id):
    result = legacy_score(path, case_id)
    report=json.loads(Path(path).read_text())
    evaluation = report["judgments"]["strict_v1"]["task_evaluation"]
    result["hard_failures"]=[k for k,v in evaluation["hard_constraints"].items() if not v]
    result["execution_valid"]=bool(result["native_audit_valid"] and not result["forbidden_state_write"] and not result["hard_failures"])
    result["strict_v1_passed"]=bool(evaluation["task_success"])
    result["task_text_physical_success"] = bool(evaluation["task_success"] and all(evaluation["hard_constraints"].values()) and result["execution_valid"])
    result["required_failures"] = [p["id"] for p in evaluation["predicates"] if not p["passed"]]
    result["score_contract"] = "all_frozen_native_strict_v1_predicates_and_every_step_hard_constraints"
    return result

from .agent import VERSION, decide_next
from .rule_revision_agent import RuleRevisionAgent
from .rule_runtime import RuleRuntime
from .adaptive_rule_protocol import expression
from .scene_profile import SceneProfileBank, scene_signature
from .transfer_phase import CertifiedRulePhaseIndex, executable_certified_rule_keys
from .immutable_hard_observation import HardSnapshotter, HardAccumulator
from .hard_trace import save_hard_trace
from rssr_augment.shared.family_autoresearch.agents import DecisionUnavailable, InvalidCompleteResponse
from rssr_augment.shared.family_autoresearch.store import BudgetExhausted, CaseBudgetExhausted
from .case import load_case, pilot_physics, rssr_servo
from .events import EventDefinition, catalog, event_trigger, signed_margin
from .session import ADAPTER_MANIPULATORS, ContinuousSession, EventAction
from .live_predicates import LivePredicates
from .mobile_event_adapter import compile_reposition_events
from .relational_event_policy import decision_context
from .immutable_rule_guard import guard as immutable_event_guard
from .progress_guard import (update_peak_and_check, backward_loaded_recontact,
                             ordered_relation_prerequisite_warning)
from .event_preview import horizon_for
from .progress_epoch import ProgressEpoch


BASE_DECISIONS = 80
WINDOW_S = 0.5
COUPLING_WARNING_RAD = .0012  # Below the immutable .002 rad hard boundary.
JOINT_MARGIN_WARNING_RAD = .05  # Above the immutable 2 mrad joint overrun.
GRIPPER_MARGIN_WARNING_RAD = .005  # Measured finger margin; native limit is unchanged.
TRANSITION_HOLD_S = .002


def episode_horizon_s(source_duration_s: float, contract: dict,
                      scene: dict) -> float:
    """Budget complete physical transitions from source time and scene scale."""
    if not np.isfinite(source_duration_s) or source_duration_s <= 0:
        raise ValueError('Invalid source duration')
    # Existing successful relation trials first satisfied every predicate at
    # about 49 s; retain their 60 s stability window while extending ordered
    # operations by the actual travel and measured scene distances below.
    horizon = max(60., source_duration_s)
    objects = {row['id']: row for row in scene.get('objects', [])}
    for predicate in contract['predicates']:
        if predicate['kind'] != 'open_insert_close':
            continue
        container = objects[predicate['container']]
        payload = objects[predicate['object']]
        joint = container.get('joint') or {}
        limits = joint.get('range')
        if limits is None:
            raise ValueError('Ordered articulation lacks travel range')
        travel = abs(float(limits[1]) - float(limits[0]))
        fixture = objects.get(container.get('parent_id'), container)
        source = np.asarray(payload['position_m'], dtype=float)
        target = np.asarray(container['position_m'], dtype=float)
        if not (np.isfinite(source).all() and np.isfinite(target).all()):
            raise ValueError('Invalid ordered-task geometry')
        lateral = float(np.linalg.norm((target - source)[:2]))
        side_and_mouth = (.5 * float(fixture['size_m'][0]) +
                          .5 * float(container['size_m'][1]))
        rim = (float(target[2]) + .5 * float(container['size_m'][2]) +
               .5 * float(payload['size_m'][2]) + .03)
        vertical = max(0., rim - float(source[2]))
        # Two slide traversals, an indirect carried-object path, and bounded
        # contact/release/stability segments. This is a diagnostic budget,
        # never a target-motion command or a relaxation of native success.
        estimate = (source_duration_s + 2. * travel / .025 +
                    (lateral + side_and_mouth + vertical) / .04 + 40.)
        horizon = max(horizon, estimate)
    return float(np.ceil(horizon * 2.) / 2.)


def new_uncontacted_gripper_closure(history: list[dict], obs: dict,
                                   last_replan: str | None) -> bool:
    """True only at a fresh contact/grasp rollback with measured closure."""
    if len(history) < 2 or last_replan != 'rule_rollback':
        return False
    latest, previous = history[-1], history[-2]
    hand = latest.get('selected_hand')
    target_id = latest.get('selected_object_id')
    if (latest['event_id'].split(':', 1)[0] not in ('contact', 'grasp')
            or not hand or not target_id
            or hand not in latest.get('gripper_opening_m', {})
            or hand not in previous.get('gripper_opening_m', {})
            or any(obs['finger_contact'][target_id][hand])):
        return False
    return (float(previous['gripper_opening_m'][hand]) -
            float(latest['gripper_opening_m'][hand]) > 1e-4)


def repeated_event_cycle(history: list[dict], now_s: float) -> dict | None:
    """Detect three recent A/B alternations from observed event history."""
    transitions = []
    for row in history[-48:]:
        event_id = row['event_id'].split(':rule:', 1)[0]
        if not transitions or transitions[-1][0] != event_id:
            transitions.append((event_id, float(row['time_s']), row))
    if len(transitions) < 6:
        return None
    recent = transitions[-6:]
    names = [row[0] for row in recent]
    if (names[0] == names[2] == names[4] and
            names[1] == names[3] == names[5] and
            names[0] != names[1] and
            now_s - recent[0][1] <= 8.):
        before = recent[0][2]['live_predicates']
        after = recent[-1][2]['live_predicates']
        delta = {key: round(float(after[key]['instantaneous_margin']) -
                            float(before[key]['instantaneous_margin']), 6)
                 for key in before if key in after}
        return {'event_ids': names[:2], 'repetitions': 3,
                'span_s': round(now_s - recent[0][1], 4),
                'native_margin_delta': delta}
    return None


def _replay(case: dict, output: Path,
            records: list[tuple[float, np.ndarray, np.ndarray, str]],
            *, stop_s: float | None = None) -> dict:
    times = np.array([row[0] for row in records], dtype=np.float64)
    commands = np.stack([row[1] for row in records])
    velocities = np.stack([row[2] for row in records])
    labels = np.array([row[3] for row in records], dtype="U40")
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Continuous executor emitted incomplete commands")
    command_path = output / "issued_robot_commands.npz"
    np.savez_compressed(command_path, time_s=times, qpos=commands,
                        qvel=velocities, event=labels)
    original = rssr_servo.JointServo
    replay_index = 0
    replay_hard_records = []
    replay_reader = replay_accumulator = replay_data = None

    class ReplayServo(original):
        def __init__(self, model, acts, kind):
            super().__init__(model, acts, kind)
            self.model = model

        def apply(self, data, desired, velocity):
            nonlocal replay_index, replay_reader, replay_accumulator, replay_data
            if replay_reader is None:
                replay_reader = HardSnapshotter(self.model)
                replay_accumulator = HardAccumulator()
            replay_data = data
            instant = replay_reader(data)
            replay_hard_records.append(instant)
            replay_accumulator.observe(instant)
            if replay_index >= len(times) or \
                    abs(float(data.time) - float(times[replay_index])) > 1e-7 or \
                    desired.shape != commands[replay_index].shape:
                raise ValueError("Independent robot-command replay diverged")
            desired[:], velocity[:] = commands[replay_index], velocities[replay_index]
            replay_index += 1
            return super().apply(data, desired, velocity)

    rssr_servo.JointServo = ReplayServo
    try:
        pilot_physics.rollout(case["config"], "sim2", output / "replay",
                              case["scene"], target_qpos=case["target_qpos"],
                              stop_s=stop_s)
    finally:
        rssr_servo.JointServo = original
    if replay_reader is None:
        raise ValueError('Independent replay lacks hard observations')
    final_hard = replay_reader(replay_data)
    replay_hard_records.append(final_hard)
    replay_hard_summary = replay_accumulator.observe(final_hard)
    save_hard_trace(output / 'replay/immutable_hard_trace.npz', replay_hard_records)
    with np.load(output / 'execute/immutable_hard_trace.npz', allow_pickle=False) as a, np.load(output / 'replay/immutable_hard_trace.npz', allow_pickle=False) as b:
        hard_same = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    if replay_index != len(times) or file_sha(output / "replay/scene.xml") != \
            case["mobile_scene_sha256"] or file_sha(output / "replay/scene.xml") != \
            file_sha(output / "execute/scene.xml"):
        raise ValueError("Independent replay omitted commands or changed scene")
    if json.loads((output / 'replay/task_contract.json').read_text()) != case['contract']:
        raise ValueError('Independent replay changed corrected task contract')
    with np.load(output / "execute/rollout.npz", allow_pickle=False) as a, \
            np.load(output / "replay/rollout.npz", allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Independent replay changed native trace schema')
        same_by_array={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
        same=all(same_by_array.values())
    return {"exact": bool(same), "every_step_hard_trace_exact": bool(hard_same),
            "every_step_hard_summary": replay_hard_summary, "exact_by_array": same_by_array, "score": score_report(output / "replay/report.json",
                                                       case["case_id"]),
            "command_path": str(command_path),
            "command_sha256": file_sha(command_path),
            "every_step_hard_trace_sha256": file_sha(output / 'replay/immutable_hard_trace.npz')}


def legacy_seq343_validate_startup(strategy_path: Path, case_id: str) -> dict:
    """Validate the actual launch binding without reserving API or physics work."""
    strategy = json.loads(strategy_path.read_text())
    if strategy.get('allow_s1_virtual_base_experiment') is not False:
        raise ValueError('Fixed-model contact package forbids virtual-base actions')
    if (strategy.get("executor_module") != __package__ or
            strategy.get("agent_version") != VERSION or
            strategy["campaign"]["model_profile"] != "gpt6luna"):
        raise ValueError("Frozen strategy does not match this event executor/version")
    run = Path(strategy["new_run"])
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("strategy_seq") != strategy["seq"] or
            manifest.get("executor_module") != __package__ or
            manifest.get("agent_version") != VERSION or
            manifest["cases"] != strategy["cases"] or
            manifest["pilot_cases"] != strategy["campaign"]["pilot_cases"] or
            case_id not in strategy["campaign"]["pilot_cases"]):
        raise ValueError("Frozen manifest/strategy/case launch binding differs")
    if manifest["strategy_sha256"] != file_sha(strategy_path):
        raise ValueError("Frozen strategy changed")
    for path, expected in strategy["implementation_sha256"].items():
        if file_sha(path) != expected:
            raise ValueError("Frozen event implementation changed: " + path)
    knowledge_path = Path(strategy["evidence"]["empty_knowledge_file"])
    if file_sha(knowledge_path) != strategy["evidence"]["empty_knowledge_sha256"]:
        raise ValueError("Frozen empty initial knowledge changed")
    knowledge = json.loads(knowledge_path.read_text())
    if knowledge["operators"] or knowledge["experiences"]:
        raise ValueError("This campaign must start with empty knowledge")
    seed = strategy['training_profile_seed']
    seed_path = Path(seed['manifest_file'])
    if (file_sha(seed_path) != seed['manifest_sha256'] or
            manifest.get('training_profile_seed_sha256') != seed['manifest_sha256']):
        raise ValueError('Frozen learned profile manifest changed')
    seed_manifest = json.loads(seed_path.read_text())
    if (seed_manifest['source_strategy_seq'] != 189 or
            seed_manifest['source_strategy_sha256'] != seed['source_strategy_sha256'] or
            seed_manifest['source_primary_audit_sha256'] != seed['source_primary_audit_sha256']):
        raise ValueError('Learned profile source changed')
    if (len(strategy['cases']) != 150 or
            len(set(strategy['cases'])) != 150 or
            set(strategy['campaign']['pilot_cases']) != set(strategy['cases'])):
        raise ValueError('Full150 launch cohort differs')
    if (file_sha(Path(seed['source_strategy_file'])) != seed['source_strategy_sha256'] or
            file_sha(Path(seed['source_primary_audit_file'])) != seed['source_primary_audit_sha256']):
        raise ValueError('Learned profile source evidence changed')
    if (file_sha(seed['longitudinal_selection_audit_file']) !=
            seed['longitudinal_selection_audit_sha256'] or
            seed_manifest.get('longitudinal_selection_audit_sha256') !=
                seed['longitudinal_selection_audit_sha256']):
        raise ValueError('Longitudinal failed-case selection audit changed')
    selection_audit = json.loads(Path(seed['longitudinal_selection_audit_file']).read_text())
    source_selected = seed_manifest['selected_failed_case_ids']
    source_failed = {row['case_id'] for row in selection_audit['rows']
                     if not row['cumulative_success']}
    if (not selection_audit.get('complete') or selection_audit.get('errors') or
            not source_selected or len(source_selected) != len(set(source_selected)) or
            not set(source_selected) <= source_failed):
        raise ValueError('Pinned source failed-case cohort differs')
    additional_sources = seed_manifest.get('additional_source_evidence', [])
    if additional_sources != seed.get('additional_source_evidence', []):
        raise ValueError('Additional trained source evidence differs')
    allowed_source_seqs = {int(seed_manifest['source_strategy_seq'])}
    for extra in additional_sources:
        for key in ('strategy', 'primary_audit', 'memory_audit', 'profile_reuse',
                    'provider_audit', 'trajectory_audit', 'cost_audit'):
            path = Path(extra[key + '_file'])
            if file_sha(path) != extra[key + '_sha256']:
                raise ValueError('Additional trained source file changed: ' + key)
        primary = json.loads(Path(extra['primary_audit_file']).read_text())
        if (primary.get('strategy_seq') != extra['strategy_seq'] or
                not primary.get('complete_coverage') or
                not primary.get('trained_subset_verified') or
                primary.get('errors') or primary.get('unsettled')):
            raise ValueError('Additional trained source primary audit failed')
        allowed_source_seqs.add(int(extra['strategy_seq']))
    if seed_manifest['source_training_samples'] != len(seed_manifest['samples']):
        raise ValueError('Trained source sample count differs')
    for item in seed_manifest['samples']:
        if (int(item.get('source_strategy_seq', seed_manifest['source_strategy_seq']))
                not in allowed_source_seqs or
                file_sha(run / item['relative_path']) != item['sha256'] or
                file_sha(item['source_path']) != item['source_sha256']):
            raise ValueError('Frozen learned profile sample changed: ' + item['relative_path'])
    pilot_seed = strategy['prior_pilot_profile_seed']
    pilot_path = Path(pilot_seed['manifest_file'])
    if (pilot_path != run / 'prior_pilot_profile_manifest.json' or
            file_sha(pilot_path) != pilot_seed['manifest_sha256'] or
            manifest.get('prior_pilot_profile_manifest_sha256') !=
                pilot_seed['manifest_sha256']):
        raise ValueError('Frozen prior pilot profile manifest changed')
    pilot_manifest = json.loads(pilot_path.read_text())
    source_rows = pilot_manifest['source_strategies']
    source_strategies = {int(item['seq']): item for item in source_rows}
    if (pilot_manifest['schema'] != 'rssr.prior_pilot_profile_manifest.v3' or
            not source_strategies or len(source_strategies) != len(source_rows) or
            any(seq >= strategy['seq'] for seq in source_strategies) or
            len(pilot_manifest['samples']) != pilot_manifest['source_training_samples'] or
            pilot_manifest['source_training_samples'] < 1 or
            any(file_sha(item['strategy_file']) != item['strategy_sha256'] or
                file_sha(item['state_file']) != item['state_sha256']
                for item in source_strategies.values())):
        raise ValueError('Prior pilot profile source inventory differs')
    for source_seq, source_info in source_strategies.items():
        source_strategy = json.loads(Path(source_info['strategy_file']).read_text())
        if source_strategy['seq'] != source_seq:
            raise ValueError('Prior pilot source strategy sequence differs')
        source_state = json.loads(Path(source_info['state_file']).read_text())
        if any(row['status'] == 'started'
               for row in source_state['reservations'].values()):
            raise ValueError('Prior pilot source still has unsettled cost')
    seen_pilot = set()
    for item in pilot_manifest['samples']:
        relative = Path(item['relative_path'])
        if (relative.is_absolute() or '..' in relative.parts or
                relative.parts[0] != 'prior_pilot_scene_profiles' or
                item['source_strategy_seq'] not in source_strategies or
                (item['source_strategy_seq'], item['source_case_id']) in seen_pilot or
                file_sha(run / relative) != item['sha256'] or
                file_sha(item['source_path']) != item['source_sha256'] or
                item['sha256'] != item['source_sha256'] or
                file_sha(item['receipt_path']) != item['receipt_sha256'] or
                file_sha(item['independent_audit_path']) != item['independent_audit_sha256']):
            raise ValueError('Prior pilot sample or audit changed: ' + str(relative))
        seen_pilot.add((item['source_strategy_seq'], item['source_case_id']))
        sample = json.loads((run / relative).read_text())
        receipt = json.loads(Path(item['receipt_path']).read_text())
        audit = json.loads(Path(item['independent_audit_path']).read_text())
        # Recent independent audits share the same outcome and every-step
        # evidence fields even when their experiment sequence changes.  Route
        # by those verified fields so a newly audited prior pilot is not
        # misread as a legacy posture-only audit.
        if (str(audit.get('schema', '')).startswith('rssr.seq') and
                str(audit.get('schema', '')).endswith('independent_audit.v1') and
                all(key in audit for key in
                    ('attempt', 'hard_step_count', 'accepted_recomputed',
                     'native_task_goal'))):
            audited_outcome = (
                audit.get('attempt') == f"attempt_{item['attempt_index']:02d}" and
                not audit.get('errors') and
                audit.get('max_chassis_world_translation_m') == 0. and
                audit.get('max_chassis_world_z_displacement_m') == 0. and
                audit.get('max_chassis_world_quaternion_distance') == 0. and
                audit.get('max_issued_torso_native_range_violation_rad') == 0. and
                audit.get('max_issued_torso_coupling_residual_rad', 1.) <= .001 and
                audit.get('hard_step_count', 0) > 0 and
                audit.get('accepted_recomputed') == sample['strict_replay_accepted'] and
                audit.get('native_task_goal') == sample['native_task_success'] and
                (not sample['strict_replay_accepted'] or
                 audit.get('independent_replay_exact') is True))
        else:
            audited_outcome = (
                audit['attempt_index'] == item['attempt_index'] and
                audit['receipt_sha256'] == item['receipt_sha256'] and
                not audit['unsettled'] and
                audit['reference_command_samples'] == 0 and
                (audit.get('torso_command_fixed') is True or
                 (audit.get('base_translation_and_heading_fixed') is True and
                  audit.get('torso_command_bounded_0p50rad') is True and
                  audit.get('torso_physical_bounded_0p515rad') is True and
                  audit.get('torso_command_coupling_residual_rad', 1.) <= .001 and
                  audit.get('torso_command_max_speed_rad_s', 1.) <= .251)) and
                not audit.get('errors') and
                audit['accepted_with_user_posture_constraint'] == sample['strict_replay_accepted'] and
                audit['native_goal'] == sample['native_task_success'])
        if (sample['case_id'] != item['source_case_id'] or
                sample['scene_signature'] != relative.parent.name or
                sample['receipt_sha256'] != item['receipt_sha256'] or
                receipt['strategy_seq'] != item['source_strategy_seq'] or
                receipt['strategy_sha256'] != source_strategies[
                    item['source_strategy_seq']]['strategy_sha256'] or
                receipt['case_id'] != item['official_case_id'] or
                audit['case_id'] != item['official_case_id'] or
                not audited_outcome):
            raise ValueError('Prior pilot source outcome differs: ' + str(relative))
    CertifiedRulePhaseIndex(
        Path(seed['source_rule_phase_index_file']),
        seed['source_rule_phase_index_sha256'],
        seed['manifest_sha256'])
    Store(run).validate_events()
    return strategy


from rssr_augment.bootstrap import validate_startup


def execute(strategy_path: Path, case_id: str, output: Path) -> dict:
    strategy = validate_startup(strategy_path, case_id)
    mobile_plan_budget = strategy.get('mobile_plan_budget', 6)
    if type(mobile_plan_budget) is not int or not 1 <= mobile_plan_budget <= 24:
        raise ValueError('Mobile plan budget must be between 1 and 24')
    run = Path(strategy["new_run"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    store = Store(run)
    store.validate_events()
    case = load_case(strategy, case_id, output)
    events = catalog(case["contract"], case["scene"],
                     manipulators=ADAPTER_MANIPULATORS)
    signature, binding = scene_signature(case['contract'], case['scene'])
    profile_bank = SceneProfileBank(run, signature, binding)
    scene_profile = profile_bank.context()
    phase_seed = strategy['training_profile_seed']
    phase_index = CertifiedRulePhaseIndex(
        Path(phase_seed['source_rule_phase_index_file']),
        phase_seed['source_rule_phase_index_sha256'],
        phase_seed['manifest_sha256'])
    source_duration_s = float(np.load(case["trial"] / "command_targets.npz")["time_s"][-1])
    stop_s = episode_horizon_s(source_duration_s, case['contract'], case['scene'])
    decision_limit = max(BASE_DECISIONS, int(np.ceil(stop_s / .75)))
    reserve_key = "full:event_v10:" + digest(str(output.resolve()))[:20] + ":execute"
    store.reserve("full", reserve_key, {"case_id": case_id,
                                      "role": "continuous_event_execute",
                                      "output": str(output / "execute")})
    monitor = LivePredicates(case["contract"], case["scene"])
    session = ContinuousSession(pilot_physics, rssr_servo, case["config"],
                                case["scene"], output / "execute",
                                case["target_qpos"], stop_s=stop_s, observer=monitor)
    finished = False
    decisions, history = 0, []
    physical_previews: list[dict] = []
    preview_step_count = 0
    no_more_decisions=False
    reached: set[str] = set()
    last_stalled = None
    last_replan = None
    warning_cooldown_until: dict[str, float] = {}
    warning_hand_cooldown_until: dict[str, float] = {}
    agent_errors = []
    terminal_hold_active = False
    terminal_hold_since_s = None
    terminal_hold_prior_no_more_decisions = False
    terminal_hold_transitions = []
    phase_seed_attempted: set[tuple[str, str, str]] = set()
    phase_seed_activated = []
    rules = RuleRuntime(case, events, output / 'rule_journal', file_sha(strategy_path), scene_profile)
    # Keep rule revision available after early contact setup: recorded
    # trajectories exhausted twelve ordinary reviews before later native
    # dependencies became active. Preserve four additional reviews for a
    # measured late A/B cycle.
    # Reserve revisions for newly measured actuator progress after early
    # approach reviews have used the ordinary budget.
    reviser = RuleRevisionAgent(store, output / 'rule_revisions', max_revisions=48)
    progress_epoch = ProgressEpoch(case['contract'])
    last_cycle_reviewed_at: dict[tuple[str, str], float] = {}
    last_contact_bridge_reviewed_at: dict[tuple[str, str, str], float] = {}
    profile_supplied_to_rule_agent = False
    reviewed_phase_candidates: set[tuple[str, str, str]] = set()
    mobile_model = None
    mobile_plan_count = 0
    mobile_plan_signature = None
    mobile_planned_groups = set()

    def refresh_mobile_capability(measured: dict, reason: str) -> None:
        """Replan only at paused observations; never change a selected event."""
        nonlocal mobile_plan_count, mobile_plan_signature
        if (mobile_model is None or rules.stack or
                mobile_plan_count >= mobile_plan_budget):
            return
        if any(any(bool(value) for value in hands.values())
               for hands in measured['hand_contact'].values()):
            return
        task_events = [event for event in events
                       if event.action.event != 'reposition_base']
        feasible, _ = decision_context(
            task_events, measured, case['contract'], history)
        groups = {(event.action.hand, event.action.object_id)
                  for event in feasible if event.action.event == 'approach'}
        if not groups:
            return
        old_mobile = [event for event in events
                      if event.action.event == 'reposition_base']
        fresh_old = any(immutable_event_guard(
            event, measured, case['contract'], events)[0]
            for event in old_mobile)
        failed_reach = reason in ('stagnation', 'event_timeout',
                                  'protected_predicate_lost', 'rule_rollback')
        if (reason != 'initial' and not (groups - mobile_planned_groups) and
                not (failed_reach and not fresh_old)):
            return
        signature = digest({
            'groups': sorted(groups),
            'base': [round(float(value) / .02) for value in
                     measured['base_joint_qpos']],
            'arms': {hand: [round(float(value) / .08) for value in
                            measured['arm_joint_qpos'][hand]]
                     for hand, _ in sorted(groups)},
            'objects': {obj: [round(float(value) / .015) for value in
                              measured['object_position_m'][obj]]
                        for _, obj in sorted(groups)}})
        if signature == mobile_plan_signature:
            return
        proposals, certificates = compile_reposition_events(
            mobile_model, measured, task_events, feasible)
        if len({event.event_id for event in proposals}) != len(proposals):
            raise ValueError('Mobile capability event ID collision')
        old_by_id = {event.event_id: index for index, event in enumerate(events)}
        rebound = []
        for proposal in proposals:
            if proposal.event_id in old_by_id:
                index = old_by_id[proposal.event_id]
                rebound.append({'event_id': proposal.event_id,
                                'old_sha256': digest(events[index].as_packet()),
                                'new_sha256': digest(proposal.as_packet())})
                events[index] = proposal
            else:
                events.append(proposal)
        plan_path = output / f'mobile_capability_plan_{mobile_plan_count:02d}.json'
        pinned_json(plan_path, {
            'schema': 'rssr.mobile_capability_plan.v1',
            'reason': reason, 'time_s': float(measured['time_s']),
            'candidate_scene_sha256': case['mobile_scene_sha256'],
            'robot_physics_steps_for_planning': 0,
            'source_observation_signature_sha256': signature,
            'proposals': [event.as_packet() for event in proposals],
            'rebound_event_versions': rebound,
            'certificates': certificates})
        rules.log('mobile_capability_plan', {
            'time_s': measured['time_s'], 'reason': reason,
            'plan_file': str(plan_path), 'plan_sha256': file_sha(plan_path),
            'proposed_event_ids': [event.event_id for event in proposals],
            'rebound_event_versions': rebound,
            'selection_by_rule_or_execution_agent_required': True})
        mobile_plan_count += 1
        mobile_plan_signature = signature
        mobile_planned_groups.update(groups)

    def learned_rule_memory(measured: dict) -> list[dict]:
        """Rank source rules by measured phase, condition and event availability."""
        ranked = []
        current_event_ids = {event.event_id for event in events}
        for index, source in enumerate(scene_profile.get('eligible_reuse_portfolio', [])):
            try:
                patch = profile_bank.initial_patch(
                    {**scene_profile, 'eligible_reuse': source},
                    rules.policy_sha256, '0' * 64)
                applicable = [rule['id'] for rule in patch['rules']
                              if expression(rule['when'], measured)]
                source_phase = phase_index.describe(source, measured, case['contract'])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            unavailable = sorted({offer['event_id'] for rule in patch['rules']
                                  for offer in rule['offer']
                                  if offer['event_id'] not in current_event_ids})
            phase_ready = {row['rule_id'] for row in source_phase
                           if row['source_prerequisites_met_now'] is True}
            ranked.append((not unavailable and bool(phase_ready.intersection(applicable)),
                           not unavailable, bool(phase_ready), bool(applicable), index, {
                'source_case_id': source['source_case_id'],
                'source_proposal_sha256': source['source_proposal_sha256'],
                'eligibility': source['eligibility'],
                'applicable_rule_ids_now': applicable,
                'source_phase_evidence': source_phase,
                'offered_event_ids_unavailable_now': unavailable,
                'rules_rebound_to_current_scene': patch['rules'],
            }))
        ranked.sort(key=lambda row: row[:5], reverse=True)
        return [row[5] for row in ranked[:6]]

    model_errors = (DecisionUnavailable, InvalidCompleteResponse, ValueError, BudgetExhausted, CaseBudgetExhausted)
    try:
        initial = obs = session.start()
        articulation_peaks = {}
        articulation_last_fractions = {}
        articulation_regression_latched = {}
        update_peak_and_check(case['contract'], obs, articulation_peaks,
                              articulation_last_fractions,
                              articulation_regression_latched)
        if strategy.get('allow_s1_virtual_base_experiment') is True:
            candidate_scene = output / 'mobile_model_preflight' / 'scene.xml'
            if file_sha(candidate_scene) != case['mobile_scene_sha256']:
                raise ValueError('Mobile capability candidate model changed')
            mobile_model = mujoco.MjModel.from_xml_path(str(candidate_scene))
            refresh_mobile_capability(obs, 'initial')
        seen_passed_predicates = {key for key, value in obs['live_predicates'].items()
                                  if value['passed']}
        profile_seed_activated = False
        pending_reuse_patch = None
        prior_rule_reviewed = False
        # A source rule may have become effective only after several native
        # relationships were established. Seed it directly only when those
        # measured source prerequisites and its current condition are true.
        # Other certified rules remain evidence for the RuleAgent to revise.
        reusable = None
        for candidate in reversed(scene_profile.get('eligible_reuse_portfolio', [])):
            try:
                source_phase = phase_index.describe(candidate, obs, case['contract'])
                ready_ids = {row['rule_id'] for row in source_phase
                             if row['source_prerequisites_met_now'] is True}
                if not ready_ids:
                    continue
                candidate_patch = profile_bank.initial_patch(
                    {**scene_profile, 'eligible_reuse': candidate},
                    rules.policy_sha256, '0' * 64)
                current_event_ids = {event.event_id for event in events}
                seed_rule_ids = {rule['id'] for rule in candidate_patch['rules']
                                 if rule['id'] in ready_ids and
                                 all(offer['event_id'] in current_event_ids
                                     for offer in rule['offer']) and
                                 expression(rule['when'], obs)}
                if not seed_rule_ids:
                    continue
                reusable = {**candidate,
                            'rules': [rule for rule in candidate['rules']
                                      if rule['id'] in seed_rule_ids]}
                break
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        if reusable and obs['immutable_hard']['all_hard_passed']:
            seed_dir = output / 'profile_seed'
            pinned_json(seed_dir / 'packet.json', {
                'schema': 'rssr.scene_profile_seed_packet.v1',
                'scene_signature': signature,
                'start_profile_sha256': scene_profile['profile_sha256'],
                'source_case_id': reusable['source_case_id'],
                'source_proposal_sha256': reusable['source_proposal_sha256'],
                'source_binding': reusable['source_binding'],
                'current_binding': binding,
                'eligibility': reusable['eligibility'],
                'external_historical_success_oracle_used': False})
            try:
                seed_patch = profile_bank.initial_patch(
                    {**scene_profile, 'eligible_reuse': reusable}, rules.policy_sha256,
                    file_sha(seed_dir / 'packet.json'))
                pinned_json(seed_dir / 'proposal.json', seed_patch)
                profile_seed_activated = rules.activate(seed_patch, seed_dir, obs, history)
                if not profile_seed_activated:
                    pending_reuse_patch = seed_patch
            except (KeyError, TypeError, ValueError) as error:
                rules.log('profile_seed_rejected', {'time_s': obs['time_s'],
                    'source_case_id': reusable['source_case_id'],
                    'source_proposal_sha256': reusable['source_proposal_sha256'],
                    'reason': str(error), 'physical_state_reset': False})
        current: EventDefinition | None = None
        last_margin = None
        stalled_windows = 0
        action_elapsed = 0.0
        while not session.done:
            # The last selected event is not a standing plan. Once the online
            # decision budget is exhausted, replace it with a measured hold;
            # otherwise an old approach can keep commanding the robot until
            # the physical horizon even though no agent can revise it.
            if decisions >= decision_limit and current is not None and \
                    current.action.event != 'hold':
                no_more_decisions = True
                current = None
                last_replan = 'decision_budget_exhausted'
            posture_ok_now = bool(
                obs['immutable_hard']['hard_constraints']['waist_coupling_within_1mrad'] and
                obs['immutable_hard']['hard_constraints']['limited_joints_within_2mrad_or_2mm'])
            task_complete_now = bool(
                obs['live_predicates'] and
                all(row['passed'] for row in obs['live_predicates'].values()) and
                obs['immutable_hard']['all_hard_passed'] and posture_ok_now)
            if terminal_hold_active and not task_complete_now:
                terminal_hold_active = False
                terminal_hold_since_s = None
                terminal_hold_transitions.append({
                    'time_s': float(obs['time_s']), 'event': 'task_regressed',
                    'failed_predicate_ids': [
                        key for key, row in obs['live_predicates'].items()
                        if not row['passed']]})
                no_more_decisions = (terminal_hold_prior_no_more_decisions or
                                     decisions >= decision_limit or
                                     not obs['immutable_hard']['all_hard_passed'])
                current = None
                last_replan = 'protected_predicate_lost'
            elif not terminal_hold_active and task_complete_now:
                # Preserve a measured complete task with the existing hold
                # primitive. Resume online decisions if a task predicate
                # regresses before the terminal dwell is verified.
                terminal_hold_active = True
                terminal_hold_since_s = float(obs['time_s'])
                terminal_hold_prior_no_more_decisions = no_more_decisions
                terminal_hold_transitions.append({
                    'time_s': float(obs['time_s']), 'event': 'task_complete_hold'})
                no_more_decisions = True
                current = None
            if (terminal_hold_active and task_complete_now and
                    terminal_hold_since_s is not None and
                    float(obs['time_s'])-terminal_hold_since_s >= 1.0):
                # The complete native predicate set has remained true during
                # an actual one-second physical hold. End this trajectory at
                # the next audited step; do not demand an unrelated source
                # video duration after the task has finished.
                session.request_terminal_stop()
                terminal_hold_transitions.append({
                    'time_s': float(obs['time_s']),
                    'event': 'task_complete_terminal_stop_requested',
                    'verified_hold_s': float(obs['time_s'])-terminal_hold_since_s})
                hold = next(e for e in events if e.action.event == 'hold')
                obs, _ = session.advance(hold.action, WINDOW_S,
                                         action_epoch=decisions)
                continue
            if current is None:
                if (not obs['immutable_hard']['all_hard_passed'] or
                        not posture_ok_now):
                    no_more_decisions = True
                if (not terminal_hold_active and not no_more_decisions and
                        obs['immutable_hard']['all_hard_passed']):
                    refresh_mobile_capability(obs, last_replan or 'new_dependency')
                available, rule_context = rules.candidates(obs, history)
                available_ids = {event.event_id for event in available}
                rule_memory = learned_rule_memory(obs)
                phase_candidates = executable_certified_rule_keys(
                    rule_memory, available_ids)
                phase_seed_just_activated = False
                # Certified rules may first become applicable after contact or
                # another measured task relation. Activate one at that phase,
                # using the same validated patch and rollback monitor as the
                # initial seed. The rule content remains agent-authored.
                if (not terminal_hold_active and not no_more_decisions and
                        not rules.stack and obs['immutable_hard']['all_hard_passed']):
                    sources = {(row['source_case_id'], row['source_proposal_sha256']): row
                               for row in scene_profile.get('eligible_reuse_portfolio', [])}
                    for memory in rule_memory:
                        source_key = (memory['source_case_id'],
                                      memory['source_proposal_sha256'])
                        source = sources.get(source_key)
                        if source is None:
                            continue
                        for rule in memory['rules_rebound_to_current_scene']:
                            key = (*source_key, rule['id'])
                            if key not in phase_candidates or key in phase_seed_attempted:
                                continue
                            phase_seed_attempted.add(key)
                            seed_dir = output / 'phase_seed' / (
                                f'activation_{len(phase_seed_attempted):03d}')
                            source_phase = next(row for row in memory['source_phase_evidence']
                                                if row['rule_id'] == rule['id'])
                            pinned_json(seed_dir / 'packet.json', {
                                'schema': 'rssr.certified_rule_phase_seed.v1',
                                'source_case_id': source_key[0],
                                'source_proposal_sha256': source_key[1],
                                'source_rule_id': rule['id'],
                                'source_phase_evidence': source_phase,
                                'observation_time_s': float(obs['time_s']),
                                'current_predicate_passed_ids': sorted(
                                    row_id for row_id, state in obs['live_predicates'].items()
                                    if state['passed']),
                                'available_event_ids': sorted(available_ids),
                                'current_policy_sha256': rules.policy_sha256,
                                'scene_signature': signature,
                                'physical_state_reset': False})
                            try:
                                patch = profile_bank.initial_patch(
                                    {**scene_profile, 'eligible_reuse': source},
                                    rules.policy_sha256, file_sha(seed_dir / 'packet.json'))
                                patch['rules'] = [candidate for candidate in patch['rules']
                                                  if candidate['id'] == rule['id']]
                                if len(patch['rules']) != 1:
                                    raise ValueError('Certified phase rule is not unique')
                                pinned_json(seed_dir / 'proposal.json', patch)
                                if rules.activate(patch, seed_dir, obs, history):
                                    phase_seed_activated.append({
                                        'source_case_id': source_key[0],
                                        'source_proposal_sha256': source_key[1],
                                        'source_rule_id': rule['id'],
                                        'revision': str(seed_dir),
                                        'time_s': float(obs['time_s'])})
                                    reviewed_phase_candidates.add(key)
                                    phase_seed_just_activated = True
                                    available, rule_context = rules.candidates(obs, history)
                                    break
                            except (KeyError, TypeError, ValueError, IndexError) as error:
                                if not (seed_dir / 'activation.json').exists():
                                    pinned_json(seed_dir / 'activation.json', {
                                        'activated': False,
                                        'reason': str(error),
                                        'time_s': float(obs['time_s']),
                                        'physical_state_reset': False})
                                rules.log('phase_seed_rejected', {
                                    'time_s': obs['time_s'],
                                    'source_case_id': source_key[0],
                                    'source_rule_id': rule['id'],
                                    'reason': str(error),
                                    'physical_state_reset': False})
                        if phase_seed_just_activated:
                            break
                # A phase-qualified seed gets one physical decision before a
                # competing transfer review can replace it at the same instant.
                new_phase_candidates = (sorted(phase_candidates - reviewed_phase_candidates)
                    if not phase_seed_just_activated and
                    (decisions > 0 or not profile_seed_activated) else [])
                all_passed = all(row['passed'] for row in obs['live_predicates'].values())
                progress_context = progress_epoch.observe(obs)
                progress_stall_due = bool(
                    not no_more_decisions and not phase_seed_just_activated and
                    decisions < decision_limit and
                    progress_epoch.stall_review_due(
                        float(obs['time_s']), reviser.revisions, all_passed))
                current_passed = {key for key, value in obs['live_predicates'].items()
                                  if value['passed']}
                newly_passed = sorted(current_passed - seen_passed_predicates)
                seen_passed_predicates.update(current_passed)
                cycle = repeated_event_cycle(history, float(obs['time_s']))
                cycle_key = tuple(cycle['event_ids']) if cycle else None
                cycle_due = bool(cycle and not newly_passed and
                    float(obs['time_s']) - last_cycle_reviewed_at.get(cycle_key, -1e9) >= 3.)
                bridge_keys = [
                    (gap['object_id'], gap['hand'], gap['contact_template_event_id'])
                    for gap in rule_context['contact_bridge_gaps']
                    if gap['rule_offset_within_envelope']]
                bridge_due_keys = [key for key in bridge_keys
                    if float(obs['time_s']) -
                    last_contact_bridge_reviewed_at.get(key, -1e9) >= 4.]
                bridge_due = bool(bridge_due_keys)
                # A closing gripper without target contact is a new physical
                # observation. Let RuleAgent revise its own time window even
                # when repeated approach reviews have spent the normal quota.
                gripper_progress_due = new_uncontacted_gripper_closure(
                    history, obs, last_replan)
                applicable_prior_rule_ids = []
                if (pending_reuse_patch is not None and not prior_rule_reviewed
                        and decisions > 0 and not rules.stack
                        and obs['immutable_hard']['all_hard_passed']):
                    try:
                        applicable_prior_rule_ids = [rule['id'] for rule in
                            pending_reuse_patch['rules'] if expression(rule['when'], obs)]
                    except (KeyError, TypeError, ValueError, IndexError) as error:
                        prior_rule_reviewed = True
                        rules.log('pending_reuse_condition_invalid', {
                            'time_s': obs['time_s'], 'reason': str(error),
                            'source_case_id': reusable['source_case_id'],
                            'source_proposal_sha256': reusable['source_proposal_sha256'],
                            'physical_state_reset': False})
                prior_rule_condition_now = bool(applicable_prior_rule_ids)
                needs_revision = ((not profile_supplied_to_rule_agent and
                    bool(scene_profile.get('eligible_reuse_portfolio')) and
                    (not profile_seed_activated or decisions > 0)) or
                    (decisions == 0 and not profile_seed_activated) or
                    prior_rule_condition_now or
                    bool(new_phase_candidates) or
                    cycle_due or
                    progress_stall_due or
                    bridge_due or
                    gripper_progress_due or
                    rule_context['dead_end'] or
                    rule_context['rule_application']['dead_end'] or
                    bool(newly_passed) or
                    last_replan in ('stagnation', 'event_timeout', 'contact_lost',
                                    'target_contact_started',
                                    'non_target_contact',
                                    'target_nonpad_contact',
                                    'protected_predicate_lost', 'rule_rollback',
                                    'gripper_coupling_warning',
                                    'ordered_prerequisite_at_risk',
                                    'articulation_regression'))
                if (not phase_seed_just_activated and not no_more_decisions
                        and decisions < decision_limit
                        and not all_passed and needs_revision
                        and reviser.revisions < (48 if progress_stall_due or newly_passed else
                                                36 if gripper_progress_due else
                                                28 if cycle_due else 24)):
                    if cycle_due:
                        last_cycle_reviewed_at[cycle_key] = float(obs['time_s'])
                    if bridge_due:
                        for key in bridge_due_keys:
                            last_contact_bridge_reviewed_at[key] = float(obs['time_s'])
                    if prior_rule_condition_now:
                        prior_rule_reviewed = True
                        rules.log('prior_reusable_rule_condition_now_true', {
                            'time_s': obs['time_s'],
                            'source_case_id': reusable['source_case_id'],
                            'source_proposal_sha256': reusable['source_proposal_sha256'],
                            'applicable_rule_ids': applicable_prior_rule_ids,
                            'physical_state_reset': False})
                    reviewed_phase_candidates.update(phase_candidates)
                    full_profile = (not profile_supplied_to_rule_agent or
                                    bool(newly_passed) or prior_rule_condition_now or
                                    bool(new_phase_candidates))
                    rule_profile = {key: value for key, value in scene_profile.items()
                                    if key not in ('eligible_reuse_portfolio',
                                                   'eligible_reuse')}
                    rule_profile['certified_rule_memory'] = rule_memory
                    rule_profile['recent_samples'] = (
                        rule_profile['recent_samples'][-5:] if full_profile else [])
                    revision_activated = False
                    try:
                        patch, revision = reviser.propose(case_id, obs, history, case['contract'],
                            events, rules.policy_sha256, {'reason': (
                                'reusable_rule_condition_now_true' if prior_rule_condition_now
                                else 'source_phase_candidate_became_executable'
                                if new_phase_candidates
                                else 'measured_gripper_closure_without_target_contact'
                                if gripper_progress_due
                                else 'unresolved_native_progress_stalled'
                                if progress_stall_due
                                else 'repeated_event_cycle' if cycle_due
                                else 'measured_approach_without_paired_contact' if bridge_due
                                else last_replan or (
                                    'scene_cold_start_or_same_scene_profile_revision'
                                    if decisions == 0 else (
                                        'native_predicate_newly_passed' if newly_passed
                                        else 'no_feasible_event'))),
                                'newly_passed_native_predicates': newly_passed,
                                'ordered_prerequisite_warning': (
                                    history[-1].get('ordered_prerequisite_warning')
                                    if last_replan == 'ordered_prerequisite_at_risk'
                                    and history else None),
                                'repeated_event_cycle': cycle if cycle_due else None,
                                'measured_task_stall': (
                                    progress_context if progress_stall_due else None),
                                'contact_bridge_review_due': bridge_due,
                                'decision_context': rule_context,
                                'prior_reusable_source_case_id': (
                                    reusable['source_case_id'] if prior_rule_condition_now else None),
                                'prior_reusable_source_proposal_sha256': (
                                    reusable['source_proposal_sha256'] if prior_rule_condition_now else None),
                                'prior_reusable_rule_ids_currently_applicable':
                                    applicable_prior_rule_ids,
                                'newly_executable_certified_rule_keys': [
                                    {'source_case_id': source_id,
                                     'source_proposal_sha256': proposal_sha,
                                     'rule_id': rule_id}
                                    for source_id, proposal_sha, rule_id in new_phase_candidates],
                                'scene_profile': rule_profile},
                            scene=case['scene'],
                            source_task_text=case['source_task_text'])
                        profile_supplied_to_rule_agent = True
                        revision_activated = rules.activate(patch, revision, obs, history)
                    except model_errors as error:
                        entry = {'role': 'runtime_rule_revision', 'time_s': obs['time_s'],
                                 'type': type(error).__name__, 'reason': str(error),
                                 'simulation_reset': False}
                        agent_errors.append(entry)
                        rules.log('revision_unavailable_or_rejected', entry)
                    available, rule_context = rules.candidates(obs, history)
                    if revision_activated and reviser.revisions < (36 if gripper_progress_due else 24):
                        # A freshly activated rule can create a measured
                        # approach/contact gap at the same physical pose. The
                        # pre-activation review could not see that candidate.
                        # Offer it once before taking another physical step.
                        post_bridge_keys = [
                            (gap['object_id'], gap['hand'],
                             gap['contact_template_event_id'])
                            for gap in rule_context['contact_bridge_gaps']
                            if gap['rule_offset_within_envelope']]
                        new_bridge_keys = [key for key in post_bridge_keys
                            if key not in bridge_keys and float(obs['time_s']) -
                            last_contact_bridge_reviewed_at.get(key, -1e9) >= 4.]
                        if new_bridge_keys:
                            for key in new_bridge_keys:
                                last_contact_bridge_reviewed_at[key] = float(obs['time_s'])
                            try:
                                bridge_patch, bridge_revision = reviser.propose(
                                    case_id, obs, history, case['contract'],
                                    events, rules.policy_sha256, {
                                        'reason': 'contact_gap_after_rule_activation',
                                        'contact_bridge_review_due': True,
                                        'decision_context': rule_context,
                                        'scene_profile': rule_profile,
                                    }, scene=case['scene'],
                                    source_task_text=case['source_task_text'])
                                rules.activate(bridge_patch, bridge_revision,
                                               obs, history)
                            except model_errors as error:
                                entry = {'role': 'post_activation_contact_bridge_revision',
                                         'time_s': obs['time_s'],
                                         'type': type(error).__name__,
                                         'reason': str(error),
                                         'simulation_reset': False}
                                agent_errors.append(entry)
                                rules.log('revision_unavailable_or_rejected', entry)
                            available, rule_context = rules.candidates(obs, history)
                excluded = sorted(event.event_id for event in available
                    if (warning_cooldown_until.get(event.event_id, -1.) >
                        float(obs['time_s']) or
                        (event.action.hand is not None and
                         warning_hand_cooldown_until.get(event.action.hand, -1.) >
                         float(obs['time_s']))))
                if excluded:
                    available = [event for event in available
                                 if event.event_id not in excluded]
                    rule_context = {**rule_context,
                        'safety_cooldown_excluded_event_ids': excluded,
                        'safety_cooldown_basis': 'recent_measured_joint_or_gripper_warning',
                        'safety_cooldown_hands': sorted(hand for hand, until in
                            warning_hand_cooldown_until.items() if until >
                            float(obs['time_s']))}
                    rules.log('safety_cooldown', {
                        'time_s': float(obs['time_s']),
                        'excluded_event_ids': excluded,
                        'physical_state_reset': False})
                guarded = {event.event_id: reason for event in available
                           if (reason := backward_loaded_recontact(
                               event, case['contract'], obs, articulation_peaks))}
                if guarded:
                    available = [event for event in available
                                 if event.event_id not in guarded]
                    rule_context = {**rule_context,
                        'progress_guard_excluded_events': guarded,
                        'progress_guard_basis': 'measured_contact_and_open_fraction',
                        'articulation_peaks': dict(articulation_peaks)}
                    rules.log('progress_guard_exclusion', {
                        'time_s': float(obs['time_s']),
                        'excluded_events': guarded,
                        'physical_state_reset': False})
                if no_more_decisions or decisions >= decision_limit:
                    current = next(e for e in events if e.action.event == 'hold')
                elif not available and (excluded or guarded):
                    current = next(e for e in events if e.action.event == 'hold')
                else:
                    try:
                        current = decide_next(case, obs, initial, available, history,
                            store, output / 'decisions' / f'event_{decisions:02d}', rule_context)
                        decisions += 1
                        preview_horizon = horizon_for(current.action.event)
                        target_contacts = obs.get('finger_contact', {}).get(
                            current.action.object_id, {}).get(current.action.hand, ())
                        approach_reviews = sum(row.get('action_event') == 'approach'
                                               for row in physical_previews)
                        contact_reviews = sum(row.get('action_event') in ('contact', 'grasp')
                                              for row in physical_previews)
                        loaded_reviews = sum(row.get('action_event') not in
                                             ('approach', 'contact', 'grasp')
                                             for row in physical_previews)
                        recent_obstacle = any(
                            row.get('stop_reason') == 'non_target_contact' and
                            row.get('selected_hand') == current.action.hand
                            for row in history[-8:]) or bool(obs.get(
                                'gripper_scene_contacts', {}).get(current.action.hand))
                        eligible_preview = (
                            (current.action.event == 'approach' and
                             recent_obstacle and approach_reviews < 2) or
                            (current.action.event in ('contact', 'grasp') and
                             contact_reviews < 8) or
                            (current.action.event not in ('approach', 'contact', 'grasp') and
                             len(target_contacts) == 2 and all(target_contacts) and
                             loaded_reviews < 6))
                        if (preview_horizon is not None and
                                len(physical_previews) < 16 and eligible_preview):
                            try:
                                preview = session.preview(current.action,
                                                          preview_horizon,
                                                          contract=case['contract'])
                                preview_step_count += int(preview[
                                    'simulated_physics_steps'])
                                preview_path = output / 'physical_previews' / (
                                    f'preview_{len(physical_previews):02d}.json')
                                pinned_json(preview_path, preview)
                                physical_previews.append({
                                    'path': str(preview_path),
                                    'sha256': file_sha(preview_path),
                                    'event_id': current.event_id,
                                    'action_event': current.action.event,
                                    'predicted_risk': preview['predicted_risk'],
                                    'source_time_s': preview['source_time_s'],
                                    'steps': preview['simulated_physics_steps']})
                                if preview['source_state_modified']:
                                    raise AssertionError('Preview modified live source state')
                                if preview['predicted_risk']:
                                    alternatives = [event for event in available
                                        if event.event_id != current.event_id]
                                    if alternatives and decisions < decision_limit:
                                        reason = {key: preview[key] for key in (
                                            'source_sha256', 'simulated_elapsed_s',
                                            'bilateral_fraction',
                                            'first_persistent_contact_loss_s',
                                            'start_progress', 'end_progress',
                                            'initial_stage_remaining_m',
                                            'final_stage_remaining_m',
                                            'initial_approach_remaining_m',
                                            'final_approach_remaining_m',
                                            'dependent_articulation',
                                            'predicted_dependency_regression',
                                            'last_half_second_progress_rate',
                                            'late_loaded_obstruction',
                                            'late_noncarried_gripper_force_n',
                                            'late_object_displacement_m',
                                            'feature_frame_pad_displacement_m',
                                            'first_persistent_non_target_contact_s',
                                            'non_target_object_ids',
                                            'maximum_non_target_penetration_m',
                                            'predicted_hard_passed')}
                                        reason['late_contact_pairs'] = [
                                            pair for row in preview.get('trace', [])[-2:]
                                            for pair in row.get('top_contact_pairs', [])[:3]
                                            if pair.get('normal_force_n', 0) >= 5.]
                                        reason['excluded_event_id'] = current.event_id
                                        try:
                                            patch, revision = reviser.propose(
                                                case_id, obs, history, case['contract'],
                                                events, rules.policy_sha256, {
                                                    'reason': 'isolated_physical_preview_risk',
                                                    'physical_preview_risk': reason,
                                                    'decision_context': rule_context,
                                                    'scene_profile': scene_profile,
                                                }, scene=case['scene'],
                                                source_task_text=case['source_task_text'])
                                            rules.activate(patch, revision, obs, history)
                                        except model_errors as error:
                                            entry = {'role': 'preview_rule_revision',
                                                     'time_s': obs['time_s'],
                                                     'type': type(error).__name__,
                                                     'reason': str(error),
                                                     'simulation_reset': False}
                                            agent_errors.append(entry)
                                            rules.log('revision_unavailable_or_rejected', entry)
                                        revised, revised_context = rules.candidates(obs, history)
                                        revised_alternatives = [event for event in revised
                                            if event.event_id != reason['excluded_event_id']]
                                        if revised_alternatives:
                                            alternatives = revised_alternatives
                                        rule_context = {**revised_context,
                                            'physical_preview_risk': reason}
                                        current = decide_next(case, obs, initial,
                                            alternatives, history, store,
                                            output / 'decisions' /
                                            f'event_{decisions:02d}', rule_context)
                                        decisions += 1
                                    elif not preview['predicted_hard_passed']:
                                        rules.log('preview_hard_rejected_no_alternative', {
                                            'time_s': float(obs['time_s']),
                                            'event_id': current.event_id,
                                            'preview_sha256': file_sha(preview_path),
                                            'physical_state_reset': False})
                                        current = next(e for e in events
                                                       if e.action.event == 'hold')
                            except (RuntimeError, ValueError) as error:
                                entry = {'role': 'isolated_physical_preview',
                                         'time_s': obs['time_s'],
                                         'type': type(error).__name__,
                                         'reason': str(error),
                                         'simulation_reset': False}
                                agent_errors.append(entry)
                                rules.log('preview_unavailable', entry)
                        rules.selected(current, rule_context, obs)
                    except model_errors as error:
                        entry = {'role': 'next_physical_event', 'time_s': obs['time_s'],
                                 'type': type(error).__name__, 'reason': str(error),
                                 'simulation_reset': False}
                        agent_errors.append(entry)
                        rules.log('execution_agent_unavailable', entry)
                        no_more_decisions = True
                        current = next(e for e in events if e.action.event == 'hold')
                last_margin = (-1.0 if current.action.event in ('clearance', 'retreat')
                               else signed_margin(current, obs, initial,
                                                  case['contract']))
                current_policy_sha256 = rules.policy_sha256
                stalled_windows, action_elapsed = 0, 0.0
                last_replan = None
            previous_time = obs["time_s"]
            last_coupling_residual = [float(
                obs['instantaneous_gripper_coupling_residual_rad'])]
            last_arm_torso_margin = [float(obs['arm_torso_min_joint_margin_rad'])]
            last_gripper_margin = [float(obs['gripper_min_joint_margin_rad'])]
            prior_contact = None
            if current.action.object_id is not None and current.action.hand is not None:
                prior_contact = any(obs["finger_contact"][current.action.object_id]
                                       [current.action.hand])
            # Contact impulses near a mesh edge can flicker for a few native
            # steps. Keep the current short event during a brief dropout, but
            # let the online agent replan if loss persists. The immutable hard
            # checks above still stop immediately on a safety regression.
            contact_loss_since_s = [None]
            contact_loss_grace_s = 0.025
            native_trigger = event_trigger(current, initial, case["contract"])
            ordered_warning = [None]

            def replan_trigger(measured: dict) -> str | None:
                if not measured['immutable_hard']['all_hard_passed']:
                    return 'hard_regression'
                arm_torso_margin = float(measured['arm_torso_min_joint_margin_rad'])
                approaching_native_limit = (
                    last_arm_torso_margin[0] >= JOINT_MARGIN_WARNING_RAD >
                    arm_torso_margin)
                last_arm_torso_margin[0] = arm_torso_margin
                if approaching_native_limit and current.action.event != 'hold':
                    return 'joint_margin_warning'
                gripper_margin = float(measured['gripper_min_joint_margin_rad'])
                approaching_gripper_limit = (
                    last_gripper_margin[0] >= GRIPPER_MARGIN_WARNING_RAD >
                    gripper_margin)
                last_gripper_margin[0] = gripper_margin
                if approaching_gripper_limit and current.action.event != 'hold':
                    return 'gripper_joint_margin_warning'
                residual = float(measured[
                    'instantaneous_gripper_coupling_residual_rad'])
                rising_warning = (last_coupling_residual[0] <
                                  COUPLING_WARNING_RAD <= residual)
                last_coupling_residual[0] = residual
                if (rising_warning and current.action.event != 'hold' and
                        current.action.hand in ADAPTER_MANIPULATORS):
                    return 'gripper_coupling_warning'
                if (current.action.event == 'approach' and
                        current.action.object_id is not None and
                        current.action.hand is not None and
                        any(measured['finger_contact'][current.action.object_id]
                            [current.action.hand])):
                    return 'target_contact_started'
                if monitor.lost_protected(measured):
                    return "protected_predicate_lost"
                regression = update_peak_and_check(
                    case['contract'], measured, articulation_peaks,
                    articulation_last_fractions,
                    articulation_regression_latched)
                if regression is not None:
                    return regression
                ordered_warning[0] = ordered_relation_prerequisite_warning(
                    current, case['contract'], measured)
                if ordered_warning[0] is not None:
                    return 'ordered_prerequisite_at_risk'
                if (current.action.event in ('approach', 'contact') and
                        current.action.object_id is not None and
                        current.action.hand is not None and
                        not any(measured['finger_contact']
                                [current.action.object_id][current.action.hand]) and
                        any(c['object_id'] == current.action.object_id for c in
                            measured['gripper_scene_contacts'][current.action.hand])):
                    return 'target_nonpad_contact'
                result = native_trigger(measured)
                if result is not None:
                    return result
                if (current.action.event in ('approach', 'contact', 'grasp') and
                        any(c['object_id'] != current.action.object_id for c in
                            measured['gripper_scene_contacts'][current.action.hand])):
                    return 'non_target_contact'
                # A grip that changes while carrying or lifting is a new
                # physical branch.  Stop the short action and let K3 select
                # the next event using this exact, still-live simulator state.
                if prior_contact and current.action.event in (
                        "lift", "transport", "align", "clearance", "open_drawer",
                        "close_drawer", "push_close"):
                    sides = measured['finger_contact'][current.action.object_id][current.action.hand]
                    holding = any(sides) if current.action.event == 'push_close' else all(sides)
                    if holding:
                        contact_loss_since_s[0] = None
                    else:
                        now_s = float(measured['time_s'])
                        if contact_loss_since_s[0] is None:
                            contact_loss_since_s[0] = now_s
                        elif now_s - contact_loss_since_s[0] >= contact_loss_grace_s:
                            return 'contact_lost'
                return None

            # Once the frozen hard audit is irreversibly violated and decisions
            # are exhausted, retain bounded robot hold commands to the native
            # terminal report without thousands of identical replan alerts.
            trigger = replan_trigger if not no_more_decisions and decisions < decision_limit else None
            obs, reason = session.advance(current.action, WINDOW_S, trigger,
                                          action_epoch=decisions)
            acquisition_settle = None
            # A control segment can end on the first few bilateral-contact
            # frames. Finish the original 100ms contact prerequisite before
            # handing this safe acquisition boundary back to policy rollback.
            # This advances the same physical session with the same contact
            # action, forces, hard monitor and issued-command clock.
            h, o = current.action.hand, current.action.object_id
            fresh_cup_contact = (
                reason == 'duration' and current.action.event in ('contact', 'grasp')
                and h in ('left', 'right')
                and obs.get('object_geometry', {}).get(o, {}).get('shape') == 'cup'
                and all(obs['finger_contact'][o][h])
                and 0 <= obs.get('bilateral_contact_seconds', {}).get(o, {}).get(h, 0.) < .10
                and obs['immutable_hard']['all_hard_passed']
                and not any(c['object_id'] != o for c in obs['gripper_scene_contacts'][h]))
            if fresh_cup_contact:
                acquisition_settle = {'start_time_s': float(obs['time_s']),
                    'initial_bilateral_contact_seconds': float(obs['bilateral_contact_seconds'][o][h]),
                    'maximum_extra_physical_duration_s': .12}
                obs, reason = session.advance(current.action, .12, trigger,
                                              action_epoch=decisions)
                acquisition_settle.update(end_time_s=float(obs['time_s']),
                    final_bilateral_contact_seconds=float(obs['bilateral_contact_seconds'][o][h]),
                    stop_reason=reason)
            warning_observation = None
            if reason in ('gripper_coupling_warning', 'joint_margin_warning',
                          'gripper_joint_margin_warning'):
                warning_cooldown_until[current.event_id] = (
                    float(obs['time_s']) + 1.0)
                if reason == 'gripper_coupling_warning' and current.action.hand is not None:
                    warning_hand_cooldown_until[current.action.hand] = (
                        float(obs['time_s']) + 1.0)
                warning_observation = {
                    'time_s': float(obs['time_s']),
                    'type': reason,
                    'arm_torso_min_joint_margin_rad': float(
                        obs['arm_torso_min_joint_margin_rad']),
                    'limiting_arm_torso_joint': obs['limiting_arm_torso_joint'],
                    'gripper_min_joint_margin_rad': float(
                        obs['gripper_min_joint_margin_rad']),
                    'limiting_gripper_joint': obs['limiting_gripper_joint'],
                    'instantaneous_gripper_coupling_residual_rad': float(
                        obs['instantaneous_gripper_coupling_residual_rad']),
                    'cumulative_hard_passed': bool(
                        obs['immutable_hard']['all_hard_passed']),
                }
                # Stay in the same native physics session. A brief robot hold
                # lets the measured transient settle before the agent decides
                # the next physical event; no object or simulator state write.
                obs, hold_reason = session.advance(
                    EventAction(event='hold', hand=current.action.hand),
                    TRANSITION_HOLD_S,
                    lambda measured: ('hard_regression' if not
                        measured['immutable_hard']['all_hard_passed'] else None),
                    action_epoch=decisions)
                if hold_reason == 'hard_regression':
                    reason = 'hard_regression'
            elapsed = max(0.0, obs["time_s"] - previous_time)
            action_elapsed += elapsed
            margin = signed_margin(current, obs, initial, case["contract"])
            rule_rolled_back = rules.observe(obs, reason)
            progress = margin - last_margin
            if progress < 0.001 and current.action.event != "reference":
                stalled_windows += 1
            else:
                stalled_windows = 0
            history.append({"event_id": current.event_id,
                            "base_certificate_sha256": current.action.base_certificate_sha256,
                            "selected_object_id": current.action.object_id,
                            "selected_hand": current.action.hand,
                            "rule_policy_sha256": current_policy_sha256,
                            "immutable_hard": obs["immutable_hard"],
                            "time_s": round(obs["time_s"], 4),
                            "signed_margin": round(margin, 5),
                            "delta_margin": round(progress, 5),
                            "stop_reason": reason,
                            "transition_warning": warning_observation,
                            "acquisition_settle": acquisition_settle,
                            "ordered_prerequisite_warning": ordered_warning[0],
                            "instantaneous_gripper_coupling_residual_rad": float(
                                obs['instantaneous_gripper_coupling_residual_rad']),
                            "finger_contact": obs["finger_contact"],
                            "hand_contact": obs["hand_contact"],
                            "gripper_opening_m": dict(obs["gripper_opening_m"]),
                            "gripper_scene_contacts": obs["gripper_scene_contacts"],
                            "live_predicates": obs["live_predicates"],
                            "object_position_m": {k: v.tolist() for k,v in obs["object_position_m"].items()},
                            "hard_penetration_m": round(
                                obs["max_robot_table_penetration_m"], 6)})
            last_margin = margin
            if reason == "event_reached":
                reached.add(current.event_id)
                monitor.protect(current.source_predicate, obs)
                current = None
                last_stalled = None
            elif reason in ("hard_regression", "posture_regression",
                            "contact_lost", "protected_predicate_lost",
                            "gripper_coupling_warning", "joint_margin_warning",
                            "gripper_joint_margin_warning",
                            "ordered_prerequisite_at_risk",
                            "articulation_regression",
                            "target_nonpad_contact"):
                if reason in ('hard_regression', 'posture_regression'):
                    no_more_decisions=True
                current = None
                last_stalled = history[-1]["event_id"]
                last_replan = reason
            elif rule_rolled_back:
                history[-1]['stop_reason'] = 'rule_rollback'
                last_replan = 'rule_rollback'
                current = None
            elif ((stalled_windows >= 3 or action_elapsed >= 8.0) and
                  not (current.action.event == "hold" and
                       all(row["passed"] for row in obs["live_predicates"].values()))):
                history[-1]["stop_reason"] = "stagnation" if stalled_windows >= 3 else "event_timeout"
                last_stalled = current.event_id
                last_replan = history[-1]["stop_reason"]
                current = None
        session.join()
        session.finish()
        obs = session.observe()
        finished = True
    finally:
        if not finished:
            session.abort()
            store.settle(reserve_key, "interrupted", str(output / "execute"))
    store.settle(reserve_key, "complete", str(output / "execute/report.json"))
    if (file_sha(output / "execute/scene.xml") != case["mobile_scene_sha256"] or
            file_sha(output / "execute/scene.xml") != case["baseline_scene_sha256"]):
        raise ValueError("Continuous episode changed the frozen scene")
    if json.loads((output / 'execute/task_contract.json').read_text()) != case['contract']:
        raise ValueError('Continuous episode changed corrected task contract')
    native = score_report(output / "execute/report.json", case_id)
    posture = {
        'schema': 'rssr.s1_native_range_coupled_waist_gate.v1',
        'joint_names': ['astribot_torso_joint_1', 'astribot_torso_joint_2'],
        'initial_qpos_rad': session.torso_start_qpos.tolist(),
        'maximum_every_step_command_excursion_rad':
            session.torso_command_max_excursion_rad.tolist(),
        'maximum_every_step_physical_excursion_rad':
            session.torso_actual_max_excursion_rad.tolist(),
        'command_envelope': 'both original S1 torso joint ranges with 0.07 rad command margin',
        'physical_gate': 'every-step native joint limits and 1 mrad waist coupling',
        'every_step_servo_observations_including_final_state': True,
        'passed': bool(obs['immutable_hard']['hard_constraints']['waist_coupling_within_1mrad'] and
                       obs['immutable_hard']['hard_constraints']['limited_joints_within_2mrad_or_2mm']),
    }
    atomic_json(output / 'torso_posture_gate.json', posture)
    executed_horizon_s = float(session.command_records[-1][0])
    replay_stop_s = (stop_s if not session.terminal_stop_requested else
                     (len(session.command_records)-1-1e-6)*session.timestep_s)
    report_quality=json.loads((output/'execute/report.json').read_text())
    head_view_quality=report_quality['head_view_audit']
    quality_passed=bool(head_view_quality['passed'])
    atomic_json(output/'head_view_quality_gate.json',head_view_quality)
    replay = None
    if (quality_passed and native["task_text_physical_success"] and
            obs["immutable_hard"]["all_hard_passed"] and posture['passed']):
        replay_key = "full:event_v10:" + digest(str(output.resolve()))[:20] + ":replay"
        store.reserve("full", replay_key, {"case_id": case_id,
                                          "role": "independent_exact_replay",
                                          "output": str(output / "replay")})
        replay_done = False
        try:
            replay = _replay(case, output, session.command_records,
                             stop_s=replay_stop_s)
            replay_done = True
        finally:
            store.settle(replay_key, "complete" if replay_done else "interrupted",
                         str(output / "replay/report.json"))
    else:
        # Preserve the actual command stream for physical failure analysis too.
        rows = session.command_records
        np.savez_compressed(output / "issued_robot_commands.npz",
                            time_s=np.array([x[0] for x in rows]),
                            qpos=np.stack([x[1] for x in rows]),
                            qvel=np.stack([x[2] for x in rows]),
                            event=np.array([x[3] for x in rows], dtype="U40"))
    atomic_json(output / 'contact_ik_plan_records.json',
                {'schema': 'rssr.contact_ik_plan_records.v1',
                 'records': session.contact_ik_plan_records})
    receipt = {"schema": "rssr.retarget.continuous_event_receipt.v1",
               "case_id": case_id, "strategy_seq": strategy["seq"],
               "strategy_sha256": file_sha(strategy_path),
               "agent_version": VERSION,
               "baseline_scene_sha256": case["baseline_scene_sha256"],
               "mobile_scene_sha256": case["mobile_scene_sha256"],
               "s1_virtual_planar_base_experiment_enabled": bool(
                   strategy.get('allow_s1_virtual_base_experiment') is True),
               "mobile_plan_budget": mobile_plan_budget,
               "mobile_capability_plan_records": [
                   {'path': str(output / f'mobile_capability_plan_{index:02d}.json'),
                    'sha256': file_sha(output / f'mobile_capability_plan_{index:02d}.json')}
                   for index in range(mobile_plan_count)],
               "baseline_command_targets_sha256": case["baseline_command_targets_sha256"],
               "event_history": history,
               "event_decisions": decisions,
               "physical_previews": physical_previews,
               "copied_state_physics_steps": preview_step_count,
               "rule_revisions_requested": reviser.revisions,
               "progress_epoch": progress_epoch.snapshot(float(obs['time_s'])),
               "rule_runtime": rules.summary(),
               "scene_signature": signature,
               "scene_binding": binding,
               "start_scene_profile_sha256": scene_profile['profile_sha256'],
               "prior_same_scene_case_ids": scene_profile['completed_same_scene_cases'],
               "profile_seed_activated": profile_seed_activated,
               "phase_seed_attempted": len(phase_seed_attempted),
               "phase_seed_activated": phase_seed_activated,
               "agent_errors_retained": agent_errors,
               "contact_ik_plan_records_sha256": file_sha(
                   output / 'contact_ik_plan_records.json'),
               "contact_ik_plan_count": len(session.contact_ik_plan_records),
               "online_every_step_hard_snapshot": obs["immutable_hard"],
               "immutable_hard_trace_sha256": file_sha(output / 'execute/immutable_hard_trace.npz'),
               "torso_posture_gate": posture,
               "torso_posture_gate_sha256": file_sha(output / 'torso_posture_gate.json'),
               "same_simulation_state": True, "horizon_s": stop_s,
               "executed_horizon_s": executed_horizon_s,
               "task_complete_early_terminal": session.terminal_stop_requested,
               "decision_limit": decision_limit,
               "source_duration_s": source_duration_s,
               "terminal_hold_transitions": terminal_hold_transitions,
               "official_source": case["official_source"],
               "historical_success_labels_used_for_decisions": True,
               "authorized_prior_training_profile_used": True,
               "training_profile_source_strategy_seq": 189,
               "training_profile_source_strategy_seqs":
                   scene_profile['seed_source_strategy_seqs'],
               "training_profile_manifest_sha256": strategy['training_profile_seed']['manifest_sha256'],
               "training_profile_source_rule_phase_index_sha256":
                   strategy['training_profile_seed']['source_rule_phase_index_sha256'],
               "object_state_written_by_agent": False,
               "issued_robot_commands_sha256": file_sha(
                   output / "issued_robot_commands.npz"),
               "native": native, "replay": replay,
               "head_view_quality": head_view_quality,
               "accepted": bool(quality_passed and posture['passed'] and replay and replay["exact"] and replay["every_step_hard_trace_exact"] and
                                obs["immutable_hard"]["all_hard_passed"] and replay["every_step_hard_summary"]["all_hard_passed"] and
                                replay["score"]["task_text_physical_success"])}
    atomic_json(output / "receipt.json", receipt)
    profile_sample = profile_bank.append(receipt, output / 'rule_journal',
                                         output / 'rule_revisions', scene_profile,
                                         case['contract'])
    atomic_json(output / 'scene_profile_sample_link.json',
                {'path': str(profile_sample), 'sha256': file_sha(profile_sample),
                 'scene_signature': signature})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        strategy = validate_startup(args.strategy, args.case_id)
        print(json.dumps({"strategy_seq": strategy["seq"], "case_id": args.case_id,
                          "executor_module": __package__, "agent_version": VERSION,
                          "startup_valid": True, "new_physics_or_api_calls": 0}))
        return
    receipt=execute(args.strategy,args.case_id,args.output)
    print(json.dumps({k:receipt[k] for k in ("case_id","accepted","event_decisions","horizon_s")},ensure_ascii=False))


if __name__ == "__main__":
    main()
