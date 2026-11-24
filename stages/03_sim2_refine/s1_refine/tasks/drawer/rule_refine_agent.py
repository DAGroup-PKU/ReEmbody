"""Deterministic, success-preserving drawer refinement through frozen v7.

The rule agent verifies a complete measured32 Sim2 and a completely verified v7
refinement, preserves that result as rollback, then reproduces it in a fresh
directory before trying a small, measured release-height optimization. It does
not call an API, change a controller/validator, render, or admit publication.

Run ``python rule_refine_agent.py plan --help``. ``plan`` adds no physics or
files; ``run`` invokes the existing maintained ``run.py drawer`` executor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
DEFAULT_PYTHON = Path(sys.executable)
SCHEMA = 'rssr.drawer.rule_refine_agent.v1'
CONTROLLER = 'source_conditioned_v7'
PHASE_CHAINS = {
    'handle': ('approach_handle', 'grasp_handle', 'open_drawer', 'hold_open',
               'close_drawer', 'release_handle'),
    'payload': ('preapproach_payload', 'approach_payload', 'grasp_payload',
                'lift_payload', 'approach_rim_payload', 'cross_rim_payload',
                'lower_payload', 'release_payload', 'withdraw_payload'),
}
PREDICATE_PHASES = {
    'initial_outside': ('initial_state',),
    'inside': ('lower_payload',),
    'object_contact': ('release_payload',),
    'open_insert_close': ('open_drawer', 'lower_payload', 'close_drawer'),
    'stable': ('terminal_window',),
    'released': ('release_payload', 'release_handle', 'terminal_window'),
}
PARAMETERS = {
    'payload_speed': ('requested_robot_target_speeds_m_s', 'payload'),
    'drawer_speed': ('requested_robot_target_speeds_m_s', 'drawer'),
    'placement_side_margin': ('placement_side_margin_m',),
    'withdraw_rise': ('withdraw_rise_m',),
    'release_drop_height': ('controlled_release_drop_height_m',),
    'placement_local_y': ('placement_local_y_m',),
    'payload_preapproach_height': ('payload_preapproach_height_m',),
    'open_fraction_target': ('requested_open_fraction',),
}
LAYERS = {
    'issued_robot_commands.npz': 'actual_issued_joint_commands',
    'drawer_command_timeline.npz': 'controller_cartesian_reference_and_jaw_intent',
    'execute/rollout.npz': 'actual_observed_native_state',
    'execute/immutable_hard_trace.npz': 'actual_every_native_step_hard_measurements',
    'native_grasp_observations.npz': 'actual_native_loaded_grasp_observations',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_pins(pins):
    require(isinstance(pins, dict) and bool(pins), 'Missing source pins')
    for path, expected in pins.items():
        require(Path(path).is_absolute() and sha(path) == expected,
                'Missing or changed pinned evidence: ' + path)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def progress(output, status, outcomes, candidate=None):
    value = dict(schema=SCHEMA, status=status,
        updated_utc=datetime.now(timezone.utc).isoformat(),
        completed_native_trials=len(outcomes),
        verified_native_trials=sum(row.get('verified') is True for row in outcomes),
        current_candidate=candidate['id'] if candidate else None,
        api_calls=0, rendering_started=False)
    temporary = output / 'progress.tmp'
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, output / 'progress.json')


def load_tool(name):
    path = ROOT / 'orchestration' / (name + '.py')
    spec = importlib.util.spec_from_file_location('_rule_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def phases(native):
    receipt = read(Path(native) / 'receipt.json')
    return receipt.get('event_history') or []


def phase_dag(contract, history, physical_end_s):
    """Bind every original predicate and complete branch, rejecting omissions.

    The contract does not encode a phase graph. Edges below are the explicit
    v7 adapter interpretation of its unchanged open/insert/close semantics.
    Original predicates remain the judge; these edges add no task success.
    """
    require(contract.get('task_id') == 'dm6', 'Expected complete native dm6')
    predicates = contract.get('predicates', [])
    ids = [row['id'] for row in predicates]
    require(len(ids) == len(set(ids)) and bool(ids), 'Duplicate/missing predicates')
    require({row['kind'] for row in predicates} == set(PREDICATE_PHASES),
            'Unmapped original task predicate; extend the adapter explicitly')
    require(bool(contract.get('hard_constraints')), 'Original hard constraints absent')
    require(all(float(row['window_s']) == 1. for row in predicates
                if row['kind'] in ('stable', 'released')),
            'Original one-second stability/release windows changed')
    expected = {name for chain in PHASE_CHAINS.values() for name in chain}
    require(len(history) == len(expected) and {r['phase'] for r in history} == expected,
            'Source must contain every complete drawer phase exactly once')
    table = {r['phase']: r for r in history}
    nodes = {'initial_state': {'kind': 'actual_initial_state', 'start_s': 0., 'end_s': 0.}}
    edges = []
    for branch, chain in PHASE_CHAINS.items():
        previous = 'initial_state'
        for phase in chain:
            row = table[phase]
            start, end = float(row['start_s']), float(row['end_s'])
            require(row.get('branch') == branch and row.get('stop_reason') == 'event_reached'
                    and 0 <= start < end <= physical_end_s, 'Incomplete source phase: ' + phase)
            if previous != 'initial_state':
                require(float(table[previous]['end_s']) <= start + 1e-8,
                        'Source branch order differs: ' + phase)
            nodes[phase] = dict(kind='controller_phase', branch=branch,
                                start_s=start, end_s=end, hand=row['hand'])
            edges.append([previous, phase])
            previous = phase
    edges += [['open_drawer', 'lower_payload'], ['withdraw_payload', 'close_drawer']]
    require(table['open_drawer']['end_s'] <= table['lower_payload']['start_s'] + 1e-8
            and table['withdraw_payload']['end_s'] <= table['close_drawer']['start_s'] + 1e-8,
            'Source phase/task order differs from this adapter')
    last_release = max(float(table[p]['end_s']) for p in ('release_handle', 'withdraw_payload'))
    require(last_release <= physical_end_s - 1. + 1e-8, 'Source lacks original final window')
    nodes['terminal_window'] = dict(kind='original_stability_and_release_window',
                                  start_s=physical_end_s - 1., end_s=physical_end_s)
    edges += [[p, 'terminal_window'] for p in ('release_handle', 'withdraw_payload')]
    for predicate in predicates:
        key = 'predicate:' + predicate['id']
        nodes[key] = dict(kind='original_predicate', predicate=predicate)
        edges += [[phase, key] for phase in PREDICATE_PHASES[predicate['kind']]]
        for dependency in predicate.get('depends_on', []):
            require(dependency in ids, 'Unknown explicit predicate dependency')
            edges.append(['predicate:' + dependency, key])
    remaining = set(nodes)
    order = []
    while remaining:
        ready = sorted(node for node in remaining
                       if not any(b == node and a in remaining for a, b in edges))
        require(bool(ready), 'Task/phase dependency cycle')
        order.extend(ready)
        remaining.difference_update(ready)
    return dict(nodes=nodes, edges=edges, topological_order=order,
                original_contract_id=contract['contract_id'],
                hard_constraints=contract['hard_constraints'],
                interpretation='v7 explicit semantic adapter; native task predicates remain unchanged')


def metrics(native):
    observation = read(Path(native) / 'drawer_dual_observations.json')
    evidence = observation.get('controlled_release_evidence') or {}
    height = evidence.get('actual_drop_height_m')
    require(isinstance(height, (int, float)) and math.isfinite(height) and height >= 0
            and evidence.get('complete_xy_footprint_inside') is True,
            'Missing actual complete-footprint controlled-release evidence')
    return dict(actual_release_drop_height_m=float(height),
                all_branches_complete_s=max(float(r['end_s']) for r in phases(native)),
                actual_physical_end_s=float(read(Path(native) / 'receipt.json')['actual_physical_end_s']))


def command_for(plan, candidate, output):
    command = [plan['python'], '-u', str(ROOT / 'run.py'), 'drawer',
               '--controller', CONTROLLER, '--baseline-native', plan['baseline_native'],
               '--baseline-handoff', plan['baseline_handoff'],
               '--prepared-case', plan['prepared_case'], '--carry-reference', plan['carry_reference'],
               '--output', str(output)]
    for name, value in sorted(candidate['parameters'].items()):
        command += ['--' + name.replace('_', '-'), format(value, '.17g')]
    return command


def build_plan(args):
    native = args.successful_native.resolve(strict=True)
    require(native.is_absolute(), 'Use an explicit successful native directory')
    # Existing combined verifier rechecks task, hard, grasp, alias, clearance,
    # actual issued commands, all source pins and independent exact replay.
    proof = load_tool('drawer_render_watch').combine(native)
    receipt, started = read(native / 'receipt.json'), read(native / 'drawer_run_started.json')
    require(receipt.get('controller') == 'rssr.drawer.source_conditioned_refine.v7'
            and started.get('schema') == 'rssr.drawer.source_conditioned_refine.v7',
            'This adapter explicitly supports verified source_conditioned_v7 only')
    quality = read(native / 'drawer_quality_receipt.json')
    handoff = Path(args.baseline_handoff or quality['source_baseline_handoff']).resolve(strict=True)
    baseline = load_tool('drawer_source_conditioned_quality_v1').verify_target_baseline(handoff)
    baseline_native = Path(baseline['source_native']).resolve(strict=True)
    require(baseline_native == Path(receipt['source_baseline_native']).resolve(strict=True),
            'Successful refined result and target baseline have different sources')
    contract = read(native / 'execute/task_contract.json')
    require(contract == read(baseline_native / 'execute/task_contract.json'),
            'Successful refined and Sim2 task contracts differ')
    setup = read(baseline_native / 'source_conditioned_adaptation.json')
    reference_receipt = read(Path(setup['source_reference']).with_name('reference_receipt.json'))
    prepared = Path(reference_receipt['prepared_case']).resolve(strict=True)
    route = Path(started['source_carry_corridor_reference']).resolve(strict=True)
    parameters = {}
    for name, keys in PARAMETERS.items():
        value = started
        for key in keys:
            value = value[key]
        require(type(value) in (int, float) and math.isfinite(value), 'Invalid source parameter')
        parameters[name] = float(value)
    source_metrics = metrics(native)
    dag = phase_dag(contract, phases(native), source_metrics['actual_physical_end_s'])
    require(started['native_requested_stop_s'] == 14.965
            and started['duration_limit_s'] == 15.
            and started['immutable_action_envelope_m_s'] == .1,
            'Successful source has a different immutable horizon/action envelope')
    pins = dict(proof['input_pins'])
    for incoming in (baseline['input_pins'], started['input_pins']):
        for path, expected in incoming.items():
            require(path not in pins or pins[path] == expected, 'Inconsistent source pins')
            pins[path] = expected
    for path in [Path(__file__).resolve(), ROOT / 'run.py',
                 ROOT / 'orchestration/drawer_source_conditioned_cli_v1.py',
                 native / 'drawer_run_started.json', native / 'drawer_dual_observations.json',
                 native / 'source_conditioning.json', native / 'drawer_command_timeline.npz',
                 native / 'execute/task_contract.json', handoff, route]:
        pins[str(path)] = sha(path)
    verify_pins(pins)
    # A 2mm maximum step is a bounded new robot goal, not a relaxed task gate.
    # Its scale is also <=5% of the original measured drop. Earlier phases
    # compute the floor but overwrite its Z, so this parameter first acts in
    # lower_payload. Exact command/reference/state prefix equality checks this.
    step = min(.002, source_metrics['actual_release_drop_height_m'] * .05,
               parameters['release_drop_height'] / 4.)
    candidates = [dict(id='00_reproduce_success', kind='exact_success_reproduction',
                       parameters=parameters, optimization_window=None)]
    lower = dag['nodes']['lower_payload']
    release = dag['nodes']['release_payload']
    for index in range(1, args.max_candidates + 1):
        if step <= 1e-6:
            break
        changed = dict(parameters, release_drop_height=parameters['release_drop_height'] - index * step)
        candidates.append(dict(id=f'{index:02d}_lower_release', kind='local_success_optimization',
            parameters=changed, changed_parameter='release_drop_height',
            optimization_window=dict(first_phase='lower_payload', last_phase='release_payload',
                                     source_start_s=lower['start_s'], source_end_s=release['end_s']),
            rationale='Reduce actual measured controlled-release drop; retain the complete verified prefix',
            maximum_step_m=.002, measured_drop_scale_fraction=.05))
    plan = dict(schema=SCHEMA, method='deterministic_success_preserving_rules',
        successful_native=str(native), source_receipt_sha256=sha(native / 'receipt.json'),
        source_command_sha256=sha(native / 'issued_robot_commands.npz'),
        baseline_native=str(baseline_native), baseline_handoff=str(handoff), prepared_case=str(prepared),
        carry_reference=str(route), case_id=receipt['case_id'], python=str(args.python.resolve(strict=True)),
        controller=CONTROLLER, input_pins=pins, source_metrics=source_metrics, task_phase_dag=dag,
        evidence_layers={kind: dict(path=str(native / name), sha256=sha(native / name))
                         for name, kind in LAYERS.items()},
        candidates=candidates, retain_source_as_rollback=True,
        require_fresh_exact_reproduction_before_optimization=True,
        rule='First fully verify complete success; optimize only lower/release from actual evidence; rollback on any failure',
        earliest_failure_rule='Earlier/unknown failed gates stop late optimization; no late edit claims an earlier repair',
        original_gates=['complete_native_task_and_hard', 'single_close_and_loaded_retention',
                        'actual_duration_le_15s', 'every_step_and_interval_20mm_clearance',
                        '32state22action_alias', 'independent_exact_issued_command_replay'],
        selection_scope='One verified native result for this source hash; publication and Real2 admission remain separate',
        publication_admitted=False, api_calls=0, rendering_requested=False)
    plan['plan_identity_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return plan


def compare_recordings(source, candidate, before_s=None):
    """Compare commands, Cartesian references and observations separately."""
    import numpy as np
    checks = {}
    for relative, layer in LAYERS.items():
        with np.load(Path(source) / relative, allow_pickle=False) as left, \
                np.load(Path(candidate) / relative, allow_pickle=False) as right:
            require(set(left.files) == set(right.files), 'Recording fields differ: ' + layer)
            lt, rt = left['time_s'], right['time_s']
            lm = lt < before_s - 1e-9 if before_s is not None else np.ones(len(lt), dtype=bool)
            rm = rt < before_s - 1e-9 if before_s is not None else np.ones(len(rt), dtype=bool)
            require(bool(lm.any() and rm.any()) and np.array_equal(lt[lm], rt[rm]),
                    'Recording clocks differ: ' + layer)
            for key in left.files:
                a, b = left[key], right[key]
                if a.ndim and len(a) == len(lt):
                    require(b.ndim and len(b) == len(rt), 'Recording sample shape differs')
                    a, b = a[lm], b[rm]
                require(a.dtype == b.dtype and a.shape == b.shape
                        and a.tobytes(order='C') == b.tobytes(order='C'),
                        'Changed preserved prefix: ' + layer + ':' + key)
            checks[layer] = True
    return checks


def failure_focus(native):
    """Locate the earliest *measured* failed gate; missing evidence stays unknown.

    Terminal task predicates alone cannot timestamp loss of loaded contact.
    The actual force receipt takes priority over a late insertion symptom.
    Unknown/global alias/replay/software failures cannot authorize local repair.
    """
    native = Path(native)
    findings = []
    path = native / 'actual_grasp_receipt.json'
    if path.is_file():
        grasp = read(path)
        maximum_gap = float(grasp['role_criteria']['maximum_contact_loss_gap_s'])
        for role in grasp.get('roles', []):
            for loss in role.get('contact_loss_spans', []):
                if float(loss['duration_s']) <= maximum_gap:
                    continue
                findings.append(dict(gate='loaded_retention', role=role['role'],
                    time_s=float(loss['start_s']), evidence_path=str(path), evidence_sha256=sha(path)))
            if role.get('first_loaded_bilateral_s') is None or role.get('first_role_proof_s') is None:
                value = role.get('first_commanded_close_s')
                if value is not None:
                    findings.append(dict(gate='single_close_loaded_acquisition', role=role['role'],
                        time_s=float(value), evidence_path=str(path), evidence_sha256=sha(path)))
    hard_path = native / 'execute/immutable_hard_trace.npz'
    if hard_path.is_file():
        import numpy as np
        with np.load(hard_path, allow_pickle=False) as trace:
            ok = np.asarray(trace['measurements_valid'], dtype=bool)
            constraints = np.asarray(trace['hard_constraints'], dtype=bool)
            if constraints.ndim > 1:
                constraints = constraints.all(axis=tuple(range(1, constraints.ndim)))
            if ok.ndim > 1:
                ok = ok.all(axis=tuple(range(1, ok.ndim)))
            indices = np.flatnonzero(~(ok & constraints))
            if len(indices):
                findings.append(dict(gate='original_native_hard', time_s=float(trace['time_s'][indices[0]]),
                    evidence_path=str(hard_path), evidence_sha256=sha(hard_path)))
    history = phases(native) if (native / 'receipt.json').is_file() else []
    for item in findings:
        item['phase_windows'] = [dict(branch=r['branch'], phase=r['phase'],
            start_s=r['start_s'], end_s=r['end_s']) for r in history
            if r['start_s'] <= item['time_s'] <= r['end_s']]
    findings.sort(key=lambda row: (row['time_s'], row['gate'], row.get('role', '')))
    return dict(earliest_failed_gate=findings[0] if findings else None,
                measured_failures=findings, unknown_failure_location=not bool(findings),
                action='rollback_to_verified_source; stop this local optimization branch',
                late_phase_edits_authorized=False)


def improves(candidate_metrics, source_metrics):
    return (candidate_metrics['actual_release_drop_height_m'] <
            source_metrics['actual_release_drop_height_m'] - 1e-6
            and candidate_metrics['all_branches_complete_s'] <=
            source_metrics['all_branches_complete_s'] + .05 + 1e-8
            and candidate_metrics['all_branches_complete_s'] <=
            candidate_metrics['actual_physical_end_s'] - 1. + 1e-8)


def select_verified(source, outcomes):
    """Never select a rejected, unverified, or unmeasured attempted improvement."""
    eligible = [row for row in outcomes if row.get('verified') is True
                and row.get('kind') == 'local_success_optimization'
                and improves(row['metrics'], source['metrics'])]
    return min(eligible, key=lambda row: (row['metrics']['actual_release_drop_height_m'],
        row['metrics']['all_branches_complete_s'], row['id'])) if eligible else source


def execute(plan, output):
    output = Path(output).resolve()
    require(not output.exists() and not output.is_symlink(), 'Use a fresh rule-agent output directory')
    verify_pins(plan['input_pins'])
    output.mkdir(parents=True, exist_ok=False)
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    write_new(output / 'owner.json', dict(pid=os.getpid(), start_ticks=int(stat[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        command_sha256=hashlib.sha256(Path('/proc/self/cmdline').read_bytes()).hexdigest()))
    write_new(output / 'plan.json', plan)
    source = dict(id='preserved_verified_source', kind='preserved_success', verified=True,
                  native_case=plan['successful_native'], native_receipt_sha256=plan['source_receipt_sha256'],
                  metrics=plan['source_metrics'])
    write_new(output / 'baseline_preservation.json', dict(schema=SCHEMA, **source,
        case_id=plan['case_id'], input_pins=plan['input_pins'], publication_admitted=False))
    outcomes = []
    progress(output, 'verified_success_preserved', outcomes)
    reproduced_native = None
    verifier = load_tool('drawer_render_watch')
    for candidate in plan['candidates']:
        verify_pins(plan['input_pins'])
        directory = output / candidate['id']
        command = command_for(plan, candidate, directory / 'native')
        directory.mkdir(exist_ok=False)
        write_new(directory / 'launch.json', dict(schema=SCHEMA, command=command,
            plan_identity_sha256=plan['plan_identity_sha256'], source_receipt_sha256=plan['source_receipt_sha256'],
            input_pins=plan['input_pins'], candidate=candidate, publication_admitted=False))
        progress(output, 'native_trial', outcomes, candidate)
        with (directory / 'native.log').open('x') as log:
            child = subprocess.run(command, cwd=ROOT, env=dict(os.environ,
                PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT)
        native = directory / 'native'
        row = dict(id=candidate['id'], kind=candidate['kind'], native_case=str(native),
                   returncode=child.returncode, verified=False)
        try:
            require(child.returncode == 0, 'Existing native executor did not finish successfully')
            verify_pins(plan['input_pins'])
            proof = verifier.combine(native)
            require(read(native / 'receipt.json')['case_id'] == plan['case_id'], 'Candidate case differs')
            require(read(native / 'execute/task_contract.json') ==
                    read(Path(plan['successful_native']) / 'execute/task_contract.json'), 'Task contract changed')
            current_metrics = metrics(native)
            phase_dag(read(native / 'execute/task_contract.json'), phases(native),
                      current_metrics['actual_physical_end_s'])
            before = None if candidate['kind'] == 'exact_success_reproduction' else \
                candidate['optimization_window']['source_start_s']
            comparison_native = plan['successful_native'] if before is None else reproduced_native
            require(comparison_native is not None, 'No fully verified fresh success reproduction')
            equality = compare_recordings(comparison_native, native, before)
            if before is not None:
                lower = next(r for r in phases(native) if r['phase'] == 'lower_payload')
                require(lower['start_s'] == before, 'Local edit changed earlier phase entry clock')
            row.update(verified=True, metrics=current_metrics, unchanged_prefix_layers=equality,
                       prefix_comparison_native=str(comparison_native), prefix_before_s=before,
                       native_receipt_sha256=sha(native / 'receipt.json'), verification_checks=proof['checks'],
                       input_pins=proof['input_pins'])
            if before is None:
                reproduced_native = native
        except Exception as error:
            try:
                focus = failure_focus(native)
            except Exception as evidence_error:
                focus = dict(earliest_failed_gate=None, unknown_failure_location=True,
                    diagnostic_error=str(evidence_error), late_phase_edits_authorized=False,
                    action='rollback_to_verified_source; stop this local optimization branch')
            row.update(reason=str(error), failure_focus=focus)
        outcomes.append(row)
        write_new(directory / 'rule_verification.json', row)
        progress(output, 'candidate_verified' if row['verified'] else 'candidate_rejected',
                 outcomes, candidate)
        if not row['verified'] or (candidate['kind'] == 'local_success_optimization'
                                  and not improves(row['metrics'], plan['source_metrics'])):
            break
    verify_pins(plan['input_pins'])
    selected = select_verified(source, outcomes)
    # One immutable selection for one source. This is a refinement decision,
    # never a Real2/publication receipt or an instruction to start rendering.
    verifier.combine(Path(selected['native_case']))
    selection = dict(schema='rssr.drawer.rule_refinement_selection.v1', case_id=plan['case_id'],
        source_native=plan['successful_native'], source_receipt_sha256=plan['source_receipt_sha256'],
        source_command_sha256=plan['source_command_sha256'], selected_result=selected,
        exactly_one_verified_result=True, rollback_to_source=selected['id'] == source['id'],
        attempted_candidates=outcomes, plan_identity_sha256=plan['plan_identity_sha256'],
        selection_objective='Lower measured controlled-release drop; at most +50ms complete-branch time; unchanged native gates',
        input_pins=plan['input_pins'], publication_admitted=False, rendering_started=False, api_calls=0)
    write_new(output / 'selection.json', selection)
    progress(output, 'complete_with_verified_selection', outcomes)
    return selection


def self_test():
    """Key semantic regressions only: complete DAG, early loss, prefix, rollback."""
    import numpy as np
    checks = {}
    source = dict(id='source', metrics=dict(actual_release_drop_height_m=.04,
        all_branches_complete_s=13., actual_physical_end_s=14.9655))
    rejected = dict(id='rejected', kind='local_success_optimization', verified=False,
                    metrics=dict(actual_release_drop_height_m=.01,
                        all_branches_complete_s=12., actual_physical_end_s=14.9655))
    require(select_verified(source, [rejected]) is source, 'Rejected result replaced rollback')
    checks['failed_candidate_cannot_replace_verified_success'] = True
    with tempfile.TemporaryDirectory(prefix='drawer_rule_test_') as temporary:
        native = Path(temporary)
        write_new(native / 'receipt.json', dict(event_history=[dict(branch='payload',
            phase='lift_payload', start_s=2., end_s=6.)]))
        write_new(native / 'actual_grasp_receipt.json', dict(
            role_criteria=dict(maximum_contact_loss_gap_s=.1), roles=[dict(role='payload',
            first_loaded_bilateral_s=2., first_role_proof_s=3.,
            contact_loss_spans=[dict(start_s=4., end_s=4.2, duration_s=.2)])]))
        focus = failure_focus(native)
        require(focus['earliest_failed_gate']['phase_windows'][0]['phase'] == 'lift_payload'
                and focus['late_phase_edits_authorized'] is False, 'Early load loss was hidden')
        checks['early_loaded_loss_blocks_late_release_repair'] = True
        left, right = native / 'a', native / 'b'
        for root in (left, right):
            for relative in LAYERS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, time_s=np.array([0., 1., 2.]), values=np.array([0., 1., 2.]))
        compare_recordings(left, right, before_s=2.)
        np.savez(right / 'issued_robot_commands.npz', time_s=np.array([0., 1., 2.]),
                 values=np.array([0., 3., 2.]))
        try:
            compare_recordings(left, right, before_s=2.)
        except ValueError:
            checks['earlier_actual_command_change_rejected'] = True
        require(checks.get('earlier_actual_command_change_rejected'), 'Prefix tamper was missed')
        fake = native / 'fake_native'
        fake.mkdir()
        write_new(fake / 'receipt.json', dict(schema='rssr.sim2_refine.drawer.receipt.v1',
            accepted=True, native=dict(task_goal_met=False)))
        try:
            load_tool('drawer_render_watch').combine(fake)
        except ValueError:
            checks['accepted_label_without_full_original_task_proof_rejected'] = True
        require(checks.get('accepted_label_without_full_original_task_proof_rejected'),
                'Accepted-only result bypassed native task admission')
    # Unknown predicates cannot quietly disappear from the full-task adapter.
    try:
        phase_dag(dict(task_id='dm6', predicates=[dict(id='new', kind='unmapped')]), [], 15.)
    except ValueError:
        checks['unmapped_original_task_predicate_fails_closed'] = True
    require(checks.get('unmapped_original_task_predicate_fails_closed'), 'Task omission was accepted')
    return dict(schema=SCHEMA, passed=True, checks=checks, physics_steps=0, api_calls=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test', help='Key task/source/locality/rollback checks; no physics')
    for name in ('plan', 'run'):
        p = sub.add_parser(name, help='Validate and print only' if name == 'plan' else
                           'Execute deterministic bounded rules in a fresh output directory')
        p.add_argument('--successful-native', type=Path, required=True,
                       help='Explicit complete verified source_conditioned_v7 result to preserve')
        p.add_argument('--baseline-handoff', type=Path,
                       help='Default: exact target baseline certificate bound by successful source')
        p.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
        p.add_argument('--max-candidates', type=int, choices=(0, 1, 2), default=2,
                       help='Local optimization trials, excluding required fresh exact reproduction')
        p.add_argument('--output', type=Path, required=name == 'run')
    args = parser.parse_args(argv)
    if args.command == 'self-test':
        result = self_test()
    else:
        plan = build_plan(args)
        if args.command == 'plan':
            if args.output:
                require(not args.output.exists(), 'Planned output must be fresh')
            result = dict(plan=plan, output_created=False, physics_steps_added=0)
        else:
            result = execute(plan, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
