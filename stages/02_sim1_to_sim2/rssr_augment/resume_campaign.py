"""Resume a paused physical campaign without editing its frozen artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ROOT, BRANCH, RESULT, MODULE, read, resolve_path
from .bootstrap import validate_startup
from .layout import archive_root
from .retarget.attempts import attempts, completed_attempts
from .shared.family_autoresearch.store import Store, _event, atomic_json, digest, file_sha


def now():
    return datetime.now(timezone.utc).isoformat()


def alive_commands():
    for directory in Path('/proc').iterdir():
        if directory.name.isdigit():
            try:
                yield directory.name, (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass


def replace_prefix(value, source, destination):
    if isinstance(value, str):
        return destination + value[len(source):] if value == source or value.startswith(source + '/') else value
    if isinstance(value, list):
        return [replace_prefix(v, source, destination) for v in value]
    if isinstance(value, dict):
        return {replace_prefix(k, source, destination): replace_prefix(v, source, destination)
                for k, v in value.items()}
    return value


def implementation_pins():
    files = list((BRANCH / 'retarget').glob('*.py')) + list((BRANCH / 'vendor/runtime').rglob('*.py'))
    files += list(BRANCH.glob('*.py')) + list((BRANCH / 'shared/family_autoresearch').glob('*.py'))
    files += [BRANCH / 'shared/responses_transport.py',
              BRANCH / 'shared/joint_robot_object_v2_20260923/task_subagent.py']
    from .shared.family_autoresearch.agents import PROFILE_FILE
    files.append(PROFILE_FILE)
    return {str(p): file_sha(p) for p in sorted(set(files))}


def source_code_proof(strategy):
    archive = archive_root()
    checked = {}
    for raw, expected in strategy['implementation_sha256'].items():
        path = archive / Path(raw).relative_to(ROOT)
        if file_sha(path) != expected:
            raise ValueError('Archived implementation fingerprint changed: ' + str(path))
        checked[str(path)] = expected
    # Every controller and native runtime remains identical after module renaming.
    # case.py only adds resolution of historical paths after directory migration.
    differences = []
    for path in (BRANCH / 'retarget').glob('*.py'):
        previous = archive / 'lyc_dev_v2/retarget' / path.name
        if not previous.exists():
            raise ValueError('Missing frozen controller: ' + str(previous))
        normalized = previous.read_text().replace('from pipeline.', 'from rssr_augment.shared.')
        normalized = normalized.replace('from lyc_dev_v2.', 'from rssr_augment.')
        if normalized != path.read_text():
            if path.name != 'case.py':
                raise ValueError('Controller changed beyond namespace migration: ' + str(path))
            differences.append(path.name)
    previous_physics = archive / 'lyc_dev_v2/physical.py'
    if file_sha(previous_physics) != file_sha(BRANCH / 'physical.py'):
        raise ValueError('Physical augmentation changed')
    return {'archived_pins_verified': len(checked), 'archived_implementation_sha256': checked,
            'controller_changes': differences, 'physical_augmentation_unchanged': True,
            'transport_changes': 'module/config resolution and batch-wide concurrency limit; no retries'}


def prepare(source_experiment, destination):
    source_experiment = source_experiment.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    request = read(source_experiment / 'request.json')
    original_progress = read(source_experiment / 'progress.json')
    entries = read(source_experiment / 'runs.json')['runs']
    organization = read(ROOT / 'result/organization_manifest.json')
    pause_epoch = datetime.fromisoformat(organization['at_beijing']).timestamp()
    resume_epoch = time.time()
    pause_seconds = max(0, resume_epoch - pause_epoch)
    # Do not interfere with jobs in the separately maintained RSSR repository.
    source_runs = {str(resolve_path(row['run'])) for row in entries}
    for pid, command in alive_commands():
        if '.retarget.run_case' in command and any(path in command for path in source_runs):
            raise ValueError('Source case is still live: ' + pid)
    new_entries, proofs = [], []
    baseline = [row for row in original_progress['groups'] if row['physical_index'] < 6]
    for entry in entries:
        index = entry['physical_index']
        source = resolve_path(entry['run']).resolve()
        if index < 6:
            new_entries.append({**entry, 'run': str(source), 'preserved_complete_group': True})
            continue
        target = RESULT / 'runs/cold_start' / f'{destination.name}_physical_{index:02d}'
        proof_path = target / 'continuation.json'
        if proof_path.exists():
            proof = read(proof_path)
            if proof['source_run'] != str(source):
                raise ValueError('Existing continuation has another source')
            validate_startup(target / 'strategy.json', request['cases'][0])
        else:
            if target.exists():
                raise ValueError('Incomplete preparation requires inspection: ' + str(target))
            original = json.loads((source / 'strategy.json').read_text())
            code_proof = source_code_proof(original)
            store = Store(source)
            store.validate_events()
            state = store.read()
            preserved = {str(source / name): file_sha(source / name)
                         for name in ('strategy.json', 'manifest.json', 'state.json',
                                      'seed_profile_manifest.json', 'prior_pilot_profile_manifest.json',
                                      'source_rule_phase_index.json', 'initial_knowledge.json')}
            target.mkdir(parents=True)
            complete_count = 0
            for cid in original['cases']:
                complete = completed_attempts(source, cid)
                if len(complete) > 1:
                    raise ValueError('Multiple completed source attempts: ' + cid)
                complete_count += bool(complete)
                for attempt in attempts(source, cid):
                    linked = target / 'cases' / cid / attempt.name
                    linked.parent.mkdir(parents=True, exist_ok=True)
                    linked.symlink_to(attempt.resolve(), target_is_directory=True)
                    for name in ('receipt.json', 'scene_profile_sample_link.json', 'failure.json'):
                        if (attempt / name).is_file():
                            preserved[str(attempt / name)] = file_sha(attempt / name)
            for name in ('seed_profile_manifest.json', 'prior_pilot_profile_manifest.json',
                         'source_rule_phase_index.json', 'initial_knowledge.json'):
                shutil.copyfile(source / name, target / name)
            if (source / 'scene_profiles').exists():
                shutil.copytree(source / 'scene_profiles', target / 'scene_profiles')
                for path in (source / 'scene_profiles').rglob('sample_*.json'):
                    preserved[str(path)] = file_sha(path)
                    if file_sha(target / path.relative_to(source)) != preserved[str(path)]:
                        raise ValueError('Learned sample changed during continuation copy')
            strategy = replace_prefix(read(source / 'strategy.json'), str(source), str(target))
            strategy.update(at=now(), executor_module=MODULE, new_run=str(target),
                            implementation_sha256=implementation_pins())
            strategy['continuation'] = {'source_run': str(source), 'source_strategy_sha256': file_sha(source / 'strategy.json'),
                                        'authorization': 'user explicitly requested continue',
                                        'scope': 'unfinished physical Sim1-to-Sim2 cases only'}
            inventory = replace_prefix(read(source / 'knowledge_inventory.json'), str(source), str(target))
            inventory['continuation_source_run'] = str(source)
            atomic_json(target / 'knowledge_inventory.json', inventory)
            strategy['initial_knowledge'] = inventory
            strategy['campaign']['initial_knowledge'] = inventory
            atomic_json(target / 'strategy.json', strategy)
            manifest = json.loads((source / 'manifest.json').read_text())
            manifest.update(executor_module=MODULE, strategy_sha256=file_sha(target / 'strategy.json'),
                            continuation_source_run=str(source))
            new_store = Store(target)
            new_store.initialize(manifest, strategy['cases'])
            with new_store.transaction() as box:
                inherited = deepcopy(state)
                inherited.update(manifest_sha256=digest(manifest), status='ready')
                if inherited['started_epoch'] is not None:
                    inherited['started_epoch'] += pause_seconds
                _event(inherited, 'authorized_physical_continuation', {
                    'source_run': str(source), 'spent_preserved': state['spent'],
                    'paused_seconds_excluded_from_wall_budget': pause_seconds,
                    'pause_evidence': str(ROOT / 'result/organization_manifest.json'),
                    'complete_cases_preserved': complete_count})
                box[0] = inherited
            settlements = []
            for key, ticket in state['reservations'].items():
                if ticket['status'] != 'started':
                    continue
                output = resolve_path(ticket['metadata']['output'])
                response = output / 'response.json'
                status = 'interrupted'
                evidence = {'source_output': str(output), 'live_source_process': False,
                            'spent_preserved': True, 'automatic_duplicate_submission': False}
                if ticket['kind'] == 'api' and response.is_file():
                    saved = json.loads(response.read_text())
                    if saved.get('request_sha256') != ticket['metadata']['request_sha256']:
                        raise ValueError('Interrupted API response binding differs')
                    status = 'complete'
                    evidence.update(response_sha256=file_sha(response), outcome='saved provider response')
                new_store.settle(key, status, evidence)
                settlements.append({'key': key, 'status': status, **evidence})
            proof = {'schema': 'rssr.authorized_physical_continuation.v1', 'at': now(),
                     'source_run': str(source), 'run': str(target), 'physical_index': index,
                     'preserved_complete_cases': complete_count, 'initial_spent': state['spent'],
                     'initial_completed': state['completed'], 'preserved_files': preserved,
                     'paused_seconds_excluded_from_wall_budget': pause_seconds,
                     'settled_interrupted_operations': settlements, 'code': code_proof}
            validate_startup(target / 'strategy.json', request['cases'][0])
            for path, expected in preserved.items():
                if file_sha(path) != expected:
                    raise ValueError('Frozen source artifact changed: ' + path)
            atomic_json(proof_path, proof)
        proofs.append(proof)
        new_entries.append({**entry, 'run': str(target), 'continuation_source_run': str(source)})
        print(json.dumps({'physical_index': index, 'preserved': proof['preserved_complete_cases'],
                          'remaining': 150 - proof['preserved_complete_cases'], 'run': str(target)}), flush=True)
    atomic_json(destination / 'runs.json', {'runs': new_entries})
    atomic_json(destination / 'request.json', {**request, 'name': destination.name,
                'continuation_source_experiment': str(source_experiment), 'execution_scope': 'sim1-sim2_only'})
    atomic_json(destination / 'continuation.json', {'at': now(), 'source_experiment': str(source_experiment),
                'baseline_groups': baseline, 'resume_groups': [p['run'] for p in proofs],
                'initial_native_complete': original_progress['physical_terminal'],
                'initial_accepted': original_progress['physical_accepted'], 'expected_candidates': 1500})
    atomic_json(destination / 'runtime_control.json', {'physical_groups': 3, 'physical_cases_per_group': 7,
                'api_slots': 14, 'memory_launch_headroom_gib': 50, 'poll_seconds': 10,
                'window_seconds': 2400})


def resources():
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    memory = int(Path('/sys/fs/cgroup/memory.current').read_text())
    ceiling = int(Path('/sys/fs/cgroup/memory.max').read_text())
    return {'cpu_quota': int(quota) / int(period), 'memory_used_gib': memory / 2**30,
            'memory_headroom_gib': (ceiling - memory) / 2**30}


def run_supervisor(root):
    from .shared.family_autoresearch.agents import _profile
    # Check the local credential loader before launching expensive native cases.
    from .shared.joint_robot_object_v2_20260923.task_subagent import _client
    _client(_profile())
    with (root / 'supervisor.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = read(root / 'continuation.json')
        entries = [row for row in read(root / 'runs.json')['runs'] if row['physical_index'] >= 6]
        active, stalled = {}, {}
        atomic_json(root / 'supervisor.json', {'pid': os.getpid(), 'started_at': now(),
                    'scope': 'sim1-sim2_only', 'implementation_sha256': file_sha(Path(__file__))})
        while True:
            control = read(root / 'runtime_control.json')
            for index, (process, stream) in list(active.items()):
                if process.poll() is not None:
                    stream.close()
                    if process.returncode:
                        stalled[index] = {'returncode': process.returncode,
                                          'log': str(root / 'logs' / f'physical_{index:02d}.log')}
                    del active[index]
            rows, pending = [], []
            for entry in entries:
                index, run = entry['physical_index'], Path(entry['run'])
                strategy = read(run / 'strategy.json')
                receipts, errors, exhausted = [], [], []
                for cid in strategy['cases']:
                    complete = completed_attempts(run, cid)
                    if len(complete) > 1:
                        raise ValueError('Duplicate completed case: ' + cid)
                    if complete:
                        receipts.append(read(complete[0] / 'receipt.json'))
                    elif any((path / 'failure.json').exists() for path in attempts(run, cid)):
                        errors.append(cid)
                if (run / 'full_progress.json').is_file():
                    exhausted = [row['case_id'] for row in read(run / 'full_progress.json')['case_results']
                                 if row.get('status') == 'case_budget_exhausted']
                row = {'physical_index': index, 'run': str(run), 'physical_native_complete': len(receipts),
                       'physical_terminal': len(receipts) + len(errors) + len(exhausted),
                       'physical_accepted': sum(bool(r['accepted']) for r in receipts),
                       'execution_errors': len(errors), 'budget_exhausted_cases': exhausted,
                       'process_pid': active[index][0].pid if index in active else None}
                rows.append(row)
                if row['physical_terminal'] < len(strategy['cases']) and index not in stalled:
                    pending.append(entry)
                elif errors or exhausted:
                    stalled[index] = {'execution_errors': errors, 'budget_exhausted_cases': exhausted}
            measured = resources()
            for entry in pending:
                index, run = entry['physical_index'], Path(entry['run'])
                if index in active or len(active) >= control['physical_groups']:
                    continue
                if measured['memory_headroom_gib'] < control['memory_launch_headroom_gib']:
                    break
                validate_startup(run / 'strategy.json', read(run / 'strategy.json')['cases'][0])
                if any(v['status'] == 'started' for v in Store(run).read()['reservations'].values()):
                    stalled[index] = {'reason': 'unsettled operations require inspection'}
                    continue
                logs = root / 'logs'
                logs.mkdir(exist_ok=True)
                stream = (logs / f'physical_{index:02d}.log').open('a')
                env = dict(os.environ, PYTHONPATH=str(BRANCH.parent), MUJOCO_GL='egl',
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           RSSR_API_SLOT_DIRECTORY=str(root / 'api_slots'),
                           RSSR_API_SLOT_COUNT=str(control['api_slots']))
                command = [sys.executable, '-u', '-m', MODULE + '.run_campaign', '--strategy',
                           str(run / 'strategy.json'), '--scope', 'full', '--max-wall-seconds',
                           str(control['window_seconds'])]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[index] = (process, stream)
                print(json.dumps({'at': now(), 'launched_group': index, 'pid': process.pid}), flush=True)
                measured = resources()
            groups = binding['baseline_groups'] + rows
            complete = sum(row['physical_native_complete'] for row in groups)
            progress = {'at': now(), 'status': 'running' if active or pending else 'complete' if complete == 1500 else 'needs_inspection',
                        'expected_physical_candidates': 1500, 'physical_terminal': sum(row['physical_terminal'] for row in groups),
                        'physical_native_complete': complete,
                        'physical_accepted': sum(row['physical_accepted'] for row in groups),
                        'remaining': 1500 - complete, 'groups': groups,
                        'active_physical_indices': sorted(active),
                        'physical_processes': {str(i): p.pid for i, (p, _) in active.items()},
                        'maximum_physical_case_workers': control['physical_groups'] * control['physical_cases_per_group'],
                        'maximum_api_concurrency': control['api_slots'], 'resources': measured,
                        'stalled_groups': stalled, 'execution_scope': 'sim1-sim2_only'}
            atomic_json(root / 'progress.json', progress)
            if not active and not pending:
                atomic_json(root / 'summary.json', progress)
                return
            time.sleep(control['poll_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'supervise'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.source is None:
            parser.error('prepare requires --source')
        prepare(args.source, args.root)
    else:
        run_supervisor(args.root)


if __name__ == '__main__':
    main()
