#!/usr/bin/env python3
"""Independent refinement entry point. Candidate execution is never admission."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEFAULT_IMPLEMENTATION = PROJECT / 'sim1-sim2/s1_short_refine_v2'
DEFAULT_PYTHON = Path(sys.executable)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(pins):
    for path, expected in pins.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError('Input changed or missing: ' + path)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def registry():
    return read(ROOT / 'controllers/registry.json')


def selection(index):
    path = ROOT / 'controllers/published_selection.jsonl'
    verify({str(path): registry()['published_selection']['sha256']})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [row for row in rows if row['episode_index'] == index]
    if len(matches) != 1:
        raise ValueError('Published episode index must be in 0..110')
    return matches[0]


def strategy_source(args):
    if args.episode_index is not None:
        row = selection(args.episode_index)
        path = Path(row['strategy']['path'])
        verify({str(path): row['strategy']['sha256']})
        return path.resolve(), row
    if args.strategy is None:
        raise ValueError('Specify --strategy or --episode-index')
    return args.strategy.resolve(strict=True), None


def prepare_strategy(args):
    source, row = strategy_source(args)
    strategy = read(source)
    old_pins = strategy['implementation_sha256']
    verify(old_pins)
    versions = registry()['versions']
    inferred = {m.group(1) for p in old_pins
                if (m := re.search(r'/controller_(v\d+)/', p))}
    version = args.controller or (row['controller_version'] if row else
                                 next(iter(inferred)) if len(inferred) == 1 else None)
    if version not in versions:
        raise ValueError('Specify --controller; source controller is not unambiguous')
    # All migration pins are checked before adopting the new paths.
    for item in versions[version]['files'].values():
        verify({str(ROOT / item['path']): item['sha256']})
    validation_manifest = ROOT / 'provenance/validation_migration.json'
    validation = read(validation_manifest)
    verify({item['new_path']: item['new_sha256'] for item in validation['migrated_sources']})
    verify(validation['new_support_files'])
    pins, rewrites = {}, {}
    for path, expected in old_pins.items():
        old = Path(path)
        target = None
        if re.search(r'/controller_v\d+/', path):
            target = ROOT / versions[version]['directory'] / old.name
        elif '/work/v3_human_align/validation/' in path:
            target = ROOT / 'validation' / old.name
        if target is not None:
            if not target.is_file():
                raise ValueError('Missing migrated implementation: ' + str(target))
            pins[str(target)] = sha(target)
            rewrites[path] = str(target)
        else:
            pins[path] = expected
    # Helpers previously imported outside the controller must be pinned too.
    runtime_files = [Path(__file__), ROOT / 'controllers/registry.json', validation_manifest]
    for directory in ('validation', 'orchestration'):
        runtime_files.extend(sorted((ROOT / directory).rglob('*.py')))
    runtime_files.extend(sorted((ROOT / 'validation/_shared').rglob('*.json')))
    runtime_files.extend(sorted((ROOT / versions[version]['directory']).glob('*.py')))
    for path in runtime_files:
        pins[str(path.resolve())] = sha(path)
    strategy['implementation_sha256'] = pins
    strategy['sim2_refine_migration'] = dict(
        schema='rssr.sim2_refine.strategy_migration.v1',
        source_strategy=str(source), source_strategy_sha256=sha(source),
        source_implementation_sha256=old_pins, rewritten_paths=rewrites,
        controller_version=version,
        source_controller_version=row['controller_version'] if row else sorted(inferred),
        episode_index=row['episode_index'] if row else None,
        source_id=row['source_id'] if row else None,
        new_candidate_only=True, historical_outputs_modified=False)
    return strategy, version, row


def native_command(args, version, strategy_path, native_path, case_id):
    return [str(args.python), '-u', str(ROOT / 'controllers' / version / 'run_case.py'),
            '--strategy', str(strategy_path), '--case-id', case_id,
            '--output', str(native_path), '--implementation', str(args.implementation)]


def run_case(args):
    strategy, version, row = prepare_strategy(args)
    case_id = args.case_id or (row['case_id'] if row else None)
    if not case_id:
        raise ValueError('--case-id is required with --strategy')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh experiment directory: ' + str(output))
    args.python = args.python.resolve(strict=True)
    args.implementation = args.implementation.resolve(strict=True)
    command = native_command(args, version, output / 'strategy.json', output / 'native', case_id)
    if args.dry_run:
        return dict(dry_run=True, controller=version, command=command,
                    output_created=False, candidate_only=True)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'strategy.json', strategy)
    write_new(output / 'launch.json', dict(
        schema='rssr.sim2_refine.launch.v1', started_utc=datetime.now(timezone.utc).isoformat(),
        controller=version, case_id=case_id, command=command,
        strategy_sha256=sha(output / 'strategy.json'), candidate_only=True,
        no_hf_writes=True, launcher_sha256=sha(__file__)))
    env = dict(os.environ, PYTHONPATH=str(args.implementation), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl')
    with (output / 'native.log').open('x') as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = dict(returncode=process.returncode, native=str(output / 'native'),
                  native_receipt_exists=(output / 'native/receipt.json').is_file(),
                  candidate_only=True, publication_admitted=False)
    write_new(output / 'execution_result.json', result)
    return result


def batch(args):
    document = read(args.jobs)
    rows = document['jobs']
    if not rows or len({row['job_id'] for row in rows}) != len(rows):
        raise ValueError('Batch jobs need unique job_id values')
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError('Use a fresh batch directory: ' + str(root))
    jobs = []
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['job_id']):
            raise ValueError('job_id must be a plain directory name')
        child = argparse.Namespace(
            episode_index=row.get('episode_index'),
            strategy=Path(row['strategy']) if row.get('strategy') else None,
            controller=row.get('controller'), case_id=row.get('case_id'),
            output=root / row['job_id'], python=args.python,
            implementation=args.implementation, dry_run=True)
        plan = run_case(child)
        jobs.append((row['job_id'], child, plan))
    if args.dry_run:
        return dict(dry_run=True, jobs=[dict(job_id=i, **plan) for i, _, plan in jobs],
                    max_workers=args.workers, output_created=False)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'batch_spec.json', dict(jobs=rows, jobs_path=str(args.jobs.resolve()),
        jobs_sha256=sha(args.jobs), workers=args.workers, candidate_only=True))
    def execute(job):
        job_id, child, _ = job
        child.dry_run = False
        try:
            return dict(job_id=job_id, **run_case(child))
        except Exception as error:
            return dict(job_id=job_id, error=str(error), publication_admitted=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    result = dict(terminal=len(results), jobs=sorted(results, key=lambda r: r['job_id']),
                  candidate_only=True, no_hf_writes=True)
    write_new(root / 'batch_result.json', result)
    return result


def merge(args):
    entries, source_ids, source_orders = [], set(), set()
    for path in args.receipt:
        path = path.resolve(strict=True)
        receipt = read(path)
        if receipt.get('schema') != 'rssr.v3_human_align.admission.v1':
            raise ValueError('Unsupported task admission schema: ' + str(path))
        if receipt.get('accepted') is not True or not receipt.get('checks') or \
                not all(value is True for value in receipt['checks'].values()):
            raise ValueError('Receipt is not strictly admitted: ' + str(path))
        if not receipt.get('input_pins') or not receipt.get('publication_artifacts'):
            raise ValueError('Admission has no pinned evidence: ' + str(path))
        verify(receipt['input_pins'])
        for item in receipt['publication_artifacts'].values():
            verify({item['path']: item['sha256']})
        if receipt['source_id'] in source_ids or receipt['source_order'] in source_orders:
            raise ValueError('Duplicate logical source; select one verified result explicitly')
        source_ids.add(receipt['source_id'])
        source_orders.add(receipt['source_order'])
        entries.append(dict(admission_receipt=str(path), admission_receipt_sha256=sha(path),
            **{key: receipt[key] for key in
               ('source_id', 'source_order', 'physical_index', 'case_id', 'native_case')}))
    if args.expected_cases is not None and len(entries) != args.expected_cases:
        raise ValueError('Expected coverage does not match selected receipts')
    result = dict(schema='rssr.sim2_refine.selected_admissions.v1',
                  accepted=len(entries), expected_source_cases=args.expected_cases,
                  complete=args.expected_cases is not None and len(entries) == args.expected_cases,
                  accepted_native_cases=sorted(entries, key=lambda e: e['source_order']),
                  selection_explicit=True, no_hf_writes=True)
    write_new(args.output, result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__,
        epilog='Additional task entry points: admit --help; initial-state --help; drawer --help; drawer-rules --help; drawer-code --help.')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('status', help='Read code, controller and initial-state registry status')
    sub.add_parser('controllers', help='List preserved controller versions and published mapping')
    for name in ('prepare', 'run-case'):
        s = sub.add_parser(name, help='Prepare new pins' if name == 'prepare' else 'Run one new native candidate')
        source = s.add_mutually_exclusive_group(required=True)
        source.add_argument('--episode-index', type=int)
        source.add_argument('--strategy', type=Path)
        s.add_argument('--controller', choices=sorted(registry()['versions']))
        s.add_argument('--output', type=Path, required=True)
        if name == 'run-case':
            s.add_argument('--case-id')
            s.add_argument('--implementation', type=Path, default=DEFAULT_IMPLEMENTATION)
            s.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
            s.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('batch', help='Run explicitly listed new candidates, at most ten local CPU workers')
    s.add_argument('--jobs', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--workers', type=int, choices=range(1, 11), default=10)
    s.add_argument('--implementation', type=Path, default=DEFAULT_IMPLEMENTATION)
    s.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    s.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('merge', help='Select strict cup admissions without changing native data')
    s.add_argument('--receipt', type=Path, action='append', required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--expected-cases', type=int)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Pass the complete help and argument contract to task-specific entry points.
    delegates = {'admit': ROOT / 'orchestration/admission.py',
                 'initial-state': ROOT / 'initial_state/cli.py',
                 'stack-pose-prefix500': ROOT / 'orchestration/stack_pose_prefix500/batch_expanded_v2.py',
                 'stack-random300': ROOT / 'orchestration/stack_random300/produce_v4.py',
                 'stack-random300-case': ROOT / 'tasks/stack_cups/random300/run_case_v3.py',
                 'stack-random300-admit': ROOT / 'orchestration/stack_random300/admit_v2.py',
                 'stack-random300-batch': ROOT / 'orchestration/stack_random300/batch.py',
                 'stack-random300-sample': ROOT / 'orchestration/stack_random300/sample_v2.py',
                 'drawer-rules': ROOT / 'tasks/drawer/rule_refine_agent.py',
                 'drawer-code': ROOT / 'orchestration/drawer_luna_code_agent_v23.py',
                 'drawer-api': ROOT / 'orchestration/drawer_luna_api_v7.py',
                 'drawer': ROOT / 'orchestration/drawer_source_conditioned_cli_v1.py'}
    if argv and argv[0] in delegates:
        return subprocess.call([str(DEFAULT_PYTHON), str(delegates[argv[0]]), *argv[1:]], cwd=ROOT)
    args = parser().parse_args(argv)
    if args.command in ('status', 'controllers'):
        r = registry()
        result = dict(controller_versions=list(r['versions']),
                      published_controller_counts=r['published_selection']['controller_counts'],
                      published_episodes=r['published_selection']['episodes'],
                      runtime=str(DEFAULT_IMPLEMENTATION), native_python=str(DEFAULT_PYTHON),
                      initial_state_registry=read(ROOT / 'initial_state/registry_pointer.json'),
                      extra_commands=list(delegates),
                      admission_scope='v3 stack_cups only; drawer requires its own adapter and gates')
    elif args.command == 'prepare':
        strategy, version, _ = prepare_strategy(args)
        write_new(args.output, strategy)
        result = dict(prepared=True, controller=version, strategy=str(args.output.resolve()),
                      strategy_sha256=sha(args.output), implementation_pins=len(strategy['implementation_sha256']))
    elif args.command == 'run-case':
        result = run_case(args)
    elif args.command == 'batch':
        result = batch(args)
    else:
        result = merge(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'run-case':
        return result.get('returncode', 0)
    if args.command == 'batch' and not args.dry_run:
        return 0 if all(not r.get('error') and r.get('returncode') == 0 for r in result['jobs']) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
